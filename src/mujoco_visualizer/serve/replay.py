"""Trajectory sources: where a replaying viewer gets its state, instead of ``mj_step``.

This is the seam that makes replay symmetric with physics. ``Session`` already separates who
owns *stepping and state* (``backend``) from who owns *rendering and description*
(``Session``); a saved rollout is simply a different answer to the first question.

Everything is held in RAM on purpose. Measured on an 8.25 GB rollout: all 198 clips of policy
qpos is 127 MB and loads in 0.53 s, while a single rendered frame costs 5.4-41 ms. A viewer
with a reference overlay holds a second array of the same shape, so the resident total is
~254 MB, both frozen for the life of the process. So per-access
speed is irrelevant (a warm HDF5 chunk read was already 0.19 ms) and the thing worth buying
is not speed but *shareability*: an immutable numpy array needs no lock, so the render thread
and a background export thread can both read it, which an h5py handle cannot offer -- h5py
serialises on a global lock and is unsafe to interleave with other HDF5 C-library callers.
Immutability is enforced: the internal array is frozen, and qpos() returns a copy per call.
"""

from pathlib import Path
from typing import Optional, Protocol, Union, runtime_checkable

import numpy as np

__all__ = ["TrajectorySource", "CtrlTrajectorySource", "ArrayTrajectorySource"]


@runtime_checkable
class TrajectorySource(Protocol):
    """What the replay loop needs from any trajectory store."""

    @property
    def n_clips(self) -> int:
        ...

    def clip_length(self, clip: int) -> int:
        ...

    def qpos(self, clip: int, frame: int) -> np.ndarray:
        ...


@runtime_checkable
class CtrlTrajectorySource(TrajectorySource, Protocol):
    """A :class:`TrajectorySource` that can ALSO supply a per-frame ctrl vector.

    Deliberately a SEPARATE, derived protocol rather than two more members bolted directly
    onto :class:`TrajectorySource` itself. ``TrajectorySource`` is ``@runtime_checkable``, and
    ``isinstance`` against a runtime-checkable ``Protocol`` requires structural presence of
    EVERY declared member -- so putting ``has_ctrl``/``ctrl`` there would make them MANDATORY
    for ``isinstance(src, TrajectorySource)`` to hold at all, silently narrowing what counts as
    a valid ``TrajectorySource`` past what an optional channel should ever require (a legacy
    three-member source would stop satisfying its own protocol). Consumers never assert
    ``isinstance(src, CtrlTrajectorySource)`` either: ``SimLoop`` checks availability with
    ``getattr(source, "has_ctrl", False)`` (see ``loop.py``), a plain attribute probe that
    works identically whether or not a source's class happens to be declared against this
    protocol. This type exists to NAME the richer shape for typing/documentation, not to gate
    anything at runtime.
    """

    @property
    def has_ctrl(self) -> bool:
        """Whether this source can supply a per-frame ctrl vector via :meth:`ctrl`.

        An explicit query, checked by a consumer BEFORE ever calling :meth:`ctrl` -- never
        discovered by calling it and catching whatever a ctrl-less source raises, which could
        not be told apart from a real bug in a source that DOES claim to have ctrl.
        """
        ...

    def ctrl(self, clip: int, frame: int) -> np.ndarray:
        """The actuator command recorded for this frame. Only ever called when
        :attr:`has_ctrl` is True; a source with none need not implement this meaningfully."""
        ...


class ArrayTrajectorySource:
    """A ``(n_clips, n_frames, nq)`` array of joint positions, already in memory.

    ``lengths`` matters for padded datasets: the reference rollouts store a uniform padded
    frame count per clip, so a shorter clip's tail is repeated padding. Returning those
    frames as if they were real is exactly the kind of failure that looks like the fly
    freezing at the end of an episode rather than like a bug, so an out-of-length frame
    raises instead.

    ``ctrl``, when given, is a second ``(n_clips, n_frames, nu)`` array recorded alongside
    ``qpos`` -- the policy's muscle commands for each frame, held with exactly the same
    immutability discipline (frozen, copy-per-call) as ``qpos``. Omitting it (the default)
    leaves :attr:`has_ctrl` False, which is the explicit signal a consumer checks before ever
    calling :meth:`ctrl` -- so a plain qpos-only source built the old way is untouched and
    stays a fully valid ``TrajectorySource``.
    """

    def __init__(
        self,
        qpos: np.ndarray,
        lengths: Optional[np.ndarray] = None,
        ctrl: Optional[np.ndarray] = None,
    ):
        arr = np.asarray(qpos)
        if arr.ndim != 3:
            raise ValueError(
                f"qpos must be 3-D (n_clips, n_frames, nq); got shape {arr.shape}"
            )
        # Copy and freeze to enforce immutability: callers cannot mutate the shared store.
        self._qpos = np.array(arr, copy=True)
        self._qpos.setflags(write=False)
        n_clips, n_frames, _ = arr.shape
        if lengths is None:
            self._lengths = np.full(n_clips, n_frames, dtype=np.int64)
        else:
            lengths = np.asarray(lengths, dtype=np.int64)
            if lengths.shape != (n_clips,):
                raise ValueError(
                    f"lengths must have shape ({n_clips},) to match qpos; got {lengths.shape}"
                )
            self._lengths = np.minimum(lengths, n_frames)

        self._ctrl: Optional[np.ndarray] = None
        if ctrl is not None:
            carr = np.asarray(ctrl)
            if carr.ndim != 3:
                raise ValueError(
                    f"ctrl must be 3-D (n_clips, n_frames, nu); got shape {carr.shape}"
                )
            if carr.shape[:2] != (n_clips, n_frames):
                raise ValueError(
                    f"ctrl's (n_clips, n_frames) {carr.shape[:2]} must match qpos's "
                    f"{(n_clips, n_frames)}"
                )
            # Same freeze-and-copy discipline as `_qpos` above, for the same reason: an
            # immutable array needs no lock, so the render thread and export thread can both
            # read it safely.
            self._ctrl = np.array(carr, copy=True)
            self._ctrl.setflags(write=False)

    @classmethod
    def from_h5(
        cls,
        path: Union[str, Path],
        qpos_key: str = "qpos",
        lengths_key: Optional[str] = None,
    ) -> "ArrayTrajectorySource":
        """Read an HDF5 rollout fully into memory and close the file.

        The file handle is deliberately NOT retained: after this returns, nothing in the
        process holds an HDF5 object, so no thread can serialise on h5py's global lock.
        """
        import h5py  # local: h5py is not a hard dependency of this package

        with h5py.File(str(path), "r") as handle:
            if qpos_key not in handle:
                raise KeyError(
                    f"{path}: no dataset {qpos_key!r} (available at root: "
                    f"{sorted(handle.keys())})"
                )
            qpos = handle[qpos_key][:]
            lengths = handle[lengths_key][:] if lengths_key else None
        return cls(qpos, lengths)

    @property
    def n_clips(self) -> int:
        return int(self._qpos.shape[0])

    @property
    def nq(self) -> int:
        return int(self._qpos.shape[2])

    @property
    def has_ctrl(self) -> bool:
        return self._ctrl is not None

    @property
    def nu(self) -> int:
        if self._ctrl is None:
            raise ValueError("this source has no ctrl channel; check has_ctrl first")
        return int(self._ctrl.shape[2])

    def ctrl(self, clip: int, frame: int) -> np.ndarray:
        """The actuator command recorded for this frame. Raises if this source was built
        with no ``ctrl`` array -- callers must check :attr:`has_ctrl` first, per the
        protocol's explicit-query contract, rather than relying on this to signal absence."""
        if self._ctrl is None:
            raise ValueError(
                "this source has no ctrl channel (has_ctrl is False); check has_ctrl "
                "before calling ctrl()"
            )
        self._check_clip(clip)
        length = int(self._lengths[clip])
        if not 0 <= frame < length:
            raise IndexError(
                f"frame {frame} out of range for clip {clip} (length {length})"
            )
        # Always a copy, for the same reason qpos() always is: the source is a shared,
        # supposedly-frozen store, read from more than one thread.
        return np.array(self._ctrl[clip, frame], dtype=np.float64)

    def clip_length(self, clip: int) -> int:
        self._check_clip(clip)
        return int(self._lengths[clip])

    def qpos(self, clip: int, frame: int) -> np.ndarray:
        self._check_clip(clip)
        length = int(self._lengths[clip])
        if not 0 <= frame < length:
            raise IndexError(
                f"frame {frame} out of range for clip {clip} (length {length})"
            )
        # Always copy to ensure callers cannot reach the shared store via in-place mutations.
        # The per-call copy cost is invisible against a 5.4-41 ms render, and the gain is
        # that a consumer writing in place (e.g. normalising before MjData assignment) cannot
        # corrupt the trajectory every other thread reads.
        return np.array(self._qpos[clip, frame], dtype=np.float64)

    def _check_clip(self, clip: int) -> None:
        if not 0 <= clip < self.n_clips:
            raise IndexError(f"clip {clip} out of range (have {self.n_clips} clips)")
