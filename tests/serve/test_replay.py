"""ArrayTrajectorySource: the in-RAM qpos store the replay loop reads from.

No GL, no HDF5 (except the one round-trip test), no fly. Bounds behaviour is tested
explicitly because an out-of-range frame reaching Session.set_qpos is how a viewer ends up
rendering another clip's pose and calling it this clip's.
"""

import numpy as np
import pytest

from mujoco_visualizer.serve.replay import ArrayTrajectorySource, TrajectorySource


def make_qpos(n_clips=3, n_frames=5, nq=4):
    """Distinct value per (clip, frame, dof) so a mix-up is visible, not plausible."""
    return (
        np.arange(n_clips * n_frames * nq, dtype=np.float32).reshape(n_clips, n_frames, nq)
    )


def test_satisfies_protocol():
    src = ArrayTrajectorySource(make_qpos())
    assert isinstance(src, TrajectorySource)


def test_shape_accessors():
    src = ArrayTrajectorySource(make_qpos(n_clips=3, n_frames=5, nq=4))
    assert src.n_clips == 3
    assert src.nq == 4
    assert src.clip_length(0) == 5
    assert src.clip_length(2) == 5


def test_qpos_returns_the_requested_frame_as_float64():
    q = make_qpos()
    src = ArrayTrajectorySource(q)
    got = src.qpos(1, 3)
    np.testing.assert_array_equal(got, q[1, 3])
    # MuJoCo's data.qpos is float64; handing it float32 forces a per-frame upcast.
    assert got.dtype == np.float64


def test_per_clip_lengths_are_honoured():
    src = ArrayTrajectorySource(make_qpos(n_frames=5), lengths=np.array([5, 2, 3]))
    assert src.clip_length(1) == 2


def test_frame_beyond_clip_length_raises_even_though_the_array_is_padded():
    # The array has 5 frames but clip 1 is only 2 long: frames 2..4 are padding and must
    # not be silently returned as if they were real.
    src = ArrayTrajectorySource(make_qpos(n_frames=5), lengths=np.array([5, 2, 3]))
    with pytest.raises(IndexError, match="frame 3 out of range"):
        src.qpos(1, 3)


@pytest.mark.parametrize("clip,frame", [(-1, 0), (3, 0), (0, -1), (0, 5)])
def test_out_of_range_raises_indexerror(clip, frame):
    src = ArrayTrajectorySource(make_qpos(n_clips=3, n_frames=5))
    with pytest.raises(IndexError):
        src.qpos(clip, frame)


def test_rejects_wrong_dimensionality():
    with pytest.raises(ValueError, match="3-D"):
        ArrayTrajectorySource(np.zeros((5, 4), dtype=np.float32))


def test_rejects_lengths_mismatch():
    with pytest.raises(ValueError, match="lengths"):
        ArrayTrajectorySource(make_qpos(n_clips=3), lengths=np.array([5, 5]))


def test_from_h5_round_trip(tmp_path):
    h5py = pytest.importorskip("h5py")
    q = make_qpos()
    path = tmp_path / "roll.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("qpos", data=q)
        f.create_dataset("rollout_lengths", data=np.array([5, 2, 3], dtype=np.int32))

    src = ArrayTrajectorySource.from_h5(path, lengths_key="rollout_lengths")
    assert src.n_clips == 3
    assert src.clip_length(1) == 2
    np.testing.assert_array_equal(src.qpos(2, 1), q[2, 1])


def test_from_h5_without_lengths_uses_full_width(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "roll.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("qpos", data=make_qpos(n_frames=7))
    src = ArrayTrajectorySource.from_h5(path)
    assert src.clip_length(0) == 7


def test_qpos_returns_copy_not_view_for_float64_source():
    """qpos() with float64 source is a copy: mutating returned array doesn't corrupt store."""
    # Use float64 fixture to ensure we hit the same-dtype path that was untested.
    q = np.arange(3 * 5 * 4, dtype=np.float64).reshape(3, 5, 4)
    src = ArrayTrajectorySource(q)

    # Get a frame and mutate it.
    got = src.qpos(1, 2)
    got[:] = 999.0  # Mutate the returned array in place.

    # Verify the source and a second qpos() call are unchanged.
    second_call = src.qpos(1, 2)
    assert not np.allclose(got, second_call), "Mutated returned array affected the store"
    np.testing.assert_array_equal(second_call, q[1, 2])


def test_external_mutation_after_construction_does_not_affect_qpos():
    """Mutating the array passed to constructor doesn't affect qpos() returns."""
    q = make_qpos()
    src = ArrayTrajectorySource(q)

    # Get the expected value before mutation.
    expected = src.qpos(1, 2)

    # Mutate the original array passed to constructor.
    q[1, 2] = 999.0

    # Verify qpos() still returns the original value.
    actual = src.qpos(1, 2)
    np.testing.assert_array_equal(actual, expected)


def test_internal_qpos_array_is_frozen():
    """self._qpos is not writable: attempted writes raise ValueError."""
    src = ArrayTrajectorySource(make_qpos())

    with pytest.raises(ValueError, match="read-only"):
        src._qpos[0, 0, 0] = 999.0
