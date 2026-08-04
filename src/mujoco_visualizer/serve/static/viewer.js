/* Generic MuJoCo stream viewer.
 *
 * Builds its entire UI from the server's scene message, so it carries no knowledge of any
 * particular model -- the same page serves the fly and humanoid.xml. A host project layers
 * extra panels via window.MJViewer rather than forking this file.
 *
 * Camera drag is throttled to one message per animation frame; the server coalesces
 * whatever still piles up, so a fast drag over a slow link cannot build a backlog of stale
 * camera positions.
 *
 * Three banners, deliberately kept separate. None of them may overwrite or hide another.
 *  - #banner holds scene.backend_warning. It is set once, from the one-time scene message,
 *    and means "the physics you are watching is not the trained dynamics" (e.g. the cpu
 *    backend drives tendons with linear force generators, not the real force-length/
 *    force-velocity muscle model). It must stay up for as long as that is true.
 *  - #errbanner holds server errors: diverged / controller / render / command. These are the
 *    only messages that explain why the simulation stopped, so they persist until the user
 *    acts (Play or Reset -- the same actions that clear the error server-side). They used to
 *    share #warnbanner, which meant the very next JPEG's showWarn(meta.warn) hid them under
 *    one frame interval and they never came back: the sim stopped with no reason on screen.
 *  - #warnbanner holds what is genuinely transient: the per-frame frame_meta.warn (a DELTA
 *    for that frame, not a session-cumulative total, so it comes and goes with the condition)
 *    and connection state.
 */
(function () {
  const cv = document.getElementById("cv");
  const ctx = cv.getContext("2d");
  const statsEl = document.getElementById("stats");
  const backendEl = document.getElementById("backend");
  const bannerEl = document.getElementById("banner");
  const errbannerEl = document.getElementById("errbanner");
  const warnbannerEl = document.getElementById("warnbanner");
  const controlsEl = document.getElementById("controls");
  const cameraEl = document.getElementById("camera");
  const settingsEl = document.getElementById("settings");

  const sceneHandlers = [];
  const frameHandlers = [];
  const errorHandlers = [];

  let ws = null;
  let pendingMeta = null;
  let playing = false;

  function send(cmd) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(cmd));
  }

  // Persistent backend-mismatch notice. Only ever called from the scene handler below, so
  // nothing per-frame (showWarn) can touch it.
  function showBackendWarning(msg) {
    if (!msg) { bannerEl.hidden = true; bannerEl.textContent = ""; return; }
    bannerEl.hidden = false;
    bannerEl.textContent = msg;
  }

  // Server error notice: diverged / controller / render / command. Its own element, with
  // priority over per-frame warnings, and NOT cleared by the arrival of the next frame --
  // these messages are the only explanation the user gets for why the sim stopped.
  function showError(msg) {
    if (!msg) { errbannerEl.hidden = true; errbannerEl.textContent = ""; return; }
    errbannerEl.hidden = false;
    errbannerEl.textContent = msg;
  }

  // Transient notice: per-frame frame warnings and connection state only.
  function showWarn(msg) {
    if (!msg) { warnbannerEl.hidden = true; return; }
    warnbannerEl.hidden = false;
    warnbannerEl.textContent = msg;
  }

  function drawJpeg(blob) {
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      if (cv.width !== img.width || cv.height !== img.height) {
        cv.width = img.width;
        cv.height = img.height;
      }
      ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);
    };
    // A truncated/corrupt JPEG (a flaky link is exactly what this viewer runs over) fires
    // onerror instead of onload -- without this handler that path never revokes its blob
    // URL, leaking one object URL per bad frame over a long streaming session.
    img.onerror = () => { URL.revokeObjectURL(url); };
    img.src = url;
  }

  // -- UI built from the scene message --------------------------------------

  function buildControls(scene) {
    controlsEl.textContent = "";
    for (const group of scene.controls.groups) {
      const det = document.createElement("details");
      const sum = document.createElement("summary");
      sum.append(document.createTextNode(group.label || group.id));

      const count = document.createElement("span");
      count.className = "count";
      count.textContent = `(${group.actuators.length})`;
      sum.append(count);

      const gain = document.createElement("input");
      gain.type = "range";
      gain.min = "0"; gain.max = "1"; gain.step = "0.01"; gain.value = "1";
      gain.title = "group gain";
      gain.style.maxWidth = "7rem";
      gain.addEventListener("input", () => {
        send({ t: "ctrl_group", group: group.id, gain: Number(gain.value) });
      });
      gain.addEventListener("click", (e) => e.preventDefault());
      sum.append(gain);
      det.append(sum);

      for (const act of group.actuators) {
        const row = document.createElement("div");
        row.className = "act";

        const name = document.createElement("span");
        name.textContent = act.name;
        name.title = act.name;

        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = String(act.lo);
        slider.max = String(act.hi);
        slider.step = String((act.hi - act.lo) / 200 || 0.01);
        slider.value = "0";

        const out = document.createElement("output");
        out.value = "0.00";

        slider.addEventListener("input", () => {
          out.value = Number(slider.value).toFixed(2);
          send({ t: "ctrl", set: { [act.name]: Number(slider.value) } });
        });

        row.append(name, slider, out);
        det.append(row);
      }
      controlsEl.append(det);
    }
  }

  function buildSelectors(scene) {
    cameraEl.textContent = "";
    for (const name of (scene.cameras || []).concat(scene.presets || [])) {
      const opt = document.createElement("option");
      opt.value = name; opt.textContent = name;
      cameraEl.append(opt);
    }
    cameraEl.onchange = () => send({ t: "camera", named: cameraEl.value });

    settingsEl.textContent = "";
    const blank = document.createElement("option");
    blank.value = ""; blank.textContent = "(current)";
    settingsEl.append(blank);
    for (const name of (scene.settings_available || [])) {
      const opt = document.createElement("option");
      opt.value = name; opt.textContent = name;
      settingsEl.append(opt);
    }
    settingsEl.onchange = () => {
      if (settingsEl.value) send({ t: "settings", load: settingsEl.value });
    };

    for (const radio of document.querySelectorAll('input[name=ctrlmode]')) {
      radio.checked = radio.value === (scene.ctrl_mode || "absolute");
      radio.onchange = () => { if (radio.checked) send({ t: "mode", ctrl: radio.value }); };
    }
  }

  // -- camera drag ----------------------------------------------------------

  let dragging = false, lastX = 0, lastY = 0, az = 90, el = -20, queued = false;

  function flushCamera() {
    queued = false;
    send({ t: "camera", az: az, el: el });
  }

  cv.addEventListener("pointerdown", (e) => {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    cv.setPointerCapture(e.pointerId);
  });
  cv.addEventListener("pointerup", (e) => {
    dragging = false;
    cv.releasePointerCapture(e.pointerId);
  });
  cv.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    az += (e.clientX - lastX) * 0.4;
    el = Math.max(-89.9, Math.min(89.9, el - (e.clientY - lastY) * 0.4));
    lastX = e.clientX; lastY = e.clientY;
    if (!queued) { queued = true; requestAnimationFrame(flushCamera); }
  });

  // -- header controls ------------------------------------------------------

  const playBtn = document.getElementById("play");
  playBtn.onclick = () => {
    playing = !playing;
    // Play and Reset are exactly the commands that clear SimLoop._error server-side, so they
    // are also what clears the banner here. Anything sooner (e.g. the next frame) would erase
    // the only explanation the user has for why the sim stopped.
    if (playing) showError(null);
    send({ t: "sim", cmd: playing ? "play" : "pause" });
    playBtn.textContent = playing ? "Pause" : "Play";
  };
  document.getElementById("stepbtn").onclick = () => send({ t: "sim", cmd: "step", n: 1 });
  document.getElementById("reset").onclick = () => {
    playing = false; playBtn.textContent = "Play";
    showError(null);
    send({ t: "sim", cmd: "reset" });
  };
  document.getElementById("substeps").onchange = (e) =>
    send({ t: "speed", substeps_per_frame: Number(e.target.value) });
  document.getElementById("quality").onchange = (e) =>
    send({ t: "stream", quality: Number(e.target.value) });

  // -- socket ---------------------------------------------------------------

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.binaryType = "blob";

    ws.onmessage = (ev) => {
      if (typeof ev.data !== "string") {
        drawJpeg(ev.data);
        if (pendingMeta) {
          // scene.backend runs alongside the reported rtf: the warp backend reports around
          // 0.08x real time, and showing which backend produced this rtf keeps that
          // slow-motion framing legible instead of looking like the viewer is just broken.
          statsEl.textContent =
            `t=${pendingMeta.sim_time.toFixed(3)}s  rtf=${pendingMeta.rtf}x  ` +
            `${pendingMeta.w}x${pendingMeta.h}  #${pendingMeta.seq}`;
          // meta.warn is a per-frame delta, so clearing the banner when it is null is
          // correct: the condition genuinely stopped. It can no longer wipe a server error,
          // which lives in its own element (#errbanner).
          showWarn(pendingMeta.warn);
          frameHandlers.forEach((fn) => fn(pendingMeta));
          pendingMeta = null;
        }
        return;
      }
      const msg = JSON.parse(ev.data);
      if (msg.t === "scene") {
        backendEl.textContent = msg.backend ? `[${msg.backend}]` : "";
        showBackendWarning(msg.backend_warning);
        buildControls(msg);
        buildSelectors(msg);
        sceneHandlers.forEach((fn) => fn(msg));
      } else if (msg.t === "frame_meta") {
        pendingMeta = msg;
      } else if (msg.t === "error") {
        showError(`${msg.kind}: ${msg.msg}`);
        if (msg.paused) { playing = false; playBtn.textContent = "Play"; }
        errorHandlers.forEach((fn) => fn(msg));
      }
    };

    ws.onclose = () => {
      showWarn("disconnected - retrying");
      setTimeout(connect, 1000);
    };
  }

  window.MJViewer = {
    connect,
    send,
    onScene: (fn) => sceneHandlers.push(fn),
    onFrame: (fn) => frameHandlers.push(fn),
    onError: (fn) => errorHandlers.push(fn),
    extRoot: () => document.getElementById("ext"),
  };

  connect();
})();
