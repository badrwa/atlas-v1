// The orb's drawing loop.  Deliberately small: the server already decided what
// state means (motion) and what mood means (hue), so this file only animates.
//
// Budget: 30 fps cap, and the loop stops entirely when the window is hidden —
// on two Skylake cores a 60 fps canvas would steal time from speech recognition.

import { CAPTION_FADE_S, PROTOCOL_VERSION } from "./protocol.js";

const FPS = 30;
const FRAME_MS = 1000 / FPS;
const FADE_MS = CAPTION_FADE_S * 1000;  // one source: atlas_ui.theme
const MAX_SEGMENTS = 96;   // the blob is a polygon, not a per-pixel field

export function startOrb({ token = "" } = {}) {
  const body = document.body;
  const stage = document.getElementById("stage");
  const glow = document.getElementById("glow");
  const panel = document.getElementById("panel");
  const caption = document.getElementById("caption");
  const thought = document.getElementById("thought");
  const confirm = document.getElementById("confirm");
  const confirmQuestion = document.getElementById("confirm-question");
  const confirmCount = document.getElementById("confirm-count");
  const moodChip = document.getElementById("mood-chip");
  const speakerChip = document.getElementById("speaker-chip");
  const micChip = document.getElementById("mic-chip");
  const cloudChip = document.getElementById("cloud-chip");
  const degradedChip = document.getElementById("degraded-chip");
  const link = document.getElementById("link");

  const ctx = stage.getContext("2d");
  const ctxGlow = glow.getContext("2d");

  const view = {
    state: "dormant",
    mood: "calm",
    hue: 174,
    energy: 0.5,
    warmth: 0.6,
    level: 0,
    caption: "",
    captionAt: 0,
    captionFinal: false,
    thought: "",
    thoughtAt: 0,
    confirmUntil: 0,
    cloud: false,
    mic: false,
    running: true,
  };

  // ── rendering ─────────────────────────────────────────────────────
  function blob(radius, wobble, phase) {
    const points = [];
    for (let i = 0; i < MAX_SEGMENTS; i += 1) {
      const angle = (i / MAX_SEGMENTS) * Math.PI * 2;
      const noise =
        Math.sin(angle * 3 + phase) * 0.5 +
        Math.sin(angle * 5 - phase * 1.3) * 0.3 +
        Math.sin(angle * 8 + phase * 0.7) * 0.2;
      const r = radius * (1 + wobble * noise);
      points.push([r * Math.cos(angle), r * Math.sin(angle)]);
    }
    return points;
  }

  function draw(now) {
    const w = stage.width;
    const h = stage.height;
    const cx = w / 2;
    const cy = h / 2;
    const hue = view.hue;
    const phase = now / 1000;

    // How much it moves: the state sets a floor, the mood and the audio level
    // push it up.  A quiet voice still animates a little — silence is not death.
    const motion = {
      dormant: 0.05, waking: 0.22, listening: 0.30, thinking: 0.16,
      confirming: 0.08, speaking: 0.34, error: 0.18, muted: 0.0, degraded: 0.06,
    }[view.state] ?? 0.08;
    const wobble = Math.min(0.34, motion * (0.5 + view.energy) + view.level * 0.22);
    const radius = Math.min(w, h) * (0.22 + view.level * 0.05);

    ctx.clearRect(0, 0, w, h);
    ctxGlow.clearRect(0, 0, w, h);

    const points = blob(radius, wobble, phase);

    ctx.save();
    ctx.translate(cx, cy);
    ctx.beginPath();
    points.forEach(([x, y], index) => (index ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    const gradient = ctx.createRadialGradient(-radius * 0.3, -radius * 0.3, radius * 0.2, 0, 0, radius * 1.4);
    gradient.addColorStop(0, `hsla(${hue}, 85%, 68%, 0.95)`);
    gradient.addColorStop(1, `hsla(${hue - 26}, 75%, 40%, 0.75)`);
    ctx.fillStyle = gradient;
    ctx.fill();
    ctx.restore();

    // The soft halo lives on its own canvas so the blob keeps a crisp edge.
    ctxGlow.save();
    ctxGlow.translate(cx, cy);
    ctxGlow.beginPath();
    ctxGlow.arc(0, 0, radius * 1.5, 0, Math.PI * 2);
    const halo = ctxGlow.createRadialGradient(0, 0, radius * 0.4, 0, 0, radius * 1.5);
    halo.addColorStop(0, `hsla(${hue}, 80%, 60%, ${0.18 + 0.3 * view.warmth})`);
    halo.addColorStop(1, "hsla(0, 0%, 0%, 0)");
    ctxGlow.fillStyle = halo;
    ctxGlow.fill();
    ctxGlow.restore();
  }

  // ── DOM (captions, chips, countdown) ──────────────────────────────
  function paint() {
    body.dataset.state = view.state;
    body.dataset.mood = view.mood;
    const root = document.documentElement.style;
    root.setProperty("--hue", String(view.hue));
    root.setProperty("--energy", String(view.energy));
    root.setProperty("--warmth", String(view.warmth));
    root.setProperty("--level", view.level.toFixed(3));

    moodChip.textContent = view.mood;
    micChip.className = `chip mic ${view.mic ? "on" : "off"}`;
    cloudChip.className = `chip cloud ${view.cloud ? "on" : "off"}`;
    cloudChip.textContent = view.cloud ? "cloud" : "local";

    // The caption card appears only when there is something to read, and the
    // panel is what makes the orb window grow into a small card.
    const showCaption = Boolean(view.caption) && view.captionAt + FADE_MS > performance.now();
    const showThought = Boolean(view.thought) && view.thoughtAt + 3000 > performance.now();
    const confirming = view.state === "confirming" && view.confirmUntil > performance.now();
    panel.hidden = !(showCaption || showThought || confirming);
    document.body.classList.toggle("panel-open", !panel.hidden);

    if (showCaption) {
      caption.textContent = view.caption;
      caption.dataset.role = "atlas";
      caption.dir = looksRtl(view.caption) ? "rtl" : "ltr";
      caption.style.opacity = view.captionFinal ? "1" : "0.92";
    } else {
      caption.textContent = "";
    }
    thought.hidden = !showThought;
    thought.textContent = view.thought;
    confirm.hidden = !confirming;
    confirmQuestion.textContent = confirmQuestion.dataset.text || "";
    if (confirming) {
      confirmCount.textContent = String(Math.max(0, Math.ceil((view.confirmUntil - performance.now()) / 1000)));
    }
    degradedChip.hidden = view.state !== "degraded";
    degradedChip.textContent = view.degraded || "degraded";
  }

  function looksRtl(text) {
    // Same question the server answers in theme.is_rtl, asked the cheap way here:
    // first strong character wins. The browser handles the rest via dir="auto".
    const match = text.match(/[A-Za-z\u0600-\u06FF]/);
    return match ? /[\u0600-\u06FF]/.test(match[0]) : false;
  }

  // ── the socket ────────────────────────────────────────────────────
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const url = `${proto}://${location.host}/ws?token=${encodeURIComponent(token)}`;
    const socket = new WebSocket(url);

    socket.addEventListener("open", () => { link.textContent = "connected"; });
    socket.addEventListener("close", () => {
      link.textContent = "reconnecting…";
      setTimeout(connect, 1500);
    });
    socket.addEventListener("error", () => { link.textContent = "bridge unreachable"; });
    socket.addEventListener("message", (event) => {
      let message;
      try {
        message = JSON.parse(event.data);
      } catch (err) {
        return;
      }
      if (message.v !== PROTOCOL_VERSION) {
        link.textContent = `protocol mismatch (${message.v})`;
        return;
      }
      apply(message);
      paint();
    });
  }

  function apply(message) {
    switch (message.kind) {
      case "state":
        view.state = message.state;
        view.mic = Boolean(message.mic_muted) === false; // muted means the orb shows mic off
        break;
      case "mood":
        view.mood = message.mood;
        view.hue = message.hue;
        view.energy = message.energy;
        view.warmth = message.warmth;
        break;
      case "level":
        view.level = message.value;
        break;
      case "caption":
        view.caption = message.text;
        view.captionAt = performance.now();
        view.captionFinal = Boolean(message.final);
        break;
      case "tool":
        if (message.phase === "started") {
          view.thought = message.spoken;
          view.thoughtAt = performance.now();
        } else if (performance.now() - view.thoughtAt > 400) {
          view.thought = "";
        }
        break;
      case "confirm":
        view.confirmUntil = performance.now() + (message.timeout_s || 10) * 1000;
        confirmQuestion.dataset.text = message.question;
        break;
      case "degraded":
        view.state = message.degraded ? "degraded" : "dormant";
        view.degraded = message.reason || "degraded";
        break;
      case "hello":
        // Hello carries no colour: it exists to prove the link and to say which
        // capabilities are live (the cloud glyph is the honest one).
        if (message.powers) {
          view.cloud = Boolean(message.powers.cloud);
          view.mic = Boolean(message.powers.mic);
        }
        break;
      default:
        break;
    }
  }

  // ── the loop ──────────────────────────────────────────────────────
  let last = 0;
  function frame(now) {
    if (!view.running) return;
    if (now - last >= FRAME_MS) {
      last = now;
      draw(now);
      paint();
    }
    requestAnimationFrame(frame);
  }

  document.addEventListener("visibilitychange", () => {
    view.running = !document.hidden;
    if (view.running) requestAnimationFrame(frame);
  });

  // Drag anywhere (frameless window) is handled by pywebview's `easy_drag`; a
  // double-click expands the card without changing anything the server knows.
  document.addEventListener("dblclick", () => {
    panel.hidden = !panel.hidden;
  });

  connect();
  requestAnimationFrame(frame);
}
