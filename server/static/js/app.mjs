// app.mjs — ClearCut front-end entry point (2026-09-11 minimal-UI
// rewrite). Owns just two pieces of state -- {file, job} -- and derives
// every visible thing from them in render(); there is no view-state
// machine with named screens anymore (booting/offline/staged/running/
// finished used to be five separate DOM blocks toggled by a `state.view`
// enum -- now there is exactly one screen, and render() only ever
// toggles [hidden]/src/text on the same fixed elements). No framework,
// no build step -- plain ES modules, same policy as the sibling
// web_cpu/ tool.
import * as api from "./api.mjs";
import { ApiError } from "./api.mjs";
import { subscribeJob } from "./sse.mjs";
import { applyStaticI18n, currentLang, loadDicts, onLangChange, setLang, t } from "./i18n.mjs";
import * as store from "./store.mjs";
import { ICONS } from "./icons.mjs";
import { initOutputFormat } from "./components/output-format.mjs";
import { initDialogs } from "./components/dialogs.mjs";

const VIDEO_EXTENSIONS = [".mp4", ".mov", ".webm"];
const NON_TERMINAL = new Set(["queued", "waiting_gpu", "running"]);
const TERMINAL = new Set(["done", "failed", "cancelled"]);
//: Mirrors tool/pipeline/keyer.py's MIN_KEY_SATURATION / probe_says_keyable
//: -- only used to decide whether to show the "CPU is slow" note, never to
//: change what the server does.
const MIN_KEY_SATURATION = 20.0;

const el = (id) => document.getElementById(id);
const els = {
  sbDot: el("sb-dot"),
  deviceToggle: el("device-toggle"), langToggle: el("lang-toggle"),
  settingsBtn: el("settings-btn"),

  paneSrc: el("pane-src"), srcVideo: el("src-video"), dropZone: el("drop-zone"),
  dropOverlay: el("drop-overlay"), dropOverlayText: el("drop-overlay-text"),
  srcMeta: el("src-meta"), clearBtn: el("clear-btn"),
  srcStrip: el("src-strip"),
  paneOut: el("pane-out"), outPreview: el("out-preview"), outVideo: el("out-video"),
  outMeta: el("out-meta"), downloadBtn: el("download-btn"), zipBtn: el("zip-btn"),
  outStrip: el("out-strip"),

  errorBanner: el("error-banner"), errorMessage: el("error-message"), errorRetryBtn: el("error-retry-btn"),
  errorDetail: el("error-detail"),

  statusLine: el("status-line"), cpuNote: el("cpu-note"),
  barTrack: el("bar-track"), barFill: el("bar-fill"), timeLine: el("time-line"),
  runBtn: el("run-btn"), icoRun: el("icon-run"), runBtnLabel: el("run-btn-label"),
  fileInput: el("file-input"),

  settingsDialog: el("settings-dialog"), settingsCloseBtn: el("settings-close-btn"),
  formatGroup: el("format-group"),
};

/** @type {{ files: File[], fileUrls: string[], activeIndex: number, job: object|null }}
 * `files`/`fileUrls` are always the same length and in drop order, which is
 * also job.clips' order ("00", "01", ... -- see server/jobs.py's
 * _dedupe_name/_new_clip_record). `activeIndex` is the ONE thing both panes
 * and both thumbnail strips key off of -- clicking a thumbnail on either
 * side changes it and both sides re-render around the same clip. */
const state = { files: [], fileUrls: [], activeIndex: 0, job: null };
let unsubscribeSse = null;
let cancelRequested = false;
let sseDisconnected = false; // true between an onDisconnect and the next successful snapshot
let etaTickerId = null; // setInterval id, running only while a job is non-terminal
let resolvedDevice = null; // /api/system's resolved_device -- which side "auto" actually lands on

// -- static icons --
el("icon-settings").innerHTML = ICONS.settings;
el("icon-upload").innerHTML = ICONS.upload;
el("icon-settings-close").innerHTML = ICONS.close;
el("icon-run").innerHTML = ICONS.play; // render() swaps this for ICONS.stop while running
el("icon-clear").innerHTML = ICONS.clear;
el("icon-download").innerHTML = ICONS.download;
el("icon-zip").innerHTML = ICONS.archive;

// Header GPU/CPU toggle: the only UI for `overrides.device` (the settings
// dialog's device <select> was removed 2026-09-23). Clicking the
// already-pressed button clears the override (back to "auto" --
// server/presets.py's build_config resolves that against this host's
// actual CUDA availability).
function syncDeviceToggle() {
  const overrideDevice = store.getOverrides().device;
  const device = overrideDevice || "auto";
  for (const btn of els.deviceToggle.querySelectorAll("button")) {
    btn.setAttribute("aria-pressed", String(btn.dataset.device === device));
    // Neither button is "pressed" while on auto, which otherwise leaves no
    // trace of which side auto is actually resolving to on this host --
    // a dotted ring (see app.css) hints at it without claiming the button
    // was explicitly chosen (aria-pressed stays reserved for that).
    btn.classList.toggle("is-auto-resolved", !overrideDevice && btn.dataset.device === resolvedDevice);
  }
}

const outputFormat = initOutputFormat(els.formatGroup, {
  getOverrides: store.getOverrides,
  setOverrides: store.setOverrides,
});
initDialogs({
  settingsBtn: els.settingsBtn, settingsDialog: els.settingsDialog, settingsCloseBtn: els.settingsCloseBtn,
}, { onOpen: outputFormat.render });

// Header language switch (moved out of the settings dialog). hx.lang
// persistence lives in i18n.mjs's setLang().
function syncLangToggle() {
  for (const btn of els.langToggle.querySelectorAll("button")) {
    btn.setAttribute("aria-pressed", String(btn.dataset.lang === currentLang()));
  }
}
els.langToggle.addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (btn && btn.dataset.lang !== currentLang()) setLang(btn.dataset.lang);
});
onLangChange(syncLangToggle);
syncLangToggle();

els.deviceToggle.addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (!btn) return;
  const wanted = btn.dataset.device;
  const overrides = { ...store.getOverrides() };
  if (overrides.device === wanted) delete overrides.device; // toggle off -> back to auto
  else overrides.device = wanted;
  store.setOverrides(overrides);
  syncDeviceToggle();
});

onLangChange(render);

// ---------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------

function activeClip() {
  return state.job && state.job.clips && state.job.clips[state.activeIndex];
}

function clipById(clipId) {
  return state.job && state.job.clips && state.job.clips.find((c) => c.id === clipId);
}

/** m:ss / h:mm:ss, zero-padded. */
function formatDuration(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  const mm = String(m).padStart(h > 0 ? 2 : 1, "0");
  const ss = String(sec).padStart(2, "0");
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

/** Coarse "about N seconds/minutes/hours" -- deliberately imprecise (the
 * estimate itself is a moving target; false precision like "3分12秒" would
 * imply an accuracy this model doesn't have). Rounds UP so the number
 * never reads as "less time than it'll actually take". */
function formatRemaining(seconds) {
  const s = Math.max(0, seconds);
  if (s < 60) return t("time.about_sec", { n: Math.ceil(s / 10) * 10 || 10 });
  if (s < 3600) return t("time.about_min", { n: Math.ceil(s / 60) });
  const h = Math.floor(s / 3600), m = Math.ceil((s % 3600) / 60);
  return t("time.about_hr", { h, m });
}

function showError(message, { retry = false, detail = "", tone = "" } = {}) {
  els.errorBanner.hidden = false;
  // "warn" = a heads-up about something the user tried (amber), not a
  // failure (red) -- e.g. dropping a new video while one is processing.
  if (tone) els.errorBanner.dataset.tone = tone;
  else delete els.errorBanner.dataset.tone;
  els.errorMessage.textContent = message;
  els.errorRetryBtn.hidden = !retry;
  // The coded message ("エンコードがタイムアウトしました") never says WHICH
  // clip or WHAT the underlying ffmpeg/CUDA error actually was -- job.error.detail
  // (already server-sanitized, see server/jobs.py's _sanitize_detail) carries
  // that, but was previously stored in state and never rendered anywhere.
  els.errorDetail.hidden = !detail;
  els.errorDetail.textContent = detail;
}

function hideError() {
  els.errorBanner.hidden = true;
}

function render() {
  const job = state.job;
  const clip = activeClip();
  const running = job && NON_TERMINAL.has(job.status);
  const finished = job && TERMINAL.has(job.status);
  const multi = state.files.length > 1;

  // Status dot: a colour-only signal (no label text) -- deliberately
  // minimal per the brief, but still named for assistive tech via
  // aria-label rather than dropping the information entirely.
  let tone = "mute", toneLabel = t("status.ready");
  if (job && job.status === "waiting_gpu") { tone = "warn"; toneLabel = t("status.waiting_gpu"); }
  else if (job && job.status === "queued") { tone = "busy"; toneLabel = t("status.queued"); }
  else if (running) { tone = "busy"; toneLabel = t("status.running"); }
  else if (job && job.status === "done") { tone = "good"; toneLabel = t("status.done"); }
  else if (job && job.status === "cancelled") { tone = "warn"; toneLabel = t("status.cancelled"); }
  else if (job && job.status === "failed") { tone = "danger"; toneLabel = t("status.failed"); }
  els.sbDot.dataset.tone = tone;
  els.sbDot.setAttribute("aria-label", toneLabel);

  // waiting_gpu and queued are otherwise INDISTINGUISHABLE from a running
  // job on screen: same amber dot as "cancelled" for waiting_gpu, same
  // progress bar + "Cancel" button either way. A ten-minute GPU wait with
  // no visible explanation was the exact confusion that motivated this --
  // this is the one place text is added purely to disambiguate machine
  // state, not to describe an error or a completed action.
  //
  // "Preparing…" covers the gap between a clip starting and its first
  // real stage tick (server/jobs.py's "prepare" stage: model load, key
  // sampling -- which used to be tens of silent seconds). A clip that is
  // running with no stage at all yet is treated the same way. Any OTHER
  // stage name, known or not, just shows the normal progress bar.
  const runningClip = job && job.clips ? job.clips.find((c) => c.status === "running") : null;
  const preparing = !!(running && job.status === "running" && runningClip
                       && (!runningClip.stage || runningClip.stage === "prepare"));
  if (runningClip && !runningClip._localStart) runningClip._localStart = Date.now();
  let statusText = "";
  if (job && (job.status === "waiting_gpu" || job.status === "queued")) statusText = toneLabel;
  else if (preparing) statusText = t("status.preparing");
  els.statusLine.hidden = !statusText;
  els.statusLine.textContent = statusText;

  const note = cpuSlowNote(job);
  els.cpuNote.hidden = !note;
  els.cpuNote.textContent = note || "";

  // Left pane: the source video for the ACTIVE file, once one is staged/
  // running/finished. .src (property) always resolves to an absolute
  // URL, so getAttribute("src") is the only way to detect "unchanged" --
  // without this a re-render() while a thumbnail is focused would keep
  // restarting playback from frame 0 even when nothing actually changed.
  const activeUrl = state.fileUrls[state.activeIndex] || "";
  const hasFiles = state.files.length > 0;
  els.srcVideo.hidden = !hasFiles;
  if (hasFiles && els.srcVideo.getAttribute("src") !== activeUrl) {
    els.srcVideo.src = activeUrl;
    els.srcVideo.play().catch(() => {});
  }
  // The click-to-choose button only exists for the EMPTY pane: once a
  // video is showing, replacing it is a drop anywhere on the pane (see the
  // pane drag handlers), and a corner button would sit on top of the
  // video's own controls.
  els.dropZone.hidden = hasFiles;
  els.srcMeta.textContent = hasFiles ? state.files[state.activeIndex].name : "";
  els.srcMeta.title = els.srcMeta.textContent;
  els.clearBtn.hidden = !hasFiles;
  els.clearBtn.disabled = !!running; // nothing to reset mid-run; Cancel first

  // Right pane: the ACTIVE clip's live preview while it's running, its
  // real output once IT (not necessarily the whole job) is done, or
  // nothing (just the checker background) otherwise -- a multi-clip job
  // still running clip 2 can already show clip 1's finished result if the
  // user clicks back to it. .just-done plays a one-shot completion glow
  // (CSS, see app.css) -- adding a class that's already present is a
  // no-op, so this doesn't replay on every render(), only the first time
  // the active clip's own result actually appears.
  els.paneOut.classList.toggle("just-done", clip && clip.status === "done");
  els.outPreview.hidden = true;
  els.outVideo.hidden = true;
  if (clip && clip.status === "done" && clip.outputs && clip.outputs.primary) {
    const rel = clip.outputs.primary;
    if (/\.mov$/i.test(rel)) {
      // Setting .src unconditionally re-triggers the media load algorithm
      // even for the SAME url -- every render() (e.g. a language switch
      // via onLangChange) restarted a finished .mov from frame 0.
      // .src (the property) always returns the browser's RESOLVED
      // absolute URL, never the relative string that was assigned, so
      // comparing against that would never match -- getAttribute("src")
      // returns the raw value actually set below.
      const url = api.fileUrl(job.id, rel);
      if (els.outVideo.getAttribute("src") !== url) {
        els.outVideo.src = url;
        els.outVideo.play().catch(() => {});
      }
      els.outVideo.hidden = false;
    } else {
      const url = api.fileUrl(job.id, rel);
      if (els.outPreview.getAttribute("src") !== url) els.outPreview.src = url;
      els.outPreview.hidden = false;
    }
  } else if (clip && clip.status === "running" && clip.preview) {
    const url = api.previewUrl(job.id, clip.id, clip._previewSeq || 0);
    if (els.outPreview.getAttribute("src") !== url) els.outPreview.src = url;
    els.outPreview.hidden = false;
  }

  // Progress bar: the mean across every clip (a job of 1 clip -- the
  // common case -- reduces to exactly that one clip's own fraction, so
  // this subsumes the old single-clip logic rather than branching on
  // `multi`). Clicking between thumbnails must NOT move this bar --
  // it lives in the job-level dock next to Cancel/ZIP/whole-job ETA, and
  // a bar that jumps every time the user looks at a different clip would
  // be read as "the job's progress just changed" when nothing did.
  // Per-clip progress is instead shown on each running thumbnail's own
  // progress line (see renderStrips()). Idle state hides the bar with
  // `visibility` (via .is-idle), not the `hidden` attribute -- the bar
  // keeps its layout height either way, so the run button below it
  // doesn't shift position the instant a job starts or ends.
  els.barTrack.classList.toggle("is-idle", !running);
  // No real stage yet (queued / waiting for the GPU / preparing): a 0%
  // bar would read as "stuck", so show an indeterminate sweep instead.
  const indeterminate = !!(running && (job.status !== "running" || preparing));
  els.barTrack.classList.toggle("is-indeterminate", indeterminate);
  if (indeterminate) els.barTrack.removeAttribute("aria-valuenow");
  else if (running && job.clips.length) {
    const fraction = job.clips.reduce((sum, c) => sum + (TERMINAL.has(c.status) ? 1 : (c.fraction || 0)), 0)
                     / job.clips.length;
    const pct = Math.min(100, Math.round(fraction * 100));
    els.barFill.style.width = `${pct}%`;
    els.barTrack.setAttribute("aria-valuenow", String(pct));
  }
  renderTimeLine(job, clip, running, multi);

  // The one action control: a play glyph + "Convert" when idle/finished-
  // with-a-file, a stop glyph + "Cancel" while a job is in flight -- never
  // both, since this UI never runs two jobs at once. Visible text (not
  // just an icon) doubles as the accessible name, so no separate
  // aria-label is needed.
  els.runBtn.classList.toggle("is-running", running); // drives the pulsing ring in app.css
  if (running) {
    els.icoRun.innerHTML = ICONS.stop;
    els.runBtnLabel.textContent = cancelRequested ? t("btn.cancelling") : t("btn.cancel");
    els.runBtn.disabled = cancelRequested;
  } else {
    els.icoRun.innerHTML = ICONS.play;
    els.runBtnLabel.textContent = t("btn.convert");
    els.runBtn.disabled = !state.files.length;
  }

  // Download (result pane, bottom-right): the ACTIVE clip's own output,
  // once IT has one -- switching clips changes what it downloads. The
  // output's file name sits on the left of the same row, so it's clear
  // what (and which format) the button saves.
  if (clip && clip.status === "done" && clip.outputs && clip.outputs.primary) {
    const rel = clip.outputs.primary;
    els.downloadBtn.href = api.fileUrl(job.id, rel, true);
    els.downloadBtn.hidden = false;
    els.outMeta.textContent = rel.split("/").pop();
  } else {
    els.downloadBtn.hidden = true;
    els.outMeta.textContent = "";
  }
  els.outMeta.title = els.outMeta.textContent;

  // ZIP ("Download all"): only once the job is over AND at least 2 clips
  // actually produced something -- a mid-run job's zip would be an
  // incomplete snapshot (JobStore.zip_path only rebuilds it as-needed for
  // a settled job; see server/jobs.py).
  const doneClips = job ? job.clips.filter((c) => c.status === "done").length : 0;
  if (finished && doneClips >= 2) {
    els.zipBtn.href = api.zipUrl(job.id);
    els.zipBtn.hidden = false;
  } else {
    els.zipBtn.hidden = true;
  }

  // render() only ever SETS the banner (for a job that ended "failed" --
  // recomputing this every call is harmless, it's idempotent); it never
  // unconditionally clears it. Clearing is an explicit action tied to
  // "starting something fresh" (addFiles(), setActive(), the Run click
  // handler, a successful boot()) -- this used to be an `else if
  // (!running) hideError()` living right here, which cleared the banner
  // on EVERY render() call, including the one the Run click handler's own
  // catch block triggers immediately after calling showError() for a
  // fetch that failed before a job even existed (job was still null, so
  // neither branch of that old logic recognized the error as current --
  // confirmed by testing: the message flashed and vanished within the
  // same tick).
  if (clip && clip.status === "failed") {
    // The ACTIVE clip's own error -- with one clip this is unchanged from
    // before; with several, job.error.detail is a generic "k of n clip(s)
    // failed" summary (server/jobs.py) that doesn't say what actually
    // went wrong with any one of them.
    const err = clip.error;
    showError(t(`err.${(err && err.code) || "UNKNOWN"}`, { code: err && err.code }),
              { detail: (err && err.detail) || "" });
  } else if (finished && job.status === "failed" && !job.clips.some((c) => c.status === "failed")) {
    // Edge case with no failed clip to point to at all (e.g. every clip
    // was rejected before it could even start) -- fall back to the job-
    // level error rather than showing nothing.
    const err = job.error;
    showError(t(`err.${(err && err.code) || "UNKNOWN"}`, { code: err && err.code }),
              { detail: (err && err.detail) || "" });
  }

  renderStrips(job, clip);
}

/** Part 2 of the 2026-09-23 plan: CPU is only practical for the
 * colour-only keyer path (a neural clip on CPU runs ~30-70 s/frame). When
 * THIS job resolved to CPU and the clip being looked at (or the one
 * running) won't be keyed, say so with the time it will take -- a heads-up
 * only; the user's device choice is never overridden. `job.config.device`
 * is the RESOLVED device ("auto" never reaches the job record). */
function clipTakesKeyer(c, config) {
  if (c.auto_keyer === true) return true;
  if (c.auto_keyer === false) return false;
  if (config.use_keyer === "off") return false;
  const p = c.probe || {};
  return !!p.bg_is_chroma_class && p.bg_key_saturation != null && p.bg_key_saturation >= MIN_KEY_SATURATION;
}

function clipRemainingS(c) {
  if (c.status === "running" && c._eta && c._eta.eta_s != null) {
    return Math.max(0, c._eta.eta_s - (Date.now() - c._eta.at) / 1000);
  }
  if (c.eta_s != null) return c.eta_s;
  if (c.predicted_s && c.predicted_s.total != null) return c.predicted_s.total;
  return null;
}

function cpuSlowNote(job) {
  if (!job || !NON_TERMINAL.has(job.status)) return null;
  const config = job.config || {};
  if (config.device !== "cpu" || !job.clips) return null;
  const active = activeClip();
  const candidates = [active, job.clips.find((c) => c.status === "running")]
    .filter((c) => c && (c.status === "pending" || c.status === "running"));
  const target = candidates.find((c) => !clipTakesKeyer(c, config));
  if (!target) return null;
  const remaining = clipRemainingS(target);
  return remaining != null ? t("cpu.slow", { v: formatRemaining(remaining) }) : t("cpu.slow_unknown");
}

/** "残り 約4分 · 経過 2:13" (+ " · 全体 約9分" once >=2 clips) while the
 * active clip is running; "全体 約9分" alone if the active clip itself
 * isn't the one running. Segments with a null value are simply omitted
 * (never "残り null"); if nothing is left to show, the whole line hides. */
function renderTimeLine(job, clip, running, multi) {
  if (!running) { els.timeLine.hidden = true; return; }
  const parts = [];
  if (clip && clip.status === "running" && clip._eta) {
    const { elapsed_s, eta_s, at } = clip._eta;
    const drift = (Date.now() - at) / 1000;
    if (eta_s != null) parts.push(t("time.remaining", { v: formatRemaining(Math.max(0, eta_s - drift)) }));
    if (elapsed_s != null) parts.push(t("time.elapsed", { v: formatDuration(elapsed_s + drift) }));
    if (multi && clip._eta.job_eta_s != null) {
      parts.push(t("time.job_remaining", { v: formatRemaining(Math.max(0, clip._eta.job_eta_s - drift)) }));
    }
  } else if (clip && clip.status === "running" && clip._localStart) {
    // No server tick yet (the first moments of "prepare"): elapsed from
    // when this page first saw the clip running, so the clock is moving
    // from the very start instead of appearing only once a tick lands.
    parts.push(t("time.elapsed", { v: formatDuration((Date.now() - clip._localStart) / 1000) }));
  } else if (multi) {
    // Active clip isn't the one running -- fall back to whichever clip
    // IS, purely to source a whole-job estimate (nothing per-clip shown).
    const runningClip = job.clips.find((c) => c.status === "running" && c._eta);
    if (runningClip) {
      const drift = (Date.now() - runningClip._eta.at) / 1000;
      if (runningClip._eta.job_eta_s != null) {
        parts.push(t("time.job_remaining", { v: formatRemaining(Math.max(0, runningClip._eta.job_eta_s - drift)) }));
      }
    }
  }
  els.timeLine.hidden = parts.length === 0;
  els.timeLine.textContent = parts.join(" · ");
}

// ---------------------------------------------------------------------
// File staging (source-pane drop, click-to-choose, Clear)
// ---------------------------------------------------------------------

function looksLikeVideo(file) {
  const lower = file.name.toLowerCase();
  return VIDEO_EXTENSIONS.some((ext) => lower.endsWith(ext)) || file.type.startsWith("video/");
}

/** A drop/pick REPLACES the whole staged set (fresh files -> fresh job,
 * never a silent retry of the old one -- same rule the old single-file
 * addFile() followed). Non-video entries are silently dropped rather than
 * blocking the rest of the batch UNLESS every entry was rejected, matching
 * how a rejected clip in an already-created job degrades (see
 * server/jobs.py's create()) rather than failing the whole drop. */
function jobInFlight() {
  return !!(state.job && NON_TERMINAL.has(state.job.status));
}

/** Drops every trace of the previous source/result from the media
 * elements and strips. Without this a replaced (or cleared) set kept
 * playing the old, already-revoked blob URL in #src-video, and a new set
 * with the SAME clip count reused thumbnails still pointing at the old
 * files (buildStrips only ran when the count changed). */
function resetMedia() {
  for (const v of [els.srcVideo, els.outVideo]) {
    v.pause();
    v.removeAttribute("src");
    v.load();
  }
  els.outPreview.removeAttribute("src");
  els.srcStrip.innerHTML = "";
  els.outStrip.innerHTML = "";
  els.paneOut.classList.remove("just-done");
}

function detachJob() {
  if (unsubscribeSse) { unsubscribeSse(); unsubscribeSse = null; }
  stopEtaTicker();
  state.job = null;
  cancelRequested = false;
}

function addFiles(fileList) {
  if (jobInFlight()) { // no swapping mid-run -- but say so, never silently ignore
    showError(t("drop.busy"), { tone: "warn" });
    return;
  }
  const files = Array.from(fileList).filter(looksLikeVideo);
  if (!files.length) {
    showError(t("drop.not_video"));
    return;
  }
  for (const url of state.fileUrls) URL.revokeObjectURL(url);
  detachJob();
  resetMedia();
  state.files = files;
  state.fileUrls = files.map((f) => URL.createObjectURL(f));
  state.activeIndex = 0;
  hideError();
  render();
}

/** Clear (source pane, bottom-right): back to the empty start screen. */
function clearAll() {
  if (jobInFlight()) return;
  for (const url of state.fileUrls) URL.revokeObjectURL(url);
  detachJob();
  resetMedia();
  state.files = [];
  state.fileUrls = [];
  state.activeIndex = 0;
  hideError();
  render();
}
els.clearBtn.addEventListener("click", clearAll);

function setActive(index) {
  if (!state.files.length) return;
  state.activeIndex = Math.max(0, Math.min(state.files.length - 1, index));
  hideError(); // moving to a different clip shouldn't keep showing a stale one's error
  render();
}

// Outside the source pane a drop must not navigate the tab to the file.
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", (e) => e.preventDefault());

els.dropZone.addEventListener("click", () => els.fileInput.click());

// The WHOLE source pane -- including over a playing video -- is the drop
// target, so a wrong file can always be replaced by dropping the right
// one on top. dragenter/dragleave fire for every child element crossed,
// hence the depth counter rather than a plain toggle.
let dragDepth = 0;
function isFileDrag(e) {
  return !!(e.dataTransfer && Array.from(e.dataTransfer.types || []).includes("Files"));
}
function setDragOver(on) {
  els.paneSrc.classList.toggle("drag-over", on);
  els.dropOverlay.hidden = !on;
  if (!on) return;
  const busy = jobInFlight();
  els.dropOverlay.dataset.tone = busy ? "warn" : "";
  els.dropOverlayText.textContent = busy ? t("drop.busy_short") : (state.files.length ? t("drop.replace") : t("drop.title"));
}
els.paneSrc.addEventListener("dragenter", (e) => {
  if (!isFileDrag(e)) return;
  e.preventDefault();
  dragDepth += 1;
  setDragOver(true);
});
els.paneSrc.addEventListener("dragover", (e) => {
  if (!isFileDrag(e)) return;
  e.preventDefault();
  e.dataTransfer.dropEffect = "copy";
});
els.paneSrc.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) setDragOver(false);
});
els.paneSrc.addEventListener("drop", (e) => {
  e.preventDefault();
  e.stopPropagation();
  dragDepth = 0;
  setDragOver(false);
  if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
});
els.fileInput.addEventListener("change", () => {
  if (els.fileInput.files.length) addFiles(els.fileInput.files);
  els.fileInput.value = "";
});

// ---------------------------------------------------------------------
// Left/right playback sync (comparing source vs result side by side)
// ---------------------------------------------------------------------
// Both media elements got native <video> `controls` (previously neither
// had any -- they just autoplayed on loop with no way to pause either
// one). When the result is itself a <video> (MOV output), play/pause on
// either element mirrors onto the other in real time -- `_syncing` guards
// against the mirrored call re-triggering its own "play"/"pause" event
// and bouncing back and forth forever.
let _syncing = false;
function _mirror(from, to) {
  from.addEventListener("play", () => {
    if (_syncing || to.hidden) return;
    _syncing = true;
    to.play().catch(() => {}).finally(() => { _syncing = false; });
  });
  from.addEventListener("pause", () => {
    if (_syncing || to.hidden) return;
    _syncing = true;
    to.pause();
    _syncing = false;
  });
}
_mirror(els.srcVideo, els.outVideo);
_mirror(els.outVideo, els.srcVideo);

// The result is far more often a GIF/WebP (<img>, the default output) than
// a MOV (<video>) -- a native animated image has NO play/pause exposed to
// JS at all (a real browser limitation, not something this app can add
// cheaply; a true per-frame pause would need decoding the GIF by hand,
// e.g. via the WebCodecs ImageDecoder API). The reachable equivalent:
// clicking the result restarts BOTH sides together from frame 0, so a
// comparison can always be re-started in lockstep even without a true
// pause. `cursor:pointer` + a11y role communicate this is clickable.
els.outPreview.style.cursor = "pointer";
els.outPreview.setAttribute("role", "button");
els.outPreview.addEventListener("click", () => {
  if (els.outPreview.hidden) return;
  els.srcVideo.currentTime = 0;
  els.srcVideo.play().catch(() => {});
  // Re-assigning the SAME src doesn't restart a GIF/WebP's animation in
  // any browser -- a new URL (even one resolving to identical bytes) is
  // required to force the decoder to start over from frame 0.
  const base = els.outPreview.src.split("?")[0];
  els.outPreview.src = `${base}?r=${Date.now()}`;
});

// ---------------------------------------------------------------------
// Thumbnail strips (only shown once >=2 files are staged)
// ---------------------------------------------------------------------

/** Builds (once per file-count change) and updates (every render) the two
 * thumbnail strips. The ACTIVE thumbnail is hidden rather than removed --
 * this keeps every <video>/<img> in the DOM permanently rather than
 * destroying/recreating it on every swap, which is what makes clicking
 * back to a clip instant (no reload) and is why "N-1 visible thumbnails"
 * falls out naturally instead of needing its own bookkeeping. */
function renderStrips(job, clip) {
  const n = state.files.length;
  els.srcStrip.hidden = n < 2;
  els.outStrip.hidden = n < 2;
  if (n < 2) return;

  if (els.srcStrip.children.length !== n) buildStrips(n);

  for (let i = 0; i < n; i++) {
    const isActive = i === state.activeIndex;
    const c = job && job.clips && job.clips[i];
    const status = c ? c.status : "pending";

    const srcThumb = els.srcStrip.children[i];
    srcThumb.hidden = isActive;
    srcThumb.setAttribute("aria-selected", String(isActive));
    srcThumb.dataset.status = status;
    srcThumb.style.setProperty("--frac", c && status === "running" ? String(c.fraction || 0) : "0");
    srcThumb.setAttribute("aria-label", t("thumb.label", { name: state.files[i].name, status: t(`status.${status}`) }));

    const outThumb = els.outStrip.children[i];
    outThumb.hidden = isActive;
    outThumb.setAttribute("aria-selected", String(isActive));
    outThumb.dataset.status = status;
    outThumb.style.setProperty("--frac", c && status === "running" ? String(c.fraction || 0) : "0");
    outThumb.setAttribute("aria-label", srcThumb.getAttribute("aria-label"));
    const img = outThumb.querySelector("img");
    if (c && c.status === "done" && c.outputs && c.outputs.primary && !/\.mov$/i.test(c.outputs.primary)) {
      const url = api.fileUrl(job.id, c.outputs.primary);
      if (img.getAttribute("src") !== url) img.src = url;
      img.hidden = false;
    } else if (c && c.status === "running" && c.preview) {
      const url = api.previewUrl(job.id, c.id, c._previewSeq || 0);
      if (img.getAttribute("src") !== url) img.src = url;
      img.hidden = false;
    } else {
      img.hidden = true;
    }
  }
  updateRovingTabindex(els.srcStrip);
  updateRovingTabindex(els.outStrip);
}

function buildStrips(n) {
  els.srcStrip.innerHTML = "";
  els.outStrip.innerHTML = "";
  for (let i = 0; i < n; i++) {
    const srcThumb = document.createElement("button");
    srcThumb.type = "button";
    srcThumb.className = "thumb";
    srcThumb.setAttribute("role", "option");
    srcThumb.dataset.index = String(i);
    const video = document.createElement("video");
    video.className = "thumb-media";
    video.muted = true; video.playsInline = true; video.preload = "metadata";
    video.src = state.fileUrls[i];
    srcThumb.appendChild(video);
    els.srcStrip.appendChild(srcThumb);

    const outThumb = document.createElement("button");
    outThumb.type = "button";
    outThumb.className = "thumb";
    outThumb.setAttribute("role", "option");
    outThumb.dataset.index = String(i);
    const img = document.createElement("img");
    img.className = "thumb-media";
    img.alt = "";
    img.hidden = true;
    img.addEventListener("error", () => {
      // The clip-just-finished preview-PNG-deleted race (known, benign --
      // see server/static/tests/e2e.test.mjs's comment on the same
      // pattern): hide rather than show a broken-image glyph.
      if ((img.getAttribute("src") || "").includes("/preview?")) img.hidden = true;
    });
    outThumb.appendChild(img);
    els.outStrip.appendChild(outThumb);
  }
}

function updateRovingTabindex(strip) {
  const visible = Array.from(strip.children).filter((c) => !c.hidden);
  visible.forEach((c, i) => c.tabIndex = i === 0 ? 0 : -1);
}

function onStripClick(e) {
  const thumb = e.target.closest(".thumb");
  if (thumb) setActive(Number(thumb.dataset.index));
}

function onStripKeydown(e) {
  if (e.key !== "ArrowLeft" && e.key !== "ArrowRight" && e.key !== "Enter" && e.key !== " ") return;
  const strip = e.currentTarget;
  const visible = Array.from(strip.children).filter((c) => !c.hidden);
  const focused = document.activeElement;
  const idx = visible.indexOf(focused);
  if (e.key === "Enter" || e.key === " ") {
    if (idx >= 0) { e.preventDefault(); setActive(Number(focused.dataset.index)); }
    return;
  }
  if (idx < 0) return;
  e.preventDefault();
  const delta = e.key === "ArrowLeft" ? -1 : 1;
  const next = visible[(idx + delta + visible.length) % visible.length];
  next.focus();
}

for (const strip of [els.srcStrip, els.outStrip]) {
  strip.addEventListener("click", onStripClick);
  strip.addEventListener("keydown", onStripKeydown);
}

// ---------------------------------------------------------------------
// Job lifecycle
// ---------------------------------------------------------------------

function startEtaTicker() {
  if (etaTickerId) return;
  // Re-renders once a second purely so the elapsed/remaining TEXT keeps
  // advancing between server ticks (SSE events are throttled -- see
  // server/jobs.py's PROGRESS_MIN_INTERVAL_S -- and encode's tick-less
  // tail can go minutes between events even with the heartbeat's 2s
  // period feeding new numbers; this just interpolates smoothly in
  // between whatever numbers are currently held).
  etaTickerId = setInterval(() => { if (state.job) render(); }, 1000);
}

function stopEtaTicker() {
  if (etaTickerId) { clearInterval(etaTickerId); etaTickerId = null; }
}

function enterRunning(job) {
  state.job = job;
  cancelRequested = false;
  render();
  if (NON_TERMINAL.has(job.status)) { subscribeToJob(job.id); startEtaTicker(); }
}

function subscribeToJob(jobId) {
  if (unsubscribeSse) unsubscribeSse();
  unsubscribeSse = subscribeJob(jobId, {
    onSnapshot: (job) => {
      // A snapshot always follows a (re)connect (server/app.py's SSE
      // endpoint sends it first thing) -- if we get here, the connection
      // that mattered just succeeded, so the "reconnecting..." banner a
      // prior onDisconnect may have raised is now stale and must be
      // cleared explicitly (render() itself never clears the banner as a
      // side effect -- see the note on render()'s error-banner block).
      if (sseDisconnected) { sseDisconnected = false; hideError(); }
      // A reconnect's snapshot is a FRESH job object from the server, so
      // every clip's _previewSeq/_eta (client-only bookkeeping, never
      // sent by the server) would otherwise vanish -- carry them over by
      // clip id so the right pane and the time line don't go blank for a
      // clip that was already showing something.
      const oldJob = state.job;
      state.job = job;
      for (const clip of job.clips || []) {
        const old = oldJob && oldJob.clips && oldJob.clips.find((c) => c.id === clip.id);
        if (old) {
          if (old._previewSeq) clip._previewSeq = old._previewSeq;
          if (old._eta) clip._eta = old._eta;
          if (old._localStart) clip._localStart = old._localStart;
        }
        // A snapshot's clip carries the server's own eta_s/elapsed_s
        // fields (job.json schema) but NOT the client-only `_eta` shape
        // renderTimeLine() reads -- without synthesizing it here, the
        // FIRST subscribe (there's no `old` to carry over from yet) shows
        // nothing until the next "progress" SSE tick or heartbeat, which
        // for a slow-ticking stage (encode's palette pass, or simply
        // this connection being the very first one) can be a highly
        // visible multi-second blank gap. job_eta_s itself isn't stored
        // on the clip (only computed per-tick server-side), so it's
        // reconstructed the same way the server does: this clip's own
        // eta_s plus every still-pending clip's predicted total.
        if (!clip._eta && clip.status === "running" && clip.eta_s != null) {
          const pendingTotal = (job.clips || []).reduce((sum, c) => sum + (
            c.id !== clip.id && c.status === "pending" && c.predicted_s ? (c.predicted_s.total || 0) : 0
          ), 0);
          clip._eta = {
            elapsed_s: clip.elapsed_s, eta_s: clip.eta_s,
            job_eta_s: clip.eta_s + pendingTotal, at: Date.now(),
          };
        }
        // A snapshot's clip.preview is the server-side PATH (truthy
        // string) if one already exists, but carries no cache-busting
        // sequence number -- synthesize one so the <img> actually
        // (re)loads it; subsequent onPreview events overwrite this with
        // the real seq.
        if (clip.preview && !clip._previewSeq) clip._previewSeq = Date.now();
      }
      render();
      if (TERMINAL.has(job.status)) {
        if (unsubscribeSse) { unsubscribeSse(); unsubscribeSse = null; }
        stopEtaTicker();
      }
    },
    onDisconnect: () => {
      sseDisconnected = true;
      showError(t("err.NETWORK"));
      render();
    },
    onProgress: (data) => {
      const clip = clipById(data.clip_id);
      if (!clip) return;
      clip.stage = data.stage;
      clip.frames_done = data.done;
      clip.frames_total = data.total;
      if (typeof data.fraction === "number") clip.fraction = data.fraction;
      if (data.eta_s != null || data.elapsed_s != null || data.job_eta_s != null) {
        clip._eta = { elapsed_s: data.elapsed_s, eta_s: data.eta_s, job_eta_s: data.job_eta_s, at: Date.now() };
      }
      render();
    },
    onPreview: (data) => {
      const clip = clipById(data.clip_id);
      if (!clip) return;
      clip.preview = true; // presence, not the path -- the endpoint resolves the current file itself
      clip._previewSeq = data.seq;
      render();
    },
    onClipStatus: (data) => {
      const clip = clipById(data.clip_id);
      if (!clip) return;
      Object.assign(clip, {
        status: data.status, timings: data.timings, outputs: data.outputs,
        metrics: data.metrics, error: data.error, preview: null,
        started_at: data.started_at, finished_at: data.finished_at, predicted_s: data.predicted_s,
      });
      render();
    },
    onJobStatus: (data) => {
      if (!state.job) return;
      state.job.status = data.status;
      if (data.error) state.job.error = data.error;
      render();
      if (TERMINAL.has(data.status)) {
        if (unsubscribeSse) { unsubscribeSse(); unsubscribeSse = null; }
        stopEtaTicker();
      }
    },
  });
}

els.runBtn.addEventListener("click", async () => {
  if (state.job && NON_TERMINAL.has(state.job.status)) {
    if (cancelRequested) return;
    cancelRequested = true;
    render();
    try { await api.cancelJob(state.job.id); } catch (_) { /* ignore -- SSE reflects the true state */ }
    return;
  }
  if (!state.files.length) return;
  els.runBtn.disabled = true;
  hideError();
  try {
    const job = await api.createJob(state.files, "quick", store.getOverrides());
    enterRunning(job);
  } catch (e) {
    showError(e instanceof ApiError ? t(`err.${e.code}`, { code: e.code }) : t("err.UNKNOWN", { code: "?" }));
    render();
  }
});

// ---------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------

async function boot() {
  try {
    await loadDicts("/static/locales");
    const sys = await api.systemStatus();
    resolvedDevice = sys.resolved_device || null;
  } catch (_) {
    applyStaticI18n();
    outputFormat.render();
    syncDeviceToggle();
    showError(t("status.offline"), { retry: true });
    return;
  }
  applyStaticI18n();
  outputFormat.render();
  syncDeviceToggle();
  syncLangToggle();
  hideError();
  render();
}

els.errorRetryBtn.addEventListener("click", boot);

boot();
