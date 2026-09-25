// store.mjs — localStorage-backed persistence for the viewer's overrides.
// Every read/write is wrapped in try/catch: localStorage can throw in a
// private window or when site data is blocked, and a lost preference
// should never break the page.
//
// The UI only exposes two overrides now: `device` (the header GPU/CPU
// toggle) and `encoder` (Settings -> Output format). Older builds also
// wrote use_keyer/max_side/use_trimap/apply_despill/qc and encoder values
// such as "supersampled_gif"; those would keep silently applying to every
// job with no control left on screen to see or undo them, so they are
// dropped once, on load (sanitizeOverrides below). The API/CLI still
// accept all of them.
const KEY_OVERRIDES = "hx.overrides";

export const DEVICES = ["cuda", "cpu"];
export const ENCODERS = ["ss_alpha_gif", "webp", "mov"];

function readJSON(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch (_) {
    return fallback;
  }
}

function writeJSON(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch (_) { /* ignore */ }
}

/** Keep only a valid `device` and `encoder`; everything else is discarded. */
export function cleanOverrides(raw) {
  const out = {};
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return out;
  if (DEVICES.includes(raw.device)) out.device = raw.device;
  if (ENCODERS.includes(raw.encoder)) out.encoder = raw.encoder;
  return out;
}

export function sanitizeOverrides() {
  let raw = null;
  try { raw = localStorage.getItem(KEY_OVERRIDES); } catch (_) { return; }
  if (raw === null) return;
  const cleaned = cleanOverrides(readJSON(KEY_OVERRIDES, {}));
  if (JSON.stringify(cleaned) !== raw) writeJSON(KEY_OVERRIDES, cleaned);
}

export function getOverrides() {
  return cleanOverrides(readJSON(KEY_OVERRIDES, {}));
}

export function setOverrides(overrides) {
  writeJSON(KEY_OVERRIDES, cleanOverrides(overrides));
}

sanitizeOverrides();
