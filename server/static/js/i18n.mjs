// i18n.mjs — flat-key dictionary i18n, modelled on the user's own Surrobot
// (KS278810/Surrobot-dev's frontend/index.html I18N/t()/setLang()): per-lang
// flat objects, {{param}} interpolation, and a silent fallback chain
// (current lang -> ja -> the raw key itself) so a missing translation never
// breaks the page, just shows Japanese or the key.
const SUPPORTED_LANGS = ["ja", "en", "zh"];
const LANG_STORAGE_KEY = "hx.lang";

let _dicts = null;
let _lang = "ja";
const _listeners = new Set();

function detectDefaultLang() {
  const nav = (navigator.language || "ja").toLowerCase();
  if (nav.startsWith("zh")) return "zh";
  if (nav.startsWith("en")) return "en";
  return "ja";
}

function readStoredLang() {
  try {
    const saved = localStorage.getItem(LANG_STORAGE_KEY);
    if (SUPPORTED_LANGS.includes(saved)) return saved;
  } catch (_) { /* localStorage unavailable (private mode etc.) */ }
  return null;
}

_lang = readStoredLang() || detectDefaultLang();

export async function loadDicts(baseUrl) {
  const entries = await Promise.all(
    SUPPORTED_LANGS.map(async (l) => [l, await fetch(`${baseUrl}/${l}.json`).then((r) => r.json())]));
  _dicts = Object.fromEntries(entries);
  applyStaticI18n();
}

export function t(key, params) {
  const dict = (_dicts && _dicts[_lang]) || {};
  const ja = (_dicts && _dicts.ja) || {};
  let s = dict[key] ?? ja[key] ?? key;
  if (params) {
    for (const k of Object.keys(params)) s = s.split(`{{${k}}}`).join(String(params[k]));
  }
  return s;
}

// Only [data-i18n] and [data-i18n-aria-label] are actually used anywhere
// in this UI -- html/title/placeholder sweeps existed here unused (zero
// matching elements), removed rather than kept "in case a future feature
// needs them" (see the plan's B13 finding).
export function applyStaticI18n(root = document) {
  root.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = t(el.getAttribute("data-i18n")); });
  root.querySelectorAll("[data-i18n-aria-label]").forEach((el) => {
    el.setAttribute("aria-label", t(el.getAttribute("data-i18n-aria-label")));
  });
  document.documentElement.lang = _lang;
}

export function onLangChange(fn) {
  _listeners.add(fn);
  return () => _listeners.delete(fn);
}

export function setLang(lang) {
  _lang = SUPPORTED_LANGS.includes(lang) ? lang : "ja";
  try { localStorage.setItem(LANG_STORAGE_KEY, _lang); } catch (_) { /* ignore */ }
  applyStaticI18n();
  for (const fn of _listeners) fn(_lang);
}

export function currentLang() {
  return _lang;
}
