// icons.mjs — inline SVG icon set. No icon font/library dependency. All
// strokes use currentColor so tone follows whatever CSS color is applied
// to the containing element. aria-hidden="true" on every icon: paired
// with an aria-label on the containing button, so the icon itself is
// redundant to a screen reader.
//
// Trimmed to just what the two-pane UI actually uses (download/archive
// came back 2026-09-23 for the result pane's own Download buttons).
const STROKE = 'width="1em" height="1em" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"';

export const ICONS = {
  upload: `<svg viewBox="0 0 24 24" ${STROKE}><path d="M12 16V4M12 4l-4 4M12 4l4 4"/><path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/></svg>`,
  settings: `<svg viewBox="0 0 24 24" ${STROKE}><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.6a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>`,
  download: `<svg viewBox="0 0 24 24" ${STROKE}><path d="M12 4v11M12 15l-4.5-4.5M12 15l4.5-4.5"/><path d="M5 19.5h14"/></svg>`,
  archive: `<svg viewBox="0 0 24 24" ${STROKE}><rect x="3.5" y="4" width="17" height="4.5" rx="1"/><path d="M5 8.5V19a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8.5"/><path d="M10 12.5h4"/></svg>`,
  // Clear = "back to empty": a counter-clockwise arrow, not a trash can
  // (nothing on the server is deleted -- only the staged view is reset).
  clear: `<svg viewBox="0 0 24 24" ${STROKE}><path d="M4 12a8 8 0 1 0 2.4-5.7"/><path d="M4 4.5v4h4"/></svg>`,
  close: `<svg viewBox="0 0 24 24" ${STROKE}><path d="M18 6L6 18M6 6l12 12"/></svg>`,
  // Filled, not stroked (like the others) -- a play/stop glyph reads better
  // solid at the small size of the run button that sits between the panes.
  play: `<svg viewBox="0 0 24 24" width="1em" height="1em" aria-hidden="true" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>`,
  stop: `<svg viewBox="0 0 24 24" width="1em" height="1em" aria-hidden="true" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>`,
};
