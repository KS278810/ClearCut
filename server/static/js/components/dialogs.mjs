// dialogs.mjs — the settings <dialog> (output format + license text).
// Native <dialog> gives Escape-to-close and focus trapping for free; done
// manually here: returning focus to the opener on close, and closing on a
// click on the backdrop (a click whose target is the <dialog> element
// itself -- .dialog-body fills the whole visible box, so any click inside
// the card lands on a descendant instead).
function openWithFocusReturn(dialog) {
  const opener = document.activeElement;
  dialog.showModal();
  dialog.addEventListener("close", () => { if (opener && opener.focus) opener.focus(); }, { once: true });
}

/**
 * @param {object} els - settingsBtn, settingsDialog, settingsCloseBtn
 * @param {{onOpen?: () => void}} [hooks]
 */
export function initDialogs(els, hooks = {}) {
  els.settingsBtn.addEventListener("click", () => {
    if (hooks.onOpen) hooks.onOpen();
    openWithFocusReturn(els.settingsDialog);
  });
  els.settingsCloseBtn.addEventListener("click", () => els.settingsDialog.close());
  els.settingsDialog.addEventListener("click", (e) => {
    if (e.target === els.settingsDialog) els.settingsDialog.close();
  });
}
