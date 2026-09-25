// output-format.mjs — Settings -> Output format (the dialog's only
// setting). The radio cards are static markup in index.html (labels via
// data-i18n, so language switches need no re-render here); this module
// only maps the checked card to `overrides.encoder` and back.
//
// GIF is the backend default (PipelineConfig's ss_alpha_gif), so choosing
// it REMOVES the override instead of writing one -- a stored value would
// otherwise pin a future default change.
const FORMAT_TO_ENCODER = { gif: null, webp: "webp", mov: "mov" };

function encoderToFormat(encoder) {
  if (encoder === "webp") return "webp";
  if (encoder === "mov") return "mov";
  return "gif"; // no override, or an explicit ss_alpha_gif
}

/**
 * @param {HTMLFieldSetElement} group - #format-group
 * @param {{getOverrides, setOverrides}} ctx
 */
export function initOutputFormat(group, ctx) {
  const radios = Array.from(group.querySelectorAll('input[name="output-format"]'));

  function render() {
    const fmt = encoderToFormat(ctx.getOverrides().encoder);
    for (const r of radios) r.checked = r.value === fmt;
  }

  for (const r of radios) {
    r.addEventListener("change", () => {
      if (!r.checked) return;
      const overrides = { ...ctx.getOverrides() };
      const encoder = FORMAT_TO_ENCODER[r.value];
      if (encoder) overrides.encoder = encoder;
      else delete overrides.encoder;
      ctx.setOverrides(overrides);
    });
  }

  render();
  return { render };
}
