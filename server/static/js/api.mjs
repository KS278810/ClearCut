// api.mjs — thin fetch wrappers for server/app.py's REST endpoints. Every
// non-2xx response is turned into an ApiError carrying the same
// language-neutral {code, detail} the server sent, so callers can render
// `err.${code}` via i18n regardless of which endpoint failed.
//
// Trimmed to what the single-file, single-job minimal UI (2026-09-11
// rewrite) actually calls -- listJobs/retryJob/deleteJob/zipUrl (history,
// re-run-in-a-different-mode, and the download-all-as-zip affordance)
// were all removed along with the history panel and mode picker.
export class ApiError extends Error {
  constructor(code, detail, status) {
    super(detail ? `${code}: ${detail}` : code);
    this.code = code;
    this.detail = detail;
    this.status = status;
  }
}

async function handle(resp) {
  if (resp.status === 204) return null;
  let body = null;
  try { body = await resp.json(); } catch (_) { /* no/invalid JSON body */ }
  if (!resp.ok) {
    const err = (body && body.error) || {};
    throw new ApiError(err.code || "E_HTTP", err.detail || resp.statusText, resp.status);
  }
  return body;
}

export async function createJob(files, mode, overrides) {
  const fd = new FormData();
  for (const file of files) fd.append("files", file, file.name);
  fd.append("mode", mode);
  if (overrides && Object.keys(overrides).length) fd.append("overrides", JSON.stringify(overrides));
  return handle(await fetch("/api/jobs", { method: "POST", body: fd }));
}

export async function cancelJob(jobId) {
  return handle(await fetch(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" }));
}

/** `relPath` is a job-relative path like "outputs/思考_matte.gif" -- each
 * segment is percent-encoded but the slashes are preserved, matching the
 * server's `files/{name:path}` route. */
export function fileUrl(jobId, relPath, download = false) {
  const encoded = relPath.split("/").map(encodeURIComponent).join("/");
  return `/api/jobs/${encodeURIComponent(jobId)}/files/${encoded}${download ? "?download=1" : ""}`;
}

/** The rolling live-preview PNG for one specific clip (see server/jobs.py's
 * `_preview` hook) -- `seq` is only ever used as a cache-busting query
 * param, its value doesn't matter to the server. `clipId` is required
 * (unlike the server's own `?clip=` default of "whichever clip is
 * running") so a multi-clip UI can show the right clip's own preview
 * even when the user is looking at a clip that isn't the active one. */
export function previewUrl(jobId, clipId, seq) {
  return `/api/jobs/${encodeURIComponent(jobId)}/preview?clip=${encodeURIComponent(clipId)}&t=${seq}`;
}

/** Job-wide ZIP of every completed clip's outputs -- shown once >=2 clips
 * in a job are done (see app.mjs's #zip-btn). */
export function zipUrl(jobId) {
  return `/api/jobs/${encodeURIComponent(jobId)}/zip`;
}

export async function systemStatus() {
  return handle(await fetch("/api/system"));
}
