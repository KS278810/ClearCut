// sse.mjs — subscribe to a job's live event stream with exponential-backoff
// reconnect. Deliberately does NOT rely on EventSource's own built-in retry:
// we want an explicit "reconnecting" callback for the UI, and a bounded
// backoff (native EventSource retries at a fixed, usually short interval
// forever). Resync-on-reconnect falls out for free: server/app.py's SSE
// endpoint always sends a `snapshot` event first, so simply reopening the
// connection re-delivers the true current state -- no separate GET needed.
export function subscribeJob(jobId, handlers) {
  let closed = false;
  let backoffMs = 1000;
  let es = null;
  let retryTimer = null;

  function connect() {
    if (closed) return;
    es = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/events`);
    es.onopen = () => {
      backoffMs = 1000;
    };
    es.onmessage = (ev) => {
      let data;
      try { data = JSON.parse(ev.data); } catch (_) { return; }
      switch (data.type) {
        case "snapshot": handlers.onSnapshot && handlers.onSnapshot(data.job); break;
        case "progress": handlers.onProgress && handlers.onProgress(data); break;
        case "preview": handlers.onPreview && handlers.onPreview(data); break;
        case "clip_status": handlers.onClipStatus && handlers.onClipStatus(data); break;
        case "job_status": handlers.onJobStatus && handlers.onJobStatus(data); break;
        default: break;
      }
    };
    es.onerror = () => {
      es.close();
      if (closed) return;
      handlers.onDisconnect && handlers.onDisconnect();
      retryTimer = setTimeout(connect, backoffMs);
      backoffMs = Math.min(backoffMs * 2, 10000);
    };
  }

  connect();
  return function unsubscribe() {
    closed = true;
    if (retryTimer) clearTimeout(retryTimer);
    if (es) es.close();
  };
}
