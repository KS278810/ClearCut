// e2e.test.mjs — Puppeteer E2E suite for server/static/, run against the
// fake-pipeline backend (fake_server.py) so every scenario finishes in
// milliseconds regardless of real GPU/CPU availability or contention.
// Run: npm install && npm test   (from this directory)
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { after, before, test } from "node:test";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";

// `.pathname` on a file:// URL is PERCENT-ENCODED (this repo's own path has
// non-ASCII directory names) -- spawn() needs a real filesystem path, not
// a URL-encoded one, so fileURLToPath() is required here, not a shortcut.
const HERE = fileURLToPath(new URL(".", import.meta.url));
const THEORY_ROOT = fileURLToPath(new URL("../../..", import.meta.url));
const SAMPLE_ROOT = fileURLToPath(new URL("../../../../sample/", import.meta.url));
const PYTHON = THEORY_ROOT + "venv/bin/python3";
const PORT = 8971;
const BASE = `http://127.0.0.1:${PORT}`;
const SAMPLE_CLIP = SAMPLE_ROOT + "dinosaur/Triceratops.mp4";
// A filename fake_server.py's fake_run_clip specifically recognizes (see
// its own comment) to block on `cancel` for ~2s instead of finishing
// instantly -- copied once here rather than renaming a shared fixture,
// so only the ONE test that actually needs the slow path pays for it.
const SLOW_CLIP = path.join(os.tmpdir(), "heroextractor_e2e_slow_clip.mp4");
fs.copyFileSync(SAMPLE_CLIP, SLOW_CLIP);
// A filename fake_run_clip recognizes to raise instead of succeeding --
// see its own comment.
const FAIL_CLIP = path.join(os.tmpdir(), "heroextractor_e2e_fail_clip.mp4");
fs.copyFileSync(SAMPLE_CLIP, FAIL_CLIP);
// A second, distinctly-named clip for the multi-file drop tests -- a
// second copy of SAMPLE_CLIP under a different name so the two clips'
// server-assigned ids ("00"/"01") map predictably to drop order without
// relying on _dedupe_name's "(2)" suffixing.
const SECOND_CLIP = path.join(os.tmpdir(), "heroextractor_e2e_second_clip.mp4");
fs.copyFileSync(SAMPLE_CLIP, SECOND_CLIP);
// fake_run_clip lingers in the "prepare" stage for this filename (~1.5s).
const PREPARING_CLIP = path.join(os.tmpdir(), "heroextractor_e2e_preparing_clip.mp4");
fs.copyFileSync(SAMPLE_CLIP, PREPARING_CLIP);
// Keyable (flat saturated backdrop per fake_validate_upload) AND slow, for
// the "no CPU note on a keyer clip" check.
const KEYABLE_SLOW_CLIP = path.join(os.tmpdir(), "heroextractor_e2e_keyable_slow_clip.mp4");
fs.copyFileSync(SAMPLE_CLIP, KEYABLE_SLOW_CLIP);
const DATA_DIR = "/tmp/heroextractor_fake_server_test_data";

let puppeteer;
try {
  ({ default: puppeteer } = await import("puppeteer"));
} catch {
  // Fall back to a machine-wide install pointed at by PUPPETEER_PATH, for a
  // machine where nobody has run `npm install` in this test directory yet.
  if (!process.env.PUPPETEER_PATH) {
    throw new Error(
      "puppeteer not found -- run `npm install` in this directory, or set " +
      "PUPPETEER_PATH to a machine-wide puppeteer.js entry point.");
  }
  ({ default: puppeteer } = await import(process.env.PUPPETEER_PATH));
}

let serverProc;
let browser;

// 60s, not 15s: on this shared machine under heavy load (load average 40+)
// just importing the server stack has been measured taking >15s.
async function waitForServer(timeoutMs = 60000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const resp = await fetch(BASE + "/api/system");
      if (resp.ok) return;
    } catch { /* not up yet */ }
    await sleep(200);
  }
  throw new Error("fake_server did not become ready in time");
}

before(async () => {
  await import("node:fs/promises").then((fs) => fs.rm(DATA_DIR, { recursive: true, force: true }));
  serverProc = spawn(PYTHON, ["-m", "uvicorn", "fake_server:app", "--host", "127.0.0.1", "--port", String(PORT)], {
    cwd: HERE,
    env: { ...process.env, FAKE_SERVER_DATA_DIR: DATA_DIR },
    stdio: ["ignore", "pipe", "pipe"],
  });
  await waitForServer();
  browser = await puppeteer.launch({ headless: true, args: ["--no-sandbox"] });
});

after(async () => {
  if (browser) await browser.close();
  if (serverProc) serverProc.kill();
});

/** `overrides` is written RAW to localStorage "hx.overrides" before the app
 * loads (default: none) -- the suite shares one browser profile, so
 * without this a device/encoder choice made by one test would leak into
 * the next. A string is stored verbatim (for the sanitization test). */
async function newPage({ overrides = {}, lang = "en" } = {}) {
  const page = await browser.newPage();
  // Seeded before any page script runs, and only on this tab's FIRST
  // document (sessionStorage survives reloads) -- so a test that reloads
  // sees what the app itself persisted, not the seed again.
  await page.evaluateOnNewDocument((o, l) => {
    if (sessionStorage.getItem("e2e.seeded")) return;
    sessionStorage.setItem("e2e.seeded", "1");
    localStorage.setItem("hx.overrides", typeof o === "string" ? o : JSON.stringify(o));
    // L7: pin the language deterministically rather than relying on
    // whatever this headless browser's navigator.language resolves to.
    localStorage.setItem("hx.lang", l);
  }, overrides, lang);
  const consoleErrors = [];
  const pageErrors = [];
  page.on("console", (msg) => { if (msg.type() === "error") consoleErrors.push(msg.text()); });
  page.on("pageerror", (err) => pageErrors.push(String(err)));
  await page.goto(BASE + "/", { waitUntil: "networkidle0", timeout: 20000 });
  await sleep(300);
  return { page, consoleErrors, pageErrors };
}

function assertNoErrors(consoleErrors, pageErrors) {
  assert.deepEqual(consoleErrors, [], "unexpected console errors");
  assert.deepEqual(pageErrors, [], "unexpected page errors");
}

test("page boots with an empty two-pane layout, no raw i18n keys, Run disabled", async () => {
  const { page, consoleErrors, pageErrors } = await newPage();
  assert.equal(await page.title(), "ClearCut");
  assert.equal(await page.$eval(".sb-brand", (el) => el.textContent), "ClearCut");
  assert.ok(await page.$eval('link[rel="icon"]', (el) => el.href.startsWith("data:image/svg+xml")));
  const runLabel = await page.$eval("#run-btn-label", (el) => el.textContent);
  assert.ok(!runLabel.startsWith("btn."), `run button showed a raw i18n key: ${runLabel}`);
  assert.equal(await page.$eval("#run-btn", (el) => el.disabled), true, "Run must start disabled (no file yet)");
  assert.equal(await page.$eval("#src-video", (el) => el.hidden), true);
  assert.equal(await page.$eval("#out-preview", (el) => el.hidden), true);
  assert.equal(await page.$eval("#error-banner", (el) => el.hidden), true);

  // Settings holds ONLY the output format: three radio cards, GIF (the
  // backend default) checked when nothing is stored, no leftover
  // device/keyer/max_side/trimap/despill/QC controls, no raw i18n keys.
  await page.click("#settings-btn");
  await page.waitForSelector("#settings-dialog[open]");
  const dialogText = await page.$eval("#settings-dialog", (el) => el.textContent);
  assert.ok(!/\b(fmt|settings|adv|btn)\./.test(dialogText), `settings dialog showed a raw i18n key: ${dialogText}`);
  const formats = await page.$$eval('#settings-dialog input[type="radio"]', (els) => els.map((e) => e.value));
  assert.deepEqual(formats, ["gif", "webp", "mov"]);
  assert.equal(await page.$eval("#fmt-gif", (el) => el.checked), true);
  assert.equal(await page.$$eval("#settings-dialog select, #settings-dialog input[type=checkbox], #settings-dialog input[type=number]",
    (els) => els.length), 0, "no other settings controls should remain");
  // Every card is a real <label for> its own radio (clicking text selects it).
  const labelled = await page.$$eval(".format-card", (els) => els.every((l) => l.htmlFor && document.getElementById(l.htmlFor)));
  assert.ok(labelled);

  assertNoErrors(consoleErrors, pageErrors);
  await page.close();
});

test("dropping a video shows it in the left pane and enables Run", async () => {
  const { page, consoleErrors, pageErrors } = await newPage();
  const input = await page.$("#file-input");
  await input.uploadFile(SAMPLE_CLIP);
  await sleep(200);

  assert.equal(await page.$eval("#src-video", (el) => el.hidden), false, "source video should now be visible");
  assert.equal(await page.$eval("#run-btn", (el) => el.disabled), false, "Run should be enabled once a file is staged");

  assertNoErrors(consoleErrors, pageErrors);
  await page.close();
});

test("dropping a non-video file shows the not_video error and leaves Run disabled", async () => {
  const { page, pageErrors } = await newPage();
  const badFile = path.join(os.tmpdir(), "heroextractor_e2e_not_a_video.txt");
  fs.writeFileSync(badFile, "hello");

  const input = await page.$("#file-input");
  await input.uploadFile(badFile);
  await sleep(200);

  assert.equal(await page.$eval("#error-banner", (el) => el.hidden), false, "expected the not_video error banner");
  const msg = await page.$eval("#error-message", (el) => el.textContent);
  assert.ok(!msg.startsWith("drop."), `error message showed a raw i18n key: ${msg}`);
  assert.equal(await page.$eval("#run-btn", (el) => el.disabled), true);

  assert.deepEqual(pageErrors, [], "unexpected page errors");
  await page.close();
});

test("running shows a live preview mid-flight, then the final result and a working Download link", async () => {
  const { page, consoleErrors, pageErrors } = await newPage();
  const input = await page.$("#file-input");
  await input.uploadFile(SLOW_CLIP); // fake_run_clip blocks ~2s for this filename -- see its own comment
  await sleep(150);
  await page.click("#run-btn");

  let sawLivePreview = false;
  for (let i = 0; i < 100; i++) {
    const hidden = await page.$eval("#out-preview", (el) => el.hidden);
    if (!hidden) {
      const src = await page.$eval("#out-preview", (el) => el.src);
      if (src.includes("/preview?")) { sawLivePreview = true; break; }
    }
    await sleep(50);
  }
  assert.ok(sawLivePreview, "expected the right pane to show a live /preview image while the clip was running");

  let done = false;
  for (let i = 0; i < 60 && !done; i++) {
    await sleep(200);
    done = !(await page.$eval("#download-btn", (el) => el.hidden));
  }
  assert.ok(done, "job did not reach done (Save never appeared)");

  const outSrc = await page.$eval("#out-preview", (el) => el.src);
  assert.ok(outSrc.includes("/files/"), `expected the final output to be served from /files/, got ${outSrc}`);
  assert.ok(!outSrc.includes("/preview?"), "right pane should show the real output, not the live preview, once done");

  const saveHref = await page.$eval("#download-btn", (el) => el.getAttribute("href"));
  assert.ok(saveHref.includes("download=1"), `expected the save link to request a download, got ${saveHref}`);
  // It lives in the RESULT pane's footer (bottom-right), not a shared dock.
  assert.ok(await page.$eval("#download-btn", (el) => !!el.closest("#side-out .pane-foot")));
  assert.equal(await page.$eval("#download-btn", (el) => el.textContent.trim()), "Download");
  const naturalWidth = await page.$eval("#out-preview", (el) => el.naturalWidth);
  assert.ok(naturalWidth > 0, "final GIF did not decode");

  assertNoErrors(consoleErrors, pageErrors);
  await page.close();
});

test("clicking the finished result restarts both panes together", async () => {
  // Regression/feature test: neither media element had any play/pause
  // affordance at all before this (both just autoplayed on loop). An
  // animated GIF (the default output) has no JS-exposed pause, so the
  // reachable equivalent is "clicking the result restarts both sides
  // together from frame 0" -- verified here by checking the source
  // video's currentTime resets and the result <img>'s src gets a fresh
  // cache-busting query (forcing the GIF to replay from its first frame).
  const { page, pageErrors } = await newPage();
  const input = await page.$("#file-input");
  await input.uploadFile(SAMPLE_CLIP);
  await sleep(150);
  await page.click("#run-btn");

  let done = false;
  for (let i = 0; i < 100 && !done; i++) {
    await sleep(100);
    done = !(await page.$eval("#download-btn", (el) => el.hidden));
  }
  assert.ok(done, "job did not reach done");

  // Let the source video play forward a bit so currentTime is non-zero,
  // then click the result and confirm both sides jumped back to the start.
  await page.evaluate(() => { document.getElementById("src-video").currentTime = 1.0; });
  const srcBefore = await page.$eval("#out-preview", (el) => el.getAttribute("src"));
  await page.click("#out-preview");
  await sleep(100);

  const currentTimeAfter = await page.$eval("#src-video", (el) => el.currentTime);
  assert.ok(currentTimeAfter < 0.5, `expected the source video to restart near 0, got ${currentTimeAfter}`);
  const srcAfter = await page.$eval("#out-preview", (el) => el.getAttribute("src"));
  assert.notEqual(srcAfter, srcBefore, "expected a fresh cache-busted src to force the GIF to replay from frame 0");

  assert.deepEqual(pageErrors, [], "unexpected page errors");
  await page.close();
});

test("cancel mid-run reaches cancelled and re-enables Run for the same file", async () => {
  const { page, consoleErrors, pageErrors } = await newPage();
  const input = await page.$("#file-input");
  await input.uploadFile(SLOW_CLIP);
  await sleep(150);
  await page.click("#run-btn");

  let sawCancelLabel = false;
  for (let i = 0; i < 100; i++) {
    const label = await page.$eval("#run-btn-label", (el) => el.textContent);
    if (label === "Cancel") { sawCancelLabel = true; break; }
    await sleep(20);
  }
  assert.ok(sawCancelLabel, "expected the Run button's label to become \"Cancel\" while a job is in flight");
  await page.click("#run-btn");

  let reEnabled = false;
  for (let i = 0; i < 100; i++) {
    if (!(await page.$eval("#run-btn", (el) => el.disabled))) { reEnabled = true; break; }
    await sleep(50);
  }
  assert.ok(reEnabled, "Run should re-enable (same staged file) once the job reaches a terminal state");
  const tone = await page.$eval("#sb-dot", (el) => el.dataset.tone);
  assert.equal(tone, "warn", "status dot should reflect the cancelled state");

  assertNoErrors(consoleErrors, pageErrors);
  await page.close();
});

/** Clicks Run and returns THIS test's own job id. Recorded against the id
 * of the most recent job BEFORE clicking: this suite doesn't reset
 * data/jobs/ between tests, so "the most recent job" can still be an
 * earlier test's job for a moment after clicking (confirmed flaky
 * otherwise). */
async function runAndGetJobId(page) {
  const priorJobId = await page.evaluate(async () => {
    const resp = await fetch("/api/jobs?limit=1");
    return (await resp.json()).jobs[0]?.id ?? null;
  });
  await page.click("#run-btn");
  let jobId = null;
  for (let i = 0; i < 50 && !jobId; i++) {
    await sleep(100);
    jobId = await page.evaluate(async (prior) => {
      const resp = await fetch("/api/jobs?limit=1");
      const id = (await resp.json()).jobs[0]?.id ?? null;
      return id && id !== prior ? id : null;
    }, priorJobId);
  }
  assert.ok(jobId, "this test's own job never appeared in the job list");
  return jobId;
}

async function jobConfig(page, jobId) {
  return page.evaluate(async (id) => (await (await fetch(`/api/jobs/${id}`)).json()).config, jobId);
}

test("Output format: WebP persists as encoder=webp and is sent with the next job; GIF removes the override", async () => {
  // Only pageErrors is checked: SAMPLE_CLIP can complete before the
  // <img>'s one preview() fetch lands (benign 404 race, see above).
  const { page, pageErrors } = await newPage();
  await page.click("#settings-btn");
  await sleep(100);
  await page.click('label[for="fmt-webp"]');
  await sleep(100);
  await page.keyboard.press("Escape");
  await sleep(100);
  assert.equal(JSON.parse(await page.evaluate(() => localStorage.getItem("hx.overrides"))).encoder, "webp");

  await (await page.$("#file-input")).uploadFile(SAMPLE_CLIP);
  await sleep(150);
  const jobId = await runAndGetJobId(page);
  assert.equal((await jobConfig(page, jobId)).encoder, "webp", "the chosen output format should reach the job");

  // Back to GIF: the backend default, so the override is REMOVED rather
  // than pinned to ss_alpha_gif.
  await page.click("#settings-btn");
  await sleep(100);
  assert.equal(await page.$eval("#fmt-webp", (el) => el.checked), true, "dialog should reflect the stored choice");
  await page.click('label[for="fmt-gif"]');
  await sleep(100);
  const stored = JSON.parse(await page.evaluate(() => localStorage.getItem("hx.overrides")));
  assert.equal(stored.encoder, undefined);

  assert.deepEqual(pageErrors, [], "unexpected page errors");
  await page.close();
});

test("the header GPU/CPU toggle persists, is sent with the job, and un-pins on a second click", async () => {
  const { page, pageErrors } = await newPage();
  await page.click('#device-toggle button[data-device="cpu"]');
  await sleep(100);
  assert.equal(
    await page.$eval('#device-toggle button[data-device="cpu"]', (el) => el.getAttribute("aria-pressed")), "true");
  assert.equal(
    await page.$eval('#device-toggle button[data-device="cuda"]', (el) => el.getAttribute("aria-pressed")), "false");
  assert.equal(JSON.parse(await page.evaluate(() => localStorage.getItem("hx.overrides"))).device, "cpu");

  await page.reload({ waitUntil: "networkidle0" });
  await sleep(300);
  assert.equal(
    await page.$eval('#device-toggle button[data-device="cpu"]', (el) => el.getAttribute("aria-pressed")), "true",
    "the choice should survive a reload");

  await (await page.$("#file-input")).uploadFile(SAMPLE_CLIP);
  await sleep(150);
  const jobId = await runAndGetJobId(page);
  assert.equal((await jobConfig(page, jobId)).device, "cpu", "the overridden device should have been sent with the job");

  // Clicking the already-pressed button clears the override -- back to
  // "auto" -- rather than toggling to the other device.
  await page.click('#device-toggle button[data-device="cpu"]');
  await sleep(100);
  const stored = JSON.parse(await page.evaluate(() => localStorage.getItem("hx.overrides")));
  assert.equal(stored.device, undefined, "clicking the pressed button should clear the override, not flip it");

  assert.deepEqual(pageErrors, [], "unexpected page errors");
  await page.close();
});

test("stale overrides from older builds are dropped on load (only device/encoder survive)", async () => {
  // An older build could store keyer/max_side/trimap/despill/QC choices
  // and an encoder value the UI no longer offers -- they'd keep silently
  // applying with no control left to see or undo them.
  const { page, pageErrors } = await newPage({
    overrides: JSON.stringify({ device: "cpu", encoder: "supersampled_gif", use_keyer: "off",
                                max_side: 512, use_trimap: true, apply_despill: false, qc: true }),
  });
  const stored = JSON.parse(await page.evaluate(() => localStorage.getItem("hx.overrides")));
  assert.deepEqual(stored, { device: "cpu" });
  assert.equal(await page.$eval("#fmt-gif", (el) => el.checked), true);

  await (await page.$("#file-input")).uploadFile(SAMPLE_CLIP);
  await sleep(150);
  const jobId = await runAndGetJobId(page);
  const config = await jobConfig(page, jobId);
  assert.equal(config.device, "cpu");
  assert.equal(config.use_keyer, "auto", "the dropped use_keyer override must not reach the job");
  assert.equal(config.encoder, "ss_alpha_gif");

  // A valid encoder is kept.
  await page.evaluate(() => localStorage.setItem("hx.overrides", JSON.stringify({ encoder: "mov", foo: 1 })));
  await page.reload({ waitUntil: "networkidle0" });
  await sleep(300);
  assert.deepEqual(JSON.parse(await page.evaluate(() => localStorage.getItem("hx.overrides"))), { encoder: "mov" });
  await page.click("#settings-btn");
  await sleep(100);
  assert.equal(await page.$eval("#fmt-mov", (el) => el.checked), true);

  assert.deepEqual(pageErrors, [], "unexpected page errors");
  await page.close();
});

test("a keyable flat-chroma clip is matted via the fast path (auto_keyer reaches the job record)", async () => {
  // See fake_server.py's fake_validate_upload/fake_run_clip: a filename
  // containing "keyable" simulates the real upload-time probe finding a
  // flat, saturated backdrop, and fake_run_clip writes key_s into timings
  // the same way a real keyer-path run_clip would -- server/jobs.py
  // derives clip["auto_keyer"] from exactly that presence.
  const KEYABLE_CLIP = path.join(os.tmpdir(), "heroextractor_e2e_keyable_clip.mp4");
  fs.copyFileSync(SAMPLE_CLIP, KEYABLE_CLIP);

  const { page, pageErrors } = await newPage();
  const priorJobId = await page.evaluate(async () => {
    const resp = await fetch("/api/jobs?limit=1");
    return (await resp.json()).jobs[0]?.id ?? null;
  });
  const input = await page.$("#file-input");
  await input.uploadFile(KEYABLE_CLIP);
  await sleep(150);
  await page.click("#run-btn");

  let jobId = null;
  for (let i = 0; i < 50 && !jobId; i++) {
    await sleep(100);
    jobId = await page.evaluate(async (prior) => {
      const resp = await fetch("/api/jobs?limit=1");
      const id = (await resp.json()).jobs[0]?.id ?? null;
      return id && id !== prior ? id : null;
    }, priorJobId);
  }
  assert.ok(jobId, "this test's own job never appeared in the job list");

  let autoKeyer = null;
  for (let i = 0; i < 50 && autoKeyer === null; i++) {
    await sleep(100);
    autoKeyer = await page.evaluate(async (id) => {
      const job = await (await fetch(`/api/jobs/${id}`)).json();
      return job.clips[0]?.auto_keyer ?? null;
    }, jobId);
  }
  assert.equal(autoKeyer, true);

  assert.deepEqual(pageErrors, [], "unexpected page errors");
  await page.close();
});

test("Escape closes the settings dialog and returns focus to its opener", async () => {
  const { page, consoleErrors, pageErrors } = await newPage();
  await page.click("#settings-btn");
  await sleep(150);
  assert.equal(await page.$eval("#settings-dialog", (el) => el.open), true);

  await page.keyboard.press("Escape");
  await sleep(150);
  assert.equal(await page.$eval("#settings-dialog", (el) => el.open), false);
  const activeId = await page.evaluate(() => document.activeElement && document.activeElement.id);
  assert.equal(activeId, "settings-btn");

  assertNoErrors(consoleErrors, pageErrors);
  await page.close();
});

test("SSE disconnect shows the network error banner, which clears once reconnected", async () => {
  // Regression test for audit A6: sse.mjs's onDisconnect hook and the
  // err.NETWORK locale key existed but app.mjs never wired them up, so a
  // dropped SSE connection left the progress bar silently stuck with no
  // indication anything was wrong.
  const { page, consoleErrors, pageErrors } = await newPage();

  // Block the EventSource connection before it's ever opened, so it fails
  // immediately (onerror) instead of succeeding and later dropping.
  await page.setRequestInterception(true);
  let blockEvents = true;
  const onRequest = (req) => {
    if (blockEvents && req.url().includes("/events")) req.abort("failed");
    else req.continue();
  };
  page.on("request", onRequest);

  const input = await page.$("#file-input");
  await input.uploadFile(SLOW_CLIP);
  await sleep(150);
  await page.click("#run-btn");

  let sawNetworkError = false;
  for (let i = 0; i < 100; i++) {
    const hidden = await page.$eval("#error-banner", (el) => el.hidden);
    if (!hidden) { sawNetworkError = true; break; }
    await sleep(50);
  }
  assert.ok(sawNetworkError, "expected the network-error banner once the SSE connection failed to open");
  const msg = await page.$eval("#error-message", (el) => el.textContent);
  assert.ok(!msg.startsWith("err."), `error message showed a raw i18n key: ${msg}`);

  // Let the next reconnect attempt through -- sse.mjs's backoff starts at
  // 1s, so give it a bit more than that before asserting recovery.
  blockEvents = false;
  let recovered = false;
  for (let i = 0; i < 100; i++) {
    if (await page.$eval("#error-banner", (el) => el.hidden)) { recovered = true; break; }
    await sleep(100);
  }
  assert.ok(recovered, "expected the network-error banner to clear once the SSE connection succeeded");

  page.off("request", onRequest);
  assert.deepEqual(pageErrors, [], "unexpected page errors");
  await page.close();
});

test("a failed job shows both the coded message and the underlying detail", async () => {
  // Regression test for the plan's B3 finding: job/clip.error.detail (the
  // actual ffmpeg/CUDA text) reached the browser's memory but nothing on
  // screen ever rendered it -- only the generic coded message showed.
  //
  // Only pageErrors is checked (not consoleErrors): FAIL_CLIP fails almost
  // instantly, and fake_run_clip fires its one preview() call before
  // raising -- the same known fake-server-speed race as the other tests
  // in this file that already use this pattern (confirmed flaky here
  // under load: a benign 404 for the by-then-superseded preview PNG).
  const { page, pageErrors } = await newPage();
  const input = await page.$("#file-input");
  await input.uploadFile(FAIL_CLIP);
  await sleep(150);
  await page.click("#run-btn");

  let sawDetail = false;
  for (let i = 0; i < 100; i++) {
    const hidden = await page.$eval("#error-detail", (el) => el.hidden);
    if (!hidden) { sawDetail = true; break; }
    await sleep(50);
  }
  assert.ok(sawDetail, "expected the error banner's detail line to appear once the job failed");

  const msg = await page.$eval("#error-message", (el) => el.textContent);
  assert.ok(!msg.startsWith("err."), `error message showed a raw i18n key: ${msg}`);
  const detail = await page.$eval("#error-detail", (el) => el.textContent);
  assert.ok(detail.includes("fake_server test failure"),
    `expected the sanitized underlying error text, got: ${detail}`);

  assert.deepEqual(pageErrors, [], "unexpected page errors");
  await page.close();
});

test("the header language switch localizes sentences but keeps buttons/labels in English", async () => {
  const { page, consoleErrors, pageErrors } = await newPage();
  const enHint = await page.$eval(".dz-title", (el) => el.textContent);
  assert.equal(await page.$eval('#lang-toggle button[data-lang="en"]', (el) => el.getAttribute("aria-pressed")), "true");

  await page.click('#lang-toggle button[data-lang="ja"]');
  await sleep(150);
  assert.equal(await page.evaluate(() => localStorage.getItem("hx.lang")), "ja");
  assert.equal(await page.$eval('#lang-toggle button[data-lang="ja"]', (el) => el.getAttribute("aria-pressed")), "true");
  assert.equal(await page.evaluate(() => document.documentElement.lang), "ja");
  const jaHint = await page.$eval(".dz-title", (el) => el.textContent);
  assert.notEqual(jaHint, enHint, "explanatory text should follow the language");
  assert.match(jaHint, /[぀-ヿ一-鿿]/, "expected Japanese text in the drop hint");

  // Buttons and labels: English in every language.
  const paneLabels = await page.$$eval(".pane-label", (els) => els.map((e) => e.textContent));
  assert.deepEqual(paneLabels, ["Source", "Result"]);
  assert.equal(await page.$eval("#run-btn-label", (el) => el.textContent), "Convert");
  assert.deepEqual(await page.$$eval("#device-toggle button", (els) => els.map((e) => e.textContent)), ["GPU", "CPU"]);
  assert.equal(await page.$eval("#clear-btn", (el) => el.textContent.trim()), "Clear");
  await page.click("#settings-btn");
  await sleep(150);
  assert.equal(await page.$eval("#settings-dialog h2", (el) => el.textContent), "Settings");
  assert.equal(await page.$eval("#format-group legend", (el) => el.textContent), "Output format");
  assert.match(await page.$eval('label[for="fmt-mov"] .fc-desc', (el) => el.textContent), /[぀-ヿ一-鿿]/,
    "format descriptions are sentences, so they follow the language");
  await page.keyboard.press("Escape");

  // zh too, then back to en for the rest of the suite's shared profile.
  await page.click('#lang-toggle button[data-lang="zh"]');
  await sleep(150);
  assert.equal(await page.$eval("#run-btn-label", (el) => el.textContent), "Convert");
  assert.equal(await page.evaluate(() => document.documentElement.lang), "zh");

  assertNoErrors(consoleErrors, pageErrors);
  await page.close();
});

/** Fires dragenter/dragover/drop carrying `names` as fake video Files on
 * the element matching `selector` -- the same events a real OS drag
 * produces (Chrome lets page script construct a DataTransfer). */
async function dropFiles(page, selector, names, { stopAfter = "drop" } = {}) {
  return page.evaluate((sel, fileNames, stop) => {
    const target = document.querySelector(sel);
    const dt = new DataTransfer();
    for (const n of fileNames) dt.items.add(new File([new Uint8Array(64)], n, { type: "video/mp4" }));
    const fire = (type) => target.dispatchEvent(new DragEvent(type, { dataTransfer: dt, bubbles: true, cancelable: true }));
    fire("dragenter");
    fire("dragover");
    if (stop === "dragover") return;
    fire("drop");
  }, selector, names, stopAfter);
}

test("dropping onto a showing video (anywhere on the source pane) replaces the staged video", async () => {
  const { page, pageErrors } = await newPage();
  await (await page.$("#file-input")).uploadFile(SAMPLE_CLIP);
  await sleep(200);
  const before = await page.$eval("#src-video", (el) => el.getAttribute("src"));
  assert.equal(await page.$eval("#src-meta", (el) => el.textContent), "Triceratops.mp4");

  // Drag feedback over the video itself.
  await dropFiles(page, "#src-video", ["replacement.mp4"], { stopAfter: "dragover" });
  await sleep(50);
  assert.equal(await page.$eval("#drop-overlay", (el) => el.hidden), false, "overlay should show while dragging");
  assert.equal(await page.$eval("#drop-overlay-text", (el) => el.textContent), "Drop to replace");
  await page.evaluate(() => document.getElementById("src-video").dispatchEvent(new DragEvent("dragleave", { bubbles: true })));
  await sleep(50);
  assert.equal(await page.$eval("#drop-overlay", (el) => el.hidden), true);

  await dropFiles(page, "#src-video", ["replacement.mp4"]);
  await sleep(200);
  const after = await page.$eval("#src-video", (el) => el.getAttribute("src"));
  assert.ok(after.startsWith("blob:") && after !== before, "the new drop should replace the video");
  assert.equal(await page.$eval("#src-meta", (el) => el.textContent), "replacement.mp4");
  assert.equal(await page.$eval("#drop-overlay", (el) => el.hidden), true);

  // After a finished job, a drop still replaces -- and clears the old result.
  await (await page.$("#file-input")).uploadFile(SAMPLE_CLIP);
  await sleep(150);
  await page.click("#run-btn");
  let done = false;
  for (let i = 0; i < 100 && !done; i++) {
    await sleep(100);
    done = !(await page.$eval("#download-btn", (el) => el.hidden));
  }
  assert.ok(done, "job did not reach done");
  await dropFiles(page, "#pane-src", ["another.mp4"]);
  await sleep(200);
  assert.equal(await page.$eval("#src-meta", (el) => el.textContent), "another.mp4");
  assert.equal(await page.$eval("#download-btn", (el) => el.hidden), true, "the old result must not linger");
  assert.equal(await page.$eval("#out-preview", (el) => el.hidden), true);
  assert.equal(await page.$eval("#run-btn", (el) => el.disabled), false);

  assert.deepEqual(pageErrors, [], "unexpected page errors");
  await page.close();
});

test("dropping while a job is running is refused with a visible message, not silently", async () => {
  const { page, pageErrors } = await newPage();
  await (await page.$("#file-input")).uploadFile(SLOW_CLIP);
  await sleep(150);
  await page.click("#run-btn");
  for (let i = 0; i < 100; i++) {
    if ((await page.$eval("#run-btn-label", (el) => el.textContent)) === "Cancel") break;
    await sleep(20);
  }
  const srcBefore = await page.$eval("#src-video", (el) => el.getAttribute("src"));

  await dropFiles(page, "#src-video", ["replacement.mp4"], { stopAfter: "dragover" });
  await sleep(50);
  assert.equal(await page.$eval("#drop-overlay", (el) => el.dataset.tone), "warn");
  assert.equal(await page.$eval("#drop-overlay-text", (el) => el.textContent), "Can't replace while processing");
  await dropFiles(page, "#src-video", ["replacement.mp4"]);
  await sleep(100);
  assert.equal(await page.$eval("#error-banner", (el) => el.hidden), false, "expected a refusal message");
  assert.equal(await page.$eval("#error-banner", (el) => el.dataset.tone), "warn");
  const msg = await page.$eval("#error-message", (el) => el.textContent);
  assert.ok(!msg.startsWith("drop.") && msg.length > 10, `expected a readable message, got ${msg}`);
  assert.equal(await page.$eval("#src-video", (el) => el.getAttribute("src")), srcBefore, "video must not change mid-run");
  assert.equal(await page.$eval("#clear-btn", (el) => el.disabled), true, "Clear is disabled while running");

  await page.click("#run-btn"); // cancel, so the next test isn't queued behind this job
  for (let i = 0; i < 100; i++) {
    if (!(await page.$eval("#run-btn", (el) => el.disabled))) break;
    await sleep(50);
  }
  assert.deepEqual(pageErrors, [], "unexpected page errors");
  await page.close();
});

test("Clear resets the source, the result and the job view back to the empty start screen", async () => {
  const { page, pageErrors } = await newPage();
  assert.equal(await page.$eval("#clear-btn", (el) => el.hidden), true, "nothing to clear yet");
  await (await page.$("#file-input")).uploadFile(SAMPLE_CLIP, SECOND_CLIP);
  await sleep(150);
  assert.equal(await page.$eval("#clear-btn", (el) => el.hidden), false);
  assert.ok(await page.$eval("#clear-btn", (el) => !!el.closest("#side-src .pane-foot")));
  await page.click("#run-btn");
  let zipShown = false;
  for (let i = 0; i < 150 && !zipShown; i++) {
    await sleep(100);
    zipShown = !(await page.$eval("#zip-btn", (el) => el.hidden));
  }
  assert.ok(zipShown, "2 done clips should show Download all (ZIP)");
  assert.equal(await page.$eval("#clear-btn", (el) => el.disabled), false);

  await page.click("#clear-btn");
  await sleep(150);
  assert.equal(await page.$eval("#src-video", (el) => el.hidden), true);
  assert.equal(await page.$eval("#src-video", (el) => el.hasAttribute("src")), false, "old blob src must be removed");
  assert.equal(await page.$eval("#drop-zone", (el) => el.hidden), false, "empty-state picker is back");
  assert.equal(await page.$eval("#out-preview", (el) => el.hidden), true);
  assert.equal(await page.$eval("#download-btn", (el) => el.hidden), true);
  assert.equal(await page.$eval("#zip-btn", (el) => el.hidden), true);
  assert.equal(await page.$eval("#src-strip", (el) => el.hidden), true);
  assert.equal(await page.$eval("#src-strip", (el) => el.children.length), 0);
  assert.equal(await page.$eval("#clear-btn", (el) => el.hidden), true);
  assert.equal(await page.$eval("#run-btn", (el) => el.disabled), true);
  assert.equal(await page.$eval("#error-banner", (el) => el.hidden), true);
  assert.equal(await page.$eval("#src-meta", (el) => el.textContent), "");

  assert.deepEqual(pageErrors, [], "unexpected page errors");
  await page.close();
});

test("the prepare stage shows a Preparing status with elapsed time and an indeterminate bar", async () => {
  const { page, pageErrors } = await newPage();
  await (await page.$("#file-input")).uploadFile(PREPARING_CLIP);
  await sleep(150);
  await page.click("#run-btn");

  let sawPreparing = false;
  for (let i = 0; i < 100 && !sawPreparing; i++) {
    const [hidden, text, indeterminate] = await page.evaluate(() => {
      const line = document.getElementById("status-line");
      return [line.hidden, line.textContent,
              document.getElementById("bar-track").classList.contains("is-indeterminate")];
    });
    if (!hidden && text === "Preparing…" && indeterminate) sawPreparing = true;
    else await sleep(30);
  }
  assert.ok(sawPreparing, "expected 'Preparing…' with an indeterminate bar during the prepare stage");
  let sawElapsed = false;
  for (let i = 0; i < 40 && !sawElapsed; i++) {
    const tl = await page.$eval("#time-line", (el) => (el.hidden ? "" : el.textContent));
    if (/\d:\d\d/.test(tl)) sawElapsed = true;
    else await sleep(50);
  }
  assert.ok(sawElapsed, "expected an elapsed clock while preparing");

  let done = false;
  for (let i = 0; i < 100 && !done; i++) {
    await sleep(100);
    done = !(await page.$eval("#download-btn", (el) => el.hidden));
  }
  assert.ok(done, "job did not reach done");
  assert.equal(await page.$eval("#status-line", (el) => el.hidden), true);
  assert.equal(await page.$eval("#bar-track", (el) => el.classList.contains("is-indeterminate")), false);

  assert.deepEqual(pageErrors, [], "unexpected page errors");
  await page.close();
});

test("CPU on a clip the keyer can't take shows a 'GPU recommended' note; a keyable clip doesn't", async () => {
  const { page, pageErrors } = await newPage({ overrides: { device: "cpu" } });
  await (await page.$("#file-input")).uploadFile(SLOW_CLIP); // probe: not keyable (see fake_validate_upload)
  await sleep(150);
  await page.click("#run-btn");
  let note = "";
  for (let i = 0; i < 100 && !note; i++) {
    note = await page.$eval("#cpu-note", (el) => (el.hidden ? "" : el.textContent));
    if (!note) await sleep(30);
  }
  assert.ok(note.includes("GPU"), `expected the CPU note, got: ${JSON.stringify(note)}`);
  assert.ok(!note.includes("{{"), "the ETA placeholder must be filled in");
  for (let i = 0; i < 100; i++) {
    if (!(await page.$eval("#run-btn", (el) => el.disabled)) &&
        (await page.$eval("#run-btn-label", (el) => el.textContent)) === "Convert") break;
    await sleep(50);
  }
  assert.equal(await page.$eval("#cpu-note", (el) => el.hidden), true, "note is only for an in-flight job");

  await (await page.$("#file-input")).uploadFile(KEYABLE_SLOW_CLIP);
  await sleep(150);
  await page.click("#run-btn");
  let sawRunning = false;
  for (let i = 0; i < 100 && !sawRunning; i++) {
    sawRunning = (await page.$eval("#run-btn-label", (el) => el.textContent)) === "Cancel";
    if (!sawRunning) await sleep(20);
  }
  assert.ok(sawRunning);
  await sleep(300);
  assert.equal(await page.$eval("#cpu-note", (el) => el.hidden), true, "the keyer path is as fast on CPU -- no note");

  assert.deepEqual(pageErrors, [], "unexpected page errors");
  await page.close();
});

// ---------------------------------------------------------------------
// Multi-clip UI (2026-09-13 plan: two-pane thumbnail strips + shared
// activeIndex). Only pageErrors is checked in most of these (not
// consoleErrors): fake_run_clip's one preview() call per clip can race
// the server's preview-file cleanup on a near-instantly-completing clip,
// producing a benign 404 in the console.
// ---------------------------------------------------------------------

test("dropping two files shows a thumbnail strip with one visible thumb per side", async () => {
  const { page, pageErrors } = await newPage();
  const input = await page.$("#file-input");
  await input.uploadFile(SAMPLE_CLIP, SECOND_CLIP);
  await sleep(200);

  assert.equal(await page.$eval("#src-strip", (el) => el.hidden), false, "strip should appear for 2+ files");
  assert.equal(await page.$eval("#out-strip", (el) => el.hidden), false);
  const visibleSrcThumbs = await page.$$eval("#src-strip .thumb", (els) => els.filter((el) => !el.hidden).length);
  assert.equal(visibleSrcThumbs, 1, "the active clip's own thumbnail should stay hidden (shown large instead)");
  const visibleOutThumbs = await page.$$eval("#out-strip .thumb", (els) => els.filter((el) => !el.hidden).length);
  assert.equal(visibleOutThumbs, 1);

  const srcVideoSrc = await page.$eval("#src-video", (el) => el.src);
  assert.ok(srcVideoSrc.startsWith("blob:"), "left pane should show the active file's object URL");

  assert.deepEqual(pageErrors, [], "unexpected page errors");
  await page.close();
});

test("clicking a thumbnail swaps the active clip on both sides", async () => {
  const { page, pageErrors } = await newPage();
  const input = await page.$("#file-input");
  await input.uploadFile(SAMPLE_CLIP, SECOND_CLIP);
  await sleep(200);

  const srcBefore = await page.$eval("#src-video", (el) => el.src);
  await page.click("#src-strip .thumb:not([hidden])");
  await sleep(150);
  const srcAfter = await page.$eval("#src-video", (el) => el.src);
  assert.notEqual(srcBefore, srcAfter, "the left pane's video should switch to the clicked clip's file");

  // The strip's visible/hidden thumb must swap in lockstep on BOTH sides
  // (one shared activeIndex) -- the previously-visible index is now
  // hidden, and vice versa, on src AND out.
  const [srcIdx, outIdx] = await Promise.all([
    page.$eval("#src-strip .thumb:not([hidden])", (el) => el.dataset.index),
    page.$eval("#out-strip .thumb:not([hidden])", (el) => el.dataset.index),
  ]);
  assert.equal(srcIdx, outIdx, "both strips must show the same clip as their (single) visible thumbnail");

  assert.deepEqual(pageErrors, [], "unexpected page errors");
  await page.close();
});

test("a pending clip's thumbnail shows nothing; its result appears once that clip finishes", async () => {
  const { page, pageErrors } = await newPage();
  const input = await page.$("#file-input");
  // SLOW_CLIP first so clip 1 is still "pending" while clip 0 runs.
  await input.uploadFile(SLOW_CLIP, SAMPLE_CLIP);
  await sleep(150);
  await page.click("#run-btn");
  await sleep(300); // clip 0 (SLOW_CLIP) should now be running, clip 1 pending

  // Switch to the still-pending second clip -- its out-pane should show
  // nothing (empty checker), not a stale preview or a broken image.
  await page.click("#out-strip .thumb:not([hidden])");
  await sleep(150);
  assert.equal(await page.$eval("#out-preview", (el) => el.hidden), true,
    "a pending clip has no output or preview to show yet");

  let bothDone = false;
  for (let i = 0; i < 100 && !bothDone; i++) {
    await sleep(100);
    bothDone = await page.evaluate(async () => {
      const resp = await fetch("/api/jobs?limit=1");
      const { jobs } = await resp.json();
      return jobs[0] && (jobs[0].status === "done" || jobs[0].status === "failed");
    });
  }
  assert.ok(bothDone, "job did not reach a terminal state");
  await sleep(200);

  // Now the second clip (already the active one) should show ITS result.
  const outSrc = await page.$eval("#out-preview", (el) => el.src);
  assert.ok(outSrc.includes("/files/"), `expected clip 1's own output, got ${outSrc}`);

  assert.deepEqual(pageErrors, [], "unexpected page errors");
  await page.close();
});

test("the time line shows elapsed/remaining while a clip is running", async () => {
  const { page, pageErrors } = await newPage();
  const input = await page.$("#file-input");
  await input.uploadFile(SLOW_CLIP);
  await sleep(150);
  await page.click("#run-btn");

  // Generous budget (matches this file's other job-completion polls, e.g.
  // "running shows a live preview..."): the very first progress tick
  // already carries elapsed_s (see server/jobs.py's _progress), so this
  // is normally near-instant, but SSE delivery on this shared, often
  // CPU-contended machine can lag well beyond a short poll window.
  let matched = false;
  for (let i = 0; i < 100; i++) {
    const hidden = await page.$eval("#time-line", (el) => el.hidden);
    if (!hidden) {
      const text = await page.$eval("#time-line", (el) => el.textContent);
      if (/\d:\d\d/.test(text)) { matched = true; break; }
    }
    await sleep(100);
  }
  assert.ok(matched, "expected the time line to show an elapsed duration like m:ss while running");

  assert.deepEqual(pageErrors, [], "unexpected page errors");
  await page.close();
});

test("Download targets the active clip and updates when the active clip changes", async () => {
  const { page, pageErrors } = await newPage();
  const input = await page.$("#file-input");
  await input.uploadFile(SAMPLE_CLIP, SECOND_CLIP);
  await sleep(150);
  await page.click("#run-btn");

  let done = false;
  for (let i = 0; i < 100 && !done; i++) {
    await sleep(100);
    done = !(await page.$eval("#download-btn", (el) => el.hidden));
  }
  assert.ok(done, "Save link never appeared for the active clip");

  const hrefBefore = await page.$eval("#download-btn", (el) => el.getAttribute("href"));
  assert.ok(hrefBefore.includes("download=1"));
  await page.click("#src-strip .thumb:not([hidden])"); // swap to the other clip
  await sleep(150);
  const hrefAfter = await page.$eval("#download-btn", (el) => el.getAttribute("href"));
  assert.notEqual(hrefBefore, hrefAfter, "Save's target should follow the active clip");

  assert.deepEqual(pageErrors, [], "unexpected page errors");
  await page.close();
});

test("the ZIP link only appears once 2+ clips in a job are done", async () => {
  const { page, pageErrors } = await newPage();

  const input1 = await page.$("#file-input");
  await input1.uploadFile(SAMPLE_CLIP);
  await sleep(150);
  await page.click("#run-btn");
  let done = false;
  for (let i = 0; i < 100 && !done; i++) {
    await sleep(50);
    done = !(await page.$eval("#download-btn", (el) => el.hidden));
  }
  assert.ok(done);
  assert.equal(await page.$eval("#zip-btn", (el) => el.hidden), true, "one clip alone must not show ZIP");

  await page.close();

  const { page: page2, pageErrors: pageErrors2 } = await newPage();
  const input2 = await page2.$("#file-input");
  await input2.uploadFile(SAMPLE_CLIP, SECOND_CLIP);
  await sleep(150);
  await page2.click("#run-btn");
  let zipShown = false;
  for (let i = 0; i < 150 && !zipShown; i++) {
    await sleep(100);
    zipShown = !(await page2.$eval("#zip-btn", (el) => el.hidden));
  }
  assert.ok(zipShown, "2 done clips should show ZIP");
  const zipHref = await page2.$eval("#zip-btn", (el) => el.getAttribute("href"));
  assert.ok(zipHref.endsWith("/zip"));
  assert.equal(await page2.$eval("#zip-btn", (el) => el.textContent.trim()), "Download all (ZIP)");
  assert.equal(await page2.$eval("#download-btn", (el) => el.hidden), false, "Download stays next to it");

  assert.deepEqual(pageErrors2, [], "unexpected page errors");
  await page2.close();
});

test("one failed clip alongside a successful one: thumbnail marks it, error banner is scoped to it", async () => {
  const { page, pageErrors } = await newPage();
  const input = await page.$("#file-input");
  await input.uploadFile(FAIL_CLIP, SAMPLE_CLIP);
  await sleep(150);
  await page.click("#run-btn");

  // Poll the DOM itself (driven by SSE), not the job-status API directly --
  // a fetch-based poll can observe the job as "done" server-side before
  // this page's own SSE-driven render() has caught up, a race that was
  // flaky here under load (fixed the same way for the ZIP-link test above).
  let bannerShown = false;
  for (let i = 0; i < 150 && !bannerShown; i++) {
    await sleep(100);
    bannerShown = !(await page.$eval("#error-banner", (el) => el.hidden));
  }
  assert.ok(bannerShown, "expected the active (failed) clip's error banner to appear");

  // The OTHER clip's own thumbnail (now visible, since clip 0 is active)
  // must end up marked "done", not "failed" -- polled: clip 1 only starts
  // once clip 0 has already failed (and the banner appeared).
  let otherStatus = "";
  for (let i = 0; i < 100 && otherStatus !== "done" && otherStatus !== "failed"; i++) {
    otherStatus = await page.$eval("#out-strip .thumb:not([hidden])", (el) => el.dataset.status);
    if (otherStatus !== "done" && otherStatus !== "failed") await sleep(50);
  }
  assert.equal(otherStatus, "done");

  // Switching to the succeeded clip must clear the error banner.
  await page.click("#out-strip .thumb:not([hidden])");
  await sleep(150);
  assert.equal(await page.$eval("#error-banner", (el) => el.hidden), true,
    "the error banner must not follow to a clip that didn't fail");

  assert.deepEqual(pageErrors, [], "unexpected page errors");
  await page.close();
});
