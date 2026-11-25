/**
 * Sheet Music Converter — Main UI Thread Controller.
 *
 * Keeps the UI thread lightweight and 60fps smooth:
 *   - Image decoding / PDF page rendering via pdfjs-dist onto <canvas>
 *   - Zero-copy Float32Array transfer to web/worker.js (ORT + mscz-rs WASM)
 *   - Live progress streaming, multi-page PDF stitching, and .mscz / .musicxml downloads
 */

const PDFJS_CDN_BASE = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@4.10.38/build/';
const SAMPLE_BEKERN =
  'clef-G-2 key+1 time-4-4 quarter G 4 quarter A 4 eighth B 4 eighth C 5 quarter D 5 bar ' +
  'half E 5 quarter D 5 quarter C 5 bar quarter B 4 quarter A 4 half G 4 bar-light-heavy';

// --- Crude Rollbar-like client error & console log forwarder (POST /__log) ---
const CLIENT_SESSION_ID = Math.random().toString(36).slice(2, 10);
function sendClientLog(level, msg, extra = null, src = 'ui') {
  try {
    const body = JSON.stringify({
      sid: CLIENT_SESSION_ID,
      level,
      src,
      msg: String(msg ?? '').slice(0, 4000),
      extra: extra ? JSON.parse(JSON.stringify(extra, (_, v) => (v instanceof Error ? { name: v.name, message: v.message, stack: v.stack } : v))) : undefined,
      href: location.href,
    });
    if (navigator.sendBeacon) {
      navigator.sendBeacon('/__log', new Blob([body], { type: 'application/json' }));
    } else {
      fetch('/__log', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body, keepalive: true }).catch(() => {});
    }
  } catch (_) {}
}
for (const lvl of ['error', 'warn']) {
  const orig = console[lvl].bind(console);
  console[lvl] = (...args) => {
    orig(...args);
    const text = args.map((a) => (a instanceof Error ? `${a.name}: ${a.message}\n${a.stack || ''}` : typeof a === 'object' ? (() => { try { return JSON.stringify(a); } catch { return String(a); } })() : String(a))).join(' ');
    sendClientLog(lvl, text);
  };
}
window.addEventListener('error', (ev) => {
  sendClientLog('error', ev.message || 'window.error', {
    filename: ev.filename,
    lineno: ev.lineno,
    colno: ev.colno,
    stack: ev.error?.stack,
  });
});
window.addEventListener('unhandledrejection', (ev) => {
  const r = ev.reason;
  sendClientLog('error', r instanceof Error ? `${r.name}: ${r.message}` : String(r), {
    type: 'unhandledrejection',
    stack: r?.stack,
  });
});
sendClientLog('info', 'client_boot', {
  ua: navigator.userAgent,
  isSecureContext: window.isSecureContext,
  crossOriginIsolated: window.crossOriginIsolated,
  hasSAB: typeof SharedArrayBuffer !== 'undefined',
  hasWebGPU: typeof navigator.gpu !== 'undefined',
});

// DOM refs
const els = {
  badgeBackend: document.getElementById('badge-backend'),
  badgeWasm: document.getElementById('badge-wasm'),
  badgeModels: document.getElementById('badge-models'),
  alertBanner: document.getElementById('alert-banner'),
  dropZone: document.getElementById('drop-zone'),
  dropPlaceholder: document.getElementById('drop-placeholder'),
  previewImg: document.getElementById('preview-img'),
  pdfThumbs: document.getElementById('pdf-thumbs'),
  imgMeta: document.getElementById('img-meta'),
  fileInput: document.getElementById('file-input'),
  cameraInput: document.getElementById('camera-input'),
  scoreTitle: document.getElementById('score-title'),
  btnConvert: document.getElementById('btn-convert'),
  btnCancel: document.getElementById('btn-cancel'),
  statusPill: document.getElementById('status-pill'),
  statusText: document.getElementById('status-text'),
  progressFill: document.getElementById('progress-fill'),
  progressTrack: document.querySelector('.progress-track'),
  statTokens: document.getElementById('stat-tokens'),
  statSystems: document.getElementById('stat-systems'),
  statSpeed: document.getElementById('stat-speed'),
  statElapsed: document.getElementById('stat-elapsed'),
  btnDownloadMscz: document.getElementById('btn-download-mscz'),
  btnDownloadXml: document.getElementById('btn-download-xml'),
  msczSize: document.getElementById('mscz-size'),
  xmlSize: document.getElementById('xml-size'),
  bekernDetails: document.getElementById('bekern-details'),
  bekernSummaryMeta: document.getElementById('bekern-summary-meta'),
  bekernText: document.getElementById('bekern-text'),
  btnRecompile: document.getElementById('btn-recompile'),
  btnCopyBekern: document.getElementById('btn-copy-bekern'),
  btnSampleBekern: document.getElementById('btn-sample-bekern'),
  workCanvas: document.getElementById('work-canvas'),
};

const state = {
  worker: null,
  nextRequestId: 1,
  pendingRequests: new Map(),
  wasmReady: false,
  modelsReady: false,
  backend: 'wasm',
  numThreads: 1,
  maxDim: 1280,
  maxTokens: 3072,
  inputKind: null, // 'image' | 'pdf'
  loadedImage: null,
  pdfDoc: null,
  pdfPageCount: 0,
  pdfThumbUrls: [],
  loadedFileStem: 'sheet-music',
  running: false,
  cancelRequested: false,
  lastMsczBytes: null,
  lastMusicXml: null,
  pdfjsLib: null,
  pdfjsPromise: null,
};

// Expose read-only result accessor for automated E2E verification
window.__smcGetResult = () => ({
  wasmReady: state.wasmReady,
  modelsReady: state.modelsReady,
  backend: state.backend,
  numThreads: state.numThreads,
  inputKind: state.inputKind,
  pdfPageCount: state.pdfPageCount,
  msczLength: state.lastMsczBytes ? state.lastMsczBytes.length : 0,
  msczMagic: state.lastMsczBytes && state.lastMsczBytes.length >= 4
    ? Array.from(state.lastMsczBytes.slice(0, 4))
    : [],
  musicXmlLength: state.lastMusicXml ? state.lastMusicXml.length : 0,
  musicXmlHead: state.lastMusicXml ? state.lastMusicXml.slice(0, 120) : '',
  bekern: els.bekernText.value,
});

function setBadge(el, text, kind = 'muted') {
  el.textContent = text;
  el.className = `badge badge-${kind}`;
}

function showAlert(msg, kind = 'warn') {
  if (!msg) {
    els.alertBanner.classList.add('hidden');
    return;
  }
  els.alertBanner.textContent = msg;
  els.alertBanner.className = `alert-banner${kind === 'error' ? ' alert-error' : kind === 'ok' ? ' alert-ok' : ''}`;
}

function setProgress(pct, statusText, pillText) {
  const clamped = Math.max(0, Math.min(100, pct));
  els.progressFill.style.width = `${clamped.toFixed(1)}%`;
  els.progressTrack.setAttribute('aria-valuenow', String(Math.round(clamped)));
  if (statusText !== undefined) els.statusText.textContent = statusText;
  if (pillText !== undefined) els.statusPill.textContent = pillText;
}

function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

function roundMult16(val, minVal, maxVal) {
  const rounded = Math.round(val / 16) * 16;
  return Math.max(minVal, Math.min(maxVal, rounded));
}

function updateConvertButtonState() {
  const hasInput =
    (state.inputKind === 'image' && Boolean(state.loadedImage)) ||
    (state.inputKind === 'pdf' && Boolean(state.pdfDoc) && state.pdfPageCount > 0);
  els.btnConvert.disabled = !(state.modelsReady && hasInput && !state.running);
}

// ---------------------------------------------------------------------------
// Web Worker lifecycle & messaging
// ---------------------------------------------------------------------------

function initWorker() {
  const worker = new Worker('./worker.js', { type: 'module' });
  state.worker = worker;

  worker.onerror = (err) => {
    console.error('Worker error:', err);
    showAlert(`Worker error: ${err.message || 'unexpected failure'}`, 'error');
  };

  worker.onmessage = (event) => {
    const msg = event.data;
    if (!msg || typeof msg.type !== 'string') return;

    switch (msg.type) {
      case 'wasm-ready':
        state.wasmReady = true;
        setBadge(els.badgeWasm, 'mscz-rs WASM: Ready', 'ok');
        break;

      case 'wasm-error':
        setBadge(els.badgeWasm, 'mscz-rs WASM: Error', 'err');
        showAlert(`Failed to initialize mscz-rs WebAssembly: ${msg.message}`, 'error');
        break;

      case 'ort-ready':
        state.backend = msg.backend;
        state.numThreads = msg.numThreads;
        setBadge(
          els.badgeBackend,
          msg.backend === 'webgpu'
            ? 'ORT: WebGPU (Worker)'
            : `ORT: WASM SIMD (${msg.numThreads}T)`,
          'ok',
        );
        setBadge(els.badgeModels, 'Models: Loading…', 'muted');
        setProgress(0, 'Loading UniOMR encoder & decoder ONNX models in worker…', 'Loading');
        break;

      case 'ort-error':
        setBadge(els.badgeBackend, 'ORT: Unavailable', 'err');
        showAlert(`ONNX Runtime Web could not be loaded: ${msg.message}`, 'error');
        break;

      case 'models-missing':
        setBadge(els.badgeModels, 'Models: Missing', 'warn');
        showAlert(
          'ONNX models not found in web/models/ yet (missing vocab.json). Export UniOMR ONNX weights to web/models/ and refresh, or click "Load Sample (Test WASM)" below to verify .mscz generation.',
          'warn',
        );
        break;

      case 'models-ready':
        state.backend = msg.backend;
        state.numThreads = msg.numThreads;
        state.maxDim = msg.maxDim || 1280;
        state.maxTokens = msg.maxTokens || 3072;
        state.modelsReady = true;
        setBadge(
          els.badgeBackend,
          state.backend === 'webgpu'
            ? 'ORT: WebGPU (Worker)'
            : `ORT: WASM SIMD (${state.numThreads}T)`,
          'ok',
        );
        setBadge(els.badgeModels, 'Models: Ready', 'ok');
        setProgress(0, 'Ready. Drop a sheet music image or PDF.', 'Ready');
        updateConvertButtonState();
        break;

      case 'models-error':
        setBadge(els.badgeModels, 'Models: Load Failed', 'err');
        showAlert(
          `Could not load UniOMR ONNX models from web/models/: ${msg.message}. You can still test .mscz compilation with "Load Sample (Test WASM)".`,
          'warn',
        );
        setProgress(0, 'Waiting for ONNX models in web/models/.', 'No Models');
        break;

      case 'progress': {
        const req = state.pendingRequests.get(msg.requestId);
        if (req && typeof req.onProgress === 'function') {
          req.onProgress(msg);
        }
        break;
      }

      case 'page-done':
      case 'compile-done': {
        const req = state.pendingRequests.get(msg.requestId);
        if (req) {
          state.pendingRequests.delete(msg.requestId);
          req.resolve(msg);
        }
        break;
      }

      case 'cancelled': {
        const req = state.pendingRequests.get(msg.requestId);
        if (req) {
          state.pendingRequests.delete(msg.requestId);
          req.resolve({ cancelled: true });
        }
        break;
      }

      case 'error':
      case 'compile-error': {
        const req = state.pendingRequests.get(msg.requestId);
        if (req) {
          state.pendingRequests.delete(msg.requestId);
          req.reject(new Error(msg.message || 'Worker operation failed'));
        }
        break;
      }

      default:
        break;
    }
  };

  worker.postMessage({ type: 'init' });
}

function workerRecognizePage(nchwFloat32, width, height, pageIndex, pageCount, onProgress) {
  const requestId = state.nextRequestId++;
  return new Promise((resolve, reject) => {
    state.pendingRequests.set(requestId, { resolve, reject, onProgress });
    state.worker.postMessage(
      {
        type: 'recognize-page',
        requestId,
        nchwBuffer: nchwFloat32.buffer,
        width,
        height,
        pageIndex,
        pageCount,
      },
      [nchwFloat32.buffer],
    );
  });
}

function workerCompileBekern(bekern, title) {
  const requestId = state.nextRequestId++;
  return new Promise((resolve, reject) => {
    state.pendingRequests.set(requestId, { resolve, reject });
    state.worker.postMessage({
      type: 'compile',
      requestId,
      bekern,
      title,
    });
  });
}

// ---------------------------------------------------------------------------
// Canvas preprocessing (main thread, ~15 ms) -> NCHW Float32Array [0, 1]
// ---------------------------------------------------------------------------

function canvasToNchw(canvas) {
  const targetW = canvas.width;
  const targetH = canvas.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const rgba = ctx.getImageData(0, 0, targetW, targetH).data;
  const hw = targetW * targetH;
  const nchw = new Float32Array(3 * hw);
  const inv255 = 1.0 / 255.0;

  for (let i = 0; i < hw; i++) {
    const p = i * 4;
    nchw[i] = rgba[p] * inv255;
    nchw[hw + i] = rgba[p + 1] * inv255;
    nchw[2 * hw + i] = rgba[p + 2] * inv255;
  }
  return { nchw, targetW, targetH };
}

function preprocessImageToNchw(img, maxDim) {
  const origW = img.naturalWidth || img.width;
  const origH = img.naturalHeight || img.height;
  const maxSide = Math.max(origW, origH, 1);
  const scale = Math.min(1.0, maxDim / maxSide);
  const targetW = roundMult16(origW * scale, 16, maxDim);
  const targetH = roundMult16(origH * scale, 16, maxDim);

  const canvas = els.workCanvas;
  canvas.width = targetW;
  canvas.height = targetH;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, targetW, targetH);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(img, 0, 0, targetW, targetH);

  const { nchw } = canvasToNchw(canvas);
  return { nchw, origW, origH, targetW, targetH };
}

// ---------------------------------------------------------------------------
// Lazy client-side PDF loader (pdfjs-dist v4 ESM + same-origin Blob module worker)
// ---------------------------------------------------------------------------

async function loadPdfJs() {
  if (state.pdfjsLib) return state.pdfjsLib;
  if (state.pdfjsPromise) return state.pdfjsPromise;

  state.pdfjsPromise = (async () => {
    const pdfjsLib = await import(`${PDFJS_CDN_BASE}pdf.min.mjs`);
    // Use a same-origin Blob module worker that imports the CDN worker script
    // so COEP: require-corp + same-origin Worker rules pass with zero console errors.
    const workerBlob = new Blob(
      [`import '${PDFJS_CDN_BASE}pdf.worker.min.mjs';`],
      { type: 'application/javascript' },
    );
    const workerUrl = URL.createObjectURL(workerBlob);
    const pdfWorker = new Worker(workerUrl, { type: 'module' });
    pdfjsLib.GlobalWorkerOptions.workerPort = pdfWorker;
    state.pdfjsLib = pdfjsLib;
    return pdfjsLib;
  })();

  try {
    return await state.pdfjsPromise;
  } catch (err) {
    state.pdfjsPromise = null;
    throw err;
  }
}

async function renderPdfPageToNchw(pdfDoc, pageNumber, maxDim) {
  const page = await pdfDoc.getPage(pageNumber);
  const baseVp = page.getViewport({ scale: 1.0 });
  const rawScale = maxDim / Math.max(baseVp.width, baseVp.height, 1);
  const targetW = roundMult16(baseVp.width * rawScale, 16, maxDim);
  const targetH = roundMult16(baseVp.height * rawScale, 16, maxDim);

  const canvas = els.workCanvas;
  canvas.width = targetW;
  canvas.height = targetH;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  // Fill white first so transparent PDFs never render black
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, targetW, targetH);

  const fitScale = Math.min(targetW / baseVp.width, targetH / baseVp.height);
  const vp = page.getViewport({ scale: fitScale });
  const offsetX = Math.floor((targetW - vp.width) / 2);
  const offsetY = Math.floor((targetH - vp.height) / 2);

  ctx.save();
  ctx.translate(offsetX, offsetY);
  await page.render({ canvasContext: ctx, viewport: vp }).promise;
  ctx.restore();

  const previewDataUrl = canvas.toDataURL('image/jpeg', 0.82);
  const { nchw } = canvasToNchw(canvas);
  return { nchw, targetW, targetH, previewDataUrl };
}

async function renderPdfThumbnails(pdfDoc) {
  els.pdfThumbs.innerHTML = '';
  state.pdfThumbUrls = [];
  const numPages = pdfDoc.numPages;

  const thumbCanvas = document.createElement('canvas');
  const thumbCtx = thumbCanvas.getContext('2d');

  for (let p = 1; p <= numPages; p++) {
    const page = await pdfDoc.getPage(p);
    const baseVp = page.getViewport({ scale: 1.0 });
    const scale = Math.min(180 / Math.max(baseVp.width, 1), 240 / Math.max(baseVp.height, 1));
    const vp = page.getViewport({ scale });
    thumbCanvas.width = Math.max(1, Math.round(vp.width));
    thumbCanvas.height = Math.max(1, Math.round(vp.height));
    thumbCtx.fillStyle = '#ffffff';
    thumbCtx.fillRect(0, 0, thumbCanvas.width, thumbCanvas.height);
    await page.render({ canvasContext: thumbCtx, viewport: vp }).promise;

    const url = thumbCanvas.toDataURL('image/jpeg', 0.8);
    state.pdfThumbUrls.push(url);

    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `pdf-thumb${p === 1 ? ' active' : ''}`;
    btn.dataset.page = String(p);
    btn.innerHTML = `<img src="${url}" alt="PDF page ${p}" /><span class="pdf-thumb-label">Page ${p}</span>`;
    btn.addEventListener('click', () => {
      setActivePdfThumb(p);
      els.previewImg.src = url;
    });
    els.pdfThumbs.appendChild(btn);
  }

  els.pdfThumbs.classList.remove('hidden');
  if (state.pdfThumbUrls[0]) {
    els.previewImg.src = state.pdfThumbUrls[0];
    els.previewImg.classList.remove('hidden');
    els.dropPlaceholder.classList.add('hidden');
  }
}

function setActivePdfThumb(pageNumber, markDoneUpTo = 0) {
  const buttons = els.pdfThumbs.querySelectorAll('.pdf-thumb');
  for (const btn of buttons) {
    const p = Number(btn.dataset.page);
    btn.classList.toggle('active', p === pageNumber);
    if (markDoneUpTo > 0) {
      btn.classList.toggle('done', p <= markDoneUpTo);
    }
  }
}

// ---------------------------------------------------------------------------
// Compilation & Downloads (via worker)
// ---------------------------------------------------------------------------

async function compileAndEnableDownloads(bekernString) {
  if (!state.wasmReady) {
    showAlert('mscz-rs WASM is not ready yet.', 'error');
    return false;
  }
  const title = (els.scoreTitle.value || state.loadedFileStem || 'Untitled Score').trim();
  const tokCount = bekernString.trim() ? bekernString.trim().split(/\s+/).length : 0;
  els.bekernSummaryMeta.textContent = `${tokCount} tokens`;

  try {
    const res = await workerCompileBekern(bekernString, title);
    const msczBytes = new Uint8Array(res.msczBuffer);
    const xmlStr = res.musicXml;

    state.lastMsczBytes = msczBytes;
    state.lastMusicXml = xmlStr;

    els.btnDownloadMscz.disabled = false;
    els.btnDownloadXml.disabled = false;
    els.msczSize.textContent = `Native MuseScore 4 archive · ${formatBytes(msczBytes.length)}`;
    els.xmlSize.textContent = `MusicXML 3.1 partwise · ${formatBytes(new TextEncoder().encode(xmlStr).length)}`;
    return true;
  } catch (err) {
    console.error('mscz-rs WASM compile failed:', err);
    state.lastMsczBytes = null;
    state.lastMusicXml = null;
    els.btnDownloadMscz.disabled = true;
    els.btnDownloadXml.disabled = true;
    showAlert(`mscz-rs compilation warning: ${err.message}`, 'error');
    return false;
  }
}

// ---------------------------------------------------------------------------
// Main conversion pipeline (Single Image OR Multi-Page PDF)
// ---------------------------------------------------------------------------

async function runConversion() {
  if (!state.modelsReady || state.running) return;
  if (state.inputKind === 'image' && !state.loadedImage) return;
  if (state.inputKind === 'pdf' && !state.pdfDoc) return;

  state.running = true;
  state.cancelRequested = false;
  showAlert('');
  els.btnConvert.disabled = true;
  els.btnCancel.classList.remove('hidden');
  els.btnDownloadMscz.disabled = true;
  els.btnDownloadXml.disabled = true;

  const t0 = performance.now();
  const maxDim = state.maxDim || 1280;
  const pageCount = state.inputKind === 'pdf' ? state.pdfPageCount : 1;

  // Smooth live elapsed clock on main thread while worker runs
  const elapsedTimer = setInterval(() => {
    const s = (performance.now() - t0) / 1000;
    els.statElapsed.textContent = `${s.toFixed(1)} s`;
  }, 100);

  try {
    const pageBekerns = [];
    let totalTokens = 0;
    let totalSystems = 0;
    let totalDecodeMs = 0;
    let anyLoopStopped = false;

    for (let pageIdx = 1; pageIdx <= pageCount; pageIdx++) {
      if (state.cancelRequested) break;

      const pagePrefix = pageCount > 1 ? `Page ${pageIdx}/${pageCount} — ` : '';
      const basePct = ((pageIdx - 1) / pageCount) * 92;
      const spanPct = 92 / pageCount;

      let nchw;
      let targetW;
      let targetH;

      if (state.inputKind === 'pdf') {
        setActivePdfThumb(pageIdx, pageIdx - 1);
        setProgress(
          basePct + spanPct * 0.05,
          `${pagePrefix}Rendering PDF page (max_dim=${maxDim})…`,
          pageCount > 1 ? `Page ${pageIdx}/${pageCount}` : 'Rendering',
        );
        const rendered = await renderPdfPageToNchw(state.pdfDoc, pageIdx, maxDim);
        nchw = rendered.nchw;
        targetW = rendered.targetW;
        targetH = rendered.targetH;
        els.previewImg.src = rendered.previewDataUrl;
        els.imgMeta.textContent = `PDF · Page ${pageIdx}/${pageCount} (${targetW}×${targetH})`;
        els.imgMeta.classList.remove('hidden');
      } else {
        setProgress(6, `Resizing image (max_dim=${maxDim})…`, 'Preprocessing');
        const prep = preprocessImageToNchw(state.loadedImage, maxDim);
        nchw = prep.nchw;
        targetW = prep.targetW;
        targetH = prep.targetH;
        els.imgMeta.textContent = `${prep.origW}×${prep.origH} → ${targetW}×${targetH}`;
        els.imgMeta.classList.remove('hidden');
      }

      if (state.cancelRequested) break;

      const pageRes = await workerRecognizePage(
        nchw,
        targetW,
        targetH,
        pageIdx,
        pageCount,
        (prog) => {
          const overallPct = basePct + (prog.pct / 100) * spanPct;
          els.statTokens.textContent = String(totalTokens + (prog.tokens || 0));
          els.statSystems.textContent = String(totalSystems + (prog.systems || 0));
          if (prog.tokPerSec > 0) {
            els.statSpeed.textContent = `${prog.tokPerSec.toFixed(1)} tok/s`;
          }
          if (prog.stage === 'encoding') {
            setProgress(
              overallPct,
              `${pagePrefix}Encoding (${state.backend.toUpperCase()} worker)…`,
              pageCount > 1 ? `Page ${pageIdx}/${pageCount}` : 'Encoding',
            );
          } else {
            setProgress(
              overallPct,
              `${pagePrefix}Decoding system ${prog.systems} (${prog.tokens} tokens)…`,
              pageCount > 1 ? `Page ${pageIdx}/${pageCount}` : 'Decoding',
            );
          }
        },
      );

      if (pageRes.cancelled || state.cancelRequested) {
        setProgress(0, 'Cancelled by user.', 'Cancelled');
        return;
      }

      if (pageRes.bekern && pageRes.bekern.trim()) {
        pageBekerns.push(pageRes.bekern.trim());
      }
      totalTokens += pageRes.tokens || 0;
      totalSystems += pageRes.systems || 0;
      totalDecodeMs += pageRes.decodeMs || 0;
      if (pageRes.stoppedByLoop) anyLoopStopped = true;

      if (state.inputKind === 'pdf') {
        setActivePdfThumb(pageIdx, pageIdx);
      }
    }

    if (state.cancelRequested) {
      setProgress(0, 'Cancelled by user.', 'Cancelled');
      return;
    }

    // Stitch multi-page bekern+ outputs cleanly with <sys> between pages
    const fullBekern = pageBekerns.join(' <sys> ');
    els.bekernText.value = fullBekern;
    els.statTokens.textContent = String(totalTokens);
    els.statSystems.textContent = String(totalSystems);

    setProgress(95, 'Compiling bekern+ to .mscz & .musicxml via mscz-rs WASM…', 'Compiling');
    const ok = await compileAndEnableDownloads(fullBekern);

    const totalS = (performance.now() - t0) / 1000;
    const decS = Math.max(0.001, totalDecodeMs / 1000);
    els.statElapsed.textContent = `${totalS.toFixed(1)} s`;
    els.statSpeed.textContent = `${(totalTokens / decS).toFixed(1)} tok/s`;

    const pageNote = pageCount > 1 ? ` across ${pageCount} PDF pages` : '';
    if (ok) {
      setProgress(
        100,
        anyLoopStopped
          ? `Done in ${totalS.toFixed(1)} s${pageNote} (trimmed tail repetition loop). Ready to download.`
          : `Done in ${totalS.toFixed(1)} s — ${totalTokens} tokens across ${totalSystems} system(s)${pageNote}.`,
        'Done',
      );
    } else {
      setProgress(100, 'Decoding finished, but mscz-rs reported a compile error (edit tokens below).', 'Warning');
    }
  } catch (err) {
    console.error('Inference failed:', err);
    showAlert(`Conversion failed: ${err.message}`, 'error');
    setProgress(0, 'Error during conversion.', 'Error');
  } finally {
    clearInterval(elapsedTimer);
    state.running = false;
    els.btnCancel.classList.add('hidden');
    updateConvertButtonState();
  }
}

// ---------------------------------------------------------------------------
// File input handling (Images + PDFs)
// ---------------------------------------------------------------------------

function isPdfFile(file) {
  return (
    file.type === 'application/pdf' ||
    file.name.toLowerCase().endsWith('.pdf')
  );
}

async function handlePdfFile(file) {
  showAlert('');
  const stem = file.name.replace(/\.[^.]+$/, '') || 'sheet-music';
  state.loadedFileStem = stem;
  if (!els.scoreTitle.value.trim()) {
    els.scoreTitle.value = stem.replace(/[_-]+/g, ' ');
  }

  state.inputKind = 'pdf';
  state.loadedImage = null;
  state.pdfDoc = null;
  state.pdfPageCount = 0;
  updateConvertButtonState();

  try {
    setProgress(0, 'Loading PDF renderer (pdf.js)…', 'Loading PDF');
    const pdfjsLib = await loadPdfJs();
    const arrayBuffer = await file.arrayBuffer();
    const loadingTask = pdfjsLib.getDocument({ data: new Uint8Array(arrayBuffer) });
    const pdfDoc = await loadingTask.promise;

    state.pdfDoc = pdfDoc;
    state.pdfPageCount = pdfDoc.numPages;

    els.imgMeta.textContent = `PDF · ${pdfDoc.numPages} page${pdfDoc.numPages === 1 ? '' : 's'}`;
    els.imgMeta.classList.remove('hidden');

    await renderPdfThumbnails(pdfDoc);
    updateConvertButtonState();
    if (state.modelsReady) {
      setProgress(
        0,
        `PDF loaded (${pdfDoc.numPages} page${pdfDoc.numPages === 1 ? '' : 's'}). Click "Convert to MuseScore" to begin.`,
        'Ready',
      );
    }
  } catch (err) {
    console.error('PDF load failed:', err);
    state.inputKind = null;
    state.pdfDoc = null;
    state.pdfPageCount = 0;
    els.pdfThumbs.classList.add('hidden');
    showAlert(`Failed to read PDF: ${err.message}`, 'error');
    setProgress(0, 'Could not open PDF file.', 'Error');
    updateConvertButtonState();
  }
}

function handleImageFile(file) {
  showAlert('');
  const stem = file.name.replace(/\.[^.]+$/, '') || 'sheet-music';
  state.loadedFileStem = stem;
  if (!els.scoreTitle.value.trim()) {
    els.scoreTitle.value = stem.replace(/[_-]+/g, ' ');
  }

  state.pdfDoc = null;
  state.pdfPageCount = 0;
  els.pdfThumbs.classList.add('hidden');
  els.pdfThumbs.innerHTML = '';

  const reader = new FileReader();
  reader.onload = () => {
    const img = new Image();
    img.onload = () => {
      state.inputKind = 'image';
      state.loadedImage = img;
      els.previewImg.src = reader.result;
      els.previewImg.classList.remove('hidden');
      els.dropPlaceholder.classList.add('hidden');
      els.imgMeta.textContent = `${img.naturalWidth}×${img.naturalHeight}`;
      els.imgMeta.classList.remove('hidden');
      updateConvertButtonState();
      if (state.modelsReady) {
        setProgress(0, 'Image loaded. Click "Convert to MuseScore" to begin.', 'Ready');
      }
    };
    img.src = reader.result;
  };
  reader.readAsDataURL(file);
}

function handleSelectedFile(file) {
  if (!file) return;
  if (isPdfFile(file)) {
    handlePdfFile(file);
  } else if (file.type.startsWith('image/')) {
    handleImageFile(file);
  } else {
    showAlert('Please select a valid image (PNG, JPEG, WebP) or PDF file.', 'warn');
  }
}

function triggerDownload(data, filename, mimeType) {
  const blob = new Blob([data], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---------------------------------------------------------------------------
// Event bindings
// ---------------------------------------------------------------------------

els.dropZone.addEventListener('click', () => els.fileInput.click());
els.dropZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    els.fileInput.click();
  }
});
els.dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  els.dropZone.classList.add('drag-over');
});
els.dropZone.addEventListener('dragleave', () => {
  els.dropZone.classList.remove('drag-over');
});
els.dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  els.dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files?.length) {
    handleSelectedFile(e.dataTransfer.files[0]);
  }
});

els.fileInput.addEventListener('change', (e) => {
  if (e.target.files?.length) handleSelectedFile(e.target.files[0]);
});
els.cameraInput.addEventListener('change', (e) => {
  if (e.target.files?.length) handleSelectedFile(e.target.files[0]);
});

els.btnConvert.addEventListener('click', runConversion);
els.btnCancel.addEventListener('click', () => {
  state.cancelRequested = true;
  if (state.worker) {
    state.worker.postMessage({ type: 'cancel' });
  }
});

els.btnDownloadMscz.addEventListener('click', () => {
  if (!state.lastMsczBytes) return;
  const stem = (state.loadedFileStem || 'score').replace(/[^a-zA-Z0-9._-]+/g, '_');
  triggerDownload(state.lastMsczBytes, `${stem}.mscz`, 'application/octet-stream');
});

els.btnDownloadXml.addEventListener('click', () => {
  if (!state.lastMusicXml) return;
  const stem = (state.loadedFileStem || 'score').replace(/[^a-zA-Z0-9._-]+/g, '_');
  triggerDownload(state.lastMusicXml, `${stem}.musicxml`, 'application/vnd.recordare.musicxml+xml');
});

els.btnRecompile.addEventListener('click', async () => {
  showAlert('');
  const txt = els.bekernText.value.trim();
  if (!txt) {
    showAlert('Enter or generate bekern+ tokens first.', 'warn');
    return;
  }
  if (await compileAndEnableDownloads(txt)) {
    setProgress(100, 'Recompiled .mscz and .musicxml in worker via mscz-rs WASM.', 'Compiled');
  }
});

els.btnCopyBekern.addEventListener('click', async () => {
  if (!els.bekernText.value) return;
  try {
    await navigator.clipboard.writeText(els.bekernText.value);
    showAlert('Copied bekern+ tokens to clipboard.', 'ok');
    setTimeout(() => showAlert(''), 2000);
  } catch {
    els.bekernText.select();
    document.execCommand('copy');
  }
});

els.btnSampleBekern.addEventListener('click', async () => {
  showAlert('');
  els.bekernDetails.open = true;
  els.bekernText.value = SAMPLE_BEKERN;
  if (!els.scoreTitle.value.trim()) {
    els.scoreTitle.value = 'WASM Sample Melody';
  }
  const toks = SAMPLE_BEKERN.split(/\s+/);
  els.statTokens.textContent = String(toks.length);
  els.statSystems.textContent = '1';
  if (await compileAndEnableDownloads(SAMPLE_BEKERN)) {
    setProgress(100, 'Compiled sample score via mscz-rs WASM worker — ready to download .mscz / .musicxml.', 'WASM Ready');
  }
});

// Boot
initWorker();
