import {
  initMsczWasm,
  bekernToMscz,
  bekernToMusicXml,
} from './wasm/mscz_wasm.js';

const ORT_CDN_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
const SAMPLE_BEKERN =
  'clef-G-2 key+1 time-4-4 quarter G 4 quarter A 4 eighth B 4 eighth C 5 quarter D 5 bar ' +
  'half E 5 quarter D 5 quarter C 5 bar quarter B 4 quarter A 4 half G 4 bar-light-heavy';

// DOM refs
const els = {
  badgeBackend: document.getElementById('badge-backend'),
  badgeWasm: document.getElementById('badge-wasm'),
  badgeModels: document.getElementById('badge-models'),
  alertBanner: document.getElementById('alert-banner'),
  dropZone: document.getElementById('drop-zone'),
  dropPlaceholder: document.getElementById('drop-placeholder'),
  previewImg: document.getElementById('preview-img'),
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
  wasmReady: false,
  modelsReady: false,
  backend: 'wasm',
  vocab: null,
  modelIo: null,
  encoderSession: null,
  decoderSession: null,
  loadedImage: null,
  loadedFileStem: 'sheet-music',
  running: false,
  cancelRequested: false,
  lastMsczBytes: null,
  lastMusicXml: null,
};

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

function normalizeVocab(raw) {
  let idToToken = [];
  if (Array.isArray(raw)) {
    idToToken = raw.slice();
  } else if (Array.isArray(raw.vocab)) {
    idToToken = raw.vocab.slice();
  } else if (Array.isArray(raw.id_to_token)) {
    idToToken = raw.id_to_token.slice();
  } else if (raw.id_to_token && typeof raw.id_to_token === 'object') {
    const entries = Object.entries(raw.id_to_token).map(([k, v]) => [Number(k), v]);
    const maxId = entries.reduce((m, [k]) => Math.max(m, k), -1);
    idToToken = new Array(maxId + 1).fill('<pad>');
    for (const [k, v] of entries) idToToken[k] = v;
  } else if (Array.isArray(raw.tokens)) {
    idToToken = raw.tokens.slice();
  } else {
    throw new Error('Unrecognized vocab.json format');
  }

  const tokenToId = {};
  for (let i = 0; i < idToToken.length; i++) {
    tokenToId[idToToken[i]] = i;
  }
  const padId = raw.pad_id ?? tokenToId['<pad>'] ?? 233;
  const bosId = raw.bos_id ?? tokenToId['<bos>'] ?? 234;
  const eosId = raw.eos_id ?? tokenToId['<eos>'] ?? 235;
  const sysId = raw.sys_id ?? tokenToId['<sys>'] ?? 236;

  return {
    idToToken,
    tokenToId,
    vocabSize: raw.vocab_size ?? idToToken.length,
    padId,
    bosId,
    eosId,
    sysId,
    maxDim: raw.max_dim ?? 1280,
    nDecLayers: raw.n_dec_layers ?? 6,
    nHeads: raw.n_heads ?? 6,
    headDim: raw.head_dim ?? 64,
  };
}

async function fetchJsonOptional(url) {
  try {
    const resp = await fetch(url, { cache: 'no-cache' });
    if (!resp.ok) return null;
    return await resp.json();
  } catch {
    return null;
  }
}

async function waitForOrt(timeoutMs = 15000) {
  const start = performance.now();
  while (typeof window.ort === 'undefined') {
    if (performance.now() - start > timeoutMs) {
      throw new Error('onnxruntime-web failed to load from CDN');
    }
    await new Promise((r) => setTimeout(r, 50));
  }
  return window.ort;
}

async function detectWebGpu() {
  if (!('gpu' in navigator)) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    return Boolean(adapter);
  } catch {
    return false;
  }
}

async function createOrtSession(ort, url, preferWebGpu) {
  const providers = preferWebGpu ? ['webgpu', 'wasm'] : ['wasm'];
  try {
    const session = await ort.InferenceSession.create(url, {
      executionProviders: providers,
      graphOptimizationLevel: 'all',
    });
    return { session, usedWebGpu: preferWebGpu };
  } catch (err) {
    if (!preferWebGpu) throw err;
    console.warn(`WebGPU session creation failed for ${url}, falling back to WASM:`, err);
    const session = await ort.InferenceSession.create(url, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    return { session, usedWebGpu: false };
  }
}

async function initRuntime() {
  // 1. Load mscz-rs WASM immediately (fast, ~300 KB)
  try {
    await initMsczWasm();
    state.wasmReady = true;
    setBadge(els.badgeWasm, 'mscz-rs WASM: Ready', 'ok');
  } catch (err) {
    console.error('mscz-rs WASM init failed:', err);
    setBadge(els.badgeWasm, 'mscz-rs WASM: Error', 'err');
    showAlert(`Failed to initialize mscz-rs WebAssembly: ${err.message}`, 'error');
  }

  // 2. Initialize ONNX Runtime Web + check WebGPU availability
  let ort;
  try {
    ort = await waitForOrt();
    ort.env.wasm.wasmPaths = ORT_CDN_BASE;
    ort.env.wasm.numThreads =
      typeof SharedArrayBuffer !== 'undefined' && navigator.hardwareConcurrency
        ? Math.min(4, navigator.hardwareConcurrency)
        : 1;
    const hasWebGpu = await detectWebGpu();
    state.backend = hasWebGpu ? 'webgpu' : 'wasm';
    setBadge(
      els.badgeBackend,
      hasWebGpu ? 'ORT: WebGPU' : `ORT: WASM (${ort.env.wasm.numThreads}T)`,
      hasWebGpu ? 'ok' : 'warn',
    );
  } catch (err) {
    console.error('ONNX Runtime Web failed:', err);
    setBadge(els.badgeBackend, 'ORT: Unavailable', 'err');
    showAlert(`ONNX Runtime Web could not be loaded: ${err.message}`, 'error');
    return;
  }

  // 3. Load vocab.json, optional model_io.json, and ONNX encoder/decoder
  setBadge(els.badgeModels, 'Models: Loading…', 'muted');
  try {
    const vocabRaw = await fetchJsonOptional('./models/vocab.json');
    if (!vocabRaw) {
      setBadge(els.badgeModels, 'Models: Missing', 'warn');
      showAlert(
        'ONNX models not found in web/models/ yet (missing vocab.json). Export UniOMR ONNX weights to web/models/ and refresh, or click "Load Sample (Test WASM)" below to verify .mscz generation.',
        'warn',
      );
      return;
    }
    state.vocab = normalizeVocab(vocabRaw);
    state.modelIo = vocabRaw.model_io_file
      ? (await fetchJsonOptional(`./models/${vocabRaw.model_io_file}`)) || {}
      : vocabRaw;

    const encoderUrl = state.modelIo.encoder_file
      ? `./models/${state.modelIo.encoder_file}`
      : './models/uni_omr_encoder.onnx';
    const decoderUrl = state.modelIo.decoder_file
      ? `./models/${state.modelIo.decoder_file}`
      : './models/uni_omr_decoder_step.onnx';

    const preferWebGpu = state.backend === 'webgpu';
    setProgress(0, 'Loading UniOMR encoder & decoder ONNX models…', 'Loading');

    const encRes = await createOrtSession(ort, encoderUrl, preferWebGpu);
    const decRes = await createOrtSession(ort, decoderUrl, encRes.usedWebGpu);

    state.encoderSession = encRes.session;
    state.decoderSession = decRes.session;
    state.backend = encRes.usedWebGpu && decRes.usedWebGpu ? 'webgpu' : 'wasm';
    state.modelsReady = true;

    setBadge(
      els.badgeBackend,
      state.backend === 'webgpu' ? 'ORT: WebGPU' : `ORT: WASM (${ort.env.wasm.numThreads}T)`,
      state.backend === 'webgpu' ? 'ok' : 'warn',
    );
    setBadge(els.badgeModels, 'Models: Ready', 'ok');
    setProgress(0, 'Ready. Drop or photograph a sheet music page.', 'Ready');
    updateConvertButtonState();
  } catch (err) {
    console.error('Model loading failed:', err);
    setBadge(els.badgeModels, 'Models: Load Failed', 'err');
    showAlert(
      `Could not load UniOMR ONNX models from web/models/: ${err.message}. You can still test .mscz compilation with "Load Sample (Test WASM)".`,
      'warn',
    );
    setProgress(0, 'Waiting for ONNX models in web/models/.', 'No Models');
  }
}

function updateConvertButtonState() {
  els.btnConvert.disabled = !(state.modelsReady && state.loadedImage && !state.running);
}

function preprocessImageToTensor(img, maxDim) {
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

  const tensor = new window.ort.Tensor('float32', nchw, [1, 3, targetH, targetW]);
  return { tensor, origW, origH, targetW, targetH };
}

function detectRepetitionLoop(tokenIds) {
  const n = tokenIds.length;
  // Check identical token run >= 24
  if (n >= 24) {
    const last = tokenIds[n - 1];
    let run = 1;
    while (run < 24 && tokenIds[n - 1 - run] === last) run++;
    if (run >= 24) return { detected: true, period: 1, cutAt: n - run };
  }
  // Check n-gram period p in 2..32 repeated 4 times
  const maxP = Math.min(32, Math.floor(n / 4));
  for (let p = 2; p <= maxP; p++) {
    let match = true;
    for (let k = 0; k < p * 3; k++) {
      if (tokenIds[n - 1 - k] !== tokenIds[n - 1 - k - p]) {
        match = false;
        break;
      }
    }
    if (match) {
      return { detected: true, period: p, cutAt: n - p * 3 };
    }
  }
  return { detected: false };
}

function trimToCleanBoundary(tokens) {
  // Cut back to the last barline or <sys> if we stopped abnormally
  for (let i = tokens.length - 1; i >= 0; i--) {
    const t = tokens[i];
    if (t === 'bar' || t.startsWith('bar-') || t === '<sys>') {
      return tokens.slice(0, i + 1);
    }
  }
  return tokens;
}

function argmaxLastSlice(logitsData, vocabSize) {
  const offset = Math.max(0, logitsData.length - vocabSize);
  let bestIdx = 0;
  let bestVal = -Infinity;
  for (let i = 0; i < vocabSize; i++) {
    const v = logitsData[offset + i];
    if (v > bestVal) {
      bestVal = v;
      bestIdx = i;
    }
  }
  return bestIdx;
}

function compileAndEnableDownloads(bekernString) {
  if (!state.wasmReady) {
    showAlert('mscz-rs WASM is not ready yet.', 'error');
    return false;
  }
  const title = (els.scoreTitle.value || state.loadedFileStem || 'Untitled Score').trim();
  const tokCount = bekernString.trim() ? bekernString.trim().split(/\s+/).length : 0;
  els.bekernSummaryMeta.textContent = `${tokCount} tokens`;

  try {
    const msczBytes = bekernToMscz(bekernString, title);
    const xmlStr = bekernToMusicXml(bekernString, title);

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

async function runConversion() {
  if (!state.modelsReady || !state.loadedImage || state.running) return;

  state.running = true;
  state.cancelRequested = false;
  showAlert('');
  els.btnConvert.disabled = true;
  els.btnCancel.classList.remove('hidden');
  els.btnDownloadMscz.disabled = true;
  els.btnDownloadXml.disabled = true;

  const t0 = performance.now();
  const maxDim = state.vocab.maxDim || state.modelIo.max_dim || 1280;
  const maxTokens = state.modelIo.max_tokens || 2048;
  const nLayers = state.vocab.nDecLayers || state.modelIo.n_dec_layers || 6;
  const nHeads = state.vocab.nHeads || state.modelIo.n_heads || 6;
  const headDim = state.vocab.headDim || state.modelIo.head_dim || 64;

  try {
    // 1. Preprocess image on canvas: preserve aspect ratio, max_dim=1280, mult of 16, NCHW [0,1]
    setProgress(6, `Resizing image (max_dim=${maxDim})…`, 'Preprocessing');
    const { tensor: imgTensor, origW, origH, targetW, targetH } = preprocessImageToTensor(
      state.loadedImage,
      maxDim,
    );
    els.imgMeta.textContent = `${origW}×${origH} → ${targetW}×${targetH}`;
    els.imgMeta.classList.remove('hidden');

    // 2. Run Encoder once -> produces precomputed cross_k_0..5, cross_v_0..5
    setProgress(15, `Running UniOMR encoder (${state.backend.toUpperCase()})…`, 'Encoding');
    await new Promise((r) => setTimeout(r, 0));

    const encOutputs = await state.encoderSession.run({ image: imgTensor });
    if (typeof imgTensor.dispose === 'function') imgTensor.dispose();

    const encElapsed = (performance.now() - t0) / 1000;
    els.statElapsed.textContent = `${encElapsed.toFixed(1)} s`;

    // 3. KV-cached greedy autoregressive decode
    const { bosId, eosId, sysId, padId, idToToken, vocabSize } = state.vocab;

    let pastK = [];
    let pastV = [];
    for (let i = 0; i < nLayers; i++) {
      pastK.push(
        new window.ort.Tensor('float32', new Float32Array(0), [1, nHeads, 0, headDim]),
      );
      pastV.push(
        new window.ort.Tensor('float32', new Float32Array(0), [1, nHeads, 0, headDim]),
      );
    }

    const generatedIds = [];
    const generatedTokens = [];
    let curId = bosId;
    let systemCount = 1;
    let stoppedByLoop = false;
    const decodeStart = performance.now();

    setProgress(22, 'Decoding notation tokens…', 'Decoding');

    for (let step = 0; step < maxTokens; step++) {
      if (state.cancelRequested) {
        setProgress(0, 'Cancelled by user.', 'Cancelled');
        break;
      }

      const feeds = {
        token_id: new window.ort.Tensor('int64', BigInt64Array.from([BigInt(curId)]), [1, 1]),
        step_pos: new window.ort.Tensor('int64', BigInt64Array.from([BigInt(step)]), [1]),
      };
      for (let i = 0; i < nLayers; i++) {
        feeds[`past_k_${i}`] = pastK[i];
        feeds[`past_v_${i}`] = pastV[i];
        feeds[`cross_k_${i}`] = encOutputs[`cross_k_${i}`];
        feeds[`cross_v_${i}`] = encOutputs[`cross_v_${i}`];
      }

      const out = await state.decoderSession.run(feeds);
      const logitsTensor = out.logits;
      const nextId = argmaxLastSlice(logitsTensor.data, vocabSize);

      // Update self-attention KV cache and dispose previous step tensors
      const nextPastK = [];
      const nextPastV = [];
      for (let i = 0; i < nLayers; i++) {
        nextPastK.push(out[`present_k_${i}`]);
        nextPastV.push(out[`present_v_${i}`]);
      }
      for (const t of pastK) if (t && typeof t.dispose === 'function') t.dispose();
      for (const t of pastV) if (t && typeof t.dispose === 'function') t.dispose();
      if (typeof logitsTensor.dispose === 'function') logitsTensor.dispose();
      pastK = nextPastK;
      pastV = nextPastV;

      if (nextId === eosId) {
        break;
      }

      curId = nextId;
      if (nextId !== padId && nextId !== bosId) {
        generatedIds.push(nextId);
        const tok = idToToken[nextId] ?? '';
        if (tok) generatedTokens.push(tok);
        if (nextId === sysId || tok === '<sys>') {
          systemCount++;
        }
      }

      // Check lightweight repetition guard
      const loopCheck = detectRepetitionLoop(generatedIds);
      if (loopCheck.detected) {
        stoppedByLoop = true;
        generatedTokens.length = Math.max(0, loopCheck.cutAt);
        break;
      }

      // Update UI every 4 tokens
      if (step % 4 === 0) {
        const elapsedS = (performance.now() - t0) / 1000;
        const decS = Math.max(0.001, (performance.now() - decodeStart) / 1000);
        const tps = (step + 1) / decS;
        const estPct = 22 + Math.min(70, (step / Math.min(maxTokens, 1024)) * 70);
        els.statTokens.textContent = String(generatedTokens.length);
        els.statSystems.textContent = String(systemCount);
        els.statSpeed.textContent = `${tps.toFixed(1)} tok/s`;
        els.statElapsed.textContent = `${elapsedS.toFixed(1)} s`;
        setProgress(estPct, `Decoding system ${systemCount} (${generatedTokens.length} tokens)…`, 'Decoding');
        await new Promise((r) => setTimeout(r, 0));
      }
    }

    // Cleanup remaining tensors
    for (const t of pastK) if (t && typeof t.dispose === 'function') t.dispose();
    for (const t of pastV) if (t && typeof t.dispose === 'function') t.dispose();
    for (const t of Object.values(encOutputs)) {
      if (t && typeof t.dispose === 'function') t.dispose();
    }

    if (state.cancelRequested) return;

    const finalTokens = stoppedByLoop ? trimToCleanBoundary(generatedTokens) : generatedTokens;
    const bekernStr = finalTokens.join(' ');
    els.bekernText.value = bekernStr;
    els.statTokens.textContent = String(finalTokens.length);
    els.statSystems.textContent = String(systemCount);

    // 4. Compile to .mscz + .musicxml via mscz-rs WASM
    setProgress(95, 'Compiling bekern+ to .mscz & .musicxml via mscz-rs WASM…', 'Compiling');
    const ok = compileAndEnableDownloads(bekernStr);

    const totalS = (performance.now() - t0) / 1000;
    const decS = Math.max(0.001, (performance.now() - decodeStart) / 1000);
    els.statElapsed.textContent = `${totalS.toFixed(1)} s`;
    els.statSpeed.textContent = `${(finalTokens.length / decS).toFixed(1)} tok/s`;

    if (ok) {
      setProgress(
        100,
        stoppedByLoop
          ? `Done in ${totalS.toFixed(1)} s (trimmed tail repetition loop). Ready to download.`
          : `Done in ${totalS.toFixed(1)} s — ${finalTokens.length} tokens across ${systemCount} system(s).`,
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
    state.running = false;
    els.btnCancel.classList.add('hidden');
    updateConvertButtonState();
  }
}

function handleImageFile(file) {
  if (!file || !file.type.startsWith('image/')) {
    showAlert('Please select a valid image file (PNG, JPEG, WebP).', 'warn');
    return;
  }
  const stem = file.name.replace(/\.[^.]+$/, '') || 'sheet-music';
  state.loadedFileStem = stem;
  if (!els.scoreTitle.value.trim()) {
    els.scoreTitle.value = stem.replace(/[_-]+/g, ' ');
  }

  const reader = new FileReader();
  reader.onload = () => {
    const img = new Image();
    img.onload = () => {
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

// Event bindings
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
    handleImageFile(e.dataTransfer.files[0]);
  }
});

els.fileInput.addEventListener('change', (e) => {
  if (e.target.files?.length) handleImageFile(e.target.files[0]);
});
els.cameraInput.addEventListener('change', (e) => {
  if (e.target.files?.length) handleImageFile(e.target.files[0]);
});

els.btnConvert.addEventListener('click', runConversion);
els.btnCancel.addEventListener('click', () => {
  state.cancelRequested = true;
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

els.btnRecompile.addEventListener('click', () => {
  showAlert('');
  const txt = els.bekernText.value.trim();
  if (!txt) {
    showAlert('Enter or generate bekern+ tokens first.', 'warn');
    return;
  }
  if (compileAndEnableDownloads(txt)) {
    setProgress(100, 'Recompiled .mscz and .musicxml in browser via mscz-rs WASM.', 'Compiled');
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

els.btnSampleBekern.addEventListener('click', () => {
  showAlert('');
  els.bekernDetails.open = true;
  els.bekernText.value = SAMPLE_BEKERN;
  if (!els.scoreTitle.value.trim()) {
    els.scoreTitle.value = 'WASM Sample Melody';
  }
  const toks = SAMPLE_BEKERN.split(/\s+/);
  els.statTokens.textContent = String(toks.length);
  els.statSystems.textContent = '1';
  if (compileAndEnableDownloads(SAMPLE_BEKERN)) {
    setProgress(100, 'Compiled sample score via mscz-rs WASM — ready to download .mscz / .musicxml.', 'WASM Ready');
  }
});

// Boot
initRuntime();
