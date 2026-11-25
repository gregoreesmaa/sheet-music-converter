/**
 * Dedicated ES module Web Worker for Sheet Music Converter.
 *
 * Offloads ALL heavy compute off the UI thread so animations, progress bars,
 * and Cancel remain 60fps responsive:
 *   - onnxruntime-web (WebGPU / multi-threaded WASM)
 *   - UniOMR encoder + KV-cached autoregressive decoder
 *   - mscz-rs WebAssembly compiler (bekern+ -> .mscz + .musicxml)
 */

import {
  initMsczWasm,
  bekernToMscz,
  bekernToMusicXml,
} from './wasm/mscz_wasm.js';

const ORT_CDN_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
const ORT_ESM_URL = `${ORT_CDN_BASE}ort.webgpu.min.mjs`;

// --- Crude Rollbar-like worker error & console forwarder (POST /__log) ---
function sendWorkerLog(level, msg, extra = null) {
  try {
    const body = JSON.stringify({ level, src: 'worker', msg: String(msg ?? '').slice(0, 4000), extra: extra || undefined });
    fetch('/__log', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body }).catch(() => {});
  } catch (_) {}
}
for (const lvl of ['error', 'warn']) {
  const orig = console[lvl].bind(console);
  console[lvl] = (...args) => {
    orig(...args);
    sendWorkerLog(lvl, args.map((a) => (a instanceof Error ? `${a.name}: ${a.message}\n${a.stack || ''}` : String(a))).join(' '));
  };
}
self.addEventListener('error', (ev) => {
  sendWorkerLog('error', ev.message || 'worker.error', { filename: ev.filename, lineno: ev.lineno, stack: ev.error?.stack });
});
self.addEventListener('unhandledrejection', (ev) => {
  const r = ev.reason;
  sendWorkerLog('error', r instanceof Error ? `${r.name}: ${r.message}` : String(r), { type: 'unhandledrejection', stack: r?.stack });
});

const state = {
  ort: null,
  wasmReady: false,
  modelsReady: false,
  backend: 'wasm',
  numThreads: 1,
  vocab: null,
  modelIo: null,
  encoderSession: null,
  decoderSession: null,
  cancelRequested: false,
};

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

async function detectWebGpu() {
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) return false;
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

function detectRepetitionLoop(tokenIds) {
  const n = tokenIds.length;
  if (n >= 24) {
    const last = tokenIds[n - 1];
    let run = 1;
    while (run < 24 && tokenIds[n - 1 - run] === last) run++;
    if (run >= 24) return { detected: true, period: 1, cutAt: n - run };
  }
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

async function handleInit() {
  // 1. Initialize mscz-rs WASM
  try {
    await initMsczWasm();
    state.wasmReady = true;
    self.postMessage({ type: 'wasm-ready' });
  } catch (err) {
    self.postMessage({ type: 'wasm-error', message: err?.message || String(err) });
  }

  // 2. Load onnxruntime-web ESM inside worker
  let ort;
  try {
    const mod = await import(ORT_ESM_URL);
    ort = mod.default || mod;
    state.ort = ort;
    ort.env.wasm.wasmPaths = ORT_CDN_BASE;
    const hw = typeof navigator !== 'undefined' && navigator.hardwareConcurrency
      ? navigator.hardwareConcurrency
      : 1;
    const threads = typeof SharedArrayBuffer !== 'undefined' ? Math.min(4, hw) : 1;
    ort.env.wasm.numThreads = threads;
    state.numThreads = threads;

    const hasWebGpu = await detectWebGpu();
    state.backend = hasWebGpu ? 'webgpu' : 'wasm';
    self.postMessage({
      type: 'ort-ready',
      backend: state.backend,
      numThreads: state.numThreads,
    });
  } catch (err) {
    self.postMessage({ type: 'ort-error', message: err?.message || String(err) });
    return;
  }

  // 3. Load vocab.json + ONNX encoder/decoder
  try {
    const vocabRaw = await fetchJsonOptional('./models/vocab.json');
    if (!vocabRaw) {
      self.postMessage({ type: 'models-missing' });
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
    const encRes = await createOrtSession(ort, encoderUrl, preferWebGpu);
    const decRes = await createOrtSession(ort, decoderUrl, encRes.usedWebGpu);

    state.encoderSession = encRes.session;
    state.decoderSession = decRes.session;
    state.backend = encRes.usedWebGpu && decRes.usedWebGpu ? 'webgpu' : 'wasm';
    state.modelsReady = true;

    self.postMessage({
      type: 'models-ready',
      backend: state.backend,
      numThreads: state.numThreads,
      maxDim: state.vocab.maxDim || state.modelIo.max_dim || 1280,
      maxTokens: state.modelIo.max_tokens || 2048,
    });
  } catch (err) {
    self.postMessage({ type: 'models-error', message: err?.message || String(err) });
  }
}

async function handleRecognizePage(msg) {
  const { requestId, nchwBuffer, width, height, pageIndex = 1, pageCount = 1 } = msg;
  if (!state.modelsReady || !state.ort) {
    self.postMessage({ type: 'error', requestId, message: 'UniOMR models are not loaded yet.' });
    return;
  }

  state.cancelRequested = false;
  const ort = state.ort;
  const maxTokens = state.modelIo.max_tokens || 2048;
  const nLayers = state.vocab.nDecLayers || state.modelIo.n_dec_layers || 6;
  const nHeads = state.vocab.nHeads || state.modelIo.n_heads || 6;
  const headDim = state.vocab.headDim || state.modelIo.head_dim || 64;
  const { bosId, eosId, sysId, padId, idToToken, vocabSize } = state.vocab;

  const t0 = performance.now();
  let imgTensor = null;
  let encOutputs = null;
  let pastK = [];
  let pastV = [];

  try {
    const nchw = new Float32Array(nchwBuffer);
    imgTensor = new ort.Tensor('float32', nchw, [1, 3, height, width]);

    self.postMessage({
      type: 'progress',
      requestId,
      stage: 'encoding',
      tokens: 0,
      systems: 0,
      tokPerSec: 0,
      elapsedMs: performance.now() - t0,
      pct: 15,
      pageIndex,
      pageCount,
    });

    encOutputs = await state.encoderSession.run({ image: imgTensor });
    if (typeof imgTensor.dispose === 'function') imgTensor.dispose();
    imgTensor = null;

    const encodeMs = performance.now() - t0;

    if (state.cancelRequested) {
      self.postMessage({ type: 'cancelled', requestId });
      return;
    }

    for (let i = 0; i < nLayers; i++) {
      pastK.push(new ort.Tensor('float32', new Float32Array(0), [1, nHeads, 0, headDim]));
      pastV.push(new ort.Tensor('float32', new Float32Array(0), [1, nHeads, 0, headDim]));
    }

    const generatedIds = [];
    const generatedTokens = [];
    let curId = bosId;
    let systemCount = 1;
    let stoppedByLoop = false;
    const decodeStart = performance.now();

    self.postMessage({
      type: 'progress',
      requestId,
      stage: 'decoding',
      tokens: 0,
      systems: systemCount,
      tokPerSec: 0,
      elapsedMs: performance.now() - t0,
      pct: 22,
      pageIndex,
      pageCount,
    });

    for (let step = 0; step < maxTokens; step++) {
      if (state.cancelRequested) {
        self.postMessage({ type: 'cancelled', requestId });
        return;
      }

      const feeds = {
        token_id: new ort.Tensor('int64', BigInt64Array.from([BigInt(curId)]), [1, 1]),
        step_pos: new ort.Tensor('int64', BigInt64Array.from([BigInt(step)]), [1]),
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

      const loopCheck = detectRepetitionLoop(generatedIds);
      if (loopCheck.detected) {
        stoppedByLoop = true;
        generatedTokens.length = Math.max(0, loopCheck.cutAt);
        break;
      }

      if (step % 4 === 0) {
        const now = performance.now();
        const decS = Math.max(0.001, (now - decodeStart) / 1000);
        const tokPerSec = (step + 1) / decS;
        const pct = 22 + Math.min(70, (step / Math.min(maxTokens, 1024)) * 70);
        self.postMessage({
          type: 'progress',
          requestId,
          stage: 'decoding',
          tokens: generatedTokens.length,
          systems: systemCount,
          tokPerSec,
          elapsedMs: now - t0,
          pct,
          pageIndex,
          pageCount,
        });
      }
    }

    const finalTokens = stoppedByLoop ? trimToCleanBoundary(generatedTokens) : generatedTokens;
    const decodeMs = performance.now() - decodeStart;
    const tokPerSec = finalTokens.length / Math.max(0.001, decodeMs / 1000);

    self.postMessage({
      type: 'page-done',
      requestId,
      bekern: finalTokens.join(' '),
      tokens: finalTokens.length,
      systems: systemCount,
      stoppedByLoop,
      tokPerSec,
      encodeMs,
      decodeMs,
      pageIndex,
      pageCount,
    });
  } catch (err) {
    self.postMessage({
      type: 'error',
      requestId,
      message: err?.message || String(err),
    });
  } finally {
    if (imgTensor && typeof imgTensor.dispose === 'function') imgTensor.dispose();
    for (const t of pastK) if (t && typeof t.dispose === 'function') t.dispose();
    for (const t of pastV) if (t && typeof t.dispose === 'function') t.dispose();
    if (encOutputs) {
      for (const t of Object.values(encOutputs)) {
        if (t && typeof t.dispose === 'function') t.dispose();
      }
    }
  }
}

function handleCompile(msg) {
  const { requestId, bekern, title } = msg;
  if (!state.wasmReady) {
    self.postMessage({
      type: 'compile-error',
      requestId,
      message: 'mscz-rs WASM is not ready yet.',
    });
    return;
  }
  try {
    const msczBytes = bekernToMscz(bekern, title);
    const musicXml = bekernToMusicXml(bekern, title);
    // Transfer the underlying ArrayBuffer zero-copy to the main thread
    const msczBuffer = msczBytes.buffer.slice(
      msczBytes.byteOffset,
      msczBytes.byteOffset + msczBytes.byteLength,
    );
    self.postMessage(
      {
        type: 'compile-done',
        requestId,
        msczBuffer,
        musicXml,
      },
      [msczBuffer],
    );
  } catch (err) {
    self.postMessage({
      type: 'compile-error',
      requestId,
      message: err?.message || String(err),
    });
  }
}

self.onmessage = async (event) => {
  const msg = event.data;
  if (!msg || typeof msg.type !== 'string') return;
  switch (msg.type) {
    case 'init':
      await handleInit();
      break;
    case 'recognize-page':
      await handleRecognizePage(msg);
      break;
    case 'compile':
      handleCompile(msg);
      break;
    case 'cancel':
      state.cancelRequested = true;
      break;
    default:
      break;
  }
};
