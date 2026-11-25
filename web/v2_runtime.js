/**
 * UniOMR v2 browser/edge inference runtime helpers (ES module, zero external deps).
 *
 * Designed to be imported directly by `web/worker.js` once v2 checkpoints are
 * exported via `tmp-train/mvd/export_uni_v2_onnx.py`:
 *   1. `selectV2Target(manifest, capabilities, preferredTarget)` — auto-selects
 *      WebGPU (preferring FP16 when `shaderF16` is supported and available),
 *      WebNN NPU (static ring-buffer KV), or WASM SIMD CPU fallback.
 *   2. `runV2DynamicDecodeLoop(...)` — zero-copy GPU-resident KV-cache slicing
 *      via `step_pos = acceptedLen` (0 `getData()` calls on KV tensors) +
 *      All-T MTP-K draft extraction at index `acceptedCount - 1` (0 wasted
 *      `T=1` fallback steps after partial accepts).
 *   3. `runV2NpuStaticDecodeLoop(...)` — allocates static ring-buffer KV
 *      tensors `(1, nKvHeads, kvMaxLen, headDim)` ONCE and steps `step_pos =
 *      0, 1, 2, ...` with 100% constant tensor shapes (zero graph recompiles).
 */

// ---------------------------------------------------------------------------
// 1. Multi-Target Auto-Selector
// ---------------------------------------------------------------------------

function inferPrecision(quantFormat) {
  const q = String(quantFormat || '').toLowerCase();
  if (q.includes('fp16')) return 'fp16';
  if (q.includes('int4')) return 'int4';
  if (q.includes('int8') || q.includes('uint8')) return 'int8';
  return 'fp32';
}

function buildTargetSelection(targetName, entry, overrides = {}) {
  const executionProvider =
    targetName === 'webgpu' ? 'webgpu' : targetName === 'npu' ? 'webnn' : 'wasm';
  const quantFormat = overrides.quantFormat ?? entry.quant_format ?? 'fp32';
  const precision = overrides.precision ?? inferPrecision(quantFormat);
  const useFp16 = overrides.useFp16 ?? precision === 'fp16';
  return {
    target: targetName,
    executionProvider,
    encoderFile: overrides.encoderFile ?? entry.encoder_file,
    decoderFile: overrides.decoderFile ?? entry.decoder_file,
    quantFormat,
    precision,
    useFp16,
    kvMode: entry.kv_mode || (targetName === 'npu' ? 'static_ring_buffer' : 'dynamic_slice'),
    mtpK: Number(entry.mtp_k ?? 1),
    staticHw: entry.static_hw ?? null,
    staticKMem: entry.static_k_mem ?? null,
    staticDecT: entry.static_dec_t ?? null,
    kvMaxLen: entry.kv_max_len ?? null,
    entry,
  };
}

function resolveWebGpuTarget(targets, shaderF16) {
  const fp16Entry = targets.webgpu_fp16 || null;
  const baseEntry = targets.webgpu || null;

  if (shaderF16) {
    if (fp16Entry) {
      return buildTargetSelection('webgpu', fp16Entry, {
        precision: 'fp16',
        useFp16: true,
      });
    }
    if (baseEntry && baseEntry.fp16_encoder_file && baseEntry.fp16_decoder_file) {
      return buildTargetSelection('webgpu', baseEntry, {
        encoderFile: baseEntry.fp16_encoder_file,
        decoderFile: baseEntry.fp16_decoder_file,
        quantFormat: 'fp16_mixed',
        precision: 'fp16',
        useFp16: true,
      });
    }
  }

  if (baseEntry) {
    const isBaseFp16 = inferPrecision(baseEntry.quant_format) === 'fp16';
    if (!isBaseFp16 || shaderF16) {
      return buildTargetSelection('webgpu', baseEntry);
    }
    // Base webgpu entry requires FP16, but adapter lacks shader-f16.
    if (targets.webgpu_int4) {
      return buildTargetSelection('webgpu', targets.webgpu_int4);
    }
    return null;
  }

  if (shaderF16 && fp16Entry) {
    return buildTargetSelection('webgpu', fp16Entry, {
      precision: 'fp16',
      useFp16: true,
    });
  }

  return null;
}

/**
 * Select the optimal UniOMR v2 ONNX target from `manifest.json` based on
 * runtime hardware capabilities and an optional user/caller preference.
 *
 * @param {object} manifest - Parsed `manifest.json` from `export_uni_v2_onnx.py`.
 * @param {{ webgpu?: boolean, shaderF16?: boolean, webnn?: boolean, wasm?: boolean }} capabilities
 * @param {'auto'|'webgpu'|'webgpu_fp16'|'npu'|'webnn'|'cpu'|'wasm'} [preferredTarget='auto']
 */
export function selectV2Target(manifest, capabilities = {}, preferredTarget = 'auto') {
  const targets = manifest && typeof manifest === 'object' ? manifest.targets || {} : {};
  const caps = {
    webgpu: Boolean(capabilities?.webgpu),
    shaderF16: Boolean(capabilities?.shaderF16),
    webnn: Boolean(capabilities?.webnn),
    wasm: capabilities?.wasm !== false,
  };

  const pref = String(preferredTarget || 'auto').trim().toLowerCase();

  // Explicit target override
  if (pref !== '' && pref !== 'auto') {
    if (pref === 'webgpu_fp16') {
      const sel = resolveWebGpuTarget(targets, true);
      if (sel) return sel;
    } else if (pref === 'webgpu') {
      const sel = resolveWebGpuTarget(targets, caps.shaderF16) ||
        (targets.webgpu ? buildTargetSelection('webgpu', targets.webgpu) : null);
      if (sel) return sel;
    } else if (pref === 'npu' || pref === 'webnn') {
      if (targets.npu) return buildTargetSelection('npu', targets.npu);
    } else if (pref === 'cpu' || pref === 'wasm') {
      if (targets.cpu) return buildTargetSelection('cpu', targets.cpu);
    }
  }

  // Auto-selection: WebGPU -> WebNN NPU -> WASM SIMD CPU
  if (caps.webgpu) {
    const wg = resolveWebGpuTarget(targets, caps.shaderF16);
    if (wg) return wg;
  }

  if (!caps.webgpu && caps.webnn && targets.npu) {
    return buildTargetSelection('npu', targets.npu);
  }

  if (targets.cpu) {
    return buildTargetSelection('cpu', targets.cpu);
  }

  // Fallback if manifest only exported a subset of targets
  if (targets.npu && caps.webnn) {
    return buildTargetSelection('npu', targets.npu);
  }
  if (targets.webgpu) {
    return buildTargetSelection('webgpu', targets.webgpu);
  }
  if (targets.npu) {
    return buildTargetSelection('npu', targets.npu);
  }

  throw new Error('UniOMR v2 manifest contains no usable targets.');
}

// ---------------------------------------------------------------------------
// Internal Tensor / Logits / FSA Helpers
// ---------------------------------------------------------------------------

function disposeTensor(t) {
  if (t && typeof t.dispose === 'function') {
    t.dispose();
  }
}

async function readLogitsData(tensor) {
  if (typeof tensor.getData === 'function') {
    return await tensor.getData();
  }
  return tensor.data;
}

function argmaxSlice(data, vocabSize, offset) {
  let bestIdx = 0;
  let bestVal = -Infinity;
  for (let i = 0; i < vocabSize; i++) {
    const v = data[offset + i];
    if (v > bestVal) {
      bestVal = v;
      bestIdx = i;
    }
  }
  return bestIdx;
}

function selectTokenAt(fsa, logitsData, vocabSize, posIndex) {
  const offset = posIndex * vocabSize;
  if (fsa && typeof fsa.selectNextToken === 'function') {
    return fsa.selectNextToken(logitsData, vocabSize, offset);
  }
  return argmaxSlice(logitsData, vocabSize, offset);
}

function stepFsa(fsa, tokenId) {
  if (fsa && typeof fsa.step === 'function') {
    fsa.step(tokenId);
  }
}

function inferKvGeometry(crossKv, opts = {}) {
  let nDecLayers = opts.nDecLayers;
  if (!Number.isInteger(nDecLayers) || nDecLayers <= 0) {
    let count = 0;
    while (crossKv && crossKv[`cross_k_${count}`]) count++;
    nDecLayers = count;
  }
  if (nDecLayers <= 0) {
    throw new Error('Could not determine nDecLayers from crossKv or options.');
  }
  const ck0 = crossKv?.cross_k_0;
  const dims = ck0?.dims || [];
  const nKvHeads = opts.nKvHeads ?? dims[1];
  const headDim = opts.headDim ?? dims[3];
  if (!Number.isInteger(nKvHeads) || nKvHeads <= 0 || !Number.isInteger(headDim) || headDim <= 0) {
    throw new Error(`Invalid KV geometry: nKvHeads=${nKvHeads}, headDim=${headDim}`);
  }
  return { nDecLayers, nKvHeads, headDim };
}

/**
 * Draft `mtpLogitsDataList.length` future tokens from All-T MTP logits at
 * position `posIndex = acceptedCount - 1` using a cloned FSA so the committed
 * FSA state is never mutated.
 */
function draftFromAllTMtp(fsa, mtpLogitsDataList, vocabSize, posIndex) {
  const draftFsa = fsa && typeof fsa.clone === 'function' ? fsa.clone() : null;
  const drafts = new Array(mtpLogitsDataList.length);
  for (let m = 0; m < mtpLogitsDataList.length; m++) {
    const tok = selectTokenAt(draftFsa, mtpLogitsDataList[m], vocabSize, posIndex);
    stepFsa(draftFsa, tok);
    drafts[m] = tok;
  }
  return drafts;
}

// ---------------------------------------------------------------------------
// 2. Dynamic-KV Decode Loop (WebGPU / CPU): Zero-Copy KV + All-T MTP-K
// ---------------------------------------------------------------------------

/**
 * Run the UniOMR v2 dynamic-KV decoder with:
 *   - **Zero-Copy GPU KV-Cache Slicing:** passes `present_k_i / present_v_i`
 *     directly back as `past_k_i / past_v_i` with `step_pos = acceptedLen`.
 *     NEVER calls `getData()` on any KV tensor.
 *   - **All-T MTP-K Draft Extraction:** reads `mtp_logits_j` at position
 *     `acceptedCount - 1` (`0 .. T-1`) so partial accepts immediately draft the
 *     next `K-1` tokens with **0 wasted `T=1` fallback steps**.
 *
 * @param {object} params
 * @param {object} params.decoderSession - ORT InferenceSession for v2 dynamic decoder.
 * @param {Record<string, object>} params.crossKv - Encoder outputs (`cross_k_0..L-1`, `cross_v_0..L-1`).
 * @param {number} params.bosId - BOS token ID.
 * @param {number} params.eosId - EOS token ID.
 * @param {number} params.maxTokens - Maximum number of generated tokens (excluding BOS/EOS).
 * @param {number} [params.mtpK=1] - Total MTP width (e.g. 4 for 1 main + 3 MTP draft heads).
 * @param {object|null} [params.fsa=null] - Optional BekernFsa grammar mask instance.
 * @param {object} params.ort - onnxruntime-web module (provides `ort.Tensor`).
 * @param {number} [params.nDecLayers] - Optional override (inferred from `crossKv`).
 * @param {number} [params.nKvHeads] - Optional override (inferred from `crossKv`).
 * @param {number} [params.headDim] - Optional override (inferred from `crossKv`).
 * @param {number} [params.vocabSize] - Optional override (inferred from `logits.dims[2]`).
 * @param {(info: object) => void} [params.onProgress] - Optional per-step callback.
 * @param {() => boolean} [params.shouldCancel] - Optional cancellation predicate.
 */
export async function runV2DynamicDecodeLoop({
  decoderSession,
  crossKv,
  bosId,
  eosId,
  maxTokens,
  mtpK = 1,
  fsa = null,
  ort,
  nDecLayers: optLayers,
  nKvHeads: optKvHeads,
  headDim: optHeadDim,
  vocabSize: optVocabSize,
  onProgress = null,
  shouldCancel = null,
}) {
  const { nDecLayers, nKvHeads, headDim } = inferKvGeometry(crossKv, {
    nDecLayers: optLayers,
    nKvHeads: optKvHeads,
    headDim: optHeadDim,
  });

  const nMtpHeads = Math.max(0, Number(mtpK || 1) - 1);
  if (fsa && typeof fsa.reset === 'function') fsa.reset();

  let pastK = new Array(nDecLayers);
  let pastV = new Array(nDecLayers);
  for (let i = 0; i < nDecLayers; i++) {
    pastK[i] = new ort.Tensor('float32', new Float32Array(0), [1, nKvHeads, 0, headDim]);
    pastV[i] = new ort.Tensor('float32', new Float32Array(0), [1, nKvHeads, 0, headDim]);
  }

  const tokenIds = [];
  let acceptedLen = 0; // Exact committed KV length passed as `step_pos`
  let curToken = bosId;
  let pendingDrafts = null; // null only on initial BOS prefill (or when nMtpHeads === 0)
  let hitEos = false;
  let cancelled = false;
  let steps = 0;
  let t1FallbackSteps = 0;
  let totalDrafted = 0;
  let totalDraftAccepted = 0;

  try {
    while (tokenIds.length < maxTokens) {
      if (typeof shouldCancel === 'function' && shouldCancel()) {
        cancelled = true;
        break;
      }

      const isPrefill = steps === 0;
      const useSpec = nMtpHeads > 0 && pendingDrafts !== null && pendingDrafts.length === nMtpHeads;
      if (!isPrefill && nMtpHeads > 0 && !useSpec) {
        t1FallbackSteps += 1;
      }

      const batchIds = useSpec ? [curToken, ...pendingDrafts] : [curToken];
      const T = batchIds.length;

      const tokenIdsTensor = new ort.Tensor(
        'int64',
        BigInt64Array.from(batchIds, (id) => BigInt(id)),
        [1, T],
      );
      const stepPosTensor = new ort.Tensor(
        'int64',
        BigInt64Array.from([BigInt(acceptedLen)]),
        [1],
      );

      const feeds = {
        token_ids: tokenIdsTensor,
        step_pos: stepPosTensor,
      };
      for (let i = 0; i < nDecLayers; i++) {
        feeds[`past_k_${i}`] = pastK[i];
        feeds[`past_v_${i}`] = pastV[i];
        feeds[`cross_k_${i}`] = crossKv[`cross_k_${i}`];
        feeds[`cross_v_${i}`] = crossKv[`cross_v_${i}`];
      }

      let out;
      try {
        out = await decoderSession.run(feeds);
      } finally {
        disposeTensor(tokenIdsTensor);
        disposeTensor(stepPosTensor);
      }
      steps += 1;

      // Zero-copy GPU KV handoff: dispose previous past_k/v and keep present_k/v
      // directly on-device without ever calling getData() or slicing on CPU.
      const nextPastK = new Array(nDecLayers);
      const nextPastV = new Array(nDecLayers);
      for (let i = 0; i < nDecLayers; i++) {
        const pk = out[`present_k_${i}`];
        const pv = out[`present_v_${i}`];
        if (pastK[i] !== pk) disposeTensor(pastK[i]);
        if (pastV[i] !== pv) disposeTensor(pastV[i]);
        nextPastK[i] = pk;
        nextPastV[i] = pv;
      }
      pastK = nextPastK;
      pastV = nextPastV;

      // Read CPU logits & All-T MTP logits, then immediately dispose them.
      const logitsTensor = out.logits;
      const vocabSize = optVocabSize ?? logitsTensor.dims[logitsTensor.dims.length - 1];
      const logitsData = await readLogitsData(logitsTensor);
      disposeTensor(logitsTensor);

      const mtpDataList = [];
      for (let m = 0; m < nMtpHeads; m++) {
        const mt = out[`mtp_logits_${m}`];
        if (mt) {
          mtpDataList.push(await readLogitsData(mt));
          disposeTensor(mt);
        }
      }

      const maxRemaining = maxTokens - tokenIds.length;
      let acceptedCount = 0;
      let lastAccepted = curToken;

      if (useSpec) {
        totalDrafted += pendingDrafts.length;
        for (let j = 0; j < T; j++) {
          if (acceptedCount >= maxRemaining) break;
          const vTok = selectTokenAt(fsa, logitsData, vocabSize, j);
          stepFsa(fsa, vTok);
          acceptedCount += 1;
          lastAccepted = vTok;
          if (vTok === eosId) {
            hitEos = true;
            break;
          }
          tokenIds.push(vTok);
          if (j < pendingDrafts.length) {
            if (vTok === pendingDrafts[j]) {
              totalDraftAccepted += 1;
            } else {
              // Partial accept: stop verifying; All-T MTP at index `j`
              // (`acceptedCount - 1`) immediately provides next drafts!
              break;
            }
          }
        }
      } else {
        const vTok = selectTokenAt(fsa, logitsData, vocabSize, 0);
        stepFsa(fsa, vTok);
        acceptedCount = 1;
        lastAccepted = vTok;
        if (vTok === eosId) {
          hitEos = true;
        } else {
          tokenIds.push(vTok);
        }
      }

      // Advance logical KV length by the exact number of accepted tokens.
      // Any rejected draft slots beyond `acceptedLen` will be sliced off
      // on-GPU by `past_k[:, :, :step_pos, :]` on the next step!
      acceptedLen += acceptedCount;

      if (typeof onProgress === 'function') {
        onProgress({
          steps,
          tokens: tokenIds.length,
          acceptedCount,
          acceptedLen,
          hitEos,
        });
      }

      if (hitEos || tokenIds.length >= maxTokens) {
        break;
      }

      curToken = lastAccepted;
      if (nMtpHeads > 0 && mtpDataList.length === nMtpHeads) {
        // All-T MTP draft extraction at index `acceptedCount - 1` (0 .. T-1).
        const draftPosIndex = Math.max(0, Math.min(T - 1, acceptedCount - 1));
        pendingDrafts = draftFromAllTMtp(fsa, mtpDataList, vocabSize, draftPosIndex);
      } else {
        pendingDrafts = null;
      }
    }
  } finally {
    for (let i = 0; i < nDecLayers; i++) {
      disposeTensor(pastK[i]);
      disposeTensor(pastV[i]);
    }
  }

  return {
    tokenIds,
    hitEos,
    cancelled,
    steps,
    acceptedLen,
    t1FallbackSteps,
    totalDrafted,
    totalDraftAccepted,
    meanAcceptRate: totalDrafted > 0 ? totalDraftAccepted / totalDrafted : 0,
  };
}

// ---------------------------------------------------------------------------
// 3. Static Ring-Buffer KV Decode Loop (WebNN NPU / CoreML ANE / Hexagon)
// ---------------------------------------------------------------------------

/**
 * Run the UniOMR v2 100%-static-shape NPU ring-buffer KV decoder.
 *
 * Allocates fixed `(1, nKvHeads, kvMaxLen, headDim)` KV tensors ONCE before
 * decoding and steps `step_pos = 0, 1, 2, ...` with constant tensor shapes on
 * every invocation (zero graph recompiles on WebNN / CoreML ANE / Hexagon).
 *
 * @param {object} params
 * @param {object} params.decoderSession - ORT InferenceSession for v2 NPU static decoder.
 * @param {Record<string, object>} params.crossKv - Static encoder outputs (`cross_k_0..L-1`, `cross_v_0..L-1`).
 * @param {number} params.kvMaxLen - Fixed static KV ring-buffer length.
 * @param {number} params.nKvHeads - Number of KV heads.
 * @param {number} params.headDim - Per-head dimension.
 * @param {number} params.nDecLayers - Number of decoder layers.
 * @param {number} params.bosId - BOS token ID.
 * @param {number} params.eosId - EOS token ID.
 * @param {number} params.maxTokens - Maximum generated tokens.
 * @param {object|null} [params.fsa=null] - Optional BekernFsa grammar mask instance.
 * @param {object} params.ort - onnxruntime-web module (provides `ort.Tensor`).
 * @param {number} [params.vocabSize] - Optional override (inferred from `logits.dims[2]`).
 * @param {(info: object) => void} [params.onProgress] - Optional per-step callback.
 * @param {() => boolean} [params.shouldCancel] - Optional cancellation predicate.
 */
export async function runV2NpuStaticDecodeLoop({
  decoderSession,
  crossKv,
  kvMaxLen,
  nKvHeads,
  headDim,
  nDecLayers,
  bosId,
  eosId,
  maxTokens,
  fsa = null,
  ort,
  vocabSize: optVocabSize,
  onProgress = null,
  shouldCancel = null,
}) {
  if (!Number.isInteger(kvMaxLen) || kvMaxLen <= 0) {
    throw new Error(`Invalid kvMaxLen: ${kvMaxLen}`);
  }
  const geom = inferKvGeometry(crossKv, { nDecLayers, nKvHeads, headDim });
  const L = geom.nDecLayers;
  const H = geom.nKvHeads;
  const D = geom.headDim;

  if (fsa && typeof fsa.reset === 'function') fsa.reset();

  // Allocate fixed static ring-buffer KV tensors ONCE.
  const kvBufSize = H * kvMaxLen * D;
  let pastK = new Array(L);
  let pastV = new Array(L);
  for (let i = 0; i < L; i++) {
    pastK[i] = new ort.Tensor('float32', new Float32Array(kvBufSize), [1, H, kvMaxLen, D]);
    pastV[i] = new ort.Tensor('float32', new Float32Array(kvBufSize), [1, H, kvMaxLen, D]);
  }

  const tokenIds = [];
  const limit = Math.min(maxTokens, kvMaxLen);
  let curToken = bosId;
  let stepPos = 0;
  let hitEos = false;
  let cancelled = false;
  let steps = 0;

  try {
    while (tokenIds.length < limit && stepPos < kvMaxLen) {
      if (typeof shouldCancel === 'function' && shouldCancel()) {
        cancelled = true;
        break;
      }

      const tokenIdsTensor = new ort.Tensor(
        'int64',
        BigInt64Array.from([BigInt(curToken)]),
        [1, 1],
      );
      const stepPosTensor = new ort.Tensor(
        'int64',
        BigInt64Array.from([BigInt(stepPos)]),
        [1],
      );

      const feeds = {
        token_ids: tokenIdsTensor,
        step_pos: stepPosTensor,
      };
      for (let i = 0; i < L; i++) {
        feeds[`past_k_${i}`] = pastK[i];
        feeds[`past_v_${i}`] = pastV[i];
        feeds[`cross_k_${i}`] = crossKv[`cross_k_${i}`];
        feeds[`cross_v_${i}`] = crossKv[`cross_v_${i}`];
      }

      let out;
      try {
        out = await decoderSession.run(feeds);
      } finally {
        disposeTensor(tokenIdsTensor);
        disposeTensor(stepPosTensor);
      }
      steps += 1;

      // Pass static ring-buffer present_k/v directly back as past_k/v (zero-copy).
      const nextPastK = new Array(L);
      const nextPastV = new Array(L);
      for (let i = 0; i < L; i++) {
        const pk = out[`present_k_${i}`];
        const pv = out[`present_v_${i}`];
        if (pastK[i] !== pk) disposeTensor(pastK[i]);
        if (pastV[i] !== pv) disposeTensor(pastV[i]);
        nextPastK[i] = pk;
        nextPastV[i] = pv;
      }
      pastK = nextPastK;
      pastV = nextPastV;

      // Dispose any unused MTP outputs if the static NPU graph exports them.
      for (const key of Object.keys(out)) {
        if (key.startsWith('mtp_logits_')) {
          disposeTensor(out[key]);
        }
      }

      const logitsTensor = out.logits;
      const vocabSize = optVocabSize ?? logitsTensor.dims[logitsTensor.dims.length - 1];
      const logitsData = await readLogitsData(logitsTensor);
      disposeTensor(logitsTensor);

      const nextTok = selectTokenAt(fsa, logitsData, vocabSize, 0);
      stepFsa(fsa, nextTok);
      stepPos += 1;

      if (nextTok === eosId) {
        hitEos = true;
      } else {
        tokenIds.push(nextTok);
        curToken = nextTok;
      }

      if (typeof onProgress === 'function') {
        onProgress({
          steps,
          tokens: tokenIds.length,
          stepPos,
          hitEos,
        });
      }

      if (hitEos) break;
    }
  } finally {
    for (let i = 0; i < L; i++) {
      disposeTensor(pastK[i]);
      disposeTensor(pastV[i]);
    }
  }

  return {
    tokenIds,
    hitEos,
    cancelled,
    steps,
    stepPos,
  };
}
