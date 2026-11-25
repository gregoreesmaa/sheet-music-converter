import assert from 'node:assert/strict';
import {
  selectV2Target,
  runV2DynamicDecodeLoop,
  runV2NpuStaticDecodeLoop,
} from './v2_runtime.js';

// ---------------------------------------------------------------------------
// Minimal Mock ORT Tensor (tracks GPU-resident KV getData() calls!)
// ---------------------------------------------------------------------------

class MockTensor {
  constructor(type, data, dims, { gpuResident = false, onGetData = null } = {}) {
    this.type = type;
    this.dims = dims.slice();
    this.gpuResident = gpuResident;
    this._data = data;
    this._onGetData = onGetData;
    this.disposed = false;
  }

  get data() {
    if (this.gpuResident) {
      throw new Error('Forbidden synchronous .data access on GPU-resident KV tensor!');
    }
    return this._data;
  }

  async getData() {
    if (this._onGetData) this._onGetData(this);
    if (this.gpuResident) {
      throw new Error('Forbidden getData() call on GPU-resident KV tensor!');
    }
    return this._data;
  }

  dispose() {
    this.disposed = true;
  }
}

const mockOrt = {
  Tensor: class extends MockTensor {
    constructor(type, data, dims) {
      super(type, data, dims, { gpuResident: false });
    }
  },
};

// Minimal FSA mock verifying clone / selectNextToken / step integration.
class MockFsa {
  constructor(vocabSize) {
    this.vocabSize = vocabSize;
    this.history = [];
  }
  reset() {
    this.history = [];
  }
  clone() {
    const c = new MockFsa(this.vocabSize);
    c.history = this.history.slice();
    return c;
  }
  selectNextToken(logitsData, vocabSize, offset = 0) {
    let best = 0;
    let bestVal = -Infinity;
    for (let i = 0; i < vocabSize; i++) {
      const v = logitsData[offset + i];
      if (v > bestVal) {
        bestVal = v;
        best = i;
      }
    }
    return best;
  }
  step(id) {
    this.history.push(id);
  }
}

function makeOneHotLogits(vocabSize, tokenSeq) {
  const T = tokenSeq.length;
  const arr = new Float32Array(T * vocabSize);
  for (let t = 0; t < T; t++) {
    arr[t * vocabSize + tokenSeq[t]] = 10.0;
  }
  return new MockTensor('float32', arr, [1, T, vocabSize]);
}

// ---------------------------------------------------------------------------
// Test 1: selectV2Target across all capability combos & explicit overrides
// ---------------------------------------------------------------------------

function testSelectV2Target() {
  const manifest = {
    schema_version: 2,
    arch: 'uni_omr_v2',
    targets: {
      webgpu: {
        target: 'webgpu',
        encoder_file: 'webgpu/uni_omr_v2_encoder.onnx',
        decoder_file: 'webgpu/uni_omr_v2_decoder.onnx',
        quant_format: 'int4_matmul_nbits_b32',
        kv_mode: 'dynamic_slice',
        mtp_k: 4,
      },
      webgpu_fp16: {
        target: 'webgpu',
        encoder_file: 'webgpu_fp16/uni_omr_v2_encoder.onnx',
        decoder_file: 'webgpu_fp16/uni_omr_v2_decoder.onnx',
        quant_format: 'fp16_mixed',
        kv_mode: 'dynamic_slice',
        mtp_k: 4,
      },
      npu: {
        target: 'npu',
        encoder_file: 'npu/uni_omr_v2_encoder.onnx',
        decoder_file: 'npu/uni_omr_v2_decoder.onnx',
        quant_format: 'int8_dequantize_linear',
        kv_mode: 'static_ring_buffer',
        static_hw: [1280, 960],
        static_k_mem: 2048,
        static_dec_t: 1,
        kv_max_len: 2048,
        mtp_k: 4,
      },
      cpu: {
        target: 'cpu',
        encoder_file: 'cpu/uni_omr_v2_encoder.onnx',
        decoder_file: 'cpu/uni_omr_v2_decoder.onnx',
        quant_format: 'int8_dynamic_quint8',
        kv_mode: 'dynamic_slice',
        mtp_k: 4,
      },
    },
  };

  // 1a. WebGPU + shaderF16 -> selects FP16 WebGPU variant
  const rWgF16 = selectV2Target(manifest, { webgpu: true, shaderF16: true, webnn: true, wasm: true });
  assert.equal(rWgF16.target, 'webgpu');
  assert.equal(rWgF16.executionProvider, 'webgpu');
  assert.equal(rWgF16.useFp16, true);
  assert.equal(rWgF16.precision, 'fp16');
  assert.equal(rWgF16.encoderFile, 'webgpu_fp16/uni_omr_v2_encoder.onnx');

  // 1b. WebGPU without shaderF16 -> selects INT4 WebGPU variant
  const rWgInt4 = selectV2Target(manifest, { webgpu: true, shaderF16: false, webnn: true, wasm: true });
  assert.equal(rWgInt4.target, 'webgpu');
  assert.equal(rWgInt4.useFp16, false);
  assert.equal(rWgInt4.precision, 'int4');
  assert.equal(rWgInt4.encoderFile, 'webgpu/uni_omr_v2_encoder.onnx');

  // 1c. WebNN-only (no WebGPU) -> selects NPU static ring-buffer target
  const rNpu = selectV2Target(manifest, { webgpu: false, shaderF16: false, webnn: true, wasm: true });
  assert.equal(rNpu.target, 'npu');
  assert.equal(rNpu.executionProvider, 'webnn');
  assert.equal(rNpu.kvMode, 'static_ring_buffer');
  assert.equal(rNpu.kvMaxLen, 2048);

  // 1d. WASM-only -> selects CPU fallback
  const rCpu = selectV2Target(manifest, { webgpu: false, shaderF16: false, webnn: false, wasm: true });
  assert.equal(rCpu.target, 'cpu');
  assert.equal(rCpu.executionProvider, 'wasm');
  assert.equal(rCpu.kvMode, 'dynamic_slice');

  // 1e. Explicit overrides ('npu', 'cpu', 'webgpu')
  const rForceNpu = selectV2Target(manifest, { webgpu: true, shaderF16: true, webnn: true }, 'npu');
  assert.equal(rForceNpu.target, 'npu');
  assert.equal(rForceNpu.executionProvider, 'webnn');

  const rForceCpu = selectV2Target(manifest, { webgpu: true, shaderF16: true, webnn: true }, 'cpu');
  assert.equal(rForceCpu.target, 'cpu');
  assert.equal(rForceCpu.executionProvider, 'wasm');

  const rForceWasmAlias = selectV2Target(manifest, { webgpu: true, shaderF16: true }, 'wasm');
  assert.equal(rForceWasmAlias.target, 'cpu');

  // 1f. Inline fp16_encoder_file on `targets.webgpu` without separate `webgpu_fp16` key
  const inlineFp16Manifest = {
    targets: {
      webgpu: {
        target: 'webgpu',
        encoder_file: 'webgpu/enc_int4.onnx',
        decoder_file: 'webgpu/dec_int4.onnx',
        fp16_encoder_file: 'webgpu/enc_fp16.onnx',
        fp16_decoder_file: 'webgpu/dec_fp16.onnx',
        quant_format: 'int4_matmul_nbits_b32',
        kv_mode: 'dynamic_slice',
      },
      cpu: manifest.targets.cpu,
    },
  };
  const rInlineF16 = selectV2Target(inlineFp16Manifest, { webgpu: true, shaderF16: true });
  assert.equal(rInlineF16.useFp16, true);
  assert.equal(rInlineF16.encoderFile, 'webgpu/enc_fp16.onnx');

  console.log('[PASS] Test 1: selectV2Target (webgpu+f16, webgpu-int4, webnn-only, wasm-only, overrides)');
}

// ---------------------------------------------------------------------------
// Test 2: runV2DynamicDecodeLoop (0 KV getData, exact step_pos rewind,
//         All-T MTP-4 draft indexing at acceptedCount - 1, 0 T=1 fallbacks!)
// ---------------------------------------------------------------------------

async function testDynamicDecodeLoop() {
  const VOCAB = 16;
  const BOS = 14;
  const EOS = 15;
  const LAYERS = 2;
  const KV_HEADS = 2;
  const HEAD_DIM = 8;

  let kvGetDataCalls = 0;
  const onKvGetData = () => {
    kvGetDataCalls += 1;
  };

  // GPU-resident cross-KV tensors
  const crossKv = {};
  for (let i = 0; i < LAYERS; i++) {
    crossKv[`cross_k_${i}`] = new MockTensor('float32', null, [1, KV_HEADS, 6, HEAD_DIM], {
      gpuResident: true,
      onGetData: onKvGetData,
    });
    crossKv[`cross_v_${i}`] = new MockTensor('float32', null, [1, KV_HEADS, 6, HEAD_DIM], {
      gpuResident: true,
      onGetData: onKvGetData,
    });
  }

  // Scripted 4-step MTP-4 scenario:
  //   Step 0 (Prefill BOS, T=1, step_pos=0):
  //     logits[0] = 1
  //     mtp[0..2] at pos 0 -> drafts [2, 3, 4]
  //   Step 1 (Verify [1, 2, 3, 4], T=4, step_pos=1):
  //     logits = [2, 9, 8, 7] -> v0=2 matches d1=2, v1=9 != d2=3 -> PARTIAL ACCEPT (acceptedCount=2)!
  //     Accepted tokens so far: [1, 2, 9], acceptedLen = 1 + 2 = 3 (present_k len was 1+4=5!).
  //     All-T MTP at index `acceptedCount - 1 = 1` gives drafts [10, 11, 12]
  //     (while index 0, 2, 3 have distinct decoy drafts to prove exact index=1 is read!).
  //   Step 2 (Verify [9, 10, 11, 12], T=4, step_pos=3 — REWOUND from past_k len 5!):
  //     logits = [10, 11, 12, 13] -> ALL 3 DRAFTS MATCH + bonus 13 (acceptedCount=4)!
  //     Accepted tokens so far: [1, 2, 9, 10, 11, 12, 13], acceptedLen = 3 + 4 = 7.
  //     All-T MTP at index `acceptedCount - 1 = 3` gives drafts [5, 6, 7].
  //   Step 3 (Verify [13, 5, 6, 7], T=4, step_pos=7):
  //     logits = [5, EOS, 1, 1] -> v0=5 matches d1=5, v1=EOS -> STOP!
  //     Final tokens: [1, 2, 9, 10, 11, 12, 13, 5]
  const trace = [];
  const decoderSession = {
    async run(feeds) {
      const stepIdx = trace.length;
      const tokenIds = Array.from(feeds.token_ids.data, (x) => Number(x));
      const stepPos = Number(feeds.step_pos.data[0]);
      const pastLen = feeds.past_k_0.dims[2];
      trace.push({ stepIdx, tokenIds, stepPos, pastLen });

      const T = tokenIds.length;
      const presentLen = stepPos + T;

      let logits;
      let mtp0;
      let mtp1;
      let mtp2;

      if (stepIdx === 0) {
        assert.deepEqual(tokenIds, [BOS]);
        assert.equal(stepPos, 0);
        logits = makeOneHotLogits(VOCAB, [1]);
        mtp0 = makeOneHotLogits(VOCAB, [2]);
        mtp1 = makeOneHotLogits(VOCAB, [3]);
        mtp2 = makeOneHotLogits(VOCAB, [4]);
      } else if (stepIdx === 1) {
        assert.deepEqual(tokenIds, [1, 2, 3, 4]);
        assert.equal(stepPos, 1);
        assert.equal(pastLen, 1);
        // Partial accept: v0=2 (matches), v1=9 (mismatch vs d2=3) -> acceptedCount=2, draft index=1
        logits = makeOneHotLogits(VOCAB, [2, 9, 8, 7]);
        // Put distinct decoys at indices 0, 2, 3; real drafts [10, 11, 12] ONLY at index 1!
        mtp0 = makeOneHotLogits(VOCAB, [0, 10, 0, 0]);
        mtp1 = makeOneHotLogits(VOCAB, [0, 11, 0, 0]);
        mtp2 = makeOneHotLogits(VOCAB, [0, 12, 0, 0]);
      } else if (stepIdx === 2) {
        // Must have immediately used All-T drafts [10, 11, 12] (NO T=1 fallback!)
        // AND must have passed step_pos=3 while past_k still has GPU length 5 (zero-copy rewind!)
        assert.deepEqual(tokenIds, [9, 10, 11, 12]);
        assert.equal(stepPos, 3);
        assert.equal(pastLen, 5, 'past_k must stay unsliced on GPU (len=5) while step_pos rewinds to 3');
        logits = makeOneHotLogits(VOCAB, [10, 11, 12, 13]);
        // All 4 accepted -> draft index = 3
        mtp0 = makeOneHotLogits(VOCAB, [0, 0, 0, 5]);
        mtp1 = makeOneHotLogits(VOCAB, [0, 0, 0, 6]);
        mtp2 = makeOneHotLogits(VOCAB, [0, 0, 0, 7]);
      } else if (stepIdx === 3) {
        assert.deepEqual(tokenIds, [13, 5, 6, 7]);
        assert.equal(stepPos, 7);
        assert.equal(pastLen, 7);
        logits = makeOneHotLogits(VOCAB, [5, EOS, 1, 1]);
        mtp0 = makeOneHotLogits(VOCAB, [0, 0, 0, 0]);
        mtp1 = makeOneHotLogits(VOCAB, [0, 0, 0, 0]);
        mtp2 = makeOneHotLogits(VOCAB, [0, 0, 0, 0]);
      } else {
        throw new Error(`Unexpected extra decode step ${stepIdx}`);
      }

      const out = {
        logits,
        mtp_logits_0: mtp0,
        mtp_logits_1: mtp1,
        mtp_logits_2: mtp2,
      };
      for (let i = 0; i < LAYERS; i++) {
        out[`present_k_${i}`] = new MockTensor('float32', null, [1, KV_HEADS, presentLen, HEAD_DIM], {
          gpuResident: true,
          onGetData: onKvGetData,
        });
        out[`present_v_${i}`] = new MockTensor('float32', null, [1, KV_HEADS, presentLen, HEAD_DIM], {
          gpuResident: true,
          onGetData: onKvGetData,
        });
      }
      return out;
    },
  };

  const fsa = new MockFsa(VOCAB);
  const res = await runV2DynamicDecodeLoop({
    decoderSession,
    crossKv,
    bosId: BOS,
    eosId: EOS,
    maxTokens: 64,
    mtpK: 4,
    fsa,
    ort: mockOrt,
  });

  assert.equal(kvGetDataCalls, 0, 'MUST have 0 getData() calls on GPU-resident KV tensors');
  assert.equal(res.t1FallbackSteps, 0, 'MUST have 0 T=1 fallback steps after partial accept');
  assert.equal(res.hitEos, true);
  assert.equal(res.steps, 4);
  assert.deepEqual(res.tokenIds, [1, 2, 9, 10, 11, 12, 13, 5]);
  assert.deepEqual(fsa.history, [1, 2, 9, 10, 11, 12, 13, 5, EOS]);
  // Verify step 2 rewound step_pos from pastLen=5 to stepPos=3
  assert.equal(trace[2].pastLen, 5);
  assert.equal(trace[2].stepPos, 3);

  console.log(
    `[PASS] Test 2: runV2DynamicDecodeLoop (0 KV getData calls, 0 T=1 fallbacks, exact step_pos rewind 5->3, All-T draft idx=1 & 3, ${res.steps} steps -> ${res.tokenIds.length} tokens)`,
  );
}

// ---------------------------------------------------------------------------
// Test 3: runV2NpuStaticDecodeLoop (100% constant KV shapes + clean <eos> stop)
// ---------------------------------------------------------------------------

async function testNpuStaticDecodeLoop() {
  const VOCAB = 16;
  const BOS = 14;
  const EOS = 15;
  const LAYERS = 2;
  const KV_HEADS = 2;
  const HEAD_DIM = 8;
  const KV_MAX_LEN = 32;

  const expectedKvDims = [1, KV_HEADS, KV_MAX_LEN, HEAD_DIM];
  const crossKv = {};
  for (let i = 0; i < LAYERS; i++) {
    crossKv[`cross_k_${i}`] = new MockTensor('float32', new Float32Array(KV_HEADS * 16 * HEAD_DIM), [1, KV_HEADS, 16, HEAD_DIM]);
    crossKv[`cross_v_${i}`] = new MockTensor('float32', new Float32Array(KV_HEADS * 16 * HEAD_DIM), [1, KV_HEADS, 16, HEAD_DIM]);
  }

  const emitted = [3, 7, 11, 4, EOS];
  const seenStepPos = [];
  const seenKvShapes = [];

  const decoderSession = {
    async run(feeds) {
      const idx = seenStepPos.length;
      assert.deepEqual(feeds.token_ids.dims, [1, 1], 'NPU token_ids shape must stay constant [1, 1]');
      assert.deepEqual(feeds.step_pos.dims, [1], 'NPU step_pos shape must stay constant [1]');
      for (let i = 0; i < LAYERS; i++) {
        assert.deepEqual(feeds[`past_k_${i}`].dims, expectedKvDims);
        assert.deepEqual(feeds[`past_v_${i}`].dims, expectedKvDims);
      }
      seenStepPos.push(Number(feeds.step_pos.data[0]));
      seenKvShapes.push(feeds.past_k_0.dims.slice());

      const out = {
        logits: makeOneHotLogits(VOCAB, [emitted[idx]]),
      };
      for (let i = 0; i < LAYERS; i++) {
        // Static ring buffer outputs keep exact same shape [1, KV_HEADS, KV_MAX_LEN, HEAD_DIM].
        out[`present_k_${i}`] = new MockTensor(
          'float32',
          new Float32Array(KV_HEADS * KV_MAX_LEN * HEAD_DIM),
          expectedKvDims,
        );
        out[`present_v_${i}`] = new MockTensor(
          'float32',
          new Float32Array(KV_HEADS * KV_MAX_LEN * HEAD_DIM),
          expectedKvDims,
        );
      }
      return out;
    },
  };

  const fsa = new MockFsa(VOCAB);
  const res = await runV2NpuStaticDecodeLoop({
    decoderSession,
    crossKv,
    kvMaxLen: KV_MAX_LEN,
    nKvHeads: KV_HEADS,
    headDim: HEAD_DIM,
    nDecLayers: LAYERS,
    bosId: BOS,
    eosId: EOS,
    maxTokens: 64,
    fsa,
    ort: mockOrt,
  });

  assert.equal(res.hitEos, true);
  assert.equal(res.steps, 5);
  assert.deepEqual(res.tokenIds, [3, 7, 11, 4]);
  assert.deepEqual(seenStepPos, [0, 1, 2, 3, 4]);
  for (const shp of seenKvShapes) {
    assert.deepEqual(shp, expectedKvDims);
  }

  console.log(
    `[PASS] Test 3: runV2NpuStaticDecodeLoop (constant ${JSON.stringify(expectedKvDims)} across ${res.steps} steps, clean <eos> stop)`,
  );
}

// ---------------------------------------------------------------------------
// Run all tests
// ---------------------------------------------------------------------------

async function main() {
  testSelectV2Target();
  await testDynamicDecodeLoop();
  await testNpuStaticDecodeLoop();
  console.log('ALL v2_runtime TESTS PASSED (3/3)');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
