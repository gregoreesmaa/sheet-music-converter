import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { performance } from 'node:perf_hooks';

// Polyfill file:// fetch so web/wasm/mscz_wasm.js loads web/wasm/mscz_wasm.wasm natively in Node
const origFetch = globalThis.fetch;
globalThis.fetch = async (url, opts) => {
  const u = typeof url === 'string' ? url : url.href;
  if (u.startsWith('file://')) {
    const buf = await fs.readFile(fileURLToPath(u));
    return {
      ok: true,
      status: 200,
      clone() { return this; },
      async arrayBuffer() { return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength); },
    };
  }
  return origFetch(url, opts);
};
// Disable instantiateStreaming in Node file:// shim so arrayBuffer fallback runs cleanly
const origInstantiateStreaming = WebAssembly.instantiateStreaming;
WebAssembly.instantiateStreaming = undefined;

import {
  initMsczWasm,
  bekernToMscz,
  bekernToMusicXml,
} from './wasm/mscz_wasm.js';
import {
  BekernFsa,
  draftMtpTokens,
  verifyMtpBatch,
  sanitizeBekernForMscz,
  compileBekernWithFallback,
  formatOrtError,
  extractPageTailState,
  applyCrossPageHeaderCarryover,
  prefillPrevStateKv,
} from './worker.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const vocabJson = JSON.parse(await fs.readFile(path.join(__dirname, 'models', 'vocab.json'), 'utf8'));
const idToToken = vocabJson.vocab;
const tokenToId = Object.fromEntries(idToToken.map((t, i) => [t, i]));
const vocabSize = idToToken.length;

function makeLogits(overrides) {
  const arr = new Float32Array(vocabSize).fill(-10.0);
  for (const [tok, val] of Object.entries(overrides)) {
    const id = tokenToId[tok];
    assert.notEqual(id, undefined, `Unknown vocab token: ${tok}`);
    arr[id] = val;
  }
  return arr;
}

function assertValidZip(bytes, label) {
  assert.ok(bytes instanceof Uint8Array && bytes.length > 100, `${label}: expected non-empty .mscz bytes`);
  assert.equal(bytes[0], 0x50, `${label}: ZIP magic[0] should be 'P'`);
  assert.equal(bytes[1], 0x4b, `${label}: ZIP magic[1] should be 'K'`);
  assert.equal(bytes[2], 0x03, `${label}: ZIP magic[2] should be 0x03`);
  assert.equal(bytes[3], 0x04, `${label}: ZIP magic[3] should be 0x04`);
}

// ---------------------------------------------------------------------------
// 1. BekernFsa grammar mask tests + latency check
// ---------------------------------------------------------------------------
{
  const fsa = new BekernFsa(idToToken);

  // At HEADER/BODY: bare 'F', bare '4', orphan 'a1', orphan 'dot', orphan 'chord', '<pad>' are forbidden
  const id0 = fsa.selectNextToken(
    makeLogits({ F: 100, '4': 99, a1: 98, dot: 97, chord: 96, '<pad>': 95, 'clef-G-2': 5 }),
    vocabSize,
  );
  assert.equal(idToToken[id0], 'clef-G-2');
  fsa.step(id0);

  // Transition to BODY on quarter -> must pick a pitch step A..G next (even if '4', 'bar', '<eos>' have higher logits)
  fsa.step(tokenToId['key+0']);
  fsa.step(tokenToId['time-4-4']);
  fsa.step(tokenToId['quarter']);
  const stepId = fsa.selectNextToken(
    makeLogits({ '4': 100, bar: 99, '<eos>': 98, dot: 97, F: 3 }),
    vocabSize,
  );
  assert.equal(idToToken[stepId], 'F');
  fsa.step(stepId);

  // NEED_OCT: immediately after 'F', ONLY octave allowed
  const octId = fsa.selectNextToken(
    makeLogits({ F: 100, bar: 99, '<eos>': 98, a1: 97, '4': 2 }),
    vocabSize,
  );
  assert.equal(idToToken[octId], '4');
  fsa.step(octId);

  // PITCH_TAIL: dot is allowed; once dot is emitted, alter (a1) is forbidden
  fsa.step(tokenToId['dot']);
  const afterDotId = fsa.selectNextToken(
    makeLogits({ a1: 100, stacc: 4 }),
    vocabSize,
  );
  assert.equal(idToToken[afterDotId], 'stacc');
  fsa.step(afterDotId);

  // Once mod (stacc) is emitted, dot is forbidden
  const afterModId = fsa.selectNextToken(
    makeLogits({ dot: 100, a1: 99, chord: 4 }),
    vocabSize,
  );
  assert.equal(idToToken[afterModId], 'chord');
  fsa.step(afterModId);

  // After chord, ONLY duration is allowed
  const chordDurId = fsa.selectNextToken(
    makeLogits({ A: 100, bar: 99, '<eos>': 98, quarter: 3 }),
    vocabSize,
  );
  assert.equal(idToToken[chordDurId], 'quarter');
  fsa.step(chordDurId);
  fsa.step(tokenToId['A']);
  fsa.step(tokenToId['4']);

  // Consecutive duplicate voice-1 is forbidden
  fsa.step(tokenToId['voice-1']);
  const dupVoiceId = fsa.selectNextToken(
    makeLogits({ 'voice-1': 100, 'voice-2': 5 }),
    vocabSize,
  );
  assert.equal(idToToken[dupVoiceId], 'voice-2');
  fsa.step(dupVoiceId);

  // Chord chain cap <= 8
  fsa.step(tokenToId['bar']);
  fsa.step(tokenToId['quarter']);
  fsa.step(tokenToId['C']);
  fsa.step(tokenToId['4']);
  for (let c = 0; c < 8; c++) {
    fsa.step(tokenToId['chord']);
    fsa.step(tokenToId['quarter']);
    fsa.step(tokenToId['E']);
    fsa.step(tokenToId['4']);
  }
  const cappedChordId = fsa.selectNextToken(makeLogits({ chord: 100, bar: 1 }), vocabSize);
  assert.equal(idToToken[cappedChordId], 'bar', 'chord chain should cap at 8');

  // Reset on <sys>
  fsa.step(tokenToId['<sys>']);
  const afterSysId = fsa.selectNextToken(makeLogits({ chord: 100, F: 99, 'clef-G-2': 1 }), vocabSize);
  assert.equal(idToToken[afterSysId], 'clef-G-2', '<sys> must reset FSA state cleanly');

  // Speed benchmark: 3072 steps
  fsa.reset();
  const benchLogits = new Float32Array(vocabSize);
  for (let i = 0; i < vocabSize; i++) benchLogits[i] = Math.sin(i * 1.3);
  const t0 = performance.now();
  for (let s = 0; s < 3072; s++) {
    const id = fsa.selectNextToken(benchLogits, vocabSize);
    fsa.step(id);
  }
  const elapsedMs = performance.now() - t0;
  const msPerStep = elapsedMs / 3072;
  assert.ok(msPerStep < 0.1, `BekernFsa too slow: ${msPerStep.toFixed(4)} ms/step`);
  console.log(`[PASS] BekernFsa rules & speed (${msPerStep.toFixed(4)} ms/step over 3072 steps)`);
}

// ---------------------------------------------------------------------------
// 2. Real mscz-rs WASM integration tests
// ---------------------------------------------------------------------------
await initMsczWasm(pathToFileURL(path.join(__dirname, 'wasm', 'mscz_wasm.wasm')));
WebAssembly.instantiateStreaming = origInstantiateStreaming;

// (a) 76-measure score with bare "F" injected in measure 76 (exact production repro!)
{
  const barToks = 'quarter C 4 quarter D 4 quarter E 4 quarter F 4 bar';
  const measures = Array.from({ length: 76 }, () => barToks);
  const bad76 = `clef-G-2 key+0 time-4-4 ${measures.join(' ')} F quarter G 4 quarter A 4 half B 4 bar-light-heavy`;

  // Confirm raw mscz-rs rejects it with the exact production error
  assert.throws(
    () => bekernToMscz(bad76, 'Repro 76'),
    /unexpected token "F" in measure 76/,
    'Raw bekernToMscz should reproduce the exact measure-76 error',
  );

  const res = compileBekernWithFallback(bad76, 'Repro 76');
  assertValidZip(res.msczBytes, '76-measure bare-F repro');
  assert.ok(res.musicXml.includes('<score-partwise'), '76-measure MusicXML should be valid');
  assert.equal(res.usedFallback, false, 'Sanitizer should clean bare F before fallback');
  console.log(`[PASS] (a) 76-measure bare "F" repro -> valid .mscz (${res.msczBytes.length} B) + MusicXML (${res.musicXml.length} chars)`);
}

// (b) Mid-note truncated sequences ending in "quarter F", "chord", "grace", "rest"
{
  const prefix = 'clef-G-2 key+1 time-4-4 quarter G 4 quarter A 4 eighth B 4 eighth C 5 quarter D 5 bar half E 5';
  for (const tail of ['quarter F', 'chord', 'chord quarter', 'chord quarter G', 'grace', 'grace eighth F', 'rest']) {
    const truncated = `${prefix} ${tail}`;
    assert.throws(() => bekernToMscz(truncated, 'Truncated'), `Raw bekernToMscz should reject dangling tail "${tail}"`);
    const res = compileBekernWithFallback(truncated, 'Truncated');
    assertValidZip(res.msczBytes, `truncated tail "${tail}"`);
    assert.ok(res.musicXml.includes('<score-partwise'));
  }
  console.log('[PASS] (b) Mid-note truncated tails (quarter F, chord, grace, rest) -> all compile to valid .mscz ZIP');
}

// (c) Normal multi-system repeats-melody sequence -> preserved identically & compiles identically
{
  const repeatsMelody =
    'clef-G-2 key+1 time-4-4 bar-repeat-left quarter G 4 quarter A 4 eighth B 4 eighth C 5 quarter D 5 bar ' +
    'half E 5 dot quarter D 5 bar-repeat-right <sys> ' +
    'clef-G-2 key+1 time-4-4 quarter C 5 stacc quarter B 4 accent quarter A 4 tenuto quarter G 4 fermata bar ' +
    'rest quarter quarter G 4 chord quarter B 4 chord quarter D 5 half G 4 bar-light-heavy';

  const sanitized = sanitizeBekernForMscz(repeatsMelody);
  assert.equal(sanitized, repeatsMelody, 'Valid multi-system repeats-melody should be unchanged by sanitizer');

  const rawMscz = bekernToMscz(repeatsMelody, 'Repeats Melody');
  const rawXml = bekernToMusicXml(repeatsMelody, 'Repeats Melody');
  const res = compileBekernWithFallback(repeatsMelody, 'Repeats Melody');
  assertValidZip(res.msczBytes, 'repeats-melody');
  assert.deepEqual(res.msczBytes, rawMscz, 'Compiled .mscz bytes must be identical');
  assert.equal(res.musicXml, rawXml, 'Compiled MusicXML must be identical');
  console.log(`[PASS] (c) Multi-system repeats-melody preserved & compiled identically (${res.msczBytes.length} B)`);
}

// (d) Measure-by-measure fallback drops only the single malformed measure and logs a warning
{
  const score =
    'clef-G-2 key+0 time-4-4 quarter C 4 half D 4 quarter E 4 bar ' +
    'quarter F 4 half G 4 quarter A 4 bar ' +
    'quarter B 4 half C 5 quarter D 5 bar-light-heavy';
  const warnings = [];
  const res = compileBekernWithFallback(score, 'Fallback Test', {
    toMscz: (txt, title) => {
      // Simulate an edge-case compiler failure on measure 2 only
      if (txt.includes('quarter F 4 half G 4')) {
        throw new Error('simulated edge-case error in measure 2');
      }
      return bekernToMscz(txt, title);
    },
    toMusicXml: bekernToMusicXml,
    onWarn: (m) => warnings.push(m),
  });
  assertValidZip(res.msczBytes, 'measure-by-measure fallback');
  assert.equal(res.usedFallback, true);
  assert.equal(res.droppedMeasures, 1);
  assert.ok(!res.bekern.includes('quarter F 4 half G 4'));
  assert.ok(res.bekern.includes('quarter C 4 half D 4'));
  assert.ok(res.bekern.includes('quarter B 4 half C 5'));
  assert.ok(warnings.length >= 2, 'Expected warning logs on fallback');
  console.log(`[PASS] (d) Measure-by-measure fallback dropped 1 malformed measure, kept 2 valid measures (${res.msczBytes.length} B)`);
}

// (e) formatOrtError formats numeric Emscripten C++ exception pointers with stage/dims/step context
{
  const msg1 = formatOrtError(75852400, { stage: 'encoding', width: 912, height: 1280, step: -1, backend: 'wasm', numThreads: 8 });
  assert.ok(msg1.includes('WASM runtime exception (likely out-of-memory, ptr=75852400)'), msg1);
  assert.ok(msg1.includes('stage=encoding') && msg1.includes('image=912x1280') && msg1.includes('backend=wasm(8T)'), msg1);

  const msg2 = formatOrtError(new Error('36622584'), { stage: 'decoding', width: 912, height: 1280, nMem: 14729, step: 42 });
  assert.ok(msg2.includes('WASM runtime exception (likely out-of-memory, ptr=36622584)'), msg2);
  assert.ok(msg2.includes('stage=decoding') && msg2.includes('n_mem=14729') && msg2.includes('step=42'), msg2);

  const msg3 = formatOrtError(new Error('invalid tensor shape'), { stage: 'encoding', width: 512, height: 256 });
  assert.ok(msg3.includes('invalid tensor shape') && msg3.includes('stage=encoding'), msg3);
  console.log('[PASS] (e) formatOrtError handles numeric WASM exception pointers and normal errors');
}

// ---------------------------------------------------------------------------
// 3. BekernFsa.clone() & MTP-4 speculative drafting / verification tests
// ---------------------------------------------------------------------------
{
  const eosId = vocabJson.eos_id;

  // (a) BekernFsa.clone() deep-copies mutable state without mutating the parent
  const fsa = new BekernFsa(idToToken);
  fsa.step(tokenToId['clef-G-2']);
  fsa.step(tokenToId['key+0']);
  fsa.step(tokenToId['time-4-4']);
  fsa.step(tokenToId['quarter']); // now in FSA_NEED_STEP

  const cloned = fsa.clone();
  cloned.step(tokenToId['C']);
  cloned.step(tokenToId['4']);
  cloned.step(tokenToId['chord']);
  cloned.step(tokenToId['quarter']);
  cloned.step(tokenToId['E']);
  cloned.step(tokenToId['4']);

  // Parent must still require a pitch step (A..G), not octave/chord
  const parentNext = fsa.selectNextToken(makeLogits({ '4': 100, bar: 99, G: 2 }), vocabSize);
  assert.equal(idToToken[parentNext], 'G', 'clone() must not mutate parent FSA state');

  // (b) draftMtpTokens drafts 3 FSA-valid tokens without mutating committed FSA
  const drafts = draftMtpTokens(
    fsa,
    [
      makeLogits({ '4': 100, C: 5 }), // FSA_NEED_STEP -> must pick 'C', not '4'
      makeLogits({ C: 100, '4': 5 }), // FSA_NEED_OCT -> must pick '4', not 'C'
      makeLogits({ F: 100, quarter: 5 }), // FSA_PITCH_TAIL -> must pick 'quarter', not bare 'F'
    ],
    vocabSize,
  );
  assert.deepEqual(
    drafts.map((id) => idToToken[id]),
    ['C', '4', 'quarter'],
    'draftMtpTokens must enforce FSA grammar across all 3 draft heads',
  );
  // Committed FSA must still be untouched (still in FSA_NEED_STEP)
  const stillStep = fsa.selectNextToken(makeLogits({ '4': 100, D: 2 }), vocabSize);
  assert.equal(idToToken[stillStep], 'D');

  // Helper to concat 4 logit slices into [1, 4, vocabSize]
  const concat4 = (a, b, c, d) => {
    const out = new Float32Array(4 * vocabSize);
    out.set(a, 0);
    out.set(b, vocabSize);
    out.set(c, 2 * vocabSize);
    out.set(d, 3 * vocabSize);
    return out;
  };

  // (c) All-4 accept case: drafts [C, 4, quarter] all match, bonus token 'D' emitted!
  const fsaAll = fsa.clone();
  const verAll = verifyMtpBatch(
    fsaAll,
    concat4(
      makeLogits({ '4': 100, C: 5 }), // j=0 -> 'C' (matches d1)
      makeLogits({ C: 100, '4': 5 }), // j=1 -> '4' (matches d2)
      makeLogits({ F: 100, quarter: 5 }), // j=2 -> 'quarter' (matches d3)
      makeLogits({ '4': 100, D: 5 }), // j=3 -> 'D' (bonus token after 'quarter')
    ),
    vocabSize,
    drafts,
    eosId,
  );
  assert.equal(verAll.allMatched, true);
  assert.equal(verAll.hitEos, false);
  assert.equal(verAll.acceptedCount, 4);
  assert.deepEqual(verAll.acceptedIds.map((id) => idToToken[id]), ['C', '4', 'quarter', 'D']);
  // FSA should now expect an octave for 'D'
  assert.equal(idToToken[fsaAll.selectNextToken(makeLogits({ bar: 100, '5': 1 }), vocabSize)], '5');

  // (d) Partial accept (2/4): d1='C' matches, d2='4' mismatches -> corrected to '5', stops at 2!
  const fsaPart = fsa.clone();
  const verPart = verifyMtpBatch(
    fsaPart,
    concat4(
      makeLogits({ C: 5 }), // j=0 -> 'C' (matches d1)
      makeLogits({ '5': 5 }), // j=1 -> '5' (mismatches d2='4'!)
      makeLogits({ quarter: 5 }), // j=2 (ignored)
      makeLogits({ D: 5 }), // j=3 (ignored)
    ),
    vocabSize,
    drafts,
    eosId,
  );
  assert.equal(verPart.allMatched, false);
  assert.equal(verPart.hitEos, false);
  assert.equal(verPart.acceptedCount, 2);
  assert.deepEqual(verPart.acceptedIds.map((id) => idToToken[id]), ['C', '5']);

  // (e) Mid-batch EOS: accepts up to and including EOS and sets hitEos=true
  const fsaEos = fsa.clone();
  const verEos = verifyMtpBatch(
    fsaEos,
    concat4(
      makeLogits({ C: 5 }), // j=0 -> 'C' (matches d1)
      makeLogits({ '4': 5 }), // j=1 -> '4' (matches d2)
      makeLogits({ '<eos>': 10 }), // j=2 -> '<eos>' (allowed in PITCH_TAIL!)
      makeLogits({ D: 5 }), // j=3 (ignored)
    ),
    vocabSize,
    drafts,
    eosId,
  );
  assert.equal(verEos.hitEos, true);
  assert.equal(verEos.allMatched, false);
  assert.equal(verEos.acceptedCount, 3);
  assert.deepEqual(verEos.acceptedIds.map((id) => idToToken[id]), ['C', '4', '<eos>']);

  console.log('[PASS] (f) BekernFsa.clone() + MTP-4 speculative drafting & verification (4/4, partial, EOS)');
}

// ---------------------------------------------------------------------------
// 4. Multi-Page Streaming Symbolic State Prefix & Cross-Page Header Carryover
// ---------------------------------------------------------------------------
{
  // (a) Single-staff multi-system tail state extraction (inherits time-*, tracks key change)
  const page1 =
    'clef-G-2 key+1 time-3-4 quarter G 4 quarter A 4 quarter B 4 bar <sys> ' +
    'clef-G-2 key-2 half B 4 quarter C 5 bar';
  assert.deepEqual(extractPageTailState(page1), ['clef-G-2', 'key-2', 'time-3-4']);

  // (b) Grand-staff tail state extraction
  const grandPage1 =
    'clef-G-2 key-1 time-4-4 quarter F 4 quarter G 4 half A 4 bar <sys> ' +
    'clef-F-4 key-1 time-4-4 whole F 2 bar <sys> ' +
    'clef-G-2 key-1 whole A 4 bar <sys> ' +
    'clef-F-4 key-1 half F 2 half C 3 bar';
  const grandTail = extractPageTailState(grandPage1);
  assert.deepEqual(grandTail, ['clef-G-2', 'key-1', 'time-4-4', '<sys>', 'clef-F-4', 'key-1', 'time-4-4']);

  // (c) Open tie, slur, hairpin, repeat, and split bar (<partial_bar>)
  const openTailPage =
    'clef-G-2 key+0 time-6-8 bar-repeat-left hairpin-cresc-start ' +
    'eighth C 4 slur-start eighth D 4 tie-start';
  assert.deepEqual(extractPageTailState(openTailPage), [
    'clef-G-2',
    'key+0',
    'time-6-8',
    'tie-start',
    'D',
    '4',
    'slur-start',
    'hairpin-cresc-start',
    'bar-repeat-left',
    '<partial_bar>',
  ]);

  // (d) Cross-page header carryover: Page 2 omits time-3-4 (and Page 2 sys 2 omits key & time)
  const page2Omitted =
    'clef-G-2 key-2 quarter D 5 half E 5 bar <sys> ' +
    'clef-G-2 quarter F 5 half G 5 bar-light-heavy';
  const carried = applyCrossPageHeaderCarryover(page2Omitted, extractPageTailState(page1));
  assert.equal(
    carried,
    'clef-G-2 key-2 time-3-4 quarter D 5 half E 5 bar <sys> ' +
      'clef-G-2 key-2 time-3-4 quarter F 5 half G 5 bar-light-heavy',
  );
  // Carried-over Page 2 compiles cleanly through real mscz-rs WASM!
  const compiledP2 = compileBekernWithFallback(carried, 'Page 2 Carried');
  assertValidZip(compiledP2.msczBytes, 'carried Page 2');
  assert.ok(compiledP2.musicXml.includes('<beats>3</beats>'), 'Page 2 MusicXML must inherit 3/4 meter');

  // (e) Grand-staff cross-page carryover: Page 2 omits time-4-4 on both staves
  const grandPage2 =
    'clef-G-2 key-1 whole G 4 bar <sys> ' +
    'clef-F-4 key-1 whole C 3 bar';
  assert.equal(
    applyCrossPageHeaderCarryover(grandPage2, grandTail),
    'clef-G-2 key-1 time-4-4 whole G 4 bar <sys> clef-F-4 key-1 time-4-4 whole C 3 bar',
  );

  // (f) prefillPrevStateKv: complete no-op when prevStateIds omitted OR <prev_state> not in vocab
  let runCalls = 0;
  const mockOrt = {
    Tensor: class MockTensor {
      constructor(type, data, dims) {
        this.type = type;
        this.data = data;
        this.dims = dims;
        this.disposed = false;
      }
      dispose() {
        this.disposed = true;
      }
    },
  };
  const mockSession = {
    async run(feeds) {
      runCalls += 1;
      const L = feeds.token_ids.dims[1];
      return {
        logits: new mockOrt.Tensor('float32', new Float32Array(0), [1, L, 10]),
        present_k_0: new mockOrt.Tensor('float32', new Float32Array(0), [1, 6, L, 64]),
        present_v_0: new mockOrt.Tensor('float32', new Float32Array(0), [1, 6, L, 64]),
      };
    },
  };
  const initK = [new mockOrt.Tensor('float32', new Float32Array(0), [1, 6, 0, 64])];
  const initV = [new mockOrt.Tensor('float32', new Float32Array(0), [1, 6, 0, 64])];
  const encOut = { cross_k_0: initK[0], cross_v_0: initV[0] };

  // No-op when prevStateIds is null
  const noop1 = await prefillPrevStateKv({
    ort: mockOrt,
    decoderSession: mockSession,
    encOutputs: encOut,
    pastK: initK,
    pastV: initV,
    nLayers: 1,
    prevStateIds: null,
    tokenToId: { ...tokenToId, '<prev_state>': 237 },
    useMtp: true,
  });
  assert.equal(noop1.prefilledLen, 0);
  assert.equal(runCalls, 0);

  // No-op when <prev_state> is absent from vocab.json (existing checkpoints)
  const noop2 = await prefillPrevStateKv({
    ort: mockOrt,
    decoderSession: mockSession,
    encOutputs: encOut,
    pastK: initK,
    pastV: initV,
    nLayers: 1,
    prevStateIds: [237, 33, 55, 63, 238],
    tokenToId,
    useMtp: true,
  });
  assert.equal(noop2.prefilledLen, 0);
  assert.equal(runCalls, 0);

  // Active 1-shot [1, L] WebGPU prefill when <prev_state> exists and prevStateIds provided
  const active = await prefillPrevStateKv({
    ort: mockOrt,
    decoderSession: mockSession,
    encOutputs: encOut,
    pastK: initK,
    pastV: initV,
    nLayers: 1,
    prevStateIds: [237, 33, 55, 63, 238],
    tokenToId: { ...tokenToId, '<prev_state>': 237, '</prev_state>': 238 },
    useMtp: true,
  });
  assert.equal(active.prefilledLen, 5);
  assert.equal(runCalls, 1, 'WebGPU prefill must execute in a single [1, L] call');
  assert.equal(active.pastK[0].dims[2], 5);
  assert.equal(active.pastV[0].dims[2], 5);

  console.log('[PASS] (g) Multi-page tail state extraction, cross-page header carryover & 1-shot KV prefill');
}

console.log('ALL TESTS PASSED');
