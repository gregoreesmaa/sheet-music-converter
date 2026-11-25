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
  sanitizeBekernForMscz,
  compileBekernWithFallback,
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

console.log('ALL TESTS PASSED');
