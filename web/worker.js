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
    if (typeof fetch !== 'function' || typeof self === 'undefined') return;
    const body = JSON.stringify({ level, src: 'worker', msg: String(msg ?? '').slice(0, 4000), extra: extra || undefined });
    fetch('/__log', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body }).catch(() => {});
  } catch (_) {}
}
if (typeof self !== 'undefined' && typeof self.addEventListener === 'function') {
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
}

const DEFAULT_MAX_TOKENS = 3072;

// ---------------------------------------------------------------------------
// Bekern+ FSA grammar mask (mirrors tmp-train/mvd/grammar.py & mscz-rs)
// ---------------------------------------------------------------------------

const CLS_SPECIAL = 1 << 0;
const CLS_EOS = 1 << 1;
const CLS_SYS = 1 << 2;
const CLS_HEADER = 1 << 3;
const CLS_BAR_CLOSE = 1 << 4;
const CLS_BAR_LEFT = 1 << 5;
const CLS_VOICE = 1 << 6;
const CLS_TUPLET_START = 1 << 7;
const CLS_TUPLET_STOP = 1 << 8;
const CLS_DYN_HAIR = 1 << 9;
const CLS_GRACE = 1 << 10;
const CLS_CHORD = 1 << 11;
const CLS_DUR = 1 << 12;
const CLS_REST = 1 << 13;
const CLS_BAR_REST = 1 << 14;
const CLS_STEP = 1 << 15;
const CLS_OCT = 1 << 16;
const CLS_ALTER = 1 << 17;
const CLS_DOT = 1 << 18;
const CLS_MOD = 1 << 19;
const CLS_TIE = 1 << 20;

const FSA_HEADER = 0;
const FSA_BODY = 1;
const FSA_NEED_DUR_NOTE = 2; // after chord or grace
const FSA_NEED_DUR_REST = 3; // after rest
const FSA_NEED_STEP = 4; // after pitched duration
const FSA_NEED_OCT = 5; // after pitch step A..G
const FSA_PITCH_TAIL = 6; // after octave
const FSA_REST_TAIL = 7; // after rest + duration

const MAX_CHORDS_PER_ONSET = 8;
const MAX_CONSECUTIVE_GRACE = 8;
const MAX_DOTS_PER_EVENT = 3;

const DUR_SET = new Set(['breve', 'whole', 'half', 'quarter', 'eighth', '16th', '32nd', '64th', '128th']);
const STEP_SET = new Set(['A', 'B', 'C', 'D', 'E', 'F', 'G']);
const OCT_SET = new Set(['-1', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']);
const ALTER_SET = new Set(['a-2', 'a-1', 'a1', 'a2']);
const MOD_SET = new Set(['slur-start', 'slur-stop', 'stacc', 'accent', 'tenuto', 'fermata']);
const TIE_SET = new Set(['tie-start', 'tie-stop']);
const BAR_CLOSE_SET = new Set(['bar', 'bar-light-heavy', 'bar-repeat-right', 'bar-double']);
const DYN_SET = new Set(['dyn-pp', 'dyn-p', 'dyn-mp', 'dyn-mf', 'dyn-f', 'dyn-ff', 'dyn-sfz']);
const HAIRPIN_SET = new Set(['hairpin-cresc-start', 'hairpin-dim-start', 'hairpin-stop']);

function isValidClef(t) {
  if (!t.startsWith('clef-')) return false;
  const parts = t.slice(5).split('-');
  if (parts.length < 2 || parts.length > 3) return false;
  if (!parts[0] || parts[0].length !== 1) return false;
  if (!/^\d+$/.test(parts[1])) return false;
  if (parts.length === 3 && parts[2] !== '8vb' && parts[2] !== '8va') return false;
  return true;
}

function isValidKey(t) {
  return /^key[+-]?\d+$/.test(t);
}

function isValidTime(t) {
  const m = /^time-(\d+)-(\d+)$/.exec(t);
  return Boolean(m && Number(m[1]) > 0 && Number(m[2]) > 0);
}

function parseVoiceNum(t) {
  if (!t.startsWith('voice-')) return 0;
  const n = Number(t.slice(6));
  return Number.isInteger(n) && n >= 1 ? n : 0;
}

function isValidTupletStart(t) {
  const m = /^tuplet-start-(\d+)-(\d+)$/.exec(t);
  return Boolean(m && Number(m[1]) > 0 && Number(m[2]) > 0);
}

function classifyBekernToken(tok) {
  if (tok === '<pad>' || tok === '<bos>') return CLS_SPECIAL;
  if (tok === '<eos>') return CLS_EOS;
  if (tok === '<sys>') return CLS_SYS;
  if (isValidClef(tok) || isValidKey(tok) || isValidTime(tok)) return CLS_HEADER;
  if (BAR_CLOSE_SET.has(tok)) return CLS_BAR_CLOSE;
  if (tok === 'bar-repeat-left') return CLS_BAR_LEFT;
  if (parseVoiceNum(tok) > 0) return CLS_VOICE;
  if (isValidTupletStart(tok)) return CLS_TUPLET_START;
  if (tok === 'tuplet-stop') return CLS_TUPLET_STOP;
  if (DYN_SET.has(tok) || HAIRPIN_SET.has(tok)) return CLS_DYN_HAIR;
  if (tok === 'grace') return CLS_GRACE;
  if (tok === 'chord') return CLS_CHORD;
  if (DUR_SET.has(tok)) return CLS_DUR;
  if (tok === 'rest') return CLS_REST;
  if (tok === 'bar-rest') return CLS_BAR_REST;
  if (STEP_SET.has(tok)) return CLS_STEP;
  if (OCT_SET.has(tok)) return CLS_OCT;
  if (ALTER_SET.has(tok)) return CLS_ALTER;
  if (tok === 'dot') return CLS_DOT;
  if (MOD_SET.has(tok)) return CLS_MOD;
  if (TIE_SET.has(tok)) return CLS_TIE;
  return 0;
}

export class BekernFsa {
  constructor(idToToken) {
    this.vocabSize = idToToken.length;
    this.tokenClass = new Uint32Array(this.vocabSize);
    this.voiceNum = new Uint16Array(this.vocabSize);
    for (let i = 0; i < this.vocabSize; i++) {
      const tok = idToToken[i] ?? '';
      this.tokenClass[i] = classifyBekernToken(tok);
      if ((this.tokenClass[i] & CLS_VOICE) !== 0) {
        this.voiceNum[i] = parseVoiceNum(tok);
      }
    }
    this.reset();
  }

  reset() {
    this.mode = FSA_HEADER;
    this.seg = 0;
    this.segLast = [false];
    this.tupletDepth = [0];
    this.pendingTuplets = 0;
    this.chordCount = 0;
    this.graceCount = 0;
    this.alterSeen = false;
    this.dotCount = 0;
    this.modOrTieSeen = false;
    this.lastTokenId = -1;
  }

  _growSeg(n) {
    while (this.segLast.length < n) {
      this.segLast.push(false);
      this.tupletDepth.push(0);
    }
  }

  _drainTuplets() {
    if (this.pendingTuplets > 0) {
      this.tupletDepth[this.seg] += this.pendingTuplets;
      this.pendingTuplets = 0;
    }
  }

  _boundaryMask(placedOverride) {
    let mask =
      CLS_HEADER |
      CLS_BAR_CLOSE |
      CLS_BAR_LEFT |
      CLS_VOICE |
      CLS_TUPLET_START |
      CLS_DYN_HAIR |
      CLS_DUR |
      CLS_REST |
      CLS_BAR_REST |
      CLS_EOS |
      CLS_SYS;
    const placed = placedOverride !== null ? placedOverride : this.segLast[this.seg];
    if (placed && this.chordCount < MAX_CHORDS_PER_ONSET) {
      mask |= CLS_CHORD;
    }
    if (this.graceCount < MAX_CONSECUTIVE_GRACE) {
      mask |= CLS_GRACE;
    }
    if (this.tupletDepth[this.seg] > 0) {
      mask |= CLS_TUPLET_STOP;
    }
    return mask;
  }

  getAllowedMask() {
    switch (this.mode) {
      case FSA_HEADER:
      case FSA_BODY:
        return this._boundaryMask(null);
      case FSA_NEED_DUR_NOTE:
      case FSA_NEED_DUR_REST:
        return CLS_DUR;
      case FSA_NEED_STEP:
        return CLS_STEP;
      case FSA_NEED_OCT:
        return CLS_OCT;
      case FSA_PITCH_TAIL: {
        let mask = this._boundaryMask(true) | CLS_MOD | CLS_TIE;
        if (!this.alterSeen && this.dotCount === 0 && !this.modOrTieSeen) {
          mask |= CLS_ALTER;
        }
        if (!this.modOrTieSeen && this.dotCount < MAX_DOTS_PER_EVENT) {
          mask |= CLS_DOT;
        }
        return mask;
      }
      case FSA_REST_TAIL: {
        let mask = this._boundaryMask(false) | CLS_MOD;
        if (!this.modOrTieSeen && this.dotCount < MAX_DOTS_PER_EVENT) {
          mask |= CLS_DOT;
        }
        return mask;
      }
      default:
        return this._boundaryMask(null);
    }
  }

  selectNextToken(logitsData, vocabSize) {
    const n = Math.min(vocabSize, this.vocabSize);
    const offset = Math.max(0, logitsData.length - vocabSize);
    const allowMask = this.getAllowedMask();
    const forbidVoiceId =
      this.lastTokenId >= 0 && (this.tokenClass[this.lastTokenId] & CLS_VOICE) !== 0
        ? this.lastTokenId
        : -1;
    const cls = this.tokenClass;

    let bestIdx = -1;
    let bestVal = -Infinity;
    for (let i = 0; i < n; i++) {
      if ((cls[i] & allowMask) === 0 || i === forbidVoiceId) continue;
      const v = logitsData[offset + i];
      if (v > bestVal) {
        bestVal = v;
        bestIdx = i;
      }
    }
    if (bestIdx >= 0) return bestIdx;

    // Fallback (should never trigger unless logits are all NaN/-Inf)
    let fbIdx = 0;
    let fbVal = -Infinity;
    for (let i = 0; i < vocabSize; i++) {
      const v = logitsData[offset + i];
      if (v > fbVal) {
        fbVal = v;
        fbIdx = i;
      }
    }
    return fbIdx;
  }

  _stepBoundary(id, c) {
    if ((c & CLS_SYS) !== 0) {
      this.reset();
      return;
    }
    if ((c & CLS_EOS) !== 0) {
      return;
    }
    if ((c & CLS_HEADER) !== 0) {
      // Remain in HEADER if still in HEADER, otherwise stay in BODY
      this.mode = this.mode === FSA_HEADER ? FSA_HEADER : FSA_BODY;
      return;
    }
    if ((c & CLS_BAR_CLOSE) !== 0) {
      this.seg = 0;
      this.segLast = [false];
      this.tupletDepth = [0];
      this.chordCount = 0;
      this.graceCount = 0;
      this.mode = FSA_BODY;
      return;
    }
    if ((c & CLS_VOICE) !== 0) {
      const vn = this.voiceNum[id] || 1;
      this._growSeg(vn);
      this.seg = vn - 1;
      this.chordCount = 0;
      this.graceCount = 0;
      this.mode = FSA_BODY;
      return;
    }
    if ((c & CLS_TUPLET_START) !== 0) {
      this.pendingTuplets += 1;
      this.mode = FSA_BODY;
      return;
    }
    if ((c & CLS_TUPLET_STOP) !== 0) {
      if (this.tupletDepth[this.seg] > 0) this.tupletDepth[this.seg] -= 1;
      this.mode = FSA_BODY;
      return;
    }
    if ((c & (CLS_DYN_HAIR | CLS_BAR_LEFT)) !== 0) {
      this.mode = FSA_BODY;
      return;
    }
    if ((c & CLS_GRACE) !== 0) {
      this.graceCount += 1;
      this.chordCount = 0;
      this.mode = FSA_NEED_DUR_NOTE;
      return;
    }
    if ((c & CLS_CHORD) !== 0) {
      this.chordCount += 1;
      this.mode = FSA_NEED_DUR_NOTE;
      return;
    }
    if ((c & CLS_DUR) !== 0) {
      this._drainTuplets();
      this.chordCount = 0;
      this.graceCount = 0;
      this.alterSeen = false;
      this.dotCount = 0;
      this.modOrTieSeen = false;
      this.mode = FSA_NEED_STEP;
      return;
    }
    if ((c & CLS_REST) !== 0) {
      this._drainTuplets();
      this.chordCount = 0;
      this.graceCount = 0;
      this.dotCount = 0;
      this.modOrTieSeen = false;
      this.mode = FSA_NEED_DUR_REST;
      return;
    }
    if ((c & CLS_BAR_REST) !== 0) {
      this._drainTuplets();
      this.segLast[this.seg] = false;
      this.chordCount = 0;
      this.graceCount = 0;
      this.mode = FSA_BODY;
    }
  }

  step(id) {
    if (id < 0 || id >= this.vocabSize) return;
    const c = this.tokenClass[id];
    if ((c & CLS_SPECIAL) !== 0) return;

    if ((c & CLS_SYS) !== 0) {
      this.reset();
      return;
    }

    switch (this.mode) {
      case FSA_HEADER:
      case FSA_BODY:
        this._stepBoundary(id, c);
        break;
      case FSA_NEED_DUR_NOTE:
        if ((c & CLS_DUR) !== 0) {
          this._drainTuplets();
          this.alterSeen = false;
          this.dotCount = 0;
          this.modOrTieSeen = false;
          this.mode = FSA_NEED_STEP;
        }
        break;
      case FSA_NEED_DUR_REST:
        if ((c & CLS_DUR) !== 0) {
          this.dotCount = 0;
          this.modOrTieSeen = false;
          this.mode = FSA_REST_TAIL;
        }
        break;
      case FSA_NEED_STEP:
        if ((c & CLS_STEP) !== 0) {
          this.mode = FSA_NEED_OCT;
        }
        break;
      case FSA_NEED_OCT:
        if ((c & CLS_OCT) !== 0) {
          this.segLast[this.seg] = true;
          this.mode = FSA_PITCH_TAIL;
        }
        break;
      case FSA_PITCH_TAIL:
        if ((c & CLS_ALTER) !== 0) {
          this.alterSeen = true;
        } else if ((c & CLS_DOT) !== 0) {
          this.dotCount += 1;
        } else if ((c & (CLS_MOD | CLS_TIE)) !== 0) {
          this.modOrTieSeen = true;
        } else {
          this.segLast[this.seg] = true;
          this.mode = FSA_BODY;
          this._stepBoundary(id, c);
        }
        break;
      case FSA_REST_TAIL:
        if ((c & CLS_DOT) !== 0) {
          this.dotCount += 1;
        } else if ((c & CLS_MOD) !== 0) {
          this.modOrTieSeen = true;
        } else {
          this.segLast[this.seg] = false;
          this.mode = FSA_BODY;
          this._stepBoundary(id, c);
        }
        break;
      default:
        break;
    }

    this.lastTokenId = id;
  }
}

// ---------------------------------------------------------------------------
// Defense-in-depth sanitizer + measure-by-measure fallback for mscz-rs WASM
// ---------------------------------------------------------------------------

function sanitizeSingleSystem(tokens) {
  const out = [];
  let seg = 0;
  let segLast = [false];
  let tupletDepth = [0];
  let pendingTuplets = 0;
  let lastVoiceTok = null;

  const growSeg = (n) => {
    while (segLast.length < n) {
      segLast.push(false);
      tupletDepth.push(0);
    }
  };
  const drainTuplets = () => {
    if (pendingTuplets > 0) {
      tupletDepth[seg] += pendingTuplets;
      pendingTuplets = 0;
    }
  };

  let i = 0;
  const n = tokens.length;
  while (i < n) {
    const t = tokens[i];
    const cls = classifyBekernToken(t);

    if (cls === 0 || (cls & (CLS_SPECIAL | CLS_EOS | CLS_SYS)) !== 0) {
      i += 1;
      continue;
    }

    if ((cls & CLS_HEADER) !== 0) {
      out.push(t);
      lastVoiceTok = null;
      i += 1;
      continue;
    }

    if ((cls & CLS_BAR_CLOSE) !== 0) {
      out.push(t);
      seg = 0;
      segLast = [false];
      tupletDepth = [0];
      lastVoiceTok = null;
      i += 1;
      continue;
    }

    if ((cls & (CLS_BAR_LEFT | CLS_DYN_HAIR)) !== 0) {
      out.push(t);
      lastVoiceTok = null;
      i += 1;
      continue;
    }

    if ((cls & CLS_VOICE) !== 0) {
      if (t !== lastVoiceTok) {
        const vn = parseVoiceNum(t);
        growSeg(vn);
        seg = vn - 1;
        out.push(t);
        lastVoiceTok = t;
      }
      i += 1;
      continue;
    }

    if ((cls & CLS_TUPLET_START) !== 0) {
      pendingTuplets += 1;
      out.push(t);
      lastVoiceTok = null;
      i += 1;
      continue;
    }

    if ((cls & CLS_TUPLET_STOP) !== 0) {
      if (tupletDepth[seg] > 0) {
        tupletDepth[seg] -= 1;
        out.push(t);
        lastVoiceTok = null;
      }
      i += 1;
      continue;
    }

    if ((cls & CLS_BAR_REST) !== 0) {
      drainTuplets();
      segLast[seg] = false;
      out.push(t);
      lastVoiceTok = null;
      i += 1;
      continue;
    }

    // Rest event: rest DUR [dot*] [mod*]
    if ((cls & CLS_REST) !== 0) {
      if (i + 1 < n && DUR_SET.has(tokens[i + 1])) {
        drainTuplets();
        segLast[seg] = false;
        out.push(t, tokens[i + 1]);
        i += 2;
        while (i < n && tokens[i] === 'dot') {
          out.push(tokens[i]);
          i += 1;
        }
        while (i < n && MOD_SET.has(tokens[i])) {
          out.push(tokens[i]);
          i += 1;
        }
        lastVoiceTok = null;
      } else {
        // Incomplete/dangling rest at tail or malformed token after rest
        i += 1;
      }
      continue;
    }

    // Pitched note event: [grace] [chord] DUR STEP OCT [alter] [dot*] [mod|tie]*
    if ((cls & (CLS_GRACE | CLS_CHORD | CLS_DUR)) !== 0) {
      let j = i;
      let hasGrace = false;
      let hasChord = false;
      if (tokens[j] === 'grace') {
        hasGrace = true;
        j += 1;
      } else if (tokens[j] === 'chord') {
        hasChord = true;
        j += 1;
      }
      if (j + 2 < n && DUR_SET.has(tokens[j]) && STEP_SET.has(tokens[j + 1]) && OCT_SET.has(tokens[j + 2])) {
        drainTuplets();
        if (hasGrace) out.push('grace');
        if (hasChord && segLast[seg]) out.push('chord');
        out.push(tokens[j], tokens[j + 1], tokens[j + 2]);
        j += 3;
        if (j < n && ALTER_SET.has(tokens[j])) {
          out.push(tokens[j]);
          j += 1;
        }
        while (j < n && tokens[j] === 'dot') {
          out.push(tokens[j]);
          j += 1;
        }
        while (j < n && (MOD_SET.has(tokens[j]) || TIE_SET.has(tokens[j]))) {
          out.push(tokens[j]);
          j += 1;
        }
        segLast[seg] = true;
        lastVoiceTok = null;
        i = j;
      } else {
        // Incomplete/truncated note (e.g. "quarter F", dangling "chord", dangling "grace")
        // Advance past the invalid prefix token(s) so any later valid tokens can recover.
        if (j < n && DUR_SET.has(tokens[j])) {
          i = j + 1;
          if (i < n && STEP_SET.has(tokens[i])) i += 1;
        } else {
          i += 1;
        }
      }
      continue;
    }

    // Stray bare step ("F"), bare octave ("4"), orphan alter/dot/mod/tie -> drop cleanly
    i += 1;
  }

  return out;
}

export function sanitizeBekernForMscz(rawText) {
  const raw = String(rawText ?? '').trim();
  if (!raw) return '';
  const allTokens = raw.split(/\s+/);
  const systems = [];
  let cur = [];
  for (const tok of allTokens) {
    if (tok === '<sys>') {
      if (cur.length > 0) {
        const clean = sanitizeSingleSystem(cur);
        if (clean.length > 0) systems.push(clean.join(' '));
        cur = [];
      }
    } else {
      cur.push(tok);
    }
  }
  if (cur.length > 0) {
    const clean = sanitizeSingleSystem(cur);
    if (clean.length > 0) systems.push(clean.join(' '));
  }
  return systems.join(' <sys> ');
}

function splitSystemIntoHeaderAndMeasures(sysText) {
  const toks = sysText.split(/\s+/).filter(Boolean);
  const header = [];
  let i = 0;
  while (i < toks.length && (classifyBekernToken(toks[i]) & CLS_HEADER) !== 0) {
    header.push(toks[i]);
    i += 1;
  }
  const measures = [];
  let cur = [];
  for (; i < toks.length; i++) {
    const t = toks[i];
    cur.push(t);
    if (BAR_CLOSE_SET.has(t)) {
      measures.push(cur);
      cur = [];
    }
  }
  if (cur.length > 0) measures.push(cur);
  return { header, measures };
}

export function compileBekernWithFallback(rawBekern, title, { toMscz = bekernToMscz, toMusicXml = bekernToMusicXml, onWarn = null } = {}) {
  const sanitized = sanitizeBekernForMscz(rawBekern);
  try {
    const msczBytes = toMscz(sanitized, title);
    const musicXml = toMusicXml(sanitized, title);
    return { msczBytes, musicXml, bekern: sanitized, usedFallback: false, droppedMeasures: 0 };
  } catch (firstErr) {
    if (typeof onWarn === 'function') {
      onWarn(`mscz-rs compile failed after sanitize (${firstErr?.message || firstErr}); retrying with measure-by-measure fallback`);
    }
    const rawSystems = sanitized.split(/\s*<sys>\s*/).filter(Boolean);
    const keptSystems = [];
    let droppedMeasures = 0;
    for (const sysText of rawSystems) {
      const { header, measures } = splitSystemIntoHeaderAndMeasures(sysText);
      const keptMeasures = [];
      for (const mToks of measures) {
        const probe = [...header, ...mToks].join(' ');
        try {
          toMscz(probe, title);
          keptMeasures.push(mToks.join(' '));
        } catch {
          droppedMeasures += 1;
        }
      }
      if (keptMeasures.length > 0) {
        keptSystems.push([...header, ...keptMeasures].join(' ').trim());
      } else if (header.length > 0) {
        keptSystems.push(header.join(' '));
      }
    }
    const recovered = keptSystems.join(' <sys> ');
    if (!recovered) throw firstErr;
    const msczBytes = toMscz(recovered, title);
    const musicXml = toMusicXml(recovered, title);
    if (typeof onWarn === 'function') {
      onWarn(`mscz-rs measure-by-measure fallback succeeded (dropped ${droppedMeasures} malformed measure(s))`);
    }
    return { msczBytes, musicXml, bekern: recovered, usedFallback: true, droppedMeasures };
  }
}

const state = {
  ort: null,
  wasmReady: false,
  modelsReady: false,
  backend: 'wasm',
  numThreads: 1,
  vocab: null,
  fsa: null,
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
      logSeverityLevel: 3,
    });
    return { session, usedWebGpu: preferWebGpu };
  } catch (err) {
    if (!preferWebGpu) throw err;
    console.warn(`WebGPU session creation failed for ${url}, falling back to WASM:`, err);
    const session = await ort.InferenceSession.create(url, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      logSeverityLevel: 3,
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
    ort.env.logLevel = 'error';
    ort.env.wasm.wasmPaths = ORT_CDN_BASE;
    ort.env.wasm.simd = true;
    const hw = typeof navigator !== 'undefined' && navigator.hardwareConcurrency
      ? navigator.hardwareConcurrency
      : 4;
    const canThread =
      typeof SharedArrayBuffer !== 'undefined' &&
      (typeof crossOriginIsolated === 'undefined' || crossOriginIsolated);
    const threads = canThread ? Math.min(8, hw) : 1;
    ort.env.wasm.numThreads = threads;
    state.numThreads = threads;

    // INT8 QUInt8 (DynamicQuantizeLinear/MatMulInteger) models run 5-7x faster
    // on multi-threaded WASM SIMD than WebGPU (which ping-pongs CPU<->GPU on
    // quantized ops). Default directly to WASM SIMD unless explicitly overridden.
    state.backend = 'wasm';
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
    state.fsa = new BekernFsa(state.vocab.idToToken);
    state.modelIo = vocabRaw.model_io_file
      ? (await fetchJsonOptional(`./models/${vocabRaw.model_io_file}`)) || {}
      : vocabRaw;

    const encoderUrl = state.modelIo.encoder_file
      ? `./models/${state.modelIo.encoder_file}`
      : './models/uni_omr_encoder.onnx';
    const decoderUrl = state.modelIo.decoder_file
      ? `./models/${state.modelIo.decoder_file}`
      : './models/uni_omr_decoder_step.onnx';

    const preferWebGpu = false;
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
      maxTokens: state.modelIo.max_tokens || DEFAULT_MAX_TOKENS,
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
  const maxTokens = state.modelIo.max_tokens || DEFAULT_MAX_TOKENS;
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
    const fsa = state.fsa || new BekernFsa(idToToken);
    fsa.reset();
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
      const nextId = fsa.selectNextToken(logitsTensor.data, vocabSize);
      fsa.step(nextId);

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

    const rawFinalTokens = stoppedByLoop ? trimToCleanBoundary(generatedTokens) : generatedTokens;
    const sanitizedBekern = sanitizeBekernForMscz(rawFinalTokens.join(' '));
    const finalTokenCount = sanitizedBekern ? sanitizedBekern.split(/\s+/).length : 0;
    const decodeMs = performance.now() - decodeStart;
    const tokPerSec = finalTokenCount / Math.max(0.001, decodeMs / 1000);

    self.postMessage({
      type: 'page-done',
      requestId,
      bekern: sanitizedBekern,
      tokens: finalTokenCount,
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
    const { msczBytes, musicXml } = compileBekernWithFallback(bekern, title, {
      onWarn: (warnMsg) => sendWorkerLog('warn', warnMsg),
    });
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

if (typeof self !== 'undefined') {
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
}
