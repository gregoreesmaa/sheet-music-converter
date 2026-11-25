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
  bekernToMsczAndMusicXml,
  validateBekernSyntax,
} from './wasm/mscz_wasm.js';

const ORT_CDN_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/';
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

  clone() {
    const copy = Object.create(BekernFsa.prototype);
    copy.vocabSize = this.vocabSize;
    copy.tokenClass = this.tokenClass;
    copy.voiceNum = this.voiceNum;
    copy.mode = this.mode;
    copy.seg = this.seg;
    copy.segLast = this.segLast.slice();
    copy.tupletDepth = this.tupletDepth.slice();
    copy.pendingTuplets = this.pendingTuplets;
    copy.chordCount = this.chordCount;
    copy.graceCount = this.graceCount;
    copy.alterSeen = this.alterSeen;
    copy.dotCount = this.dotCount;
    copy.modOrTieSeen = this.modOrTieSeen;
    copy.lastTokenId = this.lastTokenId;
    return copy;
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

  selectNextToken(logitsData, vocabSize, offset = null) {
    const n = Math.min(vocabSize, this.vocabSize);
    const base = offset !== null ? offset : Math.max(0, logitsData.length - vocabSize);
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
      const v = logitsData[base + i];
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
      const v = logitsData[base + i];
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

// ---------------------------------------------------------------------------
// Multi-Page Streaming Symbolic State Prefix & Cross-Page Header Carryover
// (JS twin of tmp-train/mvd/data.py extract_page_tail_state)
// ---------------------------------------------------------------------------

const TAIL_SPECIALS = new Set([
  '<pad>',
  '<bos>',
  '<eos>',
  '<prev_state>',
  '</prev_state>',
  '<partial_bar>',
]);

function inferStaffCount(chunks) {
  const m = chunks.length;
  if (m <= 1) return m;

  const firstClefs = chunks.map((ch) => ch.find((t) => isValidClef(t)) ?? null);
  const hasTime = chunks.map((ch) => ch.some((t) => isValidTime(t)));
  const distinctClefs = new Set(firstClefs.filter((c) => c !== null));
  if (distinctClefs.size <= 1) return 1;

  for (let p = 2; p <= m; p++) {
    if (m % p !== 0) continue;
    let ok = true;
    for (let i = 0; i < m; i++) {
      const ci = firstClefs[i];
      const cp = firstClefs[i % p];
      if (ci !== null && cp !== null && ci !== cp) {
        ok = false;
        break;
      }
    }
    if (ok) return p;
  }

  let leadTime = 0;
  for (const ht of hasTime) {
    if (ht) leadTime += 1;
    else break;
  }
  if (leadTime >= 2 && m % leadTime === 0) return leadTime;

  return m;
}

function extractSingleStaffTail(chunks) {
  let activeClef = null;
  let activeKey = null;
  let activeTime = null;
  const openTies = [];
  let slurDepth = 0;
  let openHairpin = null;
  let openRepeat = false;
  let hasBarContent = false;

  let lastStep = null;
  let lastPitch = null;

  for (const ch of chunks) {
    for (const tok of ch) {
      if (TAIL_SPECIALS.has(tok)) continue;
      if (isValidClef(tok)) {
        activeClef = tok;
        lastStep = null;
        continue;
      }
      if (isValidKey(tok)) {
        activeKey = tok;
        lastStep = null;
        continue;
      }
      if (isValidTime(tok)) {
        activeTime = tok;
        lastStep = null;
        continue;
      }
      if (BAR_CLOSE_SET.has(tok)) {
        hasBarContent = false;
        if (tok === 'bar-repeat-right') openRepeat = false;
        lastStep = null;
        continue;
      }

      hasBarContent = true;

      if (tok === 'bar-repeat-left') {
        openRepeat = true;
        lastStep = null;
        continue;
      }
      if (tok === 'hairpin-cresc-start') {
        openHairpin = 'hairpin-cresc-start';
        lastStep = null;
        continue;
      }
      if (tok === 'hairpin-dim-start') {
        openHairpin = 'hairpin-dim-start';
        lastStep = null;
        continue;
      }
      if (tok === 'hairpin-stop') {
        openHairpin = null;
        lastStep = null;
        continue;
      }
      if (tok === 'slur-start') {
        slurDepth += 1;
        continue;
      }
      if (tok === 'slur-stop') {
        if (slurDepth > 0) slurDepth -= 1;
        continue;
      }
      if (STEP_SET.has(tok)) {
        lastStep = tok;
        continue;
      }
      if (OCT_SET.has(tok) && lastStep !== null) {
        lastPitch = `${lastStep} ${tok}`;
        lastStep = null;
        continue;
      }
      if (tok === 'tie-start') {
        if (lastPitch !== null && !openTies.includes(lastPitch)) {
          openTies.push(lastPitch);
        }
        continue;
      }
      if (tok === 'tie-stop') {
        if (lastPitch !== null) {
          const idx = openTies.indexOf(lastPitch);
          if (idx >= 0) openTies.splice(idx, 1);
        }
        continue;
      }
      if (!ALTER_SET.has(tok) && tok !== 'dot') {
        lastStep = null;
      }
    }
  }

  const out = [];
  if (activeClef !== null) out.push(activeClef);
  if (activeKey !== null) out.push(activeKey);
  if (activeTime !== null) out.push(activeTime);
  for (const pitchKey of openTies) {
    const [step, oct] = pitchKey.split(' ');
    out.push('tie-start', step, oct);
  }
  if (slurDepth > 0) out.push('slur-start');
  if (openHairpin !== null) out.push(openHairpin);
  if (openRepeat) out.push('bar-repeat-left');
  if (hasBarContent) out.push('<partial_bar>');
  return out;
}

export function extractPageTailState(bekernStr) {
  const raw = String(bekernStr ?? '').trim();
  if (!raw) return [];

  const rawTokens = raw.split(/\s+/);
  const tokens = [];
  let inPrevState = false;
  for (const tok of rawTokens) {
    if (tok === '<prev_state>') {
      inPrevState = true;
      continue;
    }
    if (tok === '</prev_state>') {
      inPrevState = false;
      continue;
    }
    if (inPrevState || TAIL_SPECIALS.has(tok)) continue;
    tokens.push(tok);
  }
  if (tokens.length === 0) return [];

  const chunks = [];
  let cur = [];
  for (const tok of tokens) {
    if (tok === '<sys>') {
      if (cur.length > 0) {
        chunks.push(cur);
        cur = [];
      }
    } else {
      cur.push(tok);
    }
  }
  if (cur.length > 0) chunks.push(cur);
  if (chunks.length === 0) return [];

  const nStaves = inferStaffCount(chunks);
  const staffChunks = Array.from({ length: nStaves }, () => []);
  for (let i = 0; i < chunks.length; i++) {
    staffChunks[i % nStaves].push(chunks[i]);
  }

  const result = [];
  for (const sChunks of staffChunks) {
    const sTail = extractSingleStaffTail(sChunks);
    if (sTail.length === 0) continue;
    if (result.length > 0) result.push('<sys>');
    result.push(...sTail);
  }
  return result;
}

function parseTailStateHeaders(prevTailState) {
  const toks = Array.isArray(prevTailState)
    ? prevTailState.map((t) => String(t ?? '').trim()).filter(Boolean)
    : String(prevTailState ?? '').trim().split(/\s+/).filter(Boolean);
  if (toks.length === 0) return [];

  const staves = [];
  let cur = { clef: null, key: null, time: null };
  const flush = () => {
    if (cur.clef || cur.key || cur.time) {
      staves.push(cur);
    }
    cur = { clef: null, key: null, time: null };
  };
  for (const t of toks) {
    if (t === '<sys>') {
      flush();
    } else if (isValidClef(t)) {
      cur.clef = t;
    } else if (isValidKey(t)) {
      cur.key = t;
    } else if (isValidTime(t)) {
      cur.time = t;
    }
  }
  flush();
  return staves;
}

/**
 * Carry over omitted Page 2+ headers (time-*, and missing key* / clef-*)
 * from the previous page's extracted tail state into each system/staff chunk.
 */
export function applyCrossPageHeaderCarryover(pageBekern, prevTailState) {
  const raw = String(pageBekern ?? '').trim();
  if (!raw) return '';
  const prevStaves = parseTailStateHeaders(prevTailState);
  if (prevStaves.length === 0) return raw;

  const runningStaves = prevStaves.map((s) => ({ ...s }));
  const rawSystems = raw.split(/\s*<sys>\s*/).filter(Boolean);
  if (rawSystems.length === 0) return raw;

  const outSystems = [];
  for (let idx = 0; idx < rawSystems.length; idx++) {
    const toks = rawSystems[idx].split(/\s+/).filter(Boolean);
    if (toks.length === 0) continue;

    const staff = runningStaves[idx % runningStaves.length];
    let headerEnd = 0;
    let leadClef = null;
    let leadKey = null;
    let leadTime = null;
    while (headerEnd < toks.length) {
      const t = toks[headerEnd];
      if (isValidClef(t)) {
        leadClef = t;
        headerEnd += 1;
      } else if (isValidKey(t)) {
        leadKey = t;
        headerEnd += 1;
      } else if (isValidTime(t)) {
        leadTime = t;
        headerEnd += 1;
      } else {
        break;
      }
    }

    let finalToks;
    if (leadClef && leadKey && leadTime) {
      finalToks = toks;
    } else {
      const mergedHeader = [];
      const effClef = leadClef || staff.clef;
      const effKey = leadKey || staff.key;
      const effTime = leadTime || staff.time;
      if (effClef) mergedHeader.push(effClef);
      if (effKey) mergedHeader.push(effKey);
      if (effTime) mergedHeader.push(effTime);
      finalToks = [...mergedHeader, ...toks.slice(headerEnd)];
    }

    // Update running staff state across the chunk so mid-page changes propagate.
    for (const t of finalToks) {
      if (isValidClef(t)) staff.clef = t;
      else if (isValidKey(t)) staff.key = t;
      else if (isValidTime(t)) staff.time = t;
    }

    outSystems.push(finalToks.join(' '));
  }

  return outSystems.join(' <sys> ');
}

/**
 * Optional 1-shot KV-cache prefix prefill for `<prev_state> ... </prev_state>`.
 * Completely no-op when `prevStateIds` is omitted/empty or `<prev_state>` is
 * not present in `tokenToId` (existing checkpoints work identically).
 */
export async function prefillPrevStateKv({
  ort,
  decoderSession,
  encOutputs,
  pastK,
  pastV,
  nLayers,
  prevStateIds,
  tokenToId,
  useMtp,
}) {
  if (
    !Array.isArray(prevStateIds) ||
    prevStateIds.length === 0 ||
    !tokenToId ||
    tokenToId['<prev_state>'] === undefined
  ) {
    return { pastK, pastV, prefilledLen: 0 };
  }

  const L = prevStateIds.length;
  if (useMtp) {
    // WebGPU MTP-4 decoder accepts dynamic [1, L] token_ids -> 1-shot prefill!
    const tokenIdsTensor = new ort.Tensor(
      'int64',
      BigInt64Array.from(prevStateIds.map((id) => BigInt(id))),
      [1, L],
    );
    const stepPosTensor = new ort.Tensor('int64', BigInt64Array.from([0n]), [1]);
    const feeds = {
      token_ids: tokenIdsTensor,
      step_pos: stepPosTensor,
    };
    for (let i = 0; i < nLayers; i++) {
      feeds[`past_k_${i}`] = pastK[i];
      feeds[`past_v_${i}`] = pastV[i];
      feeds[`cross_k_${i}`] = encOutputs[`cross_k_${i}`];
      feeds[`cross_v_${i}`] = encOutputs[`cross_v_${i}`];
    }

    let out;
    try {
      out = await decoderSession.run(feeds);
    } finally {
      if (typeof tokenIdsTensor.dispose === 'function') tokenIdsTensor.dispose();
      if (typeof stepPosTensor.dispose === 'function') stepPosTensor.dispose();
    }

    const nextPastK = [];
    const nextPastV = [];
    for (let i = 0; i < nLayers; i++) {
      nextPastK.push(out[`present_k_${i}`]);
      nextPastV.push(out[`present_v_${i}`]);
    }
    for (const t of pastK) if (t && typeof t.dispose === 'function') t.dispose();
    for (const t of pastV) if (t && typeof t.dispose === 'function') t.dispose();
    if (out.logits?.dispose) out.logits.dispose();
    for (const k of ['mtp_logits_0', 'mtp_logits_1', 'mtp_logits_2']) {
      if (out[k]?.dispose) out[k].dispose();
    }
    return { pastK: nextPastK, pastV: nextPastV, prefilledLen: L };
  }

  // Standard single-step WASM decoder fallback: step through prefix tokens
  let curPastK = pastK;
  let curPastV = pastV;
  for (let p = 0; p < L; p++) {
    const tokenIdTensor = new ort.Tensor(
      'int64',
      BigInt64Array.from([BigInt(prevStateIds[p])]),
      [1, 1],
    );
    const stepPosTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(p)]), [1]);
    const feeds = {
      token_id: tokenIdTensor,
      step_pos: stepPosTensor,
    };
    for (let i = 0; i < nLayers; i++) {
      feeds[`past_k_${i}`] = curPastK[i];
      feeds[`past_v_${i}`] = curPastV[i];
      feeds[`cross_k_${i}`] = encOutputs[`cross_k_${i}`];
      feeds[`cross_v_${i}`] = encOutputs[`cross_v_${i}`];
    }
    let out;
    try {
      out = await decoderSession.run(feeds);
    } finally {
      if (typeof tokenIdTensor.dispose === 'function') tokenIdTensor.dispose();
      if (typeof stepPosTensor.dispose === 'function') stepPosTensor.dispose();
    }
    const nextPastK = [];
    const nextPastV = [];
    for (let i = 0; i < nLayers; i++) {
      nextPastK.push(out[`present_k_${i}`]);
      nextPastV.push(out[`present_v_${i}`]);
    }
    for (const t of curPastK) if (t && typeof t.dispose === 'function') t.dispose();
    for (const t of curPastV) if (t && typeof t.dispose === 'function') t.dispose();
    if (out.logits?.dispose) out.logits.dispose();
    curPastK = nextPastK;
    curPastV = nextPastV;
  }
  return { pastK: curPastK, pastV: curPastV, prefilledLen: L };
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

export function compileBekernWithFallback(
  rawBekern,
  title,
  {
    toMsczAndMusicXml = null,
    toMscz = null,
    toMusicXml = null,
    validateSyntax = null,
    onWarn = null,
  } = {},
) {
  const hasCustomSplit = typeof toMscz === 'function' || typeof toMusicXml === 'function';
  const effToMscz = toMscz ?? bekernToMscz;
  const effToMusicXml = toMusicXml ?? bekernToMusicXml;
  const compileBoth = typeof toMsczAndMusicXml === 'function'
    ? toMsczAndMusicXml
    : hasCustomSplit
      ? (txt, t) => ({ msczBytes: effToMscz(txt, t), musicXml: effToMusicXml(txt, t) })
      : bekernToMsczAndMusicXml;
  const probeValid = typeof validateSyntax === 'function'
    ? validateSyntax
    : hasCustomSplit
      ? (probe) => {
          try {
            effToMscz(probe, title);
            return true;
          } catch {
            return false;
          }
        }
      : validateBekernSyntax;

  const sanitized = sanitizeBekernForMscz(rawBekern);
  try {
    const { msczBytes, musicXml } = compileBoth(sanitized, title);
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
        let ok = false;
        try {
          ok = Boolean(probeValid(probe));
        } catch {
          ok = false;
        }
        if (ok) {
          keptMeasures.push(mToks.join(' '));
        } else {
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
    const { msczBytes, musicXml } = compileBoth(recovered, title);
    if (typeof onWarn === 'function') {
      onWarn(`mscz-rs measure-by-measure fallback succeeded (dropped ${droppedMeasures} malformed measure(s))`);
    }
    return { msczBytes, musicXml, bekern: recovered, usedFallback: true, droppedMeasures };
  }
}

// ---------------------------------------------------------------------------
// MTP-4 FSA-Verified Speculative Decoding Helpers
// ---------------------------------------------------------------------------

/**
 * Draft K-1 (typically 3) future tokens from MTP head logits using a cloned
 * BekernFsa so the committed FSA state is never mutated.
 */
export function draftMtpTokens(fsa, mtpLogitsList, vocabSize) {
  const draftFsa = fsa.clone();
  const drafts = new Array(mtpLogitsList.length);
  for (let i = 0; i < mtpLogitsList.length; i++) {
    const tok = draftFsa.selectNextToken(mtpLogitsList[i], vocabSize, 0);
    draftFsa.step(tok);
    drafts[i] = tok;
  }
  return drafts;
}

/**
 * Verify a speculative MTP-K batch [curTok, d1, ..., d_{K-1}] against the true
 * decoder logits [1, K, vocabSize], advancing `fsa` for every accepted token.
 *
 * Returns:
 *   - acceptedIds: 1..K verified token IDs (including the correction/bonus token)
 *   - acceptedCount: acceptedIds.length (KV cache should be sliced to pastLen + acceptedCount)
 *   - allMatched: true iff all K-1 drafts matched (all K KV entries valid; can reuse mtp_logits)
 *   - hitEos: true iff eosId was produced
 */
export function verifyMtpBatch(fsa, batchLogitsData, vocabSize, draftedTokens, eosId, maxRemaining = Infinity) {
  const K = draftedTokens.length + 1;
  const acceptedIds = [];
  let allMatched = true;
  let hitEos = false;

  for (let j = 0; j < K; j++) {
    if (acceptedIds.length >= maxRemaining) {
      allMatched = false;
      break;
    }
    const vTok = fsa.selectNextToken(batchLogitsData, vocabSize, j * vocabSize);
    fsa.step(vTok);
    acceptedIds.push(vTok);
    if (vTok === eosId) {
      hitEos = true;
      allMatched = j === K - 1;
      break;
    }
    if (j < draftedTokens.length && vTok !== draftedTokens[j]) {
      allMatched = false;
      break;
    }
  }

  return {
    acceptedIds,
    acceptedCount: acceptedIds.length,
    allMatched,
    hitEos,
  };
}

async function sliceKvTensor(ort, tensor, nHeads, keepLen, headDim) {
  const totalLen = tensor.dims[2];
  if (keepLen === totalLen) return tensor;
  const src = typeof tensor.getData === 'function' ? await tensor.getData() : tensor.data;
  const dst = new Float32Array(nHeads * keepLen * headDim);
  const copyPerHead = keepLen * headDim;
  const strideSrc = totalLen * headDim;
  for (let h = 0; h < nHeads; h++) {
    const srcOff = h * strideSrc;
    dst.set(src.subarray(srcOff, srcOff + copyPerHead), h * copyPerHead);
  }
  if (typeof tensor.dispose === 'function') tensor.dispose();
  return new ort.Tensor('float32', dst, [1, nHeads, keepLen, headDim]);
}

const state = {
  ort: null,
  wasmReady: false,
  modelsReady: false,
  backend: 'wasm',
  capabilities: { webgpu: false, webnn: false, wasm: true },
  numThreads: 1,
  useMtp: false,
  mtpK: 1,
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

  const nHeads = raw.n_heads ?? 6;
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
    nHeads,
    nKvHeads: raw.n_kv_heads ?? nHeads,
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

function detectWebNn() {
  return typeof navigator !== 'undefined' && 'ml' in navigator;
}

async function createOrtSession(ort, url, provider, extraOpts = {}) {
  return await ort.InferenceSession.create(url, {
    executionProviders: [provider],
    graphOptimizationLevel: 'all',
    logSeverityLevel: 3,
    ...extraOpts,
  });
}

async function warmupWebGpuSessions(ort, encSession, decSession, vocab) {
  const nLayers = vocab.nDecLayers || 6;
  const nKvHeads = vocab.nKvHeads ?? vocab.nHeads ?? 6;
  const headDim = vocab.headDim || 64;
  const bosId = vocab.bosId;

  let imgT = null;
  let encOut = null;
  let tokT1 = null;
  let posT1 = null;
  let tokT4 = null;
  let posT4 = null;
  let decOut1 = null;
  let decOut4 = null;
  const emptyK = [];
  const emptyV = [];

  try {
    // 1. Minimal encoder run (32x32)
    imgT = new ort.Tensor('float32', new Float32Array(1 * 3 * 32 * 32), [1, 3, 32, 32]);
    encOut = await encSession.run({ image: imgT });

    // 2. Minimal T=1 decoder run
    for (let i = 0; i < nLayers; i++) {
      emptyK.push(new ort.Tensor('float32', new Float32Array(0), [1, nKvHeads, 0, headDim]));
      emptyV.push(new ort.Tensor('float32', new Float32Array(0), [1, nKvHeads, 0, headDim]));
    }
    tokT1 = new ort.Tensor('int64', BigInt64Array.from([BigInt(bosId)]), [1, 1]);
    posT1 = new ort.Tensor('int64', BigInt64Array.from([0n]), [1]);
    const feed1 = { token_ids: tokT1, step_pos: posT1 };
    for (let i = 0; i < nLayers; i++) {
      feed1[`past_k_${i}`] = emptyK[i];
      feed1[`past_v_${i}`] = emptyV[i];
      feed1[`cross_k_${i}`] = encOut[`cross_k_${i}`];
      feed1[`cross_v_${i}`] = encOut[`cross_v_${i}`];
    }
    decOut1 = await decSession.run(feed1);

    // 3. Minimal T=4 MTP-4 decoder run (verifies dynamic seq_len & causal mask on WebGPU)
    tokT4 = new ort.Tensor(
      'int64',
      BigInt64Array.from([BigInt(bosId), BigInt(bosId), BigInt(bosId), BigInt(bosId)]),
      [1, 4],
    );
    posT4 = new ort.Tensor('int64', BigInt64Array.from([1n]), [1]);
    const feed4 = { token_ids: tokT4, step_pos: posT4 };
    for (let i = 0; i < nLayers; i++) {
      feed4[`past_k_${i}`] = decOut1[`present_k_${i}`];
      feed4[`past_v_${i}`] = decOut1[`present_v_${i}`];
      feed4[`cross_k_${i}`] = encOut[`cross_k_${i}`];
      feed4[`cross_v_${i}`] = encOut[`cross_v_${i}`];
    }
    decOut4 = await decSession.run(feed4);
  } finally {
    if (imgT?.dispose) imgT.dispose();
    if (tokT1?.dispose) tokT1.dispose();
    if (posT1?.dispose) posT1.dispose();
    if (tokT4?.dispose) tokT4.dispose();
    if (posT4?.dispose) posT4.dispose();
    for (const t of emptyK) if (t?.dispose) t.dispose();
    for (const t of emptyV) if (t?.dispose) t.dispose();
    for (const bag of [encOut, decOut1, decOut4]) {
      if (bag) {
        for (const t of Object.values(bag)) if (t?.dispose) t.dispose();
      }
    }
  }
}

function detectRepetitionLoop(tokenIds, idToToken = null) {
  const n = tokenIds.length;
  if (n >= 24) {
    const last = tokenIds[n - 1];
    let run = 1;
    while (run < 24 && tokenIds[n - 1 - run] === last) run++;
    if (run >= 24) return { detected: true, period: 1, cutAt: n - run };
  }
  // Never falsely kill legitimate multi-bar rests (bar-rest bar x N) or normal
  // musical ostinatos (e.g. 4-8 repeated chords/measures):
  //   - For short periods p=2..5 (like "bar-rest bar" or "quarter C 4"),
  //     require >= 32 consecutive repeats (>= 64 tokens for bar-rest bar!).
  //   - For medium/long periods p=6..32, require >= 6 consecutive repeats
  //     (and >= 48 tokens total in the loop).
  const maxP = Math.min(32, Math.floor(n / 6));
  for (let p = 2; p <= maxP; p++) {
    const minReps = p <= 5 ? Math.max(16, Math.ceil(64 / p)) : Math.max(6, Math.ceil(48 / p));
    if (n < p * (minReps + 1)) continue;
    let match = true;
    const span = p * minReps;
    for (let k = 0; k < span; k++) {
      if (tokenIds[n - 1 - k] !== tokenIds[n - 1 - k - p]) {
        match = false;
        break;
      }
    }
    if (match) {
      // Allow up to 64 consecutive bar-rest measures (e.g. orchestral parts)
      if (p === 2 && idToToken) {
        const t0 = idToToken[tokenIds[n - 1]];
        const t1 = idToToken[tokenIds[n - 2]];
        if ((t0 === 'bar' && t1 === 'bar-rest') || (t1 === 'bar' && t0 === 'bar-rest')) {
          if (n < 130) continue;
        }
      }
      return { detected: true, period: p, cutAt: n - span };
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

async function handleInit(msg = {}) {
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

    const hasWebGpu = msg?.forceBackend === 'wasm' ? false : await detectWebGpu();
    const hasWebNn = detectWebNn();
    state.capabilities = { webgpu: hasWebGpu, webnn: hasWebNn, wasm: true };
    state.backend = hasWebGpu ? 'webgpu' : 'wasm';
    self.postMessage({
      type: 'ort-ready',
      backend: state.backend,
      capabilities: state.capabilities,
      numThreads: state.numThreads,
    });
  } catch (err) {
    self.postMessage({ type: 'ort-error', message: err?.message || String(err) });
    return;
  }

  // 3. Load vocab.json + ONNX encoder/decoder (prefer WebGPU INT4 + MTP-4, fallback to WASM SIMD 8T)
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

    const webgpuEncFile = state.modelIo.webgpu_encoder_file || vocabRaw.webgpu_encoder_file || null;
    const webgpuDecFile = state.modelIo.webgpu_decoder_file || vocabRaw.webgpu_decoder_file || null;
    const webgpuMtpK = Number(state.modelIo.webgpu_mtp_k || vocabRaw.webgpu_mtp_k || 1);

    let loadedWebGpu = false;
    if (state.backend === 'webgpu' && webgpuEncFile && webgpuDecFile) {
      let wgEnc = null;
      let wgDec = null;
      try {
        // Keep cross_k/v and present_k/v resident in WebGPU buffers across
        // decode steps to eliminate ~80 MB/step CPU<->GPU copies!
        const nLayers = state.vocab.nDecLayers || 6;
        const decOutLoc = {
          logits: 'cpu',
          mtp_logits_0: 'cpu',
          mtp_logits_1: 'cpu',
          mtp_logits_2: 'cpu',
        };
        for (let i = 0; i < nLayers; i++) {
          decOutLoc[`present_k_${i}`] = 'gpu-buffer';
          decOutLoc[`present_v_${i}`] = 'gpu-buffer';
        }
        wgEnc = await createOrtSession(ort, `./models/${webgpuEncFile}`, 'webgpu', {
          preferredOutputLocation: 'gpu-buffer',
        });
        wgDec = await createOrtSession(ort, `./models/${webgpuDecFile}`, 'webgpu', {
          preferredOutputLocation: decOutLoc,
        });
        await warmupWebGpuSessions(ort, wgEnc, wgDec, state.vocab);
        state.encoderSession = wgEnc;
        state.decoderSession = wgDec;
        state.backend = 'webgpu';
        state.useMtp = webgpuMtpK >= 4;
        state.mtpK = state.useMtp ? webgpuMtpK : 1;
        loadedWebGpu = true;
      } catch (wgErr) {
        console.warn('WebGPU session/warmup failed, falling back to WASM SIMD 8T:', wgErr);
        try { if (wgEnc?.release) await wgEnc.release(); } catch (_) {}
        try { if (wgDec?.release) await wgDec.release(); } catch (_) {}
      }
    }

    if (!loadedWebGpu) {
      const encoderUrl = state.modelIo.encoder_file
        ? `./models/${state.modelIo.encoder_file}`
        : './models/uni_omr_encoder.onnx';
      const decoderUrl = state.modelIo.decoder_file
        ? `./models/${state.modelIo.decoder_file}`
        : './models/uni_omr_decoder_step.onnx';
      state.encoderSession = await createOrtSession(ort, encoderUrl, 'wasm');
      state.decoderSession = await createOrtSession(ort, decoderUrl, 'wasm');
      state.backend = 'wasm';
      state.useMtp = false;
      state.mtpK = 1;
    }

    state.modelsReady = true;

    self.postMessage({
      type: 'models-ready',
      backend: state.backend,
      capabilities: state.capabilities,
      numThreads: state.numThreads,
      useMtp: state.useMtp,
      mtpK: state.mtpK,
      maxDim: state.vocab.maxDim || state.modelIo.max_dim || 1280,
      maxTokens: state.modelIo.max_tokens || DEFAULT_MAX_TOKENS,
    });
  } catch (err) {
    self.postMessage({ type: 'models-error', message: err?.message || String(err) });
  }
}

export function formatOrtError(err, ctx = {}) {
  const raw = err instanceof Error ? (err.message || String(err)) : String(err ?? 'Unknown error');
  const isNumericPtr = typeof err === 'number' || /^\s*\d+\s*$/.test(raw);
  const parts = [];
  if (ctx.stage) parts.push(`stage=${ctx.stage}`);
  if (ctx.width && ctx.height) parts.push(`image=${ctx.width}x${ctx.height}`);
  if (typeof ctx.nMem === 'number' && ctx.nMem > 0) parts.push(`n_mem=${ctx.nMem}`);
  if (typeof ctx.step === 'number' && ctx.step >= 0) parts.push(`step=${ctx.step}`);
  if (ctx.backend) parts.push(`backend=${ctx.backend}${ctx.numThreads ? `(${ctx.numThreads}T)` : ''}`);
  const meta = parts.length > 0 ? ` [${parts.join(', ')}]` : '';
  if (isNumericPtr) {
    return `WASM runtime exception (likely out-of-memory, ptr=${raw.trim()})${meta}`;
  }
  return `${raw}${meta}`;
}

async function handleRecognizePage(msg) {
  const {
    requestId,
    nchwBuffer,
    width,
    height,
    pageIndex = 1,
    pageCount = 1,
    prevStateIds = null,
    prevTailState = null,
  } = msg;
  if (!state.modelsReady || !state.ort) {
    self.postMessage({ type: 'error', requestId, message: 'UniOMR models are not loaded yet.' });
    return;
  }

  state.cancelRequested = false;
  const ort = state.ort;
  const maxTokens = state.modelIo.max_tokens || DEFAULT_MAX_TOKENS;
  const nLayers = state.vocab.nDecLayers || state.modelIo.n_dec_layers || 6;
  const nHeads = state.vocab.nHeads || state.modelIo.n_heads || 6;
  const nKvHeads = state.vocab.nKvHeads ?? state.modelIo.n_kv_heads ?? nHeads;
  const headDim = state.vocab.headDim || state.modelIo.head_dim || 64;
  const { bosId, eosId, sysId, padId, idToToken, tokenToId, vocabSize } = state.vocab;

  const t0 = performance.now();
  let imgTensor = null;
  let encOutputs = null;
  let pastK = [];
  let pastV = [];
  let stage = 'encoding';
  let currentStep = -1;
  let nMem = 0;

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
    const ck0 = encOutputs.cross_k_0;
    if (ck0 && Array.isArray(ck0.dims) && ck0.dims.length >= 3) {
      nMem = ck0.dims[2];
    }

    const encodeMs = performance.now() - t0;

    if (state.cancelRequested) {
      self.postMessage({ type: 'cancelled', requestId });
      return;
    }

    stage = 'decoding';
    for (let i = 0; i < nLayers; i++) {
      pastK.push(new ort.Tensor('float32', new Float32Array(0), [1, nKvHeads, 0, headDim]));
      pastV.push(new ort.Tensor('float32', new Float32Array(0), [1, nKvHeads, 0, headDim]));
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

    const emitToken = (tokId) => {
      if (tokId !== padId && tokId !== bosId && tokId !== eosId) {
        generatedIds.push(tokId);
        const tok = idToToken[tokId] ?? '';
        if (tok) generatedTokens.push(tok);
        if (tokId === sysId || tok === '<sys>') {
          systemCount++;
        }
      }
    };

    const postDecodeProgress = (stepCount) => {
      const now = performance.now();
      const decS = Math.max(0.001, (now - decodeStart) / 1000);
      const tokPerSec = Math.max(1, stepCount) / decS;
      const pct = 22 + Math.min(70, (stepCount / Math.min(maxTokens, 1024)) * 70);
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
    };

    // Optional 1-shot KV-cache prefill for <prev_state> ... </prev_state>
    const prefillRes = await prefillPrevStateKv({
      ort,
      decoderSession: state.decoderSession,
      encOutputs,
      pastK,
      pastV,
      nLayers,
      prevStateIds,
      tokenToId,
      useMtp: state.useMtp,
    });
    pastK = prefillRes.pastK;
    pastV = prefillRes.pastV;
    const prefilledLen = prefillRes.prefilledLen;

    if (state.useMtp) {
      // 4-token FSA-verified MTP-4 speculative decoding (WebGPU native)
      let pendingDrafts = null;
      let totalGenerated = 0;
      let lastProgressAt = 0;

      while (totalGenerated < maxTokens) {
        currentStep = totalGenerated;
        if (state.cancelRequested) {
          self.postMessage({ type: 'cancelled', requestId });
          return;
        }

        const pastLen = pastK[0].dims[2];
        const canSpeculate =
          pendingDrafts !== null &&
          pendingDrafts.length === 3 &&
          maxTokens - totalGenerated >= 4;

        const batchIds = canSpeculate
          ? [curId, pendingDrafts[0], pendingDrafts[1], pendingDrafts[2]]
          : [curId];
        const tokenIdsTensor = new ort.Tensor(
          'int64',
          BigInt64Array.from(batchIds.map((id) => BigInt(id))),
          [1, batchIds.length],
        );
        const stepPosTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(pastLen)]), [1]);
        const feeds = {
          token_ids: tokenIdsTensor,
          step_pos: stepPosTensor,
        };
        for (let i = 0; i < nLayers; i++) {
          feeds[`past_k_${i}`] = pastK[i];
          feeds[`past_v_${i}`] = pastV[i];
          feeds[`cross_k_${i}`] = encOutputs[`cross_k_${i}`];
          feeds[`cross_v_${i}`] = encOutputs[`cross_v_${i}`];
        }

        let out;
        try {
          out = await state.decoderSession.run(feeds);
        } finally {
          if (typeof tokenIdsTensor.dispose === 'function') tokenIdsTensor.dispose();
          if (typeof stepPosTensor.dispose === 'function') stepPosTensor.dispose();
        }

        const logitsTensor = out.logits;
        const mtpTensors = [out.mtp_logits_0, out.mtp_logits_1, out.mtp_logits_2];

        let keepLen;
        let done = false;
        let nextDrafts = null;

        if (canSpeculate) {
          const ver = verifyMtpBatch(
            fsa,
            logitsTensor.data,
            vocabSize,
            pendingDrafts,
            eosId,
            maxTokens - totalGenerated,
          );
          for (const tokId of ver.acceptedIds) {
            totalGenerated++;
            if (tokId === eosId) {
              done = true;
              break;
            }
            emitToken(tokId);
            curId = tokId;
          }
          keepLen = pastLen + ver.acceptedCount;
          if (!done && ver.allMatched && mtpTensors[0] && mtpTensors[1] && mtpTensors[2]) {
            nextDrafts = draftMtpTokens(
              fsa,
              [mtpTensors[0].data, mtpTensors[1].data, mtpTensors[2].data],
              vocabSize,
            );
          }
        } else {
          const nextId = fsa.selectNextToken(logitsTensor.data, vocabSize, 0);
          fsa.step(nextId);
          totalGenerated++;
          keepLen = pastLen + 1;
          if (nextId === eosId) {
            done = true;
          } else {
            emitToken(nextId);
            curId = nextId;
            if (mtpTensors[0] && mtpTensors[1] && mtpTensors[2]) {
              nextDrafts = draftMtpTokens(
                fsa,
                [mtpTensors[0].data, mtpTensors[1].data, mtpTensors[2].data],
                vocabSize,
              );
            }
          }
        }

        const nextPastK = [];
        const nextPastV = [];
        for (let i = 0; i < nLayers; i++) {
          nextPastK.push(await sliceKvTensor(ort, out[`present_k_${i}`], nKvHeads, keepLen, headDim));
          nextPastV.push(await sliceKvTensor(ort, out[`present_v_${i}`], nKvHeads, keepLen, headDim));
        }
        for (const t of pastK) if (t && typeof t.dispose === 'function') t.dispose();
        for (const t of pastV) if (t && typeof t.dispose === 'function') t.dispose();
        if (logitsTensor?.dispose) logitsTensor.dispose();
        for (const mt of mtpTensors) if (mt?.dispose) mt.dispose();
        pastK = nextPastK;
        pastV = nextPastV;
        pendingDrafts = nextDrafts;

        if (done) break;

        const loopCheck = detectRepetitionLoop(generatedIds, idToToken);
        if (loopCheck.detected) {
          stoppedByLoop = true;
          generatedTokens.length = Math.max(0, loopCheck.cutAt);
          break;
        }

        if (totalGenerated - lastProgressAt >= 4) {
          lastProgressAt = totalGenerated;
          postDecodeProgress(totalGenerated);
        }
      }
    } else {
      // Standard single-step autoregressive decoding (WASM SIMD 8T fallback)
      for (let step = 0; step < maxTokens; step++) {
        currentStep = step;
        if (state.cancelRequested) {
          self.postMessage({ type: 'cancelled', requestId });
          return;
        }

        const tokenIdTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(curId)]), [1, 1]);
        const stepPosTensor = new ort.Tensor('int64', BigInt64Array.from([BigInt(prefilledLen + step)]), [1]);
        const feeds = {
          token_id: tokenIdTensor,
          step_pos: stepPosTensor,
        };
        for (let i = 0; i < nLayers; i++) {
          feeds[`past_k_${i}`] = pastK[i];
          feeds[`past_v_${i}`] = pastV[i];
          feeds[`cross_k_${i}`] = encOutputs[`cross_k_${i}`];
          feeds[`cross_v_${i}`] = encOutputs[`cross_v_${i}`];
        }

        let out;
        try {
          out = await state.decoderSession.run(feeds);
        } finally {
          if (typeof tokenIdTensor.dispose === 'function') tokenIdTensor.dispose();
          if (typeof stepPosTensor.dispose === 'function') stepPosTensor.dispose();
        }
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
        emitToken(nextId);

        const loopCheck = detectRepetitionLoop(generatedIds, idToToken);
        if (loopCheck.detected) {
          stoppedByLoop = true;
          generatedTokens.length = Math.max(0, loopCheck.cutAt);
          break;
        }

        if (step % 4 === 0) {
          postDecodeProgress(step + 1);
        }
      }
    }

    const rawFinalTokens = stoppedByLoop ? trimToCleanBoundary(generatedTokens) : generatedTokens;
    const sanitizedRaw = sanitizeBekernForMscz(rawFinalTokens.join(' '));
    const sanitizedBekern = prevTailState
      ? applyCrossPageHeaderCarryover(sanitizedRaw, prevTailState)
      : sanitizedRaw;
    const tailState = extractPageTailState(sanitizedBekern);
    const finalTokenCount = sanitizedBekern ? sanitizedBekern.split(/\s+/).length : 0;
    const decodeMs = performance.now() - decodeStart;
    const tokPerSec = finalTokenCount / Math.max(0.001, decodeMs / 1000);

    self.postMessage({
      type: 'page-done',
      requestId,
      bekern: sanitizedBekern,
      tailState,
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
    const errCtx = {
      stage,
      width,
      height,
      step: currentStep,
      nMem,
      pageIndex,
      pageCount,
      backend: state.backend,
      numThreads: state.numThreads,
      elapsedMs: Math.round(performance.now() - t0),
    };
    const formatted = formatOrtError(err, errCtx);
    sendWorkerLog('error', `recognize-page failed: ${formatted}`, {
      ...errCtx,
      rawError: err instanceof Error ? { name: err.name, message: err.message, stack: err.stack } : String(err),
    });
    self.postMessage({
      type: 'error',
      requestId,
      message: formatted,
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
        await handleInit(msg);
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
