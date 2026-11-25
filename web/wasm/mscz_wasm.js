/**
 * Zero-bundler ES module wrapper for `mscz-rs` compiled to `wasm32-unknown-unknown`.
 *
 * Exports:
 *   - initMsczWasm(wasmUrl?) -> Promise<void>
 *   - bekernToMscz(bekernText, title?) -> Uint8Array
 *   - bekernToMusicXml(bekernText, title?) -> string
 *   - bekernToMsczAndMusicXml(bekernText, title?) -> { msczBytes: Uint8Array, musicXml: string }
 *   - validateBekernSyntax(bekernText) -> boolean
 *   - stitchPagesBekern(pages) -> string
 */

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder('utf-8');

let wasmExports = null;
let initPromise = null;

/**
 * Load and instantiate `mscz_wasm.wasm` (idempotent).
 * @param {string|URL} [wasmUrl]
 */
export async function initMsczWasm(wasmUrl) {
  if (wasmExports) return;
  if (initPromise) return initPromise;

  const url = wasmUrl ?? new URL('./mscz_wasm.wasm', import.meta.url);
  initPromise = (async () => {
    const resp = await fetch(url);
    if (!resp.ok) {
      throw new Error(`Failed to fetch ${url}: HTTP ${resp.status}`);
    }
    let result;
    if (typeof WebAssembly.instantiateStreaming === 'function') {
      try {
        result = await WebAssembly.instantiateStreaming(resp.clone(), {});
      } catch {
        const bytes = await resp.arrayBuffer();
        result = await WebAssembly.instantiate(bytes, {});
      }
    } else {
      const bytes = await resp.arrayBuffer();
      result = await WebAssembly.instantiate(bytes, {});
    }
    wasmExports = result.instance.exports;
  })();

  try {
    await initPromise;
  } catch (err) {
    initPromise = null;
    throw err;
  }
}

function callCompiler(fnName, bekernText, title = '') {
  if (!wasmExports) {
    throw new Error('mscz-rs WASM not initialized; call await initMsczWasm() first');
  }
  const {
    memory,
    wasm_alloc,
    wasm_free,
    wasm_result_ptr,
    wasm_result_len,
    wasm_error_ptr,
    wasm_error_len,
  } = wasmExports;

  const bekernBytes = textEncoder.encode(String(bekernText ?? ''));
  const titleBytes = textEncoder.encode(String(title ?? ''));

  const bekernPtr = wasm_alloc(bekernBytes.length);
  const titlePtr = wasm_alloc(titleBytes.length);

  try {
    if (bekernBytes.length > 0) {
      new Uint8Array(memory.buffer, bekernPtr, bekernBytes.length).set(bekernBytes);
    }
    if (titleBytes.length > 0) {
      new Uint8Array(memory.buffer, titlePtr, titleBytes.length).set(titleBytes);
    }

    const outLen = wasmExports[fnName](
      bekernPtr,
      bekernBytes.length,
      titlePtr,
      titleBytes.length,
    );

    if (outLen === 0) {
      const errPtr = wasm_error_ptr();
      const errLen = wasm_error_len();
      const errMsg = errLen > 0
        ? textDecoder.decode(new Uint8Array(memory.buffer, errPtr, errLen))
        : `${fnName} failed with unknown error`;
      throw new Error(errMsg);
    }

    const resPtr = wasm_result_ptr();
    const resLen = wasm_result_len();
    // Copy out of WASM linear memory so future calls cannot overwrite it.
    return new Uint8Array(memory.buffer.slice(resPtr, resPtr + resLen));
  } finally {
    wasm_free(bekernPtr, bekernBytes.length);
    wasm_free(titlePtr, titleBytes.length);
  }
}

/**
 * Compile a `bekern+` token stream (single-system or `<sys>`-separated full page)
 * into native MuseScore 4 `.mscz` archive bytes.
 * @param {string} bekernText
 * @param {string} [title]
 * @returns {Uint8Array}
 */
export function bekernToMscz(bekernText, title = '') {
  return callCompiler('bekern_to_mscz', bekernText, title);
}

/**
 * Compile a `bekern+` token stream into a MusicXML 3.1 UTF-8 string.
 * @param {string} bekernText
 * @param {string} [title]
 * @returns {string}
 */
export function bekernToMusicXml(bekernText, title = '') {
  const bytes = callCompiler('bekern_to_musicxml', bekernText, title);
  return textDecoder.decode(bytes);
}

/**
 * Compile a `bekern+` token stream into BOTH native MuseScore 4 `.mscz` bytes
 * and a MusicXML 3.1 UTF-8 string in a single WASM parse pass.
 * Automatically falls back to separate `bekernToMscz` + `bekernToMusicXml`
 * calls if the loaded WASM binary does not export `bekern_to_mscz_and_musicxml`.
 * @param {string} bekernText
 * @param {string} [title]
 * @returns {{ msczBytes: Uint8Array, musicXml: string }}
 */
export function bekernToMsczAndMusicXml(bekernText, title = '') {
  if (!wasmExports) {
    throw new Error('mscz-rs WASM not initialized; call await initMsczWasm() first');
  }
  if (typeof wasmExports.bekern_to_mscz_and_musicxml !== 'function') {
    return {
      msczBytes: bekernToMscz(bekernText, title),
      musicXml: bekernToMusicXml(bekernText, title),
    };
  }
  const framed = callCompiler('bekern_to_mscz_and_musicxml', bekernText, title);
  if (framed.byteLength < 4) {
    throw new Error('bekern_to_mscz_and_musicxml returned truncated frame');
  }
  const view = new DataView(framed.buffer, framed.byteOffset, framed.byteLength);
  const msczLen = view.getUint32(0, true);
  if (4 + msczLen > framed.byteLength) {
    throw new Error(`bekern_to_mscz_and_musicxml invalid frame length (${msczLen} > ${framed.byteLength - 4})`);
  }
  const msczBytes = framed.slice(4, 4 + msczLen);
  const musicXml = textDecoder.decode(framed.subarray(4 + msczLen));
  return { msczBytes, musicXml };
}

/**
 * Fast parse-only syntax validator for `bekern+` streams (skips `.mscx`/MusicXML/ZIP).
 * Automatically falls back to `bekernToMscz` if the loaded WASM binary does not
 * export `validate_bekern_syntax`.
 * @param {string} bekernText
 * @returns {boolean}
 */
export function validateBekernSyntax(bekernText) {
  if (!wasmExports) {
    throw new Error('mscz-rs WASM not initialized; call await initMsczWasm() first');
  }
  if (typeof wasmExports.validate_bekern_syntax !== 'function') {
    try {
      bekernToMscz(bekernText, '');
      return true;
    } catch {
      return false;
    }
  }
  const { memory, wasm_alloc, wasm_free, validate_bekern_syntax } = wasmExports;
  const bytes = textEncoder.encode(String(bekernText ?? ''));
  const ptr = wasm_alloc(bytes.length);
  try {
    if (bytes.length > 0) {
      new Uint8Array(memory.buffer, ptr, bytes.length).set(bytes);
    }
    return validate_bekern_syntax(ptr, bytes.length) === 1;
  } finally {
    wasm_free(ptr, bytes.length);
  }
}

/**
 * Stitch an array of per-page `bekern+` token strings into a single `<sys>`-joined
 * multi-page `bekern+` stream, carrying over active `clef-*`/`key*`/`time-*` headers
 * whenever Page 2+ omits them.
 * @param {string[]} pages
 * @returns {string}
 */
export function stitchPagesBekern(pages) {
  if (!wasmExports) {
    throw new Error('mscz-rs WASM not initialized; call await initMsczWasm() first');
  }
  const {
    memory,
    wasm_alloc,
    wasm_free,
    wasm_result_ptr,
    wasm_result_len,
    wasm_error_ptr,
    wasm_error_len,
    stitch_pages_bekern,
  } = wasmExports;

  const joined = (Array.isArray(pages) ? pages : [String(pages ?? '')])
    .map((p) => String(p ?? '').replace(/\r?\n/g, ' '))
    .join('\n');
  const bytes = textEncoder.encode(joined);
  const ptr = wasm_alloc(bytes.length);

  try {
    if (bytes.length > 0) {
      new Uint8Array(memory.buffer, ptr, bytes.length).set(bytes);
    }
    const outLen = stitch_pages_bekern(ptr, bytes.length);
    if (outLen === 0) {
      const errPtr = wasm_error_ptr();
      const errLen = wasm_error_len();
      const errMsg = errLen > 0
        ? textDecoder.decode(new Uint8Array(memory.buffer, errPtr, errLen))
        : 'stitch_pages_bekern failed with unknown error';
      throw new Error(errMsg);
    }
    const resPtr = wasm_result_ptr();
    const resLen = wasm_result_len();
    return textDecoder.decode(new Uint8Array(memory.buffer.slice(resPtr, resPtr + resLen)));
  } finally {
    wasm_free(ptr, bytes.length);
  }
}
