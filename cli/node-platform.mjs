/**
 * node-platform.mjs — the few browser APIs LUCID's own file readers need, for
 * plain Node (the command-line LUCID, cli/track.mjs). Nothing here is LUCID
 * logic: it only lets the app's unchanged readers run outside a browser.
 *
 *  - `Worker`: the browser's Web Worker, on `node:worker_threads`. The app reads
 *    a `.slp` in workers: sleap-io.js's streaming reader starts a classic worker
 *    from a `blob:` URL (h5wasm loaded by `importScripts`), and the raw reader
 *    (`loading/slp-import-worker.js`) is a module worker. Both mount the user's
 *    File with h5wasm's WORKERFS, which reads through `FileReaderSync` — given to
 *    the worker by cli/node-worker-bootstrap.mjs. Messages are structured-cloned
 *    by worker_threads exactly like the browser's postMessage.
 *  - `DiskFile`: a `File` for a path on disk that reads lazily (the browser's
 *    File from a picker does the same), so a multi-GB `.slp` is never read into
 *    memory whole. It crosses into a worker as its path.
 *
 * `installNodePlatform()` sets `globalThis.Worker` (only if absent) and returns
 * the repository root URL, which stands in for `document.baseURI`.
 *
 * Depends on: cli/node-worker-bootstrap.mjs (the worker side)
 */
import fs from 'node:fs';
import path from 'node:path';
import { Worker as ThreadWorker } from 'node:worker_threads';
import { resolveObjectURL } from 'node:buffer';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT_URL = pathToFileURL(path.resolve(HERE, '..') + path.sep).href;
const BOOTSTRAP = path.join(HERE, 'node-worker-bootstrap.mjs');

/** Workers' console.log / info / debug off (the CLI's default; --verbose keeps them). */
let quietWorkers = false;
export function setWorkerConsoleQuiet(q) { quietWorkers = !!q; }

/** Marker a DiskFile becomes when posted to a worker (revived there by path). */
const DISK_FILE_TAG = '__lucidDiskFile';

/**
 * A `File` backed by a path: size and name from `stat`, bytes read on demand.
 * `slice()` returns another DiskFile over the range, like a browser File.
 */
export class DiskFile extends File {
    constructor(filePath, range) {
        const abs = path.resolve(filePath);
        super([], path.basename(abs));
        const st = range ? null : fs.statSync(abs);
        this._path = abs;
        this._start = range ? range.start : 0;
        this._end = range ? range.end : st.size;
        this._mtime = range ? range.mtime : st.mtimeMs;
    }
    get size() { return this._end - this._start; }
    get lastModified() { return Math.floor(this._mtime); }
    get path() { return this._path; }
    slice(start, end) {
        const n = this.size;
        let a = start == null ? 0 : (start < 0 ? Math.max(n + start, 0) : Math.min(start, n));
        let b = end == null ? n : (end < 0 ? Math.max(n + end, 0) : Math.min(end, n));
        if (b < a) b = a;
        return new DiskFile(this._path, { start: this._start + a, end: this._start + b, mtime: this._mtime });
    }
    _readSync() {
        const out = new Uint8Array(this.size);
        const fd = fs.openSync(this._path, 'r');
        try {
            let off = 0;
            while (off < out.length) {
                const n = fs.readSync(fd, out, off, out.length - off, this._start + off);
                if (n <= 0) break;
                off += n;
            }
        } finally { fs.closeSync(fd); }
        return out;
    }
    async arrayBuffer() { return this._readSync().buffer; }
    async bytes() { return this._readSync(); }
    async text() { return new TextDecoder().decode(this._readSync()); }
    stream() { return new Blob([this._readSync()]).stream(); }
}

// Replace every DiskFile in a message with its path (plain objects/arrays only;
// typed arrays and other values pass through to the structured clone).
function encodeFiles(v, depth) {
    if (depth > 6 || v == null || typeof v !== 'object') return v;
    if (v instanceof DiskFile) {
        return { [DISK_FILE_TAG]: true, path: v._path, start: v._start, end: v._end, mtime: v._mtime, name: v.name };
    }
    if (Array.isArray(v)) return v.map(x => encodeFiles(x, depth + 1));
    if (Object.getPrototypeOf(v) !== Object.prototype) return v;
    const out = {};
    for (const k of Object.keys(v)) out[k] = encodeFiles(v[k], depth + 1);
    return out;
}

/** The transferables worker_threads accepts (ArrayBuffers and MessagePorts). */
function transferList(t) {
    if (!Array.isArray(t)) return undefined;
    return t.filter(x => x instanceof ArrayBuffer || (x && x.constructor && x.constructor.name === 'MessagePort'));
}

/**
 * The browser's `Worker` on a worker thread. `url`: a `blob:` URL (classic
 * script — its code is read from the Blob) or a `file:` URL / path (module
 * or classic, per `opts.type`). Messages posted before the thread is up are
 * queued, as a browser queues them.
 */
export class NodeWebWorker extends EventTarget {
    constructor(url, opts) {
        super();
        this.onmessage = null;
        this.onerror = null;
        this._queue = [];
        this._thread = null;
        this._dead = false;
        const href = String(url);
        const type = (opts && opts.type) === 'module' ? 'module' : 'classic';
        if (href.startsWith('blob:')) {
            const blob = resolveObjectURL(href);
            if (!blob) throw new Error('NodeWebWorker: unknown blob URL ' + href);
            blob.text().then(code => this._start({ kind: 'classic', code: code, name: 'blob-worker' }),
                e => this._fail(e));
        } else {
            const file = href.startsWith('file:') ? fileURLToPath(href.split('?')[0]) : path.resolve(href.split('?')[0]);
            this._start({ kind: type, file: file, url: pathToFileURL(file).href, name: path.basename(file) });
        }
    }
    _start(spec) {
        if (this._dead) return;
        const t = new ThreadWorker(BOOTSTRAP, { workerData: Object.assign({ quiet: quietWorkers }, spec) });
        t.on('message', m => this._deliver(m));
        t.on('error', e => this._fail(e));
        this._thread = t;
        const q = this._queue;
        this._queue = [];
        q.forEach(([m, tr]) => t.postMessage(m, tr));
    }
    _deliver(m) {
        if (m && m.__lucidWorkerError) { this._fail(new Error(m.message)); return; }
        const ev = new MessageEvent('message', { data: m });
        if (typeof this.onmessage === 'function') this.onmessage(ev);
        this.dispatchEvent(ev);
    }
    _fail(e) {
        const ev = new Event('error');
        ev.message = (e && (e.stack || e.message)) || String(e);
        ev.error = e;
        if (typeof this.onerror === 'function') this.onerror(ev);
        this.dispatchEvent(ev);
    }
    postMessage(msg, transfer) {
        const m = encodeFiles(msg, 0), tr = transferList(transfer);
        if (this._thread) this._thread.postMessage(m, tr);
        else this._queue.push([m, tr]);
    }
    terminate() {
        this._dead = true;
        if (this._thread) this._thread.terminate();
    }
}

/**
 * Install the platform: `globalThis.Worker` (when Node has none) and the
 * repository root, returned for code that resolves files against
 * `document.baseURI` in the browser.
 * @returns {{ baseURI: string }}
 */
export function installNodePlatform() {
    if (typeof globalThis.Worker !== 'function') globalThis.Worker = NodeWebWorker;
    // No browser storage: the app's settings store then reads its defaults (or
    // the CLI's --settings). Node >= 25 has a localStorage that warns when read
    // without --localstorage-file; replace it so nothing is read or written.
    if (!process.execArgv.some(a => a.startsWith('--localstorage-file'))) {
        Object.defineProperty(globalThis, 'localStorage', { value: undefined, configurable: true, writable: true });
    }
    return { baseURI: REPO_ROOT_URL };
}
