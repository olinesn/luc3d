/**
 * node-worker-bootstrap.mjs — the inside of cli/node-platform.mjs's
 * `NodeWebWorker`: a worker thread dressed as a browser worker's global scope,
 * then the worker script itself (a module, or a classic script such as
 * sleap-io.js's blob worker).
 *
 * Provides `self`, `postMessage`, `onmessage` / `addEventListener('message')`,
 * `importScripts` (classic scripts, run in the global scope), and
 * `FileReaderSync` — which h5wasm's WORKERFS uses to read the File it mounted.
 * A File arrives as its path (cli/node-platform.mjs `DiskFile`) and is
 * revived here as a File whose `slice()` reads the disk with `fs.readSync`.
 */
import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import { createRequire } from 'node:module';
import { parentPort, workerData } from 'node:worker_threads';
import { fileURLToPath, pathToFileURL } from 'node:url';

const DISK_FILE_TAG = '__lucidDiskFile';

class WorkerDiskFile extends File {
    constructor(d) {
        super([], d.name);
        this._path = d.path; this._start = d.start; this._end = d.end; this._mtime = d.mtime;
    }
    get size() { return this._end - this._start; }
    get lastModified() { return Math.floor(this._mtime); }
    get lastModifiedDate() { return new Date(this._mtime); }
    slice(start, end) {
        const n = this.size;
        let a = start == null ? 0 : (start < 0 ? Math.max(n + start, 0) : Math.min(start, n));
        let b = end == null ? n : (end < 0 ? Math.max(n + end, 0) : Math.min(end, n));
        if (b < a) b = a;
        return new WorkerDiskFile({ path: this._path, start: this._start + a, end: this._start + b, mtime: this._mtime, name: this.name });
    }
    readSync() {
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
    async arrayBuffer() { return this.readSync().buffer; }
}

function revive(v, depth) {
    if (depth > 6 || v == null || typeof v !== 'object') return v;
    if (v[DISK_FILE_TAG]) return new WorkerDiskFile(v);
    if (Array.isArray(v)) return v.map(x => revive(x, depth + 1));
    if (Object.getPrototypeOf(v) !== Object.prototype) return v;
    for (const k of Object.keys(v)) v[k] = revive(v[k], depth + 1);
    return v;
}

if (workerData.quiet) console.log = console.info = console.debug = () => {};

const listeners = new Set();
globalThis.self = globalThis;
// Emscripten (h5wasm) asks this to tell a worker apart; WORKERFS asserts it.
globalThis.WorkerGlobalScope = globalThis.WorkerGlobalScope || class WorkerGlobalScope {};
globalThis.onmessage = null;
globalThis.postMessage = function (m, transfer) {
    const tr = Array.isArray(transfer) ? transfer.filter(x => x instanceof ArrayBuffer) : undefined;
    parentPort.postMessage(m, tr);
};
globalThis.addEventListener = function (type, fn) { if (type === 'message') listeners.add(fn); };
globalThis.removeEventListener = function (type, fn) { if (type === 'message') listeners.delete(fn); };
globalThis.close = function () { process.exit(0); };
globalThis.FileReaderSync = class FileReaderSync {
    readAsArrayBuffer(blob) {
        if (blob instanceof WorkerDiskFile) return blob.readSync().buffer;
        throw new Error('FileReaderSync (Node): only files opened from disk can be read');
    }
};
// Classic-worker scripts run in the global scope, so give them `require`
// (an Emscripten module under Node loads `fs`/`path` through it).
globalThis.require = createRequire(import.meta.url);
globalThis.importScripts = function () {
    for (const u of arguments) {
        const s = String(u);
        const file = s.startsWith('file:') ? fileURLToPath(s.split('?')[0]) : path.resolve(s.split('?')[0]);
        vm.runInThisContext(fs.readFileSync(file, 'utf8'), { filename: file });
    }
};

let ready = false;
const pending = [];
function dispatch(m) {
    const ev = { data: revive(m, 0), type: 'message' };
    if (typeof globalThis.onmessage === 'function') globalThis.onmessage(ev);
    listeners.forEach(fn => fn(ev));
}
parentPort.on('message', m => { if (ready) dispatch(m); else pending.push(m); });

process.on('uncaughtException', e => {
    parentPort.postMessage({ __lucidWorkerError: true, message: String((e && e.stack) || e) });
});

try {
    if (workerData.kind === 'module') {
        await import(workerData.url);
    } else {
        const code = workerData.code != null ? workerData.code : fs.readFileSync(workerData.file, 'utf8');
        vm.runInThisContext(code, { filename: workerData.name || 'worker.js' });
    }
} catch (e) {
    parentPort.postMessage({ __lucidWorkerError: true, message: String((e && e.stack) || e) });
}
ready = true;
pending.splice(0).forEach(dispatch);
