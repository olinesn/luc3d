/**
 * h5-status-guard.js — make the h5wasm LUC3D's own code uses throw when HDF5 says a
 * call failed.
 *
 * h5wasm's C++ returns HDF5's status and its JS drops it. So a read HDF5 gave up on (a
 * damaged compressed chunk, a zero page of a cloud file the cold-read patch could not
 * repair) hands back whatever was in the buffer, and on strings h5wasm then frees pointers
 * from that buffer, which can abort the whole module. A write HDF5 could not finish looks
 * done: the file is short, or unreadable, and nothing says so.
 *
 * This is `guardH5Status` from the `LUCID local patch (failed status)` in lib/sleap-io,
 * which guards the h5wasm instances inside the bundle (`guardH5Status`,
 * `guardH5StatusWorker`). Keep them in lockstep (tests/test-h5-status-guard.mjs runs all
 * three on the same cases). This one is for the instances outside it: the raw worker
 * (`loading/slp-import-worker.js`), the frame worker and `initH5wasm`
 * (`import-export/file-io.js`).
 *
 * Status alone doesn't cover writes. What `create_dataset` and `create_attribute` return
 * is their last cleanup's status, and `set_dataset_data`'s leaves out the H5Dclose that
 * writes the chunks it has cached. But HDF5 prints the error stack of any API call that
 * fails (to fd 2, and only then), so a write call fails if that happened during it.
 *
 * No imports, so workers and a Node test can load it.
 */

const WRITE_CALLS = ['create_dataset', 'create_vlen_str_dataset', 'create_attribute', 'create_vlen_str_attribute',
    'create_group', 'set_dataset_data', 'resize_dataset', 'flush'];
const FAILED_WRITE_MESSAGE = 'Could not finish writing the file: HDF5 reported a failed write, so it may be incomplete.';

/**
 * @param {Object} h5 The h5wasm module (`Module`, `File`), after `ready`.
 * @param {(err:Error)=>void} [onFailedRead] Told of each failed read as it happens, for
 *   a caller whose reads are try/caught to treat what a file doesn't have as absent, so
 *   that this error can't be taken for that (`throwIfFailedRead` in the raw worker).
 */
export function guardH5Status(h5, onFailedRead) {
    const calls = h5.Module;
    // A build that doesn't expose these has nothing here to guard.
    if (!calls || !h5.File) return;

    // A read that doesn't finish leaves its buffer as malloc did, and h5wasm then goes on
    // to free the strings it takes to be in there.
    let unfilled = null;
    const guardRead = (name, what) => {
        const read = calls[name];
        calls[name] = function (...args) {
            unfilled = args[args.length - 1];
            const status = read.apply(this, args);
            if (status < 0) {
                const err = new Error('Could not read ' + what(args) + ' from the file. Part of it may be damaged.');
                err.failedRead = true;
                if (onFailedRead) onFailedRead(err);
                throw err;
            }
            unfilled = null;
            return status;
        };
    };
    guardRead('get_dataset_data', (args) => String(args[1]));
    guardRead('get_attribute_data', (args) => 'the ' + args[2] + ' attribute of ' + args[1]);
    for (const name of ['reclaim_vlen_memory', 'reclaim_vlen_memory_count']) {
        const reclaim = calls[name];
        calls[name] = function (...args) {
            if (args[3] === unfilled) return 0;
            return reclaim.apply(this, args);
        };
    }

    // Status alone doesn't cover writes: see the top of this file.
    let reported = false;
    if (h5.FS) {
        const fsWrite = h5.FS.write;
        h5.FS.write = function (stream) {
            if (stream && stream.fd === 2) reported = true;
            return fsWrite.apply(this, arguments);
        };
    }
    for (const name of WRITE_CALLS) {
        const write = calls[name];
        if (!write) continue;
        calls[name] = function (...args) {
            reported = false;
            const status = write.apply(this, args);
            if (status < 0 || reported) throw new Error(FAILED_WRITE_MESSAGE);
            return status;
        };
    }
    const prototype = h5.File.prototype;
    const close = prototype.close;
    prototype.close = function () {
        reported = false;
        const status = close.call(this);
        // Nothing is at stake on a file opened for reading.
        if ((status < 0 || reported) && this.mode !== 'r') throw new Error(FAILED_WRITE_MESSAGE);
        return status;
    };
}
