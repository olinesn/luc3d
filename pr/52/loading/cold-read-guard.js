/**
 * cold-read-guard.js — Blob reads that survive cloud files still downloading.
 *
 * In a folder synced by Google Drive for desktop on macOS (File Provider), a
 * read of a part of a file that has not been downloaded yet can go wrong in
 * two ways, with nothing else amiss (both measured 2026-10-19):
 *
 *   - it FAILS: `arrayBuffer()` rejects (NotReadableError) and a Blob stream
 *     errors with `TypeError: network error`. On cold 8-camera loads this cost
 *     2 of 48 cameras their frame-accurate decoder (they stepped by `<video>`
 *     seeks after that). 30 of 31 failed reads succeeded when read again at
 *     once.
 *   - it returns a run of ZEROS with every other byte right: the first 64 KiB
 *     of a read that starts on a 1 MiB boundary of the file where nothing is
 *     downloaded yet (by `arrayBuffer()` and by stream alike), 16 KiB pages in
 *     h5wasm's `.slp` reads (which the `LUCID local patch (cold-read)` in
 *     lib/h5wasm guards). A re-read returned the real bytes every time.
 *
 * Drive fetches a video only in the ranges that are read, so every seek into a
 * part of a video nobody has looked at yet is such a read, not just the load.
 *
 * `readBlobRange` is a Blob read that handles both:
 *   - a read that throws is tried again, after `COLD_READ_RETRY_MS` delays,
 *     before the last error is rethrown;
 *   - every part of a read that came back all zero (per 4 KiB block of the
 *     FILE, clipped to the read; neighbouring blocks merged) is read again
 *     TOGETHER WITH THE 4 KiB BLOCK ON EACH SIDE OF IT, and the re-read is
 *     copied in only where the first read gave a zero byte — the same rule as
 *     h5wasm's `lucidColdReadGuard`, so a re-read can never make a read worse.
 *     The neighbours are there because a run of zeros need not start or end on
 *     a block boundary: the block where it ends part-way (zeros, then real
 *     bytes) is not all zero, so re-reading only the all-zero blocks left its
 *     zeros in place — on cold 167 MB `.slp` files through h5wasm's copy of
 *     the rule, every wrong block was that one (93 of 93). The repaired bytes
 *     are checked again the same way, at most three re-reads deep. Bytes that
 *     really are zero cost one small re-read.
 *
 * A file that has GONE AWAY (moved, or on a share that dropped) fails the same
 * way, but for good, and retrying every read then costs 4.2 s each, for nothing.
 * `createReadWatch` tells the two apart for one open file: a read goes on being
 * retried until one has used up every retry; from then on the file is `failing`,
 * every read gets one try, and the failed read is tried again every
 * FILE_RECHECK_MS until a read works, which ends it. Reads of a file still
 * downloading keep all their retries, since nearly all of them work when read
 * again (above). A dropped share can also leave a read unanswered for minutes
 * rather than failing it, so under a watch a read has a deadline as well.
 *
 * No imports, so a Node test can load it (tests/test-cold-read-guard.mjs).
 */

export const COLD_READ_BLOCK = 4096;

/** Delays before each retry of a read that threw; the error stands after the last. */
export const COLD_READ_RETRY_MS = [200, 1000, 3000];

/** While a file is failing (createReadWatch): how often its failed read is tried again, to notice it is back. */
export const FILE_RECHECK_MS = 2000;

/**
 * Under a watch, a read still unanswered after this long counts as failed, with
 * no retry. Generous: a read is at most a 1 MiB window, and a read that answers
 * later still ends the failure.
 */
export const READ_STALL_MS = 15000;

/** The same deadline once the file is already failing, when nobody should wait on it. */
export const READ_STALL_FAILING_MS = 2000;

/** The errors of reads that failed under an armed watch (isWatchedReadFailure). */
var watchedFailures = new WeakSet();

/** Is this the error of a read a watch has already seen fail? */
export function isWatchedReadFailure(err) {
    return err !== null && typeof err === 'object' && watchedFailures.has(err);
}

/** Page-wide counts (for the console and tests): zero runs re-read / repaired, reads retried / rescued. */
export const coldReadStats = { rereads: 0, recovered: 0, retries: 0, rescued: 0 };

/**
 * The all-zero parts of `bytes`, which were read from file offset `position`:
 * `[start, end)` file ranges, per 4 KiB block of the file clipped to the read,
 * neighbouring blocks merged. Empty when no block is all zero.
 * @param {Uint8Array} bytes
 * @param {number} position
 * @returns {Array<[number, number]>}
 */
export function zeroRuns(bytes, position) {
    var runs = [];
    var end = position + bytes.length;
    for (var b = position - position % COLD_READ_BLOCK; b < end; b += COLD_READ_BLOCK) {
        var lo = b > position ? b : position;
        var hi = b + COLD_READ_BLOCK < end ? b + COLD_READ_BLOCK : end;
        var zero = true;
        for (var q = lo - position, qe = hi - position; q < qe; q++) {
            if (bytes[q] !== 0) { zero = false; break; }
        }
        if (!zero) continue;
        var last = runs[runs.length - 1];
        if (last && last[1] === lo) last[1] = hi;
        else runs.push([lo, hi]);
    }
    return runs;
}

function sleep(ms) {
    return new Promise(function (r) { setTimeout(r, ms); });
}

function nameOf(blob) {
    return (blob && blob.name) || 'file';
}

/**
 * One Blob read. Under an armed watch it has a deadline, and a read of a failing
 * file with an earlier read still unanswered fails at once: on a hung share each
 * new read would only hang as well, and they would pile up.
 */
function readOnce(blob, start, end, watch) {
    if (!watch || !watch.armed) return blob.slice(start, end).arrayBuffer();
    if (watch.failing && watch.stalled > 0) {
        return Promise.reject(new Error(nameOf(blob) + ' is not answering reads'));
    }
    return new Promise(function (resolve, reject) {
        var read = blob.slice(start, end).arrayBuffer();
        var ms = watch.failing ? watch.stallFailingMs : watch.stallMs;
        var late = false;
        var timer = setTimeout(function () {
            late = true;
            watch.stalled++;
            var err = new Error(nameOf(blob) + ' did not answer a read in ' + ms + ' ms');
            watch.noteFailed(start, end, err);
            reject(err);
        }, ms);
        read.then(function (ab) {
            clearTimeout(timer);
            if (!late) { resolve(ab); return; }
            // nobody is waiting for these bytes any more, but it did answer
            watch.stalled--;
            watch.noteRead();
        }, function (e) {
            clearTimeout(timer);
            if (!late) reject(e);
            else watch.stalled--;
        });
    });
}

/**
 * `[start, end)` of `blob` as bytes, trying again when the read throws. Under a
 * watch of a file already failing there is only the one try.
 */
async function readRetrying(blob, start, end, watch) {
    for (var attempt = 0; ; attempt++) {
        try {
            var bytes = new Uint8Array(await readOnce(blob, start, end, watch));
            if (attempt > 0) {
                coldReadStats.rescued++;
                console.warn('[cold-read] ' + nameOf(blob) + ' bytes ' + start + '-' + end + ' read on try '
                    + (attempt + 1) + ' (file still downloading?)');
            }
            if (watch) watch.noteRead();
            return bytes;
        } catch (e) {
            if (attempt >= COLD_READ_RETRY_MS.length || (watch && watch.failing)) {
                if (watch) watch.noteFailed(start, end, e);
                throw e;
            }
            coldReadStats.retries++;
            await sleep(COLD_READ_RETRY_MS[attempt]);
        }
    }
}

/**
 * Re-read every all-zero part of `bytes` (read from `position` of `blob`),
 * with the 4 KiB block on each side of it, and put the real bytes in place, in
 * `bytes` itself: a re-read byte is copied in only where `bytes` holds a zero.
 * @param {Blob} blob
 * @param {Uint8Array} bytes
 * @param {number} position
 * @param {number} [depth]
 * @param {Object} [watch] - createReadWatch's, for the re-reads
 * @returns {Promise<Uint8Array>} `bytes`
 */
export async function repairZeroRuns(blob, bytes, position, depth, watch) {
    depth = depth || 0;
    var end = position + bytes.length;
    var runs = zeroRuns(bytes, position);
    for (var i = 0; i < runs.length; i++) {
        // A run need not end (or start) on a block boundary: the partly-zero
        // block on each side is read again with it.
        var from = Math.max(position, runs[i][0] - COLD_READ_BLOCK);
        var to = Math.min(end, runs[i][1] + COLD_READ_BLOCK);
        var again = await readRetrying(blob, from, to, watch);
        coldReadStats.rereads++;
        var fixed = 0;
        for (var k = 0, at = from - position, n = Math.min(again.length, to - from); k < n; k++, at++) {
            if (bytes[at] === 0 && again[k] !== 0) { bytes[at] = again[k]; fixed++; }
        }
        if (!fixed) continue;
        coldReadStats.recovered++;
        console.warn('[cold-read] ' + nameOf(blob) + ' bytes ' + from + '-' + to
            + ' first read as zeros were read again (file still downloading?)');
        // The re-read can carry a hole of its own.
        if (depth < 2) await repairZeroRuns(blob, bytes.subarray(from - position, to - position), from, depth + 1, watch);
    }
    return bytes;
}

/**
 * Bytes `[start, end)` of `blob`, retried when the read throws and with every
 * all-zero part re-read (see the module comment).
 * @param {Blob} blob
 * @param {number} start
 * @param {number} end
 * @param {Object} [watch] - createReadWatch's, when `blob` is a file kept open
 * @returns {Promise<Uint8Array>}
 */
export async function readBlobRange(blob, start, end, watch) {
    var bytes = await readRetrying(blob, start, end, watch);
    return repairZeroRuns(blob, bytes, start, 0, watch);
}

/**
 * The whole of `blob` as an ArrayBuffer, guarded like `readBlobRange` —
 * `blob.arrayBuffer()` for files that may be in a cloud-synced folder.
 * @param {Blob} blob
 * @returns {Promise<ArrayBuffer>}
 */
export async function readBlobArrayBuffer(blob) {
    return (await readBlobRange(blob, 0, blob.size)).buffer;
}

/**
 * Whether one open file can be read right now, learned from the reads made with
 * this watch (`readBlobRange`, `guardedRangeSource`) and from `check()`.
 *
 * It does nothing until `arm()`: while a file is being opened every read keeps all
 * its retries, as a failed read there costs a camera its frame-accurate decoder
 * for the whole session. Armed, a read that fails for good (or passes its
 * deadline) makes the file `failing`, and `onChange(true, err)` is called; any
 * read that works ends that, and `onChange(false)` is called.
 *
 * Any read, not only the failed part: a cloud file can be readable in the parts
 * downloaded and not in the others. The recheck reads what failed, though, since
 * reading byte 0 would call such a file back when it is not.
 *
 * @param {Blob} blob
 * @param {{ onChange?: function(boolean, Error=): void, recheckMs?: number, stallMs?: number, stallFailingMs?: number }} [opts]
 */
export function createReadWatch(blob, opts) {
    var onChange = (opts && opts.onChange) || null;
    var recheckMs = (opts && opts.recheckMs) || FILE_RECHECK_MS;
    var timer = null, lastFailed = null, checking = null;

    function recheck() {
        timer = null;
        if (watch.failing) watch.check();
    }

    var watch = {
        armed: false,
        failing: false,
        stalled: 0,   // reads past their deadline and still unanswered
        stallMs: (opts && opts.stallMs) || READ_STALL_MS,
        stallFailingMs: (opts && opts.stallFailingMs) || READ_STALL_FAILING_MS,

        arm: function () { watch.armed = true; },

        /** For good: no more rechecks and no more `onChange`. */
        stop: function () {
            watch.armed = false;
            watch.failing = false;
            lastFailed = null;
            onChange = null;
            if (timer) { clearTimeout(timer); timer = null; }
        },

        noteRead: function () {
            lastFailed = null;
            if (!watch.failing) return;
            watch.failing = false;
            if (timer) { clearTimeout(timer); timer = null; }
            if (onChange) onChange(false);
        },

        noteFailed: function (start, end, err) {
            if (!watch.armed) return;
            if (err !== null && typeof err === 'object') watchedFailures.add(err);
            lastFailed = [start, end];
            if (!timer) timer = setTimeout(recheck, recheckMs);
            if (watch.failing) return;
            watch.failing = true;
            if (onChange) onChange(true, err);
        },

        /**
         * Read the file now, for what the watch can't see (a `<video>` reads it
         * natively): what failed last, else its first byte, with the same retries
         * as any other read. Resolves to whether it could be read.
         * @returns {Promise<boolean>}
         */
        check: function () {
            if (!checking) {
                var range = lastFailed || [0, Math.min(1, blob.size)];
                checking = readRetrying(blob, range[0], range[1], watch).then(
                    function () { checking = null; return true; },
                    function () { checking = null; return false; });
            }
            return checking;
        },
    };
    return watch;
}

/** guardedRangeSource reads whole aligned windows of this size ... */
export const RANGE_WINDOW_BYTES = 1 << 20;
/** ... and keeps the most recent ones up to this many bytes. */
export const RANGE_CACHE_BYTES = 4 << 20;

/**
 * A `{ size, readRange(start, length) }` range source over `blob` whose reads
 * are guarded like `readBlobRange` — the shape sleap-io.js's
 * `MediaBunnyVideoBackend.fromRangeSource` takes (it drives mediabunny's
 * `StreamSource` with it), so the video decoder reads a File through the
 * guard instead of through mediabunny's own `BlobSource`.
 *
 * mediabunny asks a `StreamSource` for ~64–192 KiB at a time, and a Blob read
 * costs about the same however small it is: one read per request made reading
 * packets ~6x slower than `BlobSource`, which streams. So a request is served
 * from a `RANGE_WINDOW_BYTES`-aligned window read whole (one guarded read per
 * window, the last few kept), and sequential reads cost one Blob read per MiB.
 * A request is answered with a copy, so mediabunny's own cache never pins a
 * whole window.
 * @param {Blob} blob
 * @param {{ windowBytes?: number, cacheBytes?: number, watch?: Object }} [opts] - `watch`: createReadWatch's
 */
export function guardedRangeSource(blob, opts) {
    var W = (opts && opts.windowBytes) || RANGE_WINDOW_BYTES;
    var cap = (opts && opts.cacheBytes) || RANGE_CACHE_BYTES;
    var readWatch = (opts && opts.watch) || undefined;
    var size = blob.size;
    var windows = [];   // { start, end, bytes: Promise<Uint8Array> }, most recently used last

    function lookup(start, end) {
        for (var i = windows.length - 1; i >= 0; i--) {
            var w = windows[i];
            if (w.start <= start && end <= w.end) {
                if (i !== windows.length - 1) { windows.splice(i, 1); windows.push(w); }
                return w;
            }
        }
        return null;
    }

    function load(start, end) {
        var w = { start: Math.floor(start / W) * W, end: Math.min(size, Math.ceil(end / W) * W), bytes: null };
        w.bytes = readBlobRange(blob, w.start, w.end, readWatch);
        // a failed read must not stay cached: the next request reads again
        w.bytes.catch(function () { var i = windows.indexOf(w); if (i >= 0) windows.splice(i, 1); });
        windows.push(w);
        var held = 0;
        for (var i = windows.length - 1; i >= 0; i--) {
            held += windows[i].end - windows[i].start;
            if (held > cap && i < windows.length - 1) { windows.splice(0, i + 1); break; }
        }
        return w;
    }

    return {
        size: size,
        readRange: async function (start, length) {
            var end = start + length;
            var w = lookup(start, end) || load(start, end);
            var bytes = await w.bytes;
            return bytes.slice(start - w.start, end - w.start);
        },
    };
}
