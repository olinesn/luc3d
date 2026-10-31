/**
 * behavior-pool.js — compute behavior features, score rule detectors and
 * classifiers, extract training rows and train forest trees, chunk by chunk
 * (or batch by batch), on a pool of Web Workers (`pose/behavior-worker.js`).
 *
 * WHY: features are independent from one chunk of frames to the next once
 * each chunk is padded by the longest window (pose/behavior-features.js
 * `featurePadFrames`), so chunks run in parallel and only one chunk's
 * features exist at a time — the frames × features matrix is never built.
 * The workers run the same pure code as the main thread, so results do not
 * depend on where they ran.
 *
 * Usage (main thread):
 *
 *     await runBehaviorChunks(chunks, function makeJob(i) { return {kind, spec, chunk, data, plan, extra}; },
 *         function onDone(i, result) { ...copy into whole-session arrays... },
 *         { onProgress: function (done, total) {...}, isCancelled, pause });
 *
 * `onDone` may run in any order (each chunk owns its frames). A job's data is
 * made only when a worker is free for it (`makeJob`), so at most ~2 chunks per
 * worker are in memory.
 *
 * Job kinds: 'scores' (rule detectors), 'features' (feature columns), 'rows'
 * (classifier training rows, as codes), 'forest' (classifier scores) and
 * 'trees' (grow some trees of a forest; no poses). What every job of a run
 * shares — the forests to score with, the binned training matrix — goes to
 * each worker ONCE per run (`opts.shared = {key, value}`; a worker keeps the
 * last one it was sent, and gets it again only when the key changes), not
 * with every job: there is no SharedArrayBuffer without cross-origin
 * isolation, so each worker holds its own copy. `opts.releaseShared` drops
 * the copies when the run ends (a training matrix is tens of MB).
 *
 * Runs INLINE (same calls, one chunk at a time, `opts.pause()` awaited between
 * them so the page can paint) when workers are unavailable, the run is small
 * (`opts.workload` below `MIN_WORKLOAD`, where worker start-up would
 * dominate), or `window.LUCID_BEHAVIOR_WORKER === 0` (the kill switch). Set it
 * to N > 0 to force an N-worker pool regardless of size (tests). A chunk a
 * worker fails on — or a worker that dies — is redone inline.
 *
 * Pool size `min(hardwareConcurrency − 1, 8)`; workers persist between runs
 * and are terminated after 30 s idle. One run owns the pool at a time (a
 * concurrent second run goes inline).
 */
import { computeFeatureChunk } from './behavior-features.js';
import { scoreDetectorsChunk } from './behavior-rules.js';
import { extractTrainingRows, scoreForestsChunk, trainTrees } from './behavior-forest.js';

/** Below this many animal-frames a run stays inline. */
export const MIN_WORKLOAD = 40000;
/** Workers are kept between runs, then released after this idle time. */
const IDLE_TERMINATE_MS = 30000;

let pool = null;          // { workers: [{ w, busy }], idleTimer }
let poolBusy = false;

function forcedSize() {
    return (typeof window !== 'undefined' && window.LUCID_BEHAVIOR_WORKER != null) ? window.LUCID_BEHAVIOR_WORKER : undefined;
}

function poolSize() {
    const forced = forcedSize();
    if (forced != null) return Math.max(0, forced | 0);
    const hc = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 1;
    return Math.max(0, Math.min(hc - 1, 8));
}

function getPool() {
    if (typeof Worker !== 'function' || typeof document === 'undefined') return null;
    const n = poolSize();
    if (n < 1) return null;
    if (pool && pool.workers.length === n) {
        clearTimeout(pool.idleTimer);
        return pool;
    }
    if (pool) terminatePool();
    try {
        const url = new URL('pose/behavior-worker.js', document.baseURI);
        const workers = [];
        for (let i = 0; i < n; i++) workers.push({ w: new Worker(url, { type: 'module' }), busy: false });
        pool = { workers, idleTimer: null };
        return pool;
    } catch (e) {
        console.warn('[behavior-pool] workers unavailable, computing inline:', e);
        pool = null;
        return null;
    }
}

function terminatePool() {
    if (!pool) return;
    clearTimeout(pool.idleTimer);
    for (const x of pool.workers) x.w.terminate();
    pool = null;
}

/**
 * What one job computes, on whichever thread runs it (the worker runs the
 * same switch).
 * @param {Object} job
 * @param {*} [shared] the run's shared value (`opts.shared.value`)
 */
export function runJob(job, shared) {
    if (job.kind === 'trees') return trainTrees(shared, job.trees);
    const chunk = { start: job.chunk.start, end: job.chunk.end, padStart: job.chunk.padStart, padEnd: job.chunk.padEnd, data: job.data };
    if (job.kind === 'scores') return scoreDetectorsChunk(job.spec, chunk, job.plan, job.extra);
    if (job.kind === 'rows') return extractTrainingRows(job.spec, chunk, job.request);
    if (job.kind === 'forest') return scoreForestsChunk(job.spec, chunk, job.plan, shared);
    return computeFeatureChunk(job.spec, chunk, job.request);
}

/**
 * Run `count` chunk jobs.
 * @param {number} count
 * @param {function(number): Object} makeJob job i: {kind, spec, chunk, data, plan?, extra?, request?}
 * @param {function(number, Object)} onDone
 * @param {{workload?: number, onProgress?: function(number, number), isCancelled?: function(): boolean,
 *          pause?: function(): Promise, shared?: {key: string, value: *}, releaseShared?: boolean,
 *          maxWorkers?: number}} [opts]
 *   shared: sent to each worker once (see the header); maxWorkers: use at most
 *   this many (each holds a copy of `shared`)
 * @returns {Promise<{parallel: boolean, workers: number, cancelled: boolean}>}
 */
export async function runBehaviorChunks(count, makeJob, onDone, opts) {
    opts = opts || {};
    const cancelled = opts.isCancelled || (() => false);
    const progress = opts.onProgress || (() => {});
    const forced = forcedSize();
    const big = (forced != null && forced > 0) || opts.workload == null || opts.workload >= MIN_WORKLOAD;
    const P = (big && count > 1 && !poolBusy) ? getPool() : null;
    const shared = opts.shared || null;
    const sharedValue = shared ? shared.value : undefined;
    if (!P) {
        for (let i = 0; i < count; i++) {
            if (cancelled()) return { parallel: false, workers: 0, cancelled: true };
            onDone(i, runJob(makeJob(i), sharedValue));
            progress(i + 1, count);
            if (opts.pause && i + 1 < count) await opts.pause();
        }
        return { parallel: false, workers: 0, cancelled: false };
    }
    const slots = opts.maxWorkers > 0 ? P.workers.slice(0, Math.max(1, opts.maxWorkers | 0)) : P.workers;
    poolBusy = true;
    let next = 0, done = 0, failure = null, stop = false;
    const jobs = new Map();
    try {
        await new Promise((resolve) => {
            const finishIfDone = () => { if (done >= count || (stop && jobs.size === 0)) resolve(); };
            const dispatch = (slot) => {
                if (stop || failure) { finishIfDone(); return; }
                if (cancelled()) { stop = true; finishIfDone(); return; }
                if (next >= count) { finishIfDone(); return; }
                const i = next++;
                const job = makeJob(i);
                slot.busy = true;
                jobs.set(i, { job, slot });
                if (shared && slot.sharedKey !== shared.key) {
                    slot.w.postMessage({ type: 'shared', key: shared.key, value: shared.value });
                    slot.sharedKey = shared.key;
                }
                const { data, ...rest } = job;
                const list = data || [];
                slot.w.postMessage(Object.assign({ type: 'job', id: i, sharedKey: shared ? shared.key : null }, rest, { data: list }),
                    list.map((a) => a.buffer));
            };
            const settle = (slot, i, result) => {
                jobs.delete(i);
                slot.busy = false;
                try { onDone(i, result); } catch (e) { failure = failure || e; }
                done++;
                progress(done, count);
                if (done >= count) { resolve(); return; }
                dispatch(slot);
            };
            for (const slot of slots) {
                slot.w.onmessage = (e) => {
                    const m = e.data;
                    const entry = jobs.get(m.id);
                    if (!entry) return;
                    if (m.type === 'done') { settle(slot, m.id, m.result); return; }
                    console.warn('[behavior-pool] a worker failed a chunk; computing it inline:', m.message);
                    // The job's data went to the worker; rebuild it.
                    let r = null;
                    try { r = runJob(makeJob(m.id), sharedValue); } catch (err) { failure = failure || err; }
                    settle(slot, m.id, r);
                };
                slot.w.onerror = (ev) => {
                    if (ev && ev.preventDefault) ev.preventDefault();
                    // A dead worker: redo its chunk inline and hand it no more.
                    for (const [i, entry] of jobs) {
                        if (entry.slot !== slot) continue;
                        let r = null;
                        try { r = runJob(makeJob(i), sharedValue); } catch (err) { failure = failure || err; }
                        jobs.delete(i);
                        try { onDone(i, r); } catch (e) { failure = failure || e; }
                        done++;
                        progress(done, count);
                    }
                    slot.dead = true;
                    if (slots.every((s) => s.dead)) {
                        while (next < count && !failure) {
                            const i = next++;
                            try { onDone(i, runJob(makeJob(i), sharedValue)); } catch (e) { failure = failure || e; }
                            done++;
                            progress(done, count);
                        }
                    }
                    finishIfDone();
                };
            }
            for (const slot of slots) dispatch(slot);
            finishIfDone();
        });
    } finally {
        for (const slot of slots) {
            slot.w.onmessage = null; slot.w.onerror = null; slot.busy = false;
            if (opts.releaseShared && slot.sharedKey != null && !slot.dead) {
                slot.w.postMessage({ type: 'shared', key: null, value: null });
                slot.sharedKey = null;
            }
        }
        if (P.workers.some((s) => s.dead)) terminatePool();
        else if (pool === P) { clearTimeout(P.idleTimer); P.idleTimer = setTimeout(terminatePool, IDLE_TERMINATE_MS); }
        poolBusy = false;
    }
    if (failure) throw failure;
    return { parallel: true, workers: slots.length, cancelled: stop };
}
