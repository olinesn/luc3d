// pose/behavior-discovery.js — finding behaviors without labels, in the browser:
// B-SOiD's pipeline (Hsu & Yttri 2021) over the behavior features.
//
//   1. Features in 100 ms bins (pose/behavior-discovery-core.js), per animal, on
//      every session given. The sample is at most `MAX_BINS` bins across sessions
//      and animals: the bins whose seeded hash is smallest, the same bins for the
//      same poses.
//   2. Standardized, PCA to 70% of the variance (pose/behavior-embedding.js).
//   3. UMAP (pose/behavior-umap.js): n_neighbors 60, min_dist 0, from the exact
//      neighbours in PCA space (worker pool), starting from the first principal
//      components; and side by side a second layout from a random view of the same
//      data, the test the next step uses.
//   4. HDBSCAN on both layouts (pose/hdbscan.js), at the smallest-cluster size in
//      0.5–1% of the bins that gives the most clusters which the other layout also
//      holds and which happen in several separate stretches of time; any other
//      cluster is noise (these steps find clusters even in pure noise, and in one
//      long rest: pose/behavior-embedding.js). Clusters are numbered by size, largest
//      first, so state k takes the k-th state color.
//   5. A forest per cluster, "this cluster" against everything else (the
//      classifiers' trees, pose/behavior-forest.js), and one more for the bins in no
//      cluster when there are any. Whole 10 s stretches are held out to say how well
//      the forests repeat the clusters on bins they never saw (neighbouring bins are
//      near copies).
//   6. Every frame's own 100 ms bin is scored by every forest and takes the best
//      one: the same forests on the same features at every phase, B-SOiD's
//      frameshift, at the session's full frame rate. A frame with no pose has no
//      state, and neither has a frame that looks like the bins in no cluster: B-SOiD
//      gives every frame a cluster, but when a few islands stand out of a continuum
//      that would force the continuum into the islands.
//
// The result becomes a `behaviorStates` layer per session (source 'discovery',
// pose/behavior-states.js), which is what is saved; the layout, the sample and
// what stands out in each cluster stay in memory only (`discoveryResult`), since
// the same poses give them again. "Split a behavior" (A-SOiD's directed discovery)
// is the same pipeline on the frames labelled present of one behavior: its layer
// has states only on those frames.
//
// Deterministic: UMAP and the forests are seeded, the neighbours are exact, rows
// are computed in a fixed chunk layout and put in a fixed order, so the same poses
// give the same clusters however the work was spread over the pool
// (tests/e2e/behavior-discovery.mjs).
//
// Memory: only the sample is held whole (Float32Array, bins × features); poses are
// read one session at a time, and frames × features is never built. UMAP's own
// arrays live in the discovery worker's heap, which ends with the run.
//
// DOM-free (a Node test runs it end to end, inline).
//
// Depends on: pose/behavior-discovery-core.js, pose/behavior-pool.js,
//             pose/behavior-features.js, pose/behavior-forest.js,
//             pose/behavior-labels.js, pose/behavior-states.js, and when it runs
//             inline pose/behavior-embedding.js, pose/behavior-umap.js

import { binFrames, binCenters, discoveryFeatureNames, speedFeatures, heldOutAgreement, randomView } from './behavior-discovery-core.js';
import { runBehaviorChunks } from './behavior-pool.js';
import { planChunks, featurePadFrames, sliceChunkData } from './behavior-features.js';
import { DEFAULT_FOREST_PARAMS, binEdges, binColumn, finalizeForest, encodeColumn, mix32, missingFeatures } from './behavior-forest.js';
import { listBouts } from './behavior-labels.js';
import { setStateLayer, runsFromSequence, cleanLayerName, stateLayer } from './behavior-states.js';

/** At most this many bins are clustered (B-SOiD's 100 ms bins, across sessions and animals). */
export var MAX_BINS = 30000;
/** UMAP's n_neighbors and min_dist. */
export var NEIGHBORS = 60;
export var MIN_DIST = 0;
/** The smallest cluster, as shares of the clustered bins (B-SOiD's). */
export var CLUSTER_SHARE = [0.005, 0.01];
/** …and never fewer bins than this (a second of one animal's time). */
export var MIN_CLUSTER_BINS = 10;

/**
 * The fewest bins a cluster can have: UMAP with min_dist 0 clumps points in groups of
 * about n_neighbors, whatever they are, so below that a "cluster" is a piece of the
 * layout. On planted states, a smallest cluster under about half of n_neighbors broke
 * them into dozens of clumps; from there up they came out whole. (B-SOiD's shares are
 * above this at its 30,000 bins; a shorter recording needs the floor.)
 */
export function clusterFloor(n, neighbors) {
    return Math.max(MIN_CLUSTER_BINS, Math.min(neighbors, Math.round(n / 20)));
}
/** Fewer usable bins than this cannot be clustered. */
export var MIN_BINS = 50;
/** Frames per chunk (fixed: a value must not depend on how a run was split). */
var CHUNK_FRAMES = 2048;
/** Trees per cluster, and how they are spread over the pool. */
var TREES_PER_CLUSTER = 50;
var TREES_PER_JOB = 5;
var TRAIN_WORKERS = 6;
/** Rows per nearest-neighbour job. */
var KNN_JOB_ROWS = 512;
/** Bins of a stretch held out together, in seconds; and how many stretches in ten. */
var HOLDOUT_STRETCH_SECONDS = 10;
var HOLDOUT_TENTHS = 1;
/** Below this many clustered bins nothing is held out. */
var HOLDOUT_MIN_ROWS = 500;
/** What stands out in a cluster: its features this far from the mean, most first. */
var PROFILE_TOP = 5;

var SEP = '\u0001';

function fnv(str) {
    var h = 0x811c9dc5;
    for (var i = 0; i < str.length; i++) h = Math.imul(h ^ str.charCodeAt(i), 0x01000193);
    return h >>> 0;
}

function noPause() { return Promise.resolve(); }

// ----------------------------------------------------------------------------
// The whole-sample steps: in the discovery worker, or inline
// ----------------------------------------------------------------------------

/**
 * The steps that read the whole sample. One worker per run: `close()` ends it and
 * frees what it held. Inline when there are no workers, or with the behavior
 * pool's kill switch (`window.LUCID_BEHAVIOR_WORKER = 0`).
 */
function openStepRunner(isCancelled, pause) {
    var inline = typeof Worker !== 'function' || typeof document === 'undefined' ||
        (typeof window !== 'undefined' && window.LUCID_BEHAVIOR_WORKER === 0);
    var worker = null, nextId = 1, pending = null, poll = null;
    if (!inline) {
        try {
            worker = new Worker(new URL('pose/behavior-discovery-worker.js', document.baseURI), { type: 'module' });
        } catch (e) {
            console.warn('[behavior-discovery] no worker, running inline:', e);
            worker = null;
        }
    }
    var settle = function (fn, v) { var p = pending; pending = null; clearInterval(poll); poll = null; if (p) fn(p, v); };
    if (worker) {
        worker.onmessage = function (e) {
            var m = e.data;
            if (!pending || m.id !== pending.id) return;
            if (m.type === 'progress') { if (pending.onProgress) pending.onProgress(m.done, m.total); return; }
            if (m.type === 'error') settle(function (p, v) { p.reject(new Error(v)); }, m.message);
            else settle(function (p, v) { p.resolve(v); }, m.result);
        };
        worker.onerror = function (ev) {
            if (ev && ev.preventDefault) ev.preventDefault();
            settle(function (p, v) { p.reject(new Error(v)); }, (ev && ev.message) || 'the discovery worker failed');
        };
    }
    /** @returns {Promise<*>} the step's result, or null once cancelled */
    function call(message, transfer, onProgress) {
        if (!worker) return callInline(message, onProgress);
        return new Promise(function (resolve, reject) {
            var id = nextId++;
            pending = { id: id, resolve: resolve, reject: reject, onProgress: onProgress };
            // Terminating is the only way into a running step.
            poll = setInterval(function () {
                if (isCancelled()) settle(function (p) { p.resolve(null); });
            }, 200);
            worker.postMessage(Object.assign({ id: id }, message), transfer || []);
        });
    }
    async function callInline(m, onProgress) {
        if (isCancelled()) return null;
        if (m.type === 'reduce') return (await import('./behavior-embedding.js')).standardizeAndReduce(m.rows, m.M, m.F, m.opts);
        if (m.type === 'cluster') return (await import('./behavior-embedding.js')).clusterLayouts(m.E, m.E2, m.n, m.d, m.opts);
        var run = (await import('./behavior-umap.js')).prepareUmap(m.knn, m.opts);
        for (var ep = 0; ep < run.epochs; ep++) {
            run.step();
            if (ep % 5 === 4) {
                if (onProgress) onProgress(ep + 1, run.epochs);
                await pause();
                if (isCancelled()) return null;
            }
        }
        return { embedding: run.embedding() };
    }
    return {
        call: call,
        close: function () { clearInterval(poll); if (worker) worker.terminate(); worker = null; },
    };
}

// ----------------------------------------------------------------------------
// Which bins
// ----------------------------------------------------------------------------

/** One animal's frames with a pose: at least half the used nodes, and not skipped as broken tracking. */
function presentMask(ctx, a) {
    var input = ctx.input, T = input.numFrames, N = input.nodeNames.length, D = input.dim;
    var used = ctx.spec.used, arr = input.data[a], need = Math.max(1, Math.ceil(used.length / 2));
    var skip = ctx.broken && ctx.broken.masks && ctx.broken.masks[a];
    var out = new Uint8Array(T);
    for (var t = 0; t < T; t++) {
        if (skip && skip[t]) continue;
        var base = t * N * D, c = 0;
        for (var i = 0; i < used.length; i++) { var v = arr[base + used[i] * D]; if (v === v) c++; }
        if (c >= need) out[t] = 1;
    }
    return out;
}

/**
 * The frames of one behavior's present labels in a session, per animal (a mutual
 * pair: both animals; a social behavior otherwise: the actor).
 * @returns {Map<string, number[][]>} sorted, merged [start, end]
 */
export function behaviorFramesByAnimal(session, behaviorId, mutual) {
    var byAnimal = new Map();
    var add = function (a, s, e) { if (!byAnimal.has(a)) byAnimal.set(a, []); byAnimal.get(a).push([s, e]); };
    listBouts(session, { behaviorId: behaviorId, present: true }).forEach(function (b) {
        add(b.animal, b.start, b.end);
        if (mutual && b.target) add(b.target, b.start, b.end);
    });
    byAnimal.forEach(function (ivs, a) {
        ivs.sort(function (x, y) { return x[0] - y[0] || x[1] - y[1]; });
        var out = [];
        ivs.forEach(function (iv) {
            var last = out[out.length - 1];
            if (last && iv[0] <= last[1] + 1) last[1] = Math.max(last[1], iv[1]);
            else out.push([iv[0], iv[1]]);
        });
        byAnimal.set(a, out);
    });
    return byAnimal;
}

/**
 * The usable bins of one context, per animal: whole bins with a pose on at least
 * half their frames, and for a split, centered in one of the behavior's bouts.
 * @returns {Array<Int32Array>} per animal, center frames
 */
function usableBins(ctx, w, intervals) {
    var input = ctx.input, T = input.numFrames;
    var hb = w >> 1, centers = binCenters(T, w);
    return input.animals.map(function (name, a) {
        var ivs = intervals ? intervals.get(name) : null;
        if (intervals && !ivs) return new Int32Array(0);
        var pres = presentMask(ctx, a);
        var out = [], iv = 0;
        for (var k = 0; k < centers.length; k++) {
            var t = centers[k], c = 0;
            if (ivs) {
                while (iv < ivs.length && ivs[iv][1] < t) iv++;
                if (iv >= ivs.length) break;
                if (t < ivs[iv][0]) continue;
            }
            for (var u = t - hb; u < t - hb + w; u++) c += pres[u];
            if (c * 2 >= w) out.push(t);
        }
        return Int32Array.from(out);
    });
}

/** The hash of a bin: the same bin of the same animal in the same session, whatever else is there. */
function binHash(seriesHash, frame, seed) {
    return mix32(seriesHash, frame, seed) >>> 0;
}

/** The `keep`-th smallest of many values (keep ≥ 1), or Infinity when there are no more than that. */
function kthSmallest(parts, keep) {
    var total = parts.reduce(function (s, p) { return s + p.length; }, 0);
    if (total <= keep) return Infinity;
    var all = new Uint32Array(total), o = 0;
    parts.forEach(function (p) { all.set(p, o); o += p.length; });
    all.sort();
    return all[keep - 1];
}

// ----------------------------------------------------------------------------
// The run
// ----------------------------------------------------------------------------

/**
 * Find clusters in the poses of some sessions.
 *
 * @param {Object} opts
 * @param {Array<{session: Session, name: string, ctx?: Object, getCtx?: function(): ?Object}>} opts.sources
 *   behavior contexts (ui/behavior-detect.js `buildBehaviorContext`) or getters that build one, called
 *   when needed, so only one session's poses are held at a time
 * @param {string} [opts.split] a behavior id: cluster only the frames labelled present of it
 * @param {boolean} [opts.splitMutual] that behavior is mutual (both animals of a pair)
 * @param {number} [opts.maxBins]  default MAX_BINS
 * @param {number} [opts.neighbors] default NEIGHBORS
 * @param {number} [opts.minDist]  default MIN_DIST
 * @param {number[]} [opts.clusterShare] default CLUSTER_SHARE
 * @param {number} [opts.seed]     default 1
 * @param {function({stage: string, done: number, total: number, unit?: string})} [opts.onProgress]
 *   stages in order: 'read', 'bins', 'reduce', 'neighbors', 'layout', 'cluster', 'forests', 'labels'
 * @param {function(): Promise} [opts.pause] between slices of main-thread work
 * @param {function(): boolean} [opts.isCancelled]
 * @returns {Promise<{cancelled?: boolean, error?: string, clusters?: Object[], sources?: Object[], map?: Object,
 *                    report?: Object}>}
 */
export async function discoverBehaviors(opts) {
    var pause = opts.pause || noPause;
    var cancelled = opts.isCancelled || function () { return false; };
    var progress = opts.onProgress || function () {};
    var seed = opts.seed == null ? 1 : opts.seed;
    var maxBins = opts.maxBins > 0 ? Math.floor(opts.maxBins) : MAX_BINS;
    var timings = {};
    var clock = function () { return Date.now(); };
    var t0 = clock();
    var lap = function (name) { var t = clock(); timings[name] = (timings[name] || 0) + t - t0; t0 = t; };

    // ---- The sessions, and the features they all measure.
    var sources = (opts.sources || []).slice().sort(function (x, y) { return x.name < y.name ? -1 : x.name > y.name ? 1 : 0; });
    var ctxOf = function (s) { return s.ctx || (s.getCtx ? s.getCtx() : null); };
    var names = null, logFeatures = null, refMode = null;
    var leftOut = [];
    var plans = [];
    var parts = [];
    var usableTotal = 0;
    for (var si = 0; si < sources.length; si++) {
        if (cancelled()) return { cancelled: true };
        progress({ stage: 'read', done: si, total: sources.length, session: sources[si].name, unit: 'sessions' });
        var src = sources[si];
        var intervals = opts.split ? behaviorFramesByAnimal(src.session, opts.split, opts.splitMutual) : null;
        if (intervals && !intervals.size) continue;     // not labelled here: nothing to split
        var ctx = ctxOf(src);
        if (!ctx || !ctx.input.animals.length) { leftOut.push({ session: src.name, reason: 'no identified poses' }); continue; }
        if (!names) {
            names = discoveryFeatureNames(ctx.catalog);
            logFeatures = speedFeatures(ctx.catalog, names);
            refMode = ctx.input.mode;
        } else {
            var miss = missingFeatures(ctx.catalog, names, false);
            if (ctx.input.mode !== refMode) { leftOut.push({ session: src.name, reason: 'its poses are ' + ctx.input.mode.toUpperCase() }); continue; }
            if (miss.length) { leftOut.push({ session: src.name, reason: 'a different body (' + miss[0] + ')' }); continue; }
        }
        var w = binFrames(ctx.fps);
        var bins = usableBins(ctx, w, intervals);
        var hashes = bins.map(function (list, a) {
            var h = fnv(src.name + SEP + ctx.input.animals[a]), out = new Uint32Array(list.length);
            for (var i = 0; i < list.length; i++) out[i] = binHash(h, list[i], seed);
            usableTotal += list.length;
            return out;
        });
        hashes.forEach(function (h) { parts.push(h); });
        plans.push({ index: si, source: src, bins: bins, hashes: hashes, w: w, fps: ctx.fps, animals: ctx.input.animals.slice(), intervals: intervals });
        await pause();
    }
    progress({ stage: 'read', done: sources.length, total: sources.length, unit: 'sessions' });
    lap('read');
    if (!plans.length) {
        return { error: opts.split ? 'No labelled frames of this behavior have poses' : 'No session with identified poses: run Track All first', leftOut: leftOut };
    }
    if (usableTotal < MIN_BINS) {
        return { error: 'Too little to cluster: ' + usableTotal.toLocaleString('en-US') + ' bins of 100 ms have a pose, and it takes at least ' + MIN_BINS, leftOut: leftOut };
    }
    var cut = kthSmallest(parts, maxBins);
    parts = null;

    // ---- The sample's rows, in a fixed order: by session, then chunk, then row.
    var F = names.length;
    var M = 0;
    plans.forEach(function (p) {
        p.keep = p.bins.map(function (list, a) {
            var out = [], h = p.hashes[a];
            for (var i = 0; i < list.length; i++) if (h[i] <= cut) out.push(list[i]);
            M += out.length;
            return Int32Array.from(out);
        });
        p.hashes = null;
    });
    if (M > maxBins) M = maxBins;     // equal hashes at the cut: only the first ones
    var rows = new Float32Array(M * F).fill(NaN);
    var binSource = new Int16Array(M), binAnimal = new Int16Array(M), binFrame = new Int32Array(M);
    var nextRow = 0, doneBins = 0;
    progress({ stage: 'bins', done: 0, total: M, unit: 'bins' });
    for (var pi = 0; pi < plans.length; pi++) {
        if (cancelled()) return { cancelled: true };
        var plan = plans[pi];
        var c2 = ctxOf(plan.source);
        var input = c2.input, T = input.numFrames, N = input.nodeNames.length, D = input.dim;
        var chunks = planChunks(T, featurePadFrames(c2.spec), CHUNK_FRAMES);
        // Rows per chunk: [center frame, animal], and where each goes in the sample.
        var perChunk = chunks.map(function () { return []; });
        plan.keep.forEach(function (list, a) {
            for (var i = 0; i < list.length; i++) perChunk[Math.floor(list[i] / CHUNK_FRAMES)].push(list[i], a);
        });
        var todo = [];
        for (var ci = 0; ci < chunks.length && nextRow < M; ci++) {
            var pairs = perChunk[ci];
            if (!pairs.length) continue;
            var count = Math.min(pairs.length / 2, M - nextRow);
            var arr = Int32Array.from(pairs.slice(0, count * 2));
            for (var q = 0; q < count; q++) {
                binSource[nextRow + q] = plan.index; binAnimal[nextRow + q] = arr[2 * q + 1]; binFrame[nextRow + q] = arr[2 * q];
            }
            todo.push({ chunk: chunks[ci], rows: arr, at: nextRow });
            nextRow += count;
        }
        var skip = c2.broken ? c2.broken.masks : null;
        var ran = await runBehaviorChunks(todo.length, function (i) {
            return { kind: 'binrows', spec: c2.spec, chunk: todo[i].chunk, data: sliceChunkData(input.data, todo[i].chunk, N * D, skip),
                     request: { names: names, w: plan.w, rows: todo[i].rows } };
        }, function (i, res) {
            if (!res) return;
            rows.set(res.values, todo[i].at * F);
            doneBins += todo[i].rows.length / 2;
            progress({ stage: 'bins', done: doneBins, total: M, unit: 'bins' });
        }, { workload: T * input.animals.length, isCancelled: cancelled, pause: pause });
        if (ran.cancelled) return { cancelled: true };
    }
    if (nextRow < M) {
        rows = rows.slice(0, nextRow * F); binSource = binSource.slice(0, nextRow); binAnimal = binAnimal.slice(0, nextRow); binFrame = binFrame.slice(0, nextRow);
        M = nextRow;
    }
    lap('bins');
    if (cancelled()) return { cancelled: true };

    var steps = openStepRunner(cancelled, pause), steps2 = null;
    try {
        // ---- Standardize, PCA.
        progress({ stage: 'reduce', done: 0, total: 1 });
        var red = await steps.call({ type: 'reduce', rows: rows, M: M, F: F, opts: { logFeatures: logFeatures } });
        if (!red) return { cancelled: true };
        lap('reduce');
        var n = red.n;
        if (n < MIN_BINS || red.d < 1) {
            return { error: 'Too little to cluster: ' + n.toLocaleString('en-US') + ' bins of 100 ms have enough of their features', leftOut: leftOut };
        }

        // ---- Exact neighbours in PCA space, on the pool.
        var k = Math.min(opts.neighbors > 0 ? opts.neighbors : NEIGHBORS, Math.max(2, Math.floor(n / 3)));
        var indices = new Int32Array(n * k), distances = new Float32Array(n * k);
        var ranges = [];
        for (var r0 = 0; r0 < n; r0 += KNN_JOB_ROWS) ranges.push([r0, Math.min(n, r0 + KNN_JOB_ROWS)]);
        progress({ stage: 'neighbors', done: 0, total: n, unit: 'bins' });
        var doneK = 0;
        var knnRun = await runBehaviorChunks(ranges.length, function (i) {
            return { kind: 'knn', data: [], range: ranges[i], k: k };
        }, function (i, res) {
            if (!res) return;
            indices.set(res.indices, ranges[i][0] * k);
            distances.set(res.distances, ranges[i][0] * k);
            doneK += ranges[i][1] - ranges[i][0];
            progress({ stage: 'neighbors', done: doneK, total: n, unit: 'bins' });
        }, { workload: n * n, shared: { key: 'knn:' + (++_runs), value: { Y: red.Y, n: n, d: red.d } }, releaseShared: true,
             isCancelled: cancelled, pause: pause });
        if (knnRun.cancelled) return { cancelled: true };
        lap('neighbors');

        // ---- Two UMAP layouts, from the first two principal components and from a random view, side by side.
        var dims = 2;
        var init = new Float32Array(n * dims);
        if (red.d >= dims) { for (var r1 = 0; r1 < n; r1++) for (var d0 = 0; d0 < dims; d0++) init[r1 * dims + d0] = red.Y[r1 * red.d + d0]; }
        progress({ stage: 'layout', done: 0, total: 1, unit: 'epochs' });
        var knn = { indices: indices, distances: distances, n: n, k: k };
        var layoutOpts = function (s, start) { return { nComponents: dims, minDist: opts.minDist != null ? opts.minDist : MIN_DIST, seed: s, init: start }; };
        steps2 = openStepRunner(cancelled, pause);
        var layouts = await Promise.all([
            steps.call({ type: 'umap', knn: knn, opts: layoutOpts(seed, red.d >= dims ? init : null) }, null, function (d, t) { progress({ stage: 'layout', done: d, total: t, unit: 'epochs' }); }),
            steps2.call({ type: 'umap', knn: knn, opts: layoutOpts(seed + 1, randomView(red.Y, n, red.d, seed + 1)) }),
        ]);
        steps2.close();
        if (!layouts[0] || !layouts[1]) return { cancelled: true };
        var um = layouts[0];
        indices = null; distances = null; knn = null;
        lap('layout');

        // ---- HDBSCAN, keeping the clusters that repeat.
        progress({ stage: 'cluster', done: 0, total: 1 });
        var share = opts.clusterShare || CLUSTER_SHARE;
        var fpsOf = new Map(plans.map(function (p) { return [p.index, p.fps > 0 ? p.fps : 30]; }));
        var series = Int32Array.from(red.kept, function (r) { return binSource[r] * 4096 + binAnimal[r]; });
        var times = Float64Array.from(red.kept, function (r) { return binFrame[r] / fpsOf.get(binSource[r]); });
        var cl = await steps.call({ type: 'cluster', E: um.embedding, E2: layouts[1].embedding, n: n, d: dims,
            opts: { share: share, floor: clusterFloor(n, k), series: series, times: times } });
        if (!cl) return { cancelled: true };
        layouts = null;
        lap('cluster');
        steps.close();
        if (cl.count < 2) {
            return { error: (cl.count ? 'Only one cluster' : 'No cluster') + ' came back in a second layout and in several separate stretches of time, so the poses form one continuous blob here: nothing to tell apart', leftOut: leftOut,
                     report: { bins: n, usable: usableTotal, features: F, pcaDims: red.d, explained: red.explained, neighbors: k, dropped: cl.dropped,
                               minClusterSize: cl.minClusterSize, sizes: cl.sizes, counts: cl.counts } };
        }

        // ---- The map, and what stands out in each cluster.
        var kept = red.kept;
        var clusters = clusterProfiles(cl, red, names, n);
        clusters.forEach(function (c, i) { c.repeat = cl.repeat[i]; c.episodes = cl.episodes[i]; });

        // ---- A forest per cluster on the clustered bins.
        var forestRun = await fitClusterForests(rows, kept, cl, F, names, binSource, binAnimal, binFrame, plans, { seed: seed, pause: pause, isCancelled: cancelled, progress: progress });
        if (forestRun.cancelled) return { cancelled: true };
        lap('forests');
        rows = null;

        // ---- Every frame of every session.
        var results = [];
        for (var pj = 0; pj < plans.length; pj++) {
            if (cancelled()) return { cancelled: true };
            var pl = plans[pj];
            var c3 = ctxOf(pl.source);
            var out = await labelFrames(c3, pl, names, forestRun.forests, forestRun.rest, {
                pause: pause, isCancelled: cancelled,
                onProgress: function (d, t) { progress({ stage: 'labels', done: d, total: t, unit: 'frames', session: pl.source.name, step: pj + 1, steps: plans.length }); },
            });
            if (out.cancelled) return { cancelled: true };
            results.push({ session: pl.source.session, name: pl.source.name, animals: out.animals, frames: c3.input.numFrames, fps: c3.fps });
        }
        lap('labels');
        return {
            clusters: clusters,
            sources: results,
            leftOut: leftOut,
            map: {
                embedding: um.embedding, dims: dims, labels: cl.labels,
                session: Int16Array.from(kept, function (r) { return binSource[r]; }),
                animal: Int16Array.from(kept, function (r) { return binAnimal[r]; }),
                frame: Int32Array.from(kept, function (r) { return binFrame[r]; }),
                sessions: sources.map(function (s) { return s.name; }),
                sessionsOfIndex: sources.map(function (s) { return s.session; }),
                animals: plans.reduce(function (acc, p) { acc[p.index] = p.animals; return acc; }, {}),
            },
            report: {
                bins: n, usable: usableTotal, sampled: usableTotal > maxBins, features: red.features.length, pcaDims: red.d, explained: red.explained,
                neighbors: k, minDist: opts.minDist != null ? opts.minDist : MIN_DIST, minClusterSize: cl.minClusterSize, sizes: cl.sizes, counts: cl.counts,
                noise: cl.noise, dropped: cl.dropped,
                noiseShare: n ? cl.noise / n : 0, heldOut: forestRun.heldOut, heldOutAgreement: forestRun.agreement,
                trees: forestRun.trees, seed: seed, timings: timings,
            },
        };
    } finally {
        steps.close();
        if (steps2) steps2.close();
    }
}

var _runs = 0;

/** Clusters in order: bins, share of the clustered bins, and the features furthest from the mean. */
function clusterProfiles(cl, red, names, n) {
    var D = red.features.length;
    var sums = new Float64Array(cl.count * D), counts = new Int32Array(cl.count);
    for (var r = 0; r < n; r++) {
        var c = cl.labels[r];
        if (c < 0) continue;
        counts[c]++;
        var o = r * D, b = c * D;
        for (var d = 0; d < D; d++) sums[b + d] += red.Z[o + d];
    }
    var clustered = n - cl.noise, out = [];
    for (var k = 0; k < cl.count; k++) {
        var top = [];
        for (var d2 = 0; d2 < D; d2++) top.push([d2, counts[k] ? sums[k * D + d2] / counts[k] : 0]);
        top.sort(function (x, y) { return Math.abs(y[1]) - Math.abs(x[1]) || x[0] - y[0]; });
        out.push({
            state: k, bins: counts[k], share: clustered ? counts[k] / clustered : 0,
            top: top.slice(0, PROFILE_TOP).map(function (x) { return { name: names[red.features[x[0]]], z: x[1] }; }),
        });
    }
    return out;
}

/**
 * The forests that label every frame: one per cluster, and one for the bins in no
 * cluster when there are at least `MIN_CLUSTER_BINS`. Whole stretches of
 * `HOLDOUT_STRETCH_SECONDS` are held out by a seeded hash when there are enough bins.
 * @returns {Promise<{forests: Object[], rest: boolean, agreement: number, heldOut: number, trees: number, cancelled?: boolean}>}
 *   the last forest is the bins in no cluster when `rest`
 */
async function fitClusterForests(rows, kept, cl, F, names, binSource, binAnimal, binFrame, plans, o) {
    var n = kept.length, K = cl.count;
    var rest = cl.noise >= MIN_CLUSTER_BINS;
    var list = [];
    for (var r = 0; r < n; r++) if (rest || cl.labels[r] >= 0) list.push(r);
    var R = list.length;
    if (rest) K++;
    // Codes, feature-major, then bins.
    var codes = new Uint16Array(F * R), col = new Float32Array(R);
    var labels = new Int16Array(R);
    for (var f = 0; f < F; f++) {
        for (var i = 0; i < R; i++) col[i] = rows[kept[list[i]] * F + f];
        encodeColumn(col, codes, f * R, 1);
    }
    var hist = new Uint32Array(65536), lut = new Uint8Array(65536);
    var bins = new Uint8Array(F * R), nb = new Uint16Array(F), edges = new Array(F);
    for (var f2 = 0; f2 < F; f2++) {
        edges[f2] = binEdges(codes, f2 * R, R, hist);
        nb[f2] = edges[f2].length + 1;
        binColumn(codes, f2 * R, R, edges[f2], bins, f2 * R, lut);
        if ((f2 & 127) === 127) {
            o.progress({ stage: 'forests', done: 0, total: 1, unit: 'trees' });
            await o.pause();
            if (o.isCancelled()) return { cancelled: true };
        }
    }
    codes = null;
    // Train and held-out rows.
    var fpsOf = new Map(plans.map(function (p) { return [p.index, p.fps > 0 ? p.fps : 30]; }));
    var train = [], test = [];
    for (var i2 = 0; i2 < R; i2++) {
        var rr = kept[list[i2]];
        labels[i2] = cl.labels[list[i2]] >= 0 ? cl.labels[list[i2]] : K - 1;
        var stretch = Math.floor(binFrame[rr] / (HOLDOUT_STRETCH_SECONDS * fpsOf.get(binSource[rr])));
        var held = R >= HOLDOUT_MIN_ROWS && mix32(binSource[rr] * 64 + binAnimal[rr], stretch, o.seed) % 10 < HOLDOUT_TENTHS;
        (held ? test : train).push(i2);
    }
    var trainRows = Int32Array.from(train), testRows = Int32Array.from(test);
    var data = { bins: bins, labels: labels, R: R, F: F, nb: nb, train: trainRows };
    var params = { maxDepth: DEFAULT_FOREST_PARAMS.maxDepth, minLeaf: DEFAULT_FOREST_PARAMS.minLeaf };
    var jobs = [];
    for (var c = 0; c < K; c++) {
        for (var t0 = 0; t0 < TREES_PER_CLUSTER; t0 += TREES_PER_JOB) {
            var ts = [];
            for (var ti = t0; ti < Math.min(TREES_PER_CLUSTER, t0 + TREES_PER_JOB); ti++) ts.push(ti);
            jobs.push({ cluster: c, trees: ts });
        }
    }
    var treesOf = [];
    for (var c1 = 0; c1 < K; c1++) treesOf.push(new Array(TREES_PER_CLUSTER));
    var doneT = 0, total = K * TREES_PER_CLUSTER;
    o.progress({ stage: 'forests', done: 0, total: total, unit: 'trees' });
    var run = await runBehaviorChunks(jobs.length, function (i) {
        return { kind: 'clusterTrees', data: [], trees: { cluster: jobs[i].cluster, trees: jobs[i].trees, seed: o.seed, params: params } };
    }, function (i, out) {
        out.trees.forEach(function (tr, q) { treesOf[jobs[i].cluster][jobs[i].trees[q]] = tr; });
        doneT += jobs[i].trees.length;
        o.progress({ stage: 'forests', done: doneT, total: total, unit: 'trees' });
    }, { workload: R * total, shared: { key: 'clusters:' + (++_runs), value: data }, releaseShared: true,
         maxWorkers: TRAIN_WORKERS, isCancelled: o.isCancelled, pause: o.pause });
    if (run.cancelled || o.isCancelled()) return { cancelled: true };
    var agreement = heldOutAgreement(treesOf, { bins: bins, R: R, labels: labels }, testRows);
    // Raw forests, all reading the whole row of `names`.
    var index = new Map(names.map(function (nm, i3) { return [nm, i3]; }));
    var forests = treesOf.map(function (trees) {
        var fin = finalizeForest(trees, edges, names);
        var feat = fin.forest.feat, map = fin.features.map(function (nm) { return index.get(nm); });
        for (var q2 = 0; q2 < feat.length; q2++) if (feat[q2] !== 0xFFFF) feat[q2] = map[feat[q2]];
        return fin.forest;
    });
    return { forests: forests, rest: rest, agreement: agreement, heldOut: testRows.length, trees: total };
}

/**
 * The cluster of every frame of one session, as runs per animal: with `rest`, the
 * last forest stands for "in no cluster", so its frames have no state. For a split,
 * only on the behavior's frames.
 * @returns {Promise<{animals: Object<string, Int32Array>, cancelled?: boolean}>}
 */
async function labelFrames(ctx, plan, names, forests, rest, o) {
    var input = ctx.input, T = input.numFrames, N = input.nodeNames.length, D = input.dim;
    var seqs = input.animals.map(function () { return new Int16Array(T).fill(-1); });
    var chunks = planChunks(T, featurePadFrames(ctx.spec), CHUNK_FRAMES);
    var ivs = plan.intervals;
    var wanted = chunks;
    if (ivs) {
        var anyIn = function (ch) {
            var hit = false;
            ivs.forEach(function (list) { for (var i = 0; i < list.length && !hit; i++) if (list[i][0] < ch.end && list[i][1] >= ch.start) hit = true; });
            return hit;
        };
        wanted = chunks.filter(anyIn);
    }
    var done = 0, total = 0, skip = ctx.broken ? ctx.broken.masks : null;
    wanted.forEach(function (ch) { total += ch.end - ch.start; });
    var run = await runBehaviorChunks(wanted.length, function (i) {
        return { kind: 'clusterLabels', spec: ctx.spec, chunk: wanted[i], data: sliceChunkData(input.data, wanted[i], N * D, skip),
                 plan: { names: names, w: plan.w } };
    }, function (i, res) {
        if (!res) return;
        res.labels.forEach(function (arr, a) { seqs[a].set(arr, wanted[i].start); });
        done += wanted[i].end - wanted[i].start;
        o.onProgress(done, total);
    }, { workload: T * Math.max(1, input.animals.length) * 8, shared: { key: 'labels:' + (++_runs), value: forests }, releaseShared: true,
         isCancelled: o.isCancelled, pause: o.pause });
    if (run.cancelled) return { cancelled: true };
    var animals = {};
    input.animals.forEach(function (name, a) {
        var seq = seqs[a];
        if (rest) { var none = forests.length - 1; for (var t0 = 0; t0 < T; t0++) if (seq[t0] === none) seq[t0] = -1; }
        if (ivs) {
            var list = ivs.get(name) || [], at = 0;
            for (var t = 0; t < T; t++) {
                while (at < list.length && list[at][1] < t) at++;
                if (at >= list.length || t < list[at][0]) seq[t] = -1;
            }
        }
        var runs = runsFromSequence(seq);
        if (runs.length) animals[name] = runs;
    });
    return { animals: animals };
}

// ----------------------------------------------------------------------------
// Keeping a result
// ----------------------------------------------------------------------------

var _results = new Map();

/**
 * Keep a run's clusters: as a states layer in each session it read (replacing a
 * layer of that name there), and its map in memory.
 * @param {Object} result from `discoverBehaviors`
 * @param {string} name the layer's name
 * @returns {{layer: string, sessions: number, runs: number}}
 */
export function storeDiscovery(result, name) {
    var layer = cleanLayerName(name) || 'Discovery';
    var runs = 0, sessions = 0;
    (result.sources || []).forEach(function (s) {
        var r = setStateLayer(s.session, { name: layer, source: 'discovery', stateNames: {}, animals: s.animals });
        if (r.layer) { sessions++; runs += r.runs; }
    });
    _results.set(layer, { layer: layer, clusters: result.clusters, map: result.map, report: result.report, sessions: (result.sources || []).map(function (s) { return s.session; }) });
    return { layer: layer, sessions: sessions, runs: runs };
}

/**
 * What a run left in memory for a layer: `{clusters, map, report, sessions}`, or null
 * — after a reopen (only the layer is saved), or when the layer of that name is no
 * longer discovery's on this session.
 */
export function discoveryResult(session, name) {
    var r = _results.get(name);
    if (!r || r.sessions.indexOf(session) < 0) return null;
    var layer = stateLayer(session, name);
    return layer && layer.source === 'discovery' ? r : null;
}

/** An identity was renamed: the maps name the new animal from here on. */
export function renameDiscoveryAnimal(session, oldName, newName) {
    _results.forEach(function (r) {
        if (r.sessions.indexOf(session) < 0) return;
        var si = r.map.sessionsOfIndex.indexOf(session);
        var names = r.map.animals[si];
        if (!names) return;
        var at = names.indexOf(oldName);
        if (at >= 0) names[at] = newName;
    });
}

/** Forget every run's map (a new project or a load). */
export function resetDiscoveryResults() {
    _results.clear();
}
