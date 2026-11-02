// pose/behavior-discovery-core.js — the parts of behavior discovery that run on
// the behavior worker pool (pose/behavior-pool.js), one job at a time: features
// in 100 ms bins, exact nearest neighbours, one-vs-rest forests for the clusters
// and the cluster of every frame. pose/behavior-discovery.js runs the whole
// pipeline; pose/behavior-embedding.js has the steps that need the whole sample.
//
// ## Bins
//
// B-SOiD reads poses in 100 ms bins, averaging distances and summing
// displacements. The feature engine's base features are in lengths, angles and
// speeds per second, and a displacement summed over a bin is its mean speed
// times the bin's length, so every column is simply averaged over the bin. A
// bin is `binFrames(fps)` frames (the seconds are converted with the session's
// frame rate), and a value needs at least half the bin's frames.
//
// The bin of frame t is the window of w frames that holds t, with ⌊w / 2⌋ frames
// before it. Clustering reads the non-overlapping bins (frames ⌊w / 2⌋ + k·w);
// labelling reads the bin of EVERY frame, which is B-SOiD's frameshift — the same
// bins at every phase — and restores the full frame rate. A frame's window is
// centered on it, so a label is not late by half a bin.
//
// ## Chunks
//
// As everywhere in the behavior features, a job holds one padded chunk of poses
// and one animal's columns at a time; the frames × features matrix is never
// built. Only the rows asked for are returned, and values do not depend on how
// a session was cut into chunks (`binRowsChunk` is the same whole or chunked:
// tests/test-behavior-discovery.mjs).
//
// Pure, and imports only the feature engine and the forests, so the behavior
// worker runs exactly this code.
//
// Depends on: pose/behavior-features.js, pose/behavior-forest.js

import { computeFeatureChunk } from './behavior-features.js';
import { trainTree, treeScratch, treeSeed, scoreRows, scoreBinnedRows, presentFrames } from './behavior-forest.js';

/** A bin, in seconds (B-SOiD's). */
export var BIN_SECONDS = 0.1;

/** Base features that say how well an animal was tracked rather than what it did. */
var NOT_BEHAVIOR = ['missing'];

/** A bin's length in frames at this frame rate: at least one. */
export function binFrames(fps) {
    return Math.max(1, Math.round(BIN_SECONDS * (fps > 0 ? fps : 30)));
}

/** The bins of a session of T frames: their center frames, for bins that fit whole. */
export function binCenters(T, w) {
    var count = Math.floor(T / w), out = new Int32Array(count), h = w >> 1;
    for (var k = 0; k < count; k++) out[k] = h + k * w;
    return out;
}

/**
 * The features discovery clusters on: every base feature of the catalog (no
 * window statistics: the bins are the window), except those about the tracking.
 * Deterministic for a catalog.
 * @returns {string[]}
 */
export function discoveryFeatureNames(catalog) {
    return catalog.animal.filter(function (f) {
        return !f.stat && NOT_BEHAVIOR.indexOf(catalog.bases.get(f.base).kind) < 0;
    }).map(function (f) { return f.name; });
}

/**
 * Which of the features are speeds (a magnitude per second, heavy-tailed: they are
 * clustered on their log, pose/behavior-embedding.js).
 * @returns {Uint8Array} per name
 */
export function speedFeatures(catalog, names) {
    return Uint8Array.from(names, function (nm) {
        var f = catalog.byName.get(nm), base = f && catalog.bases.get(f.base);
        return base && (base.kind === 'cspeed' || base.kind === 'nspeed') ? 1 : 0;
    });
}

/** The finite mean of col[from..to] (clamped), or NaN with under half of `w` values. */
function windowMean(col, from, to, w) {
    var s = 0, c = 0;
    if (from < 0) from = 0;
    if (to >= col.length) to = col.length - 1;
    for (var i = from; i <= to; i++) {
        var v = col[i];
        if (v === v) { s += v; c++; }
    }
    return c * 2 >= w && c ? s / c : NaN;
}

/** The chunk a job reads its columns over: the frames plus a bin's reach on each side, within the padding. */
function innerChunk(chunk, w) {
    var hb = w >> 1, ha = w - 1 - hb;
    return { start: Math.max(chunk.padStart, chunk.start - hb), end: Math.min(chunk.padEnd, chunk.end + ha), padStart: chunk.padStart, padEnd: chunk.padEnd, data: chunk.data };
}

/**
 * Feature rows of some bins of one chunk (a worker job).
 * @param {Object} spec the feature spec
 * @param {{start, end, padStart, padEnd, data: Float32Array[]}} chunk
 * @param {{names: string[], w: number, rows: Int32Array}} req rows: [center frame, animal index] pairs,
 *   frames within [start, end)
 * @returns {{values: Float32Array}} row-major, rows × names, NaN = missing
 */
export function binRowsChunk(spec, chunk, req) {
    var F = req.names.length, n = req.rows.length / 2, w = req.w;
    var out = new Float32Array(n * F).fill(NaN);
    var inner = innerChunk(chunk, w);
    var hb = w >> 1, ha = w - 1 - hb, memo = {};
    var byAnimal = new Map();
    for (var i = 0; i < n; i++) {
        var a0 = req.rows[2 * i + 1];
        if (!byAnimal.has(a0)) byAnimal.set(a0, []);
        byAnimal.get(a0).push(i);
    }
    byAnimal.forEach(function (list, a) {
        var cols = computeFeatureChunk(spec, inner, { animal: req.names, only: [a] }, memo).animal[a] || {};
        for (var j = 0; j < F; j++) {
            var col = cols[req.names[j]];
            if (!col) continue;
            for (var q = 0; q < list.length; q++) {
                var r = list[q], t = req.rows[2 * r] - inner.start;
                out[r * F + j] = windowMean(col, t - hb, t + ha, w);
            }
        }
    });
    return { values: out };
}

/**
 * The cluster of every frame of one chunk (a worker job): each frame's bin is
 * scored by every cluster's forest, and the highest score wins. A frame whose
 * bin has a pose on fewer than half its frames has no cluster (−1). Forests read
 * the row of all `names` (`feat` indexes `names`, not a subset of it).
 * @param {Object} spec
 * @param {{start, end, padStart, padEnd, data: Float32Array[]}} chunk
 * @param {{names: string[], w: number}} plan
 * @param {Array<{feat, val, kid, miss, roots}>} forests one per cluster
 * @returns {{labels: Int16Array[]}} per animal, frames of [start, end)
 */
export function clusterLabelsChunk(spec, chunk, plan, forests) {
    var names = plan.names, F = names.length, w = plan.w;
    var n = chunk.end - chunk.start;
    var inner = innerChunk(chunk, w);
    var hb = w >> 1, ha = w - 1 - hb, off = chunk.start - inner.start;
    var X = new Float32Array(n * F), tmp = new Float32Array(n), best = new Float32Array(n), memo = {};
    var labels = [];
    for (var a = 0; a < chunk.data.length; a++) {
        var out = new Int16Array(n).fill(-1);
        labels.push(out);
        var present = presentFrames(spec, chunk.data[a], inner);
        var any = false;
        var usable = new Uint8Array(n);
        for (var t = 0; t < n; t++) {
            var c = 0;
            for (var u = Math.max(0, t + off - hb); u <= Math.min(present.length - 1, t + off + ha); u++) c += present[u];
            if (c * 2 >= w && c) { usable[t] = 1; any = true; }
        }
        if (!any) continue;
        var cols = computeFeatureChunk(spec, inner, { animal: names, only: [a] }, memo).animal[a] || {};
        for (var j = 0; j < F; j++) {
            var col = cols[names[j]];
            for (var t2 = 0; t2 < n; t2++) {
                X[t2 * F + j] = col ? windowMean(col, t2 + off - hb, t2 + off + ha, w) : NaN;
            }
        }
        best.fill(-Infinity);
        for (var k = 0; k < forests.length; k++) {
            scoreRows(forests[k], X, F, n, tmp);
            for (var t3 = 0; t3 < n; t3++) if (tmp[t3] > best[t3]) { best[t3] = tmp[t3]; out[t3] = k; }
        }
        for (var t4 = 0; t4 < n; t4++) if (!usable[t4]) out[t4] = -1;
    }
    return { labels: labels };
}

/**
 * Exact nearest neighbours of some rows (a worker job): `k` neighbours per row,
 * nearest first, the row itself first of all (UMAP's convention). Every row is
 * compared with every row, so it costs n × rows × dims, spread over the pool; ties
 * keep the earlier row, so the answer does not depend on how the rows were split.
 * @param {{Y: Float32Array, n: number, d: number}} shared
 * @param {number[]} range [from, to) rows
 * @returns {{indices: Int32Array, distances: Float32Array}} (to − from) × k
 */
export function knnRows(shared, range, k) {
    var Y = shared.Y, n = shared.n, d = shared.d;
    var from = range[0], to = range[1], count = to - from;
    var indices = new Int32Array(count * k), distances = new Float32Array(count * k);
    var cap = k - 1;
    var hd = new Float64Array(Math.max(1, cap)), hi = new Int32Array(Math.max(1, cap));
    for (var i = from; i < to; i++) {
        var o = i * d, size = 0, top = Infinity;
        for (var j = 0; j < n; j++) {
            if (j === i) continue;
            var q = j * d, s = 0;
            for (var m = 0; m < d && s < top; m++) { var u = Y[o + m] - Y[q + m]; s += u * u; }
            if (s >= top) continue;
            if (size < cap) {
                var c = size++;
                while (c > 0) {
                    var par = (c - 1) >> 1;
                    if (hd[par] >= s) break;
                    hd[c] = hd[par]; hi[c] = hi[par]; c = par;
                }
                hd[c] = s; hi[c] = j;
                if (size === cap) top = hd[0];
            } else if (cap > 0) {
                var c2 = 0;
                for (;;) {
                    var kid = 2 * c2 + 1;
                    if (kid >= cap) break;
                    if (kid + 1 < cap && hd[kid + 1] > hd[kid]) kid++;
                    if (hd[kid] <= s) break;
                    hd[c2] = hd[kid]; hi[c2] = hi[kid]; c2 = kid;
                }
                hd[c2] = s; hi[c2] = j;
                top = hd[0];
            }
        }
        var base = (i - from) * k;
        indices[base] = i; distances[base] = 0;
        // Pop the heap: farthest last.
        for (var p = size; p > 0; p--) {
            indices[base + p] = hi[0]; distances[base + p] = Math.sqrt(hd[0]);
            var last = p - 1;
            var sv = hd[last], iv = hi[last], c3 = 0;
            for (;;) {
                var kid2 = 2 * c3 + 1;
                if (kid2 >= last) break;
                if (kid2 + 1 < last && hd[kid2 + 1] > hd[kid2]) kid2++;
                if (hd[kid2] <= sv) break;
                hd[c3] = hd[kid2]; hi[c3] = hi[kid2]; c3 = kid2;
            }
            hd[c3] = sv; hi[c3] = iv;
        }
    }
    return { indices: indices, distances: distances };
}

/**
 * A random view of the data: its rows on two random orthonormal directions, seeded
 * (the second layout's start). Points close in the data start close, as in the first
 * two components, so nothing is torn, but the arrangement is another one.
 * @param {ArrayLike<number>} Y n × d
 * @returns {Float32Array} n × 2
 */
export function randomView(Y, n, d, seed) {
    var a = (seed >>> 0) || 1;
    var uniform = function () {
        a = (a + 0x6D2B79F5) | 0;
        var t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    var normal = function () { var u = 0; while (!u) u = uniform(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * uniform()); };
    var e1 = new Float64Array(d), e2 = new Float64Array(d), s1 = 0, s2 = 0, dot = 0;
    for (var j = 0; j < d; j++) { e1[j] = normal(); e2[j] = normal(); s1 += e1[j] * e1[j]; }
    s1 = Math.sqrt(s1) || 1;
    for (var j2 = 0; j2 < d; j2++) { e1[j2] /= s1; dot += e1[j2] * e2[j2]; }
    for (var j3 = 0; j3 < d; j3++) { e2[j3] -= dot * e1[j3]; s2 += e2[j3] * e2[j3]; }
    s2 = Math.sqrt(s2) || 1;
    for (var j4 = 0; j4 < d; j4++) e2[j4] /= s2;
    var out = new Float32Array(n * 2);
    for (var r = 0; r < n; r++) {
        var x = 0, y = 0, o = r * d;
        for (var j5 = 0; j5 < d; j5++) { x += Y[o + j5] * e1[j5]; y += Y[o + j5] * e2[j5]; }
        out[2 * r] = x; out[2 * r + 1] = y;
    }
    return out;
}

/**
 * Grow some trees of one cluster's forest, "this cluster" against the rest (a
 * worker job): the classifiers' tree (pose/behavior-forest.js, with its
 * class-balanced bootstrap), seeded from (seed, cluster, tree index) alone.
 * @param {{bins: Uint8Array, labels: Int16Array, R: number, F: number, nb: Uint16Array, train: Int32Array}} shared
 * @param {{cluster: number, trees: number[], seed: number, params: Object}} job
 * @returns {{trees: Object[]}} in job order, bin form
 */
export function trainClusterTrees(shared, job) {
    var R = shared.R, labels = shared.labels, y = new Uint8Array(R);
    for (var r = 0; r < R; r++) y[r] = labels[r] === job.cluster ? 1 : 0;
    var data = { bins: shared.bins, y: y, R: R, F: shared.F, nb: shared.nb };
    var s = treeScratch(R, shared.F);
    var out = job.trees.map(function (ti) {
        var tr = trainTree(data, shared.train, treeSeed(job.seed, job.cluster, ti), job.params, s);
        delete tr.imp;
        return tr;
    });
    return { trees: out };
}

/**
 * The share of rows whose best-scoring forest is their own cluster's, for rows
 * no forest trained on.
 * @param {Object[][]} forestsTrees per cluster, trees in bin form
 * @param {{bins: Uint8Array, R: number, labels: Int16Array}} data
 * @param {Int32Array} rows
 * @returns {number} NaN with no rows
 */
export function heldOutAgreement(forestsTrees, data, rows) {
    if (!rows.length) return NaN;
    var best = new Float32Array(rows.length).fill(-Infinity), pick = new Int16Array(rows.length).fill(-1);
    forestsTrees.forEach(function (trees, k) {
        var sc = scoreBinnedRows(trees, data, rows);
        for (var i = 0; i < rows.length; i++) if (sc[i] > best[i]) { best[i] = sc[i]; pick[i] = k; }
    });
    var hit = 0;
    for (var i2 = 0; i2 < rows.length; i2++) if (pick[i2] === data.labels[rows[i2]]) hit++;
    return hit / rows.length;
}
