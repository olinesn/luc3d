// pose/behavior-embedding.js — the steps of behavior discovery that read the
// whole sample at once: standardizing the bins, PCA, and HDBSCAN on the layout.
// They run in the discovery worker (pose/behavior-discovery-worker.js), or inline
// when there are no workers.
//
// ## Standardized, then PCA
//
// Each feature becomes a z-score over the sampled bins. Speeds go through a log
// first (`SPEED_LOG_OFFSET`): they are heavy-tailed, so in raw z-scores everything
// but a sprint sits in a fraction of a standard deviation, and on five mice the
// clusters then changed with UMAP's seed (agreement between seeds 0.62, against
// 0.97 with the log). B-SOiD does not do this; forests do not care (a log keeps
// the order). A tracking error that survives the broken-tracking check is hundreds
// of standard deviations out, and a handful of those would take over the principal
// components, so z-scores are cut at ±`Z_LIMIT`. A missing value counts as the
// feature's mean (0), and a bin missing more than `MAX_MISSING_SHARE` of its
// features sits out (the forests label it later: they take missing values as they
// are). A feature that is constant or mostly missing is left out.
//
// PCA keeps the fewest components reaching `VARIANCE_KEPT` of the variance
// (B-SOiD's 70%): `symmetricEigen` on the covariance for up to `PCA_EXACT_MAX`
// features, `fitPCA`'s top `PCA_TOP` directions past that. Components keep their
// scale (no whitening), so distances between bins stay those of the z-scores.
//
// ## Clusters that repeat
//
// UMAP with min_dist 0 followed by HDBSCAN finds clusters in anything: Gaussian
// noise with no structure at all gave 3 to 91 of them, depending on the sizes tried.
// So the count alone says nothing. What noise's clusters do not do is come back in a
// second layout of the same bins. That layout starts from another view of the data
// (`randomView`, a random linear projection) with another seed: not from random
// positions, because those tear real clusters (on five mice, half of rearing ended
// up hanging off the big blob, from one sample in seven). At each smallest-cluster
// size tried, a cluster counts only where the second layout's cluster tree holds a
// group sharing at least `MIN_REPEAT` of their bins (Jaccard, `bestTreeOverlaps`): the
// tree, not what its selection kept, because selection can settle on another level
// while the group is right there. Measured: 0.89–0.99 on five real mice; at most 0.69
// on noise, but for a whole blob matching itself, which is one cluster and says so.
// A cluster must also happen more than once. Bins of one long stretch are near copies
// of each other, so an animal resting for ten seconds makes a tight clump that repeats
// perfectly: on a two-mouse recording, 23 clusters, many of them one rest each. So a
// cluster counts only with bins from at least `MIN_EPISODES` separate stretches of time
// (`clusterEpisodes`: of one animal, more than `EPISODE_GAP_SECONDS` apart; at 2 s, one
// rest that a few bins interrupt still counted as 8), or with at least `BIG_CLUSTER_SHARE`
// of all the bins: a cluster an animal is in nearly all the time has few stretches because
// it never leaves, not because it happened once.
// The size kept is the one with the most clusters that pass both, among the sizes where
// they cover nearly as many bins as at any size — B-SOiD keeps the most clusters, which
// on noise is just the smallest size. A cluster that fails either becomes noise. Whatever
// is left may still be a posture rather than a behavior, but it is not an accident of
// the layout or of one moment.
//
// Depends on: pose/id-switch-check.js (`symmetricEigen`, `fitPCA`), pose/hdbscan.js

import { symmetricEigen, fitPCA } from './id-switch-check.js';
import { hdbscanLinkage, condenseTree, selectClusters, bestTreeOverlaps } from './hdbscan.js';

/** B-SOiD's share of the variance to keep. */
export var VARIANCE_KEPT = 0.7;
/** z-scores are cut here. */
export var Z_LIMIT = 6;
/** A bin missing more than this share of its features sits out of the clustering. */
export var MAX_MISSING_SHARE = 0.25;
/** Up to this many features, the whole covariance's eigenvalues. */
export var PCA_EXACT_MAX = 512;
/** Past that, this many directions. */
export var PCA_TOP = 64;
/** Sizes tried between the smallest and largest "smallest cluster" (B-SOiD's). */
var SIZES_TRIED = 25;
/** A speed's log: log(v + this), in BL/s (about the jitter of a still animal: 0.012–0.056 BL/s on real recordings). */
export var SPEED_LOG_OFFSET = 0.05;
/** A cluster counts when the other layout's tree holds a group sharing this share of their bins (Jaccard). */
export var MIN_REPEAT = 0.7;
/** A cluster needs bins from this many separate stretches of time… */
export var MIN_EPISODES = 5;
/** …of one animal, at least this far apart. */
export var EPISODE_GAP_SECONDS = 10;
/** …unless it holds this share of all the bins. */
export var BIG_CLUSTER_SHARE = 0.25;
/** Sizes whose repeating clusters cover this share of what the best size covers are in the running. */
var COVER_NEARLY = 0.9;

/**
 * Standardize the sample and reduce it.
 * @param {Float32Array} rows M × F, row-major, NaN = missing
 * @param {number} M
 * @param {number} F
 * @param {{variance?: number, zLimit?: number, maxMissing?: number, logFeatures?: ArrayLike<number>}} [opts]
 *   logFeatures: per feature, 1 to take its log first (speeds)
 * @returns {{Y: Float32Array, n: number, d: number, kept: Int32Array, features: Int32Array, mean: Float32Array, std: Float32Array,
 *            explained: number, spectrum: Float64Array, Z: Float32Array}}
 *   Y n × d; kept: the rows that stayed (indices into `rows`); features: the features that stayed, with their mean and std;
 *   Z: the kept rows' z-scores, n × features.length
 */
export function standardizeAndReduce(rows, M, F, opts) {
    opts = opts || {};
    var limit = opts.zLimit != null ? opts.zLimit : Z_LIMIT;
    var maxMissing = opts.maxMissing != null ? opts.maxMissing : MAX_MISSING_SHARE;
    var variance = opts.variance != null ? opts.variance : VARIANCE_KEPT;
    // Which rows have enough of their features.
    var keepRow = new Uint8Array(M), n = 0;
    var perFeature = new Int32Array(F);
    var logged = opts.logFeatures || null;
    var read = function (v, j) { return logged && logged[j] ? Math.log(v + SPEED_LOG_OFFSET) : v; };
    for (var r = 0; r < M; r++) {
        var o = r * F, miss = 0;
        for (var j = 0; j < F; j++) if (rows[o + j] !== rows[o + j]) miss++;
        if (miss <= maxMissing * F) { keepRow[r] = 1; n++; }
    }
    var kept = new Int32Array(n);
    for (var r1 = 0, q0 = 0; r1 < M; r1++) if (keepRow[r1]) kept[q0++] = r1;
    // Each feature's mean and spread over those rows.
    var sum = new Float64Array(F), sum2 = new Float64Array(F);
    for (var q = 0; q < n; q++) {
        var o1 = kept[q] * F;
        for (var j1 = 0; j1 < F; j1++) {
            var v = rows[o1 + j1];
            if (v === v) { v = read(v, j1); sum[j1] += v; sum2[j1] += v * v; perFeature[j1]++; }
        }
    }
    var featList = [], meanAll = new Float32Array(F), stdAll = new Float32Array(F);
    for (var j2 = 0; j2 < F; j2++) {
        var c = perFeature[j2];
        if (!c || c * 2 < n) continue;
        var m = sum[j2] / c, s2 = sum2[j2] / c - m * m;
        var s = s2 > 0 ? Math.sqrt(s2) : 0;
        if (!(s > 1e-6 * Math.max(1e-6, Math.abs(m)))) continue;
        meanAll[j2] = m; stdAll[j2] = s;
        featList.push(j2);
    }
    var D = featList.length;
    var features = Int32Array.from(featList);
    var Z = new Float32Array(n * D);
    for (var q1 = 0; q1 < n; q1++) {
        var o2 = kept[q1] * F, o3 = q1 * D;
        for (var d = 0; d < D; d++) {
            var f = featList[d], x = rows[o2 + f];
            if (x !== x) { Z[o3 + d] = 0; continue; }
            var z = (read(x, f) - meanAll[f]) / stdAll[f];
            Z[o3 + d] = z > limit ? limit : z < -limit ? -limit : z;
        }
    }
    var done = function (Y, dims, explained, spectrum) {
        return {
            Y: Y, n: n, d: dims, kept: kept, features: features, explained: explained, spectrum: spectrum, Z: Z,
            mean: Float32Array.from(featList, function (x2) { return meanAll[x2]; }), std: Float32Array.from(featList, function (x3) { return stdAll[x3]; }),
        };
    };
    if (!n || !D) return done(new Float32Array(0), 0, NaN, new Float64Array(0));
    // Column means again: cutting the z-scores moved them a little.
    var center = new Float64Array(D);
    for (var q2 = 0; q2 < n; q2++) for (var d1 = 0; d1 < D; d1++) center[d1] += Z[q2 * D + d1];
    for (var d2 = 0; d2 < D; d2++) center[d2] /= n;
    var dirs, values, total = 0, k;
    if (D <= PCA_EXACT_MAX) {
        var C = new Float64Array(D * D), row = new Float64Array(D);
        for (var q3 = 0; q3 < n; q3++) {
            var o4 = q3 * D;
            for (var a = 0; a < D; a++) row[a] = Z[o4 + a] - center[a];
            for (var a2 = 0; a2 < D; a2++) {
                var va = row[a2], ca = a2 * D;
                if (!va) continue;
                for (var b = a2; b < D; b++) C[ca + b] += va * row[b];
            }
        }
        for (var a3 = 0; a3 < D; a3++) { for (var b2 = a3; b2 < D; b2++) { C[a3 * D + b2] /= Math.max(1, n - 1); C[b2 * D + a3] = C[a3 * D + b2]; } }
        var eig = symmetricEigen(C, D);
        var order = Array.from({ length: D }, function (_, i) { return i; }).sort(function (p, q4) { return eig.values[q4] - eig.values[p] || p - q4; });
        values = Float64Array.from(order, function (i) { return Math.max(0, eig.values[i]); });
        for (var i0 = 0; i0 < D; i0++) total += values[i0];
        k = keepCount(values, total, variance);
        dirs = new Float64Array(D * k);
        for (var c1 = 0; c1 < k; c1++) { var src = order[c1]; for (var d3 = 0; d3 < D; d3++) dirs[d3 * k + c1] = eig.vectors[d3 * D + src]; }
    } else {
        for (var d4 = 0; d4 < D; d4++) {
            var t = 0;
            for (var q5 = 0; q5 < n; q5++) { var u = Z[q5 * D + d4] - center[d4]; t += u * u; }
            total += t / Math.max(1, n - 1);
        }
        var top = Math.min(PCA_TOP, D, Math.max(1, n - 1));
        var pca = fitPCA(Z, n, D, top);
        // Its directions come back through project(): read them off the unit vectors.
        var basis = new Float64Array(D * pca.dims), unit = new Float64Array(D), origin = pca.project(unit);
        for (var d5 = 0; d5 < D; d5++) {
            unit[d5] = 1;
            var col = pca.project(unit);
            for (var c2 = 0; c2 < pca.dims; c2++) basis[d5 * pca.dims + c2] = col[c2] - origin[c2];
            unit[d5] = 0;
        }
        var spread = new Float64Array(pca.dims);
        for (var q6 = 0; q6 < n; q6++) {
            for (var c3 = 0; c3 < pca.dims; c3++) {
                var p2 = 0;
                for (var d6 = 0; d6 < D; d6++) p2 += (Z[q6 * D + d6] - center[d6]) * basis[d6 * pca.dims + c3];
                spread[c3] += p2 * p2;
            }
        }
        var order2 = Array.from({ length: pca.dims }, function (_, i) { return i; }).sort(function (p, q7) { return spread[q7] - spread[p] || p - q7; });
        values = Float64Array.from(order2, function (i) { return spread[i] / Math.max(1, n - 1); });
        k = keepCount(values, total, variance);
        dirs = new Float64Array(D * k);
        for (var c4 = 0; c4 < k; c4++) { var s3 = order2[c4]; for (var d7 = 0; d7 < D; d7++) dirs[d7 * k + c4] = basis[d7 * pca.dims + s3]; }
    }
    var Y = new Float32Array(n * k);
    for (var q8 = 0; q8 < n; q8++) {
        var o5 = q8 * D, o6 = q8 * k;
        for (var c5 = 0; c5 < k; c5++) {
            var p3 = 0;
            for (var d8 = 0; d8 < D; d8++) p3 += (Z[o5 + d8] - center[d8]) * dirs[d8 * k + c5];
            Y[o6 + c5] = p3;
        }
    }
    var kept2 = 0;
    for (var c6 = 0; c6 < k; c6++) kept2 += values[c6];
    return done(Y, k, total > 0 ? kept2 / total : NaN, values);
}

/** The fewest components reaching `variance` of `total`: at least 2 when there are 2, at most what is there. */
function keepCount(values, total, variance) {
    var acc = 0, k = 0;
    while (k < values.length && (acc < variance * total || k < 2)) { acc += values[k]; k++; }
    return Math.max(Math.min(1, values.length), k);
}

/**
 * Sizes of "smallest cluster" to try, in points: from `share[0]` to `share[1]` of the
 * points, evenly; from `floor` to twice that where the shares fall below it.
 * @returns {number[]} ascending, each once
 */
export function clusterSizes(n, share, floor) {
    var low = Math.max(floor, share[0] * n), high = Math.max(2 * floor, share[1] * n), out = [];
    for (var i = 0; i < SIZES_TRIED; i++) {
        var s = Math.round(low + (high - low) * i / Math.max(1, SIZES_TRIED - 1));
        if (out.indexOf(s) < 0) out.push(s);
    }
    return out;
}

/**
 * How many separate stretches of time each cluster's bins come from.
 * @param {ArrayLike<number>} labels n, −1 = none
 * @param {ArrayLike<number>} series n: which recording of which animal a bin is from
 * @param {ArrayLike<number>} times n: its time, in seconds
 * @param {Int32Array} order the bins by series, then time
 * @returns {Int32Array} per cluster
 */
export function clusterEpisodes(labels, count, series, times, order, gap) {
    var out = new Int32Array(count), lastS = new Int32Array(count).fill(-1), lastT = new Float64Array(count);
    for (var q = 0; q < order.length; q++) {
        var r = order[q], k = labels[r];
        if (k < 0) continue;
        if (series[r] !== lastS[k] || times[r] - lastT[k] > gap) out[k]++;
        lastS[k] = series[r]; lastT[k] = times[r];
    }
    return out;
}

/**
 * HDBSCAN on two layouts of the same bins: of the smallest-cluster sizes tried, the
 * one with the most clusters of the first layout that the second layout's tree also
 * holds (the largest such size). A cluster that does not repeat becomes noise. Clusters
 * are numbered by how many points they hold, largest first; noise is −1.
 * @param {ArrayLike<number>} E n × d, the layout kept
 * @param {ArrayLike<number>} E2 n × d, the second opinion
 * @param {{share: number[], floor: number, minSamples?: number, minRepeat?: number, minEpisodes?: number,
 *          series?: ArrayLike<number>, times?: ArrayLike<number>}} opts
 *   series and times: each bin's recording of an animal and its time (s), for the episodes test (skipped without)
 * @returns {{labels: Int16Array, count: number, noise: number, minClusterSize: number, repeat: Float64Array,
 *            episodes: Int32Array, dropped: number, sizes: number[], counts: number[]}}
 *   repeat: per cluster, its best overlap with the second layout's tree; episodes: its separate stretches of time;
 *   dropped: clusters that failed either test
 */
export function clusterLayouts(E, E2, n, d, opts) {
    var minRepeat = opts.minRepeat != null ? opts.minRepeat : MIN_REPEAT;
    var minEpisodes = opts.minEpisodes != null ? opts.minEpisodes : MIN_EPISODES;
    var sizes = clusterSizes(n, opts.share, opts.floor);
    var series = opts.series || null, times = opts.times || null, order = null;
    if (series && times) {
        order = new Int32Array(n);
        for (var i1 = 0; i1 < n; i1++) order[i1] = i1;
        order.sort(function (x, y) { return series[x] - series[y] || times[x] - times[y] || x - y; });
    }
    var la = hdbscanLinkage(E, n, d, { minSamples: opts.minSamples || 1 });
    var lb = hdbscanLinkage(E2, n, d, { minSamples: opts.minSamples || 1 });
    var tried = sizes.map(function (s) {
        var a = selectClusters(condenseTree(la, s));
        var overlaps = bestTreeOverlaps(a.labels, a.count, condenseTree(lb, s));
        var episodes = order ? clusterEpisodes(a.labels, a.count, series, times, order, EPISODE_GAP_SECONDS) : null;
        var good = 0, size = new Int32Array(a.count), covered = 0, keep = new Uint8Array(a.count);
        for (var i0 = 0; i0 < n; i0++) if (a.labels[i0] >= 0) size[a.labels[i0]]++;
        for (var k = 0; k < a.count; k++) {
            if (overlaps[k] >= minRepeat && (!episodes || episodes[k] >= minEpisodes || size[k] >= BIG_CLUSTER_SHARE * n)) { keep[k] = 1; good++; covered += size[k]; }
        }
        return { size: s, a: a, overlaps: overlaps, episodes: episodes, keep: keep, good: good, covered: covered };
    });
    var mostCovered = tried.reduce(function (m, t) { return Math.max(m, t.covered); }, 0);
    var best = tried[0], counts = tried.map(function (t) { return t.good; });
    tried.forEach(function (t) {
        if (t.covered >= COVER_NEARLY * mostCovered && (best.covered < COVER_NEARLY * mostCovered || t.good >= best.good)) best = t;
    });
    var a = best.a;
    var tally = new Int32Array(a.count);
    for (var i = 0; i < n; i++) if (a.labels[i] >= 0) tally[a.labels[i]]++;
    var kept = [];
    for (var k2 = 0; k2 < a.count; k2++) if (best.keep[k2]) kept.push(k2);
    kept.sort(function (x, y) { return tally[y] - tally[x] || x - y; });
    var remap = new Int16Array(a.count).fill(-1);
    kept.forEach(function (old, k) { remap[old] = k; });
    var labels = new Int16Array(n), noise = 0;
    for (var i2 = 0; i2 < n; i2++) {
        labels[i2] = a.labels[i2] >= 0 ? remap[a.labels[i2]] : -1;
        if (labels[i2] < 0) noise++;
    }
    return {
        labels: labels, count: kept.length, noise: noise, minClusterSize: best.size, sizes: sizes, counts: counts,
        repeat: Float64Array.from(kept, function (old) { return best.overlaps[old]; }),
        episodes: Int32Array.from(kept, function (old) { return best.episodes ? best.episodes[old] : 0; }), dropped: a.count - kept.length,
    };
}
