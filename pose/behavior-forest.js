// pose/behavior-forest.js — random forests for behavior classifiers: one
// binary forest per behavior ("rearing" vs "not rearing"), trained on the
// frames a user labelled, scoring every frame with the share of trees that
// say "present".
//
// Pure and import-free but for the feature engine, so the behavior worker
// (pose/behavior-worker.js) runs exactly this code: it trains trees, extracts
// training rows and scores frames. The orchestration (which frames are
// samples, cross-validation, the model registry) is pose/behavior-classifier.js.
//
// ## Codes and bins
//
// A training row holds one value per feature. Values are stored as 16-bit
// CODES that keep their order (`codeOf`: the top half of a float32's bits,
// made sortable — bfloat16 resolution, about 0.4%), 0 for a missing value.
// Each feature is then cut into at most 255 quantile BINS of its codes
// (`binEdges`, `binColumn`; a value with few distinct codes gets one bin per
// code), plus bin 0 for missing — the Uint8 histogram CART of LightGBM and
// scikit-learn's HistGradientBoosting. A split "bin ≤ b" is the same test as
// "value ≤ `codeUpperValue`(the bin's upper edge)" on the raw float32 value,
// so a trained tree is turned into raw thresholds (`finalizeForest`) and
// prediction never bins: training and prediction agree exactly.
//
// ## Trees
//
// CART on the Gini impurity, as scikit-learn's RandomForestClassifier with
// class_weight='balanced_subsample' (A-SOiD's setting): each tree draws a
// bootstrap sample of the training rows, and weights each class so the two
// weigh the same in that sample; each split tries √F random features (more
// when some are constant in the node), and a missing value goes to whichever
// side helps more (learned per split; with nothing missing in the node, to
// the heavier side). A leaf holds the class-weighted share of "present", so
// 0.5 is the balanced decision whatever the labels' proportions. Depth ≤
// `maxDepth`, every leaf ≥ `minLeaf` distinct rows. Feature importance is the
// mean decrease in impurity, normalized per tree.
//
// Every tree is seeded from (forest seed, fold, tree index) alone, so a forest
// is the same whichever worker trained which tree, and the same labels and
// poses retrain the same forest, bit for bit (`forestHash`).
//
// ## A forest
//
// Raw form (what is kept, saved and run): all trees in flat typed arrays —
// `feat` (Uint16, LEAF for a leaf), `val` (Float32: the threshold, or a
// leaf's share), `kid` (Int32: the left child; the right one is next),
// `miss` (Uint8: 1 = a missing value goes left), `roots` (Int32, one per
// tree). `features` names each feature index, as the feature engine names
// them (an actor's own feature, "target · …" for the target's, relational
// ones by name).
//
// Depends on: pose/behavior-features.js

import { computeFeatureChunk, resolveFeatureName, TARGET_PREFIX } from './behavior-features.js';

/** Bumped when a forest's meaning changes (codes, bins, the tree format). */
export var FOREST_VERSION = 1;

/** The defaults: 100 trees, depth 16, leaves of at least 3 rows. */
export var DEFAULT_FOREST_PARAMS = { trees: 100, maxDepth: 16, minLeaf: 3, seed: 1 };

/** `feat` of a leaf. */
export var LEAF = 0xFFFF;
/** The code (and the bin) of a missing value. */
export var MISSING_CODE = 0;
/** Value bins per feature (1–255; bin 0 is missing). */
export var MAX_VALUE_BINS = 255;

// ----------------------------------------------------------------------------
// Codes
// ----------------------------------------------------------------------------

var _f32 = new Float32Array(1);
var _u32 = new Uint32Array(_f32.buffer);
var NEG_TINY = -1.401298464324817e-45;    // the negative float32 nearest 0

/** A float32 bit pattern's code: sortable, 1–65535; 0 for NaN; −0 is +0. */
function codeOfBits(b) {
    if ((b & 0x7fffffff) > 0x7f800000) return 0;          // NaN
    if (b === 0x80000000) b = 0;                           // −0 → +0
    return (b & 0x80000000) ? ((~b) >>> 16) & 0xffff : ((b >>> 16) | 0x8000);
}

/** The code of a value: order-keeping, 0 = missing. */
export function codeOf(v) {
    if (v !== v) return 0;
    _f32[0] = v;
    return codeOfBits(_u32[0]);
}

/**
 * The largest float32 whose code is `code`: "code ≤ c" is "value ≤ this".
 * (The negative code next to zero holds only negative values: −0 is +0.)
 */
export function codeUpperValue(code) {
    var bits;
    if (code >= 0x8000) {
        bits = (((code & 0x7fff) << 16) | 0xffff) >>> 0;
        if (bits >= 0x7f800000) return Infinity;
    } else {
        bits = (((~code) & 0xffff) << 16) >>> 0;
    }
    _u32[0] = bits;
    var v = _f32[0];
    return v === 0 && code < 0x8000 ? NEG_TINY : v;
}

/**
 * Codes of a Float32Array: `out[outStart + i * stride]` for every value.
 * @param {Float32Array} col
 * @param {Uint16Array} out
 */
export function encodeColumn(col, out, outStart, stride) {
    var u = new Uint32Array(col.buffer, col.byteOffset, col.length);
    var s = stride || 1, o = outStart || 0;
    for (var i = 0; i < u.length; i++, o += s) out[o] = codeOfBits(u[i]);
}

// ----------------------------------------------------------------------------
// Bins
// ----------------------------------------------------------------------------

/**
 * Quantile bin edges of one feature's codes: sorted, distinct codes, at most
 * MAX_VALUE_BINS − 1 of them. A code c ≠ 0 falls in bin 1 + (edges below c),
 * so bin b holds the codes in (edges[b − 2], edges[b − 1]]. With few distinct
 * codes every one gets its own bin; otherwise each bin takes about 1/255 of
 * the values (one heavy code may take several shares). Missing codes take no
 * part.
 * @param {Uint16Array} codes
 * @param {number} start
 * @param {number} len
 * @param {Uint32Array} hist scratch of 65,536, all zero (left zero)
 * @returns {Uint16Array}
 */
export function binEdges(codes, start, len, hist) {
    var n = 0, lo = 65536, hi = 0;
    for (var i = start; i < start + len; i++) {
        var c = codes[i];
        if (!c) continue;
        hist[c]++; n++;
        if (c < lo) lo = c;
        if (c > hi) hi = c;
    }
    if (!n) return new Uint16Array(0);
    var distinct = 0;
    for (var d = lo; d <= hi; d++) if (hist[d]) distinct++;
    var edges = [];
    if (distinct <= MAX_VALUE_BINS) {
        for (var e = lo; e <= hi; e++) {
            if (!hist[e]) continue;
            if (e !== hi) edges.push(e);
            hist[e] = 0;
        }
        return Uint16Array.from(edges);
    }
    var step = n / MAX_VALUE_BINS, next = step, acc = 0;
    for (var k = lo; k <= hi; k++) {
        var h = hist[k];
        if (!h) continue;
        hist[k] = 0;
        acc += h;
        if (acc >= next && k !== hi && edges.length < MAX_VALUE_BINS - 1) {
            edges.push(k);
            while (next <= acc) next += step;
        }
    }
    return Uint16Array.from(edges);
}

/**
 * Bin one feature's codes: `out[outStart + i] = bin of codes[start + i]`.
 * @param {Uint8Array} lut scratch of 65,536
 */
export function binColumn(codes, start, len, edges, out, outStart, lut) {
    // lut[c] = 1 + #(edges < c); lut[0] = 0 (missing).
    var prev = 0;
    for (var k = 0; k < edges.length; k++) {
        lut.fill(k + 1, prev, edges[k] + 1);
        prev = edges[k] + 1;
    }
    lut.fill(edges.length + 1, prev, 65536);
    lut[0] = 0;
    for (var i = 0; i < len; i++) out[outStart + i] = lut[codes[start + i]];
}

// ----------------------------------------------------------------------------
// Random numbers (seeded)
// ----------------------------------------------------------------------------

/** A 32-bit mix of integers (seeds per tree and per sample). */
export function mix32(a, b, c) {
    var h = (a | 0) ^ 0x9e3779b9;
    h = Math.imul(h ^ (h >>> 16), 0x85ebca6b) ^ (b | 0);
    h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) ^ (c | 0);
    h = Math.imul(h ^ (h >>> 16), 0x27d4eb2f);
    return (h ^ (h >>> 15)) >>> 0;
}

/** mulberry32: a small seeded generator, uniform in [0, 1). */
export function seededRandom(seed) {
    var a = seed >>> 0;
    return function () {
        a = (a + 0x6D2B79F5) | 0;
        var t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// ----------------------------------------------------------------------------
// Training one tree
// ----------------------------------------------------------------------------

/**
 * The scratch one worker reuses across its trees.
 * @param {number} R rows
 */
export function treeScratch(R, F) {
    return {
        cnt: new Int32Array(R), wt: new Float64Array(R), idx: new Int32Array(R), tmp: new Int32Array(R),
        perm: new Int32Array(F), h0: new Float64Array(256), h1: new Float64Array(256), hc: new Int32Array(256),
        imp: new Float64Array(F),
    };
}

function growNodes(t, need) {
    if (need <= t.cap) return;
    var cap = Math.max(need, t.cap * 2);
    var grow = function (Arr, old) { var a = new Arr(cap); a.set(old); return a; };
    t.feat = grow(Uint16Array, t.feat); t.bin = grow(Uint8Array, t.bin); t.miss = grow(Uint8Array, t.miss);
    t.kid = grow(Int32Array, t.kid); t.prob = grow(Float32Array, t.prob);
    t.cap = cap;
}

/**
 * Grow one tree.
 * @param {{bins: Uint8Array, y: Uint8Array, R: number, F: number, nb: Uint16Array}} data
 *   feature-major bins (`bins[f * R + r]`), labels 0/1, and each feature's
 *   highest value bin
 * @param {Int32Array} trainRows the rows it may draw from (sorted, distinct)
 * @param {number} seed this tree's seed
 * @param {{maxDepth:number, minLeaf:number, mtry?:number}} params
 * @param {Object} s scratch from `treeScratch`
 * @returns {{n:number, feat:Uint16Array, bin:Uint8Array, miss:Uint8Array, kid:Int32Array, prob:Float32Array,
 *            imp:Float64Array}} bin form; imp sums to 1 (all 0 for a single leaf)
 */
export function trainTree(data, trainRows, seed, params, s) {
    var R = data.R, F = data.F, bins = data.bins, y = data.y, nb = data.nb;
    var maxDepth = params.maxDepth, minLeaf = Math.max(1, params.minLeaf | 0);
    var mtry = Math.max(1, Math.min(F, params.mtry || Math.round(Math.sqrt(F))));
    var rand = seededRandom(seed);
    var n = trainRows.length;
    var cnt = s.cnt, wt = s.wt, idx = s.idx, tmp = s.tmp, perm = s.perm, h0 = s.h0, h1 = s.h1, hc = s.hc, imp = s.imp;
    cnt.fill(0);
    imp.fill(0);
    for (var k = 0; k < n; k++) cnt[trainRows[(rand() * n) | 0]]++;
    var n0 = 0, n1 = 0;
    for (var q = 0; q < n; q++) { var r0 = trainRows[q]; if (y[r0]) n1 += cnt[r0]; else n0 += cnt[r0]; }
    var w0 = n0 > 0 ? (n0 + n1) / (2 * n0) : 0, w1 = n1 > 0 ? (n0 + n1) / (2 * n1) : 0;
    var m = 0, W0 = 0, W1 = 0;
    for (var q2 = 0; q2 < n; q2++) {
        var r1 = trainRows[q2];
        if (!cnt[r1]) continue;
        idx[m++] = r1;
        var w = cnt[r1] * (y[r1] ? w1 : w0);
        wt[r1] = w;
        if (y[r1]) W1 += w; else W0 += w;
    }
    for (var f0 = 0; f0 < F; f0++) perm[f0] = f0;
    var t = { cap: 0, feat: new Uint16Array(0), bin: new Uint8Array(0), miss: new Uint8Array(0), kid: new Int32Array(0), prob: new Float32Array(0) };
    growNodes(t, 256);
    var nodes = 1;
    // Stack of [node, start, end, depth, W0, W1].
    var stack = [[0, 0, m, 0, W0, W1]];
    var totalGain = 0;
    while (stack.length) {
        var it = stack.pop();
        // A child's weights come from a subtraction: rounding can leave a hair below zero.
        var node = it[0], start = it[1], end = it[2], depth = it[3], a0 = it[4] > 0 ? it[4] : 0, a1 = it[5] > 0 ? it[5] : 0;
        var count = end - start, Wn = a0 + a1;
        t.prob[node] = Wn > 0 ? Math.min(1, a1 / Wn) : 0;
        t.feat[node] = LEAF;
        if (depth >= maxDepth || count < 2 * minLeaf || !(a0 > 0) || !(a1 > 0)) continue;
        var qParent = (a0 * a0 + a1 * a1) / Wn;
        var minGain = 1e-10 * Wn;
        var best = null, bestGain = minGain;
        var visited = 0;
        for (var j = 0; j < F && visited < mtry; j++) {
            var pick = j + ((rand() * (F - j)) | 0);
            var f = perm[pick]; perm[pick] = perm[j]; perm[j] = f;
            var nbf = nb[f];
            h0.fill(0, 0, nbf + 1); h1.fill(0, 0, nbf + 1); hc.fill(0, 0, nbf + 1);
            var base = f * R;
            for (var e = start; e < end; e++) {
                var r = idx[e], bv = bins[base + r];
                if (y[r]) h1[bv] += wt[r]; else h0[bv] += wt[r];
                hc[bv]++;
            }
            // Constant in this node (one bin, missing counted as one): not a try.
            var used = 0;
            for (var u = 0; u <= nbf && used < 2; u++) if (hc[u]) used++;
            if (used < 2) continue;
            visited++;
            var m0 = h0[0], m1 = h1[0], mc = hc[0];
            var T0 = a0 - m0, T1 = a1 - m1, Tc = count - mc;
            var L0 = 0, L1 = 0, Lc = 0;
            for (var b = 1; b <= nbf; b++) {
                var last = b === nbf;
                if (last) {
                    // Every value left, missing right: only when something is missing.
                    if (!mc || !Tc) break;
                    L0 = T0; L1 = T1; Lc = Tc;
                } else {
                    if (!hc[b]) continue;
                    L0 += h0[b]; L1 += h1[b]; Lc += hc[b];
                }
                var R0 = T0 - L0, R1 = T1 - L1, Rc = Tc - Lc;
                // Missing to the right.
                var la0 = L0, la1 = L1, lac = Lc, ra0 = R0 + m0, ra1 = R1 + m1, rac = Rc + mc;
                if (lac >= minLeaf && rac >= minLeaf && la0 + la1 > 0 && ra0 + ra1 > 0) {
                    var g = (la0 * la0 + la1 * la1) / (la0 + la1) + (ra0 * ra0 + ra1 * ra1) / (ra0 + ra1) - qParent;
                    if (g > bestGain) {
                        bestGain = g;
                        best = [f, b, mc ? 0 : (la0 + la1 >= ra0 + ra1 ? 1 : 0), la0, la1, ra0, ra1];
                    }
                }
                if (mc && !last) {
                    // Missing to the left.
                    var lb0 = L0 + m0, lb1 = L1 + m1, lbc = Lc + mc;
                    if (lbc >= minLeaf && Rc >= minLeaf && R0 + R1 > 0 && lb0 + lb1 > 0) {
                        var g2 = (lb0 * lb0 + lb1 * lb1) / (lb0 + lb1) + (R0 * R0 + R1 * R1) / (R0 + R1) - qParent;
                        if (g2 > bestGain) { bestGain = g2; best = [f, b, 1, lb0, lb1, R0, R1]; }
                    }
                }
            }
        }
        if (!best) continue;
        // Split: rows going left first (stable), then the right ones.
        var bf = best[0], bb = best[1], bm = best[2], bbase = bf * R;
        var nl = start, nr = 0;
        for (var p = start; p < end; p++) {
            var rr = idx[p], v = bins[bbase + rr];
            if (v === 0 ? bm === 1 : v <= bb) idx[nl++] = rr; else tmp[nr++] = rr;
        }
        for (var p2 = 0; p2 < nr; p2++) idx[nl + p2] = tmp[p2];
        growNodes(t, nodes + 2);
        var left = nodes;
        nodes += 2;
        t.feat[node] = bf; t.bin[node] = bb; t.miss[node] = bm; t.kid[node] = left;
        imp[bf] += bestGain;
        totalGain += bestGain;
        stack.push([left + 1, nl, end, depth + 1, best[5], best[6]]);
        stack.push([left, start, nl, depth + 1, best[3], best[4]]);
    }
    var outImp = new Float64Array(F);
    if (totalGain > 0) for (var fi = 0; fi < F; fi++) outImp[fi] = imp[fi] / totalGain;
    return {
        n: nodes, feat: t.feat.slice(0, nodes), bin: t.bin.slice(0, nodes), miss: t.miss.slice(0, nodes),
        kid: t.kid.slice(0, nodes), prob: t.prob.slice(0, nodes), imp: outImp,
    };
}

/** The seed of tree `index` of fold `fold` (−1 = the final forest). */
export function treeSeed(seed, fold, index) {
    return mix32(seed, fold + 7, index);
}

/**
 * Train some trees of a forest (a worker job).
 * @param {Object} data as for `trainTree`, plus `folds`: each fold's training rows
 * @param {{trees: number[], fold: number, seed: number, params: Object, trainRows?: Int32Array, rowsIndex?: number}} job
 *   `fold` seeds the trees (−1 = the final forest); the rows are `job.trainRows`, else `data.folds[job.rowsIndex]`
 * @returns {{trees: Object[], imp: Float64Array}} the trees in job order; imp summed over them
 */
export function trainTrees(data, job) {
    var s = treeScratch(data.R, data.F);
    var rows = job.trainRows || data.folds[job.rowsIndex];
    var out = [], imp = new Float64Array(data.F);
    job.trees.forEach(function (ti) {
        var tr = trainTree(data, rows, treeSeed(job.seed, job.fold, ti), job.params, s);
        for (var f = 0; f < data.F; f++) imp[f] += tr.imp[f];
        delete tr.imp;
        out.push(tr);
    });
    return { trees: out, imp: imp };
}

// ----------------------------------------------------------------------------
// Scoring binned rows (cross-validation)
// ----------------------------------------------------------------------------

/**
 * Each row's score from trees in bin form: the mean of their leaves' shares.
 * @param {Object[]} trees
 * @param {{bins: Uint8Array, R: number}} data
 * @param {Int32Array} rows
 * @returns {Float32Array}
 */
export function scoreBinnedRows(trees, data, rows) {
    var out = new Float32Array(rows.length);
    var R = data.R, bins = data.bins;
    for (var i = 0; i < rows.length; i++) {
        var r = rows[i], s = 0;
        for (var k = 0; k < trees.length; k++) {
            var tr = trees[k], nd = 0, f = tr.feat[0];
            while (f !== LEAF) {
                var v = bins[f * R + r];
                nd = tr.kid[nd] + (v === 0 ? (tr.miss[nd] ? 0 : 1) : (v <= tr.bin[nd] ? 0 : 1));
                f = tr.feat[nd];
            }
            s += tr.prob[nd];
        }
        out[i] = trees.length ? s / trees.length : NaN;
    }
    return out;
}

// ----------------------------------------------------------------------------
// The raw forest
// ----------------------------------------------------------------------------

/**
 * Trees in bin form → the raw forest over only the features its splits use.
 * @param {Object[]} trees bin form, in tree order
 * @param {Uint16Array[]} edges each feature's bin edges (`binEdges`)
 * @param {string[]} names each feature's name
 * @param {Float64Array} [impSum] importance summed over the trees
 * @returns {{features: string[], importance: Float32Array, forest: {feat, val, kid, miss, roots}}}
 */
export function finalizeForest(trees, edges, names, impSum) {
    var usedIndex = new Map();
    var total = 0;
    trees.forEach(function (tr) {
        total += tr.n;
        for (var i = 0; i < tr.n; i++) if (tr.feat[i] !== LEAF && !usedIndex.has(tr.feat[i])) usedIndex.set(tr.feat[i], -1);
    });
    var usedList = Array.from(usedIndex.keys()).sort(function (a, b) { return a - b; });
    usedList.forEach(function (f, i) { usedIndex.set(f, i); });
    var feat = new Uint16Array(total), val = new Float32Array(total), kid = new Int32Array(total), miss = new Uint8Array(total);
    var roots = new Int32Array(trees.length);
    var o = 0;
    trees.forEach(function (tr, k) {
        roots[k] = o;
        for (var i = 0; i < tr.n; i++) {
            var f = tr.feat[i];
            if (f === LEAF) { feat[o + i] = LEAF; val[o + i] = tr.prob[i]; continue; }
            feat[o + i] = usedIndex.get(f);
            var e = edges[f];
            val[o + i] = tr.bin[i] - 1 < e.length ? codeUpperValue(e[tr.bin[i] - 1]) : Infinity;
            kid[o + i] = tr.kid[i] + o;
            miss[o + i] = tr.miss[i];
        }
        o += tr.n;
    });
    var importance = new Float32Array(usedList.length);
    if (impSum && trees.length) usedList.forEach(function (f, i) { importance[i] = impSum[f] / trees.length; });
    return {
        features: usedList.map(function (f) { return names[f]; }), importance: importance,
        forest: { feat: feat, val: val, kid: kid, miss: miss, roots: roots },
    };
}

/**
 * Scores of `n` rows from a raw forest: X is row-major, `X[t * Fm + j]` the
 * value of the forest's feature j on row t (NaN = missing).
 * @returns {Float32Array} out
 */
export function scoreRows(forest, X, Fm, n, out) {
    var feat = forest.feat, val = forest.val, kid = forest.kid, miss = forest.miss, roots = forest.roots;
    var T = roots.length;
    out = out || new Float32Array(n);
    for (var t = 0; t < n; t++) {
        var base = t * Fm, s = 0;
        for (var k = 0; k < T; k++) {
            var i = roots[k], f = feat[i];
            while (f !== LEAF) {
                var v = X[base + f];
                i = kid[i] + (v !== v ? (miss[i] ? 0 : 1) : (v <= val[i] ? 0 : 1));
                f = feat[i];
            }
            s += val[i];
        }
        out[t] = T ? s / T : NaN;
    }
    return out;
}

/** A forest's node count. */
export function forestNodes(forest) {
    return forest && forest.feat ? forest.feat.length : 0;
}

function hashBytes(h, bytes) {
    for (var i = 0; i < bytes.length; i++) h = Math.imul(h ^ bytes[i], 0x01000193);
    return h >>> 0;
}

function bytesOf(a) {
    return new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
}

/**
 * A forest's identity: two 32-bit hashes over its arrays and feature names,
 * as 16 hex digits. The same labels and poses retrain the same forest, so
 * the same hash.
 */
export function forestHash(forest, features) {
    var hs = [0x811c9dc5, 0x2f6b9ad1];
    var parts = [forest.feat, forest.val, forest.kid, forest.miss, forest.roots];
    var names = new TextEncoderLike().encode((features || []).join('\u0001'));
    return hs.map(function (h0) {
        var h = h0;
        parts.forEach(function (p) { h = hashBytes(h, bytesOf(p)); });
        h = hashBytes(h, names);
        return ('0000000' + h.toString(16)).slice(-8);
    }).join('');
}

/** UTF-8 encoding without depending on TextEncoder being there (old workers, Node). */
function TextEncoderLike() {}
TextEncoderLike.prototype.encode = function (s) {
    if (typeof TextEncoder === 'function') return new TextEncoder().encode(s);
    var out = [];
    for (var i = 0; i < s.length; i++) {
        var c = s.charCodeAt(i);
        if (c < 0x80) out.push(c);
        else if (c < 0x800) out.push(0xc0 | (c >> 6), 0x80 | (c & 63));
        else out.push(0xe0 | (c >> 12), 0x80 | ((c >> 6) & 63), 0x80 | (c & 63));
    }
    return Uint8Array.from(out);
};

// ----------------------------------------------------------------------------
// Base64 of typed arrays (model files)
// ----------------------------------------------------------------------------

var B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
var B64_INV = (function () { var t = new Int16Array(256).fill(-1); for (var i = 0; i < 64; i++) t[B64.charCodeAt(i)] = i; return t; })();

/** A typed array's bytes as base64 (little-endian, as every browser lays them out). */
export function typedToBase64(arr) {
    var b = bytesOf(arr);
    if (typeof btoa === 'function') {
        var parts = [];
        for (var i = 0; i < b.length; i += 0x8000) parts.push(String.fromCharCode.apply(null, b.subarray(i, i + 0x8000)));
        return btoa(parts.join(''));
    }
    var out = [];
    for (var j = 0; j < b.length; j += 3) {
        var x = (b[j] << 16) | ((j + 1 < b.length ? b[j + 1] : 0) << 8) | (j + 2 < b.length ? b[j + 2] : 0);
        out.push(B64[(x >> 18) & 63] + B64[(x >> 12) & 63] + (j + 1 < b.length ? B64[(x >> 6) & 63] : '=') + (j + 2 < b.length ? B64[x & 63] : '='));
    }
    return out.join('');
}

/**
 * Base64 back to a typed array of `Type` (null when malformed or the length
 * is not a whole number of elements).
 */
export function base64ToTyped(str, Type) {
    if (typeof str !== 'string' || str.length % 4 || /[^A-Za-z0-9+/=]/.test(str)) return null;
    var pad = str.endsWith('==') ? 2 : str.endsWith('=') ? 1 : 0;
    var len = str.length / 4 * 3 - pad;
    if (len % Type.BYTES_PER_ELEMENT) return null;
    var buf = new ArrayBuffer(len), b = new Uint8Array(buf);
    if (typeof atob === 'function') {
        var bin;
        try { bin = atob(str); } catch (e) { return null; }
        if (bin.length !== len) return null;
        for (var k = 0; k < len; k++) b[k] = bin.charCodeAt(k);
        return new Type(buf);
    }
    var o = 0;
    for (var i = 0; i < str.length; i += 4) {
        var c0 = B64_INV[str.charCodeAt(i)], c1 = B64_INV[str.charCodeAt(i + 1)];
        var c2 = str[i + 2] === '=' ? 0 : B64_INV[str.charCodeAt(i + 2)], c3 = str[i + 3] === '=' ? 0 : B64_INV[str.charCodeAt(i + 3)];
        if (c0 < 0 || c1 < 0 || c2 < 0 || c3 < 0) return null;
        var x = (c0 << 18) | (c1 << 12) | (c2 << 6) | c3;
        if (o < len) b[o++] = (x >> 16) & 255;
        if (o < len) b[o++] = (x >> 8) & 255;
        if (o < len) b[o++] = x & 255;
    }
    return new Type(buf);
}

/**
 * Check a raw forest read from a file: every index in range, children after
 * their parent, thresholds and shares finite where they must be.
 * @returns {string|null} why it is broken, or null
 */
export function forestProblem(forest, nFeatures) {
    if (!forest || !(forest.feat instanceof Uint16Array) || !(forest.val instanceof Float32Array) ||
        !(forest.kid instanceof Int32Array) || !(forest.miss instanceof Uint8Array) || !(forest.roots instanceof Int32Array)) return 'missing arrays';
    var n = forest.feat.length;
    if (forest.val.length !== n || forest.kid.length !== n || forest.miss.length !== n || !forest.roots.length) return 'arrays of different lengths';
    for (var k = 0; k < forest.roots.length; k++) if (forest.roots[k] < 0 || forest.roots[k] >= n) return 'a root out of range';
    for (var i = 0; i < n; i++) {
        var f = forest.feat[i];
        if (f === LEAF) { if (!(forest.val[i] >= 0 && forest.val[i] <= 1)) return 'a leaf share out of range'; continue; }
        if (f >= nFeatures) return 'a feature out of range';
        if (!(forest.kid[i] > i && forest.kid[i] + 1 < n)) return 'a child out of range';
        if (forest.val[i] !== forest.val[i]) return 'a missing threshold';
    }
    return null;
}

// ----------------------------------------------------------------------------
// Feature sets
// ----------------------------------------------------------------------------

/** The target's own features a social classifier reads: its body as a whole, its head and tail base. */
var TARGET_KINDS = ['cheight', 'cspeed', 'vspeed', 'turn', 'elong', 'missing'];

/**
 * The features a classifier for a (social) behavior reads, by name, as
 * `resolveFeatureName` resolves them: every animal feature (an animal's own;
 * for a social behavior the actor's), and for a social behavior also every
 * relational feature and the target's body-wide features plus its head and
 * tail base heights and speeds ("target · …"). Deterministic for a catalog.
 * @param {Object} catalog
 * @param {boolean} social
 * @param {{head:number, tailBase:number}} roles
 * @returns {string[]}
 */
export function classifierFeatureNames(catalog, social, roles) {
    var own = catalog.animal.map(function (f) { return f.name; });
    if (!social) return own;
    var head = roles && roles.head, tail = roles && roles.tailBase;
    var target = catalog.animal.filter(function (f) {
        var base = catalog.bases.get(f.base);
        if (!base) return false;
        if (TARGET_KINDS.indexOf(base.kind) >= 0) return true;
        return (base.kind === 'height' || base.kind === 'nspeed') && (base.node === head || base.node === tail);
    }).map(function (f) { return TARGET_PREFIX + f.name; });
    return catalog.pair.map(function (f) { return f.name; }).concat(own, target);
}

/**
 * Resolve feature names to refs for the kernels: `{who: 'self'|'actor'|'target'|'pair', name}`
 * (the catalog name). Null when any name is not a feature of this catalog.
 */
export function featureRefs(catalog, names, social) {
    var out = [];
    for (var i = 0; i < names.length; i++) {
        var r = resolveFeatureName(catalog, names[i], social);
        if (!r) return null;
        out.push({ who: r.who, name: r.feature.name });
    }
    return out;
}

/** The names among `names` that are not features of this catalog. */
export function missingFeatures(catalog, names, social) {
    return names.filter(function (nm) { return !resolveFeatureName(catalog, nm, social); });
}

// ----------------------------------------------------------------------------
// Chunk kernels (worker jobs)
// ----------------------------------------------------------------------------

/**
 * Which frames of [start, end) an animal is there on: at least half its used
 * nodes (the centroid's rule in the feature engine), on the chunk's (already
 * masked) data. Frames skipped as broken tracking are all-missing, so absent.
 * @returns {Uint8Array} 1 = there
 */
export function presentFrames(spec, data, chunk) {
    var N = spec.nodeNames.length, D = spec.mode === '2d' ? 2 : 3;
    var used = spec.used || [];
    var need = Math.max(1, Math.ceil(used.length / 2));
    var n = chunk.end - chunk.start;
    var out = new Uint8Array(n);
    for (var t = 0; t < n; t++) {
        var base = (t + chunk.start - chunk.padStart) * N * D, c = 0;
        for (var i = 0; i < used.length; i++) { var v = data[base + used[i] * D]; if (v === v) c++; }
        out[t] = c >= need ? 1 : 0;
    }
    return out;
}

function splitRefs(refs) {
    var self = [], target = [], pair = [];
    refs.forEach(function (r) {
        var into = r.who === 'pair' ? pair : r.who === 'target' ? target : self;
        if (into.indexOf(r.name) < 0) into.push(r.name);
    });
    return { self: self, target: target, pair: pair };
}

function union(a, b) {
    if (!a.length) return b.slice();
    var seen = new Set(a), out = a.slice();
    b.forEach(function (x) { if (!seen.has(x)) { seen.add(x); out.push(x); } });
    return out;
}

/**
 * Training rows of one chunk, as codes (a worker job): every feature of each
 * row set, for each row `[frame, actor, target]` (target −1 when not social).
 * Animals are computed one at a time, so only one animal's columns exist at
 * once.
 * @param {Object} spec
 * @param {{start, end, padStart, padEnd, data}} chunk
 * @param {{sets: Array<{refs: Array<{who, name}>, rows: Int32Array}>}} req
 * @returns {{sets: Array<{codes: Uint16Array}>}} feature-major: `codes[j * n + i]`
 */
export function extractTrainingRows(spec, chunk, req) {
    var memo = {};
    var sets = req.sets.map(function (s) {
        var n = s.rows.length / 3;
        return { refs: s.refs, rows: s.rows, n: n, codes: new Uint16Array(s.refs.length * n), parts: splitRefs(s.refs) };
    });
    // Which animals each set needs as actor and as target, and which pairs (names merged once per animal).
    var asActor = new Map(), asTarget = new Map(), pairs = new Map();
    sets.forEach(function (s) {
        var actors = new Set(), targets = new Set();
        for (var i = 0; i < s.n; i++) {
            var a = s.rows[3 * i + 1], b = s.rows[3 * i + 2];
            actors.add(a);
            if (b >= 0) {
                targets.add(b);
                if (s.parts.pair.length && !pairs.has(a + '>' + b)) pairs.set(a + '>' + b, [a, b]);
            }
        }
        if (s.parts.self.length) actors.forEach(function (a) { asActor.set(a, union(asActor.get(a) || [], s.parts.self)); });
        if (s.parts.target.length) targets.forEach(function (b) { asTarget.set(b, union(asTarget.get(b) || [], s.parts.target)); });
    });
    var animals = Array.from(new Set(Array.from(asActor.keys()).concat(Array.from(asTarget.keys())))).sort(function (x, y) { return x - y; });
    animals.forEach(function (x) {
        var names = union(asActor.get(x) || [], asTarget.get(x) || []);
        var cols = computeFeatureChunk(spec, chunk, { animal: names, only: [x] }, memo).animal[x] || {};
        sets.forEach(function (s) {
            for (var j = 0; j < s.refs.length; j++) {
                var rf = s.refs[j];
                if (rf.who === 'pair') continue;
                var col = cols[rf.name], at = rf.who === 'target' ? 2 : 1, o = j * s.n;
                for (var i = 0; i < s.n; i++) {
                    if (s.rows[3 * i + at] !== x) continue;
                    s.codes[o + i] = col ? codeOf(col[s.rows[3 * i] - chunk.start]) : MISSING_CODE;
                }
            }
        });
    });
    var pnames = [];
    sets.forEach(function (s) { pnames = union(pnames, s.parts.pair); });
    pairs.forEach(function (pr, key) {
        var cols = computeFeatureChunk(spec, chunk, { pair: pnames, pairs: [pr] }, memo).pair[key] || {};
        sets.forEach(function (s) {
            for (var j = 0; j < s.refs.length; j++) {
                var rf = s.refs[j];
                if (rf.who !== 'pair') continue;
                var col = cols[rf.name], o = j * s.n;
                for (var i = 0; i < s.n; i++) {
                    if (s.rows[3 * i + 1] !== pr[0] || s.rows[3 * i + 2] !== pr[1]) continue;
                    s.codes[o + i] = col ? codeOf(col[s.rows[3 * i] - chunk.start]) : MISSING_CODE;
                }
            }
        });
    });
    return { sets: sets.map(function (s) { return { codes: s.codes }; }) };
}

/** Row-major values of a model's features for frames [0, n). */
function gather(refs, colsOf, n) {
    var Fm = refs.length;
    var X = new Float32Array(n * Fm);
    for (var j = 0; j < Fm; j++) {
        var col = colsOf(refs[j]);
        if (!col) { for (var t0 = 0; t0 < n; t0++) X[t0 * Fm + j] = NaN; continue; }
        for (var t = 0; t < n; t++) X[t * Fm + j] = col[t];
    }
    return X;
}

/**
 * Score every model on one chunk (a worker job): non-social models per
 * animal, social ones per ordered pair (a MUTUAL behavior once per pair,
 * a < b, the mean of its two directions). A frame without the animal (or the
 * target) is unscored (NaN), like a rule's missing feature.
 * @param {Object} spec
 * @param {{start, end, padStart, padEnd, data}} chunk
 * @param {{models: Array<{behaviorId, social, mutual, refs: Array<{who, name}>}>}} plan
 * @param {Object<string, {feat, val, kid, miss, roots}>} forests by behavior id
 * @returns {{scores: Object<string, Object<string, Float32Array>>}} key 'a' or 'a>b'
 */
export function scoreForestsChunk(spec, chunk, plan, forests) {
    var nA = chunk.data.length, n = chunk.end - chunk.start;
    var memo = {};
    var models = plan.models.filter(function (m) { return forests && forests[m.behaviorId]; });
    var lone = models.filter(function (m) { return !m.social; });
    var social = nA > 1 ? models.filter(function (m) { return m.social; }) : [];
    var present = chunk.data.map(function (d) { return presentFrames(spec, d, chunk); });
    var selfNames = [], targetNames = [], pairNames = [];
    lone.concat(social).forEach(function (m) {
        var p = splitRefs(m.refs);
        selfNames = union(selfNames, p.self); targetNames = union(targetNames, p.target); pairNames = union(pairNames, p.pair);
    });
    var scores = {};
    models.forEach(function (m) { scores[m.behaviorId] = {}; });
    var directed = {};
    social.forEach(function (m) { directed[m.behaviorId] = {}; });
    var tcols = [];
    if (social.length && targetNames.length) {
        for (var b0 = 0; b0 < nA; b0++) tcols[b0] = computeFeatureChunk(spec, chunk, { animal: targetNames, only: [b0] }, memo).animal[b0] || {};
    }
    var mask = function (out, pa, pb) {
        for (var t = 0; t < n; t++) if (!pa[t] || (pb && !pb[t])) out[t] = NaN;
        return out;
    };
    for (var a = 0; a < nA; a++) {
        var cols = selfNames.length ? (computeFeatureChunk(spec, chunk, { animal: selfNames, only: [a] }, memo).animal[a] || {}) : {};
        lone.forEach(function (m) {
            var X = gather(m.refs, function (r) { return cols[r.name]; }, n);
            scores[m.behaviorId][String(a)] = mask(scoreRows(forests[m.behaviorId], X, m.refs.length, n), present[a]);
        });
        if (!social.length) continue;
        for (var b = 0; b < nA; b++) {
            if (b === a) continue;
            var pcols = pairNames.length ? (computeFeatureChunk(spec, chunk, { pair: pairNames, pairs: [[a, b]] }, memo).pair[a + '>' + b] || {}) : {};
            social.forEach(function (m) {
                var X = gather(m.refs, function (r) {
                    return r.who === 'pair' ? pcols[r.name] : r.who === 'target' ? (tcols[b] || {})[r.name] : cols[r.name];
                }, n);
                directed[m.behaviorId][a + '>' + b] = mask(scoreRows(forests[m.behaviorId], X, m.refs.length, n), present[a], present[b]);
            });
        }
    }
    social.forEach(function (m) {
        var d = directed[m.behaviorId];
        if (!m.mutual) { scores[m.behaviorId] = d; return; }
        for (var a2 = 0; a2 < nA; a2++) {
            for (var b2 = a2 + 1; b2 < nA; b2++) {
                var x = d[a2 + '>' + b2], y = d[b2 + '>' + a2], out = new Float32Array(n);
                for (var t = 0; t < n; t++) {
                    var u = x[t], v = y[t];
                    out[t] = u !== u ? v : v !== v ? u : (u + v) / 2;
                }
                scores[m.behaviorId][a2 + '>' + b2] = out;
            }
        }
    });
    return { scores: scores };
}
