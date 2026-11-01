// pose/behavior-umap.js — UMAP through the vendored umap-js (lib/umap-js/), with
// the nearest neighbours and the fuzzy graph kept in typed arrays.
//
// umap-js is used as it is, for its smooth-distance calibration, its a/b curve
// fit and its gradient descent. Two of its parts cannot run at the size discovery
// needs (30,000 bins × 60 neighbours), so they are replaced here from outside the
// library:
//   * its sparse matrix is a Map with a "row:col" string key and an object per
//     entry — measured 1.75 GB of heap at 10,000 points, which at 30,000 would be
//     past the renderer's 4 GB cap;
//   * its nearest neighbours (random-projection trees, then descent over JS
//     arrays) are approximate and slow; `knnRows`
//     (pose/behavior-discovery-core.js) gives the exact ones, in parallel, through
//     its public `setPrecomputedKNN`.
// `TypedGraphUMAP` gives back the same graph it would have built — the fuzzy union
// a + aᵀ − a∘aᵀ with set_op_mix_ratio 1, in the same (row, col) order, so the
// same run of random numbers follows the same edges — in a small read-only object
// that answers the calls umap-js makes on its graph. It skips the search graph,
// which only umap-js's `transform` reads: new points here are labelled by forests.
//
// umap-js starts from random positions. With `init` the layout starts from those
// instead (umap-learn scales its start to ±10: so does this).
//
// This module and the library load only when discovery runs: in the discovery
// worker, or dynamically on the inline path, never at startup.
//
// Depends on: lib/umap-js/umap-js.esm.js

import { UMAP } from '../lib/umap-js/umap-js.esm.js';

/**
 * The fuzzy graph as umap-js reads it, in typed arrays, rows and columns sorted:
 * only what `initializeSimplicialSetEmbedding`, `getNEpochs` and
 * `initializeOptimization` ask of a SparseMatrix.
 */
class TypedGraph {
    constructor(n, rows, cols, vals) {
        this.nRows = n;
        this.nCols = n;
        this.rows = rows;
        this.cols = cols;
        this.vals = vals;
    }
    getValues() { return this.vals; }
    getDims() { return [this.nRows, this.nCols]; }
    map(fn) {
        var out = new Float64Array(this.vals.length);
        for (var i = 0; i < out.length; i++) out[i] = fn(this.vals[i], this.rows[i], this.cols[i]);
        return new TypedGraph(this.nRows, this.rows, this.cols, out);
    }
    /** Only the entries with a value: umap-js drops the others right after this call. */
    getAll() {
        var out = [];
        for (var i = 0; i < this.vals.length; i++) {
            if (this.vals[i]) out.push({ value: this.vals[i], row: this.rows[i], col: this.cols[i] });
        }
        return out;
    }
}

/**
 * The fuzzy union of a directed k-neighbour graph: w_ij = a_ij + a_ji − a_ij · a_ji,
 * with a_ij = 1 up to rho_i and exp(−(d − rho_i) / sigma_i) past it.
 * @param {Int32Array} indices n × k
 * @param {Float32Array} distances n × k
 * @param {ArrayLike<number>} sigmas
 * @param {ArrayLike<number>} rhos
 * @returns {TypedGraph}
 */
export function fuzzyGraph(indices, distances, sigmas, rhos, n, k) {
    // Each row's neighbours sorted by column, and who points at each row.
    var col = new Int32Array(n * k), val = new Float64Array(n * k);
    var incoming = new Int32Array(n + 1);
    for (var i = 0; i < n; i++) {
        var o = i * k, m = 0;
        for (var j = 0; j < k; j++) {
            var c = indices[o + j];
            if (c === i || c < 0) continue;
            var d = distances[o + j] - rhos[i];
            var v = d <= 0 ? 1 : Math.exp(-d / sigmas[i]);
            // Insertion by column; a duplicate cannot occur in one row.
            var at = o + m++;
            while (at > o && col[at - 1] > c) { col[at] = col[at - 1]; val[at] = val[at - 1]; at--; }
            col[at] = c; val[at] = v;
            incoming[c + 1]++;
        }
        for (var e = m; e < k; e++) col[o + e] = -1;
    }
    for (var i1 = 0; i1 < n; i1++) incoming[i1 + 1] += incoming[i1];
    var nnzIn = incoming[n];
    var fromRow = new Int32Array(nnzIn), fromVal = new Float64Array(nnzIn), next = incoming.slice(0, n);
    for (var i2 = 0; i2 < n; i2++) {
        for (var j2 = i2 * k; j2 < (i2 + 1) * k; j2++) {
            var c2 = col[j2];
            if (c2 < 0) break;
            var p = next[c2]++;
            fromRow[p] = i2; fromVal[p] = val[j2];
        }
    }
    // Merge each row's two sorted lists.
    var rows = new Int32Array(2 * nnzIn), cols = new Int32Array(2 * nnzIn), vals = new Float64Array(2 * nnzIn);
    var nz = 0;
    for (var r = 0; r < n; r++) {
        var a = r * k, aEnd = (r + 1) * k, b = incoming[r], bEnd = incoming[r + 1];
        while (a < aEnd && col[a] >= 0 && b < bEnd) {
            var ca = col[a], cb = fromRow[b];
            rows[nz] = r;
            if (ca === cb) { cols[nz] = ca; vals[nz] = val[a] + fromVal[b] - val[a] * fromVal[b]; a++; b++; }
            else if (ca < cb) { cols[nz] = ca; vals[nz] = val[a]; a++; }
            else { cols[nz] = cb; vals[nz] = fromVal[b]; b++; }
            nz++;
        }
        for (; a < aEnd && col[a] >= 0; a++, nz++) { rows[nz] = r; cols[nz] = col[a]; vals[nz] = val[a]; }
        for (; b < bEnd; b++, nz++) { rows[nz] = r; cols[nz] = fromRow[b]; vals[nz] = fromVal[b]; }
    }
    return new TypedGraph(n, rows.slice(0, nz), cols.slice(0, nz), vals.slice(0, nz));
}

class TypedGraphUMAP extends UMAP {
    fuzzySimplicialSet(X, nNeighbors) {
        var res = this.smoothKNNDistance(this.knnDistances, nNeighbors, this.localConnectivity);
        return fuzzyGraph(this.typedKnn.indices, this.typedKnn.distances, res.sigmas, res.rhos, this.typedKnn.n, this.typedKnn.k);
    }
    makeSearchFns() { /* no transform */ }
    makeSearchGraph() { return null; }
}

/** A row's view of an n × k typed array, what umap-js's smoothing reads. */
function rowViews(arr, n, k) {
    var out = new Array(n);
    for (var i = 0; i < n; i++) out[i] = arr.subarray(i * k, (i + 1) * k);
    return out;
}

/** mulberry32, the generator the forests use. */
function seeded(seed) {
    var a = seed >>> 0;
    return function () {
        a = (a + 0x6D2B79F5) | 0;
        var t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

/**
 * A UMAP layout, ready to step.
 * @param {{indices: Int32Array, distances: Float32Array, n: number, k: number}} knn
 * @param {{nComponents?: number, minDist?: number, seed?: number, nEpochs?: number, init?: Float32Array}} [opts]
 *   init: n × nComponents start positions (default: umap-js's random ones)
 * @returns {{epochs: number, step: function(): number, embedding: function(): Float32Array}}
 */
export function prepareUmap(knn, opts) {
    opts = opts || {};
    var dims = opts.nComponents || 2;
    var rand = seeded(opts.seed == null ? 1 : opts.seed);
    var umap = new TypedGraphUMAP({ nComponents: dims, nNeighbors: knn.k, minDist: opts.minDist == null ? 0 : opts.minDist,
                                    random: rand, nEpochs: opts.nEpochs || 0 });
    umap.typedKnn = knn;
    umap.setPrecomputedKNN(rowViews(knn.indices, knn.n, knn.k), rowViews(knn.distances, knn.n, knn.k));
    var epochs = umap.initializeFit({ length: knn.n });
    var pos = umap.getEmbedding();
    if (opts.init) {
        var mx = 0;
        for (var i = 0; i < opts.init.length; i++) { var av = Math.abs(opts.init[i]); if (av > mx) mx = av; }
        var scale = mx > 0 ? 10 / mx : 1;
        for (var r = 0; r < knn.n; r++) {
            for (var d = 0; d < dims; d++) pos[r][d] = opts.init[r * dims + d] * scale + 1e-4 * (rand() - 0.5);
        }
    }
    return {
        epochs: epochs,
        step: function () { return umap.step(); },
        embedding: function () {
            var out = new Float32Array(knn.n * dims), cur = umap.getEmbedding();
            for (var r2 = 0; r2 < knn.n; r2++) for (var d2 = 0; d2 < dims; d2++) out[r2 * dims + d2] = cur[r2][d2];
            return out;
        },
    };
}
