// pose/hdbscan.js — HDBSCAN clustering in typed arrays (Campello, Moulavi &
// Sander 2013; the algorithm as the Python `hdbscan` package runs it, whose
// results the Node test compares with on a fixture).
//
// A point's CORE distance is the distance to its k-th nearest other point
// (`minSamples`). The MUTUAL REACHABILITY of two points is the largest of their
// distance and their two core distances, so sparse points are pushed away from
// everything and dense regions keep their shape. Then:
//   1. the minimum spanning tree of mutual reachability — Borůvka over a kd-tree
//      (each round every component finds its cheapest edge out, pruning whole
//      subtrees that are all one component or too far), about n log n on a
//      low-dimensional embedding rather than Prim's n²;
//   2. the single-linkage hierarchy from its edges, cheapest first;
//   3. the CONDENSED tree: walking down from the root, a split where both sides
//      have at least `minClusterSize` points makes two clusters, and a smaller
//      side's points just "fall out" of the cluster at that level;
//   4. Excess of Mass selection: a cluster's stability is what its points add up
//      to while they stay in it, Σ (λ_leaving − λ_birth), with λ = 1 / distance;
//      a cluster is kept when it is at least as stable as its descendants
//      together, and the root never counts (so one big blob is all noise).
//
// Every point below a selected cluster takes its label; the rest are noise (-1).
// Steps 1–2 do not depend on `minClusterSize`, so trying several sizes
// (`clusterAtSize`) re-runs only 3–4. `bestTreeOverlaps` asks whether a condensed tree
// holds a given group of points at any level, which is how a cluster is looked for in
// another layout (pose/behavior-embedding.js).
//
// Deterministic: no random numbers; ties are settled by point order. Pure and
// import-free, and nothing here holds one JS object per point (the 4 GB heap
// rule).
//
// Depends on: nothing.

/** Below this many points a kd-tree node is a leaf. */
var LEAF_SIZE = 16;

// ----------------------------------------------------------------------------
// kd-tree
// ----------------------------------------------------------------------------

/**
 * A balanced kd-tree. Points are copied in tree order (`P`), so a node's points
 * are the consecutive positions [start[node], end[node]); `order[pos]` is the
 * point's original index. Children come after their parent in the node arrays.
 * @param {ArrayLike<number>} X n × d, row-major
 * @returns {{n, d, P: Float64Array, order: Int32Array, count: number, start: Int32Array, end: Int32Array,
 *            left: Int32Array, lo: Float64Array, hi: Float64Array}} left[node] −1 for a leaf; the right child
 *   is left[node] + (the left subtree's node count), kept in `right`
 */
export function buildKdTree(X, n, d, leafSize) {
    var leaf = Math.max(1, leafSize || LEAF_SIZE);
    var order = new Int32Array(n);
    for (var i = 0; i < n; i++) order[i] = i;
    var cap = Math.max(3, 4 * Math.ceil(n / leaf) + 3);
    var start = new Int32Array(cap), end = new Int32Array(cap), left = new Int32Array(cap).fill(-1), right = new Int32Array(cap).fill(-1);
    var lo = new Float64Array(cap * d), hi = new Float64Array(cap * d);
    var count = 0;

    function box(node, s, e) {
        var o = node * d;
        for (var j = 0; j < d; j++) { lo[o + j] = Infinity; hi[o + j] = -Infinity; }
        for (var p = s; p < e; p++) {
            var q = order[p] * d;
            for (var j2 = 0; j2 < d; j2++) {
                var v = X[q + j2];
                if (v < lo[o + j2]) lo[o + j2] = v;
                if (v > hi[o + j2]) hi[o + j2] = v;
            }
        }
    }
    /** order[k] becomes the k-th smallest of [s, e) along dim (Hoare quickselect, median-of-three pivot). */
    function select(s, e, k, dim) {
        var a = s, b = e - 1;
        while (a < b) {
            var m = (a + b) >> 1;
            var va = X[order[a] * d + dim], vm = X[order[m] * d + dim], vb = X[order[b] * d + dim];
            var pivot = va < vm ? (vm < vb ? vm : (va < vb ? vb : va)) : (va < vb ? va : (vm < vb ? vb : vm));
            var i2 = a, j2 = b;
            while (i2 <= j2) {
                while (X[order[i2] * d + dim] < pivot) i2++;
                while (X[order[j2] * d + dim] > pivot) j2--;
                if (i2 <= j2) { var t = order[i2]; order[i2] = order[j2]; order[j2] = t; i2++; j2--; }
            }
            if (k <= j2) b = j2; else if (k >= i2) a = i2; else return;
        }
    }
    function build(s, e) {
        var node = count++;
        start[node] = s; end[node] = e;
        box(node, s, e);
        if (e - s <= leaf) return node;
        var o = node * d, dim = 0, width = -1;
        for (var j = 0; j < d; j++) {
            var w = hi[o + j] - lo[o + j];
            if (w > width) { width = w; dim = j; }
        }
        if (!(width > 0)) return node;       // every point here is the same one
        var mid = (s + e) >> 1;
        select(s, e, mid, dim);
        left[node] = build(s, mid);
        right[node] = build(mid, e);
        return node;
    }
    if (n) build(0, n);
    var P = new Float64Array(n * d);
    for (var p = 0; p < n; p++) {
        var q = order[p] * d, o2 = p * d;
        for (var j3 = 0; j3 < d; j3++) P[o2 + j3] = X[q + j3];
    }
    return { n: n, d: d, P: P, order: order, count: count, start: start, end: end, left: left, right: right, lo: lo, hi: hi };
}

function boxDist2(tree, node, P, o) {
    var d = tree.d, lo = tree.lo, hi = tree.hi, b = node * d, s = 0;
    for (var j = 0; j < d; j++) {
        var v = P[o + j], u = lo[b + j];
        if (v < u) { s += (u - v) * (u - v); continue; }
        u = hi[b + j];
        if (v > u) s += (v - u) * (v - u);
    }
    return s;
}

/**
 * Squared core distances, by tree position: the squared distance from each point
 * to its k-th nearest other point. Bounded max-heap per point.
 * @returns {Float64Array}
 */
export function coreDistances2(tree, k) {
    var n = tree.n, d = tree.d, P = tree.P;
    var out = new Float64Array(n);
    var K = Math.min(Math.max(1, k | 0), Math.max(1, n - 1));
    if (n < 2) return out;
    var heap = new Float64Array(K);
    var stackN = new Int32Array(128), stackB = new Float64Array(128);
    for (var i = 0; i < n; i++) {
        var o = i * d, size = 0, top = Infinity, sp = 0;
        stackN[sp] = 0; stackB[sp++] = 0;
        while (sp) {
            var node = stackN[--sp];
            if (stackB[sp] >= top) continue;
            var l = tree.left[node];
            if (l < 0) {
                for (var j = tree.start[node]; j < tree.end[node]; j++) {
                    if (j === i) continue;
                    var q = j * d, s = 0;
                    for (var m = 0; m < d; m++) { var u = P[o + m] - P[q + m]; s += u * u; }
                    if (size < K) {
                        var c = size++;
                        while (c > 0) { var par = (c - 1) >> 1; if (heap[par] >= s) break; heap[c] = heap[par]; c = par; }
                        heap[c] = s;
                        if (size === K) top = heap[0];
                    } else if (s < top) {
                        var c2 = 0;
                        for (;;) {
                            var kid = 2 * c2 + 1;
                            if (kid >= K) break;
                            if (kid + 1 < K && heap[kid + 1] > heap[kid]) kid++;
                            if (heap[kid] <= s) break;
                            heap[c2] = heap[kid]; c2 = kid;
                        }
                        heap[c2] = s;
                        top = heap[0];
                    }
                }
                continue;
            }
            var r = tree.right[node];
            var dl = boxDist2(tree, l, P, o), dr = boxDist2(tree, r, P, o);
            if (dl <= dr) {
                if (dr < top) { stackN[sp] = r; stackB[sp++] = dr; }
                if (dl < top) { stackN[sp] = l; stackB[sp++] = dl; }
            } else {
                if (dl < top) { stackN[sp] = l; stackB[sp++] = dl; }
                if (dr < top) { stackN[sp] = r; stackB[sp++] = dr; }
            }
        }
        out[i] = size ? heap[0] : 0;
    }
    return out;
}

// ----------------------------------------------------------------------------
// The minimum spanning tree of mutual reachability (Borůvka)
// ----------------------------------------------------------------------------

function ufFind(parent, a) {
    var r = a;
    while (parent[r] !== r) r = parent[r];
    while (parent[a] !== r) { var nx = parent[a]; parent[a] = r; a = nx; }
    return r;
}

/**
 * The n − 1 edges of the minimum spanning tree over mutual reachability.
 * @param {Object} tree from `buildKdTree`
 * @param {Float64Array} core2 from `coreDistances2`
 * @returns {{a: Int32Array, b: Int32Array, w: Float64Array}} original point indices; w = mutual reachability
 */
export function mutualReachabilityMst(tree, core2) {
    var n = tree.n, d = tree.d, P = tree.P, nodes = tree.count;
    var a = new Int32Array(Math.max(0, n - 1)), b = new Int32Array(Math.max(0, n - 1)), w = new Float64Array(Math.max(0, n - 1));
    if (n < 2) return { a: a, b: b, w: w };
    var uf = new Int32Array(n), comp = new Int32Array(n);
    for (var i0 = 0; i0 < n; i0++) { uf[i0] = i0; comp[i0] = i0; }
    var minCore = new Float64Array(nodes);
    for (var nd = nodes - 1; nd >= 0; nd--) {
        var l0 = tree.left[nd];
        if (l0 < 0) {
            var mc = Infinity;
            for (var p0 = tree.start[nd]; p0 < tree.end[nd]; p0++) if (core2[p0] < mc) mc = core2[p0];
            minCore[nd] = mc;
        } else {
            minCore[nd] = Math.min(minCore[l0], minCore[tree.right[nd]]);
        }
    }
    var nodeComp = new Int32Array(nodes);
    var bestW = new Float64Array(n), bestA = new Int32Array(n), bestB = new Int32Array(n);
    var stackN = new Int32Array(128), stackB = new Float64Array(128);
    var edges = 0;
    while (edges < n - 1) {
        // Which nodes are all one component.
        for (var nd2 = nodes - 1; nd2 >= 0; nd2--) {
            var l1 = tree.left[nd2];
            if (l1 < 0) {
                var c0 = comp[tree.start[nd2]];
                for (var p1 = tree.start[nd2] + 1; p1 < tree.end[nd2]; p1++) if (comp[p1] !== c0) { c0 = -1; break; }
                nodeComp[nd2] = c0;
            } else {
                var cl = nodeComp[l1], cr = nodeComp[tree.right[nd2]];
                nodeComp[nd2] = cl === cr ? cl : -1;
            }
        }
        bestW.fill(Infinity);
        bestA.fill(-1);
        for (var i = 0; i < n; i++) {
            var c = comp[i], ci = core2[i];
            if (ci >= bestW[c]) continue;            // nothing from i can beat what c has
            var o = i * d, sp = 0;
            stackN[sp] = 0; stackB[sp++] = 0;
            while (sp) {
                var node = stackN[--sp];
                if (stackB[sp] >= bestW[c] || nodeComp[node] === c) continue;
                var l = tree.left[node];
                if (l < 0) {
                    for (var j = tree.start[node]; j < tree.end[node]; j++) {
                        var cj = comp[j];
                        if (cj === c) continue;
                        var q = j * d, s = 0;
                        for (var m = 0; m < d; m++) { var u = P[o + m] - P[q + m]; s += u * u; }
                        if (ci > s) s = ci;
                        if (core2[j] > s) s = core2[j];
                        if (s < bestW[c]) { bestW[c] = s; bestA[c] = i; bestB[c] = j; }
                        if (s < bestW[cj]) { bestW[cj] = s; bestA[cj] = i; bestB[cj] = j; }
                    }
                    continue;
                }
                var r = tree.right[node];
                var dl = boxDist2(tree, l, P, o), dr = boxDist2(tree, r, P, o);
                var bl = Math.max(dl, ci, minCore[l]), br = Math.max(dr, ci, minCore[r]);
                if (dl <= dr) {
                    if (br < bestW[c]) { stackN[sp] = r; stackB[sp++] = br; }
                    if (bl < bestW[c]) { stackN[sp] = l; stackB[sp++] = bl; }
                } else {
                    if (bl < bestW[c]) { stackN[sp] = l; stackB[sp++] = bl; }
                    if (br < bestW[c]) { stackN[sp] = r; stackB[sp++] = br; }
                }
            }
        }
        var added = 0;
        for (var c1 = 0; c1 < n; c1++) {
            if (bestA[c1] < 0) continue;
            var ra = ufFind(uf, bestA[c1]), rb = ufFind(uf, bestB[c1]);
            if (ra === rb) continue;
            uf[ra < rb ? rb : ra] = ra < rb ? ra : rb;
            a[edges] = tree.order[bestA[c1]]; b[edges] = tree.order[bestB[c1]]; w[edges] = Math.sqrt(bestW[c1]);
            edges++; added++;
        }
        if (!added) break;
        for (var i2 = 0; i2 < n; i2++) comp[i2] = ufFind(uf, i2);
    }
    return { a: a, b: b, w: w };
}

// ----------------------------------------------------------------------------
// Single linkage
// ----------------------------------------------------------------------------

/**
 * The single-linkage hierarchy of a spanning tree's edges, cheapest first. Node
 * ids below n are points; node n + i is the i-th merge.
 * @returns {{n, left: Int32Array, right: Int32Array, dist: Float64Array, size: Int32Array}}
 */
export function singleLinkage(mst, n) {
    var m = Math.max(0, n - 1);
    var order = new Int32Array(m);
    for (var i = 0; i < m; i++) order[i] = i;
    order.sort(function (x, y) { return mst.w[x] - mst.w[y] || x - y; });
    var left = new Int32Array(m), right = new Int32Array(m), dist = new Float64Array(m), size = new Int32Array(m);
    var uf = new Int32Array(2 * n), sizes = new Int32Array(2 * n);
    for (var u0 = 0; u0 < 2 * n; u0++) { uf[u0] = u0; sizes[u0] = u0 < n ? 1 : 0; }
    for (var e = 0; e < m; e++) {
        var k = order[e];
        var ra = ufFind(uf, mst.a[k]), rb = ufFind(uf, mst.b[k]);
        var made = n + e;
        left[e] = ra; right[e] = rb; dist[e] = mst.w[k];
        size[e] = sizes[ra] + sizes[rb];
        uf[ra] = made; uf[rb] = made;
        sizes[made] = size[e];
    }
    return { n: n, left: left, right: right, dist: dist, size: size };
}

// ----------------------------------------------------------------------------
// The condensed tree and selection
// ----------------------------------------------------------------------------

/**
 * The condensed tree at a smallest cluster size. Cluster ids count up from n (n
 * is the root); a cluster's id is always greater than its parent's. λ = 1 / distance;
 * a distance of 0 (identical points) counts as the smallest distance above 0.
 * @returns {{n, clusters: number, parent: Int32Array, child: Int32Array, lambda: Float64Array, size: Int32Array, rows: number}}
 */
export function condenseTree(L, minClusterSize) {
    var n = L.n, mcs = Math.max(2, minClusterSize | 0);
    var cap = 3 * n + 3;
    var parent = new Int32Array(cap), child = new Int32Array(cap), lambda = new Float64Array(cap), size = new Int32Array(cap);
    var rows = 0;
    if (n < 2) return { n: n, clusters: 1, parent: parent, child: child, lambda: lambda, size: size, rows: 0 };
    var smallest = Infinity;
    for (var e = 0; e < L.dist.length; e++) if (L.dist[e] > 0 && L.dist[e] < smallest) smallest = L.dist[e];
    if (smallest === Infinity) smallest = 1;
    var root = 2 * n - 2;
    var relabel = new Int32Array(2 * n - 1), ignore = new Uint8Array(2 * n - 1);
    relabel[root] = n;
    var nextLabel = n + 1;
    var queue = new Int32Array(2 * n - 1), head = 0, tail = 0;
    queue[tail++] = root;
    var stack = new Int32Array(n + 1);
    var countOf = function (node) { return node >= n ? L.size[node - n] : 1; };
    var fallOut = function (node, c, lam) {
        var sp = 0;
        stack[sp++] = node;
        while (sp) {
            var x = stack[--sp];
            ignore[x] = 1;
            if (x < n) { parent[rows] = c; child[rows] = x; lambda[rows] = lam; size[rows] = 1; rows++; continue; }
            stack[sp++] = L.left[x - n];
            stack[sp++] = L.right[x - n];
        }
    };
    while (head < tail) {
        var node = queue[head++];
        if (node < n || ignore[node]) continue;
        var i = node - n, l = L.left[i], r = L.right[i], c = relabel[node];
        var lam = 1 / (L.dist[i] > 0 ? L.dist[i] : smallest);
        var lc = countOf(l), rc = countOf(r);
        if (lc >= mcs && rc >= mcs) {
            relabel[l] = nextLabel++;
            parent[rows] = c; child[rows] = relabel[l]; lambda[rows] = lam; size[rows] = lc; rows++;
            relabel[r] = nextLabel++;
            parent[rows] = c; child[rows] = relabel[r]; lambda[rows] = lam; size[rows] = rc; rows++;
        } else if (lc < mcs && rc < mcs) {
            fallOut(l, c, lam);
            fallOut(r, c, lam);
        } else if (lc < mcs) {
            relabel[r] = c;
            fallOut(l, c, lam);
        } else {
            relabel[l] = c;
            fallOut(r, c, lam);
        }
        if (!ignore[l]) queue[tail++] = l;
        if (!ignore[r]) queue[tail++] = r;
    }
    return { n: n, clusters: nextLabel - n, parent: parent, child: child, lambda: lambda, size: size, rows: rows };
}

/**
 * Excess of Mass selection and the labels it gives.
 * @param {Object} cond from `condenseTree`
 * @returns {{labels: Int32Array, count: number, noise: number, stability: Float64Array}} labels 0..count−1 in
 *   cluster-id order, −1 for noise; stability of each selected cluster
 */
export function selectClusters(cond) {
    var n = cond.n, C = cond.clusters, rows = cond.rows;
    var labels = new Int32Array(n).fill(-1);
    if (C < 2) return { labels: labels, count: 0, noise: n, stability: new Float64Array(0) };
    var birth = new Float64Array(C), parentOf = new Int32Array(C).fill(-1), stab = new Float64Array(C);
    for (var r = 0; r < rows; r++) {
        if (cond.child[r] >= n) {
            var k = cond.child[r] - n;
            birth[k] = cond.lambda[r];
            parentOf[k] = cond.parent[r] - n;
        }
    }
    for (var r2 = 0; r2 < rows; r2++) {
        var p = cond.parent[r2] - n;
        stab[p] += (cond.lambda[r2] - birth[p]) * cond.size[r2];
    }
    var keep = new Uint8Array(C), sub = new Float64Array(C);
    for (var c = C - 1; c >= 1; c--) {
        if (sub[c] > stab[c]) { keep[c] = 0; stab[c] = sub[c]; } else keep[c] = 1;
        sub[parentOf[c]] += stab[c];
    }
    // Downward: a kept cluster takes everything below it.
    var top = new Int32Array(C).fill(-1), count = 0, labelOf = new Int32Array(C).fill(-1);
    var chosen = [];
    for (var c2 = 1; c2 < C; c2++) {
        var above = top[parentOf[c2]];
        if (above >= 0) { top[c2] = above; continue; }
        if (keep[c2]) { top[c2] = c2; labelOf[c2] = count++; chosen.push(stab[c2]); }
    }
    var noise = 0;
    for (var r3 = 0; r3 < rows; r3++) {
        if (cond.child[r3] >= n) continue;
        var t = top[cond.parent[r3] - n];
        if (t >= 0) labels[cond.child[r3]] = labelOf[t]; else noise++;
    }
    return { labels: labels, count: count, noise: noise, stability: Float64Array.from(chosen) };
}

/**
 * For each of some clusters (labels 0..count−1, −1 for none), its best Jaccard overlap
 * with ANY cluster of a condensed tree, the root aside: not only those its selection
 * would keep. That asks whether the tree holds this group of points at all, whichever
 * level selection would have settled on.
 * @param {ArrayLike<number>} labels n
 * @param {number} count
 * @param {Object} cond from `condenseTree`, over the same n points
 * @returns {Float64Array} per cluster
 */
export function bestTreeOverlaps(labels, count, cond) {
    var n = cond.n, C = cond.clusters, rows = cond.rows;
    var out = new Float64Array(count), sizeA = new Int32Array(count);
    for (var i = 0; i < n; i++) if (labels[i] >= 0) sizeA[labels[i]]++;
    if (C < 2) return out;
    var parentOf = new Int32Array(C).fill(-1), nodeSize = new Int32Array(C), both = new Int32Array(count * C);
    for (var r = 0; r < rows; r++) {
        var p = cond.parent[r] - n, ch = cond.child[r];
        if (ch >= n) { parentOf[ch - n] = p; continue; }
        nodeSize[p]++;
        if (labels[ch] >= 0) both[labels[ch] * C + p]++;
    }
    // Upward: a child's points are its parent's too (a child's id is greater than its parent's).
    for (var c = C - 1; c >= 1; c--) {
        var up = parentOf[c];
        nodeSize[up] += nodeSize[c];
        for (var k = 0; k < count; k++) both[k * C + up] += both[k * C + c];
    }
    for (var k2 = 0; k2 < count; k2++) {
        for (var c2 = 1; c2 < C; c2++) {
            var b = both[k2 * C + c2];
            if (!b) continue;
            var j = b / (sizeA[k2] + nodeSize[c2] - b);
            if (j > out[k2]) out[k2] = j;
        }
    }
    return out;
}

/** Clusters of an already-built hierarchy at one smallest cluster size. */
export function clusterAtSize(linkage, minClusterSize) {
    return selectClusters(condenseTree(linkage, minClusterSize));
}

/**
 * The hierarchy of n points in d dimensions (what every cluster size shares).
 * @param {ArrayLike<number>} X n × d, row-major
 * @param {{minSamples?: number}} [opts] minSamples default 1
 * @returns {Object} for `clusterAtSize`
 */
export function hdbscanLinkage(X, n, d, opts) {
    var tree = buildKdTree(X, n, d);
    var core2 = coreDistances2(tree, (opts && opts.minSamples) || 1);
    return singleLinkage(mutualReachabilityMst(tree, core2), n);
}

/**
 * HDBSCAN in one call.
 * @param {ArrayLike<number>} X n × d, row-major
 * @param {{minClusterSize: number, minSamples?: number}} opts
 * @returns {{labels: Int32Array, count: number, noise: number, stability: Float64Array}}
 */
export function hdbscan(X, n, d, opts) {
    return clusterAtSize(hdbscanLinkage(X, n, d, opts), opts.minClusterSize);
}

/** Total weight of a spanning tree (tests). */
export function mstWeight(mst) {
    var s = 0;
    for (var i = 0; i < mst.w.length; i++) s += mst.w[i];
    return s;
}
