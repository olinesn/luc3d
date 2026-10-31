/**
 * test-behavior-discovery.mjs — finding behaviors without labels
 * (pose/behavior-discovery.js and its parts), away from the app, inline.
 *
 *   1. HDBSCAN (pose/hdbscan.js): the spanning tree weighs what a brute-force Prim
 *      gives and what the Python `hdbscan` package gives, and the labels equal the
 *      package's (tests/fixtures/hdbscan/, its exact "generic" algorithm, with
 *      min_samples 1: larger values tie, and its own algorithms then differ too);
 *      Excess of Mass against leaves; a blob is all noise; identical points work.
 *   2. Nearest neighbours: exact, nearest first, the row itself first; the same
 *      however the rows are split into jobs.
 *   3. UMAP's graph (pose/behavior-umap.js): the typed graph is exactly what umap-js
 *      builds with its own sparse matrix, value for value and in the same order, so
 *      the same random numbers follow the same edges; a layout pulls planted groups
 *      apart; the same seed gives the same layout.
 *   4. Standardize and reduce (pose/behavior-embedding.js): at least 70% of the
 *      variance, the fewest components that reach it; missing values, constant
 *      features and outliers; the top-directions path past 512 features agrees with
 *      the exact one about how many components it takes; speeds through a log.
 *      4b. Clusters that repeat: overlap between labellings; a cluster only one
 *      layout has becomes noise; structureless data keeps next to none.
 *   5. Bins: a value is its bin's mean (a frame's bin holds ⌊w / 2⌋ frames before
 *      it), the seconds convert with the frame rate, values do not depend on how the
 *      session was cut into chunks.
 *   6. Planted behaviors are recovered, end to end: four kinematically different
 *      states on two animals, each cluster mostly one state, on every frame; clusters
 *      numbered by size; the same run twice gives the same runs; a state change is
 *      found within a bin of where it was planted (frameshift).
 *   7. Keeping a result: a states layer per session (source 'discovery'), a map in
 *      memory only, the layer saved and read back, the map gone after a reset.
 *   8. Split a behavior: only the labelled frames are clustered and given states, and
 *      two kinds of frames labelled alike come apart.
 *   9. Several sessions: one numbering across them; a session with another body is
 *      named as left out.
 *  10. Too little data says so; one blob of poses says so; cancelling stops.
 *
 * ESM, picked up by `tests/run-mjs-tests.mjs`.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Session, Skeleton, InstanceGroup } from '../pose/pose-data.js';
import { collectBehaviorInput, resolveBody, featureSpec, floorFrame, brokenPoseFrames } from '../pose/behavior-tensor.js';
import { catalogFor, planChunks, featurePadFrames, sliceChunkData } from '../pose/behavior-features.js';
import { seededRandom } from '../pose/behavior-forest.js';
import { resetEthogram, addBehavior } from '../pose/behavior-ethogram.js';
import { labelRange } from '../pose/behavior-labels.js';
import { stateAt, stateLayer, exportStateLayers, importStateLayers, stateColor } from '../pose/behavior-states.js';
import { buildKdTree, coreDistances2, mutualReachabilityMst, mstWeight, hdbscan, hdbscanLinkage, clusterAtSize } from '../pose/hdbscan.js';
import { knnRows, binFrames, binCenters, discoveryFeatureNames, speedFeatures, binRowsChunk } from '../pose/behavior-discovery-core.js';
import { standardizeAndReduce, clusterLayouts, clusterSizes, bestOverlaps, SPEED_LOG_OFFSET } from '../pose/behavior-embedding.js';
import { fuzzyGraph, prepareUmap } from '../pose/behavior-umap.js';
import { UMAP } from '../lib/umap-js/umap-js.esm.js';
import { discoverBehaviors, storeDiscovery, discoveryResult, resetDiscoveryResults, clusterFloor, MAX_BINS } from '../pose/behavior-discovery.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

let passed = 0, failed = 0;
const check = (cond, msg) => {
    if (cond) { passed++; console.log('  ok   ' + msg); }
    else { failed++; console.log('  FAIL ' + msg); }
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

function gaussian(rand) {
    return () => { let u = 0; while (!u) u = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand()); };
}

/** Same partition, whatever the label numbers (noise = −1 must agree). */
function samePartition(a, b) {
    const m = new Map(), r = new Map();
    for (let i = 0; i < a.length; i++) {
        if ((a[i] < 0) !== (b[i] < 0)) return false;
        if (a[i] < 0) continue;
        if (m.has(a[i]) && m.get(a[i]) !== b[i]) return false;
        if (r.has(b[i]) && r.get(b[i]) !== a[i]) return false;
        m.set(a[i], b[i]); r.set(b[i], a[i]);
    }
    return true;
}

/** Prim on the complete graph of mutual reachability: the n² answer to check Borůvka with. */
function primWeight(X, n, d, k) {
    const dist2 = (p, q) => { let s = 0; for (let j = 0; j < d; j++) { const u = X[p * d + j] - X[q * d + j]; s += u * u; } return s; };
    const core = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        const all = [];
        for (let j = 0; j < n; j++) if (j !== i) all.push(dist2(i, j));
        all.sort((x, y) => x - y);
        core[i] = all[k - 1];
    }
    const inTree = new Uint8Array(n), best = new Float64Array(n).fill(Infinity);
    best[0] = 0;
    let total = 0;
    for (let c = 0; c < n; c++) {
        let u = -1;
        for (let i = 0; i < n; i++) if (!inTree[i] && (u < 0 || best[i] < best[u])) u = i;
        inTree[u] = 1;
        total += Math.sqrt(best[u]);
        for (let v = 0; v < n; v++) {
            if (inTree[v]) continue;
            const w = Math.max(dist2(u, v), core[u], core[v]);
            if (w < best[v]) best[v] = w;
        }
    }
    return total;
}

// ---------------------------------------------------------------------------
console.log('\n-- 1. HDBSCAN --');
{
    const rand = seededRandom(5), gauss = gaussian(rand);
    for (const [n, d, k] of [[400, 2, 1], [300, 5, 3], [200, 2, 4], [64, 3, 1]]) {
        const X = new Float64Array(n * d);
        for (let i = 0; i < n * d; i++) X[i] = gauss() * (i % 7 === 0 ? 5 : 1) + (i % (2 * d) === 0 ? 8 : 0);
        const tree = buildKdTree(X, n, d);
        const mst = mutualReachabilityMst(tree, coreDistances2(tree, k));
        check(Math.abs(mstWeight(mst) - primWeight(X, n, d, k)) < 1e-6 * primWeight(X, n, d, k),
            `${n} points in ${d}D, k=${k}: the spanning tree weighs what Prim's n² answer does (${mstWeight(mst).toFixed(3)})`);
        check(mst.a.length === n - 1 && new Set(Array.from(tree.order)).size === n, 'n − 1 edges over points kept in tree order');
    }
    const fix = JSON.parse(fs.readFileSync(path.join(__dirname, 'fixtures/hdbscan/python-hdbscan.json'), 'utf8'));
    for (const c of fix.cases) {
        const rows = fix.data[c.data], d = rows[0].length, n = rows.length;
        const X = new Float64Array(rows.flat());
        const tree = buildKdTree(X, n, d);
        const w = mstWeight(mutualReachabilityMst(tree, coreDistances2(tree, c.minSamples)));
        const r = hdbscan(X, n, d, { minClusterSize: c.minClusterSize, minSamples: c.minSamples });
        check(Math.abs(w - c.mstWeight) < 1e-3, `${c.data}, size ${c.minClusterSize}, min_samples ${c.minSamples}: tree weight as Python's (${w.toFixed(3)})`);
        const count = Math.max(...c.labels) + 1;
        check(r.count === count, `the same number of clusters (${r.count})`);
        if (c.minSamples === 1) check(samePartition(Array.from(r.labels), c.labels), `labels equal to Python hdbscan ${fix.hdbscan}'s`);
    }
    // Leaves are finer than Excess of Mass: two tight blobs inside a broad one, far from another.
    const blob = (n, cx, cy, s) => { const o = []; for (let i = 0; i < n; i++) o.push(cx + s * gauss(), cy + s * gauss()); return o; };
    const pts = [...blob(120, 0, 0, 0.2), ...blob(120, 3, 0, 0.2), ...blob(200, 1.5, 0, 2.2), ...blob(300, 40, 40, 0.5)];
    const n2 = pts.length / 2, X2 = Float64Array.from(pts);
    const link = hdbscanLinkage(X2, n2, 2, { minSamples: 5 });
    const eom = clusterAtSize(link, 50), leaf = clusterAtSize(link, 50, true);
    check(eom.count >= 2 && leaf.count >= eom.count, `leaves are at least as fine as Excess of Mass (${leaf.count} against ${eom.count})`);
    check(leaf.noise >= eom.noise, `and leave at least as much noise (${leaf.noise} against ${eom.noise})`);
    // One blob: nothing beats the root's children being too small, so noise, not a cluster.
    const one = hdbscan(Float64Array.from(blob(200, 0, 0, 1)), 200, 2, { minClusterSize: 150 });
    check(one.count === 0 && one.noise === 200, 'one blob, with no split big enough: all noise (the root never counts)');
    // Identical points.
    const dup = new Float64Array(2 * 120);
    for (let i = 60; i < 120; i++) { dup[2 * i] = 10; dup[2 * i + 1] = 10; }
    const dr = hdbscan(dup, 120, 2, { minClusterSize: 20 });
    check(dr.count === 2 && dr.noise === 0, 'identical points make clusters too (distance 0 is not infinite stability)');
    check(eq(clusterSizes(10000, [0.005, 0.01], 5).slice(0, 2), [50, 52]) && eq(clusterSizes(30000, [0.005, 0.01], 60).slice(-1), [300]),
        'cluster sizes: evenly over the shares');
    check(clusterSizes(1000, [0.005, 0.01], 60)[0] === 60 && clusterSizes(1000, [0.005, 0.01], 60).slice(-1)[0] === 120, 'from the floor to twice it where the shares fall below it');
}

// ---------------------------------------------------------------------------
console.log('\n-- 2. Nearest neighbours --');
{
    const rand = seededRandom(9), n = 300, d = 6, k = 11;
    const Y = new Float32Array(n * d);
    for (let i = 0; i < Y.length; i++) Y[i] = rand() * 10;
    const whole = knnRows({ Y, n, d }, [0, n], k);
    let exact = true, selfFirst = true, sorted = true;
    for (let i = 0; i < n; i++) {
        const dists = [];
        for (let j = 0; j < n; j++) { if (j === i) continue; let s = 0; for (let m = 0; m < d; m++) s += (Y[i * d + m] - Y[j * d + m]) ** 2; dists.push([Math.sqrt(s), j]); }
        dists.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
        if (whole.indices[i * k] !== i || whole.distances[i * k] !== 0) selfFirst = false;
        for (let q = 1; q < k; q++) {
            if (Math.abs(whole.distances[i * k + q] - dists[q - 1][0]) > 1e-4) exact = false;
            if (whole.distances[i * k + q] < whole.distances[i * k + q - 1]) sorted = false;
        }
    }
    check(exact, 'exact: each row\'s neighbours are at the brute-force distances');
    check(selfFirst && sorted, 'the row itself first, then nearest first');
    const a = knnRows({ Y, n, d }, [0, 120], k), b = knnRows({ Y, n, d }, [120, n], k);
    check(eq(Array.from(a.indices).concat(Array.from(b.indices)), Array.from(whole.indices)), 'the same however the rows are split into jobs');
}

// ---------------------------------------------------------------------------
console.log('\n-- 3. UMAP --');
{
    const rand = seededRandom(3), gauss = gaussian(rand);
    const n = 400, d = 8, k = 15;
    const Y = new Float32Array(n * d), truth = new Int8Array(n);
    for (let i = 0; i < n; i++) { truth[i] = i % 4; for (let j = 0; j < d; j++) Y[i * d + j] = gauss() + (j === truth[i] ? 12 : 0); }
    const knn = knnRows({ Y, n, d }, [0, n], k);
    // The library's own graph from the same neighbours.
    const rowsOf = (arr) => Array.from({ length: n }, (_, i) => Array.from(arr.subarray(i * k, (i + 1) * k)));
    const lib = new UMAP({ nNeighbors: k, minDist: 0, random: seededRandom(1) });
    lib.setPrecomputedKNN(rowsOf(knn.indices), rowsOf(knn.distances));
    const theirs = lib.fuzzySimplicialSet({ length: n }, k, 1.0);
    const sm = lib.smoothKNNDistance(rowsOf(knn.distances), k, 1.0);
    const ours = fuzzyGraph(knn.indices, knn.distances, sm.sigmas, sm.rhos, n, k);
    const all = theirs.getAll().filter((e) => e.value);
    const mine = ours.getAll();
    check(all.length === mine.length && all.length > n, `the typed graph has umap-js's entries (${mine.length})`);
    check(all.every((e, i) => e.row === mine[i].row && e.col === mine[i].col), 'in the same (row, col) order');
    check(all.every((e, i) => Math.abs(e.value - mine[i].value) < 1e-12), 'with the same values');
    const run = prepareUmap({ indices: knn.indices, distances: knn.distances, n, k }, { nComponents: 2, minDist: 0, seed: 1, nEpochs: 200 });
    for (let e = 0; e < run.epochs; e++) run.step();
    const E = run.embedding();
    const cl = hdbscan(E, n, 2, { minClusterSize: 20 });
    check(cl.count === 4 && samePartition(Array.from(cl.labels).map((v, i) => (v < 0 ? truth[i] : v)), Array.from(truth)), 'four planted groups come out as four clusters');
    const again = prepareUmap({ indices: knn.indices, distances: knn.distances, n, k }, { nComponents: 2, minDist: 0, seed: 1, nEpochs: 200 });
    for (let e = 0; e < again.epochs; e++) again.step();
    check(eq(Array.from(again.embedding()), Array.from(E)), 'the same seed gives the same layout, bit for bit');
    const started = prepareUmap({ indices: knn.indices, distances: knn.distances, n, k }, { nComponents: 2, minDist: 0, seed: 1, nEpochs: 1, init: new Float32Array(n * 2).fill(3) });
    const pos = started.embedding();
    check(Math.max(...pos) < 10.01 && Math.min(...pos) > 9.99, 'a given start is scaled to ±10, as umap-learn scales its own');
}

// ---------------------------------------------------------------------------
console.log('\n-- 4. Standardize and reduce --');
{
    const rand = seededRandom(11), gauss = gaussian(rand);
    const M = 2000, F = 12;
    const rows = new Float32Array(M * F);
    for (let r = 0; r < M; r++) {
        const a = gauss(), b = gauss();
        for (let j = 0; j < F; j++) rows[r * F + j] = (j < 6 ? a : b) * (j + 1) * 10 + 0.2 * gauss() + 50;
    }
    const red = standardizeAndReduce(rows, M, F);
    check(red.d === 2 && red.explained > 0.9, `two latent directions take two components (${red.d}, ${(100 * red.explained).toFixed(0)}%)`);
    check(red.n === M && red.features.length === F, 'every row and feature stays');
    const flat = new Float32Array(M * F);
    for (let i = 0; i < flat.length; i++) flat[i] = gauss();
    const noise = standardizeAndReduce(flat, M, F);
    check(noise.d === Math.ceil(0.7 * F) || noise.d === Math.ceil(0.7 * F) + 1, `independent features take about 70% of them (${noise.d} of ${F})`);
    check(noise.explained >= 0.7, 'and reach at least 70%');
    const fewer = standardizeAndReduce(flat, M, F, { variance: 0.3 });
    check(fewer.d < noise.d && fewer.explained >= 0.3, 'less variance kept takes fewer components');
    // Missing values, a constant feature, a mostly missing one, an outlier.
    const messy = flat.slice();
    for (let r = 0; r < M; r++) { messy[r * F] = 7; if (r % 3) messy[r * F + 1] = NaN; messy[r * F + 2] = r % 50 ? messy[r * F + 2] : NaN; }
    for (let j = 3; j < F; j++) messy[5 * F + j] = NaN;
    messy[7 * F + 4] = 1e6;
    const m = standardizeAndReduce(messy, M, F);
    check(!Array.from(m.features).includes(0) && !Array.from(m.features).includes(1), 'a constant feature and a mostly missing one are left out');
    check(m.n === M - 1 && !Array.from(m.kept).includes(5), 'a row missing most of its features sits out');
    check(Array.from(m.Z).every(Number.isFinite), 'a missing value counts as the mean: no NaN goes on');
    check(Math.max(...m.Z) <= 6.0001, 'and an outlier is cut at 6 standard deviations');
    check(Array.from(m.Y).every(Number.isFinite), 'the components are finite');
    // Past the exact path: top directions, about the same number of components.
    const F2 = 600, M2 = 900;
    const big = new Float32Array(M2 * F2);
    for (let r = 0; r < M2; r++) {
        const lat = [gauss(), gauss(), gauss()];
        for (let j = 0; j < F2; j++) big[r * F2 + j] = lat[j % 3] * (1 + (j % 7)) + 0.3 * gauss();
    }
    const top = standardizeAndReduce(big, M2, F2);
    check(top.d === 3 && top.explained >= 0.7, `600 features (fitPCA's top directions): 3 components for 3 latent directions (${top.d}, ${(100 * top.explained).toFixed(0)}%)`);
    // Speeds: their log, so "still" and "slow" differ as much as "slow" and "fast".
    const speeds = new Float32Array(3000 * 2);
    for (let r = 0; r < 3000; r++) { const v = Math.exp(gauss() * 1.5); speeds[2 * r] = v; speeds[2 * r + 1] = v + 0.1 * gauss(); }
    const raw = standardizeAndReduce(speeds, 3000, 2), logged = standardizeAndReduce(speeds, 3000, 2, { logFeatures: [1, 0] });
    const skew = (Z, D) => { let s3 = 0; for (let r = 0; r < 3000; r++) s3 += Z[r * D] ** 3; return s3 / 3000; };
    check(Math.abs(skew(logged.Z, 2)) < 0.5 && skew(raw.Z, 2) > 2, `a logged column is about symmetric (skew ${skew(logged.Z, 2).toFixed(2)}, raw ${skew(raw.Z, 2).toFixed(2)})`);
    check(SPEED_LOG_OFFSET > 0 && logged.mean[0] < raw.mean[0], 'it is read as log(v + offset)');
}

// ---------------------------------------------------------------------------
console.log('\n-- 4b. Clusters that repeat --');
{
    const rand = seededRandom(13), gauss = gaussian(rand);
    check(eq(Array.from(bestOverlaps([0, 0, 1, 1, -1], [5 - 5, 0, 0, -1, 1], 2, 2)).map((v) => +v.toFixed(3)), [0.667, 0.25]), 'overlap is Jaccard with the best cluster of the other labelling');
    // Two real blobs in both layouts, plus clumps that are in the first layout only.
    const n = 900, E = new Float64Array(2 * n), E2 = new Float64Array(2 * n), truth = new Int8Array(n);
    for (let i = 0; i < n; i++) {
        const c = i < 500 ? 0 : i < 800 ? 1 : 2;
        truth[i] = c;
        const cx = c === 0 ? 0 : c === 1 ? 30 : -30;
        E[2 * i] = cx + 0.8 * gauss(); E[2 * i + 1] = 0.8 * gauss();
        // In the second layout the third group is scattered among the first.
        E2[2 * i] = (c === 2 ? 0 : cx) + (c === 2 ? 3 : 0.8) * gauss(); E2[2 * i + 1] = (c === 2 ? 3 : 0.8) * gauss();
    }
    const res = clusterLayouts(E, E2, n, 2, { share: [0.05, 0.1], floor: 5 });
    check(res.count === 2 && res.dropped >= 1, `the clusters that come back stay (${res.count}), the other does not (${res.dropped} dropped)`);
    check(res.labels[0] === 0 && res.labels[600] === 1 && res.labels[850] === -1, 'numbered by size, largest first, the dropped one is noise');
    check(res.labels instanceof Int16Array && res.repeat.length === 2 && res.repeat[0] > 0.7, `in a typed array, with how well each repeats (${res.repeat[0].toFixed(2)})`);
    // Nothing structured: clusters found on a blob's layout do not come back in another layout of it.
    const M = 3000, F = 60, rows = new Float32Array(M * F);
    for (let r = 0; r < M; r++) { const l = [gauss(), gauss(), gauss(), gauss(), gauss()]; for (let j = 0; j < F; j++) rows[r * F + j] = l[j % 5] * (1 + j % 3) + gauss(); }
    const red = standardizeAndReduce(rows, M, F);
    const knn = knnRows({ Y: red.Y, n: red.n, d: red.d }, [0, red.n], 60);
    const layout = (seed, start) => {
        const init = new Float32Array(red.n * 2);
        for (let i = 0; i < red.n; i++) { init[2 * i] = red.Y[i * red.d]; init[2 * i + 1] = red.Y[i * red.d + 1]; }
        const run = prepareUmap({ indices: knn.indices, distances: knn.distances, n: red.n, k: 60 }, { nComponents: 2, minDist: 0, seed, init: start ? init : null, nEpochs: 200 });
        for (let e = 0; e < run.epochs; e++) run.step();
        return run.embedding();
    };
    const A = layout(1, true), B = layout(2, false);
    const noisy = clusterLayouts(A, B, red.n, 2, { share: [0.005, 0.01], floor: 10 });
    const single = hdbscan(A, red.n, 2, { minClusterSize: 15 });
    check(single.count > noisy.count, `structureless data: ${single.count} clusters from one layout, ${noisy.count} that repeat`);
    check(noisy.count <= 2, `and at most a couple repeat, covering little (${noisy.count}, ${(100 * (1 - noisy.noise / red.n)).toFixed(0)}% of the bins)`);
}

// ---------------------------------------------------------------------------
// A world of four kinematically different states.
const NODES = ['Nose', 'Head', 'EarL', 'EarR', 'Neck', 'Trunk', 'PawL', 'PawR', 'TTI', 'Tail1'];
const EDGES = [[0, 1], [1, 2], [1, 3], [1, 4], [4, 5], [4, 6], [4, 7], [5, 8], [8, 9]];
const ALONG = [50, 36, 30, 30, 20, 0, 18, 18, -40, -70], SIDE = [0, 0, 9, -9, 0, 0, 12, -12, 0, 0], H = [14, 24, 30, 30, 30, 30, 6, 6, 14, 4];
const FPS = 30;
const STATES = ['rest', 'walk', 'rear', 'groom'];

/** `A` animals in their own places, each switching between the states every 1–5 s (or following `plan`). */
function plantedWorld(name, T, seed, A, plan) {
    const rand = seededRandom(seed), gauss = gaussian(rand);
    const s = new Session([], new Skeleton('m', NODES, EDGES), [], name);
    const truth = [];
    for (let a = 0; a < A; a++) { s.addIdentity('id_' + a); truth.push(new Int8Array(T)); }
    let gid = 1;
    for (let a = 0; a < A; a++) {
        const tr = truth[a];
        let cx = 3000 * a, cy = 0, hd = rand() * 6.28, t = 0;
        if (plan) plan(tr, a);
        else {
            while (t < T) {
                const k = Math.floor(rand() * 4), len = Math.round(FPS * (1 + rand() * 4));
                for (let i = t; i < Math.min(T, t + len); i++) tr[i] = k;
                t += len;
            }
        }
        for (let f = 0; f < T; f++) {
            const k = tr[f];
            if (k === 1) { hd += 0.02 * gauss(); cx += 8 * Math.cos(hd); cy += 8 * Math.sin(hd); }
            const g = new InstanceGroup(gid++, s.identities[a].id);
            const p = new Float64Array(NODES.length * 3);
            for (let n = 0; n < NODES.length; n++) {
                let along = ALONG[n], side = SIDE[n], z = H[n];
                if (k === 1 && (n === 6 || n === 7)) { const ph = 0.9 * f + (n === 7 ? Math.PI : 0); along += 9 * Math.sin(ph); z += 4 * Math.max(0, Math.cos(ph)); }
                if (k === 2 && n < 8) { const fr = (ALONG[n] + 40) / 90; z += 75 * fr; along = -40 + (ALONG[n] + 40) * 0.45; }
                if (k === 3 && (n === 6 || n === 7 || n === 0)) { const ph = 1.3 * f; along += (n === 0 ? -14 : -4) + 5 * Math.sin(ph); z += (n === 0 ? -5 : 14) + 3 * Math.cos(ph); side += 3 * Math.sin(ph * 0.5); }
                if (k === 0) z -= 3;
                const c = Math.cos(hd), sn = Math.sin(hd);
                p[3 * n] = cx + along * c - side * sn + gauss();
                p[3 * n + 1] = cy + along * sn + side * c + gauss();
                p[3 * n + 2] = z + gauss();
            }
            g.points3d = p;
            let list = s.instanceGroups.get(f);
            if (!list) { list = []; s.instanceGroups.set(f, list); }
            list.push(g);
        }
    }
    return { session: s, truth, T };
}

function contextOf(session, T, fps) {
    const rate = fps || FPS;
    const input = collectBehaviorInput(session, { numFrames: T });
    const body = resolveBody(input.nodeNames, input.edges, null);
    const spec = featureSpec(input, body, floorFrame([0, 0, 0], [0, 0, 1]), rate);
    return { session, input, body, spec, catalog: catalogFor(spec), fps: rate, broken: brokenPoseFrames(input, body, spec.bodyLength, rate) };
}

// ---------------------------------------------------------------------------
console.log('\n-- 5. Bins --');
{
    check(binFrames(30) === 3 && binFrames(60) === 6 && binFrames(120) === 12 && binFrames(5) === 1, '100 ms is 3 frames at 30 fps, 6 at 60, 12 at 120, and at least one');
    check(eq(Array.from(binCenters(20, 6)), [3, 9, 15]) && eq(Array.from(binCenters(9, 3)), [1, 4, 7]), 'non-overlapping bins: a frame\'s bin holds ⌊w / 2⌋ frames before it');
    const T = 3000, w = plantedWorld('bins', T, 21, 2);
    const ctx = contextOf(w.session, T);
    const N = NODES.length;
    const names = discoveryFeatureNames(ctx.catalog);
    check(names.includes('Nose height') && names.includes('centroid speed') && !names.includes('missing nodes') && !names.some((nm) => / · /.test(nm)),
        `base features, no window statistics, nothing about the tracking (${names.length})`);
    const logged = speedFeatures(ctx.catalog, names);
    check(logged[names.indexOf('centroid speed')] === 1 && logged[names.indexOf('Nose speed')] === 1 && logged[names.indexOf('turning speed')] === 0 && logged[names.indexOf('Nose height')] === 0,
        'speeds are the magnitudes: not turning, which is signed');
    const frames = [1, 4, 100, 2047, 2048, 2049, 2900];
    const req = (list, a) => { const r = new Int32Array(list.length * 2); list.forEach((f, i) => { r[2 * i] = f; r[2 * i + 1] = a; }); return r; };
    const pad = featurePadFrames(ctx.spec);
    const one = { start: 0, end: T, padStart: 0, padEnd: T, data: sliceChunkData(ctx.input.data, { padStart: 0, padEnd: T }, N * 3, ctx.broken.masks) };
    const whole = binRowsChunk(ctx.spec, one, { names, w: 3, rows: req(frames, 1) }).values;
    const chunks = planChunks(T, pad, 512);
    const piecewise = new Float32Array(whole.length);
    frames.forEach((f, i) => {
        const ch = chunks[Math.floor(f / 512)];
        const res = binRowsChunk(ctx.spec, Object.assign({}, ch, { data: sliceChunkData(ctx.input.data, ch, N * 3, ctx.broken.masks) }), { names, w: 3, rows: req([f], 1) });
        piecewise.set(res.values, i * names.length);
    });
    let mostly = 0;
    for (let i = 0; i < whole.length; i++) if (whole[i] === piecewise[i] || (whole[i] !== whole[i] && piecewise[i] !== piecewise[i])) mostly++;
    check(mostly === whole.length, 'a value does not depend on how the session was cut into chunks');
    // A height's bin value is the mean of the frames' heights.
    const j = names.indexOf('Nose height'), bl = ctx.spec.bodyLength[1];
    const mean = (ctx.input.data[1][(99 * N) * 3 + 2] + ctx.input.data[1][(100 * N) * 3 + 2] + ctx.input.data[1][(101 * N) * 3 + 2]) / 3 / bl;
    check(Math.abs(whole[2 * names.length + j] - mean) < 1e-3, `a bin's value is its frames' mean (${whole[2 * names.length + j].toFixed(4)} ≈ ${mean.toFixed(4)})`);
}

// ---------------------------------------------------------------------------
console.log('\n-- 6. Planted behaviors --');
const T6 = 9000;
const world = plantedWorld('planted', T6, 7, 2);
const ctx6 = contextOf(world.session, T6);
const stages = [];
const res6 = await discoverBehaviors({ sources: [{ session: world.session, name: 'planted', ctx: ctx6 }], onProgress: (p) => { if (stages[stages.length - 1] !== p.stage) stages.push(p.stage); } });
{
    check(!res6.error && !res6.cancelled, 'it runs' + (res6.error ? ': ' + res6.error : ''));
    check(eq(stages, ['read', 'bins', 'reduce', 'neighbors', 'layout', 'cluster', 'forests', 'labels']), `its stages, in order (${stages.join(', ')})`);
    check(res6.report.bins > 5000 && res6.report.bins <= 6000 && !res6.report.sampled, `every usable bin clustered (${res6.report.bins})`);
    check(res6.report.pcaDims >= 2 && res6.report.explained >= 0.7, `PCA keeps at least 70% of the variance (${res6.report.pcaDims} components, ${(100 * res6.report.explained).toFixed(0)}%)`);
    check(res6.clusters.length >= 4 && res6.clusters.length <= 6, `about the planted number of clusters (${res6.clusters.length})`);
    check(clusterFloor(30000, 60) === 60 && clusterFloor(600, 60) === 30 && clusterFloor(100, 33) === 10, 'a cluster is never smaller than UMAP\'s clumps, on all but very few bins');
    check(res6.clusters.every((c, i) => i === c.state && (i === 0 || res6.clusters[i - 1].bins >= c.bins)), 'numbered by size, largest first');
    check(res6.clusters.every((c) => c.top.length === 5 && c.top.every((f) => typeof f.name === 'string' && Number.isFinite(f.z))), 'each with the features furthest from their mean');
    check(res6.clusters.every((c) => c.repeat >= 0.5), `and each repeats in a second layout (${res6.clusters.map((c) => c.repeat.toFixed(2)).join(', ')})`);
    check(res6.report.heldOutAgreement > 0.95, `the forests repeat the clusters on bins they never saw (${(100 * res6.report.heldOutAgreement).toFixed(1)}%)`);
    const kept = storeDiscovery(res6, 'Discovery');
    check(kept.sessions === 1 && kept.runs > 0, `a layer on the session (${kept.runs} runs)`);
    const layer = stateLayer(world.session, 'Discovery');
    check(layer && layer.source === 'discovery', 'its source is discovery');
    // Every planted state mostly falls in its own big cluster, on every frame.
    const conf = new Map();
    let none = 0;
    for (let a = 0; a < 2; a++) {
        for (let f = 0; f < T6; f++) {
            const k = stateAt(layer, 'id_' + a, f);
            if (k < 0) { none++; continue; }
            const key = world.truth[a][f] + '>' + k;
            conf.set(key, (conf.get(key) || 0) + 1);
        }
    }
    const own = new Set();
    STATES.forEach((nm, i) => {
        const hits = [...conf.entries()].filter(([key]) => key.startsWith(i + '>')).sort((x, y) => y[1] - x[1]);
        const total = hits.reduce((s, h) => s + h[1], 0);
        const best = hits[0];
        own.add(best[0].split('>')[1]);
        check(best[1] / total > 0.9, `${nm}: ${(100 * best[1] / total).toFixed(1)}% of its frames in one cluster (${best[0].split('>')[1]})`);
    });
    check(own.size === 4, 'and each state has its own');
    check(none < 0.08 * 2 * T6, `nearly every frame has a state (${none} of ${2 * T6} do not)`);
    // Frameshift: where the planted state changes, the labels change within a bin or so.
    let found = 0, changes = 0;
    const big = new Set([...own].map(Number));
    for (let a = 0; a < 2; a++) {
        for (let f = 30; f < T6 - 30; f++) {
            if (world.truth[a][f] === world.truth[a][f - 1]) continue;
            changes++;
            const before = stateAt(layer, 'id_' + a, f - 9), after = stateAt(layer, 'id_' + a, f + 9);
            let at = -1;
            for (let u = f - 8; u <= f + 8; u++) if (stateAt(layer, 'id_' + a, u) !== before) { at = u; break; }
            if (big.has(before) && big.has(after) && before !== after && at >= 0 && Math.abs(at - f) <= 5) found++;
        }
    }
    check(changes > 50 && found / changes > 0.85, `a state change shows at full frame rate, within 5 frames (1.5 bins) of where it was planted, in ${found} of ${changes}`);
    // Deterministic.
    const again = await discoverBehaviors({ sources: [{ session: world.session, name: 'planted', ctx: ctx6 }] });
    check(eq(Array.from(again.map.embedding), Array.from(res6.map.embedding)) && eq(Array.from(again.map.labels), Array.from(res6.map.labels)), 'the same poses give the same layout and clusters');
    check(Object.keys(res6.sources[0].animals).every((a) => eq(Array.from(res6.sources[0].animals[a]), Array.from(again.sources[0].animals[a]))), 'and the same runs on every frame');
    const other = await discoverBehaviors({ sources: [{ session: world.session, name: 'planted', ctx: ctx6 }], maxBins: 2000 });
    check(other.report.bins === 2000 && other.report.sampled, `a smaller sample: exactly ${other.report.bins} bins`);
    check(other.clusters.length >= 4, `still finds the states (${other.clusters.length} clusters)`);
    check(MAX_BINS === 30000, 'at most 30,000 bins by default');
}

// ---------------------------------------------------------------------------
console.log('\n-- 7. Keeping a result --');
{
    const r = discoveryResult(world.session, 'Discovery');
    check(r && r.map.embedding.length === 2 * res6.report.bins && r.map.frame.length === res6.report.bins, 'the map is in memory: a point per bin, with its frame');
    check(r.map.animal.every((a) => a === 0 || a === 1) && r.map.labels.length === res6.report.bins, 'and its animal and cluster');
    const saved = exportStateLayers(world.session);
    check(saved.layers.length === 1 && saved.layers[0].source === 'discovery' && saved.layers[0].name === 'Discovery', 'what is saved is the layer');
    check(JSON.stringify(saved).indexOf('embedding') < 0, 'and not the map');
    const fresh = new Session([], world.session.skeleton, [], 'reopened');
    importStateLayers(fresh, JSON.parse(JSON.stringify(saved)));
    const L2 = stateLayer(fresh, 'Discovery');
    check(L2 && L2.source === 'discovery' && eq(Array.from(L2.animals.get('id_1')), Array.from(stateLayer(world.session, 'Discovery').animals.get('id_1'))), 'and it reads back exactly');
    check(discoveryResult(fresh, 'Discovery') === null, 'a reopened session has no map');
    check(typeof stateColor(0) === 'string', 'state 0 takes the first state color');
    resetDiscoveryResults();
    check(discoveryResult(world.session, 'Discovery') === null, 'a load forgets the maps');
}

// ---------------------------------------------------------------------------
console.log('\n-- 8. Split a behavior --');
{
    resetEthogram();
    const b = addBehavior({ name: 'active' }).behavior;
    // Half the time resting, then bouts labelled "active" that are either walking or grooming.
    const T = 9000;
    const w = plantedWorld('split', T, 31, 2, (tr) => {
        for (let f = 0; f < T; f++) {
            const seg = Math.floor(f / 150);
            tr[f] = seg % 2 === 0 ? 0 : (seg % 4 === 1 ? 1 : 3);
        }
    });
    for (let a = 0; a < 2; a++) {
        for (let seg = 1; seg * 150 < T; seg += 2) labelRange(w.session, b.id, 'id_' + a, '', seg * 150, Math.min(T - 1, seg * 150 + 149), true);
    }
    const ctx = contextOf(w.session, T);
    const res = await discoverBehaviors({ sources: [{ session: w.session, name: 'split', ctx }], split: b.id });
    check(!res.error, 'it runs' + (res.error ? ': ' + res.error : ''));
    check(res.report.bins < 0.6 * T / 3 * 2 && res.report.bins > 1000, `only the labelled frames' bins are clustered (${res.report.bins})`);
    check(res.clusters.length >= 2, `into candidate sub-behaviors (${res.clusters.length})`);
    storeDiscovery(res, 'Split: active');
    const layer = stateLayer(w.session, 'Split: active');
    let outside = 0, inside = 0, missing = 0;
    const conf = new Map();
    for (let f = 0; f < T; f++) {
        const k = stateAt(layer, 'id_0', f);
        const labelled = w.truth[0][f] !== 0;
        if (!labelled && k >= 0) outside++;
        if (labelled) { if (k >= 0) { inside++; const key = w.truth[0][f] + '>' + k; conf.set(key, (conf.get(key) || 0) + 1); } else missing++; }
    }
    check(outside === 0, 'no state outside the behavior\'s frames');
    check(inside > 0.9 * (inside + missing), `and one on its frames (${inside} of ${inside + missing})`);
    const walk = [...conf.entries()].filter(([k]) => k.startsWith('1>')).sort((x, y) => y[1] - x[1]);
    const groom = [...conf.entries()].filter(([k]) => k.startsWith('3>')).sort((x, y) => y[1] - x[1]);
    const wTotal = walk.reduce((s, h) => s + h[1], 0), gTotal = groom.reduce((s, h) => s + h[1], 0);
    check(walk[0][0].split('>')[1] !== groom[0][0].split('>')[1] && walk[0][1] / wTotal > 0.8 && groom[0][1] / gTotal > 0.8,
        `walking and grooming, labelled alike, come apart (${(100 * walk[0][1] / wTotal).toFixed(0)}%, ${(100 * groom[0][1] / gTotal).toFixed(0)}%)`);
    const none = await discoverBehaviors({ sources: [{ session: world.session, name: 'planted', ctx: ctx6 }], split: 'b999' });
    check(/No labelled frames/.test(none.error), 'a behavior with no labelled frames says so');
}

// ---------------------------------------------------------------------------
console.log('\n-- 9. Several sessions --');
{
    const a = plantedWorld('sess_a', 4500, 41, 2), b = plantedWorld('sess_b', 4500, 42, 2);
    const stranger = new Session([], new Skeleton('x', ['p', 'q', 'r'], [[0, 1], [1, 2]]), [], 'sess_c');
    stranger.addIdentity('id_0');
    for (let f = 0; f < 300; f++) {
        const g = new InstanceGroup(f + 1, stranger.identities[0].id);
        g.points3d = new Float64Array([f, 0, 10, f + 20, 0, 12, f + 40, 0, 8]);
        stranger.instanceGroups.set(f, [g]);
    }
    const res = await discoverBehaviors({ sources: [
        { session: a.session, name: 'sess_a', getCtx: () => contextOf(a.session, 4500) },
        { session: b.session, name: 'sess_b', getCtx: () => contextOf(b.session, 4500) },
        { session: stranger, name: 'sess_c', getCtx: () => contextOf(stranger, 300) },
    ] });
    check(!res.error, 'it runs' + (res.error ? ': ' + res.error : ''));
    check(res.sources.length === 2, 'on the sessions that measure the same features');
    check(res.leftOut.length === 1 && res.leftOut[0].session === 'sess_c' && /different body/.test(res.leftOut[0].reason), 'another body is named as left out');
    const kept = storeDiscovery(res, 'Discovery');
    check(kept.sessions === 2, 'a layer per session');
    check(new Set(Array.from(res.map.session)).size === 2, 'the sample crosses the sessions');
    // One numbering: state k means the same thing in both.
    const la = stateLayer(a.session, 'Discovery'), lb = stateLayer(b.session, 'Discovery');
    const best = (w, layer) => {
        const out = {};
        STATES.forEach((_, i) => {
            const c = new Map();
            for (let f = 0; f < 4500; f++) if (w.truth[0][f] === i) { const k = stateAt(layer, 'id_0', f); c.set(k, (c.get(k) || 0) + 1); }
            out[i] = [...c.entries()].sort((x, y) => y[1] - x[1])[0][0];
        });
        return out;
    };
    check(eq(best(a, la), best(b, lb)), `one numbering across sessions (${JSON.stringify(best(a, la))})`);
}

// ---------------------------------------------------------------------------
console.log('\n-- 10. Too little, one blob, cancel --');
{
    const w = plantedWorld('short', 90, 51, 1);
    const res = await discoverBehaviors({ sources: [{ session: w.session, name: 'short', ctx: contextOf(w.session, 90) }] });
    check(/Too little to cluster/.test(res.error), 'too few bins says so: ' + res.error);
    const noPoses = new Session([], new Skeleton('m', NODES, EDGES), [], 'empty');
    const res2 = await discoverBehaviors({ sources: [{ session: noPoses, name: 'empty', ctx: contextOf(noPoses, 100) }] });
    check(/No session with identified poses/.test(res2.error) && res2.leftOut.length === 1, 'no poses says so');
    const still = plantedWorld('still', 3000, 52, 1, (tr) => tr.fill(0));
    const res3 = await discoverBehaviors({ sources: [{ session: still.session, name: 'still', ctx: contextOf(still.session, 3000) }] });
    check(/nothing to tell apart/.test(res3.error || ''), `one behavior throughout: nothing to tell apart (${res3.error || res3.clusters.length + ' clusters'})`);
    let polls = 0;
    const res4 = await discoverBehaviors({ sources: [{ session: world.session, name: 'planted', ctx: ctx6 }], isCancelled: () => ++polls > 20 });
    check(res4.cancelled === true, 'cancelling stops it');
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
