/**
 * test-behavior-forest.mjs — the random forest behind the behavior
 * classifiers (pose/behavior-forest.js), away from the app.
 *
 *   1. codes keep the order of float32 values (negatives, ±0, ±Infinity,
 *      denormals), NaN is code 0, and "code ≤ c" is exactly
 *      "value ≤ codeUpperValue(c)";
 *   2. bins: missing values go to bin 0; at most 255 value bins; few distinct
 *      values get one bin each; many get quantile bins of near-equal counts;
 *   3. a tree learns separable data, and where a value is missing (the
 *      missing direction is learned, and missingness itself can decide);
 *   4. determinism: the same data and seed give the same forest bit for bit,
 *      however the trees are split into jobs; another seed another forest;
 *   5. class balance: a rare class (4%) is still found at the 0.5 decision;
 *   6. the raw forest (thresholds on values) scores exactly as the binned
 *      trees do;
 *   7. files: base64 round trips, the hash follows the forest, a damaged
 *      forest is caught; the classifier feature list (social: relational, the
 *      actor's, the target's body-wide features).
 *
 * ESM, picked up by `tests/run-mjs-tests.mjs`.
 */

import {
    codeOf, codeUpperValue, encodeColumn, binEdges, binColumn, trainTree, trainTrees, treeScratch, scoreBinnedRows,
    finalizeForest, scoreRows, forestHash, typedToBase64, base64ToTyped, forestProblem, classifierFeatureNames,
    featureRefs, seededRandom, MAX_VALUE_BINS, LEAF, DEFAULT_FOREST_PARAMS,
} from '../pose/behavior-forest.js';
import { featureCatalog, TARGET_PREFIX } from '../pose/behavior-features.js';

let passed = 0, failed = 0;
const check = (cond, msg) => {
    if (cond) { passed++; console.log('  ok   ' + msg); }
    else { failed++; console.log('  FAIL ' + msg); }
};

const f32 = (v) => Math.fround(v);

/** A binned matrix from feature-major float32 values. */
function binMatrix(vals, F, R) {
    const codes = new Uint16Array(F * R);
    for (let f = 0; f < F; f++) encodeColumn(vals.subarray(f * R, (f + 1) * R), codes, f * R, 1);
    const hist = new Uint32Array(65536), lut = new Uint8Array(65536);
    const bins = new Uint8Array(F * R), nb = new Uint16Array(F), edges = [];
    for (let f = 0; f < F; f++) {
        const e = binEdges(codes, f * R, R, hist);
        edges.push(e); nb[f] = e.length + 1;
        binColumn(codes, f * R, R, e, bins, f * R, lut);
    }
    return { codes, bins, nb, edges, hist };
}

function f1Of(scores, truth, rows, thr) {
    let tp = 0, fp = 0, fn = 0;
    rows.forEach((r, i) => { const p = scores[i] >= thr; if (p && truth[r]) tp++; else if (p) fp++; else if (truth[r]) fn++; });
    return 2 * tp / Math.max(1, 2 * tp + fp + fn);
}

// ---------------------------------------------------------------------------
console.log('\n-- 1. codes --');
{
    const rand = seededRandom(3);
    const special = [-Infinity, -3.4e38, -1, -1e-30, -1.401298464324817e-45, -0, 0, 1.401298464324817e-45, 1e-30, 0.5, 1, 3.4e38, Infinity];
    const vals = special.concat(Array.from({ length: 4000 }, () => f32((rand() - 0.5) * Math.pow(10, Math.floor(rand() * 12) - 6))));
    const sorted = vals.slice().sort((a, b) => a - b);
    let monotone = true;
    for (let i = 1; i < sorted.length; i++) if (codeOf(sorted[i]) < codeOf(sorted[i - 1])) monotone = false;
    check(monotone, 'codes never decrease as values increase (' + vals.length + ' values, ±Infinity, denormals)');
    check(codeOf(NaN) === 0 && special.every(v => codeOf(v) > 0), 'NaN is code 0, and no value is');
    check(codeOf(-0) === codeOf(0), '−0 and +0 share a code (they compare equal)');
    let exact = true, tried = 0;
    for (let k = 0; k < 3000; k++) {
        const v = vals[Math.floor(rand() * vals.length)];
        const c = 1 + Math.floor(rand() * 65535);
        if ((codeOf(v) <= c) !== (v <= codeUpperValue(c))) { exact = false; console.log('    mismatch', v, c, codeUpperValue(c)); break; }
        tried++;
    }
    // Every value against its own code and its neighbours.
    for (const v of vals) {
        const c = codeOf(v);
        if (!(v <= codeUpperValue(c)) || (c > 1 && v <= codeUpperValue(c - 1))) { exact = false; console.log('    own-code mismatch', v, c); break; }
    }
    check(exact, '"code ≤ c" is exactly "value ≤ codeUpperValue(c)" (' + tried + ' random pairs, and every value at its own code)');
    check(codeUpperValue(codeOf(0)) > 0 && codeUpperValue(codeOf(0) - 1) < 0, 'the edge at zero: +0\'s code bounds above zero, the code below holds only negatives');
}

// ---------------------------------------------------------------------------
console.log('\n-- 2. bins (with missing values) --');
{
    const R = 5000;
    const rand = seededRandom(11);
    const col = new Float32Array(R);
    for (let i = 0; i < R; i++) col[i] = rand() < 0.1 ? NaN : f32(Math.exp(rand() * 6));
    const { bins, nb, edges } = binMatrix(col, 1, R);
    let missingOk = true, counts = new Array(nb[0] + 1).fill(0);
    for (let i = 0; i < R; i++) {
        if ((col[i] !== col[i]) !== (bins[i] === 0)) missingOk = false;
        counts[bins[i]]++;
    }
    check(missingOk, 'missing values (10%) go to bin 0, and only they do');
    check(nb[0] === MAX_VALUE_BINS && edges[0].length === MAX_VALUE_BINS - 1, 'many distinct values: 255 value bins (' + nb[0] + ')');
    const vc = counts.slice(1), present = R - counts[0];
    const worst = Math.max(...vc.map(c => Math.abs(c - present / MAX_VALUE_BINS)));
    const perCode = new Map();
    for (let i = 0; i < R; i++) if (col[i] === col[i]) perCode.set(codeOf(col[i]), (perCode.get(codeOf(col[i])) || 0) + 1);
    const heaviest = Math.max(...perCode.values());
    check(worst <= heaviest, 'quantile bins hold near-equal counts, to within one code\'s values (each ' + (present / MAX_VALUE_BINS).toFixed(1) +
        ' ± ' + worst.toFixed(1) + '; the heaviest code holds ' + heaviest + ')');
    // Values one code apart each: the bins are equal to within one value.
    const fine = new Float32Array(R);
    for (let i = 0; i < R; i++) fine[i] = codeUpperValue(0x8000 + 0x3f00 + i);
    const fb = binMatrix(fine, 1, R);
    const fc = new Array(fb.nb[0] + 1).fill(0);
    for (let i = 0; i < R; i++) fc[fb.bins[i]]++;
    const fw = Math.max(...fc.slice(1).map(c => Math.abs(c - R / MAX_VALUE_BINS)));
    check(fw <= 1, 'with every value its own code, each bin holds ' + (R / MAX_VALUE_BINS).toFixed(1) + ' ± ' + fw.toFixed(1));
    let ordered = true;
    const idx = Array.from({ length: R }, (_, i) => i).filter(i => col[i] === col[i]).sort((a, b) => col[a] - col[b]);
    for (let i = 1; i < idx.length; i++) if (bins[idx[i]] < bins[idx[i - 1]]) ordered = false;
    check(ordered, 'bins follow the values\' order');
    const small = new Float32Array([3, 1, NaN, 2, 3, 1, 1, NaN, 2]);
    const s2 = binMatrix(small, 1, small.length);
    check(s2.nb[0] === 3 && Array.from(s2.bins).join(',') === '3,1,0,2,3,1,1,0,2', 'three distinct values: one bin each (' + Array.from(s2.bins).join(',') + ')');
    const none = binMatrix(new Float32Array([NaN, NaN, NaN]), 1, 3);
    check(none.nb[0] === 1 && none.edges[0].length === 0 && Array.from(none.bins).every(b => b === 0), 'a feature never measured: no value bin, all missing');
    check(none.hist.every(h => h === 0), 'the scratch histogram is left zero');
}

// ---------------------------------------------------------------------------
console.log('\n-- 3. a tree learns, with missing values --');
const R = 3000, F = 12;
const rand3 = seededRandom(5);
const vals = new Float32Array(F * R);
const y = new Uint8Array(R);
for (let r = 0; r < R; r++) {
    for (let f = 0; f < F; f++) vals[f * R + r] = f32(rand3() * 2 - 1);
    // Class: x0 > 0.2, or x3 missing (which happens for a third of class 1 rows).
    y[r] = vals[r] > 0.2 ? 1 : 0;
    if (y[r] && rand3() < 0.33) vals[3 * R + r] = NaN;
    if (!y[r] && rand3() < 0.1) vals[0 * R + r] = NaN;   // x0 missing for some class-0 rows
}
const M = binMatrix(vals, F, R);
const data = { bins: M.bins, y, R, F, nb: M.nb };
const trainRows = Int32Array.from({ length: R }, (_, i) => i).filter(i => i % 4 !== 0);
const testRows = Int32Array.from({ length: R }, (_, i) => i).filter(i => i % 4 === 0);
{
    const tree = trainTree(data, trainRows, 99, { maxDepth: 16, minLeaf: 3 }, treeScratch(R, F));
    check(tree.n > 1 && tree.feat[0] !== LEAF, 'the root splits (' + tree.n + ' nodes)');
    const sc = scoreBinnedRows([tree], data, testRows);
    check(f1Of(sc, y, Array.from(testRows), 0.5) > 0.95, 'held-out F1 > 0.95 on a threshold rule (' + f1Of(sc, y, Array.from(testRows), 0.5).toFixed(3) + ')');
    check(Math.abs(tree.imp.reduce((s, v) => s + v, 0) - 1) < 1e-9 && tree.imp[0] === Math.max(...tree.imp), 'importance sums to 1, x0 the most important');
    let minLeafOk = true;
    const counts = new Int32Array(tree.n);
    for (const r of trainRows) {
        let nd = 0;
        while (tree.feat[nd] !== LEAF) { const v = M.bins[tree.feat[nd] * R + r]; nd = tree.kid[nd] + (v === 0 ? (tree.miss[nd] ? 0 : 1) : (v <= tree.bin[nd] ? 0 : 1)); }
        counts[nd]++;
    }
    for (let i = 0; i < tree.n; i++) if (tree.feat[i] === LEAF && counts[i] > 0 && counts[i] < 3) minLeafOk = false;
    check(minLeafOk, 'no leaf holds fewer than min leaf (3) distinct training rows');
    // x0 missing, rows of class 0: the tree must send them where class 0 is.
    const miss0 = Array.from(testRows).filter(r => vals[r] !== vals[r]);
    const sm = scoreBinnedRows([tree], data, Int32Array.from(miss0));
    check(miss0.length > 10 && sm.every(s => s < 0.5), 'rows missing x0 (all class 0) are scored "not" (' + miss0.length + ' rows)');
}

// ---------------------------------------------------------------------------
console.log('\n-- 4. determinism --');
{
    const params = { maxDepth: 16, minLeaf: 3 };
    const job = (trees) => trainTrees(Object.assign({ folds: [trainRows] }, data), { trees, fold: -1, seed: 7, params, rowsIndex: 0 });
    const whole = job([0, 1, 2, 3, 4, 5]);
    const split = [job([0, 1]), job([2, 3, 4]), job([5])];
    const flat = split.flatMap(s => s.trees);
    const same = (a, b) => a.n === b.n && ['feat', 'bin', 'miss', 'kid', 'prob'].every(k => a[k].every((v, i) => Object.is(v, b[k][i])));
    check(whole.trees.every((t, i) => same(t, flat[i])), 'six trees trained in one job equal the same six trained in three jobs');
    const again = job([0, 1, 2, 3, 4, 5]);
    check(whole.trees.every((t, i) => same(t, again.trees[i])), 'and the same again, bit for bit');
    const other = trainTrees(Object.assign({ folds: [trainRows] }, data), { trees: [0], fold: -1, seed: 8, params, rowsIndex: 0 });
    check(!same(other.trees[0], whole.trees[0]), 'another seed grows another tree');
    const fin1 = finalizeForest(whole.trees, M.edges, Array.from({ length: F }, (_, i) => 'f' + i));
    const fin2 = finalizeForest(flat, M.edges, Array.from({ length: F }, (_, i) => 'f' + i));
    check(forestHash(fin1.forest, fin1.features) === forestHash(fin2.forest, fin2.features), 'and the same forest hash (' + forestHash(fin1.forest, fin1.features) + ')');
}

// ---------------------------------------------------------------------------
console.log('\n-- 5. a rare class --');
{
    const Rr = 6000, Fr = 8;
    const rand = seededRandom(21);
    const v = new Float32Array(Fr * Rr), yy = new Uint8Array(Rr);
    for (let r = 0; r < Rr; r++) {
        for (let f = 0; f < Fr; f++) v[f * Rr + r] = f32(rand());
        yy[r] = v[r] > 0.96 ? 1 : 0;
    }
    const Mr = binMatrix(v, Fr, Rr);
    const d = { bins: Mr.bins, y: yy, R: Rr, F: Fr, nb: Mr.nb, folds: [] };
    const tr = Int32Array.from({ length: Rr }, (_, i) => i).filter(i => i % 3 !== 0);
    const te = Array.from({ length: Rr }, (_, i) => i).filter(i => i % 3 === 0);
    const out = trainTrees(Object.assign(d, { folds: [tr] }), { trees: Array.from({ length: 20 }, (_, i) => i), fold: -1, seed: 1, params: { maxDepth: 16, minLeaf: 3 }, rowsIndex: 0 });
    const sc = scoreBinnedRows(out.trees, d, Int32Array.from(te));
    let tp = 0, fn = 0;
    te.forEach((r, i) => { if (yy[r]) { if (sc[i] >= 0.5) tp++; else fn++; } });
    check(tp / (tp + fn) > 0.9, '4% positives: recall at 0.5 is ' + (tp / (tp + fn)).toFixed(3) + ' (balanced class weights)');
}

// ---------------------------------------------------------------------------
console.log('\n-- 6. raw forest = binned trees --');
let fin;
{
    const out = trainTrees(Object.assign({ folds: [trainRows] }, data), { trees: Array.from({ length: 15 }, (_, i) => i), fold: -1, seed: 3, params: { maxDepth: 16, minLeaf: 3 }, rowsIndex: 0 });
    const names = Array.from({ length: F }, (_, i) => 'f' + i);
    fin = finalizeForest(out.trees, M.edges, names, out.imp);
    check(fin.features.length <= F && fin.features.includes('f0'), 'the raw forest names only the features it splits on (' + fin.features.length + ' of ' + F + ')');
    const Fm = fin.features.length, idx = fin.features.map(nm => +nm.slice(1));
    const X = new Float32Array(testRows.length * Fm);
    testRows.forEach((r, i) => idx.forEach((f, j) => { X[i * Fm + j] = vals[f * R + r]; }));
    const raw = scoreRows(fin.forest, X, Fm, testRows.length);
    const binned = scoreBinnedRows(out.trees, data, testRows);
    let worst = 0;
    for (let i = 0; i < raw.length; i++) worst = Math.max(worst, Math.abs(raw[i] - binned[i]));
    check(worst === 0, 'thresholds on raw values score every held-out row exactly as the binned trees (worst ' + worst + ')');
    check(Math.abs(Array.from(fin.importance).reduce((s, x) => s + x, 0) - 1) < 1e-5, 'the forest\'s importance sums to 1');
}

// ---------------------------------------------------------------------------
console.log('\n-- 7. files and feature lists --');
{
    const parts = ['feat', 'val', 'kid', 'miss', 'roots'];
    const Types = { feat: Uint16Array, val: Float32Array, kid: Int32Array, miss: Uint8Array, roots: Int32Array };
    const back = {};
    parts.forEach(k => { back[k] = base64ToTyped(typedToBase64(fin.forest[k]), Types[k]); });
    check(parts.every(k => back[k].length === fin.forest[k].length && back[k].every((v, i) => Object.is(v, fin.forest[k][i]))), 'every forest array survives base64');
    check(forestHash(back, fin.features) === forestHash(fin.forest, fin.features), 'and hashes the same');
    const tweaked = Object.assign({}, back, { val: back.val.slice() });
    tweaked.val[tweaked.val.length - 1] = 0.123;
    check(forestHash(tweaked, fin.features) !== forestHash(back, fin.features), 'one leaf changed: another hash');
    const why = forestProblem(back, fin.features.length);
    check(why === null, 'a sound forest has no problem' + (why ? ' (' + why + ')' : ''));
    const broken = Object.assign({}, back, { kid: back.kid.slice() });
    const internal = Array.from(broken.feat).findIndex(f => f !== LEAF);
    broken.kid[internal] = broken.kid.length + 5;
    check(/child out of range/.test(forestProblem(broken, fin.features.length) || ''), 'a child index out of range is caught');
    check(base64ToTyped('abc', Uint8Array) === null && base64ToTyped('AAAA', Float32Array) === null, 'malformed base64, or a length that is not whole elements: null');
    check(DEFAULT_FOREST_PARAMS.trees === 100 && DEFAULT_FOREST_PARAMS.maxDepth === 16 && DEFAULT_FOREST_PARAMS.minLeaf === 3, 'defaults: 100 trees, depth 16, min leaf 3');
    // Feature lists.
    const NODES = ['Nose', 'Head', 'Neck', 'Trunk', 'TTI', 'Tail1'];
    const spec = { mode: '3d', nodeNames: NODES, edges: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], roles: { head: 0, tailBase: 4 }, used: [0, 1, 2, 3, 4], paws: [] };
    const cat = featureCatalog(spec);
    const own = classifierFeatureNames(cat, false, spec.roles);
    check(own.length === cat.animal.length, 'an animal behavior reads every animal feature (' + own.length + ')');
    const soc = classifierFeatureNames(cat, true, spec.roles);
    const tgt = soc.filter(nm => nm.indexOf(TARGET_PREFIX) === 0);
    check(soc.length === cat.pair.length + cat.animal.length + tgt.length && tgt.length > 0 && tgt.length < cat.animal.length / 3,
        'a social one: every relational and actor feature, and the target\'s body-wide ones (' + tgt.length + ')');
    check(tgt.includes(TARGET_PREFIX + 'centroid speed') && tgt.includes(TARGET_PREFIX + 'Nose height') && !tgt.some(nm => /Neck height/.test(nm)),
        'the target\'s: centroid speed, head height — not every node');
    const refs = featureRefs(cat, soc, true);
    check(refs && refs.filter(r => r.who === 'target').length === tgt.length && refs.filter(r => r.who === 'pair').length === cat.pair.length,
        'every name resolves (pair, actor, target)');
    check(featureRefs(cat, ['no such feature'], false) === null, 'an unknown name does not');
}

console.log('\n' + (failed ? 'FAIL' : 'PASS') + ' — ' + passed + ' passed, ' + failed + ' failed');
process.exit(failed ? 1 : 0);
