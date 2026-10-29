/**
 * test-behavior-features.mjs — the behavior feature engine and its inputs,
 * away from the UI.
 *
 * pose/behavior-features.js turns poses into named features; the rules (and
 * later the classifiers) read nothing else. Pinned here:
 *   1. the catalog: readable, deterministic names; no heights in 2D; a node
 *      named like a fixed feature does not collide with it;
 *   2. window statistics (mean, std, min, max) equal a brute-force pass, with
 *      missing frames, at every half-width;
 *   3. band power: a sinusoid's power lands in its band, at its mean square;
 *      a band above fps / 2 is NaN;
 *   4. a session computed in padded chunks equals one pass;
 *   5. INVARIANCE: rotating and translating the whole world together with its
 *      floor changes no feature — with an explicit floor and with the fitted
 *      one — on noisy data;
 *   6. NaN masking: a missing node makes exactly the features that need it
 *      missing, and window statistics use what is left;
 *   7. the features mean what they say (a rear's head height, a walking
 *      speed, a turn, a target straight ahead and getting closer);
 *   8. node roles, default exclusions, body lengths and the floor fit (its
 *      tilt from the level bodies, which a tall cage's spread would get wrong).
 *
 * ESM, picked up by `tests/run-mjs-tests.mjs`.
 */

import {
    featureCatalog, windowStat, bandPower, computeFeatureChunk, planChunks, sliceChunkData, featurePadFrames,
    halfFrames, resolveFeatureName, TARGET_PREFIX,
} from '../pose/behavior-features.js';
import {
    guessNodeRoles, defaultExcludedNodes, resolveBody, bodyLengths, fitFloor, floorFrame, featureSpec, floorFromPlane,
} from '../pose/behavior-tensor.js';

let passed = 0, failed = 0;
const check = (cond, msg) => {
    if (cond) { passed++; console.log('  ok   ' + msg); }
    else { failed++; console.log('  FAIL ' + msg); }
};

// Deterministic noise.
let seed = 12345;
const rand = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
const gauss = () => { let u = 0; while (!u) u = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand()); };

const NODES = ['Nose', 'Head', 'Neck', 'Trunk', 'TTI', 'Tail1', 'Tail2'];
const EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]];
// Body layout along the heading (mm) and resting height (mm).
const ALONG = [45, 30, 15, 0, -40, -70, -100];
const HEIGHT = [15, 25, 30, 30, 15, 6, 4];
const FPS = 60;

/**
 * Three animals walking circles on the floor z = 0 (world frame), with 1 mm
 * noise; animal 0 rears on frames 200–259 and its Neck goes missing on
 * frames 400–409.
 */
function makeWorld(T) {
    const N = NODES.length, A = 3;
    const data = [];
    for (let a = 0; a < A; a++) {
        const arr = new Float32Array(T * N * 3).fill(NaN);
        const r = 200 + 80 * a, w = (0.4 + 0.15 * a) * (a === 1 ? -1 : 1), ph = a * 2.1;
        for (let t = 0; t < T; t++) {
            const ang = w * t / FPS + ph;
            const cx = 400 * a + r * Math.cos(ang), cy = r * Math.sin(ang);
            const hd = ang + (w > 0 ? Math.PI / 2 : -Math.PI / 2);
            const rearing = a === 0 && t >= 200 && t < 260;
            for (let n = 0; n < N; n++) {
                if (a === 0 && n === 2 && t >= 400 && t < 410) continue;
                let along = ALONG[n], z = HEIGHT[n];
                if (rearing && n <= 2) { z = [95, 85, 65][n]; along = ALONG[n] * 0.4; }
                const o = (t * N + n) * 3;
                arr[o] = cx + along * Math.cos(hd) + gauss();
                arr[o + 1] = cy + along * Math.sin(hd) + gauss();
                arr[o + 2] = z + gauss() * 0.5;
            }
        }
        data.push(arr);
    }
    return { mode: '3d', dim: 3, camera: null, numFrames: T, nodeNames: NODES.slice(), edges: EDGES.map(e => e.slice()),
             animals: ['id_0', 'id_1', 'id_2'], data, frames: [T, T, T], skipped: 0 };
}

function rotation(ax, ay, az) {
    const cx = Math.cos(ax), sx = Math.sin(ax), cy = Math.cos(ay), sy = Math.sin(ay), cz = Math.cos(az), sz = Math.sin(az);
    const Rx = [[1, 0, 0], [0, cx, -sx], [0, sx, cx]], Ry = [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], Rz = [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]];
    const mul = (P, Q) => P.map((row, i) => [0, 1, 2].map(j => row[0] * Q[0][j] + row[1] * Q[1][j] + row[2] * Q[2][j]));
    return mul(Rz, mul(Ry, Rx));
}
const apply = (R, t, p) => [0, 1, 2].map(i => R[i][0] * p[0] + R[i][1] * p[1] + R[i][2] * p[2] + t[i]);

function transformInput(input, R, tr) {
    const out = Object.assign({}, input, { data: input.data.map(arr => {
        const o = new Float32Array(arr.length);
        for (let i = 0; i < arr.length; i += 3) {
            if (arr[i] !== arr[i]) { o[i] = o[i + 1] = o[i + 2] = NaN; continue; }
            const q = apply(R, tr, [arr[i], arr[i + 1], arr[i + 2]]);
            o[i] = q[0]; o[i + 1] = q[1]; o[i + 2] = q[2];
        }
        return o;
    }) });
    return out;
}

function wholeChunk(spec, input) {
    return { start: 0, end: input.numFrames, padStart: 0, padEnd: input.numFrames, data: input.data };
}

/** Largest disagreement between two feature results, relative to each feature's typical size. */
function compareResults(r1, r2, names, pairNames) {
    let worst = 0, where = '';
    const cmp = (x, y, label) => {
        let scale = 0;
        for (let i = 0; i < x.length; i++) if (x[i] === x[i]) scale = Math.max(scale, Math.abs(x[i]));
        scale = Math.max(scale, 1e-3);
        for (let i = 0; i < x.length; i++) {
            const a = x[i], b = y[i];
            if ((a !== a) !== (b !== b)) { worst = Infinity; where = label + ' NaN@' + i; return; }
            if (a !== a) continue;
            const d = Math.abs(a - b) / scale;
            if (d > worst) { worst = d; where = label + '@' + i + ' ' + a + ' vs ' + b; }
        }
    };
    r1.animal.forEach((res, a) => names.forEach(n => cmp(res[n], r2.animal[a][n], 'a' + a + ' ' + n)));
    Object.keys(r1.pair).forEach(k => pairNames.forEach(n => cmp(r1.pair[k][n], r2.pair[k][n], k + ' ' + n)));
    return { worst, where };
}

// ---------------------------------------------------------------------------
console.log('\n-- 1. catalog --');
{
    const spec = { mode: '3d', nodeNames: NODES, edges: EDGES, roles: { head: 0, tailBase: 4 }, used: [0, 1, 2, 3, 4], paws: [] };
    const cat = featureCatalog(spec);
    const names = cat.animal.map(f => f.name);
    check(names.includes('Nose height') && names.includes('Nose height · mean ±0.3 s') && names.includes('centroid speed · max ±1 s'),
        'readable names: "Nose height", "Nose height · mean ±0.3 s", "centroid speed · max ±1 s"');
    check(names.includes('Nose–TTI distance') && names.includes('angle Nose–Head–Neck') && names.includes('turning speed'),
        'distances, joint angles and turning speed are there');
    check(!names.some(n => /Tail1|Tail2/.test(n)), 'excluded tail nodes take no part');
    check(cat.pair.map(f => f.name).includes('head–tail base distance · mean ±0.1 s') && cat.pair.some(f => f.name === 'target in view'),
        'pair features: head–tail base distance, target in view');
    check(names.includes('centroid speed · power 3–8 Hz') && names.includes('Nose speed · power 1–3 Hz'), 'band power of centroid and head speeds');
    check(JSON.stringify(featureCatalog(spec).animal.map(f => f.name)) === JSON.stringify(names), 'deterministic');
    const cat2d = featureCatalog(Object.assign({}, spec, { mode: '2d' }));
    check(!cat2d.animal.some(f => / height/.test(f.name)) && !cat2d.animal.some(f => f.name === 'vertical speed'), '2D: no heights, no vertical speed');
    const odd = featureCatalog({ mode: '3d', nodeNames: ['centroid', 'tail'], edges: [[0, 1]], roles: { head: 0, tailBase: 1 }, used: [0, 1], paws: [] });
    const heights = odd.animal.filter(f => !f.stat && / height/.test(f.name)).map(f => f.name);
    check(heights.includes('centroid height') && heights.includes('centroid height #2'), 'a node called "centroid" gets a suffix, not a collision');
    const r = resolveFeatureName(cat, TARGET_PREFIX + 'centroid speed', true);
    check(r && r.who === 'target' && resolveFeatureName(cat, 'min node distance', true).who === 'pair' &&
        resolveFeatureName(cat, 'min node distance', false) === null, 'social names: target · …, relational, and none outside a social behavior');
    const pawCat = featureCatalog({ mode: '3d', nodeNames: ['nose', 'l_paw', 'tail_base'], edges: [], roles: { head: 0, tailBase: 2 }, used: [0, 1, 2], paws: [1] });
    check(pawCat.animal.some(f => f.name === 'nose–l_paw distance · power 3–8 Hz'), 'head–paw distance band power');
}

// ---------------------------------------------------------------------------
console.log('\n-- 2. window statistics vs brute force --');
{
    const n = 1500;
    const x = new Float32Array(n);
    for (let i = 0; i < n; i++) x[i] = (i % 97 === 0 || (i > 600 && i < 640)) ? NaN : 50 + 10 * Math.sin(i / 17) + gauss();
    let worst = 0;
    for (const h of [1, 3, 10, 45]) {
        for (const stat of ['mean', 'std', 'min', 'max']) {
            const got = windowStat(x, h, stat);
            for (let t = 0; t < n; t++) {
                const vals = [];
                for (let j = Math.max(0, t - h); j <= Math.min(n - 1, t + h); j++) if (x[j] === x[j]) vals.push(x[j]);
                let want;
                if (!vals.length) want = NaN;
                else if (stat === 'mean') want = vals.reduce((s, v) => s + v, 0) / vals.length;
                else if (stat === 'std') { const m = vals.reduce((s, v) => s + v, 0) / vals.length; want = Math.sqrt(vals.reduce((s, v) => s + (v - m) * (v - m), 0) / vals.length); }
                else if (stat === 'min') want = Math.min(...vals);
                else want = Math.max(...vals);
                if ((want !== want) !== (got[t] !== got[t])) { worst = Infinity; continue; }
                if (want === want) worst = Math.max(worst, Math.abs(got[t] - want));
            }
        }
    }
    check(worst < 1e-4, 'mean / std / min / max at h = 1, 3, 10, 45 match brute force (worst ' + worst.toExponential(2) + ')');
    const allNaN = windowStat(new Float32Array(10).fill(NaN), 2, 'mean');
    check(allNaN.every(v => v !== v), 'a window with no finite value is NaN');
}

// ---------------------------------------------------------------------------
console.log('\n-- 3. band power --');
{
    // On one of the window's own bins (bin 5 of 61 frames at 60 fps, 4.92 Hz),
    // so nothing leaks into the neighbours as the window slides.
    const fps = 60, n = 600, A = 2, hb = halfFrames(0.5, fps), f0 = 5 * fps / (2 * hb + 1);
    const x = new Float32Array(n);
    for (let i = 0; i < n; i++) x[i] = 3 + A * Math.sin(2 * Math.PI * f0 * i / fps);
    const [b1, b2, b3] = bandPower(x, fps, hb);
    const mid = 300;
    check(Math.abs(b2[mid] - A * A / 2) < 1e-3, '4.9 Hz sinusoid: 3–8 Hz power = A²/2 = 2 (' + b2[mid].toFixed(4) + ')');
    check(b1[mid] < 1e-4 && b3[mid] < 1e-4, 'and none in 1–3 Hz (' + b1[mid].toExponential(1) + ') or 8–16 Hz (' + b3[mid].toExponential(1) + ')');
    let drift = 0;
    for (let i = hb; i < n - hb; i++) drift = Math.max(drift, Math.abs(b2[i] - A * A / 2));
    check(drift < 1e-3, 'steady along the signal: the sliding updates do not drift (' + drift.toExponential(1) + ')');
    const off = new Float32Array(n);
    for (let i = 0; i < n; i++) off[i] = A * Math.sin(2 * Math.PI * 5 * i / fps);
    const between = bandPower(off, fps, hb)[1];
    check(Math.abs(between[mid] - A * A / 2) < 0.1, 'between two bins (5 Hz) most of it stays in the band (' + between[mid].toFixed(3) + ')');
    const slow = bandPower(x, 12, halfFrames(0.5, 12));
    check(slow[2].every(v => v !== v), 'at 12 fps the 8–16 Hz band is above fps / 2: NaN');
}

// ---------------------------------------------------------------------------
console.log('\n-- 4. chunked = one pass --');
const world = makeWorld(1200);
const body = resolveBody(NODES, EDGES, null);
{
    const floor = floorFrame([0, 0, 0], [0, 0, 1]);
    const spec = featureSpec(world, body, floor, FPS);
    const cat = featureCatalog(spec);
    const names = cat.animal.filter((f, i) => i % 7 === 0).map(f => f.name).concat(['Nose height', 'centroid speed · mean ±1 s', 'turning speed · max ±0.3 s', 'centroid speed · power 1–3 Hz']);
    const pairNames = cat.pair.filter((f, i) => i % 5 === 0).map(f => f.name);
    const pairs = [[0, 1], [1, 0], [2, 0]];
    const one = computeFeatureChunk(spec, wholeChunk(spec, world), { animal: names, pair: pairNames, pairs });
    const pad = featurePadFrames(spec);
    const merged = { animal: world.data.map(() => ({})), pair: {} };
    for (const ch of planChunks(world.numFrames, pad, 257)) {
        const res = computeFeatureChunk(spec, Object.assign({}, ch, { data: sliceChunkData(world.data, ch, NODES.length * 3) }),
            { animal: names, pair: pairNames, pairs });
        res.animal.forEach((r, a) => names.forEach(nm => {
            if (!merged.animal[a][nm]) merged.animal[a][nm] = new Float32Array(world.numFrames);
            merged.animal[a][nm].set(r[nm], ch.start);
        }));
        Object.keys(res.pair).forEach(k => pairNames.forEach(nm => {
            merged.pair[k] = merged.pair[k] || {};
            if (!merged.pair[k][nm]) merged.pair[k][nm] = new Float32Array(world.numFrames);
            merged.pair[k][nm].set(res.pair[k][nm], ch.start);
        }));
    }
    const c = compareResults(one, merged, names, pairNames);
    check(c.worst < 1e-5, 'chunks of 257 frames (+' + pad + ' padding) give the one-pass values (worst ' + c.worst.toExponential(2) + ' ' + c.where + ')');
}

// ---------------------------------------------------------------------------
console.log('\n-- 5. invariance under world rotation and translation --');
{
    const R = rotation(0.7, -1.1, 2.3), tr = [5000, -3200, 1750];
    const moved = transformInput(world, R, tr);
    const specA = featureSpec(world, body, floorFrame([0, 0, 0], [0, 0, 1]), FPS);
    const specB = featureSpec(moved, body, floorFrame(tr, apply(R, [0, 0, 0], [0, 0, 1])), FPS);
    const cat = featureCatalog(specA);
    const names = cat.animal.map(f => f.name);
    const pairNames = cat.pair.map(f => f.name);
    const pairs = [[0, 1], [1, 2], [2, 0]];
    const ra = computeFeatureChunk(specA, wholeChunk(specA, world), { animal: names, pair: pairNames, pairs });
    const rb = computeFeatureChunk(specB, wholeChunk(specB, moved), { animal: names, pair: pairNames, pairs });
    const c = compareResults(ra, rb, names, pairNames);
    check(c.worst < 2e-3, 'explicit floor moved with the world: all ' + (names.length + pairNames.length) + ' features unchanged (worst ' +
        c.worst.toExponential(2) + ' ' + c.where + ')');
    const bl = specA.bodyLength.map((v, i) => Math.abs(v - specB.bodyLength[i]) / v);
    check(Math.max(...bl) < 1e-5, 'body lengths unchanged');

    const allNodes = NODES.map((n, i) => i);
    const roles = { head: 0, tailBase: 4 };
    const fitA = fitFloor(world, allNodes, roles), fitB = fitFloor(moved, allNodes, roles);
    check(fitA.tiltFrom === 'body axes' && fitB.tiltFrom === 'body axes', 'the floor takes its tilt from the level bodies');
    check(fitA && fitB, 'the floor is fitted in both worlds');
    const upB = R.map(row => row[0] * fitA.up[0] + row[1] * fitA.up[1] + row[2] * fitA.up[2]);
    check(Math.abs(upB[0] * fitB.up[0] + upB[1] * fitB.up[1] + upB[2] * fitB.up[2] - 1) < 1e-6, 'the fitted up direction moves with the world');
    const sA = featureSpec(world, body, fitA, FPS), sB = featureSpec(moved, body, fitB, FPS);
    const fa = computeFeatureChunk(sA, wholeChunk(sA, world), { animal: names, pair: pairNames, pairs });
    const fb = computeFeatureChunk(sB, wholeChunk(sB, moved), { animal: names, pair: pairNames, pairs });
    const c2 = compareResults(fa, fb, names, pairNames);
    check(c2.worst < 2e-3, 'fitted floor: all features unchanged (worst ' + c2.worst.toExponential(2) + ' ' + c2.where + ')');
    // The fitted floor agrees with the true one: up ≈ +z and the nodes' heights ≈ their planted heights.
    check(fitA.up[2] > 0.999, 'the fitted floor is the true floor: up · z = ' + fitA.up[2].toFixed(5));
    const h = fa.animal[1]['Trunk height'];
    let m = 0, k = 0;
    for (let t = 0; t < h.length; t++) if (h[t] === h[t]) { m += h[t]; k++; }
    const trunkMm = m / k * sA.bodyLength[1];
    check(Math.abs(trunkMm - 30) < 6, 'Trunk height on the fitted floor ≈ 30 mm: the lowest nodes (the tail, 4 mm) sit on it (' + trunkMm.toFixed(1) + ')');
    const plane = floorFromPlane({ centroid: [10, 20, 0], normal: [0, 0, -1] }, world, body.used);
    check(plane && plane.up[2] > 0.999, 'a plane normal pointing away from the animals is turned toward them');
}

// ---------------------------------------------------------------------------
console.log('\n-- 6. NaN masking --');
{
    const spec = featureSpec(world, body, floorFrame([0, 0, 0], [0, 0, 1]), FPS);
    const res = computeFeatureChunk(spec, wholeChunk(spec, world),
        { animal: ['Neck height', 'Nose height', 'Neck–Trunk distance', 'Neck height · mean ±0.1 s', 'missing nodes'] }).animal[0];
    const gap = [400, 409];
    let neckNaN = true, others = true;
    for (let t = gap[0]; t <= gap[1]; t++) {
        if (res['Neck height'][t] === res['Neck height'][t] || res['Neck–Trunk distance'][t] === res['Neck–Trunk distance'][t]) neckNaN = false;
        if (res['Nose height'][t] !== res['Nose height'][t]) others = false;
    }
    check(neckNaN, 'Neck features are NaN while the Neck is missing');
    check(others && res['Neck height'][399] === res['Neck height'][399], 'other features, and the Neck elsewhere, are not');
    check(res['missing nodes'][405] === 1 && res['missing nodes'][300] === 0, '"missing nodes" counts it');
    const h = halfFrames(0.1, FPS);
    const t = 405, vals = [];
    for (let j = t - h; j <= t + h; j++) if (res['Neck height'][j] === res['Neck height'][j]) vals.push(res['Neck height'][j]);
    const want = vals.reduce((s, v) => s + v, 0) / vals.length;
    check(Math.abs(res['Neck height · mean ±0.1 s'][t] - want) < 1e-5, 'a window over the gap averages the finite frames it has');
}

// ---------------------------------------------------------------------------
console.log('\n-- 7. the features mean what they say --');
{
    const spec = featureSpec(world, body, floorFrame([0, 0, 0], [0, 0, 1]), FPS);
    const BL = spec.bodyLength;
    check(Math.abs(BL[0] - 85) < 2, 'body length = median Nose–TTI distance ≈ 85 mm (' + BL[0].toFixed(1) + ')');
    const res = computeFeatureChunk(spec, wholeChunk(spec, world), {
        animal: ['Nose height', 'centroid speed · mean ±0.3 s', 'turning speed · mean ±0.3 s'],
        pair: ['target bearing', 'approach speed', 'centroid distance', 'target in view'], pairs: [[0, 1]],
    });
    const nh = res.animal[0]['Nose height'];
    check(Math.abs(nh[230] * BL[0] - 95) < 3 && Math.abs(nh[100] * BL[0] - 15) < 3, 'a rear: Nose height 95 mm vs 15 mm walking');
    // Animal 1: radius 280 mm, angular speed 0.55 rad/s -> 154 mm/s; turning 0.55 rad/s clockwise = -31.5 deg/s.
    const sp = res.animal[1]['centroid speed · mean ±0.3 s'][600] * spec.bodyLength[1];
    check(Math.abs(sp - 154) < 8, 'centroid speed ≈ 154 mm/s (' + sp.toFixed(1) + ')');
    const turn = res.animal[1]['turning speed · mean ±0.3 s'][600];
    check(Math.abs(turn + 31.5) < 3, 'turning speed ≈ −31.5 deg/s, clockwise negative (' + turn.toFixed(1) + ')');
}
{
    // Two animals on a line: A at the origin facing +x, B ahead of it, coming closer.
    const T = 200, N = NODES.length;
    const mk = (fn) => { const arr = new Float32Array(T * N * 3); for (let t = 0; t < T; t++) for (let n = 0; n < N; n++) { const p = fn(t, n); arr.set(p, (t * N + n) * 3); } return arr; };
    const input = { mode: '3d', dim: 3, numFrames: T, nodeNames: NODES, edges: EDGES, animals: ['a', 'b'],
        data: [mk((t, n) => [ALONG[n], 0, HEIGHT[n]]), mk((t, n) => [500 - t + ALONG[n], 0, HEIGHT[n]])] };
    const spec = featureSpec(input, body, floorFrame([0, 0, 0], [0, 0, 1]), 50);
    const res = computeFeatureChunk(spec, wholeChunk(spec, input),
        { animal: [], pair: ['target bearing', 'approach speed', 'target in view', 'relative heading'], pairs: [[0, 1], [1, 0]] });
    const ab = res.pair['0>1'], ba = res.pair['1>0'];
    check(ab['target bearing'][100] < 0.01 && ab['target in view'][100] === 1, 'target straight ahead: bearing 0°, in view');
    check(ba['target bearing'][100] > 179 && ba['target in view'][100] === 0, 'seen from the other one: behind (180°), not in view');
    const appr = ab['approach speed'][100] * spec.bodyLength[0];
    check(Math.abs(appr - 50) < 1, 'approach speed = 50 mm/s while the gap shrinks 1 mm per frame at 50 fps (' + appr.toFixed(2) + ')');
    check(ab['relative heading'][100] < 0.01, 'both face the same way: relative heading 0°');
}

// ---------------------------------------------------------------------------
console.log('\n-- 8. body, floor --');
{
    check(JSON.stringify(guessNodeRoles(NODES, EDGES)) === JSON.stringify({ head: 0, tailBase: 4 }), 'roles: Nose = head, TTI = tail base');
    check(JSON.stringify(defaultExcludedNodes(NODES, EDGES, { head: 0, tailBase: 4 })) === JSON.stringify([5, 6]), 'tail nodes past the base are excluded');
    const sleap = ['snout', 'ear_l', 'ear_r', 'spine', 'tail_base', 'tail_mid', 'tail_tip'];
    const roles = guessNodeRoles(sleap, [[0, 3], [1, 3], [2, 3], [3, 4], [4, 5], [5, 6]]);
    check(roles.head === 0 && roles.tailBase === 4, 'snout / tail_base');
    check(JSON.stringify(defaultExcludedNodes(sleap, [[0, 3], [3, 4], [4, 5], [5, 6]], roles)) === JSON.stringify([5, 6]), 'tail_mid, tail_tip excluded');
    const noTailBase = guessNodeRoles(['head', 'thorax', 'abdomen'], [[0, 1], [1, 2]]);
    check(noTailBase.head === 0 && noTailBase.tailBase === 2, 'no tail-base name: the node farthest from the head');
    const b = resolveBody(NODES, EDGES, { nodeRoles: { head: 'Head' }, exclude: ['Tail2'] });
    check(b.roles.head === 1 && b.roles.tailBase === 4 && JSON.stringify(b.used) === JSON.stringify([0, 1, 2, 3, 4, 5]),
        'saved settings override the guesses by node name');
    const bl = bodyLengths(world, body);
    check(bl.length === 3 && bl.every(v => Math.abs(v - 85) < 2), 'body lengths per animal');
    const flat = { mode: '3d', dim: 3, numFrames: 5, nodeNames: NODES, edges: EDGES, animals: ['x'], data: [new Float32Array(5 * 7 * 3).fill(NaN)] };
    check(fitFloor(flat, [0, 1, 2, 3, 4, 5, 6]) === null, 'no points: no fitted floor');
    // A tall, narrow cage: three animals walking small circles on shelves at
    // 0, 120 and 240 mm over one footprint. The points spread more up than
    // across, so their least-spread direction is horizontal; the bodies stay level.
    const T = 1500, N = NODES.length;
    const shelves = [0, 1, 2].map((a) => {
        const arr = new Float32Array(T * N * 3);
        for (let t = 0; t < T; t++) {
            const ang = (0.6 + 0.2 * a) * t / FPS + a, hd = ang + Math.PI / 2;
            const cx = 20 * a + 30 * Math.cos(ang), cy = 30 * Math.sin(ang);
            for (let n = 0; n < N; n++) {
                const o = (t * N + n) * 3;
                arr[o] = cx + ALONG[n] * Math.cos(hd) + gauss(); arr[o + 1] = cy + ALONG[n] * Math.sin(hd) + gauss();
                arr[o + 2] = 120 * a + HEIGHT[n] + gauss() * 0.5;
            }
        }
        return arr;
    });
    const tall = { mode: '3d', dim: 3, numFrames: T, nodeNames: NODES, edges: EDGES, animals: ['a', 'b', 'c'], data: shelves };
    const cams = { cameraCenters: [[0, 0, 1500], [600, 0, 900], [-600, 0, 900], [0, 600, 900]] };
    const bySpread = fitFloor(tall, [0, 1, 2, 3, 4, 5, 6], cams);
    const byBodies = fitFloor(tall, [0, 1, 2, 3, 4, 5, 6], Object.assign({ head: 0, tailBase: 4 }, cams));
    check(bySpread.tiltFrom === 'spread' && Math.abs(bySpread.up[2]) < 0.5,
        'without node roles the spread alone tilts it wrong in a tall cage (up · z = ' + bySpread.up[2].toFixed(2) + ')');
    check(byBodies.tiltFrom === 'body axes' && byBodies.up[2] > 0.99 && byBodies.confidence === 'ok',
        'with them, the level bodies give the true floor (up · z = ' + byBodies.up[2].toFixed(4) + ')');
    const lowest = byBodies.origin[0] * byBodies.up[0] + byBodies.origin[1] * byBodies.up[1] + byBodies.origin[2] * byBodies.up[2];
    check(Math.abs(lowest) < 20, 'on the bottom shelf, not the ones at 120 or 240 mm (' + lowest.toFixed(1) + ' mm)');
}

console.log('\n' + passed + ' passed, ' + failed + ' failed');
process.exit(failed ? 1 : 0);
