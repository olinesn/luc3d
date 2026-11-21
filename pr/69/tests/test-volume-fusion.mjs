/**
 * test-volume-fusion.mjs — Triangulate with a volumetric model's 3D (pose/volume-fusion.js through
 * `triangulateAndReproject`'s `options.volume`, and pose/volume3d.js's per-frame lookup).
 *
 * A synthetic 5-node animal at the meeting point of SLAP's 8 real cameras (tests/fixtures/dlt-behind-cameras.json),
 * projected into every camera with 0.5 px noise:
 *   - without `options.volume` the solve is untouched (no `volume` key in the result);
 *   - volume-gated triangulation drops a detection 60 px off that plain DLT keeps, and the node lands back on
 *     the truth;
 *   - a node seen by one camera takes the volume's mean;
 *   - the group is matched to the right volume animal; a volume animal far from every solve is ignored and the
 *     result is bit-identical to the plain solve;
 *   - fusion lands between DLT and the volume, nearer the sharper one;
 *   - the product of Gaussians and the node mapping by name;
 *   - pose/volume3d.js: a parsed file's rows at a frame, mapped to the skeleton's node order (missing node = NaN).
 *
 * Run: node tests/test-volume-fusion.mjs
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
let passed = 0, failed = 0;
const failures = [];
function ok(cond, msg) { if (cond) passed++; else { failed++; failures.push(msg); console.error('  ✗ ' + msg); } }
function group(name) { console.log('\n• ' + name); }

const imp = (p) => import(pathToFileURL(path.join(ROOT, p)).href);
const pd = await imp('pose/pose-data.js');
const core = await imp('pose/triangulation-core.js');
const VF = await imp('pose/volume-fusion.js');
const V3D = await imp('pose/volume3d.js');
const { triangulateAndReproject } = core;

const FIX = JSON.parse(fs.readFileSync(path.join(HERE, 'fixtures', 'dlt-behind-cameras.json'), 'utf8'));
const cams = FIX.cameras.map(c => new pd.Camera(c.name, c.matrix, c.dist, c.rvec, c.tvec, c.size));
const names = cams.map(c => c.name);
const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

// The point nearest every camera's optical axis: the arena's middle.
function arenaCentre() {
    const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], b = [0, 0, 0];
    for (const c of cams) {
        const R = c.rotationMatrix;  // world -> camera
        const t = c.tvec;
        const C = [0, 1, 2].map(i => -(R[0][i] * t[0] + R[1][i] * t[1] + R[2][i] * t[2]));
        const d = [R[2][0], R[2][1], R[2][2]];
        for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
            const m = (i === j ? 1 : 0) - d[i] * d[j];
            A[i][j] += m; b[i] += m * C[j];
        }
    }
    const inv = core.invert3x3(A);
    return [0, 1, 2].map(i => inv[i][0] * b[0] + inv[i][1] * b[1] + inv[i][2] * b[2]);
}
const centre = arenaCentre();
const OFFS = [[0, 0, 0], [25, 0, 0], [-25, 5, 0], [0, 20, 10], [10, -15, 5]];
const TRUTH = OFFS.map(o => [centre[0] + o[0], centre[1] + o[1], centre[2] + o[2]]);
const K = TRUTH.length;

// seeded noise
let seed = 12345;
function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; }
function gauss() { return Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd()); }

function native(X, c) { return cams[c].distortPoint(cams[c].project(X)); }
/** obs[c][k] = [x, y] | null */
function observe(noisePx, edit) {
    const obs = cams.map((_, c) => TRUTH.map(X => {
        const q = native(X, c);
        return [q[0] + noisePx * gauss(), q[1] + noisePx * gauss()];
    }));
    if (edit) edit(obs);
    return obs;
}
function makeGroup(obs) {
    const g = new pd.InstanceGroup(1, 0);
    obs.forEach((pts, c) => {
        if (pts.every(p => p == null)) return;
        const inst = new pd.Instance(pts.map(p => p || [NaN, NaN]), 0, 'predicted', 1);
        const nulled = pts.map((p, k) => p == null ? k : -1).filter(k => k >= 0);
        if (nulled.length) inst.nulledNodes = new Set(nulled);
        g.addInstance(names[c], inst);
    });
    return g;
}
function solve(obs, volume) {
    return triangulateAndReproject(makeGroup(obs), cams, { method: 'dlt', includedCameras: names, volume });
}
const pt = (r, k) => [r.points3d[3 * k], r.points3d[3 * k + 1], r.points3d[3 * k + 2]];
function makeVolume(animals, sd, covScale) {
    const n = animals.length;
    const means = new Float64Array(n * K * 3), covs = new Float64Array(n * K * 6);
    animals.forEach((pts, a) => pts.forEach((X, k) => {
        means.set(X, (a * K + k) * 3);
        covs.set([sd * sd, 0, 0, sd * sd, 0, sd * sd], (a * K + k) * 6);
    }));
    return { n, nNodes: K, means, covs, covScale, gateFloorPx: 12, gateK: 2, fuse: true, matchMaxMm: 30 };
}

group('without options.volume nothing changes');
{
    seed = 1;
    const obs = observe(0.5);
    const r = solve(obs, undefined);
    ok(!('volume' in r), 'a plain solve has no volume key');
    ok(TRUTH.every((X, k) => dist3(pt(r, k), X) < 2), 'the plain solve recovers the synthetic animal');
}

group('volume-gated triangulation drops a detection far from the volume');
{
    seed = 2;
    const obs = observe(0.5, o => { o[3][2] = [o[3][2][0] + 60, o[3][2][1]]; });
    const plain = solve(obs, undefined);
    const vol = makeVolume([TRUTH], 3, 0.05);
    const r = solve(obs, vol);
    const e0 = dist3(pt(plain, 2), TRUTH[2]), e1 = dist3(pt(r, 2), TRUTH[2]);
    ok(r.volume && r.volume.animal === 0, 'matched to the volume animal');
    ok(r.volume && r.volume.gated >= 1, 'the 60 px detection was gated out (gated ' + (r.volume && r.volume.gated) + ')');
    ok(e0 > 3 * e1 && e1 < 2, 'the node is back on the truth: plain ' + e0.toFixed(2) + ' -> ' + e1.toFixed(2));
    ok(r.volume && r.volume.fused === K, 'every node fused (' + (r.volume && r.volume.fused) + ')');
    ok(r.meanError != null, 'reprojections are still scored');
}

group('a node seen by one camera takes the volume mean');
{
    seed = 3;
    const obs = observe(0.5, o => { for (let c = 1; c < o.length; c++) o[c][4] = null; });
    const plain = solve(obs, undefined);
    ok(pt(plain, 4).every(Number.isNaN), 'plain DLT has no point for it');
    const target = [TRUTH[4][0] + 1, TRUTH[4][1] - 1, TRUTH[4][2]];
    const vol = makeVolume([TRUTH.map((X, k) => k === 4 ? target : X)], 3, 0.05);
    const r = solve(obs, vol);
    ok(dist3(pt(r, 4), target) < 1e-9, 'filled with the volume mean');
    ok(r.volume && r.volume.filled === 1, 'counted as filled');
}

group('matching to the frame\'s animals');
{
    seed = 4;
    const obs = observe(0.5);
    const far = TRUTH.map(X => [X[0] + 500, X[1], X[2]]);
    const r = solve(obs, makeVolume([far, TRUTH], 3, 0.05));
    ok(r.volume && r.volume.animal === 1, 'picks the animal at the group (animal ' + (r.volume && r.volume.animal) + ')');
    const plain = solve(obs, undefined);
    const rFar = solve(obs, makeVolume([far], 3, 0.05));
    ok(!('volume' in rFar), 'a volume animal far from the group is not used');
    ok(Array.from(rFar.points3d).every((v, i) => Object.is(v, plain.points3d[i])), 'and the result is bit-identical to the plain solve');
}

group('fusion weighs DLT and the volume by their uncertainty');
{
    seed = 5;
    // two cameras only, 3 px noise: DLT is poor; an exact, sharp volume pulls the node in
    const obs = observe(3, o => { for (let c = 2; c < o.length; c++) for (let k = 0; k < K; k++) o[c][k] = null; });
    const plain = solve(obs, undefined);
    const sharp = solve(obs, makeVolume([TRUTH], 1, 0.01));
    const broad = solve(obs, makeVolume([TRUTH], 30, 1));
    let e0 = 0, eS = 0, eB = 0;
    for (let k = 0; k < K; k++) { e0 += dist3(pt(plain, k), TRUTH[k]); eS += dist3(pt(sharp, k), TRUTH[k]); eB += dist3(pt(broad, k), TRUTH[k]); }
    ok(eS < 0.25 * e0, 'a sharp volume dominates: DLT ' + (e0 / K).toFixed(2) + ' -> ' + (eS / K).toFixed(2));
    ok(eB < e0 + 1e-9 && eB > eS, 'a broad volume moves DLT little: ' + (eB / K).toFixed(2));
}

group('product of Gaussians and node mapping');
{
    const I = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    const out = [0, 0, 0];
    VF.productOfGaussians([0, 0, 0], I, [2, 4, 6], I, out);
    ok(out.every((v, i) => Math.abs(v - [1, 2, 3][i]) < 1e-12), 'equal covariances -> the midpoint');
    VF.productOfGaussians([0, 0, 0], I, [2, 4, 6], I.map(v => v * 1e-6), out);
    ok(dist3(out, [2, 4, 6]) < 1e-4, 'a near-exact second estimate wins');
    const m = VF.mapVolumeNodes(['Tail', 'Nose', 'Ear_L'], ['Nose', 'Ear_L', 'Ear_R', 'Tail']);
    ok(Array.from(m).join() === '1,2,-1,0', 'skeleton -> volume nodes by name (' + Array.from(m).join() + ')');
}

group('pose/volume3d.js: one frame, in the skeleton\'s node order');
{
    const session = { skeleton: { nodes: [{ name: 'Nose' }, { name: 'Ear_L' }, { name: 'Ear_R' }] } };
    const J = 2;   // the file has Ear_L, Nose (no Ear_R)
    const data = {
        nodeNames: ['Ear_L', 'Nose'],
        frameIndices: Int32Array.from([7, 7, 9]),
        means: Float32Array.from([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]),
        covs: new Float32Array(3 * J * 6).fill(1),
        covScale: 0.05,
    };
    V3D.setSessionVolume(session, data);
    const v = V3D.volumeForFrame(session, 7);
    ok(v && v.n === 2 && v.nNodes === 3, 'frame 7: two animals, the skeleton\'s 3 nodes');
    ok(v && v.means[0] === 2 && v.means[3] === 1 && Number.isNaN(v.means[6]), 'Nose / Ear_L mapped, Ear_R missing (NaN)');
    ok(v && v.means[9] === 4 && v.means[12] === 3, 'the second animal too');
    ok(v && Math.abs(v.covScale - 0.05) < 1e-12, 'the file\'s covariance scale');
    ok(V3D.volumeForFrame(session, 8) === null, 'no row at frame 8: null');
    V3D.setVolumeSettings({ enabled: false });
    ok(V3D.volumeForFrame(session, 7) === null, 'setting off: null');
    V3D.setVolumeSettings({ enabled: true });
    const cov = V3D.volumeNodeCoverage(session);
    ok(cov.matched === 2 && cov.missing.join() === 'Ear_R', 'coverage: 2 of 3, Ear_R missing');
}

group('the volfuse study (Python) on real data: HardFight, production 2D, 24 animal-frames');
{
    // tests/fixtures/volume-fusion-fight.json: the study's inputs and outputs (vf_fuse.py's dlt, gate_s2, fuse_s2)
    // for 24 animal-frames of an 8-camera DoublePoke clip. LUCID's DLT is not the study's: it solves in a
    // camera-normalised frame (frame-invariant on purpose, pose/triangulation-core.js), undistorts differently and
    // nulls a node whose last two views still miss the threshold, so plain DLT already differs by ~0.7 mm at the
    // median. What must hold is that LUCID's fusion does what the study's did: it pulls the two implementations
    // together (both lean on the same volume), and where the study moved a node by more than 1 mm LUCID moves it the
    // same way.
    const FX = JSON.parse(fs.readFileSync(path.join(HERE, 'fixtures', 'volume-fusion-fight.json'), 'utf8'));
    const rc = FX.cameras.map(c => new pd.Camera(c.name, c.matrix, c.dist, c.rvec, c.tvec, c.size));
    const rn = rc.map(c => c.name);
    const NN = FX.node_names.length;
    function rsolve(cs, volume) {
        const g = new pd.InstanceGroup(1, 0);
        cs.obs.forEach((pts, c) => {
            if (pts.every(p => p == null)) return;
            const inst = new pd.Instance(pts.map(p => p || [NaN, NaN]), 0, 'predicted', 1);
            const nulled = pts.map((p, k) => p == null ? k : -1).filter(k => k >= 0);
            if (nulled.length) inst.nulledNodes = new Set(nulled);
            g.addInstance(rn[c], inst);
        });
        return triangulateAndReproject(g, rc, { method: 'dlt', includedCameras: rn, reprojErrorThreshold: FX.robust_px, volume });
    }
    function volOf(cs) {
        return { n: 1, nNodes: NN, means: Float64Array.from(cs.mean.flat()), covs: Float64Array.from(cs.cov6.flat()),
                 covScale: FX.cov_scale, gateFloorPx: FX.gate_floor_px, gateK: FX.gate_k, fuse: true, matchMaxMm: 30 };
    }
    const dDlt = [], dFused = [];
    let matched = 0, moved = 0, sameWay = 0;
    for (const cs of FX.cases) {
        const plain = rsolve(cs, undefined), fused = rsolve(cs, volOf(cs));
        if (fused.volume) matched++;
        for (let k = 0; k < NN; k++) {
            const sD = cs.dlt[k], sF = cs.fused[k], lD = pt(plain, k), lF = pt(fused, k);
            if (!sD || !sF || !lD.every(Number.isFinite) || !lF.every(Number.isFinite)) continue;
            dDlt.push(dist3(lD, sD)); dFused.push(dist3(lF, sF));
            const ds = [sF[0] - sD[0], sF[1] - sD[1], sF[2] - sD[2]], dl = [lF[0] - lD[0], lF[1] - lD[1], lF[2] - lD[2]];
            const ns = Math.hypot(...ds), nl = Math.hypot(...dl);
            if (ns > 1) { moved++; if (nl > 0 && (ds[0] * dl[0] + ds[1] * dl[1] + ds[2] * dl[2]) / (ns * nl) > 0.5) sameWay++; }
        }
    }
    const q = (a, p) => { const s = a.slice().sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
    console.log('    LUCID vs the study, ' + dDlt.length + ' nodes: plain DLT median ' + q(dDlt, 0.5).toFixed(3) + ' / p95 ' +
        q(dDlt, 0.95).toFixed(2) + ' mm; fused ' + q(dFused, 0.5).toFixed(3) + ' / ' + q(dFused, 0.95).toFixed(2) +
        ' mm; nodes the study moved > 1 mm: ' + moved + ', LUCID the same way (cos > 0.5): ' + sameWay);
    ok(matched >= 0.9 * FX.cases.length, 'animal-frames matched to their volume: ' + matched + '/' + FX.cases.length);
    ok(dDlt.length > 0.9 * FX.cases.length * NN, 'most nodes compared (' + dDlt.length + ')');
    ok(q(dFused, 0.5) <= q(dDlt, 0.5) && q(dFused, 0.95) <= q(dDlt, 0.95),
        'fusion brings LUCID and the study together (median and p95)');
    ok(moved >= 20 && sameWay >= 0.8 * moved, 'LUCID moves the nodes the study moved, the same way: ' + sameWay + '/' + moved);
}

console.log(`\n${passed} passed, ${failed} failed`);
if (failed) { console.error(failures.join('\n')); process.exit(1); }
