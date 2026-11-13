/**
 * test-tail-motion.mjs — tail motion from video (pose/tail-motion.js) and the
 * feature engine's channels it feeds (pose/behavior-features.js `spec.channels`),
 * away from the app.
 *
 *   1. the nodes: the tail chain from the edges (and by distance where the
 *      edges skip), the neck as the body axis, body edges without tail nodes;
 *   2. the crop track: gaps filled, angles unwrapped, Gaussian smoothing as
 *      scipy's gaussian_filter1d (nearest edges); the body axis turns UP in the
 *      crop and the tail base lands on its centre; forward ∘ inverse = identity;
 *   3. bands: a capsule's pixels (OpenCV's thick line), no pixel twice, missing
 *      ends skipped; bilinear luma with a 0 border;
 *   4. the pass on drawn frames: a tail that shakes gives high tail flicker and
 *      a still one ~0, the still body stays ~0, another animal moving over the
 *      band is left out, frames 0-1 are NaN;
 *   5. channels: log1p of the four and tail minus body flicker;
 *   6. the feature engine with channels: the catalog keeps every old feature in
 *      place and adds 13 per channel; values and window statistics; chunks =
 *      one pass; no channel data = NaN; skipped frames blanked;
 *   7. the store and contexts: channels follow animal NAMES, NaN for an animal
 *      without values, none for another frame count; the generation moves;
 *   8. a classifier learns a behavior only the channels show (cross-validated
 *      F1 well above the same classifier without them), and a model that uses
 *      them says how to get them where they are missing.
 *
 * ESM, picked up by `tests/run-mjs-tests.mjs`.
 */

import {
    tailMotionNodes, fillGaps, unwrapAngles, gaussianSmooth, cropTrack, cropSide, cropAffine, capsulePixels, sampleLuma,
    createTailMotionPass, tailMotionChannelValues, isTailMotionFeature, setTailMotion, tailMotionOf, clearTailMotion,
    tailMotionGeneration, withTailMotionChannels, medianOf, TAIL_MOTION_CHANNELS, TAIL_MOTION_RAW,
} from '../pose/tail-motion.js';
import { featureCatalog, catalogFor, computeFeatureChunk, planChunks, sliceChunkData, sliceChunkChannels, featurePadFrames } from '../pose/behavior-features.js';
import { Session, Skeleton, InstanceGroup } from '../pose/pose-data.js';
import { collectBehaviorInput, resolveBody, featureSpec, floorFrame, brokenPoseFrames } from '../pose/behavior-tensor.js';
import { resetEthogram, addBehavior } from '../pose/behavior-ethogram.js';
import { labelRange } from '../pose/behavior-labels.js';
import { trainBehaviorClassifier, predictModels, modelProblems, resetBehaviorModels } from '../pose/behavior-classifier.js';
import { seededRandom } from '../pose/behavior-forest.js';

let passed = 0, failed = 0;
const check = (cond, msg) => {
    if (cond) { passed++; console.log('  ok   ' + msg); }
    else { failed++; console.log('  FAIL ' + msg); }
};
const near = (a, b, tol) => Math.abs(a - b) <= tol;

// ---------------------------------------------------------------------------
console.log('\n1. nodes');
const HF = ['Nose', 'Ear_R', 'Ear_L', 'TTI', 'TailTip', 'Head', 'Trunk', 'Tail_0', 'Tail_1', 'Tail_2', 'Shoulder_left', 'Shoulder_right',
    'Haunch_left', 'Haunch_right', 'Neck'];
const ix = (nm) => HF.indexOf(nm);
const HF_EDGES = [['Nose', 'Head'], ['Head', 'Neck'], ['Neck', 'Trunk'], ['Trunk', 'TTI'], ['TTI', 'Tail_0'], ['Tail_0', 'Tail_1'], ['Tail_1', 'Tail_2'],
    ['Tail_2', 'TailTip'], ['Head', 'Ear_R'], ['Head', 'Ear_L'], ['Neck', 'Shoulder_left'], ['Neck', 'Shoulder_right'], ['TTI', 'Haunch_left'], ['TTI', 'Haunch_right']]
    .map(([a, b]) => [ix(a), ix(b)]);
const roles = { head: ix('Nose'), tailBase: ix('TTI') };
const tails = ['TailTip', 'Tail_0', 'Tail_1', 'Tail_2'].map(ix);
{
    const nd = tailMotionNodes(HF, HF_EDGES, roles, tails);
    check(nd && nd.chain.map((k) => HF[k]).join('>') === 'TTI>Tail_0>Tail_1>Tail_2>TailTip', 'the tail chain follows the edges: ' + (nd && nd.chain.map((k) => HF[k]).join(' > ')));
    check(nd.axis === ix('Neck') && nd.head === ix('Nose') && nd.tip === ix('TailTip'), 'the body axis is the neck (not the head, which turns); the tip is the chain\'s end');
    check(nd.bodyEdges.every(([a, b]) => tails.indexOf(a) < 0 && tails.indexOf(b) < 0) && nd.bodyEdges.length === HF_EDGES.length - 4,
        'body edges: every edge without a tail node past the base (' + nd.bodyEdges.length + ')');
    // edges that jump straight to the tip: the rest ordered by distance from the base
    const skip = HF_EDGES.filter(([a, b]) => !(tails.indexOf(a) >= 0 || tails.indexOf(b) >= 0)).concat([[ix('TTI'), ix('TailTip')]]);
    const d = { [ix('Tail_0')]: 10, [ix('Tail_1')]: 20, [ix('Tail_2')]: 30, [ix('TailTip')]: 40 };
    const nd2 = tailMotionNodes(HF, skip, roles, tails, (k) => d[k]);
    check(nd2.chain.map((k) => HF[k]).join('>') === 'TTI>Tail_0>Tail_1>Tail_2>TailTip', 'edges that skip: ordered by distance from the base');
    const noNeck = HF.map((nm) => nm === 'Neck' ? 'Nape' : nm);
    check(tailMotionNodes(noNeck, HF_EDGES, roles, tails).axis === ix('Nose'), 'no neck node: the head gives the axis');
    check(tailMotionNodes(HF, HF_EDGES, { head: 0, tailBase: -1 }, tails) === null && tailMotionNodes(HF, HF_EDGES, roles, []) === null,
        'no tail base, or nothing past it: null');
}

// ---------------------------------------------------------------------------
console.log('\n2. crop track');
{
    const x = Float64Array.from([NaN, 1, NaN, NaN, 4, NaN]);
    fillGaps(x);
    check(Array.from(x).join(',') === '1,1,2,3,4,4', 'gaps filled linearly, ends held: ' + Array.from(x).join(','));
    const a = Float64Array.from([3.0, -3.0, NaN, 3.1, -3.1]);
    unwrapAngles(a);
    check(near(a[1] - a[0], 2 * Math.PI - 6, 1e-12) && Number.isNaN(a[2]) && Math.abs(a[4] - a[3]) < 0.2, 'angles unwrapped across ±π (NaN skipped): ' + Array.from(a).map((v) => v.toFixed(2)).join(','));
    // Gaussian against brute force (nearest edges, radius 4 sigma)
    const rand = seededRandom(3);
    const v = Float64Array.from({ length: 50 }, () => rand());
    const g = gaussianSmooth(v, 2);
    const r = 8, w = []; let sw = 0;
    for (let k = -r; k <= r; k++) { w.push(Math.exp(-0.5 * k * k / 4)); sw += w[w.length - 1]; }
    let maxErr = 0;
    for (let i = 0; i < 50; i++) {
        let s = 0;
        for (let k = -r; k <= r; k++) s += w[k + r] / sw * v[Math.min(49, Math.max(0, i + k))];
        maxErr = Math.max(maxErr, Math.abs(s - g[i]));
    }
    check(maxErr < 1e-12, 'Gaussian smoothing = scipy gaussian_filter1d (nearest, truncate 4): max error ' + maxErr.toExponential(1));
    check(gaussianSmooth(Float64Array.from([5, 5, 5, 5]), 2).every((y) => near(y, 5, 1e-12)), 'a constant stays constant');
    // one animal, axis pointing right (+x): its crop turns it up, centred on the tail base
    const N = HF.length, T = 6;
    const P2 = new Float64Array(T * N * 2).fill(NaN);
    for (let t = 0; t < T; t++) {
        P2[(t * N + ix('TTI')) * 2] = 100; P2[(t * N + ix('TTI')) * 2 + 1] = 50;
        P2[(t * N + ix('Neck')) * 2] = 160; P2[(t * N + ix('Neck')) * 2 + 1] = 50;
    }
    const nd = tailMotionNodes(HF, HF_EDGES, roles, tails);
    const tr = cropTrack(P2, T, N, nd, 2);
    check(tr.any && near(tr.ang[3], Math.PI / 2, 1e-9) && near(tr.cx[3], 100, 1e-9), 'axis along +x: angle π/2 (OpenCV convention), centre the tail base');
    const af = cropAffine(tr.cx[3], tr.cy[3], tr.ang[3], 200, 256);
    const fw = (p) => [af.fwd[0] * p[0] + af.fwd[1] * p[1] + af.fwd[2], af.fwd[3] * p[0] + af.fwd[4] * p[1] + af.fwd[5]];
    const bw = (q) => [af.inv[0] * q[0] + af.inv[1] * q[1] + af.inv[2], af.inv[3] * q[0] + af.inv[4] * q[1] + af.inv[5]];
    const c = fw([100, 50]), k = fw([160, 50]);
    check(near(c[0], 128, 1e-9) && near(c[1], 128, 1e-9), 'the tail base lands on the crop centre');
    check(near(k[0], 128, 1e-9) && k[1] < 128 && near(128 - k[1], 60 * 256 / 200, 1e-9), 'the neck lands straight ABOVE it, scaled by crop / side');
    const back = bw(fw([123.4, -7.7]));
    check(near(back[0], 123.4, 1e-9) && near(back[1], -7.7, 1e-9), 'inverse ∘ forward = identity');
    const sz = cropSide([P2], T, N, { base: ix('TTI'), head: ix('Neck'), tip: ix('Neck') });
    check(near(sz.body, 60, 1e-9) && near(sz.side, 2.4 * 60, 1e-9), 'crop side = 2.4 × the larger of the body and tail lengths');
    check(near(medianOf([3, NaN, 1, 2]), 2, 0) && Number.isNaN(medianOf([NaN])), 'median of the finite values');
}

// ---------------------------------------------------------------------------
console.log('\n3. bands and sampling');
{
    const S = 64, mark = new Int32Array(S * S), list = [];
    const pts = Float64Array.from([10, 32, 50, 32]);   // two nodes: a horizontal segment, length 40
    const nAdd = capsulePixels(S, pts, [[0, 1]], 10, mark, 1, list);
    let expect = 0;
    for (let v = 0; v < S; v++) for (let u = 0; u < S; u++) {
        const kk = Math.max(0, Math.min(1, (u - 10) / 40)), du = u - (10 + kk * 40), dv = v - 32;
        if (du * du + dv * dv <= 25) expect++;
    }
    check(nAdd === expect && list.length === expect, 'a capsule: every pixel within width/2 of the segment, once (' + nAdd + ')');
    const again = capsulePixels(S, pts, [[0, 1]], 10, mark, 1, list);
    check(again === 0, 'the same round: no pixel added twice');
    check(capsulePixels(S, Float64Array.from([10, 32, NaN, 32]), [[0, 1]], 10, mark, 2, []) === 0, 'a segment with a missing end: skipped');
    const img = { Y: Uint8Array.from([0, 100, 200, 50]), W: 2, H: 2, stride: 2, off: 0 };
    check(sampleLuma(img, 1, 0) === 100 && near(sampleLuma(img, 0.5, 0.5), 87.5, 1e-9), 'bilinear luma: exact on a pixel, the mean at the centre');
    check(near(sampleLuma(img, -0.5, 0), 0, 1e-9) && near(sampleLuma(img, 1.5, 1), 25, 1e-9), 'outside the frame counts as 0 (OpenCV\'s constant border)');
}

// ---------------------------------------------------------------------------
console.log('\n4. the pass on drawn frames');
// Two "mice" seen from above, axis along +x. Mouse 0's tail shakes sideways (±6 px each frame) on frames 30-59;
// mouse 1 sits still far away. Then mouse 1 is moved onto mouse 0's tail and made to flicker.
function scene(T, W, H, opts) {
    const N = HF.length;
    const P2s = [new Float64Array(T * N * 2).fill(NaN), new Float64Array(T * N * 2).fill(NaN)];
    const frames = [];
    const body0 = { tti: [200, 150], neck: [250, 150], nose: [270, 150] };
    // overlap: mouse 1 lies just beside mouse 0's tail (its skeleton 8 px from the tail line)
    const body1 = opts.overlap ? { tti: [140, 158], neck: [190, 158], nose: [210, 158] } : { tti: [60, 40], neck: [110, 40], nose: [130, 40] };
    for (let t = 0; t < T; t++) {
        const Y = new Uint8Array(W * H);
        const put = (x, y, val) => { x = Math.round(x); y = Math.round(y); if (x >= 0 && y >= 0 && x < W && y < H) Y[y * W + x] = val; };
        const disc = (cx, cy, r, val) => { for (let y = -r; y <= r; y++) for (let x = -r; x <= r; x++) if (x * x + y * y <= r * r) put(cx + x, cy + y, val); };
        const line = (a, b, val) => { const n = Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1])) * 2; for (let i = 0; i <= n; i++) disc(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n, 1, val); };
        const mouse = (a, b, shake) => {
            // body: discs from TTI to nose; tail: 4 segments backwards (−x), bending sideways by `shake`
            for (let k = 0; k <= 10; k++) disc(b.tti[0] + (b.nose[0] - b.tti[0]) * k / 10, b.tti[1] + (b.nose[1] - b.tti[1]) * k / 10, 9, 180);
            const tail = [b.tti, [b.tti[0] - 15, b.tti[1] + shake * 0.3], [b.tti[0] - 30, b.tti[1] + shake * 0.6], [b.tti[0] - 45, b.tti[1] + shake * 0.85], [b.tti[0] - 60, b.tti[1] + shake]];
            for (let s = 0; s < 4; s++) line(tail[s], tail[s + 1], 230);
            const set = (nm, p) => { const o = (t * N + ix(nm)) * 2; P2s[a][o] = p[0]; P2s[a][o + 1] = p[1]; };
            set('TTI', b.tti); set('Neck', b.neck); set('Nose', b.nose); set('Head', [b.nose[0] - 10, b.nose[1]]); set('Trunk', [b.tti[0] + 25, b.tti[1]]);
            set('Tail_0', tail[1]); set('Tail_1', tail[2]); set('Tail_2', tail[3]); set('TailTip', tail[4]);
        };
        const shaking = t >= 30 && t < 60;
        mouse(0, body0, opts.shake0 && shaking ? (t % 2 ? 6 : -6) : 0);
        mouse(1, body1, opts.shake1 && shaking ? (t % 2 ? 6 : -6) : 0);
        if (opts.flicker1 && shaking && t % 2) disc(165, 154, 5, 255);   // mouse 1 flickers on mouse 0's tail band, within its own skeleton's reach
        frames.push({ Y, W, H, stride: W, off: 0 });
    }
    if (opts.hide1) P2s[1].fill(NaN);
    return { P2s, frames, N };
}
function runScene(opts) {
    const T = 80, sc = scene(T, 320, 240, opts);
    const nodes = tailMotionNodes(HF, HF_EDGES, roles, tails);
    const side = cropSide(sc.P2s, T, sc.N, nodes).side;
    const pass = createTailMotionPass({ P2s: sc.P2s, T, N: sc.N, nodes, side });
    sc.frames.forEach((f, t) => pass.push(t, f));
    return pass;
}
const meanOf = (raw, k, from, to) => { let s = 0, n = 0; for (let t = from; t < to; t++) { const v = raw[t * TAIL_MOTION_RAW + k]; if (Number.isFinite(v)) { s += v; n++; } } return n ? s / n : NaN; };
{
    const p = runScene({ shake0: true });
    const shakeF = meanOf(p.raw[0], 1, 32, 60), stillF = meanOf(p.raw[0], 1, 2, 28), bodyF = meanOf(p.raw[0], 3, 32, 60), otherF = meanOf(p.raw[1], 1, 32, 60);
    check(shakeF > 10 && stillF < 0.5, 'a shaking tail: tail flicker ' + shakeF.toFixed(1) + ' vs ' + stillF.toFixed(2) + ' when still');
    check(bodyF < 0.5 * shakeF, 'the body band stays quieter (' + bodyF.toFixed(2) + ')');
    check(otherF < 0.5, 'the still mouse: tail flicker ' + otherF.toFixed(2));
    check(Number.isNaN(p.raw[0][0]) && Number.isNaN(p.raw[0][TAIL_MOTION_RAW + 1]) && Number.isFinite(p.raw[0][2 * TAIL_MOTION_RAW + 1]),
        'frames 0 and 1 are NaN (no second difference yet)');
    check(p.stats.frames === 80 && p.stats.measured >= 2 * 78 - 2, 'every frame pushed, measured from frame 2 on (' + p.stats.measured + ' animal-frames)');
    // another animal flickering over the band is left out
    const withOther = runScene({ overlap: true, flicker1: true });
    const ignored = runScene({ overlap: true, flicker1: true, hide1: true });
    const a = meanOf(withOther.raw[0], 1, 32, 60), b = meanOf(ignored.raw[0], 1, 32, 60);
    check(b > 1 && a < 0.25 * b, 'another animal flickering on the tail band: left out (' + a.toFixed(2) + ' vs ' + b.toFixed(2) + ' when its skeleton is unknown)');
    let threw = false;
    try { withOther.push(5, { Y: new Uint8Array(1), W: 1, H: 1, stride: 1 }); } catch (e) { threw = /increasing order/.test(e.message); }
    check(threw, 'frames out of order are refused');
}

// ---------------------------------------------------------------------------
console.log('\n5. channels');
{
    const raw = Float32Array.from([1, 3, NaN, 1, 0, 0, 0, 0]);
    const ch = tailMotionChannelValues(raw, 2), K = TAIL_MOTION_CHANNELS.length;
    check(near(ch[0], Math.log1p(1), 1e-6) && near(ch[1], Math.log1p(3), 1e-6) && Number.isNaN(ch[2]) && near(ch[4], Math.log1p(3) - Math.log1p(1), 1e-6),
        'log1p of each raw value; tail minus body flicker; NaN stays NaN');
    check(ch.length === 2 * K && near(ch[K + 4], 0, 1e-9), K + ' channels per frame');
    check(isTailMotionFeature('tail flicker') && isTailMotionFeature('tail motion · mean ±0.3 s') && isTailMotionFeature('target · body flicker · max ±1 s') &&
        !isTailMotionFeature('tail base speed') && !isTailMotionFeature('TailTip height'), 'tail-motion features are recognised by name');
}

// ---------------------------------------------------------------------------
console.log('\n6. the feature engine with channels');
const NODES = ['Nose', 'Head', 'Neck', 'Trunk', 'TTI', 'Tail1'];
const EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]];
{
    const base = { mode: '3d', nodeNames: NODES, edges: EDGES, roles: { head: 0, tailBase: 4 }, used: [0, 1, 2, 3, 4], paws: [], fps: 30,
        floor: { origin: [0, 0, 0], up: [0, 0, 1], e1: [1, 0, 0], e2: [0, 1, 0] }, bodyLength: [80] };
    const withCh = Object.assign({}, base, { channels: TAIL_MOTION_CHANNELS });
    const c0 = featureCatalog(base), c1 = featureCatalog(withCh);
    check(c1.animal.length === c0.animal.length + 13 * TAIL_MOTION_CHANNELS.length, 'catalog: 13 features per channel (value + 3 windows × 4 statistics)');
    check(c0.animal.every((f, i) => c1.animal[i].name === f.name), 'every old feature keeps its name AND its place (models trained before are untouched)');
    check(catalogFor(base) !== catalogFor(withCh) && catalogFor(withCh).byName.has('tail flicker · mean ±0.3 s'), 'the cached catalog tells the two specs apart');
    // one animal, T frames, still poses; channel 1 = a ramp
    const T = 200, N = NODES.length, K = TAIL_MOTION_CHANNELS.length;
    const data = new Float32Array(T * N * 3);
    for (let t = 0; t < T; t++) for (let n = 0; n < N; n++) { data[(t * N + n) * 3] = 40 - 20 * n; data[(t * N + n) * 3 + 2] = 20; }
    const chan = new Float32Array(T * K).fill(NaN);
    for (let t = 0; t < T; t++) chan[t * K + 1] = t / 10;
    const names = ['tail flicker', 'tail flicker · mean ±0.3 s', 'tail flicker · max ±1 s', 'body motion'];
    const one = computeFeatureChunk(withCh, { start: 0, end: T, padStart: 0, padEnd: T, data: [data], channels: [chan] }, { animal: names });
    const r = one.animal[0];
    check(near(r['tail flicker'][57], 5.7, 1e-6), 'a channel reads its own value');
    check(near(r['tail flicker · mean ±0.3 s'][100], 10.0, 1e-5) && near(r['tail flicker · max ±1 s'][100], 13.0, 1e-5), 'its window statistics (mean ±0.3 s, max ±1 s)');
    check(r['body motion'].every(Number.isNaN), 'a channel with no values is NaN');
    const none = computeFeatureChunk(withCh, { start: 0, end: T, padStart: 0, padEnd: T, data: [data] }, { animal: names });
    check(none.animal[0]['tail flicker'].every(Number.isNaN), 'no channel data with the chunk: NaN');
    // chunks = one pass
    const pad = featurePadFrames(withCh), chunks = planChunks(T, pad, 64);
    let same = true;
    for (const ch of chunks) {
        const part = computeFeatureChunk(withCh, Object.assign({}, ch, { data: sliceChunkData([data], ch, N * 3), channels: sliceChunkChannels([chan], ch, K) }), { animal: names });
        for (const nm of names) for (let t = ch.start; t < ch.end; t++) {
            const a = part.animal[0][nm][t - ch.start], b = r[nm][t];
            if (!(a === b || (Number.isNaN(a) && Number.isNaN(b)) || Math.abs(a - b) < 1e-5)) same = false;
        }
    }
    check(same, 'computed in padded chunks = one pass (' + chunks.length + ' chunks)');
    const skip = [new Uint8Array(T)]; skip[0][30] = 1;
    const sl = sliceChunkChannels([chan], { start: 0, end: T, padStart: 20, padEnd: 40 }, K, skip);
    check(Number.isNaN(sl[0][10 * K + 1]) && near(sl[0][9 * K + 1], 2.9, 1e-6) && sliceChunkChannels(null, chunks[0], K) === null,
        'sliced channels blank the frames skipped as broken tracking; none without channels');
}

// ---------------------------------------------------------------------------
console.log('\n7. store and contexts');
const FPS = 30;
function wanderWorld(name, T, seed, rattle) {
    const rand = seededRandom(seed);
    const gauss = () => { let u = 0; while (!u) u = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand()); };
    const s = new Session([], new Skeleton('m', NODES, EDGES), [], name);
    s.addIdentity('id_0'); s.addIdentity('id_1');
    const ALONG = [45, 30, 15, 0, -40, -70], H = [15, 25, 30, 30, 15, 5];
    const pos = [[0, 0, 0], [600, 0, 1]];
    let gid = 1;
    for (let t = 0; t < T; t++) {
        s.instanceGroups.set(t, [0, 1].map((k) => {
            const p = pos[k];
            p[2] += (rand() - 0.5) * 0.15;
            p[0] += 3 * Math.cos(p[2]); p[1] += 3 * Math.sin(p[2]);
            if (Math.hypot(p[0] - 600 * k, p[1]) > 400) p[2] = Math.atan2(-p[1], 600 * k - p[0]) + (rand() - 0.5) * 0.5;
            const g = new InstanceGroup(gid++, s.identities[k].id);
            const pts = new Float64Array(NODES.length * 3);
            for (let n = 0; n < NODES.length; n++) {
                pts[3 * n] = p[0] + ALONG[n] * Math.cos(p[2]) + gauss() * 1.5;
                pts[3 * n + 1] = p[1] + ALONG[n] * Math.sin(p[2]) + gauss() * 1.5;
                pts[3 * n + 2] = H[n] + gauss();
            }
            g.points3d = pts;
            return g;
        }));
    }
    // "rattle": bouts the poses do not show; the video's tail flicker does
    const truth = [new Uint8Array(T), new Uint8Array(T)];
    const raw = [new Float32Array(T * TAIL_MOTION_RAW), new Float32Array(T * TAIL_MOTION_RAW)];
    for (let a = 0; a < 2; a++) {
        let t = 60 + Math.floor(rand() * 90);
        while (rattle && t < T - 60) {
            const len = 15 + Math.floor(rand() * 30);
            for (let k = t; k < Math.min(T, t + len); k++) truth[a][k] = 1;
            t += len + 90 + Math.floor(rand() * 240);
        }
        for (let t2 = 0; t2 < T; t2++) {
            const o = t2 * TAIL_MOTION_RAW;
            raw[a][o] = 2 + rand();                                    // tail motion
            raw[a][o + 1] = (truth[a][t2] ? 9 : 2) + 1.5 * rand();     // tail flicker
            raw[a][o + 2] = 1 + rand(); raw[a][o + 3] = 1 + rand();   // body
        }
    }
    return { session: s, truth, raw, T };
}
function contextOf(session, T) {
    const input = collectBehaviorInput(session, { numFrames: T });
    const body = resolveBody(input.nodeNames, input.edges, null);
    const spec = featureSpec(input, body, floorFrame([0, 0, 0], [0, 0, 1]), FPS);
    return { session, input, body, spec, catalog: catalogFor(spec), fps: FPS, broken: brokenPoseFrames(input, body, spec.bodyLength, FPS) };
}
const W = wanderWorld('Rattles', 9000, 5, true);
{
    const g0 = tailMotionGeneration();
    const plain = contextOf(W.session, W.T);
    check(withTailMotionChannels(plain) === plain, 'no tail motion kept: the context is unchanged');
    // the record lists the animals in another order: channels follow the NAMES
    const rec = setTailMotion(W.session, { camera: 'top', numFrames: W.T, animals: ['id_1', 'id_0', 'ghost'], raw: [W.raw[1], W.raw[0], null] });
    check(tailMotionGeneration() > g0 && tailMotionOf(W.session) === rec && rec.version === 1, 'kept in memory; the generation moves');
    const ctx = withTailMotionChannels(plain);
    check(ctx !== plain && ctx.spec.channels.length === TAIL_MOTION_CHANNELS.length && ctx.catalog.byName.has('tail flicker'), 'the context gets the channels and their catalog');
    const K = TAIL_MOTION_CHANNELS.length, a0 = plain.input.animals.indexOf('id_0');
    check(near(ctx.channels[a0][5 * K + 1], Math.log1p(W.raw[0][5 * TAIL_MOTION_RAW + 1]), 1e-6), 'each animal\'s channels by name, not by order');
    check(/^tail-motion\/1\/top\/\d+$/.test(ctx.channelsKey), 'a key saying what was measured: ' + ctx.channelsKey);
    const other = setTailMotion(W.session, { camera: 'top', numFrames: W.T - 1, animals: ['id_0'], raw: [W.raw[0]] });
    check(withTailMotionChannels(plain) === plain && other, 'a record for another frame count is not used');
    setTailMotion(W.session, { camera: 'top', numFrames: W.T, animals: ['id_0'], raw: [W.raw[0]] });
    const only0 = withTailMotionChannels(plain);
    check(only0.channels[plain.input.animals.indexOf('id_1')].every(Number.isNaN), 'an animal the record has no values for: NaN');
    const g1 = tailMotionGeneration();
    clearTailMotion(W.session);
    check(tailMotionOf(W.session) === null && tailMotionGeneration() > g1, 'cleared: gone, the generation moves');
}

// ---------------------------------------------------------------------------
console.log('\n8. a classifier learns what only the channels show');
{
    resetEthogram();
    resetBehaviorModels();
    const rattle = addBehavior({ name: 'tail rattle', key: 'r' }).behavior;
    W.truth.forEach((tr, a) => {
        let s = 0;
        for (let t = 1; t <= 6000; t++) {
            if (t === 6000 || tr[t] !== tr[s]) { labelRange(W.session, rattle.id, 'id_' + a, '', s, t - 1, !!tr[s]); s = t; }
        }
    });
    const plain = contextOf(W.session, W.T);
    const without = await trainBehaviorClassifier({ behaviorId: rattle.id, params: { trees: 30 }, sources: [{ session: W.session, name: W.session.name, ctx: plain }] });
    setTailMotion(W.session, { camera: 'top', numFrames: W.T, animals: ['id_0', 'id_1'], raw: W.raw });
    const ctx = withTailMotionChannels(plain);
    const withCh = await trainBehaviorClassifier({ behaviorId: rattle.id, params: { trees: 30 }, sources: [{ session: W.session, name: W.session.name, ctx }] });
    const f0 = without.report.cv.frame.f1, f1 = withCh.report.cv.frame.f1;
    check(!withCh.error && f1 > 0.85 && f1 > f0 + 0.3, 'cross-validated F1 with the channels ' + f1.toFixed(3) + ', without ' + f0.toFixed(3));
    check(withCh.model.features.some(isTailMotionFeature), 'the forest splits on tail-motion features');
    const pr = await predictModels(ctx, [withCh.model]);
    let tp = 0, fp = 0, fn = 0;
    W.truth.forEach((tr, a) => {
        const sc = pr.scores[rattle.id].get('id_' + a + '\u0001');
        for (let t = 6000; t < W.T; t++) { const p = sc[t] >= 0.5; if (p && tr[t]) tp++; else if (p) fp++; else if (tr[t]) fn++; }
    });
    const held = 2 * tp / Math.max(1, 2 * tp + fp + fn);
    check(held > 0.85, 'frames never labelled: F1 ' + held.toFixed(3));
    const why = modelProblems(withCh.model, plain);
    check(why.length === 1 && /Tail Motion from Video/.test(why[0]), 'on a context without tail motion the model says how to get it: ' + why[0]);
    clearTailMotion(W.session);
}

console.log('\n' + passed + ' passed, ' + failed + ' failed');
if (failed) process.exit(1);
