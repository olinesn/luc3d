// pose/tail-motion.js — pixel motion along each animal's tail in one camera's
// video: the math and the in-memory store (DOM-free; ui/tail-motion.js decodes
// the video and drives it).
//
// ## Why
//
// Tail rattle is a fast side-to-side shake of the tail. On the HardFight
// recordings the 3D tail points place the tail well enough, but their own
// shake says little about rattle: a classifier on the 3D tail features found
// it with F1 0.13 / 0.18 / 0.36 across recordings. The pixels in a band around
// that tail say more (0.44-0.51 / 0.04-0.24 / 0.49-0.57; side-session study
// 2026-11-13, step 2, whose Python this follows).
//
// ## What is measured
//
// Per animal and frame, in a square crop CROP px wide, centred on the tail base
// and turned so the body axis (tail base → neck, else the head) points up, SIDE
// × the larger of the median head–tail base and tail base–tail tip lengths
// across (so the nose and a stretched tail both fit):
//   tail band   within TAIL_BAND/2 crop px of the tail polyline (tail base →
//               each tail node past it → tip) on this frame or the two before
//               (where the tail just was), minus the pixels within OTHER_BAND/2
//               of another animal's skeleton on this frame;
//   body band   the same around the body's edges (no tail node past the base).
//   raw         mean |I_t − I_t−1| ("motion") and mean |I_t − 2 I_t−1 + I_t−2|
//               ("flicker": back-and-forth faster than the frame-to-frame
//               change) over each band, in grey levels (0-255) — NaN where a
//               band has fewer than MIN_BAND_PIXELS pixels or before frame 2.
// I_t−k is frame t−k's own crop (each frame aligned to the body at its time),
// sampled bilinearly from the frame's luma (outside the frame = 0), so the
// body holds still in the crop and what moves is the tail against it. The
// centre and the angle are smoothed over time (Gaussian, sigma
// TAIL_MOTION_SIGMA frames, gaps filled linearly first), so the crop follows
// the body rather than the poses' frame-to-frame jitter.
//
// The classifiers read five channels per animal (TAIL_MOTION_CHANNELS): log1p
// of the four raw values and tail minus body flicker (log1p of each), which
// the feature engine (pose/behavior-features.js `spec.channels`) gives the
// usual window statistics.
//
// ## Store
//
// One record per session (`setTailMotion` / `tailMotionOf`), in memory only:
// never written to the .slp. `tailMotionGeneration()` changes with every set
// or clear, so cached behavior contexts rebuild.
//
// Depends on: ./behavior-features.js (catalogFor).

import { catalogFor } from './behavior-features.js';

/** Bumped whenever what is measured changes. */
export var TAIL_MOTION_VERSION = 1;
/** Crop side in pixels. */
export var TAIL_MOTION_CROP = 256;
/** Crop side in units of the larger of the body and tail lengths. */
export var TAIL_MOTION_SIDE = 2.4;
/** Smoothing of the crop's centre and angle, in frames (Gaussian sigma). */
export var TAIL_MOTION_SIGMA = 2;
/** Band widths in crop pixels: around the tail, around the body, and another animal's skeleton (left out). */
export var TAIL_BAND = 20, BODY_BAND = 24, OTHER_BAND = 36;
/** Fewest band pixels a value is measured over. */
export var MIN_BAND_PIXELS = 20;
/** Raw values per animal-frame: tail motion, tail flicker, body motion, body flicker. */
export var TAIL_MOTION_RAW = 4;
/** The classifier channels, in order. */
export var TAIL_MOTION_CHANNELS = [
    { name: 'tail motion', unit: 'log grey' },
    { name: 'tail flicker', unit: 'log grey' },
    { name: 'body motion', unit: 'log grey' },
    { name: 'body flicker', unit: 'log grey' },
    { name: 'tail flicker over body', unit: 'log ratio' },
];

function fin(v) {
    return v === v && v !== Infinity && v !== -Infinity;
}

// ----------------------------------------------------------------------------
// Nodes
// ----------------------------------------------------------------------------

/**
 * The nodes the crop and the bands use.
 * @param {string[]} nodeNames
 * @param {Array<[number, number]>} edges
 * @param {{head: number, tailBase: number}} roles   (pose/behavior-tensor.js `resolveBody`)
 * @param {number[]} tailNodes  the tail nodes past the base (`body.defaultExcluded`)
 * @param {function(number): number} [distFromBase]  a node's typical distance from the
 *        tail base, to order tail nodes the edges do not chain
 * @returns {?{base: number, axis: number, head: number, chain: number[], tip: number,
 *            bodyEdges: Array<[number, number]>, allEdges: Array<[number, number]>}}
 *   null without a tail base, a head, or a node past the base.
 */
export function tailMotionNodes(nodeNames, edges, roles, tailNodes, distFromBase) {
    var base = roles && roles.tailBase != null ? roles.tailBase : -1;
    var head = roles && roles.head != null ? roles.head : -1;
    var tails = (tailNodes || []).filter(function (n) { return n !== base && n !== head && n >= 0 && n < nodeNames.length; });
    if (base < 0 || head < 0 || !tails.length) return null;
    var neck = -1;
    nodeNames.forEach(function (nm, i) { if (neck < 0 && /^neck$/i.test(String(nm || '').trim())) neck = i; });
    var tailSet = new Set(tails);
    // The chain: walk the edges from the base through tail nodes; the rest by distance from the base.
    var chain = [base], seen = new Set([base]), cur = base;
    for (;;) {
        var next = -1;
        for (var i = 0; i < edges.length && next < 0; i++) {
            var e = edges[i], o = e[0] === cur ? e[1] : e[1] === cur ? e[0] : -1;
            if (o >= 0 && tailSet.has(o) && !seen.has(o)) next = o;
        }
        if (next < 0) break;
        chain.push(next); seen.add(next); cur = next;
    }
    var rest = tails.filter(function (n) { return !seen.has(n); });
    if (rest.length) {
        var d = function (n) { var v = distFromBase ? distFromBase(n) : NaN; return fin(v) ? v : Infinity; };
        rest.sort(function (a, b) { return d(a) - d(b) || a - b; });
        if (chain.length === 1) chain = chain.concat(rest);
        else {
            // edges chained some; place the others by distance among them
            var all = chain.slice(1).concat(rest);
            all.sort(function (a, b) { return d(a) - d(b) || chain.indexOf(a) - chain.indexOf(b) || a - b; });
            chain = [base].concat(all);
        }
    }
    var allEdges = edges.filter(function (e) { return e[0] >= 0 && e[1] >= 0 && e[0] < nodeNames.length && e[1] < nodeNames.length && e[0] !== e[1]; })
        .map(function (e) { return [e[0], e[1]]; });
    var bodyEdges = allEdges.filter(function (e) { return !tailSet.has(e[0]) && !tailSet.has(e[1]); });
    return { base: base, axis: neck >= 0 ? neck : head, head: head, chain: chain, tip: chain[chain.length - 1],
             bodyEdges: bodyEdges, allEdges: allEdges };
}

// ----------------------------------------------------------------------------
// Crop track
// ----------------------------------------------------------------------------

/** Fill NaN runs linearly between finite neighbours (ends hold the nearest), in place. */
export function fillGaps(x) {
    var n = x.length, first = -1, last = -1;
    for (var i = 0; i < n; i++) if (fin(x[i])) { if (first < 0) first = i; last = i; }
    if (first < 0) return x;
    for (var a = 0; a < first; a++) x[a] = x[first];
    for (var b = last + 1; b < n; b++) x[b] = x[last];
    var prev = first;
    for (var t = first + 1; t <= last; t++) {
        if (!fin(x[t])) continue;
        if (t - prev > 1) {
            var v0 = x[prev], v1 = x[t];
            for (var k = prev + 1; k < t; k++) x[k] = v0 + (v1 - v0) * (k - prev) / (t - prev);
        }
        prev = t;
    }
    return x;
}

/** Make finite angles (radians) continuous over time (no 2π jumps), in place; NaNs are skipped. */
export function unwrapAngles(x) {
    var prev = NaN, off = 0;
    for (var i = 0; i < x.length; i++) {
        if (!fin(x[i])) continue;
        if (fin(prev)) {
            var d = x[i] + off - prev;
            while (d > Math.PI) { off -= 2 * Math.PI; d -= 2 * Math.PI; }
            while (d < -Math.PI) { off += 2 * Math.PI; d += 2 * Math.PI; }
        }
        x[i] += off;
        prev = x[i];
    }
    return x;
}

/**
 * Gaussian smoothing with the ends held (scipy.ndimage.gaussian_filter1d,
 * mode 'nearest', truncate 4). Finite input only.
 * @returns {Float64Array}
 */
export function gaussianSmooth(x, sigma) {
    var n = x.length, out = new Float64Array(n);
    if (!(sigma > 0)) { out.set(x); return out; }
    var r = Math.floor(4 * sigma + 0.5), w = new Float64Array(2 * r + 1), s = 0;
    for (var k = -r; k <= r; k++) { w[k + r] = Math.exp(-0.5 * k * k / (sigma * sigma)); s += w[k + r]; }
    for (var j = 0; j < w.length; j++) w[j] /= s;
    for (var i = 0; i < n; i++) {
        var acc = 0;
        for (var k2 = -r; k2 <= r; k2++) {
            var t = i + k2;
            acc += w[k2 + r] * x[t < 0 ? 0 : t >= n ? n - 1 : t];
        }
        out[i] = acc;
    }
    return out;
}

/**
 * One animal's crop on every frame: centre = its tail base, angle = the rotation
 * (radians) that turns its body axis up — OpenCV's getRotationMatrix2D
 * convention, atan2(axis_x, −axis_y) in image coordinates — both gap-filled and
 * smoothed. `valid` marks frames where the base and the axis were seen.
 * @param {Float64Array} P2  T × N × 2 image points (NaN = missing)
 * @returns {{cx: Float64Array, cy: Float64Array, ang: Float64Array, valid: Uint8Array, any: boolean}}
 */
export function cropTrack(P2, T, N, nodes, sigma) {
    var cx = new Float64Array(T), cy = new Float64Array(T), ang = new Float64Array(T), valid = new Uint8Array(T);
    var any = false;
    for (var t = 0; t < T; t++) {
        var ob = (t * N + nodes.base) * 2;
        var x = P2[ob], y = P2[ob + 1];
        var ax = NaN, ay = NaN;
        var oa = (t * N + nodes.axis) * 2;
        if (fin(P2[oa]) && fin(P2[oa + 1])) { ax = P2[oa] - x; ay = P2[oa + 1] - y; }
        if (!(fin(ax) && fin(ay)) && nodes.head !== nodes.axis) {
            var oh = (t * N + nodes.head) * 2;
            ax = P2[oh] - x; ay = P2[oh + 1] - y;
        }
        cx[t] = fin(x) && fin(y) ? x : NaN;
        cy[t] = fin(x) && fin(y) ? y : NaN;
        ang[t] = fin(ax) && fin(ay) && (ax !== 0 || ay !== 0) ? Math.atan2(ax, -ay) : NaN;
        if (fin(cx[t]) && fin(ang[t])) { valid[t] = 1; any = true; }
    }
    unwrapAngles(ang);
    fillGaps(cx); fillGaps(cy); fillGaps(ang);
    return { cx: gaussianSmooth(cx, sigma), cy: gaussianSmooth(cy, sigma), ang: gaussianSmooth(ang, sigma), valid: valid, any: any };
}

/** Median of the finite values (NaN when none). */
export function medianOf(values) {
    var v = [];
    for (var i = 0; i < values.length; i++) if (fin(values[i])) v.push(values[i]);
    if (!v.length) return NaN;
    v.sort(function (a, b) { return a - b; });
    var m = v.length >> 1;
    return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

/**
 * The crop's side in image pixels: TAIL_MOTION_SIDE × the larger of the median
 * head–tail base and tail base–tail tip lengths, over every animal and frame
 * (at most `maxSamples` frames each).
 * @param {Float64Array[]} P2s  per animal, T × N × 2
 * @returns {{body: number, tail: number, side: number}}
 */
export function cropSide(P2s, T, N, nodes, maxSamples) {
    var step = Math.max(1, Math.floor(T / Math.max(1, maxSamples || 20000)));
    var body = [], tail = [];
    P2s.forEach(function (P2) {
        for (var t = 0; t < T; t += step) {
            var ob = (t * N + nodes.base) * 2, oh = (t * N + nodes.head) * 2, ot = (t * N + nodes.tip) * 2;
            body.push(Math.hypot(P2[oh] - P2[ob], P2[oh + 1] - P2[ob + 1]));
            tail.push(Math.hypot(P2[ot] - P2[ob], P2[ot + 1] - P2[ob + 1]));
        }
    });
    var b = medianOf(body), tl = medianOf(tail);
    var big = Math.max(fin(b) ? b : 0, fin(tl) ? tl : 0);
    return { body: b, tail: tl, side: big > 0 ? TAIL_MOTION_SIDE * big : NaN };
}

/**
 * Image → crop pixels for one frame, and back: `fwd` = [a, b, c, d, e, f] with
 * u = a x + b y + c, v = d x + e y + f (OpenCV's warpAffine matrix:
 * getRotationMatrix2D((cx, cy), angle, S / side) moved so (cx, cy) lands on
 * (S/2, S/2)); `inv` the same for crop → image.
 */
export function cropAffine(cx, cy, ang, side, S) {
    var s = S / side, al = s * Math.cos(ang), be = s * Math.sin(ang), h = S / 2;
    var fwd = [al, be, h - al * cx - be * cy, -be, al, h + be * cx - al * cy];
    var s2 = s * s;
    // x = cx + (al (u − h) − be (v − h)) / s², y = cy + (be (u − h) + al (v − h)) / s²
    var inv = [al / s2, -be / s2, cx - (al * h - be * h) / s2, be / s2, al / s2, cy - (be * h + al * h) / s2];
    return { fwd: fwd, inv: inv };
}

// ----------------------------------------------------------------------------
// Bands and sampling
// ----------------------------------------------------------------------------

/**
 * Mark the pixels of an S × S grid within `width/2` of any segment (a capsule
 * per segment: OpenCV's thick line), appending each newly marked pixel's index
 * to `list`. `mark[i] === gen` means marked in this round (pass a new `gen`
 * instead of clearing). `pts`: crop coordinates, flat [u, v] per node (NaN =
 * missing; a segment with a missing end is skipped).
 * @returns {number} pixels added
 */
export function capsulePixels(S, pts, segs, width, mark, gen, list) {
    var r = width / 2, r2 = r * r, added = 0;
    for (var i = 0; i < segs.length; i++) {
        var p = segs[i][0], q = segs[i][1];
        var pu = pts[2 * p], pv = pts[2 * p + 1], qu = pts[2 * q], qv = pts[2 * q + 1];
        if (!(fin(pu) && fin(pv) && fin(qu) && fin(qv))) continue;
        var u0 = Math.max(0, Math.ceil(Math.min(pu, qu) - r)), u1 = Math.min(S - 1, Math.floor(Math.max(pu, qu) + r));
        var v0 = Math.max(0, Math.ceil(Math.min(pv, qv) - r)), v1 = Math.min(S - 1, Math.floor(Math.max(pv, qv) + r));
        if (u0 > u1 || v0 > v1) continue;
        var du = qu - pu, dv = qv - pv, L2 = du * du + dv * dv;
        for (var v = v0; v <= v1; v++) {
            for (var u = u0; u <= u1; u++) {
                var idx = v * S + u;
                if (mark[idx] === gen) continue;
                var wu = u - pu, wv = v - pv;
                var k = L2 > 0 ? (wu * du + wv * dv) / L2 : 0;
                if (k < 0) k = 0; else if (k > 1) k = 1;
                var eu = wu - k * du, ev = wv - k * dv;
                if (eu * eu + ev * ev <= r2) { mark[idx] = gen; list.push(idx); added++; }
            }
        }
    }
    return added;
}

/**
 * Bilinear luma at image point (x, y), 0 outside the frame (OpenCV's
 * INTER_LINEAR with a constant 0 border). `img`: {Y, W, H, stride, off}.
 */
export function sampleLuma(img, x, y) {
    var x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
    var W = img.W, H = img.H, Y = img.Y, st = img.stride, off = img.off || 0;
    var v00 = 0, v10 = 0, v01 = 0, v11 = 0;
    var inX0 = x0 >= 0 && x0 < W, inX1 = x0 + 1 >= 0 && x0 + 1 < W;
    if (y0 >= 0 && y0 < H) {
        var r0 = off + y0 * st;
        if (inX0) v00 = Y[r0 + x0];
        if (inX1) v10 = Y[r0 + x0 + 1];
    }
    if (y0 + 1 >= 0 && y0 + 1 < H) {
        var r1 = off + (y0 + 1) * st;
        if (inX0) v01 = Y[r1 + x0];
        if (inX1) v11 = Y[r1 + x0 + 1];
    }
    return (v00 * (1 - fx) + v10 * fx) * (1 - fy) + (v01 * (1 - fx) + v11 * fx) * fy;
}

/** Is crop pixel (u, v) within the capsules of `others` (flat [pu, pv, qu, qv] per segment, r² given)? */
function inOthers(u, v, others, n, r2) {
    for (var i = 0; i < n; i += 4) {
        var pu = others[i], pv = others[i + 1], du = others[i + 2] - pu, dv = others[i + 3] - pv;
        var wu = u - pu, wv = v - pv, L2 = du * du + dv * dv;
        var k = L2 > 0 ? (wu * du + wv * dv) / L2 : 0;
        if (k < 0) k = 0; else if (k > 1) k = 1;
        var eu = wu - k * du, ev = wv - k * dv;
        if (eu * eu + ev * ev <= r2) return true;
    }
    return false;
}

// ----------------------------------------------------------------------------
// The pass
// ----------------------------------------------------------------------------

/**
 * The per-frame computation over a whole video, fed one frame at a time in order.
 *
 *     var pass = createTailMotionPass({P2s, T, N, nodes, side});
 *     for (t = 0; t < T; t++) pass.push(t, luma);   // luma: {Y, W, H, stride, off}
 *     pass.raw   // per animal Float32Array T × TAIL_MOTION_RAW
 *
 * The pass keeps the last three frames' `luma` objects (their pixels are read
 * again for the two frames after), so a caller reusing buffers must keep at
 * least three in rotation. Frames must come in increasing order; a skipped
 * frame (never pushed) leaves NaN on it and on the two after it.
 * @param {{P2s: Float64Array[], T: number, N: number, nodes: Object, side: number,
 *          S?: number, sigma?: number}} o
 */
export function createTailMotionPass(o) {
    var S = o.S || TAIL_MOTION_CROP, T = o.T, N = o.N, nodes = o.nodes, side = o.side;
    var A = o.P2s.length;
    var tracks = o.P2s.map(function (P2) { return cropTrack(P2, T, N, nodes, o.sigma != null ? o.sigma : TAIL_MOTION_SIGMA); });
    var raw = o.P2s.map(function () { return new Float32Array(T * TAIL_MOTION_RAW).fill(NaN); });
    var tailSegs = [];
    for (var c = 1; c < nodes.chain.length; c++) tailSegs.push([nodes.chain[c - 1], nodes.chain[c]]);
    var mark = new Int32Array(S * S), gen = 0;
    var pts = new Float64Array(N * 2), opts2 = new Float64Array(N * 2);
    // per animal, the band pixel lists of the last 3 frames: [tail, body] each
    var history = o.P2s.map(function () { return []; });
    var frames = [];   // last 3 pushed: {t, luma, inv: per animal}
    var others = new Float64Array(Math.max(4, nodes.allEdges.length * 4 * Math.max(0, A - 1)));
    var rO = OTHER_BAND / 2, rO2 = rO * rO;
    var stats = { frames: 0, measured: 0 };

    function toCrop(fwd, P2, t, out) {
        for (var n = 0; n < N; n++) {
            var oi = (t * N + n) * 2, x = P2[oi], y = P2[oi + 1];
            if (fin(x) && fin(y)) { out[2 * n] = fwd[0] * x + fwd[1] * y + fwd[2]; out[2 * n + 1] = fwd[3] * x + fwd[4] * y + fwd[5]; }
            else { out[2 * n] = NaN; out[2 * n + 1] = NaN; }
        }
    }

    function push(t, luma) {
        if (!(t >= 0 && t < T)) return;
        var prevT = frames.length ? frames[frames.length - 1].t : -1;
        if (t <= prevT) throw new Error('tail motion: frames must come in increasing order (' + t + ' after ' + prevT + ')');
        if (prevT >= 0 && t !== prevT + 1) { frames.length = 0; history.forEach(function (h) { h.length = 0; }); }   // a gap: start over
        var cur = { t: t, luma: luma, inv: new Array(A) };
        stats.frames++;
        for (var a = 0; a < A; a++) {
            var tr = tracks[a];
            if (!tr.any || !(side > 0)) { history[a].push(null); if (history[a].length > 3) history[a].shift(); continue; }
            var af = cropAffine(tr.cx[t], tr.cy[t], tr.ang[t], side, S);
            cur.inv[a] = af.inv;
            toCrop(af.fwd, o.P2s[a], t, pts);
            gen++;
            var tailList = [];
            capsulePixels(S, pts, tailSegs, TAIL_BAND, mark, gen, tailList);
            gen++;
            var bodyList = [];
            capsulePixels(S, pts, nodes.bodyEdges, BODY_BAND, mark, gen, bodyList);
            history[a].push([tailList, bodyList]);
            if (history[a].length > 3) history[a].shift();
            if (frames.length < 2 || history[a].length < 3 || !history[a][0] || !history[a][1]) continue;
            var f1 = frames[frames.length - 1], f2 = frames[frames.length - 2];
            if (!f1.inv[a] || !f2.inv[a]) continue;
            // the other animals' skeletons on this frame, in this animal's crop
            var no = 0;
            for (var b = 0; b < A; b++) {
                if (b === a) continue;
                toCrop(af.fwd, o.P2s[b], t, opts2);
                for (var e = 0; e < nodes.allEdges.length; e++) {
                    var p = nodes.allEdges[e][0], q = nodes.allEdges[e][1];
                    var pu = opts2[2 * p], pv = opts2[2 * p + 1], qu = opts2[2 * q], qv = opts2[2 * q + 1];
                    if (!(fin(pu) && fin(pv) && fin(qu) && fin(qv))) continue;
                    if (Math.max(pu, qu) < -rO || Math.min(pu, qu) > S + rO || Math.max(pv, qv) < -rO || Math.min(pv, qv) > S + rO) continue;
                    others[no++] = pu; others[no++] = pv; others[no++] = qu; others[no++] = qv;
                }
            }
            var lumas = [luma, f1.luma, f2.luma], invs = [af.inv, f1.inv[a], f2.inv[a]];
            for (var which = 0; which < 2; which++) {
                gen++;
                var list = [];
                for (var h = 0; h < 3; h++) {
                    var src = history[a][h][which];
                    for (var j = 0; j < src.length; j++) { var ix = src[j]; if (mark[ix] !== gen) { mark[ix] = gen; list.push(ix); } }
                }
                var s1 = 0, s2 = 0, m = 0;
                for (var li = 0; li < list.length; li++) {
                    var idx = list[li], u = idx % S, v = (idx - u) / S;
                    if (no && inOthers(u, v, others, no, rO2)) continue;
                    var c0 = 0, c1 = 0, c2 = 0;
                    for (var k = 0; k < 3; k++) {
                        var iv = invs[k];
                        var val = sampleLuma(lumas[k], iv[0] * u + iv[1] * v + iv[2], iv[3] * u + iv[4] * v + iv[5]);
                        if (k === 0) c0 = val; else if (k === 1) c1 = val; else c2 = val;
                    }
                    s1 += Math.abs(c0 - c1);
                    s2 += Math.abs(c0 - 2 * c1 + c2);
                    m++;
                }
                if (m >= MIN_BAND_PIXELS) {
                    raw[a][t * TAIL_MOTION_RAW + 2 * which] = s1 / m;
                    raw[a][t * TAIL_MOTION_RAW + 2 * which + 1] = s2 / m;
                    if (which === 0) stats.measured++;
                }
            }
        }
        frames.push(cur);
        if (frames.length > 2) frames.shift();
    }

    return { push: push, raw: raw, tracks: tracks, stats: stats, S: S, side: side };
}

// ----------------------------------------------------------------------------
// Channels
// ----------------------------------------------------------------------------

/**
 * One animal's classifier channels from its raw values: T × 5, log1p of tail
 * motion, tail flicker, body motion and body flicker, then tail flicker minus
 * body flicker (both log1p). NaN stays NaN.
 * @param {Float32Array} raw  T × TAIL_MOTION_RAW
 * @returns {Float32Array}
 */
export function tailMotionChannelValues(raw, T) {
    var K = TAIL_MOTION_CHANNELS.length, out = new Float32Array(T * K).fill(NaN);
    for (var t = 0; t < T; t++) {
        var o = t * TAIL_MOTION_RAW;
        for (var k = 0; k < 4; k++) { var x = raw[o + k]; if (fin(x) && x >= 0) out[t * K + k] = Math.log1p(x); }
        var a = out[t * K + 1], b = out[t * K + 3];
        if (fin(a) && fin(b)) out[t * K + 4] = a - b;
    }
    return out;
}

/** Is `name` a feature measured from tail motion (a channel or one of its window statistics)? */
export function isTailMotionFeature(name) {
    var nm = String(name || '').replace(/^target · /, '');
    return TAIL_MOTION_CHANNELS.some(function (c) { return nm === c.name || nm.indexOf(c.name + ' · ') === 0; });
}

// ----------------------------------------------------------------------------
// Store
// ----------------------------------------------------------------------------

var _store = new WeakMap();
var _gen = 0, _stamp = 0;

/**
 * Keep a session's tail motion (in memory only).
 * @param {Object} session
 * @param {{camera: string, numFrames: number, animals: string[], raw: Float32Array[], side?: number,
 *          seconds?: number, framesRead?: number}} rec
 */
export function setTailMotion(session, rec) {
    if (!session || !rec) return null;
    var r = Object.assign({ version: TAIL_MOTION_VERSION }, rec, { stamp: ++_stamp, _channels: new Map() });
    _store.set(session, r);
    _gen++;
    return r;
}

/** A session's tail motion record, or null. */
export function tailMotionOf(session) {
    return (session && _store.get(session)) || null;
}

/** Forget a session's tail motion. */
export function clearTailMotion(session) {
    if (session && _store.delete(session)) _gen++;
}

/** Changes whenever any session's tail motion is set or cleared. */
export function tailMotionGeneration() {
    return _gen;
}

/**
 * A behavior context with the session's tail motion as channels: `spec.channels`
 * (names and units), `channels` (per input animal, T × 5; NaN for an animal
 * the record has no values for), `channelsKey` (what was measured: camera,
 * version, which computation) and the catalog that includes them. The context
 * unchanged when the session has none, or none for this many frames.
 * @param {Object} ctx  ui/behavior-detect.js `buildBehaviorContext`'s
 * @returns {Object}
 */
export function withTailMotionChannels(ctx) {
    var rec = ctx && ctx.session ? tailMotionOf(ctx.session) : null;
    if (!rec || !ctx.input || rec.numFrames !== ctx.input.numFrames) return ctx;
    var T = ctx.input.numFrames, K = TAIL_MOTION_CHANNELS.length;
    var data = ctx.input.animals.map(function (name) {
        var i = rec.animals.indexOf(name);
        if (i < 0 || !rec.raw[i]) return new Float32Array(T * K).fill(NaN);
        if (!rec._channels.has(i)) rec._channels.set(i, tailMotionChannelValues(rec.raw[i], T));
        return rec._channels.get(i);
    });
    var spec = Object.assign({}, ctx.spec, { channels: TAIL_MOTION_CHANNELS.map(function (c) { return { name: c.name, unit: c.unit }; }) });
    return Object.assign({}, ctx, { spec: spec, catalog: catalogFor(spec), channels: data,
        channelsKey: 'tail-motion/' + rec.version + '/' + rec.camera + '/' + rec.stamp });
}
