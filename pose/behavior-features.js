// pose/behavior-features.js — the behavior feature engine: what each animal,
// and each actor toward a target, is doing on every frame, as named numbers
// ("Nose height · mean ±0.3 s", "head–tail base distance").
//
// Pure and import-free, so the behavior worker (pose/behavior-worker.js) runs
// exactly this code, and the main thread runs it too (small sessions, "This
// frame", Node tests).
//
// ## Frame of reference
//
// Features never read absolute world coordinates. Every point is first put in
// the FLOOR frame (`spec.floor`: an origin on the floor, the up direction, and
// two horizontal axes e1, e2 with e1 × e2 = up), and every feature is then a
// height above the floor, a distance, an angle, or a difference over time.
// The horizontal axes themselves are arbitrary (pose/behavior-tensor.js builds
// them from `up` alone), so nothing reports a horizontal coordinate or an
// absolute heading — only quantities that do not change when the floor plane
// is rotated: rotating and translating the whole world together with its floor
// leaves every feature unchanged (tests/test-behavior-features.mjs).
//
// Without 3D (`spec.mode === '2d'`) the "floor" is one camera's image plane:
// no heights, and lengths are in pixels before normalization.
//
// ## Units
//
// Lengths are divided by the animal's body length (`spec.bodyLength`, its
// session-median head–tail base distance, in data units), so a threshold in
// body lengths (BL) means the same thing in millimetres, centimetres or pixels,
// for a mouse or a rat. Speeds are BL/s, angles degrees, turning deg/s.
//
// ## What is computed
//
// Per animal and frame (the base features), over the nodes in `spec.used`
// (excluded nodes — tail nodes past the base by default — take no part):
//   <node> height, centroid height              above the floor (3D only)
//   <node> forward / left                       egocentric: from the centroid,
//                                               along / across the heading
//   <a>–<b> distance                            every pair of used nodes
//   angle <a>–<b>–<c>                           at each joint of two skeleton edges
//   centroid speed, vertical speed (3D),        central differences over
//   <node> speed, turning speed                 ±SPEED_HALF_SECONDS
//   elongation, missing nodes
//   nearest animal distance / min node distance / head–head distance
// Per ordered actor → target pair (the relational features):
//   centroid distance, head–head distance, head–tail base distance (actor's
//   head to target's tail base), min node distance, target bearing (|angle| of
//   the target's centroid from the actor's head, 0 = straight ahead) and its
//   sin / cos (signed: left positive), relative heading, approach speed
//   (positive while the distance shrinks), tangential speed, target in view
//   (within ±VIEW_HALF_ANGLE of the actor's heading).
// Every base feature also has its mean, std, min and max over centered windows
// of ±WINDOW_SECONDS ("· mean ±0.3 s"), NaN-masked, in O(T) (prefix sums and
// monotonic deques). The speeds of the centroid and the head and the head–paw
// distances also have their power in BANDS ("· power 3–8 Hz", a sliding DFT
// over ±BAND_HALF_SECONDS; a band above fps/2 is NaN, so the feature list does
// not depend on the frame rate).
//
// Missing data is NaN, never filled: a feature that needs a missing node is
// NaN on that frame, and a window statistic uses the finite values it has.
//
// ## Chunks
//
// `computeFeatureChunk` works on a run of frames padded on both sides by
// `featurePadFrames(spec)`, which covers every window and difference, so a
// session computed chunk by chunk gives exactly what one pass would — and the
// frames × features matrix is never built (the renderer's heap is capped near
// 4 GB). Only the requested features are computed.
//
// Depends on: nothing.

/** Bumped whenever a feature's definition changes (models record it, Phase 3). */
export var FEATURE_SPEC_VERSION = 1;

/** Half-widths of the centered windows, in seconds. */
export var WINDOW_SECONDS = [0.1, 0.3, 1.0];
/** The window statistics, in catalog order. */
export var WINDOW_STATS = ['mean', 'std', 'min', 'max'];
/** Frequency bands (Hz, [low, high)) for band power. */
export var BANDS = [[1, 3], [3, 8], [8, 16]];
/** Speeds are central differences over ± this many seconds (at least one frame). */
export var SPEED_HALF_SECONDS = 0.05;
/** Band power reads a window of ± this many seconds. */
export var BAND_HALF_SECONDS = 0.5;
/** "target in view": within ± this many degrees of the actor's heading. */
export var VIEW_HALF_ANGLE = 60;

/** Separator between a base feature and its statistic. */
export var STAT_SEP = ' · ';
/** Prefix naming the target's own feature in a social behavior ("target · centroid speed"). */
export var TARGET_PREFIX = 'target · ';

var DEG = 180 / Math.PI;

// ----------------------------------------------------------------------------
// Small helpers
// ----------------------------------------------------------------------------

/** A window half-width in frames: round(seconds × fps), at least 1. */
export function halfFrames(seconds, fps) {
    return Math.max(1, Math.round(seconds * (fps > 0 ? fps : 30)));
}

/** How a window reads in a feature name: 0.3 -> "±0.3 s", 1 -> "±1 s". */
export function windowLabel(seconds) {
    return '±' + String(Number(seconds.toFixed(3))) + ' s';
}

/** How a band reads in a feature name: [3, 8] -> "power 3–8 Hz". */
export function bandLabel(band) {
    return 'power ' + band[0] + '–' + band[1] + ' Hz';
}

/** Frames of context a chunk needs on each side (every window and difference). */
export function featurePadFrames(spec) {
    var fps = spec.fps > 0 ? spec.fps : 30;
    var hMax = 0;
    for (var i = 0; i < WINDOW_SECONDS.length; i++) hMax = Math.max(hMax, halfFrames(WINDOW_SECONDS[i], fps));
    return Math.max(hMax, halfFrames(BAND_HALF_SECONDS, fps)) + halfFrames(SPEED_HALF_SECONDS, fps) + 1;
}

function wrapAngle(a) {
    while (a > Math.PI) a -= 2 * Math.PI;
    while (a <= -Math.PI) a += 2 * Math.PI;
    return a;
}

function fin(v) {
    return v === v && v !== Infinity && v !== -Infinity;
}

// ----------------------------------------------------------------------------
// Catalog
// ----------------------------------------------------------------------------

/**
 * Every feature of a spec, with a readable name. Deterministic for a spec, so
 * the main thread and the worker agree on names.
 *
 * @param {Object} spec {mode, nodeNames, edges, roles:{head, tailBase}, used, paws}
 * @returns {{animal: Object[], pair: Object[], byName: Map<string, Object>, bases: Map<string, Object>}}
 *   Each feature `{name, scope: 'animal'|'pair', base, stat: null|'mean'|'std'|'min'|'max'|'power',
 *   window: seconds|null, band: [lo, hi]|null, unit}`; `bases` maps a base name to
 *   `{name, scope, kind, unit, node?, a?, b?, c?, band?}`.
 */
export function featureCatalog(spec) {
    var names = spec.nodeNames || [];
    var used = spec.used || [];
    var is3d = spec.mode !== '2d';
    var roles = spec.roles || {};
    var head = roles.head != null ? roles.head : -1;
    var paws = spec.paws || [];
    var usedSet = new Set(used);
    var bases = new Map();
    var animal = [], pair = [];
    var byName = new Map();

    function addBase(name, scope, kind, unit, extra) {
        var n = name, i = 2;
        while (bases.has(n)) n = name + ' #' + (i++);
        var b = Object.assign({ name: n, scope: scope, kind: kind, unit: unit }, extra || {});
        bases.set(n, b);
        return b;
    }

    // Fixed names first, so a node called "centroid" gets the suffix instead.
    var fixedAnimal = [];
    if (is3d) fixedAnimal.push(['centroid height', 'cheight', 'BL']);
    fixedAnimal.push(['centroid speed', 'cspeed', 'BL/s', { band: true }]);
    if (is3d) fixedAnimal.push(['vertical speed', 'vspeed', 'BL/s']);
    fixedAnimal.push(['turning speed', 'turn', 'deg/s']);
    fixedAnimal.push(['elongation', 'elong', 'BL']);
    fixedAnimal.push(['missing nodes', 'missing', 'count']);
    fixedAnimal.push(['nearest animal distance', 'nearCent', 'BL']);
    fixedAnimal.push(['nearest animal min node distance', 'nearMin', 'BL']);
    fixedAnimal.push(['nearest animal head–head distance', 'nearHH', 'BL']);
    var fixedPair = [
        ['centroid distance', 'pCent', 'BL'],
        ['head–head distance', 'pHH', 'BL'],
        ['head–tail base distance', 'pHT', 'BL'],
        ['min node distance', 'pMin', 'BL'],
        ['target bearing', 'pBear', 'deg'],
        ['target bearing sin', 'pBearSin', ''],
        ['target bearing cos', 'pBearCos', ''],
        ['relative heading', 'pRelHead', 'deg'],
        ['approach speed', 'pApproach', 'BL/s'],
        ['tangential speed', 'pTang', 'BL/s'],
        ['target in view', 'pView', ''],
    ];
    var animalBases = [], pairBases = [];
    fixedAnimal.forEach(function (f) { animalBases.push(addBase(f[0], 'animal', f[1], f[2], f[3])); });
    fixedPair.forEach(function (f) { pairBases.push(addBase(f[0], 'pair', f[1], f[2], f[3])); });

    var nodeBases = [];
    if (is3d) used.forEach(function (n) { nodeBases.push(addBase(names[n] + ' height', 'animal', 'height', 'BL', { node: n })); });
    used.forEach(function (n) {
        nodeBases.push(addBase(names[n] + ' forward', 'animal', 'fwd', 'BL', { node: n }));
        nodeBases.push(addBase(names[n] + ' left', 'animal', 'left', 'BL', { node: n }));
    });
    used.forEach(function (n) {
        nodeBases.push(addBase(names[n] + ' speed', 'animal', 'nspeed', 'BL/s', { node: n, band: n === head }));
    });
    for (var i = 0; i < used.length; i++) {
        for (var j = i + 1; j < used.length; j++) {
            var a = used[i], b = used[j];
            var isPaw = (a === head && paws.indexOf(b) >= 0) || (b === head && paws.indexOf(a) >= 0);
            nodeBases.push(addBase(names[a] + '–' + names[b] + ' distance', 'animal', 'dist', 'BL', { a: a, b: b, band: isPaw }));
        }
    }
    // Joint angles: at every used node, between each two of its edges to used nodes.
    var nbrs = new Map();
    (spec.edges || []).forEach(function (e) {
        var p = e[0], q = e[1];
        if (!usedSet.has(p) || !usedSet.has(q) || p === q) return;
        if (!nbrs.has(p)) nbrs.set(p, []);
        if (!nbrs.has(q)) nbrs.set(q, []);
        if (nbrs.get(p).indexOf(q) < 0) nbrs.get(p).push(q);
        if (nbrs.get(q).indexOf(p) < 0) nbrs.get(q).push(p);
    });
    used.forEach(function (mid) {
        var ns = (nbrs.get(mid) || []).slice().sort(function (x, y) { return x - y; });
        for (var p = 0; p < ns.length; p++) {
            for (var q = p + 1; q < ns.length; q++) {
                nodeBases.push(addBase('angle ' + names[ns[p]] + '–' + names[mid] + '–' + names[ns[q]], 'animal', 'angle', 'deg',
                    { a: ns[p], b: mid, c: ns[q] }));
            }
        }
    });
    animalBases = animalBases.concat(nodeBases);

    function addFeature(list, base, stat, window, band, unit) {
        var name = base.name;
        if (stat === 'power') name += STAT_SEP + bandLabel(band);
        else if (stat) name += STAT_SEP + stat + ' ' + windowLabel(window);
        var f = { name: name, scope: base.scope, base: base.name, stat: stat, window: window, band: band, unit: unit };
        list.push(f);
        byName.set(name, f);
    }
    function expand(list, base) {
        addFeature(list, base, null, null, null, base.unit);
        WINDOW_SECONDS.forEach(function (w) {
            WINDOW_STATS.forEach(function (s) { addFeature(list, base, s, w, null, base.unit); });
        });
        if (base.band) {
            var sq = base.unit === 'BL' ? 'BL²' : '(' + base.unit + ')²';
            BANDS.forEach(function (bd) { addFeature(list, base, 'power', null, bd, sq); });
        }
    }
    animalBases.forEach(function (b) { expand(animal, b); });
    pairBases.forEach(function (b) { expand(pair, b); });
    return { animal: animal, pair: pair, byName: byName, bases: bases };
}

var _catalogCache = { key: null, value: null };

/** `featureCatalog`, cached for the last spec seen (same node list, roles and mode). */
export function catalogFor(spec) {
    var key = JSON.stringify([spec.mode, spec.nodeNames, spec.edges, spec.roles, spec.used, spec.paws]);
    if (_catalogCache.key !== key) _catalogCache = { key: key, value: featureCatalog(spec) };
    return _catalogCache.value;
}

// ----------------------------------------------------------------------------
// Window statistics and band power (exported for tests)
// ----------------------------------------------------------------------------

/**
 * Centered-window statistic of `x` (NaN = missing) for every frame: over the
 * finite values in [t − h, t + h] ∩ [0, n). NaN where there are none. O(n).
 * @param {ArrayLike<number>} x
 * @param {number} h half-width in frames
 * @param {'mean'|'std'|'min'|'max'} stat
 * @returns {Float32Array}
 */
export function windowStat(x, h, stat) {
    var out = new Float32Array(x.length);
    if (stat === 'min' || stat === 'max') slidingExtreme(x, h, stat === 'max', out);
    else windowMoments(x, h, stat === 'std' ? null : out, stat === 'std' ? out : null);
    return out;
}

/**
 * Mean and/or population std over centered windows, from prefix sums of the
 * finite values (offset by the first one, to keep the sums small).
 */
function windowMoments(x, h, meanOut, stdOut) {
    var n = x.length;
    var s1 = new Float64Array(n + 1), s2 = new Float64Array(n + 1), c = new Int32Array(n + 1);
    var x0 = NaN;
    for (var i = 0; i < n && x0 !== x0; i++) if (fin(x[i])) x0 = x[i];
    if (x0 !== x0) x0 = 0;
    for (var t = 0; t < n; t++) {
        var v = x[t];
        if (fin(v)) {
            var d = v - x0;
            s1[t + 1] = s1[t] + d; s2[t + 1] = s2[t] + d * d; c[t + 1] = c[t] + 1;
        } else {
            s1[t + 1] = s1[t]; s2[t + 1] = s2[t]; c[t + 1] = c[t];
        }
    }
    for (var u = 0; u < n; u++) {
        var lo = u - h < 0 ? 0 : u - h, hi = u + h + 1 > n ? n : u + h + 1;
        var cnt = c[hi] - c[lo];
        if (!cnt) {
            if (meanOut) meanOut[u] = NaN;
            if (stdOut) stdOut[u] = NaN;
            continue;
        }
        var m = (s1[hi] - s1[lo]) / cnt;
        if (meanOut) meanOut[u] = m + x0;
        if (stdOut) {
            var varc = (s2[hi] - s2[lo]) / cnt - m * m;
            stdOut[u] = varc > 0 ? Math.sqrt(varc) : 0;
        }
    }
}

/** Sliding min (or max) of the finite values over centered windows, by a monotonic deque. */
function slidingExtreme(x, h, isMax, out) {
    var n = x.length;
    var dq = new Int32Array(Math.max(1, n));
    var head = 0, tail = 0, next = 0;
    for (var t = 0; t < n; t++) {
        var hi = t + h < n - 1 ? t + h : n - 1;
        while (next <= hi) {
            var v = x[next];
            if (fin(v)) {
                while (tail > head && (isMax ? x[dq[tail - 1]] <= v : x[dq[tail - 1]] >= v)) tail--;
                dq[tail++] = next;
            }
            next++;
        }
        var lo = t - h;
        while (tail > head && dq[head] < lo) head++;
        out[t] = tail > head ? x[dq[head]] : NaN;
    }
}

/**
 * Power of `x` in each of BANDS over centered windows of ±hb frames: a sliding
 * DFT over the window's own frequency bins (bin m = m × fps / (2hb + 1); the
 * window's mean is bin 0, outside every band, so it never leaks in). Missing
 * frames are held at the last finite value; a window less than half finite is
 * NaN, and so is a band with no bin below fps / 2. The power of a bin is
 * 2|X|² / W², a sinusoid's mean square, so a band's power is that band's share
 * of the window's variance.
 * @returns {Float32Array[]} one column per band
 */
export function bandPower(x, fps, hb) {
    var n = x.length;
    var W = 2 * hb + 1;
    var outs = BANDS.map(function () { return new Float32Array(n); });
    var bins = BANDS.map(function (bd) {
        var ms = [];
        for (var m = 1; m <= (W - 1) >> 1; m++) {
            var f = m * fps / W;
            if (f >= bd[0] && f < bd[1] && f < fps / 2) ms.push(m);
        }
        return ms;
    });
    var all = [];
    bins.forEach(function (ms) { ms.forEach(function (m) { if (all.indexOf(m) < 0) all.push(m); }); });
    all.sort(function (a, b) { return a - b; });
    var binIdx = bins.map(function (ms) { return ms.map(function (m) { return all.indexOf(m); }); });
    // Held signal (missing = last finite value) and the finite count, padded to the window.
    var held = new Float64Array(n), ok = new Int32Array(n + 1);
    var last = NaN;
    for (var i = 0; i < n && last !== last; i++) if (fin(x[i])) last = x[i];
    for (var t = 0; t < n; t++) {
        var v = x[t];
        if (fin(v)) { last = v; ok[t + 1] = ok[t] + 1; } else ok[t + 1] = ok[t];
        held[t] = last === last ? last : 0;
    }
    var at = function (j) { return j < 0 ? held[0] : j >= n ? held[n - 1] : held[j]; };
    var K = all.length;
    var re = new Float64Array(K), im = new Float64Array(K);
    var cr = new Float64Array(K), ci = new Float64Array(K);
    for (var q = 0; q < K; q++) { cr[q] = Math.cos(2 * Math.PI * all[q] / W); ci[q] = Math.sin(2 * Math.PI * all[q] / W); }
    var power = new Float64Array(K);
    var RESYNC = 512;
    for (var u = 0; u < n; u++) {
        var s = u - hb;
        if (u % RESYNC === 0) {
            // Exact DFT of the window starting at s (bounds sliding-update drift).
            for (var k = 0; k < K; k++) {
                var r = 0, im0 = 0, w = 2 * Math.PI * all[k] / W;
                for (var j = 0; j < W; j++) {
                    var val = at(s + j);
                    r += val * Math.cos(w * j); im0 -= val * Math.sin(w * j);
                }
                re[k] = r; im[k] = im0;
            }
        } else {
            // X_{s} = ω^m (X_{s-1} − x[s−1] + x[s−1+W]), ω = e^{2πi m / W}.
            var drop = at(s - 1), add = at(s - 1 + W);
            for (var k2 = 0; k2 < K; k2++) {
                var a = re[k2] - drop + add, b = im[k2];
                re[k2] = a * cr[k2] - b * ci[k2];
                im[k2] = a * ci[k2] + b * cr[k2];
            }
        }
        var lo = s < 0 ? 0 : s, hi = s + W > n ? n : s + W;
        var good = (ok[hi] - ok[lo]) * 2 >= (hi - lo) && hi > lo;
        for (var k3 = 0; k3 < K; k3++) power[k3] = 2 * (re[k3] * re[k3] + im[k3] * im[k3]) / (W * W);
        for (var bi = 0; bi < BANDS.length; bi++) {
            if (!good || !binIdx[bi].length) { outs[bi][u] = NaN; continue; }
            var sum = 0;
            for (var mi = 0; mi < binIdx[bi].length; mi++) sum += power[binIdx[bi][mi]];
            outs[bi][u] = sum;
        }
    }
    return outs;
}

// ----------------------------------------------------------------------------
// Per-chunk animal context
// ----------------------------------------------------------------------------

/**
 * Everything the base features of one animal read, over a padded chunk of L
 * frames: points in the floor frame (u, v along the floor, z up; z = 0 in
 * 2D), the centroid, the head and tail base, the heading, and the centroid's
 * horizontal (and vertical) velocity.
 */
function animalContext(spec, data, L, bodyLength) {
    var N = spec.nodeNames.length;
    var dim = spec.mode === '2d' ? 2 : 3;
    var used = spec.used;
    var fps = spec.fps > 0 ? spec.fps : 30;
    var k = halfFrames(SPEED_HALF_SECONDS, fps);
    var F = new Float64Array(L * N * 3);
    var fl = spec.floor;
    for (var t = 0; t < L; t++) {
        for (var n = 0; n < N; n++) {
            var src = (t * N + n) * dim, dst = (t * N + n) * 3;
            var x = data[src], y = data[src + 1], z = dim === 3 ? data[src + 2] : 0;
            if (!fin(x) || !fin(y) || !fin(z)) { F[dst] = NaN; F[dst + 1] = NaN; F[dst + 2] = NaN; continue; }
            if (dim === 3 && fl) {
                var dx = x - fl.origin[0], dy = y - fl.origin[1], dz = z - fl.origin[2];
                F[dst] = dx * fl.e1[0] + dy * fl.e1[1] + dz * fl.e1[2];
                F[dst + 1] = dx * fl.e2[0] + dy * fl.e2[1] + dz * fl.e2[2];
                F[dst + 2] = dx * fl.up[0] + dy * fl.up[1] + dz * fl.up[2];
            } else {
                F[dst] = x; F[dst + 1] = y; F[dst + 2] = dim === 3 ? z : 0;
            }
        }
    }
    var minPresent = Math.max(1, Math.ceil(used.length / 2));
    var present = new Int32Array(L);
    var C = new Float64Array(L * 3);
    for (var t2 = 0; t2 < L; t2++) {
        var cx = 0, cy = 0, cz = 0, cnt = 0;
        for (var i = 0; i < used.length; i++) {
            var o = (t2 * N + used[i]) * 3;
            if (F[o] !== F[o]) continue;
            cx += F[o]; cy += F[o + 1]; cz += F[o + 2]; cnt++;
        }
        present[t2] = cnt;
        if (cnt >= minPresent) { C[t2 * 3] = cx / cnt; C[t2 * 3 + 1] = cy / cnt; C[t2 * 3 + 2] = cz / cnt; }
        else { C[t2 * 3] = NaN; C[t2 * 3 + 1] = NaN; C[t2 * 3 + 2] = NaN; }
    }
    var head = spec.roles && spec.roles.head != null ? spec.roles.head : -1;
    var tail = spec.roles && spec.roles.tailBase != null ? spec.roles.tailBase : -1;
    var hd = new Float64Array(L), fx = new Float64Array(L), fy = new Float64Array(L);
    for (var t3 = 0; t3 < L; t3++) {
        hd[t3] = NaN; fx[t3] = NaN; fy[t3] = NaN;
        if (head < 0 || tail < 0) continue;
        var ho = (t3 * N + head) * 3, to = (t3 * N + tail) * 3;
        var ax = F[ho] - F[to], ay = F[ho + 1] - F[to + 1];
        var len = Math.sqrt(ax * ax + ay * ay);
        if (!(len > 1e-9 * (bodyLength || 1))) continue;
        hd[t3] = Math.atan2(ay, ax); fx[t3] = ax / len; fy[t3] = ay / len;
    }
    // Centroid velocity: the mean displacement of the used nodes seen at both
    // ends (robust to a node dropping in or out, unlike the centroid's own).
    var V = new Float64Array(L * 3);
    var dtSpan = 2 * k / fps;
    for (var t4 = 0; t4 < L; t4++) {
        V[t4 * 3] = NaN; V[t4 * 3 + 1] = NaN; V[t4 * 3 + 2] = NaN;
        if (t4 - k < 0 || t4 + k >= L) continue;
        var vx = 0, vy = 0, vz = 0, m = 0;
        for (var i2 = 0; i2 < used.length; i2++) {
            var o0 = ((t4 - k) * N + used[i2]) * 3, o1 = ((t4 + k) * N + used[i2]) * 3;
            if (F[o0] !== F[o0] || F[o1] !== F[o1]) continue;
            vx += F[o1] - F[o0]; vy += F[o1 + 1] - F[o0 + 1]; vz += F[o1 + 2] - F[o0 + 2]; m++;
        }
        if (m >= minPresent) { V[t4 * 3] = vx / m / dtSpan; V[t4 * 3 + 1] = vy / m / dtSpan; V[t4 * 3 + 2] = vz / m / dtSpan; }
    }
    return {
        L: L, N: N, F: F, C: C, present: present, hd: hd, fx: fx, fy: fy, V: V, k: k, fps: fps, dtSpan: dtSpan,
        head: head, tail: tail, BL: bodyLength > 0 ? bodyLength : 1,
    };
}

function dist3(F, o1, G, o2) {
    var dx = F[o1] - G[o2], dy = F[o1 + 1] - G[o2 + 1], dz = F[o1 + 2] - G[o2 + 2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

// ----------------------------------------------------------------------------
// Base features
// ----------------------------------------------------------------------------

/** One animal base feature over the padded chunk. */
function animalBase(spec, base, ctx, all, nearest) {
    var L = ctx.L, N = ctx.N, F = ctx.F, C = ctx.C, BL = ctx.BL, k = ctx.k;
    var out = new Float32Array(L);
    var used = spec.used;
    var t, o, o2;
    switch (base.kind) {
    case 'height':
        for (t = 0; t < L; t++) { o = (t * N + base.node) * 3; out[t] = F[o + 2] / BL; }
        break;
    case 'cheight':
        for (t = 0; t < L; t++) out[t] = C[t * 3 + 2] / BL;
        break;
    case 'fwd': case 'left':
        for (t = 0; t < L; t++) {
            o = (t * N + base.node) * 3;
            var du = F[o] - C[t * 3], dv = F[o + 1] - C[t * 3 + 1];
            out[t] = (base.kind === 'fwd' ? du * ctx.fx[t] + dv * ctx.fy[t] : -du * ctx.fy[t] + dv * ctx.fx[t]) / BL;
        }
        break;
    case 'dist':
        for (t = 0; t < L; t++) out[t] = dist3(F, (t * N + base.a) * 3, F, (t * N + base.b) * 3) / BL;
        break;
    case 'angle':
        for (t = 0; t < L; t++) {
            var oa = (t * N + base.a) * 3, ob = (t * N + base.b) * 3, oc = (t * N + base.c) * 3;
            var ux = F[oa] - F[ob], uy = F[oa + 1] - F[ob + 1], uz = F[oa + 2] - F[ob + 2];
            var wx = F[oc] - F[ob], wy = F[oc + 1] - F[ob + 1], wz = F[oc + 2] - F[ob + 2];
            var nu = Math.sqrt(ux * ux + uy * uy + uz * uz), nw = Math.sqrt(wx * wx + wy * wy + wz * wz);
            if (!(nu > 0) || !(nw > 0)) { out[t] = NaN; continue; }
            var cs = (ux * wx + uy * wy + uz * wz) / (nu * nw);
            out[t] = Math.acos(cs > 1 ? 1 : cs < -1 ? -1 : cs) * DEG;
        }
        break;
    case 'cspeed':
        for (t = 0; t < L; t++) out[t] = Math.sqrt(ctx.V[t * 3] * ctx.V[t * 3] + ctx.V[t * 3 + 1] * ctx.V[t * 3 + 1]) / BL;
        break;
    case 'vspeed':
        for (t = 0; t < L; t++) out[t] = ctx.V[t * 3 + 2] / BL;
        break;
    case 'nspeed':
        for (t = 0; t < L; t++) {
            if (t - k < 0 || t + k >= L) { out[t] = NaN; continue; }
            out[t] = dist3(F, ((t + k) * N + base.node) * 3, F, ((t - k) * N + base.node) * 3) / ctx.dtSpan / BL;
        }
        break;
    case 'turn':
        for (t = 0; t < L; t++) {
            if (t - k < 0 || t + k >= L) { out[t] = NaN; continue; }
            out[t] = wrapAngle(ctx.hd[t + k] - ctx.hd[t - k]) / ctx.dtSpan * DEG;
        }
        break;
    case 'elong':
        for (t = 0; t < L; t++) {
            if (ctx.head < 0 || ctx.tail < 0) { out[t] = NaN; continue; }
            out[t] = dist3(F, (t * N + ctx.head) * 3, F, (t * N + ctx.tail) * 3) / BL;
        }
        break;
    case 'missing':
        for (t = 0; t < L; t++) out[t] = used.length - ctx.present[t];
        break;
    case 'nearCent': case 'nearMin': case 'nearHH':
        for (t = 0; t < L; t++) {
            var j = nearest[t];
            if (j < 0) { out[t] = NaN; continue; }
            var other = all[j];
            if (base.kind === 'nearCent') out[t] = dist3(C, t * 3, other.C, t * 3) / BL;
            else if (base.kind === 'nearHH') {
                out[t] = ctx.head < 0 ? NaN : dist3(F, (t * N + ctx.head) * 3, other.F, (t * N + ctx.head) * 3) / BL;
            } else out[t] = minNodeDistance(used, ctx, other, t) / BL;
        }
        break;
    default:
        out.fill(NaN);
    }
    for (t = 0; t < L; t++) if (!fin(out[t])) out[t] = NaN;
    return out;
}

function minNodeDistance(used, A, B, t) {
    var N = A.N, best = Infinity;
    for (var i = 0; i < used.length; i++) {
        var oa = (t * N + used[i]) * 3;
        if (A.F[oa] !== A.F[oa]) continue;
        for (var j = 0; j < used.length; j++) {
            var ob = (t * N + used[j]) * 3;
            if (B.F[ob] !== B.F[ob]) continue;
            var d = dist3(A.F, oa, B.F, ob);
            if (d < best) best = d;
        }
    }
    return best === Infinity ? NaN : best;
}

/** For each frame, the index (into `all`) of the animal nearest to `a` by centroid, or -1. */
function nearestOf(a, all, L) {
    var out = new Int16Array(L).fill(-1);
    var A = all[a];
    for (var t = 0; t < L; t++) {
        if (A.C[t * 3] !== A.C[t * 3]) continue;
        var best = Infinity;
        for (var b = 0; b < all.length; b++) {
            if (b === a || !all[b]) continue;
            var B = all[b];
            if (B.C[t * 3] !== B.C[t * 3]) continue;
            var d = dist3(A.C, t * 3, B.C, t * 3);
            if (d < best) { best = d; out[t] = b; }
        }
    }
    return out;
}

/** One actor → target base feature over the padded chunk (lengths in the actor's body lengths). */
function pairBase(spec, base, A, B) {
    var L = A.L, N = A.N, BL = A.BL, k = A.k;
    var out = new Float32Array(L);
    var t;
    // The target's centroid seen from the actor's head, in the actor's heading frame.
    function bearing(tt) {
        if (A.head < 0) return NaN;
        var ho = (tt * N + A.head) * 3;
        var du = B.C[tt * 3] - A.F[ho], dv = B.C[tt * 3 + 1] - A.F[ho + 1];
        var along = du * A.fx[tt] + dv * A.fy[tt], side = -du * A.fy[tt] + dv * A.fx[tt];
        if (!fin(along) || !fin(side) || (along === 0 && side === 0)) return NaN;
        return Math.atan2(side, along);
    }
    switch (base.kind) {
    case 'pCent':
        for (t = 0; t < L; t++) out[t] = dist3(A.C, t * 3, B.C, t * 3) / BL;
        break;
    case 'pHH':
        for (t = 0; t < L; t++) out[t] = A.head < 0 ? NaN : dist3(A.F, (t * N + A.head) * 3, B.F, (t * N + B.head) * 3) / BL;
        break;
    case 'pHT':
        for (t = 0; t < L; t++) out[t] = A.head < 0 || B.tail < 0 ? NaN : dist3(A.F, (t * N + A.head) * 3, B.F, (t * N + B.tail) * 3) / BL;
        break;
    case 'pMin':
        for (t = 0; t < L; t++) out[t] = minNodeDistance(spec.used, A, B, t) / BL;
        break;
    case 'pBear': case 'pBearSin': case 'pBearCos': case 'pView':
        for (t = 0; t < L; t++) {
            var br = bearing(t);
            if (br !== br) { out[t] = NaN; continue; }
            if (base.kind === 'pBear') out[t] = Math.abs(br) * DEG;
            else if (base.kind === 'pBearSin') out[t] = Math.sin(br);
            else if (base.kind === 'pBearCos') out[t] = Math.cos(br);
            else out[t] = Math.abs(br) * DEG <= VIEW_HALF_ANGLE ? 1 : 0;
        }
        break;
    case 'pRelHead':
        for (t = 0; t < L; t++) out[t] = Math.abs(wrapAngle(B.hd[t] - A.hd[t])) * DEG;
        break;
    case 'pApproach':
        for (t = 0; t < L; t++) {
            if (t - k < 0 || t + k >= L) { out[t] = NaN; continue; }
            var d1 = dist3(A.C, (t + k) * 3, B.C, (t + k) * 3), d0 = dist3(A.C, (t - k) * 3, B.C, (t - k) * 3);
            out[t] = -(d1 - d0) / A.dtSpan / BL;
        }
        break;
    case 'pTang':
        for (t = 0; t < L; t++) {
            var rx = B.C[t * 3] - A.C[t * 3], ry = B.C[t * 3 + 1] - A.C[t * 3 + 1];
            var rl = Math.sqrt(rx * rx + ry * ry);
            var vx = B.V[t * 3] - A.V[t * 3], vy = B.V[t * 3 + 1] - A.V[t * 3 + 1];
            if (!(rl > 0) || !fin(vx) || !fin(vy)) { out[t] = NaN; continue; }
            out[t] = Math.abs(-vx * ry / rl + vy * rx / rl) / BL;
        }
        break;
    default:
        out.fill(NaN);
    }
    for (t = 0; t < L; t++) if (!fin(out[t])) out[t] = NaN;
    return out;
}

// ----------------------------------------------------------------------------
// A chunk
// ----------------------------------------------------------------------------

/**
 * Compute the requested features for frames [start, end).
 *
 * @param {Object} spec as for `featureCatalog`, plus {fps, floor, bodyLength: number[], animals: string[]}
 * @param {{start:number, end:number, padStart:number, padEnd:number, data: Float32Array[]}} chunk
 *   `data[a]` holds animal a's points for frames [padStart, padEnd), N × dim per frame, NaN = missing
 * @param {{animal?: string[], pair?: string[], pairs?: number[][]}} request feature names
 *   (animal features for every animal; pair features for each [actor, target] in `pairs`)
 * @returns {{animal: Object<string, Float32Array>[], pair: Object<string, Object<string, Float32Array>>}}
 *   `animal[a][name]` and `pair['a>b'][name]`, each of length end − start
 */
export function computeFeatureChunk(spec, chunk, request) {
    var cat = catalogFor(spec);
    var L = chunk.padEnd - chunk.padStart;
    var off = chunk.start - chunk.padStart, len = chunk.end - chunk.start;
    var nA = chunk.data.length;
    var animalNames = (request.animal || []).filter(function (nm) { return cat.byName.has(nm) && cat.byName.get(nm).scope === 'animal'; });
    var pairNames = (request.pair || []).filter(function (nm) { return cat.byName.has(nm) && cat.byName.get(nm).scope === 'pair'; });
    var pairs = request.pairs || [];
    var contexts = new Array(nA);
    var ctxOf = function (a) {
        if (!contexts[a]) contexts[a] = animalContext(spec, chunk.data[a], L, (spec.bodyLength || [])[a]);
        return contexts[a];
    };
    var needsAll = pairNames.length > 0 || animalNames.some(function (nm) { return /^near/.test(cat.bases.get(cat.byName.get(nm).base).kind); });
    if (needsAll) for (var a0 = 0; a0 < nA; a0++) ctxOf(a0);
    var fps = spec.fps > 0 ? spec.fps : 30;
    var hb = halfFrames(BAND_HALF_SECONDS, fps);

    function derive(f, baseCol, cache) {
        if (!f.stat) return baseCol;
        if (f.stat === 'power') {
            if (!cache.power) cache.power = bandPower(baseCol, fps, hb);
            for (var bi = 0; bi < BANDS.length; bi++) {
                if (BANDS[bi][0] === f.band[0] && BANDS[bi][1] === f.band[1]) return cache.power[bi];
            }
            return new Float32Array(baseCol.length).fill(NaN);
        }
        var key = f.stat + '@' + f.window;
        if (!cache[key]) cache[key] = windowStat(baseCol, halfFrames(f.window, fps), f.stat);
        return cache[key];
    }
    function crop(col) {
        return col.slice(off, off + len);
    }

    var animalOut = [];
    for (var a = 0; a < nA; a++) {
        var res = {};
        if (animalNames.length) {
            var ctx = ctxOf(a);
            var near = needsAll ? nearestOf(a, contexts, L) : null;
            var baseCols = new Map(), caches = new Map();
            for (var i = 0; i < animalNames.length; i++) {
                var f = cat.byName.get(animalNames[i]);
                if (!baseCols.has(f.base)) {
                    baseCols.set(f.base, animalBase(spec, cat.bases.get(f.base), ctx, contexts, near));
                    caches.set(f.base, {});
                }
                res[f.name] = crop(derive(f, baseCols.get(f.base), caches.get(f.base)));
            }
        }
        animalOut.push(res);
    }
    var pairOut = {};
    for (var p = 0; p < pairs.length; p++) {
        var A = ctxOf(pairs[p][0]), B = ctxOf(pairs[p][1]);
        var pres = {};
        var pBase = new Map(), pCache = new Map();
        for (var q = 0; q < pairNames.length; q++) {
            var pf = cat.byName.get(pairNames[q]);
            if (!pBase.has(pf.base)) {
                pBase.set(pf.base, pairBase(spec, cat.bases.get(pf.base), A, B));
                pCache.set(pf.base, {});
            }
            pres[pf.name] = crop(derive(pf, pBase.get(pf.base), pCache.get(pf.base)));
        }
        pairOut[pairs[p][0] + '>' + pairs[p][1]] = pres;
    }
    return { animal: animalOut, pair: pairOut };
}

/**
 * The chunks a session of `numFrames` frames is computed in: `[start, end)`
 * with its padded range, `chunkFrames` long (the last one shorter).
 */
export function planChunks(numFrames, pad, chunkFrames) {
    var out = [];
    var size = Math.max(1, chunkFrames | 0);
    for (var s = 0; s < numFrames; s += size) {
        var e = Math.min(numFrames, s + size);
        out.push({ start: s, end: e, padStart: Math.max(0, s - pad), padEnd: Math.min(numFrames, e + pad) });
    }
    return out;
}

/**
 * Slice animal data for one chunk: `data[a]` (whole session, N × dim per
 * frame) -> a copy covering [padStart, padEnd), safe to transfer to a worker.
 * `skip[a][t]` set (pose/behavior-tensor.js `brokenPoseFrames`) blanks that
 * frame in the copy, so every feature reads it as missing.
 */
export function sliceChunkData(data, chunk, framesStride, skip) {
    return data.map(function (arr, a) {
        var out = arr.slice(chunk.padStart * framesStride, chunk.padEnd * framesStride);
        var m = skip && skip[a];
        if (m) {
            for (var t = chunk.padStart; t < chunk.padEnd; t++) {
                if (m[t]) out.fill(NaN, (t - chunk.padStart) * framesStride, (t - chunk.padStart + 1) * framesStride);
            }
        }
        return out;
    });
}

/**
 * Resolve a feature name in a context. In an animal context a name is one of
 * the animal's features; in a pair context it is a relational feature, the
 * actor's own feature, or the target's (`TARGET_PREFIX` + name).
 * @returns {{scope:'animal'|'pair', who:'self'|'actor'|'target'|'pair', feature:Object}|null}
 */
export function resolveFeatureName(catalog, name, social) {
    if (typeof name !== 'string') return null;
    var f = catalog.byName.get(name);
    if (social) {
        if (f && f.scope === 'pair') return { scope: 'pair', who: 'pair', feature: f };
        if (f && f.scope === 'animal') return { scope: 'animal', who: 'actor', feature: f };
        if (name.indexOf(TARGET_PREFIX) === 0) {
            var g = catalog.byName.get(name.slice(TARGET_PREFIX.length));
            if (g && g.scope === 'animal') return { scope: 'animal', who: 'target', feature: g };
        }
        return null;
    }
    return f && f.scope === 'animal' ? { scope: 'animal', who: 'self', feature: f } : null;
}

/** Every name a condition may use: animal features, and for a social behavior also the relational and target ones. */
export function featureNamesFor(catalog, social) {
    var out = catalog.animal.map(function (f) { return f.name; });
    if (!social) return out;
    return catalog.pair.map(function (f) { return f.name; }).concat(out, catalog.animal.map(function (f) { return TARGET_PREFIX + f.name; }));
}
