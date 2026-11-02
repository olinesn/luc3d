// pose/behavior-tensor.js — the poses the behavior features read, per animal,
// and the body and floor they are measured against.
//
// ## Input
//
// `collectBehaviorInput` builds one NaN-filled Float32Array per animal (T
// frames × N nodes × 3, about 32 MB per animal for 176k frames × 15 nodes,
// outside the V8 heap):
//   * 3D: from `session.instanceGroups` — always resident, lazy projects
//     included — keyed by `group.identityId`, as every bulk 3D collector is
//     (pose/temporal-smoothing.js `collectTrajectories`). An animal is an
//     identity NAME (labels and predictions are keyed by name); groups without
//     an identity are skipped, and a second group of one identity on a frame
//     (a proofreading defect) is counted, not used.
//   * 2D, when the session has no identified 3D (single-camera projects): one
//     camera's 2D. Resident frames are read from memory (a group member takes
//     its group's identity, an ungrouped one its per-frame identity); a lazy
//     project's other frames straight from the store columns, as
//     pose/detection-filter-stats.js `collectStore` does, so nothing is
//     hydrated. Lengths are then pixels, and there are no heights.
//
// ## Body
//
// Node roles — the head (front of the body: heading, head height, head–head
// distance) and the tail base (back of the body) — are guessed from node names
// and can be set in Behavior detection settings (`ethogram.features`, saved
// only once changed). Excluded nodes take no part in any feature; by default
// the tail nodes past the tail base (a dangling tail would dominate speeds and
// distances). A body length is each animal's session-median head–tail base
// distance, the unit every length feature is divided by.
//
// ## Floor
//
// Every feature is measured against a floor (pose/behavior-features.js): its
// up direction and a point on it. In order of preference:
//   1. a Define Planes plane chosen as the floor (its normal turned toward the
//      animals),
//   2. the Set Origin frame (its +Z is up),
//   3. a robust fit to the lower envelope of the animals' nodes (`fitFloor`).
// The two horizontal axes are built from `up` alone (`floorFrame`), since no
// feature may depend on them.
//
// ## Broken tracking
//
// `brokenPoseFrames` marks the animal-frames whose tracking is clearly broken
// — a used node far off the body, or the body jumping faster than an animal
// can move — and the features skip them (`sliceChunkData`'s masks: those
// frames read as missing). Without it a 3D point triangulated from the wrong
// detections reads as a rear, and a jump to another animal as a run.
//
// DOM-free.
//
// Depends on: pose/pose-data.js, pose/plane-fit.js, pose/behavior-features.js

import { POINT3D_STRIDE, points3dNodeCount } from './pose-data.js';
import { jacobiEigen } from './plane-fit.js';
import { halfFrames, SPEED_HALF_SECONDS } from './behavior-features.js';

// ----------------------------------------------------------------------------
// Node roles
// ----------------------------------------------------------------------------

var HEAD_PATTERNS = [/nose|snout/i, /head/i];
var TAIL_BASE_PATTERNS = [/tail.?base|base.?of.?tail|tti|tail.?root|tailbase/i];
var PAW_PATTERN = /paw|hand|wrist|forelimb|foreleg|front.?foot/i;

function findByPatterns(names, patterns, except) {
    for (var p = 0; p < patterns.length; p++) {
        for (var i = 0; i < names.length; i++) {
            if (i !== except && patterns[p].test(names[i])) return i;
        }
    }
    return -1;
}

/** Shortest edge-hop distance from `start` to every node (-1 = unreachable). */
function hops(n, edges, start) {
    var d = new Array(n).fill(-1);
    if (start < 0) return d;
    var adj = [];
    for (var i = 0; i < n; i++) adj.push([]);
    (edges || []).forEach(function (e) {
        if (e[0] >= 0 && e[0] < n && e[1] >= 0 && e[1] < n) { adj[e[0]].push(e[1]); adj[e[1]].push(e[0]); }
    });
    d[start] = 0;
    var q = [start];
    while (q.length) {
        var u = q.shift();
        adj[u].forEach(function (v) { if (d[v] < 0) { d[v] = d[u] + 1; q.push(v); } });
    }
    return d;
}

/**
 * The head and tail base guessed from node names: the head is the first node
 * named like nose / snout, else head; the tail base one named like tail base /
 * TTI, else the tail node nearest the head, else the node farthest from the
 * head along the skeleton.
 * @returns {{head:number, tailBase:number}} node indices, -1 when not found
 */
export function guessNodeRoles(nodeNames, edges) {
    var names = nodeNames || [];
    var head = findByPatterns(names, HEAD_PATTERNS, -1);
    var tail = findByPatterns(names, TAIL_BASE_PATTERNS, head);
    if (tail < 0 && head >= 0) {
        var d = hops(names.length, edges, head);
        var best = -1;
        names.forEach(function (nm, i) {
            if (i === head || !/tail/i.test(nm)) return;
            if (best < 0 || (d[i] >= 0 && (d[best] < 0 || d[i] < d[best]))) best = i;
        });
        if (best < 0) {
            names.forEach(function (nm, i) { if (i !== head && d[i] > 0 && (best < 0 || d[i] > d[best])) best = i; });
        }
        tail = best;
    }
    return { head: head, tailBase: tail };
}

/**
 * The nodes left out of features by default: tail nodes past the tail base —
 * named like "tail", other than the base itself, and (when edges say so)
 * farther from the head than the base.
 */
export function defaultExcludedNodes(nodeNames, edges, roles) {
    var names = nodeNames || [];
    var tail = roles ? roles.tailBase : -1;
    if (tail < 0) return [];
    var d = hops(names.length, edges, roles.head);
    var out = [];
    names.forEach(function (nm, i) {
        if (i === tail || i === roles.head || !/tail/i.test(nm) || TAIL_BASE_PATTERNS[0].test(nm)) return;
        if (d[tail] >= 0 && d[i] >= 0 && d[i] <= d[tail]) return;
        out.push(i);
    });
    return out;
}

/**
 * The body a session's features use: roles and excluded nodes from the saved
 * settings (by node NAME) where they name existing nodes, guessed otherwise.
 * @param {string[]} nodeNames
 * @param {number[][]} edges
 * @param {{nodeRoles?:{head?:string, tailBase?:string}, exclude?:string[]}|null} settings
 * @returns {{roles:{head, tailBase}, guessed:{head, tailBase}, used:number[], excluded:number[],
 *            defaultExcluded:number[], paws:number[], roleNames:{head?:string, tailBase?:string}}}
 */
export function resolveBody(nodeNames, edges, settings) {
    var names = nodeNames || [];
    var guessed = guessNodeRoles(names, edges);
    var roles = { head: guessed.head, tailBase: guessed.tailBase };
    var nr = settings && settings.nodeRoles;
    if (nr) {
        ['head', 'tailBase'].forEach(function (r) {
            if (typeof nr[r] === 'string') { var i = names.indexOf(nr[r]); if (i >= 0) roles[r] = i; }
        });
    }
    var defaults = defaultExcludedNodes(names, edges, roles);
    var excluded = defaults;
    if (settings && Array.isArray(settings.exclude)) {
        excluded = [];
        settings.exclude.forEach(function (nm) { var i = names.indexOf(nm); if (i >= 0 && excluded.indexOf(i) < 0) excluded.push(i); });
    }
    excluded = excluded.filter(function (i) { return i !== roles.head && i !== roles.tailBase; });
    var used = [];
    for (var i = 0; i < names.length; i++) if (excluded.indexOf(i) < 0) used.push(i);
    var paws = used.filter(function (i) { return i !== roles.head && PAW_PATTERN.test(names[i]); });
    return {
        roles: roles, guessed: guessed, used: used, excluded: excluded, defaultExcluded: defaults, paws: paws,
        roleNames: { head: roles.head >= 0 ? names[roles.head] : undefined, tailBase: roles.tailBase >= 0 ? names[roles.tailBase] : undefined },
    };
}

// ----------------------------------------------------------------------------
// Input
// ----------------------------------------------------------------------------

function identityName(session, id) {
    var ids = (session && session.identities) || [];
    for (var i = 0; i < ids.length; i++) if (ids[i] && ids[i].id === id) return ids[i].name;
    return null;
}

/** Does the session have any identified 3D? */
export function hasIdentified3d(session) {
    if (!session || !session.instanceGroups) return false;
    for (var [, groups] of session.instanceGroups) {
        if (!groups) continue;
        for (var i = 0; i < groups.length; i++) {
            var g = groups[i];
            if (g && g.points3d && g.identityId != null && g.identityId >= 0 && points3dNodeCount(g.points3d) > 0) return true;
        }
    }
    return false;
}

function newAnimalArrays(count, size) {
    var out = [];
    for (var i = 0; i < count; i++) out.push(new Float32Array(size).fill(NaN));
    return out;
}

function collect3d(session, N, numFrames) {
    var seen = new Set(), maxFrame = -1;
    for (var [f, groups] of session.instanceGroups) {
        if (!groups) continue;
        for (var i = 0; i < groups.length; i++) {
            var g = groups[i];
            if (g && g.points3d && g.identityId != null && g.identityId >= 0) { seen.add(g.identityId); if (f > maxFrame) maxFrame = f; }
        }
    }
    var T = numFrames > 0 ? numFrames : maxFrame + 1;
    var order = [];
    (session.identities || []).forEach(function (id) { if (id && seen.has(id.id) && order.indexOf(id.id) < 0) order.push(id.id); });
    var index = new Map(order.map(function (id, k) { return [id, k]; }));
    var data = newAnimalArrays(order.length, Math.max(0, T) * N * 3);
    var frames = new Array(order.length).fill(0);
    var skipped = 0;
    var filled = new Set();
    for (var [frame, list] of session.instanceGroups) {
        if (!list || frame < 0 || frame >= T) continue;
        filled.clear();
        for (var j = 0; j < list.length; j++) {
            var grp = list[j];
            if (!grp || !grp.points3d || grp.identityId == null || grp.identityId < 0) continue;
            var a = index.get(grp.identityId);
            if (a == null) continue;
            if (filled.has(a)) { skipped++; continue; }
            filled.add(a);
            var pts = grp.points3d, n = Math.min(N, points3dNodeCount(pts));
            var arr = data[a], base = frame * N * 3, any = false;
            for (var k = 0; k < n; k++) {
                var o = k * POINT3D_STRIDE, x = pts[o], y = pts[o + 1], z = pts[o + 2];
                if (x !== x || y !== y || z !== z) continue;
                arr[base + k * 3] = x; arr[base + k * 3 + 1] = y; arr[base + k * 3 + 2] = z;
                any = true;
            }
            if (any) frames[a]++;
        }
    }
    return {
        mode: '3d', dim: 3, camera: null, numFrames: T,
        animals: order.map(function (id) { return identityName(session, id); }), animalIds: order,
        data: data, frames: frames, skipped: skipped,
    };
}

/** The camera whose 2D the fallback reads: the requested one, else the first with any 2D. */
export function pick2dCamera(session, preferred) {
    var cams = (session && session.cameras) || [];
    var names = cams.map(function (c) { return c.name; });
    if (preferred && names.indexOf(preferred) >= 0) return preferred;
    var loader = session && session.lazyLoader;
    for (var i = 0; i < names.length; i++) {
        var cam = names[i];
        if (loader && loader.labelsByCam && loader.labelsByCam.has(cam)) return cam;
        if (session.frameGroups) {
            for (var [, fg] of session.frameGroups) {
                if (fg && (fg.getInstances(cam).length || fg.getUnlinkedInstances(cam).length)) return cam;
            }
        }
    }
    return names[0] || null;
}

function collect2d(session, N, numFrames, cam) {
    var rows = [];      // [frame, identityId, xy (Float64Array 2N) or store reader]
    var maxFrame = -1;
    var resident = session.frameGroups || new Map();
    // Resident frames, from memory.
    for (var [frame, fg] of resident) {
        if (!fg) continue;
        var memberId = new Map();
        var groups = session.instanceGroups ? session.instanceGroups.get(frame) : null;
        if (groups) {
            groups.forEach(function (g) {
                if (!g || g.identityId == null || g.identityId < 0) return;
                var inst = g.getInstance(cam);
                if (inst) memberId.set(inst, g.identityId);
            });
        }
        var take = function (inst) {
            if (!inst || !inst._xy) return;
            var id = memberId.has(inst) ? memberId.get(inst) : session.getIdentityIdForUnlinkedInstance
                ? session.getIdentityIdForUnlinkedInstance(cam, inst, frame) : null;
            if (id == null || id < 0) return;
            rows.push([frame, id, inst._xy]);
            if (frame > maxFrame) maxFrame = frame;
        };
        fg.getInstances(cam).forEach(take);
        fg.getUnlinkedInstances(cam).forEach(function (u) { take(u && u.instance); });
    }
    // A lazy project's other frames, from the store columns.
    var loader = session.lazyLoader;
    var labels = loader && loader.labelsByCam && loader.labelsByCam.get(cam);
    var store = labels && labels._lazyDataStore;
    if (store && typeof loader.forEachInstanceRow === 'function') {
        var idn = store.instancesData || {}, pts = store.pointsData || {}, pred = store.predPointsData || {};
        loader.forEachInstanceRow(function (cn, frameIdx, trk, info) {
            if (resident.has(frameIdx) || trk < 0) return;
            var id = session.getIdentityIdForTrack(cam, trk, frameIdx);
            if (id == null || id < 0) return;
            var j = info.instanceRow;
            var isPred = Number(idn.instance_type ? idn.instance_type[j] : 0) !== 0;
            var cols = isPred ? pred : pts;
            var p0 = Number(idn.point_id_start ? idn.point_id_start[j] : 0) || 0;
            var p1 = Number(idn.point_id_end ? idn.point_id_end[j] : 0) || 0;
            var m = Math.min(p1 - p0, N);
            var xy = new Float64Array(2 * N).fill(NaN);
            for (var k = 0; k < m; k++) {
                var x = cols.x ? Number(cols.x[p0 + k]) : NaN, y = cols.y ? Number(cols.y[p0 + k]) : NaN;
                if (isFinite(x) && isFinite(y)) { xy[2 * k] = x; xy[2 * k + 1] = y; }
            }
            rows.push([frameIdx, id, xy]);
            if (frameIdx > maxFrame) maxFrame = frameIdx;
        }, { camera: cam });
    }
    var T = numFrames > 0 ? numFrames : maxFrame + 1;
    var seen = new Set(rows.map(function (r) { return r[1]; }));
    var order = [];
    (session.identities || []).forEach(function (id) { if (id && seen.has(id.id) && order.indexOf(id.id) < 0) order.push(id.id); });
    var index = new Map(order.map(function (id, k) { return [id, k]; }));
    var data = newAnimalArrays(order.length, Math.max(0, T) * N * 2);
    var frames = new Array(order.length).fill(0);
    var used = new Set(), skipped = 0;
    rows.forEach(function (r) {
        var a = index.get(r[1]);
        if (a == null || r[0] < 0 || r[0] >= T) return;
        var key = r[0] * 4096 + a;
        if (used.has(key)) { skipped++; return; }
        used.add(key);
        var xy = r[2], arr = data[a], base = r[0] * N * 2, any = false;
        var n = Math.min(N, xy.length >> 1);
        for (var k = 0; k < n; k++) {
            var x = xy[2 * k], y = xy[2 * k + 1];
            if (x !== x || y !== y) continue;
            arr[base + 2 * k] = x; arr[base + 2 * k + 1] = y; any = true;
        }
        if (any) frames[a]++;
    });
    return {
        mode: '2d', dim: 2, camera: cam, numFrames: T,
        animals: order.map(function (id) { return identityName(session, id); }), animalIds: order,
        data: data, frames: frames, skipped: skipped,
    };
}

/**
 * Every animal's poses for the behavior features.
 * @param {Session} session
 * @param {{numFrames?: number, camera?: string, mode?: 'auto'|'3d'|'2d'}} [opts]
 * @returns {{mode:'3d'|'2d', dim:number, camera:?string, numFrames:number, nodeNames:string[], edges:number[][],
 *            animals:string[], animalIds:number[], data:Float32Array[], frames:number[], skipped:number}}
 */
export function collectBehaviorInput(session, opts) {
    opts = opts || {};
    var skel = session && session.skeleton;
    var nodeNames = skel && skel.nodes ? skel.nodes.map(function (nd) { return typeof nd === 'string' ? nd : nd && nd.name; }) : [];
    var edges = skel && skel.edges ? skel.edges.map(function (e) { return Array.isArray(e) ? [e[0], e[1]] : [e.source, e.destination]; }) : [];
    var N = nodeNames.length;
    var mode = opts.mode === '2d' || opts.mode === '3d' ? opts.mode : (hasIdentified3d(session) ? '3d' : '2d');
    var res = mode === '3d' ? collect3d(session, N, opts.numFrames)
        : collect2d(session, N, opts.numFrames, pick2dCamera(session, opts.camera));
    res.nodeNames = nodeNames;
    res.edges = edges.filter(function (e) { return e[0] >= 0 && e[0] < N && e[1] >= 0 && e[1] < N; });
    return res;
}

// ----------------------------------------------------------------------------
// Body lengths
// ----------------------------------------------------------------------------

function median(values) {
    if (!values.length) return NaN;
    values.sort(function (a, b) { return a - b; });
    var m = values.length >> 1;
    return values.length % 2 ? values[m] : (values[m - 1] + values[m]) / 2;
}

/**
 * Each animal's body length in data units: the median head–tail base distance
 * over (up to ~50,000 sampled) frames; without both roles, twice the median
 * largest distance of a used node from the others' mean; 1 when there is no
 * data at all.
 */
export function bodyLengths(input, body) {
    var N = input.nodeNames.length, D = input.dim, T = input.numFrames;
    var step = Math.max(1, Math.floor(T / 50000));
    var head = body.roles.head, tail = body.roles.tailBase;
    return input.data.map(function (arr) {
        var vals = [];
        for (var t = 0; t < T; t += step) {
            var base = t * N * D;
            if (head >= 0 && tail >= 0) {
                var s = 0, ok = true;
                for (var d = 0; d < D; d++) {
                    var u = arr[base + head * D + d] - arr[base + tail * D + d];
                    if (u !== u) { ok = false; break; }
                    s += u * u;
                }
                if (ok && s > 0) vals.push(Math.sqrt(s));
            }
        }
        var bl = median(vals);
        if (bl > 0) return bl;
        vals = [];
        for (var t2 = 0; t2 < T; t2 += step) {
            var b2 = t2 * N * D, c = [0, 0, 0], n = 0;
            body.used.forEach(function (k) {
                if (arr[b2 + k * D] !== arr[b2 + k * D]) return;
                for (var d2 = 0; d2 < D; d2++) c[d2] += arr[b2 + k * D + d2];
                n++;
            });
            if (n < 2) continue;
            var far = 0;
            body.used.forEach(function (k) {
                if (arr[b2 + k * D] !== arr[b2 + k * D]) return;
                var s2 = 0;
                for (var d3 = 0; d3 < D; d3++) { var w = arr[b2 + k * D + d3] - c[d3] / n; s2 += w * w; }
                far = Math.max(far, Math.sqrt(s2));
            });
            if (far > 0) vals.push(2 * far);
        }
        bl = median(vals);
        return bl > 0 ? bl : 1;
    });
}

// ----------------------------------------------------------------------------
// Broken tracking
// ----------------------------------------------------------------------------

/**
 * Bumped whenever what `brokenPoseFrames` skips changes: a run records it
 * (pose/behavior-predictions.js `runConfigPrint`), so runs made before go stale.
 */
export var POSE_CHECK_VERSION = 1;

/**
 * A used node farther than this from the body's middle, in body lengths, is
 * off the body. On four real recordings the farthest used node of 99% of
 * poses sat within 0.86–1.88 BL of the middle, with a thin tail of broken
 * poses running out to hundreds.
 */
export var BROKEN_NODE_BL = 1.2;

/**
 * The middle moving more than this many body lengths between frames, and
 * faster than `BROKEN_SPEED_BL_S`, is a jump no animal makes (a mouse
 * sprinting covers about 10 BL/s). The step limit keeps jitter at high frame
 * rates from counting.
 */
export var BROKEN_STEP_BL = 0.5;
export var BROKEN_SPEED_BL_S = 20;

/** Past this gap (seconds) between two frames with a middle, no jump is judged. */
var BROKEN_GAP_S = 1;

/** The median of the first `n` values of `vals`, sorted in place (n is small). */
function medianFirst(vals, n) {
    for (var i = 1; i < n; i++) {
        var v = vals[i], j = i - 1;
        while (j >= 0 && vals[j] > v) { vals[j + 1] = vals[j]; j--; }
        vals[j + 1] = v;
    }
    var h = n >> 1;
    return n % 2 ? vals[h] : (vals[h - 1] + vals[h]) / 2;
}

/**
 * The animal-frames whose tracking is clearly broken, for the features to
 * skip. A frame is broken when a used node sits more than `BROKEN_NODE_BL`
 * body lengths from the body's middle — the per-axis median of the used nodes,
 * so one stray node cannot drag it; it needs at least half of them — or when
 * that middle jumps more than `BROKEN_STEP_BL` body lengths, faster than
 * `BROKEN_SPEED_BL_S`, from the last frame that had one. Around a jump, every
 * frame a speed's central difference (± `SPEED_HALF_SECONDS`) could span
 * across it is skipped too, so no speed straddles it.
 * @param {Object} input from `collectBehaviorInput`
 * @param {Object} body from `resolveBody`
 * @param {number[]} bodyLength per animal, in data units (`bodyLengths`)
 * @param {number} fps
 * @returns {{masks: Uint8Array[], far: number[], jumps: number[], frames: number[], total: number}}
 *   `masks[a][t]` is 1 to skip; per animal, the frames with a node off the
 *   body, the jumps, and the frames skipped in all
 */
export function brokenPoseFrames(input, body, bodyLength, fps) {
    var N = input.nodeNames.length, D = input.dim, T = input.numFrames;
    var used = (body && body.used) || [];
    var rate = fps > 0 ? fps : 30;
    var k = halfFrames(SPEED_HALF_SECONDS, rate);
    var maxGap = Math.max(1, Math.round(BROKEN_GAP_S * rate));
    var need = Math.max(3, Math.ceil(used.length / 2));
    var present = new Int32Array(used.length), vals = new Float64Array(used.length);
    var mid = new Float64Array(D), prev = new Float64Array(D);
    var out = { masks: [], far: [], jumps: [], frames: [], total: 0 };
    input.data.forEach(function (arr, a) {
        var mask = new Uint8Array(T);
        var BL = bodyLength ? bodyLength[a] : NaN;
        var far = 0, jumps = 0, prevT = -1;
        if (BL > 0 && used.length >= 3) {
            var lim2 = BROKEN_NODE_BL * BL * BROKEN_NODE_BL * BL;
            for (var t = 0; t < T; t++) {
                var base = t * N * D, m = 0;
                for (var i = 0; i < used.length; i++) {
                    var o = base + used[i] * D;
                    if (arr[o] === arr[o]) present[m++] = used[i];
                }
                if (m < need) continue;
                for (var d = 0; d < D; d++) {
                    for (var j = 0; j < m; j++) vals[j] = arr[base + present[j] * D + d];
                    mid[d] = medianFirst(vals, m);
                }
                // A node off the body.
                for (var q = 0; q < m; q++) {
                    var s = 0;
                    for (var d2 = 0; d2 < D; d2++) { var u = arr[base + present[q] * D + d2] - mid[d2]; s += u * u; }
                    if (s > lim2) { mask[t] = 1; far++; break; }
                }
                // A jump of the middle from the last frame that had one: skip
                // [prevT − k + 1, t + k − 1], every frame a central difference
                // over ±k could pair across it.
                if (prevT >= 0 && t - prevT <= maxGap) {
                    var s2 = 0;
                    for (var d3 = 0; d3 < D; d3++) { var w = mid[d3] - prev[d3]; s2 += w * w; }
                    var step = Math.sqrt(s2);
                    if (step > BROKEN_STEP_BL * BL && step > BROKEN_SPEED_BL_S * BL * (t - prevT) / rate) {
                        jumps++;
                        for (var f = Math.max(0, prevT - k + 1); f <= Math.min(T - 1, t + k - 1); f++) mask[f] = 1;
                    }
                }
                prevT = t;
                for (var d4 = 0; d4 < D; d4++) prev[d4] = mid[d4];
            }
        }
        var frames = 0;
        for (var g = 0; g < T; g++) frames += mask[g];
        out.masks.push(mask); out.far.push(far); out.jumps.push(jumps); out.frames.push(frames);
        out.total += frames;
    });
    return out;
}

// ----------------------------------------------------------------------------
// Floor
// ----------------------------------------------------------------------------

function norm(v) {
    var l = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    return l > 0 ? [v[0] / l, v[1] / l, v[2] / l] : null;
}
function cross(a, b) {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function dot(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/**
 * A floor from its up direction and a point on it, with horizontal axes built
 * from `up` alone (e1 × e2 = up). No feature reads e1 or e2 themselves.
 * @returns {{origin:number[], up:number[], e1:number[], e2:number[]}|null}
 */
export function floorFrame(origin, up, extra) {
    var u = up && norm(up);
    if (!u || !origin || !origin.every(Number.isFinite)) return null;
    var ax = [Math.abs(u[0]), Math.abs(u[1]), Math.abs(u[2])];
    var a = ax[0] <= ax[1] && ax[0] <= ax[2] ? [1, 0, 0] : ax[1] <= ax[2] ? [0, 1, 0] : [0, 0, 1];
    var d = dot(a, u);
    var e1 = norm([a[0] - d * u[0], a[1] - d * u[1], a[2] - d * u[2]]);
    var e2 = cross(u, e1);
    return Object.assign({ origin: origin.slice(0, 3), up: u, e1: e1, e2: e2 }, extra || {});
}

/**
 * The body-axis cue for up: the second moments of the unit head → tail base
 * vectors (sampled frames, every animal); level bodies make their smallest
 * eigenvector vertical. `ratio` is the smallest eigenvalue over the middle
 * one (small = clear).
 * @returns {{normal:number[], ratio:number, samples:number}|null}
 */
function bodyAxisNormal(input, head, tail) {
    if (!(head >= 0) || !(tail >= 0) || head === tail || input.dim !== 3) return null;
    var N = input.nodeNames.length, T = input.numFrames;
    var step = Math.max(1, Math.floor(T * input.data.length / 60000));
    var M = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], n = 0, k = 0;
    input.data.forEach(function (arr) {
        for (var t = 0; t < T; t++, k++) {
            if (k % step) continue;
            var h = (t * N + head) * 3, b = (t * N + tail) * 3;
            var u = [arr[h] - arr[b], arr[h + 1] - arr[b + 1], arr[h + 2] - arr[b + 2]];
            var len = Math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
            if (!(len > 0)) continue;
            for (var r = 0; r < 3; r++) for (var c = 0; c < 3; c++) M[r][c] += u[r] * u[c] / (len * len);
            n++;
        }
    });
    if (n < 30) return null;
    M = M.map(function (row) { return row.map(function (v) { return v / n; }); });
    var eig = jacobiEigen(M);
    var idx = [0, 1, 2].sort(function (a, b) { return eig.eigenvalues[a] - eig.eigenvalues[b]; });
    var normal = norm(eig.eigenvectors[idx[0]]);
    if (!normal) return null;
    return { normal: normal, ratio: eig.eigenvalues[idx[0]] / Math.max(1e-30, eig.eigenvalues[idx[1]]), samples: n };
}

/**
 * The body-axis cue is used when its smallest eigenvalue is at most this
 * share of the middle one (bodies level most of the time; rearing raises it).
 */
var BODY_AXIS_MAX_RATIO = 0.6;

/** Sampled used-node points of every animal (up to `max`), as a flat Float64Array. */
function samplePoints(input, used, max) {
    var N = input.nodeNames.length, T = input.numFrames;
    var total = T * input.data.length * Math.max(1, used.length);
    var step = Math.max(1, Math.ceil(total / max));
    var out = [];
    var k = 0;
    input.data.forEach(function (arr) {
        for (var t = 0; t < T; t++) {
            for (var i = 0; i < used.length; i++, k++) {
                if (k % step) continue;
                var o = (t * N + used[i]) * 3;
                var x = arr[o], y = arr[o + 1], z = arr[o + 2];
                if (x === x && y === y && z === z) out.push(x, y, z);
            }
        }
    });
    return Float64Array.from(out);
}

/** The sign that turns `normal` toward the animals: most sampled points lie on its positive side. */
function towardAnimals(pts, origin, normal) {
    var above = 0, below = 0;
    for (var i = 0; i < pts.length; i += 3) {
        var h = (pts[i] - origin[0]) * normal[0] + (pts[i + 1] - origin[1]) * normal[1] + (pts[i + 2] - origin[2]) * normal[2];
        if (h > 0) above++; else if (h < 0) below++;
    }
    return below > above ? -1 : 1;
}

/** A floor from a fitted plane `{centroid, normal}`, its normal turned toward the animals. */
export function floorFromPlane(fit, input, used) {
    if (!fit || !fit.centroid || !fit.normal) return null;
    var n = norm(fit.normal);
    if (!n) return null;
    var s = input ? towardAnimals(samplePoints(input, used || [], 20000), fit.centroid, n) : 1;
    return floorFrame(fit.centroid, [s * n[0], s * n[1], s * n[2]], { source: 'plane' });
}

/** A floor from the Set Origin frame: its origin, +Z up. */
export function floorFromOrigin(frame) {
    if (!frame || !frame.origin || !frame.zAxis) return null;
    return floorFrame(frame.origin, frame.zAxis, { source: 'origin' });
}

function percentileSorted(sorted, q) {
    if (!sorted.length) return NaN;
    var i = Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))));
    return sorted[i];
}

/**
 * Least squares q = α a + β b + γ, refit once without the worst 20%.
 * @returns {{alpha, beta, gamma, rms}|null}
 */
function fitEnvelope(cells) {
    function solve(list) {
        if (list.length < 3) return null;
        var Saa = 0, Sab = 0, Sbb = 0, Sa = 0, Sb = 0, S1 = 0, Saq = 0, Sbq = 0, Sq = 0;
        list.forEach(function (c) {
            Saa += c[0] * c[0]; Sab += c[0] * c[1]; Sbb += c[1] * c[1]; Sa += c[0]; Sb += c[1]; S1 += 1;
            Saq += c[0] * c[2]; Sbq += c[1] * c[2]; Sq += c[2];
        });
        var M = [[Saa, Sab, Sa], [Sab, Sbb, Sb], [Sa, Sb, S1]], r = [Saq, Sbq, Sq];
        var det = M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
            M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
        if (!(Math.abs(det) > 1e-300)) return null;
        var col = function (j) {
            var A = M.map(function (row, i) { var c = row.slice(); c[j] = r[i]; return c; });
            return (A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])) / det;
        };
        return { alpha: col(0), beta: col(1), gamma: col(2) };
    }
    var p = solve(cells);
    if (!p) return null;
    var res = cells.map(function (c) { return Math.abs(c[2] - (p.alpha * c[0] + p.beta * c[1] + p.gamma)); });
    var cut = res.slice().sort(function (a, b) { return a - b; })[Math.floor(0.8 * (res.length - 1))];
    var kept = cells.filter(function (c, i) { return res[i] <= cut; });
    var p2 = solve(kept) || p;
    var ss = 0;
    kept.forEach(function (c) { var e = c[2] - (p2.alpha * c[0] + p2.beta * c[1] + p2.gamma); ss += e * e; });
    p2.rms = Math.sqrt(ss / Math.max(1, kept.length));
    return p2;
}

/**
 * The floor as the animals' lower envelope (no floor was given), from `nodes`
 * — all of them, excluded tail nodes included: the lowest points are often the
 * tail and the paws, which are what touch the floor.
 *
 * Its tilt comes first from the BODIES: quadrupeds spend most of their time
 * roughly level, so the direction most perpendicular to their head–tail base
 * axes is up, whatever the arena's shape (`opts.head` / `opts.tailBase`; the
 * smallest eigenvector of the axes' second moments, used when clearly the
 * smallest). Without both roles, or when the bodies point every way, it falls
 * back to the points' least spread (PCA) — which a small or tall cage can
 * fool: on a proofread SLAP home cage, where the mice spread about as much
 * up as across, it put "up" 75° off. The points' extent along
 * the other two is cut into a grid, and in each cell the 5th percentile of
 * height on each side gives two envelopes, each fitted with a plane
 * (trimmed least squares). The floor is the flatter one — the lowest points
 * (paws, tail base, a nose sniffing the floor) sit on it everywhere, while the
 * top of the bodies varies with posture. Which side is the floor is decided,
 * in order, by the cameras (`opts.cameraCenters`: rigs film the floor from
 * above, so at least 70% of them on one side settle it), by the way the
 * heights trail (bodies rise from the floor by rearing and climbing, never
 * sink into it: skewness of at least 0.25), then by which envelope is flatter.
 * Every step reads the data (and the cameras) only, so the result moves with
 * the world.
 *
 * @param {Object} input from `collectBehaviorInput` (3D)
 * @param {number[]} nodes node indices to read
 * @param {{cameraCenters?: number[][], head?: number, tailBase?: number}} [opts]
 * @returns {{origin, up, e1, e2, source:'fit', confidence:'ok'|'low', points:number,
 *            tiltFrom:'body axes'|'spread', upFrom:'cameras'|'heights'|'flatness'|'one envelope'}|null}
 */
export function fitFloor(input, nodes, opts) {
    if (input.mode !== '3d') return null;
    var pts = samplePoints(input, nodes, 120000);
    var n = pts.length / 3;
    if (n < 30) return null;
    var mu = [0, 0, 0];
    for (var i = 0; i < pts.length; i += 3) { mu[0] += pts[i]; mu[1] += pts[i + 1]; mu[2] += pts[i + 2]; }
    mu = mu.map(function (v) { return v / n; });
    var C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (var j = 0; j < pts.length; j += 3) {
        var d = [pts[j] - mu[0], pts[j + 1] - mu[1], pts[j + 2] - mu[2]];
        for (var r = 0; r < 3; r++) for (var c = 0; c < 3; c++) C[r][c] += d[r] * d[c];
    }
    var tr = C[0][0] + C[1][1] + C[2][2];
    if (!(tr > 0)) return null;
    C = C.map(function (row) { return row.map(function (v) { return v / tr; }); });
    var eig = jacobiEigen(C);
    var idx = [0, 1, 2].sort(function (a, b) { return eig.eigenvalues[a] - eig.eigenvalues[b]; });
    var n0 = norm(eig.eigenvectors[idx[0]]);
    var spreadRatio = eig.eigenvalues[idx[0]] / Math.max(1e-30, eig.eigenvalues[idx[1]]);
    var tiltFrom = 'spread';
    var axes = bodyAxisNormal(input, opts && opts.head, opts && opts.tailBase);
    if (axes && axes.ratio < BODY_AXIS_MAX_RATIO) { n0 = axes.normal; tiltFrom = 'body axes'; }
    // The grid's first axis: the points' widest direction along the floor.
    var wide = eig.eigenvectors[idx[2]];
    var dw = wide[0] * n0[0] + wide[1] * n0[1] + wide[2] * n0[2];
    var e1 = norm([wide[0] - dw * n0[0], wide[1] - dw * n0[1], wide[2] - dw * n0[2]]);
    if (!e1) { var mid = eig.eigenvectors[idx[1]], dm = mid[0] * n0[0] + mid[1] * n0[1] + mid[2] * n0[2]; e1 = norm([mid[0] - dm * n0[0], mid[1] - dm * n0[1], mid[2] - dm * n0[2]]); }
    if (!n0 || !e1) return null;
    var e2 = cross(n0, e1);
    var A = new Float64Array(n), B = new Float64Array(n), H = new Float64Array(n);
    for (var k = 0; k < n; k++) {
        var dx = pts[3 * k] - mu[0], dy = pts[3 * k + 1] - mu[1], dz = pts[3 * k + 2] - mu[2];
        A[k] = dx * e1[0] + dy * e1[1] + dz * e1[2];
        B[k] = dx * e2[0] + dy * e2[1] + dz * e2[2];
        H[k] = dx * n0[0] + dy * n0[1] + dz * n0[2];
    }
    var sa = Array.from(A).sort(function (x, y) { return x - y; }), sb = Array.from(B).sort(function (x, y) { return x - y; });
    var aLo = percentileSorted(sa, 0.01), aHi = percentileSorted(sa, 0.99);
    var bLo = percentileSorted(sb, 0.01), bHi = percentileSorted(sb, 0.99);
    var K = 8;
    var cellsOf = function () { var o = []; for (var q = 0; q < K * K; q++) o.push([]); return o; };
    var cellsPts = cellsOf();
    for (var m = 0; m < n; m++) {
        if (A[m] < aLo || A[m] > aHi || B[m] < bLo || B[m] > bHi) continue;
        var ci = Math.min(K - 1, Math.floor((A[m] - aLo) / ((aHi - aLo) || 1) * K));
        var cj = Math.min(K - 1, Math.floor((B[m] - bLo) / ((bHi - bLo) || 1) * K));
        cellsPts[ci * K + cj].push(m);
    }
    var envelopes = [1, -1].map(function (side) {
        var cells = [];
        cellsPts.forEach(function (list) {
            if (list.length < 20) return;
            var hs = list.map(function (q) { return side * H[q]; }).sort(function (x, y) { return x - y; });
            var as = 0, bs = 0;
            list.forEach(function (q) { as += A[q]; bs += B[q]; });
            cells.push([as / list.length, bs / list.length, percentileSorted(hs, 0.05)]);
        });
        return { side: side, cells: cells.length, fit: fitEnvelope(cells) };
    });
    var lo = envelopes[0], hi = envelopes[1];
    var pick = null, why = '';
    if (lo.fit && hi.fit) {
        // 1. Cameras look at the floor from above: most of them are on the up side.
        var cams = (opts && opts.cameraCenters) || [];
        var camUp = 0, camDown = 0;
        cams.forEach(function (c) {
            if (!c || !c.every(Number.isFinite)) return;
            var h = (c[0] - mu[0]) * n0[0] + (c[1] - mu[1]) * n0[1] + (c[2] - mu[2]) * n0[2];
            if (h > 0) camUp++; else if (h < 0) camDown++;
        });
        var camTotal = camUp + camDown;
        // 2. Bodies rise from the floor (rearing, climbing), never sink into it:
        //    the heights trail toward up.
        var m2 = 0, m3 = 0;
        for (var s = 0; s < n; s++) { m2 += H[s] * H[s]; m3 += H[s] * H[s] * H[s]; }
        var skew = m2 > 0 ? (m3 / n) / Math.pow(m2 / n, 1.5) : 0;
        // 3. The floor envelope is the flatter one: the lowest points sit on it everywhere.
        var ratio = hi.fit.rms / Math.max(1e-12, lo.fit.rms);
        if (camTotal >= 2 && Math.max(camUp, camDown) >= 0.7 * camTotal) { pick = camUp > camDown ? lo : hi; why = 'cameras'; }
        else if (Math.abs(skew) >= 0.25) { pick = skew > 0 ? lo : hi; why = 'heights'; }
        else if (ratio > 1.5 || ratio < 2 / 3) { pick = ratio > 1 ? lo : hi; why = 'flatness'; }
        else { pick = skew >= 0 ? lo : hi; why = 'heights'; }
    } else if (lo.fit || hi.fit) { pick = lo.fit ? lo : hi; why = 'one envelope'; }
    if (!pick) return null;
    var f = pick.fit, sd = pick.side;
    // side·h = α a + β b + γ  =>  (p − μ)·w = γ with w = side·n0 − α e1 − β e2.
    var w = [sd * n0[0] - f.alpha * e1[0] - f.beta * e2[0], sd * n0[1] - f.alpha * e1[1] - f.beta * e2[1],
        sd * n0[2] - f.alpha * e1[2] - f.beta * e2[2]];
    var ww = dot(w, w);
    var origin = [mu[0] + f.gamma / ww * w[0], mu[1] + f.gamma / ww * w[1], mu[2] + f.gamma / ww * w[2]];
    var clear = tiltFrom === 'body axes' || spreadRatio <= 0.25;
    var confidence = pick.cells >= 6 && clear ? 'ok' : 'low';
    return floorFrame(origin, w, { source: 'fit', confidence: confidence, points: n, upFrom: why, tiltFrom: tiltFrom });
}

// ----------------------------------------------------------------------------
// Spec
// ----------------------------------------------------------------------------

/**
 * The feature spec pose/behavior-features.js computes from.
 * @param {Object} input from `collectBehaviorInput`
 * @param {Object} body from `resolveBody`
 * @param {Object|null} floor from `floorFrame` / `fitFloor` (ignored in 2D)
 * @param {number} fps
 */
export function featureSpec(input, body, floor, fps) {
    return {
        mode: input.mode, dim: input.dim, numFrames: input.numFrames,
        nodeNames: input.nodeNames.slice(), edges: input.edges.map(function (e) { return e.slice(); }),
        roles: { head: body.roles.head, tailBase: body.roles.tailBase },
        used: body.used.slice(), paws: body.paws.slice(),
        fps: fps > 0 ? fps : 30,
        floor: input.mode === '3d' && floor ? { origin: floor.origin, up: floor.up, e1: floor.e1, e2: floor.e2 } : null,
        bodyLength: bodyLengths(input, body),
        animals: input.animals.slice(),
    };
}
