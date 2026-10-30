// pose/threshold-examples.js — the session's own nodes, for the reprojection
// error threshold card in Settings ▸ Default Triangulation
// (ui/threshold-examples-panel.js). DOM-free; Node-tested
// (tests/test-threshold-examples.mjs).
//
// `scanThresholdExamples` solves a sample of the session's frames with the
// threshold off and lists every node at least 3 cameras see, smallest worst
// camera error first. That one number decides a threshold: a node whose worst
// error is over it has a camera dropped, so the list splits in two at
// `firstOverThreshold`, and the slider only has to move an index.
//
// `solveExample` then runs the threshold's rule on one node, solve by solve, so a
// view can show what each pass did. It solves through `triangulateAndReproject`
// with the same observation mask the solver's own loop applies
// (`includedCameras`, on a one-node view of the group), so it cannot tell a
// different story from a real Triangulate. Nothing here writes to the session.
//
// Depends on: pose/triangulation-core.js (triangulateAndReproject),
// pose/lazy-residency.js (hydrateFrameMembers2d, releaseFrameMembers2d),
// pose/placeholder-cameras.js (isPlaceholderCamera).

import { triangulateAndReproject } from './triangulation-core.js';
import { hydrateFrameMembers2d, releaseFrameMembers2d } from './lazy-residency.js';
import { isPlaceholderCamera } from './placeholder-cameras.js';

/** Frames solved per scan, at most, spread evenly over the recording. */
export const EXAMPLE_MAX_FRAMES = 240;
/** Fewer cameras than this can only keep a node or lose it, never drop a camera. */
export const EXAMPLE_MIN_CAMERAS = 3;
/** A scan hands the thread back after this long. */
const SCAN_SLICE_MS = 12;

/** The session's calibrated cameras: only these can solve anything. */
export function calibratedCameras(session) {
    return ((session && session.cameras) || []).filter(function (c) {
        return c && c.projectionMatrix && !isPlaceholderCamera(c);
    });
}

/** Up to `max` frames with groups, in order, spread evenly over those frames. */
export function sampleExampleFrames(session, max) {
    var frames = [];
    if (session && session.instanceGroups) {
        session.instanceGroups.forEach(function (groups, frame) {
            if (groups && groups.length) frames.push(frame);
        });
    }
    frames.sort(function (a, b) { return a - b; });
    var limit = max || EXAMPLE_MAX_FRAMES;
    if (frames.length <= limit) return frames;
    var out = [];
    for (var i = 0; i < limit; i++) out.push(frames[Math.floor(i * frames.length / limit)]);
    return out;
}

/** Index of the first example whose worst error is over `threshold`; `examples.length` if none. */
export function firstOverThreshold(examples, threshold) {
    var a = 0, b = examples.length;
    while (a < b) {
        var m = (a + b) >> 1;
        if (examples[m].worst > threshold) b = m; else a = m + 1;
    }
    return a;
}

function bracketFrame(session, frame, read) {
    var hydrated = hydrateFrameMembers2d(session, frame) > 0;
    try {
        return read();
    } finally {
        if (hydrated) releaseFrameMembers2d(session, frame);
    }
}

function scanFrame(session, frame, cameras, included, method, out) {
    bracketFrame(session, frame, function () {
        (session.instanceGroups.get(frame) || []).forEach(function (group, groupIdx) {
            var result = triangulateAndReproject(group, cameras,
                { method: method, reprojErrorThreshold: 0, includedCameras: included });
            var worst = [], worstCam = [], seen = [];
            included.forEach(function (name) {
                var errs = result.errors[name];
                if (!errs) return;
                for (var k = 0; k < errs.length; k++) {
                    var e = errs[k];
                    if (e == null || !isFinite(e)) continue;
                    seen[k] = (seen[k] || 0) + 1;
                    if (!(worst[k] >= e)) { worst[k] = e; worstCam[k] = name; }
                }
            });
            for (var k = 0; k < worst.length; k++) {
                if (seen[k] >= EXAMPLE_MIN_CAMERAS) {
                    out.push({ frame: frame, group: groupIdx, node: k, worst: worst[k], worstCamera: worstCam[k] });
                }
            }
        });
    });
}

/**
 * Solve a sample of the session with the threshold off, a slice per task.
 *
 * @param {Object} session
 * @param {Object} opts
 * @param {'dlt'|'ba'} opts.method
 * @param {string[]} [opts.included]  cameras that may contribute (default: all calibrated)
 * @param {number} [opts.maxFrames]
 * @param {function(number, number): void} [opts.onProgress]  (frames done, frames)
 * @param {function(): boolean} [opts.isCancelled]
 * @returns {Promise<?{examples: Object[], frames: number}>} smallest worst error
 *   first; null if cancelled. An example: `{ frame, group, node, worst, worstCamera }`.
 */
export async function scanThresholdExamples(session, opts) {
    var cameras = calibratedCameras(session);
    var included = opts.included || cameras.map(function (c) { return c.name; });
    var frames = sampleExampleFrames(session, opts.maxFrames);
    var examples = [];
    var cancelled = opts.isCancelled || function () { return false; };
    if (cameras.length >= EXAMPLE_MIN_CAMERAS) {
        var sliceStart = Date.now();
        for (var i = 0; i < frames.length; i++) {
            if (cancelled()) return null;
            scanFrame(session, frames[i], cameras, included, opts.method, examples);
            if (Date.now() - sliceStart >= SCAN_SLICE_MS) {
                if (opts.onProgress) opts.onProgress(i + 1, frames.length);
                await new Promise(function (r) { setTimeout(r, 0); });
                sliceStart = Date.now();
            }
        }
    }
    if (cancelled()) return null;
    examples.sort(function (a, b) { return a.worst - b.worst; });
    return { examples: examples, frames: frames.length };
}

// The group as the solver reads it, holding one node: the same 2D, the same
// mask, a solve that costs one node.
function oneNodeView(group, k) {
    var views = new Map();
    group.instances.forEach(function (inst, cam) {
        var on = !!inst && inst.hasPoint(k) && !(inst.nulledNodes && inst.nulledNodes.has(k));
        views.set(cam, {
            numNodes: 1,
            nulledNodes: null,
            hasPoint: function () { return on; },
            getPoint: function () { return inst.getPoint(k); },
        });
    });
    return { getInstance: function (cam) { return views.get(cam) || null; } };
}

function isOver(error, threshold) { return threshold > 0 && error > threshold; }

/**
 * Run the threshold's rule on one example: after each solve, while more than two
 * cameras are left, drop only the worst camera over the threshold and solve again;
 * if the last cameras are still over, the node has no 3D.
 *
 * @param {Object} session
 * @param {{frame: number, group: number, node: number}} example
 * @param {Object} opts
 * @param {'dlt'|'ba'} opts.method
 * @param {number} opts.threshold  0 = off
 * @param {string[]} [opts.included]
 * @returns {?{
 *   skeletons: Map<string, Array>, predictions: Map<string, ?number[]>,
 *   passes: Array<{cameras: string[], errors: Object, reprojections: Object, drop: ?string}>,
 *   dropped: string[], lost: boolean, identityId: number
 * }} null if the example's group is gone. `skeletons` holds every node's 2D per
 *   camera, copied, so a lazy frame's members can go back to the store.
 */
export function solveExample(session, example, opts) {
    var cameras = calibratedCameras(session);
    var included = opts.included || cameras.map(function (c) { return c.name; });
    var method = opts.method;
    return bracketFrame(session, example.frame, function () {
        var group = (session.instanceGroups.get(example.frame) || [])[example.group];
        if (!group) return null;
        var k = example.node;
        var view = oneNodeView(group, k);
        var skeletons = new Map(), predictions = new Map();
        group.instances.forEach(function (inst, cam) {
            var pts = [];
            for (var n = 0; inst && n < inst.numNodes; n++) {
                var p = inst.hasPoint(n) ? inst.getPoint(n) : null;
                pts.push(p ? [p[0], p[1]] : null);
            }
            skeletons.set(cam, pts);
            var mine = view.getInstance(cam);
            predictions.set(cam, mine.hasPoint() ? pts[k] : null);
        });
        var kept = included.filter(function (name) { return predictions.get(name); });
        var passes = [], dropped = [], lost = false;
        for (var guard = 0; guard <= cameras.length; guard++) {
            var result = triangulateAndReproject(view, cameras,
                { method: method, reprojErrorThreshold: 0, includedCameras: kept });
            var errors = {}, reprojections = {};
            cameras.forEach(function (c) {
                var err = result.errors[c.name] ? result.errors[c.name][0] : null;
                var rep = result.reprojections[c.name] ? result.reprojections[c.name][0] : null;
                errors[c.name] = err == null ? null : err;
                reprojections[c.name] = rep ? [rep[0], rep[1]] : null;
            });
            var pass = { cameras: kept.slice(), errors: errors, reprojections: reprojections, drop: null };
            passes.push(pass);
            var solved = kept.some(function (name) { return errors[name] != null; });
            if (!solved) { lost = true; break; }
            var worst = null;
            kept.forEach(function (name) {
                if (errors[name] != null && isOver(errors[name], opts.threshold)
                    && (worst === null || errors[name] > errors[worst])) worst = name;
            });
            if (worst === null) break;
            if (kept.length <= 2) { lost = true; break; }
            pass.drop = worst;
            dropped.push(worst);
            kept = kept.filter(function (name) { return name !== worst; });
        }
        return {
            skeletons: skeletons, predictions: predictions, passes: passes,
            dropped: dropped, lost: lost, identityId: group.identityId,
        };
    });
}
