// pose/behavior-predictions.js — predicted behavior bouts, per session: what
// the rule detectors, the classifiers and promoted states (keypoint-MoSeq
// syllables, pose/behavior-states.js) say each animal did, kept for review.
//
// ## Storage
//
// Per session (a WeakMap, like the labels), one RUN per behavior — the last
// time that behavior was detected:
//
//   { behaviorId, source: 'rule' | 'model:<hash>' | 'states:<layer>' | 'file',
//     cfg,               // a print of what produced it: the detector (a model
//                        // run: its post-processing and exclusive group), the
//                        // body and floor settings, the frame rate, 2D or 3D;
//                        // a states run: its layer's runs and its states
//     states,            // a states run: the states promoted (sorted), else null
//     labels,            // a model run: the print of the labels its model
//                        // learned from (pose/behavior-classifier.js
//                        // `trainingLabelPrint`), or null (an imported model)
//     fps, mode, camera,
//     bouts: [{animal, target, start, end (inclusive), score}],   sorted
//     scores: Map<'animal\u0001target', Float32Array> | null }    per frame, in memory only
//
// Bouts are saved (as events in a `.slp`, in `metadata.lucid.behaviorPredictions`
// in a project JSON, through import-export/behavior-metadata.js and
// behavior-events.js, only when there are any); per-frame scores are not — they
// are recomputed by running again. A MUTUAL behavior's pair is stored in name
// order, as labels are. A 'file' run is one another tool wrote into the file: it
// is reviewed like any run and never goes stale.
//
// ## Stale
//
// A run is stale once what produced it changed:
//   * settings — its detector, the body it measured (the head, tail base and
//     nodes used, as resolved on the session's skeleton), the floor in effect
//     (the caller's `floorKey`: the Set Origin frame or plane fit it used), or
//     the frame rate (`runConfigPrint` differs from the run's `cfg`). A frame
//     rate nothing has said is no change (`env.fpsUnknown`: a `.slp` does not
//     save it, so a project reopened without its videos only holds a default):
//     the run is checked at the rate it recorded until something says;
//   * input — the poses it read: the session's identified 3D (or, for a 2D
//     run, its camera's 2D) no longer has the print taken when it ran
//     (`predictionInputPrint`: an order-free hash of every value, its frame and
//     its identity, so any edit, re-solve, smoothing or identity fix shows).
//   * states (a states run) — its layer was replaced, changed or removed
//     (the layer's print is in `cfg`; the states' names are not);
//   * labels (a classifier run only) — the labels its model learned from
//     changed (`env.labelPrintOf(behaviorId)` differs from the run's `labels`),
//     so a retrain would change it: Accept and Reject feed the next training;
//   * model (a classifier run only) — a newer model replaced the one that
//     made it (`env.modelHashOf(behaviorId)` differs from the run's hash; a
//     project reopened has no model until it is retrained, and retraining on
//     the same labels gives the same hash).
// Label edits do not make a rule run stale (rules do not read labels).
// Stale predictions stay visible, marked, until run again. Checking the input
// costs one pass over the poses, so it is done only when something may have
// changed (`notePossibleInputChange`, from markDirty) and is cached in
// between.
//
// ## Review
//
// A predicted bout is reviewed by labelling it: Accept labels its frames
// present (source 'a', accepted prediction), Reject labels them "not" — each
// one `labelBouts` edit, so one undo step. Its review status is read back
// from the labels, never stored.
//
// DOM-free.
//
// Depends on: pose/behavior-ethogram.js, pose/behavior-labels.js, pose/behavior-tensor.js,
//             pose/behavior-states.js, pose/pose-data.js

import { behaviorById, featureSettings, behaviorPost, exclusivePeers } from './behavior-ethogram.js';
import { boutIntervals, labelBouts, behaviorFps } from './behavior-labels.js';
import { resolveBody, POSE_CHECK_VERSION } from './behavior-tensor.js';
import { stateLayer, layerPrint, cleanLayerName } from './behavior-states.js';
import { FEATURE_SPEC_VERSION } from './behavior-features.js';
import { POINT3D_STRIDE, points3dNodeCount } from './pose-data.js';

var SEP = '\u0001';

/** Is `s` a run source: 'rule', 'model:' and a forest's 16-hex-digit hash, or 'states:' and a layer name? */
export function isRunSource(s) {
    return s === 'rule' || s === 'file' || (typeof s === 'string' && (/^model:[0-9a-f]{16}$/.test(s) ||
        (s.indexOf('states:') === 0 && cleanLayerName(s.slice(7)) === s.slice(7) && s.length > 7)));
}

/** A states run's layer name ('' for any other run). */
export function runStateLayer(run) {
    var s = run && run.source;
    return typeof s === 'string' && s.indexOf('states:') === 0 ? s.slice(7) : '';
}

/** States as a run keeps them: unique non-negative integers, sorted. */
function stateList(states) {
    var out = [];
    (Array.isArray(states) ? states : []).forEach(function (k) {
        var v = Number(k);
        if (Number.isInteger(v) && v >= 0 && out.indexOf(v) < 0) out.push(v);
    });
    return out.sort(function (a, b) { return a - b; });
}

/** A model run's forest hash ('' for a rule run). */
export function runModelHash(run) {
    return run && typeof run.source === 'string' && run.source.indexOf('model:') === 0 ? run.source.slice(6) : '';
}

/** The threshold a run's scores are decided at: its behavior's for a classifier run, 0.5 for a rule. */
export function runThreshold(run) {
    if (!runModelHash(run)) return 0.5;
    return behaviorPost(behaviorById(run.behaviorId)).threshold;
}

var _stores = new WeakMap();
var _listeners = [];

function storeOf(session) {
    if (!session) return null;
    var st = _stores.get(session);
    if (!st) {
        st = { runs: new Map(), print: null, inputStale: false, checkPending: false };
        _stores.set(session, st);
    }
    return st;
}

/** Be told after every change to a session's predictions (`fn(session, info)`). */
export function onPredictionsChange(fn) {
    if (typeof fn === 'function') _listeners.push(fn);
}

function emit(session, info) {
    for (var i = 0; i < _listeners.length; i++) {
        try { _listeners[i](session, info || {}); } catch (e) { console.error('[behavior] prediction listener failed', e); }
    }
}

/** A pair as stored: a mutual behavior's in name order. */
function orderPair(behaviorId, animal, target) {
    var b = behaviorById(behaviorId);
    if (b && b.mutual && target && target < animal) return [target, animal];
    return [animal, target || ''];
}

function keyOf(animal, target) {
    return animal + SEP + (target || '');
}

// ----------------------------------------------------------------------------
// Prints
// ----------------------------------------------------------------------------

function fnv(str) {
    var h = 0x811c9dc5;
    for (var i = 0; i < str.length; i++) h = Math.imul(h ^ str.charCodeAt(i), 0x01000193);
    return (h >>> 0).toString(16);
}

/** The body features are measured on, as resolved on the session's skeleton: [head, tail base, used nodes] by name. */
function bodyPrint(session) {
    var sk = session && session.skeleton;
    var names = sk && sk.nodes ? sk.nodes.map(function (nd) { return typeof nd === 'string' ? nd : nd && nd.name; }) : [];
    var edges = sk && sk.edges ? sk.edges : [];
    var body = resolveBody(names, edges, featureSettings());
    return [names[body.roles.head] || '', names[body.roles.tailBase] || '', body.used.map(function (i) { return names[i]; })];
}

/**
 * What a run of `behavior` depends on besides the poses: its detector, the
 * resolved body, the floor in effect, the frame rate, 2D vs 3D, and which
 * frames count as broken tracking (`POSE_CHECK_VERSION`, so runs from before
 * a change to that check go stale). A classifier run (`env.source`
 * 'model:…') depends on its post-processing and its exclusive group (the
 * peers and theirs) instead of a detector, and on the feature definitions
 * (`FEATURE_SPEC_VERSION`); its model is checked by hash, apart.
 * @param {Object} behavior
 * @param {Session} session
 * @param {{fps:number, mode?:string, camera?:string, floorKey?:string, source?:string}} env
 */
export function runConfigPrint(behavior, session, env) {
    env = env || {};
    if (typeof env.source === 'string' && env.source.indexOf('states:') === 0) {
        // Promoted states: the layer's runs (re-imported or removed: stale) and which states.
        var layer = stateLayer(session, env.source.slice(7));
        return fnv(JSON.stringify(['states', env.source, layer ? layerPrint(layer) : null, stateList(env.states),
            behavior ? [behavior.kind, !!behavior.social, !!behavior.mutual] : null]));
    }
    // The floor in effect (the caller's key); without one, the floor setting.
    var fs = featureSettings();
    var floor = env.mode === '2d' ? '' : env.floorKey != null ? env.floorKey : (fs && fs.floor ? JSON.stringify(fs.floor) : 'auto');
    var model = typeof env.source === 'string' && env.source.indexOf('model:') === 0;
    var what = model
        ? ['model', behaviorPost(behavior), behavior && behavior.group || '',
           exclusivePeers(behavior).map(function (p) { return [p.id, behaviorPost(p)]; }).sort(), FEATURE_SPEC_VERSION]
        : behavior && behavior.detector || null;
    return fnv(JSON.stringify([what, bodyPrint(session), floor,
        Math.round((env.fps || 0) * 1000) / 1000, env.mode || '', env.camera || '',
        behavior ? [behavior.kind, !!behavior.social, !!behavior.mutual] : null, POSE_CHECK_VERSION]));
}

function mix(a, b) {
    var h = Math.imul(a ^ (a >>> 16), 0x85ebca6b) ^ Math.imul((b + 0x9e3779b9) | 0, 0xc2b2ae35);
    h ^= h >>> 13;
    return Math.imul(h, 0x27d4eb2f) >>> 0;
}

/** Groups (3D) or frames (2D) between the yields of the sliced print. */
var PRINT_SLICE_GROUPS = 20000;
var PRINT_SLICE_FRAMES = 4000;

/**
 * The print, as steps: yields between slices of the walk (the async check
 * pauses there), returns the print. Every value folds into an order-free sum
 * and xor, so the slices change nothing.
 */
function* printSteps(session, mode, camera) {
    var sum = 0, xor = 0, n = 0;
    var add = function (frame, id, k, v) {
        var q = Math.round(v * 1000) | 0;
        var h = mix(q, ((frame * 64 + (id & 63)) * 128 + k) | 0);
        sum = (sum + h) >>> 0; xor = (xor ^ h) >>> 0; n++;
    };
    if (!session) return '';
    if (mode !== '2d') {
        var seen = 0;
        for (var [frame, groups] of session.instanceGroups || []) {
            if (!groups) continue;
            for (var i = 0; i < groups.length; i++) {
                var g = groups[i];
                if (!g || !g.points3d || g.identityId == null || g.identityId < 0) continue;
                var pts = g.points3d, cnt = points3dNodeCount(pts) * POINT3D_STRIDE;
                for (var j = 0; j < cnt; j++) { var v = pts[j]; if (v === v) add(frame, g.identityId, j, v); }
                if (++seen % PRINT_SLICE_GROUPS === 0) yield;
            }
        }
        return 'p3:' + n + ':' + sum + ':' + xor;
    }
    var resident = session.frameGroups || new Map();
    var k0 = 0;
    for (var [f, fg] of resident) {
        if (!fg) continue;
        var memberId = new Map();
        var gs = session.instanceGroups ? session.instanceGroups.get(f) : null;
        if (gs) gs.forEach(function (gg) { var inst = gg && gg.getInstance(camera); if (inst && gg.identityId >= 0) memberId.set(inst, gg.identityId); });
        var take = function (inst) {
            if (!inst || !inst._xy) return;
            var id = memberId.has(inst) ? memberId.get(inst)
                : session.getIdentityIdForUnlinkedInstance ? session.getIdentityIdForUnlinkedInstance(camera, inst, f) : null;
            if (id == null || id < 0) return;
            for (var k = 0; k < inst._xy.length; k++) { var x = inst._xy[k]; if (x === x) add(f, id, k, x); }
        };
        fg.getInstances(camera).forEach(take);
        fg.getUnlinkedInstances(camera).forEach(function (u) { take(u && u.instance); });
        if (++k0 % PRINT_SLICE_FRAMES === 0) yield;
    }
    var loader = session.lazyLoader;
    var labels = loader && loader.labelsByCam && loader.labelsByCam.get(camera);
    var store = labels && labels._lazyDataStore;
    var rowMap = loader && loader.frameRowByCam && loader.frameRowByCam.get(camera);
    if (store && rowMap && typeof loader.forEachInstanceRow === 'function') {
        var idn = store.instancesData || {}, pc = store.pointsData || {}, pp = store.predPointsData || {};
        var N = session.skeleton && session.skeleton.nodes ? session.skeleton.nodes.length : 0;
        var visit = function (cn, frameIdx, trk, info) {
            if (resident.has(frameIdx) || trk < 0) return;
            var id = session.getIdentityIdForTrack(camera, trk, frameIdx);
            if (id == null || id < 0) return;
            var r = info.instanceRow;
            var cols = Number(idn.instance_type ? idn.instance_type[r] : 0) !== 0 ? pp : pc;
            var p0 = Number(idn.point_id_start ? idn.point_id_start[r] : 0) || 0;
            var p1 = Number(idn.point_id_end ? idn.point_id_end[r] : 0) || 0;
            for (var k2 = 0; k2 < Math.min(p1 - p0, N); k2++) {
                var x2 = cols.x ? Number(cols.x[p0 + k2]) : NaN, y2 = cols.y ? Number(cols.y[p0 + k2]) : NaN;
                if (isFinite(x2) && isFinite(y2)) { add(frameIdx, id, 2 * k2, x2); add(frameIdx, id, 2 * k2 + 1, y2); }
            }
        };
        var last = -1;
        rowMap.forEach(function (row, fr) { if (fr > last) last = fr; });
        for (var s0 = 0; s0 <= last; s0 += PRINT_SLICE_FRAMES) {
            loader.forEachInstanceRow(visit, { camera: camera, start: s0, end: s0 + PRINT_SLICE_FRAMES });
            yield;
        }
    }
    return 'p2:' + camera + ':' + n + ':' + sum + ':' + xor;
}

/**
 * An order-free print of the poses a run reads: every finite coordinate (to
 * 1e-3 units), with its frame, identity and node, folded by sum and xor. The
 * same poses always give the same print whatever order they are visited in
 * (a lazy project's resident frames come and go). One pass over the poses
 * (~0.8 s for an hour of 5 animals × 15 nodes in 3D): `predictionInputPrintAsync`
 * spreads it over short slices.
 * @param {Session} session
 * @param {'3d'|'2d'} mode
 * @param {string} [camera] the 2D run's camera
 * @returns {string}
 */
export function predictionInputPrint(session, mode, camera) {
    var it = printSteps(session, mode, camera);
    var r = it.next();
    while (!r.done) r = it.next();
    return r.value;
}

/**
 * `predictionInputPrint` in slices, awaiting `pause()` between them.
 * @param {{pause: function(): Promise, isCancelled?: function(): boolean}} opts
 * @returns {Promise<string|null>} null when cancelled
 */
export async function predictionInputPrintAsync(session, mode, camera, opts) {
    var cancelled = (opts && opts.isCancelled) || function () { return false; };
    var it = printSteps(session, mode, camera);
    var r = it.next();
    while (!r.done) {
        await opts.pause();
        if (cancelled()) return null;
        r = it.next();
    }
    return r.value;
}

// ----------------------------------------------------------------------------
// Runs
// ----------------------------------------------------------------------------

/**
 * Store a behavior's run, replacing the last one.
 * @param {Session} session
 * @param {{behaviorId, source?, cfg, labels?, fps, mode, camera, bouts: Array<{animal,target,start,end,score}>,
 *          scores?: Map<string, Float32Array>}} run
 */
export function setBehaviorRun(session, run) {
    var st = storeOf(session);
    if (!st || !run || !behaviorById(run.behaviorId)) return;
    st.runs.set(run.behaviorId, normalizeRun(run));
}

function normalizeRun(run) {
    var bouts = (run.bouts || []).map(function (b) {
        var p = orderPair(run.behaviorId, b.animal, b.target);
        var out = { animal: p[0], target: p[1], start: b.start, end: b.end, score: Number.isFinite(b.score) ? b.score : NaN };
        if (b.ext) out.ext = b.ext;
        return out;
    }).filter(function (b) { return b.animal && b.end >= b.start && b.start >= 0; });
    bouts.sort(function (a, b) { return a.start - b.start || a.end - b.end || (a.animal < b.animal ? -1 : a.animal > b.animal ? 1 : 0); });
    var source = isRunSource(run.source) ? run.source : 'rule';
    return {
        behaviorId: run.behaviorId, source: source, cfg: run.cfg || '',
        labels: source.indexOf('model:') === 0 && typeof run.labels === 'string' && run.labels ? run.labels : null,
        states: source.indexOf('states:') === 0 ? stateList(run.states) : null,
        fps: run.fps > 0 ? run.fps : 0, mode: run.mode === '2d' ? '2d' : '3d', camera: run.camera || null,
        bouts: bouts, scores: run.scores || null,
    };
}

/**
 * Replace a run's bouts (and keep its scores): a classifier's post-processing
 * changed and its bouts were re-derived from the scores in memory, so it is
 * as fresh as before under the new settings (`cfg`).
 */
export function replaceRunBouts(session, behaviorId, bouts, cfg) {
    var st = session && _stores.get(session);
    var run = st && st.runs.get(behaviorId);
    if (!run) return false;
    st.runs.set(behaviorId, normalizeRun(Object.assign({}, run, { bouts: bouts, cfg: cfg != null ? cfg : run.cfg })));
    emit(session, { rederived: behaviorId });
    return true;
}

/**
 * Note a finished detection pass (one or more runs set): the print of the
 * poses it read becomes the session's reference, and the session is fresh.
 */
export function noteDetectionPass(session, inputPrint, info) {
    var st = storeOf(session);
    if (!st) return;
    st.print = inputPrint || null;
    st.inputStale = false;
    emit(session, Object.assign({ run: true }, info || {}));
}

/** Forget a session's predictions (all, or some behaviors'). */
export function clearBehaviorRuns(session, behaviorIds) {
    var st = session && _stores.get(session);
    if (!st || !st.runs.size) return 0;
    var n = 0;
    Array.from(st.runs.keys()).forEach(function (id) {
        if (behaviorIds && behaviorIds.indexOf(id) < 0) return;
        st.runs.delete(id);
        n++;
    });
    if (!st.runs.size) { st.print = null; st.inputStale = false; }
    if (n) emit(session, { cleared: true });
    return n;
}

/** The session's runs, in ethogram-independent insertion order. */
export function behaviorRuns(session) {
    var st = session && _stores.get(session);
    return st ? Array.from(st.runs.values()) : [];
}

/** One behavior's run, or null. */
export function behaviorRunOf(session, behaviorId) {
    var st = session && _stores.get(session);
    return (st && st.runs.get(behaviorId)) || null;
}

/** True when the session has any predicted bout. */
export function hasPredictions(session) {
    var st = session && _stores.get(session);
    if (!st) return false;
    for (var r of st.runs.values()) if (r.bouts.length) return true;
    return false;
}

/**
 * Every predicted bout, as plain records sorted by start:
 * `{behaviorId, animal, target, start, end, score, source}`.
 * @param {{behaviorId?:string, animal?:string}} [filter] an animal matches as
 *   actor, or as either animal of a mutual pair
 */
export function predictedBouts(session, filter) {
    var st = session && _stores.get(session);
    if (!st) return [];
    filter = filter || {};
    var out = [];
    st.runs.forEach(function (run) {
        if (filter.behaviorId != null && run.behaviorId !== filter.behaviorId) return;
        var b = behaviorById(run.behaviorId);
        if (!b) return;
        run.bouts.forEach(function (bt) {
            if (filter.animal != null && bt.animal !== filter.animal && !(b.mutual && bt.target === filter.animal)) return;
            out.push({ behaviorId: run.behaviorId, animal: bt.animal, target: bt.target, start: bt.start, end: bt.end,
                       score: bt.score, source: run.source });
        });
    });
    out.sort(function (a, b) { return a.start - b.start || a.end - b.end || (a.animal < b.animal ? -1 : a.animal > b.animal ? 1 : 0); });
    return out;
}

/** The per-frame score of a behavior for an animal (and target) at `frame`, or NaN (not run, not kept, unscored). */
export function predictionScoreAt(session, behaviorId, animal, target, frame) {
    var run = behaviorRunOf(session, behaviorId);
    if (!run || !run.scores) return NaN;
    var p = orderPair(behaviorId, animal, target);
    var s = run.scores.get(keyOf(p[0], p[1]));
    return s && frame >= 0 && frame < s.length ? s[frame] : NaN;
}

// ----------------------------------------------------------------------------
// Stale
// ----------------------------------------------------------------------------

/**
 * Something may have changed the poses (an edit marked the project dirty):
 * the next `checkPredictionInput` re-prints them. Cheap.
 */
export function notePossibleInputChange(session) {
    var st = session && _stores.get(session);
    if (st && st.runs.size) { st.checkPending = true; st.checkGen = (st.checkGen || 0) + 1; }
}

/** Is a re-print due? */
export function predictionCheckPending(session) {
    var st = session && _stores.get(session);
    return !!(st && st.runs.size && (st.checkPending || st.print == null));
}

/** Compare a fresh print with the reference; true when the stale flag changed. */
function settleInput(session, st, now) {
    if (st.print == null) { st.print = now; return false; }
    var stale = now !== st.print;
    if (stale === st.inputStale) return false;
    st.inputStale = stale;
    emit(session, { stale: stale });
    return true;
}

/**
 * Re-print the poses if something may have changed them (or no print exists
 * yet: predictions read from a file take the poses as loaded as their
 * reference). One pass over the poses, at once.
 * @returns {boolean} whether the input stale flag changed
 */
export function checkPredictionInput(session) {
    var st = session && _stores.get(session);
    if (!st || !st.runs.size) return false;
    if (!st.checkPending && st.print != null) return false;
    st.checkPending = false;
    var run = st.runs.values().next().value;
    return settleInput(session, st, predictionInputPrint(session, run.mode, run.camera));
}

/**
 * `checkPredictionInput` in slices, awaiting `pause()` between them; an edit
 * during the walk (a newer `notePossibleInputChange`) abandons it, and the
 * caller checks again later.
 * @returns {Promise<boolean>} whether the input stale flag changed
 */
export async function checkPredictionInputAsync(session, pause) {
    var st = session && _stores.get(session);
    if (!st || !st.runs.size) return false;
    if (!st.checkPending && st.print != null) return false;
    var gen = st.checkGen || 0;
    var run = st.runs.values().next().value;
    var now = await predictionInputPrintAsync(session, run.mode, run.camera, {
        pause: pause, isCancelled: function () { return (st.checkGen || 0) !== gen || !st.runs.size; },
    });
    if (now == null || (st.checkGen || 0) !== gen) return false;
    st.checkPending = false;
    return settleInput(session, st, now);
}

/**
 * The frame rate to check a run against: the one now, or while nothing has said
 * what it is (`env.fpsUnknown`), the one the run recorded, else the labels'.
 */
function fpsToCheck(session, run, env) {
    var now = env && env.fps;
    if (!env || !env.fpsUnknown) return now;
    return run.fps > 0 ? run.fps : (behaviorFps(session) || now);
}

/**
 * Why a session's predictions are stale.
 * @param {{fps:number, fpsUnknown?:boolean, floorKey?:string, labelPrintOf?: function(string): string,
 *          modelHashOf?: function(string): ?string}} env the frame rate (`fpsUnknown`: only a default,
 *   nothing has said what it is) and floor now, and for classifier runs the print of each
 *   behavior's training labels now and its current model's hash
 * @returns {{input: boolean, settings: string[], labels: string[], model: string[]}} behavior ids
 *   whose detector or settings changed since their run (or that no longer exist); whose training
 *   labels changed (classifier runs); whose model was replaced (classifier runs)
 */
export function predictionStaleness(session, env) {
    var st = session && _stores.get(session);
    if (!st || !st.runs.size) return { input: false, settings: [], labels: [], model: [] };
    var settings = [], labels = [], model = [];
    st.runs.forEach(function (run, id) {
        // A run another tool made (read from a file's events): nothing here produced
        // it, so no setting here can make it stale.
        if (run.source === 'file') return;
        var b = behaviorById(id);
        var now = b && runConfigPrint(b, session, { fps: fpsToCheck(session, run, env), mode: run.mode, camera: run.camera, floorKey: env && env.floorKey,
                                                   source: run.source, states: run.states });
        if (!b || now !== run.cfg) settings.push(id);
        var hash = runModelHash(run);
        if (!hash || !b) return;
        if (run.labels && env && env.labelPrintOf && env.labelPrintOf(id) !== run.labels) labels.push(id);
        var current = env && env.modelHashOf ? env.modelHashOf(id) : null;
        if (current && current !== hash) model.push(id);
    });
    return { input: !!st.inputStale, settings: settings, labels: labels, model: model };
}

/** Is this behavior's run stale (input, settings, labels or model)? */
export function isRunStale(session, behaviorId, env) {
    var s = predictionStaleness(session, env);
    return s.input || s.settings.indexOf(behaviorId) >= 0 || s.labels.indexOf(behaviorId) >= 0 || s.model.indexOf(behaviorId) >= 0;
}

/**
 * Do the session's predictions stand — some predicted bout, none of it
 * stale? Drives the blue toolbar button (pose/run-state.js). Cached input
 * state only; never walks the poses.
 * @param {{fps:number, fpsUnknown?:boolean, floorKey?:string}} env
 */
export function predictionsStand(session, env) {
    if (!hasPredictions(session)) return false;
    var s = predictionStaleness(session, env);
    if (s.input) return false;
    return behaviorRuns(session).some(function (r) {
        return r.bouts.length && s.settings.indexOf(r.behaviorId) < 0 && s.labels.indexOf(r.behaviorId) < 0 && s.model.indexOf(r.behaviorId) < 0;
    });
}

// ----------------------------------------------------------------------------
// Identities
// ----------------------------------------------------------------------------

/** An identity was renamed: its predicted bouts (as actor and target) follow. */
export function renamePredictedAnimal(session, oldName, newName) {
    var st = session && _stores.get(session);
    if (!st || !oldName || !newName || oldName === newName) return 0;
    var moved = 0;
    st.runs.forEach(function (run, id) {
        var touched = false;
        var bouts = run.bouts.map(function (b) {
            if (b.animal !== oldName && b.target !== oldName) return b;
            touched = true; moved++;
            var p = orderPair(id, b.animal === oldName ? newName : b.animal, b.target === oldName ? newName : b.target);
            return Object.assign({}, b, { animal: p[0], target: p[1] });
        });
        if (!touched) return;
        var scores = null;
        if (run.scores) {
            scores = new Map();
            run.scores.forEach(function (arr, k) {
                var parts = k.split(SEP);
                var p = orderPair(id, parts[0] === oldName ? newName : parts[0], parts[1] === oldName ? newName : parts[1]);
                scores.set(keyOf(p[0], p[1]), arr);
            });
        }
        st.runs.set(id, normalizeRun(Object.assign({}, run, { bouts: bouts, scores: scores })));
    });
    if (moved) emit(session, { identity: 'rename' });
    return moved;
}

// ----------------------------------------------------------------------------
// Review
// ----------------------------------------------------------------------------

/**
 * How the labels have settled a predicted bout: 'accepted' (every frame
 * labelled present), 'rejected' (every frame labelled "not"), 'mixed', or
 * 'unreviewed' (no frame labelled). Also the share of its frames labelled
 * "not" (`disagreement`).
 */
export function reviewStatus(session, bout) {
    var list = boutIntervals(session, bout.behaviorId, bout.animal, bout.target);
    var len = bout.end - bout.start + 1, pres = 0, abs = 0;
    // The list is sorted and non-overlapping: start at the first interval
    // ending at or after the bout's start.
    var lo = 0, hi = list.length;
    while (lo < hi) { var mid = (lo + hi) >> 1; if (list[mid].end < bout.start) lo = mid + 1; else hi = mid; }
    for (var i = lo; i < list.length; i++) {
        var iv = list[i];
        if (iv.start > bout.end) break;
        var n = Math.min(iv.end, bout.end) - Math.max(iv.start, bout.start) + 1;
        if (iv.present) pres += n; else abs += n;
    }
    var status = pres === len ? 'accepted' : abs === len ? 'rejected' : pres + abs === 0 ? 'unreviewed' : 'mixed';
    return { status: status, present: pres, absent: abs, disagreement: abs / len };
}

/** The review orders. */
export var REVIEW_ORDERS = ['uncertainty', 'disagreement', 'time'];

/**
 * Predicted bouts in review order, each with its review status.
 * @param {{order?:'uncertainty'|'disagreement'|'time', behaviorId?, animal?, unreviewedOnly?: boolean}} [opts]
 *   uncertainty: scores nearest the decision first (0.5 for a rule, the
 *   behavior's threshold for a classifier); disagreement: most frames
 *   labelled "not" first, then uncertainty; time: by start
 */
export function reviewQueue(session, opts) {
    opts = opts || {};
    var thr = new Map();
    behaviorRuns(session).forEach(function (r) { thr.set(r.behaviorId, runThreshold(r)); });
    var list = predictedBouts(session, { behaviorId: opts.behaviorId || undefined, animal: opts.animal || undefined }).map(function (b) {
        var r = reviewStatus(session, b);
        var t = thr.has(b.behaviorId) ? thr.get(b.behaviorId) : 0.5;
        return Object.assign(b, { status: r.status, disagreement: r.disagreement, threshold: t,
                                  uncertainty: Number.isFinite(b.score) ? Math.abs(b.score - t) : 1 });
    });
    if (opts.unreviewedOnly) list = list.filter(function (b) { return b.status !== 'accepted' && b.status !== 'rejected'; });
    var order = REVIEW_ORDERS.indexOf(opts.order) >= 0 ? opts.order : 'uncertainty';
    if (order === 'uncertainty') list.sort(function (a, b) { return a.uncertainty - b.uncertainty || a.start - b.start; });
    else if (order === 'disagreement') list.sort(function (a, b) { return b.disagreement - a.disagreement || a.uncertainty - b.uncertainty || a.start - b.start; });
    return list;
}

/**
 * Accept predicted bouts: label their frames present, source 'a' (accepted
 * prediction). One `labelBouts` edit — one undo step for all of them.
 * @returns {{kept:number, dropped:number}}
 */
export function acceptPredictions(session, bouts) {
    return labelBouts(session, (bouts || []).map(function (b) {
        return { behaviorId: b.behaviorId, animal: b.animal, target: b.target, start: b.start, end: b.end, present: true };
    }), { source: 'a' });
}

/** Reject predicted bouts: label their frames "not" (a manual label). One undo step. */
export function rejectPredictions(session, bouts) {
    return labelBouts(session, (bouts || []).map(function (b) {
        return { behaviorId: b.behaviorId, animal: b.animal, target: b.target, start: b.start, end: b.end, present: false };
    }), { source: 'm' });
}

// ----------------------------------------------------------------------------
// Save / load
// ----------------------------------------------------------------------------

/**
 * The `metadata.lucid.behaviorPredictions` payload, or null when there is no
 * predicted bout (nothing is written).
 */
export function exportPredictionRuns(session) {
    var st = session && _stores.get(session);
    if (!st) return null;
    var runs = [];
    st.runs.forEach(function (run) {
        if (!run.bouts.length || !behaviorById(run.behaviorId)) return;
        var o = { behavior: run.behaviorId, source: run.source, cfg: run.cfg, mode: run.mode };
        if (run.labels) o.labels = run.labels;
        if (run.states) o.states = run.states;
        if (run.camera) o.camera = run.camera;
        if (run.fps > 0) o.fps = run.fps;
        o.bouts = run.bouts.map(function (b) {
            return [b.animal, b.target, b.start, b.end, Number.isFinite(b.score) ? Math.round(b.score * 1000) / 1000 : null];
        });
        runs.push(o);
    });
    if (!runs.length) return null;
    return { v: 1, runs: runs };
}

/**
 * What a `.slp` keeps in `metadata.lucid.behaviorPredictions` beside its events:
 * each run with a bout, without the bouts (`exportPredictionEvents`). Null when
 * there is no predicted bout (nothing is written).
 */
export function exportPredictionHeaders(session) {
    var payload = exportPredictionRuns(session);
    if (!payload) return null;
    return {
        v: 2,
        runs: payload.runs.map(function (r) {
            var o = Object.assign({}, r);
            delete o.bouts;
            return o;
        }),
    };
}

/**
 * Every predicted bout of the session, one entry each, for saving as events:
 * `{behaviorId, source, animal, target, start, end, score, ext}`. Scores are rounded
 * as `exportPredictionRuns` rounds them. `ext` is what a file said about a bout read
 * from it that LUC3D does not use, written back unchanged.
 */
export function exportPredictionEvents(session) {
    var st = session && _stores.get(session);
    var out = [];
    if (!st) return out;
    st.runs.forEach(function (run) {
        if (!behaviorById(run.behaviorId)) return;
        run.bouts.forEach(function (b) {
            out.push({ behaviorId: run.behaviorId, source: run.source, animal: b.animal, target: b.target, start: b.start, end: b.end,
                       score: Number.isFinite(b.score) ? Math.round(b.score * 1000) / 1000 : NaN, ext: b.ext || null });
        });
    });
    return out;
}

/**
 * Replace a session's predictions with saved ones (a load). Runs naming a
 * behavior the ethogram does not have are dropped. Their poses are taken as
 * loaded (the first input check prints them). Does not notify.
 * @param {Map<string,string>} [idMap] file behavior id -> project id
 * @returns {{kept:number, dropped:number}} bouts
 */
export function importPredictionRuns(session, payload, idMap) {
    var st = storeOf(session);
    if (!st) return { kept: 0, dropped: 0 };
    st.runs = new Map(); st.print = null; st.inputStale = false; st.checkPending = false;
    if (!payload || typeof payload !== 'object' || payload.v !== 1 || !Array.isArray(payload.runs)) return { kept: 0, dropped: 0 };
    var kept = 0, dropped = 0;
    payload.runs.forEach(function (r) {
        if (!r || typeof r.behavior !== 'string' || !Array.isArray(r.bouts)) return;
        var id = idMap && idMap.has(r.behavior) ? idMap.get(r.behavior) : r.behavior;
        if (!behaviorById(id)) { dropped += r.bouts.length; return; }
        var bouts = [];
        r.bouts.forEach(function (b) {
            var s = Array.isArray(b) ? Math.round(Number(b[2])) : NaN, e = Array.isArray(b) ? Math.round(Number(b[3])) : NaN;
            if (!Array.isArray(b) || typeof b[0] !== 'string' || !b[0] || !(s >= 0) || !(e >= s)) { dropped++; return; }
            var bout = { animal: b[0], target: typeof b[1] === 'string' ? b[1] : '', start: s, end: e,
                         score: b[4] == null ? NaN : Number(b[4]) };
            if (b[5] && typeof b[5] === 'object') bout.ext = b[5];
            bouts.push(bout);
        });
        kept += bouts.length;
        st.runs.set(id, normalizeRun({ behaviorId: id, source: r.source, cfg: typeof r.cfg === 'string' ? r.cfg : '',
            labels: r.labels, states: r.states, fps: r.fps > 0 ? r.fps : 0, mode: r.mode, camera: r.camera, bouts: bouts }));
    });
    return { kept: kept, dropped: dropped };
}

/** Forget a session's predictions entirely. */
export function resetBehaviorPredictions(session) {
    if (session) _stores.delete(session);
}
