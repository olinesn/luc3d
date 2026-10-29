// ui/behavior-detect.js — running the behavior detectors on the active
// session: the toolbar's Behavior button (All frames), This frame, and what
// the detection settings need to show live numbers (feature columns, the
// resolved body and floor).
//
// A run reads every animal's poses once (pose/behavior-tensor.js), resolves
// the body (node roles, excluded nodes, body lengths) and the floor, then
// scores every behavior that has a detector, chunk by chunk on the worker pool
// (pose/behavior-pool.js), and turns the scores into bouts
// (pose/behavior-rules.js). Each behavior's result replaces its last run in
// pose/behavior-predictions.js, which the Review section, the timeline and the
// project file read.
//
// The floor, in order: the plane chosen as the floor in the settings; the Set
// Origin frame (+Z up) — the default when one is set; a fit to the animals'
// lowest points. A choice that cannot be honoured (the plane has no fit, the
// origin was cleared) falls back to the fit and says so.
//
// Predictions go stale when their poses change: every markDirty notes that
// they may have, and a check (one pass over the poses) runs shortly after the
// edits stop, never on a draw.
//
// Depends on: ui/app-state.js, ui/loading-overlay.js, ui/plane-definition.js,
//             ui/origin-definition.js, pose/behavior-*.js,
//             import-export/save-load.js

import { state } from './app-state.js';
import { showLoading, showLoadingProgress, hideLoading, yieldToPaint } from './loading-overlay.js';
import { planeModel, getPlanes } from './plane-definition.js';
import { originState } from './origin-definition.js';
import { behaviors, behaviorById, featureSettings } from '../pose/behavior-ethogram.js';
import {
    collectBehaviorInput, resolveBody, fitFloor, floorFromPlane, floorFromOrigin, featureSpec,
} from '../pose/behavior-tensor.js';
import { featurePadFrames, planChunks, sliceChunkData, catalogFor, TARGET_PREFIX } from '../pose/behavior-features.js';
import { planDetectors, detectorBouts, orderedPairs, combineMutual } from '../pose/behavior-rules.js';
import { runBehaviorChunks, runJob } from '../pose/behavior-pool.js';
import {
    setBehaviorRun, noteDetectionPass, runConfigPrint, predictionInputPrintAsync, notePossibleInputChange,
    predictionCheckPending, checkPredictionInputAsync,
} from '../pose/behavior-predictions.js';
import { setStatus, markDirty, onMarkDirty } from '../import-export/save-load.js';

/** Frames per chunk of a run (each padded by the longest window). */
var CHUNK_FRAMES = 8192;
/** How long after the last edit the poses are re-printed. */
var INPUT_CHECK_DELAY_MS = 600;

var SEP = '\u0001';

/** The frame rate behaviors are measured at: the session's (30 when unknown). */
export function detectionFps() {
    return state.fps > 0 ? state.fps : 30;
}

function roundJoin(v) {
    return (v || []).map(function (x) { return Math.round(x * 1000) / 1000; }).join(',');
}

/**
 * The floor in effect, as a short key (no pose is read): the Set Origin frame
 * or the chosen plane's fit, or 'fit'. A run records it, so a floor that moved
 * after it ran makes it stale.
 */
export function floorKey() {
    var s = featureSettings();
    var want = s && s.floor ? s.floor.source : 'auto';
    if (want === 'plane') {
        var fit = null;
        try { fit = planeModel().usableFitForPlane(s.floor.plane); } catch (e) { fit = null; }
        return 'plane:' + s.floor.plane + ':' + (fit ? roundJoin(fit.centroid) + ':' + roundJoin(fit.normal) : 'none');
    }
    if (want === 'auto' || want === 'origin') {
        var f = originState.frame;
        if (f) return 'origin:' + roundJoin(f.origin) + ':' + roundJoin(f.zAxis);
    }
    return 'fit';
}

/** What predictions are checked against now: the frame rate and the floor. */
export function detectionEnv() {
    return { fps: detectionFps(), floorKey: floorKey() };
}

function numFramesOf(session) {
    return session === state.session && state.totalFrames > 0 ? state.totalFrames : 0;
}

function n(x) {
    return Number(x || 0).toLocaleString('en-US');
}

// ----------------------------------------------------------------------------
// Body and floor
// ----------------------------------------------------------------------------

/** Camera centers in world coordinates (C = −Rᵀ t), for the floor fit's sense of up. */
function cameraCenters(session) {
    return ((session && session.cameras) || []).map(function (c) {
        try {
            var R = c.rotationMatrix, t = c.tvec;
            if (!R || !t) return null;
            return [0, 1, 2].map(function (j) { return -(R[0][j] * t[0] + R[1][j] * t[1] + R[2][j] * t[2]); });
        } catch (e) { return null; }
    }).filter(function (p) { return p && p.every(Number.isFinite); });
}

/** The planes a floor can be: those with a usable fit. */
export function floorPlanes() {
    var model;
    try { model = planeModel(); } catch (e) { return []; }
    return (getPlanes() || []).filter(function (p) { return !!model.usableFitForPlane(p); })
        .map(function (p) { return { id: p.id, name: p.name }; });
}

/**
 * The floor for an input, as the settings ask, with what was used.
 * @returns {{floor: Object|null, source: string, label: string, warning: string}}
 */
export function resolveFloor(input, body, settings) {
    if (input.mode !== '3d') return { floor: null, source: 'none', label: 'No floor: 2D, from camera ' + (input.camera || '?'), warning: '' };
    var want = settings && settings.floor ? settings.floor.source : 'auto';
    var warning = '';
    if (want === 'plane') {
        var id = settings.floor.plane;
        var plane = null, fit = null;
        try { plane = planeModel().getPlane(id); fit = plane && planeModel().usableFitForPlane(plane); } catch (e) { fit = null; }
        var fl = fit && floorFromPlane(fit, input, body.used);
        if (fl) return { floor: fl, source: 'plane', label: 'The plane "' + plane.name + '"', warning: '' };
        warning = plane ? 'The plane "' + plane.name + '" has no fit yet: using the fitted floor' : 'The floor plane no longer exists: using the fitted floor';
    }
    if (want === 'origin' || want === 'auto') {
        var fo = floorFromOrigin(originState.frame);
        if (fo) return { floor: fo, source: 'origin', label: 'The Set Origin frame (+Z up)', warning: '' };
        if (want === 'origin') warning = 'No origin is set: using the fitted floor';
    }
    var all = input.nodeNames.map(function (nm, i) { return i; });
    var fitted = fitFloor(input, all, { cameraCenters: cameraCenters(state.session), head: body.roles.head, tailBase: body.roles.tailBase });
    if (!fitted) return { floor: null, source: 'none', label: 'No floor: too few 3D points to fit one', warning: warning };
    var why = { cameras: 'the cameras are above it', heights: 'the bodies rise from it', flatness: 'it is the flatter side', 'one envelope': 'only one side could be fitted' }[fitted.upFrom] || '';
    return {
        floor: fitted, source: 'fit',
        label: 'Fitted to the animals’ lowest points, level with ' + (fitted.tiltFrom === 'body axes' ? 'their bodies' : 'their spread') +
            (why ? ' (up: ' + why + ')' : ''),
        warning: warning || (fitted.confidence === 'low' ? 'The fitted floor is uncertain: choose a plane or set an origin' : ''),
    };
}

/**
 * Everything a run reads, for the active session: the poses, the body, the
 * floor and the feature spec.
 * @param {Session} session
 * @param {{mode?: '3d'|'2d'}} [opts]
 * @returns {{session, input, body, floorInfo, spec, catalog, fps}}
 */
export function buildBehaviorContext(session, opts) {
    var settings = featureSettings();
    var input = collectBehaviorInput(session, { numFrames: numFramesOf(session), mode: opts && opts.mode });
    var body = resolveBody(input.nodeNames, input.edges, settings);
    var floorInfo = resolveFloor(input, body, settings);
    var fps = detectionFps();
    var spec = featureSpec(input, body, floorInfo.floor, fps);
    return { session: session, input: input, body: body, floorInfo: floorInfo, spec: spec, catalog: catalogFor(spec), fps: fps };
}

// The last context, kept a minute for This frame and the settings dialog
// (collecting an hour of 5 animals takes ~0.5 s): any edit, another session,
// other settings or another frame rate drop it.
var CONTEXT_KEEP_MS = 60000;
var _ctxCache = null;
var _editGen = 0;

function contextKey(session) {
    return JSON.stringify([featureSettings(), detectionFps(), numFramesOf(session), floorKey()]);
}

/** `buildBehaviorContext`, reusing the last one while nothing changed. */
export function cachedBehaviorContext(session) {
    var key = contextKey(session);
    var c = _ctxCache;
    if (c && c.session === session && c.gen === _editGen && c.key === key) {
        clearTimeout(c.timer);
        c.timer = setTimeout(dropContextCache, CONTEXT_KEEP_MS);
        return c.ctx;
    }
    return keepContext(session, buildBehaviorContext(session));
}

function keepContext(session, ctx) {
    dropContextCache();
    _ctxCache = { session: session, gen: _editGen, key: contextKey(session), ctx: ctx, timer: setTimeout(dropContextCache, CONTEXT_KEEP_MS) };
    return ctx;
}

function dropContextCache() {
    if (_ctxCache) clearTimeout(_ctxCache.timer);
    _ctxCache = null;
}

/** Re-resolve the body and floor of a context after a settings change (the poses are kept). */
export function refreshBehaviorContext(ctx) {
    var settings = featureSettings();
    var body = resolveBody(ctx.input.nodeNames, ctx.input.edges, settings);
    var floorInfo = resolveFloor(ctx.input, body, settings);
    var fps = detectionFps();
    var spec = featureSpec(ctx.input, body, floorInfo.floor, fps);
    return Object.assign({}, ctx, { body: body, floorInfo: floorInfo, spec: spec, catalog: catalogFor(spec), fps: fps });
}

// ----------------------------------------------------------------------------
// Running
// ----------------------------------------------------------------------------

/** The ethogram's behaviors that have a detector, as the planner takes them. */
function detectorBehaviors() {
    return behaviors().filter(function (b) { return !!b.detector; }).map(function (b) {
        return { behaviorId: b.id, social: !!b.social, mutual: !!b.mutual, point: b.kind === 'point', detector: b.detector };
    });
}

/** The animal and target names of a score key ('a' or 'a>b'). */
function namesOfKey(input, key) {
    var p = String(key).split('>');
    return [input.animals[+p[0]] || '', p.length > 1 ? (input.animals[+p[1]] || '') : ''];
}

var _running = false;
var _openSettings = null;

/** ui/behavior-panel.js registers how to open the detection settings (avoids an import cycle). */
export function setDetectionSettingsOpener(fn) {
    _openSettings = typeof fn === 'function' ? fn : null;
}

function openSettings() {
    if (_openSettings) _openSettings();
}

/**
 * Behavior ▸ Detect Behaviors, the toolbar's Behavior button: score every
 * behavior that has a detector on every frame, and keep the bouts.
 */
export async function detectAllFrames() {
    var session = state.session;
    if (!session) { setStatus('No session loaded'); return null; }
    if (_running) return null;
    var list = detectorBehaviors();
    if (!list.length) {
        setStatus(behaviors().length ? 'No behavior has a detector yet: set one in the detection settings'
            : 'No behaviors yet: add them (with detectors) in the detection settings', 'warning');
        openSettings();
        return null;
    }
    _running = true;
    var title = { title: 'Detect Behaviors', unit: 'frames' };
    try {
        showLoading('Reading poses…', { title: title.title });
        await yieldToPaint();
        var ctx = keepContext(session, buildBehaviorContext(session));
        var input = ctx.input;
        if (!input.animals.length) {
            setStatus('No identified animals to detect behaviors on: run Track All (behaviors are detected per identity)', 'warning');
            return null;
        }
        var plan = planDetectors(ctx.catalog, list);
        var social = plan.plans.filter(function (p) { return p.social; });
        var notes = plan.skipped.map(function (s) {
            var b = behaviorById(s.behaviorId);
            return (b ? b.name : s.behaviorId) + ': ' + s.problems.join('; ');
        });
        if (social.length && input.animals.length < 2) {
            notes.push(social.map(function (p) { return behaviorById(p.behaviorId).name; }).join(', ') + ': social, but only one animal');
            plan.plans = plan.plans.filter(function (p) { return !p.social; });
        }
        if (!plan.plans.length) {
            setStatus('Nothing to detect — ' + notes.join(' · '), 'warning');
            openSettings();
            return null;
        }
        var T = input.numFrames, N = input.nodeNames.length, D = input.dim;
        var chunks = planChunks(T, featurePadFrames(ctx.spec), CHUNK_FRAMES);
        var scores = {};
        plan.plans.forEach(function (p) { scores[p.behaviorId] = {}; });
        var jobPlan = { request: plan.request, plans: plan.plans };
        var detail = plan.plans.length + ' behavior' + (plan.plans.length === 1 ? '' : 's') + ' · ' +
            input.animals.length + ' animal' + (input.animals.length === 1 ? '' : 's') + ' · ' + (input.mode === '3d' ? '3D' : '2D, ' + input.camera);
        var doneFrames = 0;
        await runBehaviorChunks(chunks.length, function (i) {
            return { kind: 'scores', spec: ctx.spec, chunk: chunks[i], data: sliceChunkData(input.data, chunks[i], N * D), plan: jobPlan };
        }, function (i, res) {
            var ch = chunks[i];
            if (!res) return;
            Object.keys(res.scores).forEach(function (bid) {
                var byKey = res.scores[bid];
                Object.keys(byKey).forEach(function (k) {
                    var whole = scores[bid][k];
                    if (!whole) { whole = new Float32Array(T).fill(NaN); scores[bid][k] = whole; }
                    whole.set(byKey[k], ch.start);
                });
            });
            doneFrames += ch.end - ch.start;
        }, {
            workload: T * input.animals.length,
            onProgress: function () { showLoadingProgress('Detecting behaviors', doneFrames, T, Object.assign({ detail: detail }, title)); },
            pause: yieldToPaint,
        });
        var fps = ctx.fps;
        var env = detectionEnv();
        var counts = [], total = 0;
        plan.plans.forEach(function (p) {
            var b = behaviorById(p.behaviorId);
            var bouts = [], scoreMap = new Map();
            Object.keys(scores[p.behaviorId]).forEach(function (k) {
                var nm = namesOfKey(input, k);
                var arr = scores[p.behaviorId][k];
                scoreMap.set(p.mutual && nm[1] < nm[0] ? nm[1] + SEP + nm[0] : nm[0] + SEP + nm[1], arr);
                detectorBouts(arr, p.detector, fps, p.point).forEach(function (bt) {
                    bouts.push({ animal: nm[0], target: nm[1], start: bt[0], end: bt[1], score: bt[2] });
                });
            });
            setBehaviorRun(session, {
                behaviorId: p.behaviorId, source: 'rule',
                cfg: runConfigPrint(b, session, { fps: fps, mode: input.mode, camera: input.camera, floorKey: env.floorKey }),
                fps: fps, mode: input.mode, camera: input.camera, bouts: bouts, scores: scoreMap,
            });
            counts.push(b.name + ' ' + n(bouts.length));
            total += bouts.length;
        });
        // The poses this run read, as the reference later edits are checked against.
        showLoadingProgress('Detecting behaviors', T, T, Object.assign({ detail: detail }, title));
        var print = await predictionInputPrintAsync(session, input.mode, input.camera, { pause: slicePause });
        noteDetectionPass(session, print, { frames: T });
        markDirty();
        // That edit was this run's own: the poses it read are still current.
        if (_ctxCache && _ctxCache.ctx === ctx) _ctxCache.gen = _editGen;
        setStatus('Detected ' + n(total) + ' bout' + (total === 1 ? '' : 's') + ': ' + counts.join(', ') +
            (ctx.floorInfo.warning ? ' · ' + ctx.floorInfo.warning : '') + (notes.length ? ' · Skipped ' + notes.join(' · ') : ''),
            notes.length || ctx.floorInfo.warning ? 'warning' : 'success');
        return { bouts: total };
    } catch (e) {
        console.error('[behavior] detection failed', e);
        setStatus('Behavior detection failed: ' + (e && e.message || e), 'error');
        return null;
    } finally {
        hideLoading();
        _running = false;
    }
}

// ----------------------------------------------------------------------------
// This frame
// ----------------------------------------------------------------------------

var _readout = null;
var _readoutListeners = [];

/** The last This frame readout: `{frame, rows: [{behaviorId, animal, target, score, detected, conditions}]}` or null. */
export function frameReadout() {
    return _readout && _readout.session === state.session ? _readout : null;
}

/** Be told when the This frame readout changes. */
export function onFrameReadout(fn) {
    if (typeof fn === 'function') _readoutListeners.push(fn);
}

/**
 * This frame: evaluate every detector at the playhead and say, per animal
 * (and pair), which behaviors hold and each condition's value against its
 * threshold. Writes nothing.
 */
export function detectThisFrame() {
    var session = state.session;
    if (!session) { setStatus('No session loaded'); return null; }
    var list = detectorBehaviors();
    if (!list.length) {
        setStatus('No behavior has a detector yet: set one in the detection settings', 'warning');
        openSettings();
        return null;
    }
    var ctx = cachedBehaviorContext(session);
    var input = ctx.input;
    var f = state.currentFrame;
    if (!input.animals.length || f < 0 || f >= input.numFrames) {
        setStatus('No identified animals on this session to detect behaviors on', 'warning');
        return null;
    }
    var plan = planDetectors(ctx.catalog, list);
    if (input.animals.length < 2) plan.plans = plan.plans.filter(function (p) { return !p.social; });
    var pad = featurePadFrames(ctx.spec);
    var chunk = { start: f, end: f + 1, padStart: Math.max(0, f - pad), padEnd: Math.min(input.numFrames, f + 1 + pad) };
    var names = { animal: [], pair: [] };
    plan.plans.forEach(function (p) {
        p.conditions.forEach(function (c) {
            var into = c.who === 'pair' ? names.pair : names.animal;
            if (into.indexOf(c.name) < 0) into.push(c.name);
        });
    });
    var res = runJob({ kind: 'scores', spec: ctx.spec, chunk: chunk, data: sliceChunkData(input.data, chunk, input.nodeNames.length * input.dim),
        plan: { request: plan.request, plans: plan.plans }, extra: { columns: names } });
    var rows = [];
    plan.plans.forEach(function (p) {
        Object.keys(res.scores[p.behaviorId]).forEach(function (k) {
            var nm = namesOfKey(input, k);
            var parts = String(k).split('>');
            var score = res.scores[p.behaviorId][k][0];
            var conds = p.conditions.map(function (c) {
                var col;
                if (c.who === 'pair') col = res.columns.pair[k] && res.columns.pair[k][c.name];
                else col = res.columns.animal[c.who === 'target' ? +parts[1] : +parts[0]][c.name];
                return { feature: (c.who === 'target' ? TARGET_PREFIX : '') + c.name, op: c.op, value: c.value, unit: c.unit,
                         actual: col ? col[0] : NaN };
            });
            rows.push({ behaviorId: p.behaviorId, animal: nm[0], target: nm[1], score: score, detected: score >= 0.5, conditions: conds });
        });
    });
    rows.sort(function (a, b) { return (b.detected - a.detected) || ((b.score || 0) - (a.score || 0)); });
    _readout = { session: session, frame: f, rows: rows };
    _readoutListeners.forEach(function (fn) { try { fn(_readout); } catch (e) { console.error(e); } });
    var hits = rows.filter(function (r) { return r.detected; }).map(function (r) {
        var b = behaviorById(r.behaviorId);
        return b.name + ' · ' + r.animal + (r.target ? (b.mutual ? ' ↔ ' : ' → ') + r.target : '');
    });
    setStatus('Frame ' + n(f + 1) + ': ' + (hits.length ? hits.join(', ') : 'no behavior detected') +
        ' (details in the Behavior tab ▸ Predictions)', hits.length ? 'success' : undefined);
    return _readout;
}

// ----------------------------------------------------------------------------
// Feature columns for the settings editor
// ----------------------------------------------------------------------------

/**
 * Whole-session columns of some features, for histograms and live counts:
 * `{animal: {name: Float32Array[] (per animal)}, pair: {name: Map<'a>b', Float32Array>}}`.
 * Target-prefixed names are computed as the plain animal feature.
 * @param {Object} ctx from `buildBehaviorContext`
 * @param {string[]} names feature names (catalog names, or TARGET_PREFIX + an animal feature)
 * @param {{onProgress?: function(number, number), isCancelled?: function(): boolean}} [opts]
 */
export async function computeFeatureColumns(ctx, names, opts) {
    var input = ctx.input;
    var animalNames = [], pairNames = [];
    (names || []).forEach(function (nm) {
        var plain = nm.indexOf(TARGET_PREFIX) === 0 ? nm.slice(TARGET_PREFIX.length) : nm;
        var f = ctx.catalog.byName.get(plain);
        if (!f) return;
        var into = f.scope === 'pair' ? pairNames : animalNames;
        if (into.indexOf(plain) < 0) into.push(plain);
    });
    var T = input.numFrames, N = input.nodeNames.length, D = input.dim, A = input.animals.length;
    var out = { animal: {}, pair: {} };
    animalNames.forEach(function (nm) { out.animal[nm] = input.animals.map(function () { return new Float32Array(T).fill(NaN); }); });
    var pairs = pairNames.length && A > 1 ? orderedPairs(A) : [];
    pairNames.forEach(function (nm) {
        var m = new Map();
        pairs.forEach(function (pr) { m.set(pr[0] + '>' + pr[1], new Float32Array(T).fill(NaN)); });
        out.pair[nm] = m;
    });
    if (!animalNames.length && !pairs.length) return out;
    var chunks = planChunks(T, featurePadFrames(ctx.spec), CHUNK_FRAMES);
    await runBehaviorChunks(chunks.length, function (i) {
        return { kind: 'features', spec: ctx.spec, chunk: chunks[i], data: sliceChunkData(input.data, chunks[i], N * D),
                 request: { animal: animalNames, pair: pairNames, pairs: pairs } };
    }, function (i, res) {
        if (!res) return;
        var st = chunks[i].start;
        res.animal.forEach(function (cols, a) {
            animalNames.forEach(function (nm) { if (cols[nm]) out.animal[nm][a].set(cols[nm], st); });
        });
        Object.keys(res.pair).forEach(function (k) {
            pairNames.forEach(function (nm) { if (res.pair[k][nm]) out.pair[nm].get(k).set(res.pair[k][nm], st); });
        });
    }, { workload: T * Math.max(1, A), onProgress: opts && opts.onProgress, isCancelled: opts && opts.isCancelled, pause: yieldToPaint });
    return out;
}

export { combineMutual };

// ----------------------------------------------------------------------------
// Stale input
// ----------------------------------------------------------------------------

var _checkTimer = null;
var _checking = false;

/** A pause between slices of the background check: a task, so input and paint go first. */
function slicePause() {
    return new Promise(function (resolve) { setTimeout(resolve, 0); });
}

/**
 * Check the poses once the edits stop: in slices between tasks (an hour of
 * 5 animals is ~0.8 s of hashing), abandoned and rescheduled when another
 * edit comes in meanwhile.
 */
function scheduleInputCheck() {
    if (_checkTimer) clearTimeout(_checkTimer);
    _checkTimer = setTimeout(function () {
        _checkTimer = null;
        var s = state.session;
        if (!s || !predictionCheckPending(s)) return;
        if (_checking) { scheduleInputCheck(); return; }
        _checking = true;
        checkPredictionInputAsync(s, slicePause).catch(function (e) {
            console.error('[behavior] prediction input check failed', e);
        }).then(function () {
            _checking = false;
            if (state.session === s && predictionCheckPending(s)) scheduleInputCheck();
        });
    }, INPUT_CHECK_DELAY_MS);
}

/**
 * Wire detection once (from ui/behavior-panel.js `setupBehaviorUi`): every
 * edit may have moved the poses the predictions read, so it is noted, and
 * checked once the edits stop.
 * @param {{openSettings?: Function}} [deps]
 */
export function setupBehaviorDetection(deps) {
    if (deps && deps.openSettings) setDetectionSettingsOpener(deps.openSettings);
    onMarkDirty(function () {
        _editGen++;
        var s = state.session;
        if (!s) return;
        notePossibleInputChange(s);
        if (predictionCheckPending(s)) scheduleInputCheck();
    });
}

/**
 * Predictions read from a file take the poses as loaded as their reference:
 * print them shortly after they are first shown (ui/run-buttons.js asks on
 * every draw; this only schedules).
 */
export function ensurePredictionInputChecked(session) {
    if (session && session === state.session && predictionCheckPending(session) && !_checkTimer) scheduleInputCheck();
}
