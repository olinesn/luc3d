// pose/behavior-classifier.js — classifiers trained on behavior labels: what
// the frames a user labelled teach a random forest (pose/behavior-forest.js),
// how well it does on frames it never saw, and its predictions on every frame.
//
// One forest per behavior: "present" against "not". The samples are the
// labelled frames of an animal (for a social behavior, of an actor → target
// pair; a MUTUAL pair gives a sample in each direction, scored as their mean),
// where the animal (and the target) have a pose — at least half the used
// nodes, and not skipped as broken tracking. Unlabelled frames teach nothing.
// The frames an EXCLUSIVE peer was labelled present on count as "not" (the
// group means one at a time); the behavior's own labels win where both say
// something.
//
// ## Gate, samples, rows
//
// A behavior trains once each class has at least `GATE_FRAMES` frames and
// `GATE_BOUTS` bouts (a "not" interval, or a peer's present one, is a bout of
// "not"). Rows are capped (`rowCap`: at most `MAX_TRAINING_ROWS`, and the
// training matrix within `ROW_BUDGET_CELLS` codes): above it, each class keeps
// its frames whose seeded hash is below 2^-k (the same frames for the same
// labels; a little more labelling keeps nearly the same set), plus one frame of
// every bout that would otherwise have none (stratified by bout).
//
// Feature values for the rows are computed on the worker pool, chunk by chunk
// (`TRAIN_CHUNK_FRAMES`, a fixed layout so values never depend on how a run was
// split), and CACHED per session (by name) and chunk, keyed by a print of that
// chunk's poses, the frames skipped there and the feature spec: retraining
// after a few more labels recomputes only the chunks that changed, and a
// session that is no longer resident (a lazy multi-session project evicts the
// sessions not shown) still trains from the rows cached while it was.
//
// ## Cross-validation
//
// Never random frames (neighbouring frames are near copies: the scores would
// be inflated). Folds are, in order: whole sessions, when at least 2 sessions
// have both classes; else whole animals (pairs, for a social behavior); else
// 5 contiguous time blocks, with the frames within the features' reach of a
// held-out block left out of its training. Each fold's forest scores the rows
// it never saw; pooled over folds: frame-level precision, recall and F1, a
// precision–recall curve over thresholds, and bout-level precision, recall and
// F1 (`boutPrf`, after the behavior's post-processing). Bin edges are fitted on
// all rows (they read no label).
//
// ## Predictions
//
// `predictModels` scores every frame (chunks on the pool); `deriveBouts`
// applies each behavior's post-processing (`behaviorPost`: threshold, shortest
// bout, longest gap joined) and, within an exclusive group, keeps only the top
// behavior above its threshold on each frame. A run's source is
// 'model:<hash>'; it goes stale when the labels it learned from change
// (`trainingLabelPrint`), when a newer forest replaces its model, or when the
// settings or poses change (pose/behavior-predictions.js).
//
// Trained forests live in memory only (the registry here), never in the
// project file: the labels are saved, and retraining is seeded, so the same
// labels and poses give the same forest — and the same hash.
//
// DOM-free (Node tests run it end to end, inline).
//
// Depends on: pose/behavior-forest.js, pose/behavior-features.js,
//             pose/behavior-postprocess.js, pose/behavior-pool.js,
//             pose/behavior-ethogram.js, pose/behavior-labels.js,
//             pose/behavior-predictions.js, pose/behavior-tensor.js

import {
    FOREST_VERSION, DEFAULT_FOREST_PARAMS, binEdges, binColumn, finalizeForest, scoreBinnedRows, forestHash, mix32,
    classifierFeatureNames, featureRefs, missingFeatures, scoreForestsChunk, presentFrames,
} from './behavior-forest.js';
import { planChunks, featurePadFrames, sliceChunkData, FEATURE_SPEC_VERSION } from './behavior-features.js';
import { scoresToBouts, secondsToFrames, framePrf, prCurve, boutPrf, exclusiveScores } from './behavior-postprocess.js';
import { runBehaviorChunks, runJob } from './behavior-pool.js';
import { behaviorById, exclusivePeers, behaviorPost } from './behavior-ethogram.js';
import { listBouts, boutIntervals, onBehaviorChange } from './behavior-labels.js';
import { behaviorRunOf, runThreshold } from './behavior-predictions.js';
import { POSE_CHECK_VERSION } from './behavior-tensor.js';

/** Frames and bouts each class needs before a behavior trains. */
export var GATE_FRAMES = 20;
export var GATE_BOUTS = 2;
/** At most this many training rows per behavior. */
export var MAX_TRAINING_ROWS = 40000;
/** …and rows × features within this many codes (2 bytes each while assembled, 1 once binned). */
export var ROW_BUDGET_CELLS = 32 * 1024 * 1024;
/** Frames per chunk when computing training rows (fixed: the values must not depend on the split). */
export var TRAIN_CHUNK_FRAMES = 2048;
/** Frames per chunk when scoring (one animal's features of a chunk are in memory at a time). */
export var PREDICT_CHUNK_FRAMES = 2048;
/** Cross-validation folds at most. */
export var MAX_FOLDS = 5;
/** Trees per job when training on the pool. */
var TREES_PER_JOB = 4;
/** The row cache keeps at most this many bytes of codes (least recently used sessions go first). */
var CACHE_BUDGET_BYTES = 256 * 1024 * 1024;
/** Training workers at most (each holds a copy of the binned matrix). */
var TRAIN_WORKERS = 6;

var SEP = '\u0001';

function fnv(str) {
    var h = 0x811c9dc5;
    for (var i = 0; i < str.length; i++) h = Math.imul(h ^ str.charCodeAt(i), 0x01000193);
    return h >>> 0;
}

function hex(h) {
    return ('0000000' + (h >>> 0).toString(16)).slice(-8);
}

function noPause() { return Promise.resolve(); }

// ----------------------------------------------------------------------------
// The model registry (in memory, project-scoped)
// ----------------------------------------------------------------------------

var _models = new Map();
var _listeners = [];

/** Be told after every change to the trained models (`fn(info)`). */
export function onModelsChange(fn) {
    if (typeof fn === 'function') _listeners.push(fn);
}

function emitModels(info) {
    for (var i = 0; i < _listeners.length; i++) {
        try { _listeners[i](info || {}); } catch (e) { console.error('[behavior] model listener failed', e); }
    }
}

/** Keep a trained (or imported) model as its behavior's classifier, replacing the last one. */
export function setBehaviorModel(model) {
    if (!model || !model.behaviorId) return;
    _models.set(model.behaviorId, model);
    emitModels({ behaviorId: model.behaviorId });
}

/** A behavior's model, or null. */
export function behaviorModelOf(behaviorId) {
    return _models.get(behaviorId) || null;
}

/** Every model, in training order. */
export function behaviorModels() {
    return Array.from(_models.values());
}

/** Forget a behavior's model. */
export function removeBehaviorModel(behaviorId) {
    if (_models.delete(behaviorId)) emitModels({ behaviorId: behaviorId, removed: true });
}

/** Forget every model and every cached row (a new project, or before a load). */
export function resetBehaviorModels() {
    var had = _models.size > 0;
    _models.clear();
    _rowCache.clear();
    _cacheTick = 0;
    noteLabelsChanged();
    if (had) emitModels({ reset: true });
}

/** The hash a run of this behavior should carry now: its model's, or null without one. */
export function modelHashOf(behaviorId) {
    var m = _models.get(behaviorId);
    return m ? m.hash : null;
}

// ----------------------------------------------------------------------------
// Labels → classes
// ----------------------------------------------------------------------------

function sessionAnimalNames(session) {
    return ((session && session.identities) || []).map(function (i) { return i && i.name; }).filter(Boolean);
}

/** Sorted, merged [start, end] list minus sorted, merged `cut`. */
function subtract(ivs, cut) {
    var out = [];
    ivs.forEach(function (iv) {
        var s = iv[0], e = iv[1];
        for (var i = 0; i < cut.length && s <= e; i++) {
            var c = cut[i];
            if (c[1] < s) continue;
            if (c[0] > e) break;
            if (c[0] > s) out.push([s, c[0] - 1]);
            s = c[1] + 1;
        }
        if (s <= e) out.push([s, e]);
    });
    return out;
}

function merge(ivs) {
    ivs.sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; });
    var out = [];
    ivs.forEach(function (iv) {
        var last = out[out.length - 1];
        if (last && iv[0] <= last[1] + 1) last[1] = Math.max(last[1], iv[1]);
        else out.push([iv[0], iv[1]]);
    });
    return out;
}

/**
 * A behavior's labelled intervals in one session, as training classes: per
 * series (actor, target — a mutual pair in name order), `[start, end, y]`
 * with y 1 = present, 0 = "not". The behavior's own labels, plus "not" where
 * an exclusive peer is present on an animal its own labels say nothing about.
 * @returns {Map<string, {actor: string, target: string, ivs: number[][]}>} by 'actor\u0001target'
 */
export function labelClasses(session, b) {
    var out = new Map();
    if (!session || !b) return out;
    var get = function (actor, target) {
        var k = actor + SEP + (target || '');
        var e = out.get(k);
        if (!e) { e = { actor: actor, target: target || '', own: [], peer: [] }; out.set(k, e); }
        return e;
    };
    listBouts(session, { behaviorId: b.id }).forEach(function (bt) {
        get(bt.animal, b.social ? bt.target : '').own.push([bt.start, bt.end, bt.present ? 1 : 0]);
    });
    var peers = exclusivePeers(b);
    if (peers.length) {
        var animals = sessionAnimalNames(session);
        peers.forEach(function (p) {
            listBouts(session, { behaviorId: p.id, present: true }).forEach(function (bt) {
                var who = p.mutual && bt.target ? [bt.animal, bt.target] : [bt.animal];
                who.forEach(function (a) {
                    if (!b.social) { get(a, '').peer.push([bt.start, bt.end]); return; }
                    animals.forEach(function (x) {
                        if (x === a) return;
                        if (b.mutual) get(a < x ? a : x, a < x ? x : a).peer.push([bt.start, bt.end]);
                        else get(a, x).peer.push([bt.start, bt.end]);
                    });
                });
            });
        });
    }
    out.forEach(function (e) {
        var own = e.own.slice().sort(function (x, y) { return x[0] - y[0]; });
        var cover = merge(own.map(function (iv) { return [iv[0], iv[1]]; }));
        var neg = subtract(merge(e.peer), cover).map(function (iv) { return [iv[0], iv[1], 0]; });
        e.ivs = own.concat(neg).sort(function (x, y) { return x[0] - y[0] || x[1] - y[1]; });
        delete e.own; delete e.peer;
    });
    return out;
}

/**
 * What a behavior's labels give to train on, read from the labels alone (no
 * pose): per class, frames and bouts, over the sessions (pair-frames for a
 * social behavior).
 * @returns {{present: {frames, bouts}, absent: {frames, bouts}, sessions: number}}
 */
export function labelCounts(sessions, behaviorId) {
    var b = behaviorById(behaviorId);
    var res = { present: { frames: 0, bouts: 0 }, absent: { frames: 0, bouts: 0 }, sessions: 0 };
    if (!b) return res;
    (sessions || []).forEach(function (s) {
        var cl = labelClasses(s, b);
        if (!cl.size) return;
        res.sessions++;
        cl.forEach(function (e) {
            e.ivs.forEach(function (iv) {
                var c = iv[2] ? res.present : res.absent;
                c.frames += iv[1] - iv[0] + 1;
                c.bouts++;
            });
        });
    });
    return res;
}

/** Why counts do not pass the gate (empty = it trains). */
export function gateProblems(counts) {
    var out = [];
    [['present', 'present'], ['absent', '"not"']].forEach(function (c) {
        var k = counts[c[0]];
        var need = [];
        if (k.frames < GATE_FRAMES) need.push((GATE_FRAMES - k.frames).toLocaleString('en-US') + ' more frame' + (GATE_FRAMES - k.frames === 1 ? '' : 's'));
        if (k.bouts < GATE_BOUTS) need.push((GATE_BOUTS - k.bouts) + ' more bout' + (GATE_BOUTS - k.bouts === 1 ? '' : 's'));
        if (need.length) out.push(c[1] + ' needs ' + need.join(' and '));
    });
    return out;
}

/**
 * A print of what a behavior's classifier learns from: its labels and its
 * exclusive peers' present labels, in every session (by name). A model run
 * is stale once this differs from the print taken when its model trained.
 */
export function trainingLabelPrint(sessions, behaviorId) {
    var b = behaviorById(behaviorId);
    if (!b) return '';
    var peers = exclusivePeers(b).map(function (p) { return p.id; }).sort();
    var parts = [];
    (sessions || []).slice().sort(function (x, y) { return (x.name || '') < (y.name || '') ? -1 : (x.name || '') > (y.name || '') ? 1 : 0; })
        .forEach(function (s) {
            var rows = listBouts(s, { behaviorId: b.id }).map(function (bt) { return [bt.animal, bt.target, bt.start, bt.end, bt.present ? 1 : 0].join(','); });
            peers.forEach(function (pid) {
                listBouts(s, { behaviorId: pid, present: true }).forEach(function (bt) { rows.push([pid, bt.animal, bt.target, bt.start, bt.end].join(',')); });
            });
            if (rows.length) parts.push((s.name || '') + ':' + rows.join(';'));
        });
    var str = JSON.stringify([b.id, !!b.social, !!b.mutual, b.group || '', peers, parts]);
    return hex(fnv(str)) + hex(fnv(str + '#'));
}

// The label print, cached: a draw asks for it (is a classifier run stale?),
// and walking every session's labels on every draw would be too slow. Any
// label edit (`onBehaviorChange`), a load (`noteLabelsChanged`, from the
// metadata reader) or a reset invalidates it.
var _labelGen = 0;
var _printCache = new Map();
onBehaviorChange(function () { _labelGen++; });

/** Labels were replaced without an edit (a file was read): drop the cached prints. */
export function noteLabelsChanged() {
    _labelGen++;
}

/** `trainingLabelPrint`, cached until the labels (or the sessions, or the behavior's group) change. */
export function labelPrintOf(sessions, behaviorId) {
    var b = behaviorById(behaviorId);
    if (!b) return '';
    var key = _labelGen + '|' + (sessions || []).map(function (s) { return s && s.name; }).join(SEP) + '|' + (b.group || '') + '|' +
        exclusivePeers(b).map(function (p) { return p.id; }).join(',') + '|' + (b.social ? 1 : 0) + (b.mutual ? 1 : 0);
    var c = _printCache.get(behaviorId);
    if (c && c.key === key) return c.print;
    var print = trainingLabelPrint(sessions, behaviorId);
    _printCache.set(behaviorId, { key: key, print: print });
    return print;
}

// ----------------------------------------------------------------------------
// The row cache
// ----------------------------------------------------------------------------

// session name -> { tick, sets: Map(setKey -> {kind, names, tick, blocks: Map(chunkIndex ->
//                     {fp, index: Map(rowKey -> col), codes (feature-major), n, F})}) }
// A set is one feature list on one feature spec ('animal#…' / 'pair#…').
var _rowCache = new Map();
var _cacheTick = 0;

function cacheBytes() {
    var total = 0;
    _rowCache.forEach(function (e) { e.sets.forEach(function (st) { st.blocks.forEach(function (bl) { total += bl.codes.byteLength; }); }); });
    return total;
}

function trimCache(keepName) {
    var total = cacheBytes();
    if (total <= CACHE_BUDGET_BYTES) return;
    var names = Array.from(_rowCache.keys()).filter(function (n) { return n !== keepName; })
        .sort(function (a, b) { return _rowCache.get(a).tick - _rowCache.get(b).tick; });
    for (var i = 0; i < names.length && total > CACHE_BUDGET_BYTES; i++) {
        _rowCache.get(names[i]).sets.forEach(function (st) { st.blocks.forEach(function (bl) { total -= bl.codes.byteLength; }); });
        _rowCache.delete(names[i]);
    }
}

function cacheSet(sessionName, setKey, kind, namesKey) {
    var e = _rowCache.get(sessionName);
    if (!e) { e = { tick: 0, sets: new Map() }; _rowCache.set(sessionName, e); }
    e.tick = ++_cacheTick;
    var st = e.sets.get(setKey);
    if (!st) { st = { kind: kind, names: namesKey, tick: 0, blocks: new Map() }; e.sets.set(setKey, st); }
    st.tick = _cacheTick;
    return st.blocks;
}

/**
 * A session's most recently used cached set of this kind and feature list,
 * for training on a session with no poses resident (its rows were computed
 * while it was).
 */
function latestCachedSet(sessionName, kind, namesKey) {
    var e = _rowCache.get(sessionName);
    if (!e) return null;
    var best = null;
    e.sets.forEach(function (st) {
        if (st.kind === kind && st.names === namesKey && st.blocks.size && (!best || st.tick > best.tick)) best = st;
    });
    return best ? best.blocks : null;
}

/** How many rows are cached for a session (all feature sets), for the UI. */
export function cachedRowCount(sessionName) {
    var e = _rowCache.get(sessionName), n = 0;
    if (e) e.sets.forEach(function (st) { st.blocks.forEach(function (bl) { n += bl.n; }); });
    return n;
}

function rowKey(frame, actor, target) {
    return frame + SEP + actor + SEP + (target || '');
}

/** A print of a context's feature spec and a feature set: what makes a cached row valid besides the poses. */
function specPrint(ctx, names) {
    var sp = ctx.spec;
    return hex(fnv(JSON.stringify([FEATURE_SPEC_VERSION, POSE_CHECK_VERSION, FOREST_VERSION, sp.mode, sp.nodeNames, sp.edges, sp.roles,
        sp.used, sp.paws, sp.fps, sp.floor, sp.bodyLength, ctx.input.animals, ctx.input.numFrames > 0 ? 1 : 0]) + '|' + names.join(SEP)));
}

/** A print of one chunk's poses (every animal, the padded range) and the frames skipped there. */
function chunkPrint(ctx, chunk, setKey) {
    var input = ctx.input, N = input.nodeNames.length, D = input.dim;
    var h1 = fnv(setKey + ':' + chunk.padStart + ':' + chunk.padEnd), h2 = h1 ^ 0x5bd1e995;
    input.data.forEach(function (arr, a) {
        var u = new Uint32Array(arr.buffer, arr.byteOffset + chunk.padStart * N * D * 4, (chunk.padEnd - chunk.padStart) * N * D);
        for (var i = 0; i < u.length; i++) {
            h1 = Math.imul(h1 ^ u[i], 0x01000193);
            h2 = Math.imul(h2 ^ (u[i] + i), 0x85ebca6b);
        }
        var m = ctx.broken && ctx.broken.masks && ctx.broken.masks[a];
        if (m) for (var t = chunk.padStart; t < chunk.padEnd; t++) if (m[t]) { h1 = Math.imul(h1 ^ (t + 1), 0x01000193); h2 ^= t * 31 + a; }
    });
    return hex(h1) + hex(h2);
}

// ----------------------------------------------------------------------------
// Samples
// ----------------------------------------------------------------------------

/**
 * Is animal `a` there on frame `t` of a context: at least half its used nodes,
 * and not skipped as broken tracking.
 */
function presentAt(ctx, a, t) {
    if (a < 0) return false;
    var m = ctx.broken && ctx.broken.masks && ctx.broken.masks[a];
    if (m && m[t]) return false;
    var input = ctx.input, N = input.nodeNames.length, D = input.dim, used = ctx.spec.used;
    var arr = input.data[a], base = t * N * D, c = 0, need = Math.max(1, Math.ceil(used.length / 2));
    for (var i = 0; i < used.length; i++) if (arr[base + used[i] * D] === arr[base + used[i] * D]) c++;
    return c >= need;
}

/**
 * The eligible labelled frames of a behavior in one source, by series.
 * @returns {Array<{actor, target, a, t, hash, ivs: Array<{start, end, y, frames: Int32Array}>}>}
 */
function sourceCandidates(src, b, namesKey) {
    var out = [];
    var classes = labelClasses(src.session, b);
    var ctx = src.ctx;
    var cached = !ctx ? cachedSampleIndex(src, b, namesKey) : null;
    classes.forEach(function (e) {
        var a = ctx ? ctx.input.animals.indexOf(e.actor) : -1;
        var t = b.social ? (ctx ? ctx.input.animals.indexOf(e.target) : -1) : -1;
        var series = { actor: e.actor, target: e.target, a: a, t: t, hash: fnv((src.name || '') + SEP + e.actor + SEP + e.target), ivs: [] };
        e.ivs.forEach(function (iv) {
            var frames = [];
            for (var f = iv[0]; f <= iv[1]; f++) {
                if (ctx) {
                    if (f >= ctx.input.numFrames || !presentAt(ctx, a, f) || (b.social && !presentAt(ctx, t, f))) continue;
                } else if (!cached || !cached.has(rowKey(f, e.actor, e.target))) continue;
                frames.push(f);
            }
            series.ivs.push({ start: iv[0], end: iv[1], y: iv[2], frames: Int32Array.from(frames) });
        });
        out.push(series);
    });
    return out;
}

/** For a source without poses: the row keys its cache holds for this behavior's feature set. */
function cachedSampleIndex(src, b, namesKey) {
    var blocks = latestCachedSet(src.name, b.social ? 'pair' : 'animal', namesKey);
    if (!blocks) return null;
    var keys = new Set();
    blocks.forEach(function (bl) { bl.index.forEach(function (col, k) { keys.add(k); }); });
    return keys;
}

/** The most rows a behavior with `F` features trains on. */
export function rowCap(F, social) {
    var perRow = social ? 2 : 1;     // rows per sample (a mutual pair gives two; counted conservatively)
    return Math.max(GATE_FRAMES * 2, Math.min(MAX_TRAINING_ROWS, Math.floor(ROW_BUDGET_CELLS / Math.max(1, F) / perRow)));
}

/**
 * Pick the samples: everything under the cap, else per class the frames whose
 * seeded hash is under 2^-k, plus one frame of every bout left without any.
 * @returns {{samples: Array<[srcIndex, seriesIndex, frame, y, boutId]>, counts, rates: number[]}}
 */
function pickSamples(candidates, cap, seed) {
    var n = [0, 0], bouts = [0, 0];
    candidates.forEach(function (sc) {
        sc.series.forEach(function (se) {
            se.ivs.forEach(function (iv) { if (iv.frames.length) { n[iv.y] += iv.frames.length; bouts[iv.y]++; } });
        });
    });
    var rates = [1, 1];
    if (n[0] + n[1] > cap) {
        var small = n[0] < n[1] ? 0 : 1, large = 1 - small;
        var capSmall = Math.min(n[small], Math.floor(cap / 2)), capLarge = cap - capSmall;
        var caps = [];
        caps[small] = capSmall; caps[large] = capLarge;
        [0, 1].forEach(function (c) {
            var r = 1;
            while (n[c] * r > caps[c] && r > 1 / (1 << 20)) r /= 2;
            rates[c] = r;
        });
    }
    var samples = [];
    var boutId = 0;
    candidates.forEach(function (sc, si) {
        sc.series.forEach(function (se, ji) {
            se.ivs.forEach(function (iv) {
                var id = boutId++;
                if (!iv.frames.length) return;
                var rate = rates[iv.y];
                var kept = 0, bestF = -1, bestU = Infinity;
                for (var i = 0; i < iv.frames.length; i++) {
                    var f = iv.frames[i];
                    var u = rate >= 1 ? 0 : mix32(se.hash, f, seed) / 4294967296;
                    if (u < rate) { samples.push([si, ji, f, iv.y, id]); kept++; }
                    else if (u < bestU) { bestU = u; bestF = f; }
                }
                if (!kept && bestF >= 0) samples.push([si, ji, bestF, iv.y, id]);
            });
        });
    });
    samples.sort(function (x, y) { return x[0] - y[0] || x[1] - y[1] || x[2] - y[2]; });
    return { samples: samples, counts: { present: { frames: n[1], bouts: bouts[1] }, absent: { frames: n[0], bouts: bouts[0] } }, rates: rates };
}

// ----------------------------------------------------------------------------
// Folds
// ----------------------------------------------------------------------------

/**
 * Grouped folds for rows with a session, a group (animal or pair) and a
 * frame: by session when ≥ 2 sessions have both classes, else by group when
 * ≥ 2 groups do, else 5 contiguous time blocks (by session, then frame) whose
 * training leaves out the rows within `gap` frames of the block. Groups are
 * dealt round-robin into at most MAX_FOLDS folds. Rows of a session (group)
 * with one class only are never held out.
 * @param {{session: Int32Array, group: Int32Array, frame: Int32Array, y: Uint8Array}} rows
 * @param {number} gap
 * @returns {{scheme: 'session'|'animal'|'time', folds: Array<{test: Int32Array, train: Int32Array, label: string}>}}
 */
export function planFolds(rows, gap, labels) {
    var R = rows.y.length;
    var both = function (keyArr) {
        var seen = new Map();
        for (var r = 0; r < R; r++) {
            var k = keyArr[r], s = seen.get(k) || 0;
            seen.set(k, s | (rows.y[r] ? 2 : 1));
        }
        return Array.from(seen.keys()).filter(function (k) { return seen.get(k) === 3; }).sort(function (a, b) { return a - b; });
    };
    var mk = function (keyArr, keys, scheme, names) {
        var nf = Math.min(MAX_FOLDS, keys.length);
        var foldOf = new Map();
        keys.forEach(function (k, i) { foldOf.set(k, i % nf); });
        var folds = [];
        for (var f = 0; f < nf; f++) {
            var test = [], train = [];
            for (var r = 0; r < R; r++) (foldOf.get(keyArr[r]) === f ? test : train).push(r);
            folds.push({ test: Int32Array.from(test), train: Int32Array.from(train),
                         label: keys.filter(function (k, i) { return i % nf === f; }).map(function (k) { return names ? names[k] : String(k); }).join(', ') });
        }
        return { scheme: scheme, folds: folds };
    };
    var sess = both(rows.session);
    if (sess.length >= 2) return mk(rows.session, sess, 'session', labels && labels.sessions);
    var groups = both(rows.group);
    if (groups.length >= 2) return mk(rows.group, groups, 'animal', labels && labels.groups);
    // Time blocks: rows by (session, frame), cut into equal counts.
    var order = Array.from({ length: R }, function (_, i) { return i; }).sort(function (a, b) {
        return rows.session[a] - rows.session[b] || rows.frame[a] - rows.frame[b] || a - b;
    });
    var nb = Math.min(MAX_FOLDS, Math.max(2, Math.floor(R / 10)));
    var folds = [];
    for (var f = 0; f < nb; f++) {
        var lo = Math.floor(f * R / nb), hi = Math.floor((f + 1) * R / nb);
        if (hi <= lo) continue;
        var inBlock = new Uint8Array(R);
        var spans = new Map();     // session -> [minFrame, maxFrame] of the block
        for (var i = lo; i < hi; i++) {
            var r2 = order[i];
            inBlock[r2] = 1;
            var sp = spans.get(rows.session[r2]);
            if (!sp) spans.set(rows.session[r2], [rows.frame[r2], rows.frame[r2]]);
            else { sp[0] = Math.min(sp[0], rows.frame[r2]); sp[1] = Math.max(sp[1], rows.frame[r2]); }
        }
        var test = [], train = [];
        for (var r3 = 0; r3 < R; r3++) {
            if (inBlock[r3]) { test.push(r3); continue; }
            var s3 = spans.get(rows.session[r3]);
            if (s3 && rows.frame[r3] >= s3[0] - gap && rows.frame[r3] <= s3[1] + gap) continue;
            train.push(r3);
        }
        folds.push({ test: Int32Array.from(test), train: Int32Array.from(train), label: 'block ' + (f + 1) });
    }
    return { scheme: 'time', folds: folds };
}

// ----------------------------------------------------------------------------
// Training
// ----------------------------------------------------------------------------

/**
 * Compute (or take from the cache) the rows of one source and feature set.
 * @param {{name: string}} src
 * @param {Object} ctx the source's context
 * @param {string} setKey
 * @param {string[]} names the set's feature names
 * @param {Object[]} refs the same, resolved (`featureRefs`)
 * @param {Set<string>} keys the rows wanted (`rowKey`)
 * @returns {Promise<Map<number, Object>>} the set's cached blocks, now holding every wanted row
 */
async function sourceRows(src, ctx, setKey, names, refs, keys, opts) {
    var blocks = cacheSet(src.name, setKey, setKey.split('#')[0], opts.namesKey);
    var input = ctx.input, T = input.numFrames;
    var chunks = planChunks(T, featurePadFrames(ctx.spec), TRAIN_CHUNK_FRAMES);
    var need = new Map();
    keys.forEach(function (k) {
        var p = k.split(SEP);
        var ci = Math.floor(+p[0] / TRAIN_CHUNK_FRAMES);
        if (!need.has(ci)) need.set(ci, []);
        need.get(ci).push(p);
    });
    var todo = [];
    need.forEach(function (list, ci) {
        var ch = chunks[ci];
        if (!ch) return;
        var fp = chunkPrint(ctx, ch, setKey);
        var bl = blocks.get(ci);
        var keep = bl && bl.fp === fp ? bl : null;
        var missing = keep ? list.filter(function (p) { return !keep.index.has(rowKey(p[0], p[1], p[2])); }) : list;
        if (missing.length) todo.push({ ci: ci, chunk: ch, fp: fp, keep: keep, rows: missing });
    });
    if (!todo.length) return blocks;
    var N = input.nodeNames.length, D = input.dim, F = names.length;
    var animalIndex = new Map(input.animals.map(function (nm, i) { return [nm, i]; }));
    await runBehaviorChunks(todo.length, function (i) {
        var td = todo[i];
        var arr = new Int32Array(td.rows.length * 3);
        td.rows.forEach(function (p, j) {
            arr[3 * j] = +p[0]; arr[3 * j + 1] = animalIndex.get(p[1]); arr[3 * j + 2] = p[2] ? animalIndex.get(p[2]) : -1;
        });
        return { kind: 'rows', spec: ctx.spec, chunk: td.chunk, data: sliceChunkData(input.data, td.chunk, N * D, ctx.broken ? ctx.broken.masks : null),
                 request: { sets: [{ refs: refs, rows: arr }] } };
    }, function (i, res) {
        var td = todo[i];
        if (!res) return;
        var codes = res.sets[0].codes, n = td.rows.length, old = td.keep;
        var total = (old ? old.n : 0) + n, o = old ? old.n : 0;
        var merged = new Uint16Array(F * total);
        var index = old ? new Map(old.index) : new Map();
        for (var j = 0; j < F; j++) {
            if (old) merged.set(old.codes.subarray(j * old.n, (j + 1) * old.n), j * total);
            merged.set(codes.subarray(j * n, (j + 1) * n), j * total + o);
        }
        td.rows.forEach(function (p, r) { index.set(rowKey(p[0], p[1], p[2]), o + r); });
        blocks.set(td.ci, { fp: td.fp, index: index, codes: merged, n: total, F: F });
        _rowsComputed += n;
    }, { workload: todo.length * TRAIN_CHUNK_FRAMES * Math.max(1, input.animals.length), onProgress: opts.onChunk,
         isCancelled: opts.isCancelled, pause: opts.pause });
    return blocks;
}

var _rowsComputed = 0;

/** How many training rows had their features computed (not taken from the cache) since the page loaded. */
export function trainingRowsComputed() {
    return _rowsComputed;
}

/**
 * Train classifiers for several behaviors at once, with cross-validation.
 * Each session's poses are read twice in all (its samples, then its rows),
 * whatever the number of behaviors, and rows shared by behaviors of the same
 * kind (social or not) are computed once.
 *
 * @param {Object} opts
 * @param {string[]} opts.behaviorIds
 * @param {Array<{session: Session, name: string, ctx?: Object, getCtx?: function(): ?Object}>} opts.sources
 *   the sessions to learn from: a behavior context (pose/behavior-tensor.js input, spec, catalog,
 *   body, broken masks — ui/behavior-detect.js `buildBehaviorContext`), or a getter that builds one
 *   (called when needed, so only one session's poses are held at a time), or neither: a session
 *   whose poses are not resident trains from the rows cached while they were
 * @param {string} [opts.reference] the source whose feature list every other must measure (default:
 *   the first with poses)
 * @param {Object} [opts.params] forest parameters (DEFAULT_FOREST_PARAMS)
 * @param {boolean} [opts.crossValidate=true]
 * @param {number} [opts.maxRows] a lower cap on the rows per behavior (default `rowCap`)
 * @param {Session[]} [opts.allSessions] for the label print (default: the sources' sessions)
 * @param {function(Object)} [opts.onProgress] `{stage: 'read'|'rows'|'bins'|'trees', done, total, behaviorId?}`
 * @param {function(): Promise} [opts.pause] between slices of main-thread work
 * @param {function(): boolean} [opts.isCancelled]
 * @returns {Promise<{results: Array<{behaviorId, model?: Object, report: ?Object, error?: string}>, cancelled: boolean}>}
 */
export async function trainBehaviorClassifiers(opts) {
    var params = Object.assign({}, DEFAULT_FOREST_PARAMS, opts.params || {});
    var pause = opts.pause || noPause;
    var cancelled = opts.isCancelled || function () { return false; };
    var progress = opts.onProgress || function () {};
    var list = (opts.behaviorIds || []).map(behaviorById).filter(Boolean);
    var results = list.map(function (b) { return { behaviorId: b.id, report: null }; });
    var sources = (opts.sources || []).slice().sort(function (x, y) { return x.name < y.name ? -1 : x.name > y.name ? 1 : 0; });
    var ctxOf = function (s) { return s.ctx || (s.getCtx ? s.getCtx() : null); };
    // The reference context: what every session must measure.
    var refSrc = null, refCtx = null;
    var order = opts.reference ? sources.filter(function (s) { return s.name === opts.reference; }).concat(sources) : sources;
    for (var i0 = 0; i0 < order.length && !refCtx; i0++) {
        var c0 = ctxOf(order[i0]);
        if (c0 && c0.input.animals.length) { refSrc = order[i0]; refCtx = c0; }
    }
    if (!refCtx) {
        results.forEach(function (r) { r.error = 'No session with identified poses to learn from (run Track All, or open a labelled session)'; });
        return { results: results, cancelled: false };
    }
    var sets = {};
    ['animal', 'pair'].forEach(function (kind) {
        var names = classifierFeatureNames(refCtx.catalog, kind === 'pair', refCtx.spec.roles);
        sets[kind] = { names: names, refs: featureRefs(refCtx.catalog, names, kind === 'pair'), namesKey: hex(fnv(names.join(SEP))) };
    });
    var kindOf = function (b) { return b.social ? 'pair' : 'animal'; };
    // Phase A: each session's eligible labelled frames, per behavior.
    var cands = list.map(function () { return new Array(sources.length).fill(null); });
    var leftOut = list.map(function () { return []; });
    var setKeys = sources.map(function () { return {}; });
    for (var si = 0; si < sources.length; si++) {
        if (cancelled()) return { results: results, cancelled: true };
        progress({ stage: 'read', done: si, total: sources.length, session: sources[si].name });
        var s = sources[si];
        var anyLabels = list.some(function (b) { return labelClasses(s.session, b).size; });
        if (!anyLabels) continue;
        var ctx = s === refSrc ? refCtx : ctxOf(s);
        if (ctx && !ctx.input.animals.length) ctx = null;
        var ok = {};
        ['animal', 'pair'].forEach(function (kind) {
            if (!ctx) { ok[kind] = true; return; }
            var miss = missingFeatures(ctx.catalog, sets[kind].names, kind === 'pair');
            ok[kind] = !miss.length && ctx.input.mode === refCtx.input.mode;
            if (ok[kind]) setKeys[si][kind] = kind + '#' + specPrint(ctx, sets[kind].names);
            else setKeys[si][kind] = { why: ctx.input.mode !== refCtx.input.mode ? 'its poses are ' + ctx.input.mode.toUpperCase() : 'a different body (' + miss[0] + ')' };
        });
        list.forEach(function (b, bi) {
            if (!labelClasses(s.session, b).size) return;
            var kind = kindOf(b);
            if (!ok[kind]) { leftOut[bi].push({ session: s.name, reason: setKeys[si][kind].why }); return; }
            var series = sourceCandidates({ session: s.session, name: s.name, ctx: ctx }, b, sets[kind].namesKey);
            if (!series.some(function (se) { return se.ivs.some(function (iv) { return iv.frames.length; }); })) {
                leftOut[bi].push({ session: s.name, reason: ctx ? 'no labelled frame has a pose' : 'not open, and none of its rows are cached: open it once to include it' });
                return;
            }
            cands[bi][si] = series;
        });
        await pause();
    }
    progress({ stage: 'read', done: sources.length, total: sources.length });
    // Phase B: samples, gate, rows wanted.
    var plans = list.map(function (b, bi) {
        var kind = kindOf(b);
        var withSrc = [];
        cands[bi].forEach(function (series, si2) { if (series) withSrc.push({ si: si2, series: series }); });
        var F = sets[kind].names.length;
        var cap = rowCap(F, b.social && b.mutual);
        if (opts.maxRows > 0) cap = Math.min(cap, Math.floor(opts.maxRows / (b.social && b.mutual ? 2 : 1)));
        var picked = pickSamples(withSrc, cap, params.seed);
        var report = {
            behaviorId: b.id, name: b.name, counts: picked.counts, gate: gateProblems(picked.counts), features: F, rows: 0,
            sampled: picked.rates[0] < 1 || picked.rates[1] < 1, sessions: withSrc.map(function (w) { return sources[w.si].name; }),
            leftOut: leftOut[bi], cv: null,
        };
        results[bi].report = report;
        if (report.gate.length) { results[bi].error = 'Not enough labels: ' + report.gate.join('; '); return null; }
        if (b.social && !withSrc.some(function (w) { return w.series.some(function (se) { return se.target; }); })) {
            results[bi].error = 'A social behavior needs labels with a target'; return null;
        }
        var rowsMeta = [];   // [sourceIndex, series, frame, y, bout, dir]
        picked.samples.forEach(function (sm) {
            var w = withSrc[sm[0]], se = w.series[sm[1]];
            rowsMeta.push([w.si, se, sm[2], sm[3], sm[4], 0]);
            if (b.social && b.mutual) rowsMeta.push([w.si, se, sm[2], sm[3], sm[4], 1]);
        });
        report.rows = rowsMeta.length;
        return { b: b, kind: kind, rowsMeta: rowsMeta, report: report };
    });
    var keyOfRow = function (m) { return m[5] ? rowKey(m[2], m[1].target, m[1].actor) : rowKey(m[2], m[1].actor, m[1].target); };
    // Phase C: the rows, session by session (each session's poses read once more).
    var want = sources.map(function () { return { animal: new Set(), pair: new Set() }; });
    var totalRows = 0;
    plans.forEach(function (p) { if (p) p.rowsMeta.forEach(function (m) { want[m[0]][p.kind].add(keyOfRow(m)); }); });
    want.forEach(function (w) { totalRows += w.animal.size + w.pair.size; });
    var blocksOf = sources.map(function () { return {}; });
    var doneRows = 0;
    progress({ stage: 'rows', done: 0, total: totalRows });
    for (var sj = 0; sj < sources.length; sj++) {
        if (cancelled()) return { results: results, cancelled: true };
        var w = want[sj];
        if (!w.animal.size && !w.pair.size) continue;
        var src = sources[sj];
        var ctx2 = src === refSrc ? refCtx : ctxOf(src);
        if (ctx2 && !ctx2.input.animals.length) ctx2 = null;
        for (var kind of ['animal', 'pair']) {
            var keys = w[kind];
            if (!keys.size) continue;
            if (!ctx2) {
                blocksOf[sj][kind] = latestCachedSet(src.name, kind, sets[kind].namesKey) || new Map();
            } else {
                var base = doneRows, size = keys.size;
                blocksOf[sj][kind] = await sourceRows(src, ctx2, setKeys[sj][kind], sets[kind].names, sets[kind].refs, keys, {
                    namesKey: sets[kind].namesKey, isCancelled: cancelled, pause: pause,
                    onChunk: function (d, n) { progress({ stage: 'rows', done: base + Math.round(size * d / Math.max(1, n)), total: totalRows }); },
                });
            }
            doneRows += keys.size;
            progress({ stage: 'rows', done: doneRows, total: totalRows });
        }
    }
    trimCache(refSrc.name);
    // Phase D: per behavior, the matrix, the folds, the trees.
    for (var pi = 0; pi < plans.length; pi++) {
        var plan = plans[pi];
        if (!plan) continue;
        if (cancelled()) return { results: results, cancelled: true };
        var out = await fitBehavior(plan, sets[plan.kind], blocksOf, keyOfRow, refCtx, sources, params, {
            crossValidate: opts.crossValidate, pause: pause, isCancelled: cancelled, progress: progress,
            allSessions: opts.allSessions || sources.map(function (sx) { return sx.session; }),
        });
        if (out.cancelled) return { results: results, cancelled: true };
        results[pi].model = out.model;
        if (out.error) results[pi].error = out.error;
    }
    return { results: results, cancelled: false };
}

/** `trainBehaviorClassifiers` for one behavior: `{model?, report, error?}`. */
export async function trainBehaviorClassifier(opts) {
    var res = await trainBehaviorClassifiers(Object.assign({}, opts, { behaviorIds: [opts.behaviorId] }));
    var r = res.results[0] || { report: null, error: 'No such behavior' };
    if (res.cancelled && !r.error) r.error = 'Cancelled';
    return r;
}

/** One behavior's matrix, folds, trees and model (`trainBehaviorClassifiers` phase D). */
async function fitBehavior(plan, set, blocksOf, keyOfRow, refCtx, sources, params, o) {
    var b = plan.b, rowsMeta = plan.rowsMeta, report = plan.report;
    var social = !!b.social, mutual = social && !!b.mutual;
    var F = set.names.length, R = rowsMeta.length;
    var progress = function (stage, done, total) { o.progress({ stage: stage, done: done, total: total, behaviorId: b.id }); };
    // Codes, feature-major, then bins.
    var codes = new Uint16Array(F * R);
    var y = new Uint8Array(R);
    var missingRows = 0;
    for (var r = 0; r < R; r++) {
        var m = rowsMeta[r];
        y[r] = m[3];
        var k = keyOfRow(m);
        var blocks = blocksOf[m[0]][plan.kind];
        var bl = blocks && blocks.get(Math.floor(m[2] / TRAIN_CHUNK_FRAMES));
        var col = bl ? bl.index.get(k) : undefined;
        if (col == null || bl.F !== F) { missingRows++; continue; }
        for (var j = 0; j < F; j++) codes[j * R + r] = bl.codes[j * bl.n + col];
    }
    if (missingRows) console.warn('[behavior] ' + missingRows + ' training rows of "' + b.name + '" had no features and read as missing');
    var hist = new Uint32Array(65536), lut = new Uint8Array(65536);
    var bins = new Uint8Array(F * R), nb = new Uint16Array(F), edges = new Array(F);
    for (var f = 0; f < F; f++) {
        edges[f] = binEdges(codes, f * R, R, hist);
        nb[f] = edges[f].length + 1;
        binColumn(codes, f * R, R, edges[f], bins, f * R, lut);
        if ((f & 127) === 127) {
            progress('bins', f + 1, F);
            await o.pause();
            if (o.isCancelled()) return { cancelled: true };
        }
    }
    codes = null;
    progress('bins', F, F);
    // Each row's session, group (the animal, or the pair) and frame: the folds and the bout metrics.
    var rowSession = new Int32Array(R), rowGroup = new Int32Array(R), rowFrame = new Int32Array(R), rowSeries = new Int32Array(R);
    var groupOf = function (se) { return social ? [se.actor, se.target].sort().join(' ↔ ') : se.actor; };
    var groupNames = Array.from(new Set(rowsMeta.map(function (m) { return groupOf(m[1]); }))).sort();
    var groupIndex = new Map(groupNames.map(function (g, i) { return [g, i]; })), seriesIndex = new Map();
    for (var r2 = 0; r2 < R; r2++) {
        var m2 = rowsMeta[r2], se = m2[1];
        rowSession[r2] = m2[0]; rowGroup[r2] = groupIndex.get(groupOf(se)); rowFrame[r2] = m2[2];
        if (!seriesIndex.has(se)) seriesIndex.set(se, seriesIndex.size);
        rowSeries[r2] = seriesIndex.get(se);
    }
    var gap = featurePadFrames(refCtx.spec);
    var folding = o.crossValidate === false ? { scheme: 'none', folds: [] }
        : planFolds({ session: rowSession, group: rowGroup, frame: rowFrame, y: y }, gap,
            { sessions: sources.map(function (s) { return s.name; }), groups: groupNames });
    var folds = folding.folds.filter(function (fd) {
        var has = 0;
        for (var i = 0; i < fd.train.length; i++) has |= y[fd.train[i]] ? 2 : 1;
        return has === 3 && fd.test.length;
    });
    var all = new Int32Array(R);
    for (var r3 = 0; r3 < R; r3++) all[r3] = r3;
    var foldRows = folds.map(function (fd) { return fd.train; }).concat([all]);
    var data = { bins: bins, y: y, R: R, F: F, nb: nb, folds: foldRows };
    var finalFold = folds.length;
    var jobs = [];
    for (var fi = 0; fi <= finalFold; fi++) {
        for (var t0 = 0; t0 < params.trees; t0 += TREES_PER_JOB) {
            var ts = [];
            for (var ti = t0; ti < Math.min(params.trees, t0 + TREES_PER_JOB); ti++) ts.push(ti);
            jobs.push({ fold: fi, trees: ts });
        }
    }
    var treesOf = foldRows.map(function () { return new Array(params.trees); });
    var impParts = new Array(jobs.length);
    var doneTrees = 0, totalTrees = (finalFold + 1) * params.trees;
    progress('trees', 0, totalTrees);
    var trainParams = { maxDepth: params.maxDepth, minLeaf: params.minLeaf, mtry: params.mtry };
    var run = await runBehaviorChunks(jobs.length, function (i) {
        return { kind: 'trees', data: [], trees: { trees: jobs[i].trees, fold: jobs[i].fold === finalFold ? -1 : jobs[i].fold,
                 seed: params.seed, params: trainParams, rowsIndex: jobs[i].fold } };
    }, function (i, outp) {
        var jb = jobs[i];
        outp.trees.forEach(function (tr, k2) { treesOf[jb.fold][jb.trees[k2]] = tr; });
        if (jb.fold === finalFold) impParts[i] = outp.imp;
        doneTrees += jb.trees.length;
        progress('trees', doneTrees, totalTrees);
    }, { workload: R * params.trees, shared: { key: 'train:' + b.id + ':' + (++_trainRuns), value: data }, releaseShared: true,
         maxWorkers: TRAIN_WORKERS, isCancelled: o.isCancelled, pause: o.pause });
    if (run.cancelled || o.isCancelled()) return { cancelled: true };
    // Importance summed in job order, so it is the same however the jobs were spread.
    var impSum = new Float64Array(F);
    impParts.forEach(function (part) { if (part) for (var q = 0; q < F; q++) impSum[q] += part[q]; });
    // Cross-validation: each fold's forest on the rows it never saw.
    var oof = new Float32Array(R).fill(NaN);
    folds.forEach(function (fd, i) {
        var sc = scoreBinnedRows(treesOf[i], data, fd.test);
        for (var q = 0; q < fd.test.length; q++) oof[fd.test[q]] = sc[q];
    });
    var post = behaviorPost(b);
    var fps = refCtx.fps || refCtx.spec.fps;
    if (folds.length) {
        report.cv = evaluate(oof, y, rowsMeta, rowSeries, mutual, post, fps, b.kind === 'point');
        report.cv.scheme = folding.scheme;
        report.cv.folds = folds.map(function (fd) { return fd.label; });
        report.cv.trees = params.trees;
    } else if (o.crossValidate !== false) {
        report.cv = { scheme: folding.scheme, folds: [], note: 'too few labels to hold any out' };
    }
    var fin = finalizeForest(treesOf[finalFold], edges, set.names, impSum);
    var names = refCtx.input.nodeNames;
    var model = {
        behaviorId: b.id, name: b.name, kind: b.kind, social: social, mutual: mutual, group: b.group || '',
        hash: forestHash(fin.forest, fin.features), features: fin.features, forest: fin.forest, importance: fin.importance,
        params: { trees: params.trees, maxDepth: params.maxDepth, minLeaf: params.minLeaf, seed: params.seed,
                  mtry: params.mtry || Math.max(1, Math.round(Math.sqrt(F))) },
        versions: { forest: FOREST_VERSION, features: FEATURE_SPEC_VERSION, poseCheck: POSE_CHECK_VERSION },
        mode: refCtx.input.mode, nodes: names.slice(),
        body: { head: names[refCtx.spec.roles.head] || '', tailBase: names[refCtx.spec.roles.tailBase] || '',
                used: refCtx.spec.used.map(function (i) { return names[i]; }) },
        catalogSize: F,
        training: { sessions: report.sessions, rows: R, sampled: report.sampled, counts: report.counts, leftOut: report.leftOut },
        cv: report.cv,
        labelPrint: trainingLabelPrint(o.allSessions, b.id),
        imported: false,
    };
    report.hash = model.hash;
    report.top = topFeatures(model, 10);
    return { model: model };
}

var _trainRuns = 0;

/** A model's most important features: `[{name, importance}]`, highest first. */
export function topFeatures(model, k) {
    if (!model || !model.importance) return [];
    var idx = Array.from(model.importance, function (v, i) { return i; });
    idx.sort(function (a, b) { return model.importance[b] - model.importance[a] || a - b; });
    return idx.slice(0, k || 10).map(function (i) { return { name: model.features[i], importance: model.importance[i] }; });
}

/**
 * Pooled cross-validation metrics from out-of-fold scores: per sample (a
 * mutual pair-frame's two rows averaged), frame-level P/R/F1 at the
 * behavior's threshold and over a threshold curve; bout-level P/R/F1 from
 * each series' tested frames (a row's score held until the next tested frame
 * of its labelled interval, so rows sampled from long bouts still make bouts).
 */
function evaluate(oof, y, rowsMeta, rowSeries, mutual, post, fps, point) {
    var units = [];   // [score, y, seriesIdx, frame, bout]
    for (var r = 0; r < oof.length; r++) {
        var m = rowsMeta[r];
        if (mutual) {
            if (m[5]) continue;
            var a = oof[r], b = oof[r + 1];
            var s = a !== a ? b : b !== b ? a : (a + b) / 2;
            if (s === s) units.push([s, y[r], rowSeries[r], m[2], m[4]]);
        } else if (oof[r] === oof[r]) units.push([oof[r], y[r], rowSeries[r], m[2], m[4]]);
    }
    var scores = Float32Array.from(units.map(function (u) { return u[0]; }));
    var truth = Uint8Array.from(units.map(function (u) { return u[1]; }));
    var frame = framePrf(scores, truth, post.threshold);
    var curve = prCurve(scores, truth, 50);
    // Bouts, per series.
    var bySeries = new Map();
    units.forEach(function (u) { if (!bySeries.has(u[2])) bySeries.set(u[2], []); bySeries.get(u[2]).push(u); });
    var minFrames = Math.max(1, secondsToFrames(post.minBout, fps)), maxGap = secondsToFrames(post.maxGap, fps);
    var agg = { predicted: 0, hits: 0, truth: 0, found: 0 };
    bySeries.forEach(function (list) {
        list.sort(function (p, q) { return p[3] - q[3]; });
        var lo = list[0][3], hi = list[list.length - 1][3];
        var dense = new Float32Array(hi - lo + 1).fill(NaN);
        var truthIvs = new Map();
        for (var i = 0; i < list.length; i++) {
            var u = list[i];
            var nextSame = i + 1 < list.length && list[i + 1][4] === u[4] ? list[i + 1][3] : u[3] + 1;
            dense.fill(u[0], u[3] - lo, nextSame - lo);
            if (u[1]) {
                var tv = truthIvs.get(u[4]);
                if (!tv) truthIvs.set(u[4], [u[3], u[3]]); else tv[1] = u[3];
            }
        }
        var pred = scoresToBouts(dense, { threshold: post.threshold, minFrames: minFrames, maxGapFrames: maxGap, point: point })
            .map(function (bt) { return [bt[0] + lo, bt[1] + lo]; });
        var tr = Array.from(truthIvs.values()).sort(function (p, q) { return p[0] - q[0]; });
        var m2 = boutPrf(pred, tr);
        agg.predicted += m2.predicted; agg.hits += m2.hits; agg.truth += m2.truth; agg.found += m2.found;
    });
    var bp = agg.predicted ? agg.hits / agg.predicted : NaN, br = agg.truth ? agg.found / agg.truth : NaN;
    var bf = bp + br > 0 ? 2 * bp * br / (bp + br) : (bp === 0 || br === 0 ? 0 : NaN);
    return {
        tested: units.length, threshold: post.threshold,
        frame: { precision: frame.precision, recall: frame.recall, f1: frame.f1, tp: frame.tp, fp: frame.fp, fn: frame.fn, tn: frame.tn },
        bout: { precision: bp, recall: br, f1: bf, predicted: agg.predicted, truth: agg.truth },
        curve: curve.points, best: curve.best,
        // Kept so the threshold dialog can recompute bout metrics without retraining.
        oof: { scores: scores, truth: truth, series: Int32Array.from(units.map(function (u) { return u[2]; })),
               frames: Int32Array.from(units.map(function (u) { return u[3]; })), bouts: Int32Array.from(units.map(function (u) { return u[4]; })) },
    };
}

/**
 * Cross-validated metrics of a model at other post-processing settings, from
 * its kept out-of-fold scores (the threshold dialog's live numbers).
 * @returns {{frame: Object, bout: Object}|null}
 */
export function evaluateAt(cv, post, fps, point) {
    if (!cv || !cv.oof) return null;
    var o = cv.oof;
    var frame = framePrf(o.scores, o.truth, post.threshold);
    var bySeries = new Map();
    for (var i = 0; i < o.scores.length; i++) {
        if (!bySeries.has(o.series[i])) bySeries.set(o.series[i], []);
        bySeries.get(o.series[i]).push(i);
    }
    var minFrames = Math.max(1, secondsToFrames(post.minBout, fps)), maxGap = secondsToFrames(post.maxGap, fps);
    var agg = { predicted: 0, hits: 0, truth: 0, found: 0 };
    bySeries.forEach(function (list) {
        list.sort(function (p, q) { return o.frames[p] - o.frames[q]; });
        var lo = o.frames[list[0]], hi = o.frames[list[list.length - 1]];
        var dense = new Float32Array(hi - lo + 1).fill(NaN);
        var truthIvs = new Map();
        for (var k = 0; k < list.length; k++) {
            var u = list[k];
            var next = k + 1 < list.length && o.bouts[list[k + 1]] === o.bouts[u] ? o.frames[list[k + 1]] : o.frames[u] + 1;
            dense.fill(o.scores[u], o.frames[u] - lo, next - lo);
            if (o.truth[u]) {
                var tv = truthIvs.get(o.bouts[u]);
                if (!tv) truthIvs.set(o.bouts[u], [o.frames[u], o.frames[u]]); else tv[1] = o.frames[u];
            }
        }
        var pred = scoresToBouts(dense, { threshold: post.threshold, minFrames: minFrames, maxGapFrames: maxGap, point: point })
            .map(function (bt) { return [bt[0] + lo, bt[1] + lo]; });
        var m2 = boutPrf(pred, Array.from(truthIvs.values()).sort(function (p, q) { return p[0] - q[0]; }));
        agg.predicted += m2.predicted; agg.hits += m2.hits; agg.truth += m2.truth; agg.found += m2.found;
    });
    var bp = agg.predicted ? agg.hits / agg.predicted : NaN, br = agg.truth ? agg.found / agg.truth : NaN;
    return {
        frame: { precision: frame.precision, recall: frame.recall, f1: frame.f1 },
        bout: { precision: bp, recall: br, f1: bp + br > 0 ? 2 * bp * br / (bp + br) : (bp === 0 || br === 0 ? 0 : NaN),
                predicted: agg.predicted, truth: agg.truth },
    };
}

// ----------------------------------------------------------------------------
// Prediction
// ----------------------------------------------------------------------------

/**
 * Why a model cannot run on a context (empty = it can): a feature its forest
 * reads that this session does not have (another skeleton, 2D vs 3D, other
 * excluded nodes), or a social model with one animal.
 */
export function modelProblems(model, ctx) {
    var out = [];
    if (!model || !ctx) return ['no model'];
    var miss = missingFeatures(ctx.catalog, model.features, model.social);
    if (miss.length) out.push(miss.length.toLocaleString('en-US') + ' of its features are not measured here (' + miss[0] + (miss.length > 1 ? ', …' : '') + ')');
    if (model.social && ctx.input.animals.length < 2) out.push('social, but only one animal');
    return out;
}

/**
 * Score every frame of a context with some models.
 * @param {Object} ctx
 * @param {Object[]} models
 * @param {{onProgress?: function(number, number), pause?: function(): Promise, isCancelled?: function(): boolean}} [opts]
 * @returns {Promise<{scores: Object<string, Map<string, Float32Array>>, skipped: Array<{behaviorId, problems}>, cancelled: boolean}>}
 *   scores by behavior id, then by 'animal\u0001target' names (a mutual pair in name order)
 */
export async function predictModels(ctx, models, opts) {
    opts = opts || {};
    var input = ctx.input, T = input.numFrames, N = input.nodeNames.length, D = input.dim;
    var skipped = [], plan = [], forests = {};
    (models || []).forEach(function (m) {
        var probs = modelProblems(m, ctx);
        if (probs.length) { skipped.push({ behaviorId: m.behaviorId, problems: probs }); return; }
        plan.push({ behaviorId: m.behaviorId, social: m.social, mutual: m.mutual, refs: featureRefs(ctx.catalog, m.features, m.social) });
        forests[m.behaviorId] = m.forest;
    });
    var scores = {};
    plan.forEach(function (p) { scores[p.behaviorId] = new Map(); });
    if (!plan.length || !T) return { scores: scores, skipped: skipped, cancelled: false };
    var chunks = planChunks(T, featurePadFrames(ctx.spec), PREDICT_CHUNK_FRAMES);
    var key = 'forest:' + plan.map(function (p) { return p.behaviorId + '=' + modelHashOf(p.behaviorId); }).join(',') + ':' + (++_predictRuns);
    var done = 0;
    var res = await runBehaviorChunks(chunks.length, function (i) {
        return { kind: 'forest', spec: ctx.spec, chunk: chunks[i], data: sliceChunkData(input.data, chunks[i], N * D, ctx.broken ? ctx.broken.masks : null),
                 plan: { models: plan } };
    }, function (i, out) {
        var ch = chunks[i];
        if (!out) return;
        Object.keys(out.scores).forEach(function (bid) {
            var byKey = out.scores[bid];
            Object.keys(byKey).forEach(function (k) {
                var p = k.split('>');
                var a = input.animals[+p[0]] || '', t = p.length > 1 ? (input.animals[+p[1]] || '') : '';
                var pm = plan.find(function (x) { return x.behaviorId === bid; });
                if (pm && pm.mutual && t && t < a) { var tmp = a; a = t; t = tmp; }
                var name = a + SEP + t;
                var whole = scores[bid].get(name);
                if (!whole) { whole = new Float32Array(T).fill(NaN); scores[bid].set(name, whole); }
                whole.set(byKey[k], ch.start);
            });
        });
        done += ch.end - ch.start;
        if (opts.onProgress) opts.onProgress(done, T);
    }, { workload: T * Math.max(1, input.animals.length) * Math.max(1, plan.length) * 8, shared: { key: key, value: forests },
         releaseShared: true, pause: opts.pause, isCancelled: opts.isCancelled });
    return { scores: scores, skipped: skipped, cancelled: !!res.cancelled };
}

var _predictRuns = 0;

/**
 * Score one frame with some models (This frame): inline, on a chunk padded
 * around it.
 * @returns {{rows: Array<{behaviorId, animal, target, score}>, skipped: Array}}
 */
export function predictModelsAtFrame(ctx, models, frame) {
    var input = ctx.input;
    var plan = [], forests = {}, skipped = [];
    (models || []).forEach(function (m) {
        var probs = modelProblems(m, ctx);
        if (probs.length) { skipped.push({ behaviorId: m.behaviorId, problems: probs }); return; }
        plan.push({ behaviorId: m.behaviorId, social: m.social, mutual: m.mutual, refs: featureRefs(ctx.catalog, m.features, m.social) });
        forests[m.behaviorId] = m.forest;
    });
    var rows = [];
    if (!plan.length || frame < 0 || frame >= input.numFrames) return { rows: rows, skipped: skipped };
    var pad = featurePadFrames(ctx.spec);
    var chunk = { start: frame, end: frame + 1, padStart: Math.max(0, frame - pad), padEnd: Math.min(input.numFrames, frame + 1 + pad) };
    var out = runJob({ kind: 'forest', spec: ctx.spec, chunk: chunk,
        data: sliceChunkData(input.data, chunk, input.nodeNames.length * input.dim, ctx.broken ? ctx.broken.masks : null), plan: { models: plan } }, forests);
    plan.forEach(function (p) {
        Object.keys(out.scores[p.behaviorId] || {}).forEach(function (k) {
            var q = k.split('>');
            rows.push({ behaviorId: p.behaviorId, animal: input.animals[+q[0]] || '', target: q.length > 1 ? (input.animals[+q[1]] || '') : '',
                        score: out.scores[p.behaviorId][k][0] });
        });
    });
    return { rows: rows, skipped: skipped };
}

/**
 * Bouts from per-frame scores, for several behaviors at once: each behavior's
 * thresholds and durations, and within an exclusive group, on each frame,
 * only the member scoring highest above its own threshold (per animal; per
 * pair for social behaviors, among members alike in that).
 * @param {Object<string, Map<string, Float32Array>>} scoreMaps by behavior id, then 'animal\u0001target'
 * @param {Object<string, {threshold:number, minFrames:number, maxGapFrames:number, point:boolean, group:string, social:boolean}>} how
 * @returns {Object<string, Array<{animal, target, start, end, score}>>}
 */
export function deriveBouts(scoreMaps, how) {
    var ids = Object.keys(scoreMaps);
    var adjusted = {};
    ids.forEach(function (id) { adjusted[id] = new Map(scoreMaps[id]); });
    // Exclusive groups.
    var groups = new Map();
    ids.forEach(function (id) {
        var h = how[id];
        if (!h || !h.group) return;
        var gk = h.group + SEP + (h.social ? 's' : 'a');
        if (!groups.has(gk)) groups.set(gk, []);
        groups.get(gk).push(id);
    });
    groups.forEach(function (members) {
        if (members.length < 2) return;
        var keys = new Set();
        members.forEach(function (id) { scoreMaps[id].forEach(function (v, k) { keys.add(k); }); });
        keys.forEach(function (k) {
            var present = members.filter(function (id) { return scoreMaps[id].has(k); });
            if (present.length < 2) return;
            var out = exclusiveScores(present.map(function (id) { return { scores: scoreMaps[id].get(k), threshold: how[id].threshold }; }));
            present.forEach(function (id, i) { adjusted[id].set(k, out[i]); });
        });
    });
    var res = {};
    ids.forEach(function (id) {
        var h = how[id] || { threshold: 0.5, minFrames: 1, maxGapFrames: 0, point: false };
        var bouts = [];
        adjusted[id].forEach(function (arr, k) {
            var p = k.split(SEP);
            scoresToBouts(arr, { threshold: h.threshold, minFrames: h.minFrames, maxGapFrames: h.maxGapFrames, point: h.point }).forEach(function (bt) {
                bouts.push({ animal: p[0], target: p[1] || '', start: bt[0], end: bt[1], score: bt[2] });
            });
        });
        res[id] = bouts;
    });
    return res;
}

/** How a behavior's model scores become bouts at `fps` (for `deriveBouts`). */
export function modelBoutSettings(b, fps) {
    var p = behaviorPost(b);
    return { threshold: p.threshold, minFrames: Math.max(1, secondsToFrames(p.minBout, fps)), maxGapFrames: secondsToFrames(p.maxGap, fps),
             point: b.kind === 'point', group: b.group || '', social: !!b.social };
}

// ----------------------------------------------------------------------------
// Active learning
// ----------------------------------------------------------------------------

/**
 * Unlabelled stretches where a behavior's latest scores sit near its
 * threshold — what to label next to teach the classifier most. Frames within
 * `band` of the threshold, joined across gaps of up to `gapFrames`; the
 * stretches with the most uncertainty in all (the sum, over their frames, of
 * how far inside the band each is) first, so a long doubtful stretch comes
 * before a one-frame blip at a bout's edge — but a confident classifier, with
 * nothing else to ask, still offers its blips. Needs the run's per-frame
 * scores (in memory after a prediction in this session).
 * @param {{band?: number, gapFrames?: number, minFrames?: number}} [opts]
 * @returns {Array<{behaviorId, animal, target, start, end, score, uncertainty, kind: 'uncertain'}>}
 *   score: the stretch's mean; uncertainty: its mean distance to the threshold
 */
export function uncertainSegments(session, behaviorId, opts) {
    opts = opts || {};
    var run = behaviorRunOf(session, behaviorId);
    if (!run || !run.scores) return [];
    var thr = runThreshold(run);
    var band = opts.band != null ? opts.band : 0.15, gap = opts.gapFrames != null ? opts.gapFrames : 3, minLen = Math.max(1, opts.minFrames || 1);
    var out = [];
    run.scores.forEach(function (arr, k) {
        var p = k.split(SEP);
        var labelled = boutIntervals(session, behaviorId, p[0], p[1] || '');
        var li = 0;
        var cur = null;
        var flush = function () {
            if (cur && cur.n >= minLen) {
                out.push({ behaviorId: behaviorId, animal: p[0], target: p[1] || '', start: cur.start, end: cur.end,
                           score: cur.sum / cur.n, uncertainty: cur.dist / cur.n, mass: cur.n * band - cur.dist, kind: 'uncertain' });
            }
            cur = null;
        };
        for (var t = 0; t < arr.length; t++) {
            while (li < labelled.length && labelled[li].end < t) li++;
            var isLab = li < labelled.length && labelled[li].start <= t;
            var v = arr[t];
            var near = !isLab && v === v && Math.abs(v - thr) <= band;
            if (near) {
                if (cur && t - cur.end - 1 > gap) flush();
                if (!cur) cur = { start: t, end: t, n: 0, sum: 0, dist: 0 };
                cur.end = t; cur.n++; cur.sum += v; cur.dist += Math.abs(v - thr);
            } else if (cur && (isLab || t - cur.end > gap)) flush();
        }
        flush();
    });
    out.sort(function (a, b) { return b.mass - a.mass || a.uncertainty - b.uncertainty || a.start - b.start || (a.animal < b.animal ? -1 : 1); });
    out.forEach(function (o) { delete o.mass; });
    return out;
}

/**
 * Where a behavior's latest predicted bouts and its labels disagree: frames
 * labelled present outside every predicted bout ('missed'), and frames
 * labelled "not" inside one ('false'), as runs of at least `minFrames`; the
 * longest first.
 * @returns {Array<{behaviorId, animal, target, start, end, score, kind: 'missed'|'false'}>}
 */
export function disagreementSegments(session, behaviorId, opts) {
    var run = behaviorRunOf(session, behaviorId);
    if (!run) return [];
    var minLen = Math.max(1, (opts && opts.minFrames) || 1);
    var byKey = new Map();
    run.bouts.forEach(function (bt) {
        var k = bt.animal + SEP + bt.target;
        if (!byKey.has(k)) byKey.set(k, []);
        byKey.get(k).push([bt.start, bt.end, bt.score]);
    });
    var keys = new Set(byKey.keys());
    listBouts(session, { behaviorId: behaviorId }).forEach(function (bt) { keys.add(bt.animal + SEP + bt.target); });
    var out = [];
    keys.forEach(function (k) {
        var p = k.split(SEP);
        var pred = (byKey.get(k) || []).sort(function (x, y) { return x[0] - y[0]; });
        var predIvs = pred.map(function (x) { return [x[0], x[1]]; });
        var scoreArr = run.scores ? run.scores.get(k) : null;
        boutIntervals(session, behaviorId, p[0], p[1] || '').forEach(function (iv) {
            var pieces = iv.present ? subtract([[iv.start, iv.end]], predIvs)
                : predIvs.map(function (x) { return [Math.max(x[0], iv.start), Math.min(x[1], iv.end)]; }).filter(function (x) { return x[1] >= x[0]; });
            pieces.forEach(function (pc) {
                if (pc[1] - pc[0] + 1 < minLen) return;
                var s = NaN;
                if (scoreArr) {
                    var sum = 0, n = 0;
                    for (var f = pc[0]; f <= pc[1]; f++) { var v = scoreArr[f]; if (v === v) { sum += v; n++; } }
                    s = n ? sum / n : NaN;
                }
                out.push({ behaviorId: behaviorId, animal: p[0], target: p[1] || '', start: pc[0], end: pc[1], score: s, kind: iv.present ? 'missed' : 'false' });
            });
        });
    });
    out.sort(function (a, b) { return (b.end - b.start) - (a.end - a.start) || a.start - b.start; });
    return out;
}

/** The share of a context's frames for animal `a` that have a pose (for tests and reports). */
export function presentShare(ctx, a) {
    if (!ctx || a < 0) return 0;
    var n = 0;
    for (var t = 0; t < ctx.input.numFrames; t++) if (presentAt(ctx, a, t)) n++;
    return ctx.input.numFrames ? n / ctx.input.numFrames : 0;
}

export { presentFrames };
