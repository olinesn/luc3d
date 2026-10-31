// pose/behavior-states.js — behavioral STATES, per session: one categorical
// label per animal and frame from an unsupervised model — keypoint-MoSeq's
// syllables (import-export/kpms-io.js), the in-browser discovery's clusters
// (pose/behavior-discovery.js) — kept as named LAYERS of runs.
//
// ## Storage
//
// Per session (a WeakMap, like labels and predictions: never on FrameGroups,
// so nothing to hydrate or evict on a lazy project), a list of layers:
//
//   { name,                          unique within the session
//     source: 'keypoint-MoSeq' | 'discovery',
//     stateNames: Map<state, name>,  only the states someone named
//     animals: Map<animal NAME, Int32Array [start, end, state, start, end, state, …]>,
//     version }                      bumped by every change (caches key on it)
//
// A run covers frames start..end, inclusive. Runs are sorted, never overlap,
// and touching runs of one state are one run; frames no run covers have no
// state. Runs are the only form — 3 ints each, about 0.7 MB for an hour of
// 5 animals at 50 fps with 0.4 s syllables — never a per-frame array.
//
// States are non-negative integers (keypoint-MoSeq's syllable numbers). A
// layer's runs are never edited in place: every change builds new arrays.
//
// Animals are identity NAMES, as labels are: Rename Identity calls
// `renameStateAnimal`, and every `Session.swapIdentitiesInRange` exchanges the
// two animals' runs on its frames (`onIdentitySwap`, as the labels do).
//
// ## Saved
//
// `metadata.lucid.behaviorStates` (import-export/behavior-metadata.js), only
// when some layer has a run:
//   { v: 1, layers: [{ name, source, stateNames: {state: name},
//                      animals: {name: [[start, end, state], …]} }] }
//
// ## Summaries
//
// `stateSummary` (usage, runs, median duration per state) and
// `stateEnrichment` (per state × behavior: labelled frames present and "not")
// read runs and label intervals only — never one frame at a time. Unlabelled
// frames count for neither: P(behavior | state) = present / (present + not).
//
// DOM-free: Node tests import it directly.
//
// Depends on: pose/behavior-labels.js, pose/pose-data.js (the identity-swap hook)

import { listBouts } from './behavior-labels.js';
import { onIdentitySwap } from './pose-data.js';

/** Where a layer's states came from. */
export var STATE_SOURCES = ['keypoint-MoSeq', 'discovery'];

/**
 * State colors on the timeline: state k (keypoint-MoSeq numbers syllables by
 * how often they occur, 0 the most, and discovery its clusters by size) takes entry k; later states share
 * `OTHER_STATE_COLOR`. Chosen one at a time, each as far (CIEDE2000) from the
 * ones before it and from the gray as it could be, light enough for the dark
 * timeline: tests/test-track-identity-palette.mjs §11.
 */
export var STATE_COLORS = [
    '#4fa3ff', '#ffff00', '#00ffdd', '#ff9911', '#ff4455', '#33aa00',
    '#ffccff', '#ff00dd', '#aa9900', '#33ddff', '#bbdd99', '#55aa99',
    '#ffaa99', '#9977ff', '#ffdd99', '#00ff00', '#bb7799', '#ff5500',
    '#bb8866', '#0099bb', '#bbbbff', '#aacc00', '#44cc88', '#779966',
];
/** The color of every state past the palette. */
export var OTHER_STATE_COLOR = '#8a8f96';

var LAYER_NAME_MAX = 60;
var STATE_NAME_MAX = 60;
var MAX_STATE = 1 << 20;

var _stores = new WeakMap();
var _listeners = [];

function storeOf(session, create) {
    if (!session) return null;
    var st = _stores.get(session);
    if (!st && create) {
        st = { layers: [] };
        _stores.set(session, st);
    }
    return st || null;
}

/**
 * Be told after every change to a session's states (`fn(session, info)`).
 * Loading a file does not notify.
 */
export function onStatesChange(fn) {
    if (typeof fn === 'function') _listeners.push(fn);
}

function emit(session, info) {
    for (var i = 0; i < _listeners.length; i++) {
        try { _listeners[i](session, info || {}); } catch (e) { console.error('[behavior] states listener failed', e); }
    }
}

/** The timeline color of a state. */
export function stateColor(state) {
    return state >= 0 && state < STATE_COLORS.length ? STATE_COLORS[state] : OTHER_STATE_COLOR;
}

/** A layer name as stored: trimmed, one line, at most 60 characters ('' when nothing is left). */
export function cleanLayerName(name) {
    return String(name == null ? '' : name).replace(/\s+/g, ' ').trim().slice(0, LAYER_NAME_MAX);
}

function cleanStateName(name) {
    return String(name == null ? '' : name).replace(/\s+/g, ' ').trim().slice(0, STATE_NAME_MAX);
}

function isState(k) {
    return Number.isInteger(k) && k >= 0 && k < MAX_STATE;
}

// ----------------------------------------------------------------------------
// Runs (Int32Array triplets). All return new arrays.
// ----------------------------------------------------------------------------

/**
 * Runs from a per-frame sequence (keypoint-MoSeq's `syllable` column): equal
 * consecutive values are one run; anything that is not a non-negative
 * integer is no state. BigInt64Array and plain arrays are read too.
 * @param {ArrayLike<number|bigint>} seq
 * @param {{offset?: number, maxFrame?: number}} [opts] frame of seq[0] (default 0);
 *   frames at or past maxFrame are dropped
 * @returns {Int32Array}
 */
export function runsFromSequence(seq, opts) {
    var offset = opts && Number.isInteger(opts.offset) ? opts.offset : 0;
    var maxFrame = opts && opts.maxFrame > 0 ? opts.maxFrame : Infinity;
    var n = seq ? seq.length : 0;
    var out = [];
    var cur = -1, start = 0;
    var close = function (endIdx) {
        if (cur < 0) return;
        var s = start + offset, e = endIdx + offset;
        if (s < 0) s = 0;
        if (e >= maxFrame) e = maxFrame - 1;
        if (e >= s) out.push(s, e, cur);
    };
    for (var i = 0; i < n; i++) {
        var v = Number(seq[i]);
        var k = isState(v) ? v : -1;
        if (k === cur) continue;
        close(i - 1);
        cur = k; start = i;
    }
    close(n - 1);
    return Int32Array.from(out);
}

/**
 * Runs as stored: from triplets or `[[start, end, state], …]`, malformed ones
 * dropped, sorted, an overlapping run cut to the frames the earlier one left,
 * touching runs of one state joined.
 * @returns {{runs: Int32Array, dropped: number}}
 */
function normalizeRuns(src) {
    var list = [], dropped = 0;
    if (src instanceof Int32Array || (src && ArrayBuffer.isView(src))) {
        for (var i = 0; i + 2 < src.length; i += 3) list.push([Number(src[i]), Number(src[i + 1]), Number(src[i + 2])]);
        if (src.length % 3) dropped++;
    } else if (Array.isArray(src)) {
        src.forEach(function (r) {
            if (Array.isArray(r) && r.length >= 3) list.push([Number(r[0]), Number(r[1]), Number(r[2])]);
            else dropped++;
        });
    }
    list = list.filter(function (r) {
        var ok = Number.isInteger(r[0]) && Number.isInteger(r[1]) && r[0] >= 0 && r[1] >= r[0] && isState(r[2]);
        if (!ok) dropped++;
        return ok;
    });
    list.sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; });
    var out = [];
    var lastEnd = -1;
    list.forEach(function (r) {
        var s = Math.max(r[0], lastEnd + 1), e = r[1];
        if (e < s) { dropped++; return; }
        var m = out.length;
        if (m && out[m - 1] === r[2] && out[m - 2] === s - 1) out[m - 2] = e;
        else out.push(s, e, r[2]);
        lastEnd = e;
    });
    return { runs: Int32Array.from(out), dropped: dropped };
}

/** Join touching runs of one state (runs already sorted, not overlapping). */
function joinRuns(arr) {
    var out = [];
    for (var i = 0; i + 2 < arr.length; i += 3) {
        var m = out.length;
        if (m && out[m - 1] === arr[i + 2] && out[m - 2] === arr[i] - 1) out[m - 2] = arr[i + 1];
        else out.push(arr[i], arr[i + 1], arr[i + 2]);
    }
    return Int32Array.from(out);
}

/** A run list split at [s, e]: the parts inside and outside, each sorted. */
function cutRuns(runs, s, e) {
    var inside = [], outside = [];
    for (var i = 0; i + 2 < runs.length; i += 3) {
        var a = runs[i], b = runs[i + 1], k = runs[i + 2];
        if (b < s || a > e) { outside.push(a, b, k); continue; }
        if (a < s) outside.push(a, s - 1, k);
        inside.push(Math.max(a, s), Math.min(b, e), k);
        if (b > e) outside.push(e + 1, b, k);
    }
    return { inside: inside, outside: outside };
}

/** Two disjoint run lists as one (sorted, joined). */
function mergeDisjoint(a, b) {
    var trip = [];
    var push = function (arr) { for (var i = 0; i + 2 < arr.length; i += 3) trip.push([arr[i], arr[i + 1], arr[i + 2]]); };
    push(a); push(b);
    trip.sort(function (x, y) { return x[0] - y[0]; });
    var flat = [];
    trip.forEach(function (r) { flat.push(r[0], r[1], r[2]); });
    return joinRuns(flat);
}

/** `base` with `top` laid over it: on frames `top` covers, `top` wins. One sweep. */
function overlayRuns(base, top) {
    var rest = [], j = 0, nTop = top.length / 3;
    for (var i = 0; i + 2 < base.length; i += 3) {
        var s = base[i], e = base[i + 1], k = base[i + 2];
        while (j < nTop && top[3 * j + 1] < s) j++;
        for (var t = j; s <= e; t++) {
            if (t >= nTop || top[3 * t] > e) { rest.push(s, e, k); break; }
            if (top[3 * t] > s) rest.push(s, top[3 * t] - 1, k);
            s = top[3 * t + 1] + 1;
        }
    }
    return mergeDisjoint(rest, top);
}

// ----------------------------------------------------------------------------
// Layers
// ----------------------------------------------------------------------------

function toMap(obj) {
    if (obj instanceof Map) return new Map(obj);
    var m = new Map();
    if (obj && typeof obj === 'object') Object.keys(obj).forEach(function (k) { m.set(k, obj[k]); });
    return m;
}

function stateNamesFrom(src) {
    var out = new Map();
    toMap(src).forEach(function (v, k) {
        var s = Number(k), nm = cleanStateName(v);
        if (isState(s) && nm) out.set(s, nm);
    });
    return out;
}

function freshLayer(name, source, stateNames, animals) {
    return { name: name, source: source, stateNames: stateNames, animals: animals, version: 1, _print: null };
}

/**
 * Add a layer to a session, replacing its layer of the same name.
 * @param {Session} session
 * @param {{name: string, source?: string, stateNames?: Object|Map, animals: Object|Map}} layer
 *   animals: name -> runs (an Int32Array of triplets, or [[start, end, state], …])
 * @param {{silent?: boolean}} [opts] silent: do not notify (a load)
 * @returns {{layer: Object|null, runs: number, dropped: number}}
 */
export function setStateLayer(session, layer, opts) {
    var name = cleanLayerName(layer && layer.name);
    if (!session || !name) return { layer: null, runs: 0, dropped: 0 };
    var source = STATE_SOURCES.indexOf(layer.source) >= 0 ? layer.source : STATE_SOURCES[0];
    var animals = new Map(), runs = 0, dropped = 0;
    toMap(layer.animals).forEach(function (r, animal) {
        if (typeof animal !== 'string' || !animal) { dropped++; return; }
        var nr = normalizeRuns(r);
        dropped += nr.dropped;
        if (!nr.runs.length) return;
        animals.set(animal, nr.runs);
        runs += nr.runs.length / 3;
    });
    var st = storeOf(session, true);
    var made = freshLayer(name, source, stateNamesFrom(layer.stateNames), animals);
    var at = -1;
    for (var i = 0; i < st.layers.length; i++) if (st.layers[i].name === name) at = i;
    if (at >= 0) { made.version = st.layers[at].version + 1; st.layers[at] = made; } else st.layers.push(made);
    if (!(opts && opts.silent)) emit(session, { layer: name });
    return { layer: made, runs: runs, dropped: dropped };
}

/** Remove a layer; true when there was one. */
export function removeStateLayer(session, name) {
    var st = storeOf(session, false);
    if (!st) return false;
    var before = st.layers.length;
    st.layers = st.layers.filter(function (l) { return l.name !== name; });
    if (st.layers.length === before) return false;
    emit(session, { layer: name, removed: true });
    return true;
}

/** The session's layers, in the order they were added (read-only). */
export function stateLayers(session) {
    var st = storeOf(session, false);
    return st ? st.layers.slice() : [];
}

/** One layer by name, or null. */
export function stateLayer(session, name) {
    var st = storeOf(session, false);
    if (!st) return null;
    for (var i = 0; i < st.layers.length; i++) if (st.layers[i].name === name) return st.layers[i];
    return null;
}

/** True when the session has a layer with any run. */
export function hasStates(session) {
    return stateLayers(session).some(function (l) { return l.animals.size > 0; });
}

/**
 * Name a state of a layer (an empty name clears it). States keep their
 * numbers; only the label changes.
 * @returns {boolean} whether anything changed
 */
export function setStateName(session, layerName, state, name) {
    var layer = stateLayer(session, layerName);
    var s = Number(state);
    if (!layer || !isState(s)) return false;
    var nm = cleanStateName(name);
    if ((layer.stateNames.get(s) || '') === nm) return false;
    var names = new Map(layer.stateNames);
    if (nm) names.set(s, nm); else names.delete(s);
    layer.stateNames = names;
    layer.version++;
    emit(session, { layer: layer.name, names: true });
    return true;
}

/** A state's name in a layer, or ''. */
export function stateName(layer, state) {
    return (layer && layer.stateNames.get(state)) || '';
}

/** How a state reads: "12" or "12 · groom". */
export function stateLabel(layer, state) {
    var nm = stateName(layer, state);
    return nm ? state + ' · ' + nm : String(state);
}

/** The animals a layer has runs for, in name order. */
export function layerAnimals(layer) {
    return layer ? Array.from(layer.animals.keys()).sort() : [];
}

/** An animal's state on a frame, or -1. Binary search: never walks the runs. */
export function stateAt(layer, animal, frame) {
    var runs = layer && layer.animals.get(animal);
    if (!runs || !runs.length) return -1;
    var lo = 0, hi = runs.length / 3 - 1;
    while (lo <= hi) {
        var mid = (lo + hi) >> 1, o = mid * 3;
        if (frame < runs[o]) hi = mid - 1;
        else if (frame > runs[o + 1]) lo = mid + 1;
        else return runs[o + 2];
    }
    return -1;
}

/**
 * A print of a layer's runs, so a prediction made from it can tell the layer
 * was replaced or changed. Neither the states' names nor the animals' names
 * are in it (a sum over animals): naming a state or renaming an identity
 * leaves predictions as they were. Cached per version.
 */
export function layerPrint(layer) {
    if (!layer) return '';
    if (layer._print && layer._print.version === layer.version) return layer._print.value;
    var sum = 0;
    layer.animals.forEach(function (runs) {
        var h = 0x811c9dc5;
        for (var i = 0; i < runs.length; i++) h = Math.imul(h ^ runs[i], 0x01000193);
        sum = (sum + (h >>> 0)) >>> 0;
    });
    var value = sum.toString(16) + '.' + layer.animals.size;
    layer._print = { version: layer.version, value: value };
    return value;
}

// ----------------------------------------------------------------------------
// Summaries
// ----------------------------------------------------------------------------

function median(sorted) {
    var n = sorted.length;
    if (!n) return NaN;
    return n % 2 ? sorted[n >> 1] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
}

/**
 * Usage per state over one or more layers (one layer name across sessions):
 * frames, runs, and the median run length in frames.
 * @param {Object[]} layers
 * @returns {{frames: number, states: Array<{state, frames, runs, medianFrames, share}>}}
 *   sorted by state; share = frames / all frames with a state
 */
export function stateSummary(layers) {
    var acc = new Map(), total = 0;
    (layers || []).forEach(function (layer) {
        if (!layer) return;
        layer.animals.forEach(function (runs) {
            for (var i = 0; i + 2 < runs.length; i += 3) {
                var k = runs[i + 2], len = runs[i + 1] - runs[i] + 1;
                var a = acc.get(k);
                if (!a) { a = { state: k, frames: 0, runs: 0, lengths: [] }; acc.set(k, a); }
                a.frames += len; a.runs++; a.lengths.push(len);
                total += len;
            }
        });
    });
    var states = Array.from(acc.values()).sort(function (a, b) { return a.state - b.state; }).map(function (a) {
        var lengths = Int32Array.from(a.lengths).sort();
        return { state: a.state, frames: a.frames, runs: a.runs, medianFrames: median(lengths), share: total ? a.frames / total : 0 };
    });
    return { frames: total, states: states };
}

/** Sorted, merged [start, end] intervals. */
function mergeIntervals(ivs) {
    ivs.sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; });
    var out = [];
    ivs.forEach(function (iv) {
        var last = out[out.length - 1];
        if (last && iv[0] <= last[1] + 1) last[1] = Math.max(last[1], iv[1]);
        else out.push([iv[0], iv[1]]);
    });
    return out;
}

/** Merged intervals `a` minus merged intervals `b`. */
function subtractIntervals(a, b) {
    var out = [], j = 0;
    a.forEach(function (iv) {
        var s = iv[0], e = iv[1];
        while (j < b.length && b[j][1] < s) j++;
        var k = j;
        while (k < b.length && b[k][0] <= e) {
            if (b[k][0] > s) out.push([s, b[k][0] - 1]);
            s = Math.max(s, b[k][1] + 1);
            if (s > e) break;
            k++;
        }
        if (s <= e) out.push([s, e]);
    });
    return out;
}

/** Add, per state, the frames of `runs` that merged intervals `ivs` cover. */
function overlapByState(runs, ivs, into) {
    var i = 0, j = 0, n = runs.length / 3;
    while (i < n && j < ivs.length) {
        var o = i * 3, lo = Math.max(runs[o], ivs[j][0]), hi = Math.min(runs[o + 1], ivs[j][1]);
        if (hi >= lo) into.set(runs[o + 2], (into.get(runs[o + 2]) || 0) + hi - lo + 1);
        if (runs[o + 1] < ivs[j][1]) i++; else j++;
    }
}

/**
 * How the labels fall in each state: per state and behavior, the frames
 * labelled present and the frames labelled "not" (and not present). A frame
 * counts once per animal and behavior, whatever the target, as the Summary's
 * time budget counts it; an animal's labels are its bouts as actor, or as
 * either animal of a mutual pair. Unlabelled frames count for neither.
 *
 * @param {Array<{session: Session, layer: Object}>} items one layer name across sessions
 * @param {string[]} [behaviorIds] default: every behavior with labels
 * @returns {Map<number, Map<string, {present: number, absent: number}>>}
 */
export function stateEnrichment(items, behaviorIds) {
    var want = behaviorIds ? new Set(behaviorIds) : null;
    var out = new Map();
    var add = function (behaviorId, counts, field) {
        counts.forEach(function (frames, state) {
            var row = out.get(state);
            if (!row) { row = new Map(); out.set(state, row); }
            var c = row.get(behaviorId);
            if (!c) { c = { present: 0, absent: 0 }; row.set(behaviorId, c); }
            c[field] += frames;
        });
    };
    (items || []).forEach(function (it) {
        if (!it || !it.session || !it.layer) return;
        it.layer.animals.forEach(function (runs, animal) {
            var byBehavior = new Map();
            listBouts(it.session, { animal: animal }).forEach(function (b) {
                if (want && !want.has(b.behaviorId)) return;
                var e = byBehavior.get(b.behaviorId);
                if (!e) { e = { pres: [], abs: [] }; byBehavior.set(b.behaviorId, e); }
                (b.present ? e.pres : e.abs).push([b.start, b.end]);
            });
            byBehavior.forEach(function (e, behaviorId) {
                var pres = mergeIntervals(e.pres);
                var abs = subtractIntervals(mergeIntervals(e.abs), pres);
                var pc = new Map(), ac = new Map();
                overlapByState(runs, pres, pc);
                overlapByState(runs, abs, ac);
                add(behaviorId, pc, 'present');
                add(behaviorId, ac, 'absent');
            });
        });
    });
    return out;
}

/**
 * The bouts where an animal is in any of `states`: maximal stretches of
 * touching runs with those states (a frame with no state, or another state,
 * ends one), each with the state it spends the most frames in.
 * @param {Object} layer
 * @param {number[]} states
 * @returns {Array<{animal: string, target: string, start: number, end: number, state: number}>}
 */
export function statesToBouts(layer, states) {
    var want = new Set((states || []).map(Number));
    var out = [];
    if (!layer || !want.size) return out;
    layerAnimals(layer).forEach(function (animal) {
        var runs = layer.animals.get(animal);
        var cur = null, frames = null;
        var finish = function () {
            if (!cur) return;
            var best = -1, most = -1;
            frames.forEach(function (f, k) { if (f > most || (f === most && k < best)) { most = f; best = k; } });
            cur.state = best;
            out.push(cur);
            cur = null;
        };
        for (var i = 0; i + 2 < runs.length; i += 3) {
            var s = runs[i], e = runs[i + 1], k = runs[i + 2];
            if (!want.has(k)) { finish(); continue; }
            if (cur && s !== cur.end + 1) finish();
            if (!cur) { cur = { animal: animal, target: '', start: s, end: e, state: k }; frames = new Map(); }
            cur.end = e;
            frames.set(k, (frames.get(k) || 0) + e - s + 1);
        }
        finish();
    });
    out.sort(function (a, b) { return a.start - b.start || a.end - b.end || (a.animal < b.animal ? -1 : a.animal > b.animal ? 1 : 0); });
    return out;
}

// ----------------------------------------------------------------------------
// Identities
// ----------------------------------------------------------------------------

function replaceAnimals(session, layer, animals) {
    var st = storeOf(session, false);
    var i = st ? st.layers.indexOf(layer) : -1;
    if (i < 0) return;
    var made = freshLayer(layer.name, layer.source, layer.stateNames, animals);
    made.version = layer.version + 1;
    st.layers[i] = made;
}

/**
 * An identity was renamed: its runs follow, in every layer. Runs an orphan of
 * the new name left are overwritten where the renamed animal has a state.
 * @returns {number} layers changed
 */
export function renameStateAnimal(session, oldName, newName) {
    if (!oldName || !newName || oldName === newName) return 0;
    var changed = 0;
    stateLayers(session).forEach(function (layer) {
        var runs = layer.animals.get(oldName);
        if (!runs) return;
        var animals = new Map();
        layer.animals.forEach(function (r, a) { if (a !== oldName) animals.set(a, r); });
        animals.set(newName, animals.has(newName) ? overlayRuns(animals.get(newName), runs) : runs);
        replaceAnimals(session, layer, animals);
        changed++;
    });
    if (changed) emit(session, { identity: 'rename' });
    return changed;
}

/**
 * Two identities were exchanged on frames [start, end] (an ID-switch fix, or
 * a reassignment that swaps to the end): their states on those frames are
 * exchanged too, runs crossing an edge split there. Its own inverse.
 * @param {number} end inclusive; Infinity = to the end
 * @returns {number} layers changed
 */
export function swapStateAnimalsInRange(session, nameA, nameB, start, end) {
    if (!nameA || !nameB || nameA === nameB) return 0;
    var s = Math.max(0, Math.floor(start)), e = end;
    var changed = 0;
    stateLayers(session).forEach(function (layer) {
        var ra = layer.animals.get(nameA), rb = layer.animals.get(nameB);
        if (!ra && !rb) return;
        var ca = cutRuns(ra || new Int32Array(0), s, e), cb = cutRuns(rb || new Int32Array(0), s, e);
        if (!ca.inside.length && !cb.inside.length) return;
        var animals = new Map(layer.animals);
        var na = mergeDisjoint(ca.outside, cb.inside), nb = mergeDisjoint(cb.outside, ca.inside);
        if (na.length) animals.set(nameA, na); else animals.delete(nameA);
        if (nb.length) animals.set(nameB, nb); else animals.delete(nameB);
        replaceAnimals(session, layer, animals);
        changed++;
    });
    if (changed) emit(session, { identity: 'swap' });
    return changed;
}

// Every `Session.swapIdentitiesInRange` carries the states with the poses.
onIdentitySwap(swapStateAnimalsInRange);

// ----------------------------------------------------------------------------
// Save / load
// ----------------------------------------------------------------------------

/**
 * The `metadata.lucid.behaviorStates` payload, or null when no layer has a
 * run (nothing is written).
 */
export function exportStateLayers(session) {
    var layers = [];
    stateLayers(session).forEach(function (layer) {
        if (!layer.animals.size) return;
        var names = {};
        Array.from(layer.stateNames.keys()).sort(function (a, b) { return a - b; }).forEach(function (k) { names[k] = layer.stateNames.get(k); });
        var animals = {};
        layerAnimals(layer).forEach(function (a) {
            var runs = layer.animals.get(a), rows = [];
            for (var i = 0; i + 2 < runs.length; i += 3) rows.push([runs[i], runs[i + 1], runs[i + 2]]);
            animals[a] = rows;
        });
        layers.push({ name: layer.name, source: layer.source, stateNames: names, animals: animals });
    });
    return layers.length ? { v: 1, layers: layers } : null;
}

/**
 * Replace a session's layers with saved ones (a load). Tolerates absence and
 * garbage; does not notify.
 * @returns {{layers: number, runs: number, dropped: number}}
 */
export function importStateLayers(session, payload) {
    if (!session) return { layers: 0, runs: 0, dropped: 0 };
    _stores.delete(session);
    if (!payload || typeof payload !== 'object' || payload.v !== 1 || !Array.isArray(payload.layers)) return { layers: 0, runs: 0, dropped: 0 };
    var layers = 0, runs = 0, dropped = 0;
    payload.layers.forEach(function (l) {
        if (!l || typeof l !== 'object' || !cleanLayerName(l.name) || !l.animals || typeof l.animals !== 'object') { dropped++; return; }
        var r = setStateLayer(session, l, { silent: true });
        if (!r.layer) { dropped++; return; }
        layers++; runs += r.runs; dropped += r.dropped;
    });
    return { layers: layers, runs: runs, dropped: dropped };
}

/**
 * Forget every session's states (a new project or a load: every reader then
 * reads its session's layers back with `importStateLayers`).
 */
export function resetBehaviorStates() {
    _stores = new WeakMap();
}
