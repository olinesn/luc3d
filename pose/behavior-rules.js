// pose/behavior-rules.js — rule detectors: a behavior is detected on a frame
// when every one of its conditions on named features holds (an AND), and the
// detected frames become bouts with a minimum duration and a maximum gap.
//
// A detector lives on its ethogram behavior (`behavior.detector`, saved with
// the ethogram only when set):
//
//   { conditions: [{ feature: 'Nose height · mean ±0.1 s', op: '>', value: 0.5 }, …],
//     minDuration: 0.3,   // seconds of DETECTED frames a bout needs
//     maxGap: 0.2 }       // seconds; detected runs this close are joined first
//
// Features are named as pose/behavior-features.js names them. A social
// behavior's conditions may also use the relational features of the actor →
// target pair ("head–tail base distance") and the target's own features
// ("target · centroid speed"); unprefixed animal features are the actor's. A
// MUTUAL behavior is detected for a pair when its conditions hold in either
// direction.
//
// ## Scores
//
// Every frame gets a score in [0, 1], not just yes / no, so the review list can
// put the doubtful bouts first: each condition contributes σ(margin / scale),
// where the margin is how far the feature is past its threshold and the scale
// is 10% of the threshold (at least a small amount in the feature's unit), and
// the frame's score is the lowest of them. The decision itself stays exact: a
// frame where every condition holds scores at least 0.5, any other frame below
// 0.5, and a frame where a condition's feature is missing is unscored (NaN).
//
// Pure: the behavior worker loads it.
//
// Depends on: pose/behavior-features.js, pose/behavior-postprocess.js

import { resolveFeatureName, computeFeatureChunk, catalogFor, TARGET_PREFIX } from './behavior-features.js';
import { scoresToBouts, secondsToFrames } from './behavior-postprocess.js';

/** The comparison operators: at least / at most the value. */
export var RULE_OPS = ['>', '<'];

var MAX_CONDITIONS = 12;
var DURATION_MAX_S = 600;

/**
 * The smallest margin scale per feature unit: a condition's score goes from
 * 0.27 to 0.73 within ± one scale of its threshold.
 */
var UNIT_FLOOR = { 'BL': 0.01, 'BL/s': 0.02, 'deg': 1, 'deg/s': 2, 'count': 0.25, '': 0.01, 'BL²': 1e-4, '(BL/s)²': 1e-3 };

/** The margin scale of a condition: 10% of its threshold, at least the unit's floor. */
export function conditionScale(unit, value) {
    var floor = UNIT_FLOOR[unit] != null ? UNIT_FLOOR[unit] : 0.01;
    return Math.max(Math.abs(Number(value) || 0) * 0.1, floor);
}

/**
 * A clean copy of a detector, or null when it has no usable condition.
 * Conditions are kept in order; malformed ones are dropped.
 */
export function normalizeDetector(d) {
    if (!d || typeof d !== 'object' || !Array.isArray(d.conditions)) return null;
    var conditions = [];
    d.conditions.forEach(function (c) {
        if (!c || typeof c.feature !== 'string' || !c.feature.trim() || conditions.length >= MAX_CONDITIONS) return;
        var v = Number(c.value);
        if (!Number.isFinite(v)) return;
        conditions.push({ feature: c.feature.trim(), op: c.op === '<' ? '<' : '>', value: v });
    });
    if (!conditions.length) return null;
    var dur = function (x) {
        var n = Number(x);
        return Number.isFinite(n) && n > 0 ? Math.min(DURATION_MAX_S, n) : 0;
    };
    return { conditions: conditions, minDuration: dur(d.minDuration), maxGap: dur(d.maxGap) };
}

/**
 * What would stop a detector running on a session with this feature catalog.
 * @returns {string[]} one message per problem (empty = it can run)
 */
export function detectorProblems(detector, catalog, social) {
    var d = normalizeDetector(detector);
    if (!d) return ['No conditions'];
    var out = [];
    d.conditions.forEach(function (c, i) {
        if (!resolveFeatureName(catalog, c.feature, social)) {
            out.push('Condition ' + (i + 1) + ': "' + c.feature + '" is not a feature of this session' +
                (/ height/.test(c.feature) ? ' (heights need 3D)' : social ? '' : (catalog.pair.some(function (f) { return f.name === c.feature; }) ? ' (it describes a pair: make the behavior social)' : '')));
        }
    });
    return out;
}

// ----------------------------------------------------------------------------
// Templates
// ----------------------------------------------------------------------------

/**
 * Starting points for common behaviors. `{head}` stands for the head node's
 * name. Thresholds are in body lengths (BL) and BL/s, set for mice and rats;
 * check them against the histogram in the settings.
 */
export var DETECTOR_TEMPLATES = [
    { id: 'rearing', name: 'rearing', social: false, mutual: false,
      description: 'The head is high above the floor',
      detector: { conditions: [{ feature: '{head} height · mean ±0.1 s', op: '>', value: 0.5 }], minDuration: 0.3, maxGap: 0.2 } },
    { id: 'immobile', name: 'immobile', social: false, mutual: false,
      description: 'The body hardly moves',
      detector: { conditions: [{ feature: 'centroid speed · mean ±0.3 s', op: '<', value: 0.1 }], minDuration: 1, maxGap: 0.2 } },
    { id: 'locomotion', name: 'locomotion', social: false, mutual: false,
      description: 'The body moves along the floor',
      detector: { conditions: [{ feature: 'centroid speed · mean ±0.3 s', op: '>', value: 1 }], minDuration: 0.5, maxGap: 0.2 } },
    { id: 'contact', name: 'contact', social: true, mutual: true,
      description: 'Two animals touch: their closest nodes are within a fraction of a body length',
      detector: { conditions: [{ feature: 'min node distance', op: '<', value: 0.15 }], minDuration: 0.2, maxGap: 0.2 } },
    { id: 'nose-to-nose', name: 'nose-to-nose', social: true, mutual: true,
      description: 'The two heads are close together',
      detector: { conditions: [{ feature: 'head–head distance', op: '<', value: 0.2 }], minDuration: 0.1, maxGap: 0.1 } },
    { id: 'anogenital', name: 'anogenital approach', social: true, mutual: false,
      description: "The actor's head is at the target's tail base",
      detector: { conditions: [{ feature: 'head–tail base distance', op: '<', value: 0.25 }], minDuration: 0.2, maxGap: 0.2 } },
    { id: 'following', name: 'following', social: true, mutual: false,
      description: 'The target is ahead (within ±45°), close, and both are moving',
      detector: { conditions: [
          { feature: 'target bearing', op: '<', value: 45 },
          { feature: 'centroid distance', op: '<', value: 2 },
          { feature: 'centroid speed · mean ±0.3 s', op: '>', value: 0.5 },
          { feature: TARGET_PREFIX + 'centroid speed · mean ±0.3 s', op: '>', value: 0.5 },
      ], minDuration: 0.5, maxGap: 0.3 } },
];

/**
 * A template's detector with `{head}` filled in.
 * @param {Object} template one of DETECTOR_TEMPLATES
 * @param {{head?: string}} roleNames node names of the roles
 * @returns {{detector?: Object, error?: string}}
 */
export function instantiateTemplate(template, roleNames) {
    if (!template) return { error: 'No such template' };
    var missing = null;
    var d = JSON.parse(JSON.stringify(template.detector));
    d.conditions.forEach(function (c) {
        c.feature = c.feature.replace(/\{(\w+)\}/g, function (m, role) {
            var nm = roleNames && roleNames[role];
            if (!nm) missing = role;
            return nm || m;
        });
    });
    if (missing) return { error: 'The template needs a ' + missing + ' node: set it under Body' };
    return { detector: normalizeDetector(d) };
}

// ----------------------------------------------------------------------------
// Scoring
// ----------------------------------------------------------------------------

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

/**
 * Per-frame scores of an AND of conditions (see the header).
 * @param {Array<{op:string, value:number, unit:string}>} conditions
 * @param {Array<ArrayLike<number>>} columns one feature column per condition, all `n` long
 * @param {number} n
 * @param {Float32Array} [out]
 * @returns {Float32Array}
 */
export function scoreConditions(conditions, columns, n, out) {
    out = out || new Float32Array(n);
    var m = conditions.length;
    var scales = conditions.map(function (c) { return conditionScale(c.unit, c.value); });
    for (var t = 0; t < n; t++) {
        var low = 1, all = true, missing = m === 0;
        for (var i = 0; i < m && !missing; i++) {
            var v = columns[i][t];
            if (v !== v) { missing = true; break; }
            var c = conditions[i];
            var margin = c.op === '<' ? c.value - v : v - c.value;
            if (margin < 0) all = false;
            var s = sigmoid(margin / scales[i]);
            if (s < low) low = s;
        }
        out[t] = missing ? NaN : all ? Math.max(low, 0.5) : Math.min(low, 0.4999);
    }
    return out;
}

/**
 * Bouts of a detector from its per-frame scores.
 * @param {ArrayLike<number>} scores
 * @param {{minDuration?:number, maxGap?:number}} detector seconds
 * @param {number} fps
 * @param {boolean} [point] a point behavior: one frame per bout, at its peak
 */
export function detectorBouts(scores, detector, fps, point) {
    var d = detector || {};
    return scoresToBouts(scores, {
        threshold: 0.5,
        minFrames: Math.max(1, secondsToFrames(d.minDuration, fps)),
        maxGapFrames: secondsToFrames(d.maxGap, fps),
        point: !!point,
    });
}

/**
 * The plan for running detectors on a chunk: which features to compute, and
 * for each behavior how each condition maps to them.
 * @param {Object} catalog
 * @param {Array<{behaviorId:string, social:boolean, mutual:boolean, detector:Object}>} behaviorsIn
 * @returns {{request: {animal:string[], pair:string[]}, plans: Object[], skipped: Array<{behaviorId, problems}>}}
 */
export function planDetectors(catalog, behaviorsIn) {
    var animal = new Set(), pair = new Set();
    var plans = [], skipped = [];
    (behaviorsIn || []).forEach(function (b) {
        var d = normalizeDetector(b.detector);
        var problems = d ? detectorProblems(d, catalog, b.social) : ['No conditions'];
        if (problems.length) { skipped.push({ behaviorId: b.behaviorId, problems: problems }); return; }
        var conds = d.conditions.map(function (c) {
            var r = resolveFeatureName(catalog, c.feature, b.social);
            if (r.scope === 'pair') pair.add(r.feature.name); else animal.add(r.feature.name);
            return { name: r.feature.name, who: r.who, op: c.op, value: c.value, unit: r.feature.unit };
        });
        plans.push({ behaviorId: b.behaviorId, social: !!b.social, mutual: !!b.mutual, point: !!b.point, detector: d, conditions: conds });
    });
    return { request: { animal: Array.from(animal), pair: Array.from(pair) }, plans: plans, skipped: skipped };
}

/**
 * A mutual pair's per-frame score: the better of its two directions (a frame
 * scored in only one direction keeps that score).
 */
export function combineMutual(x, y) {
    var n = x.length;
    var out = new Float32Array(n);
    for (var t = 0; t < n; t++) {
        var u = x[t], v = y[t];
        out[t] = u !== u ? v : v !== v ? u : Math.max(u, v);
    }
    return out;
}

/** Every ordered pair of `n` animals. */
export function orderedPairs(n) {
    var out = [];
    for (var a = 0; a < n; a++) for (var b = 0; b < n; b++) if (a !== b) out.push([a, b]);
    return out;
}

/**
 * Score every planned detector on one chunk (what a worker job does).
 * @param {Object} spec feature spec (see pose/behavior-features.js)
 * @param {Object} chunk {start, end, padStart, padEnd, data}
 * @param {{request:Object, plans:Object[]}} plan from `planDetectors`
 * @param {{columns?: {animal?:string[], pair?:string[]}}} [extra] feature columns to return as well
 * @returns {{scores: Object<string, Object<string, Float32Array>>, columns: Object}}
 *   scores[behaviorId][key]: key 'a' (animal index) or 'a>b' (actor > target;
 *   for a mutual behavior a < b and the score is the better direction)
 */
export function scoreDetectorsChunk(spec, chunk, plan, extra) {
    var nA = chunk.data.length;
    var anySocial = plan.plans.some(function (p) { return p.social; });
    var pairs = anySocial || (extra && extra.columns && (extra.columns.pair || []).length) ? orderedPairs(nA) : [];
    var reqAnimal = plan.request.animal.slice(), reqPair = plan.request.pair.slice();
    if (extra && extra.columns) {
        (extra.columns.animal || []).forEach(function (nm) { if (reqAnimal.indexOf(nm) < 0) reqAnimal.push(nm); });
        (extra.columns.pair || []).forEach(function (nm) { if (reqPair.indexOf(nm) < 0) reqPair.push(nm); });
    }
    var feats = computeFeatureChunk(spec, chunk, { animal: reqAnimal, pair: reqPair, pairs: pairs });
    var n = chunk.end - chunk.start;
    var scores = {};
    plan.plans.forEach(function (p) {
        var byKey = {};
        if (!p.social) {
            for (var a = 0; a < nA; a++) {
                byKey[String(a)] = scoreConditions(p.conditions, p.conditions.map(function (c) { return feats.animal[a][c.name]; }), n);
            }
        } else {
            var directed = {};
            pairs.forEach(function (pr) {
                var key = pr[0] + '>' + pr[1];
                directed[key] = scoreConditions(p.conditions, p.conditions.map(function (c) {
                    if (c.who === 'pair') return feats.pair[key][c.name];
                    return feats.animal[c.who === 'target' ? pr[1] : pr[0]][c.name];
                }), n);
            });
            if (!p.mutual) byKey = directed;
            else {
                pairs.forEach(function (pr) {
                    if (pr[0] > pr[1]) return;
                    byKey[pr[0] + '>' + pr[1]] = combineMutual(directed[pr[0] + '>' + pr[1]], directed[pr[1] + '>' + pr[0]]);
                });
            }
        }
        scores[p.behaviorId] = byKey;
    });
    var columns = {};
    if (extra && extra.columns) {
        columns.animal = feats.animal.map(function (res) {
            var o = {};
            (extra.columns.animal || []).forEach(function (nm) { if (res[nm]) o[nm] = res[nm]; });
            return o;
        });
        columns.pair = {};
        Object.keys(feats.pair).forEach(function (k) {
            var o = {};
            (extra.columns.pair || []).forEach(function (nm) { if (feats.pair[k][nm]) o[nm] = feats.pair[k][nm]; });
            columns.pair[k] = o;
        });
    }
    return { scores: scores, columns: columns };
}

/** The catalog a feature spec yields (re-exported for callers that only import rules). */
export function catalogOf(spec) {
    return catalogFor(spec);
}

// ----------------------------------------------------------------------------
// The settings editor's live numbers
// ----------------------------------------------------------------------------

/**
 * A histogram of feature values pooled over several columns, on a robust
 * range (the 0.5th to 99.5th percentile, widened to include `include`).
 * @param {Array<ArrayLike<number>>} columns
 * @param {number} bins
 * @param {number} [include] a value the range must show (the threshold)
 * @returns {{lo:number, hi:number, counts:Uint32Array, n:number, below:number, above:number}|null}
 */
export function featureHistogram(columns, bins, include) {
    var sample = [];
    var total = 0;
    columns.forEach(function (c) { if (c) total += c.length; });
    var step = Math.max(1, Math.floor(total / 200000));
    var k = 0;
    columns.forEach(function (c) {
        if (!c) return;
        for (var i = 0; i < c.length; i++, k++) {
            if (k % step) continue;
            var v = c[i];
            if (v === v) sample.push(v);
        }
    });
    if (!sample.length) return null;
    sample.sort(function (a, b) { return a - b; });
    var lo = sample[Math.floor(0.005 * (sample.length - 1))], hi = sample[Math.ceil(0.995 * (sample.length - 1))];
    if (Number.isFinite(include)) { lo = Math.min(lo, include); hi = Math.max(hi, include); }
    if (!(hi > lo)) { hi = lo + (Math.abs(lo) * 0.1 || 1); lo -= (Math.abs(lo) * 0.1 || 1) * 0.5; }
    var pad = (hi - lo) * 0.02;
    lo -= pad; hi += pad;
    var counts = new Uint32Array(bins);
    var n = 0, below = 0, above = 0;
    columns.forEach(function (c) {
        if (!c) return;
        for (var i = 0; i < c.length; i++) {
            var v = c[i];
            if (v !== v) continue;
            n++;
            if (v < lo) { below++; continue; }
            if (v >= hi) { above++; continue; }
            counts[Math.min(bins - 1, Math.floor((v - lo) / (hi - lo) * bins))]++;
        }
    });
    return { lo: lo, hi: hi, counts: counts, n: n, below: below, above: above };
}

/**
 * Bouts and detected frames a detector gives over several contexts (animals
 * or pairs), from their feature columns — the editor's live count.
 * @param {Array<Array<ArrayLike<number>>>} contexts per context, one column per condition
 * @param {Array<{op, value, unit}>} conditions
 * @param {Object} detector (minDuration, maxGap)
 * @param {number} fps
 * @param {boolean} [point]
 * @returns {{bouts:number, frames:number}}
 */
export function previewDetector(contexts, conditions, detector, fps, point) {
    var bouts = 0, frames = 0;
    contexts.forEach(function (cols) {
        if (!cols.length || !cols[0]) return;
        var n = cols[0].length;
        var sc = scoreConditions(conditions, cols, n);
        detectorBouts(sc, detector, fps, point).forEach(function (b) {
            bouts++;
            frames += b[1] - b[0] + 1;
        });
    });
    return { bouts: bouts, frames: frames };
}
