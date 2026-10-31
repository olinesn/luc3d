// pose/behavior-postprocess.js — from per-frame scores to bouts, for every
// source of behavior predictions (rule detectors now; classifiers, keypoint-
// MoSeq states and discovery later).
//
// A frame is detected when its score is at least the threshold. Detected runs
// separated by at most `maxGapFrames` undetected (or unscored) frames are
// joined first — so a bout with one noisy frame in the middle stays whole —
// and a joined bout is kept only when its DETECTED frames number at least
// `minFrames`: the gaps it bridged do not count, so scattered one-frame
// detections never add up to a bout (on real recordings they made bouts that
// were mostly undetected). A bout's score is the mean of its frames' finite
// scores, the bridged ones included. A POINT behavior keeps one frame per
// bout: the one where its score peaks.
//
// Pure and import-free (the behavior worker loads it).
//
// Depends on: nothing.

/**
 * Bouts from per-frame scores.
 * @param {ArrayLike<number>} scores NaN = unscored
 * @param {{threshold?:number, minFrames?:number, maxGapFrames?:number, point?:boolean, start?:number, end?:number}} [opts]
 *   threshold default 0.5; minFrames (detected frames a bout needs) default 1; maxGapFrames default 0;
 *   [start, end) limits the frames read (default all)
 * @returns {Array<number[]>} `[start, end (inclusive), score]`, sorted
 */
export function scoresToBouts(scores, opts) {
    opts = opts || {};
    var thr = opts.threshold != null ? opts.threshold : 0.5;
    var minFrames = Math.max(1, Math.round(opts.minFrames || 1));
    var maxGap = Math.max(0, Math.round(opts.maxGapFrames || 0));
    var s0 = Math.max(0, opts.start || 0);
    var e0 = Math.min(scores.length, opts.end != null ? opts.end : scores.length);
    // Raw detected runs.
    var runs = [];
    var runStart = -1;
    for (var t = s0; t < e0; t++) {
        var v = scores[t];
        var on = v >= thr;
        if (on && runStart < 0) runStart = t;
        if (!on && runStart >= 0) { runs.push([runStart, t - 1]); runStart = -1; }
    }
    if (runStart >= 0) runs.push([runStart, e0 - 1]);
    // Join across short gaps, counting the detected frames each bout holds.
    var joined = [];
    for (var i = 0; i < runs.length; i++) {
        var last = joined[joined.length - 1];
        var len = runs[i][1] - runs[i][0] + 1;
        if (last && runs[i][0] - last[1] - 1 <= maxGap) { last[1] = runs[i][1]; last[2] += len; }
        else joined.push([runs[i][0], runs[i][1], len]);
    }
    var out = [];
    for (var j = 0; j < joined.length; j++) {
        var s = joined[j][0], e = joined[j][1];
        if (joined[j][2] < minFrames) continue;
        var sum = 0, n = 0, peak = s, best = -Infinity;
        for (var f = s; f <= e; f++) {
            var x = scores[f];
            if (x === x) { sum += x; n++; if (x > best) { best = x; peak = f; } }
        }
        var score = n ? sum / n : NaN;
        out.push(opts.point ? [peak, peak, best] : [s, e, score]);
    }
    return out;
}

/** Seconds to frames at `fps` (rounded; never negative). */
export function secondsToFrames(seconds, fps) {
    var v = Number(seconds) * (fps > 0 ? fps : 30);
    return Number.isFinite(v) && v > 0 ? Math.round(v) : 0;
}

// ----------------------------------------------------------------------------
// Metrics (classifier evaluation)
// ----------------------------------------------------------------------------

function prf(tp, fp, fn) {
    var p = tp + fp > 0 ? tp / (tp + fp) : NaN, r = tp + fn > 0 ? tp / (tp + fn) : NaN;
    var f1 = 2 * tp + fp + fn > 0 ? 2 * tp / (2 * tp + fp + fn) : NaN;
    return { precision: p, recall: r, f1: f1 };
}

/**
 * Frame-level precision, recall and F1 at a threshold, the decision
 * `scoresToBouts` makes (score ≥ threshold; an unscored frame is not
 * detected).
 * @param {ArrayLike<number>} scores
 * @param {ArrayLike<number>} truth 1 = present, 0 = absent
 * @returns {{tp, fp, fn, tn, precision, recall, f1}}
 */
export function framePrf(scores, truth, threshold) {
    var tp = 0, fp = 0, fn = 0, tn = 0;
    for (var i = 0; i < scores.length; i++) {
        var on = scores[i] >= threshold;
        if (truth[i]) { if (on) tp++; else fn++; } else if (on) fp++; else tn++;
    }
    return Object.assign({ tp: tp, fp: fp, fn: fn, tn: tn }, prf(tp, fp, fn));
}

/**
 * A precision–recall curve: frame-level P, R and F1 at thresholds 0, 1/steps,
 * …, 1 (default 100 steps), and the threshold with the best F1 (the lowest
 * of equal ones).
 * @returns {{points: Array<{threshold, precision, recall, f1}>, best: {threshold, precision, recall, f1}|null}}
 */
export function prCurve(scores, truth, steps) {
    var k = Math.max(1, steps || 100);
    var points = [], best = null;
    for (var i = 0; i <= k; i++) {
        var thr = i / k;
        var m = framePrf(scores, truth, thr);
        var pt = { threshold: thr, precision: m.precision, recall: m.recall, f1: m.f1 };
        points.push(pt);
        if (m.f1 === m.f1 && (!best || m.f1 > best.f1)) best = pt;
    }
    return { points: points, best: best };
}

/**
 * Bout-level precision, recall and F1: a predicted bout counts as right when
 * it overlaps a true bout by at least one frame, and a true bout as found
 * when a predicted bout overlaps it.
 * @param {Array<number[]>} predicted [start, end] inclusive, sorted by start
 * @param {Array<number[]>} truth [start, end] inclusive, sorted by start
 * @returns {{predicted, hits, truth, found, precision, recall, f1}}
 */
export function boutPrf(predicted, truth) {
    var hits = 0, found = 0;
    var j = 0;
    for (var i = 0; i < predicted.length; i++) {
        var p = predicted[i];
        while (j < truth.length && truth[j][1] < p[0]) j++;
        if (j < truth.length && truth[j][0] <= p[1]) hits++;
    }
    var k = 0;
    for (var t = 0; t < truth.length; t++) {
        var tr = truth[t];
        while (k < predicted.length && predicted[k][1] < tr[0]) k++;
        if (k < predicted.length && predicted[k][0] <= tr[1]) found++;
    }
    var precision = predicted.length ? hits / predicted.length : NaN, recall = truth.length ? found / truth.length : NaN;
    var f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : (precision === 0 || recall === 0 ? 0 : NaN);
    return { predicted: predicted.length, hits: hits, truth: truth.length, found: found, precision: precision, recall: recall, f1: f1 };
}

/**
 * An exclusive group's per-frame decision: on each frame only the member
 * scoring highest among those at or above their own threshold stays
 * detected; every other member above its threshold is pushed just under it
 * (scores below a threshold, and unscored frames, are left alone). Ties go to
 * the earlier member.
 * @param {Array<{scores: ArrayLike<number>, threshold: number}>} members one animal (or pair), same length
 * @returns {Float32Array[]} adjusted copies, in member order
 */
export function exclusiveScores(members) {
    var out = members.map(function (m) { return Float32Array.from(m.scores); });
    if (members.length < 2) return out;
    var n = out[0].length;
    for (var t = 0; t < n; t++) {
        var top = -1, topScore = -Infinity;
        for (var i = 0; i < members.length; i++) {
            var v = out[i][t];
            if (v >= members[i].threshold && v > topScore) { top = i; topScore = v; }
        }
        if (top < 0) continue;
        for (var j = 0; j < members.length; j++) {
            if (j !== top && out[j][t] >= members[j].threshold) out[j][t] = members[j].threshold - 1e-4;
        }
    }
    return out;
}
