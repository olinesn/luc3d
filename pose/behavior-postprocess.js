// pose/behavior-postprocess.js — from per-frame scores to bouts, for every
// source of behavior predictions (rule detectors now; classifiers, keypoint-
// MoSeq states and discovery later).
//
// A frame is detected when its score is at least the threshold. Detected runs
// separated by at most `maxGapFrames` undetected (or unscored) frames are
// joined first, then runs shorter than `minFrames` are dropped — the order
// that keeps a bout with one noisy frame in the middle whole. A bout's score
// is the mean of its frames' finite scores. A POINT behavior keeps one frame
// per bout: the one where its score peaks.
//
// Pure and import-free (the behavior worker loads it).
//
// Depends on: nothing.

/**
 * Bouts from per-frame scores.
 * @param {ArrayLike<number>} scores NaN = unscored
 * @param {{threshold?:number, minFrames?:number, maxGapFrames?:number, point?:boolean, start?:number, end?:number}} [opts]
 *   threshold default 0.5; minFrames default 1; maxGapFrames default 0;
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
    // Join across short gaps.
    var joined = [];
    for (var i = 0; i < runs.length; i++) {
        var last = joined[joined.length - 1];
        if (last && runs[i][0] - last[1] - 1 <= maxGap) last[1] = runs[i][1];
        else joined.push([runs[i][0], runs[i][1]]);
    }
    var out = [];
    for (var j = 0; j < joined.length; j++) {
        var s = joined[j][0], e = joined[j][1];
        if (e - s + 1 < minFrames) continue;
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
