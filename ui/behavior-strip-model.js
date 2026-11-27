// ui/behavior-strip-model.js — what the bouts strip above the seekbar shows
// (the DOM half is ui/behavior-strip.js): its lanes, its window of time, and
// where everything sits in that window.
//
// ## The lanes
//
// One lane per animal and behavior that has a bout or a running state, built
// only from the labels (`listBouts`, `runningStates`) — never from frames, so a
// lazy project's non-resident frames make no difference, and a label edit never
// walks frames. A mutual bout is stored once for its pair and listed in BOTH
// animals' lanes; a directed one (actor → target) in the actor's.
//
// ## The window
//
// The playhead stays in the middle: `half` frames show on each side of its
// frame. Zooming picks one of `STRIP_ZOOM_STEPS` (seconds each side).
//
// ## The film
//
// Nothing is drawn as the playhead moves. Bars and ticks are drawn once on a
// "film", three windows wide, and each frame only copies the film's slice under
// the window onto the screen (`filmSourceX`). The film is redrawn when the window
// would run off its end (`filmHolds`), so once per window of playback, and when
// the labels or the zoom change. A running state's bar is the one thing that
// changes shape as the playhead moves, and is drawn over the copy
// (`runningBarWidth`).
//
// Every frame has a slot one `pxPerFrame` wide, and the playhead sits in the
// middle of its frame's slot, as on the Timeline.
//
// Imports only the seekbar's ruler and tooltip helpers and the readout's number
// format, all free of the DOM, so Node tests load it directly
// (`tests/test-behavior-strip.mjs`).

import { chooseTickSteps, formatTickLabel, stepDecimals } from './seekbar-ticks.js';
import { seekbarTooltipText } from './seekbar-tooltip.js';
import { formatFrameNumber } from './frame-readout.js';

/** How much time the strip can show, in seconds each side of the playhead. */
export var STRIP_ZOOM_STEPS = [0.5, 1, 2, 3, 5, 10, 20, 30, 60, 120];
export var DEFAULT_HALF_SECONDS = 5;
/** One lane's height; the bars are 2 px shorter, so neighbouring lanes never touch. */
export var LANE_HEIGHT_PX = 14;
/** The time ruler's height. */
export var RULER_HEIGHT_PX = 15;
/** Lanes on show before the strip scrolls. */
export var MAX_VISIBLE_LANES = 8;
/** A bar is never narrower than this, or a one-frame bout would vanish when zoomed out. */
export var MIN_BAR_PX = 2;
/** A point event is a tick this wide, so two on neighbouring frames stay two. */
export var POINT_BAR_PX = 3;
/** One label character on the strip's ruler (10 px monospace). */
var RULER_CHAR_PX = 6;

// ----------------------------------------------------------------------------
// Whether it shows
// ----------------------------------------------------------------------------

/**
 * Does the strip show? Only with an ethogram, something to seek through and the
 * user's choice on: always in Label Behaviors mode, where the labels are being
 * made, and outside it once the session has labels to read.
 * @param {{on:boolean, hasEthogram:boolean, totalFrames:number, mode:boolean, hasLabels:boolean}} s
 */
export function stripWanted(s) {
    return !!(s.on && s.hasEthogram && s.totalFrames > 0 && (s.mode || s.hasLabels));
}

// ----------------------------------------------------------------------------
// Zoom
// ----------------------------------------------------------------------------

/** A stored or typed window as one of the zoom steps, or the default when it is no number. */
export function normalizeHalfSeconds(v) {
    var n = typeof v === 'number' ? v : parseFloat(v);
    if (!(n > 0) || !isFinite(n)) return DEFAULT_HALF_SECONDS;
    var best = STRIP_ZOOM_STEPS[0];
    STRIP_ZOOM_STEPS.forEach(function (s) { if (Math.abs(s - n) < Math.abs(best - n)) best = s; });
    return best;
}

/** The next zoom step: `dir` > 0 zooms in (less time), `dir` < 0 out; stops at both ends. */
export function zoomedHalfSeconds(half, dir) {
    var i = STRIP_ZOOM_STEPS.indexOf(normalizeHalfSeconds(half));
    var j = Math.max(0, Math.min(STRIP_ZOOM_STEPS.length - 1, i + (dir > 0 ? -1 : 1)));
    return STRIP_ZOOM_STEPS[j];
}

/** The zoom readout: "±5 s". */
export function formatStripWindow(half) {
    return '±' + half + ' s';
}

// ----------------------------------------------------------------------------
// Lanes
// ----------------------------------------------------------------------------

var PAINT_RANK = { absent: 0, present: 1, running: 2 };

function itemKind(it) {
    return it.running ? 'running' : it.present ? 'present' : 'absent';
}

/**
 * The strip's lanes: known animals in their order, then any other name a bout
 * mentions (a deleted identity's) by name; within an animal, the ethogram's
 * order. Each lane's items are in paint order: "not" labels under present
 * bouts, running states on top. A bout naming no ethogram behavior has no lane.
 *
 * @param {string[]} animals the session's animals, in order
 * @param {Array} ethogram the behaviors, in order
 * @param {Array} bouts as `listBouts` returns them
 * @param {Array} running as `runningStates` returns them
 * @param {function(Object): string} glyphOf how a behavior's pair reads (`pairGlyph`)
 * @returns {Array<{animal:string, behaviorId:string, name:string, color:string, point:boolean, pair:string,
 *                  items:Array<{start:number, end:number, present:boolean, running:boolean, partner:string}>}>}
 */
export function stripLanes(animals, ethogram, bouts, running, glyphOf) {
    var order = new Map();
    (ethogram || []).forEach(function (b, i) { order.set(b.id, { b: b, i: i }); });
    var lanes = new Map();
    var add = function (rec, end, present, isRunning) {
        var e = order.get(rec.behaviorId);
        if (!e) return;
        // Both animals of a mutual pair; the actor of anything else.
        var who = e.b.mutual && rec.target ? [rec.animal, rec.target] : [rec.animal];
        who.forEach(function (animal) {
            var key = animal + '\u0001' + rec.behaviorId;
            var lane = lanes.get(key);
            if (!lane) {
                lane = { animal: animal, behaviorId: rec.behaviorId, name: e.b.name, color: e.b.color, point: e.b.kind === 'point',
                         pair: glyphOf ? glyphOf(e.b) : '', order: e.i, items: [] };
                lanes.set(key, lane);
            }
            lane.items.push({ start: rec.start, end: end, present: present, running: isRunning,
                              partner: rec.target ? (animal === rec.animal ? rec.target : rec.animal) : '' });
        });
    };
    (bouts || []).forEach(function (bt) { add(bt, bt.end, !!bt.present, false); });
    (running || []).forEach(function (r) { add(r, r.start, true, true); });

    var rank = new Map();
    (animals || []).forEach(function (n, i) { rank.set(n, i); });
    var rankOf = function (n) { return rank.has(n) ? rank.get(n) : (animals || []).length; };
    var out = Array.from(lanes.values());
    out.sort(function (p, q) {
        return rankOf(p.animal) - rankOf(q.animal) || (p.animal < q.animal ? -1 : p.animal > q.animal ? 1 : 0) || p.order - q.order;
    });
    out.forEach(function (lane) {
        lane.items.sort(function (a, b) { return PAINT_RANK[itemKind(a)] - PAINT_RANK[itemKind(b)] || a.start - b.start; });
    });
    return out;
}

// ----------------------------------------------------------------------------
// Window and film
// ----------------------------------------------------------------------------

/**
 * @param {number} halfSeconds time each side of the playhead
 * @param {number} fps
 * @param {number} widthPx the strip's width
 * @returns {{half:number, visible:number, pxPerFrame:number, width:number}}
 */
export function stripScale(halfSeconds, fps, widthPx) {
    var half = Math.max(1, Math.round(halfSeconds * fps));
    var visible = 2 * half + 1;
    return { half: half, visible: visible, pxPerFrame: widthPx / visible, width: widthPx };
}

/** The frame whose slot holds `x` (px from the strip's left edge) with the playhead on `frame`. Not clamped. */
export function frameAtX(x, frame, scale) {
    return frame + Math.floor((x - scale.width / 2) / scale.pxPerFrame + 0.5);
}

/** `x` of the middle of a frame's slot, on the strip (where the playhead would sit on it). */
export function xOfFrame(f, frame, scale) {
    return scale.width / 2 + (f - frame) * scale.pxPerFrame;
}

/**
 * A film centred on `frame`: one window each side of the one on show, or less
 * when that would pass `maxWidthPx`.
 */
export function stripFilm(frame, scale, maxWidthPx) {
    var margin = scale.visible;
    if (maxWidthPx > 0) margin = Math.max(0, Math.min(margin, Math.floor((maxWidthPx - scale.width) / 2 / scale.pxPerFrame)));
    var count = scale.visible + 2 * margin;
    var start = frame - scale.half - margin;
    return { start: start, end: start + count - 1, width: count * scale.pxPerFrame };
}

/** Can the film show the window around `frame`, or must it be rebuilt? */
export function filmHolds(film, frame, scale) {
    return frame - scale.half >= film.start && frame + scale.half <= film.end;
}

/** Left edge of a frame's slot, on the film. */
function filmX(f, film, scale) {
    return (f - film.start) * scale.pxPerFrame;
}

/**
 * Where the window starts on the film with the playhead on `frame`, in whole
 * device pixels: the copy moves no pixel by a fraction, so edges stay sharp.
 */
export function filmSourceX(frame, film, scale, dpr) {
    var d = dpr > 0 ? dpr : 1;
    return Math.round(((frame - film.start + 0.5) * scale.pxPerFrame - scale.width / 2) * d);
}

/**
 * The film's rectangle for frames [start, end], clipped to the film; a narrow one
 * is widened around its middle, so a one-frame bout stays on its frame.
 * @returns {{x:number, w:number}|null} null when it is entirely off the film
 */
export function filmRect(start, end, film, scale, minPx) {
    if (end < film.start || start > film.end) return null;
    var x0 = filmX(Math.max(start, film.start), film, scale);
    var x1 = filmX(Math.min(end, film.end) + 1, film, scale);
    var w = x1 - x0, min = minPx > 0 ? minPx : MIN_BAR_PX;
    return w >= min ? { x: x0, w: w } : { x: (x0 + x1) / 2 - min / 2, w: min };
}

/** A running state's bar width with the playhead on `frame`: it ends with the playhead's slot, like the Timeline's. */
export function runningBarWidth(item, frame, film, scale) {
    var x0 = filmX(Math.max(item.start, film.start), film, scale);
    var x1 = filmX(Math.max(item.start, frame) + 1, film, scale);
    return Math.max(MIN_BAR_PX, x1 - x0);
}

/**
 * The bars on a film: `{lane, x, w, kind, partner, start, end, color, item}`, in
 * paint order. Zoomed out past one pixel a frame, bars of a lane that touch and
 * say the same thing become one, so a dense lane cannot make thousands of them.
 *
 * @param {Array} lanes from `stripLanes`
 * @param {number} cursor the playhead (a running state's end)
 */
export function filmBars(lanes, film, scale, cursor) {
    var out = [];
    var coarse = scale.pxPerFrame < 1;
    lanes.forEach(function (lane, li) {
        var prev = null;
        lane.items.forEach(function (it) {
            var end = it.running ? Math.max(it.start, cursor) : it.end;
            var kind = itemKind(it);
            var r = filmRect(it.start, end, film, scale, lane.point && kind === 'present' ? POINT_BAR_PX : MIN_BAR_PX);
            if (!r) return;
            if (lane.point && kind === 'present') { r = { x: r.x + r.w / 2 - POINT_BAR_PX / 2, w: POINT_BAR_PX }; }
            if (coarse && prev && kind !== 'running' && prev.kind === kind && prev.partner === it.partner && r.x - (prev.x + prev.w) < 1) {
                prev.w = Math.max(prev.w, r.x + r.w - prev.x);
                prev.end = Math.max(prev.end, end);
                return;
            }
            prev = { lane: li, x: r.x, w: r.w, kind: kind, partner: it.partner, start: it.start, end: end, color: lane.color, item: it };
            out.push(prev);
        });
    });
    return out;
}

/** The film's stretches outside the recording, as rectangles (frames before 0, after the last). */
export function filmOutside(film, scale, totalFrames) {
    var out = [];
    if (film.start < 0) out.push({ x: 0, w: filmX(Math.min(0, film.end + 1), film, scale) });
    if (film.end >= totalFrames) {
        var x = filmX(Math.max(totalFrames, film.start), film, scale);
        out.push({ x: x, w: film.width - x });
    }
    return out;
}

/**
 * The ruler's ticks on a film, at the times the seekbar's ruler would mark and
 * under the middle of the frame slot the playhead sits in at that time. So with
 * the readout on 00:41.000, the playhead is on the 00:41 tick.
 *
 * @returns {{step:number, minorStep:number, majors:Array<{x:number, label:string}>, minors:number[]}}
 */
export function filmTicks(film, scale, fps, totalFrames) {
    var none = { step: 0, minorStep: 0, majors: [], minors: [] };
    if (!(fps > 0) || !(totalFrames > 0) || !(scale.width > 0)) return none;
    var lastSec = (totalFrames - 1) / fps;
    var labelPx = function (s) {
        return formatTickLabel(Math.floor(lastSec / s + 1e-9) * s, stepDecimals(s)).length * RULER_CHAR_PX;
    };
    var steps = chooseTickSteps(scale.pxPerFrame * fps, 1 / fps, labelPx);
    if (!steps.step) return none;
    var fromSec = Math.max(0, film.start) / fps;
    var toSec = Math.min(film.end, totalFrames - 1) / fps;
    var atSec = function (t) { return (t * fps - film.start + 0.5) * scale.pxPerFrame; };
    var majors = [], minors = [];
    var dec = stepDecimals(steps.step);
    for (var k = Math.ceil(fromSec / steps.step - 1e-9); k * steps.step <= toSec + 1e-9; k++) {
        majors.push({ x: atSec(k * steps.step), label: formatTickLabel(k * steps.step, dec) });
    }
    if (steps.minorStep) {
        var per = Math.round(steps.step / steps.minorStep);
        for (var q = Math.ceil(fromSec / steps.minorStep - 1e-9); q * steps.minorStep <= toSec + 1e-9; q++) {
            if (q % per) minors.push(atSec(q * steps.minorStep));
        }
    }
    return { step: steps.step, minorStep: steps.minorStep, majors: majors, minors: minors };
}

// ----------------------------------------------------------------------------
// Hover
// ----------------------------------------------------------------------------

/**
 * What a lane shows on `frame`, or within `tol` frames of it (a bar widened to
 * stay visible), the topmost first; null when nothing.
 * @param {number} cursor the playhead (a running state's end)
 */
export function boutAt(lane, frame, cursor, tol) {
    var t = tol > 0 ? tol : 0;
    for (var i = lane.items.length - 1; i >= 0; i--) {
        var it = lane.items[i];
        var end = it.running ? Math.max(it.start, cursor) : it.end;
        if (frame >= it.start - t && frame <= end + t) return it;
    }
    return null;
}

/** How a bout reads: "not rear · id_1 · frames 1,200–1,260 (1.0 s)", "● chase · id_1 → id_2 · since frame 1,200". */
export function boutText(lane, it, fps) {
    var who = lane.animal + (it.partner ? lane.pair + it.partner : '');
    if (it.running) return '● ' + lane.name + ' · ' + who + ' · since frame ' + formatFrameNumber(it.start + 1);
    var n = it.end - it.start + 1;
    return (it.present ? '' : 'not ') + lane.name + ' · ' + who + ' · ' +
        (n === 1 ? 'frame ' + formatFrameNumber(it.start + 1)
            : 'frames ' + formatFrameNumber(it.start + 1) + '–' + formatFrameNumber(it.end + 1) +
              (fps > 0 ? ' (' + (n / fps).toFixed(1) + ' s)' : ''));
}

/** The tooltip: the frame a click would land on, as the seekbar says it, and the bout under the pointer. */
export function stripHoverText(frame, fps, lane, it) {
    var text = seekbarTooltipText(frame, fps);
    return lane && it ? text + ' — ' + boutText(lane, it, fps) : text;
}
