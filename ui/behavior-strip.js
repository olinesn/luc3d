// ui/behavior-strip.js — the bouts strip: a slim, zoomed-in view of the labels
// around the playhead, under the camera view and above the Timeline, lined up
// with the seekbar, so someone labelling can read the nearby bouts at a glance.
//
// It shows while the project has an ethogram and the user has not switched it
// off: always in Label Behaviors mode, and outside it once the session has
// labels (`stripWanted`). One lane per animal and behavior (lanes, window and
// geometry: ui/behavior-strip-model.js). A click seeks, leaving the video playing
// or paused as the seekbar does; Shift+drag opens the Timeline's menu that
// labels a range; the wheel zooms, Shift+wheel scrolls the lanes when there are
// more than fit.
//
// Playback must not get slower, so moving the playhead draws next to nothing:
// bars and ticks are drawn once on a hidden "film" canvas, and the per-frame call
// (`updateBehaviorStripFrame`, from `updateBehaviorFrame`) copies the slice under
// the window to the screen (`blit`), plus a rectangle per running state or range.
// The film is redrawn once per window of playback, and on a label, ethogram or
// zoom change (`requestBehaviorRepaint`). Not a CSS transform per frame: on this
// page that makes the browser re-sort all its layers (Layerize), about 1 ms a
// frame however small the element, against next to nothing for the copy
// (`tests/e2e/_bench-behavior-strip.mjs`). Nothing reads frames, so lazy projects
// need nothing resident.
//
// The on/off choice and the window are browser-local (`localStorage`), never in
// the `.slp`.
//
// Depends on: ui/app-state.js, ui/behavior-coding.js, ui/behavior-detect.js,
//             ui/behavior-strip-model.js, ui/behavior-timeline.js, ui/info-tip.js,
//             ui/plane-definition.js (`ICON_INFO`), pose/behavior-ethogram.js,
//             pose/behavior-labels.js, import-export/save-load.js

import { state } from './app-state.js';
import {
    behaviorUi, registerBehaviorRepaint, requestBehaviorRepaint, sessionAnimals, animalColor, focalAnimal, pairGlyph,
} from './behavior-coding.js';
import { detectionFps } from './behavior-detect.js';
import {
    DEFAULT_HALF_SECONDS, LANE_HEIGHT_PX, MAX_VISIBLE_LANES, MIN_BAR_PX, RULER_HEIGHT_PX, STRIP_ZOOM_STEPS,
    boutAt, filmBars, filmHolds, filmOutside, filmRect, filmSourceX, filmTicks, formatStripWindow, frameAtX,
    normalizeHalfSeconds, runningBarWidth, stripFilm, stripHoverText, stripLanes, stripScale, stripWanted,
    xOfFrame, zoomedHalfSeconds,
} from './behavior-strip-model.js';
import { ABSENT_COLOR, onTimelineRangeSelect } from './behavior-timeline.js';
import { setInfoTip } from './info-tip.js';
import { ICON_INFO } from './plane-definition.js';
import { behaviors } from '../pose/behavior-ethogram.js';
import { hasBehaviorData, listBouts, runningStates } from '../pose/behavior-labels.js';
import { setStatus } from '../import-export/save-load.js';

var ON_KEY = 'lucid.behaviorStrip';
var WINDOW_KEY = 'lucid.behaviorStripWindow';
/** A trackpad's small deltas add up to one zoom step; a mouse wheel's notch is one by itself. */
var WHEEL_STEP = 30;
/** "Not" labels: the Timeline's gray, faint (alpha 0x38 = 0.22, its alpha too). */
var ABSENT_FILL = ABSENT_COLOR + '38';
/** A canvas wider than this can lose its GPU backing; a very wide window gets a shorter film. */
var MAX_FILM_DEVICE_PX = 16000;
var OUTSIDE_FILL = 'rgba(255, 255, 255, 0.06)';
var ANIMAL_LINE = 'rgba(255, 255, 255, 0.14)';
var SELECTION_FILL = 'rgba(255, 255, 255, 0.18)';
var SELECTION_EDGE = 'rgba(255, 255, 255, 0.75)';
var ARMED_FILL = 'rgba(76, 200, 120, 0.18)';
var ARMED_EDGE = '#4cc878';
var PARTNER_FONT = '9px sans-serif';
var RULER_FONT = '10px "SF Mono", "Fira Code", "Cascadia Code", monospace';

var INFO = 'The labels around the playhead, one lane per animal and behavior. ' +
    'Click: go to that frame. Shift+drag: label a range. Wheel: zoom (Shift+wheel: scroll the lanes).';

var el = null;                  // the DOM handles (setupBehaviorStrip)
var _deps = {};
var _on = true;
var _halfSeconds = DEFAULT_HALF_SECONDS;
var _shown = false;
var _session = null;
var _frame = 0;                 // the frame on show
var _lanes = [];
var _viewLeft = 0;              // where the seekbar starts in the strip
var _viewWidth = 0;             // … and how wide it is
var _fps = 0;
var _scale = null;
var _film = null;               // which frames the film holds (`stripFilm`)
var _dpr = 1;
var _ctx = null;                // the lanes on screen, and the ruler
var _rulerCtx = null;
var _filmCtx = null;            // the hidden films they copy from
var _rulerFilmCtx = null;
var _sx = NaN;                  // where the copy on screen starts on the film
var _rulerColors = null;        // the ruler's colours, from the CSS, read once
var _builtTotal = -1;           // what the film was drawn for
var _builtFps = -1;
var _labelsHtml = null;         // what is in the DOM, so an unchanged repaint writes nothing
var _running = [];              // the running bars, drawn over the copy
var _armed = null;              // the range armed with [ ], drawn over the copy
var _revealed = null;           // the focal animal last brought into view
var _rootRect = null;           // read once per hover, so moving the pointer stays layout-free
var _bodyRect = null;
var _drag = null;               // a Shift+drag: {anchor, animal}
var _selection = null;          // its range, also while its menu is open
var _hoverShown = false;
var _tipLen = -1;               // the tooltip's half width, measured when its length changes
var _tipHalf = 0;
var _wheelSum = 0;
var _wheelAt = 0;

function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function px(n) { return n.toFixed(2) + 'px'; }

// ----------------------------------------------------------------------------
// The browser-local choices
// ----------------------------------------------------------------------------

function readChoices() {
    try {
        _on = localStorage.getItem(ON_KEY) !== '0';
        _halfSeconds = normalizeHalfSeconds(localStorage.getItem(WINDOW_KEY));
    } catch (e) { /* storage unavailable: the defaults */ }
}

/** Is the strip switched on (this browser's choice)? */
export function behaviorStripOn() {
    return _on;
}

export function setBehaviorStripOn(on) {
    _on = !!on;
    try { localStorage.setItem(ON_KEY, _on ? '1' : '0'); } catch (e) { /* storage unavailable: session only */ }
    requestBehaviorRepaint({ ui: true });
}

function setWindow(half) {
    if (half === _halfSeconds) return;
    _halfSeconds = half;
    try { localStorage.setItem(WINDOW_KEY, String(half)); } catch (e) { /* storage unavailable: session only */ }
    showWindow();
    if (_shown) repaint();
}

function showWindow() {
    el.window.textContent = formatStripWindow(_halfSeconds);
    el.zoomIn.disabled = _halfSeconds === STRIP_ZOOM_STEPS[0];
    el.zoomOut.disabled = _halfSeconds === STRIP_ZOOM_STEPS[STRIP_ZOOM_STEPS.length - 1];
}

// ----------------------------------------------------------------------------
// Painting (never per frame)
// ----------------------------------------------------------------------------

/** Line the strip up with the seekbar. */
function measure() {
    var bar = document.getElementById('seekbar');
    if (!bar || !el) return;
    var r = bar.getBoundingClientRect();
    var left = r.left - el.root.parentElement.getBoundingClientRect().left;
    if (left === _viewLeft && r.width === _viewWidth) return;
    _viewLeft = left;
    _viewWidth = r.width;
    el.root.style.setProperty('--behavior-strip-left', px(left));
    el.root.style.setProperty('--behavior-strip-width', px(r.width));
    _rootRect = null;
    if (_shown) repaint();
}

/**
 * Bring the strip up to date with the labels, the ethogram and the mode: what
 * every repaint comes to, and what `updateInfoPanel` calls after a load (a load
 * does not notify).
 */
export function refreshBehaviorStrip() {
    if (!el) return;
    var session = state.session;
    if (session !== _session) { _session = session; _frame = state.currentFrame; _film = null; }
    var want = stripWanted({
        on: _on, hasEthogram: behaviors().length > 0, totalFrames: state.totalFrames,
        mode: !!behaviorUi().mode, hasLabels: hasBehaviorData(session),
    });
    if (want && !_shown) { _frame = state.currentFrame; _film = null; }
    if (want !== _shown) { _shown = want; el.root.hidden = !want; _rootRect = null; }
    if (!want) { hideHover(); releaseFilm(); return; }
    repaint();
}

/** A hidden strip holds no film. */
function releaseFilm() {
    _film = null;
    if (_filmCtx) { _filmCtx.canvas.width = 0; _rulerFilmCtx.canvas.width = 0; }
}

/** Repaint lanes, film and position for the labels now. */
function repaint() {
    var session = state.session;
    if (!_filmCtx) {
        _filmCtx = document.createElement('canvas').getContext('2d');
        _rulerFilmCtx = document.createElement('canvas').getContext('2d');
    }
    _lanes = stripLanes(sessionAnimals(session), behaviors(), listBouts(session), runningStates(session), pairGlyph);
    paintLabels(session);
    paintFilm();
    place(_frame, true);
}

function paintLabels(session) {
    var focal = behaviorUi().mode ? focalAnimal(session) : null;
    var html = _lanes.map(function (lane, i) {
        var running = lane.items.some(function (it) { return it.running; });
        var who = lane.animal + ' · ' + lane.name;
        return '<div class="behavior-strip-label' + (i && _lanes[i - 1].animal !== lane.animal ? ' is-next-animal' : '') +
            (lane.animal === focal ? ' is-focal' : '') + '" data-lane="' + i + '" title="' + esc(who) + '">' +
            (running ? '<span class="behavior-strip-running">●</span>' : '') +
            '<span class="behavior-strip-animal" style="color:' + esc(animalColor(session, lane.animal)) + '">' + esc(lane.animal) + '</span>' +
            '<span class="behavior-strip-name">' + esc(lane.name) + '</span>' +
            '<span class="behavior-swatch" style="background:' + esc(lane.color) + '"></span></div>';
    }).join('') || '<div class="behavior-strip-label is-empty">No bouts yet</div>';
    var changed = html !== _labelsHtml;
    if (changed) {
        _labelsHtml = html;
        el.labels.innerHTML = html;
        el.root.style.setProperty('--behavior-strip-lanes', String(Math.max(1, _lanes.length)));
        el.title.textContent = _lanes.length > MAX_VISIBLE_LANES
            ? _lanes.length.toLocaleString('en-US') + ' lanes · Shift+wheel scrolls' : 'Bouts';
        _rootRect = null;
    }
    if (changed || focal !== _revealed) revealFocal(focal);
}

/**
 * With more lanes than fit, keep the lanes the keys are labelling in view: the
 * focal animal's running one, else its first, unless one of its lanes already shows.
 */
function revealFocal(focal) {
    _revealed = focal;
    if (!focal || _lanes.length <= MAX_VISIBLE_LANES) return;
    var body = el.body, top = body.scrollTop, shown = body.clientHeight;
    var mine = [];
    _lanes.forEach(function (lane, i) { if (lane.animal === focal) mine.push(i); });
    if (!mine.length) return;
    var running = mine.filter(function (i) { return _lanes[i].items.some(function (it) { return it.running; }); });
    var inView = function (i) { return i * LANE_HEIGHT_PX >= top && (i + 1) * LANE_HEIGHT_PX <= top + shown; };
    if (running.length ? inView(running[0]) : mine.some(inView)) return;
    body.scrollTop = (running.length ? running[0] : mine[0]) * LANE_HEIGHT_PX;
    _rootRect = null;
    hideHover();
}

/** Draw a canvas at a size in device pixels (a resize clears it); its CSS size keeps one to one. */
function sizeCanvas(ctx, w, h, onScreen) {
    var c = ctx.canvas;
    if (c.width === w && c.height === h) return;
    c.width = w;
    c.height = h;
    ctx.imageSmoothingEnabled = false;
    if (onScreen) {
        c.style.width = px(w / _dpr);
        c.style.height = px(h / _dpr);
    }
}

function lanesHeightPx() {
    return Math.max(1, _lanes.length) * LANE_HEIGHT_PX;
}

/** Draw the film around `_frame` (keeping it where it is when it still fits). */
function paintFilm() {
    _fps = detectionFps();
    _dpr = window.devicePixelRatio || 1;
    _builtTotal = state.totalFrames;
    _builtFps = state.fps;
    var scale = stripScale(_halfSeconds, _fps, _viewWidth);
    var same = _scale && _scale.half === scale.half && _scale.width === scale.width;
    _scale = scale;
    if (!_film || !same || !filmHolds(_film, _frame, _scale)) _film = stripFilm(_frame, _scale, MAX_FILM_DEVICE_PX / _dpr);
    var u = behaviorUi();
    _armed = u.rangeIn != null ? { start: u.rangeIn, end: u.rangeOut != null ? u.rangeOut : u.rangeIn } : null;
    drawFilm();
}

/** A bar's edges on whole device pixels: the copy never blurs them. */
function fillSnapped(ctx, x, y, w, h) {
    var x0 = Math.round(x * _dpr) / _dpr;
    var x1 = Math.max(x0 + 1 / _dpr, Math.round((x + w) * _dpr) / _dpr);
    ctx.fillRect(x0, y, x1 - x0, h);
}

function drawFilm() {
    var filmW = Math.ceil(_film.width * _dpr);
    var lanesH = Math.round(lanesHeightPx() * _dpr);
    var rulerH = Math.round(RULER_HEIGHT_PX * _dpr);
    sizeCanvas(_filmCtx, filmW, lanesH, false);
    sizeCanvas(_rulerFilmCtx, filmW, rulerH, false);
    sizeCanvas(_ctx, Math.round(_viewWidth * _dpr), lanesH, true);
    sizeCanvas(_rulerCtx, Math.round(_viewWidth * _dpr), rulerH, true);

    var c = _filmCtx;
    c.setTransform(1, 0, 0, 1, 0, 0);
    c.clearRect(0, 0, filmW, lanesH);
    c.setTransform(_dpr, 0, 0, _dpr, 0, 0);
    var r = _rulerFilmCtx;
    r.setTransform(1, 0, 0, 1, 0, 0);
    r.clearRect(0, 0, filmW, rulerH);
    r.setTransform(_dpr, 0, 0, _dpr, 0, 0);

    c.fillStyle = OUTSIDE_FILL;
    r.fillStyle = OUTSIDE_FILL;
    filmOutside(_film, _scale, state.totalFrames).forEach(function (o) {
        c.fillRect(o.x, 0, o.w, lanesHeightPx());
        r.fillRect(o.x, 0, o.w, RULER_HEIGHT_PX);
    });

    // A line between animals, as between their lane labels.
    c.fillStyle = ANIMAL_LINE;
    _lanes.forEach(function (lane, i) {
        if (i && _lanes[i - 1].animal !== lane.animal) c.fillRect(0, i * LANE_HEIGHT_PX, _film.width, 1);
    });

    var withPartner = sessionAnimals(state.session).length > 2;
    var bars = filmBars(_lanes, _film, _scale, _frame);
    _running = bars.filter(function (b) { return b.kind === 'running'; });
    bars.forEach(function (b) {
        if (b.kind === 'running') return;
        var y = b.lane * LANE_HEIGHT_PX + 1, h = LANE_HEIGHT_PX - 2;
        c.fillStyle = b.kind === 'absent' ? ABSENT_FILL : b.color;
        fillSnapped(c, b.x, y, b.w, h);
        if (!withPartner || b.kind !== 'present' || !b.partner || b.w < 26) return;
        c.save();
        c.beginPath();
        c.rect(b.x, y, b.w, h);
        c.clip();
        c.fillStyle = '#111';
        c.font = PARTNER_FONT;
        c.textBaseline = 'middle';
        c.fillText(_lanes[b.lane].pair.trim() + ' ' + b.partner, b.x + 2, y + h / 2);
        c.restore();
    });

    var ticks = filmTicks(_film, _scale, _fps, state.totalFrames);
    if (!_rulerColors) {
        var css = getComputedStyle(el.root);
        _rulerColors = { minor: css.getPropertyValue('--text-muted'), major: css.getPropertyValue('--text-secondary') };
    }
    r.fillStyle = _rulerColors.minor;
    ticks.minors.forEach(function (x) { fillSnapped(r, x - 0.5, RULER_HEIGHT_PX - 3, 1, 3); });
    r.fillStyle = _rulerColors.major;
    r.font = RULER_FONT;
    r.textAlign = 'center';
    r.textBaseline = 'alphabetic';
    ticks.majors.forEach(function (m) {
        fillSnapped(r, m.x - 0.5, RULER_HEIGHT_PX - 5, 1, 5);
        r.fillText(m.label, m.x, 9);
    });
    _sx = NaN;
}

function drawRange(range, fill, edge) {
    var box = filmRect(range.start, range.end, _film, _scale, MIN_BAR_PX);
    if (!box) return;
    var h = lanesHeightPx();
    _ctx.fillStyle = fill;
    fillSnapped(_ctx, box.x, 0, box.w, h);
    _ctx.fillStyle = edge;
    fillSnapped(_ctx, box.x, 0, 1, h);
    fillSnapped(_ctx, box.x + box.w - 1, 0, 1, h);
}

// ----------------------------------------------------------------------------
// Moving (per frame: a copy, and a rectangle per thing that changes)
// ----------------------------------------------------------------------------

/**
 * The window on `frame`. With nothing changing and the film not a whole device
 * pixel further on (most frames when zoomed far out), nothing is drawn.
 */
function blit(frame, force) {
    if (!_film || !(_viewWidth > 0)) return;
    var sx = filmSourceX(frame, _film, _scale, _dpr);
    var moved = sx !== _sx;
    var extra = _running.length || _selection || _armed;
    if (!moved && !_running.length && !force) return;
    var w = _ctx.canvas.width, h = _ctx.canvas.height;
    _ctx.setTransform(1, 0, 0, 1, 0, 0);
    _ctx.clearRect(0, 0, w, h);
    _ctx.drawImage(_filmCtx.canvas, sx, 0, w, h, 0, 0, w, h);
    if (moved || force) {
        var rh = _rulerCtx.canvas.height;
        _rulerCtx.clearRect(0, 0, w, rh);
        _rulerCtx.drawImage(_rulerFilmCtx.canvas, sx, 0, w, rh, 0, 0, w, rh);
    }
    _sx = sx;
    if (!extra) return;
    _ctx.setTransform(_dpr, 0, 0, _dpr, -sx, 0);        // film coordinates
    if (_armed) drawRange(_armed, ARMED_FILL, ARMED_EDGE);
    for (var i = 0; i < _running.length; i++) {
        var b = _running[i];
        _ctx.fillStyle = b.color;
        fillSnapped(_ctx, b.x, b.lane * LANE_HEIGHT_PX + 1, runningBarWidth(b.item, frame, _film, _scale), LANE_HEIGHT_PX - 2);
    }
    if (_selection) drawRange(_selection, SELECTION_FILL, SELECTION_EDGE);
}

function place(frame, force) {
    if (!_film || !_scale) return;
    if (frame === _frame && !force) return;
    _frame = frame;
    if (!filmHolds(_film, frame, _scale)) {
        _film = stripFilm(frame, _scale, MAX_FILM_DEVICE_PX / _dpr);
        drawFilm();
    }
    blit(frame, force);
    if (_hoverShown) hideHover();     // the frame under a still pointer just changed
}

/**
 * The per-frame call (`updateBehaviorFrame`): moves the window. A comparison and
 * a return when the strip is off or the frame is the same.
 */
export function updateBehaviorStripFrame(frame) {
    if (!el) return;
    if (state.session !== _session) refreshBehaviorStrip();
    if (!_shown) return;
    if (state.totalFrames !== _builtTotal || state.fps !== _builtFps || window.devicePixelRatio !== _dpr) {
        _frame = frame;
        paintFilm();
        place(frame, true);
        return;
    }
    place(frame, false);
}

// ----------------------------------------------------------------------------
// The pointer
// ----------------------------------------------------------------------------

function rememberRects() {
    if (_rootRect) return;
    _rootRect = el.root.getBoundingClientRect();
    _bodyRect = el.body.getBoundingClientRect();
}

/** The frame under a mouse event: the view's ends for a pointer past them. */
function frameAtEvent(e) {
    rememberRects();
    var x = Math.max(0, Math.min(_viewWidth - 0.001, e.clientX - _rootRect.left - _viewLeft));
    return Math.max(0, Math.min(state.totalFrames - 1, frameAtX(x, _frame, _scale)));
}

/** The lane under a mouse event, or null over the ruler or below the lanes. */
function laneAtEvent(e) {
    rememberRects();
    var y = e.clientY - _bodyRect.top;
    if (y < 0 || y >= _bodyRect.height) return null;
    return _lanes[Math.floor((y + el.body.scrollTop) / LANE_HEIGHT_PX)] || null;
}

function hideHover() {
    _hoverShown = false;
    el.hover.hidden = true;
    el.tip.hidden = true;
}

function showTip(text, x) {
    el.tip.textContent = text;
    el.tip.hidden = false;
    if (text.length !== _tipLen) { _tipLen = text.length; _tipHalf = el.tip.offsetWidth / 2; }
    el.tip.style.left = px(Math.max(_tipHalf, Math.min(_rootRect.width - _tipHalf, x)));
}

function onHover(e) {
    // While playing the window moves under the pointer, so the tooltip would be wrong a frame later.
    if (_drag || !_scale || state.isPlaying) return;
    rememberRects();
    var inView = e.clientX >= _rootRect.left + _viewLeft && e.clientX <= _rootRect.left + _viewLeft + _viewWidth;
    if (!inView) { if (_hoverShown) hideHover(); return; }
    var frame = frameAtEvent(e);
    var lane = laneAtEvent(e);
    // A bar can be wider than its frames (kept visible when zoomed out).
    var it = lane ? boutAt(lane, frame, _frame, MIN_BAR_PX / _scale.pxPerFrame / 2) : null;
    var x = _viewLeft + xOfFrame(frame, _frame, _scale);
    el.hover.style.left = px(x);
    el.hover.hidden = false;
    _hoverShown = true;
    showTip(stripHoverText(frame, _fps, lane, it), x);
}

function endSelection() {
    _selection = null;
    if (_film) blit(_frame, true);
}

function onDragMove(e) {
    _selection = { start: Math.min(_drag.anchor, frameAtEvent(e)), end: Math.max(_drag.anchor, frameAtEvent(e)) };
    blit(_frame, true);
}

function onDragEnd(e) {
    document.removeEventListener('mousemove', onDragMove);
    document.removeEventListener('mouseup', onDragEnd);
    onDragMove(e);
    var animal = _drag.animal, end = _selection.end;
    _drag = null;
    onTimelineRangeSelect(_selection.start, end, {
        animal: animal,
        anchor: { x: _rootRect.left + _viewLeft + xOfFrame(end, _frame, _scale), top: _rootRect.top },
        onClose: endSelection,
    });
    // Without an open menu (no animals yet) it only reported the range.
    if (!document.getElementById('behaviorRangeMenu')) endSelection();
}

function onClickEnd(e) {
    document.removeEventListener('mouseup', onClickEnd);
    if (!_scale) return;
    rememberRects();
    var inView = e.clientX >= _rootRect.left + _viewLeft && e.clientX <= _rootRect.left + _viewLeft + _viewWidth &&
        e.clientY >= _rootRect.top && e.clientY <= _rootRect.bottom;
    if (inView && _deps.seek) _deps.seek(frameAtEvent(e));
}

function onViewDown(e) {
    if (e.button !== 0 || !_scale) return;
    e.preventDefault();
    _rootRect = null;
    rememberRects();
    hideHover();
    if (!e.shiftKey) { document.addEventListener('mouseup', onClickEnd); return; }
    // The window scrolls with the playhead, so a range can only be dragged out on a still one.
    if (_deps.pause) _deps.pause();
    var lane = laneAtEvent(e);
    _drag = { anchor: frameAtEvent(e), animal: lane ? lane.animal : null };
    onDragMove(e);
    document.addEventListener('mousemove', onDragMove);
    document.addEventListener('mouseup', onDragEnd);
}

function onWheel(e) {
    if (!e.ctrlKey && e.shiftKey) {
        var before = el.body.scrollTop;
        // Shift turns a mouse wheel's deltaY into deltaX on some systems.
        el.body.scrollTop += e.deltaY || e.deltaX;
        if (el.body.scrollTop !== before) {
            e.preventDefault();
            hideHover();                    // another lane is under the pointer now
        }
        return;
    }
    e.preventDefault();
    if (e.timeStamp - _wheelAt > 300) _wheelSum = 0;
    _wheelAt = e.timeStamp;
    _wheelSum += e.deltaY;
    if (Math.abs(_wheelSum) < WHEEL_STEP) return;
    var dir = _wheelSum < 0 ? 1 : -1;       // up or pinch out: zoom in, as on the Timeline
    _wheelSum = 0;
    setWindow(zoomedHalfSeconds(_halfSeconds, dir));
}

// ----------------------------------------------------------------------------
// Setup
// ----------------------------------------------------------------------------

/**
 * Wire the strip once (`setupBehaviorUi`).
 * @param {{seek: function(number), pause: function()}} deps the seekbar's way
 *   of seeking (keeps the video playing or paused), and pausing playback
 */
export function setupBehaviorStrip(deps) {
    var byId = function (id) { return document.getElementById(id); };
    var root = byId('behaviorStrip');
    if (!root) return;
    _deps = deps || {};
    el = {
        root: root, title: byId('behaviorStripTitle'), body: byId('behaviorStripBody'), labels: byId('behaviorStripLabels'),
        hover: byId('behaviorStripHover'), tip: byId('behaviorStripTooltip'),
        window: byId('behaviorStripWindow'), zoomIn: byId('behaviorStripZoomIn'), zoomOut: byId('behaviorStripZoomOut'),
    };
    _ctx = byId('behaviorStripCanvas').getContext('2d');
    _rulerCtx = byId('behaviorStripRulerCanvas').getContext('2d');
    readChoices();
    showWindow();
    root.style.setProperty('--behavior-strip-lane-h', LANE_HEIGHT_PX + 'px');
    root.style.setProperty('--behavior-strip-ruler-h', RULER_HEIGHT_PX + 'px');
    root.style.setProperty('--behavior-strip-lanes-max', String(MAX_VISIBLE_LANES));
    var info = byId('behaviorStripInfo');
    if (info) { info.innerHTML = ICON_INFO; setInfoTip(info, INFO); }
    el.zoomIn.addEventListener('click', function () { setWindow(zoomedHalfSeconds(_halfSeconds, 1)); });
    el.zoomOut.addEventListener('click', function () { setWindow(zoomedHalfSeconds(_halfSeconds, -1)); });
    byId('behaviorStripClose').addEventListener('click', function () {
        setBehaviorStripOn(false);
        setStatus('Bouts strip hidden: "Bouts strip" in the Behavior tab brings it back');
    });
    Array.prototype.forEach.call(root.querySelectorAll('.behavior-strip-view'), function (view) {
        view.addEventListener('mousedown', onViewDown);
    });
    root.addEventListener('mousemove', onHover);
    root.addEventListener('mouseleave', function () { hideHover(); _rootRect = null; });
    root.addEventListener('wheel', onWheel, { passive: false });
    if (typeof ResizeObserver === 'function') new ResizeObserver(measure).observe(byId('seekbar'));
    measure();
    refreshBehaviorStrip();
}

registerBehaviorRepaint(function (info) {
    if (info.labels || info.ui || info.ethogram) refreshBehaviorStrip();
});
