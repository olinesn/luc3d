/**
 * ui/model-check-highlight.js — marks the selected Model Check entry in its
 * camera's view: a box around the hand-labelled animal the model misread, a ring
 * on each misread node, and, where the node was read on another labelled node, a
 * dashed line to that node (and a thin box around the neighbour it belongs to).
 *
 * Drawn on its OWN canvas in the view's `.canvas-wrapper` (`.model-check-canvas`),
 * like ui/id-switch-highlight.js: the wrapper's CSS transform carries zoom / pan /
 * rotation, the overlay redraws never clear it, and the overlay-video export does
 * not include it. Static, so it repaints only when something it depends on
 * changes: the frame, the canvas size (zoom), or the instances it was drawn from
 * (a lazy frame arriving, a node dragged). A requestAnimationFrame loop runs only
 * while the view is on the entry's frame, and does nothing on a tick where none of
 * those changed.
 *
 * Driven by ui/model-check-panel.js: `setModelCheckHighlight(target)` on
 * selection, `updateModelCheckHighlight(frame)` on every frame change (from
 * ui/ui-wiring.js `updateSeekbarVisual`, beside the ID Switches bar).
 */

import { state } from './app-state.js';
import { makeVideoToCanvasTransform } from './overlays.js';

var _target = null;     // {session, camera, frame, label, locate: () => {target, nodes, others} | null}
var _raf = 0;
var _canvases = new WeakMap();
var _drawn = null;      // {canvas, sig} — what is on screen
var _lastSig = '';

/** Colours: the misread animal (orange), its misread nodes (red), the neighbours they were read on (grey). */
export var MODEL_CHECK_COLORS = Object.freeze({ animal: '255, 140, 40', node: '255, 70, 60', other: '220, 220, 220' });

/**
 * Select what to mark, or `null` to stop.
 * @param {?{session, camera: string, frame: number, label: string,
 *   locate: () => ?{target: Instance, nodes: Array<{k: number, name: string, ref?: {inst: Instance, j: number}}>,
 *   others: Array<{inst: Instance, label: string}>}}} target
 */
export function setModelCheckHighlight(target) {
    _target = target || null;
    _lastSig = '';
    clear();
    updateModelCheckHighlight(state.currentFrame);
}

/** @returns {?{camera, frame, label}} (tests) */
export function getModelCheckHighlight() {
    return _target ? { camera: _target.camera, frame: _target.frame, label: _target.label, drawn: !!_drawn } : null;
}

/** Every frame change: start the loop on the entry's frame, clear it elsewhere. */
export function updateModelCheckHighlight(frame) {
    if (!_target || frame !== _target.frame || state.session !== _target.session) { stopLoop(); clear(); return; }
    if (!_raf && typeof requestAnimationFrame === 'function') _raf = requestAnimationFrame(tick);
}

function stopLoop() {
    if (_raf && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(_raf);
    _raf = 0;
}

function clear() {
    if (_drawn) {
        var c = _drawn.canvas;
        c.getContext('2d').clearRect(0, 0, c.width, c.height);
        _drawn = null;
    }
    _lastSig = '';
}

function viewOf(camera) {
    var views = state.views || [];
    for (var i = 0; i < views.length; i++) if ((views[i].cameraName || views[i].name) === camera) return views[i];
    return null;
}

function canvasFor(v) {
    if (!v.wrapper || typeof document === 'undefined') return null;
    var c = _canvases.get(v);
    if (!c || c.parentNode !== v.wrapper) {
        c = document.createElement('canvas');
        c.className = 'model-check-canvas';
        v.wrapper.appendChild(c);
        _canvases.set(v, c);
    }
    var zs = v.zoom && v.zoom.scale ? v.zoom.scale : 1;
    var w = Math.max(1, Math.round((v.videoWidth || 1) * zs)), h = Math.max(1, Math.round((v.videoHeight || 1) * zs));
    if (c.width !== w) c.width = w;
    if (c.height !== h) c.height = h;
    return c;
}

function pt(inst, k) {
    if (!inst || !(inst.hasPoint ? inst.hasPoint(k) : true)) return null;
    var x = inst.getX(k), y = inst.getY(k);
    return isFinite(x) && isFinite(y) ? { x: x, y: y } : null;
}

function bbox(inst) {
    var b = { x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity };
    for (var k = 0; k < (inst.numNodes || 0); k++) {
        var p = pt(inst, k);
        if (!p) continue;
        if (p.x < b.x0) b.x0 = p.x; if (p.y < b.y0) b.y0 = p.y;
        if (p.x > b.x1) b.x1 = p.x; if (p.y > b.y1) b.y1 = p.y;
    }
    return b.x1 >= b.x0 ? b : null;
}

function tick() {
    _raf = 0;
    if (!_target || state.currentFrame !== _target.frame || state.session !== _target.session) { clear(); return; }
    var v = viewOf(_target.camera);
    var c = v && canvasFor(v);
    var loc = c ? _target.locate() : null;
    // What the picture depends on; nothing to do when it has not changed.
    var sig = c ? [c.width, c.height, c.offsetWidth, v.zoom && v.zoom.scale].join(',') : '';
    if (loc && loc.target) {
        var b = bbox(loc.target);
        sig += '|' + (b ? [b.x0, b.y0, b.x1, b.y1].join(',') : '') + '|' + loc.nodes.map(function (n) {
            var p = pt(loc.target, n.k), q = n.ref ? pt(n.ref.inst, n.ref.j) : null;
            return (p ? p.x + ',' + p.y : '') + (q ? '>' + q.x + ',' + q.y : '');
        }).join(';') + '|' + loc.others.map(function (o) { var ob = bbox(o.inst); return ob ? ob.x0 + ',' + ob.y0 : ''; }).join(';');
    }
    if (sig !== _lastSig || !_drawn || _drawn.canvas !== c) {
        clear();
        if (c && loc && loc.target) paint(v, c, loc);
        _lastSig = sig;
    }
    if (typeof requestAnimationFrame === 'function') _raf = requestAnimationFrame(tick);
}

function label(ctx, text, x, y, fs, s, rgb) {
    ctx.font = '600 ' + fs + 'px system-ui, sans-serif';
    ctx.textBaseline = 'alphabetic';
    var tw = ctx.measureText(text).width;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.65)';
    ctx.fillRect(x, y - fs, tw + 8 * s, fs + 5 * s);
    ctx.fillStyle = 'rgb(' + rgb + ')';
    ctx.fillText(text, x + 4 * s, y);
}

function paint(v, c, loc) {
    var ctx = c.getContext('2d');
    var toC = makeVideoToCanvasTransform(v.videoWidth, v.videoHeight, c.width, c.height);
    var zs = v.zoom && v.zoom.scale ? v.zoom.scale : 1;
    // Lines and text in SCREEN pixels, as ui/id-switch-highlight.js does.
    var s = c.offsetWidth > 0 ? c.width / (c.offsetWidth * zs) : (toC.scale || 1), pad = 8 * s, fs = Math.round(12 * s);
    var box = function (inst, rgb, dash, width, text) {
        var b = bbox(inst);
        if (!b) return;
        var p0 = toC(b.x0, b.y0), p1 = toC(b.x1, b.y1);
        var x = p0.x - pad, y = p0.y - pad, w = p1.x - p0.x + 2 * pad, h = p1.y - p0.y + 2 * pad;
        ctx.setLineDash([]); ctx.lineWidth = width + 2 * s; ctx.strokeStyle = 'rgba(0, 0, 0, 0.55)'; ctx.strokeRect(x, y, w, h);
        ctx.setLineDash(dash); ctx.lineWidth = width; ctx.strokeStyle = 'rgb(' + rgb + ')'; ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);
        if (text) label(ctx, text, x, Math.max(fs + 2 * s, y - 5 * s), fs, s, rgb);
    };
    ctx.save();
    loc.others.forEach(function (o) { box(o.inst, MODEL_CHECK_COLORS.other, [6 * s, 5 * s], 1.5 * s, o.label); });
    box(loc.target, MODEL_CHECK_COLORS.animal, [], 2.5 * s, _target.label);
    loc.nodes.forEach(function (n) {
        var p = pt(loc.target, n.k);
        if (!p) return;
        var cp = toC(p.x, p.y), r = 7 * s;
        var q = n.ref ? pt(n.ref.inst, n.ref.j) : null;
        if (q && (q.x !== p.x || q.y !== p.y)) {
            var cq = toC(q.x, q.y);
            ctx.setLineDash([4 * s, 3 * s]); ctx.lineWidth = 1.5 * s; ctx.strokeStyle = 'rgb(' + MODEL_CHECK_COLORS.node + ')';
            ctx.beginPath(); ctx.moveTo(cp.x, cp.y); ctx.lineTo(cq.x, cq.y); ctx.stroke();
            ctx.setLineDash([]); ctx.beginPath(); ctx.arc(cq.x, cq.y, 4 * s, 0, 2 * Math.PI); ctx.stroke();
        }
        ctx.setLineDash([]);
        ctx.lineWidth = 4 * s; ctx.strokeStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.beginPath(); ctx.arc(cp.x, cp.y, r, 0, 2 * Math.PI); ctx.stroke();
        ctx.lineWidth = 2 * s; ctx.strokeStyle = 'rgb(' + MODEL_CHECK_COLORS.node + ')';
        ctx.beginPath(); ctx.arc(cp.x, cp.y, r, 0, 2 * Math.PI); ctx.stroke();
        label(ctx, n.name, cp.x + r + 2 * s, cp.y + fs / 2, Math.round(fs * 0.9), s, MODEL_CHECK_COLORS.node);
    });
    ctx.restore();
    _drawn = { canvas: c };
}
