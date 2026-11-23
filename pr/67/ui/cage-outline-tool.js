// ui/cage-outline-tool.js — trace and edit the cage's outline in each camera
// view (Define Planes ▸ Outlines), and draw it on the 2D overlays.
//
// The data is `pose/cage-outlines.js` (a `CageOutlineSet` on
// `Session.cageOutlines`, saved by `import-export/plane-metadata.js`). This
// module owns the gestures, the drawing and the panel section:
//
//   * Trace / Retrace on a view's row starts a trace on that view: each click
//     adds a vertex; clicking the first vertex (once there are 3) or Enter
//     closes it; Backspace (or a right-click) removes the last one; Esc
//     cancels. Closing REPLACES that view's outline.
//   * On a closed outline, inside Defining Plane Mode: drag a vertex to move
//     it, Shift+click an edge to insert one there, right-click a vertex to
//     delete it (an outline keeps at least 3).
//
// The pointer gestures arrive through two InteractionManager callbacks
// (`outlinePointerDown` / `outlinePointerMove`, merged in by
// `planeInteractionCallbacks`), which run before the plane-node and pose
// handling, so `ui/interaction.js` keeps no import of this feature. While a
// view is being traced, every click on it is a vertex (plane nodes there are
// not grabbable); elsewhere a plane node under the cursor wins over an outline
// vertex at the same spot, so a cage corner that is both stays draggable as a
// node.
//
// Circular imports (`plane-definition.js` imports this module back) are safe:
// every use is inside a function body.

import { state, interactionManager } from './app-state.js?v=e4b63c0b274f';
import { makeVideoToCanvasTransform } from './overlays.js?v=e4b63c0b274f';
import { overlayDrawSize } from './overlay-window.js?v=e4b63c0b274f';
import {
    CageOutlineSet, OutlineTrace, nearestOutlineVertex, nearestOutlineEdge,
} from '../pose/cage-outlines.js?v=e4b63c0b274f';
import { planeState, redraw } from './plane-definition.js?v=e4b63c0b274f';
import { planeVisibility } from './plane-visibility.js?v=e4b63c0b274f';
import { setStatus, markDirty } from '../import-export/save-load.js?v=e4b63c0b274f';
import { isTextEntryTarget } from './keyboard-target.js?v=e4b63c0b274f';

/** Outline colour: orange, apart from the plane palette's blues and greens. */
export const OUTLINE_COLOR = '#ffa726';
const OUTLINE_FILL = 'rgba(255,167,38,0.12)';
/** Hit radius for outline vertices and edges, in screen px. */
const HIT_PX = 7;
const VERTEX_PX = 3.5;

/** What the tool is doing right now. Transient; nothing here is saved. */
export const outlineState = {
    /** @type {OutlineTrace|null} the outline being traced (with `.session`) */
    trace: null,
    /** @type {number[]|null} cursor on the traced view, video px (rubber band) */
    cursor: null,
    /** @type {{viewName:string, polyId:number, index:number, moved:boolean}|null} */
    drag: null,
    /** @type {{viewName:string, polyId:number, index:number}|null} */
    hover: null,
};

let _keyHandler = null;
let _dragMove = null;
let _dragUp = null;

/** The active session's outline set, created on first use. */
export function sessionOutlines(session) {
    var s = session || state.session;
    if (!s) return null;
    if (!(s.cageOutlines instanceof CageOutlineSet)) s.cageOutlines = new CageOutlineSet();
    return s.cageOutlines;
}

function viewByName(name) {
    var views = state.views || [];
    for (var i = 0; i < views.length; i++) if (views[i].name === name) return views[i];
    return null;
}

function clampToView(viewName, x, y) {
    var v = viewByName(viewName);
    var w = v && v.videoWidth, h = v && v.videoHeight;
    if (w > 0) x = Math.max(0, Math.min(w, x));
    if (h > 0) y = Math.max(0, Math.min(h, y));
    return [x, y];
}

function fmt(n) { return Number(n).toLocaleString('en-US'); }

/** The overlay's drawing size (what `drawPlaneOverlays` maps onto): the full
 *  virtual canvas, which a windowed overlay canvas is drawn into through a
 *  transform (`ui/overlay-window.js`). */
function overlaySize(view) {
    return overlayDrawSize(view);
}

/** Is a trace live on the active session? Drops a stale one (session switched). */
function liveTrace() {
    var t = outlineState.trace;
    if (t && t.session !== state.session) { cancelOutlineTrace(true); return null; }
    return t;
}

// ============================================
// Tracing
// ============================================

/** Start tracing `viewName`'s outline (replacing any trace in progress). */
export function startOutlineTrace(viewName) {
    if (!state.session || !viewByName(viewName)) return false;
    if (outlineState.trace) cancelOutlineTrace(true);
    var t = new OutlineTrace(viewName);
    t.session = state.session;
    outlineState.trace = t;
    outlineState.cursor = null;
    _keyHandler = onTraceKeyDown;
    window.addEventListener('keydown', _keyHandler, true);
    setStatus('Tracing the outline on ' + viewName + ': click around the cage, then click the ' +
        'first point or press Enter. Backspace undoes, Esc cancels.', 'warning');
    renderOutlinesSection();
    redraw();
    return true;
}

/** Stop tracing without changing the saved outline. */
export function cancelOutlineTrace(silent) {
    if (_keyHandler) { window.removeEventListener('keydown', _keyHandler, true); _keyHandler = null; }
    var had = !!outlineState.trace;
    outlineState.trace = null;
    outlineState.cursor = null;
    if (had && !silent) setStatus('Outline tracing cancelled', 'warning');
    renderOutlinesSection();
    if (had) redraw();
}

/** Close the trace: it becomes the view's outline. */
export function closeOutlineTrace() {
    var t = outlineState.trace;
    if (!t) return false;
    if (!t.canClose()) {
        setStatus('An outline needs at least 3 points (' + t.points.length + ' so far)', 'warning');
        return false;
    }
    var set = sessionOutlines(t.session);
    var poly = set && set.replaceView(t.viewName, t.points);
    if (_keyHandler) { window.removeEventListener('keydown', _keyHandler, true); _keyHandler = null; }
    outlineState.trace = null;
    outlineState.cursor = null;
    if (!poly) {
        setStatus('Could not save the outline on ' + t.viewName, 'error');
    } else {
        markDirty();
        setStatus('Outline on ' + t.viewName + ': ' + fmt(poly.points.length) + ' points', 'success');
    }
    renderOutlinesSection();
    redraw();
    return !!poly;
}

/** Remove a view's outline. */
export function clearViewOutline(viewName) {
    var t = liveTrace();
    if (t && t.viewName === viewName) cancelOutlineTrace(true);
    var set = sessionOutlines();
    if (set && set.clearView(viewName)) {
        markDirty();
        setStatus('Cleared the outline on ' + viewName, 'success');
    }
    renderOutlinesSection();
    redraw();
}

function onTraceKeyDown(e) {
    var t = liveTrace();
    if (!t) return;
    // Typing in a field (a node name, say) keeps its own Enter and Backspace.
    if (e.key !== 'Escape' && isTextEntryTarget(e.target)) return;
    if (e.key === 'Enter') {
        closeOutlineTrace();
    } else if (e.key === 'Backspace') {
        t.undo();
        renderOutlinesSection();
        redraw();
    } else if (e.key === 'Escape') {
        cancelOutlineTrace();
    } else {
        return;
    }
    e.preventDefault();
    e.stopImmediatePropagation();
}

// ============================================
// Pointer hooks (InteractionManager callbacks)
// ============================================

/**
 * Mousedown on a view. Returns true when the outline tool took the click.
 * @param {MouseEvent} e
 * @param {string} viewName
 * @param {number} vx - video px
 * @param {number} vy
 * @param {number} displayToVideo - video px per screen px
 * @param {Object} [im] - the InteractionManager (for the plane-node priority)
 */
export function outlinePointerDown(e, viewName, vx, vy, displayToVideo, im) {
    if (!planeState.active || !state.session) return false;
    var r = HIT_PX * (displayToVideo || 1);
    var rightish = e.button === 2 || (e.button === 0 && e.ctrlKey);
    var t = liveTrace();
    if (t) {
        if (t.viewName !== viewName) return false;
        if (rightish) { t.undo(); renderOutlinesSection(); redraw(); return true; }
        if (e.button !== 0) return false;
        if (t.canClose() && t.nearFirst(vx, vy, r)) { closeOutlineTrace(); return true; }
        var p = clampToView(viewName, vx, vy);
        t.add(p[0], p[1]);
        renderOutlinesSection();
        redraw();
        return true;
    }

    var set = state.session.cageOutlines;
    if (!set || !set.polygons(viewName).length) return false;
    var hv = nearestOutlineVertex(set, viewName, vx, vy, r);
    if (hv && im && im.findNearestPlaneNode) {
        // A plane node at the same spot wins: a cage corner is often both.
        // "Same spot" is within a screen pixel, not exact equality (the two
        // hit tests compute distance differently, so a true tie can differ in
        // the last bit).
        var pn = im.findNearestPlaneNode(vx, vy, viewName);
        if (pn && pn.nodeIdx >= 0 && pn.distance <= hv.distance + (displayToVideo || 1)) hv = null;
    }
    if (rightish) {
        if (!hv) return false;
        if (set.deleteVertex(viewName, hv.polyId, hv.index)) {
            markDirty();
            outlineState.hover = null;
        } else {
            setStatus('An outline keeps at least 3 points: use Clear in the Outlines section to remove it', 'warning');
        }
        renderOutlinesSection();
        redraw();
        return true;
    }
    if (e.button !== 0) return false;
    if (hv) { beginVertexDrag(viewName, hv.polyId, hv.index); return true; }
    if (e.shiftKey) {
        var he = nearestOutlineEdge(set, viewName, vx, vy, r);
        if (he && set.insertVertex(viewName, he.polyId, he.index, he.point[0], he.point[1])) {
            markDirty();
            renderOutlinesSection();
            beginVertexDrag(viewName, he.polyId, he.index + 1);
            redraw();
            return true;
        }
    }
    return false;
}

/**
 * Mousemove on a view. Returns true when the tool consumed it (tracing this
 * view), 'hover' when the cursor is over an outline vertex (so the manager
 * shows the grab cursor), else false.
 */
export function outlinePointerMove(e, viewName, vx, vy, displayToVideo) {
    if (!planeState.active || !state.session) return false;
    var t = liveTrace();
    if (t && t.viewName === viewName) {
        outlineState.cursor = clampToView(viewName, vx, vy);
        var v = viewByName(viewName);
        if (v && v.overlayCanvas) v.overlayCanvas.style.cursor = 'crosshair';
        redraw();
        return true;
    }
    if (outlineState.drag) return false;
    var set = state.session.cageOutlines;
    var prev = outlineState.hover;
    var hv = (set && set.polygons(viewName).length)
        ? nearestOutlineVertex(set, viewName, vx, vy, HIT_PX * (displayToVideo || 1)) : null;
    outlineState.hover = hv ? { viewName: viewName, polyId: hv.polyId, index: hv.index } : null;
    var changed = !prev !== !outlineState.hover || (prev && outlineState.hover &&
        (prev.viewName !== outlineState.hover.viewName || prev.polyId !== outlineState.hover.polyId ||
         prev.index !== outlineState.hover.index));
    if (changed) redraw();
    return hv ? 'hover' : false;
}

function beginVertexDrag(viewName, polyId, index) {
    endVertexDrag();
    outlineState.drag = { viewName: viewName, polyId: polyId, index: index, moved: false };
    _dragMove = function (ev) {
        var d = outlineState.drag;
        if (!d || !interactionManager) return;
        var c = interactionManager.canvasToVideo(ev.clientX, ev.clientY, d.viewName);
        var p = clampToView(d.viewName, c[0], c[1]);
        var set = sessionOutlines();
        if (set && set.moveVertex(d.viewName, d.polyId, d.index, p[0], p[1])) {
            d.moved = true;
            redraw();
        }
    };
    _dragUp = function () {
        var d = outlineState.drag;
        endVertexDrag();
        if (d && d.moved) markDirty();
        renderOutlinesSection();
        redraw();
    };
    document.addEventListener('mousemove', _dragMove, true);
    document.addEventListener('mouseup', _dragUp, true);
    // What the InteractionManager sets for its own drags: the video cell's
    // pan / box-zoom (loading/video.js) stands down while it is set.
    window.__mvguiDragging = true;
}

function endVertexDrag() {
    if (_dragMove) document.removeEventListener('mousemove', _dragMove, true);
    if (_dragUp) document.removeEventListener('mouseup', _dragUp, true);
    if (_dragMove) window.__mvguiDragging = false;
    _dragMove = null;
    _dragUp = null;
    outlineState.drag = null;
}

// ============================================
// Drawing
// ============================================

/**
 * Draw a view's outline (and the trace in progress) on its overlay. Called
 * from `drawAllOverlays` right after `drawPlaneOverlays`, on the same canvas
 * and with the same video-to-canvas mapping. Shown with the plane bodies: always inside Defining
 * Plane Mode, outside it when Visibility ▸ Planes in 2D Views is on.
 * Vertices (the editing handles) are drawn inside the mode only.
 */
export function drawCageOutlines(view) {
    if (!view || !view.overlayCtx || !view.overlayCanvas) return;
    var set = state.session ? state.session.cageOutlines : null;
    var polys = set ? set.polygons(view.name) : [];
    var t = outlineState.trace;
    var tracing = !!(t && t.session === state.session && t.viewName === view.name);
    if (!polys.length && !tracing) return;
    var vis = planeVisibility(planeState.active);
    if (!vis.planes2d && !tracing) return;

    var drawSize = overlaySize(view);
    var videoW = view.videoWidth || drawSize.width;
    var videoH = view.videoHeight || drawSize.height;
    if (!videoW || !videoH) return;
    var tf = makeVideoToCanvasTransform(videoW, videoH, drawSize.width, drawSize.height);
    var ctx = view.overlayCtx;
    var edit = planeState.active;
    var hover = outlineState.hover;

    ctx.save();
    ctx.lineJoin = 'round';
    for (var p = 0; p < polys.length; p++) {
        var pts = polys[p].points.map(function (q) { return tf(q[0], q[1]); });
        if (pts.length < 3) continue;
        ctx.beginPath();
        ctx.moveTo(pts[0].x, pts[0].y);
        for (var i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
        ctx.closePath();
        // A retrace in progress shows the old outline faintly, so the new one reads as the edit.
        ctx.globalAlpha = tracing ? 0.35 : 1;
        ctx.fillStyle = OUTLINE_FILL;
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = OUTLINE_COLOR;
        ctx.stroke();
        if (edit && !tracing) {
            for (var k = 0; k < pts.length; k++) {
                var hot = !!(hover && hover.viewName === view.name && hover.polyId === polys[p].id &&
                    hover.index === k);
                drawVertex(ctx, pts[k], hot ? VERTEX_PX + 2 : VERTEX_PX, hot);
            }
        }
        ctx.globalAlpha = 1;
    }

    if (tracing) {
        var tp = t.points.map(function (q) { return tf(q[0], q[1]); });
        var cur = outlineState.cursor ? tf(outlineState.cursor[0], outlineState.cursor[1]) : null;
        if (tp.length) {
            ctx.beginPath();
            ctx.moveTo(tp[0].x, tp[0].y);
            for (var j = 1; j < tp.length; j++) ctx.lineTo(tp[j].x, tp[j].y);
            ctx.lineWidth = 2;
            ctx.strokeStyle = OUTLINE_COLOR;
            ctx.stroke();
            if (cur) {
                ctx.setLineDash([5, 4]);
                ctx.beginPath();
                ctx.moveTo(tp[tp.length - 1].x, tp[tp.length - 1].y);
                ctx.lineTo(cur.x, cur.y);
                if (t.canClose()) ctx.lineTo(tp[0].x, tp[0].y);
                ctx.lineWidth = 1.5;
                ctx.stroke();
                ctx.setLineDash([]);
            }
            for (var m = 0; m < tp.length; m++) drawVertex(ctx, tp[m], VERTEX_PX, false);
            if (t.canClose()) {
                // The first point is the close target.
                ctx.beginPath();
                ctx.arc(tp[0].x, tp[0].y, HIT_PX, 0, Math.PI * 2);
                ctx.lineWidth = 1.5;
                ctx.strokeStyle = '#ffffff';
                ctx.stroke();
            }
        }
    }
    ctx.restore();
}

function drawVertex(ctx, pt, r, hot) {
    ctx.beginPath();
    ctx.rect(pt.x - r, pt.y - r, 2 * r, 2 * r);
    ctx.fillStyle = hot ? '#ffffff' : OUTLINE_COLOR;
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = 'rgba(0,0,0,0.8)';
    ctx.stroke();
}

// ============================================
// Panel section (Define Planes ▸ Outlines)
// ============================================

/** One sentence each for the section's ⓘ. */
export const OUTLINES_INFO =
    'Trace the cage\'s outline (the whole area it covers) in each camera view. ' +
    'The outlines are saved with the project, for building the cage\'s 3D shape from its silhouettes.';

function smallButton(label, title, onClick, disabled) {
    var b = document.createElement('button');
    b.type = 'button';
    b.className = 'panel-btn outline-row-btn';
    b.textContent = label;
    b.title = title;
    b.disabled = !!disabled;
    b.addEventListener('click', function (e) { e.stopPropagation(); onClick(); });
    return b;
}

/** Rebuild the Outlines table from the active session's views and outlines. */
export function renderOutlinesSection() {
    var tbody = document.querySelector('#cageOutlinesTable tbody');
    if (!tbody) return;
    var table = document.getElementById('cageOutlinesTable');
    var empty = document.getElementById('cageOutlinesEmpty');
    var hint = document.getElementById('cageOutlinesHint');
    var views = state.views || [];
    var set = state.session ? sessionOutlines() : null;
    var t = outlineState.trace && outlineState.trace.session === state.session ? outlineState.trace : null;
    tbody.innerHTML = '';
    if (table) table.style.display = views.length ? '' : 'none';
    if (empty) empty.style.display = views.length ? 'none' : '';
    for (var i = 0; i < views.length; i++) {
        (function (viewName) {
            var tr = document.createElement('tr');
            tr.dataset.view = viewName;
            var nameTd = document.createElement('td');
            nameTd.className = 'outline-view-name';
            nameTd.textContent = viewName;
            nameTd.title = viewName;
            var countTd = document.createElement('td');
            countTd.className = 'outline-count';
            var n = set ? set.vertexCount(viewName) : 0;
            var tracingThis = !!(t && t.viewName === viewName);
            countTd.textContent = tracingThis ? fmt(t.points.length) + '…' : (n ? fmt(n) : '—');
            var actTd = document.createElement('td');
            var acts = document.createElement('div');
            acts.className = 'outline-actions';
            actTd.appendChild(acts);
            if (tracingThis) {
                acts.appendChild(smallButton('Close', 'Close the outline (Enter)', closeOutlineTrace,
                    !t.canClose()));
                acts.appendChild(smallButton('Cancel', 'Stop tracing, keep the saved outline (Esc)',
                    function () { cancelOutlineTrace(); }));
            } else {
                acts.appendChild(smallButton(n ? 'Retrace' : 'Trace',
                    n ? 'Trace this view\'s outline again (replaces it)' : 'Trace the cage\'s outline in this view',
                    function () { startOutlineTrace(viewName); }));
                acts.appendChild(smallButton('Clear', 'Remove this view\'s outline',
                    function () { clearViewOutline(viewName); }, !n));
            }
            tr.appendChild(nameTd);
            tr.appendChild(countTd);
            tr.appendChild(actTd);
            tbody.appendChild(tr);
        })(views[i].name);
    }
    if (hint) {
        if (t) {
            hint.textContent = 'Click around the cage on ' + t.viewName +
                '; click the first point or press Enter to close.';
        } else if (set && !set.isEmpty()) {
            hint.textContent = 'Drag a point to move it, Shift+click an edge to add one, right-click to delete.';
        } else {
            hint.textContent = '';
        }
        hint.style.display = hint.textContent ? '' : 'none';
    }
}
