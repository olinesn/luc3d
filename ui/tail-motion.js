// ui/tail-motion.js — Behavior ▸ Tail Motion from Video…: measure how fast each
// animal's tail moves in one camera's video (pose/tail-motion.js has the math),
// so the behavior classifiers can use it as five more channels per animal
// (pose/behavior-features.js `spec.channels`; attached to every behavior
// context by ui/behavior-detect.js through `withTailMotionChannels`).
//
// The poses come from the session's behavior input (ui/behavior-detect.js):
// identified 3D projected into the chosen camera (calibration, lens distortion
// included), or, without 3D, that camera's own 2D. The frames come from the
// view's frame-accurate decoder (mediabunny), streamed once in order
// (`samplesAtTimestamps` over every frame, as the image ID-switch check
// streams its samples), and each frame's luma is read from its Y plane
// (`VideoFrame.copyTo`; RGB frames are converted).
//
// The result is kept per session in memory (pose/tail-motion.js store) and is
// never saved: compute it again after reopening. Training or predicting with
// a classifier that learned tail motion, on a session without it, says so
// (pose/behavior-classifier.js `modelProblems`).
//
// Depends on: ui/app-state.js, ui/behavior-detect.js, ui/loading-overlay.js,
// ui/info-tip.js, ui/plane-definition.js (ICON_INFO), import-export/save-load.js
// (setStatus), pose/tail-motion.js,
// pose/triangulation-core.js (reprojectPointCamera, cameraDepth).

import { state } from './app-state.js';
import { cachedBehaviorContext } from './behavior-detect.js';
import { showLoadingProgress, hideLoading, yieldToPaint } from './loading-overlay.js';
import { setInfoTip } from './info-tip.js';
import { ICON_INFO } from './plane-definition.js';
import { setStatus } from '../import-export/save-load.js';
import { reprojectPointCamera, cameraDepth } from '../pose/triangulation-core.js';
import {
    tailMotionNodes, cropSide, createTailMotionPass, medianOf, setTailMotion, tailMotionOf, clearTailMotion,
    TAIL_MOTION_CHANNELS, TAIL_MOTION_CROP,
} from '../pose/tail-motion.js';

var PLANAR_Y = ['I420', 'I420A', 'I422', 'I422A', 'I444', 'I444A', 'NV12'];
var RGB_ORDER = { RGBA: [0, 1, 2], RGBX: [0, 1, 2], BGRA: [2, 1, 0], BGRX: [2, 1, 0] };
var WORK_MS = 40;        // main-thread work between paints
var PROGRESS_MS = 250;

export var TAIL_MOTION_INFO = 'Measures how fast each animal\'s tail moves against its body, in a band along the tail in one camera\'s video. ' +
    'The classifiers then get five more measures per animal (tail motion and flicker, body motion and flicker, tail over body): train them again to use them.';

function n(x) { return Number(x || 0).toLocaleString('en-US'); }

function cannot(msg) {
    var e = new Error(msg);
    e.userFacing = true;
    return e;
}

function fin(v) { return v === v && v !== Infinity && v !== -Infinity; }

/** "1 min 52 s" / "8 s". */
export function formatDuration(seconds) {
    var s = Math.max(0, Math.round(seconds));
    if (s < 60) return s + ' s';
    var m = Math.floor(s / 60);
    return m + ' min' + (s % 60 ? ' ' + (s % 60) + ' s' : '');
}

// ----------------------------------------------------------------------------
// Cameras
// ----------------------------------------------------------------------------

/**
 * The session's cameras with a loaded video: `{name, view, camera, tilt, streams}`
 * — `camera` the calibration's (null without one), `tilt` the angle in degrees
 * between its optical axis and the floor's normal (NaN without a camera or a
 * floor), `streams` whether its decoder can stream every frame (mediabunny).
 * Sorted as the views are.
 */
export function tailMotionCameras(session, ctx) {
    var views = (state.views || []).filter(function (v) { return v && v.decoder; });
    var cams = (session && session.cameras) || [];
    var up = ctx && ctx.floorInfo && ctx.floorInfo.floor ? ctx.floorInfo.floor.up : null;
    return views.map(function (v) {
        var name = v.cameraName || v.name;
        var camera = cams.find(function (c) { return c && c.name === name; }) || null;
        var tilt = NaN;
        if (camera && up) {
            try {
                var R = camera.rotationMatrix;          // world -> camera: the optical axis in world is R's third row
                var ax = R[2], nu = Math.hypot(up[0], up[1], up[2]), na = Math.hypot(ax[0], ax[1], ax[2]);
                if (nu > 0 && na > 0) tilt = Math.acos(Math.min(1, Math.abs(ax[0] * up[0] + ax[1] * up[1] + ax[2] * up[2]) / (nu * na))) * 180 / Math.PI;
            } catch (e) { /* no rotation */ }
        }
        var be = v.decoder && v.decoder._mbBackend;
        return { name: name, view: v, camera: camera, tilt: tilt,
                 streams: !!(be && be.sink && typeof be.sink.samplesAtTimestamps === 'function' && be._frameTimes) };
    });
}

/** The camera to suggest: the streamable one looking most straight down, else the first streamable. */
export function suggestTailMotionCamera(choices) {
    var ok = choices.filter(function (c) { return c.streams; });
    if (!ok.length) return null;
    var withTilt = ok.filter(function (c) { return fin(c.tilt); });
    if (withTilt.length) return withTilt.reduce(function (a, b) { return b.tilt < a.tilt ? b : a; });
    return ok[0];
}

// ----------------------------------------------------------------------------
// Poses in the camera
// ----------------------------------------------------------------------------

/**
 * Every animal's points in the camera's image: T × N × 2 (NaN = missing). 3D
 * input is projected (points behind the camera dropped); 2D input must be
 * that camera's.
 */
export function posesInCamera(input, cameraName, camera) {
    var T = input.numFrames, N = input.nodeNames.length;
    if (input.mode === '2d') {
        if (input.camera !== cameraName) {
            throw cannot('the poses here are 2D from ' + (input.camera || 'another camera') + ': measure tail motion in ' + (input.camera || 'that camera'));
        }
        return input.data.map(function (arr) { return arr; });
    }
    if (!camera) throw cannot(cameraName + ' has no calibration to place the 3D poses in its image');
    var p = [0, 0, 0];
    return input.data.map(function (arr) {
        var out = new Float64Array(T * N * 2).fill(NaN);
        for (var i = 0; i < T * N; i++) {
            var x = arr[3 * i], y = arr[3 * i + 1], z = arr[3 * i + 2];
            if (!(fin(x) && fin(y) && fin(z))) continue;
            p[0] = x; p[1] = y; p[2] = z;
            if (!(cameraDepth(p, camera) > 0)) continue;
            var q = reprojectPointCamera(p, camera);
            if (q && fin(q[0]) && fin(q[1])) { out[2 * i] = q[0]; out[2 * i + 1] = q[1]; }
        }
        return out;
    });
}

// ----------------------------------------------------------------------------
// Frames
// ----------------------------------------------------------------------------

var _fallbackCanvas = null;

/**
 * A frame's luma (Y plane), into `slot.buf` (grown as needed): `{Y, W, H, stride, off}`.
 * Planar YUV / NV12 frames copy as they are; RGB frames, or frames the browser
 * only hands out as RGB, are converted (BT.601 weights).
 */
export async function frameLuma(vf, slot) {
    var rect = vf.visibleRect;
    var W = rect ? rect.width : vf.displayWidth, H = rect ? rect.height : vf.displayHeight;
    var fmt = vf.format;
    if (fmt && PLANAR_Y.indexOf(fmt) >= 0) {
        var size = vf.allocationSize();
        if (!slot.buf || slot.buf.byteLength < size) slot.buf = new Uint8Array(size);
        var layout = await vf.copyTo(slot.buf);
        return { Y: slot.buf, W: W, H: H, stride: layout[0].stride, off: layout[0].offset };
    }
    var rgb = null, order = RGB_ORDER[fmt] || null, rowBytes = 4 * W;
    try {
        var opt = order ? undefined : { format: 'RGBA' };
        var bytes = vf.allocationSize(opt);
        if (!slot.rgb || slot.rgb.byteLength < bytes) slot.rgb = new Uint8Array(bytes);
        var lay = await vf.copyTo(slot.rgb, opt);
        rgb = slot.rgb; rowBytes = lay[0].stride; order = order || RGB_ORDER.RGBA;
        if (lay[0].offset) rgb = slot.rgb.subarray(lay[0].offset);
    } catch (e) {
        // the browser cannot convert this frame on copy: draw it instead
        if (!_fallbackCanvas || _fallbackCanvas.width !== W || _fallbackCanvas.height !== H) _fallbackCanvas = new OffscreenCanvas(W, H);
        var c = _fallbackCanvas.getContext('2d', { willReadFrequently: true });
        c.drawImage(vf, 0, 0, W, H);
        rgb = c.getImageData(0, 0, W, H).data; rowBytes = 4 * W; order = RGB_ORDER.RGBA;
    }
    if (!slot.buf || slot.buf.byteLength < W * H) slot.buf = new Uint8Array(W * H);
    var Y = slot.buf;
    for (var y = 0; y < H; y++) {
        var r = y * rowBytes, o = y * W;
        for (var x = 0; x < W; x++, r += 4) Y[o + x] = (299 * rgb[r + order[0]] + 587 * rgb[r + order[1]] + 114 * rgb[r + order[2]] + 500) / 1000 | 0;
    }
    return { Y: Y, W: W, H: H, stride: W, off: 0 };
}

// ----------------------------------------------------------------------------
// The computation
// ----------------------------------------------------------------------------

/**
 * Measure tail motion for every animal of `session` from camera `cameraName`
 * and keep it (pose/tail-motion.js `setTailMotion`).
 * @param {Object} session  the session shown (its views hold the decoders)
 * @param {string} cameraName
 * @param {{ctx?: Object, backend?: Object, onProgress?: function(number, number), isCancelled?: function(): boolean}} [opts]
 *   `backend`: a mediabunny video backend to read instead of the view's (tests)
 * @returns {Promise<Object>} the stored record, or `{cancelled: true}`
 */
export async function computeTailMotion(session, cameraName, opts) {
    opts = opts || {};
    var t0 = performance.now();
    var ctx = opts.ctx || cachedBehaviorContext(session);
    var input = ctx && ctx.input;
    if (!input || !input.animals || !input.animals.length) throw cannot('there are no animals with identities to measure');
    var choice = tailMotionCameras(session, ctx).find(function (c) { return c.name === cameraName; }) || null;
    var be = opts.backend || (choice && choice.view.decoder && choice.view.decoder._mbBackend);
    if (!be || !be.sink || typeof be.sink.samplesAtTimestamps !== 'function' || !be._frameTimes) {
        throw cannot('needs ' + cameraName + '\'s video loaded with the frame-accurate decoder');
    }
    var camera = (choice && choice.camera) || ((session.cameras || []).find(function (c) { return c && c.name === cameraName; }) || null);
    var T = input.numFrames, N = input.nodeNames.length;
    var P2s = posesInCamera(input, cameraName, camera);
    var roles = ctx.body.roles;
    var dist = function (node) {
        var v = [];
        var step = Math.max(1, Math.floor(T / 5000));
        P2s.forEach(function (P2) {
            for (var t = 0; t < T; t += step) {
                var a = (t * N + roles.tailBase) * 2, b = (t * N + node) * 2;
                v.push(Math.hypot(P2[b] - P2[a], P2[b + 1] - P2[a + 1]));
            }
        });
        return medianOf(v);
    };
    var nodes = tailMotionNodes(input.nodeNames, input.edges, roles, ctx.body.defaultExcluded, dist);
    if (!nodes) throw cannot('needs a head, a tail base and tail nodes past it in the skeleton');
    var size = cropSide(P2s, T, N, nodes);
    if (!(size.side > 0)) throw cannot('no animal is in ' + cameraName + '\'s image');
    var pass = createTailMotionPass({ P2s: P2s, T: T, N: N, nodes: nodes, side: size.side });
    var frames = Math.min(T, be._frameTimes.length);
    var ts = new Array(frames);
    for (var i = 0; i < frames; i++) ts[i] = be._frameTimes[i];
    var it = be.sink.samplesAtTimestamps(ts)[Symbol.asyncIterator]();
    var slots = [{}, {}, {}];
    var cancelled = false, read = 0;
    var workStart = performance.now(), lastProgress = 0;
    try {
        for (var t = 0; t < frames; t++) {
            var r = await it.next();
            if (r.done) break;
            var s = r.value;
            if (!s) continue;                  // a frame the video does not have: the next one starts over
            var vf = s.toVideoFrame();
            s.close();
            try {
                pass.push(t, await frameLuma(vf, slots[t % 3]));
                read++;
            } finally { vf.close(); }
            if (opts.isCancelled && opts.isCancelled()) { cancelled = true; break; }
            var now = performance.now();
            if (now - lastProgress >= PROGRESS_MS) { lastProgress = now; if (opts.onProgress) opts.onProgress(t + 1, frames); }
            if (now - workStart >= WORK_MS) { await yieldToPaint(); workStart = performance.now(); }
        }
    } finally {
        try { if (it.return) await it.return(); } catch (e) { /* ignore */ }
    }
    if (cancelled) return { cancelled: true };
    if (opts.onProgress) opts.onProgress(frames, frames);
    return setTailMotion(session, {
        camera: cameraName, numFrames: T, animals: input.animals.slice(), raw: pass.raw,
        side: size.side, bodyPx: size.body, tailPx: size.tail, crop: TAIL_MOTION_CROP,
        framesRead: read, framesInVideo: be._frameTimes.length, measured: pass.stats.measured,
        seconds: (performance.now() - t0) / 1000, tailNodes: nodes.chain.map(function (k) { return input.nodeNames[k]; }),
    });
}

// ----------------------------------------------------------------------------
// The dialog
// ----------------------------------------------------------------------------

var _open = null;

/** Behavior ▸ Tail Motion from Video…: pick the camera, then compute (progress card; Esc cancels). */
export function showTailMotionModal() {
    if (_open) return;
    var session = state.session;
    if (!session) { setStatus('Load a session first'); return; }
    var ctx;
    try { ctx = cachedBehaviorContext(session); } catch (e) { setStatus('Tail motion: ' + (e && e.message || e), 'error'); return; }
    if (!ctx || !ctx.input || !ctx.input.animals.length) {
        setStatus('Tail motion needs animals with identities (Track All, or identities on the instances)', 'warning');
        return;
    }
    var choices = tailMotionCameras(session, ctx);
    if (!choices.length) { setStatus('Tail motion needs this session\'s videos loaded', 'warning'); return; }
    var pick = suggestTailMotionCamera(choices);
    var rec = tailMotionOf(session);

    var overlay = document.createElement('div');
    overlay.className = 'plane-confirm-overlay';
    overlay.id = 'tailMotionModal';
    var modal = document.createElement('div');
    modal.className = 'plane-confirm-modal behavior-confirm-modal tail-motion-modal';
    var h = document.createElement('h3');
    h.textContent = 'Tail Motion from Video ';
    var tip = document.createElement('button');
    tip.type = 'button';
    tip.className = 'plane-section-info';
    tip.id = 'tailMotionInfo';
    tip.setAttribute('aria-label', 'About tail motion');
    tip.innerHTML = ICON_INFO;
    setInfoTip(tip, TAIL_MOTION_INFO);
    h.appendChild(tip);
    modal.appendChild(h);
    var msg = document.createElement('div');
    msg.className = 'plane-confirm-message';
    msg.textContent = 'Measure each animal\'s tail movement in one camera\'s video, for the behavior classifiers. Kept until the project closes; not saved in the file.';
    modal.appendChild(msg);

    var row = document.createElement('label');
    row.className = 'tail-motion-camera';
    row.textContent = 'Camera ';
    var sel = document.createElement('select');
    sel.id = 'tailMotionCamera';
    choices.forEach(function (c) {
        var o = document.createElement('option');
        o.value = c.name;
        o.textContent = c.name + (fin(c.tilt) ? ' — ' + Math.round(c.tilt) + '° from straight down' : '') + (c.streams ? '' : ' (video not ready)');
        o.disabled = !c.streams;
        if (pick && c.name === pick.name) o.selected = true;
        sel.appendChild(o);
    });
    row.appendChild(sel);
    modal.appendChild(row);

    var table = document.createElement('table');
    table.className = 'data-table behavior-confirm-counts';
    var rows = [['Animals', n(ctx.input.animals.length)], ['Frames', n(ctx.input.numFrames)],
                ['Poses', ctx.input.mode === '3d' ? '3D, placed in the camera' : '2D from ' + (ctx.input.camera || '—')]];
    if (rec) rows.push(['Measured now', 'from ' + rec.camera + ' in ' + formatDuration(rec.seconds || 0)]);
    rows.forEach(function (r) {
        var tr = document.createElement('tr');
        var a = document.createElement('td'); a.textContent = r[0];
        var b = document.createElement('td'); b.textContent = r[1]; b.className = 'num';
        tr.appendChild(a); tr.appendChild(b);
        table.appendChild(tr);
    });
    modal.appendChild(table);

    var actions = document.createElement('div');
    actions.className = 'modal-actions';
    var clear = null;
    if (rec) {
        clear = document.createElement('button');
        clear.textContent = 'Forget';
        clear.id = 'tailMotionClear';
        actions.appendChild(clear);
    }
    var cancel = document.createElement('button');
    cancel.textContent = 'Cancel';
    var go = document.createElement('button');
    go.className = 'primary';
    go.id = 'tailMotionCompute';
    go.textContent = rec ? 'Measure again' : 'Measure';
    go.disabled = !pick;
    actions.appendChild(cancel);
    actions.appendChild(go);
    modal.appendChild(actions);
    overlay.appendChild(modal);
    document.body.appendChild(overlay);
    _open = overlay;

    function close() {
        document.removeEventListener('keydown', onKey, true);
        if (overlay.parentNode) overlay.remove();
        _open = null;
    }
    function onKey(e) {
        if (e.key !== 'Escape') return;
        e.preventDefault();
        e.stopPropagation();
        close();
    }
    document.addEventListener('keydown', onKey, true);
    cancel.addEventListener('click', close);
    if (clear) clear.addEventListener('click', function () {
        clearTailMotion(session);
        close();
        setStatus('Tail motion forgotten: the classifiers no longer have it on this session');
    });
    go.addEventListener('click', function () {
        var cam = sel.value;
        close();
        runTailMotion(session, cam, ctx);
    });
    go.focus();
}

/**
 * Compute with the progress card; Esc cancels. Resolves the record, null when
 * cancelled or failed (the status line says which).
 */
export async function runTailMotion(session, cameraName, ctx) {
    var cancelled = false;
    var onKey = function (e) {
        if (e.key !== 'Escape') return;
        e.preventDefault();
        e.stopPropagation();
        cancelled = true;
    };
    var title = 'Tail Motion from Video';
    var label = 'Reading ' + cameraName;
    var T = ctx && ctx.input ? ctx.input.numFrames : 0;
    document.addEventListener('keydown', onKey, true);
    showLoadingProgress(label, 0, T, { title: title, detail: 'Esc cancels' });
    await yieldToPaint();
    try {
        var rec = await computeTailMotion(session, cameraName, {
            ctx: ctx,
            isCancelled: function () { return cancelled; },
            onProgress: function (d, total) { showLoadingProgress(label, d, total, { title: title, detail: 'Esc cancels' }); },
        });
        if (rec.cancelled) { setStatus('Tail motion cancelled: nothing kept', 'warning'); return null; }
        setStatus('Tail motion measured from ' + cameraName + ' for ' + n(rec.animals.length) + ' animal' + (rec.animals.length === 1 ? '' : 's') +
            ' over ' + n(rec.framesRead) + ' frames in ' + formatDuration(rec.seconds) + ': the classifiers have ' +
            n(TAIL_MOTION_CHANNELS.length) + ' more measures per animal — train them again to use them');
        return rec;
    } catch (e) {
        setStatus('Tail motion: ' + (e && e.userFacing ? e.message : 'failed — ' + String(e && e.message || e)), e && e.userFacing ? 'warning' : 'error');
        if (!(e && e.userFacing)) console.error(e);
        return null;
    } finally {
        document.removeEventListener('keydown', onKey, true);
        hideLoading();
    }
}
