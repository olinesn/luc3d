// ui/threshold-examples-panel.js — the reprojection error threshold card's right
// half: what the value being tried does to this session's own nodes.
//
// One tile per calibrated camera, all at one scale so errors compare by eye: a
// crop of that camera's video around the node's prediction, the prediction
// (white dot), the reprojection after the rule has run (blue ring), the
// reprojection error (red line), the threshold (dashed green ring) and, where a
// camera was dropped, the first solve's reprojection (grey ring).
//
// The arrows walk the session's nodes from the smallest worst-camera error to the
// largest. Moving the slider jumps to the first node the value catches, so ‹ shows
// what it keeps and › what it drops. With the threshold off, or nothing over it,
// the list opens at the largest error. Everything here is read-only: the demo
// never changes a setting or the session.
//
// The session is solved once per method, a slice per task, when the panel is
// first shown (`start`), and only the tiles of the example on screen decode video.
// A decoder's frame may be closed by the next decode, so each tile copies its crop
// at once, and only the newest request per camera runs.
//
// Depends on: pose/threshold-examples.js, ui/app-state.js (state),
// ui/frame-readout.js (formatFrameNumber).

import {
    calibratedCameras,
    firstOverThreshold,
    scanThresholdExamples,
    solveExample,
    EXAMPLE_MIN_CAMERAS,
} from '../pose/threshold-examples.js';
import { state } from './app-state.js';
import { formatFrameNumber } from './frame-readout.js';

/** A tile's canvas side, in device pixels. */
const TILE_PX = 240;
/** A crop's smallest side, in video pixels: a small error still shows its surroundings. */
const MIN_CROP = 80;
const DECODE_DELAY_MS = 90;

const PREDICTION = '#fff', REPROJECTION = '#7c93ff', FIRST_SOLVE = '#b5b5b5', ERROR = '#e74c3c', RING = '#2ecc71';

function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
}

function count(n) { return Number(n).toLocaleString('en-US'); }
function px(e) { return e.toFixed(1) + ' px'; }
function thresholdText(t) { return count(t) + ' px'; }

/** "camA", "camA and camB", "camA, camB and camC", in the order they were dropped. */
function nameList(names) {
    if (names.length < 2) return names.join('');
    return names.slice(0, -1).join(', ') + ' and ' + names[names.length - 1];
}

/**
 * The sentence under the tiles.
 * @param {Object} solved  `solveExample`'s result
 * @param {number} threshold
 */
export function exampleResultText(solved, threshold) {
    const first = solved.passes[0], last = solved.passes[solved.passes.length - 1];
    const largest = function (pass) {
        return pass.cameras.reduce(function (m, c) { return pass.errors[c] != null && pass.errors[c] > m ? pass.errors[c] : m; }, 0);
    };
    if (!(threshold > 0)) return 'With the threshold off, every camera counts. The largest error here is ' + px(largest(first)) + '.';
    const at = 'At ' + thresholdText(threshold) + ', ';
    if (solved.lost) {
        const dropped = solved.dropped.length ? nameList(solved.dropped) + (solved.dropped.length > 1 ? ' are' : ' is') + ' dropped, but the ' : 'the ';
        return at + dropped + 'last cameras still miss, so this node gets no 3D point in this frame.';
    }
    if (!solved.dropped.length) return at + 'every camera is within the threshold, so nothing is dropped. The largest error is ' + px(largest(first)) + '.';
    return at + nameList(solved.dropped) + (solved.dropped.length > 1 ? ' are' : ' is') + ' dropped and the node is solved again from '
        + count(last.cameras.length) + ' cameras. The largest error goes from ' + px(largest(first)) + ' to ' + px(largest(last)) + '.';
}

/** One camera's text under its tile, and its state. */
export function tileVerdict(solved, name, threshold, included) {
    const first = solved.passes[0], last = solved.passes[solved.passes.length - 1];
    const dropped = solved.dropped.indexOf(name) >= 0;
    if (included && included.indexOf(name) < 0) return { state: 'out', text: 'not in tracking' };
    if (!solved.predictions.get(name)) return { state: 'out', text: 'no prediction' };
    const a = first.errors[name], b = last.errors[name];
    if (dropped) return { state: 'drop', text: px(a) + ' · dropped' };
    if (solved.lost && isFinite(b) && b > threshold) return { state: 'over', text: px(b) + ' · over' };
    if (b == null) return { state: 'out', text: 'no 3D' };
    const moved = solved.dropped.length > 0 && a != null;
    return { state: threshold > 0 ? 'in' : 'off', text: moved ? a.toFixed(1) + ' → ' + px(b) : px(b) };
}

/** The crop's side for an example: every tile at one scale, so errors compare by eye. */
export function cropSide(solved, names) {
    let worst = 0;
    solved.passes.forEach(function (pass) {
        names.forEach(function (name) {
            const e = pass.errors[name];
            if (e != null && isFinite(e) && e > worst) worst = e;
        });
    });
    return Math.max(MIN_CROP, Math.ceil(worst * 2.6));
}

function drawMarks(ctx, solved, name, box, threshold, edges) {
    const scale = TILE_PX / box.side;
    const X = function (p) { return (p[0] - box.x) * scale; };
    const Y = function (p) { return (p[1] - box.y) * scale; };
    const pts = solved.skeletons.get(name);
    if (pts) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)'; ctx.lineWidth = 1.5;
        (edges || []).forEach(function (edge) {
            const a = pts[edge[0]], b = pts[edge[1]];
            if (!a || !b) return;
            ctx.beginPath(); ctx.moveTo(X(a), Y(a)); ctx.lineTo(X(b), Y(b)); ctx.stroke();
        });
        ctx.fillStyle = 'rgba(255, 255, 255, 0.55)';
        pts.forEach(function (p) { if (p) { ctx.beginPath(); ctx.arc(X(p), Y(p), 3, 0, 2 * Math.PI); ctx.fill(); } });
    }
    const pred = solved.predictions.get(name);
    const first = solved.passes[0].reprojections[name];
    const last = solved.passes[solved.passes.length - 1].reprojections[name];
    const ring = function (p, color, r, width) {
        ctx.strokeStyle = color; ctx.lineWidth = width;
        ctx.beginPath(); ctx.arc(X(p), Y(p), r, 0, 2 * Math.PI); ctx.stroke();
    };
    if (pred && threshold > 0) {
        ctx.setLineDash([7, 5]);
        ring(pred, RING, threshold * scale, 2);
        ctx.setLineDash([]);
    }
    if (first && solved.dropped.length && !solved.lost) ring(first, FIRST_SOLVE, 7, 2);
    const drawn = solved.lost ? first : last;
    if (pred && drawn) {
        ctx.strokeStyle = ERROR; ctx.lineWidth = 3; ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(X(pred), Y(pred)); ctx.lineTo(X(drawn), Y(drawn)); ctx.stroke();
    }
    if (drawn) ring(drawn, REPROJECTION, 8, 3);
    if (pred) {
        ctx.fillStyle = PREDICTION; ctx.strokeStyle = 'rgba(0, 0, 0, 0.7)'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(X(pred), Y(pred), 6, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    }
}

/**
 * @param {Object} opts
 * @param {?Object} opts.session
 * @param {function(string): boolean} [opts.isCameraIncluded]  the working Camera Views value
 * @returns {{el: HTMLElement, start: function(): void, update: function({threshold: number, method: string}): void,
 *   step: function(number): void, invalidate: function(): void, dispose: function(): void}}
 */
export function buildThresholdExamples(opts) {
    const session = opts.session;
    const root = el('div', 'thr-ex');
    root.tabIndex = 0;
    root.setAttribute('role', 'group');
    root.setAttribute('aria-label', 'Examples from this session');

    const head = el('div', 'thr-ex-head');
    head.appendChild(el('div', 'thr-ex-title', 'In this session'));
    const nav = el('div', 'thr-ex-nav');
    const prev = el('button', 'thr-ex-arrow thr-ex-prev', '‹');
    const pos = el('span', 'thr-ex-pos');
    const next = el('button', 'thr-ex-arrow thr-ex-next', '›');
    prev.type = next.type = 'button';
    prev.setAttribute('aria-label', 'Smaller error');
    next.setAttribute('aria-label', 'Larger error');
    prev.title = 'Smaller error';
    next.title = 'Larger error';
    nav.append(prev, pos, next);
    head.appendChild(nav);
    root.appendChild(head);

    const where = el('div', 'thr-ex-where');
    root.appendChild(where);
    const message = el('p', 'thr-ex-message');
    root.appendChild(message);
    const grid = el('div', 'thr-ex-grid');
    root.appendChild(grid);
    const result = el('p', 'thr-ex-result');
    root.appendChild(result);
    const summary = el('p', 'thr-ex-summary');
    root.appendChild(summary);
    const legend = el('div', 'thr-ex-legend');
    legend.innerHTML = '<span><i class="thr-key-pred"></i>prediction</span><span><i class="thr-key-rep"></i>reprojection</span>'
        + '<span><i class="thr-key-first"></i>first solve</span><span><i class="thr-key-err"></i>reprojection error</span>'
        + '<span><i class="thr-key-ring"></i>threshold</span>';
    root.appendChild(legend);

    const cameras = calibratedCameras(session);
    const names = cameras.map(function (c) { return c.name; });
    const tiles = new Map();
    names.forEach(function (name) {
        const tile = el('div', 'thr-ex-tile');
        tile.dataset.camera = name;
        const canvas = el('canvas');
        canvas.width = canvas.height = TILE_PX;
        tile.appendChild(canvas);
        tile.appendChild(el('div', 'thr-ex-cam', name));
        tile.appendChild(el('div', 'thr-ex-err'));
        grid.appendChild(tile);
        tiles.set(name, { tile: tile, canvas: canvas, crop: null, cropKey: null });
    });

    let started = false, disposed = false;
    let method = 'dlt', threshold = 0, index = -1;
    let scanToken = 0, showToken = 0, decodeTimer = 0;
    let scans = {};
    const decodeChains = new Map();

    function included() {
        return names.filter(function (n) { return !opts.isCameraIncluded || opts.isCameraIncluded(n); });
    }

    function setMessage(text) {
        message.textContent = text;
        message.hidden = !text;
        const showing = !text;
        where.hidden = grid.hidden = result.hidden = summary.hidden = legend.hidden = !showing;
        pos.textContent = '';
        prev.disabled = next.disabled = true;
    }

    function jumpToThreshold() {
        const list = scans[method] ? scans[method].examples : [];
        const at = threshold > 0 ? firstOverThreshold(list, threshold) : list.length;
        index = at < list.length ? at : list.length - 1;
    }

    function nodeName(node) {
        const nodes = session.skeleton && session.skeleton.nodes;
        return nodes && nodes[node] != null ? (nodes[node].name || String(nodes[node])) : 'node ' + node;
    }

    function animalName(id) {
        const known = session.identities && session.identities[id];
        if (known && known.name) return known.name;
        return id != null && id >= 0 ? 'animal ' + id : 'an unnamed animal';
    }

    function viewFor(name) {
        return (state.views || []).filter(function (v) {
            return v && v.name === name && v.decoder && typeof v.decoder.getFrame === 'function';
        })[0];
    }

    function drawTile(t, name, solved, box, edges) {
        const ctx = t.canvas.getContext('2d');
        ctx.fillStyle = '#111';
        ctx.fillRect(0, 0, TILE_PX, TILE_PX);
        if (t.crop) ctx.drawImage(t.crop, 0, 0);
        drawMarks(ctx, solved, name, box, threshold, edges);
    }

    function show() {
        const token = ++showToken;
        clearTimeout(decodeTimer);
        const scan = scans[method];
        if (!session || !scan) return;
        const list = scan.examples;
        if (!list.length) {
            setMessage('No node here is predicted by ' + EXAMPLE_MIN_CAMERAS + ' or more cameras in one group. '
                + 'Group the predictions across cameras (Track All) to see them here.');
            return;
        }
        const example = list[index];
        const use = included();
        const solved = solveExample(session, example, { method: method, threshold: threshold, included: use });
        if (!solved) return;
        setMessage('');
        pos.textContent = count(index + 1) + ' of ' + count(list.length);
        prev.disabled = index <= 0;
        next.disabled = index >= list.length - 1;
        where.textContent = 'Frame ' + formatFrameNumber(example.frame + 1) + ' · ' + animalName(solved.identityId) + ' · ' + nodeName(example.node);
        result.textContent = exampleResultText(solved, threshold);
        const over = threshold > 0 ? list.length - firstOverThreshold(list, threshold) : 0;
        summary.textContent = threshold > 0
            ? count(over) + ' of these ' + count(list.length) + ' nodes have a camera dropped at ' + thresholdText(threshold) + '. '
                + '‹ smaller errors, › larger. Nodes from ' + count(scan.frames) + ' frames, seen by ' + EXAMPLE_MIN_CAMERAS + ' or more cameras.'
            : 'These ' + count(list.length) + ' nodes run from the smallest worst-camera error to the largest. Nodes from '
                + count(scan.frames) + ' frames, seen by ' + EXAMPLE_MIN_CAMERAS + ' or more cameras.';

        // Every tile at one scale, centred on its own prediction (or, without one, its reprojection).
        const side = cropSide(solved, use);
        const edges = session.skeleton && session.skeleton.edges;
        const toDecode = [];
        names.forEach(function (name) {
            const t = tiles.get(name);
            const verdict = tileVerdict(solved, name, threshold, use);
            t.tile.dataset.state = verdict.state;
            t.tile.querySelector('.thr-ex-err').textContent = verdict.text;
            const centre = solved.predictions.get(name) || solved.passes[solved.passes.length - 1].reprojections[name];
            const box = centre ? { x: centre[0] - side / 2, y: centre[1] - side / 2, side: side } : { x: 0, y: 0, side: side };
            const key = example.frame + '|' + box.x + '|' + box.y + '|' + side;
            if (t.cropKey !== key) { t.crop = null; t.cropKey = key; if (centre) toDecode.push({ name: name, t: t, box: box, key: key }); }
            drawTile(t, name, solved, box, edges);
            t.redraw = function () { drawTile(t, name, solved, box, edges); };
        });
        if (toDecode.length) decodeTimer = setTimeout(function () { decodeTiles(token, example.frame, toDecode); }, DECODE_DELAY_MS);
    }

    function decodeTiles(token, frame, items) {
        items.forEach(function (it) {
            const view = viewFor(it.name);
            if (!view) return;
            const chain = decodeChains.get(it.name) || Promise.resolve();
            decodeChains.set(it.name, chain.then(async function () {
                if (token !== showToken || disposed) return;
                let image = null;
                try { image = await view.decoder.getFrame(frame); } catch (e) { image = null; }
                if (!image || token !== showToken || disposed) return;
                const crop = new OffscreenCanvas(TILE_PX, TILE_PX);
                try {
                    crop.getContext('2d').drawImage(image, it.box.x, it.box.y, it.box.side, it.box.side, 0, 0, TILE_PX, TILE_PX);
                } catch (e) { return; /* a closed frame draws nothing */ }
                if (it.t.cropKey !== it.key) return;
                it.t.crop = crop;
                it.t.redraw();
            }));
        });
    }

    async function ensureScan() {
        if (disposed || !started || !session) return;
        if (scans[method]) { jumpToThreshold(); show(); return; }
        if (cameras.length < EXAMPLE_MIN_CAMERAS) {
            setMessage('This needs a calibration with ' + EXAMPLE_MIN_CAMERAS + ' or more cameras.');
            return;
        }
        const token = ++scanToken, scanMethod = method;
        setMessage('Solving this session’s frames…');
        const scan = await scanThresholdExamples(session, {
            method: scanMethod,
            included: included(),
            isCancelled: function () { return disposed || token !== scanToken; },
            onProgress: function (done, total) {
                if (token === scanToken) setMessage('Solving this session’s frames… ' + count(done) + ' of ' + count(total));
            },
        });
        if (!scan || token !== scanToken) return;
        scans[scanMethod] = scan;
        jumpToThreshold();
        show();
    }

    function step(by) {
        const list = scans[method] ? scans[method].examples : [];
        const nextIndex = Math.max(0, Math.min(list.length - 1, index + by));
        if (!list.length || nextIndex === index) return;
        index = nextIndex;
        show();
    }

    prev.addEventListener('click', function () { step(-1); });
    next.addEventListener('click', function () { step(1); });

    setMessage(session ? 'Solving this session’s frames…' : 'Load a session to see this on its own predictions and reprojections.');

    return {
        el: root,
        /** Solve the session: call when the panel is first on screen, so opening Settings costs nothing. */
        start: function () {
            if (started) return;
            started = true;
            ensureScan();
        },
        update: function (next) {
            const m = next.method === 'ba' ? 'ba' : 'dlt', t = next.threshold > 0 ? next.threshold : 0;
            const methodChanged = m !== method, thresholdChanged = t !== threshold;
            if (!methodChanged && !thresholdChanged) return;
            method = m; threshold = t;
            if (!started) return;
            if (methodChanged) { ensureScan(); return; }
            if (scans[method]) { jumpToThreshold(); show(); }
        },
        step: step,
        /** The working Camera Views changed: what was solved no longer stands. */
        invalidate: function () {
            scans = {};
            scanToken++;
            names.forEach(function (n) { const t = tiles.get(n); t.crop = null; t.cropKey = null; });
            ensureScan();
        },
        dispose: function () {
            disposed = true;
            clearTimeout(decodeTimer);
        },
    };
}
