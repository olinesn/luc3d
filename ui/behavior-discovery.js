// ui/behavior-discovery.js — behavior discovery in the app: running it on the
// project with the progress card (Esc cancels), keeping its clusters as a states
// layer, drawing its map, and stepping through a cluster's examples.
//
// A run reads the poses of the active session, or of every session whose poses
// are resident (a lazy project's other sessions are named as left out), one at a
// time, through ui/behavior-detect.js's contexts: the same body, floor and
// broken-tracking masks as the detectors. Everything from there on is
// pose/behavior-discovery.js; only the layer is saved.
//
// The map (`drawDiscoveryMap`) is the sample's layout from the run in this page:
// a point per bin, in its cluster's state color, gray for bins in no cluster. A
// click goes to that bin's moment. It is gone after a reopen (only the layer is
// saved), and running again draws it again: the same poses give the same layout.
//
// Depends on: ui/app-state.js, ui/loading-overlay.js, ui/behavior-detect.js,
//             ui/behavior-coding.js, pose/behavior-discovery.js, pose/behavior-states.js,
//             pose/behavior-ethogram.js, import-export/save-load.js

import { state } from './app-state.js';
import { showLoading, showLoadingProgress, hideLoading, yieldToPaint } from './loading-overlay.js';
import { buildBehaviorContext, cachedBehaviorContext } from './behavior-detect.js';
import { requestBehaviorRepaint, setReviewFocus, goToBehaviorFrame, openBehaviorTab, behaviorUi } from './behavior-coding.js';
import { discoverBehaviors, storeDiscovery } from '../pose/behavior-discovery.js';
import { stateColor, stateLabel, statesToBouts, OTHER_STATE_COLOR } from '../pose/behavior-states.js';
import { behaviorById } from '../pose/behavior-ethogram.js';
import { setStatus } from '../import-export/save-load.js';

/** The layer a run makes unless told otherwise. */
export var DEFAULT_DISCOVERY_LAYER = 'Discovery';
/** A split's layer name. */
export function splitLayerName(behaviorName) {
    return 'Split: ' + behaviorName;
}

var STAGES = {
    read: 'Reading poses', bins: 'Measuring 100 ms bins', reduce: 'Reducing the features (PCA)', neighbors: 'Finding neighbours',
    layout: 'Laying out the bins, twice (UMAP)', cluster: 'Clustering (HDBSCAN)', forests: 'Growing a forest per cluster', labels: 'Labelling every frame',
};
var STAGE_ORDER = Object.keys(STAGES);

function n(x) { return Number(x || 0).toLocaleString('en-US'); }
function plural(k, one, many) { return n(k) + ' ' + (k === 1 ? one : (many || one + 's')); }
function fps() { return state.fps > 0 ? state.fps : 30; }

function sessionName(s, i) {
    return (s && s.name) || 'session ' + (i + 1);
}

var _running = false;

/** Is a run going on? */
export function isDiscovering() {
    return _running;
}

/**
 * Find behaviors in the poses, and keep them as a states layer.
 * @param {{all?: boolean, split?: string, layer?: string, maxBins?: number, neighbors?: number,
 *          clusterShare?: number[], seed?: number}} [opts]
 *   all: every session whose poses are loaded (default: the active one); split: a behavior id
 * @returns {Promise<Object|null>} the run's result, or null (nothing found, cancelled, already running)
 */
export async function runDiscovery(opts) {
    opts = opts || {};
    if (!state.session) { setStatus('No session loaded', 'warning'); return null; }
    if (_running) return null;
    _running = true;
    var title = opts.split ? 'Split a Behavior' : 'Discover Behaviors';
    var cancel = false;
    var onKey = function (e) {
        if (e.key !== 'Escape') return;
        e.preventDefault();
        e.stopPropagation();
        cancel = true;
        showLoading('Cancelling…', { title: title });
    };
    document.addEventListener('keydown', onKey, true);
    try {
        showLoading('Reading poses…', { title: title, detail: 'Esc cancels' });
        await yieldToPaint();
        var sessions = opts.all && state.sessions && state.sessions.length ? state.sessions : [state.session];
        var sources = sessions.map(function (s) {
            return {
                session: s, name: sessionName(s, (state.sessions || []).indexOf(s)),
                getCtx: function () { return s === state.session ? cachedBehaviorContext(s) : buildBehaviorContext(s); },
            };
        });
        var split = opts.split ? behaviorById(opts.split) : null;
        var res = await discoverBehaviors({
            sources: sources, split: split ? split.id : null, splitMutual: !!(split && split.mutual),
            maxBins: opts.maxBins, neighbors: opts.neighbors, clusterShare: opts.clusterShare, seed: opts.seed,
            pause: yieldToPaint, isCancelled: function () { return cancel; },
            onProgress: function (p) {
                if (cancel) return;
                var step = STAGE_ORDER.indexOf(p.stage) + 1;
                var detail = (p.session ? p.session + ' · ' : '') + 'Esc cancels';
                if (p.unit && p.total > 0) showLoadingProgress(STAGES[p.stage], p.done, p.total, { title: title, unit: p.unit, step: step, steps: STAGE_ORDER.length, detail: detail });
                else showLoading(STAGES[p.stage] + '…', { title: title, step: step, steps: STAGE_ORDER.length, detail: detail });
            },
        });
        if (res.cancelled || cancel) { setStatus(title + ' cancelled', 'warning'); return null; }
        if (res.error) {
            setStatus(title + ': ' + res.error + leftOutNote(res), 'warning');
            return null;
        }
        var name = opts.layer || (split ? splitLayerName(split.name) : DEFAULT_DISCOVERY_LAYER);
        var kept = storeDiscovery(res, name);
        openBehaviorTab();
        var sec = document.getElementById('behaviorStatesSection');
        if (sec && !sec.open) sec.open = true;
        requestBehaviorRepaint({ states: true });
        var rep = res.report;
        setStatus('Found ' + plural(res.clusters.length, 'cluster') + ' in ' + plural(rep.bins, 'bin') + ' of 100 ms' + (kept.sessions > 1 ? ' from ' + plural(kept.sessions, 'session') : '') +
            ' · "' + kept.layer + '" under States' + (Number.isFinite(rep.heldOutAgreement) ? ' · the forests repeat the clusters on ' + Math.round(100 * rep.heldOutAgreement) + '% of bins they never saw' : '') +
            leftOutNote(res), res.leftOut && res.leftOut.length ? 'warning' : 'success');
        return res;
    } catch (e) {
        console.error('[behavior] discovery failed', e);
        setStatus(title + ' failed: ' + (e && e.message || e), 'error');
        return null;
    } finally {
        document.removeEventListener('keydown', onKey, true);
        hideLoading();
        _running = false;
    }
}

function leftOutNote(res) {
    var left = res.leftOut || [];
    return left.length ? ' · Left out: ' + left.map(function (l) { return l.session + ' (' + l.reason + ')'; }).join(', ') : '';
}

// ----------------------------------------------------------------------------
// Examples
// ----------------------------------------------------------------------------

var _examples = null;

/**
 * Step to a state's next example: its runs from the longest down, whichever
 * animal, boxed on the video while the playhead is in it. Wraps at the end.
 * @param {string} word what to call a state in the status line ("Cluster")
 * @returns {Object|null} the example shown
 */
export function nextStateExample(session, layer, k, word) {
    var e = _examples;
    if (!e || e.session !== session || e.layer !== layer || e.state !== k) {
        var list = statesToBouts(layer, [k]);
        list.sort(function (a, b) { return (b.end - b.start) - (a.end - a.start) || a.start - b.start || (a.animal < b.animal ? -1 : 1); });
        e = _examples = { session: session, layer: layer, state: k, list: list, at: -1 };
    }
    if (!e.list.length) { setStatus(word + ' ' + k + ' does not occur in this session', 'warning'); return null; }
    e.at = (e.at + 1) % e.list.length;
    var it = e.list[e.at];
    showStateMoment(session, layer, k, word, it.animal, it.start, it.end);
    setStatus(word + ' ' + stateLabel(layer, k) + ': ' + it.animal + ', frames ' + n(it.start + 1) + '–' + n(it.end + 1) + ' (' +
        ((it.end - it.start + 1) / fps()).toFixed(2) + ' s) · example ' + n(e.at + 1) + ' of ' + n(e.list.length) + ', longest first');
    return it;
}

function showStateMoment(session, layer, k, word, animal, start, end) {
    setReviewFocus(session, { behaviorId: '', label: word + ' ' + stateLabel(layer, k), color: stateColor(k), animal: animal, target: '', start: start, end: end }, 'states');
    goToBehaviorFrame(start);
}

/** Drop the box of the example shown (only if an example set it). */
export function clearStateExample() {
    _examples = null;
    var r = behaviorUi().review;
    if (r && r.from === 'states') setReviewFocus(null);
}

// ----------------------------------------------------------------------------
// The map
// ----------------------------------------------------------------------------

/** Where a layout's points land on a canvas of this size. */
function mapScale(map, W, H, pad) {
    var E = map.embedding, d = map.dims, m = E.length / d;
    var x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (var i = 0; i < m; i++) {
        var x = E[i * d], y = E[i * d + 1];
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
    }
    var s = Math.min((W - 2 * pad) / Math.max(1e-9, x1 - x0), (H - 2 * pad) / Math.max(1e-9, y1 - y0));
    var ox = (W - s * (x1 - x0)) / 2, oy = (H - s * (y1 - y0)) / 2;
    return { sx: function (v) { return ox + (v - x0) * s; }, sy: function (v) { return oy + (v - y0) * s; } };
}

/**
 * Draw a run's layout: every bin in its cluster's color (gray: in no cluster),
 * `selected`'s drawn last and the others dimmed, each cluster's number at its middle.
 * @param {HTMLCanvasElement} canvas
 * @param {Object} result from `discoveryResult`
 * @param {number|null} [selected]
 */
export function drawDiscoveryMap(canvas, result, selected) {
    var map = result.map;
    var dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
    var W = Math.max(50, Math.round((canvas.clientWidth || 300) * dpr)), H = Math.max(50, Math.round((canvas.clientHeight || 220) * dpr));
    if (canvas.width !== W) canvas.width = W;
    if (canvas.height !== H) canvas.height = H;
    var g = canvas.getContext('2d');
    g.clearRect(0, 0, W, H);
    var E = map.embedding, d = map.dims, m = E.length / d, labels = map.labels;
    var sc = mapScale(map, W, H, 6 * dpr);
    var dot = Math.max(1, Math.round(1.2 * dpr));
    var K = result.clusters.length;
    var byCluster = [];
    for (var k = -1; k < K; k++) byCluster.push([]);
    for (var i = 0; i < m; i++) byCluster[labels[i] + 1].push(i);
    var dim = selected != null && selected >= 0;
    var paint = function (list, color, alpha) {
        g.globalAlpha = alpha;
        g.fillStyle = color;
        for (var j = 0; j < list.length; j++) {
            var o = list[j] * d;
            g.fillRect(sc.sx(E[o]) - dot / 2, sc.sy(E[o + 1]) - dot / 2, dot, dot);
        }
    };
    paint(byCluster[0], OTHER_STATE_COLOR, dim ? 0.15 : 0.4);
    for (var k2 = 0; k2 < K; k2++) if (k2 !== selected) paint(byCluster[k2 + 1], stateColor(k2), dim ? 0.15 : 0.8);
    if (dim && byCluster[selected + 1]) paint(byCluster[selected + 1], stateColor(selected), 1);
    g.globalAlpha = 1;
    // Each cluster's number at the median of its points.
    g.font = Math.round(11 * dpr) + 'px sans-serif';
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    g.lineWidth = 3 * dpr;
    g.strokeStyle = 'rgba(0,0,0,0.8)';
    g.fillStyle = '#ffffff';
    for (var k3 = 0; k3 < K; k3++) {
        var pts = byCluster[k3 + 1];
        if (!pts.length) continue;
        var xs = pts.map(function (p) { return E[p * d]; }).sort(function (a, b) { return a - b; });
        var ys = pts.map(function (p) { return E[p * d + 1]; }).sort(function (a, b) { return a - b; });
        var mx = sc.sx(xs[xs.length >> 1]), my = sc.sy(ys[ys.length >> 1]);
        g.strokeText(String(k3), mx, my);
        g.fillText(String(k3), mx, my);
    }
}

/**
 * The bin under a point of the canvas, in CSS pixels from its corner, or null.
 * @returns {{index: number, state: number, frame: number, animal: string, session: string, sessionIndex: number}|null}
 */
export function discoveryBinAt(canvas, result, cssX, cssY) {
    var map = result.map;
    var dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
    var W = canvas.width, H = canvas.height;
    var sc = mapScale(map, W, H, 6 * dpr);
    var E = map.embedding, d = map.dims, m = E.length / d;
    var px = cssX * dpr, py = cssY * dpr, best = -1, top = (8 * dpr) * (8 * dpr);
    for (var i = 0; i < m; i++) {
        var dx = sc.sx(E[i * d]) - px, dy = sc.sy(E[i * d + 1]) - py, s = dx * dx + dy * dy;
        if (s < top) { top = s; best = i; }
    }
    if (best < 0) return null;
    var si = map.session[best];
    return { index: best, state: map.labels[best], frame: map.frame[best], animal: (map.animals[si] || [])[map.animal[best]] || '',
             session: map.sessions[si] || '', sessionIndex: si };
}

/** Go to a bin of the map: its moment, boxed on the video. */
export function showDiscoveryBin(session, layer, bin, word) {
    if (!bin || !session) return;
    var half = Math.max(1, Math.round(0.05 * fps()));
    showStateMoment(session, layer, bin.state, word, bin.animal, Math.max(0, bin.frame - half), bin.frame + half);
    setStatus(word + ' ' + (bin.state >= 0 ? stateLabel(layer, bin.state) : 'none') + ': ' + bin.animal + ', frame ' + n(bin.frame + 1));
}
