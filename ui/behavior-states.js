// ui/behavior-states.js — behavioral states in the app: the Behavior tab's
// States section and the timeline's "States" rows (pose/behavior-states.js
// has the layers; ui/behavior-kpms-modal.js and ui/behavior-discovery.js bring
// them in).
//
// The States section shows one layer — keypoint-MoSeq's syllables, or discovery's
// clusters — of the active session, or of every session that has a layer of that
// name:
//   * a row per state: its color, number and name (editable: names are
//     written into every session's layer of that name), its share of the
//     frames, its median run length, and how the labels fall in it —
//     P(behavior | state) with the frames behind it, for one behavior or the
//     best-matching one; unlabelled frames count for neither;
//   * Next ▸ per state: the next run of it after the playhead; for a cluster, its
//     next example, the longest runs first, boxed on the video;
//   * Promote: the checked states become one behavior's predictions (source
//     `states:<layer>`, pose/behavior-predictions.js), reviewed in the
//     Predictions section like the detectors' — each bout scored by the
//     share of its main state's labelled frames that are that behavior. The
//     behavior can be a new one, named after them;
//   * Remove the layer (this session, or all), with a confirm.
// States below 0.5% of the frames are folded away until asked for.
//
// A layer discovery made in this page also has its map (a canvas, drawn by
// ui/behavior-discovery.js): every clustered bin, in its state's color. A press on
// a point goes to that moment; a click on a state's number selects it on the map.
// What stands out in each cluster is listed under its row. Only the layer is saved,
// so a reopened project has no map until discovery runs again. The tab's repaints
// rebuild the section's HTML, so the canvas lives outside it and is moved back in:
// it is drawn only when the run, the selected state or its size changed.
//
// Timeline: `stateTimelineRows` gives one row per animal of the shown layer,
// each run in its state's color (`stateColor`), cached per layer version: a
// repaint never walks frames.
//
// Depends on: ui/app-state.js, ui/behavior-coding.js, ui/behavior-confirm.js,
//             ui/behavior-detect.js, ui/behavior-discovery.js, ui/behavior-discovery-modal.js,
//             ui/behavior-kpms-modal.js, ui/behavior-train.js, ui/info-tip.js,
//             ui/plane-definition.js (the ⓘ icon), pose/behavior-states.js,
//             pose/behavior-discovery.js, pose/behavior-predictions.js, pose/behavior-ethogram.js,
//             pose/behavior-tensor.js, import-export/save-load.js

import { state } from './app-state.js';
import { sessionAnimals, animalColor, requestBehaviorRepaint, goToBehaviorFrame } from './behavior-coding.js';
import { behaviorConfirm } from './behavior-confirm.js';
import { detectionFps, sessionFps } from './behavior-detect.js';
import { showKpmsExportModal, showKpmsImportModal } from './behavior-kpms-modal.js';
import { showBehaviorDiscoveryModal } from './behavior-discovery-modal.js';
import { nextStateExample, clearStateExample, drawDiscoveryMap, discoveryBinAt, showDiscoveryBin } from './behavior-discovery.js';
import { openSessionByName } from './behavior-train.js';
import { setInfoTip } from './info-tip.js';
import { ICON_INFO } from './plane-definition.js';
import {
    stateLayers, stateLayer, removeStateLayer, setStateName, stateName, stateLabel, stateSummary, stateEnrichment,
    statesToBouts, stateColor, layerAnimals, onStatesChange,
} from '../pose/behavior-states.js';
import { discoveryResult } from '../pose/behavior-discovery.js';
import { setBehaviorRun, behaviorRunOf, runConfigPrint, runStateLayer } from '../pose/behavior-predictions.js';
import { behaviors, behaviorById, behaviorByName, addBehavior } from '../pose/behavior-ethogram.js';
import { hasIdentified3d, pick2dCamera } from '../pose/behavior-tensor.js';
import { markDirty, setStatus } from '../import-export/save-load.js';

/** The gutter name of the timeline's states group. */
export var STATES_GROUP_NAME = 'States';
/** States under this share of the frames are folded away. */
var RARE_SHARE = 0.005;
var STATE_ALPHA = 0.9;

var STATES_INFO = 'Behavioral states found without labels — keypoint-MoSeq\'s syllables, or the clusters of Discover Behaviors — and how your labels fall in each ' +
    '(unlabelled frames count for neither). Name them, or promote some to predictions of a behavior to review.';

/** Promote's choice of "a new behavior". */
var NEW_BEHAVIOR = '__new';

/** What the section shows (browser memory only). */
var ui = { layer: '', scope: 'session', behavior: '', picked: new Set(), showRare: false, promoteTo: '', selected: null };

function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function n(x) { return Number(x || 0).toLocaleString('en-US'); }
function fr(f) { return (f + 1).toLocaleString('en-US'); }
function fps() { return detectionFps(); }
function plural(k, one, many) { return n(k) + ' ' + (k === 1 ? one : (many || one + 's')); }
function pct(x) { return x >= 0.995 || x === 0 ? Math.round(100 * x) + '%' : x < 0.01 ? (100 * x).toFixed(1) + '%' : Math.round(100 * x) + '%'; }

function sessionsOf() {
    return (state.sessions && state.sessions.length) ? state.sessions : (state.session ? [state.session] : []);
}

/** What a layer calls its states. */
function stateWord(layer) {
    return layer && layer.source === 'discovery' ? 'cluster' : 'syllable';
}

function capital(word) {
    return word.charAt(0).toUpperCase() + word.slice(1);
}

/** The layer the section and timeline show for `session`: the chosen one, else the newest. */
export function shownLayer(session) {
    var layers = stateLayers(session);
    if (!layers.length) return null;
    for (var i = 0; i < layers.length; i++) if (layers[i].name === ui.layer) return layers[i];
    return layers[layers.length - 1];
}

/** Each session's layer of this name, as `{session, layer}` (the active session's only, unless counting all). */
function scopeItems(session, name, all) {
    var list = all ? sessionsOf() : [session];
    return list.map(function (s) { return { session: s, layer: stateLayer(s, name) }; }).filter(function (it) { return it.layer; });
}

// ----------------------------------------------------------------------------
// Timeline
// ----------------------------------------------------------------------------

var _rowCache = new WeakMap();

/**
 * The "States" rows for `session`: one per animal of the shown layer (the
 * session's animals first, in identity order), each run in its state's color.
 * Empty when the session has no states.
 * @returns {Array<{animal, color, lanes, segments, group}>}
 */
export function stateTimelineRows(session) {
    var layer = shownLayer(session);
    if (!layer || !layer.animals.size) return [];
    var c = _rowCache.get(layer);
    if (!c || c.version !== layer.version) {
        var segs = new Map();
        layer.animals.forEach(function (runs, animal) {
            var list = new Array(runs.length / 3);
            for (var i = 0, j = 0; i + 2 < runs.length; i += 3, j++) {
                list[j] = { start: runs[i], end: runs[i + 1], color: stateColor(runs[i + 2]), alpha: STATE_ALPHA, lane: 0 };
            }
            segs.set(animal, list);
        });
        c = { version: layer.version, segs: segs };
        _rowCache.set(layer, c);
    }
    var known = sessionAnimals(session);
    var order = known.filter(function (a) { return c.segs.has(a); })
        .concat(layerAnimals(layer).filter(function (a) { return known.indexOf(a) < 0; }));
    return order.map(function (a) {
        return { animal: a, color: animalColor(session, a), lanes: 1, segments: c.segs.get(a), group: STATES_GROUP_NAME };
    });
}

// ----------------------------------------------------------------------------
// The section
// ----------------------------------------------------------------------------

var _summaryCache = null;

/** Usage of the shown layer across the scope, cached while no layer changed. */
function summaryOf(items) {
    var key = items.map(function (it) { return it.layer.name + '@' + it.layer.version; }).join('|');
    var layers = items.map(function (it) { return it.layer; });
    if (_summaryCache && _summaryCache.key === key && _summaryCache.layers.every(function (l, i) { return l === layers[i]; })) return _summaryCache.value;
    var value = stateSummary(layers);
    _summaryCache = { key: key, layers: layers, value: value };
    return value;
}

/** Behaviors a state can be promoted to: one animal's, lasting (not a point, not social). */
function promotable() {
    return behaviors().filter(function (b) { return b.kind !== 'point' && !b.social; });
}

/**
 * The labels' verdict on a state: for one behavior, or the best-matching one
 * (the highest share present, smoothed toward 1/2 by the frames behind it).
 */
function labelCell(row, behaviorId) {
    if (!row) return { text: '–', title: 'No labelled frames in it' };
    var pick = null;
    if (behaviorId) {
        var c = row.get(behaviorId);
        if (c && c.present + c.absent) pick = { id: behaviorId, c: c };
    } else {
        var best = -1;
        row.forEach(function (c2, id) {
            var lab = c2.present + c2.absent;
            if (!lab || !behaviorById(id)) return;
            var s = (c2.present + 1) / (lab + 2);
            if (s > best) { best = s; pick = { id: id, c: c2 }; }
        });
    }
    if (!pick) return { text: '–', title: 'No labelled frames in it' };
    var b = behaviorById(pick.id), lab = pick.c.present + pick.c.absent;
    return {
        text: (behaviorId ? '' : (b ? b.name : '?') + ' ') + Math.round(100 * pick.c.present / lab) + '% (' + n(pick.c.present) + '/' + n(lab) + ')',
        title: n(pick.c.present) + ' of the ' + n(lab) + ' frames labelled either way in it are "' + (b ? b.name : '?') + '"',
    };
}

/** Render the States section for the active session (called by the Behavior tab's refresh). */
export function renderStatesSection(session) {
    var host = document.getElementById('behaviorStatesBody');
    var count = document.getElementById('behaviorStatesCount');
    if (!host) return;
    var tools = '<div class="behavior-tools">' +
        '<button class="panel-btn" data-act="kpms-export" title="Write the poses and settings keypoint-MoSeq needs, with the calls to run in Python">Export for keypoint-MoSeq…</button>' +
        '<button class="panel-btn" data-act="kpms-import" title="Bring keypoint-MoSeq\'s syllables back: results.h5 or its CSV results">Import results…</button>' +
        '<button class="panel-btn" data-act="discover" title="Find clusters in the poses with no labels, in the browser, or split a labelled behavior">Discover…</button></div>';
    var layer = session ? shownLayer(session) : null;
    if (!layer) {
        if (count) count.textContent = '0';
        host.innerHTML = tools + '<p class="table-empty">No states yet. Discover clusters here, or export the poses for keypoint-MoSeq, fit it in Python, then import its results.</p>';
        return;
    }
    if (ui.layer !== layer.name) ui.selected = null;
    ui.layer = layer.name;
    var word = stateWord(layer), found = layer.source === 'discovery' ? discoveryResult(session, layer.name) : null;
    var others = sessionsOf().filter(function (s) { return s !== session && stateLayer(s, layer.name); }).length;
    var all = ui.scope === 'all' && others > 0;
    var items = scopeItems(session, layer.name, all);
    var sum = summaryOf(items);
    if (count) count.textContent = n(sum.states.length);
    var enr = stateEnrichment(items);
    var labelled = new Set();
    enr.forEach(function (row) { row.forEach(function (c, id) { if (c.present + c.absent) labelled.add(id); }); });
    if (ui.behavior && !behaviorById(ui.behavior)) ui.behavior = '';
    var layers = stateLayers(session);
    var animals = new Set();
    items.forEach(function (it) { it.layer.animals.forEach(function (_, a) { animals.add(a); }); });
    var runs = sum.states.reduce(function (t, s) { return t + s.runs; }, 0);
    var html = tools + '<div class="behavior-states-head">' +
        '<label>States <select data-act="layer">' + layers.map(function (l) {
            return '<option value="' + esc(l.name) + '"' + (l === layer ? ' selected' : '') + '>' + esc(l.name) + '</option>';
        }).join('') + '</select></label>' +
        (others ? '<label>Counted in <select data-act="scope"><option value="session">this session</option><option value="all"' + (all ? ' selected' : '') + '>' +
            n(others + 1) + ' sessions</option></select></label>' : '') +
        '<button class="panel-btn panel-btn-danger" data-act="remove-layer" title="Forget these states (labels and predictions stay)">Remove…</button></div>';
    html += '<p class="behavior-fps-note">' + plural(sum.states.length, word) + ' · ' + plural(runs, 'run') + ' · ' + plural(animals.size, 'animal') +
        ' · ' + (sum.frames / fps()).toFixed(1) + ' s with a ' + word + ', at ' + fps() + ' fps</p>';
    if (found) html += discoveryMapHtml(found);
    html += '<label class="behavior-states-labels">Labels <select data-act="enrich"><option value="">best match</option>' +
        behaviors().map(function (b) {
            return '<option value="' + esc(b.id) + '"' + (b.id === ui.behavior ? ' selected' : '') + '>' + esc(b.name) + (labelled.has(b.id) ? '' : ' (no labels here)') + '</option>';
        }).join('') + '</select></label>';
    var rare = sum.states.filter(function (s) { return s.share < RARE_SHARE; });
    var shown = ui.showRare ? sum.states : sum.states.filter(function (s) { return s.share >= RARE_SHARE; });
    Array.from(ui.picked).forEach(function (k) { if (!sum.states.some(function (s) { return s.state === k; })) ui.picked.delete(k); });
    var isCluster = word === 'cluster';
    html += '<p class="behavior-states-legend">Each ' + word + ': its share of the frames · its median run · what the labels say in it. ' +
        (isCluster ? 'Click its number for its next example, longest first.' : 'Click its number for its next run.') + '</p><div class="behavior-states-list">';
    shown.forEach(function (s) {
        var cell = labelCell(enr.get(s.state), ui.behavior);
        var marks = found && found.clusters[s.state] ? 'found again in a second layout: ' + Math.round(100 * found.clusters[s.state].repeat) + '% · ' + plural(found.clusters[s.state].episodes, 'separate stretch', 'separate stretches') + ' · ' + standsOut(found.clusters[s.state]) : '';
        html += '<div class="behavior-state-row' + (ui.selected === s.state ? ' is-selected' : '') + '" data-state="' + s.state + '">' +
            '<input type="checkbox" data-act="pick" aria-label="Pick ' + word + ' ' + s.state + '"' + (ui.picked.has(s.state) ? ' checked' : '') + '>' +
            '<button class="behavior-state-go" data-act="next" title="' + (isCluster ? 'The next example of cluster ' + s.state + ', longest first'
                : 'The next run of syllable ' + s.state + ' after the playhead') + '">' +
            '<span class="behavior-swatch" style="background:' + stateColor(s.state) + '"></span>' + s.state + ' ▸</button>' +
            '<input type="text" class="behavior-state-name" data-act="name" maxlength="60" placeholder="name…" aria-label="Name of ' + word + ' ' + s.state +
            '" value="' + esc(stateName(layer, s.state)) + '">' +
            '<span class="behavior-state-stats" title="Share of the frames with a ' + word + ' · median run length">' + pct(s.share) + ' · ' +
            (s.medianFrames / fps()).toFixed(2) + ' s</span>' +
            '<span class="behavior-state-labels" title="' + esc(cell.title) + '">' + esc(cell.text === '–' ? 'no labels in it' : cell.text) + '</span>' +
            (marks ? '<span class="behavior-state-marks" title="How much of the cluster a second UMAP layout finds again; how many separate stretches of time its sampled bins come from; the features furthest from their mean in it, in standard deviations">' + esc(marks) + '</span>' : '') + '</div>';
    });
    html += '</div>';
    if (rare.length) {
        html += '<button class="panel-btn behavior-states-rare" data-act="rare">' + (ui.showRare ? 'Hide' : 'Show') + ' ' + plural(rare.length, 'rare ' + word) +
            ' (under 0.5% of frames)</button>';
    }
    var targets = promotable();
    if (ui.promoteTo !== NEW_BEHAVIOR && ui.promoteTo && !targets.some(function (b) { return b.id === ui.promoteTo; })) ui.promoteTo = '';
    if (!ui.promoteTo) ui.promoteTo = targets.length && !isCluster ? targets[0].id : NEW_BEHAVIOR;
    var focus = document.activeElement && host.contains(document.activeElement) && document.activeElement.dataset.act === 'name'
        ? document.activeElement.closest('[data-state]').dataset.state : null;
    html += '<div class="behavior-states-promote">Promote ' + plural(ui.picked.size, 'checked ' + word) + ' to predictions of <select data-act="promote-to">' +
        '<option value="' + NEW_BEHAVIOR + '"' + (ui.promoteTo === NEW_BEHAVIOR ? ' selected' : '') + '>a new behavior</option>' + targets.map(function (b) {
            return '<option value="' + esc(b.id) + '"' + (b.id === ui.promoteTo ? ' selected' : '') + '>' + esc(b.name) + '</option>';
        }).join('') + '</select> <button class="panel-btn" data-act="promote"' + (ui.picked.size ? '' : ' disabled') +
        ' title="Their runs become predicted bouts of that behavior, to Accept or Reject under Predictions. A new behavior takes their names.">Promote</button></div>';
    host.innerHTML = html;
    if (found) placeDiscoveryMap(host, found);
    // A repaint while a name is being typed (Tab to the next one) keeps the field focused.
    var again = focus != null && host.querySelector('.behavior-state-row[data-state="' + focus + '"] input[data-act="name"]');
    if (again) again.focus();
}

/** "Nose height +2.1 · centroid speed −1.4": what stands out in a cluster. */
function standsOut(cluster) {
    return (cluster.top || []).slice(0, 3).map(function (f) {
        return f.name + ' ' + (f.z >= 0 ? '+' : '−') + Math.abs(f.z).toFixed(1);
    }).join(' · ');
}

/** The map of a run in this page, with what it read and how well the forests repeat it. */
function discoveryMapHtml(found) {
    var r = found.report || {};
    var bits = [n(r.bins) + (r.sampled ? ' of ' + n(r.usable) : '') + ' bins of 100 ms', n(r.features) + ' features', n(r.pcaDims) + ' principal components (' + Math.round(100 * r.explained) + '% of the variance)',
                n(r.neighbors) + ' neighbors', 'smallest cluster ' + n(r.minClusterSize) + ' bins'];
    if (r.dropped) bits.push(plural(r.dropped, 'cluster') + ' dropped (not found again in a second layout, or from too few separate stretches of time)');
    if (r.noiseShare) bits.push(Math.round(100 * r.noiseShare) + '% of the bins in no cluster (gray)');
    if (Number.isFinite(r.heldOutAgreement)) bits.push('the forests repeat the clusters on ' + Math.round(100 * r.heldOutAgreement) + '% of bins they never saw');
    return '<div class="behavior-discovery-map"><div class="behavior-discovery-canvas"></div>' +
        '<p class="behavior-fps-note">' + esc(bits.join(' · ')) + '</p></div>';
}

var _map = { canvas: null, found: null, selected: null, width: 0, height: 0, dpr: 0 };

/** Put the map's canvas back in the section (it keeps its picture), redrawing it only if it would differ. */
function placeDiscoveryMap(host, found) {
    var slot = host.querySelector('.behavior-discovery-canvas');
    if (!slot) return;
    if (!_map.canvas) {
        _map.canvas = document.createElement('canvas');
        _map.canvas.setAttribute('aria-label', 'Every clustered bin laid out by UMAP, in its cluster\'s color. Press a point to go to that moment.');
    }
    var canvas = _map.canvas;
    slot.appendChild(canvas);
    var dpr = window.devicePixelRatio || 1;
    if (_map.found === found && _map.selected === ui.selected && _map.width === canvas.clientWidth && _map.height === canvas.clientHeight && _map.dpr === dpr) return;
    _map.found = found; _map.selected = ui.selected; _map.width = canvas.clientWidth; _map.height = canvas.clientHeight; _map.dpr = dpr;
    drawDiscoveryMap(canvas, found, ui.selected);
}

// ----------------------------------------------------------------------------
// Actions
// ----------------------------------------------------------------------------

/** Name a state in every session's layer of this name. */
function nameState(name, k, label) {
    var changed = 0;
    sessionsOf().forEach(function (s) { if (setStateName(s, name, k, label)) changed++; });
    return changed;
}

/** Seek to the next run of state k (any animal) after the playhead, wrapping. */
function nextRun(session, layer, k) {
    var from = state.currentFrame + 1;
    var best = null, first = null;
    layer.animals.forEach(function (runs, animal) {
        for (var i = 0; i + 2 < runs.length; i += 3) {
            if (runs[i + 2] !== k) continue;
            var r = { animal: animal, start: runs[i], end: runs[i + 1] };
            if (!first || r.start < first.start) first = r;
            if (r.start >= from) { if (!best || r.start < best.start) best = r; break; }
        }
    });
    var hit = best || first;
    if (!hit) { setStatus('Syllable ' + k + ' does not occur in this session', 'warning'); return; }
    goToBehaviorFrame(hit.start);
    var len = hit.end - hit.start + 1;
    setStatus('Syllable ' + stateLabel(layer, k) + ': ' + hit.animal + ', frames ' + fr(hit.start) + '–' + fr(hit.end) +
        ' (' + (len / fps()).toFixed(2) + ' s)' + (best ? '' : ' · back at the first'));
}

/** Promote the checked states of the shown layer to predictions of `ui.promoteTo`. */
async function promote(session) {
    var layer = shownLayer(session);
    var picked = Array.from(ui.picked).sort(function (x, y) { return x - y; });
    if (!layer || !picked.length) return;
    var word = stateWord(layer);
    var b = ui.promoteTo === NEW_BEHAVIOR ? behaviorFromStates(layer, picked, word) : behaviorById(ui.promoteTo);
    if (!b) return;
    var others = sessionsOf().filter(function (s) { return s !== session && stateLayer(s, layer.name); }).length;
    var items = scopeItems(session, layer.name, ui.scope === 'all' && others > 0);
    var source = 'states:' + layer.name;
    var replacing = items.filter(function (it) { var r = behaviorRunOf(it.session, b.id); return r && r.bouts.length && r.source !== source; });
    if (replacing.length) {
        var from = new Set(replacing.map(function (it) {
            var r = behaviorRunOf(it.session, b.id);
            return r.source === 'rule' ? 'its detector' : runStateLayer(r) ? 'other states' : 'its classifier';
        }));
        var ok = await behaviorConfirm({
            id: 'statesPromoteConfirm', title: 'Replace "' + b.name + '" predictions?',
            message: '"' + b.name + '" already has predictions from ' + Array.from(from).join(' and ') + ' in ' + plural(replacing.length, 'session') +
                '. A behavior keeps one set of predictions: promoting replaces them. Labels stay as they are.',
            confirmLabel: 'Replace',
        });
        if (!ok) return;
    }
    var enr = stateEnrichment(items, [b.id]);
    var share = function (k) {
        var c = enr.get(k) && enr.get(k).get(b.id);
        var lab = c ? c.present + c.absent : 0;
        return lab ? c.present / lab : NaN;
    };
    var total = 0;
    items.forEach(function (it) {
        var bouts = statesToBouts(it.layer, picked).map(function (bt) {
            return { animal: bt.animal, target: '', start: bt.start, end: bt.end, score: share(bt.state) };
        });
        var mode = hasIdentified3d(it.session) ? '3d' : '2d';
        setBehaviorRun(it.session, {
            behaviorId: b.id, source: source, states: picked, cfg: runConfigPrint(b, it.session, { source: source, states: picked }),
            fps: sessionFps(it.session), mode: mode, camera: mode === '2d' ? pick2dCamera(it.session) : null, bouts: bouts,
        });
        total += bouts.length;
    });
    markDirty();
    requestBehaviorRepaint({ predictions: true });
    var pred = document.getElementById('behaviorPredictionsSection');
    if (pred && !pred.open) pred.open = true;
    setStatus('Promoted ' + word + (picked.length === 1 ? ' ' : 's ') + picked.join(', ') + ' to predictions of "' + b.name + '": ' +
        plural(total, 'bout') + (items.length > 1 ? ' in ' + plural(items.length, 'session') : '') + ' · Review them under Predictions', 'success');
}

/**
 * A new ethogram entry for some states, named after them: their names, else
 * "Cluster 3". A behavior of that name already there is asked for by name instead.
 * @returns {Object|null} the behavior
 */
function behaviorFromStates(layer, picked, word) {
    var name = picked.map(function (k) { return stateName(layer, k) || capital(word) + ' ' + k; }).join(' + ').slice(0, 60);
    var there = behaviorByName(name);
    if (there) {
        if (there.kind === 'point' || there.social) { setStatus('"' + there.name + '" is not a lasting behavior of one animal: name the ' + word + ' differently', 'warning'); return null; }
        ui.promoteTo = there.id;
        return there;
    }
    var made = addBehavior({ name: name, kind: 'state', description: 'From ' + word + (picked.length === 1 ? ' ' : 's ') + picked.join(', ') + ' of "' + layer.name + '"' });
    if (made.error) { setStatus(made.error, 'warning'); return null; }
    ui.promoteTo = made.behavior.id;
    requestBehaviorRepaint({ ethogram: true });
    return made.behavior;
}

async function removeLayer(session) {
    var layer = shownLayer(session);
    if (!layer) return;
    var others = sessionsOf().filter(function (s) { return s !== session && stateLayer(s, layer.name); });
    var all = ui.scope === 'all' && others.length > 0;
    var ok = await behaviorConfirm({
        id: 'statesRemoveConfirm', title: 'Remove the states "' + layer.name + '"?',
        message: 'Their ' + stateWord(layer) + 's and names leave ' + (all ? 'all ' + n(others.length + 1) + ' sessions' : 'this session') +
            '. Labels stay; predictions promoted from them go stale.',
        confirmLabel: 'Remove',
    });
    if (!ok) return;
    (all ? [session].concat(others) : [session]).forEach(function (s) { removeStateLayer(s, layer.name); });
    ui.picked.clear();
    ui.selected = null;
    clearStateExample();
    setStatus('Removed the states "' + layer.name + '"');
}

function wireStates(host) {
    host.addEventListener('click', function (e) {
        var btn = e.target.closest('button');
        if (!btn) return;
        var session = state.session, act = btn.dataset.act;
        if (act === 'kpms-export') showKpmsExportModal();
        else if (act === 'kpms-import') showKpmsImportModal();
        else if (act === 'discover') showBehaviorDiscoveryModal();
        if (!session) return;
        if (act === 'next') {
            var row = btn.closest('[data-state]');
            var layer = shownLayer(session);
            if (!row || !layer) return;
            var k0 = Number(row.dataset.state);
            if (layer.source === 'discovery') {
                ui.selected = k0;
                nextStateExample(session, layer, k0, 'Cluster');
                requestBehaviorRepaint({ states: true });
            } else nextRun(session, layer, k0);
        } else if (act === 'promote') promote(session);
        else if (act === 'remove-layer') removeLayer(session);
        else if (act === 'rare') { ui.showRare = !ui.showRare; requestBehaviorRepaint({ states: true }); }
    });
    host.addEventListener('change', function (e) {
        var t = e.target, act = t.dataset.act, session = state.session;
        if (!session) return;
        if (act === 'layer') { ui.layer = t.value; ui.picked.clear(); ui.selected = null; clearStateExample(); requestBehaviorRepaint({ states: true }); }
        else if (act === 'scope') { ui.scope = t.value === 'all' ? 'all' : 'session'; requestBehaviorRepaint({ states: true }); }
        else if (act === 'enrich') { ui.behavior = t.value; requestBehaviorRepaint({ states: true }); }
        else if (act === 'promote-to') { ui.promoteTo = t.value; }
        else if (act === 'pick') {
            var k = Number(t.closest('[data-state]').dataset.state);
            if (t.checked) ui.picked.add(k); else ui.picked.delete(k);
            requestBehaviorRepaint({ states: true });
        } else if (act === 'name') {
            var layer = shownLayer(session);
            if (!layer) return;
            var k2 = Number(t.closest('[data-state]').dataset.state);
            var wd = stateWord(layer);
            if (nameState(layer.name, k2, t.value)) setStatus(t.value.trim() ? 'Named ' + wd + ' ' + k2 + ' "' + t.value.trim() + '"' : capital(wd) + ' ' + k2 + ' has no name now');
        }
    });
    // Enter in a name commits it (the change event) without leaving the field.
    host.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && e.target.dataset.act === 'name') { e.preventDefault(); e.target.blur(); }
    });
    // A press on the map goes to that bin's moment, in its session. On the press, not
    // the click: a repaint between press and release would lose the click.
    host.addEventListener('pointerdown', async function (e) {
        var canvas = e.target.closest ? e.target.closest('.behavior-discovery-map canvas') : null;
        var session = state.session;
        if (!canvas || !session || e.button !== 0) return;
        var layer = shownLayer(session);
        var found = layer && discoveryResult(session, layer.name);
        if (!found) return;
        var box = canvas.getBoundingClientRect();
        var bin = discoveryBinAt(canvas, found, e.clientX - box.left, e.clientY - box.top);
        if (!bin) return;
        ui.selected = bin.state >= 0 ? bin.state : null;
        var here = session.name === bin.session || sessionsOf().length < 2;
        if (!here && !(await openSessionByName(bin.session))) { setStatus('Could not open ' + bin.session, 'warning'); return; }
        showDiscoveryBin(state.session, shownLayer(state.session) || layer, bin, 'Cluster');
        requestBehaviorRepaint({ states: true });
    });
}

/**
 * Wire the section once (from the Behavior tab's setup): its events, its ⓘ,
 * and what a change to the states does — the project is dirty, and the tab
 * and the timeline repaint.
 */
export function setupBehaviorStates() {
    var host = document.getElementById('behaviorStatesBody');
    if (host) wireStates(host);
    var info = document.getElementById('behaviorStatesInfo');
    if (info) { info.innerHTML = ICON_INFO; setInfoTip(info, STATES_INFO); }
    // A map drawn while folded has no size: draw it when the section opens.
    var section = document.getElementById('behaviorStatesSection');
    if (section) section.addEventListener('toggle', function () { if (section.open) requestBehaviorRepaint({ states: true }); });
    onStatesChange(function (session, info2) {
        // A layer just imported or replaced is the one to show.
        if (info2 && info2.layer && !info2.names && !info2.removed) ui.layer = info2.layer;
        markDirty();
        requestBehaviorRepaint({ states: true });
    });
}
