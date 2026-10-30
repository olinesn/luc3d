// ui/behavior-states.js — behavioral states in the app: the Behavior tab's
// States section and the timeline's "States" rows (pose/behavior-states.js
// has the layers; ui/behavior-kpms-modal.js brings them in).
//
// The States section shows one layer — keypoint-MoSeq's syllables — of the
// active session, or of every session that has a layer of that name:
//   * a row per state: its color, number and name (editable: names are
//     written into every session's layer of that name), its share of the
//     frames, its median run length, and how the labels fall in it —
//     P(behavior | state) with the frames behind it, for one behavior or the
//     best-matching one; unlabelled frames count for neither;
//   * Next ▸ per state: the next run of it after the playhead;
//   * Promote: the checked states become one behavior's predictions (source
//     `states:<layer>`, pose/behavior-predictions.js), reviewed in the
//     Predictions section like the detectors' — each bout scored by the
//     share of its main state's labelled frames that are that behavior;
//   * Remove the layer (this session, or all), with a confirm.
// States below 0.5% of the frames are folded away until asked for.
//
// Timeline: `stateTimelineRows` gives one row per animal of the shown layer,
// each run in its state's color (`stateColor`), cached per layer version: a
// repaint never walks frames.
//
// Depends on: ui/app-state.js, ui/behavior-coding.js, ui/behavior-confirm.js,
//             ui/behavior-detect.js, ui/behavior-kpms-modal.js, ui/info-tip.js,
//             ui/plane-definition.js (the ⓘ icon), pose/behavior-states.js,
//             pose/behavior-predictions.js, pose/behavior-ethogram.js, pose/behavior-tensor.js,
//             import-export/save-load.js

import { state } from './app-state.js';
import { sessionAnimals, animalColor, requestBehaviorRepaint, goToBehaviorFrame } from './behavior-coding.js';
import { behaviorConfirm } from './behavior-confirm.js';
import { sessionFps } from './behavior-detect.js';
import { showKpmsExportModal, showKpmsImportModal } from './behavior-kpms-modal.js';
import { setInfoTip } from './info-tip.js';
import { ICON_INFO } from './plane-definition.js';
import {
    stateLayers, stateLayer, removeStateLayer, setStateName, stateName, stateLabel, stateSummary, stateEnrichment,
    statesToBouts, stateColor, layerAnimals, onStatesChange,
} from '../pose/behavior-states.js';
import { setBehaviorRun, behaviorRunOf, runConfigPrint, runStateLayer } from '../pose/behavior-predictions.js';
import { behaviors, behaviorById } from '../pose/behavior-ethogram.js';
import { hasIdentified3d, pick2dCamera } from '../pose/behavior-tensor.js';
import { markDirty, setStatus } from '../import-export/save-load.js';

/** The gutter name of the timeline's states group. */
export var STATES_GROUP_NAME = 'States';
/** States under this share of the frames are folded away. */
var RARE_SHARE = 0.005;
var STATE_ALPHA = 0.9;

var STATES_INFO = 'Behavioral states found without labels — keypoint-MoSeq\'s syllables — and how your labels fall in each ' +
    '(unlabelled frames count for neither). Name them, or promote some to predictions of a behavior to review.';

/** What the section shows (browser memory only). */
var ui = { layer: '', scope: 'session', behavior: '', picked: new Set(), showRare: false, promoteTo: '' };

function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function n(x) { return Number(x || 0).toLocaleString('en-US'); }
function fr(f) { return (f + 1).toLocaleString('en-US'); }
function fps() { return state.fps > 0 ? state.fps : 30; }
function plural(k, one, many) { return n(k) + ' ' + (k === 1 ? one : (many || one + 's')); }
function pct(x) { return x >= 0.995 || x === 0 ? Math.round(100 * x) + '%' : x < 0.01 ? (100 * x).toFixed(1) + '%' : Math.round(100 * x) + '%'; }

function sessionsOf() {
    return (state.sessions && state.sessions.length) ? state.sessions : (state.session ? [state.session] : []);
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
        '<button class="panel-btn" data-act="kpms-import" title="Bring keypoint-MoSeq\'s syllables back: results.h5 or its CSV results">Import results…</button></div>';
    var layer = session ? shownLayer(session) : null;
    if (!layer) {
        if (count) count.textContent = '0';
        host.innerHTML = tools + '<p class="table-empty">No states yet. Export the poses for keypoint-MoSeq, fit it in Python, then import its results.</p>';
        return;
    }
    ui.layer = layer.name;
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
    html += '<p class="behavior-fps-note">' + plural(sum.states.length, 'syllable') + ' · ' + plural(runs, 'run') + ' · ' + plural(animals.size, 'animal') +
        ' · ' + (sum.frames / fps()).toFixed(1) + ' s with a syllable, at ' + fps() + ' fps</p>';
    html += '<label class="behavior-states-labels">Labels <select data-act="enrich"><option value="">best match</option>' +
        behaviors().map(function (b) {
            return '<option value="' + esc(b.id) + '"' + (b.id === ui.behavior ? ' selected' : '') + '>' + esc(b.name) + (labelled.has(b.id) ? '' : ' (no labels here)') + '</option>';
        }).join('') + '</select></label>';
    var rare = sum.states.filter(function (s) { return s.share < RARE_SHARE; });
    var shown = ui.showRare ? sum.states : sum.states.filter(function (s) { return s.share >= RARE_SHARE; });
    Array.from(ui.picked).forEach(function (k) { if (!sum.states.some(function (s) { return s.state === k; })) ui.picked.delete(k); });
    html += '<p class="behavior-states-legend">Each syllable: its share of the frames · its median run · what the labels say in it. ' +
        'Click its number for its next run.</p><div class="behavior-states-list">';
    shown.forEach(function (s) {
        var cell = labelCell(enr.get(s.state), ui.behavior);
        html += '<div class="behavior-state-row" data-state="' + s.state + '">' +
            '<input type="checkbox" data-act="pick" aria-label="Pick syllable ' + s.state + '"' + (ui.picked.has(s.state) ? ' checked' : '') + '>' +
            '<button class="behavior-state-go" data-act="next" title="The next run of syllable ' + s.state + ' after the playhead">' +
            '<span class="behavior-swatch" style="background:' + stateColor(s.state) + '"></span>' + s.state + ' ▸</button>' +
            '<input type="text" class="behavior-state-name" data-act="name" maxlength="60" placeholder="name…" aria-label="Name of syllable ' + s.state +
            '" value="' + esc(stateName(layer, s.state)) + '">' +
            '<span class="behavior-state-stats" title="Share of the frames with a syllable · median run length">' + pct(s.share) + ' · ' +
            (s.medianFrames / fps()).toFixed(2) + ' s</span>' +
            '<span class="behavior-state-labels" title="' + esc(cell.title) + '">' + esc(cell.text === '–' ? 'no labels in it' : cell.text) + '</span></div>';
    });
    html += '</div>';
    if (rare.length) {
        html += '<button class="panel-btn behavior-states-rare" data-act="rare">' + (ui.showRare ? 'Hide' : 'Show') + ' ' + plural(rare.length, 'rare syllable') +
            ' (under 0.5% of frames)</button>';
    }
    var targets = promotable();
    if (ui.promoteTo && !targets.some(function (b) { return b.id === ui.promoteTo; })) ui.promoteTo = '';
    if (!ui.promoteTo && targets.length) ui.promoteTo = targets[0].id;
    var focus = document.activeElement && host.contains(document.activeElement) && document.activeElement.dataset.act === 'name'
        ? document.activeElement.closest('[data-state]').dataset.state : null;
    html += '<div class="behavior-states-promote">' + (targets.length
        ? 'Promote ' + plural(ui.picked.size, 'checked syllable') + ' to predictions of <select data-act="promote-to">' + targets.map(function (b) {
            return '<option value="' + esc(b.id) + '"' + (b.id === ui.promoteTo ? ' selected' : '') + '>' + esc(b.name) + '</option>';
        }).join('') + '</select> <button class="panel-btn" data-act="promote"' + (ui.picked.size ? '' : ' disabled') +
            ' title="Their runs become predicted bouts of that behavior, to Accept or Reject under Predictions">Promote</button>'
        : 'Add a behavior that lasts (Edit ethogram…) to promote syllables to its predictions.') + '</div>';
    host.innerHTML = html;
    // A repaint while a name is being typed (Tab to the next one) keeps the field focused.
    var again = focus != null && host.querySelector('.behavior-state-row[data-state="' + focus + '"] input[data-act="name"]');
    if (again) again.focus();
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
    var layer = shownLayer(session), b = behaviorById(ui.promoteTo);
    var picked = Array.from(ui.picked).sort(function (x, y) { return x - y; });
    if (!layer || !b || !picked.length) return;
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
    setStatus('Promoted syllable' + (picked.length === 1 ? ' ' : 's ') + picked.join(', ') + ' to predictions of "' + b.name + '": ' +
        plural(total, 'bout') + (items.length > 1 ? ' in ' + plural(items.length, 'session') : '') + ' · Review them under Predictions', 'success');
}

async function removeLayer(session) {
    var layer = shownLayer(session);
    if (!layer) return;
    var others = sessionsOf().filter(function (s) { return s !== session && stateLayer(s, layer.name); });
    var all = ui.scope === 'all' && others.length > 0;
    var ok = await behaviorConfirm({
        id: 'statesRemoveConfirm', title: 'Remove the states "' + layer.name + '"?',
        message: 'Their syllables and names leave ' + (all ? 'all ' + n(others.length + 1) + ' sessions' : 'this session') +
            '. Labels stay; predictions promoted from them go stale.',
        confirmLabel: 'Remove',
    });
    if (!ok) return;
    (all ? [session].concat(others) : [session]).forEach(function (s) { removeStateLayer(s, layer.name); });
    ui.picked.clear();
    setStatus('Removed the states "' + layer.name + '"');
}

function wireStates(host) {
    host.addEventListener('click', function (e) {
        var btn = e.target.closest('button');
        if (!btn) return;
        var session = state.session, act = btn.dataset.act;
        if (act === 'kpms-export') showKpmsExportModal();
        else if (act === 'kpms-import') showKpmsImportModal();
        if (!session) return;
        if (act === 'next') {
            var row = btn.closest('[data-state]');
            var layer = shownLayer(session);
            if (row && layer) nextRun(session, layer, Number(row.dataset.state));
        } else if (act === 'promote') promote(session);
        else if (act === 'remove-layer') removeLayer(session);
        else if (act === 'rare') { ui.showRare = !ui.showRare; requestBehaviorRepaint({ states: true }); }
    });
    host.addEventListener('change', function (e) {
        var t = e.target, act = t.dataset.act, session = state.session;
        if (!session) return;
        if (act === 'layer') { ui.layer = t.value; ui.picked.clear(); requestBehaviorRepaint({ states: true }); }
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
            if (nameState(layer.name, k2, t.value)) setStatus(t.value.trim() ? 'Named syllable ' + k2 + ' "' + t.value.trim() + '"' : 'Syllable ' + k2 + ' has no name now');
        }
    });
    // Enter in a name commits it (the change event) without leaving the field.
    host.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && e.target.dataset.act === 'name') { e.preventDefault(); e.target.blur(); }
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
    onStatesChange(function (session, info2) {
        // A layer just imported or replaced is the one to show.
        if (info2 && info2.layer && !info2.names && !info2.removed) ui.layer = info2.layer;
        markDirty();
        requestBehaviorRepaint({ states: true });
    });
}
