// ui/behavior-rules-modal.js — Behavior ▸ Detection Settings… (also the
// toolbar Behavior menu's Settings…): what the detectors measure against —
// the body and the floor — and each behavior's detector.
//
// A detector is conditions on named features that must all hold, for at
// least a minimum time (pose/behavior-rules.js). For the condition being
// edited, the dialog shows that feature's histogram over the session (every
// animal, or every pair for a social behavior) with the threshold on it, and
// under it the bouts the whole detector gives, both live as the threshold
// moves. Templates fill in common detectors, or add their behavior to the
// ethogram.
//
// Every field applies as it is changed (like the ethogram editor), marks the
// project dirty, and makes that behavior's predictions stale until it is
// detected again. Body and floor settings are saved only once changed
// (`ethogram.features`). Esc or Close leave; one scroller (the dialog).
//
// Depends on: ui/app-state.js, ui/behavior-coding.js, ui/behavior-detect.js,
//             ui/info-tip.js, ui/loading-overlay.js, ui/plane-definition.js (the ⓘ icon),
//             ui/origin-definition.js, pose/behavior-ethogram.js,
//             pose/behavior-features.js, pose/behavior-rules.js,
//             import-export/save-load.js

import { state } from './app-state.js';
import { requestBehaviorRepaint } from './behavior-coding.js';
import {
    cachedBehaviorContext, refreshBehaviorContext, computeFeatureColumns, floorPlanes, detectAllFrames, combineMutual,
} from './behavior-detect.js';
import { setInfoTip } from './info-tip.js';
import { showLoading, hideLoading, yieldToPaint } from './loading-overlay.js';
import { ICON_INFO } from './plane-definition.js';
import { originState } from './origin-definition.js';
import {
    behaviors, behaviorById, behaviorByName, addBehavior, setBehaviorDetector, featureSettings, setFeatureSettings,
} from '../pose/behavior-ethogram.js';
import { featureNamesFor, resolveFeatureName } from '../pose/behavior-features.js';
import {
    DETECTOR_TEMPLATES, instantiateTemplate, normalizeDetector, detectorProblems, featureHistogram, scoreConditions,
    detectorBouts, orderedPairs,
} from '../pose/behavior-rules.js';
import { markDirty, setStatus } from '../import-export/save-load.js';

var HIST_BINS = 60;
var LIVE_DELAY_MS = 120;

var INFO = {
    body: 'Features are measured on the body: the head and tail base set its heading and length, and every length is ' +
        'divided by the body length (BL, the median head–tail base distance) so thresholds suit any animal and unit. ' +
        'Excluded nodes take no part; by default, the tail past its base.',
    floor: 'Heights are measured above the floor and turning in its plane. By default the Set Origin frame (+Z up) when ' +
        'one is set, else a plane fitted to the animals’ lowest points.',
    detectors: 'A behavior is detected on the frames where all its conditions hold. Detections closer than the gap are ' +
        'joined, and a bout needs at least the shortest bout of detected frames. Scores near 0.5 are the borderline ' +
        'frames the Predictions review lists first.',
};

function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function n(x) { return Number(x || 0).toLocaleString('en-US'); }
function num(v, d) {
    if (!Number.isFinite(v)) return '–';
    var a = Math.abs(v);
    return v.toFixed(d != null ? d : a >= 100 ? 0 : a >= 10 ? 1 : a >= 1 ? 2 : 3);
}

/** The open dialog, or null. */
var _dlg = null;

function changed(behaviorId) {
    markDirty();
    requestBehaviorRepaint({ ethogram: true, predictions: true });
    if (behaviorId) scheduleLive(behaviorId);
}

// ----------------------------------------------------------------------------
// Rendering
// ----------------------------------------------------------------------------

function infoButton(key) {
    return '<button type="button" class="plane-section-info bd-info" data-info="' + key + '" aria-label="About this section"></button>';
}

function bodyHtml(ctx) {
    var names = ctx.input.nodeNames;
    var body = ctx.body;
    var s = featureSettings() || {};
    var roleSel = function (role, label) {
        var cur = s.nodeRoles && s.nodeRoles[role];
        var guessed = body.guessed[role];
        return '<label class="bd-role">' + label + ' <select data-act="role" data-role="' + role + '">' +
            '<option value="">Auto (' + esc(guessed >= 0 ? names[guessed] : 'none found') + ')</option>' +
            names.map(function (nm) {
                return '<option value="' + esc(nm) + '"' + (nm === cur ? ' selected' : '') + '>' + esc(nm) + '</option>';
            }).join('') + '</select></label>';
    };
    var html = '<div class="bd-roles">' + roleSel('head', 'Head') + roleSel('tailBase', 'Tail base') + '</div>';
    html += '<div class="bd-exclude"><span class="bd-label">Excluded</span>' + names.map(function (nm, i) {
        var role = i === body.roles.head || i === body.roles.tailBase;
        return '<label class="bd-node' + (role ? ' is-role' : '') + '"><input type="checkbox" data-act="exclude" data-node="' + esc(nm) + '"' +
            (body.excluded.indexOf(i) >= 0 ? ' checked' : '') + (role ? ' disabled' : '') + '> ' + esc(nm) + '</label>';
    }).join('') + (s.exclude ? ' <button class="panel-btn" data-act="exclude-default" title="Back to the default: the tail past its base">Default</button>' : '') + '</div>';
    var unit = ctx.input.mode === '3d' ? '' : ' px';
    html += '<p class="bd-note">' + (ctx.input.animals.length
        ? 'Body length: ' + ctx.input.animals.map(function (a, i) { return esc(a) + ' ' + num(ctx.spec.bodyLength[i]) + unit; }).join(', ') +
          (ctx.input.mode === '3d' ? ' (calibration units)' : '') + '. '
        : 'No identified animals yet: features are measured per identity (run Track All). ') +
        (ctx.input.mode === '3d' ? 'From the 3D.' : 'No 3D: from camera ' + esc(ctx.input.camera || '?') + '’s 2D, without heights.') + '</p>';
    return html;
}

function floorHtml(ctx) {
    if (ctx.input.mode !== '3d') return '<p class="bd-note">No floor in 2D: features are measured in the camera image.</p>';
    var s = featureSettings() || {};
    var cur = s.floor ? (s.floor.source === 'plane' ? 'plane:' + s.floor.plane : s.floor.source) : 'auto';
    var opts = [['auto', 'Automatic: the Set Origin frame if one is set, else fitted'],
        ['origin', 'The Set Origin frame (+Z up)' + (originState.frame ? '' : ' — none set')],
        ['fit', 'Fitted to the animals’ lowest points']];
    floorPlanes().forEach(function (p) { opts.push(['plane:' + p.id, 'The plane "' + p.name + '"']); });
    if (cur.indexOf('plane:') === 0 && !opts.some(function (o) { return o[0] === cur; })) opts.push([cur, 'A plane that no longer exists']);
    var info = ctx.floorInfo;
    return '<label class="bd-floor">Floor <select data-act="floor">' + opts.map(function (o) {
        return '<option value="' + esc(o[0]) + '"' + (o[0] === cur ? ' selected' : '') + '>' + esc(o[1]) + '</option>';
    }).join('') + '</select></label>' +
        '<p class="bd-note">Using: ' + esc(info.label) + '.' + (info.warning ? ' <span class="bd-warn">' + esc(info.warning) + '.</span>' : '') + '</p>';
}

function templateOptions(social) {
    return DETECTOR_TEMPLATES.filter(function (t) { return social == null || t.social === social; }).map(function (t) {
        return '<option value="' + esc(t.id) + '">' + esc(t.name) + (t.social ? (t.mutual ? ' (↔)' : ' (→)') : '') + '</option>';
    }).join('');
}

function conditionRow(c, i, unit, focused) {
    return '<tr data-i="' + i + '"' + (focused ? ' class="is-focus"' : '') + '>' +
        '<td class="bd-feat-cell"><input type="text" data-f="feature" list="' + listIdFor(c.social) + '" value="' + esc(c.feature) +
        '" spellcheck="false" autocomplete="off" title="A feature name: type to search the list"></td>' +
        '<td><select data-f="op"><option value=">"' + (c.op === '>' ? ' selected' : '') + '>above</option>' +
        '<option value="<"' + (c.op === '<' ? ' selected' : '') + '>below</option></select></td>' +
        '<td><input type="number" step="any" data-f="value" value="' + esc(c.value) + '"></td>' +
        '<td class="bd-unit">' + esc(unit || '') + '</td>' +
        '<td><button class="bd-x" data-act="remove-cond" title="Remove this condition">✕</button></td></tr>';
}

function listIdFor(social) {
    return social ? 'bdFeatSocial' : 'bdFeatAnimal';
}

function cardHtml(ctx, b) {
    var d = b.detector ? normalizeDetector(b.detector) : null;
    var focus = (_dlg && _dlg.focus.get(b.id)) || 0;
    var html = '<div class="bd-card-head"><span class="behavior-swatch" style="background:' + esc(b.color) + '"></span>' +
        '<b>' + esc(b.name) + '</b>' + (b.social ? ' <span class="behavior-tag">' + (b.mutual ? '↔' : '→') + '</span>' : '') +
        (b.kind === 'point' ? ' <span class="behavior-tag" title="A point behavior: one frame per detected bout, where its score peaks">point</span>' : '') +
        '<span class="bd-spacer"></span>' +
        '<select data-act="template" title="Fill in a common detector"><option value="">Template…</option>' + templateOptions(!!b.social) + '</select>' +
        (d ? '<button class="panel-btn" data-act="remove-detector" title="Detect this behavior no more (its labels stay)">Remove</button>' : '') +
        '</div>';
    if (!d) {
        return html + '<p class="bd-note">No detector: labelled by hand only. <button class="panel-btn" data-act="add-detector">Add a detector</button></p>';
    }
    html += '<table class="bd-conds"><tbody>' + d.conditions.map(function (c, i) {
        var r = resolveFeatureName(ctx.catalog, c.feature, !!b.social);
        return conditionRow(Object.assign({ social: !!b.social }, c), i, r ? r.feature.unit : '', i === focus);
    }).join('') + '</tbody></table>' +
        '<div class="bd-row"><button class="panel-btn" data-act="add-cond">+ Condition</button>' +
        '<label title="A bout needs at least this long detected (the gaps it joined do not count)">Shortest bout <input type="number" min="0" step="0.05" data-f="minDuration" value="' +
        esc(d.minDuration) + '"> s</label>' +
        '<label title="Detected runs this close are joined into one bout first">Join gaps up to <input type="number" min="0" step="0.05" data-f="maxGap" value="' +
        esc(d.maxGap) + '"> s</label></div>' +
        '<div class="bd-hist"><canvas class="bd-hist-canvas" width="420" height="78"></canvas><div class="bd-hist-caption"></div></div>' +
        '<div class="bd-live" role="status"></div>';
    return html;
}

function detectorsHtml(ctx) {
    var list = behaviors();
    var html = '<div class="bd-row"><label>Add a behavior from a template <select data-act="new-from-template"><option value="">Choose…</option>' +
        templateOptions(null) + '</select></label></div>';
    if (!list.length) return html + '<p class="bd-note">No behaviors yet. Add one from a template, or in Edit Ethogram….</p>';
    return html + list.map(function (b) {
        return '<div class="bd-card" data-id="' + esc(b.id) + '">' + cardHtml(ctx, b) + '</div>';
    }).join('');
}

function datalistsHtml(ctx) {
    var opt = function (nm) { return '<option value="' + esc(nm) + '"></option>'; };
    return '<datalist id="bdFeatAnimal">' + featureNamesFor(ctx.catalog, false).map(opt).join('') + '</datalist>' +
        '<datalist id="bdFeatSocial">' + featureNamesFor(ctx.catalog, true).map(opt).join('') + '</datalist>';
}

function renderAll() {
    var dlg = _dlg;
    if (!dlg) return;
    var m = dlg.modal;
    m.querySelector('#bdBody').innerHTML = bodyHtml(dlg.ctx);
    m.querySelector('#bdFloor').innerHTML = floorHtml(dlg.ctx);
    m.querySelector('#bdDetectors').innerHTML = detectorsHtml(dlg.ctx);
    m.querySelector('#bdLists').innerHTML = datalistsHtml(dlg.ctx);
    behaviors().forEach(function (b) { if (b.detector) scheduleLive(b.id, true); });
}

function renderCard(id) {
    var dlg = _dlg;
    var el = dlg && dlg.modal.querySelector('.bd-card[data-id="' + id + '"]');
    var b = behaviorById(id);
    if (!el || !b) return;
    el.innerHTML = cardHtml(dlg.ctx, b);
    if (b.detector) scheduleLive(id, true);
}

// ----------------------------------------------------------------------------
// Live numbers
// ----------------------------------------------------------------------------

/** The detector a card shows right now (its inputs, uncommitted values included). */
function cardDetector(id) {
    var dlg = _dlg;
    var el = dlg && dlg.modal.querySelector('.bd-card[data-id="' + id + '"]');
    var b = behaviorById(id);
    if (!el || !b || !b.detector) return null;
    var d = normalizeDetector(b.detector);
    if (!d) return null;
    el.querySelectorAll('tr[data-i]').forEach(function (tr) {
        var i = +tr.dataset.i, c = d.conditions[i];
        if (!c) return;
        var v = Number(tr.querySelector('[data-f="value"]').value);
        if (Number.isFinite(v)) c.value = v;
        c.op = tr.querySelector('[data-f="op"]').value === '<' ? '<' : '>';
    });
    var md = Number(el.querySelector('[data-f="minDuration"]').value), mg = Number(el.querySelector('[data-f="maxGap"]').value);
    if (Number.isFinite(md) && md >= 0) d.minDuration = md;
    if (Number.isFinite(mg) && mg >= 0) d.maxGap = mg;
    return d;
}

function scheduleLive(id, soon) {
    var dlg = _dlg;
    if (!dlg) return;
    clearTimeout(dlg.timers.get(id));
    dlg.timers.set(id, setTimeout(function () { updateLive(id); }, soon ? 0 : LIVE_DELAY_MS));
}

/** Columns of every feature a detector reads, computing the missing ones (async). */
function columnsReady(names) {
    var dlg = _dlg;
    var missing = names.filter(function (nm) { return !dlg.columns.has(nm); });
    if (!missing.length) return true;
    var todo = missing.filter(function (nm) { return !dlg.computing.has(nm); });
    if (!todo.length) return false;
    todo.forEach(function (nm) { dlg.computing.add(nm); });
    var gen = dlg.gen;
    computeFeatureColumns(dlg.ctx, todo).then(function (res) {
        if (!_dlg || _dlg.gen !== gen) return;
        todo.forEach(function (nm) {
            _dlg.computing.delete(nm);
            if (res.animal[nm]) _dlg.columns.set(nm, { animal: res.animal[nm] });
            else if (res.pair[nm]) _dlg.columns.set(nm, { pair: res.pair[nm] });
            else _dlg.columns.set(nm, { missing: true });
        });
        behaviors().forEach(function (b) { if (b.detector) scheduleLive(b.id, true); });
    }).catch(function (e) {
        console.error('[behavior] feature columns failed', e);
        if (_dlg && _dlg.gen === gen) todo.forEach(function (nm) { _dlg.computing.delete(nm); _dlg.columns.set(nm, { missing: true }); });
    });
    return false;
}

/** One condition's column for an animal (who: self/actor/target) or a pair. */
function columnFor(cond, a, b) {
    var col = _dlg.columns.get(cond.name);
    if (!col) return null;
    if (cond.who === 'pair') return col.pair ? col.pair.get(a + '>' + b) || null : null;
    return col.animal ? col.animal[cond.who === 'target' ? b : a] || null : null;
}

function updateLive(id) {
    var dlg = _dlg;
    if (!dlg) return;
    var el = dlg.modal.querySelector('.bd-card[data-id="' + id + '"]');
    var b = behaviorById(id);
    if (!el || !b) return;
    var live = el.querySelector('.bd-live');
    var d = cardDetector(id);
    if (!d || !live) return;
    var ctx = dlg.ctx;
    var problems = detectorProblems(d, ctx.catalog, !!b.social);
    el.querySelectorAll('tr[data-i]').forEach(function (tr) {
        var c = d.conditions[+tr.dataset.i];
        var ok = c && resolveFeatureName(ctx.catalog, c.feature, !!b.social);
        tr.querySelector('[data-f="feature"]').classList.toggle('is-invalid', !ok);
        tr.querySelector('.bd-unit').textContent = ok ? ok.feature.unit : '';
    });
    if (problems.length) { live.innerHTML = '<span class="bd-warn">' + problems.map(esc).join('<br>') + '</span>'; drawHistogram(el, null); return; }
    var conds = d.conditions.map(function (c) {
        var r = resolveFeatureName(ctx.catalog, c.feature, !!b.social);
        return { name: r.feature.name, who: r.who, op: c.op, value: c.value, unit: r.feature.unit, label: c.feature };
    });
    var A = ctx.input.animals.length;
    if (!A || (b.social && A < 2)) {
        live.textContent = b.social ? 'Needs at least two identified animals.' : 'No identified animals to measure.';
        drawHistogram(el, null);
        return;
    }
    if (!columnsReady(conds.map(function (c) { return c.name; }))) {
        live.textContent = 'Measuring ' + conds.filter(function (c) { return !dlg.columns.has(c.name); }).map(function (c) { return c.label; }).join(', ') + '…';
        drawHistogram(el, null);
        return;
    }
    // The focused condition's histogram, pooled over every animal (or pair).
    var fi = Math.min(conds.length - 1, dlg.focus.get(id) || 0);
    var fc = conds[fi];
    var pooled = [];
    var col = dlg.columns.get(fc.name);
    if (col.animal) pooled = col.animal; else if (col.pair) pooled = Array.from(col.pair.values());
    drawHistogram(el, featureHistogram(pooled, HIST_BINS, fc.value), fc, fc.who === 'pair' ? 'pair' : 'animal');
    // The bouts the whole detector gives.
    var fps = ctx.fps, point = b.kind === 'point';
    var T = ctx.input.numFrames;
    var bouts = 0, frames = 0;
    var add = function (scores) {
        detectorBouts(scores, d, fps, point).forEach(function (bt) { bouts++; frames += bt[1] - bt[0] + 1; });
    };
    var score = function (a, t) {
        return scoreConditions(conds, conds.map(function (c) { return columnFor(c, a, t); }), T);
    };
    if (!b.social) for (var a = 0; a < A; a++) add(score(a, a));
    else if (!b.mutual) orderedPairs(A).forEach(function (pr) { add(score(pr[0], pr[1])); });
    else orderedPairs(A).forEach(function (pr) { if (pr[0] < pr[1]) add(combineMutual(score(pr[0], pr[1]), score(pr[1], pr[0]))); });
    var over = b.social ? (b.mutual ? A * (A - 1) / 2 : A * (A - 1)) : A;
    live.textContent = n(bouts) + ' bout' + (bouts === 1 ? '' : 's') + ' · ' + num(frames / fps, 1) + ' s in all, over ' +
        n(over) + ' ' + (b.social ? 'pair' : 'animal') + (over === 1 ? '' : 's');
}

function drawHistogram(card, h, cond, over) {
    var canvas = card.querySelector('.bd-hist-canvas');
    var cap = card.querySelector('.bd-hist-caption');
    if (!canvas) return;
    var g = canvas.getContext('2d');
    var W = canvas.width, H = canvas.height;
    g.clearRect(0, 0, W, H);
    if (!h || !cond) { if (cap) cap.textContent = ''; return; }
    var max = 0;
    for (var i = 0; i < h.counts.length; i++) max = Math.max(max, h.counts[i]);
    var x = function (v) { return (v - h.lo) / (h.hi - h.lo) * W; };
    var bw = W / h.counts.length;
    var tx = x(cond.value);
    var styles = getComputedStyle(card);
    var on = styles.getPropertyValue('--bd-hist-on').trim() || '#5fb3f9';
    var off = styles.getPropertyValue('--bd-hist-off').trim() || '#6b7280';
    for (var k = 0; k < h.counts.length; k++) {
        if (!h.counts[k]) continue;
        var mid = h.lo + (k + 0.5) * (h.hi - h.lo) / h.counts.length;
        var pass = cond.op === '<' ? mid <= cond.value : mid >= cond.value;
        g.fillStyle = pass ? on : off;
        var bh = Math.max(1, Math.sqrt(h.counts[k] / max) * (H - 14));
        g.fillRect(k * bw, H - 12 - bh, Math.max(1, bw - 1), bh);
    }
    g.fillStyle = '#f2a541';
    g.fillRect(Math.round(tx) - 1, 0, 2, H - 12);
    g.fillStyle = styles.getPropertyValue('--text-secondary').trim() || '#9aa0a6';
    g.font = '10px sans-serif';
    g.textAlign = 'left';
    g.fillText(num(h.lo), 1, H - 1);
    g.textAlign = 'right';
    g.fillText(num(h.hi), W - 1, H - 1);
    if (cap) {
        cap.textContent = cond.label + ' over ' + n(h.n) + ' ' + over + '-frames (square-root scale) · threshold ' + num(cond.value) +
            (cond.unit ? ' ' + cond.unit : '') + (h.below || h.above ? ' · ' + n(h.below + h.above) + ' outside the range shown' : '');
    }
}

// ----------------------------------------------------------------------------
// Edits
// ----------------------------------------------------------------------------

function updateSettings(fn) {
    var s = JSON.parse(JSON.stringify(featureSettings() || {}));
    fn(s);
    setFeatureSettings(s);
    _dlg.ctx = refreshBehaviorContext(_dlg.ctx);
    _dlg.columns = new Map();
    _dlg.computing = new Set();
    _dlg.gen++;
    renderAll();
    changed();
}

function commitDetector(id, mutate) {
    var b = behaviorById(id);
    if (!b) return;
    var d = normalizeDetector(b.detector) || { conditions: [], minDuration: 0, maxGap: 0 };
    mutate(d);
    // A name that is not a feature (yet) is kept, and flagged on its row.
    if (setBehaviorDetector(id, d.conditions.length ? d : null).error) setBehaviorDetector(id, null);
    changed(id);
}

function applyTemplate(id, templateId) {
    var t = DETECTOR_TEMPLATES.filter(function (x) { return x.id === templateId; })[0];
    if (!t) return;
    var r = instantiateTemplate(t, _dlg.ctx.body.roleNames);
    if (r.error) { setStatus(r.error, 'warning'); return; }
    setBehaviorDetector(id, r.detector);
    _dlg.focus.set(id, 0);
    renderCard(id);
    changed(id);
}

function newFromTemplate(templateId) {
    var t = DETECTOR_TEMPLATES.filter(function (x) { return x.id === templateId; })[0];
    if (!t) return;
    var r = instantiateTemplate(t, _dlg.ctx.body.roleNames);
    if (r.error) { setStatus(r.error, 'warning'); return; }
    var b = behaviorByName(t.name);
    if (b && b.detector) { setStatus('"' + b.name + '" already has a detector: use its Template… menu to replace it', 'warning'); return; }
    if (!b) {
        var res = addBehavior({ name: t.name, social: t.social, mutual: t.mutual, description: t.description });
        if (res.error) { setStatus(res.error, 'warning'); return; }
        b = res.behavior;
    }
    setBehaviorDetector(b.id, r.detector);
    renderAll();
    changed(b.id);
    setStatus('Added "' + b.name + '" with its template detector: check its threshold against the histogram', 'success');
}

function onClick(e) {
    var t = e.target.closest('button');
    if (!t || !_dlg) return;
    var card = t.closest('.bd-card');
    var id = card && card.dataset.id;
    var act = t.dataset.act;
    if (act === 'close') { closeDialog(); return; }
    if (act === 'detect') { closeDialog(); detectAllFrames(); return; }
    if (act === 'exclude-default') { updateSettings(function (s) { delete s.exclude; }); return; }
    if (!id) return;
    if (act === 'add-detector') {
        var b = behaviorById(id);
        var first = b.social ? 'centroid distance' : 'centroid speed · mean ±0.3 s';
        setBehaviorDetector(id, { conditions: [{ feature: first, op: b.social ? '<' : '>', value: b.social ? 1 : 0.5 }], minDuration: 0.2, maxGap: 0.1 });
        renderCard(id);
        changed(id);
    } else if (act === 'remove-detector') {
        setBehaviorDetector(id, null);
        renderCard(id);
        changed(id);
    } else if (act === 'add-cond') {
        commitDetector(id, function (d) { d.conditions.push({ feature: d.conditions.length ? d.conditions[d.conditions.length - 1].feature : 'centroid speed', op: '>', value: 0 }); });
        _dlg.focus.set(id, normalizeDetector(behaviorById(id).detector).conditions.length - 1);
        renderCard(id);
    } else if (act === 'remove-cond') {
        var i = +t.closest('tr').dataset.i;
        commitDetector(id, function (d) { d.conditions.splice(i, 1); });
        _dlg.focus.set(id, 0);
        renderCard(id);
    }
}

function onChange(e) {
    var t = e.target;
    if (!_dlg) return;
    var act = t.dataset.act;
    if (act === 'role') {
        updateSettings(function (s) {
            s.nodeRoles = s.nodeRoles || {};
            if (t.value) s.nodeRoles[t.dataset.role] = t.value; else delete s.nodeRoles[t.dataset.role];
        });
        return;
    }
    if (act === 'exclude') {
        var names = [];
        _dlg.modal.querySelectorAll('input[data-act="exclude"]').forEach(function (cb) { if (cb.checked) names.push(cb.dataset.node); });
        updateSettings(function (s) { s.exclude = names; });
        return;
    }
    if (act === 'floor') {
        updateSettings(function (s) {
            if (t.value === 'auto') delete s.floor;
            else if (t.value.indexOf('plane:') === 0) s.floor = { source: 'plane', plane: Number(t.value.slice(6)) };
            else s.floor = { source: t.value };
        });
        return;
    }
    if (act === 'new-from-template') { if (t.value) newFromTemplate(t.value); t.value = ''; return; }
    var card = t.closest('.bd-card');
    var id = card && card.dataset.id;
    if (!id) return;
    if (act === 'template') { if (t.value) applyTemplate(id, t.value); t.value = ''; return; }
    var f = t.dataset.f;
    if (!f) return;
    if (f === 'minDuration' || f === 'maxGap') {
        var v = Number(t.value);
        if (!Number.isFinite(v) || v < 0) { t.value = normalizeDetector(behaviorById(id).detector)[f]; return; }
        commitDetector(id, function (d) { d[f] = v; });
        return;
    }
    var tr = t.closest('tr[data-i]');
    if (!tr) return;
    var i = +tr.dataset.i;
    commitDetector(id, function (d) {
        var c = d.conditions[i];
        if (!c) return;
        if (f === 'feature') c.feature = t.value.trim().replace(/ - /g, '–');
        else if (f === 'op') c.op = t.value;
        else if (f === 'value') { var x = Number(t.value); if (Number.isFinite(x)) c.value = x; }
    });
}

function onInput(e) {
    var t = e.target;
    var card = t.closest && t.closest('.bd-card');
    if (!card || !_dlg) return;
    if (t.dataset.f === 'value' || t.dataset.f === 'minDuration' || t.dataset.f === 'maxGap') scheduleLive(card.dataset.id);
}

function onFocus(e) {
    var tr = e.target.closest && e.target.closest('tr[data-i]');
    var card = tr && tr.closest('.bd-card');
    if (!tr || !card || !_dlg) return;
    var id = card.dataset.id, i = +tr.dataset.i;
    if (_dlg.focus.get(id) === i) return;
    _dlg.focus.set(id, i);
    card.querySelectorAll('tr[data-i]').forEach(function (r) { r.classList.toggle('is-focus', +r.dataset.i === i); });
    scheduleLive(id, true);
}

function onKey(e) {
    if (e.key !== 'Escape') return;
    e.preventDefault();
    e.stopPropagation();
    closeDialog();
}

function closeDialog() {
    if (!_dlg) return;
    document.removeEventListener('keydown', onKey, true);
    _dlg.timers.forEach(function (tm) { clearTimeout(tm); });
    _dlg.overlay.remove();
    _dlg = null;
}

/** Behavior ▸ Detection Settings…. */
export async function showBehaviorRulesModal() {
    if (_dlg) return;
    var session = state.session;
    if (!session) { setStatus('Load a session first'); return; }
    var big = (state.totalFrames || 0) * Math.max(1, (session.identities || []).length) > 200000;
    if (big) { showLoading('Reading poses', { title: 'Behavior detection settings' }); await yieldToPaint(); }
    var ctx;
    try { ctx = cachedBehaviorContext(session); } finally { if (big) hideLoading(); }
    if (_dlg) return;
    var overlay = document.createElement('div');
    overlay.className = 'multi-frame-modal-overlay behavior-modal-overlay';
    var modal = document.createElement('div');
    modal.className = 'multi-frame-modal behavior-rules-modal';
    modal.setAttribute('role', 'dialog');
    modal.setAttribute('aria-label', 'Behavior detection settings');
    modal.innerHTML = '<h3>Behavior detection</h3>' +
        '<details class="info-section info-collapsible" open><summary><h3>Body</h3>' + infoButton('body') + '</summary><div id="bdBody"></div></details>' +
        '<details class="info-section info-collapsible" open><summary><h3>Floor</h3>' + infoButton('floor') + '</summary><div id="bdFloor"></div></details>' +
        '<details class="info-section info-collapsible" open><summary><h3>Detectors</h3>' + infoButton('detectors') + '</summary><div id="bdDetectors"></div></details>' +
        '<div id="bdLists"></div>' +
        '<div class="modal-actions"><span class="bd-spacer"></span>' +
        '<button data-act="detect" title="Close, and detect every behavior that has a detector on all frames">Detect now</button>' +
        '<button class="primary" data-act="close">Close</button></div>';
    overlay.appendChild(modal);
    document.body.appendChild(overlay);
    modal.querySelectorAll('.bd-info').forEach(function (btn) { btn.innerHTML = ICON_INFO; setInfoTip(btn, INFO[btn.dataset.info]); });
    _dlg = { overlay: overlay, modal: modal, ctx: ctx, columns: new Map(), computing: new Set(), focus: new Map(), timers: new Map(), gen: 1 };
    modal.addEventListener('click', onClick);
    modal.addEventListener('change', onChange);
    modal.addEventListener('input', onInput);
    modal.addEventListener('focusin', onFocus);
    document.addEventListener('keydown', onKey, true);
    renderAll();
}
