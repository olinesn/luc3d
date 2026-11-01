// ui/behavior-train-modal.js — Behavior ▸ Train Classifiers…: which
// behaviors can be learned from the labels, training them, and how well each
// does on labels it never saw.
//
// The table lists every behavior with its labelled frames and bouts of each
// class over the project (what the gate needs: 20 frames and 2 bouts each),
// and its classifier. Train trains the ticked ones (ui/behavior-train.js).
// Each classifier then has a card: how it was cross-validated (whole
// sessions, else whole animals, else blocks of time), frame-level and
// bout-level precision, recall and F1, the precision–recall curve with the
// current threshold on it, the threshold and durations (live numbers from the
// cross-validation, applied when let go: the bouts of every prediction in
// memory are re-derived at once), and its most important features.
//
// Esc or Close leave; one scroller (the dialog).
//
// Depends on: ui/app-state.js, ui/behavior-train.js, ui/behavior-detect.js,
//             ui/info-tip.js, ui/plane-definition.js (the ⓘ icon),
//             pose/behavior-ethogram.js, pose/behavior-classifier.js,
//             import-export/save-load.js

import { state } from './app-state.js';
import {
    classifierStatus, trainClassifiers, applyClassifierPost, exportClassifiers, showImportClassifiers, isTraining, fmt,
    lastTrainingReport, openSessionByName,
} from './behavior-train.js';
import { detectAllFrames, detectionFps } from './behavior-detect.js';
import { setInfoTip } from './info-tip.js';
import { ICON_INFO } from './plane-definition.js';
import { behaviorById, behaviorPost } from '../pose/behavior-ethogram.js';
import { evaluateAt, topFeatures, onModelsChange, GATE_FRAMES, GATE_BOUTS } from '../pose/behavior-classifier.js';
import { setStatus } from '../import-export/save-load.js';

var INFO = {
    train: 'A classifier learns a behavior from the frames you labelled — present and "not"; unlabelled frames teach ' +
        'nothing — and then predicts it on every frame. Each class needs ' + GATE_FRAMES + ' frames and ' + GATE_BOUTS +
        ' bouts. It is tested on labels it never saw: whole sessions, else whole animals, else blocks of time.',
    post: 'A frame counts as the behavior when its score reaches the threshold; shorter bouts are dropped and shorter ' +
        'gaps joined. The numbers update from the cross-validation, without retraining; letting go applies them to ' +
        'the predictions in memory.',
};

function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function n(x) { return Number(x || 0).toLocaleString('en-US'); }
function pct(v) { return Number.isFinite(v) ? Math.round(v * 100) + '%' : '–'; }

var SCHEME_TEXT = {
    session: 'by session', animal: 'by animal', time: 'by blocks of time', none: 'not cross-validated',
};

var _dlg = null;

function infoButton(key) {
    return '<button type="button" class="plane-section-info bd-info" data-info="' + key + '" aria-label="About this"></button>';
}

// ----------------------------------------------------------------------------
// Rendering
// ----------------------------------------------------------------------------

function tableHtml() {
    var rows = classifierStatus();
    if (!rows.length) return '<p class="bd-note">No behaviors yet: edit the ethogram first.</p>';
    var html = '<table class="data-table bt-table"><thead><tr><th></th><th>Behavior</th><th title="Frames and bouts labelled present">Present</th>' +
        '<th title="Frames and bouts labelled &quot;not&quot; (with the frames an exclusive peer is present on)">"Not"</th><th>Classifier</th></tr></thead><tbody>';
    rows.forEach(function (r) {
        var b = r.behavior, m = r.model;
        var ready = !r.gate.length;
        var checked = _dlg.picked.has(b.id) || (!_dlg.touched && ready);
        if (checked && ready) _dlg.picked.add(b.id);
        var status = m ? (m.imported ? 'imported' : 'trained') + (m.cv && m.cv.frame ? ' · F1 ' + fmt(m.cv.frame.f1) : '') : '–';
        html += '<tr data-id="' + esc(b.id) + '"' + (ready ? '' : ' class="is-short"') + '>' +
            '<td><input type="checkbox" data-act="pick"' + (checked && ready ? ' checked' : '') + (ready ? '' : ' disabled') + '></td>' +
            '<td><span class="behavior-swatch" style="background:' + esc(b.color) + '"></span>' + esc(b.name) +
            (b.social ? ' <span class="behavior-tag">' + (b.mutual ? '↔' : '→') + '</span>' : '') + '</td>' +
            '<td class="num">' + n(r.counts.present.frames) + ' · ' + n(r.counts.present.bouts) + '</td>' +
            '<td class="num">' + n(r.counts.absent.frames) + ' · ' + n(r.counts.absent.bouts) + '</td>' +
            '<td>' + (ready ? esc(status) : '<span class="bd-warn">' + esc(r.gate.join('; ')) + '</span>') + '</td></tr>';
    });
    return html + '</tbody></table>';
}

function metricsHtml(frame, bout) {
    return '<table class="data-table bt-metrics"><thead><tr><th></th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>' +
        '<tr><td>Frames</td><td class="num">' + pct(frame && frame.precision) + '</td><td class="num">' + pct(frame && frame.recall) + '</td><td class="num"><b>' +
        fmt(frame && frame.f1) + '</b></td></tr>' +
        '<tr><td>Bouts</td><td class="num">' + pct(bout && bout.precision) + '</td><td class="num">' + pct(bout && bout.recall) + '</td><td class="num"><b>' +
        fmt(bout && bout.f1) + '</b></td></tr></tbody></table>';
}

function cardHtml(m) {
    var b = behaviorById(m.behaviorId);
    if (!b) return '';
    var post = behaviorPost(b);
    var cv = m.cv;
    var rep = lastTrainingReport(m.behaviorId);
    var how;
    if (!cv) how = 'Not cross-validated.';
    else if (cv.note) how = 'Not cross-validated: ' + cv.note + '.';
    else how = 'Cross-validated ' + (SCHEME_TEXT[cv.scheme] || cv.scheme) + ' (' + (cv.folds || []).length + ' folds' +
        (cv.folds && cv.folds.length && cv.scheme !== 'time' ? ': ' + cv.folds.join('; ') : '') + ')' +
        (cv.tested ? ', on ' + n(cv.tested) + ' labelled frames it never saw' : '') + '.';
    var tr = m.training || {};
    var trained = m.imported ? 'Imported' + (tr.sessions && tr.sessions.length ? ', trained on ' + tr.sessions.join(', ') : '')
        : 'Trained on ' + n(tr.rows) + ' rows' + (tr.sampled ? ' (sampled)' : '') + ' from ' + (tr.sessions || []).join(', ');
    var left = (rep && rep.leftOut && rep.leftOut.length) ? '<div class="bd-note bd-warn">Left out: ' +
        rep.leftOut.map(function (l) {
            // A session whose poses are not loaded can be opened, then trained again.
            var canOpen = /open it once/.test(l.reason);
            return esc(l.session) + ' (' + esc(l.reason) + ')' + (canOpen ? ' <button class="panel-btn" data-act="open-session" data-session="' +
                esc(l.session) + '" title="Show this session so its poses are read; then Train again">Open it</button>' : '');
        }).join('; ') + '</div>' : '';
    var top = topFeatures(m, 8);
    return '<div class="bd-card bt-card" data-id="' + esc(b.id) + '">' +
        '<div class="bd-card-head"><span class="behavior-swatch" style="background:' + esc(b.color) + '"></span><b>' + esc(b.name) + '</b>' +
        '<span class="bd-spacer"></span><button class="panel-btn" data-act="predict" title="Predict ' + esc(b.name) + ' on every frame of this session">Predict all frames</button>' +
        '<button class="panel-btn" data-act="export-one" title="Save this classifier to a file">Export…</button></div>' +
        '<div class="bd-note">' + esc(trained) + '. ' + esc(how) + '</div>' + left +
        '<div class="bt-body"><div class="bt-left"><div class="bt-live">' + metricsHtml(cv && cv.frame, cv && cv.bout) + '</div>' +
        '<div class="bt-post">' +
        '<label>Threshold <input type="range" min="0" max="1" step="0.01" data-f="threshold" value="' + post.threshold + '"> <span class="bt-thr">' + post.threshold.toFixed(2) + '</span></label>' +
        '<label title="Bouts with fewer detected frames than this are dropped">Shortest bout <input type="number" min="0" step="0.05" data-f="minBout" value="' + post.minBout + '"> s</label>' +
        '<label title="Detections this close are joined into one bout">Join gaps up to <input type="number" min="0" step="0.05" data-f="maxGap" value="' + post.maxGap + '"> s</label>' +
        infoButton('post') + '</div>' +
        (cv && cv.best ? '<div class="bd-note">Best F1 on frames: ' + fmt(cv.best.f1) + ' at a threshold of ' + Number(cv.best.threshold).toFixed(2) + '.</div>' : '') +
        '</div><div class="bt-right"><canvas class="bt-curve" width="220" height="150" aria-label="Precision against recall over thresholds"></canvas>' +
        '<div class="bd-hist-caption">Precision (up) against recall (across); the dot is the threshold.</div></div></div>' +
        (top.length ? '<div class="bt-top"><span class="bd-label">Most important features:</span><ol>' + top.map(function (f) {
            return '<li>' + esc(f.name) + ' <span class="bt-imp">' + (100 * f.importance).toFixed(1) + '%</span></li>';
        }).join('') + '</ol></div>' : '') +
        '</div>';
}

function renderAll() {
    if (!_dlg) return;
    var body = _dlg.modal.querySelector('#btTable');
    if (body) body.innerHTML = tableHtml();
    renderCards();
    updateButtons();
}

function renderCards() {
    var host = _dlg.modal.querySelector('#btCards');
    if (!host) return;
    var models = classifierStatus().map(function (r) { return r.model; }).filter(Boolean);
    host.innerHTML = models.length ? models.map(cardHtml).join('') : '<p class="bd-note">No classifier yet: tick behaviors above and Train.</p>';
    host.querySelectorAll('.bd-info').forEach(function (btn) { btn.innerHTML = ICON_INFO; setInfoTip(btn, INFO[btn.dataset.info]); });
    host.querySelectorAll('.bt-card').forEach(function (card) { drawCurve(card); });
}

function updateButtons() {
    var go = _dlg.modal.querySelector('[data-act="train"]');
    if (go) {
        go.disabled = isTraining() || !_dlg.picked.size;
        go.textContent = _dlg.picked.size ? 'Train ' + _dlg.picked.size + ' classifier' + (_dlg.picked.size === 1 ? '' : 's') : 'Train';
    }
}

function cardPost(card) {
    var get = function (f) { var el = card.querySelector('[data-f="' + f + '"]'); return el ? Number(el.value) : NaN; };
    return { threshold: Math.min(1, Math.max(0, get('threshold'))), minBout: Math.max(0, get('minBout') || 0), maxGap: Math.max(0, get('maxGap') || 0) };
}

/** The live numbers of a card at its controls' settings (from the kept cross-validation scores). */
function updateLive(card) {
    var b = behaviorById(card.dataset.id);
    var m = b && classifierStatus().find(function (r) { return r.behavior.id === b.id; });
    var model = m && m.model;
    if (!model) return;
    var post = cardPost(card);
    var thrEl = card.querySelector('.bt-thr');
    if (thrEl) thrEl.textContent = post.threshold.toFixed(2);
    var live = evaluateAt(model.cv, post, detectionFps(), b.kind === 'point');
    var host = card.querySelector('.bt-live');
    if (host && live) host.innerHTML = metricsHtml(live.frame, live.bout);
    drawCurve(card, live ? live.frame : null);
}

function drawCurve(card, at) {
    var canvas = card.querySelector('.bt-curve');
    var b = behaviorById(card.dataset.id);
    var m = b && classifierStatus().find(function (r) { return r.behavior.id === b.id; });
    var cv = m && m.model && m.model.cv;
    if (!canvas) return;
    var g = canvas.getContext('2d');
    var W = canvas.width, H = canvas.height, pad = 18;
    g.clearRect(0, 0, W, H);
    var styles = getComputedStyle(card);
    var muted = styles.getPropertyValue('--text-secondary').trim() || '#9aa0a6';
    g.strokeStyle = muted;
    g.lineWidth = 1;
    g.strokeRect(pad + 0.5, 0.5, W - pad - 1, H - pad - 1);
    g.fillStyle = muted;
    g.font = '10px sans-serif';
    g.textAlign = 'left';
    g.fillText('0', pad, H - 4);
    g.textAlign = 'right';
    g.fillText('recall 1', W - 1, H - 4);
    g.save();
    g.translate(10, H - pad);
    g.rotate(-Math.PI / 2);
    g.textAlign = 'right';
    g.fillText('precision 1', H - pad, 0);
    g.restore();
    if (!cv || !Array.isArray(cv.curve) || !cv.curve.length) return;
    var x = function (r) { return pad + r * (W - pad - 2); };
    var y = function (p) { return (H - pad - 1) - p * (H - pad - 2); };
    g.strokeStyle = b.color || '#5fb3f9';
    g.lineWidth = 1.5;
    g.beginPath();
    var started = false;
    cv.curve.forEach(function (pt) {
        if (!Number.isFinite(pt.precision) || !Number.isFinite(pt.recall)) return;
        if (!started) { g.moveTo(x(pt.recall), y(pt.precision)); started = true; } else g.lineTo(x(pt.recall), y(pt.precision));
    });
    g.stroke();
    var here = at || cv.frame;
    if (here && Number.isFinite(here.precision) && Number.isFinite(here.recall)) {
        g.fillStyle = styles.getPropertyValue('--text-primary').trim() || '#fff';
        g.beginPath();
        g.arc(x(here.recall), y(here.precision), 3.5, 0, 2 * Math.PI);
        g.fill();
    }
}

// ----------------------------------------------------------------------------
// Events
// ----------------------------------------------------------------------------

async function onClick(e) {
    var t = e.target.closest('button, input');
    if (!t || !_dlg) return;
    var act = t.dataset.act;
    if (act === 'pick') {
        var id = t.closest('tr').dataset.id;
        _dlg.touched = true;
        if (t.checked) _dlg.picked.add(id); else _dlg.picked.delete(id);
        updateButtons();
        return;
    }
    if (act === 'close') { closeDialog(); return; }
    if (act === 'train') {
        var ids = Array.from(_dlg.picked);
        t.disabled = true;
        await trainClassifiers(ids);
        if (_dlg) renderAll();
        return;
    }
    if (act === 'import') { showImportClassifiers(); return; }
    if (act === 'open-session') {
        var name = t.dataset.session;
        closeDialog();
        var ok = await openSessionByName(name);
        setStatus(ok ? 'Opened ' + name + ': train again (Behavior ▸ Train Classifiers…) to include it' : 'Could not open ' + name, ok ? 'success' : 'warning');
        return;
    }
    if (act === 'export') { exportClassifiers(); return; }
    var card = t.closest('.bt-card');
    if (!card) return;
    if (act === 'export-one') { exportClassifiers([card.dataset.id]); return; }
    if (act === 'predict') {
        var bid = card.dataset.id;
        closeDialog();
        await detectAllFrames({ only: [bid] });
    }
}

function onInput(e) {
    var card = e.target.closest('.bt-card');
    if (card && e.target.dataset.f) updateLive(card);
}

function onChange(e) {
    var card = e.target.closest('.bt-card');
    if (!card || !e.target.dataset.f) return;
    var id = card.dataset.id;
    var redone = applyClassifierPost(id, cardPost(card));
    var b = behaviorById(id);
    setStatus(b.name + ': threshold ' + behaviorPost(b).threshold.toFixed(2) + ', shortest bout ' + behaviorPost(b).minBout + ' s, gaps ' +
        behaviorPost(b).maxGap + ' s' + (redone ? ' · ' + redone + ' prediction' + (redone === 1 ? '' : 's') + ' updated' : ''), 'success');
    updateLive(card);
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
    _dlg.overlay.remove();
    _dlg = null;
}

/** Behavior ▸ Train Classifiers…. */
export function showBehaviorTrainModal() {
    if (_dlg) return;
    if (!state.session) { setStatus('Load a session first'); return; }
    var overlay = document.createElement('div');
    overlay.className = 'multi-frame-modal-overlay behavior-modal-overlay';
    var modal = document.createElement('div');
    modal.className = 'multi-frame-modal behavior-rules-modal behavior-train-modal';
    modal.setAttribute('role', 'dialog');
    modal.setAttribute('aria-label', 'Train behavior classifiers');
    modal.innerHTML = '<h3>Behavior classifiers</h3>' +
        '<details class="info-section info-collapsible" open><summary><h3>Labels</h3>' + infoButton('train') + '</summary><div id="btTable"></div>' +
        '<div class="bd-row"><button class="primary" data-act="train">Train</button>' +
        '<span class="bd-note">Trains on every labelled session; a classifier then predicts its behavior in place of the detector (Behavior ▸ All frames).</span></div></details>' +
        '<details class="info-section info-collapsible" open><summary><h3>Classifiers</h3></summary><div id="btCards"></div></details>' +
        '<div class="modal-actions"><button data-act="import" title="Use classifiers saved from another project">Import…</button>' +
        '<button data-act="export" title="Save every classifier to one file">Export all…</button><span class="bd-spacer"></span>' +
        '<button class="primary" data-act="close">Close</button></div>';
    overlay.appendChild(modal);
    document.body.appendChild(overlay);
    modal.querySelectorAll('.bd-info').forEach(function (btn) { btn.innerHTML = ICON_INFO; setInfoTip(btn, INFO[btn.dataset.info]); });
    _dlg = { overlay: overlay, modal: modal, picked: new Set(), touched: false };
    modal.addEventListener('click', onClick);
    modal.addEventListener('input', onInput);
    modal.addEventListener('change', onChange);
    document.addEventListener('keydown', onKey, true);
    renderAll();
}

/** Repaint the open dialog (a model was trained or imported elsewhere). */
export function refreshBehaviorTrainModal() {
    if (_dlg) renderAll();
}

onModelsChange(function () { refreshBehaviorTrainModal(); });
