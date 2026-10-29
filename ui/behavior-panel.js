// ui/behavior-panel.js — the Behavior tab, and the wiring of the behavior
// labelling feature (`setupBehaviorUi`).
//
// The tab's three sections are static in index.html (their folds are
// browser-local, restored once here); their bodies are rendered from the
// session's bouts:
//
//   Coding    the animal being labelled (digits 1–9 in the mode), one button
//             per behavior (a click codes it at the playhead, or labels the
//             armed range; Shift-click labels the range "not"), running states,
//             the armed range, the target question, Undo, Mark reviewed…
//   Bouts     every bout, filterable; a click lands 1 s before it and offers
//             set start / end to the playhead, behavior, target, note, delete
//   Predictions  the detectors' predicted bouts to review — most uncertain,
//             most contradicted by labels, or earliest first: a click lands on a
//             bout (1 s early); its Accept labels it present (source "a"), its
//             Reject labels it "not", and both move on to the next; Accept all in range… (armed range, else all, with
//             a confirm) is one undo step. Stale predictions say why. Also the
//             This frame readout: per animal, each detector's conditions
//             against their thresholds on one frame.
//   Summary   per animal × behavior: bouts, seconds, share of the labelled
//             frames; labels of a deleted identity, to reassign or delete
//
// The buttons work with or without Label Behaviors mode, so behaviors can be
// labelled in the middle of proofreading.
//
// Depends on: ui/app-state.js, ui/behavior-coding.js, ui/behavior-mode.js,
//             ui/behavior-ethogram-modal.js, ui/behavior-io-modal.js, ui/behavior-timeline.js,
//             ui/behavior-detect.js, ui/behavior-rules-modal.js, ui/behavior-confirm.js,
//             ui/info-tip.js, ui/plane-definition.js (the ⓘ icon),
//             ui/panel-visibility.js, ui/section-state.js, ui/settings.js,
//             ui/rendering.js, pose/behavior-ethogram.js, pose/behavior-labels.js,
//             pose/behavior-predictions.js, import-export/save-load.js

import { state } from './app-state.js';
import {
    behaviorUi, codeBehavior, focalAnimal, setFocalAnimal, sessionAnimals, animalColor, selectedBehaviorId,
    selectBehavior, pickBehaviorTarget, cancelPendingTarget, clearArmedRange, armedRange, registerBehaviorRepaint,
    requestBehaviorRepaint, setBehaviorNavigator, goToBehaviorFrame, BOUT_LEAD_IN_SECONDS,
    behaviorBoxesVisible, setBehaviorBoxesVisible, pairGlyph, openBehaviorTab,
} from './behavior-coding.js';
import { toggleBehaviorMode, exitBehaviorMode, isBehaviorModeActive } from './behavior-mode.js';
import { showEthogramModal, showMarkReviewedModal } from './behavior-ethogram-modal.js';
import { showBehaviorImportModal, showBehaviorExportModal } from './behavior-io-modal.js';
import { ABSENT_COLOR } from './behavior-timeline.js';
import {
    detectAllFrames, detectThisFrame, frameReadout, onFrameReadout, detectionEnv, setupBehaviorDetection,
} from './behavior-detect.js';
import { showBehaviorRulesModal } from './behavior-rules-modal.js';
import { behaviorConfirm } from './behavior-confirm.js';
import { setInfoTip } from './info-tip.js';
import { ICON_INFO } from './plane-definition.js';
import { isInfoPanelVisible } from './panel-visibility.js';
import { persistSectionStates } from './section-state.js';
import { setHandler } from './settings.js';
import { drawAllOverlays } from './rendering.js';
import { behaviors, behaviorById } from '../pose/behavior-ethogram.js';
import {
    onBehaviorChange, listBouts, runningStates, activeAt, timeBudget, orphanedAnimals, reassignAnimal,
    deleteBout, editBout, setBehaviorCursor, canUndoBehaviorEdit, undoBehaviorEdit, isRunning, closeRunningAtCursor,
    cancelRunning, hasBehaviorData, toggleState,
} from '../pose/behavior-labels.js';
import {
    reviewQueue, acceptPredictions, rejectPredictions, predictionStaleness, hasPredictions, clearBehaviorRuns,
    onPredictionsChange, behaviorRuns,
} from '../pose/behavior-predictions.js';
import { markDirty, setStatus } from '../import-export/save-load.js';

var SECTIONS_KEY = 'lucid.behaviorSections';
var MAX_LISTED_BOUTS = 500;

var panel = { animal: '', behavior: '', showAbsent: false, selectedKey: null, stale: true };
/** The Predictions section's filters and selection (browser memory only). */
var review = { behavior: '', animal: '', order: 'uncertainty', unreviewedOnly: true, selectedKey: null, selectedIndex: -1 };

var PREDICTIONS_INFO = 'Bouts the detectors found, to check: Accept labels a bout as that behavior, Reject labels it ' +
    '"not", and each moves on to the next. The labels are what count; predictions are redone by detecting again ' +
    '(toolbar Behavior button) and go stale when the poses or the settings change.';

function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function n(x) { return Number(x || 0).toLocaleString('en-US'); }
function fr(f) { return (f + 1).toLocaleString('en-US'); }
function fps() { return state.fps > 0 ? state.fps : 30; }
function secs(frames) { return (frames / fps()).toFixed(1); }
function boutKey(b) { return [b.behaviorId, b.animal, b.target, b.start, b.end].join('|'); }

/** Show the Behavior tab with its Predictions section open (the This frame readout lives there). */
function openBehaviorTabSection() {
    openBehaviorTab();
    var sec = document.getElementById('behaviorPredictionsSection');
    if (sec && !sec.open) sec.open = true;
}

/** Is the Behavior tab on screen? */
function tabVisible() {
    var tab = document.getElementById('tabBehavior');
    return !!(tab && tab.classList.contains('active') && isInfoPanelVisible());
}

// ----------------------------------------------------------------------------
// Rendering
// ----------------------------------------------------------------------------

/** Re-render the Behavior tab for the active session (skipped while it is hidden). */
export function refreshBehaviorPanel() {
    if (!tabVisible()) { panel.stale = true; return; }
    panel.stale = false;
    var session = state.session;
    var empty = document.getElementById('behaviorEmpty');
    var modeBtn = document.getElementById('behaviorModeBtn');
    if (modeBtn) {
        modeBtn.classList.toggle('active', isBehaviorModeActive());
        modeBtn.setAttribute('aria-pressed', isBehaviorModeActive() ? 'true' : 'false');
        modeBtn.textContent = isBehaviorModeActive() ? 'Leave Label Behaviors mode' : 'Label Behaviors mode';
    }
    var msg = '';
    if (!session) msg = 'Load a session to label behaviors.';
    else if (!behaviors().length) msg = 'No behaviors yet. Edit the ethogram to list the behaviors you will label.';
    else if (!sessionAnimals(session).length) msg = 'Behaviors are labelled per animal: run Track All, or add identities (Tracks ▸ New Identity).';
    if (empty) { empty.textContent = msg; empty.hidden = !msg; }
    renderCoding(session, !!msg);
    renderBouts(session);
    renderPredictions(session);
    renderSummary(session);
}

function renderCoding(session, disabled) {
    var host = document.getElementById('behaviorCodingBody');
    if (!host) return;
    if (!session || disabled) { host.innerHTML = ''; return; }
    var ui = behaviorUi();
    var animals = sessionAnimals(session);
    var focal = focalAnimal(session);
    var html = '';
    if (ui.pending) {
        var pb = behaviorById(ui.pending.behaviorId);
        html += '<div class="behavior-pending">Target of <b>' + esc(pb ? pb.name : '?') + '</b> by ' + esc(ui.pending.animal) +
            '? Pick it below. <button class="panel-btn" data-act="cancel-target">Cancel</button></div>';
    }
    html += '<div class="behavior-animals" role="group" aria-label="Animal">' + animals.map(function (a, i) {
        var cls = 'behavior-animal' + (a === focal ? ' is-focal' : '') +
            (ui.pending && a !== ui.pending.animal ? ' is-target-choice' : '');
        return '<button class="' + cls + '" data-animal="' + esc(a) + '" style="--animal-color:' + esc(animalColor(session, a)) + '"' +
            ' title="' + (ui.pending ? 'Target: ' : 'Label ') + esc(a) + (i < 9 ? ' (' + (i + 1) + ' in the mode)' : '') + '">' +
            (i < 9 ? '<span class="behavior-digit">' + (i + 1) + '</span>' : '') + esc(a) + '</button>';
    }).join('') + '</div>';
    var range = armedRange();
    if (ui.rangeIn != null) {
        html += '<div class="behavior-armed">Range ' + fr(ui.rangeIn) + '–' + (ui.rangeOut != null ? fr(ui.rangeOut) : '…') +
            (range ? ' · a behavior labels it (Shift-click: "not")' : ' · ] sets its end') +
            ' <button class="panel-btn" data-act="clear-range">Clear</button></div>';
    }
    var running = runningStates(session);
    if (running.length) {
        html += '<div class="behavior-running">' + running.map(function (r) {
            var b = behaviorById(r.behaviorId);
            if (!b) return '';
            return '<div class="behavior-running-row" data-run="' + esc(r.behaviorId + '|' + r.animal + '|' + r.target) + '">' +
                '<span class="behavior-swatch" style="background:' + esc(b.color) + '"></span>● ' + esc(b.name) + ' · ' +
                esc(r.animal) + (r.target ? esc(pairGlyph(b)) + esc(r.target) : '') + ' since ' + fr(r.start) +
                ' <button class="panel-btn" data-act="stop-run">Stop here</button>' +
                ' <button class="panel-btn" data-act="cancel-run" title="Drop it without labelling">Discard</button></div>';
        }).join('') + '</div>';
    }
    html += '<div class="behavior-buttons">' + behaviors().map(function (b) {
        return '<button class="behavior-chip-btn" data-behavior="' + esc(b.id) + '" title="' + esc(behaviorTitle(b)) + '">' +
            '<span class="behavior-swatch" style="background:' + esc(b.color) + '"></span>' + esc(b.name) +
            (b.key ? ' <kbd>' + esc(b.key) + '</kbd>' : '') + (b.social ? ' <span class="behavior-tag">' + (b.mutual ? '↔' : '→') + '</span>' : '') + '</button>';
    }).join('') + '</div>';
    var sel = selectedBehaviorId();
    html += '<div class="behavior-tools">' +
        '<label>Timeline <select data-act="timeline-behavior"><option value="">All behaviors</option>' +
        behaviors().map(function (b) {
            return '<option value="' + esc(b.id) + '"' + (b.id === sel ? ' selected' : '') + '>' + esc(b.name) + '</option>';
        }).join('') + '</select></label>' +
        '<button class="panel-btn" data-act="undo"' + (canUndoBehaviorEdit(session) ? '' : ' disabled') + '>Undo</button>' +
        '<button class="panel-btn" data-act="reviewed" title="Turn the unlabelled frames of a range into &quot;not&quot; labels">Mark reviewed…</button>' +
        '<label title="Box the animals engaged in each behavior in the videos, labelled (this browser only)"><input type="checkbox" data-act="on-video"' +
        (behaviorBoxesVisible() ? ' checked' : '') + '> On video</label>' +
        '</div>';
    host.innerHTML = html;
    updateCodingNow(state.currentFrame);
}

function behaviorTitle(b) {
    var t = (b.kind === 'point' ? 'Code "' + b.name + '" on this frame' : 'Start / stop "' + b.name + '" at the playhead') +
        ' for the selected animal; with a range armed, label the range (Shift-click: not ' + b.name + ')';
    if (b.description) t += '\n' + b.description;
    return t;
}

/** Light the behavior buttons the focal animal is doing on `frame` (cheap; runs every frame). */
function updateCodingNow(frame) {
    var host = document.getElementById('behaviorCodingBody');
    if (!host || !state.session) return;
    var focal = focalAnimal(state.session);
    var on = new Set();
    activeAt(state.session, frame).forEach(function (a) {
        var b = behaviorById(a.behaviorId);
        if (a.animal === focal || (b && b.mutual && a.target === focal)) on.add(a.behaviorId);
    });
    host.querySelectorAll('.behavior-chip-btn').forEach(function (btn) {
        btn.classList.toggle('is-on', on.has(btn.dataset.behavior));
    });
}

function renderBouts(session) {
    var host = document.getElementById('behaviorBoutsBody');
    var count = document.getElementById('behaviorBoutsCount');
    if (!host) return;
    if (!session || (!behaviors().length && !hasBehaviorData(session))) {
        host.innerHTML = '';
        if (count) count.textContent = '0';
        return;
    }
    var filter = {};
    if (panel.animal) filter.animal = panel.animal;
    if (panel.behavior) filter.behaviorId = panel.behavior;
    if (!panel.showAbsent) filter.present = true;
    var bouts = listBouts(session, filter);
    if (count) count.textContent = n(bouts.length);
    var animals = sessionAnimals(session).concat(orphanedAnimals(session, sessionAnimals(session)));
    var html = '<div class="behavior-filters">' +
        '<select data-filter="animal"><option value="">All animals</option>' + animals.map(function (a) {
            return '<option value="' + esc(a) + '"' + (a === panel.animal ? ' selected' : '') + '>' + esc(a) + '</option>';
        }).join('') + '</select>' +
        '<select data-filter="behavior"><option value="">All behaviors</option>' + behaviors().map(function (b) {
            return '<option value="' + esc(b.id) + '"' + (b.id === panel.behavior ? ' selected' : '') + '>' + esc(b.name) + '</option>';
        }).join('') + '</select>' +
        '<label><input type="checkbox" data-filter="absent"' + (panel.showAbsent ? ' checked' : '') + '> "not" labels</label>' +
        '<button class="panel-btn" data-act="next-bout"' + (bouts.length ? '' : ' disabled') + '>Next ▸</button>' +
        '</div>';
    if (!bouts.length) {
        html += '<p class="table-empty">No bouts' + (panel.animal || panel.behavior ? ' match the filter' : ' labelled yet') + '.</p>';
        host.innerHTML = html;
        return;
    }
    html += '<div class="behavior-bout-list">';
    bouts.slice(0, MAX_LISTED_BOUTS).forEach(function (bt) {
        var b = behaviorById(bt.behaviorId);
        var k = boutKey(bt);
        var sel = k === panel.selectedKey;
        html += '<div class="behavior-bout' + (sel ? ' is-current' : '') + (bt.present ? '' : ' is-absent') + '" data-key="' + esc(k) + '">' +
            '<div class="behavior-bout-main">' +
            '<span class="behavior-swatch" style="background:' + esc(bt.present && b ? b.color : ABSENT_COLOR) + '"></span>' +
            '<span class="behavior-bout-name">' + (bt.present ? '' : 'not ') + esc(b ? b.name : '?') + '</span>' +
            '<span class="behavior-bout-who">' + esc(bt.animal) + (bt.target ? esc(pairGlyph(b)) + esc(bt.target) : '') + '</span>' +
            '<span class="behavior-bout-frames">' + (bt.start === bt.end ? fr(bt.start) : fr(bt.start) + '–' + fr(bt.end)) + '</span>' +
            '<span class="behavior-bout-dur">' + secs(bt.end - bt.start + 1) + ' s</span>' +
            (bt.source === 'a' ? '<span class="behavior-tag" title="Accepted from a prediction">acc</span>' : '') +
            (bt.source === 'i' ? '<span class="behavior-tag" title="Imported">imp</span>' : '') +
            '</div>' + (sel ? selectedHtml(session, bt) : '') + '</div>';
    });
    html += '</div>';
    if (bouts.length > MAX_LISTED_BOUTS) {
        html += '<p class="table-empty">Showing the first ' + n(MAX_LISTED_BOUTS) + ' of ' + n(bouts.length) + ' — filter to narrow.</p>';
    }
    host.innerHTML = html;
}

function selectedHtml(session, bt) {
    var b = behaviorById(bt.behaviorId);
    var animals = sessionAnimals(session);
    var html = '<div class="behavior-bout-actions">' +
        '<button class="panel-btn" data-act="set-start" title="Move the start to the playhead">Start ← playhead</button>' +
        '<button class="panel-btn" data-act="set-end" title="Move the end to the playhead">End ← playhead</button>' +
        '<select data-act="set-behavior" title="Behavior">' + behaviors().map(function (o) {
            return '<option value="' + esc(o.id) + '"' + (o.id === bt.behaviorId ? ' selected' : '') + '>' + esc(o.name) + '</option>';
        }).join('') + '</select>';
    if (b && b.social) {
        html += '<select data-act="set-target" title="Target">' + animals.filter(function (a) { return a !== bt.animal; }).map(function (a) {
            return '<option value="' + esc(a) + '"' + (a === bt.target ? ' selected' : '') + '>' + esc(pairGlyph(b).trim()) + ' ' + esc(a) + '</option>';
        }).join('') + '</select>';
    }
    html += '<input type="text" data-act="set-note" placeholder="Note" value="' + esc(bt.note || '') + '">' +
        '<button class="panel-btn panel-btn-danger" data-act="delete-bout">Delete</button></div>';
    return html;
}

// ----------------------------------------------------------------------------
// Predictions
// ----------------------------------------------------------------------------

function predKey(p) { return [p.behaviorId, p.animal, p.target, p.start, p.end].join('|'); }

function scorePct(x) { return Number.isFinite(x) ? Math.round(100 * x) + '%' : '–'; }

var STATUS_TEXT = { accepted: 'accepted', rejected: 'rejected', mixed: 'partly labelled', unreviewed: '' };

function staleHtml(session) {
    var st = predictionStaleness(session, detectionEnv());
    if (!st.input && !st.settings.length) return '';
    var why = [];
    if (st.input) why.push('the poses changed since they were detected');
    if (st.settings.length) {
        why.push('the settings of ' + st.settings.map(function (id) { var b = behaviorById(id); return b ? b.name : 'a removed behavior'; }).join(', ') + ' changed');
    }
    return '<div class="behavior-stale" role="status">Out of date: ' + esc(why.join('; ')) + '. ' +
        '<button class="panel-btn" data-act="detect-all">Detect again</button></div>';
}

function readoutHtml() {
    var r = frameReadout();
    if (!r || r === _readoutHidden) return '';
    var html = '<div class="behavior-readout"><div class="behavior-readout-head">This frame: ' + fr(r.frame) +
        (r.frame !== state.currentFrame ? ' <button class="panel-btn" data-act="readout-go">Go there</button>' : '') +
        ' <button class="panel-btn" data-act="readout-close" title="Hide this readout">✕</button></div>';
    if (!r.rows.length) return html + '<p class="table-empty">No detector could run on this frame.</p></div>';
    r.rows.forEach(function (row) {
        var b = behaviorById(row.behaviorId);
        if (!b) return;
        html += '<div class="behavior-readout-row' + (row.detected ? ' is-on' : '') + '">' +
            '<span class="behavior-swatch" style="background:' + esc(b.color) + '"></span>' +
            '<b>' + (row.detected ? '' : 'not ') + esc(b.name) + '</b> · ' + esc(row.animal) + (row.target ? esc(pairGlyph(b)) + esc(row.target) : '') +
            ' <span class="behavior-pred-score">' + (Number.isFinite(row.score) ? scorePct(row.score) : 'missing data') + '</span>' +
            '<div class="behavior-readout-conds">' + row.conditions.map(function (c) {
                var ok = Number.isFinite(c.actual) && (c.op === '<' ? c.actual <= c.value : c.actual >= c.value);
                return '<div class="' + (ok ? 'is-ok' : 'is-no') + '">' + (ok ? '✓ ' : '✗ ') + esc(c.feature) + ' = ' +
                    (Number.isFinite(c.actual) ? c.actual.toFixed(3) : 'missing') + (c.unit ? ' ' + esc(c.unit) : '') +
                    ' (' + (c.op === '<' ? 'below ' : 'above ') + c.value + ')</div>';
            }).join('') + '</div></div>';
    });
    return html + '</div>';
}

function renderPredictions(session) {
    var host = document.getElementById('behaviorPredictionsBody');
    var count = document.getElementById('behaviorPredictionsCount');
    if (!host) return;
    if (!session || !behaviors().length) {
        host.innerHTML = '';
        if (count) count.textContent = '0';
        return;
    }
    var html = '<div class="behavior-tools">' +
        '<button class="panel-btn" data-act="detect-all" title="Detect every behavior that has a detector, on all frames (the toolbar Behavior button)">Detect all frames</button>' +
        '<button class="panel-btn" data-act="detect-frame" title="Which behaviors the detectors find on this frame, and why">This frame</button>' +
        '<button class="panel-btn" data-act="detect-settings" title="The body, the floor and each behavior\'s detector">Settings…</button>' +
        (hasPredictions(session) ? '<button class="panel-btn" data-act="clear-predictions" title="Forget the predictions (labels stay)">Clear</button>' : '') +
        '</div>';
    var withDetector = behaviors().filter(function (b) { return !!b.detector; }).length;
    if (!hasPredictions(session)) {
        html += '<p class="table-empty">' + (withDetector
            ? 'Nothing detected yet: Detect all frames, then review the bouts here.'
            : 'No detectors yet: Settings… gives a behavior one (templates for rearing, contact, following…).') + '</p>';
        html += readoutHtml();
        host.innerHTML = html;
        if (count) count.textContent = '0';
        return;
    }
    html += staleHtml(session);
    var all = reviewQueue(session, {});
    var unreviewed = all.filter(function (p) { return p.status !== 'accepted' && p.status !== 'rejected'; }).length;
    if (count) count.textContent = n(unreviewed);
    var runIds = behaviorRuns(session).map(function (r) { return r.behaviorId; });
    var animals = sessionAnimals(session);
    html += '<div class="behavior-filters">' +
        '<select data-review="behavior"><option value="">All behaviors</option>' + behaviors().filter(function (b) { return runIds.indexOf(b.id) >= 0; }).map(function (b) {
            return '<option value="' + esc(b.id) + '"' + (b.id === review.behavior ? ' selected' : '') + '>' + esc(b.name) + '</option>';
        }).join('') + '</select>' +
        '<select data-review="animal"><option value="">All animals</option>' + animals.map(function (a) {
            return '<option value="' + esc(a) + '"' + (a === review.animal ? ' selected' : '') + '>' + esc(a) + '</option>';
        }).join('') + '</select>' +
        '<select data-review="order" title="Which bouts come first">' +
        [['uncertainty', 'Most uncertain first'], ['disagreement', 'Most contradicted first'], ['time', 'In time order']].map(function (o) {
            return '<option value="' + o[0] + '"' + (o[0] === review.order ? ' selected' : '') + '>' + o[1] + '</option>';
        }).join('') + '</select>' +
        '<label title="Hide bouts already accepted or rejected"><input type="checkbox" data-review="unreviewed"' + (review.unreviewedOnly ? ' checked' : '') + '> To review only</label>' +
        '</div>';
    var queue = reviewQueue(session, { order: review.order, behaviorId: review.behavior, animal: review.animal, unreviewedOnly: review.unreviewedOnly });
    html += '<div class="behavior-tools">' +
        '<button class="panel-btn" data-act="review-next"' + (queue.length ? '' : ' disabled') + ' title="Go to the next bout in this list">Next ▸</button>' +
        '<button class="panel-btn" data-act="accept-range"' + (queue.length ? '' : ' disabled') +
        ' title="Accept every listed bout inside the armed [ ] range (else all of them), as one undo step">Accept all in range…</button>' +
        '<span class="behavior-review-count">' + n(unreviewed) + ' of ' + n(all.length) + ' to review</span></div>';
    if (!queue.length) {
        html += '<p class="table-empty">' + (all.length ? 'Every predicted bout here is reviewed.' : 'No predicted bouts.') + '</p>';
    } else {
        html += '<div class="behavior-bout-list">';
        queue.slice(0, MAX_LISTED_BOUTS).forEach(function (p) {
            var b = behaviorById(p.behaviorId);
            var k = predKey(p);
            html += '<div class="behavior-bout behavior-pred is-' + p.status + (k === review.selectedKey ? ' is-current' : '') + '" data-pkey="' + esc(k) + '">' +
                '<div class="behavior-bout-main">' +
                '<span class="behavior-swatch" style="background:' + esc(b ? b.color : ABSENT_COLOR) + '"></span>' +
                '<span class="behavior-bout-name">' + esc(b ? b.name : '?') + '</span>' +
                '<span class="behavior-bout-who">' + esc(p.animal) + (p.target ? esc(pairGlyph(b)) + esc(p.target) : '') + '</span>' +
                '<span class="behavior-bout-frames">' + (p.start === p.end ? fr(p.start) : fr(p.start) + '–' + fr(p.end)) + '</span>' +
                '<span class="behavior-bout-dur">' + secs(p.end - p.start + 1) + ' s</span>' +
                '<span class="behavior-pred-score" title="How clearly the detector\'s conditions hold (50% = at the threshold)">' + scorePct(p.score) + '</span>' +
                (STATUS_TEXT[p.status] ? '<span class="behavior-tag">' + STATUS_TEXT[p.status] + '</span>' : '') +
                '</div>' + (k === review.selectedKey
                    ? '<div class="behavior-bout-actions"><button class="panel-btn" data-act="accept" title="Label this bout as ' + esc(b ? b.name : '') +
                      ', and go to the next">Accept</button><button class="panel-btn" data-act="reject" title="Label this bout &quot;not ' + esc(b ? b.name : '') +
                      '&quot;, and go to the next">Reject</button></div>'
                    : '') + '</div>';
        });
        html += '</div>';
        if (queue.length > MAX_LISTED_BOUTS) {
            html += '<p class="table-empty">Showing the first ' + n(MAX_LISTED_BOUTS) + ' of ' + n(queue.length) + ' — filter to narrow.</p>';
        }
    }
    html += readoutHtml();
    host.innerHTML = html;
}

/** The predicted bout a list row shows. */
function findPrediction(key) {
    var all = reviewQueue(state.session, {});
    for (var i = 0; i < all.length; i++) if (predKey(all[i]) === key) return all[i];
    return null;
}

function landOn(p) {
    goToBehaviorFrame(Math.max(0, p.start - Math.round(BOUT_LEAD_IN_SECONDS * fps())));
}

/** Select row `i` of the list (wrapping), and go to its bout. */
function selectReviewRow(host, i) {
    var rows = Array.from(host.querySelectorAll('.behavior-pred'));
    if (!rows.length) return;
    var row = rows[((i % rows.length) + rows.length) % rows.length];
    review.selectedKey = row.dataset.pkey;
    review.selectedIndex = rows.indexOf(row);
    var p = findPrediction(review.selectedKey);
    renderPredictions(state.session);
    if (p) landOn(p);
}

async function acceptInRange(session) {
    var queue = reviewQueue(session, { order: 'time', behaviorId: review.behavior, animal: review.animal, unreviewedOnly: review.unreviewedOnly });
    var range = armedRange();
    var take = range ? queue.filter(function (p) { return p.start >= range.start && p.end <= range.end; }) : queue;
    if (!take.length) { setStatus(range ? 'No listed bout lies inside the armed range' : 'No predicted bouts to accept'); return; }
    var frames = take.reduce(function (s, p) { return s + p.end - p.start + 1; }, 0);
    var byBehavior = new Map();
    take.forEach(function (p) { byBehavior.set(p.behaviorId, (byBehavior.get(p.behaviorId) || 0) + 1); });
    var ok = await behaviorConfirm({
        id: 'behaviorAcceptRangeConfirm',
        title: 'Accept ' + n(take.length) + ' predicted bout' + (take.length === 1 ? '' : 's') + '?',
        message: 'They become behavior labels' + (range ? ' (frames ' + fr(range.start) + '–' + fr(range.end) + ')' : ' (every listed bout)') +
            ', ' + secs(frames) + ' s in all. One Undo takes them all back.',
        rows: Array.from(byBehavior.keys()).map(function (id) { var b = behaviorById(id); return [b ? b.name : id, n(byBehavior.get(id))]; }),
        confirmLabel: 'Accept them',
    });
    if (!ok) return;
    var res = acceptPredictions(session, take);
    setStatus('Accepted ' + n(res.kept) + ' predicted bout' + (res.kept === 1 ? '' : 's') + ' as labels', 'success');
}

function wirePredictions(host) {
    host.addEventListener('click', function (e) {
        var session = state.session;
        if (!session) return;
        var t = e.target;
        if (t.closest('select, input')) return;
        var btn = t.closest('button');
        var act = btn && btn.dataset.act;
        if (act === 'detect-all') { detectAllFrames(); return; }
        if (act === 'detect-frame') { detectThisFrame(); return; }
        if (act === 'detect-settings') { showBehaviorRulesModal(); return; }
        if (act === 'clear-predictions') {
            clearBehaviorRuns(session);
            markDirty();
            setStatus('Cleared the predictions (labels stay)');
            return;
        }
        if (act === 'readout-close') { onFrameReadoutClose(); return; }
        if (act === 'readout-go') { var r = frameReadout(); if (r) goToBehaviorFrame(r.frame); return; }
        if (act === 'review-next') {
            var rows = Array.from(host.querySelectorAll('.behavior-pred'));
            var i = rows.findIndex(function (r) { return r.dataset.pkey === review.selectedKey; });
            selectReviewRow(host, i >= 0 ? i + 1 : Math.max(0, review.selectedIndex));
            return;
        }
        if (act === 'accept-range') { acceptInRange(session); return; }
        var row = t.closest('.behavior-pred');
        if (!row) return;
        var p = findPrediction(row.dataset.pkey);
        if (!p) return;
        if (act === 'accept' || act === 'reject') {
            // The selected bout's buttons: label it, then move on to the next one.
            var idx = Array.from(host.querySelectorAll('.behavior-pred')).indexOf(row);
            (act === 'accept' ? acceptPredictions : rejectPredictions)(session, [p]);
            setStatus((act === 'accept' ? 'Accepted' : 'Rejected') + ' ' + (behaviorById(p.behaviorId) || {}).name + ' · ' + p.animal +
                ' · frames ' + fr(p.start) + '–' + fr(p.end), 'success');
            renderPredictions(session);
            var left = Array.from(host.querySelectorAll('.behavior-pred'));
            var stay = left.findIndex(function (r) { return r.dataset.pkey === review.selectedKey; });
            selectReviewRow(host, stay >= 0 ? stay + 1 : idx);
            return;
        }
        if (row.dataset.pkey !== review.selectedKey) {
            review.selectedKey = row.dataset.pkey;
            review.selectedIndex = Array.from(host.querySelectorAll('.behavior-pred')).indexOf(row);
            renderPredictions(session);
        }
        landOn(p);
    });
    host.addEventListener('change', function (e) {
        var t = e.target;
        var k = t.dataset.review;
        if (!k) return;
        if (k === 'behavior') review.behavior = t.value;
        else if (k === 'animal') review.animal = t.value;
        else if (k === 'order') review.order = t.value;
        else if (k === 'unreviewed') review.unreviewedOnly = t.checked;
        review.selectedIndex = -1;
        renderPredictions(state.session);
    });
}

var _readoutHidden = null;

function onFrameReadoutClose() {
    _readoutHidden = frameReadout();
    renderPredictions(state.session);
}

function renderSummary(session) {
    var host = document.getElementById('behaviorSummaryBody');
    if (!host) return;
    var rows = session ? timeBudget(session) : [];
    if (!rows.length) { host.innerHTML = '<p class="table-empty">Nothing labelled yet.</p>'; return; }
    var order = new Map(behaviors().map(function (b, i) { return [b.id, i]; }));
    var animals = sessionAnimals(session);
    rows.sort(function (a, b) {
        var ai = animals.indexOf(a.animal), bi = animals.indexOf(b.animal);
        return (ai < 0 ? 1e9 : ai) - (bi < 0 ? 1e9 : bi) || (order.get(a.behaviorId) || 0) - (order.get(b.behaviorId) || 0);
    });
    var html = '<table class="data-table behavior-summary"><thead><tr><th>Animal</th><th>Behavior</th><th>Bouts</th>' +
        '<th>Total s</th><th>Mean s</th><th title="Present frames as a share of the frames labelled either way">Of labelled</th></tr></thead><tbody>';
    rows.forEach(function (r) {
        var b = behaviorById(r.behaviorId);
        var labelled = r.presentFrames + r.absentFrames;
        html += '<tr><td>' + esc(r.animal) + '</td><td>' + esc(b ? b.name : '?') + '</td><td class="num">' + n(r.bouts) + '</td>' +
            '<td class="num">' + secs(r.presentFrames) + '</td><td class="num">' + (r.bouts ? secs(r.boutFrames / r.bouts) : '–') + '</td>' +
            '<td class="num">' + (labelled ? Math.round(100 * r.presentFrames / labelled) + '%' : '–') + '</td></tr>';
    });
    html += '</tbody></table>';
    var orphans = orphanedAnimals(session, animals);
    if (orphans.length) {
        html += '<div class="behavior-orphans">' + orphans.map(function (o) {
            return '<div class="behavior-orphan" data-orphan="' + esc(o) + '">Labels of <b>' + esc(o) + '</b>, which no identity is called: ' +
                '<select data-act="orphan-to"><option value="">Reassign to…</option>' + animals.map(function (a) {
                    return '<option value="' + esc(a) + '">' + esc(a) + '</option>';
                }).join('') + '</select> <button class="panel-btn panel-btn-danger" data-act="orphan-delete">Delete them</button></div>';
        }).join('') + '</div>';
    }
    html += '<p class="behavior-fps-note">Seconds at ' + fps() + ' fps.</p>';
    host.innerHTML = html;
}

// ----------------------------------------------------------------------------
// Events (delegated on the static section bodies)
// ----------------------------------------------------------------------------

function findBout(key) {
    var all = listBouts(state.session);
    for (var i = 0; i < all.length; i++) if (boutKey(all[i]) === key) return all[i];
    return null;
}

function selectBoutRow(key, land) {
    panel.selectedKey = key;
    var bt = findBout(key);
    renderBouts(state.session);
    if (bt && land) goToBehaviorFrame(Math.max(0, bt.start - Math.round(BOUT_LEAD_IN_SECONDS * fps())));
}

function wireCoding(host) {
    host.addEventListener('click', function (e) {
        var session = state.session;
        var t = e.target.closest('button');
        if (!t || !session) return;
        if (t.dataset.animal) {
            if (!pickBehaviorTarget(t.dataset.animal)) setFocalAnimal(t.dataset.animal);
            return;
        }
        if (t.dataset.behavior) { codeBehavior(t.dataset.behavior, { absent: e.shiftKey }); return; }
        var act = t.dataset.act;
        if (act === 'cancel-target') cancelPendingTarget();
        else if (act === 'clear-range') clearArmedRange();
        else if (act === 'undo') { if (!undoBehaviorEdit(session)) setStatus('No behavior edit to undo'); }
        else if (act === 'reviewed') showMarkReviewedModal();
        else if (act === 'stop-run' || act === 'cancel-run') {
            var p = t.closest('[data-run]').dataset.run.split('|');
            if (act === 'cancel-run') cancelRunning(session, p[0], p[1], p[2]);
            else if (isRunning(session, p[0], p[1], p[2])) toggleState(session, p[0], p[1], p[2], state.currentFrame);
        }
    });
    host.addEventListener('change', function (e) {
        if (e.target.dataset.act === 'timeline-behavior') selectBehavior(e.target.value || null);
        else if (e.target.dataset.act === 'on-video') setBehaviorBoxesVisible(e.target.checked);
    });
}

function wireBouts(host) {
    host.addEventListener('click', function (e) {
        var session = state.session;
        if (!session) return;
        var t = e.target;
        if (t.closest('select, input')) return;
        var btn = t.closest('button');
        if (btn && btn.dataset.act === 'next-bout') {
            var rows = Array.from(host.querySelectorAll('.behavior-bout'));
            var i = rows.findIndex(function (r) { return r.dataset.key === panel.selectedKey; });
            var next = rows[i + 1] || rows[0];
            if (next) selectBoutRow(next.dataset.key, true);
            return;
        }
        var row = t.closest('.behavior-bout');
        if (!row) return;
        var bt = findBout(row.dataset.key);
        if (btn && bt) {
            var act = btn.dataset.act, res = null;
            if (act === 'set-start') res = editBout(session, bt, { start: Math.min(state.currentFrame, bt.end) });
            else if (act === 'set-end') res = editBout(session, bt, { end: Math.max(state.currentFrame, bt.start) });
            else if (act === 'delete-bout') { deleteBout(session, bt); panel.selectedKey = null; return; }
            if (res) panel.selectedKey = boutKey(Object.assign({}, bt, res));
            return;
        }
        if (row.dataset.key !== panel.selectedKey) selectBoutRow(row.dataset.key, true);
        else goToBehaviorFrame(Math.max(0, (bt ? bt.start : 0) - Math.round(BOUT_LEAD_IN_SECONDS * fps())));
    });
    host.addEventListener('change', function (e) {
        var session = state.session;
        var t = e.target;
        if (t.dataset.filter === 'animal') { panel.animal = t.value; renderBouts(session); return; }
        if (t.dataset.filter === 'behavior') { panel.behavior = t.value; renderBouts(session); return; }
        if (t.dataset.filter === 'absent') { panel.showAbsent = t.checked; renderBouts(session); return; }
        var row = t.closest('.behavior-bout');
        var bt = row && findBout(row.dataset.key);
        if (!bt) return;
        var changes = null;
        if (t.dataset.act === 'set-behavior') {
            var nb = behaviorById(t.value);
            changes = { behaviorId: t.value };
            if (nb && nb.social && !bt.target) {
                var other = sessionAnimals(session).filter(function (a) { return a !== bt.animal; })[0];
                if (other) changes.target = other;
            }
        } else if (t.dataset.act === 'set-target') changes = { target: t.value };
        else if (t.dataset.act === 'set-note') changes = { note: t.value };
        if (!changes) return;
        var res = editBout(session, bt, changes);
        if (!res) { setStatus('That change cannot be made to this bout', 'warning'); renderBouts(session); return; }
        var b2 = behaviorById(changes.behaviorId || bt.behaviorId);
        panel.selectedKey = boutKey({
            behaviorId: changes.behaviorId || bt.behaviorId, animal: bt.animal,
            target: b2 && b2.social ? (changes.target || bt.target) : '', start: res.start, end: res.end,
        });
    });
}

function wireSummary(host) {
    host.addEventListener('change', function (e) {
        if (e.target.dataset.act !== 'orphan-to' || !e.target.value) return;
        var from = e.target.closest('[data-orphan]').dataset.orphan;
        var k = reassignAnimal(state.session, from, e.target.value);
        setStatus('Moved ' + n(k) + ' bout' + (k === 1 ? '' : 's') + ' of ' + from + ' to ' + e.target.value, 'success');
    });
    host.addEventListener('click', function (e) {
        var btn = e.target.closest('button');
        if (!btn || btn.dataset.act !== 'orphan-delete') return;
        var name = btn.closest('[data-orphan]').dataset.orphan;
        var session = state.session;
        listBouts(session).filter(function (b) { return b.animal === name || b.target === name; })
            .forEach(function (b) { deleteBout(session, b); });
        setStatus('Deleted the labels of ' + name, 'success');
    });
}

// ----------------------------------------------------------------------------
// Per frame, and setup
// ----------------------------------------------------------------------------

/**
 * Called on every frame change (ui-wiring `updateSeekbarVisual`): keep the
 * save cursor at the playhead, grow running states, light the buttons.
 */
var _lastSession = null;

export function updateBehaviorFrame(frame) {
    var session = state.session;
    // A session switch ends the running states of the session left, where its
    // playhead last was (switchSession itself stays free of feature hooks).
    if (_lastSession && _lastSession !== session) closeRunningAtCursor(_lastSession);
    _lastSession = session;
    if (!session) return;
    setBehaviorCursor(session, frame);
    if (runningStates(session).length) requestBehaviorRepaint({ frame: true });
    if (tabVisible()) updateCodingNow(frame);
}

/**
 * Wire the feature once (from ui-wiring `setupMenus`).
 * @param {{closeMenus: Function, navigate: Function}} deps
 */
export function setupBehaviorUi(deps) {
    deps = deps || {};
    setBehaviorNavigator(deps.navigate);
    var closeMenus = deps.closeMenus || function () {};
    var on = function (id, fn) {
        var el = document.getElementById(id);
        if (el) el.addEventListener('click', function (e) { closeMenus(); fn(e); });
    };
    on('menuBehaviorMode', toggleBehaviorMode);
    on('menuBehaviorEthogram', showEthogramModal);
    on('menuBehaviorReviewed', showMarkReviewedModal);
    on('menuBehaviorImport', showBehaviorImportModal);
    on('menuBehaviorExport', showBehaviorExportModal);
    on('menuBehaviorDetect', detectAllFrames);
    on('menuBehaviorRules', showBehaviorRulesModal);
    on('tbBehaviorMode', toggleBehaviorMode);
    on('behaviorModeBtn', toggleBehaviorMode);
    on('behaviorModeExit', exitBehaviorMode);
    on('behaviorEthogramBtn', showEthogramModal);
    setHandler('behaviorMode', toggleBehaviorMode);
    persistSectionStates(['behaviorCodingSection', 'behaviorBoutsSection', 'behaviorPredictionsSection', 'behaviorSummarySection'], SECTIONS_KEY);
    var coding = document.getElementById('behaviorCodingBody');
    var bouts = document.getElementById('behaviorBoutsBody');
    var preds = document.getElementById('behaviorPredictionsBody');
    var summary = document.getElementById('behaviorSummaryBody');
    if (coding) wireCoding(coding);
    if (bouts) wireBouts(bouts);
    if (preds) wirePredictions(preds);
    if (summary) wireSummary(summary);
    var info = document.getElementById('behaviorPredictionsInfo');
    if (info) { info.innerHTML = ICON_INFO; setInfoTip(info, PREDICTIONS_INFO); }
    setupBehaviorDetection({ openSettings: showBehaviorRulesModal });
    var tabBtn = document.querySelector('.panel-tab[data-tab="tabBehavior"]');
    if (tabBtn) tabBtn.addEventListener('click', function () { setTimeout(refreshBehaviorPanel, 0); });

    // Every label edit: the project is dirty, and everything that shows labels repaints.
    onBehaviorChange(function () {
        markDirty();
        requestBehaviorRepaint({ labels: true });
    });
    // A detection run, cleared predictions, or predictions going stale.
    onPredictionsChange(function () { requestBehaviorRepaint({ predictions: true }); });
    onFrameReadout(function () {
        _readoutHidden = null;
        openBehaviorTabSection();
        requestBehaviorRepaint({ predictions: true });
    });
    registerBehaviorRepaint(function (info) {
        if (info.labels || info.ui || info.ethogram || info.predictions) refreshBehaviorPanel();
        // The video boxes follow the labels and the target question (`ui`);
        // the toolbar's blue Behavior button follows the predictions.
        if ((info.labels || info.ethogram || info.ui || info.predictions) && state.session) drawAllOverlays(state.currentFrame);
    });
}
