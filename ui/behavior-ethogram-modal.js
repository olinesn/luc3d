// ui/behavior-ethogram-modal.js — Behavior ▸ Edit Ethogram… (the project's
// behaviors) and Behavior ▸ Mark Range Reviewed….
//
// The ethogram editor applies each field as it is changed (a refused change
// says why and reverts), so there is no Apply step to forget; Esc or Close
// leave. Removing a behavior that has bouts asks first and deletes its bouts
// in every loaded session. Ethogram JSON can be exported and imported (merged
// by behavior name), to reuse a scheme across projects; Import also takes a
// BORIS project's ethogram (exclusions become groups where they can).
//
// Mark Range Reviewed turns the unlabelled frames of a range into "not"
// labels, for the chosen animals and behaviors — what a classifier needs to
// learn what a behavior is not.
//
// Depends on: ui/app-state.js, ui/behavior-coding.js, ui/behavior-confirm.js,
//             pose/behavior-ethogram.js, pose/behavior-labels.js,
//             import-export/behavior-io.js, import-export/save-load.js,
//             loading/cold-read-guard.js

import { state } from './app-state.js';
import { sessionAnimals, focalAnimal, armedRange, requestBehaviorRepaint } from './behavior-coding.js';
import { confirmRemoveBehavior, confirmMakeMutual } from './behavior-confirm.js';
import {
    behaviors, addBehavior, updateBehavior, removeBehavior, moveBehavior, serializeEthogram, ingestEthogram,
    behaviorById, behaviorByName, BEHAVIOR_KINDS,
} from '../pose/behavior-ethogram.js';
import { countBehaviorBouts, deleteBehaviorBouts, markReviewed, rekeyBehavior, bothWayPairs } from '../pose/behavior-labels.js';
import { parseBehaviorFile, ethogramPayload } from '../import-export/behavior-io.js';
import { markDirty, setStatus } from '../import-export/save-load.js';
import { readBlobArrayBuffer } from '../loading/cold-read-guard.js';

function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/** The open dialog's overlay, so a second open is ignored. */
var _open = null;

function openOverlay(className, html) {
    var overlay = document.createElement('div');
    overlay.className = 'multi-frame-modal-overlay behavior-modal-overlay';
    var modal = document.createElement('div');
    modal.className = 'multi-frame-modal ' + className;
    modal.innerHTML = html;
    overlay.appendChild(modal);
    document.body.appendChild(overlay);
    return { overlay: overlay, modal: modal };
}

function ethogramChanged() {
    markDirty();
    requestBehaviorRepaint({ ethogram: true });
}

// ----------------------------------------------------------------------------
// Ethogram editor
// ----------------------------------------------------------------------------

function rowHtml(b, i, n) {
    return '<tr data-id="' + esc(b.id) + '">' +
        '<td><input type="color" data-f="color" value="' + esc(b.color) + '" title="Color"></td>' +
        '<td><input type="text" data-f="name" value="' + esc(b.name) + '" maxlength="40" class="behavior-ed-name"></td>' +
        '<td><input type="text" data-f="key" value="' + esc(b.key) + '" maxlength="1" class="behavior-ed-key" title="One letter, coded in Label Behaviors mode"></td>' +
        '<td><select data-f="kind">' + BEHAVIOR_KINDS.map(function (k) {
            return '<option value="' + k + '"' + (k === b.kind ? ' selected' : '') + '>' + (k === 'state' ? 'State' : 'Point') + '</option>';
        }).join('') + '</select></td>' +
        '<td><select data-f="socialKind" title="Social: done to a target animal (→), or by a pair with no actor and target (↔ mutual)">' +
            [['none', '—'], ['directed', '→ target'], ['mutual', '↔ mutual']].map(function (o) {
                var cur = b.mutual ? 'mutual' : b.social ? 'directed' : 'none';
                return '<option value="' + o[0] + '"' + (o[0] === cur ? ' selected' : '') + '>' + o[1] + '</option>';
            }).join('') + '</select></td>' +
        '<td><input type="text" data-f="group" value="' + esc(b.group) + '" maxlength="40" class="behavior-ed-group" ' +
            'title="Behaviors sharing a group are mutually exclusive for one animal"></td>' +
        '<td><input type="text" data-f="description" value="' + esc(b.description) + '" maxlength="500" class="behavior-ed-desc"></td>' +
        '<td class="behavior-ed-move"><button data-act="up"' + (i === 0 ? ' disabled' : '') + ' title="Move up">▲</button>' +
        '<button data-act="down"' + (i === n - 1 ? ' disabled' : '') + ' title="Move down">▼</button>' +
        '<button data-act="remove" title="Remove this behavior">✕</button></td></tr>';
}

/** Behavior ▸ Edit Ethogram…. */
export function showEthogramModal() {
    if (_open) return;
    var dlg = openOverlay('behavior-ethogram-modal',
        '<h3>Ethogram</h3>' +
        '<p class="behavior-ed-lead">The behaviors you will label, for every session of this project. A <b>state</b> has a start and an end; ' +
        'a <b>point</b> is one frame. A <b>social</b> behavior is done to a target animal (→), or is <b>mutual</b> (↔): ' +
        'a pair with no actor and target, like nose-to-nose sniffing. Behaviors in the same <b>group</b> exclude each other.</p>' +
        '<div class="behavior-ed-table-wrap"><table class="data-table behavior-ed-table"><thead><tr><th></th><th>Name</th><th>Key</th>' +
        '<th>Kind</th><th>Social</th><th>Group</th><th>Description</th><th></th></tr></thead><tbody id="behaviorEdRows"></tbody></table></div>' +
        '<div class="behavior-ed-msg" id="behaviorEdMsg" role="status"></div>' +
        '<div class="modal-actions behavior-ed-actions">' +
        '<button id="behaviorEdAdd">+ Add behavior</button>' +
        '<button id="behaviorEdImport" title="Add the behaviors of an ethogram JSON file or a BORIS project (matched by name)">Import…</button>' +
        '<button id="behaviorEdExport" title="Save this ethogram as JSON, to reuse in another project">Export…</button>' +
        '<span class="behavior-ed-spacer"></span>' +
        '<button class="primary" id="behaviorEdClose">Close</button></div>' +
        '<input type="file" id="behaviorEdFile" accept=".json,.boris,application/json" hidden>');
    _open = dlg.overlay;
    var modal = dlg.modal;
    var tbody = modal.querySelector('#behaviorEdRows');
    var msg = modal.querySelector('#behaviorEdMsg');

    function render(focusId) {
        var list = behaviors();
        tbody.innerHTML = list.length ? list.map(function (b, i) { return rowHtml(b, i, list.length); }).join('')
            : '<tr><td colspan="8" class="table-empty">No behaviors yet — add the first one.</td></tr>';
        if (focusId) {
            var inp = tbody.querySelector('tr[data-id="' + focusId + '"] [data-f="name"]');
            if (inp) { inp.focus(); inp.select(); }
        }
    }
    function say(text, warn) {
        msg.textContent = text || '';
        msg.classList.toggle('is-warning', !!warn);
    }

    function close() {
        document.removeEventListener('keydown', onKey, true);
        dlg.overlay.remove();
        _open = null;
    }
    function onKey(e) {
        if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); close(); }
    }
    document.addEventListener('keydown', onKey, true);

    tbody.addEventListener('change', function (e) {
        var tr = e.target.closest('tr[data-id]');
        if (!tr) return;
        var f = e.target.dataset.f;
        if (f === 'socialKind') { changeSocialKind(tr.dataset.id, e.target.value); return; }
        var fields = {};
        fields[f] = e.target.type === 'checkbox' ? e.target.checked : e.target.value;
        var res = updateBehavior(tr.dataset.id, fields);
        if (res.error) { say(res.error, true); render(); return; }
        say('');
        // Show the stored form (a key is kept lower-case, a name trimmed) in
        // place: re-rendering the table here would take the focus away from
        // the field the user just tabbed into.
        if (e.target.type === 'text') e.target.value = res.behavior[f];
        ethogramChanged();
    });
    tbody.addEventListener('click', function (e) {
        var btn = e.target.closest('button[data-act]');
        var tr = btn && btn.closest('tr[data-id]');
        if (!tr) return;
        var id = tr.dataset.id;
        if (btn.dataset.act === 'up' || btn.dataset.act === 'down') {
            if (moveBehavior(id, btn.dataset.act === 'up' ? -1 : 1)) { ethogramChanged(); render(); }
            return;
        }
        var b = behaviorById(id);
        var sessions = (state.sessions && state.sessions.length) ? state.sessions : (state.session ? [state.session] : []);
        var total = 0, holding = 0;
        sessions.forEach(function (s) { var c = countBehaviorBouts(s, id); total += c; if (c) holding++; });
        confirmRemoveBehavior(b, total, holding).then(function (ok) {
            if (!ok) return;
            sessions.forEach(function (s) { deleteBehaviorBouts(s, id); });
            removeBehavior(id);
            ethogramChanged();
            render();
            say('Removed "' + b.name + '"' + (total ? ' and its ' + total.toLocaleString('en-US') + ' bouts' : ''));
        });
    });
    // Social kind: none / directed (→ target) / mutual (↔). Making a behavior
    // mutual merges bouts labelled both ways round (A → B and B → A), so that
    // asks first when there are any; its bouts are then stored as pairs.
    function changeSocialKind(id, kind) {
        var b = behaviorById(id);
        var sessions = (state.sessions && state.sessions.length) ? state.sessions : (state.session ? [state.session] : []);
        var toMutual = kind === 'mutual' && !b.mutual;
        var merging = 0, holding = 0;
        if (toMutual) sessions.forEach(function (s) { var c = bothWayPairs(s, id); merging += c; if (c) holding++; });
        (toMutual ? confirmMakeMutual(b, merging, holding) : Promise.resolve(true)).then(function (ok) {
            if (!ok) { render(); return; }
            var res = updateBehavior(id, { social: kind !== 'none', mutual: kind === 'mutual' });
            if (res.error) { say(res.error, true); render(); return; }
            if (toMutual) sessions.forEach(function (s) { rekeyBehavior(s, id); });
            say(kind === 'mutual' ? '"' + b.name + '" is mutual: one bout per pair' : '');
            ethogramChanged();
        });
    }

    modal.querySelector('#behaviorEdAdd').addEventListener('click', function () {
        var k = behaviors().length + 1, name = 'behavior ' + k;
        while (behaviors().some(function (b) { return b.name.toLowerCase() === name; })) name = 'behavior ' + (++k);
        var res = addBehavior({ name: name });
        if (res.error) { say(res.error, true); return; }
        ethogramChanged();
        render(res.behavior.id);
    });
    modal.querySelector('#behaviorEdExport').addEventListener('click', function () {
        var payload = serializeEthogram();
        if (!payload) { say('The ethogram is empty', true); return; }
        var blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
        var a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = 'ethogram.json';
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
        say('Saved ethogram.json');
    });
    var file = modal.querySelector('#behaviorEdFile');
    modal.querySelector('#behaviorEdImport').addEventListener('click', function () { file.click(); });
    file.addEventListener('change', function () {
        var f = file.files && file.files[0];
        if (!f) return;
        readBlobArrayBuffer(f).then(function (buf) {
            var parsed = parseBehaviorFile(f.name, new TextDecoder('utf-8').decode(new Uint8Array(buf)));
            if (parsed.error || (parsed.kind !== 'ethogram' && parsed.kind !== 'boris-project')) {
                throw new Error(parsed.error || 'it holds labels, not an ethogram (use Behavior ▸ Import Labels…)');
            }
            var before = behaviors().length, groups = [];
            if (parsed.kind === 'ethogram') ingestEthogram(parsed.ethogramPayload);
            else {
                var fresh = parsed.ethogram.filter(function (e) { return !behaviorByName(e.name); });
                var res = ethogramPayload(fresh, behaviors().map(function (b) { return b.group; }).filter(Boolean));
                ingestEthogram(res.payload);
                groups = res.groups;
            }
            var added = behaviors().length - before;
            if (added) ethogramChanged();
            render();
            say(added ? 'Added ' + added + ' behavior' + (added === 1 ? '' : 's') + ' from ' + f.name +
                (groups.length ? ' (BORIS exclusions became ' + groups.length + ' group' + (groups.length === 1 ? '' : 's') + ')' : '')
                : 'No new behaviors in ' + f.name);
        }).catch(function (err) {
            say('Could not read ' + f.name + ': ' + err.message, true);
        }).finally(function () { file.value = ''; });
    });
    modal.querySelector('#behaviorEdClose').addEventListener('click', close);
    render();
}

// ----------------------------------------------------------------------------
// Mark Range Reviewed
// ----------------------------------------------------------------------------

/** Behavior ▸ Mark Range Reviewed…. */
export function showMarkReviewedModal() {
    if (_open) return;
    var session = state.session;
    if (!session || !behaviors().length) { setStatus('Add behaviors to the ethogram first', 'warning'); return; }
    var animals = sessionAnimals(session);
    if (!animals.length) { setStatus('No animals to review: behaviors are labelled per identity', 'warning'); return; }
    var range = armedRange();
    var last = Math.max(0, (session.numFrames || state.totalFrames || 1) - 1);
    var from = range ? range.start : 0, to = range ? range.end : last;
    var focal = focalAnimal(session);
    var dlg = openOverlay('behavior-reviewed-modal',
        '<h3>Mark range reviewed</h3>' +
        '<p class="behavior-ed-lead">Frames in the range that carry no label for a behavior become <b>"not" that behavior</b>. ' +
        'Labels already there are kept.</p>' +
        '<div class="behavior-rev-range"><label>Frames <input type="number" id="behaviorRevFrom" min="1" max="' + (last + 1) + '" value="' + (from + 1) + '"></label>' +
        '<label>to <input type="number" id="behaviorRevTo" min="1" max="' + (last + 1) + '" value="' + (to + 1) + '"></label>' +
        '<button id="behaviorRevAll">Whole session</button></div>' +
        '<div class="behavior-rev-cols"><fieldset><legend>Animals</legend>' + animals.map(function (a) {
            return '<label><input type="checkbox" data-animal="' + esc(a) + '"' + (a === focal ? ' checked' : '') + '> ' + esc(a) + '</label>';
        }).join('') + '</fieldset><fieldset><legend>Behaviors</legend>' + behaviors().map(function (b) {
            return '<label><input type="checkbox" data-behavior="' + esc(b.id) + '" checked> ' + esc(b.name) + '</label>';
        }).join('') + '</fieldset></div>' +
        '<div class="modal-actions"><button id="behaviorRevCancel">Cancel</button><button class="primary" id="behaviorRevApply">Mark reviewed</button></div>');
    _open = dlg.overlay;
    var modal = dlg.modal;
    function close() {
        document.removeEventListener('keydown', onKey, true);
        dlg.overlay.remove();
        _open = null;
    }
    function onKey(e) {
        if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); close(); }
    }
    document.addEventListener('keydown', onKey, true);
    modal.querySelector('#behaviorRevAll').addEventListener('click', function () {
        modal.querySelector('#behaviorRevFrom').value = 1;
        modal.querySelector('#behaviorRevTo').value = last + 1;
    });
    modal.querySelector('#behaviorRevCancel').addEventListener('click', close);
    modal.querySelector('#behaviorRevApply').addEventListener('click', function () {
        var s = parseInt(modal.querySelector('#behaviorRevFrom').value, 10) - 1;
        var e = parseInt(modal.querySelector('#behaviorRevTo').value, 10) - 1;
        var pickA = Array.from(modal.querySelectorAll('[data-animal]:checked')).map(function (c) { return c.dataset.animal; });
        var pickB = Array.from(modal.querySelectorAll('[data-behavior]:checked')).map(function (c) { return c.dataset.behavior; });
        if (!Number.isFinite(s) || !Number.isFinite(e) || !pickA.length || !pickB.length) {
            setStatus('Pick a range, at least one animal and one behavior', 'warning');
            return;
        }
        s = Math.max(0, Math.min(last, s)); e = Math.max(0, Math.min(last, e));
        var filled = markReviewed(session, pickB, pickA, animals, Math.min(s, e), Math.max(s, e));
        close();
        setStatus(filled ? filled.toLocaleString('en-US') + ' unlabelled frame-labels became "not" labels' : 'Everything in that range was already labelled', 'success');
    });
}
