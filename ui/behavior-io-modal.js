// ui/behavior-io-modal.js — Behavior ▸ Export Labels… and Behavior ▸ Import
// Labels…: the dialogs over import-export/behavior-io.js.
//
// Export writes one table (bouts, time budget, BORIS aggregated or tabular
// events) for this session or all of them, or one per-frame table per animal —
// into a folder picked once where the browser can (`showDirectoryPicker`),
// else as separate downloads. The choices are remembered in this browser.
//
// Import reads a BORIS project, BORIS event table, LUC3D bouts table or
// ethogram JSON, then shows what it would do before it does it: which
// observation, who each subject is (an animal or Skip), the behaviors it adds
// to the ethogram, and how many bouts it labels and skips. The bouts land in
// the active session as ONE edit (`labelBouts`), so Undo takes the whole import
// back; labels already on those frames are replaced, as labelling by hand does.
//
// Depends on: ui/app-state.js, ui/behavior-coding.js, ui/behavior-detect.js, pose/behavior-ethogram.js,
//             pose/behavior-labels.js, import-export/behavior-io.js,
//             import-export/save-load.js, loading/cold-read-guard.js

import { state } from './app-state.js';
import { sessionAnimals, focalAnimal, requestBehaviorRepaint } from './behavior-coding.js';
import { sessionFps } from './behavior-detect.js';
import { behaviors, behaviorByName, ingestEthogram } from '../pose/behavior-ethogram.js';
import { labelBouts, hasBehaviorData, noteBehaviorFps } from '../pose/behavior-labels.js';
import {
    toDelimited, boutsTable, perFrameTables, timeBudgetTable, borisAggregatedTable, borisTabularTable,
    parseBehaviorFile, ethogramPayload, defaultObservation, defaultSubjectMap, newBehaviorsFor, planBehaviorImport,
} from '../import-export/behavior-io.js';
import { markDirty, setStatus } from '../import-export/save-load.js';
import { readBlobArrayBuffer } from '../loading/cold-read-guard.js';

var PREFS_KEY = 'lucid.behaviorExport';

function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function n(x) { return Number(x).toLocaleString('en-US'); }

function plural(k, one, many) { return n(k) + ' ' + (k === 1 ? one : (many || one + 's')); }

/** A name made safe for a file name. */
function safeName(s) {
    return String(s || 'session').replace(/[\\/:*?"<>|]+/g, '_').replace(/\s+/g, '_').slice(0, 80) || 'session';
}

function openOverlay(className, html) {
    var overlay = document.createElement('div');
    overlay.className = 'multi-frame-modal-overlay behavior-modal-overlay';
    var modal = document.createElement('div');
    modal.className = 'multi-frame-modal behavior-io-modal ' + className;
    modal.innerHTML = html;
    overlay.appendChild(modal);
    document.body.appendChild(overlay);
    var onKey = function (e) {
        if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); close(); }
    };
    function close() {
        document.removeEventListener('keydown', onKey, true);
        overlay.remove();
    }
    document.addEventListener('keydown', onKey, true);
    return { overlay: overlay, modal: modal, close: close };
}

function isOpen() {
    return !!document.querySelector('.behavior-io-modal');
}

function sessionsOf() {
    return (state.sessions && state.sessions.length) ? state.sessions : (state.session ? [state.session] : []);
}

function maxKey(map) {
    var m = -1;
    if (map && typeof map.forEach === 'function') map.forEach(function (_, k) { if (k > m) m = k; });
    return m;
}

/** The video files of a session (`state.videoFiles` entries). */
function sessionVideos(session) {
    var files = (session.videoFileIndices || []).map(function (i) { return state.videoFiles && state.videoFiles[i]; }).filter(Boolean);
    if (!files.length && session === state.session) {
        files = (state.videoFiles || []).filter(function (vf) { return vf && (vf.sessionIdx == null || vf.sessionIdx === state.activeSessionIdx); });
    }
    return files;
}

function videoCamera(vf) { return vf.assignedCamera || vf.name || ''; }

function videoPath(vf) { return vf ? (vf.videoPath || (vf.file && vf.file.name) || vf.name || '') : ''; }

/**
 * What the exports need to know about a session: its name, frame rate, length,
 * animals, and the video named in BORIS tables (`camera`'s, else the first).
 */
export function behaviorSessionInfo(session, camera) {
    var active = session === state.session;
    var fps = sessionFps(session);
    var frames = active ? (state.totalFrames || 0) : (session.totalFrames || 0);
    if (!frames) frames = session.lazyLoader ? session.numFrames : Math.max(maxKey(session.frameGroups), maxKey(session.instanceGroups)) + 1;
    var files = sessionVideos(session);
    var vf = null;
    for (var i = 0; i < files.length && camera; i++) if (videoCamera(files[i]) === camera) { vf = files[i]; break; }
    return { session: session, name: session.name || 'Session', fps: fps, frames: Math.max(0, frames || 0),
             animals: sessionAnimals(session), media: videoPath(vf || files[0]) };
}

function loadPrefs() {
    try { return JSON.parse(localStorage.getItem(PREFS_KEY) || '{}') || {}; } catch (e) { return {}; }
}

function savePrefs(p) {
    try { localStorage.setItem(PREFS_KEY, JSON.stringify(p)); } catch (e) { /* best effort */ }
}

function downloadText(text, filename, type) {
    var blob = new Blob([text], { type: type + ';charset=utf-8' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
}

// ----------------------------------------------------------------------------
// Export
// ----------------------------------------------------------------------------

var TABLES = [
    { id: 'bouts', name: 'Bouts', file: 'behavior-bouts', text: 'One row per labelled bout, present and "not", in frames and seconds.' },
    { id: 'frames', name: 'Per-frame table', file: 'behavior-frames', text: 'One file per animal: 1, 0 or blank (unlabelled) on every frame, a column per behavior.' },
    { id: 'budget', name: 'Time budget', file: 'behavior-time-budget', text: 'Per animal and behavior: bouts, time, and how much of the session was labelled.' },
    { id: 'boris-aggregated', name: 'BORIS aggregated events', file: 'boris-aggregated', text: 'One row per bout in BORIS’s layout. Present bouts only: BORIS has no "not" labels.' },
    { id: 'boris-tabular', name: 'BORIS tabular events', file: 'boris-tabular', text: 'START, STOP and POINT rows in BORIS’s layout, which SimBA imports. Present bouts only.' },
];

/** Behavior ▸ Export Labels…. */
export function showBehaviorExportModal() {
    if (isOpen()) return;
    var session = state.session;
    var sessions = sessionsOf();
    if (!session || !sessions.some(hasBehaviorData)) { setStatus('No behavior labels to export yet', 'warning'); return; }
    var prefs = loadPrefs();
    var table = TABLES.some(function (t) { return t.id === prefs.table; }) ? prefs.table : 'bouts';
    var cameras = sessionVideos(session).map(videoCamera).filter(Boolean);
    var fps = behaviorSessionInfo(session).fps;
    var dlg = openOverlay('behavior-export-modal',
        '<h3>Export behavior labels</h3>' +
        '<div class="behavior-io-choices" role="radiogroup" aria-label="Table">' + TABLES.map(function (t) {
            return '<label class="behavior-io-choice"><input type="radio" name="bxTable" value="' + t.id + '"' + (t.id === table ? ' checked' : '') + '>' +
                '<span><b>' + esc(t.name) + '</b><span class="behavior-io-choice-text">' + esc(t.text) + '</span></span></label>';
        }).join('') + '</div>' +
        '<div class="behavior-io-opts">' +
        (sessions.length > 1 ? '<label>Sessions <select id="bxScope"><option value="one">This session</option>' +
            '<option value="all"' + (prefs.scope === 'all' ? ' selected' : '') + '>All ' + n(sessions.length) + ' sessions</option></select></label>' : '') +
        (cameras.length > 1 ? '<label id="bxCamRow" title="The video BORIS tables name as their media file">Video <select id="bxCam">' + cameras.map(function (c) {
            return '<option value="' + esc(c) + '"' + (c === prefs.camera ? ' selected' : '') + '>' + esc(c) + '</option>';
        }).join('') + '</select></label>' : '') +
        '<label>Separator <select id="bxSep"><option value=",">Comma (.csv)</option>' +
        '<option value="tab"' + (prefs.sep === 'tab' ? ' selected' : '') + '>Tab (.tsv)</option></select></label></div>' +
        '<p class="behavior-ed-lead">Seconds at ' + (fps > 0 ? +fps.toFixed(3) : '?') + ' fps. Frame numbers count from 0, ' +
        'as SLEAP and pandas do: the timeline’s frame 1 is frame 0.</p>' +
        '<div class="modal-actions"><button id="bxCancel">Cancel</button><button class="primary" id="bxGo">Export</button></div>');
    var modal = dlg.modal;
    var camRow = modal.querySelector('#bxCamRow');
    var tableNow = function () { var r = modal.querySelector('input[name="bxTable"]:checked'); return r ? r.value : 'bouts'; };
    var showCam = function () { if (camRow) camRow.hidden = !/^boris/.test(tableNow()); };
    modal.querySelector('.behavior-io-choices').addEventListener('change', showCam);
    showCam();
    modal.querySelector('#bxCancel').addEventListener('click', dlg.close);
    modal.querySelector('#bxGo').addEventListener('click', function () {
        var t = tableNow();
        var scopeSel = modal.querySelector('#bxScope'), camSel = modal.querySelector('#bxCam');
        var choice = {
            table: t, scope: scopeSel ? scopeSel.value : 'one', camera: camSel ? camSel.value : '',
            sep: modal.querySelector('#bxSep').value === 'tab' ? 'tab' : 'comma',
        };
        savePrefs(choice);
        // The folder picker needs the click's user activation: ask before any work.
        var dirPick = t === 'frames' && typeof window.showDirectoryPicker === 'function'
            ? window.showDirectoryPicker({ mode: 'readwrite' }) : null;
        dlg.close();
        runExport(choice, dirPick);
    });
}

async function runExport(choice, dirPick) {
    var sep = choice.sep === 'tab' ? '\t' : ',';
    var ext = choice.sep === 'tab' ? '.tsv' : '.csv';
    var type = choice.sep === 'tab' ? 'text/tab-separated-values' : 'text/csv';
    var pick = choice.scope === 'all' ? sessionsOf() : [state.session];
    var infos = pick.filter(Boolean).map(function (s) { return behaviorSessionInfo(s, choice.camera); });
    var meta = TABLES.filter(function (t) { return t.id === choice.table; })[0] || TABLES[0];
    var base = choice.scope === 'all' && infos.length > 1 ? 'all-sessions' : safeName(infos[0] && infos[0].name);
    try {
        if (choice.table !== 'frames') {
            var rows = choice.table === 'budget' ? timeBudgetTable(infos)
                : choice.table === 'boris-aggregated' ? borisAggregatedTable(infos)
                : choice.table === 'boris-tabular' ? borisTabularTable(infos)
                : boutsTable(infos);
            var name = base + '_' + meta.file + ext;
            downloadText(toDelimited(rows, sep), name, type);
            setStatus('Exported ' + name + ' (' + plural(rows.length - 1, 'row') + ')', 'success');
            return;
        }
        var files = [];
        infos.forEach(function (info) {
            perFrameTables(info, sep).forEach(function (tbl) {
                files.push({ name: safeName(info.name) + '_' + safeName(tbl.animal) + '_' + meta.file + ext, text: tbl.text });
            });
        });
        if (!files.length) { setStatus('No behavior labels to export', 'warning'); return; }
        var dir = null;
        if (dirPick) {
            try { dir = await dirPick; } catch (err) {
                if (err && err.name === 'AbortError') { setStatus('Export cancelled', 'info'); return; }
                throw err;
            }
        }
        for (var i = 0; i < files.length; i++) {
            if (dir) {
                var fh = await dir.getFileHandle(files[i].name, { create: true });
                var w = await fh.createWritable();
                await w.write(files[i].text);
                await w.close();
            } else {
                downloadText(files[i].text, files[i].name, type);
            }
        }
        setStatus('Exported ' + plural(files.length, 'per-frame table') + (dir ? ' to ' + dir.name : ''), 'success');
    } catch (err) {
        console.error('[behavior-export]', err);
        setStatus('Export failed: ' + (err && err.message ? err.message : err), 'error');
    }
}

// ----------------------------------------------------------------------------
// Import
// ----------------------------------------------------------------------------

function groupsInUse() {
    return behaviors().map(function (b) { return b.group; }).filter(Boolean);
}

function ethogramChanged() {
    markDirty();
    requestBehaviorRepaint({ ethogram: true });
}

/** Behavior ▸ Import Labels…: pick a file, then the import dialog. */
export function showBehaviorImportModal() {
    if (isOpen()) return;
    var input = document.createElement('input');
    input.type = 'file';
    input.accept = '.boris,.json,.csv,.tsv,.txt';
    input.hidden = true;
    document.body.appendChild(input);
    input.addEventListener('cancel', function () { input.remove(); });
    input.addEventListener('change', function () {
        var f = input.files && input.files[0];
        input.remove();
        if (f) importBehaviorFile(f);
    });
    input.click();
}

/**
 * Read a behavior file and open the import dialog for it.
 * @param {File} file
 */
export async function importBehaviorFile(file) {
    var text;
    try {
        text = new TextDecoder('utf-8').decode(new Uint8Array(await readBlobArrayBuffer(file)));
    } catch (err) {
        setStatus('Could not read ' + file.name + ': ' + (err && err.message ? err.message : err), 'error');
        return;
    }
    var parsed = parseBehaviorFile(file.name, text);
    if (parsed.error) { setStatus(file.name + ': ' + parsed.error, 'error'); return; }
    openImportDialog(file.name, parsed);
}

function openImportDialog(fileName, parsed) {
    if (isOpen()) return;
    var session = state.session;
    var animals = session ? sessionAnimals(session) : [];
    var timed = parsed.kind !== 'lucid-bouts';
    var observations = parsed.observations.filter(function (o) { return o.bouts.length || o.unpaired; });
    var canLabel = !!session && observations.length > 0 && animals.length > 0;
    var info = session ? behaviorSessionInfo(session) : { fps: 0, frames: 0, name: '' };
    // The file's own ethogram (a BORIS project or an ethogram JSON): what it would add.
    var fileEntries = parsed.ethogram ? parsed.ethogram.filter(function (e) { return !behaviorByName(e.name); }) : [];
    var jsonNew = parsed.ethogramPayload ? (parsed.ethogramPayload.behaviors || []).filter(function (e) {
        return e && typeof e.name === 'string' && e.name.trim() && !behaviorByName(e.name);
    }).length : 0;
    var ui = {
        obs: observations.length ? Math.max(0, observations.indexOf(parsed.observations[defaultObservation(parsed, info.name,
            session ? sessionVideos(session).map(videoPath) : [])])) : -1,
        map: null, addBehaviors: true, labels: canLabel,
    };
    var obsNow = function () { return ui.obs >= 0 ? observations[ui.obs] : null; };
    var resetMap = function () { var o = obsNow(); ui.map = o ? defaultSubjectMap(o, animals, focalAnimal(session)) : {}; };
    resetMap();

    var html = '<h3>Import behavior labels</h3>' +
        '<p class="behavior-io-file"><b>' + esc(fileName) + '</b> · ' + esc(parsed.label) + '</p>';
    if (parsed.kind === 'ethogram' || parsed.kind === 'boris-project') {
        var count = parsed.kind === 'ethogram' ? jsonNew : fileEntries.length;
        html += '<label class="behavior-io-check"><input type="checkbox" id="biAdd"' + (count ? ' checked' : ' disabled') + '> ' +
            (count ? 'Add its ' + plural(count, 'new behavior') + ' to the ethogram' : 'Every behavior in it is already in the ethogram') + '</label>' +
            '<p class="behavior-io-detail" id="biGroups"></p>';
    }
    if (observations.length) {
        if (!session || !animals.length) {
            html += '<p class="behavior-io-warn">' + (session ? 'This session has no animals to label: behaviors are labelled per identity.'
                : 'Open a session to import its labels.') + '</p>';
        } else {
            html += (parsed.kind === 'boris-project'
                ? '<label class="behavior-io-check"><input type="checkbox" id="biLabels" checked> Import labels into <b>' + esc(info.name) + '</b></label>'
                : '<p class="behavior-io-into">Into session <b>' + esc(info.name) + '</b></p>') +
                '<div class="behavior-io-body" id="biBody">' +
                (observations.length > 1 ? '<label class="behavior-io-obs">' + (timed ? 'Observation' : 'Session in the file') + ' <select id="biObs">' +
                    observations.map(function (o, i) {
                        return '<option value="' + i + '"' + (i === ui.obs ? ' selected' : '') + '>' + esc(o.id) + ' (' + plural(o.bouts.length, 'bout') + ')</option>';
                    }).join('') + '</select></label>' : '') +
                '<p class="behavior-io-detail" id="biFps"></p>' +
                '<table class="data-table behavior-io-subjects"><thead><tr><th>In the file</th><th>Animal</th></tr></thead><tbody id="biSubjects"></tbody></table>' +
                '<label class="behavior-io-check" id="biNewRow"><input type="checkbox" id="biNew" checked> <span id="biNewText"></span></label>' +
                '<p class="behavior-io-summary" id="biSummary" role="status"></p>' +
                '<p class="behavior-ed-lead">Labels already on those frames are replaced. Undo takes the whole import back.</p></div>';
        }
    }
    html += '<div class="modal-actions"><button id="biCancel">Cancel</button><button class="primary" id="biGo">Import</button></div>';
    var dlg = openOverlay('behavior-import-modal', html);
    var modal = dlg.modal;
    var $ = function (sel) { return modal.querySelector(sel); };

    function subjectLabel(s) { return s === '' ? '(no subject)' : s; }

    function renderSubjects() {
        var o = obsNow(), body = $('#biSubjects');
        if (!o || !body) return;
        body.innerHTML = o.subjects.map(function (s, i) {
            return '<tr><td>' + esc(subjectLabel(s)) + '</td><td><select data-subject="' + i + '"><option value="">Skip</option>' +
                animals.map(function (a) { return '<option value="' + esc(a) + '"' + (ui.map[s] === a ? ' selected' : '') + '>' + esc(a) + '</option>'; }).join('') +
                '</select></td></tr>';
        }).join('') || '<tr><td colspan="2" class="table-empty">No subjects</td></tr>';
    }

    /** Behaviors the chosen observation's bouts use that the ethogram lacks, beyond a BORIS project's own. */
    function extraEntries() {
        var o = obsNow();
        if (!o) return [];
        return newBehaviorsFor(parsed, o, animals).filter(function (e) { return fileEntries.indexOf(e) < 0; });
    }

    /** Everything the import would add to the ethogram (an ethogram JSON is ingested as it is). */
    function addingEntries() {
        var out = parsed.kind === 'boris-project' && ui.addBehaviors ? fileEntries.slice() : [];
        if (ui.labels && ui.addNew !== false) extraEntries().forEach(function (e) { out.push(e); });
        return out;
    }

    function plan() {
        var o = obsNow();
        if (!o) return null;
        return planBehaviorImport(parsed, o, { fps: info.fps, frames: info.frames, animals: animals, subjectMap: ui.map,
                                               newBehaviors: addingEntries() });
    }

    function update() {
        var add = $('#biAdd');
        ui.addBehaviors = add ? add.checked && !add.disabled : true;
        var lab = $('#biLabels');
        ui.labels = canLabel && (lab ? lab.checked : true);
        var bodyEl = $('#biBody');
        if (bodyEl) bodyEl.classList.toggle('is-off', !ui.labels);
        var groupsEl = $('#biGroups');
        if (groupsEl && parsed.kind === 'boris-project') {
            var g = ethogramPayload(fileEntries, groupsInUse());
            groupsEl.textContent = (g.groups.length ? 'BORIS exclusions become ' + plural(g.groups.length, 'exclusive group') + ': ' +
                g.groups.map(function (x) { return x.name + ' (' + x.members.join(', ') + ')'; }).join('; ') + '. ' : '') +
                (g.unkept.length ? 'Not kept (a group excludes all its behaviors from each other): ' + g.unkept.join(', ') + '.' : '');
        }
        var go = $('#biGo');
        var p = ui.labels ? plan() : null;
        if (p) {
            var o = obsNow();
            var fpsEl = $('#biFps');
            if (fpsEl) {
                fpsEl.textContent = timed
                    ? 'Seconds become frames at ' + (info.fps > 0 ? +info.fps.toFixed(3) : '?') + ' fps' +
                      (o.fps > 0 && info.fps > 0 && Math.abs(o.fps - info.fps) / info.fps > 0.005 ? ' (the file says ' + +o.fps.toFixed(3) + ' fps)' : '') +
                      (o.offset ? '; its time offset of ' + o.offset + ' s is taken off.' : '.')
                    : 'Frames are taken as they are.';
            }
            var extra = extraEntries();
            var newRow = $('#biNewRow');
            if (newRow) {
                newRow.hidden = !extra.length;
                $('#biNewText').textContent = 'Add ' + plural(extra.length, 'behavior') + ' the ethogram does not have: ' +
                    extra.map(function (e) { return e.name + (e.kind === 'point' ? ' (point)' : '') + (e.mutual ? ' (↔)' : e.social ? ' (→)' : ''); }).join(', ');
            }
            var s = p.skipped, parts = [];
            if (s.unpaired) parts.push(plural(s.unpaired, 'unpaired event'));
            if (s.subject) parts.push(plural(s.subject, 'bout') + ' of skipped subjects');
            if (s.target) parts.push(plural(s.target, 'social bout') + ' with no target animal');
            if (s.behavior) parts.push(plural(s.behavior, 'bout') + ' of behaviors not added');
            if (s.range) parts.push(plural(s.range, 'bout') + ' outside the video');
            $('#biSummary').textContent = (p.bouts.length ? plural(p.bouts.length, 'bout') + ' will be labelled.' : 'Nothing to label.') +
                (parts.length ? ' Skipped: ' + parts.join(', ') + '.' : '');
        }
        var addsSomething = (parsed.kind === 'ethogram' && jsonNew > 0) || (parsed.kind === 'boris-project' && fileEntries.length > 0);
        go.disabled = !((p && p.bouts.length) || (ui.addBehaviors && addsSomething));
    }

    function apply() {
        var before = behaviors().length;
        var p = ui.labels ? plan() : null;            // planned BEFORE the new behaviors exist, as previewed
        if (parsed.kind === 'ethogram') {
            if (ui.addBehaviors) ingestEthogram(parsed.ethogramPayload);
        } else {
            var entries = addingEntries();
            if (entries.length) ingestEthogram(ethogramPayload(entries, groupsInUse()).payload);
        }
        var added = behaviors().length - before;
        if (added) ethogramChanged();
        var res = { kept: 0, dropped: 0 };
        if (p && p.bouts.length) {
            res = labelBouts(session, p.bouts.map(function (bt) {
                var b = behaviorByName(bt.behavior);
                return { behaviorId: b ? b.id : '', animal: bt.animal, target: bt.target, start: bt.start, end: bt.end,
                         present: bt.present, note: bt.note };
            }), { source: 'i' });
            if (res.kept) noteBehaviorFps(session, info.fps);
        }
        dlg.close();
        var said = [];
        if (res.kept) said.push(plural(res.kept, 'bout') + ' labelled');
        if (added) said.push(plural(added, 'behavior') + ' added to the ethogram');
        setStatus(said.length ? 'Imported ' + fileName + ': ' + said.join(', ') : 'Nothing imported from ' + fileName,
            said.length ? 'success' : 'warning');
    }

    modal.addEventListener('change', function (e) {
        var t = e.target;
        if (t.id === 'biObs') { ui.obs = parseInt(t.value, 10) || 0; resetMap(); renderSubjects(); }
        else if (t.dataset && t.dataset.subject != null) {
            var o = obsNow();
            if (o) ui.map[o.subjects[+t.dataset.subject]] = t.value;
        } else if (t.id === 'biNew') ui.addNew = t.checked;
        update();
    });
    $('#biCancel').addEventListener('click', dlg.close);
    $('#biGo').addEventListener('click', apply);
    renderSubjects();
    update();
}
