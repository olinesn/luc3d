/**
 * ui/model-check-panel.js — File ▸ Load Model Check… and the right panel's
 * **Model Check** tab: the animals a model misread, from the `scan.json` that
 * `scripts/sleap_explain_error.py --sweep` writes, to step through while
 * proofreading.
 *
 * Loading reads the file (`parseModelCheck`, pose/model-check.js), matches each
 * of its cameras to the project's (`matchScanCameras`: by video file name, then
 * camera name; the user picks when more than one fits) and reports how many rows
 * matched. The tab lists the misread animals, worst first, filterable by split and
 * cause. Selecting one goes to its frame on its camera (switching session and solo
 * view when needed) and marks the animal (ui/model-check-highlight.js); its
 * misread nodes are listed with what each was read on. The row's `animal` is
 * matched to an instance by `matchAnimals` (file order, checked against node
 * positions), and the entry says how, or why it could not be.
 *
 * Nothing goes into the project: the check lives in this page until it is closed
 * or the page reloads, and the filters are browser-local (`localStorage`).
 *
 * Kept a leaf like ui/id-switch-modal.js: navigation (frame, view, session) is
 * registered by ui/ui-wiring.js through `setModelCheckNavigator`.
 */

import { state } from './app-state.js';
import { setStatus } from '../import-export/save-load.js';
import { groupDisplayName } from '../pose/pose-data.js';
import { formatFrameNumber } from './frame-readout.js';
import { readBlobArrayBuffer } from '../loading/cold-read-guard.js';
import {
    parseModelCheck, scanCameras, matchScanCameras, scanCameraKey, rowKey, reviewEntries, rowCauses,
    matchAnimals, describeNode, formatPx, baseName, frameRegion, CAUSE_LABELS, CAUSE_TEXT, SPLITS, MATCH_STATUS_TEXT,
} from '../pose/model-check.js';
import { setModelCheckHighlight, updateModelCheckHighlight } from './model-check-highlight.js';

var FILTER_KEY = 'modelCheckFilter';
var _check = null;   // {fileName, loadedAt, rows, skipped, camMatch: Map, byFrame: Map, current, sessions}
var _nav = {};

/** ui-wiring registers `{goToFrame(frame), showView(camera), switchSession(idx) -> Promise, zoomTo(camera, x0, y0, x1, y1)}`. */
export function setModelCheckNavigator(nav) { _nav = nav || {}; }

/** @returns the loaded check (tests), or null. */
export function getModelCheck() { return _check; }

function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
        return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
}
function count(n) { return Number(n).toLocaleString('en-US'); }
function plural(n, one, many) { return count(n) + ' ' + (n === 1 ? one : (many || one + 's')); }

function readFilter() {
    try {
        var f = JSON.parse(localStorage.getItem(FILTER_KEY) || '{}');
        return { split: typeof f.split === 'string' ? f.split : 'all', cause: typeof f.cause === 'string' ? f.cause : 'all', zoom: f.zoom !== false };
    } catch (e) { return { split: 'all', cause: 'all', zoom: true }; }
}
function writeFilter(f) {
    try { localStorage.setItem(FILTER_KEY, JSON.stringify(f)); } catch (e) { /* best effort */ }
}

function projectSessions() {
    if (state.sessions && state.sessions.length) return state.sessions;
    return state.session ? [state.session] : [];
}

/** One entry per (session, camera): the names the project knows for its video. */
export function projectCameras() {
    var sessions = projectSessions(), out = [];
    sessions.forEach(function (s, si) {
        (s.cameras || []).forEach(function (c) {
            var names = new Set();
            // What a loaded .slp says each camera's video is: `fileVideos` (Load SLP), or the
            // lazy loader's per-camera video (Load Project).
            var lv = s.lazyLoader && s.lazyLoader.videos && s.lazyLoader.videos.get ? s.lazyLoader.videos.get(c.name) : null;
            [s.fileVideos && s.fileVideos.get ? s.fileVideos.get(c.name) : null, lv].forEach(function (fv) {
                if (fv) [fv.sourceFilename, fv.filename].forEach(function (n) {
                    if (typeof n === 'string' && n && n !== '.') names.add(baseName(n));
                });
            });
            (state.videoFiles || []).forEach(function (vf, i) {
                if (!vf || vf.assignedCamera !== c.name) return;
                var mine = sessions.length <= 1 || (s.videoFileIndices || []).indexOf(i) >= 0 || vf.sessionIdx === si;
                if (!mine) return;
                if (vf.file && vf.file.name) names.add(vf.file.name);
                if (vf.videoPath) names.add(baseName(vf.videoPath));
            });
            out.push({ id: si + ':' + c.name, session: s, sessionIdx: si, sessionName: s.name || ('Session ' + (si + 1)),
                camera: c.name, videos: Array.from(names) });
        });
    });
    return out;
}

function framesOf(session) {
    var n = session === state.session ? state.totalFrames : session.totalFrames;
    return n > 0 ? n : 0;
}

function nodeNamesOf(skel) {
    return ((skel && skel.nodes) || []).map(function (n) { return typeof n === 'string' ? n : (n && n.name); });
}

/**
 * A camera-frame's hand-labelled instances, in file order as far as the project
 * knows it: the resident ones (sorted by their store row when they all have one —
 * a lazy project), else the lazy store's rows, which ARE file order.
 * @returns {Array<{inst: ?Instance, points: Object}>}
 */
export function userInstancesAt(session, cam, frame) {
    var names = nodeNamesOf(session.skeleton);
    var fg = session.frameGroups && session.frameGroups.get(frame);
    var list = [];
    if (fg) {
        var seen = new Set();
        var add = function (i) { if (i && i.type === 'user' && !seen.has(i)) { seen.add(i); list.push(i); } };
        (fg.getInstances ? fg.getInstances(cam) : []).forEach(add);
        (fg.getUnlinkedInstances ? fg.getUnlinkedInstances(cam) : []).forEach(function (u) { add(u && u.instance); });
        // The file's order: a lazy project's store row, an eager load's `_fileRow` (pose-data.js
        // `tagFileRow`). Only when every instance has one — a label added since has neither.
        ['_rawInstIndex', '_fileRow'].some(function (f) {
            if (list.length < 2 || !list.every(function (i) { return Number.isInteger(i[f]); })) return false;
            list.sort(function (a, b) { return a[f] - b[f]; });
            return true;
        });
    }
    if (list.length) {
        return list.map(function (inst) {
            var pts = {};
            names.forEach(function (n, k) {
                if (inst.hasPoint(k)) pts[n] = [inst.getX(k), inst.getY(k)];
            });
            return { inst: inst, points: pts };
        });
    }
    var loader = session.lazyLoader;
    if (!loader || typeof loader.getFrameSync !== 'function') return [];
    var fm = null;
    try { fm = loader.getFrameSync(frame); } catch (e) { fm = null; }
    var rows = fm && fm.get ? fm.get(cam) || [] : [];
    var storeNames = loader.skeleton ? nodeNamesOf(loader.skeleton) : names;
    return rows.filter(function (d) { return d && d.type === 'user'; }).map(function (d) {
        var pts = {};
        (d.points || []).forEach(function (p, k) { if (p && storeNames[k] != null) pts[storeNames[k]] = p; });
        return { inst: null, points: pts };
    });
}

/** The display name of an instance in the project: its identity, else its track. */
function instanceName(session, inst, cam, frame) {
    if (!inst) return null;
    var groups = (session.instanceGroups && session.instanceGroups.get(frame)) || [];
    for (var i = 0; i < groups.length; i++) {
        if (groups[i].getInstance && groups[i].getInstance(cam) === inst) {
            var n = groupDisplayName(session, groups[i], frame);
            return n === 'group' ? null : n;
        }
    }
    if (session.getIdentityIdForUnlinkedInstance) {
        var id = session.getIdentityIdForUnlinkedInstance(cam, inst, frame);
        var ident = id != null && id >= 0 && session.getIdentity ? session.getIdentity(id) : null;
        if (ident && ident.name) return ident.name;
    }
    if (inst.trackIdx != null && inst.trackIdx >= 0) return (session.tracks && session.tracks[inst.trackIdx]) || ('Track ' + inst.trackIdx);
    return null;
}

/** Everything known about one row now: its camera match and its instance match. */
function resolveRow(row) {
    var cm = _check.camMatch.get(scanCameraKey(row));
    var out = { row: row, cam: cm && cm.status === 'matched' ? cm.cam : null, camReason: cm ? cm.reason : '', status: 'unmatched', reason: '',
        checks: 0, index: -1, inst: null, name: null, other: function () { return null; } };
    if (!out.cam) { out.reason = 'Camera not matched: ' + (out.camReason || 'skipped'); return out; }
    if (projectSessions().indexOf(out.cam.session) < 0) { out.cam = null; out.reason = 'The project changed since this check was loaded: load it again'; return out; }
    var frames = framesOf(out.cam.session);
    if (frames && row.frame >= frames) {
        out.reason = 'Frame ' + formatFrameNumber(row.frame + 1) + ' is past the end of ' + out.cam.camera + ' (' + plural(frames, 'frame') + ')';
        return out;
    }
    var insts = userInstancesAt(out.cam.session, out.cam.camera, row.frame);
    var frameRows = _check.byFrame.get(scanCameraKey(row) + '|' + row.frame) || [row];
    var m = matchAnimals(frameRows, insts).get(row.animal);
    out.insts = insts;
    out.status = m ? m.status : 'unmatched';
    out.reason = m ? m.reason : '';
    out.checks = m ? m.checks : 0;
    out.index = m ? m.index : -1;
    out.inst = out.index >= 0 && insts[out.index] ? insts[out.index].inst : null;
    out.name = out.inst ? instanceName(out.cam.session, out.inst, out.cam.camera, row.frame) : null;
    // A verified row's distances were checked against the other animals it names, in file order too.
    out.other = function (b) { return out.status === 'verified' && insts[b] ? insts[b] : null; };
    return out;
}

// ---- loading ------------------------------------------------------------------

/** File ▸ Load Model Check…: pick a scan.json and load it. */
export async function openModelCheckFile(pick) {
    if (!projectSessions().length) { setStatus('Open the project the model check was run on first', 'warning'); return; }
    var files = pick ? await pick() : [];
    if (!files || !files.length) { setStatus('No file selected', 'warning'); return; }
    var file = files[0];
    var text;
    try {
        text = new TextDecoder().decode(await readBlobArrayBuffer(file));
    } catch (e) {
        setStatus('Could not read ' + file.name + ': ' + (e && e.message || e), 'error');
        return;
    }
    return loadModelCheckText(text, file.name);
}

/**
 * Load a check from its text (also the e2e entry point). Asks the user about
 * cameras that more than one project camera fits.
 * @returns {Promise<?{matched, total, unmatched}>}
 */
export async function loadModelCheckText(text, fileName) {
    var parsed;
    try { parsed = parseModelCheck(text); }
    catch (e) { setStatus((fileName || 'Model check') + ': ' + e.message, 'error'); return null; }
    var scams = scanCameras(parsed.rows);
    var camMatch = matchScanCameras(scams, projectCameras());
    var ambiguous = scams.filter(function (s) { return camMatch.get(s.key).status === 'ambiguous'; });
    if (ambiguous.length) {
        var picks = await askCameras(ambiguous, camMatch);
        if (!picks) { setStatus('Model check not loaded', 'warning'); return null; }
        ambiguous.forEach(function (s) {
            var p = picks.get(s.key), m = camMatch.get(s.key);
            if (p) camMatch.set(s.key, { status: 'matched', cam: p, candidates: m.candidates, how: 'chosen', reason: '' });
            else camMatch.set(s.key, { status: 'unmatched', cam: null, candidates: m.candidates, how: '', reason: 'skipped when asked (' + m.reason + ')' });
        });
    }
    var byFrame = new Map();
    parsed.rows.forEach(function (r) {
        var k = scanCameraKey(r) + '|' + r.frame;
        if (!byFrame.has(k)) byFrame.set(k, []);
        byFrame.get(k).push(r);
    });
    _check = { fileName: fileName || 'scan.json', loadedAt: new Date(), rows: parsed.rows, skipped: parsed.skipped,
        camMatch: camMatch, byFrame: byFrame, current: null };
    var rep = matchReport();
    openModelCheckPanel();
    refreshModelCheckPanel();
    var misread = parsed.rows.filter(function (r) { return r.misread; }).length;
    setStatus('Model check: ' + count(rep.matched) + ' of ' + plural(rep.total, 'row') + ' matched to the project' +
        (rep.total - rep.matched ? ', ' + count(rep.total - rep.matched) + ' not' : '') + '; ' +
        plural(misread, 'misread animal') + ' to review', rep.matched ? 'success' : 'warning');
    return { matched: rep.matched, total: rep.total, unmatched: rep.total - rep.matched };
}

/** Rows whose camera (and frame) matched, and why the others did not, per scan camera. */
function matchReport() {
    var total = 0, matched = 0, why = [], how = {};
    scanCameras(_check.rows).forEach(function (s) {
        var m = _check.camMatch.get(s.key);
        var frames = m.cam ? framesOf(m.cam.session) : 0;
        var past = m.cam && frames ? s.rows.filter(function (r) { return r.frame >= frames; }).length : 0;
        total += s.rows.length;
        if (m.status === 'matched') {
            matched += s.rows.length - past;
            how[m.how] = (how[m.how] || 0) + 1;
            if (past) why.push({ cam: s, n: past, reason: 'frames past the end of ' + m.cam.camera + ' (' + plural(frames, 'frame') + ')' });
        } else {
            why.push({ cam: s, n: s.rows.length, reason: m.reason });
        }
    });
    return { total: total, matched: matched, why: why, how: how };
}

/** The question for cameras more than one project camera fits. Resolves to Map(key -> cam | null), or null on Cancel. */
function askCameras(ambiguous, camMatch) {
    return new Promise(function (resolve) {
        var overlay = document.createElement('div');
        overlay.className = 'multi-frame-modal-overlay';
        overlay.innerHTML = '<div class="multi-frame-modal model-check-ask" role="dialog" aria-labelledby="modelCheckAskTitle">' +
            '<h3 id="modelCheckAskTitle">Which camera did the check read?</h3>' +
            '<p>More than one camera in the project fits ' + (ambiguous.length === 1 ? 'this camera' : 'these cameras') + ' of the model check.</p>' +
            ambiguous.map(function (s, i) {
                var m = camMatch.get(s.key);
                return '<label class="model-check-ask-row"><span><b>' + esc(s.camera || '(no folder)') + '</b> · ' + esc(s.video) +
                    ' · ' + plural(s.rows.length, 'animal') + '<br><small>' + esc(m.reason) + '</small></span>' +
                    '<select data-i="' + i + '">' + m.candidates.map(function (c, j) {
                        return '<option value="' + j + '">' + esc(c.sessionName + ' · ' + c.camera) + '</option>';
                    }).join('') + '<option value="skip">Skip this camera</option></select></label>';
            }).join('') +
            '<div class="modal-actions"><button id="modelCheckAskCancel">Cancel</button>' +
            '<button id="modelCheckAskOk" class="primary">Use these</button></div></div>';
        document.body.appendChild(overlay);
        var done = function (v) { document.removeEventListener('keydown', onKey, true); overlay.remove(); resolve(v); };
        // Every key stops at the dialog (the app's shortcuts must not act under it); Esc cancels.
        var onKey = function (e) { e.stopPropagation(); if (e.key === 'Escape') { e.preventDefault(); done(null); } };
        document.addEventListener('keydown', onKey, true);
        overlay.querySelector('#modelCheckAskCancel').addEventListener('click', function () { done(null); });
        overlay.addEventListener('mousedown', function (e) { if (e.target === overlay) done(null); });
        overlay.querySelector('#modelCheckAskOk').addEventListener('click', function () {
            var picks = new Map();
            overlay.querySelectorAll('select[data-i]').forEach(function (sel) {
                var s = ambiguous[+sel.dataset.i], m = camMatch.get(s.key);
                picks.set(s.key, sel.value === 'skip' ? null : m.candidates[+sel.value]);
            });
            done(picks);
        });
        overlay.querySelector('#modelCheckAskOk').focus();
    });
}

/** Forget the loaded check. */
export function closeModelCheck() {
    _check = null;
    setModelCheckHighlight(null);
    refreshModelCheckPanel();
}

// ---- the tab ----------------------------------------------------------------------

/** Show the right panel (if hidden) on the Model Check tab. */
export function openModelCheckPanel() {
    var wrap = document.getElementById('infoPanelWrapper');
    if (wrap && wrap.classList.contains('collapsed')) { var t = document.getElementById('infoPanelToggleBtn'); if (t) t.click(); }
    var tab = document.querySelector('.panel-tab[data-tab="tabModelCheck"]');
    if (tab) tab.click();
}

function time12(d) {
    return d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true });
}

function entries() { return _check ? reviewEntries(_check.rows, readFilter()) : []; }

var STATUS_ICON = { verified: '✓', order: '·', positions: '↔', unmatched: '⚠' };

function entryHtml(row, cur) {
    var r = resolveRow(row);
    var who = 'animal ' + row.animal + (r.name ? ' · ' + esc(r.name) : '');
    var where = r.cam ? esc(r.cam.camera) + (projectSessions().length > 1 ? ' <small>(' + esc(r.cam.sessionName) + ')</small>' : '') : esc(row.camera || row.video);
    var nodes = row.nodes.map(function (n) { return n.node; });
    var brief = !isFinite(row.maxPx) ? 'not detected at all'
        : nodes.length ? plural(nodes.length, 'body part') + ': ' + esc(nodes.slice(0, 3).join(', ')) + (nodes.length > 3 ? ' and ' + (nodes.length - 3) + ' more' : '')
        : 'median ' + formatPx(row.medianPx);
    var statusText = MATCH_STATUS_TEXT[r.status] + (r.reason ? ': ' + r.reason : '');
    var html = '<div class="model-check-row status-' + r.status + (cur ? ' is-current' : '') + '" data-key="' + esc(rowKey(row)) + '">' +
        '<div class="model-check-line1"><span class="model-check-where">' + where + ' · frame ' + formatFrameNumber(row.frame + 1) + '</span>' +
        '<span class="model-check-who">' + who + '</span>' +
        '<span class="model-check-px" title="Worst misread node">' + (isFinite(row.maxPx) ? formatPx(row.maxPx) : 'missed') + '</span></div>' +
        '<div class="model-check-line2"><span class="model-check-split">' + esc(row.split) + '</span> · ' + brief +
        ' <span class="model-check-status" title="' + esc(statusText) + '">' + STATUS_ICON[r.status] + '</span></div>';
    if (cur) {
        var nameOf = function (b) {
            var o = r.other(b);
            return o && o.inst ? instanceName(r.cam.session, o.inst, r.cam.camera, row.frame) : null;
        };
        html += '<ul class="model-check-nodes">' + row.nodes.map(function (n) {
            return '<li><div class="model-check-node">' + esc(describeNode(n, nameOf)) + '</div>' +
                '<div class="model-check-nodemeta"><span class="model-check-cause cause-' + esc(n.cause) + '" title="Read on ' + esc(CAUSE_TEXT[n.cause] || n.cause) + '">' +
                esc(CAUSE_LABELS[n.cause] || n.cause) + '</span>' +
                '<span class="model-check-npx">' + formatPx(n.px) + ' from its label' +
                (n.ref ? ', ' + formatPx(n.ref.px) + ' from ' + (n.ref.own ? 'its own ' : "animal " + n.ref.animal + "'s ") + esc(n.ref.node) : '') + '</span></div></li>';
        }).join('') + '</ul>' +
            (!isFinite(row.maxPx) ? '<p class="model-check-note">The model found no animal here: there is no reading to explain.</p>' : '') +
            '<p class="model-check-note model-check-match">' + esc(statusText) +
            (r.checks && (r.status === 'verified' || r.status === 'positions') ? ' (' + plural(r.checks, 'distance') + ' checked)' : '') + '.' +
            (r.status === 'order' ? ' Nothing in the check gives a position to confirm it is the right animal.' : '') + '</p>';
    }
    return html + '</div>';
}

function optionsHtml(values, cur, labelOf, countOf) {
    return values.map(function (v) {
        return '<option value="' + esc(v) + '"' + (v === cur ? ' selected' : '') + '>' + esc(labelOf(v)) + (countOf ? ' (' + countOf(v) + ')' : '') + '</option>';
    }).join('');
}

/** (Re)render the tab. Cheap; called after a load, a filter change and a selection. */
export function refreshModelCheckPanel() {
    var host = typeof document !== 'undefined' && document.getElementById('modelCheckPanel');
    if (!host) return;
    if (!_check) {
        host.innerHTML = '<div class="info-section"><h3>Model check</h3><p class="table-empty">No model check loaded.</p>' +
            '<p class="model-check-intro">Load the <code>scan.json</code> that <code>scripts/sleap_explain_error.py --sweep</code> writes ' +
            'to step through the hand-labelled animals the model misread, with what each body part was read on.</p>' +
            '<div class="id-switch-run"><button class="panel-btn" id="modelCheckLoadBtn">Load Model Check…</button></div></div>';
        host.querySelector('#modelCheckLoadBtn').addEventListener('click', function () {
            var m = document.getElementById('menuLoadModelCheck'); if (m) m.click();
        });
        return;
    }
    var f = readFilter(), rows = _check.rows, misread = rows.filter(function (r) { return r.misread; });
    var rep = matchReport();
    var list = entries();
    var splits = SPLITS.filter(function (s) { return rows.some(function (r) { return r.split === s; }); });
    var causes = Object.keys(CAUSE_LABELS).filter(function (c) { return misread.some(function (r) { return rowCauses(r).has(c); }); });
    var howText = Object.keys(rep.how).map(function (h) {
        return plural(rep.how[h], 'camera') + ' ' + ({ video: 'by video file name', camera: 'by camera name', chosen: 'chosen by you' }[h] || h);
    }).join(', ');
    var at = list.findIndex(function (r) { return rowKey(r) === _check.current; });
    host.innerHTML = '<div class="info-section model-check-head"><h3>Model check</h3>' +
        '<div class="model-check-file">' + esc(_check.fileName) + ' · loaded ' + time12(_check.loadedAt) + '</div>' +
        splits.map(function (s) {
            var all = rows.filter(function (r) { return r.split === s; }), bad = all.filter(function (r) { return r.misread; });
            return '<div class="model-check-summary"><b>' + esc(s) + ':</b> ' + count(bad.length) + ' of ' + plural(all.length, 'animal') + ' misread</div>';
        }).join('') +
        '<div class="model-check-summary">Matched ' + count(rep.matched) + ' of ' + plural(rep.total, 'row') + ' to the project' +
        (howText ? ' (' + howText + ')' : '') + '.</div>' +
        (rep.why.length ? '<details class="id-switch-about model-check-unmatched"><summary>' + plural(rep.total - rep.matched, 'row') + ' not matched</summary><ul>' +
            rep.why.map(function (w) {
                return '<li><b>' + esc(w.cam.camera || '(no folder)') + '</b> (' + esc(w.cam.video) + '), ' + plural(w.n, 'row') + ': ' + esc(w.reason) + '</li>';
            }).join('') + '</ul></details>' : '') +
        (_check.skipped ? '<div class="model-check-summary">' + plural(_check.skipped, 'row') + ' in the file could not be read and were skipped.</div>' : '') +
        '</div>' +
        '<div class="id-switch-toolbar model-check-toolbar">' +
        '<select id="modelCheckSplit" title="Show the animals of one split">' +
        optionsHtml(['all'].concat(splits), f.split, function (v) { return v === 'all' ? 'All splits' : v; },
            function (v) { return reviewEntries(rows, { split: v, cause: 'all' }).length; }) + '</select>' +
        '<select id="modelCheckCause" title="Show the animals with a node misread this way">' +
        optionsHtml(['all'].concat(causes), f.cause, function (v) { return v === 'all' ? 'All causes' : CAUSE_LABELS[v]; },
            function (v) { return reviewEntries(rows, { split: f.split, cause: v }).length; }) + '</select>' +
        '<span class="model-check-pos">' + (at >= 0 ? count(at + 1) + ' of ' : '') + plural(list.length, 'animal') + '</span>' +
        '<button class="panel-btn" id="modelCheckPrev" title="Previous entry (Shift+M)"' + (list.length ? '' : ' disabled') + '>◂ Prev</button>' +
        '<button class="panel-btn id-switch-next" id="modelCheckNext" title="Next entry (M)"' + (list.length ? '' : ' disabled') + '>Next ▸</button>' +
        '<label class="model-check-zoom" title="Zoom the view in on each animal you select (the animal and the ones its body parts were read on)">' +
        '<input type="checkbox" id="modelCheckZoom"' + (f.zoom ? ' checked' : '') + '> Zoom in</label>' +
        '<button class="panel-btn" id="modelCheckClose" title="Forget this model check">Close</button>' +
        '</div>' +
        '<div class="model-check-list" id="modelCheckList">' +
        (list.length ? list.map(function (r) { return entryHtml(r, rowKey(r) === _check.current); }).join('')
            : '<p class="table-empty">' + (misread.length ? 'No misread animal matches these filters.' : 'The model read every animal correctly.') + '</p>') +
        '</div>';
    host.querySelector('#modelCheckSplit').addEventListener('change', function (e) {
        var nf = readFilter(); nf.split = e.target.value; writeFilter(nf); refreshModelCheckPanel();
    });
    host.querySelector('#modelCheckCause').addEventListener('change', function (e) {
        var nf = readFilter(); nf.cause = e.target.value; writeFilter(nf); refreshModelCheckPanel();
    });
    host.querySelector('#modelCheckZoom').addEventListener('change', function (e) {
        var nf = readFilter(); nf.zoom = !!e.target.checked; writeFilter(nf);
    });
    host.querySelector('#modelCheckPrev').addEventListener('click', function () { stepModelCheck(-1); });
    host.querySelector('#modelCheckNext').addEventListener('click', function () { stepModelCheck(1); });
    host.querySelector('#modelCheckClose').addEventListener('click', function () {
        closeModelCheck();
        setStatus('Closed the model check', 'success');
    });
    host.querySelector('#modelCheckList').addEventListener('click', function (e) {
        var el = e.target.closest('.model-check-row');
        if (!el) return;
        var row = list.find(function (r) { return rowKey(r) === el.dataset.key; });
        if (row) goToEntry(row);
    });
    var curEl = host.querySelector('.model-check-row.is-current');
    if (curEl && curEl.scrollIntoView) curEl.scrollIntoView({ block: 'nearest' });
}

/** Next (+1) / previous (-1) entry of the filtered list, wrapping. The M / Shift+M shortcuts. */
export function stepModelCheck(dir) {
    if (!_check) { setStatus('No model check loaded: File ▸ Load Model Check…', 'warning'); return; }
    var list = entries();
    if (!list.length) { setStatus('No misread animal matches the Model Check filters', 'warning'); return; }
    var at = list.findIndex(function (r) { return rowKey(r) === _check.current; });
    var next = at < 0 ? (dir > 0 ? 0 : list.length - 1) : (at + dir + list.length) % list.length;
    return goToEntry(list[next]);
}

/** Select an entry: go to its session, camera and frame, and mark the animal. */
export async function goToEntry(row) {
    _check.current = rowKey(row);
    var r = resolveRow(row);
    if (!r.cam) {
        setModelCheckHighlight(null);
        refreshModelCheckPanel();
        setStatus('Model check: ' + r.reason, 'warning');
        return;
    }
    if (r.cam.session !== state.session && _nav.switchSession) {
        await _nav.switchSession(r.cam.sessionIdx);
        r = resolveRow(row);
    }
    if (_nav.showView) _nav.showView(r.cam.camera);
    if (_nav.goToFrame) _nav.goToFrame(row.frame);
    setModelCheckHighlight({
        session: r.cam.session, camera: r.cam.camera, frame: row.frame,
        label: 'animal ' + row.animal + (r.name ? ' · ' + r.name : '') + ' · ' + (isFinite(row.maxPx) ? formatPx(row.maxPx) : 'missed'),
        locate: function () { return locate(row); },
        onLocated: function (loc) { if (readFilter().zoom) frameAnimal(r.cam.camera, loc); },
    });
    refreshModelCheckPanel();
    var list = entries(), at = list.indexOf(row);
    var where = 'Model check' + (at >= 0 ? ' ' + count(at + 1) + ' of ' + count(list.length) : '') + ': ' + r.cam.camera +
        ', frame ' + formatFrameNumber(row.frame + 1) + ', animal ' + row.animal + (r.name ? ' (' + r.name + ')' : '');
    var hidden = r.cam.session._hiddenCameras;
    if (hidden && (hidden.has ? hidden.has(r.cam.camera) : hidden.indexOf && hidden.indexOf(r.cam.camera) >= 0)) {
        setStatus(where + ': ' + r.cam.camera + ' is hidden (Visibility tab), show it to see this animal', 'warning');
    } else if (r.status === 'unmatched') {
        setStatus(where + ': ' + r.reason, 'warning');
    } else {
        setStatus(where, 'success');
    }
}

/** The video-pixel box of an instance's placed nodes, or null. */
function instanceBox(inst) {
    var b = null;
    for (var k = 0; k < (inst.numNodes || 0); k++) {
        if (inst.hasPoint && !inst.hasPoint(k)) continue;
        var x = inst.getX(k), y = inst.getY(k);
        if (!isFinite(x) || !isFinite(y)) continue;
        if (!b) b = { x0: x, y0: y, x1: x, y1: y };
        else { b.x0 = Math.min(b.x0, x); b.y0 = Math.min(b.y0, y); b.x1 = Math.max(b.x1, x); b.y1 = Math.max(b.y1, y); }
    }
    return b;
}

function frameAnimal(camera, loc) {
    if (!_nav.zoomTo) return;
    var g = frameRegion([instanceBox(loc.target)].concat(loc.others.map(function (o) { return instanceBox(o.inst); })));
    if (g) _nav.zoomTo(camera, g.x0, g.y0, g.x1, g.y1);
}

/** What the highlight draws, from the instances as they are now (null while the frame is not loaded). */
function locate(row) {
    var r = resolveRow(row);
    if (!r.cam || !r.inst) return null;
    var names = nodeNamesOf(r.cam.session.skeleton);
    var others = [], seen = new Set();
    var nodes = row.nodes.map(function (n) {
        var k = names.indexOf(n.node), out = { k: k, name: n.node };
        var ref = n.ref;
        if (ref) {
            var b = ref.own ? row.animal : ref.animal;
            var o = b === row.animal ? { inst: r.inst } : r.other(b);
            var j = names.indexOf(ref.node);
            if (o && o.inst && j >= 0) {
                out.ref = { inst: o.inst, j: j };
                if (b !== row.animal && !seen.has(b)) {
                    seen.add(b);
                    var nm = instanceName(r.cam.session, o.inst, r.cam.camera, row.frame);
                    others.push({ inst: o.inst, label: 'animal ' + b + (nm ? ' · ' + nm : '') });
                }
            }
        }
        return out;
    }).filter(function (n) { return n.k >= 0; });
    return { target: r.inst, nodes: nodes, others: others };
}

/** Every frame change (ui-wiring `updateSeekbarVisual`). */
export function updateModelCheckFrame(frame) {
    updateModelCheckHighlight(frame);
}
