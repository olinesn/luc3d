// ui/behavior-kpms-modal.js — Behavior ▸ Export for keypoint-MoSeq… and
// Behavior ▸ Import keypoint-MoSeq Results…: the round trip to keypoint-MoSeq,
// which runs in the user's own Python (import-export/kpms-io.js has the
// formats).
//
// Export writes one pose table per animal of this session (or every session
// whose poses are loaded) and luc3d-kpms.yml — keypoint-MoSeq's settings and
// the calls to run — into a folder picked once (`showDirectoryPicker`,
// requested inside the click), else as separate downloads. The poses are the
// behavior features' (ui/behavior-detect.js `buildBehaviorContext`): 3D in
// the floor frame, the body's used nodes as `use_bodyparts`, its head and tail
// base as anterior and posterior, broken tracking scored 0; 2D from one
// camera when there is no 3D. All frames, or a stretch (the armed [ ] range by
// default); a stretch's first frame is recorded in the manifest.
//
// Import reads a results folder or files: `results.h5`, the CSV results
// folder, `luc3d-kpms.yml` (where each recording came from, and its first
// frame) and `syll_info.csv` (syllable names). It shows each recording with
// the session and animal it goes to — the manifest's, else the name's, else
// asked — then stores the syllables as a states layer in each session
// (pose/behavior-states.js), replacing a layer of the same name there.
//
// Depends on: ui/app-state.js, ui/behavior-coding.js, ui/behavior-detect.js,
//             ui/behavior-io-modal.js, ui/loading-overlay.js, pose/behavior-states.js,
//             pose/behavior-tensor.js, pose/behavior-ethogram.js, import-export/kpms-io.js,
//             import-export/save-load.js, import-export/file-io.js, loading/cold-read-guard.js

import { state } from './app-state.js';
import { sessionAnimals, armedRange, openBehaviorTab } from './behavior-coding.js';
import { buildBehaviorContext, cachedBehaviorContext } from './behavior-detect.js';
import { behaviorSessionInfo } from './behavior-io-modal.js';
import { showLoading, showLoadingProgress, hideLoading, yieldToPaint, createProgressPacer } from './loading-overlay.js';
import { setStateLayer, stateLayer, cleanLayerName, runsFromSequence } from '../pose/behavior-states.js';
import { resolveBody, hasIdentified3d, pick2dCamera } from '../pose/behavior-tensor.js';
import { featureSettings } from '../pose/behavior-ethogram.js';
import {
    KPMS_MANIFEST, kpmsBodypartNames, recordingNames, coordinateDecimals, poseCsvChunks, kpmsManifest,
    parseKpmsManifest, readResultsH5, parseResultsCsv, recordingOfCsv, parseSyllableNames, matchRecordings,
} from '../import-export/kpms-io.js';
import { setStatus } from '../import-export/save-load.js';
import { initH5wasm } from '../import-export/file-io.js';
import { readBlobArrayBuffer } from '../loading/cold-read-guard.js';

var PREFS_KEY = 'lucid.kpmsExport';
var ROWS_PER_CHUNK = 4096;
var DEFAULT_LAYER = 'keypoint-MoSeq';
var WALK_DEPTH = 4;
var WALK_MAX_ENTRIES = 20000;

function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function n(x) { return Number(x).toLocaleString('en-US'); }
function plural(k, one, many) { return n(k) + ' ' + (k === 1 ? one : (many || one + 's')); }

function openOverlay(className, html) {
    var overlay = document.createElement('div');
    overlay.className = 'multi-frame-modal-overlay behavior-modal-overlay';
    var modal = document.createElement('div');
    modal.className = 'multi-frame-modal behavior-io-modal kpms-modal ' + className;
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
    return !!document.querySelector('.kpms-modal');
}

function sessionsOf() {
    return (state.sessions && state.sessions.length) ? state.sessions : (state.session ? [state.session] : []);
}

function loadPrefs() {
    try { return JSON.parse(localStorage.getItem(PREFS_KEY) || '{}') || {}; } catch (e) { return {}; }
}

function savePrefs(p) {
    try { localStorage.setItem(PREFS_KEY, JSON.stringify(p)); } catch (e) { /* best effort */ }
}

function downloadParts(parts, filename, type) {
    var blob = new Blob(parts, { type: type + ';charset=utf-8' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
}

function nodeNamesOf(session) {
    var sk = session && session.skeleton;
    return sk && sk.nodes ? sk.nodes.map(function (nd) { return typeof nd === 'string' ? nd : nd && nd.name; }) : [];
}

function edgesOf(session) {
    var sk = session && session.skeleton;
    return sk && sk.edges ? sk.edges.map(function (e) { return Array.isArray(e) ? [e[0], e[1]] : [e.source, e.destination]; }) : [];
}

/**
 * The keypoint-MoSeq body of a skeleton: written names, used nodes, anterior
 * and posterior (the head and tail base; without them the first and last used
 * node), skeleton edges by written name.
 */
function kpmsBody(nodeNames, edges, body) {
    var kn = kpmsBodypartNames(nodeNames);
    var used = body.used.length ? body.used : nodeNames.map(function (_, i) { return i; });
    var head = body.roles.head >= 0 && used.indexOf(body.roles.head) >= 0 ? body.roles.head : used[0];
    var tail = body.roles.tailBase >= 0 && used.indexOf(body.roles.tailBase) >= 0 ? body.roles.tailBase : used[used.length - 1];
    return {
        names: kn.names, renamed: kn.renamed, used: used,
        anterior: head != null ? [kn.names[head]] : [], posterior: tail != null && tail !== head ? [kn.names[tail]] : [],
        guessedEnds: !(body.roles.head >= 0) || !(body.roles.tailBase >= 0),
        skeleton: edges.filter(function (e) { return kn.names[e[0]] && kn.names[e[1]]; }).map(function (e) { return [kn.names[e[0]], kn.names[e[1]]]; }),
    };
}

// ----------------------------------------------------------------------------
// Export
// ----------------------------------------------------------------------------

/** Behavior ▸ Export for keypoint-MoSeq…. */
export function showKpmsExportModal() {
    if (isOpen()) return;
    var session = state.session;
    if (!session) { setStatus('No session loaded', 'warning'); return; }
    if (!sessionAnimals(session).length) { setStatus('No animals to export: run Track All, or add identities', 'warning'); return; }
    var sessions = sessionsOf();
    var prefs = loadPrefs();
    var frames = behaviorSessionInfo(session).frames;
    var armed = armedRange();
    var names = nodeNamesOf(session), edges = edgesOf(session);
    var body = resolveBody(names, edges, featureSettings());
    var kb = kpmsBody(names, edges, body);
    var is3d = hasIdentified3d(session);
    var renamed = Object.keys(kb.renamed);
    var from = armed ? armed.start + 1 : 1, to = armed ? armed.end + 1 : Math.max(1, frames);
    var dlg = openOverlay('kpms-export-modal',
        '<h3>Export for keypoint-MoSeq</h3>' +
        '<p class="behavior-ed-lead">keypoint-MoSeq finds behavioral syllables in poses, in your own Python. This writes a pose ' +
        'table per animal and <b>' + KPMS_MANIFEST + '</b> — its settings and the calls to run. Then import its results.</p>' +
        '<div class="behavior-io-opts">' +
        (sessions.length > 1 ? '<label>Sessions <select id="kxScope"><option value="one">This session</option>' +
            '<option value="all"' + (prefs.scope === 'all' ? ' selected' : '') + '>All ' + n(sessions.length) + ' sessions</option></select></label>' : '') +
        '</div>' +
        '<div class="kpms-frames" role="radiogroup" aria-label="Frames">' +
        '<label><input type="radio" name="kxFrames" value="all"' + (armed ? '' : ' checked') + '> All frames</label>' +
        '<label><input type="radio" name="kxFrames" value="range"' + (armed ? ' checked' : '') + '> Frames ' +
        '<input type="number" id="kxFrom" min="1" step="1" value="' + from + '"> to ' +
        '<input type="number" id="kxTo" min="1" step="1" value="' + to + '"></label></div>' +
        '<p class="behavior-io-detail" id="kxBody">' +
        (is3d ? '3D, in the floor frame (z up), in calibration units' : '2D (no 3D): camera ' + esc(pick2dCamera(session) || '?') + ', in pixels') +
        ' · keypoint-MoSeq models ' + n(kb.used.length) + ' of ' + n(names.length) + ' nodes' +
        (kb.used.length < names.length ? ' (' + esc(body.excluded.map(function (i) { return names[i]; }).join(', ')) + ' left out, as in Detection Settings)' : '') +
        ' · heading from ' + esc(kb.anterior.join(', ') || '?') + ' to ' + esc(kb.posterior.join(', ') || '?') +
        (renamed.length ? ' · node names written with letters, digits and _ only (' + esc(renamed.map(function (r) { return r + ' → ' + kb.renamed[r]; }).join(', ')) + ')' : '') +
        '</p>' +
        '<p class="behavior-io-detail">' + (typeof window.showDirectoryPicker === 'function'
            ? 'Choose an empty folder: the calls read every .csv in it.' : 'Your browser saves each file as a download: put them in one empty folder.') + '</p>' +
        '<div class="modal-actions"><button id="kxCancel">Cancel</button><button class="primary" id="kxGo">Export…</button></div>');
    var modal = dlg.modal;
    var rangeOn = function () { var r = modal.querySelector('input[name="kxFrames"]:checked'); return r && r.value === 'range'; };
    ['kxFrom', 'kxTo'].forEach(function (id) {
        modal.querySelector('#' + id).addEventListener('focus', function () { modal.querySelector('input[name="kxFrames"][value="range"]').checked = true; });
    });
    modal.querySelector('#kxCancel').addEventListener('click', dlg.close);
    modal.querySelector('#kxGo').addEventListener('click', function () {
        var scopeSel = modal.querySelector('#kxScope');
        var choice = { scope: scopeSel ? scopeSel.value : 'one', first: 0, last: Infinity };
        if (rangeOn()) {
            var a = Math.round(Number(modal.querySelector('#kxFrom').value)), b = Math.round(Number(modal.querySelector('#kxTo').value));
            if (!(a >= 1) || !(b >= a)) { setStatus('Frames: give a first frame of 1 or more and a last frame after it', 'warning'); return; }
            choice.first = a - 1; choice.last = b - 1;
        }
        savePrefs({ scope: choice.scope });
        // The folder picker needs the click's user activation: ask before any work.
        var dirPick = typeof window.showDirectoryPicker === 'function' ? window.showDirectoryPicker({ mode: 'readwrite' }) : null;
        dlg.close();
        runKpmsExport(choice, dirPick);
    });
}

/** Any finite coordinate of animal `a` in frames first..last? */
function hasDataIn(input, a, first, last) {
    if (first === 0 && last >= input.numFrames - 1) return input.frames[a] > 0;
    var stride = input.nodeNames.length * input.dim, arr = input.data[a];
    for (var i = first * stride, end = (last + 1) * stride; i < end; i++) if (arr[i] === arr[i]) return true;
    return false;
}

/**
 * Build the export's files and write them.
 * @param {{scope: 'one'|'all', first: number, last: number}} choice frames (0-based, last inclusive or Infinity)
 * @param {Promise<FileSystemDirectoryHandle>|null} dirPick
 * @param {{write?: function(string, string[]): Promise}} [opts] write: where files go instead (tests)
 * @returns {Promise<{files: string[], leftOut: string[]}|null>}
 */
export async function runKpmsExport(choice, dirPick, opts) {
    var title = 'Export for keypoint-MoSeq';
    var pick = choice.scope === 'all' ? sessionsOf() : [state.session];
    try {
        showLoading('Reading poses…', { title: title });
        await yieldToPaint();
        var plans = [], leftOut = [], mode = null;
        for (var i = 0; i < pick.length; i++) {
            var s = pick[i], name = s.name || 'session ' + (i + 1);
            showLoading('Reading poses of ' + name + '…', { title: title, step: i + 1, steps: pick.length });
            await yieldToPaint();
            var ctx = s === state.session ? cachedBehaviorContext(s) : buildBehaviorContext(s);
            if (!ctx.input.animals.length) { leftOut.push(name + ' (poses not loaded: open it to include it)'); continue; }
            if (mode && ctx.input.mode !== mode) { leftOut.push(name + ' (' + ctx.input.mode.toUpperCase() + ', the others ' + mode.toUpperCase() + ')'); continue; }
            mode = mode || ctx.input.mode;
            plans.push({ session: s, name: name, ctx: ctx });
        }
        if (!plans.length) {
            setStatus('Nothing to export: no animal has poses' + (leftOut.length ? ' · ' + leftOut.join(', ') : ''), 'warning');
            return null;
        }
        var ctx0 = plans[0].ctx, input0 = ctx0.input;
        var kb = kpmsBody(input0.nodeNames, input0.edges, ctx0.body);
        var pairs = [], empty = [];
        plans.forEach(function (p) {
            var inp = p.ctx.input, last = Math.min(choice.last, inp.numFrames - 1), first = Math.min(choice.first, last);
            inp.animals.forEach(function (animal, a) {
                if (!(last >= first) || !hasDataIn(inp, a, first, last)) { empty.push(p.name + ' · ' + animal); return; }
                pairs.push({ plan: p, animal: animal, a: a, first: first, last: last });
            });
        });
        if (!pairs.length) { setStatus('Nothing to export: no animal has poses in those frames', 'warning'); return null; }
        var names = recordingNames(pairs.map(function (q) { return { session: q.plan.name, animal: q.animal }; }));
        var bls = [];
        plans.forEach(function (p) { (p.ctx.spec.bodyLength || []).forEach(function (v) { if (v > 0) bls.push(v); }); });
        bls.sort(function (x, y) { return x - y; });
        var decimals = coordinateDecimals(bls.length ? bls[bls.length >> 1] : NaN);
        var fpsSet = Array.from(new Set(plans.map(function (p) { return Math.round(p.ctx.fps * 1000) / 1000; })));
        var floors = plans.map(function (p) { return p.ctx.floorInfo && p.ctx.floorInfo.floor; });

        var write = opts && opts.write;
        var dir = null;
        if (!write && dirPick) {
            try { dir = await dirPick; } catch (err) {
                if (err && err.name === 'AbortError') { setStatus('Export cancelled', 'info'); return null; }
                throw err;
            }
        }
        var put = async function (fileName, chunks, onChunk) {
            var parts = [];
            var w = dir ? await (await dir.getFileHandle(fileName, { create: true })).createWritable() : null;
            try {
                for (var c of chunks) {
                    if (w) await w.write(c); else parts.push(c);
                    if (onChunk) await onChunk();
                }
            } catch (err) {
                if (w) await w.abort().catch(function () {});
                throw err;
            }
            if (w) await w.close();
            else if (write) await write(fileName, parts);
            else downloadParts(parts, fileName, fileName.endsWith('.csv') ? 'text/csv' : 'text/yaml');
        };

        var totalRows = pairs.reduce(function (t, q) { return t + q.last - q.first + 1; }, 0), doneRows = 0;
        var pacer = createProgressPacer();
        var broken = 0;
        for (var k = 0; k < pairs.length; k++) {
            var q = pairs[k], c2 = q.plan.ctx, fl = floors[plans.indexOf(q.plan)];
            var skip = c2.broken && c2.broken.masks ? c2.broken.masks[q.a] : null;
            if (skip) for (var t = q.first; t <= q.last; t++) broken += skip[t] ? 1 : 0;
            var fileName = names[k] + '.csv';
            var left = q.last - q.first + 1, first = true;
            await put(fileName, poseCsvChunks({
                dim: c2.input.dim, bodyparts: kb.names, data: c2.input.data[q.a], nodeCount: c2.input.nodeNames.length,
                first: q.first, last: q.last, floor: fl || null, skip: skip, decimals: decimals, rowsPerChunk: ROWS_PER_CHUNK,
            }), async function () {
                if (first) { first = false; return; }          // the header
                var rows = Math.min(ROWS_PER_CHUNK, left);
                left -= rows; doneRows += rows;
                if (pacer.due()) {
                    showLoadingProgress('Writing ' + fileName, doneRows, totalRows, { title: title, unit: 'rows', step: k + 1, steps: pairs.length });
                    await pacer.yield();
                }
            });
        }
        var notes = [];
        if (fpsSet.length > 1) notes.push('The sessions differ in frame rate (' + fpsSet.join(', ') + ' fps): fps is the first one\'s.');
        if (mode === '3d' && floors.some(function (f) { return !f; })) notes.push('No floor was found for some sessions: their coordinates are the world frame\'s.');
        if (kb.guessedEnds) notes.push('No head or tail base node was found: the heading runs from the first to the last modelled node.');
        var c0 = plans[0].ctx;
        var manifest = kpmsManifest({
            format: mode === '2d' ? 'deeplabcut' : 'anipose', dim: mode === '2d' ? 2 : 3,
            units: mode === '2d' ? 'px' : 'calibration units', coordinates: mode === '2d' ? 'image' : (floors.every(Boolean) ? 'floor' : 'world'),
            camera: c0.input.camera, fps: c0.fps, bodyparts: kb.names, useBodyparts: kb.used.map(function (i) { return kb.names[i]; }),
            anterior: kb.anterior, posterior: kb.posterior, skeleton: kb.skeleton, renamed: kb.renamed, notes: notes,
            recordings: pairs.map(function (q, i) {
                return { name: names[i], session: q.plan.name, animal: q.animal, firstFrame: q.first, frames: q.last - q.first + 1 };
            }),
            exportedAt: new Date().toISOString().slice(0, 10),
        });
        await put(KPMS_MANIFEST, [manifest]);
        var files = names.map(function (nm) { return nm + '.csv'; }).concat([KPMS_MANIFEST]);
        var skipped = leftOut.concat(empty.length ? [plural(empty.length, 'animal') + ' with no poses in those frames'] : []);
        setStatus('Exported ' + plural(pairs.length, 'pose table') + ' and ' + KPMS_MANIFEST + (dir ? ' to ' + dir.name : '') +
            (broken ? ' (broken tracking on ' + n(broken) + ' animal-frames scored 0)' : '') +
            ' · Next: run the calls in ' + KPMS_MANIFEST + ' in Python, then Behavior ▸ Import keypoint-MoSeq Results…' +
            (skipped.length ? ' · Left out: ' + skipped.join(', ') : ''), skipped.length || notes.length ? 'warning' : 'success');
        return { files: files, leftOut: skipped };
    } catch (err) {
        console.error('[kpms-export]', err);
        setStatus('Export for keypoint-MoSeq failed: ' + (err && err.message ? err.message : err), 'error');
        return null;
    } finally {
        hideLoading();
    }
}

// ----------------------------------------------------------------------------
// Import: finding and reading the results
// ----------------------------------------------------------------------------

function baseName(p) { return String(p).replace(/^.*\//, ''); }
function dirName(p) { var i = String(p).lastIndexOf('/'); return i < 0 ? '' : p.slice(0, i); }

/** Every file under a directory handle, as {path, getFile}, a few levels deep. */
async function walkDirectory(handle) {
    var out = [];
    var walk = async function (h, prefix, depth) {
        for await (var entry of h.values()) {
            if (out.length >= WALK_MAX_ENTRIES) return;
            var p = prefix ? prefix + '/' + entry.name : entry.name;
            if (entry.kind === 'file') {
                (function (e) { out.push({ path: p, getFile: function () { return e.getFile(); } }); })(entry);
            } else if (entry.kind === 'directory' && depth < WALK_DEPTH && entry.name.charAt(0) !== '.') {
                await walk(entry, p, depth + 1);
            }
        }
    };
    await walk(handle, '', 0);
    return out;
}

/** Files from an <input type="file"> (a folder's carry their relative path). */
function inputEntries(files) {
    return Array.from(files || []).map(function (f) {
        var p = f.webkitRelativePath || f.name;
        // A folder input's paths start with the folder's own name: drop it.
        if (f.webkitRelativePath) p = p.replace(/^[^/]+\//, '');
        return { path: p, getFile: function () { return Promise.resolve(f); } };
    });
}

/**
 * What a set of files holds: the manifest, and each place results are —
 * a results.h5, or a folder of CSV results — with the syll_info.csv beside it.
 * In a folder only `results.h5` and the CSVs of folders named `results` are
 * results (never checkpoint.h5, nor the exported pose tables); files picked
 * one by one are taken as they are.
 * @param {Array<{path: string, getFile: Function}>} entries
 * @param {{picked?: boolean}} [o] picked: chosen as files, not found in a folder
 * @returns {{manifest: Object|null, sources: Array<{kind:'h5'|'csv', label:string, entries:Object[], dir:string, info:Object|null}>}}
 */
function findSources(entries, o) {
    var picked = !!(o && o.picked);
    var manifest = null, sources = [], csvByDir = new Map(), info = new Map();
    entries.forEach(function (e) {
        var b = baseName(e.path), lb = b.toLowerCase();
        if (lb === KPMS_MANIFEST && (!manifest || e.path.split('/').length < manifest.path.split('/').length)) manifest = e;
        else if (lb === 'syll_info.csv') info.set(dirName(e.path), e);
        else if (lb === 'results.h5' || (picked && /\.(h5|hdf5)$/.test(lb))) sources.push({ kind: 'h5', label: e.path, entries: [e], dir: dirName(e.path) });
        else if (/\.csv$/.test(lb)) {
            var d = dirName(e.path);
            if (!csvByDir.has(d)) csvByDir.set(d, []);
            csvByDir.get(d).push(e);
        }
    });
    csvByDir.forEach(function (list, d) {
        if (baseName(d) === 'results' || picked) {
            sources.push({ kind: 'csv', label: (d ? d + '/ ' : '') + '(' + plural(list.length, 'CSV file') + ')', entries: list, dir: dirName(d) || d });
        }
    });
    sources.forEach(function (s) { s.info = info.get(s.dir) || info.get('') || null; });
    return { manifest: manifest, sources: sources };
}

async function readText(entry) {
    var f = await entry.getFile();
    return new TextDecoder('utf-8').decode(new Uint8Array(await readBlobArrayBuffer(f)));
}

/**
 * Read one source's syllables.
 * @returns {Promise<{recordings: Array<{name, syllables: Int32Array}>, skipped: string[]}>}
 */
async function readSource(src) {
    var title = 'Import keypoint-MoSeq results';
    if (src.kind === 'h5') {
        var f = await src.entries[0].getFile();
        showLoading('Reading ' + f.name + ' (' + n(Math.round(f.size / 1048576)) + ' MB)…', { title: title });
        await yieldToPaint();
        var bytes = new Uint8Array(await readBlobArrayBuffer(f));
        var h5 = await initH5wasm();
        var path = '/kpms-results-' + Date.now() + '.h5';
        h5.FS.writeFile(path, bytes);
        bytes = null;
        try { return readResultsH5(h5, path); } finally { try { h5.FS.unlink(path); } catch (e) { /* gone */ } }
    }
    var recordings = [], skipped = [];
    var pacer = createProgressPacer();
    for (var i = 0; i < src.entries.length; i++) {
        var e = src.entries[i];
        if (pacer.due() || i === 0) {
            showLoadingProgress('Reading CSV results', i, src.entries.length, { title: title, unit: 'files' });
            await pacer.yield();
        }
        var r = parseResultsCsv(await readText(e));
        if (r.error) skipped.push(baseName(e.path)); else recordings.push({ name: recordingOfCsv(e.path), syllables: r.syllables });
    }
    return { recordings: recordings, skipped: skipped };
}

// ----------------------------------------------------------------------------
// Import: the dialog
// ----------------------------------------------------------------------------

/** Every (session, animal) a recording can go to. */
function importCandidates() {
    var out = [];
    sessionsOf().forEach(function (s, i) {
        sessionAnimals(s).forEach(function (a) { out.push({ session: i, sessionName: s.name || 'Session ' + (i + 1), animal: a }); });
    });
    return out;
}

/** Behavior ▸ Import keypoint-MoSeq Results…. */
export function showKpmsImportModal() {
    if (isOpen()) return;
    if (!state.session) { setStatus('No session loaded', 'warning'); return; }
    if (!importCandidates().length) { setStatus('No animals to import into: run Track All, or add identities', 'warning'); return; }
    var dlg = openOverlay('kpms-import-modal',
        '<h3>Import keypoint-MoSeq results</h3>' +
        '<p class="behavior-ed-lead">Choose the folder you exported to (it holds ' + KPMS_MANIFEST + ' and keypoint-MoSeq\'s ' +
        'kpms_project), or the results themselves: <b>results.h5</b>, or the CSV files of its results folder — with ' +
        KPMS_MANIFEST + ' if you can, which says where each recording starts.</p>' +
        '<div class="behavior-tools"><button class="panel-btn" id="kiFolder">Choose folder…</button>' +
        '<button class="panel-btn" id="kiFiles">Choose files…</button></div>' +
        '<div id="kiBody"></div>' +
        '<div class="modal-actions"><button id="kiCancel">Cancel</button><button class="primary" id="kiGo" disabled>Import</button></div>');
    var modal = dlg.modal;
    var view = { found: null, source: 0, manifest: null, manifestError: '', results: null, names: null, rows: [], layer: DEFAULT_LAYER };
    modal.querySelector('#kiCancel').addEventListener('click', dlg.close);

    var load = async function (entries, picked) {
        if (!entries.length) return;
        view.found = findSources(entries, { picked: picked });
        view.manifest = null; view.manifestError = '';
        if (view.found.manifest) {
            try {
                var p = parseKpmsManifest(await readText(view.found.manifest));
                if (p.error) view.manifestError = p.error; else view.manifest = p.value;
            } catch (err) { view.manifestError = err && err.message ? err.message : String(err); }
        }
        if (!view.found.sources.length) {
            view.results = null;
            render('No keypoint-MoSeq results there: look for results.h5, or the CSV files in a folder named results.');
            return;
        }
        // The newest results.h5 first (a project can hold several models).
        var stamps = await Promise.all(view.found.sources.map(async function (s) {
            try { return (await s.entries[0].getFile()).lastModified || 0; } catch (e) { return 0; }
        }));
        view.found.sources.forEach(function (s, i) { s.stamp = stamps[i] + (s.kind === 'h5' ? 1 : 0); });
        view.found.sources.sort(function (a, b) { return b.stamp - a.stamp; });
        view.source = 0;
        await readChosen();
    };

    // Only the latest read applies (the source can be switched while one runs).
    var readGen = 0;
    var readChosen = async function () {
        var gen = ++readGen;
        var src = view.found.sources[view.source];
        var results, names = null;
        try {
            results = await readSource(src);
            if (src.info) { try { names = parseSyllableNames(await readText(src.info)); } catch (e) { names = null; } }
        } catch (err) {
            if (gen !== readGen) return;
            console.error('[kpms-import]', err);
            view.results = null;
            render('Could not read ' + src.label + ': ' + (err && err.message ? err.message : err));
            return;
        } finally {
            if (gen === readGen) hideLoading();
        }
        if (gen !== readGen) return;
        view.results = results;
        view.names = names;
        var cands = importCandidates();
        view.rows = matchRecordings(view.results.recordings.map(function (r) { return r.name; }), cands, view.manifest).map(function (m, i) {
            return Object.assign(m, { syllables: view.results.recordings[i].syllables });
        });
        render('');
    };

    var candidates = importCandidates();
    var render = function (message) {
        var body = modal.querySelector('#kiBody');
        var go = modal.querySelector('#kiGo');
        if (message || !view.results) {
            body.innerHTML = message ? '<p class="behavior-io-warn">' + esc(message) + '</p>' : '';
            go.disabled = true;
            return;
        }
        var srcs = view.found.sources;
        var html = srcs.length > 1
            ? '<label class="behavior-io-obs">Results <select id="kiSource">' + srcs.map(function (s, i) {
                return '<option value="' + i + '"' + (i === view.source ? ' selected' : '') + '>' + esc(s.label) + '</option>';
            }).join('') + '</select></label>'
            : '<p class="behavior-io-file"><b>' + esc(srcs[0].label) + '</b></p>';
        html += '<p class="behavior-io-detail">' + (view.manifest ? KPMS_MANIFEST + ': where each recording came from, and its first frame.'
            : view.manifestError ? KPMS_MANIFEST + ' could not be read (' + esc(view.manifestError) + '): recordings are matched by name.'
            : 'No ' + KPMS_MANIFEST + ': recordings are matched by name, from frame 1 unless you say otherwise.') +
            (view.names && view.names.size ? ' syll_info.csv names ' + plural(view.names.size, 'syllable') + '.' : '') + '</p>';
        html += '<table class="data-table kpms-recordings"><thead><tr><th>Recording</th><th class="num">Frames</th><th>Session · animal</th>' +
            '<th title="The session frame its first row is (frames count from 1, as on the timeline)">From frame</th></tr></thead><tbody>';
        view.rows.forEach(function (r, i) {
            html += '<tr class="' + (r.candidate < 0 ? 'kpms-unmatched' : '') + '"><td title="' + esc(r.name) + '">' + esc(r.name) + '</td>' +
                '<td class="num">' + n(r.syllables.length) + '</td>' +
                '<td><select data-row="' + i + '" data-act="to"><option value="-1">Skip</option>' + candidates.map(function (c, j) {
                    return '<option value="' + j + '"' + (j === r.candidate ? ' selected' : '') + '>' + esc(c.sessionName + ' · ' + c.animal) + '</option>';
                }).join('') + '</select></td>' +
                '<td><input type="number" min="1" step="1" data-row="' + i + '" data-act="from" value="' + (r.firstFrame + 1) + '"></td></tr>';
        });
        html += '</tbody></table>';
        html += '<label class="kpms-layer">Name these states <input type="text" id="kiLayer" maxlength="60" value="' + esc(view.layer) + '"></label>';
        html += '<p class="behavior-io-detail" id="kiSummary"></p>';
        body.innerHTML = html;
        summarize();
    };

    var summarize = function () {
        var el = modal.querySelector('#kiSummary');
        if (!el || !view.results) return;
        var go = modal.querySelector('#kiGo');
        var used = view.rows.filter(function (r) { return r.candidate >= 0; });
        var sessions = new Set(used.map(function (r) { return candidates[r.candidate].session; }));
        var states = new Set();
        used.forEach(function (r) { for (var i = 0; i < r.syllables.length; i++) if (r.syllables[i] >= 0) states.add(r.syllables[i]); });
        var name = cleanLayerName(view.layer);
        var replaces = Array.from(sessions).filter(function (i) { return stateLayer(sessionsOf()[i], name); }).length;
        var clipped = used.filter(function (r) {
            var s = sessionsOf()[candidates[r.candidate].session];
            return r.firstFrame + r.syllables.length > behaviorSessionInfo(s).frames;
        }).length;
        var mismatch = used.filter(function (r) { return r.frames > 0 && r.frames !== r.syllables.length; }).length;
        var pairs = new Set(used.map(function (r) { return r.candidate; }));
        el.innerHTML = (used.length ? plural(used.length, 'recording') + ' into ' + plural(sessions.size, 'session') + ' · ' + plural(states.size, 'syllable') : 'Choose where each recording goes.') +
            (view.rows.length - used.length ? ' · ' + n(view.rows.length - used.length) + ' skipped' : '') +
            (replaces ? ' · replaces the states named "' + esc(name) + '" in ' + plural(replaces, 'session') : '') +
            (pairs.size < used.length ? '<br><span class="behavior-io-warn">Two recordings go to the same animal: they are combined, the one starting earlier winning where they overlap.</span>' : '') +
            (clipped ? '<br><span class="behavior-io-warn">' + plural(clipped, 'recording') + ' run past the end of the session: those rows are left out. Check its first frame.</span>' : '') +
            (mismatch ? '<br><span class="behavior-io-warn">' + plural(mismatch, 'recording') + ' differ in length from what was exported: are these the results of that export?</span>' : '');
        go.disabled = !used.length || !name;
    };

    modal.addEventListener('change', function (e) {
        var t = e.target;
        if (t.id === 'kiSource') { view.source = Number(t.value) || 0; readChosen(); return; }
        if (t.dataset.act === 'to') view.rows[Number(t.dataset.row)].candidate = Number(t.value);
        if (t.dataset.act === 'from') view.rows[Number(t.dataset.row)].firstFrame = Math.max(0, Math.round(Number(t.value) || 1) - 1);
        summarize();
    });
    modal.addEventListener('input', function (e) {
        if (e.target.id === 'kiLayer') { view.layer = e.target.value; summarize(); }
    });

    modal.querySelector('#kiFolder').addEventListener('click', function () {
        if (typeof window.showDirectoryPicker === 'function') {
            window.showDirectoryPicker({ mode: 'read' }).then(async function (h) {
                showLoading('Looking in ' + h.name + '…', { title: 'Import keypoint-MoSeq results' });
                await yieldToPaint();
                var entries;
                try { entries = await walkDirectory(h); } finally { hideLoading(); }
                await load(entries);
            }).catch(function (err) {
                if (err && err.name === 'AbortError') return;
                render('Could not read the folder: ' + (err && err.message ? err.message : err));
            });
            return;
        }
        pickWithInput({ folder: true }, function (files) { load(inputEntries(files)); });
    });
    modal.querySelector('#kiFiles').addEventListener('click', function () {
        pickWithInput({ accept: '.h5,.hdf5,.csv,.yml,.yaml', multiple: true }, function (files) { load(inputEntries(files), true); });
    });
    modal.querySelector('#kiGo').addEventListener('click', function () {
        var res = applyKpmsImport(view.rows.map(function (r) {
            var c = r.candidate >= 0 ? candidates[r.candidate] : null;
            return c ? { session: sessionsOf()[c.session], animal: c.animal, firstFrame: r.firstFrame, syllables: r.syllables } : null;
        }).filter(Boolean), { layer: view.layer, names: view.names });
        if (res) dlg.close();
    });
}

function pickWithInput(o, done) {
    var input = document.createElement('input');
    input.type = 'file';
    if (o.folder) input.webkitdirectory = true;
    if (o.multiple) input.multiple = true;
    if (o.accept) input.accept = o.accept;
    input.hidden = true;
    document.body.appendChild(input);
    input.addEventListener('cancel', function () { input.remove(); });
    input.addEventListener('change', function () {
        var files = input.files;
        input.remove();
        if (files && files.length) done(files);
    });
    input.click();
}

/**
 * Store recordings' syllables as a states layer in their sessions, replacing
 * a layer of that name in each (one per session; nothing else changes).
 * @param {Array<{session: Session, animal: string, firstFrame: number, syllables: ArrayLike<number>}>} items
 * @param {{layer?: string, names?: Map<number,string>|null}} [opts]
 * @returns {{sessions: number, recordings: number, runs: number, states: number}|null}
 */
export function applyKpmsImport(items, opts) {
    var name = cleanLayerName(opts && opts.layer) || DEFAULT_LAYER;
    var bySession = new Map();
    (items || []).forEach(function (it) {
        if (!it || !it.session || !it.animal) return;
        var frames = behaviorSessionInfo(it.session).frames;
        var runs = runsFromSequence(it.syllables, { offset: it.firstFrame || 0, maxFrame: frames > 0 ? frames : Infinity });
        if (!bySession.has(it.session)) bySession.set(it.session, new Map());
        var animals = bySession.get(it.session);
        // A second recording of the same animal: both, the one starting earlier winning where they overlap (setStateLayer cuts overlaps).
        var prev = animals.get(it.animal);
        animals.set(it.animal, prev ? Int32Array.from(Array.from(prev).concat(Array.from(runs))) : runs);
    });
    if (!bySession.size) { setStatus('Nothing to import: choose where the recordings go', 'warning'); return null; }
    var names = opts && opts.names ? opts.names : null;
    var runs = 0, states = new Set(), recs = 0;
    bySession.forEach(function (animals, session) {
        var stateNames = {};
        if (names) names.forEach(function (v, k) { stateNames[k] = v; });
        var r = setStateLayer(session, { name: name, source: 'keypoint-MoSeq', stateNames: stateNames, animals: animals });
        runs += r.runs;
        recs += animals.size;
        if (r.layer) r.layer.animals.forEach(function (rr) { for (var i = 2; i < rr.length; i += 3) states.add(rr[i]); });
    });
    openBehaviorTab();
    var sec = document.getElementById('behaviorStatesSection');
    if (sec && !sec.open) sec.open = true;
    setStatus('Imported keypoint-MoSeq syllables "' + name + '": ' + plural(recs, 'animal') + ' in ' + plural(bySession.size, 'session') + ', ' +
        plural(states.size, 'syllable') + ' · See them in the Behavior tab\'s States and on the timeline', 'success');
    return { sessions: bySession.size, recordings: recs, runs: runs, states: states.size };
}
