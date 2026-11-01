// import-export/behavior-io.js — behavior labels to and from files: the
// tables other tools read (bouts, per-frame, time budget, and BORIS's
// aggregated and tabular event layouts) and the files labels come from (a
// BORIS project, BORIS event tables, LUC3D's own bouts table, an ethogram
// JSON).
//
// Frames are the truth (pose/behavior-labels.js). Seconds are frames / fps at
// the session's frame rate, and every table carries the fps it used. Frame
// numbers in files count from 0, as SLEAP and pandas do (the timeline shows
// frame index 0 as frame 1). A bout's end time is when its last frame ends,
// (end + 1) / fps — what BORIS records for a state stopped on the next frame —
// so frames survive the trip through seconds: start = round(t × fps),
// end = round(t_stop × fps) − 1. BORIS tables' `Image index` columns hold the
// bout's first and LAST frame: SimBA takes frames from them and labels START
// to STOP inclusive (`BORIS_appender.py`, `range(start, stop + 1)`), so the
// frame after the bout would give every bout one frame too many there.
//
// Exports take one `info` per session — { session, name, fps, frames,
// animals, media } — which ui/behavior-io-modal.js builds from app state, so
// this module stays DOM-free (Node tests import it). Running states are
// exported as ending at the playhead, as a save writes them.
//
// Imports are two steps that change nothing: `parseBehaviorFile` reads a file
// into observations of bouts, and `planBehaviorImport` maps one observation
// onto a session (subjects → animals, seconds → frames, behaviors by name) and
// says what it would do. The caller adds new behaviors (`ingestEthogram` of
// `ethogramPayload`) and writes the bouts with `labelBouts` — one undo step.
//
// Depends on: pose/behavior-ethogram.js, pose/behavior-labels.js

import { behaviors, behaviorById, behaviorByName } from '../pose/behavior-ethogram.js';
import { exportBoutRows, timeBudget } from '../pose/behavior-labels.js';

var SEP = '\u0001';
var SOURCE_NAMES = { m: 'manual', a: 'accepted', i: 'imported' };

/** BORIS's name for events coded with no subject selected. */
export var NO_FOCAL_SUBJECT = 'No focal subject';

// ----------------------------------------------------------------------------
// Delimited text (RFC 4180)
// ----------------------------------------------------------------------------

/** One cell, quoted when it holds the separator, a quote, a line break or edge spaces. */
export function delimitedCell(v, sep) {
    if (v == null) return '';
    var s = typeof v === 'number' ? (Number.isFinite(v) ? String(v) : '') : String(v);
    if (s.indexOf('"') >= 0 || s.indexOf(sep) >= 0 || /[\r\n]/.test(s) || /^\s|\s$/.test(s)) {
        return '"' + s.replace(/"/g, '""') + '"';
    }
    return s;
}

/** Rows of cells as delimited text with CRLF line ends. */
export function toDelimited(rows, sep) {
    var out = new Array(rows.length);
    for (var i = 0; i < rows.length; i++) {
        var r = rows[i], line = new Array(r.length);
        for (var j = 0; j < r.length; j++) line[j] = delimitedCell(r[j], sep);
        out[i] = line.join(sep);
    }
    return out.join('\r\n') + '\r\n';
}

/** The separator a table most likely uses: tab, comma or semicolon (most frequent in its first lines). */
export function detectSeparator(text) {
    var lines = String(text || '').split(/\r\n|\n|\r/).filter(function (l) { return l.trim(); }).slice(0, 10);
    var best = '\t', bestN = -1;
    ['\t', ',', ';'].forEach(function (sep) {
        var n = 0;
        lines.forEach(function (l) { n += l.split(sep).length - 1; });
        if (n > bestN) { best = sep; bestN = n; }
    });
    return best;
}

/**
 * Delimited text as rows of strings: RFC 4180 quotes ("" inside quotes is a
 * quote), CRLF, LF or CR line ends, a leading byte-order mark dropped.
 * @param {string} text
 * @param {string} [sep] detected when left out
 */
export function parseDelimited(text, sep) {
    text = String(text || '');
    if (text.charCodeAt(0) === 0xFEFF) text = text.slice(1);
    if (!sep) sep = detectSeparator(text);
    var rows = [], row = [], cell = '', quoted = false, n = text.length;
    for (var i = 0; i < n; i++) {
        var c = text[i];
        if (quoted) {
            if (c !== '"') { cell += c; continue; }
            if (text[i + 1] === '"') { cell += '"'; i++; } else quoted = false;
            continue;
        }
        if (c === '"' && cell === '') { quoted = true; continue; }
        if (c === sep) { row.push(cell); cell = ''; continue; }
        if (c === '\r' || c === '\n') {
            row.push(cell); rows.push(row); row = []; cell = '';
            if (c === '\r' && text[i + 1] === '\n') i++;
            continue;
        }
        cell += c;
    }
    if (cell !== '' || row.length) { row.push(cell); rows.push(row); }
    return rows;
}

// ----------------------------------------------------------------------------
// Numbers and names
// ----------------------------------------------------------------------------

/** Seconds with 3 decimals (BORIS's) up to 100 fps and 6 above, so every frame survives the round trip. */
function seconds(frames, fps) {
    return fps > 0 ? (frames / fps).toFixed(fps > 100 ? 6 : 3) : '';
}

function fpsCell(fps) {
    return fps > 0 ? String(+fps.toFixed(3)) : '';
}

function norm(s) {
    return String(s == null ? '' : s).trim().toLowerCase();
}

/** The first number in a cell ("29.97", "29.97;29.97", "12,5" with a decimal comma), or NaN. */
function num(s) {
    var t = String(s == null ? '' : s).trim();
    if (/^-?\d+,\d+$/.test(t)) t = t.replace(',', '.');
    var m = /-?\d+(?:\.\d+)?(?:e[-+]?\d+)?/i.exec(t);
    return m ? parseFloat(m[0]) : NaN;
}

function isNoSubject(s) {
    var n = norm(s);
    return !n || n === norm(NO_FOCAL_SUBJECT);
}

/** BORIS modifier text as values: sets are joined by "|", and "None"/"NA" mean none. */
function splitModifiers(s) {
    if (s == null) return [];
    return String(s).split('|').map(function (m) { return m.trim(); })
        .filter(function (m) { var n = m.toLowerCase(); return m && n !== 'none' && n !== 'na' && n !== 'null'; });
}

/** BORIS's `excluded` field: codes joined by commas (or a list). */
function splitCodes(v) {
    if (Array.isArray(v)) return v.map(function (x) { return String(x).trim(); }).filter(Boolean);
    return typeof v === 'string' ? v.split(',').map(function (x) { return x.trim(); }).filter(Boolean) : [];
}

function baseName(p) {
    var s = String(p || '').split(/[\\/]/).pop();
    return s.replace(/\.[^.]+$/, '');
}

// ----------------------------------------------------------------------------
// Exports
// ----------------------------------------------------------------------------

/** How many frames a session's tables cover: the video's, or past the last bout when that is longer. */
function tableFrames(info, rows) {
    var last = -1;
    rows.forEach(function (r) { if (r[4] > last) last = r[4]; });
    return Math.max(info.frames > 0 ? Math.floor(info.frames) : 0, last + 1);
}

/** The session's animals, then any other name that has labels (an identity deleted or renamed away). */
function animalsOf(info, rows) {
    var out = (info.animals || []).slice();
    rows.forEach(function (r) {
        if (r[1] && out.indexOf(r[1]) < 0) out.push(r[1]);
        if (r[2] && out.indexOf(r[2]) < 0) out.push(r[2]);
    });
    return out;
}

function socialKind(b) {
    return b.mutual ? 'mutual' : b.social ? 'directed' : '';
}

/** The bouts table's columns. */
export var BOUT_COLUMNS = ['session', 'behavior', 'kind', 'social', 'animal', 'target', 'present',
    'start_frame', 'end_frame', 'frames', 'start_s', 'end_s', 'duration_s', 'source', 'note', 'fps'];

/**
 * Every bout, present (1) and "not" (0), one row each, sessions in order. A
 * mutual pair is one row, its animals in name order.
 * @param {Array<Object>} infos
 * @returns {Array<Array>} rows, header first
 */
export function boutsTable(infos) {
    var rows = [BOUT_COLUMNS.slice()];
    (infos || []).forEach(function (info) {
        exportBoutRows(info.session).forEach(function (r) {
            var b = behaviorById(r[0]);
            if (!b) return;
            var n = r[4] - r[3] + 1;
            rows.push([info.name, b.name, b.kind, socialKind(b), r[1], r[2], r[5], r[3], r[4], n,
                seconds(r[3], info.fps), seconds(r[4] + 1, info.fps), seconds(n, info.fps),
                SOURCE_NAMES[r[6]] || 'manual', r[7] || '', fpsCell(info.fps)]);
        });
    });
    return rows;
}

/** A per-frame column's header: the behavior, with the partner of a social one. */
export function perFrameColumnName(b, partner) {
    return partner ? b.name + (b.mutual ? ' <-> ' : ' -> ') + partner : b.name;
}

/**
 * One table per animal of a session: a row per frame (from 0), a column per
 * behavior — per partner for a social one — holding 1 (doing it), 0 (labelled
 * "not") or nothing (unlabelled). A mutual pair shows in both animals'
 * tables, a directed behavior in the actor's. Animals with no label get no
 * table.
 * @param {Object} info
 * @param {string} sep
 * @returns {Array<{animal:string, columns:string[], text:string}>}
 */
export function perFrameTables(info, sep) {
    var rows = exportBoutRows(info.session);
    if (!rows.length) return [];
    var byKey = new Map();
    rows.forEach(function (r) {
        var k = r[0] + SEP + r[1] + SEP + r[2];
        if (!byKey.has(k)) byKey.set(k, []);
        byKey.get(k).push(r);
    });
    var frames = tableFrames(info, rows);
    var names = animalsOf(info, rows);
    var decimals = info.fps > 100 ? 6 : 3;
    var out = [];
    names.forEach(function (animal) {
        var cols = [];
        behaviors().forEach(function (b) {
            if (!b.social) { cols.push({ name: perFrameColumnName(b, ''), key: b.id + SEP + animal + SEP }); return; }
            names.forEach(function (other) {
                if (other === animal) return;
                var a = animal, t = other;
                if (b.mutual && t < a) { a = other; t = animal; }       // pairs are stored in name order
                cols.push({ name: perFrameColumnName(b, other), key: b.id + SEP + a + SEP + t });
            });
        });
        if (!cols.some(function (c) { return byKey.has(c.key); })) return;
        var values = cols.map(function (c) {
            var v = new Int8Array(frames).fill(-1);
            (byKey.get(c.key) || []).forEach(function (r) { v.fill(r[5] ? 1 : 0, r[3], r[4] + 1); });
            return v;
        });
        var lines = new Array(frames + 1);
        lines[0] = ['frame', 'time_s'].concat(cols.map(function (c) { return c.name; }))
            .map(function (h) { return delimitedCell(h, sep); }).join(sep);
        for (var f = 0; f < frames; f++) {
            var line = f + sep + (info.fps > 0 ? (f / info.fps).toFixed(decimals) : '');
            for (var j = 0; j < values.length; j++) {
                var x = values[j][f];
                line += x < 0 ? sep : sep + x;
            }
            lines[f + 1] = line;
        }
        out.push({ animal: animal, columns: cols.map(function (c) { return c.name; }), text: lines.join('\r\n') + '\r\n' });
    });
    return out;
}

/** The time budget table's columns. */
export var BUDGET_COLUMNS = ['session', 'animal', 'behavior', 'bouts', 'present_frames', 'present_s', 'mean_bout_s',
    'absent_frames', 'unlabelled_frames', 'session_frames', 'present_pct', 'fps'];

/**
 * Per session × animal × behavior — every animal, every behavior: bouts, time
 * doing it (a frame counts once however many bouts cover it), frames labelled
 * "not", and frames nobody labelled. No time doing something means "not seen
 * doing it" only where `unlabelled_frames` is 0.
 * @param {Array<Object>} infos
 * @returns {Array<Array>} rows, header first
 */
export function timeBudgetTable(infos) {
    var rows = [BUDGET_COLUMNS.slice()];
    (infos || []).forEach(function (info) {
        var saved = exportBoutRows(info.session);
        var frames = tableFrames(info, saved);
        var tb = new Map(timeBudget(info.session, { includeRunning: true }).map(function (r) {
            return [r.animal + SEP + r.behaviorId, r];
        }));
        animalsOf(info, saved).forEach(function (animal) {
            behaviors().forEach(function (b) {
                var r = tb.get(animal + SEP + b.id) || { bouts: 0, presentFrames: 0, absentFrames: 0, boutFrames: 0 };
                rows.push([info.name, animal, b.name, r.bouts, r.presentFrames, seconds(r.presentFrames, info.fps),
                    r.bouts ? seconds(r.boutFrames / r.bouts, info.fps) : '', r.absentFrames,
                    Math.max(0, frames - r.presentFrames - r.absentFrames), frames,
                    frames ? (100 * r.presentFrames / frames).toFixed(2) : '', fpsCell(info.fps)]);
            });
        });
    });
    return rows;
}

/** A session's present bouts (BORIS has no "not" labels), sorted by start. */
function presentBouts(info) {
    var out = [];
    exportBoutRows(info.session).forEach(function (r) {
        var b = behaviorById(r[0]);
        if (b && r[5]) out.push({ b: b, subject: r[1], target: r[2], start: r[3], end: r[4], note: r[7] || '' });
    });
    return out;
}

function withSessions(infos) {
    return (infos || []).map(function (info) {
        var saved = exportBoutRows(info.session);
        return { info: info, bouts: presentBouts(info), duration: seconds(tableFrames(info, saved), info.fps) };
    });
}

function anyTarget(all) {
    return all.some(function (x) { return x.bouts.some(function (bt) { return !!bt.target; }); });
}

/**
 * BORIS 8's tabular events layout (Export events ▸ Tabular events): a START
 * and a STOP row per state bout, a POINT row per point; the target of a social
 * behavior is its modifier (a mutual pair under its first animal). Only present
 * bouts: BORIS has no "not" labels. START's Time is start / fps and STOP's is
 * (end + 1) / fps, as BORIS times them; `Image index` is the first and the last
 * frame, which is what SimBA's BORIS import (it reads this layout as CSV)
 * turns back into exactly the bout.
 * @returns {Array<Array>} rows, header first
 */
export function borisTabularTable(infos) {
    var all = withSessions(infos);
    var mod = anyTarget(all);
    var rows = [['Observation id', 'Observation date', 'Description', 'Observation duration', 'Observation type', 'Source',
        'Time offset (s)', 'Media duration (s)', 'FPS', 'Subject', 'Behavior', 'Behavioral category']
        .concat(mod ? ['Modifier #1'] : [])
        .concat(['Behavior type', 'Time', 'Media file name', 'Image index', 'Image file path', 'Comment'])];
    all.forEach(function (x) {
        var info = x.info, events = [];
        x.bouts.forEach(function (bt) {
            // `at`: the frame whose start is the event's time; `index`: its Image index.
            if (bt.b.kind === 'point') { events.push({ bt: bt, type: 'POINT', at: bt.start, index: bt.start, order: 2, note: bt.note }); return; }
            events.push({ bt: bt, type: 'START', at: bt.start, index: bt.start, order: 1, note: bt.note });
            events.push({ bt: bt, type: 'STOP', at: bt.end + 1, index: bt.end, order: 0, note: '' });
        });
        // A bout stopping where another starts is closed first, as BORIS lists them.
        events.sort(function (a, b) { return a.at - b.at || a.order - b.order; });
        events.forEach(function (ev) {
            var bt = ev.bt;
            rows.push([info.name, '', '', x.duration, 'Media file', info.media || '', 0, x.duration, fpsCell(info.fps),
                bt.subject, bt.b.name, bt.b.group || '']
                .concat(mod ? [bt.target || ''] : [])
                .concat([ev.type, seconds(ev.at, info.fps), info.media || '', ev.index, '', ev.note]));
        });
    });
    return rows;
}

/**
 * BORIS 8's aggregated events layout (Export events ▸ Aggregated events): one
 * row per present bout, Start (s) to Stop (s) — a point has Stop = Start and
 * no duration, as BORIS writes it. `Image index start` / `stop` are the first
 * and last frame, as in the tabular layout.
 * @returns {Array<Array>} rows, header first
 */
export function borisAggregatedTable(infos) {
    var all = withSessions(infos);
    var mod = anyTarget(all);
    var rows = [['Observation id', 'Observation date', 'Description', 'Observation type', 'Source', 'Time offset (s)',
        'Coding duration', 'Media duration (s)', 'FPS (frame/s)', 'Subject', 'Observation duration by subject by observation',
        'Behavior', 'Behavioral category']
        .concat(mod ? ['Modifier #1'] : [])
        .concat(['Behavior type', 'Start (s)', 'Stop (s)', 'Duration (s)', 'Media file name', 'Image index start',
            'Image index stop', 'Image file path start', 'Image file path stop', 'Comment start', 'Comment stop'])];
    all.forEach(function (x) {
        var info = x.info;
        x.bouts.forEach(function (bt) {
            var point = bt.b.kind === 'point';
            var stop = point ? bt.start : bt.end + 1;               // the time's frame; the Image index is bt.end
            rows.push([info.name, '', '', 'Media file', info.media || '', 0, x.duration, x.duration, fpsCell(info.fps),
                bt.subject, x.duration, bt.b.name, bt.b.group || '']
                .concat(mod ? [bt.target || ''] : [])
                .concat([point ? 'POINT' : 'STATE', seconds(bt.start, info.fps), seconds(stop, info.fps),
                    point ? 'NA' : seconds(bt.end - bt.start + 1, info.fps), info.media || '', bt.start, bt.end, '', '', bt.note, '']));
        });
    });
    return rows;
}

// ----------------------------------------------------------------------------
// Reading files
// ----------------------------------------------------------------------------

function fail(msg) {
    return { error: msg, kind: '', label: '', ethogram: null, ethogramPayload: null, observations: [], subjects: [] };
}

function timedBout(ev, kind, t0, t1, note) {
    return { subject: ev.subject, behavior: ev.behavior, modifiers: ev.modifiers, target: '', mutual: false,
             kind: kind, t0: t0, t1: t1, present: true, note: note || '' };
}

/**
 * Events into bouts. An event of type START / STOP / POINT says what it is
 * (BORIS tables); an untyped one toggles (a BORIS project: a state's first
 * event starts it, the next one with the same subject, behavior and modifiers
 * stops it) unless `isPoint` says it is a point. A start that never stops, or
 * a stop with no start, is unpaired and skipped.
 */
function pairEvents(events, isPoint) {
    var sorted = events.slice().sort(function (a, b) { return a.t - b.t; });       // stable: file order on ties
    var open = new Map(), bouts = [], unpaired = 0;
    sorted.forEach(function (ev) {
        var type = ev.type || (isPoint && isPoint(ev) ? 'POINT' : '');
        if (type === 'POINT') { bouts.push(timedBout(ev, 'point', ev.t, ev.t, ev.note)); return; }
        var k = ev.subject + SEP + norm(ev.behavior) + SEP + ev.modifiers.join('|');
        var o = open.get(k);
        if (type === 'START' || (!type && !o)) {
            if (o) unpaired++;
            open.set(k, ev);
            return;
        }
        if (!o) { unpaired++; return; }
        open.delete(k);
        bouts.push(timedBout(o, 'state', o.t, ev.t, o.note || ev.note));
    });
    unpaired += open.size;
    bouts.sort(function (a, b) { return a.t0 - b.t0; });
    return { bouts: bouts, unpaired: unpaired };
}

/** The subjects of `bouts` ('' = none), in order of appearance, plus `extra` names. */
function distinctSubjects(bouts, extra) {
    var out = [];
    var add = function (s) { if (out.indexOf(s) < 0) out.push(s); };
    bouts.forEach(function (bt) { add(bt.subject); if (bt.target) add(bt.target); });
    (extra || []).forEach(add);
    return out;
}

/**
 * A behavior whose every bout names another subject in a modifier is social:
 * that is how BORIS users code a target. Marks `entries` in place.
 */
function markSocialFromModifiers(entries, bouts, subjectNames) {
    var known = new Set(subjectNames.map(norm));
    entries.forEach(function (e) {
        var mine = bouts.filter(function (bt) { return norm(bt.behavior) === norm(e.name); });
        e.social = mine.length > 0 && mine.every(function (bt) {
            return bt.modifiers.some(function (m) { return known.has(norm(m)) && norm(m) !== norm(bt.subject); });
        });
    });
}

function parseBorisProject(pj) {
    var conf = pj.behaviors_conf && typeof pj.behaviors_conf === 'object' ? pj.behaviors_conf : {};
    var entries = Object.keys(conf).sort(function (a, b) { return (+a) - (+b); })
        .map(function (k) { return conf[k]; })
        .filter(function (e) { return e && typeof e.code === 'string' && e.code.trim(); })
        .map(function (e) {
            return {
                name: e.code.trim(), kind: /point/i.test(String(e.type || '')) ? 'point' : 'state',
                key: typeof e.key === 'string' ? e.key : '', color: typeof e.color === 'string' ? e.color : '',
                description: typeof e.description === 'string' ? e.description : '',
                category: typeof e.category === 'string' ? e.category.trim() : '',
                excluded: splitCodes(e.excluded), social: false, mutual: false, group: '',
            };
        });
    var subjConf = pj.subjects_conf && typeof pj.subjects_conf === 'object' ? pj.subjects_conf : {};
    var subjectNames = Object.keys(subjConf).sort(function (a, b) { return (+a) - (+b); })
        .map(function (k) { return subjConf[k] && subjConf[k].name; })
        .filter(function (n) { return typeof n === 'string' && n.trim(); })
        .map(function (n) { return n.trim(); });
    var pointCodes = new Set(entries.filter(function (e) { return e.kind === 'point'; }).map(function (e) { return norm(e.name); }));
    var obsAll = pj.observations && typeof pj.observations === 'object' ? pj.observations : {};
    var allBouts = [];
    var observations = Object.keys(obsAll).map(function (id) {
        var o = obsAll[id] || {};
        var fps = NaN;
        var fpsMap = o.media_info && o.media_info.fps;
        if (fpsMap && typeof fpsMap === 'object') {
            for (var p in fpsMap) { var v = num(fpsMap[p]); if (v > 0) { fps = v; break; } }
        }
        var media = '';
        if (o.file && typeof o.file === 'object') {
            for (var pl in o.file) { if (Array.isArray(o.file[pl]) && o.file[pl].length) { media = String(o.file[pl][0]); break; } }
        }
        var events = (Array.isArray(o.events) ? o.events : []).filter(Array.isArray).map(function (ev) {
            return { t: num(ev[0]), subject: isNoSubject(ev[1]) ? '' : String(ev[1]).trim(),
                     behavior: String(ev[2] == null ? '' : ev[2]).trim(), modifiers: splitModifiers(ev[3]),
                     note: ev[4] == null ? '' : String(ev[4]), type: '' };
        }).filter(function (ev) { return Number.isFinite(ev.t) && ev.behavior; });
        var res = pairEvents(events, function (ev) { return pointCodes.has(norm(ev.behavior)); });
        res.bouts.forEach(function (bt) { allBouts.push(bt); });
        return { id: id, fps: fps, offset: num(o['time offset']) || 0, media: media, bouts: res.bouts,
                 unpaired: res.unpaired, subjects: distinctSubjects(res.bouts, subjectNames) };
    });
    markSocialFromModifiers(entries, allBouts, subjectNames.concat(distinctSubjects(allBouts)));
    return { kind: 'boris-project', label: 'BORIS project', ethogram: entries, ethogramPayload: null,
             observations: observations, subjects: subjectNames };
}

/** Key/value rows above an old BORIS table's header ("Observation id <tab> obs1", "Player #1 <tab> path"). */
function preamble(rows) {
    var meta = { id: '', offset: 0, media: '' };
    rows.forEach(function (r) {
        var k = norm(r[0]), v = (r[1] || '').trim();
        if (!v) return;
        if (k === 'observation id') meta.id = v;
        else if (k === 'time offset (s)' || k === 'time offset') meta.offset = num(v) || 0;
        else if (/^player #?1$/.test(k) && !meta.media) meta.media = v;
    });
    return meta;
}

function parseLucidBouts(name, head, body) {
    var col = function (h) { return head.indexOf(h); };
    var c = {
        session: col('session'), behavior: col('behavior'), kind: col('kind'), social: col('social'), animal: col('animal'),
        target: col('target'), present: col('present'), start: col('start_frame'), end: col('end_frame'), note: col('note'),
    };
    var get = function (r, j) { return j >= 0 && j < r.length ? r[j] : ''; };
    var groups = new Map(), bad = new Map();
    body.forEach(function (r) {
        var id = get(r, c.session).trim() || baseName(name);
        if (!groups.has(id)) { groups.set(id, []); bad.set(id, 0); }
        var beh = get(r, c.behavior).trim(), animal = get(r, c.animal).trim();
        var s = num(get(r, c.start)), e = c.end >= 0 ? num(get(r, c.end)) : s;
        if (!beh || !animal || !Number.isFinite(s) || !Number.isFinite(e)) { bad.set(id, bad.get(id) + 1); return; }
        var p = norm(get(r, c.present));
        groups.get(id).push({
            subject: animal, target: get(r, c.target).trim(), behavior: beh, modifiers: [],
            kind: norm(get(r, c.kind)) === 'point' ? 'point' : 'state', mutual: norm(get(r, c.social)) === 'mutual',
            social: !!norm(get(r, c.social)),
            start: Math.round(s), end: Math.round(e), present: ['0', 'false', 'no', 'not', 'absent'].indexOf(p) < 0,
            note: get(r, c.note),
        });
    });
    var observations = Array.from(groups.keys()).map(function (id) {
        var bouts = groups.get(id);
        return { id: id, fps: NaN, offset: 0, media: '', bouts: bouts, unpaired: bad.get(id), subjects: distinctSubjects(bouts) };
    });
    return { kind: 'lucid-bouts', label: 'LUC3D bouts table', ethogram: null, ethogramPayload: null,
             observations: observations, subjects: [] };
}

function parseTable(name, text) {
    var rows = parseDelimited(text).filter(function (r) { return r.some(function (c) { return c.trim() !== ''; }); });
    var hi = -1;
    for (var i = 0; i < rows.length && i < 200; i++) {
        var cells = rows[i].map(norm);
        if (cells.indexOf('behavior') >= 0 &&
            (cells.indexOf('time') >= 0 || cells.indexOf('start (s)') >= 0 || cells.indexOf('start_frame') >= 0)) { hi = i; break; }
    }
    if (hi < 0) {
        if (rows.length && rows[0].map(norm).indexOf('frame') >= 0) {
            return fail('This is a per-frame table; import a bouts table or BORIS events instead');
        }
        return fail('Not a table LUC3D can read: it needs a Behavior column with times or frames');
    }
    var head = rows[hi].map(norm);
    var body = rows.slice(hi + 1);
    if (head.indexOf('start_frame') >= 0 && head.indexOf('animal') >= 0) return parseLucidBouts(name, head, body);
    var col = function () {
        for (var a = 0; a < arguments.length; a++) { var j = head.indexOf(arguments[a]); if (j >= 0) return j; }
        return -1;
    };
    var get = function (r, j) { return j >= 0 && j < r.length ? r[j] : ''; };
    var meta = preamble(rows.slice(0, hi));
    var aggregated = col('start (s)') >= 0;
    var c = {
        obs: col('observation id'), subject: col('subject'), behavior: col('behavior'), fps: col('fps', 'fps (frame/s)'),
        offset: col('time offset (s)', 'time offset'), media: col('media file name', 'media file path', 'media file'),
        comment: col('comment start', 'comment'), type: col('behavior type', 'status'),
        time: col('time'), start: col('start (s)'), stop: col('stop (s)'),
    };
    if (!aggregated && c.type < 0) return fail('A BORIS tabular table needs a "Behavior type" (or "Status") column');
    var mods = head.map(function (h, j) { return /^modifier/.test(h) ? j : -1; }).filter(function (j) { return j >= 0; });
    var groups = new Map();
    body.forEach(function (r) {
        var id = get(r, c.obs).trim() || meta.id || baseName(name);
        var g = groups.get(id);
        if (!g) { g = { id: id, fps: NaN, offset: meta.offset, media: meta.media, events: [], bouts: [], unpaired: 0 }; groups.set(id, g); }
        var fps = num(get(r, c.fps));
        if (!(g.fps > 0) && fps > 0) g.fps = fps;
        var off = num(get(r, c.offset));
        if (c.offset >= 0 && Number.isFinite(off)) g.offset = off;
        var media = get(r, c.media).trim();
        if (!g.media && media) g.media = media;
        var beh = get(r, c.behavior).trim();
        if (!beh) return;
        var ev = {
            subject: isNoSubject(get(r, c.subject)) ? '' : get(r, c.subject).trim(), behavior: beh,
            modifiers: [], note: get(r, c.comment), t: NaN, type: '',
        };
        mods.forEach(function (j) { splitModifiers(get(r, j)).forEach(function (m) { ev.modifiers.push(m); }); });
        if (aggregated) {
            var point = /point/i.test(get(r, c.type));
            var t0 = num(get(r, c.start)), t1 = point ? t0 : num(get(r, c.stop));
            if (!Number.isFinite(t0) || !Number.isFinite(t1)) { g.unpaired++; return; }
            g.bouts.push(timedBout(ev, point ? 'point' : 'state', t0, t1, ev.note));
            return;
        }
        ev.t = num(get(r, c.time));
        ev.type = norm(get(r, c.type)).toUpperCase();
        if (!Number.isFinite(ev.t) || ['START', 'STOP', 'POINT'].indexOf(ev.type) < 0) { g.unpaired++; return; }
        g.events.push(ev);
    });
    var observations = Array.from(groups.values()).map(function (g) {
        if (!aggregated) {
            var res = pairEvents(g.events);
            g.bouts = res.bouts;
            g.unpaired += res.unpaired;
        } else {
            g.bouts.sort(function (a, b) { return a.t0 - b.t0; });
        }
        return { id: g.id, fps: g.fps, offset: g.offset || 0, media: g.media, bouts: g.bouts, unpaired: g.unpaired,
                 subjects: distinctSubjects(g.bouts) };
    });
    return { kind: aggregated ? 'boris-aggregated' : 'boris-tabular',
             label: aggregated ? 'BORIS aggregated events' : 'BORIS tabular events',
             ethogram: null, ethogramPayload: null, observations: observations, subjects: [] };
}

/**
 * Read a behavior file. Recognized: a BORIS project (`.boris`), an ethogram
 * JSON (Edit Ethogram ▸ Export), BORIS tabular or aggregated events (TSV or
 * CSV, BORIS 7 or 8 layout), and LUC3D's bouts table. Changes nothing.
 *
 * @param {string} name the file's name (an observation id when the file has none)
 * @param {string} text
 * @returns {{kind:string, label:string, error?:string,
 *            ethogram:Array|null, ethogramPayload:Object|null, subjects:string[],
 *            observations:Array<{id:string, fps:number, offset:number, media:string,
 *                                bouts:Array, unpaired:number, subjects:string[]}>}}
 *   `ethogram`: a BORIS project's behaviors, `{name, kind, key, color,
 *   description, category, excluded, social}`; `ethogramPayload`: an ethogram
 *   JSON as `ingestEthogram` takes it. A bout is `{subject, behavior,
 *   modifiers, target, mutual, kind, present, note}` plus `t0`/`t1` (seconds,
 *   BORIS) or `start`/`end` (frames, LUC3D). `subjects` lists who to map ('' =
 *   no subject).
 */
export function parseBehaviorFile(name, text) {
    text = String(text || '');
    if (text.charCodeAt(0) === 0xFEFF) text = text.slice(1);
    var trimmed = text.trim();
    if (!trimmed) return fail('The file is empty');
    if (trimmed[0] === '{') {
        var obj;
        try { obj = JSON.parse(trimmed); } catch (e) { return fail('Not valid JSON (' + e.message + ')'); }
        if (obj && obj.behaviors_conf) return parseBorisProject(obj);
        var eth = obj && obj.ethogram ? obj.ethogram : obj;
        if (eth && eth.v === 1 && Array.isArray(eth.behaviors)) {
            return { kind: 'ethogram', label: 'Ethogram', ethogram: null, ethogramPayload: eth, observations: [], subjects: [] };
        }
        return fail('Not a BORIS project or an ethogram');
    }
    return parseTable(name, text);
}

// ----------------------------------------------------------------------------
// Planning an import
// ----------------------------------------------------------------------------

/**
 * Behavior entries as an `ingestEthogram` payload (it adds what is new, by
 * name). BORIS exclusions become groups where they form cliques — a LUC3D
 * group excludes all its members from each other, BORIS excludes pairs — named
 * after the members' shared BORIS category when they have one; a name the
 * project's groups already use gets a number.
 *
 * @param {Array} entries `{name, kind, key, color, description, social, mutual, group, category, excluded}`
 * @param {string[]} groupsInUse the ethogram's group names
 * @returns {{payload:Object, groups:Array<{name:string, members:string[]}>, unkept:string[]}}
 *   unkept: behaviors whose exclusions could not become a group
 */
export function ethogramPayload(entries, groupsInUse) {
    entries = entries || [];
    var index = new Map(entries.map(function (e, i) { return [norm(e.name), i]; }));
    var adj = entries.map(function () { return new Set(); });
    entries.forEach(function (e, i) {
        (e.excluded || []).forEach(function (x) {
            var j = index.get(norm(x));
            if (j != null && j !== i) { adj[i].add(j); adj[j].add(i); }
        });
    });
    var used = new Set((groupsInUse || []).map(norm));
    entries.forEach(function (e) { if (e.group) used.add(norm(e.group)); });
    var groupOf = entries.map(function (e) { return e.group || ''; });
    var seen = entries.map(function () { return false; });
    var groups = [], unkept = [];
    for (var i = 0; i < entries.length; i++) {
        if (seen[i] || !adj[i].size) continue;
        var comp = [], stack = [i];
        seen[i] = true;
        while (stack.length) {
            var v = stack.pop();
            comp.push(v);
            adj[v].forEach(function (w) { if (!seen[w]) { seen[w] = true; stack.push(w); } });
        }
        comp.sort(function (a, b) { return a - b; });
        var clique = comp.every(function (a) { return comp.every(function (b) { return a === b || adj[a].has(b); }); });
        if (!clique) { comp.forEach(function (k) { unkept.push(entries[k].name); }); continue; }
        var cats = new Set(comp.map(function (k) { return entries[k].category || ''; }));
        var base = cats.size === 1 && !cats.has('') ? Array.from(cats)[0] : 'exclusive';
        var gname = base, n = 2;
        while (used.has(norm(gname))) gname = base + ' ' + (n++);
        used.add(norm(gname));
        comp.forEach(function (k) { groupOf[k] = gname; });
        groups.push({ name: gname, members: comp.map(function (k) { return entries[k].name; }) });
    }
    return {
        payload: {
            v: 1,
            behaviors: entries.map(function (e, k) {
                var o = { id: 'x' + (k + 1), name: e.name, kind: e.kind === 'point' ? 'point' : 'state' };
                if (e.key) o.key = e.key;
                if (/^#[0-9a-f]{6}$/i.test(e.color || '')) o.color = e.color;
                if (e.social) o.social = true;
                if (e.social && e.mutual) o.mutual = true;
                if (groupOf[k]) o.group = groupOf[k];
                if (e.description) o.description = e.description;
                return o;
            }),
        },
        groups: groups, unkept: unkept,
    };
}

/**
 * The observation an import should default to: the one named like the
 * session, else the one coded on one of its videos, else the first with bouts.
 * @returns {number} an index into `parsed.observations`
 */
export function defaultObservation(parsed, sessionName, mediaNames) {
    var obs = parsed.observations || [];
    var byName = obs.findIndex(function (o) { return sessionName && norm(o.id) === norm(sessionName); });
    if (byName >= 0) return byName;
    var media = new Set((mediaNames || []).map(function (m) { return norm(baseName(m)); }).filter(Boolean));
    var byMedia = obs.findIndex(function (o) { return o.media && media.has(norm(baseName(o.media))); });
    if (byMedia >= 0) return byMedia;
    var withBouts = obs.findIndex(function (o) { return o.bouts.length > 0; });
    return withBouts >= 0 ? withBouts : 0;
}

/**
 * Who each subject of an observation is, by default: the animal of the same
 * name; "no subject" → `focal`; and a file with one subject into a session with
 * one animal pairs them. Anything else is left unmapped ('' = skip).
 * @returns {Object<string,string>}
 */
export function defaultSubjectMap(obs, animals, focal) {
    var byNorm = new Map((animals || []).map(function (a) { return [norm(a), a]; }));
    var map = {};
    (obs.subjects || []).forEach(function (s) {
        map[s] = s === '' ? (focal || (animals && animals[0]) || '') : (byNorm.get(norm(s)) || '');
    });
    var named = (obs.subjects || []).filter(function (s) { return s !== ''; });
    if (named.length === 1 && animals && animals.length === 1 && !map[named[0]]) map[named[0]] = animals[0];
    return map;
}

/**
 * The behaviors an observation uses that the ethogram does not have, as they
 * would be added: a BORIS project's own entry when it has one, else inferred
 * from the bouts — a point when every bout is one, social when every bout has a
 * target (a LUC3D table) or names another subject or animal in a modifier.
 * @returns {Array<Object>} entries for `ethogramPayload`
 */
export function newBehaviorsFor(parsed, obs, animals) {
    var known = new Set((obs.subjects || []).concat(animals || []).map(norm));
    var byName = new Map();
    (obs.bouts || []).forEach(function (bt) {
        if (behaviorByName(bt.behavior)) return;
        var k = norm(bt.behavior);
        if (!byName.has(k)) byName.set(k, { name: bt.behavior, bouts: [] });
        byName.get(k).bouts.push(bt);
    });
    var fromFile = new Map(((parsed && parsed.ethogram) || []).map(function (e) { return [norm(e.name), e]; }));
    return Array.from(byName.values()).map(function (x) {
        var e = fromFile.get(norm(x.name));
        if (e) return e;
        var social = x.bouts.every(function (bt) {
            return !!bt.target || bt.modifiers.some(function (m) { return known.has(norm(m)) && norm(m) !== norm(bt.subject); });
        });
        return {
            name: x.name, kind: x.bouts.every(function (bt) { return bt.kind === 'point'; }) ? 'point' : 'state',
            social: social, mutual: social && x.bouts.every(function (bt) { return bt.mutual; }),
            key: '', color: '', description: '', group: '', category: '', excluded: [],
        };
    });
}

/**
 * What importing observation `obs` into a session would do. Changes nothing.
 *
 * Seconds become frames at `opts.fps` after taking off the observation's time
 * offset (BORIS adds it to every event time): start = round(t × fps), a
 * state's end = round(t_stop × fps) − 1. A social bout's target is a LUC3D
 * table's `target`, or the first modifier that names another animal (through
 * the subject map, or by the animal's own name).
 *
 * @param {Object} parsed `parseBehaviorFile`'s result
 * @param {Object} obs one of `parsed.observations`
 * @param {{fps:number, frames:number, animals:string[], subjectMap:Object<string,string>,
 *          newBehaviors:Array<Object>|null}} opts
 *   subjectMap: subject ('' = none) → animal, '' to skip; newBehaviors: the
 *   entries that will be added (`newBehaviorsFor`), or null to skip their bouts
 * @returns {{bouts:Array<{behavior:string, animal:string, target:string, start:number, end:number,
 *                         present:boolean, note:string}>,
 *            skipped:{unpaired:number, subject:number, target:number, behavior:number, range:number}}}
 */
export function planBehaviorImport(parsed, obs, opts) {
    opts = opts || {};
    var animals = opts.animals || [];
    var animalByNorm = new Map(animals.map(function (a) { return [norm(a), a]; }));
    var map = opts.subjectMap || {};
    var mapByNorm = new Map(Object.keys(map).map(function (k) { return [norm(k), map[k]]; }));
    var toAnimal = function (name) {
        if (Object.prototype.hasOwnProperty.call(map, name)) return map[name] || '';
        if (mapByNorm.has(norm(name))) return mapByNorm.get(norm(name)) || '';
        return animalByNorm.get(norm(name)) || '';
    };
    var adding = new Map((opts.newBehaviors || []).map(function (e) { return [norm(e.name), e]; }));
    var timed = parsed.kind !== 'lucid-bouts';
    var fps = opts.fps > 0 ? opts.fps : 0;
    var frames = opts.frames > 0 ? Math.floor(opts.frames) : 0;
    var offset = timed ? (obs.offset || 0) : 0;
    var skipped = { unpaired: obs.unpaired || 0, subject: 0, target: 0, behavior: 0, range: 0 };
    var out = [];
    (obs.bouts || []).forEach(function (bt) {
        var b = behaviorByName(bt.behavior) || adding.get(norm(bt.behavior));
        if (!b) { skipped.behavior++; return; }
        var animal = toAnimal(bt.subject);
        if (!animal) { skipped.subject++; return; }
        var target = '';
        if (b.social) {
            if (bt.target) target = toAnimal(bt.target);
            else {
                for (var i = 0; i < bt.modifiers.length && !target; i++) {
                    var a = toAnimal(bt.modifiers[i]);
                    if (a && a !== animal) target = a;
                }
            }
            if (!target || target === animal) { skipped.target++; return; }
        }
        var start, end;
        if (timed) {
            if (!fps) { skipped.range++; return; }
            start = Math.round((bt.t0 - offset) * fps);
            end = bt.kind === 'point' ? start : Math.max(start, Math.round((bt.t1 - offset) * fps) - 1);
        } else {
            start = Math.min(bt.start, bt.end);
            end = Math.max(bt.start, bt.end);
        }
        if (end < 0 || (frames && start > frames - 1)) { skipped.range++; return; }
        start = Math.max(0, start);
        if (frames) end = Math.min(frames - 1, end);
        out.push({ behavior: b.name, animal: animal, target: target, start: start, end: end,
                   present: bt.present !== false, note: bt.note || '' });
    });
    return { bouts: out, skipped: skipped };
}
