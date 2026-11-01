// import-export/kpms-io.js — keypoint-MoSeq's file formats, both ways: the
// pose tables and settings LUC3D writes for it, and the syllables it writes
// back. keypoint-MoSeq runs in the user's own Python; nothing of it is
// vendored (its licence is academic-only) — only its formats are followed,
// as keypoint-MoSeq 0.8.4 reads and writes them.
//
// ## Out (Behavior ▸ Export for keypoint-MoSeq…)
//
// One table per session × animal, named `<session>__<animal>.csv`
// (`recordingName`): keypoint-MoSeq names a recording after its file, so the
// name comes back with the results and finds its session and animal again.
//   * 3D: anipose's layout, which `load_keypoints(…, 'anipose')` reads:
//     `{bodypart}_x, _y, _z, _score` per node, then `fnum` (the frame, from
//     0). Its reader finds bodyparts with the pattern `(\w+)_x,\1_y,\1_z`,
//     so node names are written with letters, digits and `_` only
//     (`kpmsBodypartNames`). The coordinates are in the FLOOR frame — z the
//     height above the floor, x and y along it — because keypoint-MoSeq
//     turns each pose about z to align its heading.
//   * 2D (no 3D): DeepLabCut's single-animal layout (`'deeplabcut'`): three
//     header rows — scorer, bodyparts, coords (x, y, likelihood) — and the
//     frame as the index column.
// The score / likelihood is 1.0 where LUC3D has the node and 0.0 where it
// does not (its coordinates are left empty) or where the tracking is clearly
// broken (pose/behavior-tensor.js `brokenPoseFrames`: the coordinates are
// kept, and keypoint-MoSeq treats a 0 as missing). Written with the point,
// so pandas reads a float column: keypoint-MoSeq's DeepLabCut reader puts
// NaN into the likelihoods and fails on an integer column.
//
// `luc3d-kpms.yml` (`KPMS_MANIFEST`) goes with them: the keypoint-MoSeq
// settings (bodyparts, use_bodyparts — the nodes the behavior features use,
// tail past its base left out — anterior and posterior bodyparts from the
// node roles, skeleton, fps), each recording's session, animal and first
// frame, and the keypoint-MoSeq calls to run. Every value is written as a
// JSON-style flow value, so the file is plain YAML for Python and
// `parseKpmsManifest` reads it back without a YAML library.
//
// ## In (Behavior ▸ Import keypoint-MoSeq Results…)
//
//   * `results.h5` (`extract_results`): a group per recording holding
//     `syllable` (int64), `latent_state`, `centroid`, `heading`; only the
//     syllables are read (`readResultsH5`, through the vendored h5wasm).
//     Files from older keypoint-MoSeq versions hold `syllables_reindexed`
//     (preferred) or `syllables` instead.
//   * the CSV results folder (`save_results_as_csv`): `<recording>.csv` with
//     a `syllable` column (`parseResultsCsv`).
//   * `syll_info.csv` beside them: syllable names (`parseSyllableNames`).
// Row i of a recording is frame `first_frame + i` of its session (0 unless a
// stretch was exported; the manifest says).
//
// DOM-free: Node tests import it directly.
//
// Depends on: import-export/behavior-io.js (`parseDelimited`)

import { parseDelimited } from './behavior-io.js';

/** The settings file written beside the pose tables. */
export var KPMS_MANIFEST = 'luc3d-kpms.yml';
/** Joins session and animal in a recording name. */
export var RECORDING_SEP = '__';
/** The DeepLabCut "scorer" written into 2D tables. */
export var KPMS_SCORER = 'LUC3D';
var MANIFEST_VERSION = 1;

// ----------------------------------------------------------------------------
// Names
// ----------------------------------------------------------------------------

/**
 * Node names as keypoint-MoSeq can read them back: anipose's reader matches
 * `\w+_x`, so anything but letters, digits and `_` becomes `_`; clashes get
 * `_2`, `_3`….
 * @param {string[]} nodeNames
 * @returns {{names: string[], renamed: Object<string,string>}} renamed: LUC3D name -> written name, only where they differ
 */
export function kpmsBodypartNames(nodeNames) {
    var used = new Set(), names = [], renamed = {};
    (nodeNames || []).forEach(function (nm, i) {
        var base = String(nm == null ? '' : nm).replace(/[^A-Za-z0-9_]+/g, '_').replace(/^_+|_+$/g, '') || 'node' + i;
        var out = base, k = 2;
        while (used.has(out)) out = base + '_' + k++;
        used.add(out);
        names.push(out);
        if (out !== nm) renamed[nm] = out;
    });
    return { names: names, renamed: renamed };
}

/**
 * One part of a recording name: safe as a file name and an HDF5 group name on
 * every system — letters, digits, `.`, `-` and single `_` — so `__` only ever
 * joins session and animal.
 */
export function recordingPart(s, fallback) {
    var out = String(s == null ? '' : s).replace(/[^A-Za-z0-9._-]+/g, '_').replace(/_{2,}/g, '_')
        .replace(/^[._]+|_+$/g, '');
    return out || fallback || 'x';
}

/** The recording (and file) name of an animal in a session: `<session>__<animal>`. */
export function recordingName(sessionName, animal) {
    return recordingPart(sessionName, 'session') + RECORDING_SEP + recordingPart(animal, 'animal');
}

/**
 * Recording names for a list of (session, animal) pairs, made unique (`-2`,
 * `-3`… on a clash, which only names that differ in characters a file name
 * cannot hold can cause).
 * @param {Array<{session: string, animal: string}>} pairs
 * @returns {string[]}
 */
export function recordingNames(pairs) {
    var used = new Set();
    return (pairs || []).map(function (p) {
        var base = recordingName(p.session, p.animal), out = base, k = 2;
        while (used.has(out.toLowerCase())) out = base + '-' + k++;
        used.add(out.toLowerCase());
        return out;
    });
}

// ----------------------------------------------------------------------------
// Pose tables
// ----------------------------------------------------------------------------

/**
 * Digits after the point for coordinates whose bodies are `bodyLength` long:
 * about 1/10,000 of a body length (3 for 60 mm, 6 for 0.06 m, 2 for 300 px).
 */
export function coordinateDecimals(bodyLength) {
    if (!(bodyLength > 0) || !Number.isFinite(bodyLength)) return 3;
    return Math.max(0, Math.min(9, Math.ceil(4 - Math.log10(bodyLength))));
}

function fmt(v, p) {
    return '' + Math.round(v * p) / p;
}

/**
 * One animal's pose table as text chunks: the header, then rows in blocks.
 * @param {{dim: 2|3, bodyparts: string[], data: Float32Array, nodeCount: number, first: number, last: number,
 *          nodes?: number[], floor?: {origin:number[], e1:number[], e2:number[], up:number[]}|null,
 *          skip?: Uint8Array|null, decimals?: number, rowsPerChunk?: number}} o
 *   data: T × nodeCount × dim (NaN = missing); nodes: which node of the data
 *   each bodypart is (default: the first bodyparts.length, in order);
 *   floor: 3D written in the floor frame; skip: frames scored 0 (broken tracking)
 * @returns {Generator<string>}
 */
export function* poseCsvChunks(o) {
    var D = o.dim === 2 ? 2 : 3, N = o.nodeCount, bps = o.bodyparts || [];
    var nodes = o.nodes || bps.map(function (_, i) { return i; });
    var p = Math.pow(10, Number.isInteger(o.decimals) ? o.decimals : 3);
    var block = o.rowsPerChunk > 0 ? o.rowsPerChunk : 4096;
    var fl = D === 3 && o.floor ? o.floor : null;
    var data = o.data, skip = o.skip || null;
    if (D === 3) {
        var head = [];
        bps.forEach(function (b) { head.push(b + '_x', b + '_y', b + '_z', b + '_score'); });
        head.push('fnum');
        yield head.join(',') + '\n';
    } else {
        var r1 = ['scorer'], r2 = ['bodyparts'], r3 = ['coords'];
        bps.forEach(function (b) { r1.push(KPMS_SCORER, KPMS_SCORER, KPMS_SCORER); r2.push(b, b, b); r3.push('x', 'y', 'likelihood'); });
        yield r1.join(',') + '\n' + r2.join(',') + '\n' + r3.join(',') + '\n';
    }
    var out = '';
    var rows = 0;
    for (var t = o.first; t <= o.last; t++) {
        var broken = skip && skip[t] ? 1 : 0;
        var line = D === 3 ? '' : String(t);
        for (var j = 0; j < nodes.length; j++) {
            var base = (t * N + nodes[j]) * D;
            var x = data[base], y = data[base + 1], z = D === 3 ? data[base + 2] : 0;
            var sep = D === 3 && j === 0 ? '' : ',';
            if (x !== x || y !== y || z !== z) { line += sep + (D === 3 ? ',,,0.0' : ',,0.0'); continue; }
            if (fl) {
                var dx = x - fl.origin[0], dy = y - fl.origin[1], dz = z - fl.origin[2];
                x = fl.e1[0] * dx + fl.e1[1] * dy + fl.e1[2] * dz;
                var y2 = fl.e2[0] * dx + fl.e2[1] * dy + fl.e2[2] * dz;
                z = fl.up[0] * dx + fl.up[1] * dy + fl.up[2] * dz;
                y = y2;
            }
            line += sep + fmt(x, p) + ',' + fmt(y, p) + (D === 3 ? ',' + fmt(z, p) : '') + (broken ? ',0.0' : ',1.0');
        }
        out += (D === 3 ? line + ',' + t : line) + '\n';
        if (++rows === block) { yield out; out = ''; rows = 0; }
    }
    if (out) yield out;
}

/** A whole pose table as one string (small tables, tests). */
export function poseCsv(o) {
    var s = '';
    for (var c of poseCsvChunks(o)) s += c;
    return s;
}

// ----------------------------------------------------------------------------
// The manifest (luc3d-kpms.yml)
// ----------------------------------------------------------------------------

/** A value as YAML flow (JSON with spaces after `,` and `:`, which PyYAML reads). */
function flow(v) {
    if (Array.isArray(v)) return '[' + v.map(flow).join(', ') + ']';
    if (v && typeof v === 'object') {
        return '{' + Object.keys(v).map(function (k) { return JSON.stringify(k) + ': ' + flow(v[k]); }).join(', ') + '}';
    }
    if (typeof v === 'number') return Number.isFinite(v) ? String(v) : 'null';
    if (typeof v === 'boolean' || v == null) return String(v);
    return JSON.stringify(String(v));
}

/** The keypoint-MoSeq calls written into the manifest (Python). */
export function kpmsCalls(o) {
    var fps = o.fps > 0 ? o.fps : 30;
    var target = Math.max(1, Math.round(0.4 * fps));
    return [
        '# keypoint-MoSeq, step by step (https://keypoint-moseq.readthedocs.io). Run it',
        '# in a Python environment with keypoint-MoSeq installed, from the folder',
        '# this file is in. Fitting takes a while: try a few iterations first.',
        'import os, yaml, numpy as np',
        'import matplotlib',
        "matplotlib.use('Agg')   # plots go to files, no windows (a script would wait on each); delete in Jupyter",
        'import keypoint_moseq as kpms',
        '',
        "folder = '.'   # the folder LUC3D exported to",
        "luc3d = yaml.safe_load(open(os.path.join(folder, '" + KPMS_MANIFEST + "')))",
        "project_dir = os.path.join(folder, 'kpms_project')",
        'config = lambda: kpms.load_config(project_dir)',
        "if not os.path.exists(os.path.join(project_dir, 'config.yml')):",
        "    kpms.setup_project(project_dir, **{k: luc3d[k] for k in (",
        "        'bodyparts', 'use_bodyparts', 'anterior_bodyparts', 'posterior_bodyparts', 'skeleton', 'fps')})",
        '',
        "coordinates, confidences, bodyparts = kpms.load_keypoints(os.path.join(folder, '*.csv'), luc3d['format'])",
        'coordinates, confidences = kpms.outlier_removal(coordinates, confidences, project_dir, overwrite=True, **config())',
        'data, metadata = kpms.format_data(coordinates, confidences, **config())',
        '',
        'pca = kpms.fit_pca(**data, **config())',
        'kpms.save_pca(pca, project_dir)',
        'latent_dim = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.9) + 1)   # 90% of the variance',
        "kpms.update_config(project_dir, latent_dim=latent_dim, sigmasq_loc=kpms.estimate_sigmasq_loc(",
        "    data['Y'], data['mask'], filter_size=int(luc3d['fps'])))",
        '',
        '# kappa sets how long syllables last: aim for a median of about 0.4 s (' + target + ' frames',
        '# at ' + (+fps.toFixed(3)) + ' fps). Raise kappa for longer syllables, lower it for shorter',
        '# ones; the full model needs a kappa 10-100x lower than the AR-HMM.',
        'AR_ITERS, FULL_ITERS = 50, 500',
        'KAPPA_AR, KAPPA_FULL = 1e6, 1e4',
        'model = kpms.init_model(data, pca=pca, **config())',
        'model = kpms.update_hypparams(model, kappa=KAPPA_AR)',
        'model, model_name = kpms.fit_model(model, data, metadata, project_dir, ar_only=True, num_iters=AR_ITERS)',
        'model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name, iteration=AR_ITERS)',
        'model = kpms.update_hypparams(model, kappa=KAPPA_FULL)',
        'model = kpms.fit_model(model, data, metadata, project_dir, model_name, ar_only=False,',
        '                       start_iter=current_iter, num_iters=current_iter + FULL_ITERS)[0]',
        '',
        'kpms.reindex_syllables_in_checkpoint(project_dir, model_name)   # syllable 0 = the most used',
        'model, data, metadata, current_iter = kpms.load_checkpoint(project_dir, model_name)',
        'results = kpms.extract_results(model, metadata, project_dir, model_name)',
        'kpms.save_results_as_csv(results, project_dir, model_name)',
        '# Then in LUC3D: Behavior > Import keypoint-MoSeq Results..., and choose this folder.',
    ].join('\n') + '\n';
}

/**
 * The text of `luc3d-kpms.yml`.
 * @param {{format: 'anipose'|'deeplabcut', dim: number, units: string, coordinates: string, fps: number,
 *          bodyparts: string[], useBodyparts: string[], anterior: string[], posterior: string[],
 *          skeleton: string[][], renamed?: Object, recordings: Array<{name, session, animal, firstFrame, frames}>,
 *          exportedAt?: string, notes?: string[]}} o
 */
export function kpmsManifest(o) {
    var fps = o.fps > 0 ? Math.round(o.fps * 1000) / 1000 : 30;
    var lines = [
        '# Poses exported by LUC3D for keypoint-MoSeq' + (o.exportedAt ? ' on ' + o.exportedAt : '') + '.',
        '#',
        '# One table per session and animal, named <session>__<animal>.csv, in',
        '# ' + (o.format === 'deeplabcut' ? "DeepLabCut's 2D layout (load_keypoints(..., 'deeplabcut'))."
            : "anipose's 3D layout (load_keypoints(..., 'anipose'))."),
        o.dim === 3
            ? '# Coordinates: ' + (o.coordinates === 'floor' ? 'the floor frame (z = height above the floor; keypoint-MoSeq aligns headings about z)'
                : 'the calibration\'s world frame (no floor was found)') + ', in ' + (o.units || 'calibration units') + '.'
            : '# Coordinates: image pixels of camera ' + (o.camera || '?') + '.',
        '# Score: 1 where LUC3D has the node, 0 where it does not or the tracking is clearly broken.',
    ];
    (o.notes || []).forEach(function (n) { lines.push('# ' + n); });
    lines.push(
        '#',
        '# The settings below are keypoint-MoSeq\'s (pass them to kpms.setup_project);',
        '# `recordings` tells LUC3D where each table came from, to read the results back.',
        '# `calls` is the keypoint-MoSeq run, to copy into Python.',
        'luc3d_kpms: ' + MANIFEST_VERSION,
        'format: ' + flow(o.format === 'deeplabcut' ? 'deeplabcut' : 'anipose'),
        'dims: ' + flow(o.dim === 2 ? 2 : 3),
        'units: ' + flow(o.units || (o.dim === 2 ? 'px' : 'calibration units')),
        'coordinates: ' + flow(o.coordinates || (o.dim === 2 ? 'image' : 'world')),
        'fps: ' + flow(fps),
        'bodyparts: ' + flow(o.bodyparts || []),
        'use_bodyparts: ' + flow(o.useBodyparts || []),
        'anterior_bodyparts: ' + flow(o.anterior || []),
        'posterior_bodyparts: ' + flow(o.posterior || []),
        'skeleton: ' + flow(o.skeleton || []));
    var renamed = o.renamed || {};
    if (Object.keys(renamed).length) {
        lines.push('# LUC3D node name -> the name in the tables (letters, digits and _ only).');
        lines.push('node_names: ' + flow(renamed));
    }
    lines.push('recordings:');
    (o.recordings || []).forEach(function (r) {
        lines.push('  ' + JSON.stringify(r.name) + ': ' + flow({ session: r.session, animal: r.animal, first_frame: r.firstFrame || 0, frames: r.frames || 0 }));
    });
    lines.push('calls: |');
    kpmsCalls({ fps: fps }).replace(/\n$/, '').split('\n').forEach(function (l) { lines.push(l ? '  ' + l : ''); });
    return lines.join('\n') + '\n';
}

/** Split `"key": value` (or `key: value`) into [key, value text], or null. */
function splitKeyValue(line) {
    if (line.charAt(0) === '"') {
        for (var i = 1; i < line.length; i++) {
            if (line.charAt(i) === '\\') { i++; continue; }
            if (line.charAt(i) !== '"') continue;
            var rest = line.slice(i + 1).replace(/^\s*/, '');
            if (rest.charAt(0) !== ':') return null;
            return [JSON.parse(line.slice(0, i + 1)), rest.slice(1).trim()];
        }
        return null;
    }
    var m = /^([^:#]+):(?:\s+(.*))?$/.exec(line);
    return m ? [m[1].trim(), (m[2] || '').trim()] : null;
}

function scalar(text) {
    try { return JSON.parse(text); } catch (e) { /* not JSON */ }
    var bare = text.replace(/\s+#.*$/, '');
    try { return JSON.parse(bare); } catch (e) { /* plain */ }
    return bare.replace(/^'(.*)'$/, '$1');
}

/**
 * Read `luc3d-kpms.yml` back — what `kpmsManifest` writes (top-level
 * `key: value` lines, the `recordings` block and the `calls` literal).
 * @returns {{value?: Object, error?: string}} value.recordings: name -> {session, animal, first_frame, frames}
 */
export function parseKpmsManifest(text) {
    try {
        var lines = String(text || '').replace(/^﻿/, '').split(/\r\n|\n|\r/);
        var out = {};
        var i = 0;
        while (i < lines.length) {
            var line = lines[i++];
            if (!line.trim() || /^\s/.test(line) || line.charAt(0) === '#') continue;
            var kv = splitKeyValue(line);
            if (!kv) continue;
            var key = kv[0], rest = kv[1];
            if (/^\|[-+]?$/.test(rest)) {
                var block = [];
                while (i < lines.length && (!lines[i].trim() || /^\s/.test(lines[i]))) block.push(lines[i++].replace(/^ {2}/, ''));
                out[key] = block.join('\n').replace(/\n+$/, '') + '\n';
            } else if (rest === '' || rest.charAt(0) === '#') {
                var obj = {};
                while (i < lines.length && (!lines[i].trim() || /^\s/.test(lines[i]))) {
                    var l = lines[i++].trim();
                    if (!l || l.charAt(0) === '#') continue;
                    var e = splitKeyValue(l);
                    if (e) obj[e[0]] = scalar(e[1]);
                }
                out[key] = obj;
            } else {
                out[key] = scalar(rest);
            }
        }
        if (out.luc3d_kpms !== MANIFEST_VERSION) return { error: 'not a LUC3D keypoint-MoSeq file (no luc3d_kpms: 1)' };
        if (!out.recordings || typeof out.recordings !== 'object') out.recordings = {};
        return { value: out };
    } catch (err) {
        return { error: 'could not read it: ' + (err && err.message ? err.message : err) };
    }
}

// ----------------------------------------------------------------------------
// Results
// ----------------------------------------------------------------------------

/** Syllables as an Int32Array (-1 = none), from any typed array — BigInt64Array included — or array. */
export function syllableArray(value) {
    var n = value ? value.length : 0;
    var out = new Int32Array(n);
    for (var i = 0; i < n; i++) {
        var v = Number(value[i]);
        out[i] = Number.isInteger(v) && v >= 0 && v < 2147483647 ? v : -1;
    }
    return out;
}

/** The dataset of a recording group that holds its syllables, or null. */
function syllableDataset(group) {
    var names = ['syllable', 'syllables_reindexed', 'syllables'];
    for (var i = 0; i < names.length; i++) {
        var ds = group.get(names[i]);
        if (ds && ds.type === 'Dataset') return ds;
    }
    return null;
}

/**
 * The syllables of every recording in a keypoint-MoSeq `results.h5`.
 * @param {Object} h5 the h5wasm module (ready)
 * @param {string} path the file in h5wasm's filesystem
 * @returns {{recordings: Array<{name: string, syllables: Int32Array}>, skipped: string[]}}
 *   skipped: root entries that are not recordings
 */
export function readResultsH5(h5, path) {
    var f = new h5.File(path, 'r');
    var recordings = [], skipped = [];
    try {
        // A recording is a group with a syllable dataset; a name with "/" in
        // it (h5py makes nested groups of it) is followed a few levels down.
        var walk = function (group, prefix, depth) {
            group.keys().forEach(function (k) {
                var g = group.get(k), name = prefix ? prefix + '/' + k : k;
                if (!g || g.type !== 'Group') { if (!prefix) skipped.push(k); return; }
                var ds = syllableDataset(g);
                if (ds) {
                    if (!ds.shape || ds.shape.length !== 1) { skipped.push(name); return; }
                    recordings.push({ name: name, syllables: syllableArray(ds.value) });
                } else if (depth < 3) {
                    walk(g, name, depth + 1);
                } else {
                    skipped.push(name);
                }
            });
        };
        walk(f, '', 0);
    } finally {
        f.close();
    }
    return { recordings: recordings, skipped: skipped };
}

/**
 * One recording's syllables from a CSV results file (`save_results_as_csv`).
 * @returns {{syllables?: Int32Array, error?: string}}
 */
export function parseResultsCsv(text) {
    text = String(text || '');
    if (text.charCodeAt(0) === 0xFEFF) text = text.slice(1);
    var nl = text.indexOf('\n');
    var header = (nl < 0 ? text : text.slice(0, nl)).replace(/\r$/, '');
    var cols = header.split(',').map(function (c) { return c.trim().replace(/^"(.*)"$/, '$1'); });
    var col = cols.indexOf('syllable');
    if (col < 0) col = cols.indexOf('syllables_reindexed');
    if (col < 0) col = cols.indexOf('syllables');
    if (col < 0) return { error: 'no syllable column' };
    var vals = [];
    var pos = nl < 0 ? text.length : nl + 1;
    while (pos < text.length) {
        var end = text.indexOf('\n', pos);
        if (end < 0) end = text.length;
        var line = text.slice(pos, end);
        pos = end + 1;
        if (line.charCodeAt(line.length - 1) === 13) line = line.slice(0, -1);
        if (!line) continue;
        var cell;
        if (col === 0) { var c0 = line.indexOf(','); cell = c0 < 0 ? line : line.slice(0, c0); } else cell = line.split(',')[col];
        var v = cell == null || cell === '' ? NaN : Number(cell);
        vals.push(Number.isInteger(v) && v >= 0 ? v : -1);
    }
    return { syllables: Int32Array.from(vals) };
}

/** A CSV results file's recording name: its file name without `.csv`. */
export function recordingOfCsv(fileName) {
    return String(fileName || '').replace(/^.*[\\/]/, '').replace(/\.csv$/i, '');
}

/**
 * Syllable names from keypoint-MoSeq's `syll_info.csv` (columns syllable,
 * label, …): every non-empty label.
 * @returns {Map<number, string>}
 */
export function parseSyllableNames(text) {
    var rows = parseDelimited(text, ',');
    var out = new Map();
    if (!rows.length) return out;
    var head = rows[0].map(function (c) { return String(c).trim(); });
    var cs = head.indexOf('syllable'), cl = head.indexOf('label');
    if (cs < 0 || cl < 0) return out;
    for (var i = 1; i < rows.length; i++) {
        var s = Number(String(rows[i][cs] || '').trim()), lab = String(rows[i][cl] || '').trim();
        if (Number.isInteger(s) && s >= 0 && lab) out.set(s, lab);
    }
    return out;
}

// ----------------------------------------------------------------------------
// Matching recordings to sessions and animals
// ----------------------------------------------------------------------------

/**
 * Where each recording goes: its session and animal, and the frame its first
 * row is. In order: the manifest's record (exact session and animal names);
 * the name `recordingName` gives a (session, animal) of the project (also at
 * the end of a longer name, as keypoint-MoSeq's `path_in_name` makes them);
 * the parts either side of `__` matched to session and animal names, ignoring
 * case. Otherwise unmatched: the dialog asks.
 *
 * @param {string[]} names recording names in the results
 * @param {Array<{session: number, sessionName: string, animal: string}>} candidates every (session, animal) of the project
 * @param {Object} [manifest] a parsed manifest's value
 * @returns {Array<{name, candidate: number, firstFrame: number, frames: number, how: 'manifest'|'name'|'parts'|''}>}
 *   candidate: index into `candidates`, -1 when unmatched; frames: the manifest's row count (0 when unknown)
 */
export function matchRecordings(names, candidates, manifest) {
    var recs = (manifest && manifest.recordings) || {};
    var cands = candidates || [];
    var computed = recordingNames(cands.map(function (c) { return { session: c.sessionName, animal: c.animal }; }));
    var byName = new Map();
    computed.forEach(function (nm, i) { if (!byName.has(nm.toLowerCase())) byName.set(nm.toLowerCase(), i); });
    return (names || []).map(function (name) {
        var r = recs[name];
        if (r && typeof r === 'object') {
            for (var i = 0; i < cands.length; i++) {
                if (cands[i].sessionName === r.session && cands[i].animal === r.animal) {
                    return { name: name, candidate: i, firstFrame: Math.max(0, Math.round(Number(r.first_frame) || 0)),
                             frames: Math.max(0, Math.round(Number(r.frames) || 0)), how: 'manifest' };
                }
            }
        }
        var frames = r && typeof r === 'object' ? Math.max(0, Math.round(Number(r.frames) || 0)) : 0;
        var first = r && typeof r === 'object' ? Math.max(0, Math.round(Number(r.first_frame) || 0)) : 0;
        var low = String(name).toLowerCase();
        if (byName.has(low)) return { name: name, candidate: byName.get(low), firstFrame: first, frames: frames, how: 'name' };
        for (var j = 0; j < computed.length; j++) {
            var c = computed[j].toLowerCase();
            var at = low.length - c.length;
            if (at > 0 && low.slice(at) === c && /[-_/\\.]/.test(low.charAt(at - 1))) {
                return { name: name, candidate: j, firstFrame: first, frames: frames, how: 'name' };
            }
        }
        var cut = String(name).lastIndexOf(RECORDING_SEP);
        if (cut > 0) {
            var sp = name.slice(0, cut).toLowerCase(), ap = name.slice(cut + RECORDING_SEP.length).toLowerCase();
            for (var k = 0; k < cands.length; k++) {
                var cs = recordingPart(cands[k].sessionName, 'session').toLowerCase(), ca = recordingPart(cands[k].animal, 'animal').toLowerCase();
                var sOk = sp === cs || sp.slice(-cs.length - 1) === '-' + cs || sp === String(cands[k].sessionName).toLowerCase();
                if (sOk && (ap === ca || ap === String(cands[k].animal).toLowerCase())) {
                    return { name: name, candidate: k, firstFrame: first, frames: frames, how: 'parts' };
                }
            }
        }
        return { name: name, candidate: -1, firstFrame: first, frames: frames, how: '' };
    });
}
