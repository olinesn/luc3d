/**
 * pose/model-check.js — a model check (`scan.json`, written by
 * `scripts/sleap_explain_error.py --sweep`) read and matched to the open project.
 * DOM-free and import-free, so the parsing and matching are unit-tested in Node
 * (`tests/test-model-check.mjs`); the tab is `ui/model-check-panel.js`.
 *
 * The file is a list of rows, one per hand-labelled animal the sweep read:
 * `video_index`, `video` (file name), `camera` (the video's parent folder),
 * `frame`, `animal` (its index among the frame's user instances, in file order),
 * `split`, `median_px`, `max_px` (NaN: never detected), `bar_px`, `misread`,
 * optional `name`, and `misread_nodes: {node: {px, cause, read_on}}`, where
 * `read_on` is "animal 2's Ear_L (5.4 px)", "its own Neck (10.2 px)" or null.
 * Python's json writes NaN as a bare `NaN`, which JSON.parse refuses, so the
 * reader turns bare NaN / Infinity into null first and reads null as NaN.
 *
 * Matching has two steps:
 * 1. A scan CAMERA (video_index + video + camera) to a project camera
 *    (`matchScanCameras`): by video file name when the name is distinctive, then
 *    by camera name; ambiguous ones are left for the user to pick. A generic
 *    name (campy writes every video as `0.mp4`) never matches on its own.
 * 2. A row's `animal` to an instance (`matchAnimals`): the camera-frame's user
 *    instances in file order, as the sweep counted them, then checked against
 *    node positions. The file gives no positions, but each `read_on` does give a
 *    distance: node k was read `px` from its own label and `d` from the label it
 *    names, so by the triangle inequality the two labels are between `px - d`
 *    and `px + d` apart. When file order fails that check, the instances that
 *    pass it are searched; a row is matched only when exactly one fits.
 */

/** What each cause means, as the sweep's tally says it (lower case, after "read on"). */
export var CAUSE_TEXT = Object.freeze({
    neighbour_same: "a neighbour's same body part",
    neighbour_other: "a neighbour's other body part",
    swap: 'its own mirror node (left/right swap)',
    own_other: 'another of its own nodes',
    imprecise: 'near its own label, just past the bar',
    background: 'no labelled node (background)',
    not_read: 'not read at all',
});

/** Short names for the cause filter, in the sweep's order; `not_detected` = the animal was never found. */
export var CAUSE_LABELS = Object.freeze({
    neighbour_same: 'Neighbour, same part',
    neighbour_other: 'Neighbour, other part',
    swap: 'Left/right swap',
    own_other: 'Own other part',
    imprecise: 'Imprecise',
    background: 'Background',
    not_read: 'Not read',
    not_detected: 'Not detected',
});

export var SPLITS = Object.freeze(['held-out', 'training', 'unknown']);

/** A video name too generic to identify a camera on its own (campy: `0.mp4`). */
export var GENERIC_VIDEO_NAME = /^\d+\.[A-Za-z0-9]+$/;

/** Slack on the distance checks, px: `read_on` distances are printed to 0.1 px. */
export var DISTANCE_TOLERANCE = 0.1;

/** Above this many instances on a camera-frame, no search past file order. */
var MAX_SEARCH_INSTANCES = 12;

/**
 * Bare NaN / Infinity / -Infinity outside strings -> null, so JSON.parse reads
 * what Python's json.dumps writes.
 * @param {string} text
 * @returns {string}
 */
export function jsonWithoutNaN(text) {
    var out = '', i = 0, n = text.length, inStr = false;
    while (i < n) {
        var c = text[i];
        if (inStr) {
            out += c;
            if (c === '\\') { if (i + 1 < n) out += text[i + 1]; i += 2; continue; }
            if (c === '"') inStr = false;
            i++;
            continue;
        }
        if (c === '"') { inStr = true; out += c; i++; continue; }
        if (c === 'N' && text.startsWith('NaN', i)) { out += 'null'; i += 3; continue; }
        if (c === 'I' && text.startsWith('Infinity', i)) { out += 'null'; i += 8; continue; }
        if (c === '-' && text.startsWith('-Infinity', i)) { out += 'null'; i += 9; continue; }
        out += c; i++;
    }
    return out;
}

function num(x) { return typeof x === 'number' && isFinite(x) ? x : NaN; }
function int(x) { return typeof x === 'number' && Number.isInteger(x) ? x : NaN; }

/**
 * `read_on` text -> {own, animal, node, px}, or null (background, or a text
 * this reader does not know).
 * @param {?string} text - "animal 2's Ear_L (5.4 px)" | "its own Neck (10.2 px)"
 */
export function parseReadOn(text) {
    if (typeof text !== 'string') return null;
    var m = /^(?:its own|animal (\d+)'s) (.+) \((-?[\d.]+) px\)$/.exec(text.trim());
    if (!m) return null;
    return { own: m[1] == null, animal: m[1] == null ? null : parseInt(m[1], 10), node: m[2], px: parseFloat(m[3]) };
}

function normalizeRow(r) {
    if (!r || typeof r !== 'object') return null;
    var frame = int(r.frame), animal = int(r.animal);
    if (typeof r.video !== 'string' || !(frame >= 0) || !(animal >= 0)) return null;
    var nodes = [];
    var mn = r.misread_nodes && typeof r.misread_nodes === 'object' ? r.misread_nodes : {};
    Object.keys(mn).forEach(function (name) {
        var v = mn[name] || {};
        var readOn = typeof v.read_on === 'string' ? v.read_on : null;
        nodes.push({
            node: name,
            px: num(v.px),
            cause: typeof v.cause === 'string' ? v.cause : 'unknown',
            readOn: readOn,
            ref: parseReadOn(readOn),
        });
    });
    nodes.sort(function (a, b) { return (isFinite(b.px) ? b.px : Infinity) - (isFinite(a.px) ? a.px : Infinity) || 0; });
    var maxPx = num(r.max_px);
    return {
        videoIndex: Number.isInteger(r.video_index) ? r.video_index : -1,
        video: r.video,
        camera: typeof r.camera === 'string' ? r.camera : '',
        frame: frame,
        animal: animal,
        split: SPLITS.indexOf(r.split) >= 0 ? r.split : 'unknown',
        name: typeof r.name === 'string' ? r.name : null,
        medianPx: num(r.median_px),
        maxPx: maxPx,
        barPx: num(r.bar_px),
        misread: typeof r.misread === 'boolean' ? r.misread : (nodes.length > 0 || !isFinite(maxPx)),
        nodes: nodes,
    };
}

/**
 * Read a scan.json's text.
 * @param {string} text
 * @returns {{rows: Object[], skipped: number}} rows normalized (camelCase, NaN
 *   for missing numbers, `nodes` sorted by error, worst first)
 * @throws {Error} with a plain message when the text is not a scan
 */
export function parseModelCheck(text) {
    var data;
    try { data = JSON.parse(jsonWithoutNaN(String(text))); }
    catch (e) { throw new Error('This is not a JSON file (' + e.message + ')'); }
    if (data && !Array.isArray(data) && Array.isArray(data.rows)) data = data.rows;
    if (!Array.isArray(data)) throw new Error('This is not a model check: scan.json is a list of rows, one per hand-labelled animal');
    var rows = [], skipped = 0;
    data.forEach(function (r) { var n = normalizeRow(r); if (n) rows.push(n); else skipped++; });
    if (!rows.length) throw new Error('No rows with a video, frame and animal in this file');
    return { rows: rows, skipped: skipped };
}

/** A row's stable key (the sweep's own `name` is not unique across video indices). */
export function rowKey(row) {
    return row.videoIndex + '|' + row.video + '|' + row.camera + '|' + row.frame + '|' + row.animal;
}

/** The scan camera a row was read on. */
export function scanCameraKey(row) {
    return row.videoIndex + '|' + row.video + '|' + row.camera;
}

/** The scan's cameras: [{key, video, camera, videoIndex, rows}] in first-seen order. */
export function scanCameras(rows) {
    var by = new Map();
    rows.forEach(function (r) {
        var k = scanCameraKey(r);
        if (!by.has(k)) by.set(k, { key: k, video: r.video, camera: r.camera, videoIndex: r.videoIndex, rows: [] });
        by.get(k).rows.push(r);
    });
    return Array.from(by.values());
}

/** A path's last component. */
export function baseName(p) {
    return String(p == null ? '' : p).replace(/^.*[\/\\]/, '');
}

function lower(s) { return String(s).toLowerCase(); }

/**
 * Match each scan camera to a project camera.
 *
 * @param {Array<{key, video, camera}>} scanCams - from `scanCameras`
 * @param {Array<{id, camera, videos: string[]}>} projectCams - one per (session,
 *   camera); `videos` are the video file names the project knows for it (loaded
 *   files and the names a loaded .slp stored)
 * @returns {Map<string, {status: 'matched'|'ambiguous'|'unmatched', cam, candidates, how, reason}>}
 *   keyed by scan camera key. `how`: 'video' | 'camera'. An ambiguous one lists
 *   its `candidates` for the user to choose from.
 */
export function matchScanCameras(scanCams, projectCams) {
    var nameCount = new Map();
    scanCams.forEach(function (s) { var k = lower(s.video); nameCount.set(k, (nameCount.get(k) || 0) + 1); });
    var out = new Map();
    scanCams.forEach(function (s) {
        var v = lower(s.video);
        var distinctive = !!v && !GENERIC_VIDEO_NAME.test(s.video) && nameCount.get(v) === 1;
        var byVideo = distinctive ? projectCams.filter(function (p) {
            return (p.videos || []).some(function (n) { return lower(n) === v; });
        }) : [];
        if (byVideo.length > 1) {
            var narrowed = byVideo.filter(function (p) { return p.camera === s.camera; });
            if (narrowed.length === 1) byVideo = narrowed;
        }
        if (byVideo.length === 1) { out.set(s.key, { status: 'matched', cam: byVideo[0], candidates: byVideo, how: 'video', reason: '' }); return; }
        if (byVideo.length > 1) {
            out.set(s.key, { status: 'ambiguous', cam: null, candidates: byVideo, how: 'video',
                reason: byVideo.length + ' cameras in the project have a video named ' + s.video });
            return;
        }
        var byCam = projectCams.filter(function (p) { return p.camera === s.camera; });
        if (!byCam.length && s.camera) byCam = projectCams.filter(function (p) { return lower(p.camera) === lower(s.camera); });
        if (!byCam.length) {
            out.set(s.key, { status: 'unmatched', cam: null, candidates: [], how: '',
                reason: 'no camera named ' + (s.camera || '(none)') + (distinctive ? ' and no video named ' + s.video : '') + ' in the project' });
            return;
        }
        if (byCam.length > 1) {
            out.set(s.key, { status: 'ambiguous', cam: null, candidates: byCam, how: 'camera',
                reason: byCam.length + ' cameras in the project are named ' + s.camera });
            return;
        }
        // One camera of that name. If the project knows its video and it is
        // another distinctive file, the check may have read another recording.
        var known = (byCam[0].videos || []).filter(function (n) { return n && !GENERIC_VIDEO_NAME.test(n); });
        if (distinctive && known.length) {
            out.set(s.key, { status: 'ambiguous', cam: null, candidates: byCam, how: 'camera',
                reason: s.camera + "'s video in the project is " + known[0] + ', the check read ' + s.video });
            return;
        }
        out.set(s.key, { status: 'matched', cam: byCam[0], candidates: byCam, how: 'camera', reason: '' });
    });
    return out;
}

/** Every cause a row's misread nodes show; `not_detected` for an animal never found. */
export function rowCauses(row) {
    var s = new Set();
    if (!isFinite(row.maxPx)) s.add('not_detected');
    row.nodes.forEach(function (n) { s.add(n.cause); });
    return s;
}

/** The worst error, for sorting: an animal never detected is the worst. */
export function worstPx(row) {
    return isFinite(row.maxPx) ? row.maxPx : Infinity;
}

/**
 * The misread rows to list, filtered and sorted worst first (ties: camera, frame, animal).
 * @param {Object[]} rows
 * @param {{split?: string, cause?: string}} [filter] - 'all' or missing = no filter
 */
export function reviewEntries(rows, filter) {
    var f = filter || {};
    return rows.filter(function (r) {
        if (!r.misread) return false;
        if (f.split && f.split !== 'all' && r.split !== f.split) return false;
        if (f.cause && f.cause !== 'all' && !rowCauses(r).has(f.cause)) return false;
        return true;
    }).sort(function (a, b) {
        return (worstPx(b) - worstPx(a)) || (a.camera < b.camera ? -1 : a.camera > b.camera ? 1 : 0) ||
            (a.frame - b.frame) || (a.animal - b.animal);
    });
}

/** "12.3 px". */
export function formatPx(px) {
    return isFinite(px) ? (Math.round(px * 10) / 10).toFixed(1) + ' px' : '—';
}

/**
 * One misread node in words: "Ear_L read on animal 2's Ear_L".
 * @param {Object} n - a row's `nodes` entry
 * @param {(b: number) => ?string} [animalName] - a name for animal b (e.g. its
 *   identity in the project), shown after the sweep's index
 */
export function describeNode(n, animalName) {
    var ref = n.ref;
    if (n.cause === 'not_read') return n.node + ' not read';
    if (n.cause === 'imprecise') return n.node + ' read near its own label, just past the bar';
    if (!ref) return n.node + ' read on ' + (n.readOn || 'the background');
    if (ref.own) return n.node + ' read on its own ' + ref.node;
    var nm = animalName ? animalName(ref.animal) : null;
    return n.node + ' read on animal ' + ref.animal + (nm ? ' (' + nm + ')' : '') + "'s " + ref.node;
}

/**
 * The checks one row's misread nodes allow: each node must be labelled on the
 * animal, a named node on the animal it names, and the two labels must be
 * between `px - d` and `px + d` apart (`dist` checks).
 */
export function rowConstraints(row) {
    var out = [];
    row.nodes.forEach(function (n) {
        out.push({ a: row.animal, k: n.node });
        var ref = n.ref;
        if (!ref || !isFinite(n.px) || !isFinite(ref.px)) return;
        var b = ref.own ? row.animal : ref.animal;
        if (b === row.animal && ref.node === n.node) return;          // read on its own label: no distance
        out.push({ a: b, k: ref.node });
        out.push({ a: row.animal, k: n.node, b: b, j: ref.node,
            lo: Math.max(0, n.px - ref.px) - DISTANCE_TOLERANCE, hi: n.px + ref.px + DISTANCE_TOLERANCE });
    });
    return out;
}

function pointOf(inst, name) {
    var p = inst && inst.points ? inst.points[name] : null;
    return p && isFinite(p[0]) && isFinite(p[1]) ? p : null;
}

function passes(c, map, instances) {
    var ia = map.get(c.a);
    if (ia == null || !instances[ia]) return false;
    var p = pointOf(instances[ia], c.k);
    if (!p) return false;
    if (c.b == null) return true;
    var ib = map.get(c.b);
    if (ib == null || !instances[ib]) return false;
    var q = pointOf(instances[ib], c.j);
    if (!q) return false;
    var d = Math.hypot(p[0] - q[0], p[1] - q[1]);
    return d >= c.lo && d <= c.hi;
}

/**
 * Match each misread row of ONE camera-frame to an instance.
 *
 * @param {Object[]} frameRows - every scan row of the camera-frame (misread or
 *   not: their count is how many hand-labelled animals the sweep saw there)
 * @param {Array<{points: Object<string, number[]>}>} instances - the project's
 *   hand-labelled instances of that camera-frame, in file order as far as the
 *   project knows it; `points[nodeName] = [x, y]` for each labelled node
 * @returns {Map<number, {index, status, reason, checks}>} by `animal`, for the
 *   misread rows: `index` into `instances` or -1; `status` 'verified' (file order,
 *   node positions agree), 'order' (file order; the row gives no distance to
 *   check), 'positions' (the only instance whose nodes fit, not file order's), or
 *   'unmatched'; `checks` = how many distances were checked.
 */
export function matchAnimals(frameRows, instances) {
    var out = new Map();
    var counted = frameRows.reduce(function (m, r) { return Math.max(m, r.animal + 1); }, frameRows.length);
    var M = instances.length;
    frameRows.forEach(function (row) {
        if (!row.misread) return;
        var cons = rowConstraints(row);
        var dists = cons.filter(function (c) { return c.b != null; }).length;
        var res = { index: -1, status: 'unmatched', reason: '', checks: dists };
        out.set(row.animal, res);
        if (!M) { res.reason = 'no hand-labelled animal on this frame in the project'; return; }
        var sameCount = M === counted;
        if (sameCount) {
            var ident = new Map();
            for (var i = 0; i < M; i++) ident.set(i, i);
            if (cons.every(function (c) { return passes(c, ident, instances); })) {
                res.index = row.animal;
                res.status = dists ? 'verified' : 'order';
                return;
            }
        }
        if (!dists) {
            res.reason = sameCount
                ? 'animal ' + row.animal + ' here has no label on ' + row.nodes.map(function (n) { return n.node; }).join(', ')
                : 'the check counted ' + counted + ' hand-labelled animals on this frame, the project has ' + M;
            return;
        }
        if (M > MAX_SEARCH_INSTANCES) {
            res.reason = 'file order does not fit the node positions, and there are too many animals to search';
            return;
        }
        // Who else the row's distances name, then every injective assignment of them.
        var involved = [row.animal];
        cons.forEach(function (c) { [c.a, c.b].forEach(function (x) { if (x != null && involved.indexOf(x) < 0) involved.push(x); }); });
        var fits = new Set(), map = new Map(), used = new Array(M).fill(false);
        (function assign(d) {
            if (d === involved.length) {
                if (cons.every(function (c) { return passes(c, map, instances); })) fits.add(map.get(row.animal));
                return;
            }
            for (var j = 0; j < M; j++) {
                if (used[j]) continue;
                used[j] = true; map.set(involved[d], j);
                assign(d + 1);
                used[j] = false; map.delete(involved[d]);
            }
        })(0);
        if (fits.size === 1) {
            res.index = fits.values().next().value;
            res.status = 'positions';
            res.reason = sameCount ? 'file order does not fit; the node positions fit this animal only'
                : 'the check counted ' + counted + ' animals here, the project has ' + M + '; the node positions fit this one only';
            return;
        }
        res.reason = fits.size
            ? 'the node positions fit ' + fits.size + ' of the animals here'
            : 'no hand-labelled animal here fits the distances the check measured (labels edited since?)';
    });
    return out;
}

/** Plain text for a match status, for the list and its tooltips. */
export var MATCH_STATUS_TEXT = Object.freeze({
    verified: 'Matched in file order; node positions agree',
    order: 'Matched in file order (no distance to check it against)',
    positions: 'Matched by node positions',
    unmatched: 'Not matched to an instance',
    pending: 'Not checked yet: the frame is not loaded',
});

/** Smallest side of the zoomed region, video px: a lone curled-up animal still shows its surroundings. */
export var MODEL_CHECK_MIN_FRAME_PX = 240;

/**
 * The region to zoom to for a located entry: the animal and the animals its
 * nodes were read on, with a margin of 60% of the larger animal side, at least
 * MODEL_CHECK_MIN_FRAME_PX square, centred on them; null without a box.
 */
export function frameRegion(boxes) {
    var b = null, side = 0;
    boxes.forEach(function (x) {
        if (!x) return;
        side = Math.max(side, x.x1 - x.x0, x.y1 - x.y0);
        b = b ? { x0: Math.min(b.x0, x.x0), y0: Math.min(b.y0, x.y0), x1: Math.max(b.x1, x.x1), y1: Math.max(b.y1, x.y1) }
            : { x0: x.x0, y0: x.y0, x1: x.x1, y1: x.y1 };
    });
    if (!b) return null;
    var m = 0.6 * side, cx = (b.x0 + b.x1) / 2, cy = (b.y0 + b.y1) / 2;
    var hw = Math.max((b.x1 - b.x0) / 2 + m, MODEL_CHECK_MIN_FRAME_PX / 2), hh = Math.max((b.y1 - b.y0) / 2 + m, MODEL_CHECK_MIN_FRAME_PX / 2);
    return { x0: cx - hw, y0: cy - hh, x1: cx + hw, y1: cy + hh };
}

/**
 * Places a column of labels as close as it can to the heights they name, none
 * closer than `gap` to the next, inside [lo, hi] when they fit there. `ys` are
 * the wanted heights in any order; returns the placed heights in the same order.
 * Top-down pass pushing each label below the one above, then a bottom-up pass
 * pulling the column back inside `hi`. Pure (tests).
 */
export function spreadLabels(ys, gap, lo, hi) {
    var order = ys.map(function (y, i) { return i; }).sort(function (a, b) { return ys[a] - ys[b]; });
    var out = ys.slice(), prev = -Infinity;
    order.forEach(function (i) { out[i] = Math.max(ys[i], lo, prev + gap); prev = out[i]; });
    var next = Infinity;
    for (var k = order.length - 1; k >= 0; k--) {
        var i = order[k];
        out[i] = Math.min(out[i], hi, next - gap);
        next = out[i];
    }
    return out;
}
