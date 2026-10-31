/**
 * lazy-store-rows.js — another file's poses, added to a lazy project's columnar store.
 *
 * On a lazy project the store IS the project. Every frame is hydrated from it, a save
 * writes it whole, and everything that holds frames in memory is free to drop one with
 * no user instance, because it can always be rebuilt: playback eviction
 * (`pose/lazy-residency.js`), the sweeps' window release (`sweepLazyFrameWindows`), the
 * save's pruning (`buildSessionRefGraph`). So a pose File ▸ Add SLP brings has to be a
 * store row. Held anywhere else, it is gone the next time one of those runs.
 *
 * Rows are only ever APPENDED. No existing row moves or changes, so `frameRowByCam`,
 * `_rawInstIndex` and the file-wide group refs all stay right, and each added video gets
 * an id of its own, so its frames cannot collide with another video's (the writer keeps
 * the first row it sees for a `video:frame`).
 *
 * All or nothing, in two steps. `allocateStoreRows` allocates every column up front and
 * touches nothing, so running out of memory leaves the project as it was. Then
 * `commitStoreRows` only fills what is already allocated, and swaps the columns in without
 * a break. The old columns stay until the swap, so on a shared store this needs room for
 * the store twice over, once, for a moment (pages are not touched until they are copied).
 *
 * `SioLazyLoader.prepareAddedRows` / `commitAddedRows` drive this. Import-free and
 * DOM-free so `tests/test-lazy-store-rows.mjs` can run it in Node.
 */

// What an added row always writes, whatever the store and the file hold. Any other
// column either has is carried too.
var FRAME_COLUMNS = ['frame_id', 'video', 'frame_idx', 'instance_id_start', 'instance_id_end'];
var INSTANCE_COLUMNS = ['instance_id', 'instance_type', 'frame_id', 'skeleton', 'track', 'from_predicted',
    'score', 'point_id_start', 'point_id_end', 'tracking_score'];
var POINT_COLUMNS = ['x', 'y', 'visible', 'complete'];
var PRED_POINT_COLUMNS = ['x', 'y', 'visible', 'complete', 'score'];
// Instance columns `commitStoreRows` works out rather than copies.
var DERIVED_INSTANCE_COLUMNS = {
    instance_id: 1, frame_id: 1, skeleton: 1, track: 1, from_predicted: 1, point_id_start: 1, point_id_end: 1,
};

/**
 * The `formatId` of a store made for added rows. Anything below 1.1 would have
 * `materializeFrame` shift every point by half a pixel, and below 1.2 ignore
 * `tracking_score`; this module always writes current coordinates.
 */
export var ADDED_STORE_FORMAT = 1.4;

/**
 * Below this `formatId`, `materializeFrame` shifts every row's points by half a pixel
 * (old SLEAP coordinates). Added rows hold current coordinates, so a store this old
 * would show them half a pixel off.
 */
export var OLDEST_STORE_FORMAT = 1.1;

function isColumn(col) {
    return !!col && typeof col.length === 'number';
}

function tableRows(table) {
    var rows = 0;
    for (var k in table) if (isColumn(table[k]) && table[k].length > rows) rows = table[k].length;
    return rows;
}

function unionOf(a, b) {
    var seen = new Set(a);
    for (var i = 0; i < b.length; i++) seen.add(b[i]);
    return Array.from(seen);
}

/**
 * Space for `extra` more rows in every column of `table`, plus any in `keys` it lacks.
 * An array under two keys stays one array. Changes nothing yet.
 */
function allocateTable(table, keys, extra) {
    var rows = tableRows(table);
    var grown = {};
    var byColumn = new Map();
    var all = unionOf(Object.keys(table).filter(function (k) { return isColumn(table[k]); }), keys);
    for (var i = 0; i < all.length; i++) {
        var col = table[all[i]];
        var out = col ? byColumn.get(col) : undefined;
        if (!out) {
            out = new Float64Array(rows + extra);
            if (col) byColumn.set(col, out);
        }
        grown[all[i]] = out;
    }
    return { table: table, rows: rows, grown: grown };
}

/** Every column of an allocated table gets its old rows and takes its old one's place. */
function installTable(plan) {
    var copied = new Set();
    for (var key in plan.grown) {
        var out = plan.grown[key];
        var col = plan.table[key];
        if (col && !copied.has(out)) {
            copied.add(out);
            if (col instanceof Float64Array) out.set(col);
            else for (var i = 0; i < col.length; i++) out[i] = Number(col[i]);   // also BigInt64Array
        }
        // The property, never the object: readers take `store.instancesData.<col>` fresh.
        plan.table[key] = out;
    }
}

/**
 * The frame rows of a store, split by video and in ascending video id. A file can list
 * one video frame twice; the store keeps one row per `video:frame` (the writer would
 * keep the first and lose the second), so a repeat's instances join the first's.
 *
 * @param {Object} source - a `LazyDataStore`, or `storeFromParsedFrames`
 * @returns {Array<{raw: number, rows: number[], extraRows: Map<number, number[]>|null}>}
 *   `rows` in file order; `extraRows` maps a row to the repeats that join it
 */
export function sourceRowsByVideo(source) {
    var fd = source.framesData || {};
    var frameIdxCol = fd.frame_idx || [];
    var videoCol = fd.video || [];
    var byVideo = new Map();
    var firstRowOf = new Map();   // raw video id -> Map(frameIdx -> first row)
    for (var r = 0; r < frameIdxCol.length; r++) {
        var raw = videoCol.length ? Number(videoCol[r]) : 0;
        var block = byVideo.get(raw);
        if (!block) {
            block = { raw: raw, rows: [], extraRows: null };
            byVideo.set(raw, block);
            firstRowOf.set(raw, new Map());
        }
        var seen = firstRowOf.get(raw);
        var f = Number(frameIdxCol[r]);
        var first = seen.get(f);
        if (first === undefined) {
            seen.set(f, r);
            block.rows.push(r);
            continue;
        }
        if (!block.extraRows) block.extraRows = new Map();
        var repeats = block.extraRows.get(first);
        if (repeats) repeats.push(r);
        else block.extraRows.set(first, [r]);
    }
    return Array.from(byVideo.values()).sort(function (a, b) { return a.raw - b.raw; });
}

/** Rows one block adds to each table. */
function countBlockRows(source, block) {
    var fd = source.framesData || {};
    var idn = source.instancesData || {};
    var counts = { frames: block.rows.length, instances: 0, points: 0, predPoints: 0 };
    function countRow(r) {
        var start = Number(fd.instance_id_start[r]) || 0;
        var end = Number(fd.instance_id_end[r]) || 0;
        for (var j = start; j < end; j++) {
            counts.instances++;
            var n = Math.max(0, (Number(idn.point_id_end[j]) || 0) - (Number(idn.point_id_start[j]) || 0));
            if (Number(idn.instance_type ? idn.instance_type[j] : 0) === 1) counts.predPoints += n;
            else counts.points += n;
        }
    }
    for (var i = 0; i < block.rows.length; i++) {
        countRow(block.rows[i]);
        var repeats = block.extraRows && block.extraRows.get(block.rows[i]);
        if (repeats) for (var m = 0; m < repeats.length; m++) countRow(repeats[m]);
    }
    return counts;
}

/**
 * Allocate room in `store` for `blocks`' rows. Changes nothing, so a failure here leaves
 * the project as it was.
 *
 * @param {Object} store - the store to grow (`{}` tables for a new one)
 * @param {Array} blocks - `sourceRowsByVideo` entries, each with `id` set: the video id
 *   its rows carry in this store, unused by anything already there
 * @param {Object} source
 * @returns {Object} for `commitStoreRows`
 */
export function allocateStoreRows(store, blocks, source) {
    var add = { frames: 0, instances: 0, points: 0, predPoints: 0 };
    for (var b = 0; b < blocks.length; b++) {
        var c = countBlockRows(source, blocks[b]);
        add.frames += c.frames;
        add.instances += c.instances;
        add.points += c.points;
        add.predPoints += c.predPoints;
    }
    function keysOf(table, required) {
        return unionOf(required, Object.keys(table || {}).filter(function (k) { return isColumn(table[k]); }));
    }
    return {
        store: store,
        blocks: blocks,
        frames: allocateTable(store.framesData, FRAME_COLUMNS, add.frames),
        instances: allocateTable(store.instancesData, keysOf(source.instancesData, INSTANCE_COLUMNS), add.instances),
        points: allocateTable(store.pointsData, keysOf(source.pointsData, POINT_COLUMNS), add.points),
        predPoints: allocateTable(store.predPointsData, keysOf(source.predPointsData, PRED_POINT_COLUMNS), add.predPoints),
        newRowOf: new Int32Array(tableRows(source.instancesData || {})).fill(-1),
    };
}

/**
 * Write the blocks' rows and swap the grown columns into the store, all in one go.
 * Each block gets `rowMap`: its video frame -> its new frame row.
 *
 * @param {Object} plan - `allocateStoreRows`
 * @param {Object} source
 * @param {Object} rules
 * @param {Int32Array} rules.trackMap - the file's track id -> the id the store uses; a
 *   value outside it has no track (as `materializeFrame` reads it)
 * @param {Array<{id: number, order: number[]|null}>} rules.skeletons - per file skeleton
 *   id: the store's id for it and, when its nodes are listed in another order,
 *   order[fileNode] = storeNode. A row with an unknown id is on the first, as
 *   `materializeFrame` reads it.
 */
export function commitStoreRows(plan, source, rules) {
    var sf = source.framesData || {};
    var si = source.instancesData || {};
    var F = plan.frames.grown, I = plan.instances.grown, P = plan.points.grown, Q = plan.predPoints.grown;
    var nextFrame = plan.frames.rows, nextInst = plan.instances.rows;
    var nextPoint = plan.points.rows, nextPred = plan.predPoints.rows;
    var firstNewInst = nextInst;
    var newRowOf = plan.newRowOf;
    var trackMap = rules.trackMap;

    // [out, src, what to add to each value]: old SLEAP coordinates become current ones.
    var coordShift = source.formatId < OLDEST_STORE_FORMAT ? -0.5 : 0;
    function pointCopies(grown, table) {
        var copies = [];
        for (var key in grown) {
            if (!table || !isColumn(table[key])) continue;
            copies.push([grown[key], table[key], key === 'x' || key === 'y' ? coordShift : 0]);
        }
        return copies;
    }
    var userCopies = pointCopies(P, source.pointsData);
    var predCopies = pointCopies(Q, source.predPointsData);
    var instCopies = [];
    for (var ik in I) {
        if (DERIVED_INSTANCE_COLUMNS[ik] || !isColumn(si[ik])) continue;
        // `materializeFrame` reads no tracking score below format 1.2.
        if (ik === 'tracking_score' && source.formatId < 1.2) continue;
        instCopies.push([I[ik], si[ik]]);
    }

    function writePoints(copies, from, n, base, order) {
        for (var c = 0; c < copies.length; c++) {
            var dst = copies[c][0], src = copies[c][1], shift = copies[c][2];
            for (var k = 0; k < n; k++) dst[base + (order ? order[k] : k)] = Number(src[from + k]) + shift;
        }
    }

    function writeInstances(r, frameRow) {
        var start = Number(sf.instance_id_start[r]) || 0;
        var end = Number(sf.instance_id_end[r]) || 0;
        for (var j = start; j < end; j++) {
            var row = nextInst++;
            newRowOf[j] = row;
            for (var c = 0; c < instCopies.length; c++) instCopies[c][0][row] = Number(instCopies[c][1][j]);
            I.instance_id[row] = row;
            I.frame_id[row] = frameRow;
            var skeleton = rules.skeletons[si.skeleton ? Number(si.skeleton[j]) : 0] || rules.skeletons[0];
            I.skeleton[row] = skeleton.id;
            var track = si.track ? Number(si.track[j]) : -1;
            I.track[row] = track >= 0 && track < trackMap.length ? trackMap[track] : -1;
            I.from_predicted[row] = si.from_predicted ? Number(si.from_predicted[j]) : -1;   // file's row: resolved below
            var from = Number(si.point_id_start[j]) || 0;
            var n = Math.max(0, (Number(si.point_id_end[j]) || 0) - from);
            var order = skeleton.order && skeleton.order.length === n ? skeleton.order : null;
            var base;
            if (Number(si.instance_type ? si.instance_type[j] : 0) === 1) {
                base = nextPred;
                nextPred += n;
                writePoints(predCopies, from, n, base, order);
            } else {
                base = nextPoint;
                nextPoint += n;
                writePoints(userCopies, from, n, base, order);
            }
            I.point_id_start[row] = base;
            I.point_id_end[row] = base + n;
        }
    }

    for (var b = 0; b < plan.blocks.length; b++) {
        var block = plan.blocks[b];
        block.rowMap = new Map();
        for (var i = 0; i < block.rows.length; i++) {
            var r = block.rows[i];
            var frameRow = nextFrame++;
            var frameIdx = Number(sf.frame_idx[r]);
            F.frame_id[frameRow] = frameRow;
            F.video[frameRow] = block.id;
            F.frame_idx[frameRow] = frameIdx;
            F.instance_id_start[frameRow] = nextInst;
            writeInstances(r, frameRow);
            var repeats = block.extraRows && block.extraRows.get(r);
            if (repeats) for (var m = 0; m < repeats.length; m++) writeInstances(repeats[m], frameRow);
            F.instance_id_end[frameRow] = nextInst;
            block.rowMap.set(frameIdx, frameRow);
        }
    }

    // A user instance's link to the prediction it came from, among the rows written.
    for (var k = firstNewInst; k < nextInst; k++) {
        var fp = I.from_predicted[k];
        I.from_predicted[k] = fp >= 0 && fp < newRowOf.length && newRowOf[fp] >= 0 ? newRowOf[fp] : -1;
    }

    installTable(plan.frames);
    installTable(plan.instances);
    installTable(plan.points);
    installTable(plan.predPoints);
}

/**
 * A video id nothing in `store` uses: past every row's, annotation's and negative
 * frame's id, and past the store's own list, because `videoIndexFor` falls back to a
 * raw id as an index.
 */
export function firstFreeVideoId(store) {
    var top = (store.videos || []).length - 1;
    var col = (store.framesData || {}).video || [];
    for (var r = 0; r < col.length; r++) if (Number(col[r]) > top) top = Number(col[r]);
    function noteKey(key) {
        var s = String(key);
        var sep = s.indexOf(':');
        var id = sep > 0 ? Number(s.slice(0, sep)) : NaN;
        if (id > top) top = id;
    }
    [store._roiByFrame, store._maskByFrame, store._bboxByFrame, store._centroidByFrame, store._labelImageByFrame].forEach(function (annMap) {
        if (annMap) for (var key of annMap.keys()) noteKey(key);
    });
    if (store.negativeFrames) for (var neg of store.negativeFrames) noteKey(neg);
    return top + 1;
}

/**
 * The poses of an eagerly parsed file (`parseSlpH5`: an analysis `.h5`) in the columns
 * of a store, so they take the same route as a `.slp`.
 *
 * @param {Object} slpData - `frames`, `skeleton`, `tracks`
 * @returns {Object} what `sourceRowsByVideo` and `commitStoreRows` read
 */
export function storeFromParsedFrames(slpData) {
    var frames = slpData.frames || [];
    var nodes = (slpData.skeleton && slpData.skeleton.nodes) || [];
    var nNodes = nodes.length;
    var nInst = 0, nUser = 0, nPred = 0;
    for (var a = 0; a < frames.length; a++) {
        for (var b = 0; b < frames[a].instances.length; b++) {
            nInst++;
            if (frames[a].instances[b].type === 'user') nUser += nNodes;
            else nPred += nNodes;
        }
    }
    function table(names, rows) {
        var t = {};
        for (var i = 0; i < names.length; i++) t[names[i]] = new Float64Array(rows);
        return t;
    }
    var F = table(FRAME_COLUMNS, frames.length);
    var I = table(INSTANCE_COLUMNS, nInst);
    var P = table(POINT_COLUMNS, nUser);
    var Q = table(PRED_POINT_COLUMNS, nPred);
    var inst = 0, user = 0, pred = 0;
    for (var f = 0; f < frames.length; f++) {
        var frame = frames[f];
        F.frame_id[f] = f;
        F.video[f] = frame.videoIdx;
        F.frame_idx[f] = frame.frameIdx;
        F.instance_id_start[f] = inst;
        for (var j = 0; j < frame.instances.length; j++) {
            var d = frame.instances[j];
            var predicted = d.type !== 'user';
            var score = predicted ? (d.score || 0) : 0;
            var T = predicted ? Q : P;
            var base = predicted ? pred : user;
            for (var k = 0; k < nNodes; k++) {
                var p = d.points ? d.points[k] : null;
                T.x[base + k] = p ? p[0] : NaN;
                T.y[base + k] = p ? p[1] : NaN;
                T.visible[base + k] = p && !(d.occluded && d.occluded[k]) ? 1 : 0;
                T.complete[base + k] = p ? 1 : 0;
                if (predicted) {
                    // A node with no score of its own takes the instance's, as every writer does.
                    var own = d.pointScores ? Number(d.pointScores[k]) : NaN;
                    T.score[base + k] = own === own ? own : score;
                }
            }
            if (predicted) pred += nNodes;
            else user += nNodes;
            I.instance_id[inst] = inst;
            I.instance_type[inst] = predicted ? 1 : 0;
            I.frame_id[inst] = f;
            I.track[inst] = d.trackIdx != null && d.trackIdx >= 0 ? d.trackIdx : -1;
            I.from_predicted[inst] = -1;
            I.score[inst] = score;
            I.point_id_start[inst] = base;
            I.point_id_end[inst] = base + nNodes;
            inst++;
        }
        F.instance_id_end[f] = inst;
    }
    return {
        framesData: F, instancesData: I, pointsData: P, predPointsData: Q,
        skeletons: [{ nodeNames: nodes }],
        tracks: slpData.tracks || [],
        formatId: ADDED_STORE_FORMAT,
    };
}
