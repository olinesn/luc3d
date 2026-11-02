/**
 * unopened-sessions.js — the recording sessions of a file that a loader leaves closed.
 *
 * A `.slp` can hold several recording sessions. Two loaders show only the first: a lazy
 * open (`handleLoadProjectSlpLazy`, so any `.slp` over 200 MB from File ▸ Open) and the
 * single-`.slp` folder loader (`handleLoadSessionFolderSingleSlp`, eager). The others are
 * still the user's data, so a save writes them back with their cameras, groups, 3D,
 * calibration, identities and every key LUC3D does not read. They cannot be edited.
 *
 * Their poses are not here. After a lazy open they are rows of the one columnar store;
 * after an eager read they are `session.outsideFrames`. Either way a save keeps them on
 * their own videos (`outside-frames.js`).
 *
 * ## In memory
 *
 * The reader builds one object graph per session (a Map, arrays and a Float64Array per
 * group). `keepUnopenedSessions` turns each closed session into typed columns, the same
 * tables `/session_data` holds, and lets its graph go. Nothing boxed per group stays
 * resident for as long as the project is open.
 *
 * ## At save
 *
 * `unopenedSessionsForSave` gives the writer a `RecordingSession` whose frame groups are
 * built one at a time as it walks them. A group member names a frame by its position in
 * the file and an instance by its place within that frame. A save renumbers the frames
 * (a corrected frame goes first and its store row is skipped; another session's rows
 * shift it), so each member goes through `outputRow` on every save. The kept columns
 * stay in the file's numbering, so saving twice works.
 *
 * A member's place within its frame only moves on a video the open session shares with
 * a closed one: `renumberUnopenedMembers` follows a delete there. Nothing else edits a
 * store row a closed session names, because those rows are in no camera's row map.
 *
 * ## After an eager read (`kept.eager`)
 *
 * `keepUnopenedSessionsOfRead` takes `parseSlpViaSleapIO`'s result. Three things differ.
 * The file rows are `outsideFrames`' (`rowAt`, `outsideFramesForSave`). A member on a
 * video the open session has too holds the pose itself (`seatMembersOnOpenVideos`),
 * because the eager writer lists a frame's instances in its own order: it follows that
 * pose through edits to wherever the writer puts it (`instancePlacesForSave`), and goes
 * if the pose is deleted. And a skeleton node edit reaches these sessions, because it
 * re-shapes their poses (`kept.eager.addNode`, `kept.eager.removeNode`).
 *
 * DOM-free (tests: `tests/test-unopened-sessions.mjs`, e2e
 * `tests/e2e/lazy-keeps-other-sessions.mjs`). A file with one session never gets here,
 * so it saves byte-identically (`tests/e2e/save-golden-digest.mjs`).
 */

import { fileSessionNames, fileSessionProvenance } from './slp-provenance.js';

// Per-group flags.
var HAS_3D = 1;         // this group has 3D; without it, `instance3d` stays undefined
var PREDICTED_3D = 2;   // `pointScores` holds its rows' scores (`pred_points_3d`)
var HAS_SOURCE = 4;     // `unsmoothed` holds its rows (Temporal Smoothing's source)

// `metadata.lucid` keys every session of a LUC3D project holds identically
// (`plane-metadata.js`). `planePlacements` is per session.
var PROJECT_PLANE_KEYS = ['planeNodes', 'planes', 'meshObjects', 'planeOrigin'];

var encoder = new TextEncoder();
var decoder = new TextDecoder();

function now() {
    return (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
}

/** JSON per row in one byte buffer, so a million short strings stay out of the JS heap. */
function newMetaColumn(n) {
    return { bytes: new Uint8Array(4096), end: new Uint32Array(n), used: 0, any: false };
}

function metaPut(col, i, meta) {
    if (meta && Object.keys(meta).length > 0) {
        var text = JSON.stringify(meta);
        // UTF-8 takes at most 3 bytes per UTF-16 unit.
        var need = col.used + text.length * 3;
        if (need > col.bytes.length) {
            var grown = new Uint8Array(Math.max(need, col.bytes.length * 2));
            grown.set(col.bytes.subarray(0, col.used));
            col.bytes = grown;
        }
        col.used += encoder.encodeInto(text, col.bytes.subarray(col.used)).written;
        col.any = true;
    }
    col.end[i] = col.used;
}

function metaDone(col) {
    return col.any ? { bytes: col.bytes.slice(0, col.used), end: col.end } : null;
}

function metaAt(col, i) {
    if (!col) return {};
    var start = i > 0 ? col.end[i - 1] : 0;
    if (start === col.end[i]) return {};
    return JSON.parse(decoder.decode(col.bytes.subarray(start, col.end[i])));
}

function rows3dOf(points) {
    return ArrayBuffer.isView(points) ? Math.floor(points.length / 3) : points.length;
}

/** `points` (flat, or boxed rows from a file older than `/session_data`) into `out` at row `at`. */
function putPoints3d(out, at, points) {
    if (ArrayBuffer.isView(points)) {
        out.set(points, at * 3);
        return;
    }
    for (var r = 0; r < points.length; r++) {
        for (var c = 0; c < 3; c++) {
            var v = points[r] ? points[r][c] : NaN;
            out[(at + r) * 3 + c] = v == null ? NaN : v;
        }
    }
}

// A file older than `/session_data` holds a group's and a frame's metadata as extra
// keys beside these, and the reader keeps only `metadata`.
var INLINE_FRAME_KEYS = new Set(['frame_idx', 'frameIdx', 'instance_groups', 'labeled_frame_by_camera', 'metadata']);
var INLINE_GROUP_KEYS = new Set(['camcorder_to_lf_and_inst_idx_map', 'instances', 'score', 'points',
    'instance_3d_score', 'instance_3d_point_scores', 'identity_idx', 'metadata']);

function withInlineKeys(meta, record, known) {
    if (!record) return meta;
    var out = null;
    for (var key in record) {
        if (known.has(key)) continue;
        if (!out) out = {};
        out[key] = record[key];
    }
    return out ? Object.assign(out, meta) : meta;
}

function inlineRecordsByFrame(dict) {
    var list = dict && Array.isArray(dict.frame_group_dicts) ? dict.frame_group_dicts : null;
    if (!list) return null;
    var byFrame = new Map();
    list.forEach(function (rec) {
        byFrame.set(Number(rec.frame_idx != null ? rec.frame_idx : (rec.frameIdx != null ? rec.frameIdx : 0)), rec);
    });
    return byFrame;
}

function lucidSkeletonOf(dict) {
    var lucid = dict && dict.metadata && dict.metadata.lucid;
    return (lucid && lucid.skeleton) || null;
}

function projectPlaneKeysOf(dict) {
    var lucid = dict && dict.metadata && dict.metadata.lucid;
    if (!lucid) return '{}';
    var kept = {};
    for (var k = 0; k < PROJECT_PLANE_KEYS.length; k++) {
        if (lucid[PROJECT_PLANE_KEYS[k]] !== undefined) kept[PROJECT_PLANE_KEYS[k]] = lucid[PROJECT_PLANE_KEYS[k]];
    }
    return JSON.stringify(kept);
}

/**
 * One closed session, as columns.
 *
 * @param {Object} typed - the reader's `RecordingSession`. Its frame groups are drained.
 * @param {Object} dict - its `sessions_json` entry, as the file holds it
 * @param {Object} file - see `keepUnopenedSessions`
 * @param {string} firstPlaneKeys - `projectPlaneKeysOf` the file's first session
 */
async function keepOneSession(typed, dict, file, firstPlaneKeys) {
    var nFrames = 0, nGroups = 0, nMembers = 0, nPoints = 0;
    var anyPredicted = false, anySource = false;
    var frameKeys = [];
    for (var entry of typed.frameGroups) {
        var fg = entry[1];
        // The writer leaves out a frame with no group, and no writer makes one.
        if (!fg.instanceGroups.length) continue;
        frameKeys.push(entry[0]);
        nFrames++;
        for (var gi = 0; gi < fg.instanceGroups.length; gi++) {
            var ig = fg.instanceGroups[gi];
            nGroups++;
            if (ig._instanceRefsByCamera) nMembers += ig._instanceRefsByCamera.size;
            var pts = ig.points;
            if (pts) {
                nPoints += rows3dOf(pts);
                if (ig.instance3d && ig.instance3d.pointScores != null) anyPredicted = true;
                if (ig.instance3d && ig.instance3d.lucidUnsmoothed) anySource = true;
            }
        }
    }

    var cameras = typed.cameraGroup.cameras;
    var cameraIndex = new Map(cameras.map(function (c, i) { return [c, i]; }));
    var record = {
        // What the file's `lucid_sessions` provenance is keyed by (slp-provenance.js).
        name: file.names[file.index],
        fileProvenance: fileSessionProvenance(file.provenance, file.names, file.index),
        cameras: cameras,
        cameraGroupMetadata: typed.cameraGroup.metadata,
        videos: [],
        frameIdx: new Float64Array(nFrames),
        groupEnd: new Uint32Array(nFrames),
        frameMeta: null,
        identity: new Int32Array(nGroups),
        score: new Float64Array(nGroups),
        score3d: new Float64Array(nGroups),
        flags: new Uint8Array(nGroups),
        memberEnd: new Uint32Array(nGroups),
        pointEnd: new Uint32Array(nGroups),
        groupMeta: null,
        memberCamera: new Uint16Array(nMembers),
        memberRow: new Float64Array(nMembers),
        memberInst: new Int32Array(nMembers),
        points3d: new Float64Array(nPoints * 3),
        pointScores: anyPredicted ? new Float64Array(nPoints) : null,
        unsmoothed: anySource ? new Float64Array(nPoints * 3).fill(NaN) : null,
        // Members whose row is on a video the open session has too.
        membersOnOpenVideos: new Uint32Array(0),
        // What the file says about this session that the writer does not model: its
        // calibration dict with any extra camera key, its metadata, any other key.
        fileDict: null,
        // Its copy of the project's planes was the first session's when read.
        planesFollowFirst: false,
        // Its copy of the skeleton was too, and its poses follow a node edit (eager only).
        skeletonFollowsFirst: false,
    };

    for (var videoEntry of typed.videoByCamera) {
        var cameraPos = cameraIndex.get(videoEntry[0]);
        if (cameraPos === undefined) continue;
        record.videos.push({ camera: cameraPos, id: file.videoIdOf(videoEntry[1]), video: videoEntry[1] });
    }

    var frameMeta = newMetaColumn(nFrames), groupMeta = newMetaColumn(nGroups);
    var inlineRecords = inlineRecordsByFrame(dict);
    var g = 0, m = 0, p = 0;
    var lastYield = now();
    for (var f = 0; f < nFrames; f++) {
        var frameGroup = typed.frameGroups.get(frameKeys[f]);
        var inlineFrame = inlineRecords ? inlineRecords.get(frameKeys[f]) : null;
        var inlineGroups = inlineFrame && Array.isArray(inlineFrame.instance_groups) ? inlineFrame.instance_groups : [];
        record.frameIdx[f] = frameGroup.frameIdx;
        metaPut(frameMeta, f, withInlineKeys(frameGroup.metadata, inlineFrame, INLINE_FRAME_KEYS));
        for (var i = 0; i < frameGroup.instanceGroups.length; i++, g++) {
            var group = frameGroup.instanceGroups[i];
            var identity = group.identity ? file.identityIndex.get(group.identity) : undefined;
            record.identity[g] = identity === undefined ? -1 : identity;
            record.score[g] = group.score == null ? NaN : group.score;
            var i3d = group.instance3d;
            record.score3d[g] = i3d && i3d.score != null ? i3d.score : NaN;
            if (group._instanceRefsByCamera) {
                for (var ref of group._instanceRefsByCamera) {
                    var camPos = cameraIndex.get(ref[0]);
                    if (camPos === undefined) continue;
                    record.memberCamera[m] = camPos;
                    record.memberRow[m] = ref[1][0];
                    record.memberInst[m] = ref[1][1];
                    m++;
                }
            }
            record.memberEnd[g] = m;
            var points = group.points;
            if (points) {
                var n3d = rows3dOf(points);
                putPoints3d(record.points3d, p, points);
                var flags = HAS_3D;
                if (i3d && i3d.pointScores != null) {
                    flags |= PREDICTED_3D;
                    for (var s = 0; s < n3d; s++) record.pointScores[p + s] = i3d.pointScores[s];
                }
                if (i3d && i3d.lucidUnsmoothed) {
                    flags |= HAS_SOURCE;
                    record.unsmoothed.set(i3d.lucidUnsmoothed, p * 3);
                }
                record.flags[g] = flags;
                p += n3d;
            }
            record.pointEnd[g] = p;
            metaPut(groupMeta, g, withInlineKeys(group.metadata, inlineGroups[i], INLINE_GROUP_KEYS));
        }
        record.groupEnd[f] = g;
        typed.frameGroups.delete(frameKeys[f]);   // the graph shrinks as the columns fill
        if (now() - lastYield > 40) {
            await new Promise(function (r) { setTimeout(r, 0); });
            lastYield = now();
        }
    }
    record.frameMeta = metaDone(frameMeta);
    record.groupMeta = metaDone(groupMeta);

    if (dict) {
        var kept = Object.assign({}, dict);
        delete kept.frame_group_dicts;
        delete kept.fg_start;
        delete kept.fg_end;
        delete kept.camcorder_to_video_idx_map;
        record.fileDict = encoder.encode(JSON.stringify(kept));
        record.planesFollowFirst = !file.keepOwnPlanes && projectPlaneKeysOf(dict) !== '{}' && projectPlaneKeysOf(dict) === firstPlaneKeys;
        record.skeletonFollowsFirst = !!file.firstSkeleton && JSON.stringify(lucidSkeletonOf(dict)) === file.firstSkeleton;
    }

    // A member's video is its row's, whatever its camera's says.
    if (file.openVideoIds.size > 0) {
        var onOpen = [];
        for (var mi = 0; mi < m; mi++) {
            if (file.openVideoIds.has(file.rowVideoId(record.memberRow[mi]))) onOpen.push(mi);
        }
        record.membersOnOpenVideos = Uint32Array.from(onOpen);
    }
    return record;
}

/**
 * At load: turn every session but the first into columns and release the reader's
 * graphs. `labels.sessions` keeps only the first afterwards.
 *
 * @param {Object} labels - the reader's lazy `Labels` (read with `rawSessions: true`)
 * @param {Object} opts
 * @param {Set<number>} opts.openVideoIds - store video ids of the open session's cameras
 * @param {(row: number) => number} [opts.rowVideoId] - a file row's video id, for a read
 *   with no store
 * @param {boolean} [opts.keepOwnPlanes] - a closed session's planes never follow the open
 *   session's (they are not the same project's)
 * @param {boolean} [opts.skeletonFollowsFirst] - a closed session's copy of the skeleton
 *   follows the open session's
 * @returns {Promise<{sessions: Array, identities: Array}|null>} null when the file has
 *   no other session. `identities` is the file's catalog: what `identity` columns index.
 */
export async function keepUnopenedSessions(labels, opts) {
    var typed = labels.sessions || [];
    if (typed.length < 2) return null;
    var rawDicts = labels.rawSessionsJson || [];
    var store = labels._lazyDataStore;
    var videoCol = (store && store.framesData && store.framesData.video) || [];
    // A video's position in `labels.videos` is its id in the frames table unless the
    // file numbers them sparsely (`video0`, `video2`).
    var idOfPosition = new Map();
    if (store && typeof store.videoIndexFor === 'function') {
        var last;
        for (var r = 0; r < videoCol.length; r++) {
            var id = Number(videoCol[r]);
            if (id !== last) { idOfPosition.set(store.videoIndexFor(id), id); last = id; }
        }
    }
    var videos = labels.videos || [];
    var file = {
        names: fileSessionNames(rawDicts),
        provenance: labels.provenance,
        identityIndex: new Map((labels.identities || []).map(function (id, i) { return [id, i]; })),
        openVideoIds: (opts && opts.openVideoIds) || new Set(),
        videoIdOf: function (video) {
            var position = videos.indexOf(video);
            return idOfPosition.has(position) ? idOfPosition.get(position) : position;
        },
        rowVideoId: (opts && opts.rowVideoId) || function (row) { return Number(videoCol[row]); },
        keepOwnPlanes: !!(opts && opts.keepOwnPlanes),
        firstSkeleton: opts && opts.skeletonFollowsFirst && lucidSkeletonOf(rawDicts[0]) ? JSON.stringify(lucidSkeletonOf(rawDicts[0])) : null,
        index: 0,
    };
    var firstPlaneKeys = projectPlaneKeysOf(rawDicts[0]);
    var kept = [];
    for (var si = 1; si < typed.length; si++) {
        file.index = si;
        kept.push(await keepOneSession(typed[si], rawDicts[si], file, firstPlaneKeys));
    }
    typed.length = 1;
    return { sessions: kept, identities: labels.identities || [] };
}

/**
 * Can an eager read keep its later sessions whole? Only the typed reader gives them: the
 * raw worker (`parseSlpH5`) has no `_typedSession`.
 * @param {Object} slpData - `parseSlpViaSleapIO` / `parseSlpH5` result
 * @returns {boolean}
 */
export function readKeepsUnopenedSessions(slpData) {
    var dicts = (slpData && slpData.sessions) || [];
    return dicts.length > 1 && dicts.every(function (d) { return !!d._typedSession; });
}

/**
 * `keepUnopenedSessions` for an eager read, once `readKeepsUnopenedSessions` said yes.
 *
 * @param {Object} slpData - `parseSlpViaSleapIO` result. Its later sessions are drained.
 * @param {Object} opts
 * @param {Map<string, number>} opts.videoIdByCam - camera name -> video index, the open
 *   session's
 * @param {boolean} [opts.keepOwnPlanes] - see `keepUnopenedSessions`
 * @returns {Promise<Object>} what `keepUnopenedSessions` gives, with `eager` set
 */
export async function keepUnopenedSessionsOfRead(slpData, opts) {
    var dicts = slpData.sessions;
    var frames = slpData.frames || [];
    // `frames` is in file order and leaves out rows with no instance, which no member names.
    var videoOfRow = new Int32Array(frames.length ? frames[frames.length - 1].fileRow + 1 : 0).fill(-1);
    for (var f = 0; f < frames.length; f++) videoOfRow[frames[f].fileRow] = frames[f].videoIdx;
    var kept = await keepUnopenedSessions({
        sessions: dicts.map(function (d) { return d._typedSession; }),
        rawSessionsJson: dicts.map(function (d) {
            var raw = Object.assign({}, d);
            delete raw._typedSession;
            delete raw._typedIdentities;
            return raw;
        }),
        videos: (slpData.videos || []).map(function (v) { return v._typedVideo; }),
        identities: dicts[0]._typedIdentities,
        provenance: slpData.provenance,
    }, {
        openVideoIds: new Set(opts.videoIdByCam.values()),
        rowVideoId: function (row) { return videoOfRow[row]; },
        keepOwnPlanes: opts.keepOwnPlanes,
        skeletonFollowsFirst: true,
    });
    kept.eager = {
        videoIdByCam: opts.videoIdByCam,
        // File row -> the open session's pose at each place in that row, for rows a
        // closed session's member names on a video the open session has too.
        posesOfRows: new Map(),
        // What the file's events know each video by (`fileEventsOf`), by its place in the
        // file: a kept bout goes on the entry its video has (`outsideFramesForSave`).
        videoRecords: slpData.behaviorEvents ? slpData.behaviorEvents.videos : null,
        // For `Session.propagateNodeAdded`/`propagateNodeRemoved` (`pose-data.js` imports nothing).
        addNode: function () { addNodeToUnopenedSessions(kept); },
        removeNode: function (nodeIdx) { removeNodeFromUnopenedSessions(kept, nodeIdx); },
    };
    return kept;
}

/**
 * How far `pose` is from a file's `points`, or Infinity when they are not one pose. Nodes
 * only one side has count against it, because the reader drops a pose matching a grouped
 * one on the nodes both have (`restoreGroupingAndUnlink`), and then that one stands for it.
 */
function poseGap(pose, points) {
    if (pose.numNodes !== points.length) return Infinity;
    var gap = 0, shared = 0, onlyOneSide = 0;
    for (var n = 0; n < points.length; n++) {
        var p = points[n], has = pose.hasPoint(n);
        if (!p || !has) { if (p || has) onlyOneSide++; continue; }
        var d = Math.max(Math.abs(pose.getX(n) - p[0]), Math.abs(pose.getY(n) - p[1]));
        if (d > 0.5) return Infinity;
        gap += d;
        shared++;
    }
    if (shared === 0 && onlyOneSide > 0) return Infinity;
    return gap + onlyOneSide;
}

/**
 * After an eager read: tie each member on a video the open session has too to the open
 * session's pose, once its instances are final. A group's poses are rebuilt from the file
 * and ungrouped ones keep their own objects, so the file's coordinates say which is which.
 *
 * @param {Object} kept - `keepUnopenedSessionsOfRead`
 * @param {Array} frames - `slpData.frames`
 * @param {(frame: Object) => Array} posesOfFrame - the open session's `Instance`s on that
 *   file frame's camera and frame number
 */
export function seatMembersOnOpenVideos(kept, frames, posesOfFrame) {
    if (!hasMembersOnOpenVideos(kept)) return;
    var frameOfRow = new Map(frames.map(function (f) { return [f.fileRow, f]; }));
    var seated = kept.eager.posesOfRows;
    kept.sessions.forEach(function (record) {
        record.membersOnOpenVideos.forEach(function (m) {
            var row = record.memberRow[m], k = record.memberInst[m];
            var frame = frameOfRow.get(row);
            var filePose = frame && frame.instances[k];
            if (!filePose || (seated.has(row) && seated.get(row)[k])) return;
            var best = null, bestGap = Infinity;
            posesOfFrame(frame).forEach(function (pose) {
                var gap = poseGap(pose, filePose.points);
                if (gap < bestGap) { best = pose; bestGap = gap; }
            });
            if (!best) return;
            if (!seated.has(row)) seated.set(row, []);
            seated.get(row)[k] = best;
        });
    });
}

/**
 * The open session's poses a save must find: for `buildSlpLabelsAllViews`, which sets
 * `[labeled frame, place in it]` on each one it writes.
 * @param {Object|null} kept - `session.unopenedSessions`
 * @returns {Map<Object, (number[]|null)>|null} null when no member needs one
 */
export function instancePlacesForSave(kept) {
    if (!kept || !kept.eager || kept.eager.posesOfRows.size === 0) return null;
    var places = new Map();
    kept.eager.posesOfRows.forEach(function (poses) {
        poses.forEach(function (pose) { places.set(pose, null); });
    });
    return places;
}

/**
 * Copy one group's 3D rows into new columns: `rowsOf(n)` says which of its `n` rows go,
 * in order, with -1 for a new row, NaN.
 */
function reshapePoints(record, rowsOf) {
    var nGroups = record.pointEnd.length;
    var newEnd = new Uint32Array(nGroups);
    var total = 0, start = 0, g;
    for (g = 0; g < nGroups; g++) {
        var n = record.pointEnd[g] - start;
        total += n > 0 ? rowsOf(n).length : 0;
        newEnd[g] = total;
        start = record.pointEnd[g];
    }
    var points3d = new Float64Array(total * 3);
    var pointScores = record.pointScores ? new Float64Array(total) : null;
    var unsmoothed = record.unsmoothed ? new Float64Array(total * 3) : null;
    var at = 0;
    start = 0;
    for (g = 0; g < nGroups; g++) {
        var count = record.pointEnd[g] - start;
        var rows = count > 0 ? rowsOf(count) : [];
        for (var r = 0; r < rows.length; r++, at++) {
            var from = start + rows[r];
            for (var c = 0; c < 3; c++) {
                points3d[at * 3 + c] = rows[r] < 0 ? NaN : record.points3d[from * 3 + c];
                if (unsmoothed) unsmoothed[at * 3 + c] = rows[r] < 0 ? NaN : record.unsmoothed[from * 3 + c];
            }
            if (pointScores) pointScores[at] = rows[r] < 0 ? NaN : record.pointScores[from];
        }
        start = record.pointEnd[g];
    }
    record.pointEnd = newEnd;
    record.points3d = points3d;
    record.pointScores = pointScores;
    record.unsmoothed = unsmoothed;
}

/**
 * A skeleton node was added, after an eager read: the closed sessions' poses got an empty
 * slot (`Session.propagateNodeAdded`), so each group's 3D gets one row, NaN. Every other
 * row keeps its value and its index.
 * @param {Object} kept - `session.unopenedSessions`, with `eager` set
 */
function addNodeToUnopenedSessions(kept) {
    kept.sessions.forEach(function (record) {
        reshapePoints(record, function (n) {
            var rows = [];
            for (var r = 0; r < n; r++) rows.push(r);
            rows.push(-1);
            return rows;
        });
    });
}

/**
 * A skeleton node was removed, after an eager read: 3D rows and hidden-node lists are
 * indexed by node, like the poses `Session.propagateNodeRemoved` re-shaped.
 * @param {Object} kept - `session.unopenedSessions`, with `eager` set
 * @param {number} nodeIdx
 */
function removeNodeFromUnopenedSessions(kept, nodeIdx) {
    kept.sessions.forEach(function (record) {
        reshapePoints(record, function (n) {
            var rows = [];
            for (var r = 0; r < n; r++) if (r !== nodeIdx) rows.push(r);
            return rows;
        });
        if (record.groupMeta) record.groupMeta = groupMetaWithoutNode(record, nodeIdx);
    });
}

function groupMetaWithoutNode(record, nodeIdx) {
    var nGroups = record.memberEnd.length;
    var out = newMetaColumn(nGroups);
    for (var g = 0; g < nGroups; g++) {
        var meta = metaAt(record.groupMeta, g);
        var perCamera = meta.lucid && meta.lucid.instanceMeta;
        for (var cam in perCamera || {}) {
            var hidden = perCamera[cam].nulledNodes;
            if (!Array.isArray(hidden)) continue;
            perCamera[cam].nulledNodes = hidden.filter(function (k) { return k !== nodeIdx; })
                .map(function (k) { return k > nodeIdx ? k - 1 : k; });
        }
        metaPut(out, g, meta);
    }
    return metaDone(out);
}

function* frameGroupsOf(SIO, record, lf, inst, identityOf) {
    var g = 0, m = 0, p = 0;
    for (var f = 0; f < record.frameIdx.length; f++) {
        var groups = [];
        for (; g < record.groupEnd[f]; g++) {
            var refs = null;
            for (; m < record.memberEnd[g]; m++) {
                if (lf[m] < 0) continue;
                if (!refs) refs = new Map();
                refs.set(record.cameras[record.memberCamera[m]], [lf[m], inst[m]]);
            }
            var instance3d = undefined;
            var pEnd = record.pointEnd[g], flags = record.flags[g];
            if (flags & HAS_3D) {
                var score3d = record.score3d[g] === record.score3d[g] ? record.score3d[g] : undefined;
                var points = record.points3d.subarray(p * 3, pEnd * 3);
                instance3d = (flags & PREDICTED_3D)
                    ? new SIO.PredictedInstance3D({ points: points, skeleton: null, score: score3d, pointScores: record.pointScores.subarray(p, pEnd) })
                    : new SIO.Instance3D({ points: points, skeleton: null, score: score3d });
                if (flags & HAS_SOURCE) instance3d.lucidUnsmoothed = record.unsmoothed.subarray(p * 3, pEnd * 3);
            }
            p = pEnd;
            groups.push(new SIO.InstanceGroup({
                instanceRefsByCamera: refs || undefined,
                score: record.score[g] === record.score[g] ? record.score[g] : undefined,
                instance3d: instance3d,
                identity: record.identity[g] >= 0 ? identityOf(record.identity[g]) : undefined,
                metadata: metaAt(record.groupMeta, g),
            }));
        }
        yield new SIO.FrameGroup({ frameIdx: record.frameIdx[f], instanceGroups: groups, metadata: metaAt(record.frameMeta, f) });
    }
}

/**
 * The file's catalog entries the kept sessions' groups name, in catalog order.
 * @param {{sessions: Array}} kept
 * @returns {number[]}
 */
function catalogEntriesUsed(kept) {
    var used = new Set();
    kept.sessions.forEach(function (record) {
        var col = record.identity;
        var last = -1;
        for (var g = 0; g < col.length; g++) {
            if (col[g] !== last) { if (col[g] >= 0) used.add(col[g]); last = col[g]; }
        }
    });
    return Array.from(used).sort(function (a, b) { return a - b; });
}

/**
 * The catalog entries the open session itself lists: same place in the file's catalog
 * (`Identity.id`, set by the loaders from that order) and still the same name. Its
 * header identity then keeps what else the entry held.
 *
 * Poses can name identities too (`detection-labels.js`), matched by name wherever the
 * name is, so a name they brought into the header is that entry's as well. Without
 * that a closed session's group would write a second entry under the same name.
 *
 * @param {{identities: Array}} kept
 * @param {Array} identities - `session.identities`
 * @param {Map<number, Object>} headerOf - identity id -> the header's identity
 * @param {Map<string, Object>|null} [namedByPoses] - name -> the header's identity
 * @returns {Map<Object, Object>} catalog entry -> the header's identity
 */
export function identityEntriesOfOpenSession(kept, identities, headerOf, namedByPoses) {
    var out = new Map();
    (identities || []).forEach(function (identity) {
        var entry = kept.identities[identity.id];
        var header = headerOf.get(identity.id);
        if (!entry || !header || entry.name !== identity.name) return;
        out.set(entry, header);
        if (!header.metadata || Object.keys(header.metadata).length === 0) header.metadata = entry.metadata;
    });
    if (namedByPoses) {
        kept.identities.forEach(function (entry) {
            if (!out.has(entry) && namedByPoses.has(entry.name)) out.set(entry, namedByPoses.get(entry.name));
        });
    }
    return out;
}

/**
 * At save: the closed sessions as the writer takes them. Must run before the writer
 * opens: it adds the videos and identities they need to the lists the header is built
 * from.
 *
 * A kept group's identity stays the catalog entry it was. When the open session lists
 * that same entry (same place in the file's catalog, same name), both sessions' groups
 * write one index, which is what says they are one animal. Renaming it in the open
 * session splits it: the closed session keeps the name it had.
 *
 * @param {{sessions: Array, identities: Array}} kept - `keepUnopenedSessions`
 * @param {Object} opts
 * @param {Object} opts.SIO - window.SleapIO
 * @param {(id: number, video: Object) => Object} opts.headerVideo - the header's video
 *   for a store video id, adding one when the header has none
 * @param {(row: number) => (number|undefined)} opts.outputRow - where a store row lands
 *   in this save; undefined when it is not written as itself
 * @param {(row: number, inst: number) => (number[]|null)} [opts.movedRef] - for a row
 *   `outputRow` does not have: `[lf, inst]` when a corrected frame replaced it, null
 *   when the member has nothing left to point at (it is dropped; the group stays)
 * @param {Map<Object, Object>} [opts.identityOfEntry] - catalog entry -> the header
 *   identity the open session wrote for it
 * @param {Array} opts.identities - the header's identity list (appended to)
 * @param {Object} [opts.firstLucid] - the open session's `metadata.lucid` as this save
 *   writes it: where a kept copy of the project's planes comes from
 * @returns {Array} `RecordingSession`s for `openSlpWriter`, in file order
 */
export function unopenedSessionsForSave(kept, opts) {
    var SIO = opts.SIO;
    var entries = kept.identities || [];
    var entryOf = opts.identityOfEntry || new Map();
    var headerIdentity = function (entry) { return entryOf.get(entry) || entry; };
    catalogEntriesUsed(kept).forEach(function (index) {
        var header = headerIdentity(entries[index]);
        if (header && opts.identities.indexOf(header) < 0) opts.identities.push(header);
    });
    var identityOf = function (index) { return headerIdentity(entries[index]); };

    return kept.sessions.map(function (record) {
        var nM = record.memberRow.length;
        var lf = new Float64Array(nM);
        var inst = new Int32Array(nM);
        for (var m = 0; m < nM; m++) {
            var row = record.memberRow[m], k = record.memberInst[m];
            var out = k < 0 ? undefined : opts.outputRow(row);
            if (out !== undefined) {
                lf[m] = out;
                inst[m] = k;
                continue;
            }
            var moved = k >= 0 && opts.movedRef ? opts.movedRef(row, k) : null;
            lf[m] = moved ? moved[0] : -1;
            inst[m] = moved ? moved[1] : -1;
        }

        var session = new SIO.RecordingSession({
            cameraGroup: new SIO.CameraGroup({ cameras: record.cameras, metadata: record.cameraGroupMetadata }),
            frameGroupByFrameIdx: {
                values: function () { return frameGroupsOf(SIO, record, lf, inst, identityOf); },
            },
        });
        record.videos.forEach(function (v) {
            session.addVideo(opts.headerVideo(v.id, v.video), record.cameras[v.camera]);
        });
        if (record.fileDict) session.lucidFileDictJson = fileDictForSave(record, opts.firstLucid);
        return session;
    });
}

/**
 * This save's text of the session's own dict. Its copy of the project's planes follows
 * the open session's, because every session of a LUC3D project holds the same and an
 * eager open reads the first it finds: a plane deleted in the open session must not
 * come back from a closed one's copy. Its copy of the skeleton follows after an eager
 * read, because an eager open builds that session from it and a node edit re-shaped its
 * poses.
 */
function fileDictForSave(record, firstLucid) {
    var text = decoder.decode(record.fileDict);
    if ((!record.planesFollowFirst && !record.skeletonFollowsFirst) || !firstLucid) return text;
    var dict = JSON.parse(text);
    var lucid = dict.metadata.lucid;
    if (record.planesFollowFirst) {
        PROJECT_PLANE_KEYS.forEach(function (key) {
            if (firstLucid[key] !== undefined) lucid[key] = firstLucid[key];
            else delete lucid[key];
        });
    }
    if (record.skeletonFollowsFirst) lucid.skeleton = firstLucid.skeleton;
    return JSON.stringify(dict);
}

/**
 * The open session deleted store instance rows: follow them in every kept member on a
 * video it shares with a closed session. Runs before the store's frame ranges change.
 *
 * @param {{sessions: Array}} kept
 * @param {Object} framesData - the store's frame columns, ranges still in the old numbering
 * @param {Uint8Array} kill - per old instance row: 1 = deleted
 * @param {Uint32Array} survBefore - new index of each surviving old instance row
 */
export function renumberUnopenedMembers(kept, framesData, kill, survBefore) {
    kept.sessions.forEach(function (record) {
        var list = record.membersOnOpenVideos;
        for (var i = 0; i < list.length; i++) {
            var m = list[i];
            if (record.memberInst[m] < 0) continue;
            var start = Number(framesData.instance_id_start[record.memberRow[m]]);
            var old = start + record.memberInst[m];
            record.memberInst[m] = kill[old] ? -1 : survBefore[old] - survBefore[start];
        }
    });
}

/**
 * Every store video id the kept sessions name, with frames or not.
 * @param {{sessions: Array}} kept
 * @returns {number[]}
 */
export function unopenedVideoIds(kept) {
    var ids = [];
    kept.sessions.forEach(function (record) {
        record.videos.forEach(function (v) { ids.push(v.id); });
    });
    return ids;
}

/**
 * The sentence a load's final status adds for a file with more sessions.
 * @param {number} fileSessions - sessions in the file
 * @returns {string}
 */
export function unopenedSessionsNote(fileSessions) {
    return 'Only the first of ' + fileSessions + ' sessions in this file is open: saving keeps the others as they are.';
}

/** Does any kept session have a member on a video the open session has too? */
export function hasMembersOnOpenVideos(kept) {
    return !!kept && kept.sessions.some(function (record) { return record.membersOnOpenVideos.length > 0; });
}

/**
 * `sessions` in save order with the closed sessions each one carries: what the file's
 * provenance is keyed by (`buildSaveProvenance`). Each entry has `name` and `fileProvenance`.
 * @param {Array} sessions
 * @returns {Array}
 */
export function withUnopenedSessions(sessions) {
    var out = [];
    (sessions || []).forEach(function (session) {
        out.push(session);
        if (session && session.unopenedSessions) out.push.apply(out, session.unopenedSessions.sessions);
    });
    return out;
}
