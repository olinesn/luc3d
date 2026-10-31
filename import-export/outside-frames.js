/**
 * outside-frames.js — poses on a video no recording session names.
 *
 * A `.slp` can hold frames on a video that no session's camera map lists: a SLEAP
 * project with extra videos, or, when a loader opens only the first session of a
 * multi-session file, the other sessions' videos. LUC3D shows a session's cameras,
 * so these frames are not shown. They are still the user's data, so a save keeps
 * them, each on its own video. It never drops them, never moves them onto a camera's
 * video, and never leaves their rows pointing at a video the file does not list
 * (Python's `load_slp` raises IndexError on that).
 *
 * The lazy path keeps them where they already are, in its columnar store:
 * `buildSessionRefGraph` gives each such video a place in the header, after the
 * session's cameras, and `SioLazyLoader.remapTracksFromIdentity` keeps their track
 * names when the session's track list changes. File ▸ Add SLP puts its own there too
 * (`SioLazyLoader.commitAddedRows`, loading/lazy-store-rows.js).
 *
 * The eager path has no store, so this module holds them for it:
 *  - `keepOutsideFrames` at load, on one session (`session.outsideFrames`);
 *  - `outsideFramesForSave` at save, as sleap-io.js objects for `buildSlpBytes`.
 * A kept pose is an ordinary `Instance`, so the file gets the values an unlinked
 * instance would write. A skeleton node edit re-shapes it with the rest of the
 * project (`Session.propagateNodeAdded`/`propagateNodeRemoved`). Track edits do not
 * touch it: it holds the file's track names, not the session's indices.
 *
 * Nothing here is written to `metadata.lucid`, and a file with no such frames saves
 * byte-identically (`tests/e2e/save-golden-digest.mjs`).
 */

import { Instance } from '../pose/pose-data.js';
import { makePooledSioInstances } from './file-io.js';

function newHeaderVideo(SIO, filename, shape) {
    var videoShape = shape || [0, 0, 0, 1];
    var video = new SIO.Video({
        filename: filename,
        backendMetadata: { type: 'MediaVideo', shape: videoShape, filename: filename },
        openBackend: false,
    });
    video.shape = videoShape;
    return video;
}

/**
 * The video a header lists for frames outside a session. The file's own record when
 * it has a usable one, so the filename, shape and everything else LUC3D does not
 * read survive. A save does not copy embedded images, so such a video is replaced by
 * its source video. A video the file does not list at all (`id` past the end of
 * `videos_json`, as older LUC3D saves left it) gets a placeholder, so the poses stay
 * apart from every other video's and Python can open the file.
 *
 * @param {Object} SIO - window.SleapIO
 * @param {Object|null|undefined} video - the reader's `SIO.Video` for this id
 * @param {number} id - the video's id in the file's `frames` table
 * @returns {Object} a `SIO.Video`
 */
export function headerVideoForOutsideFrames(SIO, video, id) {
    if (video && !video.hasEmbeddedImages) return video;
    if (video && video.sourceVideo) return video.sourceVideo;
    return newHeaderVideo(SIO, 'unknown_video_' + id + '.mp4', video && video.shape);
}

/**
 * `headerVideoForOutsideFrames` for a `slpData.videos` record. Only the raw worker's
 * records have no typed video to keep: they get a new one under the file's name.
 *
 * @param {Object} SIO - window.SleapIO
 * @param {Object|null|undefined} record
 * @param {number} id - the video's id in the file's `frames` table
 * @returns {Object} a `SIO.Video`
 */
export function headerVideoOfRecord(SIO, record, id) {
    if (record && !record._typedVideo && record.filename && !record.embedded) {
        return newHeaderVideo(SIO, record.filename, record.shape);
    }
    return headerVideoForOutsideFrames(SIO, record && record._typedVideo, id);
}

/**
 * Eager load: put the frames on videos no opened session reads on `session`, after
 * any it already holds (File ▸ Add SLP can bring more).
 *
 * @param {Session} session - the project's first session
 * @param {Object} slpData - `parseSlpViaSleapIO` / `parseSlpH5` result
 * @param {Set<number>} claimedVideos - video indices some opened session reads
 * @param {number[]|null} [nodeReorderMap] - reorderMap[fileNodeIdx] = sessionNodeIdx,
 *   when the file's skeleton lists the session's nodes in another order
 */
export function keepOutsideFrames(session, slpData, claimedVideos, nodeReorderMap) {
    function inSessionOrder(values, blank) {
        if (!values || !nodeReorderMap) return values;
        var out = new Array(nodeReorderMap.length).fill(blank);
        for (var ri = 0; ri < nodeReorderMap.length && ri < values.length; ri++) out[nodeReorderMap[ri]] = values[ri];
        return out;
    }
    var kept = session.outsideFrames;
    var slotOfVideo = new Map();
    // A kept pose's `trackIdx` indexes `kept.tracks`, which holds only the names
    // kept poses use: a track deleted from the session does not come back on save
    // unless one of them is on it.
    function keptTrackSlot(trackIdx) {
        // `null >= 0` is true in JS.
        var name = trackIdx != null && trackIdx >= 0 ? (slpData.tracks || [])[trackIdx] : undefined;
        if (name === undefined) return -1;
        var trackSlot = kept.tracks.indexOf(name);
        if (trackSlot < 0) { trackSlot = kept.tracks.length; kept.tracks.push(name); }
        return trackSlot;
    }
    var frames = slpData.frames || [];
    for (var fi = 0; fi < frames.length; fi++) {
        var fd = frames[fi];
        if (claimedVideos.has(fd.videoIdx)) continue;
        if (!kept) kept = session.outsideFrames = { videos: [], tracks: [], frames: [] };
        var slot = slotOfVideo.get(fd.videoIdx);
        if (slot === undefined) {
            slot = kept.videos.length;
            slotOfVideo.set(fd.videoIdx, slot);
            var record = (slpData.videos || [])[fd.videoIdx] || null;
            kept.videos.push({
                id: fd.videoIdx,
                typed: record && record._typedVideo ? record._typedVideo : null,
                // Only the raw worker's records lack `_typedVideo`.
                filename: record && !record.embedded ? record.filename : null,
                shape: record ? record.shape : null,
            });
        }
        var instances = [];
        for (var ii = 0; ii < fd.instances.length; ii++) {
            var d = fd.instances[ii];
            var inst = new Instance(inSessionOrder(d.points, null), keptTrackSlot(d.trackIdx), d.type || 'user', d.score || 0);
            if (d.occluded) inst.setOccludedFrom(inSessionOrder(d.occluded, false));
            if (d.pointScores) inst.setPointScores(inSessionOrder(d.pointScores, NaN));
            instances.push(inst);
        }
        kept.frames.push({ video: slot, frameIdx: fd.frameIdx, instances: instances });
    }
}

/**
 * Per-camera eager load: a camera is one video, and its file's frames on any other
 * video are kept aside. The camera's own video is the first that file lists with
 * frames: the rule `SioLazyLoader.open` applies, so a folder reads the same on
 * either path.
 *
 * @param {Object} col - the worker's columnar poses (`buildColumnarFrames`)
 * @returns {number|undefined} the camera's video id; undefined when `col` has no frames
 */
export function ownVideoOfColumnar(col) {
    var own;
    for (var f = 0; f < col.nFrames; f++) {
        var v = col.videoIdx ? col.videoIdx[f] : 0;
        if (own === undefined || v < own) own = v;
    }
    return own;
}

/**
 * `keepOutsideFrames` for a camera file's columnar poses: every frame not on `own`.
 *
 * @param {Session} session
 * @param {Object} slpData - the worker's result, `columnar` set
 * @param {number} own - `ownVideoOfColumnar`
 */
export function keepOutsideColumnarFrames(session, slpData, own) {
    var col = slpData.columnar;
    var nn = col.numNodes;
    var frames = [];
    for (var f = 0; f < col.nFrames; f++) {
        if (!col.videoIdx || col.videoIdx[f] === own) continue;
        var instances = [];
        for (var i = col.instOffsets[f]; i < col.instOffsets[f + 1]; i++) {
            var points = new Array(nn), occluded = new Array(nn);
            for (var k = 0; k < nn; k++) {
                var x = col.xy[(i * nn + k) * 2], y = col.xy[(i * nn + k) * 2 + 1];
                points[k] = x === x && y === y ? [x, y] : null;
                occluded[k] = !!col.occluded[i * nn + k];
            }
            var predicted = col.type[i] === 1;
            instances.push({
                points: points, occluded: occluded, trackIdx: col.trackIdx[i], score: col.score[i],
                type: predicted ? 'predicted' : 'user',
                pointScores: predicted && col.pointScore ? col.pointScore.subarray(i * nn, (i + 1) * nn) : null,
            });
        }
        frames.push({ frameIdx: col.frameIdx[f], videoIdx: col.videoIdx[f], instances: instances });
    }
    if (frames.length > 0) {
        keepOutsideFrames(session, { frames: frames, tracks: slpData.tracks, videos: slpData.videos }, new Set([own]));
    }
}

/**
 * Eager save: `session`'s kept frames as sleap-io.js objects.
 *
 * @param {Session} session
 * @param {Object} skeleton - the `SIO.Skeleton` of this session's Labels
 * @returns {{videos: Array, tracks: Array, labeledFrames: Array}|null}
 */
export function outsideFramesForSave(session, skeleton) {
    var SIO = window.SleapIO;
    // A lazy session on this path (a project mixing lazy and eager sessions): its
    // store holds them, and the eager writer would otherwise leave them behind.
    var loader = session.lazyLoader;
    if (loader && typeof loader.forEachOutsideRow === 'function') return lazyOutsideFramesForSave(SIO, loader);
    var kept = session.outsideFrames;
    if (!kept) return null;
    var videos = kept.videos.map(function (v) {
        return !v.typed && v.filename ? newHeaderVideo(SIO, v.filename, v.shape)
            : headerVideoForOutsideFrames(SIO, v.typed, v.id);
    });
    var tracks = kept.tracks.map(function (name) { return new SIO.Track(name); });
    var newSioInstance = makePooledSioInstances(SIO, skeleton, session.skeleton.nodes.length);
    var labeledFrames = kept.frames.map(function (f) {
        return new SIO.LabeledFrame({
            video: videos[f.video],
            frameIdx: f.frameIdx,
            instances: f.instances.map(function (inst) {
                return newSioInstance(inst, inst.trackIdx >= 0 ? tracks[inst.trackIdx] : null);
            }),
        });
    });
    return { videos: videos, tracks: tracks, labeledFrames: labeledFrames };
}

function lazyOutsideFramesForSave(SIO, loader) {
    var headerVideoOf = new Map();
    var trackByName = new Map();
    var labeledFrames = [];
    loader.forEachOutsideRow(function (store, row, id) {
        var lf = store.materializeFrame(row);
        if (!lf) return;
        if (!headerVideoOf.has(lf.video)) headerVideoOf.set(lf.video, headerVideoForOutsideFrames(SIO, lf.video, id));
        lf.instances.forEach(function (inst) {
            if (inst.track && !trackByName.has(inst.track.name)) trackByName.set(inst.track.name, inst.track);
        });
        labeledFrames.push(new SIO.LabeledFrame({ video: headerVideoOf.get(lf.video), frameIdx: lf.frameIdx, instances: lf.instances }));
    });
    if (labeledFrames.length === 0) return null;
    return { videos: Array.from(headerVideoOf.values()), tracks: Array.from(trackByName.values()), labeledFrames: labeledFrames };
}

/**
 * The sentence a load's final status adds: what the file holds that is not on
 * screen, and what a save does with it. Empty when there is nothing to say.
 *
 * @param {{frames: number, videos: number}} counts - frames and videos outside the
 *   opened sessions
 * @param {number} [fileSessions] - sessions in the file, for a loader that opens only
 *   the first
 * @returns {string}
 */
export function outsideFramesNote(counts, fileSessions) {
    if (fileSessions > 1) {
        return 'Only the first of ' + fileSessions + ' sessions in this file is open: saving keeps the ' +
            'other sessions\' poses, but not their cameras or 3D.';
    }
    if (!counts.frames) return '';
    return counts.frames.toLocaleString('en-US') + ' labeled ' + (counts.frames === 1 ? 'frame' : 'frames') +
        ' on ' + counts.videos + ' ' + (counts.videos === 1 ? 'video' : 'videos') +
        ' in no session ' + (counts.frames === 1 ? 'is' : 'are') + ' not shown; saving keeps ' +
        (counts.frames === 1 ? 'it' : 'them') + '.';
}

/** `session.outsideFrames` as `outsideFramesNote` counts it. */
export function keptOutsideCounts(session) {
    var kept = session && session.outsideFrames;
    return { frames: kept ? kept.frames.length : 0, videos: kept ? kept.videos.length : 0 };
}
