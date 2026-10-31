/**
 * outside-frames.js — poses a `.slp` holds that LUC3D does not show.
 *
 * Two kinds, both still the user's data, so a save keeps them:
 *
 * 1. On a video no recording session names: a SLEAP project with extra videos, or,
 *    when a loader opens only the first session of a multi-session file, the other
 *    sessions' videos. LUC3D shows a session's cameras. Each stays on its own video:
 *    never dropped, never moved onto a camera's video, and never left pointing at a
 *    video the file does not list (Python's `load_slp` raises IndexError on that).
 *
 * 2. On a skeleton the session does not show, on any video. LUC3D shows one skeleton
 *    per session and `Instance` stores 2D by node index, so reading such a pose would
 *    put its values under the session's node names and drop its extra nodes. Each
 *    stays on its own skeleton, with its own nodes. One on a camera's own video stays
 *    on that video, in the camera's frame. On the eager path a group linking such a
 *    pose cannot be held, so that load refuses (`groupsAcrossSkeletonsRefusal`).
 *
 * The lazy path keeps both where they already are, in its columnar store:
 * `buildSessionRefGraph` gives each such video and skeleton a place in the header,
 * and `SioLazyLoader.remapTracksFromIdentity` keeps their track names when the
 * session's track list changes. Kind 2 is flagged row by row
 * (loading/poses-not-shown.js), so nothing reads it by position and an edit beside it
 * does not touch it. A group linking such a pose stays whole on
 * `session.outsideGroups`, with its poses flagged too (`keepGroupAside`), so a lazy
 * load has nothing to refuse.
 *
 * The eager path has no store, so this module holds them for it, on
 * `session.outsideFrames`:
 *  - `frames` (kind 1, on the session's skeleton): `keepOutsideFrames`. Ordinary
 *    `Instance`s, so the file gets the values an unlinked instance would write, tracker
 *    score and model identity included (`detection-labels.js`). A skeleton node edit
 *    re-shapes them with the rest of the project
 *    (`Session.propagateNodeAdded`/`propagateNodeRemoved`). Track edits do not touch
 *    them: they hold the file's track names, not the session's indices.
 *  - `otherSkeletonFrames` (kind 2): `keepOtherSkeletonFrames`. The reader's own
 *    sleap-io.js instances, written back as they are, so nothing LUC3D does not read
 *    is lost. Node edits and track edits do not touch them.
 * `outsideFramesForSave` hands both to `buildSlpBytes`.
 *
 * Nothing here is written to `metadata.lucid`, and a file with no such frames saves
 * byte-identically (`tests/e2e/save-golden-digest.mjs`).
 */

import { Instance, pooledPoints3d } from '../pose/pose-data.js';
import { makePooledSioInstances } from './file-io.js';
import { shownSkeletonEntry } from '../loading/poses-not-shown.js';
import { identityCatalogFinder, writeFileLabels } from './detection-labels.js';

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
 * Every skeleton of a file, with the poses on it; `[0]` is `slpData.skeleton` and its
 * `frames` are `slpData.frames`. A file with one skeleton lists none (the raw worker,
 * an analysis `.h5`), so this fills that in.
 *
 * @param {Object} slpData
 * @returns {Array<{name: string, nodes: string[], edges: number[][], frames: Array}>}
 */
export function fileSkeletons(slpData) {
    if (!slpData.skeletons) {
        slpData.skeletons = [{
            name: slpData.skeleton ? slpData.skeleton.name : 'skeleton',
            nodes: slpData.skeleton ? slpData.skeleton.nodes : [],
            edges: slpData.skeleton ? slpData.skeleton.edges : [],
            frames: slpData.frames || [],
        }];
    }
    return slpData.skeletons;
}

/**
 * The videos a file's session reads: its camera map's, else one per camera in order
 * (`videosClaimedBy`), else all of them for a file with no session.
 */
function videosOfFileSession(slpData, sessionIdx) {
    var sess = slpData.sessions && slpData.sessions[sessionIdx];
    var mapped = sess ? Object.values(sess.camcorder_to_video_idx_map || {}) : [];
    if (mapped.length > 0) return new Set(mapped);
    var nCameras = sess ? Object.keys(sess.calibration || {}).filter(function (k) { return k !== 'metadata'; }).length : 0;
    var n = nCameras > 0 ? nCameras : (slpData.videos || []).length;
    var all = new Set();
    for (var i = 0; i < n; i++) all.add(i);
    return all;
}

/**
 * The entry of `slpData.skeletons` a file's session shows: the one its LUC3D record
 * names (a project can hold sessions on different skeletons), else the first with a
 * pose on the session's videos, else the first.
 *
 * @param {Object} slpData
 * @param {number} sessionIdx - index into `slpData.sessions`; a file with none: 0
 * @returns {{name: string, nodes: string[], edges: number[][], frames: Array}}
 */
export function sessionSkeletonEntry(slpData, sessionIdx) {
    var sess = slpData.sessions && slpData.sessions[sessionIdx];
    var videos = null;
    return shownSkeletonEntry(fileSkeletons(slpData), sess && sess.metadata && sess.metadata.lucid && sess.metadata.lucid.skeleton, function (entry) {
        if (!videos) videos = videosOfFileSession(slpData, sessionIdx);
        return entry.frames.some(function (f) { return videos.has(f.videoIdx); });
    });
}

/**
 * Why a load must refuse, or null. A group linking a pose on a skeleton its session
 * does not show cannot be held: `Instance` stores 2D by the session's node index, and
 * leaving the pose out would change the group on the next save. Run this before the
 * load touches anything.
 *
 * Without a typed session (the raw reader, a legacy array entry) groups point at a
 * pose by its position in the file's frame. Those positions no longer hold once the
 * other skeleton's poses are set aside, so any such group refuses too.
 *
 * A refusing loader resolves `false`, as for any file that did not open: the menu
 * records the save target after that, and Save would write the open project over it.
 *
 * @param {Object} slpData
 * @param {number} nSessions - how many of the file's sessions this loader opens
 * @returns {string|null} the sentence for the status line
 */
export function groupsAcrossSkeletonsRefusal(slpData, nSessions) {
    if (fileSkeletons(slpData).length <= 1) return null;
    var groups = 0;
    for (var si = 0; si < nSessions; si++) {
        var sess = slpData.sessions && slpData.sessions[si];
        if (!sess) continue;
        var own = sessionSkeletonEntry(slpData, si);
        if (!sess._typedSession) {
            var dicts = sess.frame_group_dicts || [];
            for (var di = 0; di < dicts.length; di++) groups += (dicts[di].instance_groups || []).length;
            continue;
        }
        for (var frameGroup of sess._typedSession.frameGroups.values()) {
            for (var igi = 0; igi < frameGroup.instanceGroups.length; igi++) {
                for (var member of frameGroup.instanceGroups[igi].instanceByCamera) {
                    var skeletonEntry = member[1] && slpData.skeletonEntryOf.get(member[1].skeleton);
                    if (skeletonEntry && skeletonEntry !== own) { groups++; break; }
                }
            }
        }
    }
    if (groups === 0) return null;
    return 'Not opened: ' + groups.toLocaleString('en-US') + (groups === 1 ? ' group' : ' groups') +
        ' in this file ' + (groups === 1 ? 'links' : 'link') + ' poses on different skeletons, and LUC3D shows one skeleton per session.';
}

/**
 * Lazy load: does this group of the file link a pose the session does not show?
 *
 * @param {Object} typedIG - sleap-io.js `InstanceGroup`, its refs as read
 * @param {SioLazyLoader} loader
 * @returns {boolean}
 */
export function groupLinksPoseNotShown(typedIG, loader) {
    var refs = typedIG._instanceRefsByCamera;
    if (!refs || !loader || typeof loader.isPoseNotShown !== 'function') return false;
    for (var entry of refs) {
        if (loader.isPoseNotShown(entry[1][0], entry[1][1])) return true;
    }
    return false;
}

/**
 * Lazy load: set one of the file's groups aside whole, on `session.outsideGroups`.
 * `Instance` stores 2D by the session's node index, so its member on another skeleton
 * cannot be shown, and leaving that member out would change the group on the next save.
 * So none of its poses is shown or edited, on any skeleton, and the save writes the
 * group back as it was, its 3D included.
 *
 * Each member is held by its store frame row and its position in that frame, the two
 * numbers a file's group refers to it by; a delete keeps the position right
 * (`shiftOutsideGroupMembers`).
 *
 * @param {Session} session
 * @param {number} frameIdx
 * @param {Object} typedIG - sleap-io.js `InstanceGroup`, its refs as read
 * @param {SioLazyLoader} loader
 */
export function keepGroupAside(session, frameIdx, typedIG, loader) {
    var members = new Map();
    for (var entry of typedIG._instanceRefsByCamera) {
        var row = entry[1][0], offset = entry[1][1];
        members.set(entry[0].name, { row: row, offset: offset });
        loader.setPoseNotShown(row, offset);
    }
    var instance3d = typedIG.instance3d;
    // Flat, and in the slab pool, like a shown group's: one ArrayBuffer per group is what full GCs sweep.
    if (instance3d && instance3d.points) instance3d.points = pooledPoints3d(instance3d.points);
    if (!session.outsideGroups) session.outsideGroups = new Map();
    if (!session.outsideGroups.has(frameIdx)) session.outsideGroups.set(frameIdx, []);
    session.outsideGroups.get(frameIdx).push({
        members: members, instance3d: instance3d, identity: typedIG.identity,
        score: typedIG.score, metadata: typedIG.metadata,
    });
}

/** How many groups `session.outsideGroups` holds. */
export function outsideGroupCount(session) {
    var n = 0;
    if (session && session.outsideGroups) {
        for (var groups of session.outsideGroups.values()) n += groups.length;
    }
    return n;
}

function emptyOutsideFrames() {
    return { videos: [], tracks: [], frames: [], otherSkeletonFrames: [] };
}

// One place per video of one file, whichever call asks, because a video listed twice
// is two videos: a file's `videos` list -> `session.outsideFrames` -> (index -> place).
var keptVideoPlaces = new WeakMap();

/**
 * This video's place in `kept.videos`.
 *
 * @param {Object} kept - `session.outsideFrames`
 * @param {Object} slpData
 * @param {number} videoIdx
 * @param {string|null} camera - the camera whose video this is; null when it has none
 */
function keptVideoSlot(kept, slpData, videoIdx, camera) {
    var list = slpData.videos || [];
    if (!keptVideoPlaces.has(list)) keptVideoPlaces.set(list, new WeakMap());
    var ofList = keptVideoPlaces.get(list);
    if (!ofList.has(kept)) ofList.set(kept, new Map());
    var places = ofList.get(kept);
    if (!places.has(videoIdx)) {
        var record = list[videoIdx] || null;
        places.set(videoIdx, kept.videos.length);
        kept.videos.push({
            id: videoIdx,
            typed: record && record._typedVideo ? record._typedVideo : null,
            // Only the raw worker's records lack `_typedVideo`.
            filename: record && !record.embedded ? record.filename : null,
            shape: record ? record.shape : null,
            camera: camera,
        });
    }
    return places.get(videoIdx);
}

/**
 * Eager load: put the frames on videos no opened session reads on `session`, after
 * any it already holds (File ▸ Add SLP can bring more).
 *
 * @param {Session} session - the project's first session
 * @param {Object} slpData - `parseSlpViaSleapIO` / `parseSlpH5` result; only its
 *   `frames` are read, so they must be on `session`'s skeleton
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
        if (!kept) kept = session.outsideFrames = emptyOutsideFrames();
        var slot = keptVideoSlot(kept, slpData, fd.videoIdx, null);
        var instances = [];
        for (var ii = 0; ii < fd.instances.length; ii++) {
            var d = fd.instances[ii];
            var inst = new Instance(inSessionOrder(d.points, null), keptTrackSlot(d.trackIdx), d.type || 'user', d.score || 0);
            if (d.occluded) inst.setOccludedFrom(inSessionOrder(d.occluded, false));
            if (d.pointScores) inst.setPointScores(inSessionOrder(d.pointScores, NaN));
            if (d.trackingScore) inst.trackingScore = d.trackingScore;
            if (d.fileLabels) inst.fileLabels = d.fileLabels;
            instances.push(inst);
        }
        kept.frames.push({ video: slot, frameIdx: fd.frameIdx, instances: instances });
    }
}

/**
 * The sleap-io.js skeleton of a `slpData.skeletons` entry. The typed reader's own; the
 * raw worker's record is built into one once, so its poses share it.
 */
function sioSkeletonOf(SIO, entry) {
    if (!entry.typed) {
        entry.typed = new SIO.Skeleton({
            nodes: entry.nodes, edges: entry.edges || [], symmetries: entry.symmetries || [], name: entry.name,
        });
    }
    return entry.typed;
}

/** A pose from the raw worker as the sleap-io.js instance the typed reader would make. */
function sioInstanceFromRow(SIO, d, skeleton, track) {
    var predicted = d.type === 'predicted';
    var names = skeleton.nodeNames;
    var points = d.points.map(function (p, k) {
        var point = {
            xy: p ? [p[0], p[1]] : [NaN, NaN],
            visible: !!p && !(d.occluded && d.occluded[k]),
            complete: d.complete ? !!d.complete[k] : !!p,
            name: names[k],
        };
        if (predicted) point.score = d.pointScores ? d.pointScores[k] : NaN;
        return point;
    });
    var options = { points: points, skeleton: skeleton, track: track, trackingScore: d.trackingScore || 0 };
    if (predicted) options.score = d.score || 0;
    return predicted ? new SIO.PredictedInstance(options) : new SIO.Instance(options);
}

/**
 * Note who reads these videos, for `keepOtherSkeletonFrames`. The first session to
 * read a video has its poses, as it does for the poses it shows.
 *
 * @param {Map} readers - video index -> { session, entry, camera }
 * @param {Session} session
 * @param {Object} entry - the `fileSkeletons` entry `session` shows
 * @param {Set<number>} claimedVideos - the video indices `session` reads
 * @param {Object<string, string>} videoIdxToCameraName
 * @param {Camera[]} cameras - by video index, when the file names no video for them
 */
export function noteVideoReaders(readers, session, entry, claimedVideos, videoIdxToCameraName, cameras) {
    claimedVideos.forEach(function (v) {
        if (readers.has(v)) return;
        readers.set(v, {
            session: session, entry: entry,
            camera: videoIdxToCameraName[v] || (cameras[v] ? cameras[v].name : 'cam_' + v),
        });
    });
}

/**
 * Eager load: set aside the poses on a skeleton their session does not show. A pose
 * on a camera's video goes on the session reading that video, with the camera's name;
 * one on a video no opened session reads goes on `fallback`, unless it is on
 * `fallbackEntry`, whose poses `keepOutsideFrames` holds.
 *
 * @param {Object} slpData - `parseSlpViaSleapIO` / `parseSlpH5` result
 * @param {Map} readers - `noteVideoReaders`
 * @param {Session} fallback
 * @param {Object} fallbackEntry - the `fileSkeletons` entry `fallback` shows
 */
export function keepOtherSkeletonFrames(slpData, readers, fallback, fallbackEntry) {
    var entries = fileSkeletons(slpData);
    if (entries.length <= 1) return;
    var SIO = window.SleapIO;
    var trackByName = new Map();
    function trackOf(trackIdx) {
        // `null >= 0` is true in JS.
        var name = trackIdx != null && trackIdx >= 0 ? (slpData.tracks || [])[trackIdx] : undefined;
        if (name === undefined) return null;
        if (!trackByName.has(name)) trackByName.set(name, new SIO.Track(name));
        return trackByName.get(name);
    }
    var builtByRow = new Map();     // file row -> instance, raw worker's poses only
    var fromPredicted = [];
    entries.forEach(function (entry) {
        entry.frames.forEach(function (fd) {
            var reader = readers.get(fd.videoIdx) || null;
            if (entry === (reader ? reader.entry : fallbackEntry)) return;
            var session = reader ? reader.session : fallback;
            var kept = session.outsideFrames || (session.outsideFrames = emptyOutsideFrames());
            kept.otherSkeletonFrames.push({
                video: keptVideoSlot(kept, slpData, fd.videoIdx, reader ? reader.camera : null),
                frameIdx: fd.frameIdx,
                instances: fd.instances.map(function (d) {
                    if (d._typed) return d._typed;
                    var inst = sioInstanceFromRow(SIO, d, sioSkeletonOf(SIO, entry), trackOf(d.trackIdx));
                    if (d.row !== undefined) builtByRow.set(d.row, inst);
                    if (d.fromPredicted >= 0) fromPredicted.push([inst, d.fromPredicted]);
                    return inst;
                }),
            });
        });
    });
    fromPredicted.forEach(function (link) {
        var predicted = builtByRow.get(link[1]);
        if (predicted && predicted instanceof SIO.PredictedInstance && !(link[0] instanceof SIO.PredictedInstance)) link[0].fromPredicted = predicted;
    });
}

/**
 * Per-camera eager load: a camera is one video, and its file's frames on any other
 * video are kept aside. The camera's own video is the first that file lists with
 * frames, on any skeleton: the rule `SioLazyLoader.open` applies, so a folder reads
 * the same on either path.
 *
 * @param {Object} col - the worker's columnar poses (`buildColumnarFrames`)
 * @param {Object} [slpData] - the worker's result, for the poses on its other skeletons
 * @returns {number|undefined} the camera's video id; undefined when it has no frames
 */
export function ownVideoOfColumnar(col, slpData) {
    var own;
    for (var f = 0; f < col.nFrames; f++) {
        var v = col.videoIdx ? col.videoIdx[f] : 0;
        if (own === undefined || v < own) own = v;
    }
    return lowestVideoOnAnySkeleton(own, slpData);
}

function lowestVideoOnAnySkeleton(own, slpData) {
    if (slpData) {
        fileSkeletons(slpData).forEach(function (entry) {
            entry.frames.forEach(function (fd) {
                if (fd.videoIdx !== undefined && (own === undefined || fd.videoIdx < own)) own = fd.videoIdx;
            });
        });
    }
    return own;
}

/**
 * `ownVideoOfColumnar` for a typed read's frames (a camera file whose detections carry
 * identities or categories: only the typed reader has those).
 *
 * @param {Array<{videoIdx: number}>} frames
 * @param {Object} [slpData] - that read's result, for the poses on its other skeletons
 * @returns {number|undefined}
 */
export function ownVideoOfFrames(frames, slpData) {
    var own;
    for (var f = 0; f < frames.length; f++) {
        var v = frames[f].videoIdx;
        if (v !== undefined && (own === undefined || v < own)) own = v;
    }
    return lowestVideoOnAnySkeleton(own, slpData);
}

/**
 * A camera file's poses on another skeleton: on its own video they stay with the
 * camera, on any other they are outside.
 *
 * @param {Session} session
 * @param {Object} slpData - that file's read
 * @param {number} own - `ownVideoOfColumnar` / `ownVideoOfFrames`
 * @param {string} camName - the camera this file is
 */
export function keepCameraOtherSkeletonFrames(session, slpData, own, camName) {
    var first = fileSkeletons(slpData)[0];
    keepOtherSkeletonFrames(slpData, new Map([[own, { session: session, entry: first, camera: camName }]]), session, first);
}

/**
 * `keepOutsideFrames` for a camera file's columnar poses: every frame not on `own`,
 * and every pose on another skeleton.
 *
 * @param {Session} session
 * @param {Object} slpData - the worker's result, `columnar` set
 * @param {number} own - `ownVideoOfColumnar`
 * @param {string} camName - the camera this file is
 */
export function keepOutsideColumnarFrames(session, slpData, own, camName) {
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
                trackingScore: col.trackingScore ? col.trackingScore[i] : 0,
            });
        }
        frames.push({ frameIdx: col.frameIdx[f], videoIdx: col.videoIdx[f], instances: instances });
    }
    if (frames.length > 0) {
        keepOutsideFrames(session, { frames: frames, tracks: slpData.tracks, videos: slpData.videos }, new Set([own]));
    }
    keepCameraOtherSkeletonFrames(session, slpData, own, camName);
}

/**
 * Eager save: `session`'s kept frames as sleap-io.js objects. There is one row per
 * video and frame, because a lazy reopen's save keeps only the first row of each: a
 * kept pose on a camera's video goes into the frame `labels` already has there, so
 * this adds to `labels.labeledFrames`. What the camera's video is called follows the
 * camera, like the session's own poses. A camera the session has since lost leaves
 * the pose on the file's own record of that video.
 *
 * `new SIO.Labels` lists a kept pose's skeleton after the session's, by itself.
 *
 * @param {Session} session
 * @param {Object} labels - this session's `buildSlpLabelsAllViews` result. An identity
 *   only kept poses name is added to its `identities`.
 * @returns {{videos: Array, tracks: Array, labeledFrames: Array}|null} only what
 *   `labels` does not list yet
 */
export function outsideFramesForSave(session, labels) {
    var SIO = window.SleapIO;
    var findIdentity = identityCatalogFinder(SIO, labels.identities);
    // A lazy session on this path (a project mixing lazy and eager sessions): its
    // store holds them, and the eager writer would otherwise leave them behind.
    var loader = session.lazyLoader;
    if (loader && typeof loader.forEachOutsideRow === 'function') return lazyOutsideFramesForSave(SIO, session, labels, findIdentity);
    var kept = session.outsideFrames;
    if (!kept) return null;
    var videos = [];
    var videoOfSlot = kept.videos.map(function (v) {
        var camIdx = v.camera != null ? session.cameras.findIndex(function (c) { return c.name === v.camera; }) : -1;
        if (camIdx >= 0 && labels.videos[camIdx]) return labels.videos[camIdx];
        var video = !v.typed && v.filename ? newHeaderVideo(SIO, v.filename, v.shape)
            : headerVideoForOutsideFrames(SIO, v.typed, v.id);
        videos.push(video);
        return video;
    });
    var tracks = kept.tracks.map(function (name) { return new SIO.Track(name); });
    var newSioInstance = makePooledSioInstances(SIO, labels.skeletons[0], session.skeleton.nodes.length);
    var labeledFrames = kept.frames.map(function (f) {
        return new SIO.LabeledFrame({
            video: videoOfSlot[f.video],
            frameIdx: f.frameIdx,
            instances: f.instances.map(function (inst) {
                var sioInst = newSioInstance(inst, inst.trackIdx >= 0 ? tracks[inst.trackIdx] : null);
                writeFileLabels(sioInst, inst, findIdentity);
                return sioInst;
            }),
        });
    });
    if (kept.otherSkeletonFrames.length > 0) {
        var frameAt = new Map();    // video -> (frameIdx -> LabeledFrame)
        function frameMap(video) {
            if (!frameAt.has(video)) frameAt.set(video, new Map());
            return frameAt.get(video);
        }
        labels.labeledFrames.concat(labeledFrames).forEach(function (lf) { frameMap(lf.video).set(lf.frameIdx, lf); });
        var trackNames = new Set(kept.tracks);
        kept.otherSkeletonFrames.forEach(function (f) {
            var video = videoOfSlot[f.video];
            var lf = frameMap(video).get(f.frameIdx);
            if (!lf) {
                lf = new SIO.LabeledFrame({ video: video, frameIdx: f.frameIdx, instances: [] });
                frameMap(video).set(f.frameIdx, lf);
                labeledFrames.push(lf);
            }
            f.instances.forEach(function (inst) {
                lf.instances.push(inst);
                if (inst.track && !trackNames.has(inst.track.name)) {
                    trackNames.add(inst.track.name);
                    tracks.push(inst.track);
                }
                // The writer resolves a pose's identity by object, in this save's list.
                if (inst.identity) inst.identity = findIdentity(inst.identity);
            });
        });
    }
    return { videos: videos, tracks: tracks, labeledFrames: labeledFrames };
}

function lazyOutsideFramesForSave(SIO, session, labels, findIdentity) {
    var loader = session.lazyLoader;
    var headerVideoOf = new Map();
    var trackByName = new Map();
    var labeledFrames = [];
    function noteTrack(inst) {
        if (inst.track && !trackByName.has(inst.track.name)) trackByName.set(inst.track.name, inst.track);
    }
    loader.forEachOutsideRow(function (store, row, id) {
        var lf = store.materializeFrame(row);
        if (!lf) return;
        if (!headerVideoOf.has(lf.video)) headerVideoOf.set(lf.video, headerVideoForOutsideFrames(SIO, lf.video, id));
        lf.instances.forEach(function (inst) {
            noteTrack(inst);
            if (inst.identity) inst.identity = findIdentity(inst.identity);
        });
        labeledFrames.push(new SIO.LabeledFrame({ video: headerVideoOf.get(lf.video), frameIdx: lf.frameIdx, instances: lf.instances }));
    });
    // This writer has only the resident frames of a lazy camera: the poses the session
    // does not show come from the store, into the camera's frame (one row per video
    // and frame), each on its own skeleton. A member of a group kept aside on the
    // session's skeleton goes on `labels`' copy of it.
    var frameAt = new Map();    // video -> (frameIdx -> LabeledFrame)
    function frameFor(video, frameIdx) {
        if (!frameAt.has(video)) frameAt.set(video, new Map());
        var lf = frameAt.get(video).get(frameIdx);
        if (!lf) {
            lf = new SIO.LabeledFrame({ video: video, frameIdx: frameIdx, instances: [] });
            frameAt.get(video).set(frameIdx, lf);
            labeledFrames.push(lf);
        }
        return lf;
    }
    labels.labeledFrames.forEach(function (lf) {
        if (!frameAt.has(lf.video)) frameAt.set(lf.video, new Map());
        frameAt.get(lf.video).set(lf.frameIdx, lf);
    });
    var placed = new Map();     // "camera:row:offset" -> { inst, lf }
    session.cameras.forEach(function (cam, camIdx) {
        var camLabels = loader.labelsByCam.get(cam.name);
        var rowMap = loader.frameRowByCam.get(cam.name);
        if (!camLabels || !rowMap || !labels.videos[camIdx]) return;
        var otherIds = loader.otherSkeletonIds(camLabels);
        var skeletons = camLabels._lazyDataStore.skeletons || [];
        for (var [frameIdx, row] of rowMap) {
            loader.posesNotShownAt(cam.name, frameIdx).forEach(function (pose) {
                var inst = pose.inst;
                if (!otherIds || !otherIds.has(skeletons.indexOf(inst.skeleton))) inst.skeleton = labels.skeletons[0];
                if (inst.identity) inst.identity = findIdentity(inst.identity);
                noteTrack(inst);
                var lf = frameFor(labels.videos[camIdx], frameIdx);
                lf.instances.push(inst);
                placed.set(cam.name + ':' + row + ':' + pose.offset, { inst: inst, lf: lf });
            });
        }
    });
    var sioSession = labels.sessions && labels.sessions[0];
    if (session.outsideGroups && sioSession) {
        var sioCamOf = new Map();
        sioSession.cameraGroup.cameras.forEach(function (c) { sioCamOf.set(c.name, c); });
        for (var [groupFrame, groups] of session.outsideGroups) {
            for (var g = 0; g < groups.length; g++) {
                var instanceByCamera = new Map();
                var lfByCamera = new Map();
                for (var [memberCam, member] of groups[g].members) {
                    var sioCam = sioCamOf.get(memberCam);
                    var at = placed.get(memberCam + ':' + member.row + ':' + member.offset);
                    if (!sioCam || !at) continue;
                    instanceByCamera.set(sioCam, at.inst);
                    lfByCamera.set(sioCam, at.lf);
                }
                if (instanceByCamera.size === 0) continue;
                var sioFrameGroup = sioSession.frameGroups.get(groupFrame);
                if (!sioFrameGroup) {
                    sioFrameGroup = new SIO.FrameGroup({ frameIdx: groupFrame, instanceGroups: [], labeledFrameByCamera: new Map() });
                    sioSession.frameGroups.set(groupFrame, sioFrameGroup);
                }
                lfByCamera.forEach(function (lf, c) { sioFrameGroup.labeledFrameByCamera.set(c, lf); });
                sioFrameGroup.instanceGroups.push(new SIO.InstanceGroup({
                    instanceByCamera: instanceByCamera, instance3d: groups[g].instance3d,
                    identity: groups[g].identity ? findIdentity(groups[g].identity) : undefined, score: groups[g].score, metadata: groups[g].metadata,
                }));
            }
        }
    }
    // Poses that all joined frames `labels` has add no frame, but their tracks still count.
    if (labeledFrames.length === 0 && trackByName.size === 0) return null;
    return { videos: Array.from(headerVideoOf.values()), tracks: Array.from(trackByName.values()), labeledFrames: labeledFrames };
}

/**
 * The sentence a load's final status adds: what the file holds that is not on
 * screen, and what a save does with it. Empty when there is nothing to say.
 *
 * @param {{frames: number, videos: number, cameraPoses: (number|undefined),
 *   cameraSkeletons: (string[]|undefined), groups: (number|undefined)}} counts - frames
 *   and videos outside the opened sessions; poses on a camera's video kept for being on
 *   another skeleton; groups kept aside for linking one (`keepGroupAside`)
 * @param {number} [fileSessions] - sessions in the file, for a loader that opens only
 *   the first
 * @returns {string}
 */
export function outsideFramesNote(counts, fileSessions) {
    var notes = [];
    if (fileSessions > 1) {
        notes.push('Only the first of ' + fileSessions + ' sessions in this file is open: saving keeps the ' +
            'other sessions\' poses, but not their cameras or 3D.');
    } else if (counts.frames) {
        notes.push(counts.frames.toLocaleString('en-US') + ' labeled ' + (counts.frames === 1 ? 'frame' : 'frames') +
            ' on ' + counts.videos + ' ' + (counts.videos === 1 ? 'video' : 'videos') +
            ' in no session ' + (counts.frames === 1 ? 'is' : 'are') + ' not shown; saving keeps ' +
            (counts.frames === 1 ? 'it' : 'them') + '.');
    }
    if (counts.cameraPoses) {
        var one = counts.cameraPoses === 1;
        var named = (counts.cameraSkeletons || []).map(function (n) { return '"' + n + '"'; }).join(', ');
        notes.push(counts.cameraPoses.toLocaleString('en-US') + (one ? ' pose' : ' poses') + ' on a camera\'s video ' +
            (one ? 'is' : 'are') + ' on another skeleton' + (named ? ' (' + named + ')' : '') + ' and ' + (one ? 'is' : 'are') +
            ' not shown; saving keeps ' + (one ? 'it' : 'them') + ' on that skeleton.');
    }
    if (counts.groups) {
        var oneGroup = counts.groups === 1;
        notes.push(counts.groups.toLocaleString('en-US') + (oneGroup ? ' group' : ' groups') + ' with poses on another skeleton ' +
            (oneGroup ? 'is' : 'are') + ' not shown, nor are the poses ' + (oneGroup ? 'it links' : 'they link') +
            '; saving keeps ' + (oneGroup ? 'it' : 'them') + '.');
    }
    return notes.join(' ');
}

/**
 * What `session.outsideFrames` holds, as `outsideFramesNote` counts it.
 *
 * @param {Session|Session[]} sessions
 */
export function keptOutsideCounts(sessions) {
    var counts = { frames: 0, videos: 0, cameraPoses: 0, cameraSkeletons: [] };
    [].concat(sessions).forEach(function (session) {
        var kept = session && session.outsideFrames;
        if (!kept) return;
        // A frame holding poses of both kinds is one frame.
        var outside = new Set();
        kept.frames.forEach(function (f) { outside.add(f.video + ':' + f.frameIdx); });
        kept.otherSkeletonFrames.forEach(function (f) {
            if (kept.videos[f.video].camera == null) { outside.add(f.video + ':' + f.frameIdx); return; }
            counts.cameraPoses += f.instances.length;
            f.instances.forEach(function (inst) {
                var name = inst.skeleton && inst.skeleton.name;
                if (name && counts.cameraSkeletons.indexOf(name) < 0) counts.cameraSkeletons.push(name);
            });
        });
        counts.frames += outside.size;
        counts.videos += kept.videos.filter(function (v) { return v.camera == null; }).length;
    });
    return counts;
}
