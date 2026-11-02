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
 * instance would write, tracker score and model identity included
 * (`detection-labels.js`). A skeleton node edit re-shapes it with the rest of the
 * project (`Session.propagateNodeAdded`/`propagateNodeRemoved`). Track edits do not
 * touch it: it holds the file's track names, not the session's indices.
 *
 * A behavior event kept on such a video goes on the entry its poses have, so the
 * file's `frames` and `events` name one place. Both paths say which entry that is by
 * the file's own record of the video (`listedVideos`, `buildBehaviorEvents`), never by
 * its filename: campy names every video `0.mp4`.
 *
 * Nothing here is written to `metadata.lucid`, and a file with no such frames saves
 * byte-identically (`tests/e2e/save-golden-digest.mjs`).
 *
 * The other sessions themselves (cameras, groups, 3D) are `unopened-sessions.js`.
 */

import { Instance } from '../pose/pose-data.js';
import { keptVideoRecords } from './behavior-events.js';
import { makePooledSioInstances } from './file-io.js';
import { hasMembersOnOpenVideos, identityEntriesOfOpenSession, unopenedSessionsForSave } from './unopened-sessions.js';
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
 * What an eager load holds of a video it keeps something on. `record` is the file's
 * own entry (`slpData.videos[id]`): one object per video per file, so it tells a video
 * apart where an id cannot, as File ▸ Add SLP brings a second file's ids.
 *
 * @param {Object} slpData - `parseSlpViaSleapIO` / `parseSlpH5` result
 * @param {number} id - the video's id in the file
 */
export function keptVideoOf(slpData, id) {
    var record = (slpData.videos || [])[id] || null;
    return {
        id: id,
        record: record,
        typed: record && record._typedVideo ? record._typedVideo : null,
        // What this file's events know the video by (`fileEventsOf`): the very
        // object a kept event of this file holds. A video File ▸ Add SLP brought
        // has another file's record, so even at the same place in its own list
        // it is never taken for theirs.
        eventRecord: slpData.behaviorEvents ? slpData.behaviorEvents.videos[id] : null,
    };
}

/** The `SIO.Video` a header lists for `keptVideoOf`'s result. */
export function headerVideoOfKept(SIO, kept) {
    return kept.typed ? headerVideoForOutsideFrames(SIO, kept.typed, kept.id) : headerVideoOfRecord(SIO, kept.record, kept.id);
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
 * @param {{ownFile?: boolean}} [opts] - `ownFile`: `slpData` is the file whose other
 *   sessions `session` carries, so their group members name these frames' file rows
 */
export function keepOutsideFrames(session, slpData, claimedVideos, nodeReorderMap, opts) {
    var ownFile = !!(opts && opts.ownFile);
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
            var keptVideo = keptVideoOf(slpData, fd.videoIdx);
            keptVideo.ownFile = ownFile;
            kept.videos.push(keptVideo);
        }
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
        var keptFrame = { video: slot, frameIdx: fd.frameIdx, instances: instances };
        if (ownFile) keptFrame.fileRow = fd.fileRow;
        kept.frames.push(keptFrame);
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
 * `ownVideoOfColumnar` for a typed read's frames (a camera file whose detections carry
 * identities or categories: only the typed reader has those).
 *
 * @param {Array<{videoIdx: number}>} frames
 * @returns {number|undefined}
 */
export function ownVideoOfFrames(frames) {
    var own;
    for (var f = 0; f < frames.length; f++) {
        var v = frames[f].videoIdx;
        if (v !== undefined && (own === undefined || v < own)) own = v;
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
                trackingScore: col.trackingScore ? col.trackingScore[i] : 0,
            });
        }
        frames.push({ frameIdx: col.frameIdx[f], videoIdx: col.videoIdx[f], instances: instances });
    }
    if (frames.length > 0) {
        keepOutsideFrames(session, { frames: frames, tracks: slpData.tracks, videos: slpData.videos }, new Set([own]));
    }
}

/**
 * Eager save: `session`'s kept frames as sleap-io.js objects. Called before the
 * session's Labels are built, because its events must go on these videos
 * (`buildSlpLabelsAllViews`), and the frames need those Labels' skeleton and identities.
 *
 * @param {Session} session
 * @returns {{videos: Array, tracks: Array, listedVideos: Map<Object, Object>, placedVideos: Map<Object, Object>, labeledFramesFor: function(Object, Object[]): Array, rowAt: (Map<number, number>|undefined), videoOfId: (Map<number, Object>|undefined), recordOfCamera: (Map<string, Object>|undefined)}|null}
 *   `listedVideos`: the file's record of a video (`fileEventsOf`) -> its entry here;
 *   `placedVideos`: its record in the read (`keptVideoOf`) -> its entry here, for a kept
 *   suggestion on it (kept-suggestions.js);
 *   `labeledFramesFor(skeleton, identities)`: the frames, on the `SIO.Skeleton` and the
 *   `SIO.Identity` list of the session's Labels. An identity only these poses name is
 *   added to that list. `rowAt`/`videoOfId`, for a session carrying its file's other
 *   sessions: file row -> place in those frames, video id -> its header video. A lazy
 *   session's are its store's. `recordOfCamera`, after an eager read: camera name -> the
 *   file's record of its video, for `buildSlpLabelsAllViews`, which makes those entries.
 */
export function outsideFramesForSave(session) {
    var SIO = window.SleapIO;
    // A lazy session on this path (a project mixing lazy and eager sessions): its
    // store holds them, and the eager writer would otherwise leave them behind.
    var loader = session.lazyLoader;
    if (loader && typeof loader.forEachOutsideRow === 'function') return lazyOutsideFramesForSave(SIO, session, loader);
    var closed = session.unopenedSessions && session.unopenedSessions.eager ? session.unopenedSessions : null;
    var kept = session.outsideFrames || (closed ? { videos: [], tracks: [], frames: [] } : null);
    if (!kept) return null;
    var listedVideos = new Map();
    var placedVideos = new Map();
    var videos = kept.videos.map(function (v) {
        var video = headerVideoOfKept(SIO, v);
        if (v.eventRecord) listedVideos.set(v.eventRecord, video);
        placedVideos.set(v.record, video);
        return video;
    });
    var tracks = kept.tracks.map(function (name) { return new SIO.Track(name); });
    var out = {
        videos: videos, tracks: tracks, listedVideos: listedVideos, placedVideos: placedVideos,
        labeledFramesFor: function (skeleton, identities) {
            var findIdentity = identityCatalogFinder(SIO, identities);
            var newSioInstance = makePooledSioInstances(SIO, skeleton, session.skeleton.nodes.length);
            return kept.frames.map(function (f) {
                return new SIO.LabeledFrame({
                    video: videos[f.video],
                    frameIdx: f.frameIdx,
                    instances: f.instances.map(function (inst) {
                        var sioInst = newSioInstance(inst, inst.trackIdx >= 0 ? tracks[inst.trackIdx] : null);
                        writeFileLabels(sioInst, inst, findIdentity);
                        return sioInst;
                    }),
                });
            });
        },
    };
    if (closed) listClosedSessionVideos(SIO, closed, kept, out);
    return out;
}

/**
 * After an eager read: what a save of the session's closed sessions needs from its kept
 * frames (`unopenedSessionsForEagerSave`), and an entry for each video they name that has
 * none yet. A closed session's video has its entry here even with no frame, so a bout or
 * a suggestion on it goes on the entry that session names. A camera's video is listed by
 * the session's Labels, so it only says which record that entry is for.
 */
function listClosedSessionVideos(SIO, closed, kept, out) {
    var eventRecords = closed.eager.eventRecords || [];
    var readRecords = closed.eager.readRecords || [];
    out.rowAt = new Map();
    out.videoOfId = new Map();
    kept.videos.forEach(function (v, i) { if (v.ownFile) out.videoOfId.set(v.id, out.videos[i]); });
    kept.frames.forEach(function (f, i) { if (f.fileRow !== undefined) out.rowAt.set(f.fileRow, i); });
    out.recordOfCamera = new Map();
    closed.eager.videoIdByCam.forEach(function (id, camName) {
        if (eventRecords[id]) out.recordOfCamera.set(camName, eventRecords[id]);
    });
    var cameraIds = new Set(closed.eager.videoIdByCam.values());
    closed.sessions.forEach(function (record) {
        record.videos.forEach(function (v) {
            if (cameraIds.has(v.id) || out.videoOfId.has(v.id)) return;
            var video = headerVideoForOutsideFrames(SIO, v.video, v.id);
            out.videos.push(video);
            out.videoOfId.set(v.id, video);
            if (eventRecords[v.id]) out.listedVideos.set(eventRecords[v.id], video);
            if (readRecords[v.id]) out.placedVideos.set(readRecords[v.id], video);
        });
    });
}

function lazyOutsideFramesForSave(SIO, session, loader) {
    // A project file's store lists its videos as the file does, which is where this
    // session's kept events come from. A per-camera store is another file: no events.
    var records = loader._sharedStore ? keptVideoRecords(session) : new Map();
    var listedVideos = new Map();
    var headerVideoOf = new Map();
    var trackByName = new Map();
    var labeledFrames = [];
    var rowAt = new Map();
    var videoOfId = new Map();
    loader.forEachOutsideRow(function (store, row, id) {
        var lf = store.materializeFrame(row);
        if (!lf) return;
        if (!headerVideoOf.has(lf.video)) {
            var headerVideo = headerVideoForOutsideFrames(SIO, lf.video, id);
            headerVideoOf.set(lf.video, headerVideo);
            var record = records.get(store.videos.indexOf(lf.video));
            if (record) listedVideos.set(record, headerVideo);
        }
        videoOfId.set(id, headerVideoOf.get(lf.video));
        lf.instances.forEach(function (inst) {
            if (inst.track && !trackByName.has(inst.track.name)) trackByName.set(inst.track.name, inst.track);
        });
        rowAt.set(row, labeledFrames.length);
        labeledFrames.push(new SIO.LabeledFrame({ video: headerVideoOf.get(lf.video), frameIdx: lf.frameIdx, instances: lf.instances }));
    });
    if (labeledFrames.length === 0) return null;
    return {
        videos: Array.from(headerVideoOf.values()), tracks: Array.from(trackByName.values()), listedVideos: listedVideos,
        placedVideos: headerVideoOf, rowAt: rowAt, videoOfId: videoOfId,
        labeledFramesFor: function (skeleton, identities) {
            var findIdentity = identityCatalogFinder(SIO, identities);
            labeledFrames.forEach(function (lf) {
                lf.instances.forEach(function (inst) {
                    if (inst.identity) inst.identity = findIdentity(inst.identity);
                });
            });
            return labeledFrames;
        },
    };
}

/**
 * The eager writer's save of a session carrying its file's other sessions, as
 * `unopenedSessionsForSave` gives them. Their poses are `outside`'s frames, so that is
 * where their members go.
 *
 * This writer lists a frame's instances in its own order. A member on a video the open
 * session has too holds its pose after an eager read (`kept.eager`), and `places` says
 * where that pose went; a pose no longer in the project takes the member with it. A lazy
 * session (a project mixing lazy and eager sessions) has no such pose, so then nothing
 * is saved, rather than a session losing its links with no word.
 *
 * @param {Object} SIO - window.SleapIO
 * @param {Session} session - its `unopenedSessions` is set
 * @param {Object} labels - `buildSlpLabelsAllViews` of it
 * @param {Object|null} outside - `outsideFramesForSave` of it
 * @param {number} outsideBase - where `outside.labeledFrames` start in the file
 * @param {Array} videos - the file's video list (appended to)
 * @param {Array} identities - the file's identity list (appended to)
 * @param {Map<Object, (number[]|null)>|null} [places] - `instancePlacesForSave`, as
 *   `buildSlpLabelsAllViews` filled it in
 * @returns {Array} `RecordingSession`s
 */
export function unopenedSessionsForEagerSave(SIO, session, labels, outside, outsideBase, videos, identities, places) {
    var kept = session.unopenedSessions;
    var loader = session.lazyLoader;
    if (!kept.eager && !loader) {
        throw new Error('Cannot save: "' + session.name + '" is no longer loaded, and its file holds ' +
            kept.sessions.length + ' more session(s) that only it can write back. Open the file again, then save.');
    }
    if (!kept.eager && hasMembersOnOpenVideos(kept)) {
        throw new Error('Cannot save: this project mixes lazily-loaded and fully-loaded sessions, and a session in the file of "' +
            session.name + '" shares a video with it. Save that file on its own.');
    }
    var headerOf = new Map();
    var listed = session.identities || [];
    listed.forEach(function (identity, i) { headerOf.set(identity.id, labels.identities[i]); });
    var headerVideoOfId = new Map(outside ? outside.videoOfId : []);
    var videoIdByCam = kept.eager ? kept.eager.videoIdByCam : loader.videoIdByCam;
    session.cameras.forEach(function (c, i) {
        if (videoIdByCam && videoIdByCam.has(c.name)) headerVideoOfId.set(videoIdByCam.get(c.name), labels.videos[i]);
    });
    var sessionBase = outsideBase - labels.labeledFrames.length;
    return unopenedSessionsForSave(kept, {
        SIO: SIO,
        identities: identities,
        // `outsideFramesForSave` added what these poses name after the session's own.
        identityOfEntry: identityEntriesOfOpenSession(kept, listed, headerOf, new Map(labels.identities.slice(listed.length).map(function (identity) {
            return [identity.name, identity];
        }))),
        firstLucid: labels.sessions[0].metadata.lucid,
        headerVideo: function (id, video) {
            if (!headerVideoOfId.has(id)) {
                var added = headerVideoForOutsideFrames(SIO, video, id);
                headerVideoOfId.set(id, added);
                videos.push(added);
            }
            return headerVideoOfId.get(id);
        },
        outputRow: function (row) {
            var at = outside ? outside.rowAt.get(row) : undefined;
            return at === undefined ? undefined : outsideBase + at;
        },
        movedRef: places ? function (row, inst) {
            var poses = kept.eager.posesOfRows.get(row);
            var place = poses && poses[inst] ? places.get(poses[inst]) : null;
            return place ? [sessionBase + place[0], place[1]] : null;
        } : null,
    });
}

/**
 * The sentence a load's final status adds: what the file holds that is not on
 * screen, and what a save does with it. Empty when there is nothing to say.
 *
 * @param {{frames: number, videos: number}} counts - frames and videos outside the
 *   opened sessions
 * @param {number} [fileSessions] - sessions in the file, for a loader that opens only
 *   the first and keeps only the others' poses (the single-`.slp` folder loader on the
 *   raw worker's read, which has no typed sessions to keep whole)
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
