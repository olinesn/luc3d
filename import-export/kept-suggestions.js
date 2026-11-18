/**
 * kept-suggestions.js — a `.slp`'s suggestions (`suggestions_json`, SLEAP's frames to
 * label next) through a save.
 *
 * LUC3D has no screen for them, so all a save has to do is leave them alone: each on
 * its own video and frame, in the file's order, its group as the file holds it.
 *
 * The trap is the video. The vendored `writeSuggestions` finds a suggestion's video in
 * the header by object and writes -1 for one the header lacks, which Python reads as
 * the LAST video in the list: the wrong video, with no error. The header holds fresh
 * objects (a session's cameras, always, and the videos of kept poses), so a suggestion
 * is never handed along as read. It goes through `suggestionsForSave`, which sends it
 * to the header's video:
 *  - on a camera's video: that camera's entry, whatever it is called now (an attached
 *    video renames it), and on a camera with no poses too;
 *  - on a video with kept poses (a video no session names, a session the loader did
 *    not open): their entry (outside-frames.js);
 *  - on anything else: its own place in the header, after those, and the session's
 *    bouts on that video share it. A camera that is gone leaves its suggestions on the
 *    video the file named.
 *
 * Where they come from:
 *  - An eager load keeps a list on the session (`keepSuggestions`,
 *    `session.keptSuggestions`). A suggestion on a video the opened sessions do not
 *    read is held by the first of them, as kept poses are.
 *  - A lazy session reads them from its reader's `Labels` at save
 *    (`lazySuggestionRows`), so a lazy session on the eager writer's path keeps them too.
 *
 * Nothing here is written to `metadata.lucid`, and a file with no suggestions saves
 * byte-identically (`tests/e2e/save-golden-digest.mjs`).
 */

import { keptVideoOf, headerVideoOfKept } from './outside-frames.js';
import { keptVideoRecords } from './behavior-events.js';

// Rows of one file sit on the sessions that read their videos. This puts them back in
// the file's order at save, and rows of a file added later after them.
var nextRowOrder = 0;
// A lazy session's rows are made at save, in its reader's order: they go first, in
// session order (`Array.sort` keeps equal keys in order).
var LAZY_ROW_ORDER = -1;

/**
 * Eager load: put `slpData.suggestions` on the sessions opened from it.
 *
 * @param {Object} slpData - `parseSlpViaSleapIO` / `parseSlpH5` result
 * @param {Array<{session: Session, cameraByVideo: Object<string, string>}>} opened - the
 *   sessions this load opened from the file, each with its cameras' video indices (as
 *   `videoIdxToCameraName`). The first holds a suggestion on a video none of them reads.
 */
export function keepSuggestions(slpData, opened) {
    var suggestions = slpData.suggestions || [];
    if (suggestions.length === 0 || opened.length === 0) return;
    var keptVideos = new Map();   // video index -> its kept record, one per video of this file
    for (var i = 0; i < suggestions.length; i++) {
        var sug = suggestions[i];
        var holder = opened[0].session, camera = null;
        for (var o = 0; o < opened.length; o++) {
            var name = opened[o].cameraByVideo[sug.videoIdx];
            if (name !== undefined) { holder = opened[o].session; camera = name; break; }
        }
        var video = keptVideos.get(sug.videoIdx);
        if (!video) {
            video = keptVideoOf(slpData, sug.videoIdx);
            keptVideos.set(sug.videoIdx, video);
        }
        if (!holder.keptSuggestions) holder.keptSuggestions = [];
        holder.keptSuggestions.push({ camera: camera, video: video, frameIdx: sug.frameIdx, group: sug.group, order: nextRowOrder++ });
    }
}

/** A video moved to another session takes its suggestions along. */
export function moveKeptSuggestions(fromSession, toSession, cameraName) {
    var rows = fromSession.keptSuggestions;
    if (!rows) return;
    var moved = rows.filter(function (r) { return r.camera === cameraName; });
    if (moved.length === 0) return;
    fromSession.keptSuggestions = rows.filter(function (r) { return r.camera !== cameraName; });
    toSession.keptSuggestions = (toSession.keptSuggestions || []).concat(moved);
}

/**
 * A lazy session's suggestions, from its reader's `Labels`, each with the camera whose
 * video it is on. The rule for a camera's video is the one `SioLazyLoader` reads the
 * camera by: the session's camera map on a project file; on a camera's own file, the
 * first video it lists with frames.
 */
function lazySuggestionRows(session) {
    var loader = session.lazyLoader;
    // The worker-backed loader (a SLEAP analysis .h5) has none to keep.
    if (!loader.labelsByCam) return [];
    // A project file's store lists its videos as the file does, which is where this
    // session's kept events come from. A per-camera store is another file: no events.
    var eventRecords = loader._sharedStore ? keptVideoRecords(session) : new Map();
    var rows = [];
    function addRows(labels, cameraOfVideo) {
        var suggestions = (labels && labels.suggestions) || [];
        for (var i = 0; i < suggestions.length; i++) {
            var sug = suggestions[i];
            var camera = cameraOfVideo.get(sug.video);
            var id = labels.videos.indexOf(sug.video);
            rows.push({
                camera: camera === undefined ? null : camera,
                video: { id: id, record: sug.video, typed: sug.video, eventRecord: eventRecords.get(id) || null },
                frameIdx: sug.frameIdx,
                group: sug.metadata ? sug.metadata.group : null,
                order: LAZY_ROW_ORDER,
            });
        }
    }
    if (loader._sharedStore) {
        var shared = loader._projectLabels || Array.from(loader.labelsByCam.values())[0];
        var sharedCameras = new Map();
        session.cameras.forEach(function (c) {
            var own = shared && loader.videoIdByCam ? (shared.videos || [])[loader.videoIdByCam.get(c.name)] : null;
            if (own) sharedCameras.set(own, c.name);
        });
        addRows(shared, sharedCameras);
        return rows;
    }
    session.cameras.forEach(function (c) {
        var labels = loader.labelsByCam.get(c.name);
        if (!labels || !labels._lazyDataStore) return;
        var ownVideoId = loader._ownVideoByCam ? loader._ownVideoByCam.get(c.name) : undefined;
        var own = (labels.videos || [])[ownVideoId !== undefined ? labels._lazyDataStore.videoIndexFor(ownVideoId) : 0];
        addRows(labels, new Map(own ? [[own, c.name]] : []));
    });
    return rows;
}

/**
 * A session's suggestions for a save, in two steps, because the writers need the first
 * before the camera videos of the second exist. Called now, it gives each video only a
 * suggestion names its place. A session's events must go on those entries too, and the
 * eager writer builds them with the session's Labels.
 *
 * @param {Session} session
 * @param {{videos: Object[], listedVideos: Map<Object, Object>, placedVideos: Map<Object, Object>}} listed
 *   what the header holds so far, added to here: its `SIO.Video`s, the file's record of a
 *   video -> its entry for the session's events (`buildBehaviorEvents`), and the read's
 *   record of a video -> its entry (`outsideFramesForSave`)
 * @returns {function(Object[]): Array} given the header's video per `session.cameras`
 *   entry, the session's suggestions, for `suggestionsInFileOrder`
 */
export function suggestionsForSave(session, listed) {
    var SIO = window.SleapIO;
    var rows = (session.lazyLoader ? lazySuggestionRows(session) : []).concat(session.keptSuggestions || []);
    var cameraNames = new Set(session.cameras.map(function (c) { return c.name; }));
    rows.forEach(function (row) {
        if (row.camera !== null && cameraNames.has(row.camera)) return;
        if (listed.placedVideos.has(row.video.record)) return;
        var video = headerVideoOfKept(SIO, row.video);
        listed.placedVideos.set(row.video.record, video);
        // Two embedded records can share one source video, and a closed session's
        // video can already be in the header.
        if (listed.videos.indexOf(video) < 0) listed.videos.push(video);
        if (row.video.eventRecord) listed.listedVideos.set(row.video.eventRecord, video);
    });
    return function (cameraVideos) {
        var videoOfCamera = new Map();
        session.cameras.forEach(function (c, i) {
            if (!videoOfCamera.has(c.name)) videoOfCamera.set(c.name, cameraVideos[i]);
        });
        return rows.map(function (row) {
            return {
                order: row.order,
                frame: new SIO.SuggestionFrame({
                    video: (row.camera !== null && videoOfCamera.get(row.camera)) || listed.placedVideos.get(row.video.record),
                    frameIdx: row.frameIdx,
                    // As the file holds it (the reader's `.group` is a string). A file with
                    // none reads as 0 in Python; left out, the writer would write "default".
                    group: row.group != null ? row.group : 0,
                }),
            };
        });
    };
}

/**
 * `suggestionsForSave`'s results as `SIO.SuggestionFrame`s, in the order their files held
 * them, for the `Labels` or the writer's header.
 */
export function suggestionsInFileOrder(entries) {
    return entries.slice().sort(function (a, b) { return a.order - b.order; }).map(function (e) { return e.frame; });
}
