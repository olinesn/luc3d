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
 *  - on anything else: its own place in the header, after those. A camera that is gone
 *    leaves its suggestions on the video the file named.
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
            video = keptVideoOf((slpData.videos || [])[sug.videoIdx] || null, sug.videoIdx);
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
    var rows = [];
    function addRows(labels, cameraOfVideo) {
        var suggestions = (labels && labels.suggestions) || [];
        for (var i = 0; i < suggestions.length; i++) {
            var sug = suggestions[i];
            var camera = cameraOfVideo.get(sug.video);
            rows.push({
                camera: camera === undefined ? null : camera,
                video: { id: labels.videos.indexOf(sug.video), record: sug.video, typed: sug.video },
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
 * Every suggestion of `sessions`, each on a video of `allVideos`, in the order their
 * files held them. A video only a suggestion names is appended to `allVideos`.
 *
 * @param {Array<{session: Session, cameraVideos: Object[], placedVideos: ?Iterable}>} sessions
 *   each session with the header's video per `session.cameras` entry, and what is
 *   already placed: [the file's record of a video, the header's video] pairs
 *   (`outsideFramesForSave`'s `placedVideos`)
 * @param {Object[]} allVideos - the header's `SIO.Video` list so far
 * @returns {Object[]} `SIO.SuggestionFrame`s, for the `Labels` or the writer's header
 */
export function suggestionsForSave(sessions, allVideos) {
    var SIO = window.SleapIO;
    var placed = new Map();
    sessions.forEach(function (s) {
        if (s.placedVideos) for (var pair of s.placedVideos) placed.set(pair[0], pair[1]);
    });
    var entries = [];
    sessions.forEach(function (s) {
        var rows = (s.session.lazyLoader ? lazySuggestionRows(s.session) : []).concat(s.session.keptSuggestions || []);
        if (rows.length === 0) return;
        var videoOfCamera = new Map();
        s.session.cameras.forEach(function (c, i) {
            if (!videoOfCamera.has(c.name)) videoOfCamera.set(c.name, s.cameraVideos[i]);
        });
        rows.forEach(function (row) {
            var video = row.camera === null ? undefined : videoOfCamera.get(row.camera);
            if (!video) video = placed.get(row.video.record);
            if (!video) {
                video = headerVideoOfKept(SIO, row.video);
                placed.set(row.video.record, video);
                // Two embedded records can share one source video, and a closed session's
                // video can already be in the header.
                if (allVideos.indexOf(video) < 0) allVideos.push(video);
            }
            entries.push({
                order: row.order,
                frame: new SIO.SuggestionFrame({
                    video: video,
                    frameIdx: row.frameIdx,
                    // As the file holds it (the reader's `.group` is a string). A file with
                    // none reads as 0 in Python; left out, the writer would write "default".
                    group: row.group != null ? row.group : 0,
                }),
            });
        });
    });
    entries.sort(function (a, b) { return a.order - b.order; });
    return entries.map(function (e) { return e.frame; });
}
