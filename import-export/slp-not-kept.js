/**
 * slp-not-kept.js — what a `.slp` holds that a LUC3D save leaves out.
 *
 * A save writes poses, 3D, identities and behavior labels. Python sleap-io can also put
 * ROIs, masks, boxes, centroids, label images, negative frames and per-row metadata in a
 * file (`LUCID local patch (slp 3.0)` lists them as `labels.lucidNotKept`). Overwriting the
 * opened file would destroy those with nothing said, so `quickSave` asks where to save
 * instead, and the load's status says why. Once saved elsewhere, the project's file holds
 * only what a save keeps, and the next save goes there as usual.
 *
 * Each loader that reads a `.slp` for a session sets `session.fileNotKept`, beside
 * `session.fileProvenance`. In memory only. Import-free so a Node test can load it.
 */

/** Whether any of these sessions came from a file holding what a save leaves out. */
export function holdsNotKept(sessions) {
    return (sessions || []).some(function (s) { return !!(s && s.fileNotKept && s.fileNotKept.length); });
}

/** The project now lives in a file a save wrote, which holds none of it. */
export function clearNotKept(sessions) {
    (sessions || []).forEach(function (s) { if (s) s.fileNotKept = null; });
}

/**
 * What the load's status says, or `''`.
 *
 * @param {Array<{fileNotKept: ?string[]}>} sessions
 * @returns {string}
 */
export function notKeptNote(sessions) {
    var names = [];
    (sessions || []).forEach(function (s) {
        ((s && s.fileNotKept) || []).forEach(function (name) {
            if (names.indexOf(name) < 0) names.push(name);
        });
    });
    if (names.length === 0) return '';
    return 'A save leaves out this file\'s ' + names.join(', ') + ', so Save asks where to save and this file stays as it is';
}
