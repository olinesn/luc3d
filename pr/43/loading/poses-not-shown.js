/**
 * poses-not-shown.js — which poses of a lazy project's store its session does not show.
 *
 * LUC3D shows one skeleton per session and `Instance` stores 2D by node index, so a
 * pose on another skeleton is never read by position: its values would land under the
 * session's node names and its extra nodes would go (import-export/outside-frames.js).
 * A group linking such a pose cannot be held either, so every pose of that group goes
 * aside with it (`keepGroupAside`).
 *
 * A store holding any of these carries one extra instance column, `NOT_SHOWN`, set on
 * each such row. A column because a delete compacts every column of `instancesData`
 * and a multi-session save keeps them together, so the flags move with their rows. The
 * writers read named columns only, so it never reaches a file. A store with none has
 * no such column and pays nothing.
 *
 * Import-free: `SioLazyLoader` and the Node tests that load it need no more than this.
 */

var NOT_SHOWN = 'lucid_not_shown';

/**
 * The flag column of `store`, one entry per instance row, or null when the session
 * shows every pose in it.
 *
 * @param {Object} store - a sleap-io.js `LazyDataStore`
 * @returns {Uint8Array|null}
 */
export function notShownRows(store) {
    return (store && store.instancesData && store.instancesData[NOT_SHOWN]) || null;
}

/**
 * Flag one instance row. Creates the column on the first.
 *
 * @param {Object} store
 * @param {number} instanceRow
 */
export function markRowNotShown(store, instanceRow) {
    var idn = store.instancesData;
    var flags = idn[NOT_SHOWN];
    if (!flags) {
        var col = idn.instance_type || idn.point_id_start;
        flags = idn[NOT_SHOWN] = new Uint8Array(col ? col.length : 0);
    }
    if (instanceRow >= 0 && instanceRow < flags.length) flags[instanceRow] = 1;
}

/**
 * Flag every row, for a store whose whole file the session does not show.
 *
 * @param {Object} store
 */
export function markAllRowsNotShown(store) {
    var idn = store.instancesData;
    var col = idn.instance_type || idn.point_id_start;
    if (!idn[NOT_SHOWN]) idn[NOT_SHOWN] = new Uint8Array(col ? col.length : 0);
    idn[NOT_SHOWN].fill(1);
}

/**
 * Take every flag off, and the column with it: a store with none pays nothing.
 *
 * @param {Object} store
 */
export function clearRowsNotShown(store) {
    delete store.instancesData[NOT_SHOWN];
}

function edgeKeys(edges) {
    return (edges || []).map(function (e) { return e[0] + '-' + e[1]; }).sort().join(',');
}

/**
 * The skeleton a session shows, of the different ones its file holds: the one its LUC3D
 * record names (a project can hold sessions on different skeletons), else the first with
 * a pose on the session's videos, else the first.
 *
 * @param {Array<{nodes: string[], edges: number[][]}>} entries - one per different
 *   skeleton, in the file's order, except that an unused one is never first
 * @param {{nodes: Array, edges: Array}|null|undefined} record - `metadata.lucid.skeleton`
 * @param {(entry: Object) => boolean} onSessionVideos - does it have a pose on them?
 * @returns {Object} one of `entries`
 */
export function shownSkeletonEntry(entries, record, onSessionVideos) {
    if (entries.length === 1) return entries[0];
    if (record && record.nodes) {
        var names = JSON.stringify(record.nodes.map(function (n) { return typeof n === 'string' ? n : (n.name || ''); }));
        var sameNodes = entries.filter(function (e) { return JSON.stringify(e.nodes) === names; });
        var sameEdges = sameNodes.filter(function (e) { return edgeKeys(e.edges) === edgeKeys(record.edges); });
        if (sameEdges.length > 0) return sameEdges[0];
        if (sameNodes.length > 0) return sameNodes[0];
    }
    for (var i = 0; i < entries.length; i++) {
        if (onSessionVideos(entries[i])) return entries[i];
    }
    return entries[0];
}
