/**
 * folder-skeleton.js — which skeleton a per-camera session folder shows, and what each
 * camera's file does about it. Both routes of `handleLoadSessionFolderPerCamera` use it,
 * so a folder reads the same on either.
 *
 * LUC3D shows one skeleton per session and `Instance` stores 2D by node index, so a
 * camera's poses must line up with the session's nodes by NAME. Three cases:
 *
 *  - Same nodes in the same order: shown as they are. Edges do not count: a session
 *    draws one set, and the values stay under their own names.
 *  - Same nodes in another order: shown, read by name (`validateSkeletonCompatibility`,
 *    the rule File ▸ Add SLP applies to a second file). Read by position, nose and tail
 *    would swap.
 *  - Other nodes: not shown, and a save keeps the poses on their own skeleton
 *    (import-export/outside-frames.js). A folder is a load of several files, but a pose
 *    gets the same promise whichever file it came from.
 *
 * The file's first skeleton with a pose decides for the camera: a file holding poses on
 * two skeletons shows the first's, as long as that is the session's.
 *
 * Node-order code here touches the file's rows in memory only. It never writes a file.
 */

import { validateSkeletonCompatibility } from '../import-export/slp-merge.js';

function nodeSetKey(skeleton) {
    return JSON.stringify(skeleton.nodes.slice().sort());
}

/**
 * The skeleton a folder's session shows: the one the folder declares (`skeleton.json`),
 * else the node set most cameras have poses on. One odd camera does not hide the others.
 * A tie goes to the first camera by name, whose node order the session then takes; so
 * does a folder with no poses at all. Nothing here depends on which file opened first.
 *
 * @param {Array<{camName: string, skeleton: ({nodes: string[]}|null), hasPoses: boolean}>} cameras
 * @param {{nodes: string[]}|null} [declared]
 * @returns {Object|null} that camera's `skeleton` (or `declared`); null when none has nodes
 */
export function chooseFolderSkeleton(cameras, declared) {
    if (declared && declared.nodes && declared.nodes.length > 0) return declared;
    var groups = new Map();     // node set -> { votes, first, firstWithPoses }
    cameras.slice().sort(function (a, b) {
        return a.camName < b.camName ? -1 : a.camName > b.camName ? 1 : 0;
    }).forEach(function (cam) {
        if (!cam.skeleton || !cam.skeleton.nodes || cam.skeleton.nodes.length === 0) return;
        var key = nodeSetKey(cam.skeleton);
        var group = groups.get(key);
        if (!group) groups.set(key, group = { votes: 0, first: cam.skeleton, firstWithPoses: null });
        if (cam.hasPoses) {
            group.votes++;
            if (!group.firstWithPoses) group.firstWithPoses = cam.skeleton;
        }
    });
    var best = null;
    groups.forEach(function (group) { if (!best || group.votes > best.votes) best = group; });
    return best ? (best.firstWithPoses || best.first) : null;
}

/**
 * What a camera's skeleton means against the session's.
 *
 * @param {{nodes: string[]}|null} shown - `chooseFolderSkeleton`
 * @param {{nodes: string[]}|null} skeleton - the camera file's first skeleton with a pose
 * @returns {{onOtherNodes: boolean, reorderMap: (number[]|null)}} `reorderMap[fileIdx] =
 *   sessionIdx`, when the same nodes are in another order
 */
export function cameraSkeletonFit(shown, skeleton) {
    if (!shown || !skeleton || !skeleton.nodes || skeleton.nodes.length === 0) {
        return { onOtherNodes: false, reorderMap: null };
    }
    var fit = validateSkeletonCompatibility(shown, skeleton);
    return { onOtherNodes: !!fit.error, reorderMap: fit.reorderMap };
}

/** One instance's per-node values in the session's order; missing ones are `blank`. */
function inSessionOrder(values, reorderMap, blank) {
    if (!values) return values;
    var out = new Array(reorderMap.length).fill(blank);
    for (var k = 0; k < reorderMap.length && k < values.length; k++) out[reorderMap[k]] = values[k];
    return out;
}

/**
 * Put one camera's parsed poses in the session's node order, in place, so everything
 * after this reads them by position as usual.
 *
 * @param {Object} slpData - the worker's result: `columnar`, or nested `frames`
 * @param {number[]} reorderMap - `cameraSkeletonFit`
 */
export function readCameraPosesByName(slpData, reorderMap) {
    var col = slpData.columnar;
    if (col) {
        var nn = col.numNodes;
        var xy = new Float64Array(nn * 2);
        var occluded = new Uint8Array(nn);
        var scores = new Float64Array(nn);
        for (var i = 0; i < col.nInstances; i++) {
            var base = i * nn;
            for (var k = 0; k < nn; k++) {
                var to = reorderMap[k];
                xy[to * 2] = col.xy[(base + k) * 2];
                xy[to * 2 + 1] = col.xy[(base + k) * 2 + 1];
                occluded[to] = col.occluded[base + k];
                scores[to] = col.pointScore[base + k];
            }
            col.xy.set(xy, base * 2);
            col.occluded.set(occluded, base);
            col.pointScore.set(scores, base);
        }
    }
    (slpData.frames || []).forEach(function (fd) {
        fd.instances.forEach(function (d) {
            d.points = inSessionOrder(d.points, reorderMap, null);
            d.occluded = inSessionOrder(d.occluded, reorderMap, false);
            d.pointScores = inSessionOrder(d.pointScores, reorderMap, NaN);
        });
    });
}

/**
 * The same, for a lazy store's point tables, on the rows of `skeletonIds`. A row whose
 * point count is not its skeleton's is left as it is: its nodes cannot be told apart.
 *
 * @param {Object} store - a sleap-io.js `LazyDataStore`
 * @param {Set<number>} skeletonIds
 * @param {number[]} reorderMap - `cameraSkeletonFit`
 */
export function reorderStoreNodes(store, skeletonIds, reorderMap) {
    var idn = store.instancesData || {};
    var starts = idn.point_id_start || [];
    var n = reorderMap.length;
    var held = new Array(n);
    var COLUMNS = ['x', 'y', 'visible', 'complete', 'score'];
    for (var j = 0; j < starts.length; j++) {
        if (!skeletonIds.has(Number(idn.skeleton ? idn.skeleton[j] : 0))) continue;
        var start = Number(starts[j]);
        if (Number(idn.point_id_end[j]) - start !== n) continue;
        // `materializeFrame`'s rule: type 0 is a user instance.
        var table = Number(idn.instance_type ? idn.instance_type[j] : 0) === 0 ? store.pointsData : store.predPointsData;
        for (var c = 0; c < COLUMNS.length; c++) {
            var column = table[COLUMNS[c]];
            if (!column) continue;
            for (var k = 0; k < n; k++) held[k] = column[start + k];
            for (var m = 0; m < n; m++) column[start + reorderMap[m]] = held[m];
        }
    }
}

function nodeList(nodes) {
    var MAX = 6;
    return nodes.slice(0, MAX).join(', ') + (nodes.length > MAX ? ', … ' + nodes.length + ' in all' : '');
}

/**
 * The sentence a load's final status adds for cameras showing no poses because their
 * files are on other nodes. What happens to those poses is `outsideFramesNote`'s.
 *
 * @param {string[]} cameras
 * @param {{nodes: string[]}|null} shown - what the session shows
 * @returns {string} empty when there are none
 */
export function camerasOnOtherNodesNote(cameras, shown) {
    if (cameras.length === 0) return '';
    var one = cameras.length === 1;
    return cameras.join(', ') + (one ? ' shows' : ' show') + ' no poses: ' + (one ? 'its file is' : 'their files are') +
        ' on other nodes than this session\'s (' + nodeList(shown ? shown.nodes : []) + ').';
}

/**
 * The same for analysis `.h5` cameras opened lazily. Their poses stay in the workers
 * that read them, so a save has nothing to keep: those files are not read at all.
 *
 * @param {string[]} cameras
 * @param {{nodes: string[]}|null} shown - what the session shows
 * @returns {string} empty when there are none
 */
export function camerasRefusedNote(cameras, shown) {
    if (cameras.length === 0) return '';
    var one = cameras.length === 1;
    return 'Not read: the annotation ' + (one ? 'file' : 'files') + ' of ' + cameras.join(', ') + ', on other nodes than this session\'s (' +
        nodeList(shown ? shown.nodes : []) + '). A save will not include ' + (one ? 'it' : 'them') + '.';
}
