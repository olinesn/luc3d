/**
 * test-folder-skeleton.mjs — the rule a per-camera folder's two routes share: which
 * skeleton the session shows, and what a camera's file does about it.
 *
 * Run: node tests/test-folder-skeleton.mjs
 */
import assert from 'node:assert/strict';
import {
    camerasOnOtherNodesNote, camerasRefusedNote, cameraSkeletonFit, chooseFolderSkeleton,
    readCameraPosesByName, reorderStoreNodes,
} from '../loading/folder-skeleton.js';
import { clearRowsNotShown, markAllRowsNotShown, markRowNotShown, notShownRows } from '../loading/poses-not-shown.js';

const log = console.log;
console.log = () => {};   // validateSkeletonCompatibility says when it builds a reorder map
let passed = 0;
function test(name, fn) {
    fn();
    passed++;
    log('  ✓ ' + name);
}

const sk = (...nodes) => ({ nodes });
const cam = (camName, skeleton, hasPoses = true) => ({ camName, skeleton, hasPoses });

test('every camera on one skeleton: the first by name', () => {
    const a = sk('nose', 'tail'), b = sk('nose', 'tail');
    assert.equal(chooseFolderSkeleton([cam('camB', b), cam('camA', a)]), a);
});

test('the node set most cameras have poses on wins, so one odd camera hides only itself', () => {
    const odd = sk('x', 'y', 'z'), b = sk('nose', 'tail'), c = sk('nose', 'tail');
    assert.equal(chooseFolderSkeleton([cam('camA', odd), cam('camB', b), cam('camC', c)]), b);
});

test('a tie goes to the first camera by name, whichever order they arrive in', () => {
    const a = sk('nose', 'tail'), b = sk('a', 'b', 'c');
    assert.equal(chooseFolderSkeleton([cam('camB', b), cam('camA', a)]), a);
    assert.equal(chooseFolderSkeleton([cam('camA', a), cam('camB', b)]), a);
});

test('another node order is the same node set, and the first by name sets the order', () => {
    const a = sk('nose', 'tail'), b = sk('tail', 'nose'), c = sk('tail', 'nose');
    assert.equal(chooseFolderSkeleton([cam('camC', c), cam('camB', b), cam('camA', a)]), a);
});

test('a camera with no poses has no vote and does not set the order', () => {
    const empty = sk('x', 'y'), b = sk('nose', 'tail');
    assert.equal(chooseFolderSkeleton([cam('camA', empty, false), cam('camB', b)]), b);
    const reversed = sk('tail', 'nose');
    assert.equal(chooseFolderSkeleton([cam('camA', reversed, false), cam('camB', b)]), b);
});

test('no poses anywhere: still a skeleton, the first by name', () => {
    const a = sk('nose', 'tail'), b = sk('a', 'b');
    assert.equal(chooseFolderSkeleton([cam('camB', b, false), cam('camA', a, false)]), a);
});

test('a skeleton the folder declares wins over any vote', () => {
    const declared = sk('a', 'b', 'c');
    assert.equal(chooseFolderSkeleton([cam('camA', sk('nose', 'tail')), cam('camB', sk('nose', 'tail'))], declared), declared);
    const none = sk();
    const a = sk('nose', 'tail');
    assert.equal(chooseFolderSkeleton([cam('camA', a)], none), a);
});

test('no skeleton with nodes: none', () => {
    assert.equal(chooseFolderSkeleton([cam('camA', null), cam('camB', sk())]), null);
    assert.equal(chooseFolderSkeleton([]), null);
});

test('a camera against the session: same order, other order, other nodes', () => {
    const shown = sk('nose', 'tail', 'ear');
    assert.deepEqual(cameraSkeletonFit(shown, sk('nose', 'tail', 'ear')), { onOtherNodes: false, reorderMap: null });
    assert.deepEqual(cameraSkeletonFit(shown, sk('ear', 'nose', 'tail')), { onOtherNodes: false, reorderMap: [2, 0, 1] });
    assert.equal(cameraSkeletonFit(shown, sk('nose', 'tail')).onOtherNodes, true);
    assert.equal(cameraSkeletonFit(shown, sk('nose', 'tail', 'paw')).onOtherNodes, true);
    assert.equal(cameraSkeletonFit(shown, sk('nose', 'nose', 'tail')).onOtherNodes, true);
});

test('nothing to compare against: not on other nodes', () => {
    assert.deepEqual(cameraSkeletonFit(null, sk('a')), { onOtherNodes: false, reorderMap: null });
    assert.deepEqual(cameraSkeletonFit(sk('a'), null), { onOtherNodes: false, reorderMap: null });
    assert.deepEqual(cameraSkeletonFit(sk('a'), sk()), { onOtherNodes: false, reorderMap: null });
});

test('columnar poses take the session\'s order, every per-node value with its node', () => {
    const col = {
        numNodes: 2, nInstances: 2,
        xy: new Float64Array([1, 2, 3, 4, 5, 6, NaN, NaN]),
        occluded: new Uint8Array([1, 0, 0, 0]),
        pointScore: new Float64Array([0.1, 0.2, 0.3, 0.4]),
    };
    readCameraPosesByName({ columnar: col, frames: [] }, [1, 0]);
    assert.deepEqual([...col.xy], [3, 4, 1, 2, NaN, NaN, 5, 6]);
    assert.deepEqual([...col.occluded], [0, 1, 0, 0]);
    assert.deepEqual([...col.pointScore], [0.2, 0.1, 0.4, 0.3]);
});

test('nested poses take it too, and what a pose does not carry stays absent', () => {
    const with_ = { points: [[1, 2], null, [5, 6]], occluded: [true, false, false], pointScores: [0.1, 0.2, 0.3] };
    const without = { points: [[1, 2], [3, 4], null] };
    readCameraPosesByName({ frames: [{ instances: [with_, without] }] }, [2, 0, 1]);
    assert.deepEqual(with_.points, [null, [5, 6], [1, 2]]);
    assert.deepEqual(with_.occluded, [false, false, true]);
    assert.deepEqual(with_.pointScores, [0.2, 0.3, 0.1]);
    assert.deepEqual(without.points, [[3, 4], null, [1, 2]]);
    assert.equal(without.occluded, undefined);
    assert.equal(without.pointScores, undefined);
});

function store() {
    return {
        instancesData: {
            // user on skeleton 0, prediction on skeleton 0, prediction on skeleton 1, a short row on 0
            skeleton: new Float64Array([0, 0, 1, 0]),
            instance_type: new Float64Array([0, 1, 1, 0]),
            point_id_start: new Float64Array([0, 0, 2, 2]),
            point_id_end: new Float64Array([2, 2, 4, 3]),
        },
        pointsData: { x: new Float64Array([1, 2, 7]), y: new Float64Array([3, 4, 8]), visible: new Float64Array([1, 0, 1]), complete: new Float64Array([0, 1, 1]) },
        predPointsData: { x: new Float64Array([5, 6, 9, 10]), y: new Float64Array([7, 8, 11, 12]), visible: new Float64Array(4).fill(1), complete: new Float64Array(4), score: new Float64Array([0.5, 0.6, 0.7, 0.8]) },
    };
}

test('a store\'s rows take it in both point tables, on the named skeleton only', () => {
    const s = store();
    reorderStoreNodes(s, new Set([0]), [1, 0]);
    assert.deepEqual([...s.pointsData.x], [2, 1, 7]);
    assert.deepEqual([...s.pointsData.y], [4, 3, 8]);
    assert.deepEqual([...s.pointsData.visible], [0, 1, 1]);
    assert.deepEqual([...s.pointsData.complete], [1, 0, 1]);
    assert.deepEqual([...s.predPointsData.x], [6, 5, 9, 10]);
    assert.deepEqual([...s.predPointsData.score], [0.6, 0.5, 0.7, 0.8]);
    assert.deepEqual([...s.predPointsData.y], [8, 7, 11, 12]);
});

test('doing it twice is doing nothing, so a re-decision from the current order is safe', () => {
    const s = store();
    reorderStoreNodes(s, new Set([0, 1]), [1, 0]);
    reorderStoreNodes(s, new Set([0, 1]), [1, 0]);
    assert.deepEqual(s.pointsData, store().pointsData);
    assert.deepEqual(s.predPointsData, store().predPointsData);
});

test('a whole camera\'s rows flagged, and back: no flag, no column, so a shown store pays nothing', () => {
    const s = store();
    markRowNotShown(s, 1);
    markAllRowsNotShown(s);
    assert.deepEqual([...notShownRows(s)], [1, 1, 1, 1]);
    clearRowsNotShown(s);
    assert.equal(notShownRows(s), null);
    assert.equal(Object.keys(s.instancesData).length, Object.keys(store().instancesData).length);
    markAllRowsNotShown(s);
    assert.deepEqual([...notShownRows(s)], [1, 1, 1, 1]);
});

test('the status names the cameras and what the session shows', () => {
    assert.equal(camerasOnOtherNodesNote([], sk('a')), '');
    assert.equal(camerasRefusedNote([], sk('a')), '');
    assert.equal(camerasOnOtherNodesNote(['camB'], sk('nose', 'tail')), 'camB shows no poses: its file is on other nodes than this session\'s (nose, tail).');
    assert.equal(camerasOnOtherNodesNote(['camB', 'camC'], sk('a', 'b', 'c', 'd', 'e', 'f', 'g')),
        'camB, camC show no poses: their files are on other nodes than this session\'s (a, b, c, d, e, f, … 7 in all).');
    assert.equal(camerasRefusedNote(['camB'], sk('nose', 'tail')), 'Not read: the annotation file of camB, on other nodes than this session\'s (nose, tail). A save will not include it.');
});

console.log = log;
console.log(`\n${passed} passed`);
