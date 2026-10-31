/**
 * test-lazy-store-rows.mjs — `loading/lazy-store-rows.js`, File ▸ Add SLP's way into a
 * lazy project's columnar store, on hand-built stores.
 *
 * The promise: added rows only ever go after the existing ones, so no existing row moves
 * or changes, and until the commit nothing has changed at all. The real store, the real
 * writer and Python are in `tests/e2e/add-slp-lazy-project.mjs`.
 *
 * Run: node tests/test-lazy-store-rows.mjs
 */
import {
    sourceRowsByVideo, allocateStoreRows, commitStoreRows, firstFreeVideoId, storeFromParsedFrames,
} from '../loading/lazy-store-rows.js';

let fails = 0;
const check = (c, m, got) => {
    console.log((c ? '  ✓ ' : '  ✗ ') + m + (!c && got !== undefined ? ' — got ' + JSON.stringify(got) : ''));
    if (!c) fails++;
};
const eq = (a, b) => JSON.stringify(Array.from(a)) === JSON.stringify(Array.from(b));

/** A store with two frames on video 0: one prediction, one user instance, 2 nodes each. */
function projectStore() {
    return {
        videos: [{ filename: 'a.mp4' }],
        skeletons: [{ nodeNames: ['nose', 'tail'] }],
        tracks: [{ name: 'mouse1' }],
        formatId: 1.4,
        framesData: {
            frame_id: new Float64Array([0, 1]), video: new Float64Array([0, 0]), frame_idx: new Float64Array([5, 6]),
            instance_id_start: new Float64Array([0, 1]), instance_id_end: new Float64Array([1, 2]),
        },
        instancesData: {
            instance_id: new Float64Array([0, 1]), instance_type: new Float64Array([1, 0]), frame_id: new Float64Array([0, 1]),
            skeleton: new Float64Array([0, 0]), track: new Float64Array([0, 0]), from_predicted: new Float64Array([-1, 0]),
            score: new Float64Array([0.9, 0]), point_id_start: new Float64Array([0, 0]), point_id_end: new Float64Array([2, 2]),
            tracking_score: new Float64Array([0.5, 0]),
        },
        pointsData: {
            x: new Float64Array([1, 2]), y: new Float64Array([3, 4]), visible: new Float64Array([1, 1]), complete: new Float64Array([1, 1]),
        },
        predPointsData: {
            x: new Float64Array([5, 6]), y: new Float64Array([7, 8]), visible: new Float64Array([1, 0]),
            complete: new Float64Array([1, 1]), score: new Float64Array([0.1, 0.2]),
        },
    };
}

/**
 * A file: video 0 frame 7 holds a prediction and a user instance linked to it, video 3
 * frame 7 one prediction, then video 0 frame 7 listed AGAIN with one more prediction,
 * and frame 8 on video 0.
 */
function fileStore() {
    return {
        videos: [{ filename: 'x.mp4' }],
        skeletons: [{ nodeNames: ['tail', 'nose'] }],
        tracks: [{ name: 'mouse1' }, { name: 'mouse9' }],
        formatId: 1.4,
        framesData: {
            frame_id: new Float64Array([0, 1, 2, 3]),
            video: new Float64Array([0, 3, 0, 0]),
            frame_idx: new Float64Array([7, 7, 7, 8]),
            instance_id_start: new Float64Array([0, 2, 3, 4]),
            instance_id_end: new Float64Array([2, 3, 4, 5]),
        },
        instancesData: {
            instance_type: new Float64Array([1, 0, 1, 1, 1]),
            skeleton: new Float64Array([0, 0, 0, 0, 0]),
            track: new Float64Array([1, 0, -1, 7, 0]),
            from_predicted: new Float64Array([-1, 0, -1, -1, -1]),
            score: new Float64Array([0.7, 0, 0.6, 0.5, 0.4]),
            point_id_start: new Float64Array([0, 0, 2, 4, 6]),
            point_id_end: new Float64Array([2, 2, 4, 6, 8]),
            tracking_score: new Float64Array([0.25, 0, 0.25, 0.25, 0.25]),
        },
        pointsData: {
            x: new Float64Array([100, NaN]), y: new Float64Array([101, NaN]), visible: new Float64Array([1, 0]), complete: new Float64Array([1, 0]),
        },
        predPointsData: {
            x: new Float64Array([10, 11, 20, 21, 30, 31, 40, 41]), y: new Float64Array([12, 13, 22, 23, 32, 33, 42, 43]),
            visible: new Float64Array([1, 0, 1, 1, 1, 1, 1, 1]), complete: new Float64Array([1, 1, 1, 1, 1, 1, 1, 1]),
            score: new Float64Array([0.11, 0.12, 0.21, 0.22, 0.31, 0.32, 0.41, 0.42]),
        },
    };
}

const snapshot = store => JSON.stringify([store.framesData, store.instancesData, store.pointsData, store.predPointsData],
    (k, v) => (v instanceof Float64Array ? Array.from(v, x => (x === x ? x : 'nan')) : v));

function addFile(store, source, ids) {
    const blocks = sourceRowsByVideo(source);
    blocks.forEach(b => { b.id = ids[b.raw]; });
    const before = snapshot(store);
    const plan = allocateStoreRows(store, blocks, source);
    check(snapshot(store) === before, 'allocating room changes nothing');
    commitStoreRows(plan, source, {
        trackMap: Int32Array.from([0, 1]),
        skeletons: [{ id: 0, order: [1, 0] }],
    });
    return blocks;
}

console.log('A. Rows go after the existing ones');
{
    const store = projectStore();
    const old = projectStore();
    const blocks = addFile(store, fileStore(), { 0: 1, 3: 2 });

    const prefixHolds = (table, oldTable, n) => Object.keys(oldTable).every(k => eq(table[k].subarray(0, n), oldTable[k].subarray(0, n)));
    check(prefixHolds(store.framesData, old.framesData, 2), 'existing frame rows are untouched');
    check(prefixHolds(store.instancesData, old.instancesData, 2), 'existing instance rows are untouched');
    check(prefixHolds(store.pointsData, old.pointsData, 2), 'existing user points are untouched');
    check(prefixHolds(store.predPointsData, old.predPointsData, 2), 'existing predicted points are untouched');

    check(blocks.length === 2 && blocks[0].raw === 0 && blocks[1].raw === 3, 'one block per video, by id', blocks.map(b => b.raw));
    check(eq(store.framesData.video, [0, 0, 1, 1, 2]), 'each row is on its video\'s new id', Array.from(store.framesData.video));
    check(eq(store.framesData.frame_idx, [5, 6, 7, 8, 7]), 'frame numbers are kept', Array.from(store.framesData.frame_idx));
    check(blocks[0].rowMap.get(7) === 2 && blocks[0].rowMap.get(8) === 3 && blocks[1].rowMap.get(7) === 4, 'each block maps its frames to their new rows');

    // A frame listed twice is one row, with all its instances.
    check(store.framesData.instance_id_start[2] === 2 && store.framesData.instance_id_end[2] === 5, 'a frame listed twice keeps all its instances, in one row');
    check(eq(store.instancesData.instance_type, [1, 0, 1, 0, 1, 1, 1]), 'instance types', Array.from(store.instancesData.instance_type));

    // Tracks: 1 -> 1, 0 -> 0, -1 stays, a track id past the file's list has no track.
    check(eq(store.instancesData.track, [0, 0, 1, 0, -1, 0, -1]),
        'track ids go through the map, and one past the file\'s list has no track', Array.from(store.instancesData.track));

    // Nodes: the file lists them tail, nose; the store nose, tail.
    const pred = store.predPointsData;
    check(pred.x[2] === 11 && pred.x[3] === 10, 'a prediction\'s nodes are in the store\'s order', Array.from(pred.x));
    check(pred.visible[2] === 0 && pred.visible[3] === 1, 'visibility goes with its node', Array.from(pred.visible));
    check(pred.score[2] === 0.12 && pred.score[3] === 0.11, 'and so does each node\'s score', Array.from(pred.score));
    const user = store.pointsData;
    check(user.x[3] === 100 && user.x[2] !== user.x[2], 'a user instance\'s nodes too', Array.from(user.x));

    // Point ranges point at the new rows; ranges in the old rows do not move.
    const idn = store.instancesData;
    check(eq(idn.point_id_start, [0, 0, 2, 2, 4, 6, 8]) && eq(idn.point_id_end, [2, 2, 4, 4, 6, 8, 10]), 'point ranges point at the new rows', Array.from(idn.point_id_start));
    check(idn.instance_id[4] === 4 && idn.frame_id[4] === 2, 'an instance row knows its row and its frame');
    check(idn.from_predicted[3] === 2, 'a user instance still links to its prediction, at its new row', Array.from(idn.from_predicted));
    check(idn.from_predicted[1] === 0, 'an existing link is not touched');
    check(idn.tracking_score[2] === 0.25, 'the tracking score comes along');
    check(idn.score[2] === 0.7, 'the instance score comes along');
    check(store.framesData.frame_id[4] === 4, 'a frame row knows its row');
}

console.log('B. A skeleton that does not fit keeps its own shape');
{
    const store = projectStore();
    const source = fileStore();
    source.skeletons.push({ nodeNames: ['a', 'b', 'c'] });
    source.instancesData.skeleton[4] = 1;
    source.predPointsData.x = Float64Array.from([10, 11, 20, 21, 30, 31, 40, 41, 42]);
    for (const k of ['y', 'visible', 'complete', 'score']) source.predPointsData[k] = Float64Array.from([...source.predPointsData[k], 1]);
    source.instancesData.point_id_end[4] = 9;
    const blocks = sourceRowsByVideo(source);
    blocks.forEach((b, i) => { b.id = i + 1; });
    commitStoreRows(allocateStoreRows(store, blocks, source), source, {
        trackMap: Int32Array.from([0, 1]),
        skeletons: [{ id: 0, order: [1, 0] }, { id: 1, order: null }],
    });
    const row = store.instancesData.skeleton.length - 2;   // frame 8: the file's fourth frame row
    check(store.instancesData.skeleton[row] === 1, 'its rows keep their skeleton', Array.from(store.instancesData.skeleton));
    const from = store.instancesData.point_id_start[row];
    check(eq(store.predPointsData.x.subarray(from, from + 3), [40, 41, 42]), 'and their node order', Array.from(store.predPointsData.x));
}

console.log('C. Old SLEAP coordinates');
{
    const store = projectStore();
    const source = fileStore();
    source.formatId = 1.0;
    addFile(store, source, { 0: 1, 3: 2 });
    check(store.predPointsData.x[3] === 9.5, 'a file older than format 1.1 is shifted to today\'s coordinates', Array.from(store.predPointsData.x));
    check(store.predPointsData.score[3] === 0.11, 'only x and y');
    check(store.instancesData.tracking_score[2] === 0, 'and has no tracking score below 1.2', Array.from(store.instancesData.tracking_score));
}

console.log('D. A new, empty store');
{
    const store = {
        videos: [], skeletons: [], tracks: [], formatId: 1.4,
        framesData: {}, instancesData: {}, pointsData: {}, predPointsData: {},
    };
    const source = fileStore();
    const blocks = sourceRowsByVideo(source);
    blocks[0].id = 0;
    blocks[1].id = 1;
    commitStoreRows(allocateStoreRows(store, blocks, source), source, { trackMap: Int32Array.from([0, 1]), skeletons: [{ id: 0, order: null }] });
    check(eq(store.framesData.video, [0, 0, 1]), 'every column exists and every row is there', Array.from(store.framesData.video));
    check(store.instancesData.instance_id.length === 5, 'instance columns', store.instancesData.instance_id.length);
    check(store.pointsData.x.length === 2 && store.predPointsData.x.length === 8, 'point columns');
}

console.log('E. Columns as other readers leave them');
{
    const store = projectStore();
    // A plain array, a view of a longer buffer (`deleteInstanceRows`), an integer column.
    store.framesData.frame_idx = [5, 6];
    store.instancesData.score = new Float64Array([0.9, 0, 0.123]).subarray(0, 2);
    store.instancesData.track = Int32Array.from([0, 0]);
    // One array under two keys stays one array.
    store.framesData.alias = store.framesData.video;
    const source = fileStore();
    const blocks = sourceRowsByVideo(source);
    blocks.forEach((b, i) => { b.id = i + 1; });
    commitStoreRows(allocateStoreRows(store, blocks, source), source, { trackMap: Int32Array.from([0, 1]), skeletons: [{ id: 0, order: null }] });
    check(eq(store.framesData.frame_idx, [5, 6, 7, 8, 7]), 'a plain array grows', Array.from(store.framesData.frame_idx));
    check(store.instancesData.score[1] === 0 && store.instancesData.score[2] === 0.7, 'a view grows from its own end, not its buffer\'s', Array.from(store.instancesData.score));
    check(store.framesData.alias === store.framesData.video, 'an array under two keys stays one array');
    check(store.framesData instanceof Object && store.framesData.frame_idx instanceof Float64Array, 'every grown column is f64');
}

console.log('F. A free video id');
{
    const store = projectStore();
    check(firstFreeVideoId(store) === 1, 'past the last used', firstFreeVideoId(store));
    store.videos = [{}, {}, {}];
    check(firstFreeVideoId(store) === 3, 'past the list: a raw id can fall back to an index', firstFreeVideoId(store));
    store.framesData.video = new Float64Array([0, 9]);
    check(firstFreeVideoId(store) === 10, 'past a sparse id', firstFreeVideoId(store));
    store._roiByFrame = new Map([['12:3', []]]);
    check(firstFreeVideoId(store) === 13, 'past an annotation\'s', firstFreeVideoId(store));
    store.negativeFrames = new Set(['20:1']);
    check(firstFreeVideoId(store) === 21, 'past a negative frame\'s', firstFreeVideoId(store));
}

console.log('G. What detections carry outside the columns');
{
    const store = projectStore();
    const female = { name: 'female', metadata: { strain: 'a' } };
    store.identities = [female];
    store.categoryCatalog = ['mouse'];
    store._instanceIdentityLinks = new Map([[0, [0, 0.1]]]);
    const source = fileStore();
    const male = { name: 'male' };
    source.identities = [{ name: 'female' }, male];
    source.categoryCatalog = ['rat', 'mouse'];
    source._instanceIdentityLinks = new Map([[0, [0, 0.25]], [4, [1, 0.5]], [2, [9, 0.9]]]);
    source._instanceEmbeddings = new Map([[3, { vector: [1, 2] }]]);
    source._instanceCategoryLinks = new Map([[0, [0, 0.75]], [4, [1, 0.8]]]);
    source._instanceCategoryEmbeddings = new Map([[4, { vector: [3] }]]);
    const blocks = sourceRowsByVideo(source);
    blocks.forEach((b, i) => { b.id = i + 1; });
    commitStoreRows(allocateStoreRows(store, blocks, source), source, { trackMap: Int32Array.from([0, 1]), skeletons: [{ id: 0, order: null }] });
    // New rows: file row 0 -> 2, 1 -> 3, 3 -> 4 (a repeat of its frame), 4 -> 5, 2 -> 6.
    const links = store._instanceIdentityLinks;
    check(eq(links.get(0), [0, 0.1]), 'a row already there keeps its label', links.get(0));
    check(eq(links.get(2), [0, 0.25]), 'an identity the store has is found by name', links.get(2));
    check(eq(links.get(5), [1, 0.5]) && store.identities[1] === male, 'one it lacks joins its list, with what it holds', links.get(5));
    check(store.identities[0] === female, 'and the store\'s own stays as it was');
    check(!links.has(6), 'a link to no identity is not carried, as a reader ignores it');
    check(store._instanceEmbeddings.get(4).vector[1] === 2, 'an embedding follows its row');
    check(eq(store._instanceCategoryLinks.get(2), [1, 0.75]) && store.categoryCatalog[1] === 'rat', 'a category joins its list', store.categoryCatalog);
    check(eq(store._instanceCategoryLinks.get(5), [0, 0.8]), 'and is found by value', store._instanceCategoryLinks.get(5));
    check(store._instanceCategoryEmbeddings.get(5).vector[0] === 3, 'a category embedding follows its row');

    // Only the rows this store takes: a file split between stores splits its labels.
    const other = { videos: [], skeletons: [], tracks: [], formatId: 1.4, framesData: {}, instancesData: {}, pointsData: {}, predPointsData: {} };
    const fresh = fileStore();
    fresh.identities = [{ name: 'female' }];
    fresh._instanceIdentityLinks = new Map([[0, [0, 0.25]], [2, [0, 0.5]]]);
    const split = sourceRowsByVideo(fresh);
    split[0].id = 0;
    commitStoreRows(allocateStoreRows(other, [split[0]], fresh), fresh, { trackMap: Int32Array.from([0, 1]), skeletons: [{ id: 0, order: null }] });
    check(other._instanceIdentityLinks.size === 1 && other._instanceIdentityLinks.has(0), 'a row another store takes leaves its label behind', [...other._instanceIdentityLinks.keys()]);

    // A file with none adds nothing to a store with none.
    const plain = projectStore();
    commitStoreRows(allocateStoreRows(plain, sourceRowsByVideo(fileStore()).map((b, i) => Object.assign(b, { id: i + 1 })), fileStore()), fileStore(), { trackMap: Int32Array.from([0, 1]), skeletons: [{ id: 0, order: null }] });
    check(plain._instanceIdentityLinks === undefined && plain.identities === undefined, 'a file with no labels adds none');
}

console.log('H. An eagerly parsed file');
{
    const slpData = {
        skeleton: { nodes: ['nose', 'tail'] },
        tracks: ['t0'],
        frames: [{
            frameIdx: 4, videoIdx: 1,
            instances: [
                { type: 'predicted', trackIdx: 0, score: 0.8, points: [[1, 2], null], occluded: [true, false], pointScores: new Float64Array([0.3, NaN]) },
                { type: 'user', trackIdx: -1, score: 0, points: [[3, 4], [5, 6]], occluded: [false, false], pointScores: null },
                { type: 'predicted', trackIdx: null, score: 0.4, points: [[7, 8], [9, 10]], occluded: [false, false] },
            ],
        }],
    };
    const source = storeFromParsedFrames(slpData);
    const q = source.predPointsData;
    check(eq(Array.from(source.framesData.video), [1]) && source.framesData.frame_idx[0] === 4, 'the frame row');
    check(eq(source.instancesData.instance_type, [1, 0, 1]), 'instance types', Array.from(source.instancesData.instance_type));
    check(eq(source.instancesData.track, [0, -1, -1]), 'no track is -1, however the parser said it', Array.from(source.instancesData.track));
    check(q.x[0] === 1 && q.visible[0] === 0 && q.complete[0] === 1, 'an occluded node keeps its place and is not visible');
    check(q.x[1] !== q.x[1] && q.visible[1] === 0 && q.complete[1] === 0, 'a node with no place is NaN');
    check(q.score[0] === 0.3 && q.score[1] === 0.8, 'a node with no score of its own takes the instance\'s', Array.from(q.score));
    check(q.score[2] === 0.4, 'and so does every node when there are no node scores', Array.from(q.score));
    check(source.pointsData.x[1] === 5 && source.pointsData.visible[1] === 1, 'a user instance\'s points');
    const blocks = sourceRowsByVideo(source);
    check(blocks.length === 1 && blocks[0].raw === 1, 'it splits by video like any store');
}

console.log(fails ? `\n${fails} check(s) FAILED` : '\nAll checks passed');
process.exit(fails ? 1 : 0);
