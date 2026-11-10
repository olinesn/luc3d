/**
 * test-model-check.mjs — reading a model check's scan.json and matching it to a
 * project (pose/model-check.js): Python's bare NaN, malformed rows, the
 * `read_on` text, matching scan cameras by video then camera name (never on a
 * generic campy `0.mp4` alone), sorting and filtering, and matching a row's
 * `animal` to an instance in file order, checked against node positions.
 * The tab itself is covered by tests/e2e/model-check-review.mjs.
 *
 * Run:  node tests/test-model-check.mjs
 */
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const mc = await import(pathToFileURL(path.join(HERE, '..', 'pose', 'model-check.js')).href);

let passed = 0, failed = 0; const failures = [];
function eq(a, e, m) {
    if (a === e) passed++;
    else { failed++; failures.push(`${m} (expected ${JSON.stringify(e)}, got ${JSON.stringify(a)})`); console.error('  ✗ ' + m); }
}
function ok(c, m) { eq(!!c, true, m); }
function throws(fn, re, m) {
    try { fn(); failed++; failures.push(m + ' (did not throw)'); console.error('  ✗ ' + m); }
    catch (e) { ok(re.test(e.message), m + ' — message: ' + e.message); }
}

// ---- jsonWithoutNaN / parseModelCheck ---------------------------------------
eq(mc.jsonWithoutNaN('[NaN, -Infinity, Infinity, "NaN in a string", "a\\"NaN"]'),
    '[null, null, null, "NaN in a string", "a\\"NaN"]', 'bare NaN/Infinity become null; strings untouched (escaped quote too)');

const row = (o) => Object.assign({
    video_index: 0, video: 'cam1.mp4', camera: 'Cam1', frame: 10, animal: 0, split: 'held-out',
    median_px: 3, max_px: 4, bar_px: 10, misread: false, misread_nodes: {},
}, o);
const text = JSON.stringify([
    row({ animal: 0 }),
    row({ animal: 1, misread: true, max_px: 30, misread_nodes: {
        Ear_L: { px: 30, cause: 'neighbour_same', read_on: "animal 0's Ear_L (2.0 px)" },
        Nose: { px: 12, cause: 'background', read_on: null },
    } }),
    { video: 'x.mp4' },                        // no frame: skipped
    'garbage',
]).replace('"max_px":4', '"max_px":NaN');
const parsed = mc.parseModelCheck(text);
eq(parsed.rows.length, 2, 'two usable rows');
eq(parsed.skipped, 2, 'malformed rows are counted, not fatal');
ok(Number.isNaN(parsed.rows[0].maxPx), 'NaN max_px reads as NaN (never detected)');
eq(parsed.rows[1].nodes[0].node, 'Ear_L', 'misread nodes sorted worst first');
eq(parsed.rows[1].nodes[1].ref, null, 'background read_on has no reference');
eq(parsed.rows[1].nodes[0].ref.animal, 0, "read_on names animal 0");
eq(mc.parseModelCheck(JSON.stringify({ rows: [row({})] })).rows.length, 1, '{rows: [...]} is accepted too');
throws(() => mc.parseModelCheck('{"a": 1}'), /not a model check/, 'an object without rows is refused');
throws(() => mc.parseModelCheck('not json'), /not a JSON file/, 'not JSON');
throws(() => mc.parseModelCheck('[{"video": "a"}]'), /No rows/, 'nothing usable');
const missingSplit = mc.parseModelCheck(JSON.stringify([row({ split: 'weird' })])).rows[0];
eq(missingSplit.split, 'unknown', 'unknown split names read as unknown');

// ---- parseReadOn -------------------------------------------------------------
const r1 = mc.parseReadOn("animal 2's Ear_L (5.4 px)");
eq(JSON.stringify(r1), JSON.stringify({ own: false, animal: 2, node: 'Ear_L', px: 5.4 }), 'neighbour reference');
const r2 = mc.parseReadOn('its own Shoulder right (10.0 px)');
eq(r2.own && r2.node === 'Shoulder right' && r2.px === 10, true, 'own reference, node name with a space');
eq(mc.parseReadOn(null), null, 'null');
eq(mc.parseReadOn('somewhere'), null, 'unknown text');

// ---- describeNode --------------------------------------------------------------
const nodes = parsed.rows[1].nodes;
eq(mc.describeNode(nodes[0]), "Ear_L read on animal 0's Ear_L", 'neighbour sentence');
eq(mc.describeNode(nodes[0], (b) => 'id_' + (b + 3)), "Ear_L read on animal 0 (id_3)'s Ear_L", 'with the project name');
eq(mc.describeNode(nodes[1]), 'Nose read on the background', 'background sentence');
eq(mc.describeNode({ node: 'Neck', cause: 'imprecise', ref: { own: true, node: 'Neck', px: 10.2 } }),
    'Neck read near its own label, just past the bar', 'imprecise sentence');
eq(mc.describeNode({ node: 'Neck', cause: 'swap', ref: { own: true, node: 'Ear', px: 1 } }), 'Neck read on its own Ear', 'own sentence');

// ---- matchScanCameras ------------------------------------------------------------
const scan = (video, camera, videoIndex) => ({ key: videoIndex + '|' + video + '|' + camera, video, camera, videoIndex });
const P = [
    { id: 's0:Camera1_topB', camera: 'Camera1_topB', videos: ['[2026]-s#0066-Camera1_topB.mp4'] },
    { id: 's0:Camera2_topC', camera: 'Camera2_topC', videos: [] },
    { id: 's1:Camera2_topC', camera: 'Camera2_topC', videos: [] },
    { id: 's0:Cam5', camera: 'Cam5', videos: ['other-recording.mp4'] },
    { id: 's0:cam0', camera: 'cam0', videos: ['0.mp4'] },
    { id: 's1:cam0', camera: 'cam0', videos: ['0.mp4'] },
    { id: 's0:cam9', camera: 'cam9', videos: ['0.mp4'] },
    { id: 's0:Side', camera: 'Side', videos: [] },
];
const S = [
    scan('[2026]-s#0066-Camera1_topB.mp4', 'renamedFolder', 1),   // video name wins over the folder
    scan('x.mp4', 'Camera2_topC', 2),                             // camera in two sessions: ask
    scan('y.mp4', 'Cam5', 3),                                     // camera's video differs: ask
    scan('0.mp4', 'cam9', 4),                                     // campy: generic name never matches alone; camera does
    scan('0.mp4', 'cam0', 5),                                     // campy, two sessions: ask
    scan('z.mp4', 'Camera0_mid', 6),                              // nothing
    scan('q.mp4', 'SIDE', 7),                             // case-insensitive camera fallback
];
const M = mc.matchScanCameras(S, P);
const st = (k) => M.get(S[k].key);
eq(st(0).status + ':' + st(0).cam.id + ':' + st(0).how, 'matched:s0:Camera1_topB:video', 'distinctive video name');
eq(st(1).status + ':' + st(1).candidates.length, 'ambiguous:2', 'same camera name in two sessions is asked');
eq(st(2).status, 'ambiguous', "the camera's known video is another file: asked");
ok(/other-recording/.test(st(2).reason), 'the reason names the project video');
eq(st(3).status + ':' + st(3).cam.id + ':' + st(3).how, 'matched:s0:cam9:camera', '0.mp4 matches by camera, not video');
eq(st(4).status + ':' + st(4).candidates.length, 'ambiguous:2', 'campy 0.mp4 in two sessions: asked, not matched on the name');
eq(st(5).status, 'unmatched', 'no camera, no video');
ok(/Camera0_mid/.test(st(5).reason), 'unmatched reason names the camera');
eq(st(6).status + ':' + (st(6).cam && st(6).cam.id), 'matched:s0:Side', 'case-insensitive camera name');
// A video name the scan uses for two cameras is not distinctive.
const dup = mc.matchScanCameras([scan('a.mp4', 'Cam5', 0), scan('a.mp4', 'Camera1_topB', 1)],
    [{ id: 'A', camera: 'Cam5', videos: ['a.mp4'] }, { id: 'B', camera: 'Camera1_topB', videos: [] }]);
eq([...dup.values()].map((m) => m.status + ':' + (m.cam && m.cam.id)).join(','), 'matched:A,matched:B',
    'a repeated video name falls back to the camera name');

// ---- reviewEntries: filter + worst first ----------------------------------------------
const rows = mc.parseModelCheck(JSON.stringify([
    row({ animal: 0, misread: true, max_px: 12, split: 'training', misread_nodes: { Nose: { px: 12, cause: 'swap', read_on: 'its own Ear (1.0 px)' } } }),
    row({ animal: 1, misread: true, max_px: 40, misread_nodes: { Nose: { px: 40, cause: 'background', read_on: null } } }),
    row({ animal: 2, misread: true, max_px: NaN }),
    row({ animal: 3, misread: false, max_px: 3 }),
]).replace(/"max_px":null/, '"max_px":NaN')).rows;
eq(mc.reviewEntries(rows).map((r) => r.animal).join(','), '2,1,0', 'misread only, never-detected first, then worst error');
eq(mc.reviewEntries(rows, { split: 'training' }).map((r) => r.animal).join(','), '0', 'split filter');
eq(mc.reviewEntries(rows, { cause: 'background' }).map((r) => r.animal).join(','), '1', 'cause filter');
eq(mc.reviewEntries(rows, { cause: 'not_detected' }).map((r) => r.animal).join(','), '2', 'not detected filter');
eq(mc.formatPx(32.5333), '32.5 px', 'formatPx');

// ---- matchAnimals ------------------------------------------------------------------------
// Three animals, labels as the sweep saw them (file order). Animal 1's Ear_L was
// read 20 px from its own Ear_L and 2.0 px from animal 0's Ear_L, so the two
// Ear_L labels are 18..22 px apart; its Nose 15 px off, 3.0 px from its own Ear_R.
const A0 = { Ear_L: [100, 100], Ear_R: [110, 100], Nose: [105, 90] };
const A1 = { Ear_L: [100, 120], Ear_R: [110, 120], Nose: [105, 135] };   // Ear_L 20 px from A0's
const A2 = { Ear_L: [300, 300], Ear_R: [310, 300], Nose: [305, 290] };
const frameRows = mc.parseModelCheck(JSON.stringify([
    row({ animal: 0 }),
    row({ animal: 1, misread: true, max_px: 20, misread_nodes: {
        Ear_L: { px: 20, cause: 'neighbour_same', read_on: "animal 0's Ear_L (2.0 px)" },
        Nose: { px: 15, cause: 'own_other', read_on: 'its own Ear_R (3.0 px)' },   // |Nose-Ear_R| = 15.8: in [12, 18]
    } }),
    row({ animal: 2, misread: true, max_px: 14, misread_nodes: { Nose: { px: 14, cause: 'background', read_on: null } } }),
])).rows;
const inst = (pts) => ({ points: pts });
let res = mc.matchAnimals(frameRows, [inst(A0), inst(A1), inst(A2)]);
eq(res.get(1).status + ':' + res.get(1).index + ':' + res.get(1).checks, 'verified:1:2', 'file order, both distances agree');
eq(res.get(2).status + ':' + res.get(2).index, 'order:2', 'background only: file order, nothing to check');
ok(!res.has(0), 'only misread rows are matched');

// The project holds the same animals in another order: positions find animal 1.
res = mc.matchAnimals(frameRows, [inst(A2), inst(A0), inst(A1)]);
eq(res.get(1).status + ':' + res.get(1).index, 'positions:2', 'reordered: the only instance whose nodes fit');
eq(res.get(2).status + ':' + res.get(2).index, 'order:2', 'no distance: file order is all there is (and is not verified)');

// An extra animal labelled since: counts differ.
const A3 = { Ear_L: [500, 500], Ear_R: [510, 500], Nose: [505, 490] };
res = mc.matchAnimals(frameRows, [inst(A0), inst(A1), inst(A2), inst(A3)]);
eq(res.get(1).status + ':' + res.get(1).index, 'positions:1', 'count differs: positions still find it');
eq(res.get(2).status, 'unmatched', 'count differs and no distance: unmatched');
ok(/counted 3 .* project has 4/.test(res.get(2).reason), 'reason says the counts');

// Labels moved since the check: nothing fits.
const moved = { Ear_L: [100, 160], Ear_R: [110, 160], Nose: [105, 175] };
res = mc.matchAnimals(frameRows, [inst(A0), inst(moved), inst(A2)]);
eq(res.get(1).status, 'unmatched', 'edited labels: unmatched');
ok(/edited since/.test(res.get(1).reason), 'reason suggests the labels were edited');

// A misread node missing from the instance in file order.
res = mc.matchAnimals(frameRows, [inst(A0), inst(A1), inst({ Ear_L: [300, 300] })]);
eq(res.get(2).status, 'unmatched', 'misread node not labelled: unmatched');
ok(/no label on Nose/.test(res.get(2).reason), 'reason names the node');

// Two instances fit equally: ambiguous, never a guess.
const twin = { Ear_L: [100, 120], Ear_R: [110, 120], Nose: [105, 135] };
res = mc.matchAnimals(frameRows, [inst(A0), inst(moved), inst(twin), inst(twin)]);
eq(res.get(1).status, 'unmatched', 'two fitting instances: unmatched');
ok(/fit 2 of the animals/.test(res.get(1).reason), 'reason says how many fit');

// No instance at all.
res = mc.matchAnimals(frameRows, []);
ok(/no hand-labelled animal/.test(res.get(1).reason), 'empty frame');

// Tolerance: a distance printed to 0.1 px still passes at the edge.
const edge = mc.parseModelCheck(JSON.stringify([
    row({ animal: 0 }),
    row({ animal: 1, misread: true, misread_nodes: { Ear_L: { px: 17.96, cause: 'neighbour_same', read_on: "animal 0's Ear_L (2.0 px)" } } }),
])).rows;
eq(mc.matchAnimals(edge, [inst(A0), inst(A1)]).get(1).status, 'verified', 'a 0.04 px rounding slack passes');

// ---- The region an entry zooms to ----
eq(mc.frameRegion([]), null, 'no box, no region');
eq(mc.frameRegion([null]), null, 'a box-less animal, no region');
// One 200 x 40 animal: a margin of 120 px each side (60% of its larger side).
let g = mc.frameRegion([{ x0: 100, y0: 200, x1: 300, y1: 240 }]);
eq([g.x0, g.x1, g.y0, g.y1].join(), [-20, 420, 80, 360].join(), 'margin of 60% of the larger side, centred');
// The animal and the neighbour its nodes were read on: both inside.
g = mc.frameRegion([{ x0: 100, y0: 100, x1: 160, y1: 400 }, { x0: 500, y0: 300, x1: 560, y1: 350 }]);
ok(g.x0 < 100 && g.x1 > 560 && g.y0 < 100 && g.y1 > 400, 'covers both animals');
eq([g.x0, g.y1].join(), [100 - 180, 400 + 180].join(), 'margin from the larger animal (300 px side)');
// A tiny animal still gets the minimum square.
g = mc.frameRegion([{ x0: 10, y0: 10, x1: 12, y1: 12 }]);
eq([g.x1 - g.x0, g.y1 - g.y0].join(), [mc.MODEL_CHECK_MIN_FRAME_PX, mc.MODEL_CHECK_MIN_FRAME_PX].join(), 'tiny animal: minimum square');

console.log(`\ntest-model-check: ${passed} passed, ${failed} failed`);
if (failed) { failures.forEach((f) => console.error(' - ' + f)); process.exit(1); }
