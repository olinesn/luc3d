/**
 * test-behavior-rules.mjs — rule detectors and the predictions they make,
 * away from the UI.
 *
 * pose/behavior-rules.js decides, frame by frame, whether each behavior's
 * conditions hold; pose/behavior-postprocess.js turns that into bouts; and
 * pose/behavior-predictions.js keeps them for review. Pinned here:
 *   1. detectors: normalization, problems, templates (with the head role);
 *   2. scores: exact decisions at the threshold, the soft margin around it,
 *      NaN = unscored;
 *   3. scores to bouts: gaps joined before short bouts are dropped; a point
 *      behavior's bout is its peak frame;
 *   4. a whole session scored chunk by chunk equals one pass, and a mutual
 *      behavior takes the better direction; a planted rear gives exact bouts;
 *   5. the editor's numbers: histogram, live count;
 *   6. predictions: runs, review status from labels, Accept / Reject as one
 *      undo step each, review order, staleness (settings and poses), renames,
 *      save / load.
 *
 * ESM, picked up by `tests/run-mjs-tests.mjs`.
 */

import {
    normalizeDetector, detectorProblems, DETECTOR_TEMPLATES, instantiateTemplate, scoreConditions, detectorBouts,
    planDetectors, scoreDetectorsChunk, featureHistogram, previewDetector, combineMutual, conditionScale,
} from '../pose/behavior-rules.js';
import { scoresToBouts, secondsToFrames } from '../pose/behavior-postprocess.js';
import { featureCatalog, featurePadFrames, planChunks, sliceChunkData } from '../pose/behavior-features.js';
import { resolveBody, floorFrame, featureSpec } from '../pose/behavior-tensor.js';
import { resetEthogram, addBehavior, setBehaviorDetector, setFeatureSettings, behaviorById } from '../pose/behavior-ethogram.js';
import { undoBehaviorEdit, listBouts, labelRange } from '../pose/behavior-labels.js';
import {
    setBehaviorRun, noteDetectionPass, predictedBouts, hasPredictions, reviewStatus, reviewQueue, acceptPredictions,
    rejectPredictions, runConfigPrint, predictionStaleness, predictionsStand, notePossibleInputChange,
    checkPredictionInput, predictionInputPrint, renamePredictedAnimal, exportPredictionRuns, importPredictionRuns,
    clearBehaviorRuns, predictionScoreAt,
} from '../pose/behavior-predictions.js';
import { Session, Skeleton, InstanceGroup } from '../pose/pose-data.js';

let passed = 0, failed = 0;
const check = (cond, msg) => {
    if (cond) { passed++; console.log('  ok   ' + msg); }
    else { failed++; console.log('  FAIL ' + msg); }
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const NODES = ['Nose', 'Head', 'Neck', 'Trunk', 'TTI'];
const EDGES = [[0, 1], [1, 2], [2, 3], [3, 4]];
const ALONG = [45, 30, 15, 0, -40];
const HEIGHT = [15, 25, 30, 30, 15];

// ---------------------------------------------------------------------------
console.log('\n-- 1. detectors --');
{
    const d = normalizeDetector({ conditions: [{ feature: ' Nose height ', op: '>', value: '0.5' }, { feature: '', op: '<', value: 1 },
        { feature: 'x', op: '!', value: 'abc' }], minDuration: -2, maxGap: 0.25 });
    check(eq(d, { conditions: [{ feature: 'Nose height', op: '>', value: 0.5 }], minDuration: 0, maxGap: 0.25 }),
        'normalized: names trimmed, numbers parsed, bad conditions and negative durations dropped');
    check(normalizeDetector({ conditions: [] }) === null && normalizeDetector(null) === null, 'no condition: no detector');
    const spec = { mode: '3d', nodeNames: NODES, edges: EDGES, roles: { head: 0, tailBase: 4 }, used: [0, 1, 2, 3, 4], paws: [] };
    const cat = featureCatalog(spec);
    check(detectorProblems({ conditions: [{ feature: 'Nose height', op: '>', value: 1 }] }, cat, false).length === 0, 'a known feature: no problem');
    const p = detectorProblems({ conditions: [{ feature: 'min node distance', op: '<', value: 1 }] }, cat, false);
    check(p.length === 1 && /make the behavior social/.test(p[0]), 'a pair feature on a non-social behavior is flagged, saying why');
    const cat2d = featureCatalog(Object.assign({}, spec, { mode: '2d' }));
    check(/heights need 3D/.test(detectorProblems({ conditions: [{ feature: 'Nose height', op: '>', value: 1 }] }, cat2d, false)[0]),
        'a height in 2D is flagged');
    const ids = DETECTOR_TEMPLATES.map(t => t.id);
    check(['rearing', 'immobile', 'locomotion', 'contact', 'nose-to-nose', 'anogenital', 'following'].every(id => ids.includes(id)),
        'templates: rearing, immobile, locomotion, contact, nose-to-nose, anogenital approach, following');
    const rear = instantiateTemplate(DETECTOR_TEMPLATES.find(t => t.id === 'rearing'), { head: 'Nose', tailBase: 'TTI' });
    check(rear.detector && rear.detector.conditions[0].feature === 'Nose height · mean ±0.1 s', 'the rearing template takes the head node');
    check(rear.detector.conditions[1].feature === 'TTI height · mean ±0.1 s' && rear.detector.conditions[1].op === '<' && rear.detector.conditions[1].value === 0.45,
        'and the tail base below 0.45 BL: a mouse climbing the wall is not rearing');
    check(/needs a tail base node/.test(instantiateTemplate(DETECTOR_TEMPLATES.find(t => t.id === 'rearing'), { head: 'Nose' }).error),
        'without a tail base it says so, in words');
    check(rear.detector.conditions[0].value === 0.7, 'rearing: the head more than 0.7 BL above the floor (below it, few bouts on real recordings were clear rears)');
    const n2n = DETECTOR_TEMPLATES.find(t => t.id === 'nose-to-nose').detector;
    check(n2n.minDuration === 0.2 && n2n.maxGap === 0.2, 'nose-to-nose: at least 0.2 s, gaps of 0.2 s joined (like contact and anogenital approach)');
    const fol = DETECTOR_TEMPLATES.find(t => t.id === 'following').detector.conditions;
    check(fol.some(c => c.feature === 'relative heading' && c.op === '<' && c.value === 60),
        'following: the two headed within 60° of each other (crossing or meeting head-on is not following)');
    check(DETECTOR_TEMPLATES.find(t => t.id === 'immobile').detector.conditions[0].value === 0.1, 'immobile: still below 0.1 BL/s');
    check(/head node/.test(instantiateTemplate(DETECTOR_TEMPLATES.find(t => t.id === 'rearing'), {}).error), 'without a head it says so');
    const every = DETECTOR_TEMPLATES.every(t => {
        const r = instantiateTemplate(t, { head: 'Nose', tailBase: 'TTI' });
        return r.detector && detectorProblems(r.detector, cat, t.social).length === 0;
    });
    check(every, 'every template is valid on a 3D skeleton');
}

// ---------------------------------------------------------------------------
console.log('\n-- 2. scores --');
{
    const conds = [{ op: '>', value: 0.5, unit: 'BL' }, { op: '<', value: 2, unit: 'BL/s' }];
    const a = Float32Array.from([0.4, 0.5, 0.6, 0.9, NaN, 0.9]);
    const b = Float32Array.from([1, 1, 1, 2.5, 1, NaN]);
    const s = scoreConditions(conds, [a, b], 6);
    check(s[0] < 0.5 && s[1] >= 0.5 && s[2] >= 0.5 && s[3] < 0.5, 'at least 0.5 exactly where every condition holds (threshold inclusive)');
    check(s[4] !== s[4] && s[5] !== s[5], 'a missing feature: unscored');
    check(s[2] > 0.5 && s[2] < 0.99 && s[1] === 0.5, 'near the threshold the score is near 0.5');
    check(Math.abs(conditionScale('BL', 0.5) - 0.05) < 1e-12 && conditionScale('deg', 0) === 1, 'the margin scale: 10% of the threshold, with a floor per unit');
    const far = scoreConditions([{ op: '>', value: 0.5, unit: 'BL' }], [Float32Array.from([5])], 1);
    check(far[0] > 0.999, 'far past the threshold: about 1');
}

// ---------------------------------------------------------------------------
console.log('\n-- 3. scores to bouts --');
{
    const on = (pattern) => Float32Array.from(pattern.split('').map(c => c === '1' ? 0.9 : c === '?' ? NaN : 0.1));
    check(eq(scoresToBouts(on('0011100111000')).map(b => [b[0], b[1]]), [[2, 4], [7, 9]]), 'runs of detected frames');
    check(eq(scoresToBouts(on('0011100111000'), { maxGapFrames: 2 }).map(b => [b[0], b[1]]), [[2, 9]]), 'a gap of 2 joined (maxGap 2)');
    check(eq(scoresToBouts(on('0011?00111000'), { maxGapFrames: 3 }).map(b => [b[0], b[1]]), [[2, 9]]), 'unscored frames count as gap');
    check(eq(scoresToBouts(on('0100111110001'), { minFrames: 3 }).map(b => [b[0], b[1]]), [[4, 8]]), 'bouts shorter than minFrames dropped');
    check(eq(scoresToBouts(on('0111011100'), { minFrames: 6, maxGapFrames: 1 }).map(b => [b[0], b[1]]), [[1, 7]]),
        'joined first: two runs of 3 make one bout of 6 detected frames (each alone too short)');
    check(scoresToBouts(on('0111011100'), { minFrames: 7, maxGapFrames: 1 }).length === 0,
        'the gap it bridged does not count toward the minimum (6 detected frames < 7, though it spans 7)');
    check(scoresToBouts(on('01000001000001'), { minFrames: 4, maxGapFrames: 6 }).length === 0,
        'scattered one-frame detections never add up to a bout');
    const peak = scoresToBouts(Float32Array.from([0.1, 0.6, 0.9, 0.7, 0.2]), { point: true });
    check(eq(peak, [[2, 2, Math.fround(0.9)]]), 'a point behavior: one frame, at the peak');
    const sc = scoresToBouts(Float32Array.from([0.6, 0.8, 1.0]));
    check(Math.abs(sc[0][2] - 0.8) < 1e-6, 'a bout scores the mean of its frames');
    check(secondsToFrames(0.25, 30) === 8 && secondsToFrames(-1, 30) === 0, 'seconds to frames at the session fps');
    check(eq(detectorBouts(on('0111011100'), { minDuration: 0.2, maxGap: 0.05 }, 20).map(b => [b[0], b[1]]), [[1, 7]]),
        'detector durations are seconds: 0.05 s gap = 1 frame at 20 fps, 0.2 s = 4 frames');
}

// ---------------------------------------------------------------------------
// A small world: two animals on the floor; animal 0 rears on 300–359; they
// meet nose to nose on 600–650 (animal 1 comes to face animal 0).
function makeWorld(T) {
    const N = NODES.length;
    const data = [new Float32Array(T * N * 3), new Float32Array(T * N * 3)];
    for (let t = 0; t < T; t++) {
        for (let a = 0; a < 2; a++) {
            let cx, cy, hd;
            if (t >= 600 && t <= 650) { cx = a === 0 ? 0 : 100; cy = 0; hd = a === 0 ? 0 : Math.PI; }
            else { cx = a * 500 + 100 * Math.cos(t / 50); cy = 100 * Math.sin(t / 50); hd = t / 50 + Math.PI / 2; }
            for (let n = 0; n < N; n++) {
                const o = (t * N + n) * 3;
                const rear = a === 0 && t >= 300 && t < 360 && n <= 2;
                data[a][o] = cx + ALONG[n] * Math.cos(hd);
                data[a][o + 1] = cy + ALONG[n] * Math.sin(hd);
                data[a][o + 2] = rear ? [90, 80, 60][n] : HEIGHT[n];
            }
        }
    }
    return { mode: '3d', dim: 3, numFrames: T, nodeNames: NODES, edges: EDGES, animals: ['A', 'B'], data };
}

console.log('\n-- 4. detectors on a session --');
const world = makeWorld(1000);
const body = resolveBody(NODES, EDGES, null);
const spec = featureSpec(world, body, floorFrame([0, 0, 0], [0, 0, 1]), 30);
const cat = featureCatalog(spec);
const behaviorsIn = [
    { behaviorId: 'rear', social: false, mutual: false, detector: { conditions: [{ feature: 'Nose height', op: '>', value: 0.5 }], minDuration: 0, maxGap: 0 } },
    { behaviorId: 'n2n', social: true, mutual: true, detector: { conditions: [{ feature: 'head–head distance', op: '<', value: 0.3 }], minDuration: 0, maxGap: 0 } },
    { behaviorId: 'face', social: true, mutual: false, detector: { conditions: [{ feature: 'target bearing', op: '<', value: 30 },
        { feature: 'centroid distance', op: '<', value: 2 }], minDuration: 0, maxGap: 0 } },
];
const plan = planDetectors(cat, behaviorsIn);
check(plan.plans.length === 3 && plan.skipped.length === 0, 'three detectors planned');
check(eq(plan.request.animal.sort(), ['Nose height']) && eq(plan.request.pair.sort(), ['centroid distance', 'head–head distance', 'target bearing']),
    'only the features they read are requested');
const one = scoreDetectorsChunk(spec, { start: 0, end: 1000, padStart: 0, padEnd: 1000, data: world.data }, plan);
const merged = {};
for (const ch of planChunks(1000, featurePadFrames(spec), 97)) {
    const r = scoreDetectorsChunk(spec, Object.assign({}, ch, { data: sliceChunkData(world.data, ch, NODES.length * 3) }), plan);
    for (const bid of Object.keys(r.scores)) for (const k of Object.keys(r.scores[bid])) {
        merged[bid] = merged[bid] || {};
        if (!merged[bid][k]) merged[bid][k] = new Float32Array(1000);
        merged[bid][k].set(r.scores[bid][k], ch.start);
    }
}
let same = true;
for (const bid of Object.keys(one.scores)) for (const k of Object.keys(one.scores[bid])) {
    const x = one.scores[bid][k], y = merged[bid][k];
    for (let t = 0; t < 1000; t++) if (!(x[t] === y[t] || (x[t] !== x[t] && y[t] !== y[t]) || Math.abs(x[t] - y[t]) < 1e-6)) { same = false; break; }
}
check(same, 'scored in chunks of 97 frames = one pass');
check(eq(Object.keys(one.scores.rear).sort(), ['0', '1']) && eq(Object.keys(one.scores.n2n), ['0>1']) &&
    eq(Object.keys(one.scores.face).sort(), ['0>1', '1>0']), 'keys: animals; a mutual pair once (a < b); directed pairs both ways');
const rearBouts = detectorBouts(one.scores.rear['0'], behaviorsIn[0].detector, 30);
check(eq(rearBouts.map(b => [b[0], b[1]]), [[300, 359]]) && detectorBouts(one.scores.rear['1'], behaviorsIn[0].detector, 30).length === 0,
    'a planted rear (frames 300–359) gives exactly that bout, for that animal only');
check(eq(detectorBouts(one.scores.n2n['0>1'], behaviorsIn[1].detector, 30).map(b => [b[0], b[1]]), [[600, 650]]), 'nose-to-nose: the meeting');
const f01 = detectorBouts(one.scores.face['0>1'], behaviorsIn[2].detector, 30).map(b => [b[0], b[1]]);
const f10 = detectorBouts(one.scores.face['1>0'], behaviorsIn[2].detector, 30).map(b => [b[0], b[1]]);
check(eq(f01, [[600, 650]]) && eq(f10, [[600, 650]]), 'facing each other: both directions');
const cm = combineMutual(Float32Array.from([0.2, NaN, 0.7, NaN]), Float32Array.from([0.6, 0.4, 0.1, NaN]));
check(Math.abs(cm[0] - 0.6) < 1e-6 && Math.abs(cm[1] - 0.4) < 1e-6 && Math.abs(cm[2] - 0.7) < 1e-6 && cm[3] !== cm[3],
    'a mutual pair takes the better direction (and a direction it has)');

// ---------------------------------------------------------------------------
console.log('\n-- 5. the editor numbers --');
{
    const cols = [Float32Array.from([1, 2, 3, NaN]), Float32Array.from([4, 5])];
    const h = featureHistogram(cols, 10, 10);
    check(h && h.n === 5 && h.hi > 10, 'histogram: finite values pooled; the range includes the threshold');
    const nose = scoreDetectorsChunk(spec, { start: 0, end: 1000, padStart: 0, padEnd: 1000, data: world.data },
        planDetectors(cat, []), { columns: { animal: ['Nose height'] } }).columns.animal.map(c => c['Nose height']);
    const live = previewDetector(nose.map(c => [c]), [{ op: '>', value: 0.5, unit: 'BL' }], { minDuration: 0, maxGap: 0 }, 30);
    check(live.bouts === 1 && live.frames === 60, 'live count: 1 bout, 60 frames');
    const live2 = previewDetector(nose.map(c => [c]), [{ op: '>', value: 0.5, unit: 'BL' }], { minDuration: 3, maxGap: 0 }, 30);
    check(live2.bouts === 0, 'a minimum of 3 s drops it');
}

// ---------------------------------------------------------------------------
console.log('\n-- 6. predictions --');
{
    resetEthogram();
    const rear = addBehavior({ name: 'rear', key: 'r' }).behavior;
    const n2n = addBehavior({ name: 'nose-to-nose', key: 'n', social: true, mutual: true }).behavior;
    setBehaviorDetector(rear.id, { conditions: [{ feature: 'Nose height', op: '>', value: 0.5 }] });
    setBehaviorDetector(n2n.id, { conditions: [{ feature: 'head–head distance', op: '<', value: 0.3 }] });
    const session = { name: 'S', skeleton: { nodes: NODES, edges: EDGES } };
    check(!hasPredictions(session) && predictedBouts(session).length === 0, 'nothing predicted yet');
    const scores = new Map([['A\u0001', Float32Array.from([0.1, 0.9, 0.9, 0.2])]]);
    setBehaviorRun(session, { behaviorId: rear.id, cfg: runConfigPrint(rear, session, { fps: 30, mode: '3d' }), fps: 30, mode: '3d', camera: null,
        bouts: [{ animal: 'A', target: '', start: 10, end: 20, score: 0.55 }, { animal: 'B', target: '', start: 30, end: 40, score: 0.95 },
                { animal: 'A', target: '', start: 50, end: 52, score: 0.7 }], scores });
    setBehaviorRun(session, { behaviorId: n2n.id, cfg: runConfigPrint(n2n, session, { fps: 30, mode: '3d' }), fps: 30, mode: '3d',
        bouts: [{ animal: 'B', target: 'A', start: 60, end: 70, score: 0.8 }] });
    noteDetectionPass(session, 'print-1');
    check(hasPredictions(session) && predictedBouts(session).length === 4, 'runs stored: four predicted bouts');
    check(eq(predictedBouts(session, { behaviorId: n2n.id }).map(b => [b.animal, b.target]), [['A', 'B']]), 'a mutual pair is stored in name order');
    check(predictedBouts(session, { animal: 'B' }).length === 2, 'filter by animal: as actor, or either animal of a mutual pair');
    check(Math.abs(predictionScoreAt(session, rear.id, 'A', '', 1) - 0.9) < 1e-6 && predictionScoreAt(session, rear.id, 'B', '', 1) !== predictionScoreAt(session, rear.id, 'B', '', 1),
        'per-frame scores are kept in memory');
    check(predictionsStand(session, { fps: 30 }), 'they stand');
    // Review order.
    check(eq(reviewQueue(session, {}).map(b => b.start), [10, 50, 60, 30]), 'most uncertain first (score nearest 0.5)');
    check(eq(reviewQueue(session, { order: 'time' }).map(b => b.start), [10, 30, 50, 60]), 'or in time order');
    // Accept / reject.
    const q = reviewQueue(session, { order: 'time' });
    acceptPredictions(session, [q[0]]);
    check(eq(listBouts(session).map(b => [b.start, b.end, b.present, b.source]), [[10, 20, true, 'a']]), 'Accept labels it present, source "a"');
    check(reviewStatus(session, q[0]).status === 'accepted', 'its review status reads back: accepted');
    rejectPredictions(session, [q[1]]);
    check(reviewStatus(session, q[1]).status === 'rejected' && listBouts(session, { present: false }).length === 1, 'Reject labels it "not": rejected');
    check(eq(reviewQueue(session, { unreviewedOnly: true }).map(b => b.start), [50, 60]), 'reviewed bouts leave the to-review list');
    labelRange(session, rear.id, 'A', '', 50, 50, false);
    const mixed = reviewQueue(session, { order: 'disagreement' });
    check(mixed[0].start === 30 && reviewStatus(session, q[2]).status === 'mixed', 'most contradicted first; partly labelled = mixed');
    undoBehaviorEdit(session); undoBehaviorEdit(session);
    check(reviewStatus(session, q[1]).status === 'unreviewed', 'one undo takes back one Reject');
    acceptPredictions(session, q);
    check(listBouts(session, { present: true }).length === 4, 'Accept many…');
    undoBehaviorEdit(session);
    check(listBouts(session, { present: true }).length === 1, '…is one undo step');
    // Staleness: settings.
    setBehaviorDetector(rear.id, { conditions: [{ feature: 'Nose height', op: '>', value: 0.6 }] });
    let st = predictionStaleness(session, { fps: 30 });
    check(eq(st.settings, [rear.id]) && !st.input && predictionsStand(session, { fps: 30 }), 'a changed detector: that behavior is stale (the other still stands)');
    setFeatureSettings({ nodeRoles: { head: 'Nose' } });
    check(predictionStaleness(session, { fps: 30 }).settings.length === 1, 'the head set to the node already guessed: nothing more is stale');
    setFeatureSettings({ nodeRoles: { head: 'Head' } });
    st = predictionStaleness(session, { fps: 30 });
    check(st.settings.length === 2 && !predictionsStand(session, { fps: 30 }), 'another head node: every run is stale, nothing stands');
    setFeatureSettings(null);
    check(predictionStaleness(session, { fps: 30, floorKey: 'origin:0,0,0:0,0,1' }).settings.length === 2, 'a floor that moved (another key): stale');
    check(predictionStaleness(session, { fps: 25 }).settings.length === 2, 'another frame rate: stale too');
    setBehaviorDetector(rear.id, { conditions: [{ feature: 'Nose height', op: '>', value: 0.5 }] });
    check(predictionStaleness(session, { fps: 30 }).settings.length === 0, 'settings back as they were: fresh again');
    // Renames.
    renamePredictedAnimal(session, 'A', 'Zed');
    check(eq(predictedBouts(session, { behaviorId: n2n.id }).map(b => [b.animal, b.target]), [['B', 'Zed']]) &&
        predictedBouts(session, { animal: 'Zed' }).length === 3 && Math.abs(predictionScoreAt(session, rear.id, 'Zed', '', 1) - 0.9) < 1e-6,
        'a renamed identity carries its predictions and scores (a mutual pair re-sorted)');
    // Save / load.
    const payload = JSON.parse(JSON.stringify(exportPredictionRuns(session)));
    check(payload.v === 1 && payload.runs.length === 2 && payload.runs[0].bouts[0].length === 5 && payload.runs[0].fps === 30,
        'saved: v1, one run per behavior, [animal, target, start, end, score] rows, fps');
    const fresh = { name: 'S2', skeleton: session.skeleton };
    const r = importPredictionRuns(fresh, payload, new Map());
    check(r.kept === 4 && r.dropped === 0 && eq(predictedBouts(fresh), predictedBouts(session)), 'read back exactly');
    check(predictionsStand(fresh, { fps: 30 }), 'read back: they stand (same detectors and settings)');
    const remapped = { name: 'S3' };
    importPredictionRuns(remapped, { v: 1, runs: [{ behavior: 'b99', bouts: [['A', '', 1, 2, 0.5]] }, { behavior: n2n.id, bouts: [['A', 'B', 5, 3, 1], 'x'] }] }, new Map([['b99', rear.id]]));
    check(predictedBouts(remapped).length === 1 && predictedBouts(remapped)[0].behaviorId === rear.id, 'ids remapped; garbage rows dropped');
    check(importPredictionRuns({}, null).kept === 0 && importPredictionRuns({}, { v: 2 }).kept === 0, 'garbage payloads ignored');
    clearBehaviorRuns(fresh);
    check(!hasPredictions(fresh) && exportPredictionRuns(fresh) === null, 'cleared: nothing to save');
}

console.log('\n-- 7. the poses a run read --');
{
    resetEthogram();
    const rear = addBehavior({ name: 'rear' }).behavior;
    const session = new Session([], new Skeleton('s', NODES, EDGES), [], 'P');
    const id0 = session.addIdentity('A').id, id1 = session.addIdentity('B').id;
    for (let t = 0; t < 50; t++) {
        session.instanceGroups.set(t, [id0, id1].map((id, k) => {
            const g = new InstanceGroup(t * 2 + k, id);
            g.points3d = Float64Array.from({ length: 15 }, (_, i) => t + i + 100 * k);
            return g;
        }));
    }
    const p1 = predictionInputPrint(session, '3d');
    const reordered = new Session([], new Skeleton('s', NODES, EDGES), [], 'P');
    reordered.identities = session.identities;
    Array.from(session.instanceGroups.keys()).reverse().forEach(t => reordered.instanceGroups.set(t, session.instanceGroups.get(t).slice().reverse()));
    check(predictionInputPrint(reordered, '3d') === p1, 'the print does not depend on visiting order');
    setBehaviorRun(session, { behaviorId: rear.id, cfg: runConfigPrint(rear, session, { fps: 30, mode: '3d' }), fps: 30, mode: '3d', bouts: [{ animal: 'A', start: 1, end: 2, score: 1 }] });
    noteDetectionPass(session, p1);
    notePossibleInputChange(session);
    checkPredictionInput(session);
    check(predictionsStand(session, { fps: 30 }), 'an edit that changed no pose: still fresh');
    session.instanceGroups.get(20)[0].points3d[4] += 0.5;
    notePossibleInputChange(session);
    checkPredictionInput(session);
    check(predictionStaleness(session, { fps: 30 }).input && !predictionsStand(session, { fps: 30 }), 'a pose moved half a unit: stale');
    session.instanceGroups.get(20)[0].points3d[4] -= 0.5;
    notePossibleInputChange(session);
    checkPredictionInput(session);
    check(!predictionStaleness(session, { fps: 30 }).input, 'moved back: fresh');
    const g = session.instanceGroups.get(30);
    const tmp = g[0].identityId; g[0].identityId = g[1].identityId; g[1].identityId = tmp;
    notePossibleInputChange(session);
    checkPredictionInput(session);
    check(predictionStaleness(session, { fps: 30 }).input, 'two identities swapped on a frame: stale');
}

console.log('\n' + passed + ' passed, ' + failed + ' failed');
process.exit(failed ? 1 : 0);
