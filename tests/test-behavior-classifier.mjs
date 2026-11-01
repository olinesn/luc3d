/**
 * test-behavior-classifier.mjs — classifiers trained on behavior labels
 * (pose/behavior-classifier.js), end to end and inline, away from the app.
 *
 *   1. labels → classes: a behavior's own labels, plus "not" where an
 *      exclusive peer is present (own labels win); counts and the gate
 *      (20 frames and 2 bouts per class);
 *   2. grouped folds: by session, else animal, else contiguous time blocks
 *      whose training leaves out the frames near the held-out block — never
 *      random frames;
 *   3. metrics: frame P/R/F1, the PR curve's best threshold, bout P/R/F1;
 *   4. a planted kinematic behavior ("dart": a burst of speed) is learned:
 *      cross-validated F1 > 0.9 (by animal), and F1 > 0.9 on frames never
 *      labelled; speed features come out on top; retraining gives the same
 *      forest (hash); a second session makes it cross-validate by session;
 *   5. training rows are cached: retraining computes none again, new labels
 *      compute only theirs;
 *   6. the cap: above it, rows are sampled (the same rows every time), with
 *      every bout represented;
 *   7. a social (mutual) behavior trains on pair samples in both directions
 *      and cross-validates by time blocks (one pair);
 *   8. post-processing: exclusive groups keep the top behavior per frame;
 *   9. model runs: stale when the labels they learned from change, or when a
 *      newer model replaces theirs; saved with their hash and label print;
 *  10. model files round trip (same hash, same scores) and refuse a session
 *      whose body differs;
 *  11. active learning: uncertain stretches (unlabelled, near the threshold)
 *      and disagreements (missed / false) between labels and predicted bouts.
 *
 * ESM, picked up by `tests/run-mjs-tests.mjs`.
 */

import { Session, Skeleton, InstanceGroup } from '../pose/pose-data.js';
import { collectBehaviorInput, resolveBody, featureSpec, floorFrame, brokenPoseFrames } from '../pose/behavior-tensor.js';
import { catalogFor } from '../pose/behavior-features.js';
import { resetEthogram, addBehavior, behaviorById, setBehaviorPost, updateBehavior } from '../pose/behavior-ethogram.js';
import { labelRange, listBouts } from '../pose/behavior-labels.js';
import {
    labelClasses, labelCounts, gateProblems, planFolds, trainBehaviorClassifier, trainBehaviorClassifiers, predictModels,
    deriveBouts, modelBoutSettings, trainingLabelPrint, labelPrintOf, setBehaviorModel, modelHashOf, resetBehaviorModels,
    uncertainSegments, disagreementSegments, trainingRowsComputed, topFeatures, evaluateAt, GATE_FRAMES, GATE_BOUTS,
} from '../pose/behavior-classifier.js';
import { framePrf, prCurve, boutPrf } from '../pose/behavior-postprocess.js';
import {
    setBehaviorRun, predictionStaleness, exportPredictionRuns, importPredictionRuns, runConfigPrint, behaviorRunOf,
} from '../pose/behavior-predictions.js';
import { buildModelFile, parseModelFile, modelFileProblems, adoptModelFile } from '../import-export/behavior-model-io.js';
import { seededRandom } from '../pose/behavior-forest.js';

let passed = 0, failed = 0;
const check = (cond, msg) => {
    if (cond) { passed++; console.log('  ok   ' + msg); }
    else { failed++; console.log('  FAIL ' + msg); }
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const NODES = ['Nose', 'Head', 'Neck', 'Trunk', 'TTI', 'Tail1'];
const EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]];
const ALONG = [45, 30, 15, 0, -40, -70], H = [15, 25, 30, 30, 15, 5];
const FPS = 30;
const SEP = '\u0001';

/**
 * Two mice wandering in their own boxes; each "darts" (4× its walking speed)
 * for 0.5–1.5 s every 3–11 s. Returns the session and the truth per animal.
 */
function dartWorld(name, T, seed) {
    const rand = seededRandom(seed);
    const gauss = () => { let u = 0; while (!u) u = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand()); };
    const truth = [new Uint8Array(T), new Uint8Array(T)];
    for (let a = 0; a < 2; a++) {
        let t = 60 + Math.floor(rand() * 90);
        while (t < T - 60) {
            const len = 15 + Math.floor(rand() * 30);
            for (let k = t; k < Math.min(T, t + len); k++) truth[a][k] = 1;
            t += len + 90 + Math.floor(rand() * 240);
        }
    }
    const s = new Session([], new Skeleton('m', NODES, EDGES), [], name);
    s.addIdentity('id_0'); s.addIdentity('id_1');
    const pos = [[0, 0, 0], [600, 0, 1]];
    let gid = 1;
    for (let t = 0; t < T; t++) {
        s.instanceGroups.set(t, [0, 1].map(k => {
            const p = pos[k];
            const v = truth[k][t] ? 12 : 3;
            p[2] += (rand() - 0.5) * 0.15;
            p[0] += v * Math.cos(p[2]); p[1] += v * Math.sin(p[2]);
            const dx = p[0] - 600 * k, dy = p[1];
            if (Math.hypot(dx, dy) > 400) p[2] = Math.atan2(-dy, -dx) + (rand() - 0.5) * 0.5;
            const g = new InstanceGroup(gid++, s.identities[k].id);
            const pts = new Float64Array(NODES.length * 3);
            for (let n = 0; n < NODES.length; n++) {
                pts[3 * n] = p[0] + ALONG[n] * Math.cos(p[2]) + gauss() * 1.5;
                pts[3 * n + 1] = p[1] + ALONG[n] * Math.sin(p[2]) + gauss() * 1.5;
                pts[3 * n + 2] = H[n] + gauss();
            }
            g.points3d = pts;
            return g;
        }));
    }
    return { session: s, truth, T };
}

/** A behavior context for a session, as ui/behavior-detect.js builds one (floor z = 0). */
function contextOf(session, T) {
    const input = collectBehaviorInput(session, { numFrames: T });
    const body = resolveBody(input.nodeNames, input.edges, null);
    const spec = featureSpec(input, body, floorFrame([0, 0, 0], [0, 0, 1]), FPS);
    return { session, input, body, spec, catalog: catalogFor(spec), fps: FPS, broken: brokenPoseFrames(input, body, spec.bodyLength, FPS) };
}

/** Label frames [0, end) of each animal by the truth, as runs (present / "not"). */
function labelByTruth(session, behaviorId, truth, start, end) {
    truth.forEach((tr, a) => {
        let s = start;
        for (let t = start + 1; t <= end; t++) {
            if (t === end || tr[t] !== tr[s]) { labelRange(session, behaviorId, 'id_' + a, '', s, t - 1, !!tr[s]); s = t; }
        }
    });
}

function heldOutF1(scoreMap, truth, from) {
    let tp = 0, fp = 0, fn = 0;
    truth.forEach((tr, a) => {
        const sc = scoreMap.get('id_' + a + SEP);
        for (let t = from; t < tr.length; t++) {
            const p = sc[t] >= 0.5;
            if (p && tr[t]) tp++; else if (p) fp++; else if (tr[t]) fn++;
        }
    });
    return 2 * tp / Math.max(1, 2 * tp + fp + fn);
}

// ---------------------------------------------------------------------------
console.log('\n-- 1. labels → classes, gate --');
{
    resetEthogram();
    const s = new Session([], new Skeleton('m', NODES, EDGES), [], 'Classes');
    s.addIdentity('a'); s.addIdentity('b');
    const walk = addBehavior({ name: 'walk', group: 'move' }).behavior;
    const rear = addBehavior({ name: 'rear', group: 'move' }).behavior;
    labelRange(s, rear.id, 'a', '', 100, 119, true);
    labelRange(s, rear.id, 'a', '', 200, 209, false);
    labelRange(s, walk.id, 'a', '', 150, 179, true);       // a peer: "not rear" on 150–179
    labelRange(s, walk.id, 'a', '', 205, 214, true);       // overlaps rear's own "not" on 205–209 (own wins)
    const cl = labelClasses(s, rear);
    const ivs = cl.get('a' + SEP).ivs;
    check(eq(ivs, [[100, 119, 1], [150, 179, 0], [200, 209, 0], [210, 214, 0]]),
        'own labels, plus "not" from the exclusive peer where rear has none (' + JSON.stringify(ivs) + ')');
    const c = labelCounts([s], rear.id);
    check(c.present.frames === 20 && c.present.bouts === 1 && c.absent.frames === 30 + 10 + 5 && c.absent.bouts === 3, 'counts per class: frames and bouts');
    const g = gateProblems(c);
    check(g.length === 1 && /present needs 1 more bout/.test(g[0]), 'the gate: ' + JSON.stringify(g));
    labelRange(s, rear.id, 'b', '', 300, 304, true);
    const g2 = gateProblems(labelCounts([s], rear.id));
    check(g2.length === 0 && GATE_FRAMES === 20 && GATE_BOUTS === 2, 'two present bouts, 25 frames: it trains');
}

// ---------------------------------------------------------------------------
console.log('\n-- 2. grouped folds --');
{
    const R = 400;
    const mk = (sessionOf, groupOf, yOf) => ({
        session: Int32Array.from({ length: R }, (_, i) => sessionOf(i)), group: Int32Array.from({ length: R }, (_, i) => groupOf(i)),
        frame: Int32Array.from({ length: R }, (_, i) => i % 200), y: Uint8Array.from({ length: R }, (_, i) => yOf(i)),
    });
    const bySess = planFolds(mk(i => (i < 200 ? 0 : 1), i => i % 2, i => (i % 10 < 3 ? 1 : 0)), 5, { sessions: ['A', 'B'] });
    check(bySess.scheme === 'session' && bySess.folds.length === 2 && eq(bySess.folds.map(f => f.label), ['A', 'B']), 'two sessions with both classes: by session');
    check(bySess.folds.every(f => Array.from(f.test).every(r => (r < 200) === (f.label === 'A')) && f.train.length + f.test.length === R), 'a fold holds out one whole session');
    const one = mk(i => 0, i => (i < 200 ? 0 : 1), i => (i % 10 < 3 ? 1 : 0));
    const byAnimal = planFolds(one, 5, { groups: ['id_0', 'id_1'] });
    check(byAnimal.scheme === 'animal' && byAnimal.folds.length === 2, 'one session, two animals with both classes: by animal');
    const onlyOne = mk(i => 0, i => (i < 200 ? 0 : 1), i => (i < 200 && i % 10 < 3 ? 1 : 0));
    const byTime = planFolds(onlyOne, 5, {});
    check(byTime.scheme === 'time' && byTime.folds.length === 5, 'one animal with both classes: 5 time blocks');
    // Blocks are contiguous in (session, frame), and their neighbours within the gap are not trained on.
    const ok = byTime.folds.every(f => {
        const frames = Array.from(f.test).map(r => onlyOne.frame[r]);
        const lo = Math.min(...frames), hi = Math.max(...frames);
        const contiguous = Array.from(f.test).length === new Set(Array.from(f.test)).size;
        const purged = Array.from(f.train).every(r => onlyOne.frame[r] < lo - 5 || onlyOne.frame[r] > hi + 5);
        return contiguous && purged;
    });
    check(ok, 'each block\'s training leaves out every frame within the gap (5) of it');
    const test0 = Array.from(byTime.folds[0].test);
    check(test0.every((r, i) => i === 0 || onlyOne.frame[r] >= onlyOne.frame[test0[i - 1]] || r >= 200), 'never random frames: a block is a run of time');
}

// ---------------------------------------------------------------------------
console.log('\n-- 3. metrics --');
{
    const m = framePrf([0.9, 0.6, 0.4, 0.2, NaN], [1, 0, 1, 0, 1], 0.5);
    check(m.tp === 1 && m.fp === 1 && m.fn === 2 && m.tn === 1 && Math.abs(m.f1 - 2 / 5) < 1e-12, 'frame P/R/F1 at a threshold; unscored = not detected');
    const c = prCurve([0.9, 0.8, 0.3, 0.2], [1, 1, 0, 0], 10);
    check(c.points.length === 11 && c.best.f1 === 1 && c.best.threshold > 0.3 && c.best.threshold <= 0.8, 'PR curve: the best F1 (1) between the classes (' + c.best.threshold + ')');
    const b = boutPrf([[0, 5], [20, 25], [40, 41]], [[3, 10], [22, 23], [60, 70]]);
    check(b.hits === 2 && b.found === 2 && Math.abs(b.precision - 2 / 3) < 1e-12 && Math.abs(b.recall - 2 / 3) < 1e-12, 'bouts: overlap counts as found (2 of 3 each way)');
}

// ---------------------------------------------------------------------------
console.log('\n-- 4. a planted kinematic behavior --');
resetEthogram();
resetBehaviorModels();
const W = dartWorld('Darts', 9000, 7);
const dart = addBehavior({ name: 'dart', key: 'd' }).behavior;
const LABEL_END = 6000;
labelByTruth(W.session, dart.id, W.truth, 0, LABEL_END);
const ctxW = contextOf(W.session, W.T);
const src = () => [{ session: W.session, name: W.session.name, ctx: ctxW }];
let model;
{
    const t0 = Date.now();
    const res = await trainBehaviorClassifier({ behaviorId: dart.id, sources: src() });
    const ms = Date.now() - t0;
    check(!res.error && res.model, 'trained with the defaults (100 trees) in ' + ms + ' ms, inline' + (res.error ? ': ' + res.error : ''));
    model = res.model;
    const cv = res.report.cv;
    check(cv.scheme === 'animal' && eq(cv.folds, ['id_0', 'id_1']), 'cross-validated by animal (one session, two animals): ' + cv.scheme + ' ' + JSON.stringify(cv.folds));
    check(cv.frame.f1 > 0.9, 'cross-validated frame F1 ' + cv.frame.f1.toFixed(3) + ' > 0.9 (P ' + cv.frame.precision.toFixed(3) + ', R ' + cv.frame.recall.toFixed(3) + ')');
    check(cv.bout.f1 > 0.9, 'cross-validated bout F1 ' + cv.bout.f1.toFixed(3) + ' > 0.9');
    check(cv.curve.length > 10 && cv.best && cv.best.f1 >= cv.frame.f1 - 1e-9, 'a PR curve, its best F1 ' + cv.best.f1.toFixed(3) + ' at ' + cv.best.threshold);
    const top = topFeatures(model, 5).map(f => f.name);
    check(top.filter(nm => /speed/.test(nm)).length >= 4, 'the top features are speeds: ' + top.join(' | '));
    check(model.features.length <= res.report.features && model.hash.length === 16, 'the forest reads ' + model.features.length + ' of ' + res.report.features + ' features; hash ' + model.hash);
    const pr = await predictModels(ctxW, [model]);
    const f1 = heldOutF1(pr.scores[dart.id], W.truth, LABEL_END);
    check(f1 > 0.9, 'frames never labelled (6,000–8,999): F1 ' + f1.toFixed(3) + ' > 0.9');
    check(pr.scores[dart.id].size === 2 && pr.scores[dart.id].get('id_0' + SEP).length === W.T, 'one score per frame per animal');
    const live = evaluateAt(cv, { threshold: 0.9, minBout: 0, maxGap: 0 }, FPS, false);
    check(live && live.frame.recall <= cv.frame.recall && live.frame.precision >= cv.frame.precision - 1e-9, 'a higher threshold: no more recall, no less precision (live, no retraining)');
    const again = await trainBehaviorClassifier({ behaviorId: dart.id, sources: src() });
    check(again.model.hash === model.hash, 'retrained on the same labels: the same forest (' + again.model.hash + ')');
    // A second session: cross-validation by session.
    const W2 = dartWorld('Darts 2', 4000, 99);
    labelByTruth(W2.session, dart.id, W2.truth, 0, 4000);
    const two = await trainBehaviorClassifier({ behaviorId: dart.id, params: { trees: 30 },
        sources: src().concat([{ session: W2.session, name: W2.session.name, ctx: contextOf(W2.session, W2.T) }]) });
    check(two.report.cv.scheme === 'session' && eq(two.report.cv.folds, ['Darts', 'Darts 2']) && two.report.cv.frame.f1 > 0.9,
        'two labelled sessions: cross-validated by session (F1 ' + two.report.cv.frame.f1.toFixed(3) + ')');
    check(eq(two.report.sessions, ['Darts', 'Darts 2']), 'and it names the sessions it learned from');
    // A session with labels but no poses resident, nothing cached: left out, by name.
    const ghost = new Session([], new Skeleton('m', NODES, EDGES), [], 'Ghost');
    ghost.addIdentity('id_0');
    labelRange(ghost, dart.id, 'id_0', '', 0, 30, true);
    const withGhost = await trainBehaviorClassifier({ behaviorId: dart.id, params: { trees: 10 }, crossValidate: false,
        sources: src().concat([{ session: ghost, name: 'Ghost', ctx: null }]) });
    check(withGhost.model && withGhost.report.leftOut.length === 1 && withGhost.report.leftOut[0].session === 'Ghost' &&
        /open it once/.test(withGhost.report.leftOut[0].reason), 'a labelled session with no poses and nothing cached is left out, by name');
}

// ---------------------------------------------------------------------------
console.log('\n-- 5. training rows are cached --');
{
    const before = trainingRowsComputed();
    await trainBehaviorClassifier({ behaviorId: dart.id, sources: src(), params: { trees: 5 }, crossValidate: false });
    check(trainingRowsComputed() === before, 'retraining on the same labels computes no row again');
    labelRange(W.session, dart.id, 'id_0', '', 6000, 6099, false);
    const mid = trainingRowsComputed();
    await trainBehaviorClassifier({ behaviorId: dart.id, sources: src(), params: { trees: 5 }, crossValidate: false });
    const added = trainingRowsComputed() - mid;
    check(added === 100, '100 more labelled frames: exactly those 100 rows computed (' + added + ')');
    // The cache serves a session whose poses are gone (a lazy project's other session).
    const cachedOnly = await trainBehaviorClassifier({ behaviorId: dart.id, params: { trees: 5 }, crossValidate: false,
        sources: [{ session: W.session, name: W.session.name, ctx: null }, { session: new Session([], new Skeleton('m', NODES, EDGES), [], 'Other'), name: 'Other', ctx: contextOf(dartWorld('Other', 300, 3).session, 300) }] });
    check(cachedOnly.model && cachedOnly.report.rows === 12100 && cachedOnly.report.sessions.indexOf('Darts') >= 0,
        'a session with no poses resident trains from its cached rows (' + cachedOnly.report.rows + ' rows)');
    labelRange(W.session, dart.id, 'id_0', '', 6000, 6099, null);
}

// ---------------------------------------------------------------------------
console.log('\n-- 6. the cap --');
{
    const a = await trainBehaviorClassifier({ behaviorId: dart.id, sources: src(), params: { trees: 20 }, maxRows: 2000 });
    const b = await trainBehaviorClassifier({ behaviorId: dart.id, sources: src(), params: { trees: 20 }, maxRows: 2000 });
    const bouts = listBouts(W.session, { behaviorId: dart.id }).length;
    check(a.report.sampled && a.report.rows <= 2000 + bouts && a.report.rows > 900, 'over the cap: sampled to ' + a.report.rows + ' rows (cap 2,000 plus at most one per bout)');
    check(a.model.hash === b.model.hash, 'the same rows every time (same forest)');
    check(a.report.cv.frame.f1 > 0.85, 'and still learns it (cross-validated F1 ' + a.report.cv.frame.f1.toFixed(3) + ')');
}

// ---------------------------------------------------------------------------
console.log('\n-- 7. a social (mutual) behavior --');
{
    // Mouse 0 walks a small circle; mouse 1 a big one that passes it every lap: they "meet".
    const T = 6000;
    const s = new Session([], new Skeleton('m', NODES, EDGES), [], 'Meet');
    s.addIdentity('id_0'); s.addIdentity('id_1');
    const rand = seededRandom(17);
    let gid = 1;
    const pose = (cx, cy, hd) => {
        const pts = new Float64Array(NODES.length * 3);
        for (let n = 0; n < NODES.length; n++) {
            pts[3 * n] = cx + ALONG[n] * Math.cos(hd) + (rand() - 0.5) * 2;
            pts[3 * n + 1] = cy + ALONG[n] * Math.sin(hd) + (rand() - 0.5) * 2;
            pts[3 * n + 2] = H[n] + (rand() - 0.5);
        }
        return pts;
    };
    const heads = [];
    for (let t = 0; t < T; t++) {
        const a0 = t / 30 * 0.6, a1 = t / 30 * 0.5;
        const p0 = [40 * Math.cos(a0), 40 * Math.sin(a0), a0 + Math.PI / 2];
        const p1 = [330 + 300 * Math.cos(a1 + Math.PI), 300 * Math.sin(a1 + Math.PI), a1 + Math.PI + Math.PI / 2];
        const g0 = new InstanceGroup(gid++, s.identities[0].id); g0.points3d = pose(p0[0], p0[1], p0[2]);
        const g1 = new InstanceGroup(gid++, s.identities[1].id); g1.points3d = pose(p1[0], p1[1], p1[2]);
        s.instanceGroups.set(t, [g0, g1]);
        heads.push([g0.points3d[0], g0.points3d[1], g1.points3d[0], g1.points3d[1]]);
    }
    const meet = addBehavior({ name: 'meet', social: true, mutual: true }).behavior;
    const truth = Uint8Array.from(heads, h => (Math.hypot(h[0] - h[2], h[1] - h[3]) < 100 ? 1 : 0));
    let st = 0;
    for (let t = 1; t <= 4000; t++) {
        if (t === 4000 || truth[t] !== truth[st]) { labelRange(s, meet.id, 'id_1', 'id_0', st, t - 1, !!truth[st]); st = t; }
    }
    check(listBouts(s, { behaviorId: meet.id })[0].animal === 'id_0', 'a mutual pair is stored in name order');
    const ctx = contextOf(s, T);
    const res = await trainBehaviorClassifier({ behaviorId: meet.id, sources: [{ session: s, name: 'Meet', ctx }], params: { trees: 40 } });
    check(!res.error, 'trained' + (res.error ? ': ' + res.error : ''));
    const c = labelCounts([s], meet.id);
    check(res.report.rows === 2 * (c.present.frames + c.absent.frames), 'a sample in each direction: ' + res.report.rows + ' rows for ' + (c.present.frames + c.absent.frames) + ' pair-frames');
    check(res.report.cv.scheme === 'time' && res.report.cv.folds.length === 5 && res.report.cv.frame.f1 > 0.9,
        'one pair: 5 time blocks, cross-validated F1 ' + res.report.cv.frame.f1.toFixed(3));
    check(res.model.features.some(nm => /head–head|centroid distance|min node/.test(nm)), 'it reads relational features');
    const pr = await predictModels(ctx, [res.model]);
    const sc = pr.scores[meet.id];
    check(sc.size === 1 && sc.has('id_0' + SEP + 'id_1'), 'scored once per pair, in name order (the mean of both directions)');
    let tp = 0, fp = 0, fn = 0;
    for (let t = 4000; t < T; t++) { const p = sc.get('id_0' + SEP + 'id_1')[t] >= 0.5; if (p && truth[t]) tp++; else if (p) fp++; else if (truth[t]) fn++; }
    check(2 * tp / (2 * tp + fp + fn) > 0.9, 'frames never labelled: F1 ' + (2 * tp / (2 * tp + fp + fn)).toFixed(3));
}

// ---------------------------------------------------------------------------
console.log('\n-- 8. exclusive groups --');
{
    resetEthogram();
    const a = addBehavior({ name: 'groom', group: 'g' }).behavior, b = addBehavior({ name: 'rest', group: 'g' }).behavior;
    const sa = new Float32Array(20).fill(0.1), sb = new Float32Array(20).fill(0.1);
    for (let t = 0; t < 10; t++) sa[t] = 0.9;
    for (let t = 5; t < 15; t++) sb[t] = 0.7;
    const how = { [a.id]: modelBoutSettings(a, FPS), [b.id]: modelBoutSettings(b, FPS) };
    const out = deriveBouts({ [a.id]: new Map([['m' + SEP, sa]]), [b.id]: new Map([['m' + SEP, sb]]) }, how);
    check(eq(out[a.id].map(x => [x.start, x.end]), [[0, 9]]) && eq(out[b.id].map(x => [x.start, x.end]), [[10, 14]]),
        'groom 0–9 (higher) keeps its frames; rest only 10–14');
    updateBehavior(b.id, { group: '' });
    const free = deriveBouts({ [a.id]: new Map([['m' + SEP, sa]]), [b.id]: new Map([['m' + SEP, sb]]) },
        { [a.id]: modelBoutSettings(a, FPS), [b.id]: modelBoutSettings(behaviorById(b.id), FPS) });
    check(eq(free[b.id].map(x => [x.start, x.end]), [[5, 14]]), 'out of the group: rest 5–14 untouched');
    setBehaviorPost(a.id, { threshold: 0.95, minBout: 0.5, maxGap: 0 });
    check(eq(behaviorById(a.id).post, { threshold: 0.95, minBout: 0.5 }), 'post-processing keeps only what differs from the defaults');
    setBehaviorPost(a.id, { threshold: 0.5, minBout: 0, maxGap: 0 });
    check(behaviorById(a.id).post === undefined, 'all defaults: nothing kept (never written)');
}

// ---------------------------------------------------------------------------
console.log('\n-- 9. model runs: stale on labels, on a newer model --');
{
    resetEthogram();
    resetBehaviorModels();
    const dart2 = addBehavior({ name: 'dart', key: 'd' }).behavior;
    labelByTruth(W.session, dart2.id, W.truth, 0, 3000);
    const res = await trainBehaviorClassifier({ behaviorId: dart2.id, sources: src(), params: { trees: 20 }, allSessions: [W.session] });
    setBehaviorModel(res.model);
    const sessions = [W.session];
    const env = { fps: FPS, floorKey: 'k', modelHashOf, labelPrintOf: (id) => labelPrintOf(sessions, id) };
    const source = 'model:' + res.model.hash;
    setBehaviorRun(W.session, { behaviorId: dart2.id, source, labels: res.model.labelPrint, fps: FPS, mode: '3d',
        cfg: runConfigPrint(dart2, W.session, { fps: FPS, mode: '3d', floorKey: 'k', source }), bouts: [{ animal: 'id_0', target: '', start: 5, end: 9, score: 0.8 }] });
    let st = predictionStaleness(W.session, env);
    check(!st.input && !st.settings.length && !st.labels.length && !st.model.length, 'fresh after the run');
    labelRange(W.session, dart2.id, 'id_1', '', 5000, 5009, true);
    st = predictionStaleness(W.session, env);
    check(eq(st.labels, [dart2.id]), 'a label edit: stale — the labels it learned from changed');
    labelRange(W.session, dart2.id, 'id_1', '', 5000, 5009, null);
    st = predictionStaleness(W.session, env);
    check(!st.labels.length, 'the edit undone: fresh again (the print is of the labels, not of the edits)');
    setBehaviorPost(dart2.id, { threshold: 0.7 });
    check(predictionStaleness(W.session, env).settings.indexOf(dart2.id) >= 0, 'its threshold changed: stale settings');
    setBehaviorPost(dart2.id, null);
    setBehaviorModel(Object.assign({}, res.model, { hash: 'ffffffffffffffff' }));
    check(eq(predictionStaleness(W.session, env).model, [dart2.id]), 'a newer model replaced it: stale');
    setBehaviorModel(res.model);
    const saved = exportPredictionRuns(W.session);
    check(saved.runs[0].source === source && saved.runs[0].labels === res.model.labelPrint, 'saved with its model hash and its label print');
    const reopened = new Session([], W.session.skeleton, [], 'Darts');
    importPredictionRuns(reopened, JSON.parse(JSON.stringify(saved)));
    resetBehaviorModels();
    const st2 = predictionStaleness(reopened, { fps: FPS, floorKey: 'k', modelHashOf, labelPrintOf: (id) => labelPrintOf([W.session], id) });
    check(!st2.labels.length && !st2.model.length && behaviorRunOf(reopened, dart2.id).source === source,
        'reopened without the model (not saved): still fresh while its labels are the same');
    check(trainingLabelPrint([W.session], dart2.id) === res.model.labelPrint, 'the print is the same labels\' print');
}

// ---------------------------------------------------------------------------
console.log('\n-- 10. model files --');
{
    resetEthogram();
    resetBehaviorModels();
    const d = addBehavior({ name: 'dart', key: 'd', color: '#5fb3f9' }).behavior;
    setBehaviorPost(d.id, { threshold: 0.6 });
    labelByTruth(W.session, d.id, W.truth, 0, 3000);
    const res = await trainBehaviorClassifier({ behaviorId: d.id, sources: src(), params: { trees: 20 } });
    const file = JSON.parse(JSON.stringify(buildModelFile([res.model], { fps: FPS })));
    check(file.format === 'lucid-behavior-model' && file.featureSpecVersion === 1 && eq(file.roles, { head: 'Nose', tailBase: 'TTI' }) &&
        eq(file.windows, [0.1, 0.3, 1]), 'the file records the feature-spec version, the roles, the windows in seconds');
    const parsed = parseModelFile(file);
    check(parsed.models.length === 1 && parsed.models[0].model.hash === res.model.hash, 'read back: the same forest hash');
    // Into a fresh project: the behavior comes with its post-processing.
    resetEthogram();
    resetBehaviorModels();
    check(modelFileProblems(parsed, ctxW).blocking.length === 0, 'it fits a session with the same body');
    const adopted = adoptModelFile(parsed);
    const b = behaviorById(adopted[0].behaviorId);
    check(b && b.name === 'dart' && b.key === 'd' && eq(b.post, { threshold: 0.6 }), 'the ethogram gains "dart", with its key and threshold');
    const p1 = await predictModels(ctxW, [res.model]), p2 = await predictModels(ctxW, [adopted[0]]);
    const s1 = p1.scores[d.id].get('id_0' + SEP), s2 = p2.scores[adopted[0].behaviorId].get('id_0' + SEP);
    check(s1.every((v, i) => Object.is(v, s2[i])), 'and scores every frame exactly as before');
    check(adopted[0].imported && adopted[0].labelPrint === null, 'an imported model: no label print (this project\'s labels did not train it)');
    const other = Object.assign({}, ctxW, { spec: Object.assign({}, ctxW.spec, { roles: { head: 1, tailBase: 4 } }) });
    check(modelFileProblems(parsed, other).blocking.some(p => /head \/ tail base/.test(p)), 'another head: refused, saying why');
    const bad = JSON.parse(JSON.stringify(file));
    bad.behaviors[0].model.forest.val = bad.behaviors[0].model.forest.val.replace(/^.{8}/, 'AAAAAAAA');
    check(parseModelFile(bad).problems.some(p => /does not match its hash|damaged/.test(p)), 'a tampered forest is caught');
    check(parseModelFile({ format: 'x' }).problems.length === 1, 'not a classifier file: said so');
}

// ---------------------------------------------------------------------------
console.log('\n-- 11. active learning --');
{
    resetEthogram();
    const d = addBehavior({ name: 'dart' }).behavior;
    const s = new Session([], new Skeleton('m', NODES, EDGES), [], 'AL');
    s.addIdentity('m');
    const sc = new Float32Array(200).fill(0.05);
    for (let t = 20; t < 40; t++) sc[t] = 0.55;     // near the threshold, unlabelled
    for (let t = 60; t < 70; t++) sc[t] = 0.45;     // near, but labelled below
    for (let t = 100; t < 130; t++) sc[t] = 0.95;   // sure
    labelRange(s, d.id, 'm', '', 60, 69, false);
    const source = 'model:0123456789abcdef';
    setBehaviorRun(s, { behaviorId: d.id, source, fps: FPS, mode: '3d', cfg: '', scores: new Map([['m' + SEP, sc]]),
        bouts: deriveBouts({ [d.id]: new Map([['m' + SEP, sc]]) }, { [d.id]: modelBoutSettings(d, FPS) })[d.id] });
    const un = uncertainSegments(s, d.id);
    check(un.length === 1 && un[0].start === 20 && un[0].end === 39 && Math.abs(un[0].uncertainty - 0.05) < 1e-6,
        'uncertain: the unlabelled stretch near the threshold (20–39), not the labelled one nor the sure one');
    labelRange(s, d.id, 'm', '', 100, 109, false);    // "not" inside a predicted bout
    labelRange(s, d.id, 'm', '', 150, 159, true);     // present where nothing is predicted
    const dis = disagreementSegments(s, d.id);
    check(eq(dis.map(x => [x.kind, x.start, x.end]), [['false', 100, 109], ['missed', 150, 159]]) ||
        eq(dis.map(x => [x.kind, x.start, x.end]), [['false', 100, 109], ['missed', 150, 159]].reverse()),
        'disagreements: predicted but labelled "not" (100–109), labelled but not predicted (150–159)');
    labelRange(s, d.id, 'm', '', 20, 39, true);
    check(uncertainSegments(s, d.id).length === 0, 'once labelled, the stretch is no longer offered');
}

console.log('\n' + (failed ? 'FAIL' : 'PASS') + ' — ' + passed + ' passed, ' + failed + ' failed');
process.exit(failed ? 1 : 0);
