/**
 * behavior-classifier.mjs — behavior classifiers in the real app: training on
 * labels, cross-validation, predictions, active learning, thresholds, files.
 *
 *  1. Train: two mice "dart" (a burst of 4× speed for 0.5–1.5 s) at random
 *     times; the first 4,000 of 6,000 frames are labelled by the truth. The
 *     Classifiers section and the Train Classifiers dialog show the labels
 *     against the gate; Train gives a card with cross-validated precision,
 *     recall and F1 (> 0.9) by animal, a PR curve and the top features
 *     (speeds); training on a forced 3-worker pool gives the same forest as
 *     inline, bit for bit (same hash).
 *  2. Predict: the card's Predict all frames scores every frame with the
 *     classifier (source model:<hash>): on the 2,000 frames never labelled,
 *     F1 > 0.9; "rear" (a rule) is still detected by its rule; the toolbar
 *     Behavior button is blue; This frame shows the classifier's score
 *     against its threshold.
 *  3. Threshold: moving it in the dialog updates the numbers live and, let
 *     go, re-derives the bouts from the scores in memory at once (the
 *     prediction stays fresh); it is kept in the ethogram (`post`).
 *  4. Active learning: Next uncertain ▸ shows an unlabelled stretch near the
 *     threshold and moves the playhead there; Accept labels it present
 *     (source "a") and moves on; the predictions are then stale (the labels
 *     the classifier learned from changed); Next disagreement ▸ finds a
 *     planted contradiction; training again freshens the predictions.
 *  5. Save → reopen: the classifier run keeps its hash and label print, the
 *     threshold comes back; with the classifier gone (never saved) the
 *     predictions still stand, and retraining gives the same hash.
 *  6. Files: Export Classifiers… downloads a .lucid-behavior.json; imported
 *     into a project without the behavior, it brings the behavior (with its
 *     threshold) and scores every frame as before.
 *  7. Esc closes the dialog; no page errors.
 *  8. No video: trained and predicted at 60 fps, saved, reopened with the app at
 *     the default 30: retraining measures at the 60 on record, so it gives the
 *     same forest and the predictions still stand; the threshold dialog's live
 *     numbers and an exported model file use that rate too.
 *
 * Run: node tests/e2e/behavior-classifier.mjs   (spawns its own http.server; PORT= to move it)
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8231);

let fails = 0;
const check = (c, m) => { console.log((c ? '  ✓ ' : '  ✗ ') + m); if (!c) fails++; };
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1200));

let browser;
try {
    browser = await chromium.launch();
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, acceptDownloads: true });
    const page = await context.newPage();
    const errs = [];
    page.on('pageerror', e => errs.push('pageerror: ' + String(e).slice(0, 300)));
    page.on('console', m => { if (m.type() === 'error') errs.push('console: ' + m.text().slice(0, 300)); });
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state, null, { timeout: 30000 });

    // The world: two mice wandering in their boxes, darting now and then;
    // mouse 0 rears on 700–760 (a rule detects that).
    await page.evaluate(async () => {
        const pd = await import('/pose/pose-data.js');
        const AS = await import('/ui/app-state.js');
        const NODES = ['Nose', 'Head', 'Neck', 'Trunk', 'TTI', 'Tail1'];
        const ALONG = [45, 30, 15, 0, -40, -70], H = [15, 25, 30, 30, 15, 5];
        const T = 6000;
        let seed = 7;
        const rand = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
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
        const session = new pd.Session([], new pd.Skeleton('m', NODES, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]), [], 'Darts');
        session.addIdentity('id_0'); session.addIdentity('id_1');
        const pos = [[0, 0, 0], [600, 0, 1]];
        let gid = 1;
        for (let t = 0; t < T; t++) {
            session.instanceGroups.set(t, [0, 1].map(k => {
                const p = pos[k];
                const v = truth[k][t] ? 12 : 3;
                p[2] += (rand() - 0.5) * 0.15;
                p[0] += v * Math.cos(p[2]); p[1] += v * Math.sin(p[2]);
                const dx = p[0] - 600 * k, dy = p[1];
                if (Math.hypot(dx, dy) > 400) p[2] = Math.atan2(-dy, -dx) + (rand() - 0.5) * 0.5;
                const g = new pd.InstanceGroup(gid++, session.identities[k].id);
                const pts = new Float64Array(NODES.length * 3);
                const rear = k === 0 && t >= 700 && t <= 760;
                for (let n = 0; n < NODES.length; n++) {
                    pts[3 * n] = p[0] + ALONG[n] * Math.cos(p[2]) + gauss() * 1.5;
                    pts[3 * n + 1] = p[1] + ALONG[n] * Math.sin(p[2]) + gauss() * 1.5;
                    pts[3 * n + 2] = (rear && n <= 2 ? [95, 85, 65][n] : H[n]) + gauss();
                }
                g.points3d = pts;
                return g;
            }));
        }
        window.__truth = truth;
        AS.state.sessions = [session]; AS.state.activeSessionIdx = 0; AS.state.session = session;
        AS.state.totalFrames = T; AS.state.currentFrame = 0; AS.state.views = []; AS.state.fps = 30;
        AS.timeline.setTotalFrames(T); AS.timeline.setData(session);
        (await import('/ui/info-panel.js')).updateInfoPanel();
        const eth = await import('/pose/behavior-ethogram.js');
        const L = await import('/pose/behavior-labels.js');
        const dart = eth.addBehavior({ name: 'dart', key: 'd' }).behavior;
        const rear = eth.addBehavior({ name: 'rear', key: 'r' }).behavior;
        eth.setBehaviorDetector(rear.id, { conditions: [{ feature: 'Nose height', op: '>', value: 0.5 }], minDuration: 0, maxGap: 0 });
        // Label the first 4,000 frames of both mice by the truth (one undoable edit per run).
        truth.forEach((tr, a) => {
            let s = 0;
            for (let t = 1; t <= 4000; t++) {
                if (t === 4000 || tr[t] !== tr[s]) { L.labelRange(session, dart.id, 'id_' + a, '', s, t - 1, !!tr[s]); s = t; }
            }
        });
        (await import('/ui/behavior-coding.js')).requestBehaviorRepaint({ ethogram: true, labels: true });
    });
    const openTab = () => page.evaluate(async () => {
        (await import('/ui/behavior-coding.js')).openBehaviorTab();
        const s = document.getElementById('behaviorClassifiersSection'); if (s) s.open = true;
        (await import('/ui/behavior-panel.js')).refreshBehaviorPanel();
    });
    const isBlue = () => page.evaluate(() => document.getElementById('tbBehavior').classList.contains('active'));
    const heldOutF1 = () => page.evaluate(async () => {
        const P = await import('/pose/behavior-predictions.js');
        const s = (await import('/ui/app-state.js')).state.session;
        const run = P.behaviorRunOf(s, 'b1');
        let tp = 0, fp = 0, fn = 0;
        window.__truth.forEach((tr, a) => {
            const sc = run.scores.get('id_' + a + '\u0001');
            for (let t = 4000; t < tr.length; t++) { const p = sc[t] >= 0.5; if (p && tr[t]) tp++; else if (p) fp++; else if (tr[t]) fn++; }
        });
        return { f1: 2 * tp / Math.max(1, 2 * tp + fp + fn), source: run.source };
    });

    console.log('\n-- 1. train --');
    await openTab();
    await page.waitForSelector('.behavior-classifiers', { timeout: 10000 });
    const row = await page.evaluate(() => Array.from(document.querySelectorAll('.behavior-classifiers tbody tr')).map(r => r.textContent));
    check(row.length === 2 && /dart/.test(row[0]) && /\d+ \/ \d+/.test(row[0]) && /–/.test(row[0]) && /rear.*detector/.test(row[1]),
        'the Classifiers section lists dart (bouts present / "not", no classifier yet) and rear (its detector): ' + JSON.stringify(row));
    await page.click('[data-menu="behavior"]');
    await page.click('#menuBehaviorTrain');
    await page.waitForSelector('.behavior-train-modal .bt-table', { timeout: 10000 });
    const table = await page.evaluate(() => Array.from(document.querySelectorAll('.bt-table tbody tr')).map(r => ({
        text: r.textContent, checked: !!(r.querySelector('input[data-act="pick"]') || {}).checked, disabled: !!(r.querySelector('input[data-act="pick"]') || {}).disabled })));
    check(table[0].checked && !table[0].disabled && /dart/.test(table[0].text), 'the dialog: dart is ready and ticked');
    check(table[1].disabled && /present needs/.test(table[1].text), 'rear has no labels: not ready, saying what it needs');
    await page.evaluate(() => { window.LUCID_BEHAVIOR_WORKER = 0; });
    await page.click('.behavior-train-modal [data-act="train"]');
    await page.waitForSelector('.bt-card[data-id="b1"] .bt-metrics', { timeout: 120000 });
    const card = await page.evaluate(() => {
        const c = document.querySelector('.bt-card[data-id="b1"]');
        const cells = Array.from(c.querySelectorAll('.bt-metrics tbody tr')).map(r => Array.from(r.querySelectorAll('td')).map(td => td.textContent));
        return { note: c.querySelector('.bd-note').textContent, cells, top: Array.from(c.querySelectorAll('.bt-top li')).map(li => li.textContent),
                 curve: !!c.querySelector('canvas.bt-curve') };
    });
    check(/Cross-validated by animal \(2 folds: id_0; id_1\)/.test(card.note), 'cross-validated by animal: "' + card.note.slice(0, 120) + '…"');
    const f1Frames = parseFloat(card.cells[0][3]), f1Bouts = parseFloat(card.cells[1][3]);
    check(f1Frames > 0.9 && f1Bouts > 0.9, 'the card shows F1 > 0.9 on frames (' + f1Frames + ') and bouts (' + f1Bouts + ')');
    check(card.curve && card.top.length >= 5 && card.top.slice(0, 5).filter(t => /speed/.test(t)).length >= 4, 'a PR curve, and the top features are speeds: ' + card.top.slice(0, 3).join(' | '));
    const hashInline = await page.evaluate(async () => (await import('/pose/behavior-classifier.js')).behaviorModelOf('b1').hash);
    await page.evaluate(() => { window.LUCID_BEHAVIOR_WORKER = 3; });
    const hashPool = await page.evaluate(async () => {
        const T = await import('/ui/behavior-train.js');
        await T.trainClassifiers(['b1']);
        return (await import('/pose/behavior-classifier.js')).behaviorModelOf('b1').hash;
    });
    await page.evaluate(() => { delete window.LUCID_BEHAVIOR_WORKER; });
    check(hashPool === hashInline, 'trained on a forced 3-worker pool: the same forest as inline (' + hashInline + ')');

    console.log('\n-- 2. predict --');
    check(!(await isBlue()), 'not blue before predicting');
    await page.click('.bt-card[data-id="b1"] [data-act="predict"]');
    await page.waitForFunction(() => document.getElementById('tbBehavior').classList.contains('active'), null, { timeout: 60000 });
    check(!(await page.$('.behavior-train-modal')), 'Predict all frames closes the dialog and runs; the toolbar button turns blue');
    const pred = await heldOutF1();
    check(/^model:[0-9a-f]{16}$/.test(pred.source) && pred.source === 'model:' + hashInline, 'dart\'s run is the classifier\'s (' + pred.source + ')');
    check(pred.f1 > 0.9, 'on the 2,000 frames never labelled: F1 ' + pred.f1.toFixed(3) + ' > 0.9');
    // The toolbar button runs both: rear by its rule, dart by its classifier.
    await page.click('#tbBehavior');
    await page.waitForFunction(() => /rear \d/.test(document.getElementById('statusText').textContent), null, { timeout: 60000 });
    const both = await page.evaluate(async () => {
        const P = await import('/pose/behavior-predictions.js');
        const s = (await import('/ui/app-state.js')).state.session;
        return { rear: P.behaviorRunOf(s, 'b2').source, rearBouts: P.behaviorRunOf(s, 'b2').bouts.map(b => [b.animal, b.start, b.end]),
                 dart: P.behaviorRunOf(s, 'b1').source, status: document.getElementById('statusText').textContent };
    });
    check(both.rear === 'rule' && eq(both.rearBouts, [['id_0', 700, 760]]) && both.dart === pred.source,
        'the Behavior button: rear by its rule (700–760), dart by its classifier');
    check(/rear 1, dart \d[\d,]* \(classifier\)/.test(both.status), 'the status says which came from a classifier: "' + both.status.slice(0, 90) + '"');
    await page.evaluate(async () => (await import('/pose/initialization.js')).navigateToFrame(100));
    await page.hover('#tbBehavior');
    await page.click('#tbBehaviorFrame');
    await page.waitForSelector('.behavior-readout-row', { timeout: 10000 });
    const readout = await page.evaluate(() => Array.from(document.querySelectorAll('.behavior-readout-row')).map(r => r.textContent));
    check(readout.some(r => /dart/.test(r) && /classifier score \d+% \(threshold 50%\)/.test(r)), 'This frame: dart\'s classifier score against its threshold');

    console.log('\n-- 3. threshold --');
    await page.click('[data-menu="behavior"]');
    await page.click('#menuBehaviorTrain');
    await page.waitForSelector('.bt-card[data-id="b1"] input[data-f="threshold"]', { timeout: 10000 });
    const before = await page.evaluate(async () => {
        const P = await import('/pose/behavior-predictions.js');
        const s = (await import('/ui/app-state.js')).state.session;
        return { bouts: P.behaviorRunOf(s, 'b1').bouts.length, recall: document.querySelector('.bt-card[data-id="b1"] .bt-metrics tbody tr td:nth-child(3)').textContent };
    });
    await page.evaluate(() => {
        const el = document.querySelector('.bt-card[data-id="b1"] input[data-f="threshold"]');
        el.value = '0.97';
        el.dispatchEvent(new Event('input', { bubbles: true }));
    });
    const liveRecall = await page.evaluate(() => document.querySelector('.bt-card[data-id="b1"] .bt-metrics tbody tr td:nth-child(3)').textContent);
    check(parseFloat(liveRecall) < parseFloat(before.recall), 'moving the threshold to 0.97 updates the numbers live (recall ' + before.recall + ' → ' + liveRecall + ')');
    await page.evaluate(() => {
        const el = document.querySelector('.bt-card[data-id="b1"] input[data-f="threshold"]');
        el.dispatchEvent(new Event('change', { bubbles: true }));
    });
    const after = await page.evaluate(async () => {
        const P = await import('/pose/behavior-predictions.js');
        const eth = await import('/pose/behavior-ethogram.js');
        const det = await import('/ui/behavior-detect.js');
        const s = (await import('/ui/app-state.js')).state.session;
        return { bouts: P.behaviorRunOf(s, 'b1').bouts.length, post: eth.behaviorById('b1').post, stale: P.isRunStale(s, 'b1', det.detectionEnv()) };
    });
    check(eq(after.post, { threshold: 0.97 }) && after.bouts !== before.bouts && !after.stale,
        'let go: kept in the ethogram ({threshold: 0.97}); the bouts re-derived at once (' + before.bouts + ' → ' + after.bouts + '), still fresh');
    await page.evaluate(() => {
        const el = document.querySelector('.bt-card[data-id="b1"] input[data-f="threshold"]');
        el.value = '0.5';
        el.dispatchEvent(new Event('change', { bubbles: true }));
    });
    check(await page.evaluate(async () => (await import('/pose/behavior-ethogram.js')).behaviorById('b1').post === undefined),
        'back to 0.5: nothing kept (the default is never written)');
    await page.keyboard.press('Escape');
    check(!(await page.$('.behavior-train-modal')), 'Esc closes the dialog');

    console.log('\n-- 4. active learning --');
    await openTab();
    await page.selectOption('select[data-act="al-behavior"]', 'b1');
    await page.click('[data-act="al-uncertain"]');
    await page.waitForSelector('.behavior-al-item', { timeout: 10000 });
    const item = await page.evaluate(async () => {
        const T = await import('/ui/behavior-train.js');
        const C = await import('/ui/behavior-coding.js');
        const AS = await import('/ui/app-state.js');
        const it = T.activeLearning().current;
        return { it, frame: AS.state.currentFrame, text: document.querySelector('.behavior-al-item').textContent, focus: !!C.behaviorUi().review };
    });
    check(item.it && item.it.kind === 'uncertain' && item.it.start >= 4000 && Math.abs(item.it.score - 0.5) <= 0.15,
        'Next uncertain ▸: an unlabelled stretch scoring near 0.5 (' + (item.it && [item.it.animal, item.it.start, item.it.end, item.it.score.toFixed(2)].join(' ')) + ')');
    check(item.frame >= item.it.start - 31 && item.frame <= item.it.start && item.focus, 'the playhead lands 1 s before it, boxed on the video');
    check(/dart\?/.test(item.text) && /uncertain/.test(item.text), 'shown with Accept / Reject / Skip');
    const staleOf = () => page.evaluate(async () => {
        const P = await import('/pose/behavior-predictions.js');
        const det = await import('/ui/behavior-detect.js');
        const s = (await import('/ui/app-state.js')).state.session;
        return [P.isRunStale(s, 'b1', det.detectionEnv()), P.isRunStale(s, 'b2', det.detectionEnv())];
    });
    check(eq(await staleOf(), [false, false]), 'both predictions stand before labelling');
    await page.click('.behavior-al-item [data-act="al-accept"]');
    const accepted = await page.evaluate(async (it) => {
        const L = await import('/pose/behavior-labels.js');
        const s = (await import('/ui/app-state.js')).state.session;
        return L.listBouts(s, { behaviorId: 'b1' }).filter(b => b.start === it.start && b.end === it.end && b.animal === it.animal).map(b => [b.present, b.source]);
    }, item.it);
    check(eq(accepted, [[true, 'a']]), 'Accept: the stretch is labelled present (source "a")');
    check(eq(await staleOf(), [true, false]), 'now dart\'s classifier run is stale (the labels it learned from changed); rear\'s rule run is not');
    const staleText = await page.evaluate(async () => {
        const s = document.getElementById('behaviorPredictionsSection'); if (s) s.open = true;
        (await import('/ui/behavior-panel.js')).refreshBehaviorPanel();
        return (document.querySelector('.behavior-stale') || {}).textContent || '';
    });
    check(/the labels dart learned from changed \(train again\)/.test(staleText), 'the Predictions section says why: "' + staleText.slice(0, 80) + '…"');
    // A planted contradiction: "not" over a predicted bout in the unlabelled part.
    const planted = await page.evaluate(async () => {
        const P = await import('/pose/behavior-predictions.js');
        const L = await import('/pose/behavior-labels.js');
        const s = (await import('/ui/app-state.js')).state.session;
        const bt = P.behaviorRunOf(s, 'b1').bouts.filter(b => b.start > 4500).sort((a, b) => (b.end - b.start) - (a.end - a.start))[0];
        L.labelRange(s, 'b1', bt.animal, '', bt.start, bt.end, false);
        return [bt.animal, bt.start, bt.end];
    });
    await page.click('[data-act="al-disagree"]');
    const dis = await page.evaluate(async () => (await import('/ui/behavior-train.js')).activeLearning().current);
    check(dis && dis.kind === 'false' && dis.animal === planted[0] && dis.start === planted[1] && dis.end === planted[2],
        'Next disagreement ▸: the predicted bout labelled "not" (' + planted.join(' ') + ')');
    await page.click('.behavior-al-item [data-act="al-skip"]');
    await page.evaluate(async () => { const L = await import('/pose/behavior-labels.js'); L.undoBehaviorEdit((await import('/ui/app-state.js')).state.session); });
    await page.evaluate(async () => {
        const T = await import('/ui/behavior-train.js');
        await T.trainClassifiers(['b1']);
        await (await import('/ui/behavior-detect.js')).detectAllFrames();
    });
    check(eq(await staleOf(), [false, false]) && await isBlue(), 'train and predict again: fresh, blue');

    console.log('\n-- 5. save → reopen --');
    const rt = await page.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        const fileio = await import('/import-export/file-io.js');
        const meta = await import('/import-export/behavior-metadata.js');
        const eth = await import('/pose/behavior-ethogram.js');
        const P = await import('/pose/behavior-predictions.js');
        const C = await import('/pose/behavior-classifier.js');
        const pd = await import('/pose/pose-data.js');
        const det = await import('/ui/behavior-detect.js');
        const s = AS.state.session;
        eth.setBehaviorPost('b1', { threshold: 0.55, minBout: 0.1 });
        const T = await import('/ui/behavior-train.js');
        T.applyClassifierPost('b1', { threshold: 0.55, minBout: 0.1, maxGap: 0 });
        const hash = C.behaviorModelOf('b1').hash;
        const before = P.behaviorRunOf(s, 'b1');
        const cam = new pd.Camera('cam0', [[1000, 0, 128], [0, 1000, 128], [0, 0, 1]], [0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0], [256, 256]);
        s.cameras = [cam];
        const views = [{ name: 'cam0', videoWidth: 256, videoHeight: 256, frameCount: AS.state.totalFrames }];
        const files = [{ name: 'cam0', assignedCamera: 'cam0', videoPath: 'cam0.mp4' }];
        const bytes = await window.SleapIO.saveSlpToBytes(fileio.buildSlpLabelsAllViews(s, views, files));
        const slp = await fileio.parseSlpViaSleapIO(new File([bytes], 'darts.slp'), () => {});
        const lucid = slp.sessions[0].metadata.lucid;
        meta.resetBehaviorState();
        const fresh = new pd.Session([], s.skeleton, [], 'Darts');
        fresh.identities = s.identities; fresh.instanceGroups = s.instanceGroups;
        meta.readBehaviorMetadata(fresh, lucid, slp.behaviorEvents);
        const run = P.behaviorRunOf(fresh, 'b1');
        AS.state.sessions = [fresh]; AS.state.session = fresh;
        const env = det.detectionEnv();
        const res = { hash, source: run && run.source, labels: run && run.labels === before.labels, bouts: run && run.bouts.length === before.bouts.length,
                      post: eth.behaviorById('b1').post, model: C.behaviorModelOf('b1'), stands: P.predictionsStand(fresh, env) };
        await T.trainClassifiers(['b1']);
        res.retrained = C.behaviorModelOf('b1').hash;
        res.standsAfter = P.predictionsStand(fresh, det.detectionEnv());
        return res;
    });
    check(rt.source === 'model:' + rt.hash && rt.labels && rt.bouts, 'the classifier\'s run comes back with its hash, its label print and its bouts');
    check(eq(rt.post, { threshold: 0.55, minBout: 0.1 }), 'and the threshold and shortest bout (ethogram post)');
    check(rt.model === null && rt.stands, 'the classifier itself is not saved, and the predictions still stand');
    check(rt.retrained === rt.hash && rt.standsAfter, 'retrained after the reopen: the same forest (' + rt.retrained + '), still standing');

    console.log('\n-- 6. files --');
    const [download] = await Promise.all([
        page.waitForEvent('download', { timeout: 30000 }),
        page.evaluate(async () => (await import('/ui/behavior-train.js')).exportClassifiers()),
    ]);
    const fname = download.suggestedFilename();
    const text = readFileSync(await download.path(), 'utf8');
    check(fname === 'dart.lucid-behavior.json' && JSON.parse(text).format === 'lucid-behavior-model', 'Export Classifiers… downloads ' + fname);
    const imp = await page.evaluate(async (text) => {
        const meta = await import('/import-export/behavior-metadata.js');
        const eth = await import('/pose/behavior-ethogram.js');
        const C = await import('/pose/behavior-classifier.js');
        const P = await import('/pose/behavior-predictions.js');
        const AS = await import('/ui/app-state.js');
        const s = AS.state.session;
        await (await import('/ui/behavior-detect.js')).detectAllFrames();
        const before = P.behaviorRunOf(s, 'b1');
        const scoresBefore = Array.from(before.scores.get('id_1\u0001').subarray(4000, 4100));
        meta.resetBehaviorState();
        P.clearBehaviorRuns(s);
        const T = await import('/ui/behavior-train.js');
        const res = await T.importClassifierFile(new File([text], 'dart.lucid-behavior.json'));
        const b = eth.behaviors()[0];
        await (await import('/ui/behavior-detect.js')).detectAllFrames();
        const run = P.behaviorRunOf(s, b.id);
        return { res, name: b.name, post: b.post, imported: C.behaviorModelOf(b.id).imported, source: run.source,
                 same: Array.from(run.scores.get('id_1\u0001').subarray(4000, 4100)).every((v, i) => Object.is(v, scoresBefore[i])) };
    }, text);
    check(imp.res.imported === 1 && imp.name === 'dart' && eq(imp.post, { threshold: 0.55, minBout: 0.1 }) && imp.imported,
        'imported into a project without it: dart is back, with its threshold, as an imported classifier');
    check(imp.source === 'model:' + rt.hash && imp.same, 'and it scores every frame exactly as before');

    console.log('\n-- 8. no video --');
    const noVideo = await page.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        const fileio = await import('/import-export/file-io.js');
        const meta = await import('/import-export/behavior-metadata.js');
        const eth = await import('/pose/behavior-ethogram.js');
        const P = await import('/pose/behavior-predictions.js');
        const C = await import('/pose/behavior-classifier.js');
        const T = await import('/ui/behavior-train.js');
        const det = await import('/ui/behavior-detect.js');
        const pd = await import('/pose/pose-data.js');
        const s = AS.state.session;
        const id = eth.behaviors()[0].id;
        AS.state.fps = 60;
        P.clearBehaviorRuns(s);
        await T.trainClassifiers([id]);
        await det.detectAllFrames();
        const hash = C.behaviorModelOf(id).hash;
        const made = P.behaviorRunOf(s, id);
        const views = [{ name: 'cam0', videoWidth: 256, videoHeight: 256, frameCount: AS.state.totalFrames }];
        const files = [{ name: 'cam0', assignedCamera: 'cam0', videoPath: 'cam0.mp4' }];
        const bytes = await window.SleapIO.saveSlpToBytes(fileio.buildSlpLabelsAllViews(s, views, files));
        const slp = await fileio.parseSlpViaSleapIO(new File([bytes], 'sixty.slp'), () => {});
        meta.resetBehaviorState();
        const fresh = new pd.Session([], s.skeleton, [], 'Darts');
        fresh.identities = s.identities; fresh.instanceGroups = s.instanceGroups;
        meta.readBehaviorMetadata(fresh, slp.sessions[0].metadata.lucid, slp.behaviorEvents);
        AS.state.sessions = [fresh]; AS.state.session = fresh; AS.state.fps = 30;
        const res = { id, hash, madeFps: made.fps, rate: det.detectionFps(), stands: P.predictionsStand(fresh, det.detectionEnv()) };
        await T.trainClassifiers([id]);
        res.retrained = C.behaviorModelOf(id).hash;
        res.stale = P.predictionStaleness(fresh, det.detectionEnv());
        res.fps = AS.state.fps;
        return res;
    });
    check(noVideo.madeFps === 60 && noVideo.rate === 60 && noVideo.fps === 30 && noVideo.stands, 'made at 60 fps, reopened at state.fps 30: measured at the 60 on record, and they stand');
    check(noVideo.retrained === noVideo.hash && !noVideo.stale.model.length && !noVideo.stale.settings.length,
        'retrained: the same forest as at 60 fps (' + noVideo.retrained + '), so no "newer classifier"');
    await page.click('[data-menu="behavior"]');
    await page.click('#menuBehaviorTrain');
    const minBout = `.bt-card[data-id="${noVideo.id}"] input[data-f="minBout"]`;
    await page.waitForSelector(minBout, { timeout: 10000 });
    // Darts last 15–45 frames: at 60 fps a 0.6 s shortest bout keeps few of them, at 30 most.
    await page.fill(minBout, '0.6');
    const live = await page.evaluate(async (id) => {
        const C = await import('/pose/behavior-classifier.js');
        const post = { threshold: 0.55, minBout: 0.6, maxGap: 0 };
        const at = (fps) => Math.round(100 * C.evaluateAt(C.behaviorModelOf(id).cv, post, fps, false).bout.recall);
        return { shown: document.querySelector('.bt-card[data-id="' + id + '"] .bt-live .bt-metrics tbody tr:nth-child(2) td:nth-child(3)').textContent, at60: at(60), at30: at(30) };
    }, noVideo.id);
    check(live.at60 !== live.at30 && parseFloat(live.shown) === live.at60,
        'the threshold dialog\'s live bout recall uses that rate too: ' + live.shown + ' (at 30 fps it would be ' + live.at30 + '%)');
    await page.keyboard.press('Escape');
    const [sixtyFile] = await Promise.all([
        page.waitForEvent('download', { timeout: 30000 }),
        page.evaluate(async () => (await import('/ui/behavior-train.js')).exportClassifiers()),
    ]);
    check(JSON.parse(readFileSync(await sixtyFile.path(), 'utf8')).fps === 60, 'and an exported model file records it');

    console.log('\n-- page errors --');
    check(errs.length === 0, 'no page errors' + (errs.length ? ': ' + errs.join(' | ') : ''));
} catch (e) {
    console.error(e);
    fails++;
} finally {
    if (browser) await browser.close();
    server.kill();
}
console.log(fails ? `\nFAIL (${fails})` : '\nPASS');
process.exit(fails ? 1 : 0);
