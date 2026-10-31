/**
 * behavior-rules.mjs — rule detectors and the predictions review, in the real
 * app.
 *
 *  1. A planted rearing gives exact bouts: animal 0's nose rises above half a
 *     body length on frames 500–620 only, and "Nose height > 0.5" detects
 *     exactly that bout; the same run on a forced 3-worker pool gives the same
 *     per-frame scores, bit for bit; a social (mutual) detector finds a
 *     planted nose-to-nose meeting.
 *  2. The toolbar: the Behavior button runs all frames and turns blue while
 *     the predictions stand; This frame fills the Predictions section's
 *     readout (conditions against thresholds); Defining Plane Mode locks the
 *     run items.
 *  3. Review: the Predictions section lists the bouts; the selected one is
 *     boxed on the video (dashed, "rear?") while the playhead is in it; Accept
 *     labels one present (source "a") and Reject labels one "not", each one
 *     undo step; Accept all in range asks first.
 *  4. Stale: moving a pose (an edit that marks the project dirty) makes the
 *     predictions stale — banner, button no longer blue; detecting again
 *     clears it. Changing a detector makes that behavior stale.
 *  5. The settings dialog: opens from the menu, shows the live bout count
 *     for the detector, follows a threshold change, adds a behavior from a
 *     template, closes on Esc.
 *  6. Save → reopen: predictions, detectors and feature settings come back
 *     through the eager writer; a project with none writes no key.
 *  7. 2D fallback: a single-camera session with no 3D detects from its 2D.
 *  8. Reopened without its videos: predictions made at 60 fps stand through both
 *     real readers, though `state.fps` is the default 30 (a `.slp` does not save
 *     the rate) and stays so. A changed rule, body or floor still reads stale;
 *     so does a rate that is really different: a 10 fps video, the fps box, a
 *     campy manifest. The 60 fps video keeps them fresh. Everything measured in
 *     seconds uses the rate on record: Detect gives the saved bouts and the same
 *     per-frame scores, bit for bit, and This frame and the settings dialog the
 *     same numbers, with `state.fps` still 30. Other sessions go by what they
 *     recorded too; a label edit keeps the labels' rate. What the Behavior UI
 *     shows in seconds follows (the tab, a bout's lead-in, the timeline's menu,
 *     Label Behaviors mode, States), and so do real files: an export's seconds
 *     and an import's frames.
 *
 * Run: node tests/e2e/behavior-rules.mjs   (spawns its own http.server; PORT= to move it)
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8197);
const VIDEO_60 = path.join(repoRoot, 'tests', 'fixtures', 'barcode-60', 'barcode-60.mp4');
const VIDEO_10 = path.join(repoRoot, 'tests', 'fixtures', 'bframes-test', 'bframes-test.mp4');

let fails = 0;
const check = (c, m) => { console.log((c ? '  ✓ ' : '  ✗ ') + m); if (!c) fails++; };
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1200));

let browser;
try {
    browser = await chromium.launch();
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    const errs = [];
    page.on('pageerror', e => errs.push('pageerror: ' + String(e).slice(0, 300)));
    page.on('console', m => { if (m.type() === 'error') errs.push('console: ' + m.text().slice(0, 300)); });
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state, null, { timeout: 30000 });

    // Two mice walking circles on the floor (z = 0); mouse 0 rears on 500–620;
    // on 1500–1560 they stand nose to nose, having walked there over the 2 s
    // before and walking back after (a teleport would read as broken tracking).
    await page.evaluate(async () => {
        const pd = await import('/pose/pose-data.js');
        const AS = await import('/ui/app-state.js');
        const NODES = ['Nose', 'Head', 'Neck', 'Trunk', 'TTI', 'Tail1'];
        const ALONG = [45, 30, 15, 0, -40, -70], H = [15, 25, 30, 30, 15, 5];
        const T = 3000;
        const session = new pd.Session([], new pd.Skeleton('m', NODES, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]), [], 'Rules');
        for (let i = 0; i < 2; i++) session.addIdentity('id_' + i);
        let gid = 1;
        const walk = (k, t) => { const ang = 0.5 * t / 30 + k * 3; return [700 * k + 200 * Math.cos(ang), 200 * Math.sin(ang), ang + Math.PI / 2]; };
        // Where they meet, and the step before (noses 0.47 BL apart, one 30 mm step away).
        const MEET = [[0, 0, 0], [100, 0, Math.PI]], NEAR = [[0, 0, 0], [130, 0, Math.PI]];
        for (let t = 0; t < T; t++) {
            session.instanceGroups.set(t, [0, 1].map(k => {
                const g = new pd.InstanceGroup(gid++, session.identities[k].id);
                const p = new Float64Array(NODES.length * 3);
                let cx, cy, hd;
                if (t >= 1500 && t <= 1560) [cx, cy, hd] = MEET[k];
                else if (t >= 1440 && t < 1500) {
                    const f = (t - 1440) / 59, w = walk(k, 1440);
                    cx = w[0] + f * (NEAR[k][0] - w[0]); cy = w[1] + f * (NEAR[k][1] - w[1]); hd = NEAR[k][2];
                } else if (t > 1560 && t <= 1620) {
                    const f = (t - 1561) / 59, w = walk(k, 1620);
                    cx = NEAR[k][0] + f * (w[0] - NEAR[k][0]); cy = NEAR[k][1] + f * (w[1] - NEAR[k][1]); hd = NEAR[k][2];
                } else [cx, cy, hd] = walk(k, t);
                const rear = k === 0 && t >= 500 && t <= 620;
                for (let n = 0; n < NODES.length; n++) {
                    p[3 * n] = cx + ALONG[n] * Math.cos(hd); p[3 * n + 1] = cy + ALONG[n] * Math.sin(hd);
                    p[3 * n + 2] = rear && n <= 2 ? [95, 85, 65][n] : H[n];
                }
                // Broken tracking: mouse 1's nose 400 mm up (≈ 4.7 BL) on 2000–2010.
                if (k === 1 && t >= 2000 && t <= 2010) p[2] = 400;
                g.points3d = p;
                return g;
            }));
        }
        AS.state.sessions = [session]; AS.state.activeSessionIdx = 0; AS.state.session = session;
        AS.state.totalFrames = T; AS.state.currentFrame = 0; AS.state.views = []; AS.state.fps = 30;
        AS.timeline.setTotalFrames(T); AS.timeline.setData(session);
        (await import('/ui/info-panel.js')).updateInfoPanel();
        const eth = await import('/pose/behavior-ethogram.js');
        const rear = eth.addBehavior({ name: 'rear', key: 'r' }).behavior;
        eth.setBehaviorDetector(rear.id, { conditions: [{ feature: 'Nose height', op: '>', value: 0.5 }], minDuration: 0, maxGap: 0 });
        const n2n = eth.addBehavior({ name: 'nose-to-nose', key: 'n', social: true, mutual: true }).behavior;
        eth.setBehaviorDetector(n2n.id, { conditions: [{ feature: 'head–head distance', op: '<', value: 0.3 }], minDuration: 0.5, maxGap: 0 });
        (await import('/ui/behavior-coding.js')).requestBehaviorRepaint({ ethogram: true });
    });
    const predicted = () => page.evaluate(async () => {
        const P = await import('/pose/behavior-predictions.js');
        return P.predictedBouts((await import('/ui/app-state.js')).state.session).map(b => [b.behaviorId, b.animal, b.target, b.start, b.end]);
    });
    const scoresHash = () => page.evaluate(async () => {
        const P = await import('/pose/behavior-predictions.js');
        const s = (await import('/ui/app-state.js')).state.session;
        let h = 0x811c9dc5, n = 0;
        P.behaviorRuns(s).forEach(r => Array.from(r.scores.keys()).sort().forEach(k => {
            const u = new Uint32Array(r.scores.get(k).buffer);
            for (let i = 0; i < u.length; i++) { h = Math.imul(h ^ u[i], 0x01000193); n++; }
        }));
        return (h >>> 0) + ':' + n;
    });
    const isBlue = () => page.evaluate(() => document.getElementById('tbBehavior').classList.contains('active'));
    const goTo = (f) => page.evaluate(async (f) => (await import('/pose/initialization.js')).navigateToFrame(f), f);

    console.log('\n-- 1. a planted rearing gives exact bouts --');
    check(!(await isBlue()), 'before any run the Behavior button is not blue');
    await page.evaluate(() => { window.LUCID_BEHAVIOR_WORKER = 0; });
    await page.click('#tbBehavior');
    await page.waitForFunction(() => document.getElementById('tbBehavior').classList.contains('active'), null, { timeout: 30000 });
    const inline = await predicted();
    check(eq(inline.filter(b => b[0] === 'b1'), [['b1', 'id_0', '', 500, 620]]),
        'rear: exactly frames 500–620, mouse 0 only (mouse 1\'s stray nose on 2000–2010 is skipped, not a rear)');
    const detectStatus = await page.evaluate(() => document.getElementById('statusText').textContent);
    check(/skipped broken tracking on [\d.]+% of frames/.test(detectStatus), 'the status says what it skipped (' + (detectStatus.match(/skipped[^·]*/) || [''])[0].trim() + ')');
    check(eq(inline.filter(b => b[0] === 'b2'), [['b2', 'id_0', 'id_1', 1500, 1560]]), 'nose-to-nose (mutual, 0.5 s minimum): the meeting, once per pair ' +
        JSON.stringify(inline.filter(b => b[0] === 'b2')));
    const hInline = await scoresHash();
    await page.evaluate(() => { window.LUCID_BEHAVIOR_WORKER = 3; });
    const parallel = await page.evaluate(async () => {
        const det = await import('/ui/behavior-detect.js');
        const pool = await import('/pose/behavior-pool.js');
        const orig = pool.runBehaviorChunks;
        await det.detectAllFrames();
        return true;
    });
    check(eq(await predicted(), inline), 'on a forced 3-worker pool: the same bouts');
    check((await scoresHash()) === hInline, 'and the same per-frame scores, bit for bit (' + hInline + ')');
    await page.evaluate(() => { delete window.LUCID_BEHAVIOR_WORKER; });

    console.log('\n-- 2. toolbar --');
    check(await isBlue(), 'blue while the predictions stand');
    await goTo(560);
    await page.hover('#tbBehavior');
    await page.click('#tbBehaviorFrame');
    await page.waitForSelector('.behavior-readout-row.is-on', { timeout: 10000 });
    const readout = await page.evaluate(() => Array.from(document.querySelectorAll('.behavior-readout-row')).map(r => r.textContent));
    check(readout.length >= 1 && /rear/.test(readout[0]) && /id_0/.test(readout[0]) && /✓ Nose height = /.test(readout[0]),
        'This frame (560): rear · id_0, with its condition met (' + (readout[0] || '').slice(0, 70) + ')');
    check(readout.some(r => /not rear/.test(r) && /id_1/.test(r) && /✗ Nose height/.test(r)), 'and mouse 1: not rear, condition not met');
    await goTo(2005);
    await page.click('#tbBehaviorFrame').catch(async () => { await page.hover('#tbBehavior'); await page.click('#tbBehaviorFrame'); });
    await page.waitForFunction(() => /2,006|2006/.test((document.querySelector('.behavior-readout-head') || {}).textContent || ''), null, { timeout: 10000 });
    const strayRows = await page.evaluate(() => Array.from(document.querySelectorAll('.behavior-readout-row')).map(r => r.textContent));
    check(strayRows.some(r => /rear/.test(r) && /id_1/.test(r) && /tracking broken here/.test(r)),
        'This frame on the stray nose (2005): mouse 1 is skipped, said so');
    await goTo(560);
    check(await page.evaluate(() => document.querySelector('.panel-tab[data-tab="tabBehavior"]').classList.contains('active')),
        'the Behavior tab opens to show it');
    const lock = await page.evaluate(async () => {
        const pd = await import('/ui/plane-definition.js');
        pd.togglePlaneMode();
        await new Promise(r => setTimeout(r, 100));
        (await import('/ui/rendering.js')).drawAllOverlays((await import('/ui/app-state.js')).state.currentFrame);
        const ids = ['tbBehavior', 'tbBehaviorFrame', 'tbBehaviorAllFrames'];
        const locked = ids.map(id => document.getElementById(id).disabled);
        const modeLive = !document.getElementById('tbBehaviorMode').disabled;
        pd.togglePlaneMode();
        await new Promise(r => setTimeout(r, 100));
        (await import('/ui/rendering.js')).drawAllOverlays((await import('/ui/app-state.js')).state.currentFrame);
        return { locked, modeLive, after: ids.map(id => document.getElementById(id).disabled) };
    });
    check(lock.locked.every(Boolean) && lock.after.every(d => !d), 'Defining Plane Mode disables Behavior, This frame, All frames (and lifts it)');
    check(lock.modeLive, 'Label Behaviors mode stays live in the menu');

    console.log('\n-- 3. review --');
    await page.evaluate(() => { const s = document.getElementById('behaviorPredictionsSection'); if (s) s.open = true; });
    await page.waitForSelector('.behavior-pred', { timeout: 10000 });
    const listed = await page.$$eval('.behavior-pred', rows => rows.length);
    check(listed === 2, 'the Predictions section lists both predicted bouts');
    await page.click('.behavior-pred[data-pkey^="b1|"] .behavior-bout-main');
    const focusBox = await page.evaluate(async () => {
        const C = await import('/ui/behavior-coding.js');
        const s = (await import('/ui/app-state.js')).state.session;
        const inside = C.behaviorBoxesAt(s, 550, { live: true });
        const outside = C.behaviorBoxesAt(s, 700, { live: true });
        const exported = C.behaviorBoxesAt(s, 550, {});
        const it = inside && inside.items.find(i => i.review);
        return { animals: it && it.animals, text: it && it.parts.map(p => p[0]).join(''), dashed: it && it.dashed,
                 outside: !!(outside && outside.items.some(i => i.review)), exported: !!(exported && exported.items.some(i => i.review)) };
    });
    check(eq(focusBox.animals, ['id_0']) && focusBox.dashed && /^rear\? · id_0 · \d+%$/.test(focusBox.text),
        'the selected bout is boxed on the video, dashed: "' + focusBox.text + '"');
    check(!focusBox.outside && !focusBox.exported, 'only while the playhead is in it, and never in an exported video');
    const kept = await page.evaluate(async () => {
        const C = await import('/ui/behavior-coding.js');
        const P = await import('/ui/behavior-panel.js');
        P.refreshBehaviorPanel();
        await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
        return !!C.behaviorUi().review;
    });
    check(kept, 'a repaint of the panel keeps it');
    await page.click('.behavior-pred[data-pkey^="b1|"] [data-act="accept"]');
    const afterAccept = await page.evaluate(async () => {
        const L = await import('/pose/behavior-labels.js');
        return L.listBouts((await import('/ui/app-state.js')).state.session).map(b => [b.behaviorId, b.animal, b.start, b.end, b.present, b.source]);
    });
    check(eq(afterAccept, [['b1', 'id_0', 500, 620, true, 'a']]), 'Accept: rear 500–620 is a label, source "a" (accepted prediction)');
    await page.waitForFunction(() => document.querySelectorAll('.behavior-pred').length === 1, null, { timeout: 5000 });
    check(true, 'and leaves the to-review list');
    check(await page.evaluate(() => !!document.querySelector('.behavior-pred.is-current[data-pkey^="b2|"]')),
        'and moves on to the next bout, selected');
    await page.click('.behavior-pred[data-pkey^="b2|"] [data-act="reject"]');
    const afterReject = await page.evaluate(async () => {
        const L = await import('/pose/behavior-labels.js');
        return L.listBouts((await import('/ui/app-state.js')).state.session, { behaviorId: 'b2' }).map(b => [b.animal, b.target, b.start, b.end, b.present]);
    });
    check(eq(afterReject, [['id_0', 'id_1', 1500, 1560, false]]), 'Reject: nose-to-nose 1500–1560 labelled "not"');
    await page.evaluate(async () => {
        const L = await import('/pose/behavior-labels.js');
        const s = (await import('/ui/app-state.js')).state.session;
        L.undoBehaviorEdit(s); L.undoBehaviorEdit(s);
    });
    await page.waitForFunction(() => document.querySelectorAll('.behavior-pred').length === 2, null, { timeout: 5000 });
    check(true, 'two undos take both back: both listed again');
    await page.click('[data-act="accept-range"]');
    await page.waitForSelector('#behaviorAcceptRangeConfirm', { timeout: 5000 });
    const confirmText = await page.textContent('#behaviorAcceptRangeConfirm');
    check(/Accept 2 predicted bouts\?/.test(confirmText), 'Accept all in range asks first ("Accept 2 predicted bouts?")');
    await page.click('#behaviorAcceptRangeConfirm [data-confirm]');
    const all = await page.evaluate(async () => {
        const L = await import('/pose/behavior-labels.js');
        const s = (await import('/ui/app-state.js')).state.session;
        const n = L.listBouts(s, { present: true }).length;
        L.undoBehaviorEdit(s);
        return [n, L.listBouts(s).length];
    });
    check(all[0] === 2 && all[1] === 0, 'it accepts both, as one undo step');

    console.log('\n-- 4. stale --');
    await page.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        // Tail1 (excluded from features, so nothing new is detected) moves sideways.
        AS.state.session.instanceGroups.get(700)[0].points3d[15] += 40;
        (await import('/import-export/save-load.js')).markDirty();
    });
    await page.waitForFunction(() => !document.getElementById('tbBehavior').classList.contains('active'), null, { timeout: 10000 });
    check(true, 'a pose moved: the button is no longer blue');
    await page.waitForSelector('.behavior-stale', { timeout: 5000 });
    check(/the poses changed/.test(await page.textContent('.behavior-stale')), 'the Predictions section says they are out of date, and why');
    await page.click('.behavior-stale [data-act="detect-all"]');
    await page.waitForFunction(() => document.getElementById('tbBehavior').classList.contains('active'), null, { timeout: 30000 });
    check(!(await page.$('.behavior-stale')), 'Detect again: fresh, blue');
    await page.evaluate(async () => {
        const eth = await import('/pose/behavior-ethogram.js');
        eth.setBehaviorDetector('b1', { conditions: [{ feature: 'Nose height', op: '>', value: 0.6 }] });
        (await import('/ui/behavior-coding.js')).requestBehaviorRepaint({ ethogram: true, predictions: true });
    });
    await page.waitForSelector('.behavior-stale', { timeout: 5000 });
    check(/settings of rear changed/.test(await page.textContent('.behavior-stale')), 'a changed detector: "the settings of rear changed"');
    await page.evaluate(async () => {
        const eth = await import('/pose/behavior-ethogram.js');
        eth.setBehaviorDetector('b1', { conditions: [{ feature: 'Nose height', op: '>', value: 0.5 }], minDuration: 0, maxGap: 0 });
        (await import('/ui/behavior-coding.js')).requestBehaviorRepaint({ ethogram: true, predictions: true });
    });

    console.log('\n-- 5. settings dialog --');
    await page.click('[data-menu="behavior"]');
    await page.click('#menuBehaviorRules');
    await page.waitForSelector('.behavior-rules-modal', { timeout: 10000 });
    await page.waitForFunction(() => /1 bout/.test((document.querySelector('.bd-card[data-id="b1"] .bd-live') || {}).textContent || ''), null, { timeout: 20000 })
        .catch(async (e) => { console.log('live:', await page.evaluate(() => Array.from(document.querySelectorAll('.bd-live')).map(x => x.textContent))); throw e; });
    check(true, 'rear\'s card shows the live count: 1 bout, as detected');
    const hist = await page.evaluate(() => (document.querySelector('.bd-card[data-id="b1"] .bd-hist-caption') || {}).textContent || '');
    check(/Nose height over [\d,]+ animal-frames/.test(hist) && /threshold 0\.5/.test(hist), 'with the feature\'s histogram and the threshold (' + hist.slice(0, 60) + '…)');
    await page.fill('.bd-card[data-id="b1"] input[data-f="value"]', '2');
    await page.waitForFunction(() => /^0 bouts/.test((document.querySelector('.bd-card[data-id="b1"] .bd-live') || {}).textContent || ''), null, { timeout: 5000 });
    check(true, 'a threshold of 2 BL: 0 bouts, live, before it is committed');
    await page.fill('.bd-card[data-id="b1"] input[data-f="value"]', '0.5');
    await page.press('.bd-card[data-id="b1"] input[data-f="value"]', 'Tab');
    await page.selectOption('select[data-act="new-from-template"]', 'contact');
    await page.waitForSelector('.bd-card[data-id="b3"]', { timeout: 5000 });
    const contact = await page.evaluate(async () => {
        const eth = await import('/pose/behavior-ethogram.js');
        const b = eth.behaviorById('b3');
        return b && [b.name, b.social, b.mutual, b.detector.conditions[0].feature];
    });
    check(eq(contact, ['contact', true, true, 'min node distance']), 'a template adds "contact" (mutual) with its detector');
    const tipOff = await page.evaluate(() => (document.getElementById('bdReprojTip') || {}).textContent || '');
    check(/reprojection error threshold of about 25 px/.test(tipOff) && /Tracking Wizard/.test(tipOff),
        'with no reprojection error threshold set, the Body section tips to set one (about 25 px)');
    await page.keyboard.press('Escape');
    check(!(await page.$('.behavior-rules-modal')), 'Esc closes it');
    await page.evaluate(async () => (await import('/ui/settings.js')).setTrackingThresholds({ reprojErrorThreshold: 25 }));
    await page.click('[data-menu="behavior"]');
    await page.click('#menuBehaviorRules');
    await page.waitForSelector('.behavior-rules-modal #bdBody .bd-note', { timeout: 10000 });
    check(!(await page.$('#bdReprojTip')), 'with one set (25 px), no tip');
    await page.keyboard.press('Escape');
    await page.evaluate(async () => (await import('/ui/settings.js')).setTrackingThresholds({ reprojErrorThreshold: 0 }));
    await page.evaluate(async () => {
        const eth = await import('/pose/behavior-ethogram.js');
        eth.removeBehavior('b3');
    });

    console.log('\n-- 6. save → reopen --');
    const rt = await page.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        const fileio = await import('/import-export/file-io.js');
        const meta = await import('/import-export/behavior-metadata.js');
        const eth = await import('/pose/behavior-ethogram.js');
        const P = await import('/pose/behavior-predictions.js');
        const pd = await import('/pose/pose-data.js');
        const s = AS.state.session;
        eth.setFeatureSettings({ nodeRoles: { head: 'Nose' } });
        const before = P.predictedBouts(s).map(b => [b.behaviorId, b.animal, b.target, b.start, b.end, Math.round(b.score * 1000) / 1000]);
        const cam = new pd.Camera('cam0', [[1000, 0, 128], [0, 1000, 128], [0, 0, 1]], [0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0], [256, 256]);
        s.cameras = [cam];
        const views = [{ name: 'cam0', videoWidth: 256, videoHeight: 256, frameCount: AS.state.totalFrames }];
        const files = [{ name: 'cam0', assignedCamera: 'cam0', videoPath: 'cam0.mp4' }];
        const bytes = await window.SleapIO.saveSlpToBytes(fileio.buildSlpLabelsAllViews(s, views, files));
        const slp = await fileio.parseSlpViaSleapIO(new File([bytes], 'rules.slp'), () => {});
        const lucid = slp.sessions[0].metadata.lucid;
        meta.resetBehaviorState();
        const fresh = new pd.Session([], s.skeleton, [], 'reopened');
        meta.readBehaviorMetadata(fresh, lucid, slp.behaviorEvents);
        const after = P.predictedBouts(fresh).map(b => [b.behaviorId, b.animal, b.target, b.start, b.end, b.score]);
        const res = { before, after, detector: eth.behaviorById('b1') && eth.behaviorById('b1').detector, features: eth.featureSettings(),
                      stands: P.predictionsStand(fresh, (await import('/ui/behavior-detect.js')).detectionEnv()) };
        // A project with no behavior data writes none of the keys.
        meta.resetBehaviorState();
        const plain = new pd.Session([cam], s.skeleton, [], 'plain');
        plain.instanceGroups = s.instanceGroups;
        const bytes2 = await window.SleapIO.saveSlpToBytes(fileio.buildSlpLabelsAllViews(plain, views, files));
        const l2 = (await fileio.parseSlpViaSleapIO(new File([bytes2], 'plain.slp'), () => {})).sessions[0].metadata.lucid;
        res.plainKeys = meta.BEHAVIOR_METADATA_KEYS.filter(k => Object.prototype.hasOwnProperty.call(l2, k));
        return res;
    });
    check(rt.after.length === 2 && eq(rt.after, rt.before), 'the predicted bouts and their scores come back (' + rt.after.length + ')');
    check(rt.detector && rt.detector.conditions[0].feature === 'Nose height' && eq(rt.features, { nodeRoles: { head: 'Nose' } }),
        'the detector and the changed feature setting come back');
    check(rt.stands, 'and they still stand (same detector and settings)');
    check(rt.plainKeys.length === 0, 'a project with no behavior data writes none of the behavior keys');

    console.log('\n-- 7. 2D fallback --');
    const twoD = await page.evaluate(async () => {
        const pd = await import('/pose/pose-data.js');
        const AS = await import('/ui/app-state.js');
        const eth = await import('/pose/behavior-ethogram.js');
        const meta = await import('/import-export/behavior-metadata.js');
        const det = await import('/ui/behavior-detect.js');
        const P = await import('/pose/behavior-predictions.js');
        meta.resetBehaviorState();
        const NODES = ['nose', 'head', 'trunk', 'tail_base'];
        const cam = new pd.Camera('top', [[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], [0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 1000], [640, 480]);
        const s = new pd.Session([cam], new pd.Skeleton('m', NODES, [[0, 1], [1, 2], [2, 3]]), ['t0', 't1'], 'TwoD');
        const ids = [s.addIdentity('m1').id, s.addIdentity('m2').id];
        const T = 1200;
        for (let t = 0; t < T; t++) {
            const fg = new pd.FrameGroup(t);
            for (let k = 0; k < 2; k++) {
                // m1 runs (40 px per frame) on 300–399, otherwise both sit still.
                const x = 100 + 200 * k + (k === 0 && t >= 300 && t < 400 ? (t - 300) * 4 : k === 0 && t >= 400 ? 400 : 0);
                const pts = NODES.map((n, i) => [x + 30 - 15 * i, 200 + (t % 2) * 0.2]);
                const inst = new pd.Instance(pts, k, 'predicted', 0.9);
                fg.addInstance('top', inst);
                s.setFrameIdentity(t, 'top', k, ids[k]);
            }
            s.addFrameGroup(fg);
        }
        AS.state.sessions = [s]; AS.state.activeSessionIdx = 0; AS.state.session = s; AS.state.totalFrames = T; AS.state.fps = 30;
        const run = eth.addBehavior({ name: 'run' }).behavior;
        eth.setBehaviorDetector(run.id, { conditions: [{ feature: 'centroid speed · mean ±0.1 s', op: '>', value: 1 }], minDuration: 0.5, maxGap: 0 });
        const ctx = det.buildBehaviorContext(s);
        await det.detectAllFrames();
        const bouts = P.predictedBouts(s).map(b => [b.animal, b.start, b.end]);
        return { mode: ctx.input.mode, camera: ctx.input.camera, animals: ctx.input.animals, heights: ctx.catalog.byName.has('nose height'), bouts };
    });
    check(twoD.mode === '2d' && twoD.camera === 'top' && eq(twoD.animals, ['m1', 'm2']), 'no 3D: the features read camera "top"\'s 2D, per identity');
    check(!twoD.heights, 'without heights');
    check(twoD.bouts.length === 1 && twoD.bouts[0][0] === 'm1' && Math.abs(twoD.bouts[0][1] - 300) <= 4 && Math.abs(twoD.bouts[0][2] - 399) <= 4,
        'm1 running on 300–399 is detected (' + JSON.stringify(twoD.bouts) + ')');

    console.log('\n-- 8. reopened without its videos --');
    const openApp = async () => {
        const p = await browser.newPage({ viewport: { width: 1440, height: 900 } });
        p.on('pageerror', e => errs.push('pageerror: ' + String(e).slice(0, 300)));
        await p.goto(`http://localhost:${PORT}/index.html`);
        await p.waitForFunction(() => window.__lucid && window.__lucid.state && window.SleapIO, null, { timeout: 30000 });
        return p;
    };
    // Two mice on two cameras; mouse 0 rears on 500–620. Detected at 60 fps, saved by the real writer.
    const maker = await openApp();
    const made = await maker.evaluate(async () => {
        const pd = await import('/pose/pose-data.js');
        const AS = await import('/ui/app-state.js');
        const fileio = await import('/import-export/file-io.js');
        const eth = await import('/pose/behavior-ethogram.js');
        const det = await import('/ui/behavior-detect.js');
        const P = await import('/pose/behavior-predictions.js');
        const NODES = ['Nose', 'Head', 'Neck', 'Trunk', 'TTI', 'Tail1'];
        const ALONG = [45, 30, 15, 0, -40, -70], H = [15, 25, 30, 30, 15, 5];
        const T = 1200, CAMS = ['cam0', 'cam1'];
        const mtx = [[1000, 0, 128], [0, 1000, 128], [0, 0, 1]];
        const cams = CAMS.map((n, i) => new pd.Camera(n, mtx, [0, 0, 0, 0, 0], [0, 0.1 * i, 0], [10 * i, 0, -1000], [256, 256]));
        const s = new pd.Session(cams, new pd.Skeleton('m', NODES, [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]), ['t0', 't1'], 'Sixty');
        const ids = ['id_0', 'id_1'].map(nm => s.addIdentity(nm).id);
        for (let t = 0; t < T; t++) {
            const fg = new pd.FrameGroup(t);
            s.addFrameGroup(fg);
            s.instanceGroups.set(t, ids.map((id, k) => {
                const g = new pd.InstanceGroup(t * 10 + k + 1, id);
                for (const cn of CAMS) {
                    const inst = new pd.Instance(NODES.map((_, n) => [50 + 2 * n + 80 * k + t * 0.1, 60 + n]), k, 'predicted', 0.9);
                    g.addInstance(cn, inst);
                    fg.addInstance(cn, inst);
                }
                const ang = t / 120 + k * 3, cx = 700 * k + 200 * Math.cos(ang), cy = 200 * Math.sin(ang), hd = ang + Math.PI / 2;
                const rear = k === 0 && t >= 500 && t <= 620;
                const p = new Float64Array(NODES.length * 3);
                for (let n = 0; n < NODES.length; n++) {
                    p[3 * n] = cx + ALONG[n] * Math.cos(hd); p[3 * n + 1] = cy + ALONG[n] * Math.sin(hd);
                    p[3 * n + 2] = rear && n <= 2 ? [95, 85, 65][n] : H[n];
                }
                g.points3d = p;
                return g;
            }));
        }
        AS.state.sessions = [s]; AS.state.activeSessionIdx = 0; AS.state.session = s;
        AS.state.totalFrames = T; AS.state.currentFrame = 0; AS.state.views = []; AS.state.fps = 60;
        AS.timeline.setTotalFrames(T); AS.timeline.setData(s);
        const rear = eth.addBehavior({ name: 'rear', key: 'r' }).behavior;
        eth.setBehaviorDetector(rear.id, { conditions: [{ feature: 'Nose height', op: '>', value: 0.5 }], minDuration: 0.1, maxGap: 0 });
        // Two the rate decides. The 121 frames of rearing are 2 s at 60 fps, 4 s at 30; the
        // mice walk at about 1.2 body lengths a second at 60 fps, 0.6 at 30. So at 60 neither
        // has a bout (and no run in the file), at 30 both do.
        const longRear = eth.addBehavior({ name: 'long rear', key: 'l' }).behavior;
        eth.setBehaviorDetector(longRear.id, { conditions: [{ feature: 'Nose height', op: '>', value: 0.5 }], minDuration: 3, maxGap: 0 });
        const slow = eth.addBehavior({ name: 'slow', key: 'w' }).behavior;
        eth.setBehaviorDetector(slow.id, { conditions: [{ feature: 'centroid speed', op: '<', value: 1 }], minDuration: 0, maxGap: 0 });
        await det.detectAllFrames();
        (await import('/ui/behavior-coding.js')).codeBehavior(rear.id, { animal: 'id_1', start: 100, end: 110 });
        const views = CAMS.map(n => ({ name: n, videoWidth: 256, videoHeight: 32, frameCount: T }));
        const files = CAMS.map(n => ({ name: n, assignedCamera: n, videoPath: n + '.mp4' }));
        const u = new Uint8Array(await window.SleapIO.saveSlpToBytes(fileio.buildSlpLabelsAllViews(s, views, files)));
        let bin = '';
        for (let i = 0; i < u.length; i += 8192) bin += String.fromCharCode.apply(null, u.subarray(i, i + 8192));
        return { bouts: P.predictedBouts(s).map(b => [b.animal, b.start, b.end]), runFps: P.behaviorRunOf(s, rear.id).fps, slp: btoa(bin) };
    });
    /** Everything measured in seconds, at the rate in effect: Detect, This frame (at frame 100), the settings' columns. */
    const measure = (p) => p.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        const P = await import('/pose/behavior-predictions.js');
        const det = await import('/ui/behavior-detect.js');
        const s = AS.state.session;
        // A short clip sizes the project; with no video the poses do (`totalFrames` can be 0).
        AS.state.currentFrame = AS.state.totalFrames > 0 ? Math.min(100, AS.state.totalFrames - 1) : 100;
        const readout = det.detectThisFrame().rows.filter(r => r.behaviorId === 'b3').map(r => [r.animal, r.conditions[0].actual]).sort();
        const cols = (await det.computeFeatureColumns(det.cachedBehaviorContext(s), ['centroid speed'])).animal['centroid speed'];
        const speed = cols.map(c => { let sum = 0, n = 0; for (const v of c) if (v === v) { sum += v; n++; } return sum / n; });
        await det.detectAllFrames();
        let h = 0x811c9dc5, n = 0;
        P.behaviorRuns(s).slice().sort((a, b) => (a.behaviorId < b.behaviorId ? -1 : 1)).forEach(r => Array.from(r.scores.keys()).sort().forEach(k => {
            const u = new Uint32Array(r.scores.get(k).buffer);
            for (let i = 0; i < u.length; i++) { h = Math.imul(h ^ u[i], 0x01000193); n++; }
        }));
        return { rate: det.detectionFps(), readout, speed, scores: (h >>> 0) + ':' + n, runFps: P.behaviorRuns(s).map(r => r.fps),
                 bouts: P.predictedBouts(s).map(b => [b.behaviorId, b.animal, b.start, b.end]), status: document.getElementById('statusText').textContent };
    });
    const asMade = await measure(maker);
    await maker.close();
    /** What the Behavior UI says in seconds, as the user sees it. Leaves a labelled rear on 200–259. */
    const shows = (p) => p.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        const C = await import('/ui/behavior-coding.js');
        const M = await import('/ui/behavior-mode.js');
        const St = await import('/pose/behavior-states.js');
        const s = AS.state.session;
        const painted = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
        const text = (sel) => (document.querySelector(sel) || {}).textContent || '';
        const tab = document.querySelector('.panel-tab[data-tab="tabBehavior"]');
        if (tab && !tab.classList.contains('active')) tab.click();
        (await import('/ui/behavior-panel.js')).refreshBehaviorPanel();
        await painted();
        const out = { note: text('#behaviorSummaryBody .behavior-fps-note'), predicted: text('.behavior-pred[data-pkey^="b1|"] .behavior-bout-dur') };
        let went = null;
        C.setBehaviorNavigator(f => { went = f; });
        document.querySelector('.behavior-pred[data-pkey^="b1|"] .behavior-bout-main').click();
        out.leadIn = went;
        (await import('/ui/behavior-timeline.js')).onTimelineRangeSelect(100, 159, { animal: 'id_0' });
        out.range = text('.behavior-range-head');
        document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
        M.toggleBehaviorMode();
        AS.state.currentFrame = 200;
        C.codeBehavior('b1', { animal: 'id_0' });
        AS.state.currentFrame = 259;
        C.requestBehaviorRepaint({ ui: true });
        await painted();
        out.mode = text('#behaviorModeHint');
        M.toggleBehaviorMode();
        St.setStateLayer(s, { name: 'kpms', source: 'keypoint-MoSeq', animals: { id_0: [[0, 59, 1], [60, 119, 2]], id_1: [[0, 119, 1]] } });
        C.requestBehaviorRepaint({ states: true });
        await painted();
        out.states = text('#behaviorStatesBody .behavior-fps-note');
        out.stateRun = text('#behaviorStatesBody .behavior-state-row[data-state="2"] .behavior-state-stats');
        document.querySelector('#behaviorStatesBody .behavior-state-row[data-state="2"] button[data-act="next"]').click();
        out.stateStatus = document.getElementById('statusText').textContent;
        St.removeStateLayer(s, 'kpms');
        C.setBehaviorNavigator((await import('/pose/initialization.js')).navigateToFrame);
        return out;
    });
    const behaviorMenu = async (p, id) => {
        await p.click('.menu-item[data-menu="behavior"]');
        await p.click('#' + id);
    };
    check(eq(made.bouts, [['id_0', 500, 620]]) && made.runFps === 60, 'detected at 60 fps: the planted rear');
    check(asMade.rate === 60 && asMade.readout[0][1] > 1.1 && asMade.readout[0][1] < 1.3, 'walking at ' + asMade.readout[0][1].toFixed(2) + ' body lengths a second, at 60 fps');
    check(!/rate on record/.test(asMade.status), 'the status has nothing to add at the rate the fps box holds');

    /** Open the saved file through a real reader, going on without videos at its prompt. */
    const reopen = (p, route) => p.evaluate(async ({ route, slp }) => {
        const skip = setInterval(() => {
            if (document.getElementById('videoSelectOverlay')) document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
        }, 50);
        const file = new File([Uint8Array.from(atob(slp), c => c.charCodeAt(0))], 'sixty.slp');
        if (route === 'eager') await (await import('/import-export/slp-import.js')).handleLoadSlpFile(file);
        else await (await import('/loading/session-loader.js')).handleLoadProjectSlpLazy(file);
        clearInterval(skip);
    }, { route, slp: made.slp });
    /** The real File ▸ Load Videos button, with this clip as both cameras' video. */
    const loadVideos = async (p, clip) => {
        const buffer = fs.readFileSync(clip);
        p.once('filechooser', fc => fc.setFiles(['cam0', 'cam1'].map(nm => ({ name: nm + '.mp4', mimeType: 'video/mp4', buffer }))));
        await p.evaluate(() => document.getElementById('btnLoadVideos').click());
        await p.waitForFunction(() => window.__lucid.state.views.filter(v => v.decoder && v.decoder.samples && v.decoder.samples.length).length === 2 &&
            document.getElementById('loadingOverlay').classList.contains('hidden'), null, { timeout: 60000 });
    };
    /** What the app says now: the stale check, the blue button and the banner, as the user sees them. */
    const look = (p) => p.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        const P = await import('/pose/behavior-predictions.js');
        const det = await import('/ui/behavior-detect.js');
        const s = AS.state.session;
        for (const t0 = performance.now(); P.predictionCheckPending(s) && performance.now() - t0 < 20000;) await new Promise(r => setTimeout(r, 50));
        const tab = document.querySelector('.panel-tab[data-tab="tabBehavior"]');
        if (tab && !tab.classList.contains('active')) tab.click();
        (await import('/ui/rendering.js')).drawAllOverlays(AS.state.currentFrame);
        (await import('/ui/behavior-panel.js')).refreshBehaviorPanel();
        const env = det.detectionEnv();
        const banner = document.querySelector('.behavior-stale');
        return { fps: AS.state.fps, rate: det.detectionFps(), shown: document.getElementById('fpsDisplay').textContent, views: AS.state.views.length, lazy: !!s.lazyLoader,
                 note: (document.querySelector('#behaviorSummaryBody .behavior-fps-note') || {}).textContent, fileRate: (await import('/ui/behavior-io-modal.js')).behaviorSessionInfo(s).fps,
                 runFps: P.behaviorRunOf(s, 'b1').fps, bouts: P.predictedBouts(s).length, input: P.predictionStaleness(s, env).input,
                 settings: P.predictionStaleness(s, env).settings, stands: P.predictionsStand(s, env),
                 blue: document.getElementById('tbBehavior').classList.contains('active'), banner: banner ? banner.textContent : '' };
    });
    const isFresh = (v) => !v.settings.length && !v.input && v.stands && v.blue && !v.banner;
    const isStale = (v) => eq(v.settings, ['b1']) && !v.stands && !v.blue && /settings of rear changed/.test(v.banner);
    const change = (p, what) => p.evaluate(async (what) => {
        const eth = await import('/pose/behavior-ethogram.js');
        if (what === 'rule') eth.setBehaviorDetector('b1', { conditions: [{ feature: 'Nose height', op: '>', value: 0.6 }], minDuration: 0.1, maxGap: 0 });
        if (what === 'rule back') eth.setBehaviorDetector('b1', { conditions: [{ feature: 'Nose height', op: '>', value: 0.5 }], minDuration: 0.1, maxGap: 0 });
        if (what === 'body') eth.setFeatureSettings({ nodeRoles: { head: 'Head' } });
        if (what === 'floor') eth.setFeatureSettings({ floor: { source: 'plane', plane: 99 } });
        if (what === 'settings back') eth.setFeatureSettings(null);
    }, what);

    for (const route of ['eager', 'lazy']) {
        const p = await openApp();
        await reopen(p, route);
        const v0 = await look(p);
        check(v0.lazy === (route === 'lazy') && v0.views === 0 && v0.bouts === 1 && v0.runFps === 60, route + ' reopen, no video: the run and its 60 fps came back');
        check(v0.fps === 30 && /^30\.0 fps/.test(v0.shown), route + ': state.fps stays the default 30, and the fps box says so');
        check(isFresh(v0), route + ': not called out of date: blue, no banner ' + JSON.stringify(v0.settings) + ' ' + v0.banner);
        for (const [what, back] of [['rule', 'rule back'], ['body', 'settings back'], ['floor', 'settings back']]) {
            await change(p, what);
            const v = await look(p);
            check(isStale(v), route + ': a changed ' + what + ' still reads stale');
            await change(p, back);
            check(isFresh(await look(p)), route + ': and put back, fresh');
        }
        await loadVideos(p, VIDEO_60);
        const v1 = await look(p);
        check(Math.abs(v1.fps - 60) < 0.01, route + ': the 60 fps video sets state.fps to 60 (' + v1.fps + ')');
        check(isFresh(v1), route + ': and the run stays fresh');
        await p.close();

        // With no video, everything measured in seconds uses the rate on record.
        const d = await openApp();
        await reopen(d, route);
        check((await look(d)).rate === 60, route + ': measured at the 60 on record, not the 30 in state.fps');
        await d.click('[data-menu="behavior"]');
        await d.click('#menuBehaviorRules');
        await d.waitForFunction(() => ['b1', 'b2', 'b3'].every(id => /^[\d,]+ bouts?/.test((document.querySelector('.bd-card[data-id="' + id + '"] .bd-live') || {}).textContent || '')),
            null, { timeout: 20000 });
        const live = await d.evaluate(() => ['b1', 'b2', 'b3'].map(id => document.querySelector('.bd-card[data-id="' + id + '"] .bd-live').textContent));
        check(/^1 bout · 2\.0 s in all/.test(live[0]) && /^0 bouts/.test(live[1]) && /^0 bouts/.test(live[2]),
            route + ': the settings dialog counts at 60 fps: "' + live[0].slice(0, 22) + '", and 0 for the two the rate decides');
        await d.keyboard.press('Escape');
        const seen = await shows(d);
        check(/^Seconds at 60 fps/.test(seen.note) && seen.predicted === '2.0 s', route + ': the Behavior tab counts seconds at 60: the 121-frame bout is ' + seen.predicted + ', "' + seen.note + '"');
        check(seen.leadIn === 440, route + ': opening that bout leads in by a second of video: frame ' + seen.leadIn + ' for a bout at 500');
        check(/60 frames \(1\.0 s\)$/.test(seen.range), route + ': the timeline\'s menu: "' + seen.range + '"');
        check(/● rear · id_0 1\.0 s/.test(seen.mode), route + ': Label Behaviors mode: "' + seen.mode.split('   ·   ').pop() + '"');
        check(/4\.0 s with a syllable, at 60 fps$/.test(seen.states) && / · 1\.00 s$/.test(seen.stateRun) && /\(1\.00 s\)/.test(seen.stateStatus),
            route + ': States: "' + seen.states.split(' · ').pop() + '", a 60-frame run is ' + seen.stateRun.split(' · ').pop());

        // Files. The label on id_1 is frames 100–110, and the seconds are frames / 60.
        await behaviorMenu(d, 'menuBehaviorExport');
        await d.waitForSelector('.behavior-export-modal');
        const exportSays = await d.textContent('.behavior-export-modal .behavior-ed-lead');
        await d.check('input[name="bxTable"][value="bouts"]');
        await d.selectOption('#bxSep', ',');
        const download = d.waitForEvent('download');
        await d.click('#bxGo');
        const exported = fs.readFileSync(await (await download).path(), 'utf8').split('\r\n').find(l => /,id_1,/.test(l)) || '';
        check(/^Seconds at 60 fps/.test(exportSays) && /,100,110,11,1\.667,1\.850,0\.183,.*,60$/.test(exported),
            route + ': an export says 60 fps, and writes its seconds at that: ' + exported.split(',').slice(-9).join(','));
        // A file in seconds, with no frame rate of its own: its frames depend on the rate.
        const chooser = d.waitForEvent('filechooser');
        await behaviorMenu(d, 'menuBehaviorImport');
        await (await chooser).setFiles({ name: 'seconds.csv', mimeType: 'text/csv',
            buffer: Buffer.from('Observation id,Subject,Behavior,Behavior type,Start (s),Stop (s)\r\nSixty,id_0,rear,STATE,1.000,2.000\r\n') });
        await d.waitForSelector('.behavior-import-modal');
        const importSays = await d.textContent('#biFps');
        await d.click('#biGo');
        const imported = await d.evaluate(async () => {
            const L = await import('/pose/behavior-labels.js');
            const s = (await import('/ui/app-state.js')).state.session;
            return { rows: L.exportBoutRows(s).filter(r => r[6] === 'i').map(r => [r[1], r[3], r[4]]), labelsFps: L.behaviorFps(s) };
        });
        check(/^Seconds become frames at 60 fps/.test(importSays) && eq(imported.rows, [['id_0', 60, 119]]),
            route + ': an import turns 1–2 s into frames ' + JSON.stringify(imported.rows) + ' at 60 fps: "' + importSays + '"');
        check(imported.labelsFps === 60, route + ': and the labels record 60');
        const m = await measure(d);
        check(m.readout.length === 2 && eq(m.readout, asMade.readout), route + ': This frame reads what it read at 60 fps');
        check(eq(m.speed, asMade.speed), route + ': so do the settings\' feature columns');
        check(eq(m.bouts, [['b1', 'id_0', 500, 620]]) && m.runFps.every(f => f === 60), route + ': Detect gives the saved bouts, and each run records 60');
        check(m.scores === asMade.scores, route + ': and the same per-frame scores, bit for bit (' + m.scores + ')');
        check(/no video: at 60 fps, the rate on record/.test(m.status), route + ': the status says which rate: "' + m.status.slice(0, 80) + '"');
        const v4 = await look(d);
        check(v4.fps === 30 && /^30\.0 fps/.test(v4.shown), route + ': state.fps and the fps box stay at 30 (playback)');
        check(isFresh(v4), route + ': the new run is fresh');
        await loadVideos(d, VIDEO_60);
        check(isFresh(await look(d)), route + ': and stays fresh when the 60 fps video loads');
        await d.close();

        const q = await openApp();
        await reopen(q, route);
        check(isFresh(await look(q)), route + ': fresh again, reopened');
        await loadVideos(q, VIDEO_10);
        const v2 = await look(q);
        check(Math.abs(v2.fps - 10) < 0.01, route + ': a 10 fps video sets state.fps to 10 (' + v2.fps + ')');
        check(isStale(v2), route + ': a real difference in frame rate: stale, and the banner says so');
        // The next project, with no video, is not judged by this one's rate (a lazy open keeps `state.fps`).
        await reopen(q, route);
        const v3 = await look(q);
        check(v3.views === 0 && isFresh(v3), route + ': that file opened again in the same page, no video: fresh at state.fps ' + v3.fps);
        await q.close();
    }

    // The fps box is how a project without video says its rate.
    const typeFps = async (p, value) => {
        await p.dblclick('#fpsDisplay');
        await p.fill('#fpsDisplay input', String(value));
        await p.press('#fpsDisplay input', 'Enter');
    };
    const box = await openApp();
    await reopen(box, 'eager');
    await typeFps(box, 25);
    const typed = await look(box);
    check(typed.fps === 25 && isStale(typed), 'no video, 25 typed in the fps box: stale');
    check(typed.note === 'Seconds at 25 fps.' && typed.fileRate === 25, 'and the Behavior tab and files go by the 25 typed, not the 60 on record');
    await typeFps(box, 60);
    const typed60 = await look(box);
    check(typed60.fps === 60 && isFresh(typed60), '60 typed: fresh');
    await typeFps(box, 30);
    const typed30 = await look(box);
    check(typed30.fps === 30 && isStale(typed30), '30 typed is said, not a default: stale');
    const atTyped = await measure(box);
    check(atTyped.rate === 30 && atTyped.runFps.every(f => f === 30) && atTyped.bouts.some(b => b[0] === 'b2'),
        'and Detect measures at the 30 typed, not the 60 on record: the 121 frames are 4 s now, so "long rear" has its bout');
    await box.close();

    // So does a campy recording's manifest, even at 30.
    const camp = await openApp();
    await reopen(camp, 'eager');
    await camp.evaluate(() => { window.__lucid.state.session.campyRecording = { frameRate: 30 }; });
    const atManifest = await look(camp);
    check(atManifest.rate === 30 && isStale(atManifest), 'no video, a campy manifest saying 30 fps: measured at 30, and stale');
    await camp.close();

    // A real rate outranks what is on record.
    const vid = await openApp();
    await reopen(vid, 'eager');
    await loadVideos(vid, VIDEO_10);
    const lookAt10 = await look(vid);
    check(lookAt10.note === 'Seconds at 10 fps.' && lookAt10.fileRate === 10, 'a 10 fps video: the Behavior tab and files go by it, not the 60 on record');
    const at10 = await measure(vid);
    check(at10.rate === 10 && at10.runFps.every(f => f === 10), 'a 10 fps video: Detect measures at 10, and each run records it');
    await vid.close();

    // Sessions in the background go by the same rule, and a label edit keeps the labels' rate.
    const multi = await openApp();
    await reopen(multi, 'eager');
    const others = await multi.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        const P = await import('/pose/behavior-predictions.js');
        const L = await import('/pose/behavior-labels.js');
        const det = await import('/ui/behavior-detect.js');
        const pd = await import('/pose/pose-data.js');
        const s = AS.state.session;
        const other = new pd.Session(s.cameras, s.skeleton, [], 'Other');
        other.identities = s.identities; other.instanceGroups = s.instanceGroups; other.totalFrames = AS.state.totalFrames;
        P.setBehaviorRun(other, { behaviorId: 'b1', cfg: 'x', fps: 25, mode: '3d', bouts: [{ animal: 'id_0', target: '', start: 1, end: 2, score: 0.9 }] });
        AS.state.sessions = [s, other];
        const out = { onRecord: det.sessionFps(other) };
        other.fps = 30;
        out.leftAtDefault = det.sessionFps(other);
        other.typedFps = 30;
        out.typed = det.sessionFps(other);
        other.typedFps = 0; other.campyRecording = { frameRate: 50 };
        out.manifest = det.sessionFps(other);
        other.campyRecording = null; other.fps = 40; AS.state.videoFiles = [{ name: 'v', sessionIdx: 1 }];
        out.video = det.sessionFps(other);
        AS.state.videoFiles = []; other.fps = 30;
        out.nothing = det.sessionFps(new pd.Session(s.cameras, s.skeleton, [], 'Empty'));
        out.exportRate = (await import('/ui/behavior-io-modal.js')).behaviorSessionInfo(other).fps;
        await det.detectAllSessions();
        out.runs = [P.behaviorRunOf(s, 'b1').fps, P.behaviorRunOf(other, 'b1').fps];
        out.status = document.getElementById('statusText').textContent;
        AS.state.sessions = [s];
        P.clearBehaviorRuns(s);
        out.labels = L.behaviorFps(s);
        out.fromLabels = det.detectionFps();
        (await import('/ui/behavior-coding.js')).codeBehavior('b1', { animal: 'id_0', start: 10, end: 20 });
        out.labelsAfter = L.behaviorFps(s);
        out.fps = AS.state.fps;
        return out;
    });
    await multi.close();
    check(others.onRecord === 25, 'another session on record at 25 fps, with no video: 25');
    check(others.leftAtDefault === 25, 'left at the default 30, still 25');
    check(others.typed === 30, 'unless 30 was typed for it');
    check(others.manifest === 50 && others.video === 40, 'or its manifest (50) or its video (40) said otherwise');
    check(others.nothing === 60, 'a session with nothing on record takes the app\'s rate');
    check(others.exportRate === 25, 'exporting all sessions writes that session\'s seconds at its own 25');
    check(eq(others.runs, [60, 25]) && /in 2 sessions · no video: at 60 fps, the rate on record/.test(others.status),
        'Detect on all sessions measures each at its own rate (' + others.runs.join(', ') + '), and says so');
    check(others.labels === 60 && others.fromLabels === 60, 'with no runs left, the labels\' 60 from the file is the rate on record');
    check(others.labelsAfter === 60 && others.fps === 30, 'a label edit with no video keeps it (state.fps stays 30)');

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
