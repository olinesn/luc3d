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
 *     campy manifest. The 60 fps video keeps them fresh.
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
        await det.detectAllFrames();
        const views = CAMS.map(n => ({ name: n, videoWidth: 256, videoHeight: 32, frameCount: T }));
        const files = CAMS.map(n => ({ name: n, assignedCamera: n, videoPath: n + '.mp4' }));
        const u = new Uint8Array(await window.SleapIO.saveSlpToBytes(fileio.buildSlpLabelsAllViews(s, views, files)));
        let bin = '';
        for (let i = 0; i < u.length; i += 8192) bin += String.fromCharCode.apply(null, u.subarray(i, i + 8192));
        return { bouts: P.predictedBouts(s).map(b => [b.animal, b.start, b.end]), runFps: P.behaviorRunOf(s, rear.id).fps, slp: btoa(bin) };
    });
    await maker.close();
    check(eq(made.bouts, [['id_0', 500, 620]]) && made.runFps === 60, 'detected at 60 fps: the planted rear');

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
        return { fps: AS.state.fps, shown: document.getElementById('fpsDisplay').textContent, views: AS.state.views.length, lazy: !!s.lazyLoader,
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
    await typeFps(box, 60);
    const typed60 = await look(box);
    check(typed60.fps === 60 && isFresh(typed60), '60 typed: fresh');
    await typeFps(box, 30);
    const typed30 = await look(box);
    check(typed30.fps === 30 && isStale(typed30), '30 typed is said, not a default: stale');
    await box.close();

    // So does a campy recording's manifest, even at 30.
    const camp = await openApp();
    await reopen(camp, 'eager');
    await camp.evaluate(() => { window.__lucid.state.session.campyRecording = { frameRate: 30 }; });
    check(isStale(await look(camp)), 'no video, a campy manifest saying 30 fps: stale');
    await camp.close();

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
