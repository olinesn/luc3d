/**
 * behavior-label-roundtrip.mjs — behavior labels survive save -> reopen, in
 * the real app, through BOTH `.slp` writers.
 *
 * A `.slp` carries bouts and predicted bouts as sleap-io events
 * (`import-export/behavior-events.js`) and what only LUC3D reads under
 * `metadata.lucid` (`import-export/behavior-metadata.js`). The Node tests
 * `tests/test-behavior-events.mjs` and `tests/test-behavior-metadata.mjs` pin the
 * payloads; this one pins the seam: that both writers call them, and that the file
 * reads back through the real `parseSlpViaSleapIO` adapter the importer uses.
 *
 *  1. Eager writer (`buildSlpLabelsAllViews` + `saveSlpToBytes`): the ethogram
 *     and every bout — present, "not", social with a target, a note, an
 *     accepted source — come back exactly, and a state still running at save
 *     time is written as ending at the playhead.
 *  2. Streaming writer (`saveAllSessionsStreaming`, two lazy sessions): each
 *     session reopens with its own bouts and the same ethogram.
 *  3. A project with no ethogram and no bouts writes neither key, so untouched
 *     projects keep their bytes (`tests/e2e/save-golden-digest.mjs`).
 *  4. Files other tools wrote (`tests/fixtures/behavior-events/`, from sleap-app and
 *     Python sleap-io; `behavior-cross-app` makes the first) read in, and saved again
 *     lose nothing.
 *  Both writers also carry the detection settings and predictions (Phase 2):
 *  a behavior's detector, the changed feature settings, and each session's
 *  predicted bouts with their scores; and each session's behavioral states
 *  (Phase 4: a keypoint-MoSeq layer's runs and syllable names).
 *
 * Run: node tests/e2e/behavior-label-roundtrip.mjs   (spawns its own http.server; PORT= to move it)
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8193);

let fails = 0;
const check = (c, m) => { console.log((c ? '  ✓ ' : '  ✗ ') + m); if (!c) fails++; };
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1200));

let browser;
try {
    browser = await chromium.launch();
    const page = await browser.newPage();
    const errs = [];
    page.on('pageerror', e => errs.push('pageerror: ' + String(e).slice(0, 300)));
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state, { timeout: 20000 });

    const rt = await page.evaluate(async () => {
        const pd = await import('/pose/pose-data.js');
        const AS = await import('/ui/app-state.js');
        const fileio = await import('/import-export/file-io.js');
        const saveLoad = await import('/import-export/save-load.js');
        const lazyMod = await import('/loading/sio-lazy-loader.js');
        const events = await import('/import-export/behavior-events.js');
        const meta = await import('/import-export/behavior-metadata.js');
        const eth = await import('/pose/behavior-ethogram.js');
        const L = await import('/pose/behavior-labels.js');
        const P = await import('/pose/behavior-predictions.js');
        const S = await import('/pose/behavior-states.js');
        const { Skeleton, Camera, Instance, InstanceGroup, FrameGroup, Session } = pd;
        const SioLazyLoader = lazyMod.SioLazyLoader || lazyMod.default;
        const NODES = 4;
        const skelOf = () => new Skeleton('skeleton', Array.from({ length: NODES }, (_, i) => 'n' + i),
            Array.from({ length: NODES - 1 }, (_, i) => [i, i + 1]));
        const SPECS = [
            { name: 'sessA', cams: ['a_cam0', 'a_cam1'], nFrames: 60 },
            { name: 'sessB', cams: ['b_cam0', 'b_cam1'], nFrames: 40 },
        ];
        function buildSession(spec) {
            const mtx = [[1000, 0, 128], [0, 1000, 128], [0, 0, 1]];
            const cams = spec.cams.map((cn, ci) => new Camera(cn, mtx, [0, 0, 0, 0, 0], [0, 0.1 * ci, 0], [10 * ci, 0, 0], [256, 256]));
            const session = new Session(cams, skelOf(), ['t0', 't1'], spec.name);
            const ids = ['mouseA', 'mouseB'].map(n => session.addIdentity(n).id);
            for (let f = 0; f < spec.nFrames; f++) {
                const fg = new FrameGroup(f); session.addFrameGroup(fg);
                session.instanceGroups.set(f, ids.map((id, k) => {
                    const g = new InstanceGroup(f * 10 + k + 1, id);
                    for (const cn of spec.cams) {
                        const pts = Array.from({ length: NODES }, (_, n) => [50 + n + f + 40 * k, 60 + n + f]);
                        const inst = new Instance(pts, k, 'predicted', 0.9);
                        g.addInstance(cn, inst); fg.addInstance(cn, inst);
                    }
                    const p = new Float64Array(NODES * 3);
                    for (let n = 0; n < NODES; n++) { p[3 * n] = n + f * 0.1 + 50 * k; p[3 * n + 1] = n * 2; p[3 * n + 2] = 1 + n; }
                    g.points3d = p;
                    return g;
                }));
            }
            return session;
        }
        function viewsFor(spec) { return spec.cams.map(cn => ({ name: cn, videoWidth: 256, videoHeight: 256, frameCount: spec.nFrames })); }
        function videoFilesFor(spec) { return spec.cams.map(cn => ({ name: cn, assignedCamera: cn, videoPath: cn + '.mp4' })); }

        function defineEthogram() {
            meta.resetBehaviorState();
            const rear = eth.addBehavior({ name: 'rear', key: 'r', description: 'both forepaws off the floor' }).behavior;
            const groom = eth.addBehavior({ name: 'groom', key: 'g', group: 'posture' }).behavior;
            const chase = eth.addBehavior({ name: 'chase', key: 'c', social: true }).behavior;
            const bite = eth.addBehavior({ name: 'bite', key: 'b', kind: 'point', social: true }).behavior;
            eth.setBehaviorDetector(rear.id, { conditions: [{ feature: 'n0 height · mean ±0.1 s', op: '>', value: 0.5 }], minDuration: 0.2 });
            eth.setFeatureSettings({ nodeRoles: { head: 'n0', tailBase: 'n3' } });
            return { rear, groom, chase, bite };
        }
        function labelA(s, b) {
            L.labelRange(s, b.rear.id, 'mouseA', '', 5, 20, true, { note: 'tall' });
            L.markReviewed(s, [b.rear.id], ['mouseA'], ['mouseA', 'mouseB'], 0, 40);
            L.labelRange(s, b.groom.id, 'mouseB', '', 10, 30, true, { source: 'a' });
            L.labelRange(s, b.chase.id, 'mouseA', 'mouseB', 41, 50, true);
            L.labelRange(s, b.bite.id, 'mouseB', 'mouseA', 33, 33, true);
            L.noteBehaviorFps(s, 50);
            L.toggleState(s, b.groom.id, 'mouseA', '', 52);          // still running at save time
            L.setBehaviorCursor(s, 57);
            S.setStateLayer(s, { name: 'keypoint-MoSeq', source: 'keypoint-MoSeq', stateNames: { 2: 'groom' },
                animals: { mouseA: [[0, 9, 2], [10, 59, 0]], mouseB: [[5, 30, 1]] } });
        }
        function labelB(s, b) {
            L.labelRange(s, b.groom.id, 'mouseA', '', 0, 9, true);
            P.setBehaviorRun(s, { behaviorId: b.chase.id, cfg: 'c', fps: 30, mode: '3d',
                bouts: [{ animal: 'mouseB', target: 'mouseA', start: 3, end: 8, score: 0.66 }] });
            S.setStateLayer(s, { name: 'kpms B', source: 'keypoint-MoSeq', animals: { mouseA: [[0, 39, 7]] } });
        }
        function predictA(s, b) {
            P.setBehaviorRun(s, { behaviorId: b.rear.id, cfg: 'r', fps: 30, mode: '3d',
                bouts: [{ animal: 'mouseA', target: '', start: 5, end: 9, score: 0.9 }, { animal: 'mouseB', target: '', start: 30, end: 33, score: 0.51 }] });
        }
        const predRows = (s) => P.predictedBouts(s).map(p => [p.behaviorId, p.animal, p.target, p.start, p.end, p.score]);
        async function reopen(bytes, filename) {
            const slpData = await fileio.parseSlpViaSleapIO(new File([bytes], filename), () => {});
            meta.resetBehaviorState();
            const out = [];
            for (const sd of (slpData.sessions || [])) {
                const lucid = (sd.metadata && sd.metadata.lucid) || {};
                const fresh = new Session([], skelOf(), [], 'reopened');
                meta.readBehaviorMetadata(fresh, lucid, slpData.behaviorEvents, events.sessionVideoIdxs(sd));
                out.push({
                    sessionName: lucid.sessionName,
                    hasEthogramKey: Object.prototype.hasOwnProperty.call(lucid, 'ethogram'),
                    hasLabelsKey: Object.prototype.hasOwnProperty.call(lucid, 'behaviorLabels'),
                    fps: lucid.behaviorLabels ? lucid.behaviorLabels.fps : null,
                    boutsInLucid: !!(lucid.behaviorLabels && lucid.behaviorLabels.bouts),
                    predictedBoutsInLucid: !!(lucid.behaviorPredictions && lucid.behaviorPredictions.runs.some(r => r.bouts)),
                    rows: L.exportBoutRows(fresh),
                    predictions: predRows(fresh),
                    states: S.exportStateLayers(fresh),
                    hasStatesKey: Object.prototype.hasOwnProperty.call(lucid, 'behaviorStates'),
                    otherKeys: ['sessionName', 'identities', 'skeleton', 'tracks'].every(k => Object.prototype.hasOwnProperty.call(lucid, k)),
                });
            }
            return { sessions: out, ethogram: eth.serializeEthogram(), features: eth.featureSettings(), fileEvents: slpData.behaviorEvents };
        }

        const res = {};
        // ---- 1. eager writer
        {
            const b = defineEthogram();
            const s = buildSession(SPECS[0]);
            labelA(s, b);
            predictA(s, b);
            res.expectedA = L.exportBoutRows(s);
            res.expectedPredA = predRows(s);
            res.expectedStatesA = S.exportStateLayers(s);
            res.expectedEthogram = eth.serializeEthogram();
            res.stillRunning = L.isRunning(s, b.groom.id, 'mouseA', '');
            AS.state.sessions = [s]; AS.state.session = s; AS.state.activeSessionIdx = 0;
            AS.state.views = viewsFor(SPECS[0]); AS.state.videoFiles = videoFilesFor(SPECS[0]);
            const labels = fileio.buildSlpLabelsAllViews(s, viewsFor(SPECS[0]), videoFilesFor(SPECS[0]));
            res.eager = await reopen(await window.SleapIO.saveSlpToBytes(labels), 'eager.slp');
        }
        // ---- 2. streaming writer over two lazy sessions
        async function perCamFixture(spec, camName, camIdx) {
            const mtx = [[1000, 0, 128], [0, 1000, 128], [0, 0, 1]];
            const cam = new Camera(camName, mtx, [0, 0, 0, 0, 0], [0, 0.1 * camIdx, 0], [10 * camIdx, 0, 0], [256, 256]);
            const s = new Session([cam], skelOf(), ['t0', 't1'], spec.name);
            for (let f = 0; f < spec.nFrames; f++) {
                const fg = new FrameGroup(f); s.addFrameGroup(fg);
                for (let k = 0; k < 2; k++) {
                    const pts = Array.from({ length: NODES }, (_, n) => [50 + n + f + 40 * k, 60 + n + f]);
                    fg.addInstance(camName, new Instance(pts, k, 'predicted', 0.9));
                }
            }
            const labels = fileio.buildSlpLabels(s, camName, false, { videoWidth: 256, videoHeight: 256, frameCount: spec.nFrames, videoPath: camName + '.mp4' });
            return new File([await window.SleapIO.saveSlpToBytes(labels)], camName + '.slp');
        }
        {
            const b = defineEthogram();
            const sessions = SPECS.map(buildSession);
            labelA(sessions[0], b);
            predictA(sessions[0], b);
            labelB(sessions[1], b);
            res.expectedStream = sessions.map(s => L.exportBoutRows(s));
            res.expectedStreamPred = sessions.map(predRows);
            res.expectedStreamStates = sessions.map(s => S.exportStateLayers(s));
            for (let i = 0; i < sessions.length; i++) {
                const loader = new SioLazyLoader();
                for (let ci = 0; ci < SPECS[i].cams.length; ci++) await loader.open(SPECS[i].cams[ci], await perCamFixture(SPECS[i], SPECS[i].cams[ci], ci));
                sessions[i].lazyLoader = loader;
            }
            AS.state.sessions = sessions; AS.state.session = sessions[0]; AS.state.activeSessionIdx = 0;
            AS.state.views = SPECS.flatMap(viewsFor); AS.state.videoFiles = SPECS.flatMap(videoFilesFor);
            try { res.streaming = await reopen(await saveLoad.saveAllSessionsStreaming(sessions), 'streaming.slp'); }
            catch (e) { res.streamErr = String((e && e.stack) || e); }
        }
        // ---- 3. untouched project
        {
            meta.resetBehaviorState();
            const s = buildSession(SPECS[0]);
            AS.state.sessions = [s]; AS.state.session = s; AS.state.activeSessionIdx = 0;
            const labels = fileio.buildSlpLabelsAllViews(s, viewsFor(SPECS[0]), videoFilesFor(SPECS[0]));
            res.untouched = await reopen(await window.SleapIO.saveSlpToBytes(labels), 'untouched.slp');
        }
        // ---- 4. files other tools wrote, through the real adapter
        async function fromOtherTool(fixture) {
            const bytes = await (await fetch('/tests/fixtures/behavior-events/' + fixture)).arrayBuffer();
            const slpData = await fileio.parseSlpViaSleapIO(new File([bytes], fixture), () => {});
            meta.resetBehaviorState();
            const s = new Session([], skelOf(), [], 'other tool');
            const read = meta.readBehaviorMetadata(s, (slpData.sessions[0] && slpData.sessions[0].metadata || {}).lucid || null,
                slpData.behaviorEvents, null);
            const out = { read, behaviors: eth.behaviors().map(b => ({ name: b.name, key: b.key, social: b.social, kind: b.kind, color: b.color })),
                          rows: L.exportBoutRows(s).map(r => [eth.behaviorById(r[0]).name].concat(r.slice(1))),
                          predicted: P.predictedBouts(s).map(p => [eth.behaviorById(p.behaviorId).name, p.animal, p.start, p.end, p.score]) };
            // And saved again, nothing of theirs is lost.
            const video = new window.SleapIO.Video({ filename: 'again.mp4', openBackend: false });
            const again = events.buildBehaviorEvents(window.SleapIO, [{ session: s, video, identities: [], tracks: [] }]);
            out.saved = again.events.map(e => [e.type.name, e.startFrame, e.endFrame, e.subject ? e.subject.name : '', e.target ? e.target.name : '',
                                               e.isPredicted, e.source, e.isPredicted && e.scores ? e.scores.length : 0]);
            return out;
        }
        res.fromApp = await fromOtherTool('sleap_app_behavior.slp');
        res.fromPython = await fromOtherTool('py_written_26_events.slp');
        return res;
    });

    console.log('\n-- 4. files other tools wrote --');
    const fa = rt.fromApp;
    check(fa.behaviors.map(b => b.name).join() === 'rear,chase' && fa.behaviors[0].key === 'r' && fa.behaviors[1].social, 'sleap-app: its ethogram');
    check(eq(fa.rows.map(r => r.slice(0, 7)), [
        ['rear', 'mouse1', '', 10, 19, 1, 'm'],
        ['chase', 'mouse2', 'mouse1', 30, 39, 1, 'm'],
        ['rear', 'mouse2', '', 50, 59, 0, 'm'],
    ]), 'sleap-app: its bouts on tracks, a "not" as a "not"');
    const fp = rt.fromPython;
    check(fp.behaviors.map(b => b.name).join() === 'attack,rear,never used', 'Python: the catalog, an unused type too');
    check(fp.behaviors[0].social && fp.behaviors[0].color === '#e6194b', 'Python: a target makes it social; its color comes along');
    check(fp.rows.some(r => r[0] === 'attack' && r[1] === 'mouse1' && r[2] === 'mouse2' && r[3] === 100 && r[4] === 140 && r[6] === 'm'), 'Python: a bout between tracks');
    check(fp.rows.some(r => r[0] === 'rear' && r[1] === 'animal_A' && r[3] === 50), 'Python: a bout on an identity');
    check(fp.predicted.length === 0, 'Python: its predictions have no target or no animal, so they are not offered for review');
    check(fp.saved.some(e => e[0] === 'attack' && e[5] && e[6] === 'classifier' && e[7] === 5), 'saved again: they are written back, with where they came from and their framewise scores');
    check(fp.saved.some(e => e[0] === 'rear' && e[1] === 300 && e[2] === 301 && e[3] === ''), 'saved again: an event with no animal is written back');
    check(fp.saved.some(e => e[0] === 'rear' && e[1] === 7 && e[3] === '' && e[5]), 'and a prediction with no animal');

    console.log('\n-- 1. eager writer --');
    const e0 = rt.eager.sessions[0];
    check(rt.eager.sessions.length === 1 && e0.hasEthogramKey && e0.hasLabelsKey, 'one session, with what only LUC3D reads');
    const fe = rt.eager.fileEvents;
    check(!e0.boutsInLucid && !e0.predictedBoutsInLucid, 'the bouts are not in metadata.lucid');
    check(!!fe && fe.types.map(t => t.name).join() === 'rear,groom,chase,bite,not rear', `the file has the shared event catalog (${fe && fe.types.map(t => t.name).join()})`);
    check(!!fe && fe.rows.length === rt.expectedA.length + rt.expectedPredA.length && fe.rows.every(r => r.video === 0),
        'every bout is one event, on the session\'s first camera');
    check(!!fe && fe.rows.some(r => r.type === 'not rear' && !r.predicted) && fe.rows.some(r => r.type === 'rear' && r.predicted && r.source === 'rule' && r.score === 0.9),
        'a "not" label has its own type; a prediction its source and score');
    check(eq(rt.eager.ethogram, rt.expectedEthogram), 'the ethogram reads back exactly (ids, kinds, social, group, description)');
    check(eq(e0.rows, rt.expectedA), `every bout reads back exactly (${e0.rows.length} rows)`);
    check(rt.stillRunning && e0.rows.some(r => r[1] === 'mouseA' && r[3] === 52 && r[4] === 57),
        'a state still running at save time is saved as ending at the playhead');
    check(e0.fps === 50, `the coding fps is recorded (got ${e0.fps})`);
    check(e0.rows.some(r => r[5] === 0) && e0.rows.some(r => r[2] === 'mouseB') && e0.rows.some(r => r[6] === 'a') && e0.rows.some(r => r[7] === 'tall'),
        '"not" labels, targets, sources and notes are all in the file');
    check(e0.otherKeys, 'the other metadata.lucid keys are untouched');
    check(eq(e0.predictions, rt.expectedPredA) && e0.predictions.length === 2, 'the predicted bouts and their scores read back (eager)');
    check(rt.eager.ethogram.behaviors[0].detector && eq(rt.eager.features, { nodeRoles: { head: 'n0', tailBase: 'n3' } }),
        'and so do the detector and the changed feature settings');
    check(e0.hasStatesKey && eq(e0.states, rt.expectedStatesA) && rt.expectedStatesA.layers[0].stateNames[2] === 'groom',
        'and the behavioral states: every run and syllable name (eager)');

    console.log('\n-- 2. streaming writer --');
    if (rt.streamErr) {
        check(false, 'saveAllSessionsStreaming threw: ' + rt.streamErr.slice(0, 400));
    } else {
        check(rt.streaming.sessions.length === 2, `both sessions in the file (got ${rt.streaming.sessions.length})`);
        check(rt.streaming.sessions.every(s => s.hasEthogramKey), 'the settings only LUC3D reads are in every session');
        check(!!rt.streaming.fileEvents && rt.streaming.fileEvents.types.some(t => t.name === 'rear'), 'the file has the shared event catalog');
        check(eq(rt.streaming.sessions.map(s => s.rows), rt.expectedStream), 'each session reopens with its own bouts');
        check(eq(rt.streaming.sessions.map(s => s.predictions), rt.expectedStreamPred) && rt.expectedStreamPred[1].length === 1,
            'and its own predicted bouts (streaming)');
        check(eq(rt.streaming.sessions.map(s => s.states), rt.expectedStreamStates) && rt.expectedStreamStates[1].layers[0].name === 'kpms B',
            'and its own behavioral states (streaming)');
    }

    console.log('\n-- 3. untouched project --');
    const u = rt.untouched.sessions[0];
    check(!!u && !u.hasEthogramKey && !u.hasLabelsKey && !u.hasStatesKey, 'no ethogram, bouts or states: none of their keys is written');
    check(rt.untouched.fileEvents === null, 'and no events in the file');
    check(!!u && u.otherKeys, 'and the file still carries the other metadata.lucid keys');

    check(errs.length === 0, `no page errors (${errs.join(' | ').slice(0, 300)})`);
    console.log(fails === 0 ? '\nPASS' : `\nFAIL — ${fails} check(s)`);
    await browser.close();
    browser = null;
    process.exitCode = fails === 0 ? 0 : 1;
} finally {
    if (browser) await browser.close().catch(() => {});
    server.kill();
}
