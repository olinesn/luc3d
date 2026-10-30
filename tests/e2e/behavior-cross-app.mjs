/**
 * behavior-cross-app.mjs — behavior labels travel between LUC3D and sleap-app through
 * the `.slp`, both ways, in the two real apps, with Python sleap-io reading the result.
 *
 * Both apps store labels as sleap-io events (SLP 2.6): `import-export/behavior-events.js`
 * here, `src/lib/behaviorEvents.ts` in sleap-app, the same model and layout as
 * `sleap_io/io/slp.py`. Nothing but the file passes between the apps.
 *
 *  A. A multi-camera LUC3D project: labelled in LUC3D (a state, a "not", a social one,
 *     a mutual one, a point, a note, a detector, predicted bouts, keypoint-MoSeq states)
 *     -> saved -> opened in sleap-app, where the panel shows it on BOTH cameras -> there,
 *     through the real UI: a prediction accepted, a behavior added, a bout coded with
 *     keys, one deleted -> saved -> Python reads it -> reopened in LUC3D's real loader,
 *     which has sleap-app's edits and everything only LUC3D reads (detector, run settings,
 *     states) -> edited there -> saved -> opened in sleap-app again.
 *
 *  B. A plain SLEAP project (written by Python sleap-io: tracks, no session, no
 *     identities): labelled in sleap-app -> saved -> opened in LUC3D, whose animals are
 *     its tracks -> edited there -> saved -> opened in sleap-app again, where the bouts
 *     are still on the same tracks -> Python reads it.
 *
 * Needs a checkout of sleap-app linked to a sleap-io.js with events: SLEAP_APP_DIR (default
 * ../sleap-app beside the main LUC3D checkout), and `uv` for Python. Skips without them.
 *
 * Run: node tests/e2e/behavior-cross-app.mjs
 *      (spawns http.server on PORT and sleap-app's dev server on APP_PORT)
 * WRITE_FIXTURES=<dir> also writes the two files the repos keep as fixtures, so each tests
 * the other's file without the other's code (`luc3d_behavior.slp` in sleap-app's
 * tests/fixtures, `sleap_app_behavior.slp` in tests/fixtures/behavior-events here).
 */
import { chromium } from 'playwright';
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8194);
const APP_PORT = Number(process.env.APP_PORT || 5183);
const SLEAP_APP_DIR = process.env.SLEAP_APP_DIR || path.join(os.homedir(), 'Desktop', 'sleap-app');
const PY_SPEC = process.env.SLEAP_IO_SPEC || 'sleap-io==0.9.5';
const MOD = process.platform === 'darwin' ? 'Meta' : 'Control';

const hasApp = fs.existsSync(path.join(SLEAP_APP_DIR, 'src', 'lib', 'behaviorEvents.ts'));
const hasUv = spawnSync('uv', ['--version'], { stdio: 'ignore' }).status === 0;
const hasBun = spawnSync('bun', ['--version'], { stdio: 'ignore' }).status === 0;
if (!hasApp || !hasUv || !hasBun) {
    console.log(`SKIP: needs sleap-app with behavior labels at ${SLEAP_APP_DIR} (${hasApp}), uv (${hasUv}) and bun (${hasBun})`);
    process.exit(0);
}

let fails = 0;
const check = (c, m, extra) => {
    console.log((c ? '  ✓ ' : '  ✗ ') + m + (!c && extra !== undefined ? '  got: ' + JSON.stringify(extra).slice(0, 400) : ''));
    if (!c) fails++;
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'behavior-cross-app-'));

/** What Python sleap-io reads from a file, as plain data. */
function pythonReads(file) {
    const script = `
import json, sys, sleap_io as sio
labels = sio.load_slp(sys.argv[1], open_videos=False)
def who(p):
    return None if p is None else [type(p).__name__, p.name]
print(json.dumps({
    "types": [t.name for t in labels.event_types],
    "videos": len(labels.videos),
    "events": sorted(([e.type.name, e.start_frame, e.end_frame, who(e.subject), who(e.target), e.is_predicted, e.source]
                      for e in labels.events), key=lambda r: (r[1], r[2], r[0], str(r[3]))),
}))
`;
    const r = spawnSync('uv', ['run', '--quiet', '--with', PY_SPEC, 'python', '-I', '-c', script, file], { encoding: 'utf8' });
    if (r.status !== 0) throw new Error('python failed: ' + r.stderr);
    return JSON.parse(r.stdout.trim().split('\n').pop());
}

/** A plain SLEAP project, as Python sleap-io writes one: tracks, no session, no identities. */
function pythonPlainProject(file) {
    const script = `
import sys, numpy as np, sleap_io as sio
skeleton = sio.Skeleton(nodes=["nose", "tail"], edges=[("nose", "tail")])
video = sio.Video(filename="plain.mp4", open_backend=False)
video.backend_metadata["shape"] = (100, 64, 64, 1)
t1, t2 = sio.Track(name="mouse1"), sio.Track(name="mouse2")
frames = []
for f in range(100):
    frames.append(sio.LabeledFrame(video=video, frame_idx=f, instances=[
        sio.PredictedInstance.from_numpy(np.array([[10.0 + f / 10, 10.0], [20.0, 20.0]]), skeleton=skeleton, track=t1, score=0.9, point_scores=np.array([0.9, 0.9])),
        sio.PredictedInstance.from_numpy(np.array([[40.0, 40.0 + f / 10], [50.0, 50.0]]), skeleton=skeleton, track=t2, score=0.9, point_scores=np.array([0.9, 0.9])),
    ]))
sio.Labels(labeled_frames=frames, videos=[video], skeletons=[skeleton], tracks=[t1, t2]).save(sys.argv[1], embed=False)
`;
    const r = spawnSync('uv', ['run', '--quiet', '--with', PY_SPEC, 'python', '-I', '-c', script, file], { encoding: 'utf8' });
    if (r.status !== 0) throw new Error('python failed: ' + r.stderr);
}

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
const appServer = spawn('bun', ['run', 'dev', '--port', String(APP_PORT), '--strictPort'], { cwd: SLEAP_APP_DIR, stdio: 'ignore', detached: true });
const stopServers = () => {
    server.kill();
    try { process.kill(-appServer.pid); } catch { /* gone */ }
};

async function waitForHttp(url) {
    for (let i = 0; i < 120; i++) {
        try { const r = await fetch(url); if (r.ok) return; } catch { /* not yet */ }
        await new Promise(r => setTimeout(r, 500));
    }
    throw new Error('no answer from ' + url);
}

/** Bytes into a page's global, across the Node/page boundary. */
async function putBytes(page, name, bytes) {
    await page.evaluate(([n, b64]) => {
        const bin = atob(b64);
        const u = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
        window[n] = u;
    }, [name, Buffer.from(bytes).toString('base64')]);
}
async function getBytes(page, name) {
    const b64 = await page.evaluate((n) => {
        const u = window[n];
        let s = '';
        for (let i = 0; i < u.length; i += 32768) s += String.fromCharCode.apply(null, u.subarray(i, i + 32768));
        return btoa(s);
    }, name);
    return Buffer.from(b64, 'base64');
}

// ----- LUC3D page ------------------------------------------------------------------
async function openLuc3d(browser) {
    const page = await browser.newPage();
    page.on('pageerror', e => { check(false, 'LUC3D page error: ' + String(e).slice(0, 300)); });
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state && window.SleapIO, { timeout: 30000 });
    return page;
}

/** LUC3D's real loader, dismissing the prompts it raises. */
async function luc3dLoad(page, bytes, name) {
    await putBytes(page, '__file', bytes);
    await page.evaluate((n) => {
        window.__loadDone = false;
        (async () => {
            await (await import('/import-export/slp-import.js')).handleLoadSlpFile(new File([window.__file], n));
        })().then(() => { window.__loadDone = true; }, e => { window.__loadDone = 'ERR ' + (e && e.stack || e); });
    }, name);
    for (let i = 0; i < 240; i++) {
        const s = await page.evaluate(() => {
            const b = [...document.querySelectorAll('button')].find(x => x.offsetParent && /^(Skip|Later|Cancel|Not now)$/i.test((x.textContent || '').trim()));
            if (b) { b.click(); return { done: false }; }
            return { done: window.__loadDone, prompt: !!document.getElementById('videoSelectOverlay') };
        });
        if (s.prompt) await page.keyboard.press('Escape');
        if (s.done) return s.done;
        await page.waitForTimeout(250);
    }
    return 'timeout';
}

/** What the open LUC3D session holds. */
const luc3dState = (page) => page.evaluate(async () => {
    const L = await import('/pose/behavior-labels.js');
    const P = await import('/pose/behavior-predictions.js');
    const S = await import('/pose/behavior-states.js');
    const eth = await import('/pose/behavior-ethogram.js');
    const coding = await import('/ui/behavior-coding.js');
    const s = window.__lucid.state.session;
    const nameOf = (id) => (eth.behaviorById(id) || {}).name;
    return {
        behaviors: eth.behaviors().map(b => ({ id: b.id, name: b.name, key: b.key, social: b.social, mutual: b.mutual, kind: b.kind, detector: !!b.detector, description: b.description })),
        features: eth.featureSettings(),
        animals: coding.sessionAnimals(s),
        bouts: L.exportBoutRows(s).map(r => [nameOf(r[0]), r[1], r[2], r[3], r[4], r[5], r[6], r[7]]),
        predicted: P.predictedBouts(s).map(b => [nameOf(b.behaviorId), b.animal, b.target, b.start, b.end, b.score]),
        runs: (P.exportPredictionHeaders(s) || { runs: [] }).runs.map(r => ({ behavior: nameOf(r.behavior), source: r.source, cfg: r.cfg })),
        states: S.exportStateLayers(s),
    };
});

/** File ▸ Save as the app does it without a file handle. */
async function luc3dSave(page) {
    await page.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        const sl = await import('/import-export/save-load.js');
        const cams = [];
        for (const s of AS.state.sessions) for (const c of s.cameras) if (!cams.includes(c.name)) cams.push(c.name);
        AS.state.views = cams.map(n => ({ name: n, videoWidth: 256, videoHeight: 256, frameCount: 100 }));
        AS.state.videoFiles = cams.map(n => ({ name: n, assignedCamera: n, videoPath: n + '.mp4' }));
        const orig = URL.createObjectURL;
        let blob = null;
        URL.createObjectURL = (b) => { blob = b; return orig.call(URL, b); };
        try { await sl.saveProjectSlp(); } finally { URL.createObjectURL = orig; }
        if (!blob) throw new Error('saveProjectSlp produced no file: ' + (document.getElementById('statusText') || {}).textContent);
        window.__saved = new Uint8Array(await blob.arrayBuffer());
    });
    return await getBytes(page, '__saved');
}

// ----- sleap-app page --------------------------------------------------------------
async function openSleapApp(browser) {
    const context = await browser.newContext({ acceptDownloads: true });
    const page = await context.newPage();
    // Saving falls back to a download, which a test can catch (no native file picker).
    await page.addInitScript(() => {
        delete Window.prototype.showSaveFilePicker;
        delete window.showSaveFilePicker;
    });
    const errs = [];
    page.on('pageerror', e => errs.push(String(e).slice(0, 300)));
    page.pageErrors = errs;
    await page.goto(`http://localhost:${APP_PORT}/`);
    await page.waitForFunction(() => window.sleap && window.sleap.store, { timeout: 60000 });
    return page;
}

/** sleap-app's real loader, dismissing the prompts it raises (no video on disk here). */
async function appLoad(page, bytes, name) {
    await putBytes(page, '__file', bytes);
    await page.evaluate((n) => {
        window.__loadDone = false;
        window.sleap.loadProjectFromFile(new File([window.__file], n))
            .then((ok) => { window.__loadDone = ok === true ? true : 'not loaded'; }, e => { window.__loadDone = 'ERR ' + e; });
    }, name);
    for (let i = 0; i < 240; i++) {
        const s = await page.evaluate(() => {
            const dialog = document.querySelector('[role="dialog"], [role="alertdialog"]');
            if (dialog) {
                const b = [...dialog.querySelectorAll('button')].find(x => /skip|discard|continue|don.t save|cancel|close/i.test(x.textContent || ''));
                if (b) { b.click(); return { done: false }; }
            }
            return { done: window.__loadDone };
        });
        if (s.done) return s.done;
        await page.waitForTimeout(250);
    }
    return 'timeout';
}

/** What sleap-app's open project holds, read the way its panel reads it. */
const appState = (page) => page.evaluate(() => {
    const { labels, video } = window.sleap.store.getState();
    const who = (p) => (p ? p.name : '');
    return {
        types: labels.eventTypes.map(t => ({ name: t.name, metadata: t.metadata, description: t.description })),
        videos: labels.videos.length,
        sessions: labels.sessions.length,
        tracks: labels.tracks.map(t => t.name),
        identities: labels.identities.map(i => i.name),
        videoName: video && String(video.filename),
        lucidKeys: labels.sessions.map(s => Object.keys((s.metadata || {}).lucid || {}).sort()),
        events: labels.events.map(e => [e.type.name, e.startFrame, e.endFrame, who(e.subject), who(e.target), e.isPredicted, e.source, e.metadata.note || '']),
    };
});

async function openBehaviorPanel(page) {
    if (await page.locator('[data-testid="behavior-panel"]').count()) return;
    await page.locator('[data-panel-id="behavior"]').click();
    await page.waitForSelector('[data-testid="behavior-panel"]', { timeout: 10000 });
    // The icon strip widens over the panel while the mouse is on it.
    await page.mouse.move(300, 300);
}

async function appAddBehavior(page, name, key, pairLabel) {
    await page.locator('[data-testid="behavior-add"]').click();
    await page.fill('#behavior-name', name);
    if (key) await page.fill('#behavior-key', key);
    if (pairLabel) {
        await page.locator('[data-testid="behavior-pair"]').click();
        await page.getByRole('option', { name: pairLabel }).click();
    }
    await page.locator('[data-testid="behavior-save"]').click();
    await page.waitForFunction((n) => window.sleap.store.getState().labels.eventTypes.some(t => t.name === n), name);
}

const appGoTo = (page, f) => page.evaluate((frame) => window.sleap.store.getState().setFrameIdx(frame), f);

/** Save with the app's own shortcut; the bytes come back as a download. */
async function appSave(page) {
    const [download] = await Promise.all([
        page.waitForEvent('download', { timeout: 30000 }),
        page.keyboard.press(`${MOD}+s`),
    ]);
    return fs.readFileSync(await download.path());
}

const sortRows = (rows) => rows.slice().sort((a, b) => a[1] - b[1] || a[2] - b[2] || String(a[0]).localeCompare(String(b[0])) || String(a[3]).localeCompare(String(b[3])));

let browser;
try {
    await waitForHttp(`http://localhost:${PORT}/index.html`);
    await waitForHttp(`http://localhost:${APP_PORT}/`);
    browser = await chromium.launch();
    const luc3d = await openLuc3d(browser);
    const app = await openSleapApp(browser);

    // =================================================================
    console.log('\n-- A1. a multi-camera project labelled in LUC3D, saved --');
    await luc3d.evaluate(async () => {
        const pd = await import('/pose/pose-data.js');
        const AS = await import('/ui/app-state.js');
        const meta = await import('/import-export/behavior-metadata.js');
        const eth = await import('/pose/behavior-ethogram.js');
        const L = await import('/pose/behavior-labels.js');
        const P = await import('/pose/behavior-predictions.js');
        const S = await import('/pose/behavior-states.js');
        const fileio = await import('/import-export/file-io.js');
        const { Skeleton, Camera, Instance, InstanceGroup, FrameGroup, Session } = pd;
        const NODES = 4, CAMS = ['cam0', 'cam1'], N = 100;
        const mtx = [[1000, 0, 128], [0, 1000, 128], [0, 0, 1]];
        const cams = CAMS.map((cn, ci) => new Camera(cn, mtx, [0, 0, 0, 0, 0], [0, 0.1 * ci, 0], [10 * ci, 0, 0], [256, 256]));
        const skel = new Skeleton('skeleton', Array.from({ length: NODES }, (_, i) => 'n' + i), Array.from({ length: NODES - 1 }, (_, i) => [i, i + 1]));
        const s = new Session(cams, skel, ['t0', 't1'], 'sessA');
        const ids = ['mouseA', 'mouseB'].map(n => s.addIdentity(n).id);
        for (let f = 0; f < N; f++) {
            const fg = new FrameGroup(f); s.addFrameGroup(fg);
            s.instanceGroups.set(f, ids.map((id, k) => {
                const g = new InstanceGroup(f * 10 + k + 1, id);
                for (const cn of CAMS) {
                    const inst = new Instance(Array.from({ length: NODES }, (_, n) => [50 + n + f / 10 + 40 * k, 60 + n]), k, 'predicted', 0.9);
                    g.addInstance(cn, inst); fg.addInstance(cn, inst);
                }
                const p = new Float64Array(NODES * 3);
                for (let n = 0; n < NODES; n++) { p[3 * n] = n + f * 0.01 + 50 * k; p[3 * n + 1] = n * 2; p[3 * n + 2] = 1 + n; }
                g.points3d = p;
                return g;
            }));
        }
        meta.resetBehaviorState();
        const rear = eth.addBehavior({ name: 'rear', key: 'r', description: 'both forepaws off the floor' }).behavior;
        const chase = eth.addBehavior({ name: 'chase', key: 'c', social: true }).behavior;
        const sniff = eth.addBehavior({ name: 'nose-to-nose', key: 'n', social: true, mutual: true }).behavior;
        const bite = eth.addBehavior({ name: 'bite', key: 'b', kind: 'point', social: true }).behavior;
        eth.setBehaviorDetector(rear.id, { conditions: [{ feature: 'n0 height · mean ±0.1 s', op: '>', value: 0.5 }], minDuration: 0.2 });
        eth.setFeatureSettings({ nodeRoles: { head: 'n0', tailBase: 'n3' } });
        L.labelRange(s, rear.id, 'mouseA', '', 5, 20, true, { note: 'tall' });
        L.labelRange(s, rear.id, 'mouseA', '', 21, 25, false);
        L.labelRange(s, chase.id, 'mouseA', 'mouseB', 41, 50, true);
        L.labelRange(s, sniff.id, 'mouseB', 'mouseA', 60, 65, true);
        L.labelRange(s, bite.id, 'mouseB', 'mouseA', 70, 70, true);
        L.noteBehaviorFps(s, 50);
        P.setBehaviorRun(s, { behaviorId: rear.id, cfg: 'the-rule-print', fps: 30, mode: '3d',
            bouts: [{ animal: 'mouseB', target: '', start: 30, end: 33, score: 0.51 }, { animal: 'mouseA', target: '', start: 80, end: 84, score: 0.9 }] });
        S.setStateLayer(s, { name: 'keypoint-MoSeq', source: 'keypoint-MoSeq', stateNames: { 2: 'groom' }, animals: { mouseA: [[0, 9, 2], [10, 59, 0]] } });
        AS.state.sessions = [s]; AS.state.session = s; AS.state.activeSessionIdx = 0;
        const views = CAMS.map(cn => ({ name: cn, videoWidth: 256, videoHeight: 256, frameCount: N }));
        const vfs = CAMS.map(cn => ({ name: cn, assignedCamera: cn, videoPath: cn + '.mp4' }));
        AS.state.views = views; AS.state.videoFiles = vfs;
        window.__saved = await window.SleapIO.saveSlpToBytes(fileio.buildSlpLabelsAllViews(s, views, vfs));
    });
    const fileA1 = await getBytes(luc3d, '__saved');
    if (process.env.WRITE_FIXTURES) fs.writeFileSync(path.join(process.env.WRITE_FIXTURES, 'luc3d_behavior.slp'), fileA1);
    const before = await luc3dState(luc3d);
    check(before.bouts.length === 5 && before.predicted.length === 2, 'five labelled bouts and two predicted ones');

    console.log('\n-- A2. opened in sleap-app --');
    check(await appLoad(app, fileA1, 'from-luc3d.slp') === true, 'sleap-app opens it');
    let a = await appState(app);
    check(a.videos === 2 && a.sessions === 1, 'two cameras, one recording session', a);
    check(eq(a.identities, ['mouseA', 'mouseB']), 'its animals are the identities', a.identities);
    check(eq(a.types.map(t => t.name), ['rear', 'chase', 'nose-to-nose', 'bite', 'not rear']), 'the ethogram, and a type for "not" labels', a.types.map(t => t.name));
    check(a.types[0].description === 'both forepaws off the floor' && a.types[0].metadata.key === 'r', 'descriptions and keys come along');
    check(a.types[1].metadata.directed === 'true' && a.types[2].metadata.mutual === 'true' && a.types[3].metadata.kind === 'point', 'directed, mutual, point');
    check(eq(sortRows(a.events).filter(r => !r[5]).map(r => r.slice(0, 5)), [
        ['rear', 5, 20, 'mouseA', ''],
        ['not rear', 21, 25, 'mouseA', ''],
        ['chase', 41, 50, 'mouseA', 'mouseB'],
        ['nose-to-nose', 60, 65, 'mouseA', 'mouseB'],
        ['bite', 70, 70, 'mouseB', 'mouseA'],
    ]), 'every bout, frame for frame (the mutual pair in name order)', sortRows(a.events));
    check(a.events.some(r => r[0] === 'rear' && r[7] === 'tall'), 'and the note');
    check(a.events.filter(r => r[5]).every(r => r[6] === 'rule'), 'the predictions say where they came from');
    await openBehaviorPanel(app);
    check(await app.locator('[data-testid="behavior-row"]').count() === 4, 'the panel lists the four behaviors, not the "not" type');
    check(await app.locator('[data-testid="bout-row"]').count() === 5, 'and the five bouts');
    check(await app.locator('[data-testid="prediction-row"]').count() === 2, 'and the two predictions');
    // The same bouts on the other camera: a bout belongs to the session.
    await app.evaluate(() => { const st = window.sleap.store.getState(); st.setVideo(st.labels.videos[1]); });
    await app.waitForTimeout(150);
    check(await app.locator('[data-testid="bout-row"]').count() === 5, 'the second camera shows them too');
    check(await app.locator('[data-testid="behavior-band"]').count() === 1, 'and draws them under the seekbar');

    console.log('\n-- A3. edited in sleap-app, through its UI --');
    if (process.env.SHOTS) await app.screenshot({ path: path.join(process.env.SHOTS, 'a3-before.png') });
    await app.locator('[data-testid="prediction-row"]').first().getByRole('button', { name: 'Accept' }).click();
    await app.waitForFunction(() => window.sleap.store.getState().labels.events.some(e => e.source === 'accepted_prediction'));
    check(true, 'a prediction accepted');
    await appAddBehavior(app, 'groom', 'g');
    check(true, 'a behavior added');
    await app.locator('[data-testid="behavior-mode"]').click();
    await appGoTo(app, 86);
    await app.keyboard.press('2');                       // mouseB
    await app.keyboard.press('g');
    await appGoTo(app, 90);
    await app.keyboard.press('g');
    await app.waitForFunction(() => window.sleap.store.getState().labels.events.some(e => e.type.name === 'groom'));
    await appGoTo(app, 92);
    await app.keyboard.press('b');                       // a directed point: the other animal is its target
    await app.waitForFunction(() => window.sleap.store.getState().labels.events.filter(e => e.type.name === 'bite').length === 2);
    await app.keyboard.press('Escape');
    a = await appState(app);
    check(a.events.some(r => r[0] === 'groom' && r[1] === 86 && r[2] === 90 && r[3] === 'mouseB'), 'a state coded with its key, on the animal the digit picked', a.events);
    check(a.events.some(r => r[0] === 'bite' && r[1] === 92 && r[3] === 'mouseB' && r[4] === 'mouseA'), 'a directed point, its target the other animal');
    // Deleting the first bout from the list, and undoing nothing else.
    await app.locator('[data-testid="bout-row"]').filter({ hasText: 'nose-to-nose' }).getByRole('button', { name: 'Delete bout' }).click();
    await app.waitForFunction(() => !window.sleap.store.getState().labels.events.some(e => e.type.name === 'nose-to-nose'));
    check(true, 'a bout deleted');
    await app.keyboard.press(`${MOD}+z`);
    await app.waitForFunction(() => window.sleap.store.getState().labels.events.some(e => e.type.name === 'nose-to-nose'));
    check(true, 'and undone with the app\'s own undo');
    await app.locator('[data-testid="bout-row"]').filter({ hasText: 'nose-to-nose' }).getByRole('button', { name: 'Delete bout' }).click();
    await app.waitForFunction(() => !window.sleap.store.getState().labels.events.some(e => e.type.name === 'nose-to-nose'));
    const fileA2 = await appSave(app);
    check(fileA2.length > 1000, `saved (${fileA2.length.toLocaleString('en-US')} bytes)`);
    check(app.pageErrors.length === 0, 'no page errors in sleap-app', app.pageErrors);

    console.log('\n-- A4. Python sleap-io reads what sleap-app saved --');
    const pathA2 = path.join(tmp, 'a2.slp');
    fs.writeFileSync(pathA2, fileA2);
    let py = pythonReads(pathA2);
    check(eq(py.types, ['rear', 'chase', 'nose-to-nose', 'bite', 'not rear', 'groom']), 'the catalog', py.types);
    check(py.events.some(e => eq(e.slice(0, 6), ['groom', 86, 90, ['Identity', 'mouseB'], null, false])), 'sleap-app\'s new bout, on the identity', py.events);
    check(py.events.some(e => eq(e.slice(0, 6), ['rear', 5, 20, ['Identity', 'mouseA'], null, false])), 'LUC3D\'s bout');
    check(py.events.filter(e => e[5]).length === 2, 'and both predictions, still predictions');

    console.log('\n-- A5. back in LUC3D, through its real loader --');
    check(await luc3dLoad(luc3d, fileA2, 'back.slp') === true, 'LUC3D opens it');
    let l = await luc3dState(luc3d);
    check(l.behaviors.map(b => b.name).join() === 'rear,chase,nose-to-nose,bite,groom', 'the ethogram, with sleap-app\'s behavior', l.behaviors.map(b => b.name));
    check(l.behaviors[0].id === before.behaviors[0].id && l.behaviors[0].key === 'r' && l.behaviors[0].description === 'both forepaws off the floor', 'LUC3D\'s behaviors keep their ids, keys and descriptions');
    check(l.behaviors[2].mutual && l.behaviors[1].social && !l.behaviors[1].mutual && l.behaviors[3].kind === 'point', 'and what kind they are');
    check(l.behaviors[0].detector && eq(l.features, { nodeRoles: { head: 'n0', tailBase: 'n3' } }), 'the detector and feature settings came through sleap-app untouched');
    check(eq(l.states, before.states) && l.states.layers[0].stateNames[2] === 'groom', 'so did the keypoint-MoSeq states');
    check(l.runs.some(r => r.behavior === 'rear' && r.source === 'rule' && r.cfg === 'the-rule-print'), 'and the settings print of the predictions', l.runs);
    check(eq(l.predicted.map(p => p.slice(0, 5)).sort(), before.predicted.map(p => p.slice(0, 5)).sort()), 'both predicted bouts', l.predicted);
    const byName = (name) => l.bouts.filter(b => b[0] === name);
    check(eq(byName('groom'), [['groom', 'mouseB', '', 86, 90, 1, 'm', '']]), 'sleap-app\'s coded bout', byName('groom'));
    check(byName('bite').some(b => b[1] === 'mouseB' && b[2] === 'mouseA' && b[3] === 92), 'its directed point');
    check(byName('nose-to-nose').length === 0, 'its deletion');
    check(byName('rear').some(b => b[3] === 5 && b[4] === 20 && b[5] === 1 && b[6] === 'm' && b[7] === 'tall'), 'LUC3D\'s bout and note');
    check(byName('rear').some(b => b[3] === 21 && b[4] === 25 && b[5] === 0), 'the "not" label is "not" again');
    check(byName('rear').some(b => b[1] === 'mouseB' && b[3] === 30 && b[4] === 33 && b[5] === 1 && b[6] === 'a')
        || byName('rear').some(b => b[1] === 'mouseA' && b[3] === 80 && b[4] === 84 && b[5] === 1 && b[6] === 'a'), 'the accepted prediction is an accepted label', byName('rear'));

    console.log('\n-- A6. edited in LUC3D, saved, opened in sleap-app again --');
    await luc3d.evaluate(async () => {
        const L = await import('/pose/behavior-labels.js');
        const eth = await import('/pose/behavior-ethogram.js');
        const s = window.__lucid.state.session;
        L.labelRange(s, eth.behaviorByName('groom').id, 'mouseA', '', 94, 96, true, { note: 'from LUC3D' });
    });
    const fileA3 = await luc3dSave(luc3d);
    check(await appLoad(app, fileA3, 'again.slp') === true, 'sleap-app opens it');
    a = await appState(app);
    check(a.events.some(r => r[0] === 'groom' && r[1] === 94 && r[2] === 96 && r[3] === 'mouseA' && r[7] === 'from LUC3D'), 'LUC3D\'s new bout and note', a.events);
    check(a.events.some(r => r[0] === 'groom' && r[1] === 86 && r[3] === 'mouseB'), 'sleap-app\'s earlier bout');
    check(a.types.find(t => t.name === 'groom').metadata.key === 'g', 'the key sleap-app gave its behavior');

    // =================================================================
    console.log('\n-- B1. a plain SLEAP project labelled in sleap-app --');
    const pathB0 = path.join(tmp, 'plain.slp');
    pythonPlainProject(pathB0);
    check(await appLoad(app, fs.readFileSync(pathB0), 'plain.slp') === true, 'sleap-app opens a file Python wrote');
    a = await appState(app);
    check(a.sessions === 0 && a.identities.length === 0 && eq(a.tracks, ['mouse1', 'mouse2']), 'no session, no identities, two tracks', a);
    await openBehaviorPanel(app);
    await appAddBehavior(app, 'rear', 'r');
    await appAddBehavior(app, 'chase', 'c', 'One animal to another');
    await app.locator('[data-testid="behavior-mode"]').click();
    await appGoTo(app, 10);
    await app.keyboard.press('r');
    await appGoTo(app, 19);
    await app.keyboard.press('r');
    await app.keyboard.press('2');
    await appGoTo(app, 30);
    await app.keyboard.press('c');
    await appGoTo(app, 39);
    await app.keyboard.press('c');
    await app.waitForFunction(() => window.sleap.store.getState().labels.events.length === 2);
    // A range selected on the seekbar, labelled "not".
    await app.evaluate(() => window.sleap.store.getState().set('frameRange', [50, 59]));
    await app.keyboard.press('Shift+R');
    await app.waitForFunction(() => window.sleap.store.getState().labels.events.length === 3);
    await app.evaluate(() => window.sleap.store.getState().set('frameRange', null));
    await app.keyboard.press('Escape');
    a = await appState(app);
    check(eq(sortRows(a.events).map(r => r.slice(0, 5)), [
        ['rear', 10, 19, 'mouse1', ''],
        ['chase', 30, 39, 'mouse2', 'mouse1'],
        ['not rear', 50, 59, 'mouse2', ''],
    ]), 'coded on tracks, with keys', sortRows(a.events));
    const fileB1 = await appSave(app);
    if (process.env.WRITE_FIXTURES) fs.writeFileSync(path.join(process.env.WRITE_FIXTURES, 'sleap_app_behavior.slp'), fileB1);

    console.log('\n-- B2. opened in LUC3D, whose animals are the tracks --');
    check(await luc3dLoad(luc3d, fileB1, 'from-app.slp') === true, 'LUC3D opens it');
    l = await luc3dState(luc3d);
    check(eq(l.animals, ['mouse1', 'mouse2']), 'the tracks are its animals', l.animals);
    check(l.behaviors.map(b => b.name).join() === 'rear,chase' && l.behaviors[0].key === 'r' && l.behaviors[1].social, 'the ethogram', l.behaviors);
    check(eq(l.bouts.map(r => r.slice(0, 7)), [
        ['rear', 'mouse1', '', 10, 19, 1, 'm'],
        ['chase', 'mouse2', 'mouse1', 30, 39, 1, 'm'],
        ['rear', 'mouse2', '', 50, 59, 0, 'm'],
    ]), 'every bout, the "not" as a "not"', l.bouts);
    await luc3d.evaluate(async () => {
        const L = await import('/pose/behavior-labels.js');
        const eth = await import('/pose/behavior-ethogram.js');
        const s = window.__lucid.state.session;
        L.labelRange(s, eth.behaviorByName('rear').id, 'mouse2', '', 70, 79, true, { note: 'from LUC3D' });
    });
    const fileB2 = await luc3dSave(luc3d);

    console.log('\n-- B3. back in sleap-app, then Python --');
    check(await appLoad(app, fileB2, 'from-luc3d-2.slp') === true, 'sleap-app opens it');
    a = await appState(app);
    check(a.events.some(r => r[0] === 'rear' && r[1] === 70 && r[3] === 'mouse2' && r[7] === 'from LUC3D'), 'LUC3D\'s bout', a.events);
    check(a.events.some(r => r[0] === 'rear' && r[1] === 10 && r[3] === 'mouse1'), 'and sleap-app\'s');
    const pathB2 = path.join(tmp, 'b2.slp');
    fs.writeFileSync(pathB2, fileB2);
    py = pythonReads(pathB2);
    check(py.events.some(e => eq(e.slice(0, 6), ['rear', 10, 19, ['Track', 'mouse1'], null, false])), 'Python reads the bout on the Track', py.events);
    check(py.events.some(e => eq(e.slice(0, 6), ['not rear', 50, 59, ['Track', 'mouse2'], null, false])), 'and the "not" as its own type');
    check(py.events.some(e => eq(e.slice(0, 6), ['rear', 70, 79, ['Track', 'mouse2'], null, false])), 'and LUC3D\'s bout');
    check(app.pageErrors.length === 0, 'no page errors in sleap-app', app.pageErrors);

    console.log(fails === 0 ? '\nPASS' : `\nFAIL — ${fails} check(s)`);
    process.exitCode = fails === 0 ? 0 : 1;
} finally {
    if (browser) await browser.close().catch(() => {});
    stopServers();
    fs.rmSync(tmp, { recursive: true, force: true });
}
