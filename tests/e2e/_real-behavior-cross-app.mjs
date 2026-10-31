/**
 * _real-behavior-cross-app.mjs — behavior labels between LUC3D and sleap-app on
 * a real recording, as a user would move them:
 *   make        LUC3D: load a per-camera session folder, Track All, Triangulate All
 *               (at REPROJ px), the rule templates + two hand-coded behaviors, Detect,
 *               accept / reject some predictions, hand-label bouts, a states layer the
 *               size of a real keypoint-MoSeq fit, Save. Writes OUT/luc3d.slp and
 *               OUT/luc3d.json (what the session holds).
 *   reopen      LUC3D, fresh page: open OUT/luc3d.slp, compare with OUT/luc3d.json.
 *   python      Python sleap-io and h5py read OUT/luc3d.slp's events.
 *   app         sleap-app: open OUT/luc3d.slp, compare, attach the videos, open the
 *               Behavior panel, step frames, code a bout with keys, accept and reject
 *               predictions, delete a bout, undo, Save. Writes OUT/app.slp and
 *               OUT/app.json.
 *   reopen-app  LUC3D, fresh page: open OUT/app.slp, compare with OUT/app.json, and
 *               with the poses and the settings of OUT/luc3d.json.
 *   reopen-in-app  sleap-app, fresh page: open OUT/app.slp, its own save, compare with OUT/app.json.
 *   python-app  Python sleap-io reads OUT/app.slp's events.
 *   diff        h5py: every dataset of OUT/luc3d.slp against OUT/app.slp.
 * Every stage reports time, the JS heap, the renderer's and all of Chrome's resident
 * memory (ps, every 500 ms) and long tasks, in OUT/summary.json, and survives a crashed
 * tab as a result. Never writes into the dataset folder.
 *
 * Not a test (real dataset, real Chrome for HEVC). Headless, so no browser lock. Usage:
 *   DATASET=<folder> NANIMALS=5 OUT=<dir> node tests/e2e/_real-behavior-cross-app.mjs
 *   env: STAGES=make,reopen,python,app,reopen-app,reopen-in-app,python-app,diff  EXCLUDE=<regex of files to leave out>
 *        REPROJ=25  SLEAP_APP_DIR=~/Desktop/sleap-app  SHOTS=1 (screenshots into OUT)
 *        APP_PROD=1 (sleap-app's production build, not its dev server: `bun run build` first)
 *        PROFILE=1 (a .cpuprofile of each step in sleap-app)  STOP_AFTER=edits
 *        SINGLE=1 (DATASET is one camera's folder: its .slp opened as a plain SLEAP file, no
 *        Track All, Triangulate All or Detect, which need identities: the animals are its tracks)
 */
import { chromium } from 'playwright';
import { spawn, spawnSync, execSync } from 'node:child_process';
import net from 'node:net';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { answerTrackSetup } from './fixtures/track-setup.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const DATASET = process.env.DATASET;
const NANIMALS = process.env.NANIMALS ?? '';
const OUT = process.env.OUT;
const EXCLUDE = process.env.EXCLUDE || '(^|/)\\.|/troubleshooting/';
const REPROJ = Number(process.env.REPROJ ?? 25);
const STAGES = (process.env.STAGES || 'make,reopen,python,app,reopen-app,reopen-in-app,python-app,diff').split(',');
const SLEAP_APP_DIR = process.env.SLEAP_APP_DIR || path.join(os.homedir(), 'Desktop', 'sleap-app');
const SHOTS = process.env.SHOTS === '1';
const PROFILE = process.env.PROFILE === '1';   // a .cpuprofile of each step of sleap-app into OUT
const STOP_AFTER = process.env.STOP_AFTER || '';  // end the app stage after this step (for profiling)
const SINGLE = process.env.SINGLE === '1';
const PY_SPEC = process.env.PY_SPEC || 'sleap-io==0.9.5';
if (!DATASET || !OUT) { console.error('set DATASET and OUT'); process.exit(2); }
if (path.resolve(OUT).startsWith(path.resolve(DATASET))) { console.error('OUT must not be inside the dataset'); process.exit(2); }
fs.mkdirSync(OUT, { recursive: true });
const LUC3D_SLP = path.join(OUT, 'luc3d.slp'), APP_SLP = path.join(OUT, 'app.slp');
const log = (m) => process.stdout.write(`[${new Date().toISOString().slice(11, 19)}] ${m}\n`);
const mb = (b) => Math.round(b / 1048576);
const summaryFile = path.join(OUT, 'summary.json');
const summary = fs.existsSync(summaryFile) ? JSON.parse(fs.readFileSync(summaryFile, 'utf8')) : { dataset: DATASET, reproj: REPROJ, stages: {} };
const saveSummary = () => fs.writeFileSync(summaryFile, JSON.stringify(summary, null, 1));
const fails = [];
const check = (ok, what, detail) => {
    log((ok ? '  ok   ' : '  FAIL ') + what + (!ok && detail !== undefined ? ' — ' + JSON.stringify(detail).slice(0, 600) : ''));
    if (!ok) fails.push(what);
    return ok;
};

const freePort = () => new Promise((res, rej) => {
    const s = net.createServer();
    s.on('error', rej);
    s.listen(0, () => { const p = s.address().port; s.close(() => res(p)); });
});

/** Chrome's memory by process kind, found by its profile folder. */
function chromeMemory(profileDir) {
    let out;
    try { out = execSync('ps -A -o pid=,ppid=,rss=,command=', { maxBuffer: 64 << 20 }).toString(); } catch (e) { return null; }
    const procs = [];
    for (const line of out.split('\n')) {
        const m = line.trim().match(/^(\d+)\s+(\d+)\s+(\d+)\s+(.*)$/);
        if (m) procs.push({ pid: +m[1], ppid: +m[2], rss: +m[3] * 1024, cmd: m[4] });
    }
    const root = procs.find((p) => p.cmd.includes(profileDir) && !/--type=/.test(p.cmd));
    if (!root) return null;
    const kids = new Map();
    for (const p of procs) { if (!kids.has(p.ppid)) kids.set(p.ppid, []); kids.get(p.ppid).push(p); }
    const all = [], q = [root];
    while (q.length) { const p = q.shift(); all.push(p); for (const c of kids.get(p.pid) || []) q.push(c); }
    return {
        renderer: Math.max(0, ...all.filter((p) => /--type=renderer/.test(p.cmd)).map((p) => p.rss)),
        gpu: all.filter((p) => /--type=gpu-process/.test(p.cmd)).reduce((s, p) => s + p.rss, 0),
        total: all.reduce((s, p) => s + p.rss, 0),
    };
}

/** A headless real Chrome with its own profile, memory sampling, a crash flag and a stage timer. */
async function openBrowser(stageName, url, init) {
    const profileDir = fs.mkdtempSync(path.join(os.tmpdir(), 'lucid-cross-app-'));
    const context = await chromium.launchPersistentContext(profileDir, {
        headless: true, channel: 'chrome', viewport: { width: 1480, height: 900 }, acceptDownloads: true,
        args: ['--window-size=1500,950', '--enable-precise-memory-info', '--js-flags=--expose-gc'],
    });
    const page = context.pages()[0] || await context.newPage();
    const b = { context, page, profileDir, crashed: false, errors: [] };
    page.on('crash', () => { b.crashed = true; log('*** the tab crashed ***'); });
    page.on('pageerror', (e) => { b.errors.push(String(e).slice(0, 300)); log('pageerror: ' + String(e).slice(0, 300)); });
    await page.addInitScript(() => {
        window.__lt = [];
        try {
            new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__lt.push([e.startTime, e.duration]); })
                .observe({ type: 'longtask', buffered: true });
        } catch (e) { /* no long-task timing */ }
        window.__heap = { peak: 0 };
        setInterval(() => { if (performance.memory) window.__heap.peak = Math.max(window.__heap.peak, performance.memory.usedJSHeapSize); }, 100);
    });
    if (init) await init(page);
    await page.goto(url);
    const rows = summary.stages[stageName] = [];
    let cdp = null;
    if (PROFILE) {
        cdp = await context.newCDPSession(page);
        await cdp.send('Profiler.enable');
        await cdp.send('Profiler.setSamplingInterval', { interval: 500 });
    }
    /** One measured step: wall time, heap peak, long tasks, Chrome's memory. */
    b.step = async (name, fn) => {
        const profiling = PROFILE && !/^open /.test(name);
        if (profiling) await cdp.send('Profiler.start');
        const pk = { renderer: 0, gpu: 0, total: 0 };
        const sample = () => { const m = chromeMemory(profileDir); if (m) { pk.renderer = Math.max(pk.renderer, m.renderer); pk.gpu = Math.max(pk.gpu, m.gpu); pk.total = Math.max(pk.total, m.total); } };
        sample();
        const timer = setInterval(sample, 500);
        const t0p = await page.evaluate(() => { if (window.gc) window.gc(); window.__heap.peak = performance.memory.usedJSHeapSize; return performance.now(); });
        const t0 = Date.now();
        let out, error = null;
        try { out = await fn(); } catch (e) { error = String(e && e.message || e).slice(0, 500); }
        const ms = Date.now() - t0;
        clearInterval(timer);
        sample();
        if (profiling && !b.crashed) {
            const { profile } = await cdp.send('Profiler.stop');
            fs.writeFileSync(path.join(OUT, 'profile-' + name.replace(/\W+/g, '-') + '.cpuprofile'), JSON.stringify(profile));
        }
        const row = { name, ms, rendererPeakMB: mb(pk.renderer), gpuPeakMB: mb(pk.gpu), chromePeakMB: mb(pk.total) };
        if (b.crashed) { row.crashed = true; } else {
            const after = await page.evaluate((t) => {
                window.__heap.peak = Math.max(window.__heap.peak, performance.memory.usedJSHeapSize);
                const lt = window.__lt.filter((e) => e[0] >= t);
                return { heapPeak: window.__heap.peak, heapEnd: performance.memory.usedJSHeapSize, longTasks: lt.length,
                         longestTaskMs: Math.round(lt.reduce((m, e) => Math.max(m, e[1]), 0)) };
            }, t0p).catch(() => null);
            if (after) Object.assign(row, { heapPeakMB: mb(after.heapPeak), heapEndMB: mb(after.heapEnd), longTasks: after.longTasks, longestTaskMs: after.longestTaskMs });
        }
        if (error) row.error = error;
        if (out && out.info) row.info = out.info;
        rows.push(row);
        saveSummary();
        log(`${name}: ${(ms / 1000).toFixed(2)} s · heap peak ${row.heapPeakMB} MB · renderer ${row.rendererPeakMB} MB · Chrome ${row.chromePeakMB} MB · ` +
            `long tasks ${row.longTasks} (longest ${row.longestTaskMs} ms)` + (row.info ? ' · ' + JSON.stringify(row.info).slice(0, 400) : '') + (error ? ' · ERROR ' + error : ''));
        if (error || b.crashed) throw new Error(name + ': ' + (error || 'the tab crashed'));
        return out;
    };
    b.close = async () => {
        await context.close().catch(() => {});
        try { fs.rmSync(profileDir, { recursive: true, force: true }); } catch (e) { /* left in tmp */ }
    };
    return b;
}

const shot = async (page, name) => { if (SHOTS) await page.screenshot({ path: path.join(OUT, name + '.png') }).catch(() => {}); };

// ----- what a LUC3D session holds ----------------------------------------------------
const luc3dState = (page) => page.evaluate(async () => {
    const L = await import('./pose/behavior-labels.js');
    const P = await import('./pose/behavior-predictions.js');
    const S = await import('./pose/behavior-states.js');
    const eth = await import('./pose/behavior-ethogram.js');
    const coding = await import('./ui/behavior-coding.js');
    const det = await import('./ui/behavior-detect.js');
    const s = window.__lucid.state.session;
    const nameOf = (id) => (eth.behaviorById(id) || {}).name;
    let groups = 0, with3d = 0;
    s.instanceGroups.forEach((gs) => { for (const g of gs) { groups++; if (g.points3d) with3d++; } });
    return {
        session: s.name,
        cameras: s.cameras.map((c) => c.name),
        frames: window.__lucid.state.totalFrames,
        fps: window.__lucid.state.fps,
        lazy: !!s.lazyLoader,
        groups, with3d,
        behaviors: eth.behaviors().map((b) => ({ name: b.name, key: b.key, social: !!b.social, mutual: !!b.mutual, kind: b.kind, detector: b.detector || null, description: b.description })),
        features: eth.featureSettings(),
        animals: coding.sessionAnimals(s),
        bouts: L.exportBoutRows(s).map((r) => [nameOf(r[0]), r[1], r[2], r[3], r[4], r[5], r[6], r[7]]),
        // A save keeps a score to three decimals (`exportPredictionRuns`).
        predicted: P.predictedBouts(s).map((b) => [nameOf(b.behaviorId), b.animal, b.target, b.start, b.end, Number.isFinite(b.score) ? Math.round(b.score * 1000) / 1000 : null, b.source]),
        runs: (P.exportPredictionHeaders(s) || { runs: [] }).runs.map((r) => ({ behavior: nameOf(r.behavior), source: r.source, cfg: r.cfg, fps: r.fps, mode: r.mode })),
        statesJson: JSON.stringify(S.exportStateLayers(s)),
        stale: (() => { try { return P.predictionStaleness(s, det.detectionEnv()); } catch (e) { return String(e); } })(),
    };
});

const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const sortRows = (rows) => rows.slice().sort((a, b) => { const x = JSON.stringify(a), y = JSON.stringify(b); return x < y ? -1 : x > y ? 1 : 0; });
/** The rows two forms of the labels agree on: behavior, animal, target, start, end, present. */
const boutKey = (r) => r.slice(0, 6);
const predKey = (r) => r.slice(0, 5);

/** The bouts sleap-app's events say, as LUC3D rows. */
function appRowsAsLuc3d(appState) {
    const bouts = [], preds = [];
    const negates = new Map(appState.types.filter((t) => t.metadata.negates).map((t) => [t.name, t.metadata.negates]));
    for (const e of appState.events) {
        const [type, start, end, subject, target, predicted] = e;
        if (predicted) preds.push([type, subject, target, start, end]);
        else if (negates.has(type)) bouts.push([negates.get(type), subject, target, start, end, 0]);
        else bouts.push([type, subject, target, start, end, 1]);
    }
    return { bouts, preds };
}

// ----- LUC3D: open a saved file on a fresh page ---------------------------------------
/** Answer the prompts a LUC3D loader raises, so it returns. */
const dismissLoaderPrompts = (page) => page.evaluate(() => {
    // "Skip — Load Videos Later" lets the loader return.
    setInterval(() => {
        for (const x of document.querySelectorAll('button')) {
            const t = x.textContent.trim();
            if ((t === 'Continue' || t.startsWith('Skip') || t === 'Later' || t === 'Not now' || t === 'Cancel' || t === 'Close' || t === 'OK') && x.offsetParent !== null) x.click();
        }
        if (document.getElementById('videoSelectOverlay')) document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    }, 250);
});

async function luc3dOpenSaved(b, file, stepName) {
    const page = b.page;
    await page.waitForFunction(() => window.__lucid && window.SleapIO, null, { timeout: 60000 });
    await page.evaluate(() => {
        const inp = document.createElement('input');
        inp.type = 'file'; inp.id = '__reloadPicker'; inp.style.cssText = 'position:fixed;left:-9999px';
        document.body.appendChild(inp);
    });
    await dismissLoaderPrompts(page);
    await page.setInputFiles('#__reloadPicker', file);
    await b.step(stepName, async () => {
        await page.evaluate((single) => {
            window.__reload = { done: false, err: null };
            const f = document.getElementById('__reloadPicker').files[0];
            const load = single
                ? import('./import-export/slp-import.js').then((m) => m.handleLoadSlpFile(f))
                : import('./loading/session-loader.js').then((sl) => sl.handleLoadProjectSlpLazy(f));
            load.then((r) => { window.__reload.done = true; window.__reload.result = r; })
                .catch((e) => { window.__reload.err = String(e && e.stack || e).slice(0, 400); });
        }, SINGLE);
        for (;;) {
            if (b.crashed) throw new Error('the tab crashed');
            const r = await page.evaluate(() => window.__reload).catch(() => null);
            if (r && r.err) throw new Error(r.err);
            if (r && r.done) break;
            await new Promise((res) => setTimeout(res, 500));
        }
        return { info: await page.evaluate(() => ({ status: document.getElementById('statusText').textContent.slice(0, 200) })) };
    });
    // The re-print that decides stale runs is sliced between tasks.
    await page.evaluate(async () => {
        const P = await import('./pose/behavior-predictions.js');
        const s = window.__lucid.state.session;
        const t0 = performance.now();
        while (P.predictionCheckPending(s) && performance.now() - t0 < 180000) await new Promise((r) => setTimeout(r, 100));
    });
}

// ----- the stages ----------------------------------------------------------------------
async function stageMake(port) {
    const b = await openBrowser('make', `http://localhost:${port}/index.html`, async (page) => {
        page.on('dialog', (d) => d.accept(String(NANIMALS)));
        await answerTrackSetup(page, NANIMALS);
    });
    const { page } = b;
    try {
        await page.waitForFunction(() => window.__lucid && window.__lucid.state, null, { timeout: 60000 });
        await page.evaluate(async (reproj) => {
            const st = await import('./ui/settings.js');
            st.setTrackingThresholds({ autoSwitchCheck: 0, autoImageSwitchCheck: 0, reprojErrorThreshold: reproj });
            const inp = document.createElement('input');
            inp.type = 'file'; inp.multiple = true; inp.webkitdirectory = true; inp.id = '__dir';
            inp.style.cssText = 'position:fixed;left:-9999px'; document.body.appendChild(inp);
            setInterval(() => { for (const x of document.querySelectorAll('button')) { const t = x.textContent.trim(); if ((t === 'Continue' || t === 'OK') && x.offsetParent !== null) x.click(); } }, 300);
        }, REPROJ);
        const clickStep = async (button, re, what) => {
            const prev = await page.evaluate(() => document.getElementById('statusText').textContent);
            await page.evaluate((id) => document.getElementById(id).click(), button);
            for (const t0 = Date.now(); ;) {
                const st = await page.evaluate(() => ({ s: document.getElementById('statusText').textContent, hidden: document.getElementById('loadingOverlay').classList.contains('hidden') }));
                if (st.hidden && st.s !== prev && re.test(st.s)) return { info: { status: st.s.slice(0, 160) } };
                if (st.hidden && /error|failed/i.test(st.s) && st.s !== prev) throw new Error(what + ': ' + st.s);
                if (Date.now() - t0 > 60 * 60000) throw new Error(what + ' timed out');
                await new Promise((r) => setTimeout(r, 250));
            }
        };
        if (SINGLE) {
            const slp = fs.readdirSync(DATASET).filter((n) => /\.slp$/.test(n))[0];
            if (!slp) throw new Error('no .slp in ' + DATASET);
            await page.evaluate(() => {
                const inp = document.createElement('input');
                inp.type = 'file'; inp.id = '__one'; inp.style.cssText = 'position:fixed;left:-9999px';
                document.body.appendChild(inp);
            });
            await dismissLoaderPrompts(page);
            await page.setInputFiles('#__one', path.join(DATASET, slp));
            await b.step('load one camera', async () => {
                await page.evaluate(() => {
                    window.__one = { done: false, err: null };
                    import('./import-export/slp-import.js').then((m) => m.handleLoadSlpFile(document.getElementById('__one').files[0]))
                        .then(() => { window.__one.done = true; }, (e) => { window.__one.err = String(e && e.stack || e).slice(0, 400); });
                });
                await page.waitForFunction(() => window.__one.done || window.__one.err, null, { timeout: 30 * 60000, polling: 500 });
                const err = await page.evaluate(() => window.__one.err);
                if (err) throw new Error(err);
                return { info: await page.evaluate(() => { const s = window.__lucid.state.session, st = window.__lucid.state;
                    return { cams: s.cameras.length, frames: st.totalFrames, fps: st.fps, lazy: !!s.lazyLoader, nodes: s.skeleton.nodes.length, tracks: s.tracks.length, name: s.name }; }) };
            });
        } else {
        await b.step('load the recording', async () => {
            await page.setInputFiles('#__dir', DATASET);
            await page.evaluate(async (ex) => {
                const sl = await import('./loading/session-loader.js');
                const re = new RegExp(ex);
                await sl.handleLoadSessionFolderPerCamera(Array.from(document.getElementById('__dir').files).filter((f) => !re.test(f.webkitRelativePath || f.name)), false);
            }, EXCLUDE);
            await page.waitForFunction(() => window.__lucid.state.session && window.__lucid.state.session.cameras && window.__lucid.state.session.cameras.length > 1 &&
                document.getElementById('loadingOverlay').classList.contains('hidden'), null, { timeout: 60 * 60000, polling: 1000 });
            await new Promise((r) => setTimeout(r, 3000));
            return { info: await page.evaluate(() => { const s = window.__lucid.state.session, st = window.__lucid.state;
                return { cams: s.cameras.length, frames: st.totalFrames, fps: st.fps, lazy: !!s.lazyLoader, nodes: s.skeleton.nodes.length, name: s.name }; }) };
        });
        await b.step('Track All', () => clickStep('tbTrackAll', /Assigned \d+ identities/, 'Track All'));
        await page.waitForSelector('#trackSummaryClose', { timeout: 10000 }).then(() => page.click('#trackSummaryClose')).catch(() => {});
        await b.step('Triangulate All', () => clickStep('tbTriangulateAll', /^(Triangulated|Grouped) /, 'Triangulate All'));
        }

        // The ethogram: the rule templates, with keys, and two behaviors only hands label.
        await page.evaluate(async () => {
            const T = await import('./pose/behavior-tensor.js');
            const eth = await import('./pose/behavior-ethogram.js');
            const rules = await import('./pose/behavior-rules.js');
            const st = window.__lucid.state;
            const input = T.collectBehaviorInput(st.session, { numFrames: Math.min(st.totalFrames, 1000) });
            const roleNames = T.resolveBody(input.nodeNames, input.edges, eth.featureSettings()).roleNames;
            const keys = 'rlimnfacdeh';
            rules.DETECTOR_TEMPLATES.forEach((t, i) => {
                const made = eth.addBehavior({ name: t.name, social: t.social, mutual: t.mutual, description: t.description, key: keys[i] }).behavior;
                eth.setBehaviorDetector(made.id, rules.instantiateTemplate(t, roleNames).detector);
            });
            eth.addBehavior({ name: 'groom', key: 'g', description: 'hand-coded in this test' });
            eth.addBehavior({ name: 'bite', key: 'b', kind: 'point', social: true });
        });
        await b.step('Detect (all frames)', () => page.evaluate(async () => {
            const det = await import('./ui/behavior-detect.js');
            await det.detectAllFrames();
            const P = await import('./pose/behavior-predictions.js');
            return { info: { status: document.getElementById('statusText').textContent.slice(0, 200), predicted: P.predictedBouts(window.__lucid.state.session).length } };
        }));
        await b.step('review and hand labels', () => page.evaluate(async () => {
            const P = await import('./pose/behavior-predictions.js');
            const L = await import('./pose/behavior-labels.js');
            const S = await import('./pose/behavior-states.js');
            const eth = await import('./pose/behavior-ethogram.js');
            const coding = await import('./ui/behavior-coding.js');
            const st = window.__lucid.state, s = st.session;
            const all = P.predictedBouts(s);
            // Every 6th bout accepted, every 11th rejected, spread over the recording.
            const accepted = all.filter((x, i) => i % 6 === 0), rejected = all.filter((x, i) => i % 11 === 5);
            P.acceptPredictions(s, accepted);
            P.rejectPredictions(s, rejected);
            // Hand labels, in stretches the detectors leave alone: groom present and "not", bite points.
            const animals = coding.sessionAnimals(s);
            const groom = eth.behaviorByName('groom'), bite = eth.behaviorByName('bite');
            // With no video loaded the frame count is only in the poses.
            let T = st.totalFrames;
            if (!T) s.frameGroups.forEach((fg, f) => { T = Math.max(T, f + 1); });
            if (!T && s.lazyLoader) T = s.lazyLoader.nFrames;
            const hand = [];
            for (let i = 0; i < 300; i++) {
                const a = animals[i % animals.length];
                const start = Math.floor((i + 0.1) * T / 300);
                const end = Math.min(T - 1, start + 40 + (i % 17) * 11);
                hand.push({ behaviorId: groom.id, animal: a, start, end, present: i % 5 !== 4, note: i % 25 === 0 ? 'note ' + i : '' });
            }
            for (let i = 0; i < 80; i++) {
                const a = animals[i % animals.length], tg = animals[(i + 1) % animals.length];
                hand.push({ behaviorId: bite.id, animal: a, target: tg, start: Math.floor((i + 0.5) * T / 80), end: Math.floor((i + 0.5) * T / 80), present: true });
            }
            L.labelBouts(s, hand, { source: 'm' });
            L.noteBehaviorFps(s, st.fps);
            // A states layer the size of a real keypoint-MoSeq fit.
            const lay = { name: 'keypoint-MoSeq', source: 'keypoint-MoSeq', stateNames: { 0: 'walk' }, animals: {} };
            animals.slice(0, 8).forEach((a, ai) => {
                const runs = [];
                let f = 0, k = 0;
                while (f < T) { const n = 8 + ((k * 7919 + ai * 104729) % 37); runs.push([f, Math.min(T - 1, f + n - 1), (k * 31 + ai) % 30]); f += n; k++; }
                lay.animals[a] = runs;
            });
            S.setStateLayer(s, lay);
            return { info: { predicted: all.length, accepted: accepted.length, rejected: rejected.length, hand: hand.length, bouts: L.exportBoutRows(s).length,
                             stateRuns: Object.values(lay.animals).reduce((n, r) => n + r.length, 0) } };
        }));
        // File ▸ Save as the app does it without a save-file picker.
        await b.step('Save', async () => {
            const [dl] = await Promise.all([
                page.waitForEvent('download', { timeout: 60 * 60000 }),
                page.evaluate(async () => (await import('./import-export/save-load.js')).saveProjectSlp()),
            ]);
            await dl.saveAs(LUC3D_SLP);
            return { info: { MB: +(fs.statSync(LUC3D_SLP).size / 1048576).toFixed(1), status: await page.evaluate(() => document.getElementById('statusText').textContent.slice(0, 120)) } };
        });
        const state = await luc3dState(page);
        fs.writeFileSync(path.join(OUT, 'luc3d.json'), JSON.stringify(state));
        log(`session ${state.session}: ${state.cameras.length} cameras, ${state.frames} frames, ${state.animals.length} animals, ${state.groups} groups, ` +
            `${state.bouts.length} labelled bouts, ${state.predicted.length} predicted bouts, runs stale: ${JSON.stringify(state.stale)}`);
        summary.file = { bytes: fs.statSync(LUC3D_SLP).size };
        saveSummary();
    } finally { await b.close(); }
}

async function compareLuc3d(state, want, what, opts) {
    if (state.groups === undefined) state.groups = 0;
    check(eq(state.animals, want.animals), `${what}: the same animals`, [state.animals, want.animals]);
    check(state.frames === want.frames, `${what}: the same frame count`, [state.frames, want.frames]);
    check(eq(state.cameras, want.cameras), `${what}: the same cameras`, [state.cameras, want.cameras]);
    check(state.groups === want.groups && state.with3d === want.with3d, `${what}: every 3D group came back (${state.groups.toLocaleString('en-US')})`, [state.groups, state.with3d, want.groups, want.with3d]);
    check(eq(state.behaviors, want.behaviors), `${what}: the ethogram, keys and detectors`, [state.behaviors, want.behaviors]);
    check(eq(state.features, want.features), `${what}: the feature settings`);
    check(state.statesJson === want.statesJson, `${what}: the states layer, exactly (${(want.statesJson.length / 1024).toFixed(0)} KB)`);
    if (opts && opts.rowsOnly) return;
    check(eq(sortRows(state.bouts.map(boutKey)), sortRows(want.bouts.map(boutKey))), `${what}: every labelled bout, frame for frame (${want.bouts.length.toLocaleString('en-US')})`);
    check(eq(sortRows(state.bouts), sortRows(want.bouts)), `${what}: with its source and note`);
    check(eq(sortRows(state.predicted), sortRows(want.predicted)), `${what}: every predicted bout with its score (${want.predicted.length.toLocaleString('en-US')})`);
    check(eq(state.runs, want.runs), `${what}: every run's source and settings print`);
}

async function stageReopen(port) {
    const want = JSON.parse(fs.readFileSync(path.join(OUT, 'luc3d.json'), 'utf8'));
    const b = await openBrowser('reopen', `http://localhost:${port}/index.html`);
    try {
        await luc3dOpenSaved(b, LUC3D_SLP, 'open luc3d.slp in LUC3D');
        const state = await luc3dState(b.page);
        fs.writeFileSync(path.join(OUT, 'reopen.json'), JSON.stringify(state));
        await compareLuc3d(state, want, 'LUC3D reopen');
        check(!(state.stale && (state.stale.input || state.stale.poses)), 'LUC3D reopen: the predictions do not read stale', state.stale);
        summary.reopenStale = state.stale;
    } finally { await b.close(); }
}

function stagePython(file = LUC3D_SLP, stageName = 'python') {
    const want = file === LUC3D_SLP
        ? JSON.parse(fs.readFileSync(path.join(OUT, 'luc3d.json'), 'utf8'))
        : (() => { const r = appRowsAsLuc3d(JSON.parse(fs.readFileSync(path.join(OUT, 'app.json'), 'utf8'))); return { bouts: r.bouts, predicted: r.preds }; })();
    const py = `
import json, sys, time, h5py, numpy as np
out = {}
t0 = time.time()
with h5py.File(sys.argv[1], 'r') as f:
    out['datasets'] = {k: list(f[k].shape) for k in f.keys() if isinstance(f[k], h5py.Dataset)}
    out['format'] = str(f['metadata'].attrs.get('format_id'))
    ev = f['events']
    out['h5_events'] = int(ev['start_frame'].shape[0]) if 'start_frame' in ev else int(ev.shape[0])
out['h5_seconds'] = round(time.time() - t0, 2)
t0 = time.time()
import sleap_io as sio
labels = sio.load_slp(sys.argv[1], open_videos=False, lazy=True)
out['sio_seconds'] = round(time.time() - t0, 2)
out['sio_version'] = sio.__version__
out['sio_videos'] = len(labels.videos)
out['sio_types'] = [t.name for t in labels.event_types]
out['sio_events'] = len(labels.events)
out['sio_user'] = sum(1 for e in labels.events if not e.is_predicted)
out['sio_predicted'] = sum(1 for e in labels.events if e.is_predicted)
print(json.dumps(out))
`;
    const t0 = Date.now();
    const r = spawnSync('uv', ['run', '--quiet', '--with', PY_SPEC, '--with', 'h5py', 'python', '-I', '-c', py, file], { encoding: 'utf8', maxBuffer: 64 << 20 });
    if (r.status !== 0) { check(false, 'Python reads ' + path.basename(file), r.stderr.slice(-800)); return; }
    const o = JSON.parse(r.stdout.trim().split('\n').pop());
    summary.stages[stageName] = Object.assign({ ms: Date.now() - t0 }, o);
    saveSummary();
    log('python: ' + path.basename(file) + ' ' + JSON.stringify(o).slice(0, 700));
    check(o.sio_user === want.bouts.length, `Python sleap-io ${o.sio_version} reads every labelled bout (${o.sio_user.toLocaleString('en-US')})`, [o.sio_user, want.bouts.length]);
    check(o.sio_predicted === want.predicted.length, `and every predicted bout (${o.sio_predicted.toLocaleString('en-US')})`, [o.sio_predicted, want.predicted.length]);
}

/** Every event of sleap-app's open project. */
const appState = (page) => page.evaluate(() => {
    const { labels, video } = window.sleap.store.getState();
    const who = (p) => (p ? p.name : '');
    return {
        types: labels.eventTypes.map((t) => ({ name: t.name, metadata: t.metadata, description: t.description })),
        videos: labels.videos.length,
        videoNames: labels.videos.map((v) => String(v.filename)),
        sessions: labels.sessions.length,
        tracks: labels.tracks.map((t) => t.name),
        identities: labels.identities.map((i) => i.name),
        videoName: video && String(video.filename),
        labeledFrames: labels.labeledFrames.length,
        lucidKeys: labels.sessions.map((s) => Object.keys((s.metadata || {}).lucid || {}).sort()),
        events: labels.events.map((e) => [e.type.name, e.startFrame, e.endFrame, who(e.subject), who(e.target), e.isPredicted, e.source, e.metadata.note || '']),
    };
});

/** A repaint after an action: the time until two animation frames have passed. */
const settle = (page) => page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r(performance.now())))));

/** sleap-app's dev server, or its production build, on a free port. */
async function startSleapApp() {
    const port = await freePort();
    const args = process.env.APP_PROD === '1'
        ? ['x', 'vite', 'preview', '--port', String(port), '--strictPort']
        : ['run', 'dev', '--port', String(port), '--strictPort'];
    const server = spawn('bun', args, { cwd: SLEAP_APP_DIR, stdio: 'ignore', detached: true });
    const url = `http://localhost:${port}/`;
    for (let i = 0; i < 120; i++) {
        try { const r = await fetch(url); if (r.ok) break; } catch (e) { /* not yet */ }
        await new Promise((r) => setTimeout(r, 500));
    }
    return { url, stop: () => { try { process.kill(-server.pid); } catch (e) { /* gone */ } } };
}

/** Open a file in sleap-app through its own loader, as a measured step. */
async function appOpen(b, file, stepName, answer = /skip|discard|continue|don.t save|close/i) {
    const { page } = b;
    await page.waitForFunction(() => window.sleap && window.sleap.store, null, { timeout: 120000 });
    await page.evaluate(() => {
        const inp = document.createElement('input');
        inp.type = 'file'; inp.id = '__pick'; inp.style.cssText = 'position:fixed;left:-9999px';
        document.body.appendChild(inp);
    });
    await page.setInputFiles('#__pick', file);
    await b.step(stepName, async () => {
        await page.evaluate(() => {
            window.__open = { done: false, ok: null };
            window.sleap.loadProjectFromFile(document.getElementById('__pick').files[0])
                .then((ok) => { window.__open.ok = ok; window.__open.done = true; }, (e) => { window.__open.ok = 'ERR ' + e; window.__open.done = true; });
        });
        for (;;) {
            if (b.crashed) throw new Error('the tab crashed');
            const r = await page.evaluate((src) => {
                const dialog = document.querySelector('[role="dialog"], [role="alertdialog"]');
                if (dialog) { const x = [...dialog.querySelectorAll('button')].find((y) => new RegExp(src, 'i').test(y.textContent || '')); if (x) x.click(); }
                return window.__open;
            }, answer.source).catch(() => null);
            if (r && r.done) { if (r.ok !== true) throw new Error('sleap-app did not open it: ' + r.ok); break; }
            // A dev server reload (a file of the app or of sleap-io.js changed) drops the open.
            if (r === undefined) throw new Error('the page reloaded during the open');
            await new Promise((res) => setTimeout(res, 500));
        }
        return { info: await page.evaluate(() => { const { labels } = window.sleap.store.getState();
            return { videos: labels.videos.length, labeledFrames: labels.labeledFrames.length, events: labels.events.length, types: labels.eventTypes.length }; }) };
    });
}

async function stageApp() {
    const want = JSON.parse(fs.readFileSync(path.join(OUT, 'luc3d.json'), 'utf8'));
    const app = await startSleapApp();
    const b = await openBrowser('app', app.url, async (page) => {
        // Saving falls back to a download, which a test can catch (no native file picker).
        await page.addInitScript(() => { delete Window.prototype.showSaveFilePicker; delete window.showSaveFilePicker; });
        page.on('dialog', (d) => d.accept().catch(() => {}));
    });
    const { page } = b;
    const MOD = 'Meta';
    try {
        await appOpen(b, LUC3D_SLP, 'open luc3d.slp in sleap-app');
        await page.evaluate(() => {
            const vids = document.createElement('input');
            vids.type = 'file'; vids.multiple = true; vids.id = '__vids'; vids.style.cssText = 'position:fixed;left:-9999px';
            document.body.appendChild(vids);
        });
        await shot(page, 'app-opened');
        const a0 = await appState(page);
        check(a0.videos === want.cameras.length, 'sleap-app: the cameras', [a0.videos]);
        check(eq(a0.identities.length ? a0.identities : a0.tracks, want.animals), 'sleap-app: the same animals', [a0.identities.length, a0.tracks.length]);
        const rows0 = appRowsAsLuc3d(a0);
        check(eq(sortRows(rows0.bouts), sortRows(want.bouts.map(boutKey))), `sleap-app: every labelled bout, frame for frame (${want.bouts.length.toLocaleString('en-US')})`,
            { app: rows0.bouts.length, luc3d: want.bouts.length });
        check(eq(sortRows(rows0.preds), sortRows(want.predicted.map(predKey))), `sleap-app: every predicted bout (${want.predicted.length.toLocaleString('en-US')})`,
            { app: rows0.preds.length, luc3d: want.predicted.length });

        // A production build keeps its modules to itself, so it runs without the videos.
        if (process.env.APP_PROD !== '1') await b.step('attach the videos', async () => {
            const files = [];
            const walk = (d) => { for (const n of fs.readdirSync(d)) { const p = path.join(d, n); if (fs.statSync(p).isDirectory()) walk(p); else if (/\.mp4$/i.test(n)) files.push(p); } };
            walk(DATASET);
            await page.setInputFiles('#__vids', files);
            const info = await page.evaluate(async () => {
                const R = await import('/src/lib/resolveVideos.ts');
                const { labels } = window.sleap.store.getState();
                const picked = Array.from(document.getElementById('__vids').files);
                const res = [];
                for (const v of labels.videos) {
                    const base = String(v.filename).split(/[\\/]/).pop().toLowerCase();
                    const f = picked.find((x) => x.name.toLowerCase() === base);
                    res.push({ name: base.slice(0, 40), found: !!f, ok: f ? await R.assignVideoBackend(v, f, { silent: true }) : false, issue: R.videoIssue(v) });
                }
                window.sleap.store.getState().markVideoUpdated();
                return res;
            });
            return { info: { attached: info.filter((x) => x.ok).length, of: info.length, problems: info.filter((x) => !x.ok).slice(0, 3) } };
        });
        await page.waitForTimeout(2000);
        await shot(page, 'app-videos');

        await b.step('open the Behavior panel', async () => {
            await page.locator('[data-panel-id="behavior"]').click();
            await page.waitForSelector('[data-testid="behavior-panel"]', { timeout: 120000 });
            await page.mouse.move(300, 300);
            await settle(page);
            return { info: { behaviors: await page.locator('[data-testid="behavior-row"]').count(), bouts: await page.locator('[data-testid="bout-row"]').count(),
                             predictions: await page.locator('[data-testid="prediction-row"]').count(), band: await page.locator('[data-testid="behavior-band"]').count() } };
        });
        await shot(page, 'app-panel');
        await b.step('step 30 frames', async () => {
            await page.evaluate(() => window.sleap.store.getState().setFrameIdx(1000));
            await settle(page);
            const t0 = await page.evaluate(() => performance.now());
            for (let i = 0; i < 30; i++) { await page.keyboard.press('ArrowRight'); await settle(page); }
            const t1 = await page.evaluate(() => performance.now());
            return { info: { msPerFrame: +((t1 - t0) / 30).toFixed(1), frame: await page.evaluate(() => window.sleap.store.getState().frameIdx) } };
        });
        await shot(page, 'app-stepped');
        await b.step('code a bout with keys', async () => {
            await page.locator('[data-testid="behavior-mode"]').click();
            await page.evaluate(() => window.sleap.store.getState().setFrameIdx(2000));
            await settle(page);
            await page.keyboard.press('2');
            const t0 = await page.evaluate(() => performance.now());
            await page.keyboard.press('g');
            await settle(page);
            await page.evaluate(() => window.sleap.store.getState().setFrameIdx(2040));
            await settle(page);
            await page.keyboard.press('g');
            // It may join a bout the animal already has there, so look for the frames.
            await page.waitForFunction(() => {
                const { labels } = window.sleap.store.getState();
                const animal = (labels.identities.length ? labels.identities : labels.tracks)[1];
                return labels.events.some((e) => !e.isPredicted && e.type.name === 'groom' && e.subject === animal && e.startFrame <= 2000 && e.endFrame >= 2040);
            }, null, { timeout: 120000 });
            await settle(page);
            const t1 = await page.evaluate(() => performance.now());
            await shot(page, 'app-labelling');
            await page.keyboard.press('Escape');
            await settle(page);
            return { info: { ms: Math.round(t1 - t0) } };
        });
        await shot(page, 'app-coded');
        const editSeq = () => window.sleap.store.getState().editSeq;
        const afterEdit = async (act) => {
            const n0 = await page.evaluate(editSeq);
            const t0 = await page.evaluate(() => performance.now());
            await act();
            await page.waitForFunction((v) => window.sleap.store.getState().editSeq !== v, n0, { timeout: 120000 });
            await settle(page);
            return { info: { ms: Math.round((await page.evaluate(() => performance.now())) - t0) } };
        };
        // Clicked in the page: Playwright's role locators walk every row's accessible name.
        const clickRow = (name, sel, btn) => b.step(name, () => afterEdit(() => page.evaluate(([q, label]) => {
            for (const row of document.querySelectorAll(q)) {
                const x = [...row.querySelectorAll('button')].find((y) => (y.getAttribute('aria-label') || y.textContent || '').trim() === label);
                if (x) { x.click(); return; }
            }
            throw new Error('no ' + label + ' button');
        }, [sel, btn])));
        if (rows0.preds.length) {
            await clickRow('accept a prediction', '[data-testid="prediction-row"]', 'Accept');
            await clickRow('reject a prediction', '[data-testid="prediction-row"]', 'Reject');
        }
        await clickRow('delete a bout', '[data-testid="bout-row"]', 'Delete bout');
        await b.step('undo', () => afterEdit(() => page.keyboard.press(`${MOD}+z`)));
        await shot(page, 'app-edited');
        if (STOP_AFTER === 'edits') return;
        // The crash-recovery draft: a beat after the last edit, up to a minute on a big project.
        await b.step('idle while the draft autosaves', async () => {
            await page.waitForTimeout(70000);
            return { info: await page.evaluate(() => new Promise((resolve) => {
                const s = window.sleap.store.getState();
                const open = indexedDB.open('sleap-app-behavior-drafts', 1);
                open.onupgradeneeded = () => open.result.createObjectStore('drafts');
                open.onsuccess = () => {
                    const req = open.result.transaction('drafts').objectStore('drafts').getAll();
                    req.onsuccess = () => { open.result.close(); resolve({ fullDraft: s.labelsDraftPath || null, labelDrafts: req.result.length, events: req.result[0] ? req.result[0].events.length : 0 }); };
                };
            })) };
        });
        const before = await appState(page);
        fs.writeFileSync(path.join(OUT, 'app.json'), JSON.stringify(before));
        // Close the app with the labels unsaved, open the same file: they are offered back.
        if (before.labeledFrames >= 200000) {
            await page.evaluate(() => { window.onbeforeunload = null; });
            await page.reload();
            await appOpen(b, LUC3D_SLP, 'open luc3d.slp again, restore the labels', /restore/i);
            const restored = await appState(page);
            check(eq(sortRows(restored.events), sortRows(before.events)), `sleap-app: the unsaved labels came back after closing it (${before.events.length.toLocaleString('en-US')})`,
                { restored: restored.events.length, before: before.events.length });
            await page.evaluate(() => { document.activeElement && document.activeElement.blur(); });
        }
        await b.step('Save', async () => {
            const [dl] = await Promise.all([
                page.waitForEvent('download', { timeout: 60 * 60000 }),
                page.keyboard.press(`${MOD}+s`),
            ]);
            await dl.saveAs(APP_SLP);
            return { info: { MB: +(fs.statSync(APP_SLP).size / 1048576).toFixed(1) } };
        });
        check(b.errors.length === 0, 'sleap-app: no page errors', b.errors.slice(0, 3));
    } finally {
        await b.close();
        app.stop();
    }
}

/** sleap-app reads back its own save: the compound tables, gzipped from a million rows. */
async function stageReopenInApp() {
    const edited = JSON.parse(fs.readFileSync(path.join(OUT, 'app.json'), 'utf8'));
    const app = await startSleapApp();
    const b = await openBrowser('reopen-in-app', app.url);
    try {
        await appOpen(b, APP_SLP, 'open app.slp in sleap-app');
        const again = await appState(b.page);
        const a = appRowsAsLuc3d(again), w = appRowsAsLuc3d(edited);
        check(eq(sortRows(a.bouts), sortRows(w.bouts)), `sleap-app reopen: every labelled bout (${w.bouts.length.toLocaleString('en-US')})`);
        check(eq(sortRows(a.preds), sortRows(w.preds)), `sleap-app reopen: every predicted bout (${w.preds.length.toLocaleString('en-US')})`);
        check(again.labeledFrames === edited.labeledFrames, `sleap-app reopen: every frame (${edited.labeledFrames.toLocaleString('en-US')})`);
    } finally {
        await b.close();
        app.stop();
    }
}

async function stageReopenApp(port) {
    const want = JSON.parse(fs.readFileSync(path.join(OUT, 'luc3d.json'), 'utf8'));
    const edited = JSON.parse(fs.readFileSync(path.join(OUT, 'app.json'), 'utf8'));
    const b = await openBrowser('reopen-app', `http://localhost:${port}/index.html`);
    try {
        await luc3dOpenSaved(b, APP_SLP, 'open app.slp in LUC3D');
        const state = await luc3dState(b.page);
        fs.writeFileSync(path.join(OUT, 'reopen-app.json'), JSON.stringify(state));
        await compareLuc3d(state, want, 'after sleap-app', { rowsOnly: true });
        const rows = appRowsAsLuc3d(edited);
        check(eq(sortRows(state.bouts.map(boutKey)), sortRows(rows.bouts)), `after sleap-app: every labelled bout as sleap-app left it (${rows.bouts.length.toLocaleString('en-US')})`,
            { luc3d: state.bouts.length, app: rows.bouts.length });
        check(eq(sortRows(state.predicted.map(predKey)), sortRows(rows.preds)), `after sleap-app: every predicted bout (${rows.preds.length.toLocaleString('en-US')})`,
            { luc3d: state.predicted.length, app: rows.preds.length });
        check(!(state.stale && (state.stale.input || state.stale.poses)), 'after sleap-app: the predictions do not read stale (the poses are the same)', state.stale);
        summary.reopenAppStale = state.stale;
    } finally { await b.close(); }
}

function stageDiff() {
    const py = `
import hashlib, json, sys, h5py, numpy as np
def walk(f):
    out = {}
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            h = hashlib.sha1()
            shape = obj.shape
            if shape and shape[0] > 0:
                step = 2000000
                for i in range(0, shape[0], step):
                    a = obj[i:i+step]
                    h.update(a.tobytes() if a.dtype.kind != 'O' else json.dumps([str(x) for x in a]).encode())
            elif not shape:
                v = obj[()]
                h.update(str(v).encode())
            out[name] = dict(shape=list(shape), dtype=str(obj.dtype), compression=obj.compression, sha=h.hexdigest()[:12])
    f.visititems(visit)
    return out
with h5py.File(sys.argv[1], 'r') as a, h5py.File(sys.argv[2], 'r') as b:
    A, B = walk(a), walk(b)
rows = []
for k in sorted(set(A) | set(B)):
    x, y = A.get(k), B.get(k)
    rows.append(dict(name=k, a=x, b=y, same=bool(x and y and x['sha'] == y['sha'] and x['shape'] == y['shape'])))
print(json.dumps(rows))
`;
    const r = spawnSync('uv', ['run', '--quiet', '--with', 'h5py', '--with', 'numpy', 'python', '-I', '-c', py, LUC3D_SLP, APP_SLP], { encoding: 'utf8', maxBuffer: 64 << 20 });
    if (r.status !== 0) { check(false, 'h5py compares the two files', r.stderr.slice(-800)); return; }
    const rows = JSON.parse(r.stdout.trim().split('\n').pop());
    summary.stages.diff = rows;
    saveSummary();
    for (const row of rows) log(`${row.same ? 'same ' : 'DIFF '} ${row.name}: ${row.a ? row.a.shape + ' ' + row.a.dtype + ' ' + row.a.compression : '(absent)'} -> ${row.b ? row.b.shape + ' ' + row.b.dtype + ' ' + row.b.compression : '(absent)'}`);
}

// ----- run -------------------------------------------------------------------------------
const port = await freePort();
const server = spawn('python3', ['-m', 'http.server', String(port)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise((r) => setTimeout(r, 1500));
log(`LUC3D from ${repoRoot} on port ${port}; stages ${STAGES.join(', ')}`);
try {
    for (const name of STAGES) {
        log('=== ' + name);
        try {
            if (name === 'make') await stageMake(port);
            else if (name === 'reopen') await stageReopen(port);
            else if (name === 'python') stagePython();
            else if (name === 'python-app') stagePython(APP_SLP, 'python-app');
            else if (name === 'app') await stageApp();
            else if (name === 'reopen-app') await stageReopenApp(port);
            else if (name === 'reopen-in-app') await stageReopenInApp();
            else if (name === 'diff') stageDiff();
            else throw new Error('unknown stage ' + name);
        } catch (e) {
            check(false, 'stage ' + name, String(e && e.message || e));
            summary.stages[name + ':failed'] = String(e && e.message || e);
            saveSummary();
        }
    }
} finally {
    server.kill();
    summary.fails = fails;
    saveSummary();
    log(fails.length ? `${fails.length} FAILED: ${fails.join(' | ')}` : 'ALL OK');
    log('wrote ' + summaryFile);
}
process.exit(fails.length ? 1 : 0);
