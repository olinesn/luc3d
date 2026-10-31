/**
 * _real-behavior-detect.mjs — behavior detection (Phase 2) on a real
 * recording, as a user would run it: load a per-camera session folder, Track
 * All, Triangulate All, give the ethogram the seven rule templates, then:
 *   - time each stage: reading the poses (the tensor), the body and the
 *     floor, the body lengths, each template on its own (features + scores on
 *     the worker pool), every template together (Detect, the real path),
 *     This frame, the stale check after an edit, the Predictions list;
 *   - sample memory: the main thread's JS heap (performance.memory) and the
 *     resident size of the renderer (main thread + workers), the GPU process
 *     and all of Chrome (ps, every 500 ms);
 *   - count main-thread long tasks per stage (responsiveness);
 *   - count bouts and durations per template and animal;
 *   - visit SHOTS bouts per template — highest, middle and lowest score — in
 *     the view that shows the animal largest, solo, cropped around the review
 *     box drawn on the video, and lay them out as a contact sheet.
 * Writes OUT/<label>.json, OUT/<label>-*.png and OUT/<label>-contact-sheet.png.
 *
 * Not a test (real dataset, real Chrome for HEVC, a display). Takes the
 * browser lock. Usage:
 *   DATASET=<folder> NANIMALS=5 LABEL=05mice node tests/e2e/_real-behavior-detect.mjs
 *   env: PORT=8770  EXCLUDE=<regex of files to leave out>  OUT=<dir> (default verify/behavior-real)
 *        SHOTS=3  TRI=dlt  REPROJ=25 (Tracking Wizard reprojection error threshold, px, for
 *        Triangulate All: a node's view that far off is dropped and the node re-solved, or
 *        dropped from 3D below two good views; 0 = off. Without it, real recordings keep
 *        large tracking errors that read as behavior.)
 */
import { chromium } from 'playwright';
import { spawn, execSync } from 'node:child_process';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { answerTrackSetup } from './fixtures/track-setup.js';
import { acquireBrowserLock } from '../../scripts/browser-lock.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const DATASET = process.env.DATASET;
const NANIMALS = process.env.NANIMALS ?? '';
const LABEL = process.env.LABEL || path.basename(DATASET || 'dataset');
const PORT = Number(process.env.PORT || 8770);
const EXCLUDE = process.env.EXCLUDE || '(^|/)\\.|/troubleshooting/';
const OUT = process.env.OUT || path.join(repoRoot, 'verify', 'behavior-real');
const SHOTS = Number(process.env.SHOTS ?? 3);
const TRI = process.env.TRI || 'dlt';
const REPROJ = Number(process.env.REPROJ ?? 25);
if (!DATASET) { console.error('set DATASET'); process.exit(2); }
fs.mkdirSync(OUT, { recursive: true });
const log = (m) => process.stdout.write(`[${new Date().toISOString().slice(11, 19)}] ${m}\n`);
const mb = (b) => Math.round(b / 1048576);

// ---- process memory (ps): Chrome's tree, found by its profile folder
const profileDir = fs.mkdtempSync(path.join(os.tmpdir(), 'lucid-behavior-real-'));
function chromeMemory() {
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
    const renderers = all.filter((p) => /--type=renderer/.test(p.cmd));
    return {
        renderer: Math.max(0, ...renderers.map((p) => p.rss)),
        gpu: all.filter((p) => /--type=gpu-process/.test(p.cmd)).reduce((s, p) => s + p.rss, 0),
        total: all.reduce((s, p) => s + p.rss, 0),
    };
}
let memStage = null;
const memPeaks = {};
const memTimer = setInterval(() => {
    if (!memStage) return;
    const m = chromeMemory();
    if (!m) return;
    const pk = memPeaks[memStage] || (memPeaks[memStage] = { renderer: 0, gpu: 0, total: 0 });
    pk.renderer = Math.max(pk.renderer, m.renderer); pk.gpu = Math.max(pk.gpu, m.gpu); pk.total = Math.max(pk.total, m.total);
}, 500);

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise((r) => setTimeout(r, 1500));
const release = await acquireBrowserLock({ label: '_real-behavior-detect ' + LABEL });
const result = { label: LABEL, dataset: DATASET, stages: [], templates: [], bouts: {}, shots: [] };
let context;
try {
    context = await chromium.launchPersistentContext(profileDir, {
        headless: false, channel: 'chrome', viewport: { width: 1480, height: 860 },
        args: ['--window-size=1500,950', '--enable-precise-memory-info', '--js-flags=--expose-gc'],
    });
    const page = context.pages()[0] || await context.newPage();
    page.on('pageerror', (e) => log('pageerror: ' + String(e).slice(0, 300)));
    page.on('dialog', (d) => d.accept(String(NANIMALS)));
    await answerTrackSetup(page, NANIMALS);
    await page.addInitScript(() => {
        // Behavior workers spawned and jobs posted; long tasks; the JS heap's peak.
        window.__bw = { spawned: 0, jobs: 0 };
        const Real = window.Worker;
        window.Worker = function (url, opts) {
            const w = new Real(url, opts);
            if (/behavior-worker\.js/.test(String(url))) {
                window.__bw.spawned++;
                const post = w.postMessage.bind(w);
                w.postMessage = (m, t) => { if (m && m.type === 'job') window.__bw.jobs++; return post(m, t); };
            }
            return w;
        };
        window.Worker.prototype = Real.prototype;
        window.__lt = [];
        try {
            new PerformanceObserver((l) => { for (const e of l.getEntries()) window.__lt.push([e.startTime, e.duration]); })
                .observe({ type: 'longtask', buffered: true });
        } catch (e) { /* no long-task timing */ }
        window.__heap = { peak: 0 };
        setInterval(() => { if (performance.memory) window.__heap.peak = Math.max(window.__heap.peak, performance.memory.usedJSHeapSize); }, 100);
    });
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state, null, { timeout: 60000 });
    await page.evaluate(async (reproj) => {
        const st = await import('./ui/settings.js');
        st.setTrackingThresholds({ autoSwitchCheck: 0, autoImageSwitchCheck: 0, reprojErrorThreshold: reproj });
        const inp = document.createElement('input');
        inp.type = 'file'; inp.multiple = true; inp.webkitdirectory = true; inp.id = '__dir';
        inp.style.cssText = 'position:fixed;left:-9999px'; document.body.appendChild(inp);
        setInterval(() => { for (const b of document.querySelectorAll('button')) { const t = b.textContent.trim(); if ((t === 'Continue' || t === 'OK') && b.offsetParent !== null) b.click(); } }, 300);
    }, REPROJ);
    log('Triangulate All with a reprojection error threshold of ' + REPROJ + ' px' + (REPROJ > 0 ? '' : ' (off)'));

    /** One measured stage: wall time, heap peak, long tasks, Chrome's memory. */
    const stage = async (name, fn) => {
        const t0p = await page.evaluate(() => { if (window.gc) window.gc(); window.__heap.peak = performance.memory.usedJSHeapSize; return performance.now(); });
        memStage = name;
        const m0 = chromeMemory();
        memPeaks[name] = { renderer: m0 ? m0.renderer : 0, gpu: m0 ? m0.gpu : 0, total: m0 ? m0.total : 0 };
        const t0 = Date.now();
        const out = await fn();
        const ms = Date.now() - t0;
        const m1 = chromeMemory();
        if (m1) { const pk = memPeaks[name]; pk.renderer = Math.max(pk.renderer, m1.renderer); pk.gpu = Math.max(pk.gpu, m1.gpu); pk.total = Math.max(pk.total, m1.total); }
        memStage = null;
        const after = await page.evaluate((t0p) => {
            window.__heap.peak = Math.max(window.__heap.peak, performance.memory.usedJSHeapSize);
            const lt = window.__lt.filter((e) => e[0] >= t0p);
            return { heapPeak: window.__heap.peak, heapEnd: performance.memory.usedJSHeapSize,
                     longTasks: lt.length, longTaskMs: Math.round(lt.reduce((s, e) => s + e[1], 0)), longestTaskMs: Math.round(lt.reduce((m, e) => Math.max(m, e[1]), 0)) };
        }, t0p);
        const pk = memPeaks[name];
        const row = { name, ms, heapPeakMB: mb(after.heapPeak), heapEndMB: mb(after.heapEnd), rendererPeakMB: mb(pk.renderer),
                      gpuPeakMB: mb(pk.gpu), chromePeakMB: mb(pk.total), longTasks: after.longTasks, longTaskMs: after.longTaskMs,
                      longestTaskMs: after.longestTaskMs, info: out && out.info };
        result.stages.push(row);
        log(`${name}: ${(ms / 1000).toFixed(2)} s · heap peak ${row.heapPeakMB} MB · renderer ${row.rendererPeakMB} MB · Chrome ${row.chromePeakMB} MB · ` +
            `long tasks ${row.longTasks} (${row.longTaskMs} ms, longest ${row.longestTaskMs} ms)` + (row.info ? ' · ' + JSON.stringify(row.info).slice(0, 300) : ''));
        return out;
    };
    const clickStep = async (button, re, what) => {
        const prev = await page.evaluate(() => document.getElementById('statusText').textContent);
        await page.evaluate((id) => document.getElementById(id).click(), button);
        const t0 = Date.now();
        for (;;) {
            const st = await page.evaluate(() => ({ s: document.getElementById('statusText').textContent, hidden: document.getElementById('loadingOverlay').classList.contains('hidden') }));
            if (st.hidden && st.s !== prev && re.test(st.s)) return { info: { status: st.s.slice(0, 160) } };
            if (st.hidden && /error|failed/i.test(st.s) && st.s !== prev) throw new Error(what + ': ' + st.s);
            if (Date.now() - t0 > 60 * 60000) throw new Error(what + ' timed out');
            await new Promise((r) => setTimeout(r, 250));
        }
    };

    // ---- load, Track All, Triangulate All
    await stage('load', async () => {
        await page.setInputFiles('#__dir', DATASET);
        await page.evaluate(async (ex) => {
            const sl = await import('./loading/session-loader.js');
            const re = new RegExp(ex);
            await sl.handleLoadSessionFolderPerCamera(Array.from(document.getElementById('__dir').files).filter((f) => !re.test(f.webkitRelativePath || f.name)), false);
        }, EXCLUDE);
        await page.waitForFunction(() => window.__lucid.state.session && window.__lucid.state.session.cameras && window.__lucid.state.session.cameras.length > 1 &&
            document.getElementById('loadingOverlay').classList.contains('hidden'), null, { timeout: 60 * 60000, polling: 1000 });
        await new Promise((r) => setTimeout(r, 3000));
        return { info: await page.evaluate(() => { const s = window.__lucid.state.session; const st = window.__lucid.state;
            return { cams: s.cameras.length, frames: st.totalFrames, fps: st.fps, lazy: !!s.lazyLoader, nodes: s.skeleton.nodes.length }; }) };
    });
    await stage('Track All', () => clickStep('tbTrackAll', /Assigned \d+ identities/, 'Track All'));
    // Track All's summary dialog would cover the videos.
    await page.waitForSelector('#trackSummaryClose', { timeout: 10000 }).then(() => page.click('#trackSummaryClose')).catch(() => {});
    await page.evaluate(async (m) => { (await import('./ui/settings.js')).setDefaultTriangulationMethod(m); }, TRI);
    await stage('Triangulate All', () => clickStep('tbTriangulateAll', /^(Triangulated|Grouped) /, 'Triangulate All'));

    // ---- behavior: the stages one by one
    await stage('tensor (read the poses)', () => page.evaluate(async () => {
        const T = await import('./pose/behavior-tensor.js');
        const st = window.__lucid.state;
        const input = T.collectBehaviorInput(st.session, { numFrames: st.totalFrames });
        window.__bi = { input };
        return { info: { mode: input.mode, animals: input.animals, frames: input.frames, skipped: input.skipped, numFrames: input.numFrames,
                         nodes: input.nodeNames.length, MB: Math.round(input.data.reduce((s, a) => s + a.byteLength, 0) / 1048576) } };
    }));
    await stage('body + floor', () => page.evaluate(async () => {
        const T = await import('./pose/behavior-tensor.js');
        const det = await import('./ui/behavior-detect.js');
        const eth = await import('./pose/behavior-ethogram.js');
        const input = window.__bi.input;
        const body = T.resolveBody(input.nodeNames, input.edges, eth.featureSettings());
        const floorInfo = det.resolveFloor(input, body, eth.featureSettings());
        Object.assign(window.__bi, { body, floorInfo });
        const f = floorInfo.floor;
        return { info: { head: body.roleNames.head, tailBase: body.roleNames.tailBase, excluded: body.excluded.map((i) => input.nodeNames[i]),
                         floor: floorInfo.label, warning: floorInfo.warning, up: f && f.up.map((v) => +v.toFixed(3)), confidence: f && f.confidence } };
    }));
    await stage('body lengths (spec)', () => page.evaluate(async () => {
        const T = await import('./pose/behavior-tensor.js');
        const det = await import('./ui/behavior-detect.js');
        const b = window.__bi;
        b.spec = T.featureSpec(b.input, b.body, b.floorInfo.floor, det.detectionFps());
        return { info: { fps: b.spec.fps, bodyLength: b.spec.bodyLength.map((v) => +v.toFixed(1)) } };
    }));
    await stage('broken tracking check', () => page.evaluate(async () => {
        const T = await import('./pose/behavior-tensor.js');
        const b = window.__bi;
        b.broken = T.brokenPoseFrames(b.input, b.body, b.spec.bodyLength, b.spec.fps);
        const tracked = b.input.frames.reduce((s, x) => s + x, 0);
        return { info: { skipped: b.broken.total, pctOfTracked: +(100 * b.broken.total / Math.max(1, tracked)).toFixed(2),
                         nodeOffBody: b.broken.far.reduce((s, x) => s + x, 0), jumps: b.broken.jumps.reduce((s, x) => s + x, 0) } };
    }));
    await page.evaluate(async () => {
        const eth = await import('./pose/behavior-ethogram.js');
        const rules = await import('./pose/behavior-rules.js');
        const roleNames = window.__bi.body.roleNames;
        rules.DETECTOR_TEMPLATES.forEach((t) => {
            const b = eth.behaviorByName(t.name) || eth.addBehavior({ name: t.name, social: t.social, mutual: t.mutual, description: t.description }).behavior;
            eth.setBehaviorDetector(b.id, rules.instantiateTemplate(t, roleNames).detector);
        });
        (await import('./ui/behavior-coding.js')).requestBehaviorRepaint({ ethogram: true });
    });
    const templates = await page.evaluate(async () => (await import('./pose/behavior-ethogram.js')).behaviors().map((b) => ({ id: b.id, name: b.name })));
    for (const t of templates) {
        await stage('template: ' + t.name, () => page.evaluate(async (id) => {
            const F = await import('./pose/behavior-features.js');
            const R = await import('./pose/behavior-rules.js');
            const pool = await import('./pose/behavior-pool.js');
            const eth = await import('./pose/behavior-ethogram.js');
            const LO = await import('./ui/loading-overlay.js');
            const b = eth.behaviorById(id), x = window.__bi, input = x.input;
            const catalog = F.catalogFor(x.spec);
            const plan = R.planDetectors(catalog, [{ behaviorId: b.id, social: b.social, mutual: b.mutual, point: b.kind === 'point', detector: b.detector }]);
            if (!plan.plans.length) return { info: { skipped: plan.skipped } };
            const T = input.numFrames, N = input.nodeNames.length, D = input.dim;
            const chunks = F.planChunks(T, F.featurePadFrames(x.spec), 8192);
            const whole = {};
            const jobs0 = window.__bw.jobs;
            const r = await pool.runBehaviorChunks(chunks.length, (i) => ({ kind: 'scores', spec: x.spec, chunk: chunks[i],
                data: F.sliceChunkData(input.data, chunks[i], N * D, x.broken.masks), plan: { request: plan.request, plans: plan.plans } }),
            (i, res) => {
                const sc = res.scores[b.id];
                for (const k of Object.keys(sc)) { if (!whole[k]) whole[k] = new Float32Array(T).fill(NaN); whole[k].set(sc[k], chunks[i].start); }
            }, { workload: T * input.animals.length, pause: LO.yieldToPaint });
            let bouts = 0, frames = 0;
            for (const k of Object.keys(whole)) for (const bt of R.detectorBouts(whole[k], b.detector, x.spec.fps, b.kind === 'point')) { bouts++; frames += bt[1] - bt[0] + 1; }
            return { info: { features: plan.request.animal.concat(plan.request.pair), contexts: Object.keys(whole).length, bouts,
                             seconds: +(frames / x.spec.fps).toFixed(1), parallel: r.parallel, workers: r.workers, jobs: window.__bw.jobs - jobs0, chunks: chunks.length } };
        }, t.id));
    }
    // Where each template's features sit on this recording: percentiles over
    // every animal (or pair) and frame, to judge the default thresholds.
    result.percentiles = (await stage('feature percentiles', () => page.evaluate(async () => {
        const F = await import('./pose/behavior-features.js');
        const det = await import('./ui/behavior-detect.js');
        const eth = await import('./pose/behavior-ethogram.js');
        const x = window.__bi;
        const ctx = { input: x.input, spec: x.spec, catalog: F.catalogFor(x.spec), broken: x.broken };
        const names = [];
        for (const b of eth.behaviors()) for (const c of (b.detector ? b.detector.conditions : [])) if (!names.includes(c.feature)) names.push(c.feature);
        const cols = await det.computeFeatureColumns(ctx, names);
        const out = {};
        for (const nm of names) {
            const plain = nm.indexOf(F.TARGET_PREFIX) === 0 ? nm.slice(F.TARGET_PREFIX.length) : nm;
            const arrs = cols.animal[plain] || (cols.pair[plain] ? Array.from(cols.pair[plain].values()) : []);
            let n = 0;
            for (const a of arrs) for (let i = 0; i < a.length; i++) if (a[i] === a[i]) n++;
            const all = new Float32Array(n);
            let k = 0;
            for (const a of arrs) for (let i = 0; i < a.length; i++) if (a[i] === a[i]) all[k++] = a[i];
            all.sort();
            const q = (f) => n ? +all[Math.min(n - 1, Math.floor(f * (n - 1)))].toFixed(3) : null;
            out[nm] = { n, p1: q(0.01), p5: q(0.05), p25: q(0.25), p50: q(0.5), p75: q(0.75), p95: q(0.95), p99: q(0.99) };
        }
        return { info: out };
    }))).info;
    for (const [nm, v] of Object.entries(result.percentiles || {})) {
        log(`percentiles · ${nm}: p1 ${v.p1} · p5 ${v.p5} · p25 ${v.p25} · p50 ${v.p50} · p75 ${v.p75} · p95 ${v.p95} · p99 ${v.p99} (${v.n} values)`);
    }
    await page.evaluate(() => { window.__bi = null; });
    const detect = await stage('Detect (all templates, the real path)', () => page.evaluate(async () => {
        const det = await import('./ui/behavior-detect.js');
        await det.detectAllFrames();
        return { info: { status: document.getElementById('statusText').textContent.slice(0, 200), workersSpawned: window.__bw.spawned } };
    }));
    result.workers = detect.info.workersSpawned;
    await stage('This frame', () => page.evaluate(async () => {
        const det = await import('./ui/behavior-detect.js');
        const r = det.detectThisFrame();
        return { info: { rows: r ? r.rows.length : 0, detected: r ? r.rows.filter((x) => x.detected).length : 0 } };
    }));
    await stage('stale check after an edit (sliced)', () => page.evaluate(async () => {
        const P = await import('./pose/behavior-predictions.js');
        const SL = await import('./import-export/save-load.js');
        const s = window.__lucid.state.session;
        SL.markDirty();
        const t0 = performance.now();
        while (P.predictionCheckPending(s) && performance.now() - t0 < 120000) await new Promise((r) => setTimeout(r, 50));
        return { info: { checkedMs: Math.round(performance.now() - t0), stale: P.predictionStaleness(s, (await import('./ui/behavior-detect.js')).detectionEnv()) } };
    }));
    await stage('Predictions list (render)', () => page.evaluate(async () => {
        const C = await import('./ui/behavior-coding.js');
        C.openBehaviorTab();
        const sec = document.getElementById('behaviorPredictionsSection');
        if (sec) sec.open = true;
        const P = (await import('./ui/behavior-panel.js'));
        const t0 = performance.now();
        P.refreshBehaviorPanel();
        return { info: { renderMs: Math.round(performance.now() - t0), rows: document.querySelectorAll('.behavior-pred').length } };
    }));

    // ---- bouts per template and animal
    result.bouts = await page.evaluate(async () => {
        const P = await import('./pose/behavior-predictions.js');
        const eth = await import('./pose/behavior-ethogram.js');
        const st = window.__lucid.state, fps = st.fps > 0 ? st.fps : 30;
        const out = {};
        for (const b of eth.behaviors()) {
            const list = P.predictedBouts(st.session, { behaviorId: b.id });
            const by = {};
            for (const p of list) {
                const who = p.animal + (p.target ? (b.mutual ? ' ↔ ' : ' → ') + p.target : '');
                (by[who] = by[who] || []).push((p.end - p.start + 1) / fps);
            }
            const stats = (d) => { const s = d.slice().sort((a, c) => a - c); return { bouts: s.length, totalS: +s.reduce((a, c) => a + c, 0).toFixed(1),
                medianS: +s[Math.floor(s.length / 2)].toFixed(2), maxS: +s[s.length - 1].toFixed(2) }; };
            // How much of each bout its detected frames cover (gaps joined in count against it).
            let sparse = 0;
            for (const p of list) {
                const sc = P.predictionScoreAt;
                let on = 0;
                for (let f = p.start; f <= p.end; f++) if (sc(st.session, b.id, p.animal, p.target, f) >= 0.5) on++;
                if (on * 2 < p.end - p.start + 1) sparse++;
            }
            out[b.name] = { total: list.length, sparse, perAnimal: Object.fromEntries(Object.entries(by).map(([k, d]) => [k, stats(d)])) };
        }
        return out;
    });
    for (const [name, v] of Object.entries(result.bouts)) {
        log(`bouts · ${name}: ${v.total} (${v.sparse} less than half detected)` + Object.entries(v.perAnimal).map(([k, s]) => `\n      ${k}: ${s.bouts} bouts, ${s.totalS} s (median ${s.medianS} s, longest ${s.maxS} s)`).join(''));
    }

    // ---- screenshots: SHOTS bouts per template, solo view, cropped around the review box
    if (SHOTS > 0) {
        await page.evaluate(async () => { (await import('./ui/ui-wiring.js')).enterSingleViewMode(); });
        const picks = await page.evaluate(async (n) => {
            const P = await import('./pose/behavior-predictions.js');
            const eth = await import('./pose/behavior-ethogram.js');
            const s = window.__lucid.state.session;
            const out = [];
            for (const b of eth.behaviors()) {
                const list = P.predictedBouts(s, { behaviorId: b.id }).sort((a, c) => (c.score || 0) - (a.score || 0));
                if (!list.length) continue;
                const idx = [...new Set([0, Math.floor((list.length - 1) / 2), list.length - 1])].slice(0, n);
                for (const i of idx) out.push(Object.assign({ name: b.name, rank: i === 0 ? 'highest' : i === list.length - 1 ? 'lowest' : 'middle' }, list[i]));
            }
            return out;
        }, SHOTS);
        for (const [k, p] of picks.entries()) {
            const mid = Math.floor((p.start + p.end) / 2);
            // Nothing may cover the video: no hover preview (the mouse rests on
            // the info panel), and no dialog left open.
            await page.mouse.move(1440, 600);
            await page.evaluate(() => {
                const c = document.getElementById('trackSummaryClose');
                if (c) c.click();
                document.querySelectorAll('.multi-frame-modal-overlay, .plane-confirm-overlay').forEach((o) => o.remove());
            });
            const where = await page.evaluate(async ({ p, mid }) => {
                const C = await import('./ui/behavior-coding.js');
                const init = await import('./pose/initialization.js');
                const W = await import('./ui/ui-wiring.js');
                const st = window.__lucid.state, s = st.session;
                C.setReviewFocus(s, p);
                await init.navigateToFrame(mid);
                await new Promise((r) => setTimeout(r, 900));
                // The view that shows the animals largest: their 2D detection and reprojection.
                const names = p.target ? [p.animal, p.target] : [p.animal];
                const groups = (s.instanceGroups.get(mid) || []).filter((g) => { const id = s.getIdentity(g.identityId); return id && names.includes(id.name); });
                let best = null;
                for (const v of st.views) {
                    const box = { x0: Infinity, y0: Infinity, x1: -Infinity, y1: -Infinity };
                    let found = 0;
                    for (const g of groups) {
                        let any = false;
                        for (const inst of [g.getInstance(v.name), g.getReprojectedInstance && g.getReprojectedInstance(v.name)]) {
                            if (!inst || !inst._xy) continue;
                            for (let i = 0; i < inst._xy.length; i += 2) {
                                const x = inst._xy[i], y = inst._xy[i + 1];
                                if (!(x === x) || !(y === y)) continue;
                                box.x0 = Math.min(box.x0, x); box.x1 = Math.max(box.x1, x); box.y0 = Math.min(box.y0, y); box.y1 = Math.max(box.y1, y);
                                any = true;
                            }
                        }
                        if (any) found++;
                    }
                    if (found < names.length || !(box.x1 > box.x0)) continue;
                    const vw = v.videoWidth || 1, vh = v.videoHeight || 1;
                    if (box.x0 < 0 || box.y0 < 0 || box.x1 > vw || box.y1 > vh) continue;
                    const area = (box.x1 - box.x0) * (box.y1 - box.y0) / (vw * vh);
                    if (!best || area > best.area) best = { view: v.name, box, area, vw, vh };
                }
                if (!best) return null;
                W.setSoloView(best.view);
                await new Promise((r) => setTimeout(r, 1500));
                const v = st.views.find((x) => x.name === best.view);
                const cv = v.overlayCanvas || v.canvas;
                const rect = cv.getBoundingClientRect();
                const O = await import('./ui/overlays.js');
                const tf = O.makeVideoToCanvasTransform(best.vw, best.vh, cv.width, cv.height);
                const a = tf(best.box.x0, best.box.y0), b = tf(best.box.x1, best.box.y1);
                const sx = rect.width / cv.width, sy = rect.height / cv.height;
                return { view: best.view, rect: { x: rect.left, y: rect.top, w: rect.width, h: rect.height },
                         box: { x0: rect.left + a.x * sx, y0: rect.top + a.y * sy, x1: rect.left + b.x * sx, y1: rect.top + b.y * sy } };
            }, { p, mid });
            if (!where) { log(`shot ${p.name} ${p.animal} ${p.start}-${p.end}: no view shows the animal`); continue; }
            // A 4:3 crop around the box, with room for the label above it.
            const bw = where.box.x1 - where.box.x0, bh = where.box.y1 - where.box.y0;
            let w = Math.max(bw * 2.2, 260), h = Math.max(bh * 2.2 + 40, 195);
            if (w / h > 4 / 3) h = w * 3 / 4; else w = h * 4 / 3;
            w = Math.min(w, where.rect.w); h = Math.min(h, where.rect.h);
            let x = (where.box.x0 + where.box.x1) / 2 - w / 2, y = (where.box.y0 + where.box.y1) / 2 - h / 2 - 15;
            x = Math.max(where.rect.x, Math.min(x, where.rect.x + where.rect.w - w));
            y = Math.max(where.rect.y, Math.min(y, where.rect.y + where.rect.h - h));
            const file = path.join(OUT, `${LABEL}-${String(k).padStart(2, '0')}-${p.name.replace(/\W+/g, '_')}.png`);
            await page.screenshot({ path: file, clip: { x, y, width: w, height: h } });
            result.shots.push({ file, template: p.name, rank: p.rank, animal: p.animal, target: p.target, start: p.start, end: p.end,
                                score: +(p.score || 0).toFixed(2), view: where.view });
            log(`shot ${p.name} (${p.rank}) ${p.animal}${p.target ? '/' + p.target : ''} frames ${p.start + 1}–${p.end + 1} score ${(p.score || 0).toFixed(2)} in ${where.view}`);
        }
        await page.evaluate(async () => {
            (await import('./ui/behavior-coding.js')).setReviewFocus(null);
            (await import('./ui/ui-wiring.js')).setGridMode();
        });
        // The contact sheet: one row per template.
        if (result.shots.length) {
            const rows = new Map();
            for (const s of result.shots) { if (!rows.has(s.template)) rows.set(s.template, []); rows.get(s.template).push(s); }
            const fps = await page.evaluate(() => window.__lucid.state.fps || 30);
            const tile = (s) => `<figure><img src="data:image/png;base64,${fs.readFileSync(s.file).toString('base64')}">` +
                `<figcaption>${s.animal}${s.target ? ' / ' + s.target : ''} · ${(s.start / fps).toFixed(1)}–${((s.end + 1) / fps).toFixed(1)} s · ` +
                `score ${Math.round(s.score * 100)}% (${s.rank}) · ${s.view}</figcaption></figure>`;
            const html = `<html><body style="margin:0;background:#1a1a1a;color:#ddd;font:12px -apple-system,sans-serif;padding:10px">
                <div style="font-size:14px;margin:0 0 8px">${LABEL}: detected bouts (dashed box = the bout under review)</div>
                ${[...rows].map(([t, ss]) => `<div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:8px">
                    <div style="width:110px;font-weight:600;padding-top:4px">${t}</div>${ss.map(tile).join('')}</div>`).join('')}
                <style>figure{margin:0;width:300px}img{width:300px;height:225px;object-fit:contain;background:#000;display:block}
                figcaption{padding:2px 0 0;color:#aaa;font-size:11px}</style></body></html>`;
            const sheet = await context.newPage();
            await sheet.setViewportSize({ width: 1100, height: 400 });
            await sheet.setContent(html);
            const sheetFile = path.join(OUT, `${LABEL}-contact-sheet.png`);
            await sheet.screenshot({ path: sheetFile, fullPage: true });
            await sheet.close();
            result.contactSheet = sheetFile;
            log('contact sheet: ' + sheetFile);
        }
    }
} catch (e) {
    log('FAILED: ' + (e && e.stack || e));
    result.error = String(e && e.message || e);
} finally {
    clearInterval(memTimer);
    fs.writeFileSync(path.join(OUT, LABEL + '.json'), JSON.stringify(result, null, 1));
    if (context) await context.close().catch(() => {});
    await release();
    server.kill();
    try { fs.rmSync(profileDir, { recursive: true, force: true }); } catch (e) { /* left in tmp */ }
    log('wrote ' + path.join(OUT, LABEL + '.json'));
}
