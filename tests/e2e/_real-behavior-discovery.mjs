/**
 * _real-behavior-discovery.mjs — Behavior ▸ Discover Behaviors… on a real recording,
 * as a user would run it: load a per-camera session folder, Track All, Triangulate
 * All at a 25 px reprojection error threshold, detect the seven rule templates (for
 * comparison only), then discover through the real menu, and:
 *   - time the run, stage by stage, with the main thread's JS heap (performance.memory),
 *     the renderer's resident size (workers included), all of Chrome (ps, every 500 ms)
 *     and main-thread long tasks;
 *   - per cluster: its share of the frames, runs, median run, how well it repeats, the
 *     features that stand out, and how the rule detectors' bouts fall in it;
 *   - run again from another seed (SEED2=1, the default) and say how many frames keep
 *     their cluster;
 *   - save the map, and a contact sheet of SHOTS examples per cluster (its longest, a
 *     typical and a short-ish run, at their middle frame, in the view that shows the
 *     animal largest), to judge by eye whether the clusters are behaviors.
 * Writes OUT/<label>-discovery.json, OUT/<label>-discovery-map.png and
 * OUT/<label>-discovery-sheet.png. Never writes into the data folder.
 *
 * Not a test (real dataset, real Chrome for HEVC). Headless unless HEADED=1 (which takes
 * the browser lock). Usage:
 *   DATASET=<folder> NANIMALS=5 LABEL=05mice node tests/e2e/_real-behavior-discovery.mjs
 *   env: PORT=8772  EXCLUDE=<regex of files to leave out>  OUT=<dir> (default verify/behavior-real)
 *        SHOTS=4  TRI=dlt  REPROJ=25  SEED2=1  HEADED=0
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
const PORT = Number(process.env.PORT || 8772);
const EXCLUDE = process.env.EXCLUDE || '(^|/)\\.|/troubleshooting/';
const OUT = process.env.OUT || path.join(repoRoot, 'verify', 'behavior-real');
const SHOTS = Number(process.env.SHOTS ?? 4);
const TRI = process.env.TRI || 'dlt';
const REPROJ = Number(process.env.REPROJ ?? 25);
const SEED2 = (process.env.SEED2 ?? '1') !== '0';
const HEADED = process.env.HEADED === '1';
if (!DATASET) { console.error('set DATASET'); process.exit(2); }
if (path.resolve(OUT).startsWith(path.resolve(DATASET))) { console.error('OUT must be outside the data folder'); process.exit(2); }
fs.mkdirSync(OUT, { recursive: true });
const log = (m) => process.stdout.write(`[${new Date().toISOString().slice(11, 19)}] ${m}\n`);
const mb = (b) => Math.round(b / 1048576);

const profileDir = fs.mkdtempSync(path.join(os.tmpdir(), 'lucid-discovery-real-'));
// ---- process memory (ps): Chrome's tree, found by its profile folder
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
const release = HEADED ? await acquireBrowserLock({ label: '_real-behavior-discovery ' + LABEL }) : async () => {};
const result = { label: LABEL, dataset: DATASET, stages: [], clusters: [], shots: [] };
let context;
try {
    context = await chromium.launchPersistentContext(profileDir, {
        headless: !HEADED, channel: 'chrome', viewport: { width: 1480, height: 860 },
        args: ['--window-size=1500,950', '--enable-precise-memory-info', '--js-flags=--expose-gc'],
    });
    const page = context.pages()[0] || await context.newPage();
    page.on('pageerror', (e) => log('pageerror: ' + String(e).slice(0, 300)));
    page.on('dialog', (d) => d.accept(String(NANIMALS)));
    await answerTrackSetup(page, NANIMALS);
    await page.addInitScript(() => {
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

    // ---- the rule templates, only to compare the clusters with
    await stage('Detect (seven templates, for comparison)', () => page.evaluate(async () => {
        const eth = await import('./pose/behavior-ethogram.js');
        const rules = await import('./pose/behavior-rules.js');
        const det = await import('./ui/behavior-detect.js');
        const tensor = await import('./pose/behavior-tensor.js');
        const st = window.__lucid.state;
        const sk = st.session.skeleton;
        const names = sk.nodes.map((n) => n.name || n);
        const edges = sk.edges.map((e) => (Array.isArray(e) ? [e[0], e[1]] : [e.source, e.destination]));
        const body = tensor.resolveBody(names, edges, eth.featureSettings());
        rules.DETECTOR_TEMPLATES.forEach((t) => {
            const b = eth.behaviorByName(t.name) || eth.addBehavior({ name: t.name, social: t.social, mutual: t.mutual, description: t.description }).behavior;
            eth.setBehaviorDetector(b.id, rules.instantiateTemplate(t, body.roleNames).detector);
        });
        await det.detectAllFrames();
        return { info: { status: document.getElementById('statusText').textContent.slice(0, 200) } };
    }));

    // ---- Discover, through the menu
    const prevStatus = await page.evaluate(() => document.getElementById('statusText').textContent);
    await stage('Discover Behaviors (the real path)', async () => {
        await page.evaluate(() => document.getElementById('menuBehaviorDiscover').click());
        await page.click('#bdiGo');
        await page.waitForFunction((prev) => {
            const s = document.getElementById('statusText').textContent;
            return s !== prev && /Found \d+ clusters?|Discover Behaviors/.test(s) && document.getElementById('loadingOverlay').classList.contains('hidden');
        }, prevStatus, { timeout: 30 * 60000, polling: 500 });
        return { info: { status: await page.evaluate(() => document.getElementById('statusText').textContent.slice(0, 300)) } };
    });
    const summary = await page.evaluate(async () => {
        const D = await import('./pose/behavior-discovery.js');
        const S = await import('./pose/behavior-states.js');
        const P = await import('./pose/behavior-predictions.js');
        const eth = await import('./pose/behavior-ethogram.js');
        const st = window.__lucid.state, s = st.session, fps = st.fps > 0 ? st.fps : 30;
        const res = D.discoveryResult(s, 'Discovery'), layer = S.stateLayer(s, 'Discovery');
        if (!res || !layer) return { none: true, status: document.getElementById('statusText').textContent };
        const sum = S.stateSummary([layer]);
        const T = st.totalFrames;
        // How the rule detectors' bouts fall in each cluster, by animal-frame.
        const animals = Array.from(layer.animals.keys());
        const masks = {};
        for (const b of eth.behaviors()) {
            const m = {};
            animals.forEach((a) => { m[a] = new Uint8Array(T); });
            for (const p of P.predictedBouts(s, { behaviorId: b.id })) {
                for (const who of (b.social ? [p.animal, p.target] : [p.animal])) if (who && m[who]) m[who].fill(1, p.start, p.end + 1);
            }
            masks[b.name] = m;
        }
        const per = new Map();
        let framesWithPose = 0;
        animals.forEach((a) => {
            const runs = layer.animals.get(a);
            for (let i = 0; i + 2 < runs.length; i += 3) {
                let row = per.get(runs[i + 2]);
                if (!row) { row = { frames: 0, rules: {} }; for (const nm of Object.keys(masks)) row.rules[nm] = 0; per.set(runs[i + 2], row); }
                for (let f = runs[i]; f <= runs[i + 1]; f++) {
                    row.frames++;
                    for (const nm of Object.keys(masks)) row.rules[nm] += masks[nm][a][f];
                }
            }
        });
        const base = {};
        for (const nm of Object.keys(masks)) { let c = 0; animals.forEach((a) => { const m = masks[nm][a]; for (let f = 0; f < T; f++) c += m[f]; }); base[nm] = c / (T * animals.length); }
        return {
            report: res.report, fps, T, animals, frames: sum.frames, base,
            clusters: sum.states.map((st2) => {
                const c = res.clusters[st2.state] || {}, row = per.get(st2.state);
                return { state: st2.state, share: st2.share, framesShareOfAll: st2.frames / (T * animals.length), runs: st2.runs, medianS: st2.medianFrames / fps, repeat: c.repeat, episodes: c.episodes, bins: c.bins,
                         top: (c.top || []).map((f) => [f.name, +f.z.toFixed(2)]),
                         rules: Object.fromEntries(Object.entries(row.rules).map(([nm, v]) => [nm, +(v / row.frames).toFixed(3)])) };
            }),
        };
    });
    if (summary.none) {
        log('No clusters: ' + summary.status);
        result.none = summary.status;
        // The same run again, to see what it found at each size.
        result.noneDetail = await page.evaluate(async () => {
            const D = await import('./pose/behavior-discovery.js');
            const det = await import('./ui/behavior-detect.js');
            const s = window.__lucid.state.session;
            const res = await D.discoverBehaviors({ sources: [{ session: s, name: s.name || 'session 1', getCtx: () => det.cachedBehaviorContext(s) }] });
            return { sessionName: s.name, error: res.error, report: res.report, clusters: res.clusters && res.clusters.map((c) => [c.bins, +c.repeat.toFixed(2)]) };
        });
        log('again: ' + JSON.stringify(result.noneDetail));
    }
    else {
        result.report = summary.report; result.clusters = summary.clusters; result.base = summary.base;
        log('report: ' + JSON.stringify(summary.report));
        log(`frames with a cluster: ${(100 * summary.frames / (summary.T * summary.animals.length)).toFixed(1)}%`);
        log('rule detectors overall: ' + Object.entries(summary.base).map(([nm, v]) => nm + ' ' + (100 * v).toFixed(1) + '%').join(' · '));
        for (const c of summary.clusters) {
            const rr = Object.entries(c.rules).sort((a, b) => b[1] - a[1]).slice(0, 4).map(([nm, v]) => nm + ' ' + Math.round(100 * v) + '%').join(', ');
            log(`cluster ${c.state}: ${(100 * c.share).toFixed(1)}% of frames · ${c.runs} runs · median ${c.medianS.toFixed(2)} s · repeats ${c.repeat != null ? c.repeat.toFixed(2) : '?'} · ${c.episodes} stretches | ${rr} | ${c.top.slice(0, 4).map((f) => f[0] + ' ' + f[1]).join(', ')}`);
        }
        // Another seed: how many frames keep their cluster (best match by frames).
        if (SEED2) {
            const prev2 = await page.evaluate(() => document.getElementById('statusText').textContent);
            await stage('Discover again, another seed', async () => {
                await page.evaluate(async () => {
                    const B = await import('./ui/behavior-discovery.js');
                    await B.runDiscovery({ seed: 2, layer: 'Discovery seed 2' });
                });
                return { info: { status: await page.evaluate(() => document.getElementById('statusText').textContent.slice(0, 200)) } };
            });
            result.seed2 = await page.evaluate(async () => {
                const S = await import('./pose/behavior-states.js');
                const s = window.__lucid.state.session, T = window.__lucid.state.totalFrames;
                const a = S.stateLayer(s, 'Discovery'), b = S.stateLayer(s, 'Discovery seed 2');
                if (!a || !b) return null;
                const conf = new Map(), sizeA = new Map(), sizeB = new Map();
                let both = 0;
                a.animals.forEach((_, name) => {
                    for (let f = 0; f < T; f += 2) {
                        const x = S.stateAt(a, name, f), y = S.stateAt(b, name, f);
                        if (x >= 0) sizeA.set(x, (sizeA.get(x) || 0) + 1);
                        if (y >= 0) sizeB.set(y, (sizeB.get(y) || 0) + 1);
                        if (x >= 0 && y >= 0) { const k = x + '>' + y; conf.set(k, (conf.get(k) || 0) + 1); both++; }
                    }
                });
                const best = {};
                sizeA.forEach((n, x) => {
                    let top = 0, at = -1;
                    conf.forEach((v, k) => { const [p, q] = k.split('>').map(Number); if (p === x) { const j = v / (n + sizeB.get(q) - v); if (j > top) { top = j; at = q; } } });
                    best[x] = { jaccard: +top.toFixed(3), to: at };
                });
                S.removeStateLayer(s, 'Discovery seed 2');
                return { clusters: sizeB.size, best };
            });
            log('another seed: ' + JSON.stringify(result.seed2));
        }
        // The map.
        await page.evaluate(async () => {
            const C = await import('./ui/behavior-coding.js');
            const P = await import('./ui/behavior-panel.js');
            C.openBehaviorTab();
            const sec = document.getElementById('behaviorStatesSection');
            if (sec) sec.open = true;
            P.refreshBehaviorPanel();
            document.querySelector('.behavior-discovery-map canvas').scrollIntoView();
        });
        await new Promise((r) => setTimeout(r, 500));
        const mapFile = path.join(OUT, `${LABEL}-discovery-map.png`);
        await (await page.$('#behaviorStatesBody')).screenshot({ path: mapFile }).catch((e) => log('map screenshot: ' + e));
        result.map = mapFile;
    }

    // ---- examples of each cluster, solo view, cropped around the box
    if (SHOTS > 0 && !summary.none) {
        await page.evaluate(async () => { (await import('./ui/ui-wiring.js')).enterSingleViewMode(); });
        const picks = await page.evaluate(async (n) => {
            const S = await import('./pose/behavior-states.js');
            const s = window.__lucid.state.session, L = S.stateLayer(s, 'Discovery');
            const out = [];
            const states = S.stateSummary([L]).states.map((x) => x.state);
            for (const k of states) {
                const list = S.statesToBouts(L, [k]).sort((a, b) => (b.end - b.start) - (a.end - a.start) || a.start - b.start);
                if (!list.length) continue;
                const at = [0, Math.floor(list.length * 0.1), Math.floor(list.length * 0.3), Math.floor(list.length * 0.5)];
                [...new Set(at)].slice(0, n).forEach((i) => out.push(Object.assign({ state: k, rank: i === 0 ? 'longest' : 'p' + Math.round(100 * (1 - i / list.length)) }, list[i])));
            }
            return out;
        }, SHOTS);
        for (const [k, p] of picks.entries()) {
            const mid = Math.floor((p.start + p.end) / 2);
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
                C.setReviewFocus(s, { behaviorId: '', label: 'Cluster ' + p.state, color: '#ffffff', animal: p.animal, target: '', start: p.start, end: p.end }, 'states');
                await init.navigateToFrame(mid);
                await new Promise((r) => setTimeout(r, 900));
                const groups = (s.instanceGroups.get(mid) || []).filter((g) => { const id = s.getIdentity(g.identityId); return id && id.name === p.animal; });
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
                    if (!found || !(box.x1 > box.x0)) continue;
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
            if (!where) { log(`shot cluster ${p.state} ${p.animal} ${p.start}-${p.end}: no view shows the animal`); continue; }
            const bw = where.box.x1 - where.box.x0, bh = where.box.y1 - where.box.y0;
            let w = Math.max(bw * 2.2, 260), h = Math.max(bh * 2.2 + 40, 195);
            if (w / h > 4 / 3) h = w * 3 / 4; else w = h * 4 / 3;
            w = Math.min(w, where.rect.w); h = Math.min(h, where.rect.h);
            let x = (where.box.x0 + where.box.x1) / 2 - w / 2, y = (where.box.y0 + where.box.y1) / 2 - h / 2 - 15;
            x = Math.max(where.rect.x, Math.min(x, where.rect.x + where.rect.w - w));
            y = Math.max(where.rect.y, Math.min(y, where.rect.y + where.rect.h - h));
            const file = path.join(OUT, `${LABEL}-cluster${p.state}-${String(k).padStart(3, '0')}.png`);
            await page.screenshot({ path: file, clip: { x, y, width: w, height: h } });
            result.shots.push({ file, state: p.state, rank: p.rank, animal: p.animal, start: p.start, end: p.end, view: where.view });
            log(`shot cluster ${p.state} (${p.rank}) ${p.animal} frames ${p.start + 1}–${p.end + 1} in ${where.view}`);
        }
        await page.evaluate(async () => {
            (await import('./ui/behavior-coding.js')).setReviewFocus(null);
            (await import('./ui/ui-wiring.js')).setGridMode();
        });
        if (result.shots.length) {
            const rows = new Map();
            for (const s of result.shots) { if (!rows.has(s.state)) rows.set(s.state, []); rows.get(s.state).push(s); }
            const fps = summary.fps;
            const tile = (s) => `<figure><img src="data:image/png;base64,${fs.readFileSync(s.file).toString('base64')}">` +
                `<figcaption>${s.animal} · ${(s.start / fps).toFixed(1)}–${((s.end + 1) / fps).toFixed(1)} s (${((s.end - s.start + 1) / fps).toFixed(2)} s) · ${s.rank} · ${s.view}</figcaption></figure>`;
            const info = (k) => { const c = summary.clusters.find((x) => x.state === k); return c ? `${(100 * c.share).toFixed(1)}% · median ${c.medianS.toFixed(2)} s · repeats ${c.repeat != null ? c.repeat.toFixed(2) : '?'} · ${c.episodes} stretches<br>${c.top.slice(0, 3).map((f) => f[0] + ' ' + f[1]).join('<br>')}` : ''; };
            const html = `<html><body style="margin:0;background:#1a1a1a;color:#ddd;font:12px -apple-system,sans-serif;padding:10px">
                <div style="font-size:14px;margin:0 0 8px">${LABEL}: examples of each cluster</div>
                ${[...rows].map(([t, ss]) => `<div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:8px">
                    <div style="width:170px;padding-top:4px"><b>Cluster ${t}</b><br>${info(t)}</div>${ss.map(tile).join('')}</div>`).join('')}
                <style>figure{margin:0;width:260px}img{width:260px;height:195px;object-fit:contain;background:#000;display:block}
                figcaption{padding:2px 0 0;color:#aaa;font-size:11px}</style></body></html>`;
            const sheet = await context.newPage();
            await sheet.setViewportSize({ width: 1300, height: 400 });
            await sheet.setContent(html);
            const sheetFile = path.join(OUT, `${LABEL}-discovery-sheet.png`);
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
    fs.writeFileSync(path.join(OUT, LABEL + '-discovery.json'), JSON.stringify(result, null, 1));
    if (context) await context.close().catch(() => {});
    await release();
    server.kill();
    try { fs.rmSync(profileDir, { recursive: true, force: true }); } catch (e) { /* left in tmp */ }
    log('wrote ' + path.join(OUT, LABEL + '-discovery.json'));
}
