/**
 * _bench-behavior-classifier.mjs — time training and running behavior
 * classifiers on a synthetic hour of 5 animals × 15 nodes at 50 fps (as
 * `_bench-behavior-detect.mjs`), and the heap around them. Two behaviors:
 * "rear" (per animal) and "near" (mutual, per pair), each labelled by its
 * truth over the first 6 minutes (so the row cap samples). An investigation
 * tool, not an assertion.
 *
 * Run: PORT=8747 node tests/e2e/_bench-behavior-classifier.mjs [frames] [animals] [labelledFrames]
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8199);
const FRAMES = Number(process.argv[2] || 180000), ANIMALS = Number(process.argv[3] || 5), LABELLED = Number(process.argv[4] || 18000);
const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1200));
let browser;
try {
    browser = await chromium.launch({ args: ['--enable-precise-memory-info', '--js-flags=--expose-gc'] });
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    page.on('pageerror', e => console.log('pageerror', String(e).slice(0, 300)));
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state, null, { timeout: 30000 });
    const r = await page.evaluate(async ({ T, A, LAB }) => {
        const pd = await import('/pose/pose-data.js');
        const AS = await import('/ui/app-state.js');
        const eth = await import('/pose/behavior-ethogram.js');
        const L = await import('/pose/behavior-labels.js');
        const det = await import('/ui/behavior-detect.js');
        const TR = await import('/ui/behavior-train.js');
        const C = await import('/pose/behavior-classifier.js');
        const P = await import('/pose/behavior-predictions.js');
        const N = 15;
        const names = ['nose', 'ear_l', 'ear_r', 'neck', 'spine1', 'spine2', 'spine3', 'tail_base', 'paw_fl', 'paw_fr', 'paw_hl', 'paw_hr', 'tail1', 'tail2', 'tail_tip'];
        const edges = [[0, 1], [0, 2], [0, 3], [3, 4], [4, 5], [5, 6], [6, 7], [3, 8], [3, 9], [6, 10], [6, 11], [7, 12], [12, 13], [13, 14]];
        const along = [50, 40, 40, 30, 15, 0, -15, -40, 25, 25, -25, -25, -60, -80, -100];
        const side = [0, 8, -8, 0, 0, 0, 0, 0, 10, -10, 10, -10, 0, 0, 0];
        const hgt = [15, 25, 25, 28, 30, 30, 28, 15, 2, 2, 2, 2, 4, 3, 2];
        const s = new pd.Session([], new pd.Skeleton('m', names, edges), [], 'bench');
        const ids = [];
        for (let k = 0; k < A; k++) ids.push(s.addIdentity('id_' + k).id);
        const rearTruth = [], centers = [];
        for (let k = 0; k < A; k++) { rearTruth.push(new Uint8Array(T)); centers.push(new Float32Array(T * 2)); }
        let gid = 0;
        for (let t = 0; t < T; t++) {
            s.instanceGroups.set(t, ids.map((id, k) => {
                const g = new pd.InstanceGroup(gid++, id);
                const ang = 0.3 * t / 50 + k * 1.3, cx = 200 * k + 150 * Math.cos(ang), cy = 150 * Math.sin(ang), hd = ang + Math.PI / 2;
                const rear = (t + 997 * k) % 1500 < 60;
                rearTruth[k][t] = rear ? 1 : 0;
                centers[k][2 * t] = cx; centers[k][2 * t + 1] = cy;
                const p = new Float64Array(N * 3);
                for (let n = 0; n < N; n++) {
                    p[3 * n] = cx + along[n] * Math.cos(hd) - side[n] * Math.sin(hd) + Math.random();
                    p[3 * n + 1] = cy + along[n] * Math.sin(hd) + side[n] * Math.cos(hd) + Math.random();
                    p[3 * n + 2] = (rear && n <= 4 ? hgt[n] + 60 : hgt[n]) + Math.random() * 0.5;
                }
                g.points3d = pd.pooledPoints3d ? pd.pooledPoints3d(p) : p;
                return g;
            }));
        }
        AS.state.sessions = [s]; AS.state.activeSessionIdx = 0; AS.state.session = s;
        AS.state.totalFrames = T; AS.state.currentFrame = 0; AS.state.views = []; AS.state.fps = 50;
        AS.timeline.setTotalFrames(T); AS.timeline.setData(s);
        const rear = eth.addBehavior({ name: 'rear' }).behavior;
        const near = eth.addBehavior({ name: 'near', social: true, mutual: true }).behavior;
        const label = (bid, a, b, truth) => {
            let st = 0;
            for (let t = 1; t <= LAB; t++) {
                if (t === LAB || truth[t] !== truth[st]) { L.labelRange(s, bid, a, b, st, t - 1, !!truth[st]); st = t; }
            }
        };
        for (let k = 0; k < A; k++) label(rear.id, 'id_' + k, '', rearTruth[k]);
        for (let a = 0; a < A; a++) {
            for (let b = a + 1; b < A; b++) {
                const tr = new Uint8Array(T);
                for (let t = 0; t < LAB; t++) tr[t] = Math.hypot(centers[a][2 * t] - centers[b][2 * t], centers[a][2 * t + 1] - centers[b][2 * t + 1]) < 120 ? 1 : 0;
                label(near.id, 'id_' + a, 'id_' + b, tr);
            }
        }
        if (window.gc) window.gc();
        const heap0 = performance.memory.usedJSHeapSize;
        let peak = heap0;
        const iv = setInterval(() => { peak = Math.max(peak, performance.memory.usedJSHeapSize); }, 50);
        const stages = {};
        const t0 = performance.now();
        const sessions = [s];
        const res = await C.trainBehaviorClassifiers({
            behaviorIds: [rear.id, near.id], sources: [{ session: s, name: 'bench', getCtx: () => det.cachedBehaviorContext(s) }],
            allSessions: sessions, onProgress: (p) => { const k = p.stage; stages[k] = stages[k] || [performance.now(), 0]; stages[k][1] = performance.now(); },
        });
        const trainMs = performance.now() - t0;
        res.results.forEach(x => { if (x.model) C.setBehaviorModel(x.model); });
        const peakTrain = peak;
        const t1 = performance.now();
        const t2a = performance.now();
        const again = await C.trainBehaviorClassifiers({ behaviorIds: [rear.id], sources: [{ session: s, name: 'bench', getCtx: () => det.cachedBehaviorContext(s) }],
            allSessions: sessions, params: { trees: 1 }, crossValidate: false });
        const cachedMs = performance.now() - t2a;
        peak = performance.memory.usedJSHeapSize;
        const t2 = performance.now();
        await det.detectAllFrames();
        const predictMs = performance.now() - t2;
        clearInterval(iv);
        if (window.gc) window.gc();
        const heap1 = performance.memory.usedJSHeapSize;
        const t3 = performance.now();
        det.detectThisFrame();
        const frameMs = performance.now() - t3;
        // What the retained heap is: drop the row cache and the models, then the predictions.
        C.resetBehaviorModels();
        if (window.gc) window.gc();
        const heapNoCache = performance.memory.usedJSHeapSize;
        // Held-out accuracy (after the labelled minutes).
        const run = P.behaviorRunOf(s, rear.id);
        let tp = 0, fp = 0, fn = 0;
        for (let k = 0; k < A; k++) {
            const sc = run.scores.get('id_' + k + '\u0001');
            for (let t = LAB; t < T; t++) { const pp = sc[t] >= 0.5; if (pp && rearTruth[k][t]) tp++; else if (pp) fp++; else if (rearTruth[k][t]) fn++; }
        }
        return {
            trainMs: Math.round(trainMs), stages: Object.fromEntries(Object.entries(stages).map(([k, v]) => [k, Math.round(v[1] - v[0])])),
            retrainCachedRowsMs: Math.round(cachedMs), predictMs: Math.round(predictMs), frameMs: Math.round(frameMs),
            heap0: Math.round(heap0 / 1e6), peakTrain: Math.round(peakTrain / 1e6), peakPredict: Math.round(peak / 1e6), heap1: Math.round(heap1 / 1e6),
            heapWithoutRowCacheAndModels: Math.round(heapNoCache / 1e6),
            reports: res.results.map(x => ({ id: x.behaviorId, error: x.error, rows: x.report && x.report.rows, sampled: x.report && x.report.sampled,
                features: x.report && x.report.features, used: x.model && x.model.features.length, nodes: x.model && x.model.forest.feat.length,
                cv: x.report && x.report.cv && { scheme: x.report.cv.scheme, f1: x.report.cv.frame && x.report.cv.frame.f1, bout: x.report.cv.bout && x.report.cv.bout.f1 } })),
            heldOutRearF1: 2 * tp / Math.max(1, 2 * tp + fp + fn),
            status: (document.getElementById('statusText') || {}).textContent, cores: navigator.hardwareConcurrency,
        };
    }, { T: FRAMES, A: ANIMALS, LAB: LABELLED });
    console.log(JSON.stringify(r, null, 1));
} finally {
    if (browser) await browser.close();
    server.kill();
}
