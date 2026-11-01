/**
 * _bench-behavior-detect.mjs — time Detect Behaviors (every template) on a
 * synthetic hour of 5 animals × 15 nodes at 50 fps, and the heap around it.
 * An investigation tool, not an assertion.
 *
 * Run: PORT=8746 node tests/e2e/_bench-behavior-detect.mjs [frames] [animals]
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8198);
const FRAMES = Number(process.argv[2] || 180000), ANIMALS = Number(process.argv[3] || 5);
const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1200));
let browser;
try {
    browser = await chromium.launch({ args: ['--enable-precise-memory-info', '--js-flags=--expose-gc'] });
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    page.on('pageerror', e => console.log('pageerror', String(e).slice(0, 300)));
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state, null, { timeout: 30000 });
    const r = await page.evaluate(async ({ T, A }) => {
        const pd = await import('/pose/pose-data.js');
        const AS = await import('/ui/app-state.js');
        const eth = await import('/pose/behavior-ethogram.js');
        const rules = await import('/pose/behavior-rules.js');
        const det = await import('/ui/behavior-detect.js');
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
        let gid = 0;
        for (let t = 0; t < T; t++) {
            s.instanceGroups.set(t, ids.map((id, k) => {
                const g = new pd.InstanceGroup(gid++, id);
                const ang = 0.3 * t / 50 + k * 1.3, cx = 300 * k + 150 * Math.cos(ang), cy = 150 * Math.sin(ang), hd = ang + Math.PI / 2;
                const rear = (t + 997 * k) % 1500 < 60;
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
        rules.DETECTOR_TEMPLATES.forEach(t => {
            const b = eth.addBehavior({ name: t.name, social: t.social, mutual: t.mutual }).behavior;
            eth.setBehaviorDetector(b.id, rules.instantiateTemplate(t, { head: 'nose', tailBase: 'tail_base' }).detector);
        });
        if (window.gc) window.gc();
        const heap0 = performance.memory.usedJSHeapSize;
        let peak = heap0;
        const iv = setInterval(() => { peak = Math.max(peak, performance.memory.usedJSHeapSize); }, 50);
        const t0 = performance.now();
        await det.detectAllFrames();
        const ms = performance.now() - t0;
        clearInterval(iv);
        if (window.gc) window.gc();
        const heap1 = performance.memory.usedJSHeapSize;
        const t1 = performance.now();
        det.detectThisFrame();
        const frameMs = performance.now() - t1;
        const counts = {};
        P.predictedBouts(s).forEach(b => { counts[b.behaviorId] = (counts[b.behaviorId] || 0) + 1; });
        return { ms: Math.round(ms), frameMs: Math.round(frameMs), heap0: Math.round(heap0 / 1e6), peak: Math.round(peak / 1e6), heap1: Math.round(heap1 / 1e6),
                 counts, status: (document.getElementById('statusText') || {}).textContent, cores: navigator.hardwareConcurrency };
    }, { T: FRAMES, A: ANIMALS });
    console.log(JSON.stringify(r, null, 1));
} finally {
    if (browser) await browser.close();
    server.kill();
}
