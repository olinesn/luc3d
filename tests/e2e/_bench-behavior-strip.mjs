/**
 * _bench-behavior-strip.mjs — what the bouts strip above the seekbar costs while
 * a video plays. Investigation tool, not an assertion.
 *
 * Plays through the app's real playback loop (`VideoController.startPlayback`,
 * real overlays, seekbar, Timeline and info panel) on 8 views at 60 fps, with the
 * strip off and on, both without and with a state running, alternating so a slow
 * moment of the machine hits them all. The decoders are fakes (a colour fill per
 * view, the frame from the clock), because video decoding is the same either way
 * and this asks only what the strip adds; run headless, so nothing fights for the
 * screen.
 *
 * For each pass, from a Chrome trace of the renderer's main thread: milliseconds
 * per second in script, style, layout, pre-paint, paint and layerize, with how
 * many layouts and paints ran, plus raster work on the other threads, and the
 * display-frame intervals. Then two one-off costs: drawing the film (once per
 * window of playback), and a new lane appearing during playback (the views get
 * shorter).
 *
 * Headless has no real display, so frame intervals here say little about
 * smoothness on a screen: for that run _bench-playback.mjs (headed, real data).
 *
 *   SECONDS=8 ROUNDS=2 CAMS=8 node tests/e2e/_bench-behavior-strip.mjs
 *   TIMELINE=closed       play with the Timeline collapsed
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8774);
const SECONDS = Number(process.env.SECONDS || 8);
const ROUNDS = Number(process.env.ROUNDS || 2);
const CAMS = Number(process.env.CAMS || 8);
const N = 216000;       // an hour at 60 fps

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1200));

const fmt = (x, d = 2) => x.toFixed(d).padStart(8);
const quantile = (a, q) => { const s = a.slice().sort((p, q2) => p - q2); return s.length ? s[Math.min(s.length - 1, Math.floor(q * s.length))] : 0; };

let browser;
try {
    browser = await chromium.launch();
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    page.on('pageerror', e => console.log('  [pageerror]', String(e).slice(0, 300)));
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state, null, { timeout: 30000 });

    await page.evaluate(async ({ CAMS, N }) => {
        const pd = await import('/pose/pose-data.js');
        const AS = await import('/ui/app-state.js');
        const sp = await import('/ui/sessions-panes.js');
        const K = [[600, 0, 320], [0, 600, 240], [0, 0, 1]];
        const cams = [];
        for (let i = 0; i < CAMS; i++) cams.push(new pd.Camera('cam' + i, K, [0, 0, 0, 0, 0], [0, 0.1 * i, 0], [20 * i, 0, 0], [640, 480]));
        const session = new pd.Session(cams, new pd.Skeleton('sk', ['a', 'b'], [[0, 1]]), [], 'Bench');
        session.addIdentity('id_0'); session.addIdentity('id_1');
        const fake = (vi) => {
            const src = document.createElement('canvas');
            src.width = 64; src.height = 48;
            const d = {
                samples: new Array(N), _fps: 60, _videoReady: true, playing: false, pos: 0, t0: 0,
                async getFrame() { return createImageBitmap(src); },
                drawCurrentFrame(ctx, w, h) { ctx.fillStyle = `rgb(${d.getCurrentFrameIndex() % 256},${30 * vi},7)`; ctx.fillRect(0, 0, w, h); return true; },
                getCurrentFrameIndex() { return d.playing ? Math.min(N - 1, d.pos + Math.floor((performance.now() - d.t0) * 60 / 1000)) : d.pos; },
                seekNativeSettled(f) { d.pos = f; return Promise.resolve(); },
                playNative() { d.playing = true; d.t0 = performance.now(); },
                pauseNative() { if (d.playing) d.pos = d.getCurrentFrameIndex(); d.playing = false; },
                releaseFrames() {},
            };
            return d;
        };
        AS.state.sessions = [session]; AS.state.session = session; AS.state.activeSessionIdx = 0;
        AS.state.totalFrames = N; AS.state.fps = 60; AS.state.currentFrame = 0;
        AS.state.views = cams.map((c, i) => ({ name: c.name, videoWidth: 640, videoHeight: 480, canvas: null, decoder: fake(i) }));
        AS.state.videoFiles = cams.map(c => ({ name: c.name, assignedCamera: c.name }));
        AS.state.viewMode = 'grid'; AS.state.singleViewIndex = 0;
        AS.paneManager.clearAll();
        sp.populateViewStrip();
        AS.paneManager.addAllViewsAsGrid();
        const eth = await import('/pose/behavior-ethogram.js');
        eth.addBehavior({ name: 'attack', key: 'a', social: true });
        eth.addBehavior({ name: 'tail rattle', key: 't' });
        eth.addBehavior({ name: 'sniff', key: 's', social: true, mutual: true });
        eth.addBehavior({ name: 'rear', key: 'r' });
        const L = await import('/pose/behavior-labels.js');
        // A bout every 3 s or so, through the stretch the passes play.
        const bouts = [];
        for (let f = 0; f < 60000; f += 90 + (f % 37)) {
            const pick = (f / 90 | 0) % 4;
            if (pick === 0) bouts.push({ behaviorId: 'b1', animal: 'id_0', target: 'id_1', start: f, end: f + 40, present: true });
            else if (pick === 1) bouts.push({ behaviorId: 'b2', animal: 'id_1', start: f, end: f + 30, present: true });
            else if (pick === 2) bouts.push({ behaviorId: 'b3', animal: 'id_0', target: 'id_1', start: f, end: f + 50, present: true });
            else bouts.push({ behaviorId: 'b1', animal: 'id_0', target: 'id_1', start: f, end: f + 25, present: false });
        }
        L.labelBouts(session, bouts);
        AS.timeline.setTotalFrames(N);
        AS.timeline.setData(session);
        (await import('/ui/info-panel.js')).updateInfoPanel();
        (await import('/ui/behavior-coding.js')).requestBehaviorRepaint({ labels: true, ethogram: true });
    }, { CAMS, N });
    await page.waitForTimeout(500);
    if (process.env.TIMELINE === 'closed') {
        await page.click('#timelineToggleBtn');
        await page.waitForTimeout(300);
    }
    const nBouts = await page.evaluate(async () => (await import('/pose/behavior-labels.js')).listBouts(window.__lucid.state.session).length);
    console.log(`${CAMS} views at 60 fps, ${nBouts} bouts, 2 animals, ${SECONDS} s a pass, Timeline ${process.env.TIMELINE === 'closed' ? 'closed' : 'open'}`);

    const WATCH = ['FunctionCall', 'UpdateLayoutTree', 'Layout', 'PrePaint', 'Paint', 'Layerize', 'Commit'];

    /** One pass: play from `start`, tracing. */
    const pass = async (mode, start) => {
        await page.evaluate(async ({ mode, start }) => {
            const AS = await import('/ui/app-state.js');
            const S = await import('/ui/behavior-strip.js');
            const L = await import('/pose/behavior-labels.js');
            const session = AS.state.session;
            S.setBehaviorStripOn(!mode.startsWith('off'));
            L.cancelRunning(session, 'b4', 'id_0', '');
            await AS.videoController.seekToFrame(start);
            if (mode.endsWith('+run')) L.toggleState(session, 'b4', 'id_0', '', start);
            await new Promise(r => setTimeout(r, 700));       // the views settle if the strip came or went
        }, { mode, start });
        await browser.startTracing(page, { categories: ['devtools.timeline', 'disabled-by-default-devtools.timeline'] });
        const rafs = await page.evaluate(async ({ seconds }) => {
            const AS = await import('/ui/app-state.js');
            AS.videoController.startPlayback();
            const t = [];
            const end = performance.now() + seconds * 1000;
            await new Promise(res => {
                const tick = (now) => { t.push(now); if (now < end) requestAnimationFrame(tick); else res(); };
                requestAnimationFrame(tick);
            });
            AS.videoController.stopPlayback();
            return { t, frame: AS.state.currentFrame };
        }, { seconds: SECONDS });
        const trace = JSON.parse((await browser.stopTracing()).toString());
        const events = trace.traceEvents || trace;
        const main = new Set(events.filter(e => e.ph === 'M' && e.name === 'thread_name' && e.args && e.args.name === 'CrRendererMain').map(e => e.pid + ':' + e.tid));
        const sums = {}, counts = {};
        let raster = 0, rasterN = 0;
        for (const e of events) {
            if (e.ph !== 'X' || !e.dur) continue;
            if (e.name === 'RasterTask' || e.name === 'ImageDecodeTask') { raster += e.dur; rasterN++; }
            if (!main.has(e.pid + ':' + e.tid) || !WATCH.includes(e.name)) continue;
            sums[e.name] = (sums[e.name] || 0) + e.dur;
            counts[e.name] = (counts[e.name] || 0) + 1;
        }
        const iv = [];
        for (let i = 1; i < rafs.t.length; i++) iv.push(rafs.t[i] - rafs.t[i - 1]);
        return { mode, sums, counts, raster, rasterN, iv, frame: rafs.frame };
    };

    const results = { off: [], on: [], 'off+run': [], 'on+run': [] };
    let start = 3000;
    for (let r = 0; r < ROUNDS; r++) {
        const modes = Object.keys(results);
        if (r % 2) modes.reverse();     // neither always first
        for (const mode of modes) {
            const res = await pass(mode, start);
            results[mode].push(res);
            start = res.frame + 600;
            console.log(`  round ${r + 1} ${mode.padEnd(8)} ended on frame ${res.frame.toLocaleString('en-US')}`);
        }
    }

    const perSec = (list, name) => list.reduce((a, x) => a + (x.sums[name] || 0), 0) / 1000 / (list.length * SECONDS);
    const countPerSec = (list, name) => list.reduce((a, x) => a + (x.counts[name] || 0), 0) / (list.length * SECONDS);
    console.log('\nMain thread, ms per second of playback (count per second in brackets):');
    console.log(' '.repeat(16) + Object.keys(results).map(m => m.padStart(17)).join(''));
    for (const name of WATCH) {
        console.log(name.padEnd(16) + Object.values(results).map(list => (fmt(perSec(list, name)) + ' [' + countPerSec(list, name).toFixed(1).padStart(5) + ']').padStart(17)).join(''));
    }
    const sumAll = (list) => WATCH.filter(n => n !== 'Commit').reduce((a, n) => a + perSec(list, n), 0);
    console.log('all of it'.padEnd(16) + Object.values(results).map(list => fmt(sumAll(list)).padStart(17)).join(''));
    console.log('raster, others'.padEnd(16) + Object.values(results).map(list => (fmt(list.reduce((a, x) => a + x.raster, 0) / 1000 / (list.length * SECONDS)) + ' [' + (list.reduce((a, x) => a + x.rasterN, 0) / (list.length * SECONDS)).toFixed(1).padStart(5) + ']').padStart(17)).join(''));
    console.log('\nDisplay frames, ms between them:');
    for (const [m, list] of Object.entries(results)) {
        const iv = list.flatMap(x => x.iv);
        console.log(`  ${m.padEnd(8)} median ${quantile(iv, 0.5).toFixed(1)}  p95 ${quantile(iv, 0.95).toFixed(1)}  p99 ${quantile(iv, 0.99).toFixed(1)}  max ${Math.max(...iv).toFixed(1)}  over 33 ms: ${iv.filter(x => x > 33).length} of ${iv.length}`);
    }

    // ---- one-off: the film's rebuild, and a new lane during playback ----------
    const slide = await page.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        const S = await import('/ui/behavior-strip.js');
        const L = await import('/pose/behavior-labels.js');
        S.setBehaviorStripOn(true);
        L.cancelRunning(AS.state.session, 'b4', 'id_0', '');
        await new Promise(r => requestAnimationFrame(r));
        const js = [], layout = [];
        let f = 3000;
        for (let i = 0; i < 40; i++) {
            f += 1250;      // past the film's end: every call rebuilds it
            const t0 = performance.now();
            S.updateBehaviorStripFrame(f);
            const t1 = performance.now();
            document.body.offsetHeight;     // what the rebuild leaves for the frame's layout to do
            js.push(t1 - t0); layout.push(performance.now() - t1);
            await new Promise(r => requestAnimationFrame(r));
        }
        return { js, layout };
    });
    const q50 = (a) => quantile(a, 0.5).toFixed(2), mx = (a) => Math.max(...a).toFixed(2);
    console.log(`\nA film redrawn (once per window of playback): script median ${q50(slide.js)} ms, max ${mx(slide.js)}; layout after it median ${q50(slide.layout)} ms, max ${mx(slide.layout)}`);

    const lane = await page.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        const L = await import('/pose/behavior-labels.js');
        const measure = async (ms) => {
            const iv = []; let last = performance.now();
            const end = last + ms;
            await new Promise(res => {
                const tick = (now) => { iv.push(now - last); last = now; if (now < end) requestAnimationFrame(tick); else res(); };
                requestAnimationFrame(tick);
            });
            return Math.max(...iv);
        };
        AS.videoController.startPlayback();
        await measure(500);
        const before = await measure(1000);
        const hBefore = document.getElementById('behaviorStrip').getBoundingClientRect().height;
        L.labelRange(AS.state.session, 'b4', 'id_1', '', AS.state.currentFrame, AS.state.currentFrame + 60, true);
        const after = await measure(1000);
        const hAfter = document.getElementById('behaviorStrip').getBoundingClientRect().height;
        AS.videoController.stopPlayback();
        return { before, after, hBefore, hAfter };
    });
    console.log(`A new lane during playback (strip ${lane.hBefore} → ${lane.hAfter} px): longest frame ${lane.before.toFixed(1)} ms before, ${lane.after.toFixed(1)} ms after`);
} finally {
    if (browser) await browser.close();
    server.kill();
}
