/**
 * behavior-tail-motion.mjs — Behavior ▸ Tail Motion from Video in the real app:
 * a real H.264 video decoded by the frame-accurate decoder (mediabunny, the
 * same `samplesAtTimestamps` stream the image check uses), its luma read from
 * each VideoFrame, the tail bands placed from 3D poses projected through a
 * calibration camera, and the result used by the behavior classifiers.
 *
 * The video is drawn here (ffmpeg, libx264 with B-frames): two mice seen from
 * above; mouse id_0's tail shakes sideways on three stretches; id_1 sits still.
 * The session's 3D poses are the drawn ones on the floor (z = 0), seen by a
 * camera 1,000 mm above it whose image is the floor in millimetres.
 * Asserted:
 *  1. Every frame is read; id_0's tail flicker is high where its tail shakes and
 *     ~0 elsewhere; id_1's is ~0; the tail chain and the crop are reported.
 *  2. The behavior context then carries the five channels: its catalog has them,
 *     and the feature columns computed on a forced 3-worker pool equal the ones
 *     computed inline (the channels travel to the workers).
 *  3. A "tail rattle" classifier trained on labels of the shaking stretches finds
 *     them (cross-validated F1 > 0.9) with tail motion, and the same forest
 *     (hash) trains on 3 workers as inline; without tail motion it cannot
 *     (F1 < 0.5), and the model then says to measure tail motion first.
 *  4. The menu item is there, and without loaded videos the dialog does not
 *     open: the status line says what is missing. Forget clears the record.
 *
 * Needs ffmpeg (libx264); skipped with a note if absent.
 * Run: node tests/e2e/behavior-tail-motion.mjs
 */
import { chromium } from 'playwright';
import { spawn, spawnSync } from 'node:child_process';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8297);
let fails = 0;
const check = (c, m) => { console.log((c ? '  ✓ ' : '  ✗ ') + m); if (!c) fails++; };

// ---------------------------------------------------------------------------
// The video and its poses
// ---------------------------------------------------------------------------
const W = 320, H = 240, T = 600;
const NODES = ['Nose', 'Ear_R', 'Ear_L', 'TTI', 'TailTip', 'Head', 'Trunk', 'Tail_0', 'Tail_1', 'Tail_2', 'Shoulder_left', 'Shoulder_right',
    'Haunch_left', 'Haunch_right', 'Neck'];
const ix = (nm) => NODES.indexOf(nm);
const EDGES = [['Nose', 'Head'], ['Head', 'Neck'], ['Neck', 'Trunk'], ['Trunk', 'TTI'], ['TTI', 'Tail_0'], ['Tail_0', 'Tail_1'], ['Tail_1', 'Tail_2'],
    ['Tail_2', 'TailTip'], ['Head', 'Ear_R'], ['Head', 'Ear_L'], ['Neck', 'Shoulder_left'], ['Neck', 'Shoulder_right'], ['TTI', 'Haunch_left'],
    ['TTI', 'Haunch_right']].map(([a, b]) => [ix(a), ix(b)]);
const SHAKES = [[60, 120], [240, 300], [420, 480]];
const shaking = (t) => SHAKES.some(([a, b]) => t >= a && t < b);
const MICE = [{ tti: [200, 150], dir: 1 }, { tti: [70, 60], dir: 1 }];

function mousePoints(m, shake) {
    const [x, y] = m.tti;
    const p = {
        TTI: [x, y], Trunk: [x + 25, y], Neck: [x + 50, y], Head: [x + 62, y], Nose: [x + 70, y],
        Ear_R: [x + 58, y + 7], Ear_L: [x + 58, y - 7], Shoulder_left: [x + 45, y - 9], Shoulder_right: [x + 45, y + 9],
        Haunch_left: [x + 8, y - 10], Haunch_right: [x + 8, y + 10],
        Tail_0: [x - 15, y + shake * 0.3], Tail_1: [x - 30, y + shake * 0.6], Tail_2: [x - 45, y + shake * 0.85], TailTip: [x - 60, y + shake],
    };
    return NODES.map((nm) => p[nm]);
}
const poses = [];   // [t][mouse] -> points
const raw = Buffer.alloc(W * H * T);
for (let t = 0; t < T; t++) {
    const off = t * W * H;
    const put = (x, y, v) => { x = Math.round(x); y = Math.round(y); if (x >= 0 && y >= 0 && x < W && y < H) raw[off + y * W + x] = Math.max(raw[off + y * W + x], v); };
    const disc = (cx, cy, r, v) => { for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) if (dx * dx + dy * dy <= r * r) put(cx + dx, cy + dy, v); };
    const line = (a, b, v) => { const n = Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1])) * 2; for (let i = 0; i <= n; i++) disc(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n, 1, v); };
    for (let i = 0; i < W * H; i++) raw[off + i] = 30;   // a dim floor
    const pts = MICE.map((m, k) => mousePoints(m, k === 0 && shaking(t) ? (t % 2 ? 6 : -6) : 0));
    pts.forEach((P) => {
        for (let k = 0; k <= 10; k++) disc(P[ix('TTI')][0] + (P[ix('Nose')][0] - P[ix('TTI')][0]) * k / 10, P[ix('TTI')][1], 9, 190);
        const chain = ['TTI', 'Tail_0', 'Tail_1', 'Tail_2', 'TailTip'].map((nm) => P[ix(nm)]);
        for (let s = 0; s < 4; s++) line(chain[s], chain[s + 1], 235);
    });
    poses.push(pts);
}
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'lucid-tail-motion-'));
const video = path.join(tmp, 'top.mp4');
const enc = spawnSync('ffmpeg', ['-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', W + 'x' + H, '-r', '60', '-i', 'pipe:0',
    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '12', '-g', '60', '-pix_fmt', 'yuv420p', video], { input: raw, maxBuffer: 1 << 30 });
if (enc.status !== 0 || !fs.existsSync(video)) {
    console.log('  (ffmpeg with libx264 unavailable — skipped)');
    fs.rmSync(tmp, { recursive: true, force: true });
    process.exit(0);
}

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise((r) => setTimeout(r, 1200));

let browser;
try {
    browser = await chromium.launch();
    const page = await browser.newPage({ viewport: { width: 1500, height: 900 } });
    page.on('pageerror', (e) => { console.log('  [pageerror]', String(e).slice(0, 300)); fails++; });
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state && window.SleapIO && window.SleapIO.MediaBunnyVideoBackend, null, { timeout: 30000 });
    await page.evaluate(() => {
        const inp = document.createElement('input');
        inp.type = 'file'; inp.id = '__vid';
        document.body.appendChild(inp);
    });
    await page.setInputFiles('#__vid', video);

    // the session: 3D on the floor, a camera whose image is the floor in mm
    await page.evaluate(async ({ NODES, EDGES, poses, T }) => {
        const pd = await import('./pose/pose-data.js');
        const AS = await import('./ui/app-state.js');
        const cam = new pd.Camera('top', [[1000, 0, 0], [0, 1000, 0], [0, 0, 1]], [0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 1000], [320, 240]);
        const s = new pd.Session([cam], new pd.Skeleton('mouse', NODES, EDGES), [], 'Tail test');
        s.addIdentity('id_0'); s.addIdentity('id_1');
        let gid = 1;
        for (let t = 0; t < T; t++) {
            s.instanceGroups.set(t, [0, 1].map((k) => {
                const g = new pd.InstanceGroup(gid++, s.identities[k].id);
                const pts = new Float64Array(NODES.length * 3);
                poses[t][k].forEach((p, n) => { pts[3 * n] = p[0]; pts[3 * n + 1] = p[1]; pts[3 * n + 2] = 0; });
                g.points3d = pts;
                return g;
            }));
        }
        s.totalFrames = T;
        AS.state.sessions = [s]; AS.state.activeSessionIdx = 0; AS.state.session = s;
        AS.state.totalFrames = T; AS.state.currentFrame = 0; AS.state.fps = 60; AS.state.views = [];
        window.__s = s;
    }, { NODES, EDGES: EDGES, poses, T });

    console.log('\n1. Reading the video');
    const r1 = await page.evaluate(async () => {
        const TM = await import('./ui/tail-motion.js');
        const PT = await import('./pose/tail-motion.js');
        const file = document.getElementById('__vid').files[0];
        const be = await window.SleapIO.MediaBunnyVideoBackend.fromBlob(file, file.name, {});
        const ticks = [];
        const rec = await TM.computeTailMotion(window.__s, 'top', { backend: be, onProgress: (d, n) => ticks.push([d, n]) });
        be.close();
        const flick = (a, from, to) => {
            let s = 0, n = 0;
            for (let t = from; t < to; t++) { const v = rec.raw[a][t * PT.TAIL_MOTION_RAW + 1]; if (Number.isFinite(v)) { s += v; n++; } }
            return n ? s / n : NaN;
        };
        const a0 = rec.animals.indexOf('id_0'), a1 = rec.animals.indexOf('id_1');
        return { framesRead: rec.framesRead, camera: rec.camera, chain: rec.tailNodes, side: rec.side, bodyPx: rec.bodyPx,
                 shake: [flick(a0, 65, 120), flick(a0, 245, 300), flick(a0, 425, 480)], still: flick(a0, 130, 230), other: flick(a1, 0, 600),
                 ticks: ticks.length, lastTick: ticks[ticks.length - 1], kept: PT.tailMotionOf(window.__s) === rec };
    });
    check(r1.framesRead === 600 && r1.kept && r1.lastTick && r1.lastTick[0] === 600, 'every frame read (' + r1.framesRead + '), progress reported, the record kept');
    check(r1.chain.join('>') === 'TTI>Tail_0>Tail_1>Tail_2>TailTip', 'the tail chain: ' + r1.chain.join(' > '));
    check(Math.abs(r1.bodyPx - 70) < 0.5 && Math.abs(r1.side - 2.4 * 70) < 1, 'the 3D poses land where they were drawn: body ' + r1.bodyPx.toFixed(1) + ' px, crop side ' + r1.side.toFixed(1) + ' px');
    check(r1.shake.every((v) => v > 8) && r1.still < 1, 'id_0\'s tail flicker while shaking ' + r1.shake.map((v) => v.toFixed(1)).join(' / ') + ', still ' + r1.still.toFixed(2));
    check(r1.other < 1, 'id_1 (still): ' + r1.other.toFixed(2));

    console.log('\n2. The behavior context and its feature columns');
    const r2 = await page.evaluate(async () => {
        const D = await import('./ui/behavior-detect.js');
        const ctx = D.buildBehaviorContext(window.__s);
        const names = ['tail flicker', 'tail flicker · mean ±0.3 s', 'body motion'];
        window.LUCID_BEHAVIOR_WORKER = 0;
        const inline = await D.computeFeatureColumns(ctx, names);
        window.LUCID_BEHAVIOR_WORKER = 3;
        const pooled = await D.computeFeatureColumns(ctx, names);
        delete window.LUCID_BEHAVIOR_WORKER;
        let same = true, finite = 0;
        names.forEach((nm) => inline.animal[nm].forEach((col, a) => col.forEach((v, t) => {
            const w = pooled.animal[nm][a][t];
            if (Number.isFinite(v)) finite++;
            if (!(v === w || (Number.isNaN(v) && Number.isNaN(w)))) same = false;
        })));
        return { channels: (ctx.spec.channels || []).map((c) => c.name), has: ctx.catalog.byName.has('tail flicker · max ±1 s'), key: ctx.channelsKey, same, finite };
    });
    check(r2.channels.length === 5 && r2.has && /^tail-motion\/1\/top\//.test(r2.key), 'the context carries the 5 channels (' + r2.channels.join(', ') + ')');
    check(r2.same && r2.finite > 3000, 'feature columns on 3 workers = inline (' + r2.finite.toLocaleString('en-US') + ' finite values)');

    console.log('\n3. A classifier with and without tail motion');
    const r3 = await page.evaluate(async () => {
        const E = await import('./pose/behavior-ethogram.js');
        const L = await import('./pose/behavior-labels.js');
        const CLS = await import('./pose/behavior-classifier.js');
        const D = await import('./ui/behavior-detect.js');
        const PT = await import('./pose/tail-motion.js');
        const s = window.__s;
        const b = E.addBehavior({ name: 'tail rattle', key: 'r' }).behavior;
        const runs = [[0, 59, false], [60, 119, true], [120, 239, false], [240, 299, true], [300, 419, false], [420, 479, true], [480, 599, false]];
        runs.forEach(([a, z, present]) => L.labelRange(s, b.id, 'id_0', '', a, z, present));
        L.labelRange(s, b.id, 'id_1', '', 0, 599, false);
        const train = async (workers) => {
            window.LUCID_BEHAVIOR_WORKER = workers;
            CLS.resetBehaviorModels();
            const ctx = D.buildBehaviorContext(s);
            const res = await CLS.trainBehaviorClassifier({ behaviorId: b.id, params: { trees: 40 }, sources: [{ session: s, name: s.name, ctx }] });
            delete window.LUCID_BEHAVIOR_WORKER;
            return res;
        };
        const inline = await train(0), pooled = await train(3);
        const rec = PT.tailMotionOf(s);
        PT.clearTailMotion(s);
        const without = await train(0);
        const why = CLS.modelProblems(inline.model, D.buildBehaviorContext(s));
        PT.setTailMotion(s, rec);   // keep it for the dialog check
        return { f1: inline.report && inline.report.cv.frame.f1, scheme: inline.report && inline.report.cv.scheme, err: inline.error || null,
                 same: inline.model && pooled.model && inline.model.hash === pooled.model.hash, hash: inline.model && inline.model.hash,
                 f1Without: without.report ? without.report.cv.frame.f1 : NaN, errWithout: without.error || null, why };
    });
    check(!r3.err && r3.f1 > 0.9, 'with tail motion: cross-validated F1 ' + (r3.f1 || 0).toFixed(3) + ' (' + r3.scheme + ')' + (r3.err ? ' — ' + r3.err : ''));
    check(r3.same, 'the same forest on 3 workers as inline (' + r3.hash + ')');
    check(r3.f1Without < 0.5, 'without it the poses cannot tell: F1 ' + (Number.isFinite(r3.f1Without) ? r3.f1Without.toFixed(3) : String(r3.errWithout)));
    check(r3.why.length === 1 && /Tail Motion from Video/.test(r3.why[0]), 'on a context without tail motion the model says so: ' + r3.why[0]);

    console.log('\n4. The menu and the dialog');
    const label = await page.evaluate(() => { const el = document.getElementById('menuBehaviorTailMotion'); return el && el.textContent.trim(); });
    check(label === 'Tail Motion from Video…', 'Behavior ▸ ' + label);
    await page.evaluate(() => document.getElementById('menuBehaviorTailMotion').click());
    await page.waitForTimeout(200);
    const r4 = await page.evaluate(() => ({ open: !!document.getElementById('tailMotionModal'), status: (document.getElementById('statusText') || document.getElementById('status') || {}).textContent || '' }));
    check(!r4.open && /videos loaded/.test(r4.status), 'no video loaded in a view: no dialog, the status says why ("' + r4.status.trim().slice(0, 80) + '")');
    // with a view whose decoder streams: the dialog lists the camera, Esc closes it, Forget clears
    await page.evaluate(async () => {
        const AS = await import('./ui/app-state.js');
        const file = document.getElementById('__vid').files[0];
        const be = await window.SleapIO.MediaBunnyVideoBackend.fromBlob(file, file.name, {});
        AS.state.views = [{ name: 'top', cameraName: 'top', decoder: { _mbBackend: be, getFrame() { return null; } } }];
    });
    await page.evaluate(() => document.getElementById('menuBehaviorTailMotion').click());
    await page.waitForSelector('#tailMotionModal');
    const r5 = await page.evaluate(() => {
        const sel = document.getElementById('tailMotionCamera');
        return { options: Array.from(sel.options).map((o) => o.textContent), info: !!document.querySelector('#tailMotionModal #tailMotionInfo[data-infotip]'),
                 forget: !!document.getElementById('tailMotionClear'), go: document.getElementById('tailMotionCompute').textContent };
    });
    check(r5.options.length === 1 && /^top/.test(r5.options[0]) && r5.info, 'the dialog lists the camera (' + r5.options.join(', ') + ') with its ⓘ');
    check(r5.forget && r5.go === 'Measure again', 'a measurement is kept: "Forget" and "Measure again"');
    await page.keyboard.press('Escape');
    check(await page.evaluate(() => !document.getElementById('tailMotionModal')), 'Esc closes it');
    await page.evaluate(() => document.getElementById('menuBehaviorTailMotion').click());
    await page.waitForSelector('#tailMotionClear');
    await page.click('#tailMotionClear');
    check(await page.evaluate(async () => { const PT = await import('./pose/tail-motion.js'); return PT.tailMotionOf(window.__s) === null; }), 'Forget clears it');
} finally {
    if (browser) await browser.close().catch(() => {});
    server.kill();
    fs.rmSync(tmp, { recursive: true, force: true });
}
console.log(fails ? `\n${fails} failure(s)` : '\nall passed');
process.exit(fails ? 1 : 0);
