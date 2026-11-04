/**
 * behavior-running-timeline.mjs — a running behavior state on the Timeline:
 * its bar grows to the playhead, and the Timeline redraws for it only when it
 * grows by a device pixel, never while collapsed.
 *
 * It used to redraw on every frame while a state ran (rows rebuilt, the whole
 * canvas repainted, collapsed or not): playback on 8 views fell from about 60
 * to about 43 fps.
 *
 * An hour at 60 fps (216,000 frames), two animals, device pixel ratio 2, no
 * video: Play steps the frames on a 60 fps timer and the Timeline's playhead
 * follows at ~10 Hz, as in playback. The state is started with its key in
 * Label Behaviors mode. `Timeline.redraw` is wrapped to count redraws; the bar
 * is read off the canvas around the playhead.
 *
 *  1. The whole hour on screen: 3 s of playback redraws a handful of times
 *     (the bar grows 0.012 device px a frame), not once a frame; the bar ends
 *     at the playhead; the mode bar's clock still runs.
 *  2. Stepping a frame at a time there: no redraw until the bar grows a pixel.
 *  3. Collapsed, at a pixel a frame: playback redraws nothing; opened again,
 *     the bar reaches the playhead at once.
 *  4. Zoomed in (20 px a frame): each step grows the bar to the playhead, a
 *     step back shrinks it, a click on the Timeline moves it.
 *  5. Zooming in after steps that redrew nothing: the bar reaches the playhead
 *     (it is drawn to the playhead, not to where the last redraw left it).
 *  6. Stopping the state, a label edit and a behavior selection repaint the
 *     rows as before; with no state running, steps redraw nothing.
 *
 * Run: node tests/e2e/behavior-running-timeline.mjs   (spawns its own http.server; PORT= to move it)
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8203);
const T = 216000;       // an hour at 60 fps

let fails = 0;
const check = (c, m) => { console.log((c ? '  ✓ ' : '  ✗ ') + m); if (!c) fails++; };

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1200));

let browser;
try {
    browser = await chromium.launch();
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });
    const errs = [];
    page.on('pageerror', e => errs.push('pageerror: ' + String(e).slice(0, 300)));
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state, { timeout: 20000 });

    // Two animals, an ethogram, one bout far from where the state runs.
    await page.evaluate(async (T) => {
        const pd = await import('/pose/pose-data.js');
        const AS = await import('/ui/app-state.js');
        const eth = await import('/pose/behavior-ethogram.js');
        const L = await import('/pose/behavior-labels.js');
        const session = new pd.Session([], new pd.Skeleton('m', ['a', 'b'], [[0, 1]]), [], 'Running');
        session.addIdentity('id_0'); session.addIdentity('id_1');
        AS.state.sessions = [session]; AS.state.activeSessionIdx = 0; AS.state.session = session;
        AS.state.totalFrames = T; AS.state.currentFrame = 0; AS.state.views = []; AS.state.fps = 60;
        AS.timeline.setTotalFrames(T); AS.timeline.setData(session);
        (await import('/ui/info-panel.js')).updateInfoPanel();
        eth.addBehavior({ name: 'rear', key: 'r' });
        eth.addBehavior({ name: 'groom', key: 'g' });
        L.labelRange(session, 'b1', 'id_1', '', 150000, 150600, true);
        (await import('/ui/behavior-coding.js')).requestBehaviorRepaint({ ethogram: true });
        AS.timeline.refreshTracks(session);
        const tl = AS.timeline, redraw = tl.redraw.bind(tl);
        window.__redraws = 0;
        tl.redraw = function () { window.__redraws++; return redraw(); };
    }, T);

    const rafs = (n = 2) => page.evaluate((n) => new Promise(r => { const f = () => (--n > 0 ? requestAnimationFrame(f) : r()); requestAnimationFrame(f); }), n);
    const redraws = () => page.evaluate(() => window.__redraws);
    const frame = () => page.evaluate(async () => (await import('/ui/app-state.js')).state.currentFrame);
    const goTo = (f) => page.evaluate(async (f) => (await import('/pose/initialization.js')).navigateToFrame(f), f);
    const bouts = () => page.evaluate(async () => {
        const L = await import('/pose/behavior-labels.js'); const AS = await import('/ui/app-state.js');
        return L.listBouts(AS.state.session, { animal: 'id_0' }).map(b => [b.behaviorId, b.start, b.end]);
    });
    // Press Play, wait, press it again; the frames played.
    const play = async (ms) => {
        const f0 = await frame();
        await page.click('#btnPlay');
        await page.waitForTimeout(ms);
        await page.click('#btnPlay');
        await rafs();
        return (await frame()) - f0;
    };
    // `perFrame` CSS px a frame (0: the whole hour), the playhead `lead` frames from the left edge.
    const view = (perFrame, lead) => page.evaluate(async ({ perFrame, lead }) => {
        const tl = (await import('/ui/app-state.js')).timeline;
        const w = tl._cssWidth - tl.LEFT_MARGIN - tl.RIGHT_PADDING;
        tl.setZoom(perFrame ? tl._totalFrames * perFrame / w : 1);
        tl._scrollFrame = tl._currentFrame - lead;
        tl._clampScroll();
        tl.redraw();
    }, { perFrame, lead });
    // The canvas on id_0's row: rear's color 1 CSS px left of the Timeline's
    // playhead (`at`), and 3 px past the end of the playhead's frame (`past`).
    const probe = () => page.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        const eth = await import('/pose/behavior-ethogram.js');
        const tl = AS.timeline;
        let y = tl._layout.trackAreaTop, prev = null, rowY = null;
        for (const row of tl._trackSegments) {
            if (prev != null && row.cameraName !== prev) y += tl.VIEW_GROUP_GAP;
            if (row._isBehavior && row.trackName === 'id_0') { rowY = y; break; }
            y += tl.TRACK_ROW_HEIGHT + tl.TRACK_ROW_GAP; prev = row.cameraName;
        }
        const want = [1, 3, 5].map(i => parseInt(eth.behaviorById('b1').color.slice(i, i + 2), 16));
        const d = tl._effDpr, Y = Math.floor((rowY + tl.TRACK_ROW_HEIGHT / 2 - tl._bandTop) * d);
        const isRear = (x) => {
            const p = tl._ctx.getImageData(Math.floor(x * d), Y, 1, 1).data;
            return Math.abs(p[0] - want[0]) + Math.abs(p[1] - want[1]) + Math.abs(p[2] - want[2]) < 30;
        };
        return { at: isRear(tl._frameToX(tl._currentFrame + 0.5) - 1), past: isRear(tl._frameToX(tl._currentFrame + 1) + 3) };
    });
    const reaches = (p) => p.at && !p.past;

    console.log('\n-- 1. the whole hour on screen, playing with a state running --');
    await page.mouse.click(300, 300);
    await page.evaluate(async () => (await import('/ui/behavior-mode.js')).enterBehaviorMode());
    await goTo(3000);
    await page.keyboard.press('1');
    await page.keyboard.press('r');
    await goTo(6000);
    await rafs();
    const running = await page.evaluate(async () => (await import('/pose/behavior-labels.js'))
        .runningStates((await import('/ui/app-state.js')).state.session).map(r => [r.behaviorId, r.animal, r.start]));
    check(JSON.stringify(running) === '[["b1","id_0",3000]]', `rear runs for id_0 from frame 3,000 (${JSON.stringify(running)})`);
    const clock0 = await page.textContent('#behaviorModeHint');
    let r0 = await redraws();
    const played1 = await play(3000);
    const n1 = (await redraws()) - r0;
    check(played1 >= 30, `playback ran (${played1} frames)`);
    check(n1 <= 6, `the Timeline redrew ${n1} times in ${played1} frames (it was once a frame)`);
    let p = await probe();
    check(reaches(p), `the bar ends at the playhead (${JSON.stringify(p)})`);
    const clock1 = await page.textContent('#behaviorModeHint');
    check(clock1 !== clock0, `the mode bar's clock still runs ("${clock0}" → "${clock1}")`);

    console.log('\n-- 2. stepping a frame at a time, the whole hour on screen --');
    r0 = await redraws();
    for (let i = 0; i < 30; i++) { await page.keyboard.press('ArrowRight'); await rafs(); }
    const n2 = (await redraws()) - r0;
    check(n2 <= 1, `30 steps redrew the Timeline ${n2} times (the bar grew under a device pixel; it was 30)`);
    p = await probe();
    check(reaches(p), `the bar ends at the playhead (${JSON.stringify(p)})`);

    console.log('\n-- 3. collapsed --');
    await view(1, 100);
    await rafs();
    await page.click('#timelineToggleBtn');
    await rafs(3);
    check(await page.evaluate(() => document.getElementById('timelineContainer').classList.contains('collapsed')), 'the Timeline is collapsed');
    r0 = await redraws();
    const played3 = await play(1000);
    const n3 = (await redraws()) - r0;
    check(played3 >= 10, `playback ran (${played3} frames)`);
    check(n3 === 0, `collapsed at a pixel a frame, the Timeline did not redraw (${n3} redraws in ${played3} frames)`);
    await page.click('#timelineToggleBtn');
    await rafs(3);
    p = await probe();
    check(reaches(p), `opened again, the bar reaches the playhead at once (${JSON.stringify(p)})`);

    console.log('\n-- 4. zoomed in, 20 px a frame --');
    await view(20, 10);
    await rafs();
    r0 = await redraws();
    const steps = [];
    for (let i = 0; i < 5; i++) {
        await page.keyboard.press('ArrowRight');
        await rafs();
        steps.push(reaches(await probe()));
    }
    const n4 = (await redraws()) - r0;
    check(steps.every(Boolean), `each step grows the bar to the playhead (${JSON.stringify(steps)})`);
    check(n4 >= 5, `redrawing for it (${n4} redraws in 5 steps)`);
    await page.keyboard.press('ArrowLeft');
    await page.keyboard.press('ArrowLeft');
    await rafs();
    p = await probe();
    check(reaches(p), `two steps back shrink it to the playhead (${JSON.stringify(p)})`);
    const click = await page.evaluate(async () => {
        const tl = (await import('/ui/app-state.js')).timeline;
        const rect = tl._canvas.getBoundingClientRect();
        let y = tl._layout.trackAreaTop, prev = null, rowY = null;
        for (const row of tl._trackSegments) {
            if (prev != null && row.cameraName !== prev) y += tl.VIEW_GROUP_GAP;
            if (row._isBehavior && row.trackName === 'id_0') { rowY = y; break; }
            y += tl.TRACK_ROW_HEIGHT + tl.TRACK_ROW_GAP; prev = row.cameraName;
        }
        const x = tl._frameToX(tl._currentFrame + 10.2);
        return { x: rect.left + x, y: rect.top + rowY - tl._bandTop + 5, to: tl._clampFrame(tl._xToFrame(x)) };
    });
    await page.mouse.click(click.x, click.y);
    await rafs();
    check(await frame() === click.to, `a click on the Timeline goes 10 frames on (${await frame()} = ${click.to})`);
    p = await probe();
    check(reaches(p), `and the bar follows it there (${JSON.stringify(p)})`);

    console.log('\n-- 5. zooming in after steps that redrew nothing --');
    await view(0, 0);
    await rafs();
    r0 = await redraws();
    for (let i = 0; i < 20; i++) { await page.keyboard.press('ArrowRight'); await rafs(1); }
    await rafs();
    const n5 = (await redraws()) - r0;
    await view(20, 30);
    await rafs();
    p = await probe();
    check(reaches(p), `after 20 steps and ${n5} redraws, zooming in draws the bar to the playhead (${JSON.stringify(p)})`);

    console.log('\n-- 6. labels, selection, and no state running --');
    const stopAt = await frame();
    r0 = await redraws();
    await page.keyboard.press('r');
    await rafs();
    check(JSON.stringify(await bouts()) === JSON.stringify([['b1', 3000, stopAt]]), `r stops rear: one bout 3,000–${stopAt.toLocaleString('en-US')} (${JSON.stringify(await bouts())})`);
    check((await redraws()) > r0, 'stopping the state repaints the rows');
    r0 = await redraws();
    for (let i = 0; i < 5; i++) { await page.keyboard.press('ArrowRight'); await rafs(); }
    check((await redraws()) === r0, 'with no state running, steps redraw nothing');
    p = await probe();
    check(!p.at, 'and the stopped bout stays where it ended');
    r0 = await redraws();
    await page.evaluate(async (f) => {
        const L = await import('/pose/behavior-labels.js'); const AS = await import('/ui/app-state.js');
        L.labelRange(AS.state.session, 'b2', 'id_0', '', f + 100, f + 200, true);
    }, stopAt);
    await rafs();
    check((await redraws()) > r0, 'a label edit repaints the rows');
    r0 = await redraws();
    await page.evaluate(async () => (await import('/ui/behavior-coding.js')).selectBehavior('b2'));
    await rafs();
    const shown = await page.evaluate(async () => (await import('/ui/app-state.js')).timeline._trackSegments
        .find(r => r._isBehavior && r.trackName === 'id_0').segments.map(s => [s.start, s.end]));
    check((await redraws()) > r0 && JSON.stringify(shown) === JSON.stringify([[stopAt + 100, stopAt + 200]]),
        `selecting groom repaints the rows to show only it (${JSON.stringify(shown)})`);
    await page.evaluate(async () => (await import('/ui/behavior-coding.js')).selectBehavior(null));

    check(errs.length === 0, `no page errors (${errs.join(' | ').slice(0, 300)})`);
    console.log(fails === 0 ? '\nPASS' : `\nFAIL — ${fails} check(s)`);
    await browser.close();
    browser = null;
    process.exitCode = fails === 0 ? 0 : 1;
} finally {
    if (browser) await browser.close().catch(() => {});
    server.kill();
}
