/**
 * behavior-strip.mjs — the bouts strip under the camera view, in the real app,
 * through real key presses and mouse events.
 *
 *  1. No ethogram: no strip, and the layout is what it was.
 *  2. An ethogram with no labels: only in Label Behaviors mode; outside it, once
 *     the session has labels.
 *  3. Under the camera view and its handle, above the Timeline (on the controls bar
 *     with the Timeline closed), lined up with the seekbar: the view spans exactly its
 *     width, the playhead in its middle, at 1440 px, Timeline open or closed, and resized.
 *  4. One lane per animal and behavior, labelled; bars at their frames in the
 *     behavior's color; "not" faint; a mutual pair on both animals' lanes, a
 *     directed bout on the actor's; the window follows the playhead.
 *  5. A running state grows to the playhead.
 *  6. A click seeks there and stays paused; the tooltip names the frame the click
 *     lands on and the bout under the pointer.
 *  7. Shift+drag opens the Timeline's menu for the lane's animal; a behavior labels
 *     the range; Esc closes it; "Use as [ ] range" shows the armed range. The
 *     Timeline's own Shift+drag still opens its menu above the Timeline.
 *  8. Zoom (wheel, trackpad, buttons) and on/off: browser-local, remembered over a
 *     reload, and bring back through the Behavior tab. A label edit never moves them.
 *  9. Playback draws next to nothing: a moved frame copies the films to the
 *     screen and touches nothing else; the films are drawn once per window.
 * 10. Nothing reads frames: the strip works with frames that would throw if read.
 * 11. A click while playing keeps playing (video-less and real-controller paths);
 *     Shift+drag pauses first, because the window moves under a playing video.
 * 12. It follows the session; a project reopened through both readers (with the
 *     info panel collapsed on one) shows its labels with no click or frame change,
 *     with its real videos a click seeks, and New Project takes the strip away.
 * 13. More lanes than fit: the strip stays slim and says so, Shift+wheel scrolls
 *     them, and picking an animal brings its lanes into view.
 * 14. At devicePixelRatio 2: a pixel per device pixel, sharp edges.
 *
 * Run: node tests/e2e/behavior-strip.mjs   (spawns its own http.server; PORT= to move it)
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8773);

let fails = 0;
const check = (c, m) => { console.log((c ? '  ✓ ' : '  ✗ ') + m); if (!c) fails++; };
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const near = (a, b, tol) => Math.abs(a - b) <= tol;

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1200));

let browser;
try {
    browser = await chromium.launch();
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await ctx.newPage();
    const errs = [];
    page.on('pageerror', e => errs.push('pageerror: ' + String(e).slice(0, 300)));
    page.on('console', m => {
        if (m.type() !== 'error') return;
        const t = m.text();
        if (/Failed to load resource|net::ERR|404/.test(t)) return;   // absent demo assets
        errs.push('console.error: ' + t.slice(0, 300));
    });

    const FPS = 30, T = 6000;
    // A camera-less project of three animals (frames step without a decoder).
    const project = () => page.evaluate(async ({ FPS, T }) => {
        const pd = await import('/pose/pose-data.js');
        const AS = await import('/ui/app-state.js');
        const session = new pd.Session([], new pd.Skeleton('m', ['Nose', 'Tail'], [[0, 1]]), [], 'Strip');
        for (let i = 0; i < 3; i++) session.addIdentity('id_' + i);
        AS.state.sessions = [session]; AS.state.activeSessionIdx = 0; AS.state.session = session;
        AS.state.totalFrames = T; AS.state.currentFrame = 0; AS.state.views = []; AS.state.fps = FPS;
        AS.timeline.setTotalFrames(T); AS.timeline.setData(session);
        (await import('/ui/info-panel.js')).updateInfoPanel();
    }, { FPS, T });
    const boot = async () => {
        await page.goto(`http://localhost:${PORT}/index.html`);
        await page.waitForFunction(() => window.__lucid && window.__lucid.state, null, { timeout: 30000 });
        await project();
    };
    const addBehaviors = () => page.evaluate(async () => {
        const eth = await import('/pose/behavior-ethogram.js');
        eth.addBehavior({ name: 'rear', key: 'r' });                                // b1
        eth.addBehavior({ name: 'attack', key: 'a', social: true });                // b2
        eth.addBehavior({ name: 'sniff', key: 's', social: true, mutual: true });   // b3
        eth.addBehavior({ name: 'jump', key: 'j', kind: 'point' });                 // b4
        (await import('/ui/behavior-coding.js')).requestBehaviorRepaint({ ethogram: true });
        await new Promise(r => requestAnimationFrame(r));
    });
    const paint = () => page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
    const frame = () => page.evaluate(() => window.__lucid.state.currentFrame);
    const goTo = async (f) => {
        await page.evaluate(async (f) => (await import('/pose/initialization.js')).navigateToFrame(f), f);
        await paint();
    };
    const label = async (...args) => {
        await page.evaluate(async (args) => {
            const L = await import('/pose/behavior-labels.js');
            L.labelRange((await import('/ui/app-state.js')).state.session, ...args);
        }, args);
        await paint();
    };
    const shown = () => page.evaluate(() => !document.getElementById('behaviorStrip').hidden);
    const box = (sel) => page.evaluate((sel) => {
        const e = document.querySelector(sel);
        if (!e) return null;
        const r = e.getBoundingClientRect();
        return { x: r.left, y: r.top, w: r.width, h: r.height, r: r.right, b: r.bottom };
    }, sel);
    const laneNames = () => page.$$eval('.behavior-strip-label', els => els.map(e => e.textContent));
    // What is drawn on each lane, read back as runs of one colour along a pixel row near
    // the top of the bar (above any text): {lane, x, w (screen px), r, g, b, alpha}.
    const runs = (minAlpha = 0.1) => page.evaluate((minAlpha) => {
        const c = document.getElementById('behaviorStripCanvas');
        const ctx = c.getContext('2d');
        const box = c.getBoundingClientRect();
        const d = c.width / box.width;
        const out = [];
        for (let lane = 0; lane < Math.round(c.height / (14 * d)); lane++) {
            const row = ctx.getImageData(0, Math.round((lane * 14 + 2) * d), c.width, 1).data;
            const same = (i, j) => row[4 * i] === row[4 * j] && row[4 * i + 1] === row[4 * j + 1] && row[4 * i + 2] === row[4 * j + 2] && row[4 * i + 3] === row[4 * j + 3];
            for (let i = 0; i < c.width;) {
                let j = i + 1;
                while (j < c.width && same(i, j)) j++;
                if (row[4 * i + 3] / 255 >= minAlpha) {
                    out.push({ lane, x: box.left + i / d, w: (j - i) / d, r: row[4 * i], g: row[4 * i + 1], b: row[4 * i + 2], alpha: row[4 * i + 3] / 255 });
                }
                i = j;
            }
        }
        return out;
    }, minAlpha);
    // One pixel column of a canvas at a screen x, top to bottom: alpha of each CSS row.
    const column = (id, x) => page.evaluate(({ id, x }) => {
        const c = document.getElementById(id);
        const box = c.getBoundingClientRect();
        const d = c.width / box.width;
        const col = c.getContext('2d').getImageData(Math.round((x - box.left) * d), 0, 1, c.height).data;
        const out = [];
        for (let y = 0; y < box.height; y++) out.push(col[4 * Math.floor((y + 0.5) * d) + 3] / 255);
        return out;
    }, { id, x });
    const hex = (m) => '#' + [m.r, m.g, m.b].map(v => v.toString(16).padStart(2, '0')).join('');
    const startingAt = (list, x, lane) => list.find(m => near(m.x, x, 1.01) && (lane == null || m.lane === lane));
    // The Shift+drag's range (white) or the armed range (green), read from a lane.
    const rangeRuns = async (lane, isIt) => (await runs(0.1)).filter(m => m.lane === lane && isIt(m));
    const isWhite = (m) => m.r > 240 && m.g > 240 && m.b > 240;
    const isGreen = (m) => m.g > 150 && m.r < 130 && m.b < 170;
    const colorOf = (id) => page.evaluate(async (id) => (await import('/pose/behavior-ethogram.js')).behaviorById(id).color, id);
    // Where a frame's slot sits on screen, with the playhead on `cur`.
    const slotGeom = async (halfSeconds = 5) => {
        const v = await box('.behavior-strip-lanes');
        const half = Math.round(halfSeconds * FPS);
        const ppf = v.w / (2 * half + 1);
        return { v, ppf, left: (f, cur) => v.x + v.w / 2 + (f - cur - 0.5) * ppf, mid: (f, cur) => v.x + v.w / 2 + (f - cur) * ppf };
    };
    const clearLabels = () => page.evaluate(async () => {
        const L = await import('/pose/behavior-labels.js');
        L.resetBehaviorLabels((await import('/ui/app-state.js')).state.session);
        (await import('/ui/behavior-coding.js')).requestBehaviorRepaint({ labels: true });
        await new Promise(r => requestAnimationFrame(r));
    });

    await boot();

    console.log('\n-- 1. no ethogram --');
    check(!(await shown()), 'no strip without an ethogram');
    const noEthogramBar = await box('.controls-bar');
    const noEthogramSeek = await box('#seekbar');
    const noEthogramTimeline = await box('#timelineContainer');
    await page.mouse.click(300, 300);
    await page.keyboard.press('Meta+Shift+L');
    await paint();
    check(!(await shown()), 'nor in Label Behaviors mode');
    await page.keyboard.press('Escape');
    await paint();

    console.log('\n-- 2. when it shows --');
    await addBehaviors();
    check(!(await shown()), 'an ethogram with no labels, outside the mode: hidden');
    check(eq(await box('#seekbar'), noEthogramSeek), 'and the seekbar has not moved');
    await page.keyboard.press('Meta+Shift+L');
    await paint();
    check(await shown(), 'in Label Behaviors mode it shows');
    check(eq(await laneNames(), ['No bouts yet']), 'and says there are no bouts yet');
    await page.keyboard.press('Escape');
    await paint();
    check(!(await shown()), 'leaving the mode with nothing labelled hides it again');
    await label('b1', 'id_0', '', 2900, 2950, true);
    check(await shown(), 'outside the mode it shows once the session has labels');
    await clearLabels();
    check(!(await shown()), 'and goes when they are gone');
    await label('b1', 'id_0', '', 2900, 2950, true);
    await goTo(3000);

    console.log('\n-- 3. under the camera view, above the Timeline, lined up with the seekbar --');
    const alignedWith = async (what) => {
        await paint();
        const v = await box('.behavior-strip-lanes');
        const ruler = await box('.behavior-strip-ruler');
        const seek = await box('#seekbar');
        const strip = await box('#behaviorStrip');
        const handle = await box('#timelineResizeHandle');
        const tl = await box('#timelineContainer');
        const ph = await box('#behaviorStripPlayhead');
        check(near(v.x, seek.x, 0.51) && near(v.w, seek.w, 0.51), `${what}: the lanes span exactly the seekbar's width (${v.x.toFixed(1)}, ${v.w.toFixed(1)} and ${seek.x.toFixed(1)}, ${seek.w.toFixed(1)})`);
        check(near(ruler.x, seek.x, 0.51) && near(ruler.w, seek.w, 0.51), `${what}: and so does its ruler`);
        check(near(handle.b, strip.y, 0.51) && near(strip.b, tl.y, 0.51), `${what}: the strip sits right under the camera view's handle, on top of the Timeline`);
        check(near(ph.x + ph.w / 2, seek.x + seek.w / 2, 1), `${what}: the playhead is in the middle of the view`);
    };
    await alignedWith('Timeline open');
    const fits = async (what) => {
        const app = await page.evaluate(() => {
            const r = document.querySelector('.status-bar').getBoundingClientRect();
            const grid = document.querySelector('.video-grid-section').getBoundingClientRect();
            const c = document.querySelector('.app-container');
            return { bottom: r.bottom, gridH: grid.height, h: window.innerHeight, scrollY: c.scrollHeight - c.clientHeight };
        });
        check(near(app.bottom, app.h, 0.51) && app.scrollY <= 0, `${what}: the status bar is still at the window's bottom, nothing pushed off`);
        check(app.gridH > 200, `${what}: the camera view keeps its room (${Math.round(app.gridH)} px)`);
    };
    await fits('Timeline open');
    const openStrip = await box('#behaviorStrip');
    const openTimeline = await box('#timelineContainer');
    const openBar = await box('.controls-bar');
    check(openStrip.b <= openTimeline.y + 0.51 && near(openTimeline.b, openBar.y, 0.51), 'the strip is above the Timeline, and the Timeline still sits on the controls bar');
    check(near(openTimeline.h, noEthogramTimeline.h, 40) && openTimeline.h > 20, 'and the Timeline still has its height');
    const mainBottom = await page.evaluate(() => document.querySelector('.main-content').getBoundingClientRect().bottom);
    check(near(mainBottom, (await box('#timelineResizeHandle')).y, 0.51), 'nothing comes between the camera view and its handle');
    await page.click('#timelineToggleBtn');
    await paint();
    await alignedWith('Timeline closed');
    await fits('Timeline closed');
    const closedStrip = await box('#behaviorStrip');
    check(near(closedStrip.b, (await box('.controls-bar')).y, 0.51), 'with the Timeline closed the strip sits directly on the controls bar, over the seekbar');
    await page.click('#timelineToggleBtn');
    await paint();
    const seekWithStrip = await box('#seekbar');
    check(near(seekWithStrip.x, noEthogramSeek.x, 1) && near(seekWithStrip.w, noEthogramSeek.w, 1) && near(seekWithStrip.b, noEthogramSeek.b, 1), 'the seekbar stayed where it was without the strip');
    check(near((await box('.controls-bar')).h, noEthogramBar.h, 0.51), 'and the controls bar kept its height');
    await page.setViewportSize({ width: 1000, height: 700 });
    await alignedWith('at 1000 px');
    await page.setViewportSize({ width: 1440, height: 900 });
    await alignedWith('back at 1440 px');
    const toolbar = await page.evaluate(() => { const t = document.getElementById('toolbar'); return { s: t.scrollWidth, c: t.clientWidth }; });
    check(toolbar.s <= toolbar.c, 'the toolbar still fits a 1440 px window');

    console.log('\n-- 4. lanes and bars --');
    await clearLabels();
    await label('b1', 'id_0', '', 2900, 2950, true);                 // rear
    await label('b1', 'id_0', '', 2951, 2980, false);                // not rear
    await label('b2', 'id_1', 'id_2', 3010, 3090, true);             // attack, directed
    await label('b3', 'id_2', 'id_0', 3020, 3060, true);             // sniff, mutual: stored as (id_0, id_2)
    await label('b4', 'id_1', '', 3000, 3000, true);                 // a point
    await goTo(3000);
    await paint();
    const lanes = await laneNames();
    check(eq(lanes, ['id_0rear', 'id_0sniff', 'id_1attack', 'id_1jump', 'id_2sniff']), 'one lane per animal and behavior, the animal and behavior readable: ' + lanes.join(', '));
    const stored = await page.evaluate(async () => {
        const L = await import('/pose/behavior-labels.js');
        return L.listBouts((await import('/ui/app-state.js')).state.session, { behaviorId: 'b3' }).length;
    });
    const g = await slotGeom();
    const b = await runs();
    check(stored === 1 && b.filter(m => near(m.x, g.left(3020, 3000), 1.01)).length === 2, 'a mutual pair is stored once and shown on both animals\' lanes');
    check(!lanes.includes('id_2attack'), 'a directed bout only on the actor\'s lane');
    const rear = startingAt(b, g.left(2900, 3000), 0);
    check(!!rear, `a bar starts at its first frame (${g.left(2900, 3000).toFixed(1)}: ${JSON.stringify(b.filter(m => m.lane === 0).map(m => Math.round(m.x)))})`);
    check(near(rear.x + rear.w, g.left(2951, 3000), 1.01), 'and ends with its last');
    check(hex(rear) === (await colorOf('b1')).toLowerCase(), 'in the behavior\'s ethogram color');
    check(rear.alpha === 1, 'present bouts are solid');
    const notBar = startingAt(b, g.left(2951, 3000), 0);
    check(notBar && notBar.alpha < 0.4 && near(notBar.x + notBar.w, g.left(2981, 3000), 1.01), '"not" labels are faint, on their own frames');
    const col = await column('behaviorStripCanvas', rear.x + rear.w / 2);
    check(col[0] === 0 && col[1] === 1 && col[12] === 1 && col[13] === 0, 'bars are slim, inside their lane');
    const attack = startingAt(b, g.left(3010, 3000), 2);
    check(!!attack && hex(attack) === (await colorOf('b2')).toLowerCase(), 'each bar in its own lane, its own color');
    const textPixels = (y) => page.evaluate(({ x0, x1, y, c0 }) => {
        const c = document.getElementById('behaviorStripCanvas');
        const box = c.getBoundingClientRect();
        const row = c.getContext('2d').getImageData(Math.round(x0 - box.left), Math.round(y), Math.round(x1 - x0), 1).data;
        let other = 0;
        for (let i = 0; i < row.length; i += 4) if (Math.abs(row[i] - c0.r) + Math.abs(row[i + 1] - c0.g) + Math.abs(row[i + 2] - c0.b) > 60) other++;
        return other;
    }, { x0: attack.x, x1: attack.x + attack.w, y, c0: { r: attack.r, g: attack.g, b: attack.b } });
    const onText = await textPixels(2 * 14 + 7);
    check(onText > 3 && onText < attack.w / 4, `with more than two animals a directed bar names its target (${onText} pixels of text along it)`);
    const jump = b.find(m => m.lane === 3);
    check(jump.w <= 4 && near(jump.x + jump.w / 2, g.mid(3000, 3000), 1.51), 'a point is a tick on its frame, under the playhead here');
    const labels = await page.$$eval('.behavior-strip-label', els => els.map(e => e.getBoundingClientRect().top));
    const v0 = await box('.behavior-strip-lanes');
    check(labels.every((y, i) => near(y, v0.y + i * 14, 1)), 'each label lines up with its lane');
    const animalColor = await page.$eval('.behavior-strip-label .behavior-strip-animal', e => e.style.color);
    check(animalColor !== '', 'the animal in its identity color');
    await goTo(3010);
    const moved = startingAt(await runs(), g.left(2900, 3010), 0);
    check(!!moved && near(rear.x - moved.x, 10 * g.ppf, 1.01), 'the bars scroll with the playhead, which stays in the middle');
    await goTo(5000);
    check((await runs()).length === 0, 'far from any bout, none is on show');
    await goTo(0);
    const beforeStart = (await runs(0.01)).filter(m => m.x < v0.x + v0.w / 2 - 1);
    check(beforeStart.length > 0 && beforeStart.every(m => m.alpha < 0.1) && near(beforeStart[0].x + beforeStart[0].w, v0.x + v0.w / 2 - g.ppf / 2, 1.51), 'before the recording starts, the stretch is marked');
    await goTo(3000);
    const rulerH = (await box('#behaviorStripRulerCanvas')).h;
    const tickAt = await column('behaviorStripRulerCanvas', g.mid(3000, 3000));
    check(tickAt[rulerH - 1] > 0 && tickAt[rulerH - 5] > 0, 'the ruler has a major tick under the playhead at 01:40.000, on frame 3,000 at 30 fps');
    check(tickAt.slice(0, 9).some(a => a > 0), 'with its label above it');
    const offTick = await column('behaviorStripRulerCanvas', g.mid(3005, 3000));
    check(offTick[rulerH - 4] === 0 && offTick[rulerH - 5] === 0, 'and none a sixth of a second on');

    console.log('\n-- 5. a running state grows to the playhead --');
    await clearLabels();
    await goTo(1000);
    await page.mouse.click(300, 300);
    await page.keyboard.press('Meta+Shift+L');
    await page.keyboard.press('r');
    await paint();
    check(eq(await laneNames(), ['●id_0rear']), 'its lane appears at once, marked running');
    const g2 = await slotGeom();
    const run0 = (await runs())[0];
    check(near(run0.x, g2.left(1000, 1000), 1.01), 'and its bar starts on the playhead');
    for (let i = 0; i < 20; i++) await page.keyboard.press('ArrowRight');
    await paint();
    const run1 = (await runs())[0];
    check(near(run1.x + run1.w, g2.left(1021, 1020), 1.51), `it reaches the playhead's slot (${(run1.x + run1.w).toFixed(1)}, ${g2.left(1021, 1020).toFixed(1)})`);
    check(near(run1.x, g2.left(1000, 1020), 1.01), 'and still starts where it started');
    check(near(run1.w - run0.w, 20 * g2.ppf, 1.51), 'twenty frames later, twenty frames longer');
    await page.keyboard.press('r');
    await paint();
    const done = await runs();
    check(done.length === 1 && near(done[0].x, g2.left(1000, 1020), 1.01) && near(done[0].x + done[0].w, g2.left(1021, 1020), 1.51), 'stopping it leaves an ordinary bar');
    check(eq(await laneNames(), ['id_0rear']), 'and takes the running mark off its lane');
    await page.keyboard.press('Escape');
    await paint();

    console.log('\n-- 6. click and hover --');
    await clearLabels();
    await label('b2', 'id_0', 'id_1', 3020, 3040, true);
    await goTo(3000);
    const g3 = await slotGeom();
    const laneY = g3.v.y + 7;
    await page.mouse.move(g3.mid(3030, 3000), laneY);
    await paint();
    const tip = await page.textContent('#behaviorStripTooltip');
    check(tip === 'Frame 3,031 · 01:41.000 — attack · id_0 → id_1 · frames 3,021–3,041 (0.7 s)', 'the tooltip names the frame and the bout: ' + tip);
    const hover = await box('#behaviorStripHover');
    check(near(hover.x, g3.mid(3030, 3000), 1.01), 'a line marks the frame a click would land on');
    await page.mouse.move(g3.mid(2990, 3000), laneY);
    check((await page.textContent('#behaviorStripTooltip')) === 'Frame 2,991 · 01:39.667', 'off a bout, only the frame');
    const tipBox = await box('#behaviorStripTooltip');
    const stripTop = (await box('#behaviorStrip')).y;
    check(tipBox.b <= stripTop, 'above the strip');
    await page.mouse.click(g3.mid(3030, 3000), laneY);
    await paint();
    check(await frame() === 3030, 'a click on a lane seeks to that frame');
    check(!(await page.evaluate(() => window.__lucid.state.isPlaying)), 'and stays paused');
    const landed = (await runs())[0];
    const g3b = await slotGeom();
    check(near(landed.x, g3b.left(3020, 3030), 1.01), 'the strip follows, with the playhead on the frame');
    check(near(await page.evaluate(() => parseFloat(document.getElementById('seekbarThumb').style.left)), 3030 / (T - 1) * 100, 1e-3), 'and so does the seekbar\'s thumb');
    await page.mouse.click(g3b.mid(3000, 3030), (await box('.behavior-strip-ruler')).y + 7);
    await paint();
    check(await frame() === 3000, 'a click on the ruler too');
    const fromHere = await frame();
    await page.mouse.move(g3b.mid(3010, 3000), laneY);
    await page.mouse.down();
    await page.mouse.move(300, 300, { steps: 3 });
    await page.mouse.up();
    await paint();
    check(await frame() === fromHere, 'a press let go off the strip goes nowhere');
    await page.mouse.move(300, 300);
    await paint();
    check(await page.evaluate(() => document.getElementById('behaviorStripHover').hidden && document.getElementById('behaviorStripTooltip').hidden), 'leaving hides the line and the tooltip');
    await page.mouse.move(g3b.mid(3005, 3000), laneY);
    await goTo(3001);
    check(await page.evaluate(() => document.getElementById('behaviorStripTooltip').hidden), 'a tooltip that would now be wrong goes when the frame changes');
    await page.mouse.move(300, 300);

    console.log('\n-- 7. Shift+drag labels a range --');
    await clearLabels();
    await label('b1', 'id_0', '', 2800, 2810, true);
    await label('b1', 'id_1', '', 2800, 2810, true);
    await goTo(3000);
    const g4 = await slotGeom();
    const secondLane = g4.v.y + 14 + 7;
    const drag = async (from, to, y) => {
        await page.mouse.move(g4.mid(from, 3000), y);
        await page.keyboard.down('Shift');
        await page.mouse.down();
        await page.mouse.move(g4.mid(to, 3000), y, { steps: 4 });
        await page.mouse.up();
        await page.keyboard.up('Shift');
        await paint();
    };
    await drag(2950, 3050, secondLane);
    check(await page.isVisible('#behaviorRangeMenu'), 'it opens the label-range menu');
    check((await page.textContent('.behavior-range-head')).startsWith('Frames 2,951–3,051'), 'for the dragged frames');
    check(await page.$eval('.behavior-range-animal', e => e.value) === 'id_1', 'and the animal of the lane the drag began on');
    const menu = await box('#behaviorRangeMenu');
    const stripNow = await box('#behaviorStrip');
    check(menu.b <= stripNow.y && menu.x >= 0 && menu.r <= 1440, 'above the strip, on screen');
    const sel = await rangeRuns(1, isWhite);
    check(sel.length >= 2 && near(sel[0].x, g4.left(2950, 3000), 1.01) && near(sel[sel.length - 1].x + sel[sel.length - 1].w, g4.left(3051, 3000), 1.01), 'the selection shows while it is open');
    await page.click('#behaviorRangeMenu [data-behavior="b1"]');
    await paint();
    const added = await page.evaluate(async () => {
        const L = await import('/pose/behavior-labels.js');
        return L.listBouts((await import('/ui/app-state.js')).state.session, { behaviorId: 'b1', animal: 'id_1' }).map(x => [x.start, x.end, x.present]);
    });
    check(eq(added, [[2800, 2810, true], [2950, 3050, true]]), 'a behavior button labels the range');
    check((await rangeRuns(1, isWhite)).length === 0, 'and the selection goes');
    const fresh = startingAt(await runs(), g4.left(2950, 3000), 1);
    check(!!fresh && near(fresh.x + fresh.w, g4.left(3051, 3000), 1.01), 'the new bout shows at once');
    await drag(3055, 3065, g4.v.y + 7);
    check(await page.$eval('.behavior-range-animal', e => e.value) === 'id_0', 'the first lane\'s animal for a drag there');
    await page.keyboard.press('Escape');
    await paint();
    check(!(await page.$('#behaviorRangeMenu')), 'Esc closes the menu');
    check((await rangeRuns(0, isWhite)).length === 0, 'and takes the selection with it');
    check(await frame() === 3000, 'a drag never seeks');
    await drag(3055, 3065, (await box('.behavior-strip-ruler')).y + 7);
    check(await page.isVisible('#behaviorRangeMenu'), 'from the ruler too');
    await page.click('#behaviorRangeMenu .behavior-range-actions .panel-btn:first-child');
    await paint();
    const armed = await rangeRuns(0, isGreen);
    check(armed.length >= 1 && near(armed[0].x, g4.left(3055, 3000), 1.01), '"Use as [ ] range" shows the armed range on the strip');
    await goTo(3005);
    const armedMoved = await rangeRuns(0, isGreen);
    check(armedMoved.length >= 1 && near(armedMoved[0].x, g4.left(3055, 3005), 1.01), 'it scrolls with the rest');
    await page.evaluate(async () => (await import('/ui/behavior-coding.js')).clearArmedRange());
    await paint();
    check((await rangeRuns(0, isGreen)).length === 0, 'and clearing it takes it off');
    await goTo(3000);
    // In Label Behaviors mode, where this happens: Esc is the menu's before it is the mode's.
    await page.mouse.click(300, 300);
    await page.keyboard.press('Meta+Shift+L');
    await paint();
    await drag(3055, 3065, g4.v.y + 7);
    check(await page.isVisible('#behaviorRangeMenu'), 'in Label Behaviors mode too');
    await page.keyboard.press('Escape');
    await paint();
    check(!(await page.$('#behaviorRangeMenu')), 'Esc closes the menu');
    check(await page.evaluate(async () => (await import('/ui/behavior-mode.js')).isBehaviorModeActive()), 'and stays in the mode');
    await page.keyboard.press('Escape');
    await paint();
    check(!(await page.evaluate(async () => (await import('/ui/behavior-mode.js')).isBehaviorModeActive())), 'the next Esc leaves it');
    // A drag that leaves the strip keeps going, to the window's edge.
    await page.mouse.move(g4.mid(3060, 3000), g4.v.y + 7);
    await page.keyboard.down('Shift');
    await page.mouse.down();
    await page.mouse.move(1430, 200, { steps: 4 });
    await page.mouse.up();
    await page.keyboard.up('Shift');
    await paint();
    check((await page.textContent('.behavior-range-head')).startsWith('Frames 3,061–3,151'), 'a drag past the strip\'s end stops at the last frame on show: ' + (await page.textContent('.behavior-range-head')));
    await page.keyboard.press('Escape');
    await paint();
    // The Timeline's own Shift+drag is untouched.
    const tl = await box('#timelineContainer canvas');
    await page.mouse.move(tl.x + 400, tl.y + 20);
    await page.keyboard.down('Shift');
    await page.mouse.down();
    await page.mouse.move(tl.x + 500, tl.y + 20, { steps: 3 });
    await page.mouse.up();
    await page.keyboard.up('Shift');
    await paint();
    check(await page.isVisible('#behaviorRangeMenu'), 'the Timeline\'s Shift+drag still opens its menu');
    const tlMenu = await box('#behaviorRangeMenu');
    check(tlMenu.b <= tl.y + 0.51, 'above the Timeline, as before');
    await page.keyboard.press('Escape');
    await paint();

    console.log('\n-- 8. zoom and on/off stay in this browser --');
    const storedWindow = () => page.evaluate(() => localStorage.getItem('lucid.behaviorStripWindow'));
    check((await page.textContent('#behaviorStripWindow')) === '±5 s', 'about ±5 s by default');
    await page.click('#behaviorStripZoomOut');
    await paint();
    check((await page.textContent('#behaviorStripWindow')) === '±10 s', 'the − button shows more time');
    const g10 = await slotGeom(10);
    const wideBar = startingAt(await runs(), g10.left(2950, 3000), 1);
    check(!!wideBar && near(wideBar.w, 101 * g10.ppf, 1.01), 'and the bars redraw to the new scale');
    check(await storedWindow() === '10', 'remembered in this browser');
    await page.click('#behaviorStripZoomIn');
    await page.click('#behaviorStripZoomIn');
    await paint();
    check((await page.textContent('#behaviorStripWindow')) === '±3 s', 'the + button shows less');
    const v = await box('.behavior-strip-lanes');
    await page.mouse.move(v.x + 100, v.y + 5);
    await page.mouse.wheel(0, -100);
    await paint();
    check((await page.textContent('#behaviorStripWindow')) === '±2 s', 'a wheel notch up zooms in');
    await page.mouse.wheel(0, 100);
    await paint();
    await page.mouse.wheel(0, 100);
    await paint();
    check((await page.textContent('#behaviorStripWindow')) === '±5 s', 'down zooms out, a step a notch');
    for (let i = 0; i < 4; i++) await page.evaluate(() => {
        document.querySelector('.behavior-strip-lanes').dispatchEvent(new WheelEvent('wheel', { deltaY: -12, bubbles: true, cancelable: true }));
    });
    await paint();
    check((await page.textContent('#behaviorStripWindow')) === '±3 s', 'a trackpad\'s small deltas add up to one step');
    await page.evaluate(() => new Promise(r => setTimeout(r, 350)));
    await page.click('#behaviorStripZoomOut');
    await paint();
    check(await frame() === 3000, 'zooming keeps the playhead where it is');
    for (let i = 0; i < 12; i++) if (!(await page.isDisabled('#behaviorStripZoomIn'))) await page.click('#behaviorStripZoomIn');
    check(await page.isDisabled('#behaviorStripZoomIn') && (await page.textContent('#behaviorStripWindow')) === '±0.5 s', 'zoom stops at its ends');
    for (let i = 0; i < 12; i++) if (!(await page.isDisabled('#behaviorStripZoomOut'))) await page.click('#behaviorStripZoomOut');
    check(await page.isDisabled('#behaviorStripZoomOut') && (await page.textContent('#behaviorStripWindow')) === '±120 s', 'at both ends');
    await page.evaluate(async () => (await import('/ui/behavior-coding.js')).openBehaviorTab());
    await page.evaluate(async () => (await import('/ui/behavior-panel.js')).refreshBehaviorPanel());
    await page.click('#behaviorStripZoomIn');
    await paint();
    const windowNow = await page.textContent('#behaviorStripWindow');
    check(await page.isChecked('[data-act="strip"]'), 'the Behavior tab says it is on');
    await page.click('#behaviorStripClose');
    await paint();
    check(!(await shown()), 'the × hides it');
    check((await page.textContent('#statusText')).includes('Behavior tab'), 'and says how to bring it back');
    check(await page.evaluate(() => localStorage.getItem('lucid.behaviorStrip')) === '0', 'remembered in this browser');
    check(!(await page.isChecked('[data-act="strip"]')), 'the tab\'s box follows');
    await page.keyboard.press('Meta+Shift+L');
    await paint();
    check(!(await shown()), 'off means off in the mode too');
    await page.keyboard.press('Escape');
    await paint();
    await page.reload();
    await page.waitForFunction(() => window.__lucid && window.__lucid.state, null, { timeout: 30000 });
    await project();
    await addBehaviors();
    await label('b1', 'id_0', '', 100, 200, true);
    check(!(await shown()), 'off survives a reload');
    await page.evaluate(async () => (await import('/ui/behavior-coding.js')).openBehaviorTab());
    await page.evaluate(async () => (await import('/ui/behavior-panel.js')).refreshBehaviorPanel());
    await page.click('[data-act="strip"]');
    await paint();
    check(await shown(), 'the Behavior tab\'s box brings it back');
    check((await page.textContent('#behaviorStripWindow')) === windowNow, 'and the window survives a reload');
    const written = await page.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        const meta = await import('/import-export/behavior-metadata.js');
        const lucid = {};
        meta.writeBehaviorMetadata(AS.state.session, lucid);
        return JSON.stringify(lucid);
    });
    check(!/strip/i.test(written) && !/window/i.test(written), 'nothing about it goes in the project');
    await goTo(150);
    const atThirty = (await runs())[0];
    await page.dblclick('#fpsDisplay');
    await page.keyboard.type('60');
    await page.keyboard.press('Enter');
    await paint();
    const atSixty = (await runs())[0];
    check(near(atSixty.w * 2, atThirty.w, 2), 'a new frame rate in the fps box redraws the window at once: the same frames take half the room');
    await page.evaluate(() => { localStorage.removeItem('lucid.behaviorStripWindow'); localStorage.removeItem('lucid.behaviorStrip'); });
    await page.reload();
    await page.waitForFunction(() => window.__lucid && window.__lucid.state, null, { timeout: 30000 });
    await project();
    await addBehaviors();
    await label('b1', 'id_0', '', 2900, 2950, true);
    await label('b2', 'id_0', 'id_1', 3100, 3200, true);
    await label('b1', 'id_1', '', 3010, 3020, false);
    await goTo(3000);

    console.log('\n-- 9. playback paints nothing --');
    const cdp = await ctx.newCDPSession(page);
    await cdp.send('Performance.enable');
    const metric = async (name) => (await cdp.send('Performance.getMetrics')).metrics.find(m => m.name === name).value;
    const layouts0 = await metric('LayoutCount');
    const steps = await page.evaluate(async () => {
        const S = await import('/ui/behavior-strip.js');
        const strip = document.getElementById('behaviorStrip');
        let domWrites = 0;
        const mo = new MutationObserver(r => { domWrites += r.length; });
        mo.observe(strip, { childList: true, subtree: true, attributes: true, characterData: true });
        // Every drawing call on the strip's canvases: the two on screen, and the hidden films.
        const P = CanvasRenderingContext2D.prototype;
        const calls = { screen: {}, film: {} };
        const held = {};
        for (const name of ['drawImage', 'fillRect', 'fillText', 'clearRect']) {
            held[name] = P[name];
            P[name] = function (...args) {
                const c = this.canvas;
                const where = !c.isConnected ? 'film' : c.id.startsWith('behaviorStrip') ? 'screen' : null;
                if (where) calls[where][name] = (calls[where][name] || 0) + 1;
                return held[name].apply(this, args);
            };
        }
        let f = 3000, filmDrawn = 0;
        const perStep = new Set();
        // 300 display frames, one step of 3 video frames each: 900 frames in all, 3 windows.
        for (let i = 0; i < 300; i++) {
            await new Promise(r => requestAnimationFrame(r));
            const before = JSON.stringify(calls.screen);
            const filmBefore = calls.film.fillRect || 0;
            f += 3;
            S.updateBehaviorStripFrame(f);
            if ((calls.film.fillRect || 0) !== filmBefore) filmDrawn++;
            perStep.add((calls.screen.drawImage || 0) - (JSON.parse(before).drawImage || 0));
        }
        await new Promise(r => setTimeout(r, 0));
        Object.assign(P, held);
        mo.disconnect();
        return { domWrites, calls, filmDrawn, perStep: Array.from(perStep), f };
    });
    const layouts = (await metric('LayoutCount')) - layouts0;
    check(eq(steps.perStep, [2]), `every step copies the two films to the screen, and only that (${JSON.stringify(steps.perStep)} copies a step)`);
    check(!steps.calls.screen.fillRect && !steps.calls.screen.fillText, 'nothing else is drawn on screen when nothing is running');
    check(steps.filmDrawn >= 1 && steps.filmDrawn <= 4, `the films are drawn only when the window runs off them (${steps.filmDrawn} times in 900 frames, 3 windows)`);
    check(steps.domWrites === 0, 'and the page itself is never touched');
    check(layouts < 30, `no layout from a moved frame (${layouts} layouts in 300 steps)`);
    const cost = await page.evaluate(async () => {
        const S = await import('/ui/behavior-strip.js');
        const t0 = performance.now();
        let f = 3000;
        for (let i = 0; i < 2000; i++) { f = 3000 + (i % 200); S.updateBehaviorStripFrame(f); }
        return (performance.now() - t0) / 2000;
    });
    check(cost < 1, `a frame's update costs next to nothing (${(cost * 1000).toFixed(1)} µs on the main thread)`);
    await page.evaluate(async () => (await import('/ui/behavior-strip.js')).setBehaviorStripOn(false));
    await paint();
    const offCost = await page.evaluate(async () => {
        const S = await import('/ui/behavior-strip.js');
        const t0 = performance.now();
        for (let i = 0; i < 20000; i++) S.updateBehaviorStripFrame(3000 + (i % 200));
        return (performance.now() - t0) / 20000;
    });
    check(offCost < 0.05, `and off it is a comparison and a return (${(offCost * 1000).toFixed(2)} µs)`);
    await page.evaluate(async () => (await import('/ui/behavior-strip.js')).setBehaviorStripOn(true));
    await goTo(3000);

    console.log('\n-- 10. nothing reads frames --');
    const noFrames = await page.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        const S = await import('/ui/behavior-strip.js');
        const s = AS.state.session;
        const held = { frameGroups: s.frameGroups, instanceGroups: s.instanceGroups };
        const trap = new Proxy(new Map(), { get() { throw new Error('the strip read frames'); } });
        s.frameGroups = trap; s.instanceGroups = trap;
        const before = document.getElementById('behaviorStripLabels').textContent;
        let error = null;
        try {
            S.refreshBehaviorStrip();
            for (let f = 3000; f < 4000; f += 50) S.updateBehaviorStripFrame(f);
        } catch (e) { error = String(e); }
        s.frameGroups = held.frameGroups; s.instanceGroups = held.instanceGroups;
        S.updateBehaviorStripFrame(3000);
        return { error, before, after: document.getElementById('behaviorStripLabels').textContent };
    });
    check(!noFrames.error, 'lanes and film come from the labels alone: ' + (noFrames.error || 'no frame was read'));
    check(noFrames.before === noFrames.after && noFrames.after.includes('rear'), 'and the same lanes show');

    console.log('\n-- 11. clicks and drags during playback --');
    const playing = () => page.evaluate(() => !!window.__lucid.state.isPlaying);
    await goTo(3000);
    await page.click('#btnPlay');
    check(await playing(), '(video-less playback running)');
    const g5 = await slotGeom();
    let f0 = await frame();
    await page.mouse.click(g5.v.x + g5.v.w / 4, g5.v.y + 7);
    await paint();
    const afterClick = await frame();
    check(await playing(), 'a click keeps a playing project playing, as the seekbar does');
    check(afterClick < f0, `from where it was clicked (${f0} → ${afterClick})`);
    await page.waitForFunction((f) => window.__lucid.state.currentFrame > f, afterClick, { timeout: 10000 });
    check(true, 'and the frame keeps advancing');
    await page.mouse.move(g5.v.x + 200, g5.v.y + 7);
    await page.keyboard.down('Shift');
    await page.mouse.down();
    check(!(await playing()), 'Shift+drag pauses: the window would move under the pointer');
    await page.mouse.up();
    await page.keyboard.up('Shift');
    await paint();
    await page.keyboard.press('Escape');
    await paint();

    // The real video path: VideoController + fake decoders, as seekbar-click-keeps-playback.
    await page.evaluate(async (T) => {
        const pd = await import('/pose/pose-data.js');
        const AS = await import('/ui/app-state.js');
        const sp = await import('/ui/sessions-panes.js');
        const K = [[600, 0, 320], [0, 600, 240], [0, 0, 1]];
        const cams = ['camA', 'camB'].map((n, i) => new pd.Camera(n, K, [0, 0, 0, 0, 0], [0, 0.2 * i, 0], [20 * i, 0, 0], [640, 480]));
        const session = new pd.Session(cams, new pd.Skeleton('sk', ['a', 'b'], [[0, 1]]), [], 'Video');
        session.addIdentity('id_0'); session.addIdentity('id_1');
        const fake = () => {
            const src = document.createElement('canvas');
            src.width = 64; src.height = 48;
            const d = {
                samples: new Array(T), _fps: 30, _videoReady: true, playing: false, pos: 0, t0: 0,
                async getFrame() { return createImageBitmap(src); },
                drawCurrentFrame() { return true; },
                getCurrentFrameIndex() { return d.playing ? Math.min(T - 1, d.pos + Math.floor((performance.now() - d.t0) * 30 / 1000)) : d.pos; },
                seekNativeSettled(f) { d.pos = f; return Promise.resolve(); },
                playNative() { d.playing = true; d.t0 = performance.now(); },
                pauseNative() { if (d.playing) d.pos = d.getCurrentFrameIndex(); d.playing = false; },
                releaseFrames() {},
            };
            return d;
        };
        AS.state.sessions = [session]; AS.state.session = session; AS.state.activeSessionIdx = 0;
        AS.state.totalFrames = T; AS.state.fps = 30; AS.state.currentFrame = 0;
        AS.state.views = cams.map(c => ({ name: c.name, videoWidth: 640, videoHeight: 480, canvas: null, decoder: fake() }));
        AS.state.videoFiles = cams.map(c => ({ name: c.name, assignedCamera: c.name }));
        AS.state.viewMode = 'grid'; AS.state.singleViewIndex = 0;
        AS.paneManager.clearAll();
        sp.populateViewStrip();
        AS.paneManager.addAllViewsAsGrid();
        const L = await import('/pose/behavior-labels.js');
        L.labelRange(session, 'b1', 'id_0', '', 2000, 2100, true);
        (await import('/ui/info-panel.js')).updateInfoPanel();
        (await import('/ui/behavior-coding.js')).requestBehaviorRepaint({ labels: true });
    }, T);
    await paint();
    await page.evaluate(() => window.__lucid.videoController.seekToFrame(2050));
    await paint();
    const g6 = await slotGeom();
    await page.mouse.click(g6.mid(2010, await frame()), g6.v.y + 7);
    await page.waitForFunction(() => {
        const vc = window.__lucid.videoController;
        return !vc._isSeeking && vc._scrubTarget === null && !vc.resumePending;
    }, null, { timeout: 10000 });
    check(await frame() === 2010, 'with video: a click lands on its frame');
    check(!(await playing()), 'and a paused video stays paused');
    await page.evaluate(() => window.__lucid.videoController.startPlayback());
    await page.waitForFunction(() => window.__lucid.state.isPlaying, null, { timeout: 5000 });
    const f1 = await frame();
    const g7 = await slotGeom();
    await page.mouse.click(g7.v.x + g7.v.w / 4, g7.v.y + 7);
    await page.waitForFunction(() => window.__lucid.state.isPlaying && !window.__lucid.videoController.resumePending, null, { timeout: 10000 });
    const f2 = await frame();
    check(f2 < f1, `with video: a click keeps it playing, from there (${f1} → ${f2})`);
    await page.waitForFunction((f) => window.__lucid.state.currentFrame > f, f2, { timeout: 10000 });
    check(true, 'and the frame keeps advancing');
    // It follows the real playback loop, and lands with it.
    await page.evaluate(() => window.__lucid.videoController.seekToFrame(2040));
    await paint();
    await page.evaluate(() => window.__lucid.videoController.startPlayback());
    await page.waitForFunction(() => window.__lucid.state.isPlaying && window.__lucid.state.currentFrame > 2044, null, { timeout: 10000 });
    const whilePlaying = (await runs())[0];
    await page.waitForFunction(() => window.__lucid.state.currentFrame > 2056, null, { timeout: 10000 });
    const laterPlaying = (await runs())[0];
    check(laterPlaying.x < whilePlaying.x - 2 * g7.ppf, `the strip moves with playback (${whilePlaying.x.toFixed(1)} → ${laterPlaying.x.toFixed(1)})`);
    await page.evaluate(() => window.__lucid.videoController.stopPlayback());
    await paint();
    const stoppedOn = await frame();
    const stopped = (await runs())[0];
    check(near(stopped.x, g7.left(2000, stoppedOn), 1.01), `and stops with its bars where the frame says (frame ${stoppedOn})`);

    console.log('\n-- 12. the session, and a reopened file --');
    const switched = await page.evaluate(async () => {
        const pd = await import('/pose/pose-data.js');
        const AS = await import('/ui/app-state.js');
        const L = await import('/pose/behavior-labels.js');
        const S = await import('/ui/behavior-strip.js');
        const first = AS.state.session;
        const other = new pd.Session([], first.skeleton, [], 'Other');
        other.addIdentity('id_7');
        L.labelRange(other, 'b4', 'id_7', '', 2100, 2100, true);
        AS.state.session = other;
        S.updateBehaviorStripFrame(AS.state.currentFrame);       // what a frame change after a switch does
        const lanes = document.getElementById('behaviorStripLabels').textContent;
        AS.state.session = first;
        S.updateBehaviorStripFrame(AS.state.currentFrame);
        return { lanes, back: document.getElementById('behaviorStripLabels').textContent };
    });
    check(switched.lanes === 'id_7jump', 'another session shows its own lanes');
    check(switched.back === 'id_0rear', 'and coming back, the first session\'s');
    const closed0 = await page.evaluate(() => document.getElementById('behaviorStrip').hidden);
    check(!closed0, '(still on show)');
    // A project saved with labels, reopened through both real readers, with its videos (the
    // real Load Videos) and, on a lazy project, without: no frame is resident to read from.
    const saved = await page.evaluate(async () => {
        const pd = await import('/pose/pose-data.js');
        const AS = await import('/ui/app-state.js');
        const L = await import('/pose/behavior-labels.js');
        const fileio = await import('/import-export/file-io.js');
        const NODES = ['Nose', 'Tail'], CAMS = ['cam0', 'cam1'], T = 60;
        const mtx = [[1000, 0, 128], [0, 1000, 128], [0, 0, 1]];
        const cams = CAMS.map((n, i) => new pd.Camera(n, mtx, [0, 0, 0, 0, 0], [0, 0.1 * i, 0], [10 * i, 0, -1000], [256, 256]));
        const s = new pd.Session(cams, new pd.Skeleton('m', NODES, [[0, 1]]), ['t0', 't1'], 'Saved');
        const ids = ['id_0', 'id_1'].map(nm => s.addIdentity(nm).id);
        for (let t = 0; t < T; t++) {
            const fg = new pd.FrameGroup(t);
            s.addFrameGroup(fg);
            s.instanceGroups.set(t, ids.map((id, k) => {
                const g = new pd.InstanceGroup(t * 10 + k + 1, id);
                for (const cn of CAMS) {
                    const inst = new pd.Instance(NODES.map((_, n) => [50 + 2 * n + 80 * k, 60 + n]), k, 'predicted', 0.9);
                    g.addInstance(cn, inst);
                    fg.addInstance(cn, inst);
                }
                g.points3d = new Float64Array(6);
                return g;
            }));
        }
        L.labelRange(s, 'b1', 'id_0', '', 10, 30, true);
        L.labelRange(s, 'b3', 'id_1', 'id_0', 40, 50, true);
        AS.state.sessions = [s]; AS.state.activeSessionIdx = 0; AS.state.session = s; AS.state.totalFrames = T;
        const views = CAMS.map(n => ({ name: n, videoWidth: 256, videoHeight: 32, frameCount: T }));
        const files = CAMS.map(n => ({ name: n, assignedCamera: n, videoPath: n + '.mp4' }));
        const u = new Uint8Array(await window.SleapIO.saveSlpToBytes(fileio.buildSlpLabelsAllViews(s, views, files)));
        let bin = '';
        for (let i = 0; i < u.length; i += 8192) bin += String.fromCharCode.apply(null, u.subarray(i, i + 8192));
        return btoa(bin);
    });
    const clip = fs.readFileSync(path.join(repoRoot, 'tests', 'fixtures', 'barcode-60', 'barcode-60.mp4'));
    for (const route of ['eager', 'lazy']) {
        const p = await browser.newPage({ viewport: { width: 1440, height: 900 } });
        p.on('pageerror', e => errs.push(route + ' pageerror: ' + String(e).slice(0, 300)));
        await p.goto(`http://localhost:${PORT}/index.html`);
        await p.waitForFunction(() => window.__lucid && window.__lucid.state && window.SleapIO, null, { timeout: 30000 });
        // The strip is no part of the info panel: one route with the panel collapsed.
        if (route === 'lazy') await p.click('#infoPanelToggleBtn');
        await p.evaluate(async ({ route, slp }) => {
            const skip = setInterval(() => {
                if (document.getElementById('videoSelectOverlay')) document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
            }, 50);
            const file = new File([Uint8Array.from(atob(slp), c => c.charCodeAt(0))], 'saved.slp');
            if (route === 'eager') await (await import('/import-export/slp-import.js')).handleLoadSlpFile(file);
            else await (await import('/loading/session-loader.js')).handleLoadProjectSlpLazy(file);
            clearInterval(skip);
        }, { route, slp: saved });
        await p.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
        const noVideo = await p.evaluate(() => ({ lazy: !!window.__lucid.state.session.lazyLoader, frames: window.__lucid.state.totalFrames,
                                                  hidden: document.getElementById('behaviorStrip').hidden, lanes: document.getElementById('behaviorStripLabels').textContent }));
        check(noVideo.lazy === (route === 'lazy'), `${route}: reopened that way (${noVideo.frames} frames without its videos)`);
        check(noVideo.frames > 0 ? (!noVideo.hidden && noVideo.lanes === 'id_0rearid_0sniffid_1sniff') : noVideo.hidden,
            `${route}, no video: ` + (noVideo.frames > 0 ? 'its labels show at once, with nothing to do' : 'nothing to seek through, so no strip yet'));
        p.once('filechooser', fc => fc.setFiles(['cam0', 'cam1'].map(nm => ({ name: nm + '.mp4', mimeType: 'video/mp4', buffer: clip }))));
        await p.evaluate(() => document.getElementById('btnLoadVideos').click());
        await p.waitForFunction(() => window.__lucid.state.views.filter(v => v.decoder && v.decoder.samples && v.decoder.samples.length).length === 2 &&
            document.getElementById('loadingOverlay').classList.contains('hidden'), null, { timeout: 60000 });
        await p.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
        const withVideo = await p.evaluate(() => {
            const r = document.querySelector('.behavior-strip-lanes').getBoundingClientRect();
            return { hidden: document.getElementById('behaviorStrip').hidden, lanes: document.getElementById('behaviorStripLabels').textContent, x: r.left, y: r.top, w: r.width };
        });
        check(!withVideo.hidden && withVideo.lanes === 'id_0rearid_0sniffid_1sniff', `${route}: with its videos, the reopened labels are on the strip: ${withVideo.lanes}`);
        // ±5 s at 60 fps: 300 frames each side of the playhead, 601 in all.
        const ppf = withVideo.w / 601;
        await p.mouse.click(withVideo.x + withVideo.w / 2 + 20 * ppf, withVideo.y + 7);
        await p.waitForFunction(() => {
            const vc = window.__lucid.videoController;
            return !vc._isSeeking && vc._scrubTarget === null && window.__lucid.state.currentFrame === 20;
        }, null, { timeout: 15000 }).catch(() => {});
        check(await p.evaluate(() => window.__lucid.state.currentFrame) === 20, `${route}: a click seeks a real video, lazily or not`);
        await p.evaluate(async () => (await import('/import-export/save-load.js')).newProject(true));
        await p.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
        check(await p.evaluate(() => document.getElementById('behaviorStrip').hidden), `${route}: File ▸ New Project takes the strip away`);
        await p.close();
    }
    const closed = await page.evaluate(async () => {
        (await import('/ui/app-state.js')).state.session = null;
        (await import('/ui/behavior-strip.js')).updateBehaviorStripFrame(0);
        return document.getElementById('behaviorStrip').hidden;
    });
    check(closed, 'no session, no strip');

    console.log('\n-- 13. more lanes than fit --');
    await boot();
    await page.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        const pd = await import('/pose/pose-data.js');
        const eth = await import('/pose/behavior-ethogram.js');
        const L = await import('/pose/behavior-labels.js');
        const s = AS.state.session;
        s.addIdentity('id_3'); s.addIdentity('id_4');
        for (const n of ['rear', 'groom', 'walk', 'dig']) eth.addBehavior({ name: n, key: n[0] });
        for (let a = 0; a < 5; a++) for (let b = 1; b <= 4; b++) L.labelRange(s, 'b' + b, 'id_' + a, '', 100 + 5 * a, 200 + 7 * b, true);
        (await import('/ui/info-panel.js')).updateInfoPanel();
        (await import('/ui/behavior-coding.js')).requestBehaviorRepaint({ labels: true });
    });
    await goTo(150);
    const body = () => page.evaluate(() => {
        const b = document.getElementById('behaviorStripBody');
        const r = b.getBoundingClientRect();
        const visible = [...document.querySelectorAll('.behavior-strip-label')].filter(l => {
            const q = l.getBoundingClientRect();
            return q.top >= r.top - 0.5 && q.bottom <= r.bottom + 0.5;
        }).map(l => l.textContent);
        return { top: b.scrollTop, height: r.height, whole: b.scrollHeight, visible };
    });
    const b0 = await body();
    check(b0.whole === 20 * 14 && b0.height === 8 * 14, `the strip stays slim: 8 of the 20 lanes on show (${b0.height} of ${b0.whole} px)`);
    check((await page.textContent('#behaviorStripTitle')).startsWith('20 lanes'), 'and it says how many there are, and how to reach them');
    const v13 = await box('.behavior-strip-lanes');
    await page.mouse.move(v13.x + 50, (await box('#behaviorStripBody')).y + 5);
    await page.keyboard.down('Shift');
    await page.mouse.wheel(0, 56);
    await page.keyboard.up('Shift');
    await paint();
    const b1 = await body();
    check(b1.top === 56 && (await page.textContent('#behaviorStripWindow')) === '±5 s', 'Shift+wheel scrolls the lanes, and does not zoom');
    await page.mouse.move(v13.x + v13.w / 2, (await box('#behaviorStripBody')).y + 5);
    const scrolled = await page.textContent('#behaviorStripTooltip');
    check(scrolled.includes('rear · id_1'), 'the pointer then reads the scrolled lanes: ' + scrolled);
    await page.mouse.click(300, 300);
    await page.keyboard.press('Meta+Shift+L');
    await page.keyboard.press('5');
    await paint();
    const b2 = await body();
    check(b2.visible.length === 8 && b2.visible.some(t => t.startsWith('id_4')), 'picking an animal brings its lanes into view: ' + b2.visible[0] + ' …');
    const before5 = b2.top;
    await page.keyboard.press('4');
    await paint();
    const b3 = await body();
    check(b3.visible.some(t => t.startsWith('id_3')) && b3.top === before5, 'an animal already in view moves nothing');
    await page.keyboard.press('Escape');
    await paint();

    console.log('\n-- 14. on a sharper screen --');
    const hi = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });
    const hp = await hi.newPage();
    hp.on('pageerror', e => errs.push('pageerror: ' + String(e).slice(0, 300)));
    await hp.goto(`http://localhost:${PORT}/index.html`);
    await hp.waitForFunction(() => window.__lucid && window.__lucid.state, null, { timeout: 30000 });
    const sharp = await hp.evaluate(async ({ FPS, T }) => {
        const pd = await import('/pose/pose-data.js');
        const AS = await import('/ui/app-state.js');
        const session = new pd.Session([], new pd.Skeleton('m', ['Nose', 'Tail'], [[0, 1]]), [], 'Sharp');
        session.addIdentity('id_0');
        AS.state.sessions = [session]; AS.state.activeSessionIdx = 0; AS.state.session = session;
        AS.state.totalFrames = T; AS.state.currentFrame = 0; AS.state.views = []; AS.state.fps = FPS;
        AS.timeline.setTotalFrames(T); AS.timeline.setData(session);
        (await import('/pose/behavior-ethogram.js')).addBehavior({ name: 'rear', key: 'r' });
        (await import('/pose/behavior-labels.js')).labelRange(session, 'b1', 'id_0', '', 2950, 3010, true);
        (await import('/pose/initialization.js')).navigateToFrame(3001);
        (await import('/ui/info-panel.js')).updateInfoPanel();
        await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
        const c = document.getElementById('behaviorStripCanvas');
        const box = c.getBoundingClientRect();
        const row = c.getContext('2d').getImageData(0, 4, c.width, 1).data;
        const alphas = new Set();
        for (let i = 3; i < row.length; i += 4) alphas.add(row[i]);
        return { width: c.width, css: box.width, height: c.height, alphas: Array.from(alphas) };
    }, { FPS, T });
    await hi.close();
    check(sharp.width === Math.round(sharp.css * 2) && sharp.height === 28, `the canvas has a pixel per device pixel (${sharp.width} for ${sharp.css} px)`);
    check(sharp.alphas.every(a => a === 0 || a === 255), 'and a bar\'s edges are sharp: no half-covered pixels');

    check(errs.length === 0, 'no page errors' + (errs.length ? ': ' + errs.slice(0, 3).join(' | ') : ''));
} catch (e) {
    console.error(e);
    fails++;
} finally {
    if (browser) await browser.close();
    server.kill();
}

console.log(fails ? `\n✗ ${fails} failed` : '\n✓ behavior strip OK');
process.exit(fails ? 1 : 0);
