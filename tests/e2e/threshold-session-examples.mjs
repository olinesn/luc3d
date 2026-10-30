/**
 * threshold-session-examples.mjs — the threshold card's "In this session" panel
 * (ui/threshold-examples-panel.js) in Settings ▸ Default Triangulation.
 *
 * Asserted, on a synthetic 3-camera project with one badly wrong prediction and a
 * stand-in video decoder:
 *  1. Opening Settings on another panel solves nothing; the triangulation panel
 *     does, then shows a fixed set of examples, opening on a large error.
 *  2. The slider never changes the example, the tiles' cameras or the images, and
 *     decodes nothing: only the verdicts move, here dropping the wrong camera.
 *  3. Neither does the DLT / Refined switch, though the errors are its own.
 *  4. The arrows walk the set, also from the keyboard, and the ends disable them.
 *  5. An example decodes its frame from each camera's video, and the newest request
 *     wins when arrows are clicked quickly.
 *  6. Trying values changes no setting and nothing in the session.
 *  7. The large tile starts on the largest error, a click on a small tile enlarges
 *     it, and the pick holds through the slider, the switch and the arrows.
 *  8. The card keeps one size, its text and panel side by side, stacked in a narrow
 *     window.
 *  9. No session: says so, and the figure still works.
 *
 * Run: node tests/e2e/threshold-session-examples.mjs   (PORT=… to move the server)
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8322);
let fails = 0;
const check = (c, m) => { console.log((c ? '  ✓ ' : '  ✗ ') + m); if (!c) fails++; };

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1200));

let browser;
try {
    browser = await chromium.launch();
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errs = [];
    page.on('pageerror', e => errs.push(String(e).slice(0, 300)));
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state, null, { timeout: 30000 });

    const open = (panel) => page.evaluate(async (p) => (await import('/ui/settings-modal.js')).showSettingsModal(p), panel);
    const close = () => page.evaluate(() => document.querySelector('.settings-modal-close').click());
    const read = () => page.evaluate(() => {
        const q = (s) => document.querySelector('.thr-ex ' + s);
        return {
            pos: q('.thr-ex-pos').textContent,
            where: q('.thr-ex-where').textContent,
            result: q('.thr-ex-result').textContent,
            summary: q('.thr-ex-summary').textContent,
            message: q('.thr-ex-message').hidden ? '' : q('.thr-ex-message').textContent,
            prev: q('.thr-ex-prev').disabled,
            next: q('.thr-ex-next').disabled,
            tiles: Array.from(document.querySelectorAll('.thr-ex-tile')).map(t => ({ cam: t.dataset.camera, state: t.dataset.state, err: t.querySelector('.thr-ex-err').textContent })),
        };
    });
    const ready = () => page.waitForFunction(() => { const m = document.querySelector('.thr-ex-message'); return m && m.hidden; }, null, { timeout: 30000 });
    const tryBox = (v) => page.fill('.tri-thr [data-try="threshold"] .fig-num', String(v));

    const setup = await page.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        const { buildBaRigFixture } = await import('/tests/e2e/fixtures/ba-rig-fixture.js');
        const F = await buildBaRigFixture({ frames: 24, cycle: 8 });
        const s = F.mkSession('T', { eager: true });
        // Every group must exist with all its members for the panel to find them.
        const TRI = await import('/pose/triangulation.js');
        AS.state.sessions = [s]; AS.state.activeSessionIdx = 0; AS.state.session = s;
        AS.state.totalFrames = 24; AS.state.currentFrame = 0;
        AS.state.triangulationResults = new Map();
        await TRI.triangulateAllFrames('dlt');
        // One badly wrong prediction: frame 7, first group, node 2, first camera.
        const g = s.instanceGroups.get(7)[0];
        const cam = s.cameras[0].name;
        const inst = g.getInstance(cam);
        const p = inst.getPoint(2);
        inst.setPoint(2, p[0] + 180, p[1] - 90);
        // A stand-in video: each camera's frame says which one it is in its colour.
        window.__decodes = [];
        const canvas = (cam, f) => {
            const c = new OffscreenCanvas(1280, 1024), x = c.getContext('2d');
            x.fillStyle = `rgb(${(f * 9) % 255}, 200, ${40 * s.cameras.findIndex(k => k.name === cam)})`;
            x.fillRect(0, 0, 1280, 1024);
            return c;
        };
        AS.state.views = s.cameras.map(c => ({
            name: c.name,
            decoder: { getFrame: async (f) => { window.__decodes.push(c.name + ':' + f); return canvas(c.name, f); } },
        }));
        window.__T = { AS, s, snapshot: () => JSON.stringify([...s.instanceGroups.get(7)[0].instances].map(([n, i]) => [n, Array.from(i._xy)])) };
        return window.__T.snapshot();
    });
    const window0 = setup;

    const heldStill = () => page.evaluate(() => ({
        where: document.querySelector('.thr-ex-where').textContent,
        pos: document.querySelector('.thr-ex-pos').textContent,
        cams: Array.from(document.querySelectorAll('.thr-ex-tile')).map(t => t.dataset.camera).join(),
        // A corner of each tile: past the marks, so it is the image.
        image: Array.from(document.querySelectorAll('.thr-ex-tile canvas')).map(c => Array.from(c.getContext('2d').getImageData(3, 3, 1, 1).data).join('.')).join(' '),
        decodes: window.__decodes.length,
    }));
    const settled = async () => { await page.waitForTimeout(350); };

    console.log('1. opening');
    await open('wizard');
    check(await page.evaluate(() => window.__decodes.length) === 0, 'another panel decodes nothing');
    await page.click('.settings-nav-item[data-panel="triangulation"]');
    await ready();
    let r = await read();
    check(r.tiles.length === 3, `a tile per camera, here fewer than six (${r.tiles.length})`);
    check(r.pos === 'Example 12 of 12', `a fixed set of 12, opening on the first large error (${r.pos})`);
    check(/Frame 8 /.test(r.where), `here the wrong prediction, on frame 8 (${r.where})`);
    check(/threshold off/.test(r.result) && r.tiles.every(t => t.state === 'off'), `nothing is dropped with it off (${r.result})`);

    console.log('2. the slider holds the example still');
    const beforeSetting = await page.evaluate(async () => (await import('/ui/settings.js')).getTrackingThreshold('reprojErrorThreshold'));
    await settled();
    const still = await heldStill();
    check(still.image.split(' ')[0] !== '17.17.17.255', `the tiles show the video (${still.image})`);
    for (const t of [640, 200, 100, 40, 10, 2, 0]) {
        await tryBox(t);
        await page.waitForTimeout(40);
        const now = await heldStill();
        check(now.where === still.where && now.pos === still.pos && now.cams === still.cams,
            `at ${t} px: the same node and cameras (${now.where})`);
        check(now.image === still.image, `at ${t} px: the same images`);
    }
    await settled();
    check((await heldStill()).decodes === still.decodes, `and nothing is decoded again (${still.decodes} decodes)`);
    await tryBox(200);
    await page.waitForTimeout(40);
    r = await read();
    check(!r.tiles.some(t => t.state === 'drop') && /nothing is dropped/.test(r.result), `at 200 px nothing is dropped (${r.result})`);
    await tryBox(40);
    await page.waitForTimeout(40);
    r = await read();
    check(r.tiles.some(t => t.state === 'drop') && /is dropped/.test(r.result), `at 40 px, on the same example, a camera is (${r.tiles.map(t => t.state)}; ${r.result})`);
    check(/At 40 px, 1 of [\d,]+ nodes solved here has a camera dropped/.test(r.summary), `and the summary counts the session (${r.summary})`);

    console.log('3. so does DLT / Refined');
    const dltErrs = r.tiles.map(t => t.err).join();
    await page.click('.tri-thr [data-demo-method="ba"]');
    await page.waitForTimeout(40);
    let now = await heldStill();
    r = await read();
    check(now.where === still.where && now.cams === still.cams && now.image === still.image, `Refined: the same node, cameras and images (${now.where})`);
    check(r.tiles.map(t => t.err).join() !== dltErrs, `with its own errors (${r.tiles.map(t => t.err).join()} vs ${dltErrs})`);
    await page.waitForFunction(() => !/Counting/.test(document.querySelector('.thr-ex-summary').textContent), null, { timeout: 10000 });
    now = await heldStill();
    check(now.image === still.image && now.pos === still.pos, 'and its count arriving changes nothing else');
    await page.click('.tri-thr [data-demo-method="dlt"]');
    await page.waitForTimeout(40);
    now = await heldStill();
    r = await read();
    check(now.image === still.image && r.tiles.map(t => t.err).join() === dltErrs, 'back to DLT: as it was');
    await settled();
    check((await heldStill()).decodes === still.decodes, 'with nothing decoded again');

    console.log('4. the arrows');
    await page.click('.thr-ex-prev');
    r = await read();
    check(r.pos === 'Example 11 of 12' && !r.tiles.some(t => t.state === 'drop'), `‹ goes to a smaller error (${r.pos})`);
    await page.focus('.thr-ex');
    await page.keyboard.press('ArrowRight');
    r = await read();
    check(r.pos === 'Example 12 of 12' && r.next, `the → key goes back, and the end disables › (${r.pos})`);
    check(await page.evaluate(() => !!document.querySelector('.settings-overlay')), 'the arrow keys leave the window open');
    for (let i = 0; i < 12; i++) await page.evaluate(() => document.querySelector('.thr-ex-prev').click());
    r = await read();
    check(r.prev && !r.next && r.pos === 'Example 1 of 12', `the other end disables ‹ (${r.pos})`);

    console.log('5. video');
    await settled();
    await page.evaluate(() => { window.__decodes.length = 0; });
    await page.click('.thr-ex-next');
    await page.waitForFunction(() => window.__decodes.length >= 3, null, { timeout: 5000 });
    await settled();
    const f2 = (await read()).where.match(/Frame ([\d,]+)/)[1].replace(/,/g, '') - 1;
    const seen = await page.evaluate(() => window.__decodes.slice());
    check(seen.length === 3 && seen.every(k => k.endsWith(':' + f2)), `each camera decodes the example's frame (${seen.join()})`);
    await page.evaluate(() => { window.__decodes.length = 0; for (let i = 0; i < 10; i++) document.querySelector('.thr-ex-next').click(); });
    await settled();
    const n = await page.evaluate(() => window.__decodes.length);
    check(n <= 3, `quick clicks decode only the last example (${n} decodes)`);

    console.log('6. read-only');
    const after = await page.evaluate(async () => ({ snap: window.__T.snapshot(), set: (await import('/ui/settings.js')).getTrackingThreshold('reprojErrorThreshold') }));
    check(after.snap === window0 && after.set === beforeSetting, 'the session and the setting are untouched');

    console.log('7. the large tile');
    for (let i = 0; i < 12; i++) await page.evaluate(() => document.querySelector('.thr-ex-next').click());
    await tryBox(40);
    await page.waitForTimeout(50);
    const large = () => page.evaluate(() => {
        const l = document.querySelector('.thr-ex-large');
        return {
            cam: l.dataset.camera, state: l.dataset.state, name: l.querySelector('.thr-ex-cam').textContent, err: l.querySelector('.thr-ex-err').textContent,
            pressed: Array.from(document.querySelectorAll('.thr-ex-tile[aria-pressed="true"]')).map(t => t.dataset.camera),
            w: l.querySelector('canvas').getBoundingClientRect().width,
            small: document.querySelector('.thr-ex-tile canvas').getBoundingClientRect().width,
            leftOf: document.querySelector('.thr-ex-grid').getBoundingClientRect().right <= l.getBoundingClientRect().left,
        };
    });
    let L = await large();
    r = await read();
    const dropped = r.tiles.find(t => t.state === 'drop');
    check(dropped && L.cam === dropped.cam && L.state === 'drop', `it starts on the largest error, here the dropped camera (${L.cam})`);
    check(L.pressed.length === 1 && L.pressed[0] === L.cam, `and that tile is marked (${L.pressed})`);
    check(L.leftOf && L.w > L.small * 1.5, `small tiles on the left, the large one on the right (${Math.round(L.small)} and ${Math.round(L.w)} px)`);
    await tryBox(0);
    await page.waitForTimeout(40);
    const offCam = (await large()).cam;
    check(offCam === L.cam, `the slider does not move it (${offCam})`);
    await tryBox(40);
    await page.waitForTimeout(40);
    const other = r.tiles.find(t => t.cam !== L.cam);
    await page.click(`.thr-ex-tile[data-camera="${other.cam}"]`);
    L = await large();
    check(L.cam === other.cam && L.name === other.cam && L.err === other.err && L.pressed[0] === other.cam, `clicking a tile enlarges it (${L.cam}: ${L.err})`);
    await tryBox(10);
    await page.click('.tri-thr [data-demo-method="ba"]');
    await page.waitForTimeout(40);
    await page.click('.thr-ex-prev');
    L = await large();
    check(L.cam === other.cam, `the pick holds through the slider, the switch and the arrows (${L.cam})`);
    await page.click('.tri-thr [data-demo-method="dlt"]');
    await page.click('.thr-ex-next');
    await settled();
    check(await page.evaluate(() => document.querySelector('.thr-ex-large canvas').getContext('2d').getImageData(4, 4, 1, 1).data[1]) > 100, 'the large tile draws the video too');

    console.log('8. layout');
    const layout = () => page.evaluate(() => {
        const a = document.querySelector('.tri-thr-intro').getBoundingClientRect(), b = document.querySelector('.thr-ex').getBoundingClientRect();
        const c = document.querySelector('.tri-thr').getBoundingClientRect();
        return { left: a.right <= b.left, top: Math.round(a.top - b.top), half: Math.round(100 * a.width / c.width), h: Math.round(c.height), w: Math.round(c.width), stacked: b.top >= a.bottom };
    });
    const l = await layout();
    check(l.left && l.top === 0 && l.half > 40 && l.half < 55, `text on the left, about half width, the panel beside it (${JSON.stringify(l)})`);
    const sizes = new Set();
    for (const t of [0, 5, 12, 40, 90, 200, 640]) {
        await tryBox(t);
        for (const dir of ['.thr-ex-prev', '.thr-ex-next', '.thr-ex-next']) {
            await page.evaluate((s) => document.querySelector(s).click(), dir);
            const x = await layout();
            sizes.add(x.w + 'x' + x.h);
        }
    }
    check(sizes.size === 1, `one card size across values and examples (${[...sizes].join(' | ')})`);
    await page.setViewportSize({ width: 800, height: 900 });
    await page.waitForTimeout(100);
    const narrow = await layout();
    check(narrow.stacked, `stacked in a narrow window (${JSON.stringify(narrow)})`);
    await page.setViewportSize({ width: 1440, height: 1000 });
    await close();

    console.log('9. no session');
    await page.evaluate(() => { const S = window.__T.AS.state; S.sessions = []; S.session = null; S.activeSessionIdx = -1; S.views = []; });
    await open('triangulation');
    r = await read();
    check(/Load a session/.test(r.message) && r.tiles.length === 0, `says so (${r.message})`);
    await tryBox(25);
    const big = await page.textContent('.tri-thr .fig-big');
    check(/^25 px/.test(big), `and the figure still works (${big})`);
    await close();

    check(errs.length === 0, 'no page errors' + (errs.length ? ': ' + errs.join(' | ') : ''));
} finally {
    if (browser) await browser.close();
    server.kill();
}
console.log(fails ? `${fails} FAILED` : 'all passed');
process.exit(fails ? 1 : 0);
