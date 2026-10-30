/**
 * threshold-session-examples.mjs — the threshold card's "In this session" panel
 * (ui/threshold-examples-panel.js) in Settings ▸ Default Triangulation.
 *
 * Asserted, on a synthetic 3-camera project with one badly wrong prediction and a
 * stand-in video decoder:
 *  1. Opening Settings on another panel solves nothing; the triangulation panel
 *     does, then shows one tile per camera.
 *  2. Off opens at the largest error. Moving the slider jumps to the first node the
 *     value catches: its camera is marked dropped, and › and ‹ walk larger and
 *     smaller errors, also from the keyboard.
 *  3. The ends of the list disable the arrows.
 *  4. Trying Refined solves again; the shown errors are Refined's.
 *  5. Tiles show the example's frame from each camera's video, and the newest
 *     request wins when arrows are clicked quickly.
 *  6. Trying values changes no setting and nothing in the session.
 *  7. The card keeps one size as examples change, its text and panel side by side,
 *     stacked in a narrow window.
 *  8. No session: says so, and the figure still works.
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

    await page.evaluate(async () => {
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
    });

    console.log('1. opening');
    await open('wizard');
    const idle = await page.evaluate(() => window.__decodes.length);
    check(idle === 0, 'another panel decodes nothing');
    await page.click('.settings-nav-item[data-panel="triangulation"]');
    await ready();
    let r = await read();
    check(r.tiles.length === 3, `one tile per camera (${r.tiles.length})`);
    const total = +r.pos.split(' of ')[1].replace(/,/g, '');
    check(total > 100, `the session's nodes are listed (${r.pos})`);

    console.log('2. the slider and the arrows');
    check(r.pos.startsWith(r.pos.split(' of ')[1] + ' of') && r.next, `off opens at the largest error (${r.pos})`);
    check(/Frame 8 /.test(r.where) && /· [^·]+$/.test(r.where), `which is the wrong prediction, on frame 8 (${r.where})`);
    check(/threshold off/.test(r.result) && r.tiles.every(t => t.state === 'off'), `nothing is dropped with it off (${r.result})`);
    const beforeSetting = await page.evaluate(async () => (await import('/ui/settings.js')).getTrackingThreshold('reprojErrorThreshold'));
    await tryBox(40);
    await page.waitForTimeout(50);
    r = await read();
    const at40 = +r.pos.split(' of ')[0].replace(/,/g, '');
    check(r.tiles.some(t => t.state === 'drop') && /is dropped/.test(r.result), `at 40 px the first node it catches has a camera dropped (${r.tiles.map(t => t.state)}; ${r.result})`);
    check(/have a camera dropped at 40 px/.test(r.summary), `and the summary counts them (${r.summary})`);
    const worstAt = (tiles) => Math.max(...tiles.map(t => parseFloat(t.err)).filter(Number.isFinite));
    check(worstAt(r.tiles) > 0, 'the tiles give the errors');
    await page.click('.thr-ex-prev');
    r = await read();
    check(+r.pos.split(' of ')[0].replace(/,/g, '') === at40 - 1 && !r.tiles.some(t => t.state === 'drop'), `‹ shows a node 40 px keeps (${r.pos})`);
    await page.focus('.thr-ex');
    await page.keyboard.press('ArrowRight');
    r = await read();
    check(+r.pos.split(' of ')[0].replace(/,/g, '') === at40, `the → key goes back (${r.pos})`);
    check(await page.evaluate(() => !!document.querySelector('.settings-overlay')), 'and the arrow keys leave the window open');

    console.log('3. the ends');
    await tryBox(0);
    await page.waitForTimeout(50);
    r = await read();
    check(r.next && !r.prev, `at the largest: › disabled (${r.pos})`);
    for (let i = 0; i < total + 1; i++) await page.evaluate(() => document.querySelector('.thr-ex-prev').click());
    r = await read();
    check(r.prev && !r.next && r.pos.startsWith('1 of'), `at the smallest: ‹ disabled (${r.pos})`);

    console.log('4. Refined');
    await page.click('.tri-thr [data-demo-method="ba"]');
    await ready();
    r = await read();
    const baLast = r.tiles.map(t => t.err).join();
    await page.click('.tri-thr [data-demo-method="dlt"]');
    await ready();
    const dltLast = (await read()).tiles.map(t => t.err).join();
    check(baLast !== dltLast, `Refined shows its own errors (${baLast} vs ${dltLast})`);

    console.log('5. video');
    await page.evaluate(() => { window.__decodes.length = 0; });
    await tryBox(40);
    await page.waitForFunction(() => window.__decodes.length >= 3, null, { timeout: 5000 });
    await page.waitForTimeout(150);
    const seen = await page.evaluate(() => {
        const c = document.querySelector('.thr-ex-tile canvas');
        const d = c.getContext('2d').getImageData(4, 4, 1, 1).data;
        return { decodes: window.__decodes.slice(), g: d[1] };
    });
    const f40 = (await read()).where.match(/Frame ([\d,]+)/)[1].replace(/,/g, '') - 1;
    check(seen.decodes.every(k => k.endsWith(':' + f40)), `each camera decodes the example's frame (${seen.decodes.join()})`);
    check(seen.g > 100, `and the tile draws it (green ${seen.g})`);
    await page.evaluate(() => { window.__decodes.length = 0; for (let i = 0; i < 30; i++) document.querySelector('.thr-ex-next').click(); });
    await page.waitForTimeout(400);
    const n = await page.evaluate(() => window.__decodes.length);
    check(n <= 3, `30 quick clicks decode only the last example (${n} decodes)`);

    console.log('6. read-only');
    const snap = await page.evaluate(() => window.__T.snapshot());
    await tryBox(12);
    await page.click('.thr-ex-prev');
    const after = await page.evaluate(async () => ({ snap: window.__T.snapshot(), set: (await import('/ui/settings.js')).getTrackingThreshold('reprojErrorThreshold') }));
    check(after.snap === snap && after.set === beforeSetting, 'the session and the setting are untouched');

    console.log('7. layout');
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

    console.log('8. no session');
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
