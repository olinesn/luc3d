/**
 * model-check-review.mjs — File ▸ Load Model Check… and the Model Check tab
 * (ui/model-check-panel.js, ui/model-check-highlight.js), in the real app.
 *
 * Fixture: tests/fixtures/sleap-multiview/multiview.slp (8 cameras, 3 frames, 2
 * hand-labelled animals) opened through Load SLP, with a view per camera (no
 * video), and tests/fixtures/model-check/scan.json (make_fixture.py): a model
 * check of that file whose distances come from its own labels. Six misread
 * animals: top f1 a1 (Ear_L read on animal 0's Ear_L + Nose on the background),
 * side f2 a0 (Nose on its own Head), mid f0 a1 (never detected), back f0 a0 (a
 * distance its labels do not fit) and two on Camera0_mid, which the project has
 * not got. Asserted:
 *  1. The menu item picks the file; the status line and the tab report 42 of 44
 *     rows matched (8 cameras by video file name) and name the 2 that are not.
 *  2. The list holds the misread animals, worst first (never detected first).
 *  3. M / Shift+M step through it (wrapping); an entry goes to its frame, solos
 *     its camera in single-view mode, lists each misread node with what it was
 *     read on, and says how its animal was matched: top f1 a1 verified by node
 *     positions; back f0 a0 not matched (labels do not fit); mid f0 a1 by file
 *     order only.
 *  4. The selected animal is marked on its view's own canvas, around that
 *     animal and not the other; leaving the frame clears it. The view zooms in
 *     on it and the neighbour it was read on (unless Zoom in is off), and the
 *     overlay names no node there (the rings name the misread ones).
 *  5. Split and cause filters narrow the list and are kept in localStorage.
 *  6. Loading changes nothing in the project (not marked unsaved).
 *  7. A camera more than one project camera fits is asked about; Esc cancels
 *     the load, a pick matches it.
 *  8. Lazy project (Load Project): rows on frames not loaded are matched from
 *     the store, and the mark appears once the frame is hydrated.
 *  9. Close forgets it, and brings back the 3D viewer and Timeline that
 *     loading hid (1.).
 * 10. An eager load (Load SLP) names a camera-frame's animals in the file's
 *     order, as the lazy store does (`_fileRow`, pose-data.js `tagFileRow`): the
 *     eager reader lists grouped instances before ungrouped ones.
 *
 * Run: node tests/e2e/model-check-review.mjs     (SHOT=/path.png saves a screenshot of an entry)
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8293);
const SLP = fs.readFileSync(path.join(repoRoot, 'tests/fixtures/sleap-multiview/multiview.slp'));
const SCAN_PATH = path.join(repoRoot, 'tests/fixtures/model-check/scan.json');
const SCAN = fs.readFileSync(SCAN_PATH, 'utf8');
let fails = 0;
const check = (c, m, extra) => {
    console.log((c ? '  ✓ ' : '  ✗ ') + m + (!c && extra !== undefined ? '  ' + JSON.stringify(extra).slice(0, 700) : ''));
    if (!c) fails++;
};

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1200));

let browser;
try {
    browser = await chromium.launch();
    const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
    const errs = [];
    page.on('pageerror', e => errs.push('pageerror: ' + String(e).slice(0, 300)));
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.SleapIO && window.h5wasm && window.__lucid, null, { timeout: 120000 });
    await page.evaluate((b64) => { window.__file = Uint8Array.from(atob(b64), c => c.charCodeAt(0)); }, SLP.toString('base64'));

    async function waitLoad() {
        for (let i = 0; i < 400; i++) {
            const s = await page.evaluate(() => {
                const b = [...document.querySelectorAll('button')].find(x => /Skip|Later/i.test(x.textContent || ''));
                if (b) b.click();
                return { done: window.__loadDone, prompt: !!document.getElementById('videoSelectOverlay') };
            });
            if (s.prompt) await page.keyboard.press('Escape');
            if (s.done) return s.done;
            await page.waitForTimeout(100);
        }
        return 'timeout';
    }
    // Views for every camera, with no video (the overlays and the mark need a view).
    const addViews = () => page.evaluate(async () => {
        const AS = window.__lucid, sp = await import('/ui/sessions-panes.js');
        const s = AS.state.session;
        AS.paneManager.clearAll();
        AS.state.views = s.cameras.map(c => ({ name: c.name, videoWidth: 1280, videoHeight: 1024, canvas: null }));
        AS.state.totalFrames = 3; AS.state.fps = 30;
        sp.populateViewStrip();
        AS.paneManager.addAllViewsAsGrid();
        // Zoom is the video controller's, which a load without videos leaves unset: build one as startup does.
        if (!AS.videoController) {
            const { VideoController } = await import('/loading/video.js'), { setVideoController } = await import('/ui/app-state.js');
            setVideoController(new VideoController(AS.state, { log: () => {} }));
        }
        AS.state.isDirty = false;
    });
    const loadSlp = async (lazy) => {
        await page.evaluate((lazy) => {
            window.__loadDone = false;
            (async () => {
                try {
                    if (lazy) await (await import('/loading/session-loader.js')).handleLoadProjectSlpLazy(new File([window.__file], 'multiview.slp'));
                    else await (await import('/import-export/slp-import.js')).handleLoadSlpFile(new File([window.__file], 'multiview.slp'));
                    window.__loadDone = true;
                } catch (e) { window.__loadDone = 'ERR ' + String(e && e.stack || e).slice(0, 400); }
            })();
        }, lazy);
        return waitLoad();
    };
    const panelText = () => page.evaluate(() => document.getElementById('modelCheckPanel').innerText);
    const entries = () => page.evaluate(() => [...document.querySelectorAll('#modelCheckList .model-check-row')].map(r => ({
        where: r.querySelector('.model-check-where').textContent, who: r.querySelector('.model-check-who').textContent,
        cur: r.classList.contains('is-current'), status: [...r.classList].find(c => c.startsWith('status-')),
    })));
    // An entry's own click (the list re-renders on every selection, so no coordinates).
    const clickEntry = (i) => page.evaluate((i) => document.querySelectorAll('#modelCheckList .model-check-row')[i].click(), i);
    const status = () => page.evaluate(() => document.getElementById('statusText').textContent);
    const frame = () => page.evaluate(() => window.__lucid.state.currentFrame);
    // The mark on each view's canvas: opaque pixels near each animal (bounding boxes from the session).
    const mark = () => page.evaluate(() => {
        const AS = window.__lucid, s = AS.state.session, f = AS.state.currentFrame;
        return AS.state.views.map(v => {
            const c = v.wrapper && v.wrapper.querySelector('.model-check-canvas');
            if (!c || !c.width) return { view: v.name, px: 0 };
            const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
            let px = 0, x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
            for (let y = 0; y < c.height; y++) for (let x = 0; x < c.width; x++) {
                if (!d[(y * c.width + x) * 4 + 3]) continue;
                px++; x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
            }
            const sc = c.width / 1280;
            return { view: v.name, px, box: px ? [x0 / sc, y0 / sc, x1 / sc, y1 / sc] : null };
        }).filter(m => m.px);
    });
    const userBoxes = (cam) => page.evaluate((cam) => {
        const s = window.__lucid.state.session, fg = s.frameGroups.get(window.__lucid.state.currentFrame);
        const list = [...fg.getInstances(cam), ...fg.getUnlinkedInstances(cam).map(u => u.instance)].filter(i => i.type === 'user');
        return list.map(i => { const xs = [], ys = []; for (let k = 0; k < i.numNodes; k++) if (i.hasPoint(k)) { xs.push(i.getX(k)); ys.push(i.getY(k)); }
            return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)]; });
    }, cam);

    // Every labelled camera-frame's hand-labelled instances, in the order the tab names them
    // "animal 0, 1, …" (first labelled node's x), and whether each carries its file row.
    const fileOrder = () => page.evaluate(async () => {
        const P = await import('/ui/model-check-panel.js'), s = window.__lucid.state.session, out = {};
        let tagged = 0, users = 0;
        for (const c of s.cameras) for (let f = 0; f < 3; f++) {
            const list = P.userInstancesAt(s, c.name, f);
            out[c.name + ':' + f] = list.map(u => Object.values(u.points)[0][0].toFixed(2)).join(',');
            list.forEach(u => { if (u.inst) { users++; if (Number.isInteger(u.inst._fileRow) || Number.isInteger(u.inst._rawInstIndex)) tagged++; } });
        }
        return { out, tagged, users };
    });

    console.log('\n-- eager project --');
    check(await loadSlp(false) === true, 'multiview.slp opens');
    await addViews();
    const eagerOrder = await fileOrder();
    check(eagerOrder.users > 0 && eagerOrder.tagged === eagerOrder.users,
        `every hand-labelled instance knows its row in the file (${eagerOrder.tagged} of ${eagerOrder.users})`);
    check(await page.evaluate(() => document.getElementById('modelCheckPanel').innerText.includes('No model check loaded')),
        'the tab says nothing is loaded yet');

    // ---- 1. the menu item and the report
    const panes = () => page.evaluate(() => ({
        viewport3d: !document.getElementById('viewport3dContainer').classList.contains('collapsed'),
        timeline: !document.getElementById('timelineContainer').classList.contains('collapsed'),
    }));
    const panesBefore = await panes();
    const chooser = page.waitForEvent('filechooser');
    await page.evaluate(() => document.getElementById('menuLoadModelCheck').click());
    await (await chooser).setFiles(SCAN_PATH);
    await page.waitForFunction(() => /Model check:/.test(document.getElementById('statusText').textContent), null, { timeout: 10000 });
    check(/42 of 44 rows matched to the project, 2 not; 6 misread animals/.test(await status()), 'status line reports matched and not', await status());
    check(await page.evaluate(() => document.querySelector('.panel-tab[data-tab="tabModelCheck"]').classList.contains('active')), 'the Model Check tab opens');
    let text = await panelText();
    check(/Matched 42 of 44 rows to the project \(8 cameras by video file name\)/.test(text), 'the tab says how the cameras matched', text.slice(0, 400));
    check(await page.evaluate(() => /Camera0_mid.*2 rows: no camera named Camera0_mid and no video named Camera0_mid\.mp4/
        .test(document.querySelector('.model-check-unmatched').textContent)), 'the unmatched camera is named, with why');
    check(/held-out:\s*4 of 4 animals misread/i.test(text) && /training:\s*2 of 40 animals misread/i.test(text), 'per-split counts');
    check(await page.evaluate(() => !window.__lucid.state.isDirty), 'loading the check leaves the project unchanged (not unsaved)');
    const panesLoaded = await panes();
    check(panesBefore.viewport3d && !panesLoaded.viewport3d && !panesLoaded.timeline,
        'loading hides the 3D viewer and the Timeline, for the video', { panesBefore, panesLoaded });

    // ---- 2. worst first
    let e = await entries();
    check(e.map(x => x.where.replace(/\s+/g, ' ')).join(' | ') === 'mid · frame 1 | back · frame 1 | top · frame 2 | Camera0_mid · frame 1 | Camera0_mid · frame 1 | side · frame 3',
        'misread animals only, never-detected first, then worst error', e.map(x => x.where));

    // ---- 3. stepping with the keyboard
    await page.mouse.click(500, 400);                       // focus the app, not a select
    await page.keyboard.press('m');
    await page.waitForTimeout(300);
    e = await entries();
    check(e[0].cur && await frame() === 0, 'M selects the first entry and goes to its frame');
    text = await panelText();
    check(/Matched in file order \(no distance to check it against\)/.test(text) && /found no animal here/.test(text),
        'a never-detected animal: matched by file order only, and says there is no reading', text.slice(-600));
    await page.keyboard.press('m'); await page.waitForTimeout(300);
    e = await entries();
    check(e[1].cur && e[1].status === 'status-unmatched', 'M: the next entry; labels that do not fit its distances are not matched');
    check(/labels edited since/.test(await panelText()), 'and it says why');
    check((await mark()).length === 0, 'an unmatched animal is not marked');
    await page.keyboard.press('m'); await page.waitForTimeout(400);
    e = await entries();
    check(e[2].cur && e[2].status === 'status-verified' && await frame() === 1, 'top frame 2: verified, on frame index 1');
    text = await panelText();
    check(/Ear_L read on animal 0 \(track_0\)'s Ear_L/.test(text), "names the neighbour the node was read on, with its project name", text.slice(-700));
    check(/Nose read on the background/.test(text), 'a background read');
    check(/node positions agree \(1 distance checked\)/.test(text), 'says how many distances were checked');
    check(/animal 1 · track_1/.test(e[2].who), 'which mouse: the sweep index and its project name', e[2].who);
    await page.waitForTimeout(200);
    const m1 = await mark(), boxes = await userBoxes('top');
    const inside = (b, B, pad) => b[0] >= B[0] - pad && b[1] >= B[1] - pad && b[2] <= B[2] + pad && b[3] <= B[3] + pad;
    check(m1.length === 1 && m1[0].view === 'top', 'the mark is on the camera of the entry only', m1.map(m => m.view));
    // The mark covers animal 1 (the target, with its label above) and reaches animal 0's Ear_L (the dashed line + its box).
    check(m1.length === 1 && boxes.length === 2 && m1[0].box[2] >= boxes[1][2] && m1[0].box[0] <= boxes[1][0] + 1,
        'it surrounds the misread animal', { mark: m1[0] && m1[0].box, boxes });
    // Zoom in: the view frames the animal and the one its Ear_L was read on, both whole on screen.
    const zoomOf = (cam) => page.evaluate((cam) => { const v = window.__lucid.state.views.find(w => w.name === cam); return v.zoom ? v.zoom.scale : 1; }, cam);
    const onScreen = (cam, B) => page.evaluate(([cam, B]) => {
        const v = window.__lucid.state.views.find(w => w.name === cam);
        const cell = v.canvas.closest('.video-cell').getBoundingClientRect(), c = v.canvas.getBoundingClientRect();
        const sx = (x) => c.left + x / v.videoWidth * c.width, sy = (y) => c.top + y / v.videoHeight * c.height;
        return B.every(b => sx(b[0]) >= cell.left && sx(b[2]) <= cell.right && sy(b[1]) >= cell.top && sy(b[3]) <= cell.bottom);
    }, [cam, B]);
    const z1 = await zoomOf('top');
    check(z1 > 1.2, 'selecting an entry zooms in on its animal', z1);
    check(await onScreen('top', boxes), 'the animal and its neighbour are both whole in the view');
    if (process.env.SHOT) await page.screenshot({ path: process.env.SHOT });
    await page.keyboard.press('ArrowRight'); await page.waitForTimeout(300);
    check((await mark()).length === 0, 'leaving the frame clears the mark');
    await page.keyboard.press('ArrowLeft'); await page.waitForTimeout(300);
    check((await mark()).length === 1, 'coming back redraws it');
    // Node names: the overlay names no node on the entry's view and frame (the rings name the misread
    // ones); the other views keep the Visibility setting (user label size 12 by default).
    const overlayNames = () => page.evaluate(async () => {
        const AS = window.__lucid, names = AS.state.session.skeleton.nodes.map(n => n.name || n);
        const seen = new Map(), orig = CanvasRenderingContext2D.prototype.fillText;
        CanvasRenderingContext2D.prototype.fillText = function (t) {
            const v = AS.state.views.find(w => w.overlayCanvas === this.canvas);
            if (v && names.includes(t)) seen.set(v.name, (seen.get(v.name) || 0) + 1);
            return orig.apply(this, arguments);
        };
        try { (await import('/ui/rendering.js')).drawAllOverlays(AS.state.currentFrame); }
        finally { CanvasRenderingContext2D.prototype.fillText = orig; }
        return Object.fromEntries(seen);
    });
    let nm = await overlayNames();
    check(!nm.top && Object.keys(nm).length > 0, "no overlay node names on the entry's view, the others keep theirs", nm);
    await page.keyboard.press('ArrowRight'); await page.waitForTimeout(300);
    nm = await overlayNames();
    check(nm.top > 0, 'off its frame the view names its nodes again', nm);
    await page.keyboard.press('ArrowLeft'); await page.waitForTimeout(300);
    await page.keyboard.press('Shift+M'); await page.waitForTimeout(300);
    e = await entries();
    check(e[1].cur, 'Shift+M goes back one');
    await page.keyboard.press('Shift+M'); await page.keyboard.press('Shift+M'); await page.waitForTimeout(300);
    e = await entries();
    check(e[5].cur && await frame() === 2, 'Shift+M wraps to the last entry');
    // single-view mode: the entry's camera is solo'd
    await page.keyboard.press('v'); await page.waitForTimeout(300);
    await clickEntry(2); await page.waitForTimeout(400);
    check(await page.evaluate(() => window.__lucid.state.views[window.__lucid.state.singleViewIndex].name) === 'top',
        'in single-view mode the entry solos its camera');
    await page.keyboard.press('g'); await page.waitForTimeout(300);
    // a click on an entry whose camera did not match says so
    await clickEntry(3); await page.waitForTimeout(200);
    check(/Camera not matched/.test(await status()), 'an entry on an unmatched camera says why', [await status(), await entries()]);

    // Zoom in off (browser-local): a selection leaves the zoom alone.
    await page.click('#modelCheckZoom');
    check(await page.evaluate(() => JSON.parse(localStorage.getItem('modelCheckFilter')).zoom) === false, 'the Zoom in choice is browser-local');
    await page.evaluate(() => { const AS = window.__lucid; AS.state.views.forEach(v => AS.videoController ? AS.videoController.resetZoom(v) : null); });
    const z0 = await zoomOf('top');
    await clickEntry(2); await page.waitForTimeout(400);
    check(await zoomOf('top') === z0, 'with Zoom in off, selecting does not zoom', [z0, await zoomOf('top')]);
    await page.click('#modelCheckZoom');

    // ---- 5. filters
    await page.selectOption('#modelCheckSplit', 'training');
    e = await entries();
    check(e.length === 2, 'split filter: the two training animals', e.length);
    await page.selectOption('#modelCheckSplit', 'all');
    await page.selectOption('#modelCheckCause', 'background');
    e = await entries();
    check(e.length === 1 && /top/.test(e[0].where), 'cause filter: the background read', e);
    check(await page.evaluate(() => JSON.parse(localStorage.getItem('modelCheckFilter')).cause) === 'background', 'filters are browser-local');
    await page.selectOption('#modelCheckCause', 'all');

    // ---- 6. still nothing to save
    check(await page.evaluate(() => !window.__lucid.state.isDirty), 'stepping and filtering leave the project unchanged');

    // ---- 7. an ambiguous camera is asked about
    // (text, not JSON.parse: the fixture holds Python's bare NaN, as a real scan does)
    const odd = SCAN.replace(/"video": "min_session_top\.mp4",(\s*)"camera": "videos"/g, '"video": "renamed-top.mp4",$1"camera": "top"');
    check(odd !== SCAN, 'variant fixture built');
    let pending = page.evaluate(async (t) => (await import('/ui/model-check-panel.js')).loadModelCheckText(t, 'odd.json'), odd);
    await page.waitForSelector('.model-check-ask', { timeout: 5000 });
    check(/top's video in the project is min_session_top\.mp4, the check read renamed-top\.mp4/.test(await page.textContent('.model-check-ask')),
        'the question says why it asks');
    await page.keyboard.press('Escape');
    check(await pending === null && !(await page.$('.model-check-ask')), 'Esc cancels the load');
    pending = page.evaluate(async (t) => (await import('/ui/model-check-panel.js')).loadModelCheckText(t, 'odd.json'), odd);
    await page.waitForSelector('.model-check-ask');
    await page.click('#modelCheckAskOk');
    const r7 = await pending;
    check(r7 && r7.matched === 42, 'picking the camera matches it', r7);
    check(/1 camera chosen by you/.test(await panelText()), 'the tab says it was chosen');

    // ---- 9. close
    await page.click('#modelCheckClose');
    check(/No model check loaded/.test(await panelText()), 'Close forgets the check');
    check((await mark()).length === 0, 'and clears the mark');
    const panesClosed = await panes();
    check(panesClosed.viewport3d === panesBefore.viewport3d && panesClosed.timeline === panesBefore.timeline,
        'Close brings back the panes it hid', { panesBefore, panesClosed });

    // ---- 8. lazy project
    console.log('\n-- lazy project --');
    check(await loadSlp(true) === true, 'multiview.slp opens lazily');
    await addViews();
    const lazy = await page.evaluate(() => !!window.__lucid.state.session.lazyLoader);
    check(lazy, 'the session is lazy');
    const lazyOrder = await fileOrder();   // nothing resident yet: the store's rows, i.e. the file's order
    check(JSON.stringify(lazyOrder.out) === JSON.stringify(eagerOrder.out),
        'an eager load names the animals in the same (file) order as the lazy store', { eager: eagerOrder.out, lazy: lazyOrder.out });
    await page.evaluate(async (t) => (await import('/ui/model-check-panel.js')).loadModelCheckText(t, 'scan.json'), SCAN);
    e = await entries();
    check(e[2].status === 'status-verified', 'a row is verified from the store before its frame is loaded', [e.map(x => x.status), await status(), (await panelText()).slice(0, 600)]);
    await clickEntry(2);
    await page.waitForFunction(() => window.__lucid.state.currentFrame === 1, null, { timeout: 5000 }).catch(() => {});
    let lm = [];
    for (let i = 0; i < 30 && !lm.length; i++) { await page.waitForTimeout(100); lm = await mark(); }
    check(lm.length === 1 && lm[0].view === 'top', 'the mark appears once the frame is hydrated', lm.map(m => m.view));
    check(/node positions agree/.test(await panelText()), 'and the entry is still verified on the resident frame');

    check(errs.length === 0, 'no page errors', errs);
} finally {
    if (browser) await browser.close();
    server.kill();
}
console.log(fails ? `\n${fails} FAILED` : '\nall passed');
process.exit(fails ? 1 : 0);
