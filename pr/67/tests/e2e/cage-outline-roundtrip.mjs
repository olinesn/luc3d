/**
 * cage-outline-roundtrip.mjs — Define Planes ▸ Outlines, driven with the mouse
 * and the keyboard on a real view, then saved and reopened.
 *
 * The outline is the cage's silhouette in one camera view. The cage-scan tools
 * read it from `metadata.lucid.cageOutlines` to build the cage's 3D shape, so
 * what is pinned here is (a) the gestures put the vertices where the clicks
 * were, in video pixels, (b) the gestures do not take plane-node editing away,
 * and (c) the file carries exactly the documented format and nothing for a
 * project that has no outline.
 *
 *   1. The section lists the session's views, and fits a 1440 px window.
 *   2. Trace: click, Backspace undoes, clicking the first vertex closes;
 *      vertices land where the clicks were; the project is marked unsaved.
 *   3. The outline is drawn on the view's overlay.
 *   4. Edit: drag a vertex, Shift+click an edge inserts, right-click deletes,
 *      never below 3.
 *   5. Retrace then Esc: the saved outline is untouched.
 *   6. Plane nodes: a plane node on top of an outline vertex still drags as a
 *      node; while tracing, a click on a plane node is an outline vertex.
 *   7. Leaving Defining Plane Mode cancels a trace.
 *   8. Save As (the real writer) -> parse -> read: the same points, in the
 *      documented format; a session without outlines writes no key.
 *
 * Run: node cage-outline-roundtrip.mjs   (spawns its own http.server; PORT=8791 by default)
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8791);
const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1200));

let fails = 0;
const check = (c, m) => { console.log((c ? '  ✓ ' : '  ✗ ') + m); if (!c) fails++; };

let browser;
try {
    browser = await chromium.launch();
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    page.on('pageerror', e => { console.log('  [pageerror]', String(e).slice(0, 300)); fails++; });
    page.on('console', m => {
        if (m.type() !== 'error') return;
        const t = m.text();
        if (/Failed to load resource|net::ERR|404/.test(t)) return;
        console.log('  [console.error]', t.slice(0, 300));
        fails++;
    });

    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state && window.SleapIO, null, { timeout: 30000 });
    await page.evaluate(async () => { await (await import('/pose/initialization.js')).loadDemoSession(); });
    await page.waitForFunction(() => {
        const s = window.__lucid.state;
        return s.session && s.views.length && s.views.every(v => v.videoWidth > 0 && v.overlayCanvas);
    }, null, { timeout: 60000 });
    await page.waitForTimeout(500);

    // A view whose overlay is really on screen and on top.
    const viewName = await page.evaluate(async () => {
        const P = await import('/ui/plane-definition.js');
        if (!P.isPlaneModeActive()) P.togglePlaneMode();
        for (const v of window.__lucid.state.views) {
            const r = v.overlayCanvas.getBoundingClientRect();
            if (r.width < 100 || r.height < 80) continue;
            const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
            if (hit === v.overlayCanvas) return v.name;
        }
        return null;
    });
    check(!!viewName, `a view is on screen to draw on (${viewName})`);
    if (!viewName) throw new Error('no visible view');

    /** Client coords of a video point on the chosen view (no rotation). */
    const toClient = (vx, vy) => page.evaluate(({ vn, vx, vy }) => {
        const v = window.__lucid.state.views.find(x => x.name === vn);
        const c = v.canvas || v.overlayCanvas;
        const r = c.getBoundingClientRect();
        return [r.left + vx * r.width / v.videoWidth, r.top + vy * r.height / v.videoHeight];
    }, { vn: viewName, vx, vy });
    const geom = await page.evaluate((vn) => {
        const v = window.__lucid.state.views.find(x => x.name === vn);
        const c = v.canvas || v.overlayCanvas;
        const r = c.getBoundingClientRect();
        return { W: v.videoWidth, H: v.videoHeight, perPx: v.videoWidth / r.width };
    }, viewName);
    const TOL = geom.perPx * 1.5 + 0.5;   // one rounded screen pixel, in video px
    const outline = () => page.evaluate((vn) => {
        const set = window.__lucid.state.session.cageOutlines;
        const polys = set ? set.polygons(vn) : [];
        return polys.map(p => p.points.map(q => [q[0], q[1]]));
    }, viewName);
    const tool = () => page.evaluate(async () => {
        const T = await import('/ui/cage-outline-tool.js');
        return { tracing: !!T.outlineState.trace, n: T.outlineState.trace ? T.outlineState.trace.points.length : 0,
                 status: document.getElementById('statusText').textContent,
                 dirty: window.__lucid.state.isDirty === true };
    });
    const near = (a, b) => Math.abs(a[0] - b[0]) <= TOL && Math.abs(a[1] - b[1]) <= TOL;
    const clickVideo = async (vx, vy, opts) => {
        const [x, y] = await toClient(vx, vy);
        const mods = (opts && opts.modifiers) || [];
        for (const m of mods) await page.keyboard.down(m);
        await page.mouse.click(x, y, opts && opts.button ? { button: opts.button } : undefined);
        for (const m of mods) await page.keyboard.up(m);
    };

    // ===================================================================
    console.log('\n-- 1. the section --');
    const sec = await page.evaluate(() => {
        const rows = Array.from(document.querySelectorAll('#cageOutlinesTable tbody tr'));
        const panel = document.getElementById('planePanel');
        const btns = Array.from(document.querySelectorAll('#cageOutlinesTable button'));
        const info = document.getElementById('cageOutlinesInfo');
        return {
            views: rows.map(r => r.dataset.view),
            labels: rows.map(r => Array.from(r.querySelectorAll('button')).map(b => b.textContent)),
            stateViews: window.__lucid.state.views.map(v => v.name),
            summary: document.querySelector('#cageOutlinesDetails > summary').textContent.trim(),
            isDetails: document.getElementById('cageOutlinesDetails').tagName,
            infoTip: info ? info.getAttribute('aria-label') || '' : '',
            panelOverflow: panel.scrollWidth - panel.clientWidth,
            clipped: btns.filter(b => b.scrollWidth > b.clientWidth + 1).map(b => b.textContent),
            fontSize: getComputedStyle(document.querySelector('#cageOutlinesDetails > summary')).fontSize,
        };
    });
    check(JSON.stringify(sec.views) === JSON.stringify(sec.stateViews), `one row per view (${JSON.stringify(sec.views)})`);
    check(sec.labels.every(l => l.join() === 'Trace,Clear'), 'each row offers Trace and Clear');
    check(sec.isDetails === 'DETAILS' && sec.summary === 'Outlines', 'a collapsible section titled Outlines');
    check(/outline/i.test(sec.infoTip), `its ⓘ explains it ("${sec.infoTip.slice(0, 60)}…")`);
    check(sec.fontSize === '14px', `section title matches the panel's (${sec.fontSize})`);
    check(sec.panelOverflow <= 0 && sec.clipped.length === 0,
        `nothing overflows at 1440 px (overflow ${sec.panelOverflow}, clipped ${JSON.stringify(sec.clipped)})`);

    // ===================================================================
    console.log('\n-- 2. trace --');
    const { W, H } = geom;
    const P0 = [0.22 * W, 0.2 * H], P1 = [0.62 * W, 0.24 * H], P2 = [0.66 * W, 0.7 * H], P3 = [0.26 * W, 0.66 * H];
    await page.evaluate(() => (window.__lucid.state.isDirty = false));
    await page.click(`#cageOutlinesTable tr[data-view="${viewName}"] button:has-text("Trace")`);
    let t = await tool();
    check(t.tracing && /Tracing the outline/.test(t.status), `Trace starts a trace ("${t.status.slice(0, 50)}…")`);
    await clickVideo(...P0);
    await clickVideo(...P1);
    await clickVideo(...P2);
    await clickVideo(0.5 * W, 0.9 * H);           // a mistake…
    t = await tool();
    check(t.n === 4, `four clicks, four vertices (${t.n})`);
    await page.keyboard.press('Backspace');       // …undone
    t = await tool();
    check(t.n === 3, `Backspace removes the last (${t.n})`);
    const rowCount = await page.evaluate((vn) =>
        document.querySelector(`#cageOutlinesTable tr[data-view="${vn}"] .outline-count`).textContent, viewName);
    check(rowCount === '3…', `the row counts the trace in progress ("${rowCount}")`);
    await clickVideo(...P3);
    await clickVideo(P0[0] + 1, P0[1] + 1);       // the first vertex closes it
    t = await tool();
    let o = await outline();
    check(!t.tracing && o.length === 1 && o[0].length === 4, `clicking the first vertex closes a 4-point outline (${o.map(p => p.length)})`);
    check(o[0] && [P0, P1, P2, P3].every((p, i) => near(o[0][i], p)),
        `vertices are where the clicks were, in video px (±${TOL.toFixed(2)}): ${JSON.stringify(o[0] && o[0].map(p => p.map(v => +v.toFixed(1))))}`);
    check(t.dirty, 'closing an outline marks the project unsaved');

    // ===================================================================
    console.log('\n-- 3. drawn --');
    const drawn = await page.evaluate(async ({ vn, a, b }) => {
        const P = await import('/ui/plane-definition.js');
        const { makeVideoToCanvasTransform } = await import('/ui/overlays.js');
        P.redraw();
        await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
        const v = window.__lucid.state.views.find(x => x.name === vn);
        const ds = { width: v.overlayCanvas.width, height: v.overlayCanvas.height };
        const tf = makeVideoToCanvasTransform(v.videoWidth, v.videoHeight, ds.width, ds.height);
        const m = tf((a[0] + b[0]) / 2, (a[1] + b[1]) / 2);
        const ctx = v.overlayCanvas.getContext('2d');
        const img = ctx.getImageData(Math.round(m.x) - 4, Math.round(m.y) - 4, 9, 9).data;
        let orange = 0;
        for (let i = 0; i < img.length; i += 4) {
            if (img[i] > 200 && img[i + 1] > 120 && img[i + 1] < 210 && img[i + 2] < 120 && img[i + 3] > 150) orange++;
        }
        return orange;
    }, { vn: viewName, a: P0, b: P1 });
    check(drawn > 0, `the outline's edge is drawn in orange on the overlay (${drawn} px)`);

    // ===================================================================
    console.log('\n-- 4. edit --');
    {
        await page.evaluate(() => (window.__lucid.state.isDirty = false));
        const [x1, y1] = await toClient(...P1);
        await page.mouse.move(x1, y1);
        await page.mouse.down();
        await page.mouse.move(x1 + 10, y1 + 8, { steps: 3 });
        await page.mouse.move(x1 + 20, y1 + 16, { steps: 3 });
        await page.mouse.up();
        o = await outline();
        const want = [P1[0] + 20 * geom.perPx, P1[1] + 16 * geom.perPx];
        check(near(o[0][1], want), `dragging a vertex moves it (${o[0][1].map(v => v.toFixed(1))} vs ${want.map(v => v.toFixed(1))})`);
        check(near(o[0][0], P0) && near(o[0][2], P2), 'and only that vertex');
        check((await tool()).dirty, 'a drag marks the project unsaved');
        const zoom = await page.evaluate((vn) => {
            const v = window.__lucid.state.views.find(x => x.name === vn);
            return { scale: v.zoom ? v.zoom.scale : 1, box: !!document.querySelector('.box-zoom-overlay'),
                     flag: !!window.__mvguiDragging };
        }, viewName);
        check(Math.abs(zoom.scale - 1) < 1e-9 && !zoom.box && !zoom.flag,
            `the view did not box-zoom or pan under the drag (scale ${zoom.scale}, flag ${zoom.flag})`);

        const mid = [(o[0][2][0] + o[0][3][0]) / 2, (o[0][2][1] + o[0][3][1]) / 2];
        await clickVideo(mid[0], mid[1], { modifiers: ['Shift'] });
        o = await outline();
        check(o[0].length === 5 && near(o[0][3], mid), `Shift+click on an edge inserts a vertex there (${o[0].length} points)`);

        const inserted = o[0][3];
        await clickVideo(...inserted, { button: 'right' });
        o = await outline();
        check(o[0].length === 4 && !o[0].some(q => near(q, inserted)),
            `right-click deletes that vertex (${o[0].length})`);
        await clickVideo(...o[0][3], { button: 'right' });
        o = await outline();
        await clickVideo(...o[0][2], { button: 'right' });
        o = await outline();
        const st = (await tool()).status;
        check(o[0].length === 3 && /at least 3/.test(st), `never below 3 points, and it says so (${o[0].length}, "${st.slice(0, 40)}…")`);
    }

    // ===================================================================
    console.log('\n-- 5. retrace, then Esc --');
    {
        const before = await outline();
        await page.click(`#cageOutlinesTable tr[data-view="${viewName}"] button:has-text("Retrace")`);
        await clickVideo(0.1 * W, 0.1 * H);
        await clickVideo(0.3 * W, 0.1 * H);
        await page.keyboard.press('Escape');
        const t5 = await tool();
        const after = await outline();
        check(!t5.tracing && /cancelled/.test(t5.status), 'Esc cancels the trace');
        check(JSON.stringify(after) === JSON.stringify(before), 'and the saved outline is untouched');
        // Enter does nothing to outlines once the trace has ended (listener removed).
        await page.keyboard.press('Enter');
        check(JSON.stringify(await outline()) === JSON.stringify(before) && !(await tool()).tracing,
            'the trace\'s key listener is gone after Esc');
    }

    // ===================================================================
    console.log('\n-- 6. plane nodes still edit --');
    {
        const setup = await page.evaluate(async (vn) => {
            const P = await import('/ui/plane-definition.js');
            const model = P.planeModel();
            const floor = P.createPlane('floor');
            for (const n of ['a', 'b', 'c']) model.createNodeInPlane(n, floor);
            const v = window.__lucid.state.views.find(x => x.name === vn);
            P.placePlaneOnView(floor, vn, v.videoWidth * 0.45, v.videoHeight * 0.45);
            const inst = model.getInstance(vn);
            const idx = model.pool.indexOf(floor.nodeIds[0]);
            const at = [inst.getX(idx), inst.getY(idx)];
            // Put an outline vertex exactly on that node.
            const set = window.__lucid.state.session.cageOutlines;
            const poly = set.polygons(vn)[0];
            set.moveVertex(vn, poly.id, 0, at[0], at[1]);
            P.redraw();
            return { idx, at, outlineV0: poly.points[0].slice() };
        }, viewName);
        const [nx, ny] = await toClient(...setup.at);
        await page.mouse.move(nx, ny);
        await page.mouse.down();
        await page.mouse.move(nx + 15, ny + 15, { steps: 4 });
        await page.mouse.up();
        const r6 = await page.evaluate(async ({ vn, idx }) => {
            const P = await import('/ui/plane-definition.js');
            const inst = P.planeModel().getInstance(vn);
            return { node: [inst.getX(idx), inst.getY(idx)],
                     v0: window.__lucid.state.session.cageOutlines.polygons(vn)[0].points[0].slice() };
        }, { vn: viewName, idx: setup.idx });
        check(!near(r6.node, setup.at) && near(r6.node, [setup.at[0] + 15 * geom.perPx, setup.at[1] + 15 * geom.perPx]),
            `a plane node on top of an outline vertex drags as the node (${r6.node.map(v => v.toFixed(1))})`);
        check(near(r6.v0, setup.outlineV0), 'and the outline vertex under it stays put');

        // While tracing this view, the click is an outline vertex, not a node grab.
        await page.click(`#cageOutlinesTable tr[data-view="${viewName}"] button:has-text("Retrace")`);
        const [mx, my] = await toClient(...r6.node);
        await page.mouse.move(mx, my);
        await page.mouse.down();
        await page.mouse.move(mx + 20, my, { steps: 3 });
        await page.mouse.up();
        const r6b = await page.evaluate(async ({ vn, idx }) => {
            const P = await import('/ui/plane-definition.js');
            const T = await import('/ui/cage-outline-tool.js');
            const inst = P.planeModel().getInstance(vn);
            return { node: [inst.getX(idx), inst.getY(idx)], n: T.outlineState.trace ? T.outlineState.trace.points.length : -1 };
        }, { vn: viewName, idx: setup.idx });
        check(r6b.n === 1 && near(r6b.node, r6.node), `while tracing, a click on a node is a vertex and the node does not move (${r6b.n})`);
        await page.keyboard.press('Escape');
    }

    // ===================================================================
    console.log('\n-- 7. leaving the mode cancels a trace --');
    {
        const r7 = await page.evaluate(async (vn) => {
            const P = await import('/ui/plane-definition.js');
            const T = await import('/ui/cage-outline-tool.js');
            T.startOutlineTrace(vn);
            const was = !!T.outlineState.trace;
            P.togglePlaneMode();
            const now = !!T.outlineState.trace;
            P.togglePlaneMode();
            return { was, now };
        }, viewName);
        check(r7.was && !r7.now, 'exiting Defining Plane Mode ends the trace');
    }

    // ===================================================================
    console.log('\n-- 8. save, reopen --');
    {
        const r8 = await page.evaluate(async (vn) => {
            const AS = await import('/ui/app-state.js');
            const pd = await import('/pose/pose-data.js');
            const PM = await import('/import-export/plane-metadata.js');
            const CO = await import('/pose/cage-outlines.js');
            const fileio = await import('/import-export/file-io.js');
            const saveLoad = await import('/import-export/save-load.js');
            const session = AS.state.session;
            const expected = CO.serializeCageOutlines(session.cageOutlines);

            const chunks = [];
            window.showSaveFilePicker = async () => ({
                name: 'outline.slp',
                createWritable: async () => ({
                    write: async (c) => { chunks.push(new Uint8Array(c.buffer ? c.slice(0) : c)); },
                    close: async () => {}, abort: async () => {},
                }),
            });
            await saveLoad.saveAs();
            const total = chunks.reduce((n, c) => n + c.byteLength, 0);
            const bytes = new Uint8Array(total);
            let off = 0;
            for (const c of chunks) { bytes.set(c, off); off += c.byteLength; }
            const parsed = await fileio.parseSlpViaSleapIO(new File([bytes], 'outline.slp'), () => {});
            const lucid = (parsed.sessions[0].metadata || {}).lucid || {};
            const fresh = new pd.Session([], new pd.Skeleton('s', ['n0'], []), ['t0'], 'reopened');
            PM.readPlaneMetadata(fresh, lucid);

            // A session with no outline writes no key.
            const plain = new pd.Session(session.cameras, session.skeleton, ['t0'], 'plain');
            const plainLabels = fileio.buildSlpLabelsAllViews(plain, AS.state.views, AS.state.videoFiles);
            const plainBytes = await window.SleapIO.saveSlpToBytes(plainLabels);
            const plainParsed = await fileio.parseSlpViaSleapIO(new File([plainBytes], 'plain.slp'), () => {});
            const plainLucid = (plainParsed.sessions[0].metadata || {}).lucid || {};

            return {
                status: document.getElementById('statusText').textContent,
                expected, written: lucid.cageOutlines || null,
                reopened: CO.serializeCageOutlines(fresh.cageOutlines),
                plainHasKey: Object.prototype.hasOwnProperty.call(plainLucid, 'cageOutlines'),
                plainHasLucid: !!plainParsed.sessions[0].metadata,
            };
        }, viewName);
        check(/^Saved/.test(r8.status), `Save As went through ("${r8.status.slice(0, 40)}")`);
        const w = r8.written;
        check(!!w && w.version === 1 && w.views && Array.isArray(w.views[viewName]),
            `the file has cageOutlines {version: 1, views: {${viewName}: [...]}}`);
        const p = w && w.views[viewName][0];
        check(!!p && Object.keys(p).join() === 'id,name,points' && p.name === 'cage' &&
            p.points.every(q => q.length === 2 && q.every(Number.isFinite)),
            `each polygon is {id, name, points: [[x, y], ...]} (${p && Object.keys(p).join()})`);
        check(JSON.stringify(w) === JSON.stringify(r8.expected), 'the written outlines are exactly the session\'s');
        check(JSON.stringify(r8.reopened) === JSON.stringify(r8.expected), 'and read back to the same points');
        check(r8.plainHasLucid && !r8.plainHasKey, 'a session without outlines writes no cageOutlines key');
    }
} catch (err) {
    console.error('\nFATAL', err);
    fails++;
} finally {
    if (browser) await browser.close();
    server.kill();
}

console.log(fails ? `\n${fails} FAILED` : '\nPASSED');
process.exit(fails ? 1 : 0);
