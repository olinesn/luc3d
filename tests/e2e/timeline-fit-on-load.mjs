/**
 * timeline-fit-on-load.mjs — loading a session sizes the Timeline to its rows,
 * capped at 30% of the window, and Ctrl/Cmd+J keeps that height.
 *
 * Loads real session folders (calibration.toml + per-camera video and
 * predictions .slp) through `handleLoadSessionFolderPerCamera`, which ends in
 * `fitTimelineToData()` (`ui/timeline-controller.js`). Before the fix that
 * function looked for the Timeline on `state.timeline`, which nothing sets, so
 * it never ran: a project opened after a bigger one kept the bigger height.
 *  1. Fresh app -> 3 cameras x 2 tracks: the height is exactly the rows' height.
 *  2. 4 cameras x 6 tracks: the height stops at 30% of the window; the rows scroll.
 *  3. The 3-camera folder again: the height shrinks back to its rows, in one
 *     step (no in-between height is painted).
 *  4. A height the user dragged is replaced by the fit on the next load.
 *  5. Ctrl+J collapses, Ctrl+J restores the same height; the canvas matches it
 *     on the first painted frame, with one Timeline.resize().
 *  6. Expanding with no remembered height fits the rows (not the CSS default).
 *  7-9. The other project-open paths fit too, each opened while the timeline
 *     is at the 30% cap: the lazy .slp open (`handleLoadProjectSlpLazy`), the
 *     single-SLP session folder (`handleLoadSessionFolderSingleSlp`) and the
 *     JSON project (`handleLoadProject`).
 *
 * Run: node tests/e2e/timeline-fit-on-load.mjs   (PORT= to move it)
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8473);
let fails = 0;
const check = (c, m) => { console.log((c ? '  ✓ ' : '  ✗ ') + m); if (!c) fails++; };

const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'lucid-timeline-fit-'));
const calibToml = (cams) => cams.map((c, i) => `[cam_${i}]\nname = "${c}"\nsize = [ 160, 120,]\n` +
    'matrix = [ [ 150.0, 0.0, 80.0,], [ 0.0, 150.0, 60.0,], [ 0.0, 0.0, 1.0,],]\n' +
    `distortions = [ 0.0, 0.0, 0.0, 0.0, 0.0,]\nrotation = [ 0.0, ${(0.1 * i).toFixed(2)}, 0.0,]\n` +
    `translation = [ ${-30 * i}.0, 0.0, 0.0,]\n`).join('\n') + '\n[metadata]\n';

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1200));

let browser;
try {
    browser = await chromium.launch();
    const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
    const errs = [];
    page.on('pageerror', e => errs.push('pageerror: ' + String(e).slice(0, 300)));
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state && window.SleapIO, null, { timeout: 30000 });

    // One 20-frame video, and one predictions .slp per camera and track count.
    const fixtures = await page.evaluate(async () => {
        const enc = await import('/ui/video-encode.js');
        const pd = await import('/pose/pose-data.js');
        const fileio = await import('/import-export/file-io.js');
        const b64 = (buf) => { let s = ''; for (let i = 0; i < buf.length; i++) s += String.fromCharCode(buf[i]); return btoa(s); };
        const canvas = document.createElement('canvas');
        canvas.width = 160; canvas.height = 120;
        const cctx = canvas.getContext('2d');
        const wr = await enc.createMp4Writer({ canvas, width: 160, height: 120, fps: 20,
            bitrate: 300000, frameCount: 20, keyFrameEveryFrames: 5 });
        for (let i = 0; i < 20; i++) {
            cctx.fillStyle = 'rgb(' + (10 * i) + ',60,90)'; cctx.fillRect(0, 0, 160, 120);
            await wr.addFrame(i);
        }
        const video = b64(new Uint8Array(await (await wr.finish()).blob.arrayBuffer()));
        const { Skeleton, Camera, Instance, FrameGroup, Session } = pd;
        const nodes = ['a', 'b', 'c'];
        async function camSlp(cam, nTracks) {
            const camera = new Camera(cam, [[150, 0, 80], [0, 150, 60], [0, 0, 1]], [0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0], [160, 120]);
            const s = new Session([camera], new Skeleton('sk', nodes, [[0, 1], [1, 2]]),
                Array.from({ length: nTracks }, (_, i) => 'track_' + i), cam);
            for (let f = 0; f < 20; f++) {
                const fg = new FrameGroup(f);
                for (let t = 0; t < nTracks; t++) {
                    fg.addInstance(cam, new Instance(nodes.map((_, k) => [10 + t * 20 + k * 3, 30 + k * 4 + f]), t, 'predicted', 0.9));
                }
                s.addFrameGroup(fg);
            }
            const views = [{ name: cam, videoWidth: 160, videoHeight: 120, frameCount: 20 }];
            const vf = [{ name: cam, assignedCamera: cam, videoPath: cam + '/0.mp4' }];
            return b64(await window.SleapIO.saveSlpToBytes(fileio.buildSlpLabelsAllViews(s, views, vf)));
        }
        const slps = {};
        for (const nTracks of [2, 6]) for (let c = 0; c < 4; c++) slps['Cam' + c + '_' + nTracks] = await camSlp('Cam' + c, nTracks);
        // A 3-camera x 2-track project .slp (all views in one file).
        const camNames = ['Cam0', 'Cam1', 'Cam2'];
        const cameras = camNames.map((n, i) => new Camera(n, [[150, 0, 80], [0, 150, 60], [0, 0, 1]], [0, 0, 0, 0, 0],
            [0, 0.1 * i, 0], [-30 * i, 0, 0], [160, 120]));
        const proj = new Session(cameras, new Skeleton('sk', nodes, [[0, 1], [1, 2]]), ['track_0', 'track_1'], 'Project');
        for (let f = 0; f < 20; f++) {
            const fg = new FrameGroup(f);
            camNames.forEach(cn => {
                for (let t = 0; t < 2; t++) fg.addInstance(cn, new Instance(nodes.map((_, k) => [10 + t * 20 + k * 3, 30 + k * 4 + f]), t, 'predicted', 0.9));
            });
            proj.addFrameGroup(fg);
        }
        const projSlp = b64(await window.SleapIO.saveSlpToBytes(fileio.buildSlpLabelsAllViews(proj,
            camNames.map(n => ({ name: n, videoWidth: 160, videoHeight: 120, frameCount: 20 })),
            camNames.map(n => ({ name: n, assignedCamera: n, videoPath: 'videos/' + n + '.mp4' })))));
        return { video, slps, projSlp };
    });
    const video = Buffer.from(fixtures.video, 'base64');
    function writeFolder(name, nCams, nTracks) {
        const dir = path.join(tmpRoot, name);
        const cams = Array.from({ length: nCams }, (_, i) => 'Cam' + i);
        fs.mkdirSync(dir, { recursive: true });
        fs.writeFileSync(path.join(dir, 'calibration.toml'), calibToml(cams));
        for (const c of cams) {
            fs.mkdirSync(path.join(dir, c), { recursive: true });
            fs.writeFileSync(path.join(dir, c, '0.mp4'), video);
            fs.writeFileSync(path.join(dir, c, 'predictions.slp'), Buffer.from(fixtures.slps[c + '_' + nTracks], 'base64'));
        }
        return dir;
    }
    const small = writeFolder('small', 3, 2);   // 6 rows: fits under the cap
    const big = writeFolder('big', 4, 6);       // 24 rows: taller than the cap
    // Single-SLP session folder: calibration + one project .slp + videos/.
    const single = path.join(tmpRoot, 'single');
    fs.mkdirSync(path.join(single, 'videos'), { recursive: true });
    fs.writeFileSync(path.join(single, 'calibration.toml'), calibToml(['Cam0', 'Cam1', 'Cam2']));
    fs.writeFileSync(path.join(single, 'project.slp'), Buffer.from(fixtures.projSlp, 'base64'));
    for (const c of ['Cam0', 'Cam1', 'Cam2']) fs.writeFileSync(path.join(single, 'videos', c + '.mp4'), video);

    // Count Timeline.resize() calls, and record the timeline height on every frame.
    await page.evaluate(async () => {
        const T = (await import('/ui/timeline.js')).Timeline;
        const orig = T.prototype.resize;
        window.__resizes = 0;
        T.prototype.resize = function (...a) { window.__resizes++; return orig.apply(this, a); };
        window.__heights = [];
        const sample = () => {
            window.__heights.push(Math.round(document.getElementById('timelineContainer').getBoundingClientRect().height));
            requestAnimationFrame(sample);
        };
        requestAnimationFrame(sample);
    });
    const resetCounts = () => page.evaluate(() => { window.__resizes = 0; window.__heights = []; });
    const paintedHeights = () => page.evaluate(() => {
        const seq = [];
        for (const h of window.__heights) if (seq[seq.length - 1] !== h) seq.push(h);
        return seq;
    });

    async function loadFolder(dir) {
        page.once('filechooser', async (fc) => { await fc.setFiles(dir); });
        await page.evaluate(async () => {
            await (await import('/loading/session-loader.js')).handleLoadSessionFolderPerCamera();
        });
        await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
    }
    const tl = () => page.evaluate(() => {
        const c = document.getElementById('timelineContainer');
        const t = window.__lucid.timeline;
        const preferred = t.getPreferredHeight();
        const cap = Math.floor(0.3 * window.innerHeight);
        const sc = t._trackScrollEl;
        return {
            rows: t._trackSegments.length, height: c.getBoundingClientRect().height, preferred, cap,
            fit: Math.min(preferred, cap), collapsed: c.classList.contains('collapsed'),
            scrolls: sc.scrollHeight - sc.clientHeight > 2, canvasH: t._viewHeight, scrollH: sc.clientHeight,
            sessions: window.__lucid.state.sessions.length,
        };
    });
    const near = (a, b) => Math.abs(a - b) <= 1;

    // ---- 1. first load: exactly the rows ------------------------------------------------
    console.log('\n1. Fresh app, 3 cameras x 2 tracks');
    await loadFolder(small);
    let s = await tl();
    check(s.rows === 6, `6 timeline rows (${s.rows})`);
    check(s.preferred < s.cap, `precondition: the rows fit under the cap (${s.preferred} < ${s.cap}px)`);
    check(near(s.height, s.preferred), `the timeline is the rows' height (${s.height} vs ${s.preferred}px)`);
    const smallH = s.height;
    // Keep this project as a JSON project file for section 9 (saveProject()
    // downloads it; capture the Blob instead).
    await page.evaluate(async () => {
        const saveLoad = await import('/import-export/save-load.js');
        const origCreate = URL.createObjectURL;
        let blob = null;
        URL.createObjectURL = (b) => { blob = b; return 'blob:stub'; };
        try { saveLoad.saveProject(); } finally { URL.createObjectURL = origCreate; }
        window.__projectJson = blob ? await blob.text() : null;
    });

    // ---- 2. bigger project: capped --------------------------------------------------------
    console.log('\n2. 4 cameras x 6 tracks');
    await loadFolder(big);
    s = await tl();
    check(s.rows === 24, `24 timeline rows (${s.rows})`);
    check(s.preferred > s.cap, `precondition: the rows are taller than the cap (${s.preferred} > ${s.cap}px)`);
    check(near(s.height, s.cap), `the timeline stops at 30% of the window (${s.height} vs ${s.cap}px)`);
    check(s.scrolls, 'and its rows scroll');
    const bigH = s.height;

    // ---- 3. the smaller project again: shrinks, in one step -------------------------------
    console.log('\n3. The 3-camera folder again');
    await resetCounts();
    await loadFolder(small);
    s = await tl();
    check(s.rows === 6, `6 timeline rows again (${s.rows})`);
    check(near(s.height, s.preferred), `the timeline shrinks back to its rows (${s.height} vs ${s.preferred}px; was ${bigH}px)`);
    let seq = await paintedHeights();
    check(seq.length === 2 && near(seq[0], bigH) && near(seq[1], s.height),
        `it goes straight from ${bigH} to ${s.height}px, no height in between (${seq.join(' -> ')})`);

    // ---- 4. a dragged height is replaced by the next load's fit ---------------------------
    console.log('\n4. User drags the timeline to 250px, then loads');
    await page.evaluate(() => { document.getElementById('timelineContainer').style.height = '250px'; });
    await loadFolder(small);
    s = await tl();
    check(near(s.height, s.preferred), `the load fits the rows again (${s.height} vs ${s.preferred}px)`);

    // ---- 5. Ctrl+J collapse / expand ------------------------------------------------------
    console.log('\n5. Ctrl+J collapse, Ctrl+J expand');
    await page.mouse.click(800, 300);
    await page.keyboard.press('Control+j');
    await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
    s = await tl();
    check(s.collapsed && s.height === 0, `Ctrl+J collapses it (${s.height}px)`);
    await resetCounts();
    await page.keyboard.press('Control+j');
    // Read on the 2nd frame: what the 1st painted frame showed (a ResizeObserver
    // runs before paint; nothing runs between that paint and this read).
    const firstFrame = await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(() => {
        const t = window.__lucid.timeline;
        r({ canvasH: t._viewHeight, scrollH: t._trackScrollEl.clientHeight,
            height: document.getElementById('timelineContainer').getBoundingClientRect().height });
    }))));
    check(near(firstFrame.height, smallH), `Ctrl+J restores ${smallH}px (${firstFrame.height}px)`);
    check(firstFrame.canvasH === firstFrame.scrollH && firstFrame.scrollH > 0,
        `the canvas already fits it on the first painted frame (${firstFrame.canvasH} vs ${firstFrame.scrollH}px)`);
    await page.waitForTimeout(200);
    const rz = await page.evaluate(() => window.__resizes);
    check(rz === 1, `one Timeline.resize() for the expand (${rz})`);
    seq = await paintedHeights();
    check(seq.every(h => near(h, smallH)), `no other height painted after the expand (${seq.join(' -> ')})`);

    // ---- 6. expand with nothing remembered: fits the rows ---------------------------------
    console.log('\n6. Expand with no remembered height');
    await page.keyboard.press('Control+j');
    await page.evaluate(async () => {
        (await import('/ui/timeline-controller.js')).setCachedTimelineHeight(0);
    });
    await page.keyboard.press('Control+j');
    await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
    s = await tl();
    check(!s.collapsed && near(s.height, s.fit), `it opens at the rows' height (${s.height} vs ${s.fit}px), not the CSS default`);

    // ---- 7-9. the other project-open paths, each from the 30% cap -------------------------
    // Starts a load without awaiting it, then answers any prompt the way a user
    // with no videos to hand would (Escape the video picker, Skip / Later).
    async function runLoad(kind) {
        await page.evaluate((kind) => {
            window.__loadDone = false;
            (async () => {
                try {
                    if (kind === 'lazy') {
                        const bytes = Uint8Array.from(atob(window.__projSlp), c => c.charCodeAt(0));
                        await (await import('/loading/session-loader.js')).handleLoadProjectSlpLazy(new File([bytes], 'project.slp'));
                    } else if (kind === 'single') {
                        await (await import('/loading/session-loader.js')).handleLoadSessionFolderSingleSlp();
                    } else {
                        await (await import('/import-export/save-load.js')).handleLoadProject(
                            new File([window.__projectJson], 'project.json', { type: 'application/json' }));
                    }
                    window.__loadDone = true;
                } catch (e) { window.__loadDone = 'ERR ' + String(e && e.stack || e).slice(0, 400); }
            })();
        }, kind);
        for (let i = 0; i < 240; i++) {
            const st = await page.evaluate(() => {
                const b = [...document.querySelectorAll('button')].find(x => x.offsetParent && /^(Skip|Later)\b/i.test((x.textContent || '').trim()));
                if (b) b.click();
                return { done: window.__loadDone, prompt: !!document.getElementById('videoSelectOverlay') };
            });
            if (st.prompt) await page.keyboard.press('Escape');
            if (st.done) return st.done;
            await page.waitForTimeout(250);
        }
        return 'timeout';
    }
    await page.evaluate((b64) => { window.__projSlp = b64; }, fixtures.projSlp);
    for (const [n, label, kind] of [[7, 'Lazy .slp open', 'lazy'], [8, 'Single-SLP session folder', 'single'],
        [9, 'JSON project', 'json']]) {
        console.log(`\n${n}. ${label}, opened while the timeline is at the cap`);
        if (kind === 'json' && !(await page.evaluate(() => !!window.__projectJson))) {
            check(false, 'saveProject() produced the JSON project'); continue;
        }
        await loadFolder(big);
        s = await tl();
        check(near(s.height, s.cap), `precondition: the timeline is at the cap (${s.height}px)`);
        if (kind === 'single') page.once('filechooser', async (fc) => { await fc.setFiles(single); });
        const r = await runLoad(kind);
        check(r === true, `the load finished (${r})`);
        await page.evaluate(() => new Promise(res => requestAnimationFrame(() => requestAnimationFrame(res))));
        s = await tl();
        check(s.fit < s.cap - 10, `precondition: its rows need less than the cap (${s.rows} rows, ${s.fit}px)`);
        check(near(s.height, s.fit), `the timeline fits its rows (${s.height} vs ${s.fit}px)`);
    }

    check(errs.length === 0, 'no page errors' + (errs.length ? ': ' + errs.join(' | ') : ''));
} finally {
    if (browser) await browser.close();
    server.kill();
    fs.rmSync(tmpRoot, { recursive: true, force: true });
}
console.log(fails ? `\n${fails} FAILED` : '\nall passed');
process.exit(fails ? 1 : 0);
