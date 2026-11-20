/**
 * _real-model-check.mjs — investigation tool, not an assertion: load a real
 * project .slp (no video) and a real model check (scan.json) the way the Model
 * Check tab does, and print how its rows matched: cameras, then each misread
 * animal's instance (verified / file order / by positions / not matched, and why).
 *
 * Run:
 *   SLP=/path/labels.slp SCAN=/path/scan.json node tests/e2e/_real-model-check.mjs
 *   LAZY=1 …   opens the .slp as a lazy project (Load Project) instead of Load SLP
 *   VIDEOS=/dir …  answers the video prompt with every .mp4 under that folder, read in
 *                  place (the file chooser gets their paths; nothing is copied), in
 *                  headless Chrome (Playwright's Chromium cannot decode H.264)
 *   SHOTS=/dir …   with VIDEOS: selects each entry in turn and saves the view as
 *                  <dir>/entry-NN.png (the whole page for the first one); GRID=1 keeps
 *                  the grid of views instead of single-view mode
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const PORT = Number(process.env.PORT || 8297);
const SLP = process.env.SLP, SCAN = process.env.SCAN;
if (!SLP || !SCAN) { console.error('Set SLP and SCAN'); process.exit(2); }
const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1200));
const VIDEOS = process.env.VIDEOS ? (function walk(d) {
    return fs.readdirSync(d, { withFileTypes: true }).flatMap(e => e.isDirectory() ? walk(path.join(d, e.name))
        : /\.mp4$/i.test(e.name) && !e.name.startsWith('.') ? [path.join(d, e.name)] : []);
})(process.env.VIDEOS) : null;
const browser = await chromium.launch(VIDEOS ? { channel: 'chrome' } : {});
try {
    const page = await browser.newPage({ viewport: { width: 1600, height: 900 } });
    page.on('pageerror', e => console.log('pageerror', String(e).slice(0, 300)));
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.SleapIO && window.h5wasm && window.__lucid, null, { timeout: 120000 });
    await page.evaluate((b64) => { window.__file = Uint8Array.from(atob(b64), c => c.charCodeAt(0)); }, fs.readFileSync(SLP).toString('base64'));
    await page.evaluate(([name, lazy]) => {
        window.__loadDone = false;
        (async () => {
            try {
                const f = new File([window.__file], name);
                if (lazy) await (await import('/loading/session-loader.js')).handleLoadProjectSlpLazy(f);
                else await (await import('/import-export/slp-import.js')).handleLoadSlpFile(f);
                window.__loadDone = true;
            } catch (e) { window.__loadDone = 'ERR ' + String(e && e.stack || e).slice(0, 400); }
        })();
    }, [path.basename(SLP), !!process.env.LAZY]);
    let done;
    let gaveVideos = false;
    for (let i = 0; i < 3000 && !done; i++) {
        const s = await page.evaluate((withVideos) => {
            const prompt = document.getElementById('videoSelectOverlay');
            if (!(withVideos && prompt)) {
                const b = [...document.querySelectorAll('button')].find(x => /Skip|Later/i.test(x.textContent || ''));
                if (b) b.click();
            }
            return { done: window.__loadDone, prompt: !!prompt };
        }, !!VIDEOS && !gaveVideos);
        if (s.prompt && VIDEOS && !gaveVideos) {
            gaveVideos = true;
            const chooser = page.waitForEvent('filechooser');
            await page.locator('#videoSelectOverlay button', { hasText: 'Select Video Files' }).click();
            await (await chooser).setFiles(VIDEOS);
            console.log('videos given:', VIDEOS.map(v => path.basename(v)).join(', '));
        } else if (s.prompt) await page.keyboard.press('Escape');
        done = s.done;
        await page.waitForTimeout(100);
    }
    console.log('load:', done, process.env.LAZY ? '(lazy)' : '(eager)');
    const out = await page.evaluate(async (text) => {
        const P = await import('/ui/model-check-panel.js');
        const S = window.__lucid.state;
        // An ambiguous camera would open the question: pick the first candidate and say so.
        const asked = [];
        const obs = new MutationObserver(() => {
            const d = document.querySelector('.model-check-ask');
            if (d) { asked.push(d.innerText.replace(/\s+/g, ' ').slice(0, 300)); d.querySelector('#modelCheckAskOk').click(); }
        });
        obs.observe(document.body, { childList: true });
        const rep = await P.loadModelCheckText(text, 'scan.json');
        obs.disconnect();
        const cams = P.projectCameras().map(c => c.sessionName + ' · ' + c.camera + ' · ' + c.videos.join(' / '));
        const rows = [...document.querySelectorAll('#modelCheckList .model-check-row')].map(r => r.querySelector('.model-check-line1').innerText.replace(/\s+/g, ' ') + ' | ' +
            r.querySelector('.model-check-status').title);
        return { rep, cams, asked, sessions: S.sessions.length, summary: document.querySelector('.model-check-head').innerText, rows };
    }, fs.readFileSync(SCAN, 'utf8'));
    console.log('sessions:', out.sessions);
    console.log('project cameras:\n  ' + out.cams.join('\n  '));
    if (out.asked.length) console.log('asked (first candidate taken):\n  ' + out.asked.join('\n  '));
    console.log(out.summary);
    console.log('entries:\n  ' + out.rows.join('\n  '));
    const tally = {};
    out.rows.forEach(r => { const k = r.split(' | ')[1].split(':')[0]; tally[k] = (tally[k] || 0) + 1; });
    console.log('tally:', JSON.stringify(tally, null, 1));
    if (VIDEOS && process.env.SHOTS) {
        fs.mkdirSync(process.env.SHOTS, { recursive: true });
        const views = await page.evaluate(() => window.__lucid.state.views.map(v => ({ name: v.name, decoder: !!v.decoder, w: v.videoWidth, h: v.videoHeight })));
        console.log('views:', JSON.stringify(views));
        if (!process.env.GRID) { await page.keyboard.press('v'); await page.waitForTimeout(500); }
        for (let i = 0; i < out.rows.length; i++) {
            await page.evaluate((i) => document.querySelectorAll('#modelCheckList .model-check-row')[i].click(), i);
            // Wait for the frame to be decoded and the mark drawn.
            await page.waitForFunction(async () => {
                const h = (await import('/ui/model-check-highlight.js')).getModelCheckHighlight();
                return h && h.drawn;
            }, null, { timeout: 30000 }).catch(() => {});
            await page.waitForTimeout(1200);
            const info = await page.evaluate(async () => {
                const S = window.__lucid.state, h = (await import('/ui/model-check-highlight.js')).getModelCheckHighlight();
                const v = S.views.find(x => h && (x.cameraName || x.name) === h.camera && x.wrapper && x.wrapper.offsetWidth > 0);
                const r = v ? v.wrapper.closest('.dv-content-container, .view-pane, .pane') || v.wrapper : null;
                const b = r ? r.getBoundingClientRect() : null;
                return { status: document.getElementById('statusText').textContent, frame: S.currentFrame,
                    box: b ? { x: b.x, y: b.y, width: b.width, height: b.height } : null };
            });
            const file = path.join(process.env.SHOTS, 'entry-' + String(i).padStart(2, '0') + '.png');
            await page.screenshot(i === 0 || !info.box ? { path: file } : { path: file, clip: info.box });
            console.log(path.basename(file), '|', info.status);
        }
    }
    // SHOT=/path.png [ENTRY=n]: views for the cameras (no video, sized from the file's video shape),
    // single-view mode, entry n selected, a screenshot.
    if (process.env.SHOT) {
        await page.evaluate(async () => {
            const AS = window.__lucid, sp = await import('/ui/sessions-panes.js'), s = AS.state.session;
            AS.paneManager.clearAll();
            AS.state.views = s.cameras.map(c => {
                const fv = s.fileVideos && s.fileVideos.get(c.name), sh = fv && fv.shape;
                return { name: c.name, videoWidth: sh ? sh[2] : 1920, videoHeight: sh ? sh[1] : 1080, canvas: null };
            });
            AS.state.totalFrames = Math.max(...[...s.frameGroups.keys()]) + 1;
            sp.populateViewStrip();
            AS.paneManager.addAllViewsAsGrid();
        });
        await page.keyboard.press('v'); await page.waitForTimeout(300);
        await page.evaluate((i) => document.querySelectorAll('#modelCheckList .model-check-row')[i].click(), Number(process.env.ENTRY || 0));
        await page.waitForTimeout(800);
        await page.screenshot({ path: process.env.SHOT });
        console.log('screenshot:', process.env.SHOT, await page.evaluate(() => document.getElementById('statusText').textContent));
    }
} finally { await browser.close(); server.kill(); }
