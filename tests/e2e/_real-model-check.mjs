/**
 * _real-model-check.mjs — investigation tool, not an assertion: load a real
 * project .slp (no video) and a real model check (scan.json) the way the Model
 * Check tab does, and print how its rows matched: cameras, then each misread
 * animal's instance (verified / file order / by positions / not matched, and why).
 *
 * Run:
 *   SLP=/path/labels.slp SCAN=/path/scan.json node tests/e2e/_real-model-check.mjs
 *   LAZY=1 …   opens the .slp as a lazy project (Load Project) instead of Load SLP
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
const browser = await chromium.launch();
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
    for (let i = 0; i < 1200 && !done; i++) {
        const s = await page.evaluate(() => {
            const b = [...document.querySelectorAll('button')].find(x => /Skip|Later/i.test(x.textContent || ''));
            if (b) b.click();
            return { done: window.__loadDone, prompt: !!document.getElementById('videoSelectOverlay') };
        });
        if (s.prompt) await page.keyboard.press('Escape');
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
