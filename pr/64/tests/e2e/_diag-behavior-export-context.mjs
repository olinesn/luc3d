/**
 * _diag-behavior-export-context.mjs — save a real recording's behavior context
 * (the poses per animal, the feature spec, the broken-tracking masks) and its
 * rule detectors' bouts to a folder, so the discovery pipeline can be studied in
 * Node (scripts in the session's scratchpad) without loading the project again.
 * Loads like `_real-behavior-detect.mjs`: per-camera folder, Track All, Triangulate
 * All at a 25 px reprojection error threshold. Headless real Chrome (no lock).
 *
 * Not a test. Usage:
 *   DATASET=<folder> NANIMALS=5 LABEL=05mice OUT=<dir outside the data folders> node tests/e2e/_diag-behavior-export-context.mjs
 *   env: PORT=8771  EXCLUDE=<regex>  REPROJ=25
 * Writes OUT/<label>/context.json, animal<k>.bin (Float32, T × N × dim), broken<k>.bin (Uint8), runs.json.
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import http from 'node:http';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';
import { fileURLToPath } from 'node:url';
import { answerTrackSetup } from './fixtures/track-setup.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const DATASET = process.env.DATASET;
const NANIMALS = process.env.NANIMALS ?? '';
const LABEL = process.env.LABEL || path.basename(DATASET || 'dataset');
const PORT = Number(process.env.PORT || 8771);
const EXCLUDE = process.env.EXCLUDE || '(^|/)\\.|/troubleshooting/';
const OUT = process.env.OUT;
const REPROJ = Number(process.env.REPROJ ?? 25);
if (!DATASET || !OUT) { console.error('set DATASET and OUT'); process.exit(2); }
if (path.resolve(OUT).startsWith(path.resolve(DATASET))) { console.error('OUT must be outside the data folder'); process.exit(2); }
const dir = path.join(OUT, LABEL);
fs.mkdirSync(dir, { recursive: true });
const log = (m) => process.stdout.write(`[${new Date().toISOString().slice(11, 19)}] ${m}\n`);

// The page posts binary files here.
const sink = http.createServer((req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    if (req.method === 'OPTIONS') { res.setHeader('Access-Control-Allow-Headers', '*'); res.end(); return; }
    const name = path.basename(new URL(req.url, 'http://x').searchParams.get('name'));
    const out = fs.createWriteStream(path.join(dir, name));
    req.pipe(out);
    out.on('finish', () => { res.end('ok'); });
});
await new Promise((r) => sink.listen(PORT + 1000, r));

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise((r) => setTimeout(r, 1500));
const profileDir = fs.mkdtempSync(path.join(os.tmpdir(), 'lucid-export-'));
let context;
try {
    context = await chromium.launchPersistentContext(profileDir, {
        headless: true, channel: 'chrome', viewport: { width: 1480, height: 860 },
        args: ['--enable-precise-memory-info', '--js-flags=--expose-gc'],
    });
    const page = context.pages()[0] || await context.newPage();
    page.on('pageerror', (e) => log('pageerror: ' + String(e).slice(0, 300)));
    page.on('dialog', (d) => d.accept(String(NANIMALS)));
    await answerTrackSetup(page, NANIMALS);
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state, null, { timeout: 60000 });
    await page.evaluate(async (reproj) => {
        const st = await import('./ui/settings.js');
        st.setTrackingThresholds({ autoSwitchCheck: 0, autoImageSwitchCheck: 0, reprojErrorThreshold: reproj });
        const inp = document.createElement('input');
        inp.type = 'file'; inp.multiple = true; inp.webkitdirectory = true; inp.id = '__dir';
        inp.style.cssText = 'position:fixed;left:-9999px'; document.body.appendChild(inp);
        setInterval(() => { for (const b of document.querySelectorAll('button')) { const t = b.textContent.trim(); if ((t === 'Continue' || t === 'OK') && b.offsetParent !== null) b.click(); } }, 300);
    }, REPROJ);
    const clickStep = async (button, re, what) => {
        const prev = await page.evaluate(() => document.getElementById('statusText').textContent);
        await page.evaluate((id) => document.getElementById(id).click(), button);
        const t0 = Date.now();
        for (;;) {
            const st = await page.evaluate(() => ({ s: document.getElementById('statusText').textContent, hidden: document.getElementById('loadingOverlay').classList.contains('hidden') }));
            if (st.hidden && st.s !== prev && re.test(st.s)) { log(what + ': ' + st.s.slice(0, 160) + ` (${((Date.now() - t0) / 1000).toFixed(1)} s)`); return; }
            if (st.hidden && /error|failed/i.test(st.s) && st.s !== prev) throw new Error(what + ': ' + st.s);
            if (Date.now() - t0 > 60 * 60000) throw new Error(what + ' timed out');
            await new Promise((r) => setTimeout(r, 250));
        }
    };
    await page.setInputFiles('#__dir', DATASET);
    await page.evaluate(async (ex) => {
        const sl = await import('./loading/session-loader.js');
        const re = new RegExp(ex);
        await sl.handleLoadSessionFolderPerCamera(Array.from(document.getElementById('__dir').files).filter((f) => !re.test(f.webkitRelativePath || f.name)), false);
    }, EXCLUDE);
    await page.waitForFunction(() => window.__lucid.state.session && window.__lucid.state.session.cameras && window.__lucid.state.session.cameras.length > 0 &&
        document.getElementById('loadingOverlay').classList.contains('hidden'), null, { timeout: 60 * 60000, polling: 1000 });
    await new Promise((r) => setTimeout(r, 3000));
    log('loaded: ' + JSON.stringify(await page.evaluate(() => { const s = window.__lucid.state.session; const st = window.__lucid.state;
        return { cams: s.cameras.length, frames: st.totalFrames, fps: st.fps, lazy: !!s.lazyLoader, nodes: s.skeleton.nodes.map((n) => n.name || n) }; })));
    await clickStep('tbTrackAll', /Assigned \d+ identities/, 'Track All');
    await page.waitForSelector('#trackSummaryClose', { timeout: 10000 }).then(() => page.click('#trackSummaryClose')).catch(() => {});
    await page.evaluate(async () => { (await import('./ui/settings.js')).setDefaultTriangulationMethod('dlt'); });
    await clickStep('tbTriangulateAll', /^(Triangulated|Grouped) /, 'Triangulate All');
    const info = await page.evaluate(async (sinkPort) => {
        const det = await import('./ui/behavior-detect.js');
        const eth = await import('./pose/behavior-ethogram.js');
        const rules = await import('./pose/behavior-rules.js');
        const P = await import('./pose/behavior-predictions.js');
        const st = window.__lucid.state;
        const ctx = det.buildBehaviorContext(st.session);
        const post = (name, typed) => fetch(`http://localhost:${sinkPort}/?name=${name}`, { method: 'POST', body: new Blob([typed.buffer]) });
        for (let a = 0; a < ctx.input.data.length; a++) {
            await post(`animal${a}.bin`, ctx.input.data[a]);
            await post(`broken${a}.bin`, ctx.broken.masks[a]);
        }
        const roleNames = ctx.body.roleNames;
        rules.DETECTOR_TEMPLATES.forEach((t) => {
            const b = eth.behaviorByName(t.name) || eth.addBehavior({ name: t.name, social: t.social, mutual: t.mutual, description: t.description }).behavior;
            eth.setBehaviorDetector(b.id, rules.instantiateTemplate(t, roleNames).detector);
        });
        await det.detectAllFrames();
        const runs = P.exportPredictionRuns(st.session);
        const names = {};
        eth.behaviors().forEach((b) => { names[b.id] = b.name; });
        return {
            context: { mode: ctx.input.mode, dim: ctx.input.dim, numFrames: ctx.input.numFrames, nodeNames: ctx.input.nodeNames, edges: ctx.input.edges,
                       animals: ctx.input.animals, frames: ctx.input.frames, fps: ctx.fps, spec: ctx.spec, body: ctx.body, floor: ctx.floorInfo.label, brokenTotal: ctx.broken.total },
            runs, names,
        };
    }, PORT + 1000);
    fs.writeFileSync(path.join(dir, 'context.json'), JSON.stringify(info.context));
    fs.writeFileSync(path.join(dir, 'runs.json'), JSON.stringify({ runs: info.runs, names: info.names }));
    log('wrote ' + dir + ': ' + info.context.animals.length + ' animals, ' + info.context.numFrames + ' frames at ' + info.context.fps + ' fps, floor: ' + info.context.floor);
} finally {
    if (context) await context.close().catch(() => {});
    server.kill();
    sink.close();
    fs.rmSync(profileDir, { recursive: true, force: true });
}
