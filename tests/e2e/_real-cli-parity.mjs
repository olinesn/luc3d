/**
 * _real-cli-parity.mjs — tests/e2e/cli-parity.mjs's pass A on a real recording
 * folder (an investigation tool, not part of the suite): the app loads the
 * folder (Load Session Folder, videos deferred), runs Track All and the
 * toolbar's Triangulate All and saves; the CLI runs on the folder's per-camera
 * `.slp` files and calibration; both files are compared by digest.
 *
 * A large folder (over the app's 150 MB lazy threshold) is LAZY in the app —
 * windowed sweeps and the streaming writer — while the CLI loads it eagerly,
 * so this also compares those two paths.
 *
 *   node tests/e2e/_real-cli-parity.mjs <recording-folder> <animals> [pattern]
 *
 * `<recording-folder>/calibration.toml` and `<recording-folder>/<cam>/*<pattern>`
 * (default `.predictions.slp`); copy the folder locally first (a cold cloud
 * file reads slowly). Headless; prints both run times.
 */
import { chromium } from 'playwright';
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { digestSlp, firstDifference, compare3d } from './fixtures/slp-digest.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const DIR = path.resolve(process.argv[2] || '');
const ANIMALS = Number(process.argv[3] || 0);
const PATTERN = process.argv[4] || '.predictions.slp';
const PORT = Number(process.env.PORT || 8474);
if (!fs.existsSync(path.join(DIR, 'calibration.toml')) || !ANIMALS) {
    console.error('usage: node tests/e2e/_real-cli-parity.mjs <recording-folder> <animals> [pattern]');
    process.exit(2);
}
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'lucid-real-cli-'));
// Only the calibration and each camera's predictions go to the app (its folder loader would ask about the rest).
const stage = path.join(tmp, path.basename(DIR));
fs.mkdirSync(stage);
fs.copyFileSync(path.join(DIR, 'calibration.toml'), path.join(stage, 'calibration.toml'));
const camFiles = [];
for (const d of fs.readdirSync(DIR)) {
    const cd = path.join(DIR, d);
    if (!fs.statSync(cd).isDirectory()) continue;
    const f = fs.readdirSync(cd).find(n => n.endsWith(PATTERN));
    if (!f) continue;
    fs.mkdirSync(path.join(stage, d));
    fs.copyFileSync(path.join(cd, f), path.join(stage, d, f));   // a directory upload skips symlinks
    camFiles.push(path.join(stage, d, f));
}
console.log(camFiles.length + ' cameras');

let t = Date.now();
const r = spawnSync(process.execPath, [path.join(repoRoot, 'cli', 'track.mjs'), '--slp', ...camFiles,
    '--calibration', path.join(stage, 'calibration.toml'), '--num-animals', String(ANIMALS),
    '--out', path.join(tmp, 'cli.slp')], { encoding: 'utf8' });
console.log(r.stderr.trim().split('\n').map(l => '  cli: ' + l.slice(0, 200)).join('\n'));
console.log('CLI ' + ((Date.now() - t) / 1000).toFixed(1) + ' s, exit ' + r.status);

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(res => setTimeout(res, 1200));
const browser = await chromium.launch();
try {
    const ctx = await browser.newContext({ acceptDownloads: true });
    const page = await ctx.newPage();
    page.setDefaultTimeout(0);
    page.on('pageerror', e => console.log('  pageerror ' + String(e).slice(0, 200)));
    page.on('console', m => { if (/error|fail/i.test(m.text())) console.log('  console ' + m.text().slice(0, 300)); });
    // TRI_WORKERS=0: the app's Triangulate All solves inline (its kill switch), as the CLI does.
    if (process.env.TRI_WORKERS != null) {
        await page.addInitScript((n) => { window.LUCID_TRIANGULATION_WORKERS = n; }, Number(process.env.TRI_WORKERS));
    }
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state, null, { timeout: 60000 });
    t = Date.now();
    await page.evaluate(() => {
        const inp = document.createElement('input');
        inp.type = 'file'; inp.multiple = true; inp.webkitdirectory = true; inp.id = '__dir';
        document.body.appendChild(inp);
        setInterval(() => { for (const b of document.querySelectorAll('button')) {
            const x = b.textContent.trim(); if ((x === 'Continue' || x === 'OK') && b.offsetParent !== null) b.click(); } }, 300);
    });
    await page.setInputFiles('#__dir', stage);
    const info = await page.evaluate(async (n) => {
        const sl = await import('/loading/session-loader.js');
        await sl.handleLoadSessionFolderPerCamera(Array.from(document.getElementById('__dir').files), true);
        const s = window.__lucid.state.session;
        if (!s) return { error: document.getElementById('statusText').textContent };
        const tr = await import('/pose/tracker.js');
        tr.setTrackerNumAnimals(n);
        await tr.trackAll();
        document.querySelectorAll('.track-summary-modal, .modal-overlay').forEach(e => e.remove());
        return { lazy: !!s.lazyLoader, cams: s.cameras.length };
    }, ANIMALS);
    if (info.error) throw new Error('the app did not load the folder: ' + info.error);
    await page.evaluate(() => { document.getElementById('statusText').textContent = ''; document.getElementById('tbTriangulateAll').click(); });
    await page.waitForFunction(() => /^(Grouped|Triangulated)/.test(document.getElementById('statusText').textContent));
    const dl = page.waitForEvent('download');
    await page.evaluate(async () => { const sl = await import('/import-export/save-load.js'); await sl.saveProjectSlp(); });
    await (await dl).saveAs(path.join(tmp, 'app.slp'));
    console.log('app ' + ((Date.now() - t) / 1000).toFixed(1) + ' s (' + (info.lazy ? 'lazy' : 'eager') + ', ' + info.cams + ' cameras)');
} finally {
    await browser.close();
    server.kill();
}
const a = await digestSlp(path.join(tmp, 'app.slp')), b = await digestSlp(path.join(tmp, 'cli.slp'));
// The app's automatic ID-switch checks (Track All runs the size check on a
// recording of 60 s or more) write `idSwitchReview`; the CLI does not run them.
const keysLine = (d) => (d.text.match(/^lucidKeys .*$/m) || [''])[0];
if (keysLine(a) !== keysLine(b)) console.log('metadata.lucid keys differ (the CLI runs no ID-switch check):\n  app ' + keysLine(a) + '\n  cli ' + keysLine(b));
const strip = (d) => d.text.replace(/^lucidKeys .*$/mg, '');
const diff = firstDifference(strip(a), strip(b));
const c3 = compare3d(a, b, 1e-9);
console.log((diff ? 'DIFFERENT' : 'IDENTICAL') + ' identities, per-frame identities, groups, members and unlinked poses: ' +
    a.groups.toLocaleString('en-US') + ' groups, ' + a.with3d.toLocaleString('en-US') + ' with 3D, identities ' + a.identities.join(','));
if (diff) console.log(JSON.stringify(diff, null, 1));
console.log('3D: ' + c3.exact.toLocaleString('en-US') + ' of ' + c3.n.toLocaleString('en-US') + ' values bit-identical, largest relative difference ' +
    c3.maxRel.toExponential(2) + (c3.nanMismatch ? ', ' + c3.nanMismatch + ' NaN mismatches' : '') + (c3.ok ? ' (within 1e-9)' : ' (OVER 1e-9)'));
console.log('files in ' + tmp);
process.exit(diff || !c3.ok ? 1 : 0);
