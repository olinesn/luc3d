/**
 * cli-parity.mjs — the command-line LUCID (cli/track.mjs) writes the same
 * project as the app: Track All, then the toolbar's Triangulate All, then
 * File ▸ Save, in a headless browser, against the CLI in plain Node on the same
 * input. Both files are read back with the app's reader and compared as a
 * digest of every identity, per-frame identity entry, group member, unlinked
 * pose and 3D coordinate (tests/e2e/fixtures/slp-digest.mjs). The same code runs
 * on both sides: everything must be identical, and the 3D bit-identical up to
 * the engines' own last-bit differences — Chrome's and Node's V8 builds can
 * disagree by an ulp in Math.atan2 / cos / sin / exp / log, which the DLT
 * solve's Jacobi fallback uses — so 3D is held to 1e-9 relative and the
 * count of bit-identical values is reported (`compare3d`).
 *
 * Input: sample_session's four cameras (their clips, the demo rig's
 * calibration) with seeded, untidy 3-animal predictions, one `.slp` per camera
 * (tests/e2e/fixtures/cli-scene.mjs).
 *  A. Per-camera files, Load Session Folder in the app (videos deferred) vs
 *     `--slp <cam files> --calibration`: Triangulate All runs Group by
 *     Identity & Triangulate (DLT, the default).
 *  B. A project `.slp` (that folder, loaded and saved by the app untracked),
 *     File ▸ Load SLP vs `--slp project.slp`, with the Refined method
 *     (`--settings`): Triangulate All runs `triangulateAllFrames('ba')`.
 *  C. The same project, Track Frame Range 100–299 then Triangulate All vs
 *     `--frames 100-299`.
 *  D. Track All's ID-switch check by body size: a 120 s scene on a skeleton
 *     with the size nodes; the app checks at its fps (the clips' 30), the CLI
 *     at `--fps`; both save the same ID Switches checklist (`idSwitchReview`,
 *     in the digest). Without `--fps` the CLI skips the check and saves none.
 *  E. Negative control: B's CLI output against a DLT run of the CLI on the
 *     same project differs (the comparison sees the 3D), and A's against C's.
 * Not compared: video paths and shapes (the app writes the videos it opened,
 * the CLI the ones the input referenced) and provenance timestamps.
 *
 * Run: node tests/e2e/cli-parity.mjs   (PORT= to move its http server)
 */
import { chromium } from 'playwright';
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { writeCliScene } from './fixtures/cli-scene.mjs';
import { digestSlp, firstDifference, compare3d } from './fixtures/slp-digest.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8473);
let fails = 0;
const check = (c, m, extra) => {
    console.log((c ? '  ✓ ' : '  ✗ ') + m);
    if (!c) { fails++; if (extra !== undefined) console.log('      ' + JSON.stringify(extra).slice(0, 900)); }
};

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'lucid-cli-parity-'));
const sceneDir = path.join(tmp, 'scene');
const scene = await writeCliScene(sceneDir, { frames: 600 });
const camFiles = scene.cameras.map(c => path.join(sceneDir, c, c + '.predictions.slp'));
const out = (n) => path.join(tmp, n);

function cli(args) {
    const r = spawnSync(process.execPath, [path.join(repoRoot, 'cli', 'track.mjs'), ...args], { encoding: 'utf8' });
    if (r.status !== 0) console.log(r.stdout + r.stderr);
    return r;
}

// 3D values may differ by an ulp between Chrome's and Node's V8 (see compare3d).
const REL_TOL_3D = 1e-9;

async function compare(label, appFile, cliFile) {
    const a = await digestSlp(appFile), b = await digestSlp(cliFile);
    const diff = firstDifference(a.text, b.text);
    check(!diff, label + ': app and CLI files hold the same identities, per-frame identities, groups, members and ' +
        'unlinked poses (' + a.groups.toLocaleString('en-US') + ' groups, ' + a.with3d.toLocaleString('en-US') +
        ' with 3D, identities ' + a.identities.join(',') + ')', diff || undefined);
    const c = compare3d(a, b, REL_TOL_3D);
    check(c.ok, label + ': and the same 3D: ' + c.exact.toLocaleString('en-US') + ' of ' + c.n.toLocaleString('en-US') +
        ' values bit-identical' + (c.exact < c.n ? ', the rest within ' + c.maxRel.toExponential(1) + ' relative' : ''), c.ok ? undefined : c);
    return { a, b, c };
}

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1200));
let browser;
try {
    browser = await chromium.launch();

    // A fresh page per pass (its own storage, so a setting cannot leak between passes).
    async function openApp() {
        const ctx = await browser.newContext({ acceptDownloads: true, viewport: { width: 1440, height: 900 } });
        const page = await ctx.newPage();
        page.errs = [];
        page.on('pageerror', e => page.errs.push(String(e).slice(0, 300)));
        await page.goto(`http://localhost:${PORT}/index.html`);
        await page.waitForFunction(() => window.__lucid && window.__lucid.state, null, { timeout: 30000 });
        return page;
    }
    // File ▸ Save in a browser without a save-file picker: the app's own bytes, as a download.
    async function save(page, file) {
        const dl = page.waitForEvent('download', { timeout: 60000 });
        await page.evaluate(async () => { const sl = await import('/import-export/save-load.js'); await sl.saveProjectSlp(); });
        await (await dl).saveAs(file);
        return file;
    }
    async function trackAll(page, animals) {
        return page.evaluate(async (n) => {
            const tr = await import('/pose/tracker.js');
            tr.setTrackerNumAnimals(n);                  // the Track All dialog's answer
            const r = await tr.trackAll();
            document.querySelectorAll('.track-summary-modal, .modal-overlay').forEach(e => e.remove());
            return r && r.ok;
        }, animals);
    }
    // The toolbar's Triangulate All button (its handler picks the pass), until the run's status line.
    async function triangulateAll(page) {
        await page.evaluate(() => {
            document.getElementById('statusText').textContent = '';
            document.getElementById('tbTriangulateAll').click();
        });
        await page.waitForFunction(() => /^(Grouped|Triangulated)/.test(document.getElementById('statusText').textContent),
            null, { timeout: 120000 });
        return page.evaluate(() => document.getElementById('statusText').textContent);
    }
    async function loadProject(page, file) {
        await page.evaluate(() => {
            const inp = document.createElement('input');
            inp.type = 'file'; inp.id = '__proj'; inp.style.cssText = 'position:fixed;left:-9999px';
            document.body.appendChild(inp);
        });
        await page.setInputFiles('#__proj', file);
        await page.evaluate(() => {
            window.__loadDone = false;
            import('/import-export/slp-import.js').then(imp => imp.handleLoadSlpFile(document.getElementById('__proj').files[0]))
                .then(() => { window.__loadDone = true; }, e => { window.__loadDone = 'ERR ' + (e && e.stack || e); });
        });
        for (let i = 0; i < 400; i++) {
            const s = await page.evaluate(() => ({ done: window.__loadDone, prompt: !!document.getElementById('videoSelectOverlay') }));
            if (s.prompt) await page.keyboard.press('Escape');     // no videos: cancel the folder prompt
            if (s.done) return s.done;
            await page.waitForTimeout(250);
        }
        return 'timeout';
    }

    // ---- A. per-camera files, Track All, Triangulate All (DLT) ----
    console.log('\n-- A. per-camera .slp files (Load Session Folder) --');
    let page = await openApp();
    await page.evaluate(() => {
        const inp = document.createElement('input');
        inp.type = 'file'; inp.multiple = true; inp.webkitdirectory = true; inp.id = '__dir';
        inp.style.cssText = 'position:fixed;left:-9999px';
        document.body.appendChild(inp);
    });
    await page.setInputFiles('#__dir', sceneDir);
    const loaded = await page.evaluate(async () => {
        const sl = await import('/loading/session-loader.js');
        await sl.handleLoadSessionFolderPerCamera(Array.from(document.getElementById('__dir').files), true);
        const s = window.__lucid.state.session;
        return s ? { cams: s.cameras.map(c => c.name), frames: s.frameGroups.size } : null;
    });
    check(loaded && loaded.cams.length === 4 && loaded.frames === scene.frames,
        'the app loaded the folder: ' + JSON.stringify(loaded));
    const projectFile = await save(page, out('project-untracked.slp'));   // B's and C's input
    check(await trackAll(page, scene.animals), 'app: Track All finished');
    const triA = await triangulateAll(page);
    check(/^Grouped/.test(triA), 'app: Triangulate All grouped by identity (DLT): ' + triA.slice(0, 120));
    await save(page, out('A-app.slp'));
    check(page.errs.length === 0, 'app: no page errors', page.errs);
    await page.context().close();

    let r = cli(['--slp', ...camFiles, '--calibration', path.join(sceneDir, 'calibration.toml'),
        '--num-animals', String(scene.animals), '--out', out('A-cli.slp'), '--quiet']);
    check(r.status === 0, 'CLI: per-camera run exited 0');
    const A = await compare('A', out('A-app.slp'), out('A-cli.slp'));
    check(A.a.with3d > 0 && A.a.identities.length === scene.animals, 'A: tracked ' + scene.animals + ' identities with 3D');

    // ---- B. project .slp, Refined (BA) ----
    console.log('\n-- B. project .slp (File ▸ Load SLP), Refined triangulation --');
    page = await openApp();
    check((await loadProject(page, projectFile)) === true, 'app: File ▸ Load SLP opened the untracked project');
    await page.evaluate(async () => { (await import('/ui/settings.js')).setDefaultTriangulationMethod('ba'); });
    check(await trackAll(page, scene.animals), 'app: Track All finished');
    const triB = await triangulateAll(page);
    check(/^Triangulated/.test(triB) && /Refined/.test(triB), 'app: Triangulate All ran Refined on every group: ' + triB.slice(0, 120));
    await save(page, out('B-app.slp'));
    check(page.errs.length === 0, 'app: no page errors', page.errs);
    await page.context().close();

    fs.writeFileSync(out('ba.json'), JSON.stringify({ triangulationMethod: 'ba' }));
    r = cli(['--slp', projectFile, '--num-animals', String(scene.animals), '--settings', out('ba.json'), '--out', out('B-cli.slp'), '--quiet']);
    check(r.status === 0, 'CLI: project run (Refined) exited 0');
    const B = await compare('B', out('B-app.slp'), out('B-cli.slp'));

    // ---- C. Track Frame Range ----
    console.log('\n-- C. Track Frame Range 100-299 (0-based), then Triangulate All --');
    page = await openApp();
    check((await loadProject(page, projectFile)) === true, 'app: opened the untracked project');
    const rangeOk = await page.evaluate(async () => {
        const tr = await import('/pose/tracker.js');
        tr.setTrackerNumAnimals(3);
        const res = await tr.trackFrameRange(100, 299);
        return res && res.ok && res.start === 100 && res.end === 299;
    });
    check(rangeOk, 'app: Track Frame Range 100-299 finished');
    await triangulateAll(page);
    await save(page, out('C-app.slp'));
    await page.context().close();
    r = cli(['--slp', projectFile, '--num-animals', '3', '--frames', '100-299', '--out', out('C-cli.slp'), '--quiet']);
    check(r.status === 0, 'CLI: --frames run exited 0');
    const C = await compare('C', out('C-app.slp'), out('C-cli.slp'));
    // A frame's digest block: its index, its group lines (none when untracked), its unlinked line.
    check(/\n99\n  \n  U /.test(C.b.text) && /\n100\n  G /.test(C.b.text) &&
        /\n299\n  G /.test(C.b.text) && /\n300\n  \n  U /.test(C.b.text),
        'C: frames 100 and 299 are tracked, 99 and 300 are not (the range is 0-based and inclusive)');

    // ---- D. Track All's ID-switch check by body size ----
    // A 120 s scene (the check needs 60 s of tracking) on a skeleton with the size nodes.
    // The app checks at its fps box's rate (the clips' 30 fps); the CLI at --fps.
    console.log('\n-- D. ID-switch check by body size (--fps) --');
    const sizeDir = path.join(tmp, 'scene-size');
    const sizeScene = await writeCliScene(sizeDir, { frames: 3600, sizeSkeleton: true, seed: 11 });
    page = await openApp();
    await page.evaluate(() => {
        const inp = document.createElement('input');
        inp.type = 'file'; inp.multiple = true; inp.webkitdirectory = true; inp.id = '__dir';
        inp.style.cssText = 'position:fixed;left:-9999px';
        document.body.appendChild(inp);
    });
    await page.setInputFiles('#__dir', sizeDir);
    const appFps = await page.evaluate(async () => {
        const sl = await import('/loading/session-loader.js');
        await sl.handleLoadSessionFolderPerCamera(Array.from(document.getElementById('__dir').files), true);
        return window.__lucid.state.session ? window.__lucid.state.fps : null;
    });
    check(appFps > 0, 'app: loaded the 120 s scene, fps ' + appFps);
    check(await trackAll(page, sizeScene.animals), 'app: Track All finished (and ran its size check)');
    await triangulateAll(page);
    await save(page, out('D-app.slp'));
    check(page.errs.length === 0, 'app: no page errors', page.errs);
    await page.context().close();
    const sizeFiles = sizeScene.cameras.map(c => path.join(sizeDir, c, c + '.predictions.slp'));
    r = cli(['--slp', ...sizeFiles, '--calibration', path.join(sizeDir, 'calibration.toml'),
        '--num-animals', String(sizeScene.animals), '--fps', String(appFps), '--out', out('D-cli.slp')]);
    check(r.status === 0 && /ID-switch check \(body size\): .* close encounters at /.test(r.stderr),
        'CLI: --fps ran the size check: ' + ((r.stderr.match(/ID-switch check.*$/m) || [''])[0]));
    const D = await compare('D', out('D-app.slp'), out('D-cli.slp'));
    const reviewOf = (d) => (d.text.match(/^idSwitchReview (.*)$/m) || [])[1];
    const rv = reviewOf(D.a) && JSON.parse(reviewOf(D.a));
    check(rv && rv.checks && rv.checks.size && rv.checks.size.encounters > 0 && reviewOf(D.a) === reviewOf(D.b),
        'D: both saved the same ID Switches checklist (' + (rv ? rv.checks.size.encounters + ' close encounters, ' +
        rv.checks.size.points.length + ' flagged' : 'none') + ')');
    r = cli(['--slp', ...sizeFiles, '--calibration', path.join(sizeDir, 'calibration.toml'),
        '--num-animals', String(sizeScene.animals), '--out', out('D-cli-nofps.slp')]);
    check(r.status === 0 && /body size\): skipped .*--fps/.test(r.stderr) && !reviewOf(await digestSlp(out('D-cli-nofps.slp'))),
        'CLI: without --fps the check is skipped, said so, and nothing is saved for it');

    // ---- E. negative controls ----
    console.log('\n-- E. negative controls --');
    r = cli(['--slp', projectFile, '--num-animals', '3', '--out', out('B-cli-dlt.slp'), '--quiet']);
    const dlt = await digestSlp(out('B-cli-dlt.slp'));
    check(!compare3d(dlt, B.b, REL_TOL_3D).ok, 'a DLT run differs from the Refined one in 3D (the comparison sees the 3D)');
    check(firstDifference(A.b.text, C.b.text) != null, 'a frame-range run differs from a whole-project run');
} finally {
    if (browser) await browser.close();
    server.kill();
    if (!fails) fs.rmSync(tmp, { recursive: true, force: true });
    else console.log('  (files kept in ' + tmp + ')');
}
console.log(fails ? `\n${fails} check(s) FAILED` : '\nall checks passed');
process.exit(fails ? 1 : 0);
