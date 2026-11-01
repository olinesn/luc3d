/**
 * _real-kpms-roundtrip.mjs — the keypoint-MoSeq round trip on a real
 * recording, as a user would run it: load a per-camera session folder, Track
 * All, Triangulate All (at a reprojection error threshold), then
 *   1. Export for keypoint-MoSeq on every frame — timed, files counted but not
 *      kept — and on a stretch, written to OUT/export;
 *   2. fit keypoint-MoSeq briefly on the stretch in the given Python: the
 *      `calls` of the exported luc3d-kpms.yml as written, with AR_ITERS and
 *      FULL_ITERS iterations;
 *   3. Import keypoint-MoSeq Results… through the dialog (results.h5 and
 *      luc3d-kpms.yml): every frame's syllable must match the results CSVs;
 *   4. save the project with the streaming writer (SAVE=1, the default), as a
 *      download into OUT, and read its sessions_json back (h5py, in KPMS_PY):
 *      the states, through pose/behavior-states.js, must come back exactly.
 *      (Not re-read in the page: on top of the project's ~4.4 GB heap, a
 *      second parse of a ~1 GB file crashes the tab — nothing a user does.)
 *      The file is deleted afterwards.
 * Writes OUT/summary.json. Never writes into the dataset folder.
 *
 * Not a test (real dataset, real Chrome for HEVC, keypoint-MoSeq in Python).
 * Headless, so it takes no browser lock. Usage:
 *   DATASET=<folder> NANIMALS=5 KPMS_PY=<python with keypoint-moseq> OUT=<dir> node tests/e2e/_real-kpms-roundtrip.mjs
 *   env: PORT=8771  EXCLUDE=<regex of files to leave out>  REPROJ=25 (Tracking Wizard reprojection
 *        error threshold, px, for Triangulate All; 0 = off)  FROM=<first frame, from 1>  FRAMES=7200
 *        AR_ITERS=10  FULL_ITERS=10  SAVE=1  REUSE=1 (keep OUT/export's keypoint-MoSeq results from a
 *        previous run of the same stretch instead of fitting again)
 */
import { chromium } from 'playwright';
import { spawn, spawnSync } from 'node:child_process';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { answerTrackSetup } from './fixtures/track-setup.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const DATASET = process.env.DATASET;
const NANIMALS = process.env.NANIMALS ?? '';
const KPMS_PY = process.env.KPMS_PY;
const OUT = process.env.OUT;
const PORT = Number(process.env.PORT || 8771);
const EXCLUDE = process.env.EXCLUDE || '(^|/)\\.|/troubleshooting/';
const REPROJ = Number(process.env.REPROJ ?? 25);
const FRAMES = Number(process.env.FRAMES || 7200);
const AR_ITERS = Number(process.env.AR_ITERS || 10), FULL_ITERS = Number(process.env.FULL_ITERS || 10);
const SAVE = process.env.SAVE !== '0';
const REUSE = process.env.REUSE === '1';
if (!DATASET || !KPMS_PY || !OUT) { console.error('set DATASET, KPMS_PY and OUT'); process.exit(2); }
if (path.resolve(OUT).startsWith(path.resolve(DATASET))) { console.error('OUT must not be inside the dataset'); process.exit(2); }
const EXPORT = path.join(OUT, 'export');
if (!REUSE) fs.rmSync(EXPORT, { recursive: true, force: true });
fs.mkdirSync(EXPORT, { recursive: true });
const log = (m) => process.stdout.write(`[${new Date().toISOString().slice(11, 19)}] ${m}\n`);
const summary = { dataset: DATASET, reproj: REPROJ, stages: {} };
const timed = async (name, fn) => {
    const t0 = Date.now();
    const out = await fn();
    summary.stages[name] = Object.assign({ seconds: +((Date.now() - t0) / 1000).toFixed(1) }, out || {});
    log(name + ': ' + JSON.stringify(summary.stages[name]).slice(0, 400));
    return out;
};

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise((r) => setTimeout(r, 1500));
let browser;
try {
    browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-precise-memory-info'] });
    const context = await browser.newContext({ viewport: { width: 1480, height: 860 }, acceptDownloads: true });
    const page = await context.newPage();
    page.on('pageerror', (e) => log('pageerror: ' + String(e).slice(0, 300)));
    page.on('dialog', (d) => d.accept(String(NANIMALS)));
    await answerTrackSetup(page, NANIMALS);
    await page.exposeFunction('__kpmsWrite', (name, text) => { fs.writeFileSync(path.join(EXPORT, path.basename(name)), text); return text.length; });
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
            if (st.hidden && st.s !== prev && re.test(st.s)) return { status: st.s.slice(0, 160) };
            if (st.hidden && /error|failed/i.test(st.s) && st.s !== prev) throw new Error(what + ': ' + st.s);
            if (Date.now() - t0 > 60 * 60000) throw new Error(what + ' timed out');
            await new Promise((r) => setTimeout(r, 250));
        }
    };

    // ---- load, Track All, Triangulate All
    await timed('load', async () => {
        await page.setInputFiles('#__dir', DATASET);
        await page.evaluate(async (ex) => {
            const sl = await import('./loading/session-loader.js');
            const re = new RegExp(ex);
            await sl.handleLoadSessionFolderPerCamera(Array.from(document.getElementById('__dir').files).filter((f) => !re.test(f.webkitRelativePath || f.name)), false);
        }, EXCLUDE);
        await page.waitForFunction(() => window.__lucid.state.session && window.__lucid.state.session.cameras && window.__lucid.state.session.cameras.length > 1 &&
            document.getElementById('loadingOverlay').classList.contains('hidden'), null, { timeout: 60 * 60000, polling: 1000 });
        await new Promise((r) => setTimeout(r, 3000));
        return page.evaluate(() => { const s = window.__lucid.state.session, st = window.__lucid.state;
            return { cams: s.cameras.length, frames: st.totalFrames, fps: st.fps, lazy: !!s.lazyLoader, nodes: s.skeleton.nodes.length, name: s.name }; });
    });
    await timed('Track All', () => clickStep('tbTrackAll', /Assigned \d+ identities/, 'Track All'));
    await page.waitForSelector('#trackSummaryClose', { timeout: 10000 }).then(() => page.click('#trackSummaryClose')).catch(() => {});
    await timed('Triangulate All', () => clickStep('tbTriangulateAll', /^(Triangulated|Grouped) /, 'Triangulate All'));
    const info = await page.evaluate(() => ({ frames: window.__lucid.state.totalFrames, fps: window.__lucid.state.fps,
        animals: window.__lucid.state.session.identities.map((i) => i.name), session: window.__lucid.state.session.name }));
    summary.session = info;
    const from = Math.max(1, Number(process.env.FROM || Math.floor(info.frames / 2) - Math.floor(FRAMES / 2)));
    const first = from - 1, last = Math.min(info.frames - 1, first + FRAMES - 1);

    // ---- 1. export
    await timed('export, every frame (not kept)', () => page.evaluate(async () => {
        const K = await import('./ui/behavior-kpms-modal.js');
        let bytes = 0;
        const t0 = performance.now();
        const res = await K.runKpmsExport({ scope: 'one', first: 0, last: Infinity }, null, { write: async (name, parts) => { for (const p of parts) bytes += p.length; } });
        return { ms: Math.round(performance.now() - t0), files: res && res.files.length, MB: +(bytes / 1048576).toFixed(1), leftOut: res && res.leftOut,
                 heapMB: Math.round(performance.memory.usedJSHeapSize / 1048576), status: document.getElementById('statusText').textContent.slice(0, 200) };
    }));
    await timed('export, frames ' + from + '–' + (last + 1), () => page.evaluate(async ([a, b]) => {
        const K = await import('./ui/behavior-kpms-modal.js');
        const res = await K.runKpmsExport({ scope: 'one', first: a, last: b }, null, { write: async (name, parts) => { await window.__kpmsWrite(name, parts.join('')); } });
        return { files: res && res.files, leftOut: res && res.leftOut };
    }, [first, last]));
    const yml = fs.readFileSync(path.join(EXPORT, 'luc3d-kpms.yml'), 'utf8');
    summary.manifestHead = yml.split('\n').filter((l) => /^(format|dims|units|coordinates|fps|use_bodyparts|anterior_bodyparts|posterior_bodyparts):/.test(l));

    // ---- 2. keypoint-MoSeq, briefly
    const proj = path.join(EXPORT, 'kpms_project');
    const fitted = () => fs.existsSync(proj) && fs.readdirSync(proj).some((d) => fs.existsSync(path.join(proj, d, 'results.h5')));
    if (!(REUSE && fitted())) await timed('keypoint-MoSeq fit', async () => {
        const script = [
            'import os, sys, yaml',
            "calls = yaml.safe_load(open('luc3d-kpms.yml'))['calls']",
            "assert 'AR_ITERS, FULL_ITERS = 50, 500' in calls",
            "calls = calls.replace('AR_ITERS, FULL_ITERS = 50, 500', 'AR_ITERS, FULL_ITERS = " + AR_ITERS + ', ' + FULL_ITERS + "')",
            "exec(compile(calls, 'luc3d-kpms.yml (calls)', 'exec'), {})",
        ].join('\n');
        const r = spawnSync(KPMS_PY, ['-c', script], { cwd: EXPORT, encoding: 'utf8', maxBuffer: 256 << 20 });
        fs.writeFileSync(path.join(OUT, 'kpms.log'), (r.stdout || '') + '\n' + (r.stderr || ''));
        if (r.status !== 0) throw new Error('keypoint-MoSeq failed (see ' + path.join(OUT, 'kpms.log') + ')');
        return { log: path.join(OUT, 'kpms.log') };
    });
    const model = fs.readdirSync(proj).filter((d) => fs.existsSync(path.join(proj, d, 'results.h5')))[0];
    const resultsDir = path.join(proj, model, 'results');
    const expected = {};
    for (const f of fs.readdirSync(resultsDir).filter((n) => n.endsWith('.csv'))) {
        expected[f.replace(/\.csv$/, '')] = fs.readFileSync(path.join(resultsDir, f), 'utf8').trim().split('\n').slice(1).map((l) => Number(l.split(',')[0]));
    }
    summary.model = model;

    // ---- 3. import through the dialog
    await timed('import', async () => {
        await page.evaluate(async () => (await import('./ui/behavior-kpms-modal.js')).showKpmsImportModal());
        const [chooser] = await Promise.all([page.waitForEvent('filechooser'), page.click('#kiFiles')]);
        await chooser.setFiles([path.join(proj, model, 'results.h5'), path.join(EXPORT, 'luc3d-kpms.yml')]);
        await page.waitForSelector('.kpms-recordings tbody tr', { timeout: 120000 });
        const rows = await page.$$eval('.kpms-recordings tbody tr', (trs) => trs.map((tr) => [tr.cells[0].textContent, tr.cells[1].textContent,
            tr.querySelector('select').selectedOptions[0].textContent, tr.querySelector('input').value]));
        const summaryText = await page.textContent('#kiSummary');
        await page.click('#kiGo');
        await page.waitForFunction(() => /Imported keypoint-MoSeq syllables/.test(document.getElementById('statusText').textContent), null, { timeout: 120000 });
        return { rows, summary: summaryText, status: await page.textContent('#statusText') };
    });
    const check = await page.evaluate(async ([exp, man]) => {
        const S = await import('./pose/behavior-states.js');
        const st = window.__lucid.state;
        const L = S.stateLayer(st.session, 'keypoint-MoSeq');
        const bad = [];
        for (const name of Object.keys(exp)) {
            const r = man[name];
            for (let i = 0; i < exp[name].length; i++) {
                if (S.stateAt(L, r.animal, r.first_frame + i) !== exp[name][i]) { bad.push(name + '@' + i); break; }
            }
        }
        // Nothing outside the stretch.
        const outside = [];
        L.animals.forEach((runs, a) => { if (runs[0] < man[Object.keys(man)[0]].first_frame || runs[runs.length - 2] > man[Object.keys(man)[0]].first_frame + man[Object.keys(man)[0]].frames - 1) outside.push(a); });
        const sum = S.stateSummary([L]);
        const rows = window.__lucid.state && (await import('./ui/app-state.js')).timeline._trackSegments.filter((x) => x._isBehavior && x.cameraName === 'States').length;
        return { bad, outside, syllables: sum.states.length, runs: sum.states.reduce((t, s) => t + s.runs, 0),
                 top: sum.states.slice(0, 8).map((s) => [s.state, +(100 * s.share).toFixed(1), +(s.medianFrames / st.fps).toFixed(3)]),
                 timelineRows: rows, payloadKB: +(JSON.stringify(S.exportStateLayers(st.session)).length / 1024).toFixed(1) };
    }, [expected, (await import('../../import-export/kpms-io.js')).parseKpmsManifest(yml).value.recordings]);
    summary.verify = check;
    log('verify: ' + JSON.stringify(check));

    // ---- 4. save, then read the file's session metadata back
    if (SAVE) {
        const slpPath = path.join(OUT, 'project.slp');
        await timed('save (streaming writer)', async () => {
            const [dl, info] = await Promise.all([
                page.waitForEvent('download', { timeout: 60 * 60000 }),
                page.evaluate(async () => {
                    const saveLoad = await import('./import-export/save-load.js');
                    const S = await import('./pose/behavior-states.js');
                    const st = window.__lucid.state;
                    const t0 = performance.now();
                    const bytes = await saveLoad.saveAllSessionsStreaming(st.sessions);
                    const ms = Math.round(performance.now() - t0);
                    const a = document.createElement('a');
                    a.href = URL.createObjectURL(new Blob([bytes]));
                    a.download = 'project.slp';
                    document.body.appendChild(a);
                    a.click();
                    return { ms, MB: +(bytes.byteLength / 1048576).toFixed(1), states: JSON.stringify(S.exportStateLayers(st.session)) };
                }),
            ]);
            await dl.saveAs(slpPath);
            summary.savedStatesKB = +(info.states.length / 1024).toFixed(1);
            summary._savedStates = info.states;
            return { ms: info.ms, MB: info.MB };
        });
        await timed('read back (sessions_json, h5py)', async () => {
            const py = [
                'import h5py, json, sys',
                "f = h5py.File(sys.argv[1], 'r')",
                'out = []',
                "for e in f['sessions_json'][()]:",
                '    d = json.loads(e)',
                "    l = (d.get('metadata') or {}).get('lucid') or {}",
                "    out.append({'name': l.get('sessionName'), 'states': l.get('behaviorStates'), 'keys': sorted(l.keys())})",
                'print(json.dumps(out))',
            ].join('\n');
            const r = spawnSync(KPMS_PY, ['-c', py, slpPath], { encoding: 'utf8', maxBuffer: 256 << 20 });
            if (r.status !== 0) throw new Error('reading sessions_json failed: ' + r.stderr.slice(-400));
            const sessions = JSON.parse(r.stdout);
            const S = await import('../../pose/behavior-states.js');
            const stub = { name: 'reopened' };
            S.importStateLayers(stub, sessions[0].states);
            const same = JSON.stringify(S.exportStateLayers(stub)) === summary._savedStates;
            delete summary._savedStates;
            return { same, sessions: sessions.length, behaviorKeys: sessions[0].keys.filter((k) => /^(ethogram|behavior)/.test(k)) };
        });
        fs.rmSync(slpPath, { force: true });
    }
    fs.writeFileSync(path.join(OUT, 'summary.json'), JSON.stringify(summary, null, 1));
    log('wrote ' + path.join(OUT, 'summary.json'));
    const ok = check.bad.length === 0 && check.outside.length === 0 && (!SAVE || summary.stages['read back (sessions_json, h5py)'].same);
    log(ok ? 'ROUND TRIP OK' : 'ROUND TRIP FAILED');
    process.exitCode = ok ? 0 : 1;
} finally {
    if (browser) await browser.close().catch(() => {});
    server.kill();
}
