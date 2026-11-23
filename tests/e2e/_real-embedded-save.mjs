/**
 * _real-embedded-save.mjs — a real package, saved over itself. Needs a real file, so it
 * is not part of the suite (see `tests/e2e/README.md`).
 *
 * Copies the package into the browser's private storage, where a file handle behaves as
 * one on disk does, opens that copy as File ▸ Open does and edits one node. Then saves
 * over the file the images are being read from, twice, and reports the time and the
 * memory the browser's processes used. Writes the result to OUT for Python to compare:
 *
 *   node tests/e2e/_real-embedded-save.mjs ~/Desktop/labels.pkg.slp OUT_DIR [disk|memory]
 *   uv run --no-project --python 3.12 --with "sleap-io==0.9.5" --with h5py --with numpy \
 *       python tests/fixtures/slp-embedded/compare_saved.py ~/Desktop/labels.pkg.slp OUT_DIR/saved.slp
 */
import { chromium } from 'playwright';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const [SOURCE, OUT, SCRATCH] = process.argv.slice(2);
if (!SOURCE || !OUT) { console.error('usage: _real-embedded-save.mjs package.slp out_dir [disk|memory]'); process.exit(2); }
fs.mkdirSync(OUT, { recursive: true });
const PORT = Number(process.env.PORT || 8702);
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm' };

// The app's files, the source package to download, and somewhere to put the result.
const server = http.createServer((req, res) => {
    const url = new URL(req.url, 'http://localhost');
    if (req.method === 'POST') {
        const to = fs.createWriteStream(path.join(OUT, path.basename(url.pathname)));
        req.pipe(to).on('finish', () => res.end('ok'));
        return;
    }
    const file = url.pathname === '/__source' ? SOURCE : path.join(repoRoot, url.pathname === '/' ? 'index.html' : url.pathname);
    if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.statusCode = 404; res.end(); return; }
    res.setHeader('Content-Type', TYPES[path.extname(file)] || 'application/octet-stream');
    fs.createReadStream(file).pipe(res);
});
await new Promise((res) => server.listen(PORT, res));

const TAG = '--lucid-real-embedded-save=' + process.pid;
// A profile on disk: Playwright's default profile is an incognito one, which keeps the
// browser's private storage in memory and so measures nothing here.
const profile = fs.mkdtempSync(path.join(OUT, 'profile-'));
const browser = await chromium.launchPersistentContext(profile, { args: [TAG], viewport: { width: 1440, height: 900 } });

/** Resident memory of the browser's processes, in MB: the largest one, and all of them. */
function memory() {
    const rows = execFileSync('ps', ['-axo', 'pid=,ppid=,rss=,command='], { encoding: 'utf8', maxBuffer: 1 << 26 }).split('\n')
        .map((l) => l.trim().match(/^(\d+)\s+(\d+)\s+(\d+)\s+(.*)$/)).filter(Boolean)
        .map((m) => ({ pid: +m[1], ppid: +m[2], rss: +m[3], command: m[4] }));
    const mine = new Set(rows.filter((r) => r.command.includes(TAG)).map((r) => r.pid));
    for (let grew = true; grew;) {
        grew = false;
        for (const r of rows) if (mine.has(r.ppid) && !mine.has(r.pid)) { mine.add(r.pid); grew = true; }
    }
    const sizes = rows.filter((r) => mine.has(r.pid)).map((r) => r.rss / 1024);
    const kinds = {};
    for (const r of rows) if (mine.has(r.pid)) {
        const m = r.command.match(/--utility-sub-type=([^ ]+)/) || r.command.match(/--type=([^ ]+)/);
        const k = m ? m[1].split('.').pop() : 'browser';
        kinds[k] = (kinds[k] || 0) + Math.round(r.rss / 1024);
    }
    return { largest: Math.max(0, ...sizes), total: sizes.reduce((a, b) => a + b, 0), kinds };
}

let watching = false;
let peak = { largest: 0, total: 0 };
let peakKinds = {};
const watch = () => {
    if (!watching) return;
    const now = memory();
    if (now.total > peak.total) peakKinds = now.kinds;
    peak = { largest: Math.max(peak.largest, now.largest), total: Math.max(peak.total, now.total) };
    setTimeout(watch, 200);
};
const round = (m) => ({ largestMB: Math.round(m.largest), totalMB: Math.round(m.total) });

try {
    const page = await browser.newPage();
    page.on('pageerror', (e) => console.log('  [pageerror]', String(e).slice(0, 300)));
    page.on('console', (m) => { if (m.type() === 'error') console.log('  [console]', m.text().slice(0, 300)); });
    page.on('dialog', (d) => d.accept());
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state && window.SleapIO, null, { timeout: 60000 });

    console.log('copying the package into private storage…');
    const opened = await page.evaluate(async (scratch) => {
        if (scratch) window.__lucidSaveScratch = scratch;
        const root = await navigator.storage.getDirectory();
        const handle = await root.getFileHandle('real.pkg.slp', { create: true });
        const writable = await handle.createWritable();
        await (await fetch('/__source')).body.pipeTo(writable);
        window.__handle = handle;
        const t0 = performance.now();
        const { handleLoadSlpFile } = await import('/import-export/slp-import.js');
        const file = await handle.getFile();
        window.__firstFile = file;
        const ok = await handleLoadSlpFile(file);
        window.__lucid.state.slpFileHandle = handle;
        const s = window.__lucid.state;
        return {
            ok: String(ok), ms: Math.round(performance.now() - t0), sessions: s.sessions.length, videoFiles: s.videoFiles.length,
            size: file.size, status: document.getElementById('statusText').textContent,
            leftOut: s.embeddedImagesLeftOut,
        };
    }, SCRATCH || null);
    console.log('opened', JSON.stringify(opened));
    await page.waitForTimeout(1000);

    const edit = await page.evaluate(async () => {
        const s = window.__lucid.state;
        const sess = s.session;
        for (const [frame, fg] of sess.frameGroups) {
            for (const [cam, list] of fg.instances) {
                if (list.length > 0) {
                    list[0].setPoint(0, 123.25, 234.5);
                    return { session: s.activeSessionIdx, frame, cam, type: list[0].type, linked: true };
                }
            }
            for (const [cam, list] of fg.unlinkedInstances) {
                if (list.length > 0) {
                    list[0].instance.setPoint(0, 123.25, 234.5);
                    return { session: s.activeSessionIdx, frame, cam, type: list[0].instance.type, linked: false };
                }
            }
        }
        return null;
    });
    console.log('edited', JSON.stringify(edit));

    const shown = () => page.evaluate(async () => {
        const s = window.__lucid.state;
        const { videoController } = await import('/ui/app-state.js');
        const frame = s.currentFrame;
        await videoController.seekToFrame(frame);
        const out = {};
        for (const v of s.views) {
            const d = v.ctx.getImageData(0, 0, v.canvas.width, v.canvas.height).data;
            let sum = 0;
            for (let i = 0; i < d.length; i += 4097) sum += d[i];
            out[v.name] = sum;
        }
        return { frame, out, key: s.views[0].decoder.videoInfo.key };
    });

    const before = await shown();
    console.log('before', JSON.stringify(before).slice(0, 200));

    for (const pass of [1, 2]) {
        const base = memory();
        peak = { largest: 0, total: 0 };
        watching = true;
        watch();
        const result = await page.evaluate(async () => {
            const t0 = performance.now();
            const { quickSave } = await import('/import-export/save-load.js');
            await quickSave();
            const file = await window.__handle.getFile();
            let stale = null;
            try { await window.__firstFile.slice(0, 16).arrayBuffer(); stale = false; } catch (e) { stale = e.name; }
            const root = await navigator.storage.getDirectory();
            const entries = [];
            for await (const k of root.keys()) entries.push(k);
            return {
                ms: Math.round(performance.now() - t0), status: document.getElementById('statusText').textContent,
                bytes: file.size, firstFileReads: stale, entries, workers: window.__lucid.state.frameWorkers.length,
                dirty: window.__lucid.state.isDirty,
            };
        });
        watching = false;
        console.log(`save ${pass}`, JSON.stringify(result));
        console.log(`  memory before`, JSON.stringify(round(base)), 'peak', JSON.stringify(round(peak)), 'added', Math.round(peak.total - base.total), 'MB', 'before', JSON.stringify(base.kinds), 'at peak', JSON.stringify(peakKinds));
        const after = await shown();
        console.log('  shows the same', JSON.stringify(after.out) === JSON.stringify(before.out), 'key', after.key);
        await page.waitForTimeout(500);
    }

    await page.evaluate(async () => {
        const file = await window.__handle.getFile();
        await fetch('/saved.slp', { method: 'POST', body: file });
    });
    console.log('wrote', path.join(OUT, 'saved.slp'), fs.statSync(path.join(OUT, 'saved.slp')).size);
    await page.evaluate(async () => {
        const root = await navigator.storage.getDirectory();
        for await (const k of root.keys()) { try { await root.removeEntry(k); } catch (e) { /* in use */ } }
    });
    await page.close();
} finally {
    watching = false;
    await browser.close();
    server.close();
    fs.rmSync(profile, { recursive: true, force: true });
}
