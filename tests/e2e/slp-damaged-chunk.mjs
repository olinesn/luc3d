/**
 * slp-damaged-chunk.mjs — a damaged part of a compressed table is an error, never a
 * project with whatever was in memory in its place.
 *
 * HDF5 fails a read it can't finish, such as a gzip chunk that won't inflate, but h5wasm
 * drops what it returns and hands back the buffer it allocated. LUC3D gzips its big tables
 * (`LUCID local patch (luc3d gzip)`), so this is a project's 2D, grouping and 3D. Measured
 * on the code before `LUCID local patch (failed status)`, real file, real Chromium, flipping
 * the bytes of one chunk:
 *   - /frames: File ▸ Open said "SLP loaded (1 session(s), 3 nodes, 8,301 labeled frames)",
 *     and Save wrote a file with 3,726,030 values in /instances where there are 498,000.
 *     A large project said "Reopened — 8,300 grouped frames, 2,115,673 frames".
 *   - /points: 224,100 of 298,800 values wrong, "SLP loaded", and Save wrote them.
 *   - /instances: every 2D point gone, "SLP loaded".
 *   - /session_data/frame_groups: the page crashed. /session_data/instance_groups: it hung.
 *     /session_data/points_3d: "SLP loaded", the 3D wrong.
 *
 * Nothing here is faked: the project comes from the app's own writer, each file is a real
 * file on disk, picked through a real <input>, and the bytes of a chunk are flipped on disk.
 *
 *  0. The files, read healthy, and where their chunks are.
 *  1. Every chunk of a real project, on both ways the reader opens a file.
 *  2. Python's layout, with its string tables gzipped too (`gzip-everything.slp`), where
 *     h5wasm used to free pointers from a buffer that was never filled.
 *  3. A project's string tables: LUC3D can't gzip them, so what can break is the global
 *     heap that holds them.
 *  4. Other damage to a chunk: one byte flipped, zeros on disk (which the h5wasm cold-read
 *     patch can't repair).
 *  5. The app: File ▸ Open (eager and lazy), the per-camera loader, the raw worker fallback.
 *  6. The raw worker, and the analysis .h5 paths.
 *  7. Unlike a failed read of the file's bytes, HDF5 isn't stopped by this: the rest of the
 *     file still reads, and a damaged frame of a video reads as nothing.
 *  8. The other h5wasm instances: the main thread's reader, Load 3D Points, the frame worker.
 *
 * Run: node tests/e2e/slp-damaged-chunk.mjs   (starts its own server, on a free port)
 */
import { chromium } from 'playwright';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { inflateSync } from 'node:zlib';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const FIXTURES = path.join(repoRoot, 'tests', 'fixtures');
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'lucid-damaged-chunk-'));

let fails = 0;
const check = (c, m, extra) => {
    console.log((c ? '  ✓ ' : '  ✗ ') + m + (c || extra === undefined ? '' : '  ' + JSON.stringify(extra).slice(0, 500)));
    if (!c) fails++;
};

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css',
    '.json': 'application/json', '.wasm': 'application/wasm', '.mp4': 'video/mp4', '.png': 'image/png', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml' };
const server = http.createServer((req, res) => {
    const u = new URL(req.url, 'http://x');
    const p = path.join(repoRoot, decodeURIComponent(u.pathname));
    if (!p.startsWith(repoRoot) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(p)] || 'application/octet-stream', 'Cache-Control': 'no-store' });
    fs.createReadStream(p).pipe(res);
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const ORIGIN = `http://127.0.0.1:${server.address().port}`;

/** A project whose every table overflows the writer's staging buffer, so every table is gzipped. */
const FRAMES = 8300;

// ---- finding and damaging chunks ----
/** Where a file's zlib streams are: its compressed chunks. */
function zlibStreams(bytes) {
    const found = [];
    for (let i = 0; i + 2 < bytes.length; i++) {
        if (bytes[i] !== 0x78 || ((bytes[i] << 8) | bytes[i + 1]) % 31 !== 0) continue;
        try {
            const { engine, buffer } = inflateSync(bytes.subarray(i), { info: true });
            const length = engine.bytesWritten;
            if (length > 8) { found.push({ offset: i, length, inflated: buffer }); i += length - 1; }
        } catch { /* not a stream after all */ }
    }
    return found;
}
function flipped(bytes, chunk) {
    const copy = Buffer.from(bytes);
    for (let i = chunk.offset + 2; i < chunk.offset + chunk.length; i++) copy[i] ^= 0xff;
    return copy;
}
function oneByteFlipped(bytes, chunk) {
    const copy = Buffer.from(bytes);
    copy[chunk.offset + (chunk.length >> 1)] ^= 0x40;
    return copy;
}
function zeroed(bytes, chunk) {
    const copy = Buffer.from(bytes);
    copy.fill(0, chunk.offset, chunk.offset + chunk.length);
    return copy;
}

let browser, context, page;
const pageErrors = [];
let hdf5Complaints = [];

async function openPage() {
    page = await context.newPage();
    page.on('pageerror', e => pageErrors.push(String(e).slice(0, 300)));
    page.on('dialog', d => d.accept());
    page.on('console', m => { if (/HDF5-DIAG/.test(m.text())) hdf5Complaints.push(m.text().slice(0, 120)); });
    await page.addInitScript(() => {
        const Real = window.Worker;
        window.__rawWorkers = 0;
        window.Worker = class extends Real {
            constructor(url, opts) {
                if (/slp-import-worker/.test(String(url))) window.__rawWorkers++;
                super(url, opts);
            }
        };
    });
    await page.goto(`${ORIGIN}/index.html`, { waitUntil: 'domcontentloaded', timeout: 180000 });
    await page.waitForFunction(() => window.SleapIO && window.h5wasm && window.__lucid, null, { timeout: 180000 });
    await page.evaluate(() => {
        const input = document.createElement('input');
        input.type = 'file'; input.id = '__picked'; input.multiple = true; input.style.display = 'none';
        document.body.appendChild(input);
        window.__h5wasmUrl = new URL('lib/h5wasm/h5wasm.iife.js', document.baseURI).href;
        window.__b64 = (bytes) => {
            let s = ''; const C = 0x8000;
            for (let o = 0; o < bytes.length; o += C) s += String.fromCharCode.apply(null, bytes.subarray(o, o + C));
            return btoa(s);
        };
        const fromB64 = (b) => Uint8Array.from(atob(b), c => c.charCodeAt(0));
        // A read that a damaged file leaves hanging is a failure, not a stuck test.
        window.__inTime = (promise) => Promise.race([promise, new Promise((_, no) => setTimeout(() => no(new Error('HUNG')), 240000))]);
        // A real project, by the app's own writer: user and predicted instances, identities,
        // tracks, a suggestion, 3D with a smoothing source.
        window.__buildProject = async (FRAMES) => {
            const pd = await import('/pose/pose-data.js');
            const fileio = await import('/import-export/file-io.js');
            const AS = await import('/ui/app-state.js');
            const SIOAll = await import('/lib/sleap-io/index.browser.js');
            const { Skeleton, Camera, Instance, InstanceGroup, FrameGroup, Session } = pd;
            const CAMS = 2, NODES = 2;
            const camNames = Array.from({ length: CAMS }, (_, i) => 'cam' + i);
            const nodeNames = Array.from({ length: NODES }, (_, i) => 'n' + i);
            const M = [[900, 0, 256], [0, 900, 256], [0, 0, 1]];
            const cameras = camNames.map((n, i) => {
                const a = (i / CAMS) * 1.2 - 0.6;
                return new Camera(n, M, [0, 0, 0, 0, 0], [0, a, 0], [-40 * Math.sin(a), 0, 40 * (1 - Math.cos(a))], [512, 512]);
            });
            const skeleton = new Skeleton('skeleton', nodeNames, [[0, 1]]);
            const session = new Session(cameras, skeleton, ['track_0'], 'Damaged');
            session.identities = [new pd.Identity(0, 'id_0')];
            for (let f = 0; f < FRAMES; f++) {
                const fg = new FrameGroup(f);
                session.addFrameGroup(fg);
                const g = new InstanceGroup(f + 1, 0);
                camNames.forEach((cn, ci) => {
                    const inst = new Instance(nodeNames.map((_, k) => [150 + f * 0.25 + ci * 11 + k * 3, 200 + f * 0.125 + ci * 7 + k * 2]), 0, f % 2 ? 'predicted' : 'user', 0.9);
                    fg.addInstance(cn, inst);
                    g.addInstance(cn, inst);
                });
                g.points3d = new Float64Array(NODES * 3).map((_, i) => f + i * 0.5);
                session.instanceGroups.set(f, [g]);
            }
            const views = camNames.map(n => ({ name: n, videoWidth: 512, videoHeight: 512, frameCount: FRAMES }));
            const videoFiles = camNames.map(n => ({ name: n, assignedCamera: n, videoPath: n + '.mp4' }));
            AS.state.sessions = [session]; AS.state.session = session; AS.state.activeSessionIdx = 0;
            AS.state.views = views; AS.state.videoFiles = videoFiles;
            const labels = fileio.buildSlpLabelsAllViews(session, views, videoFiles);
            labels.suggestions = [new SIOAll.SuggestionFrame({ video: labels.videos[0], frameIdx: 3 })];
            for (const s of labels.sessions) for (const fg of s.frameGroups.values()) for (const g of fg.instanceGroups) {
                const p = g.instance3d && g.instance3d.points;
                if (p && p.length) g.instance3d.lucidUnsmoothed = Float64Array.from(p, v => v + 0.25);
            }
            AS.state.sessions = []; AS.state.session = null; AS.state.views = []; AS.state.videoFiles = [];
            return window.__b64(await window.SleapIO.saveSlpToBytes(labels));
        };
        // A table's raw bytes and layout, from a healthy file.
        window.__tables = (b, names) => {
            const h5 = window.h5wasm;
            const p = '/t-' + Math.random().toString(16).slice(2);
            h5.FS.writeFile(p, fromB64(b));
            const f = new h5.File(p, 'r');
            const out = {};
            for (const n of names) {
                const ds = f.get(n);
                const v = ds.value;
                out[n] = { filters: (ds.filters || []).map(x => x.id), rows: ds.shape[0], b64: window.__b64(new Uint8Array(v.buffer, v.byteOffset, v.byteLength)) };
            }
            f.close(); h5.FS.unlink(p);
            return out;
        };
        window.__digest = (labels) => {
            const store = labels._lazyDataStore;
            const lf = store ? null : labels.labeledFrames || [];
            let finite = 0, instances = 0, sum = 0;
            if (lf) {
                for (const f of lf) for (const i of f.instances) {
                    instances++;
                    for (let k = 0; k < i._xy.length; k += 2) if (isFinite(i._xy[k])) { finite++; sum += i._xy[k] + i._xy[k + 1]; }
                }
            } else {
                const t = store.instancesData;
                instances = (t.frame_id || t.instance_id || []).length;
                for (const c of [store.pointsData, store.predPointsData]) {
                    if (c && c.x) for (let k = 0; k < c.x.length; k++) if (isFinite(c.x[k])) { finite++; sum += c.x[k] + c.y[k]; }
                }
            }
            let groups = 0, with3d = 0, members = 0, sum3d = 0;
            for (const s of labels.sessions) for (const fg of s.frameGroups.values()) for (const g of fg.instanceGroups) {
                groups++;
                const p = g.instance3d && g.instance3d.points;
                if (p && p.length) { with3d++; for (let i = 0; i < p.length; i++) { const v = +p[i]; if (isFinite(v)) sum3d += v; } }
                members += g.instanceByCamera ? g.instanceByCamera.size : 0;
            }
            return JSON.stringify({ frames: store ? (store.framesData.frame_id || []).length : lf.length, instances, finite, sum,
                videos: labels.videos.length, tracks: labels.tracks.length, suggestions: labels.suggestions.length,
                sessions: labels.sessions.length, identities: labels.identities.length, groups, with3d, members, sum3d });
        };
        window.__read = async (options) => {
            const f = document.getElementById('__picked').files[0];
            const out = {};
            try {
                const labels = await window.__inTime(window.SleapIO.readSlpStreaming(f, { openVideos: false, rawSessions: true, h5wasmUrl: window.__h5wasmUrl, ...options }));
                out.digest = window.__digest(labels);
            } catch (e) {
                out.threw = String(e && e.message || e).slice(0, 200);
                out.failedRead = !!(e && e.failedRead);
            }
            return out;
        };
    });
}

/**
 * Survives the page going down or never coming back, which some damaged files did to it
 * (garbage row counts hold the main thread): that's a result too.
 */
async function guarded(fn) {
    let timer;
    const stuck = new Promise((_, no) => { timer = setTimeout(() => no(new Error('page closed: stuck')), 300000); });
    const running = fn();
    try {
        return await Promise.race([running, stuck]);
    } catch (e) {
        if (!/crashed|closed/i.test(String(e))) throw e;
        console.log('    (the page ' + (/stuck/.test(String(e)) ? 'is stuck' : 'crashed') + '; opening a new one)');
        running.catch(() => {});
        page.close().catch(() => {});
        await openPage();
        return { crashed: String(e).slice(0, 80) };
    } finally {
        clearTimeout(timer);
    }
}

async function pick(buf, name = 'picked.slp') {
    const dir = fs.mkdtempSync(path.join(tmp, 'p-'));
    const p = path.join(dir, name);
    fs.writeFileSync(p, buf);
    await page.setInputFiles('#__picked', p);
    return p;
}
const readWith = (buf, options = {}, name) => guarded(async () => {
    await pick(buf, name);
    return await page.evaluate((options) => window.__read(options), options);
});
/** What a reader must say about a damaged table. `table` null: any table. */
const saysDamaged = (message, table) => new RegExp(`Could not read (the \\S+ attribute of )?/?${table ? table.replace('/', '\\/') : '\\S+'} from the file\\. Part of it may be damaged\\.`).test(message || '');

try {
    browser = await chromium.launch();
    context = await browser.newContext({ acceptDownloads: false });
    await openPage();

    // ================================================================= 0
    console.log('\n-- 0. the files, read healthy --');
    const project = Buffer.from(await page.evaluate((n) => window.__buildProject(n), FRAMES), 'base64');
    const TABLES = ['frames', 'instances', 'points', 'pred_points', 'session_data/frame_groups', 'session_data/instance_groups',
        'session_data/instance_group_members', 'session_data/points_3d', 'session_data/unsmoothed_points_3d'];
    const tables = await page.evaluate(([b, n]) => window.__tables(b, n), [project.toString('base64'), TABLES]);
    for (const t of TABLES) check(tables[t].filters.includes(1), `${t}: ${tables[t].rows.toLocaleString('en-US')} rows, gzipped`, tables[t].filters);

    // Whose chunk is each zlib stream: its first bytes are somewhere in that table's.
    const raw = Object.fromEntries(TABLES.map(t => [t, Buffer.from(tables[t].b64, 'base64')]));
    const chunks = zlibStreams(project).map(c => ({ ...c, table: TABLES.find(t => raw[t].indexOf(c.inflated.subarray(0, 64)) >= 0) }));
    const byTable = {};
    for (const c of chunks) (byTable[c.table] = byTable[c.table] || []).push(c);
    check(!byTable.undefined && TABLES.every(t => byTable[t] && byTable[t].length >= 1),
        `${chunks.length} chunks, each found in its table`, Object.entries(byTable).map(([k, v]) => [k, v.length]));

    const healthy = await readWith(project);
    const counts = JSON.parse(healthy.digest || '{}');
    check(counts.groups === FRAMES && counts.with3d === FRAMES && counts.frames === FRAMES * 2 && counts.finite === FRAMES * 4,
        'the project opens, with every group, its 3D and its 2D', healthy);
    const healthyLazy = await readWith(project, { lazy: true });
    check(!!healthyLazy.digest, 'and opens lazily', healthyLazy);
    const everything = fs.readFileSync(path.join(FIXTURES, 'slp-damaged-chunk', 'gzip-everything.slp'));
    const everythingChunks = JSON.parse(fs.readFileSync(path.join(FIXTURES, 'slp-damaged-chunk', 'gzip-everything.chunks.json'), 'utf8'));
    const healthyEverything = await readWith(everything);
    check(JSON.parse(healthyEverything.digest || '{}').frames === 60 && JSON.parse(healthyEverything.digest || '{}').tracks > 0,
        'gzip-everything.slp opens: Python\'s layout, its string tables gzipped too', healthyEverything);
    check(hdf5Complaints.length === 0, 'HDF5 has nothing to report on a healthy file', hdf5Complaints.slice(0, 3));

    // ================================================================= 1
    console.log('\n-- 1. every chunk of a real project, in turn --');
    for (const [label, options] of [['the reader', {}], ['the reader, opening lazily', { lazy: true }]]) {
        const wrong = [];
        for (const c of chunks) {
            const r = await readWith(flipped(project, c), options);
            if (!(saysDamaged(r.threw, c.table) && r.failedRead)) wrong.push({ table: c.table, at: c.offset, r });
        }
        check(wrong.length === 0, `${label}: each of the ${chunks.length} chunks is an error naming its table`,
            { wrong: wrong.length, of: chunks.length, first: wrong.slice(0, 2) });
    }

    // ================================================================= 2
    console.log('\n-- 2. Python\'s layout, string tables gzipped too --');
    for (const [table, list] of Object.entries(everythingChunks)) {
        for (const [offset, length] of list) {
            const bad = flipped(everything, { offset, length });
            const r = await readWith(bad);
            check(saysDamaged(r.threw, table) && r.failedRead, `the reader, a chunk of /${table} at ${offset}: an error naming it`, r);
        }
    }

    // ================================================================= 3
    console.log('\n-- 3. a project\'s string tables --');
    // h5wasm can't gzip strings, so they sit in global heaps, and what can make HDF5 fail on
    // them is a heap it can't parse. Its strings' bytes have no checksum (see CLAUDE.md).
    const heaps = [];
    for (let at = project.indexOf('GCOL'); at >= 0; at = project.indexOf('GCOL', at + 4)) heaps.push(at);
    check(heaps.length > 0, `the project has global heaps (${heaps.length})`);
    for (const at of heaps) {
        const bad = Buffer.from(project);
        bad[at] ^= 0xff;
        const r = await readWith(bad);
        check(saysDamaged(r.threw) && r.failedRead, `the global heap at ${at} can't be parsed: an error, not leftovers`, r);
    }

    // ================================================================= 4
    console.log('\n-- 4. other damage to a chunk --');
    for (const t of ['points', 'pred_points', 'session_data/points_3d', 'session_data/instance_groups']) {
        const r = await readWith(oneByteFlipped(project, byTable[t][0]));
        check(saysDamaged(r.threw, t) && r.failedRead, `/${t}, one byte flipped: an error`, r);
        // The cold-read patch reads these zeros again, gets zeros again, and hands them on.
        const z = await readWith(zeroed(project, byTable[t][0]));
        check(saysDamaged(z.threw, t) && z.failedRead, `/${t}, the chunk all zeros on disk: an error`, z);
    }

    // ================================================================= 5
    console.log('\n-- 5. the app --');
    /** Run a real loader on the picked file, dismissing what it asks; report state + status. */
    const openInApp = (fn, buf, name) => guarded(async () => {
        await pick(buf, name);
        await page.evaluate((fn) => {
            window.__rawWorkers = 0;
            window.__loadDone = false;
            (async () => {
                const f = document.getElementById('__picked').files[0];
                if (fn === 'eager') await (await import('/import-export/slp-import.js')).handleLoadSlpFile(f);
                else await (await import('/loading/session-loader.js')).handleLoadProjectSlpLazy(f);
            })().then(() => { window.__loadDone = true; }, e => { window.__loadDone = 'ERR ' + (e && e.stack || e); });
        }, fn);
        let done = 'timeout';
        for (let i = 0; i < 4800; i++) {
            const st = await page.evaluate(() => {
                const b = [...document.querySelectorAll('button')].find(x => x.offsetParent && /^(Skip|Later|Cancel|Not now)$/i.test((x.textContent || '').trim()));
                if (b) { b.click(); return { clicked: true }; }
                return { done: window.__loadDone, prompt: !!document.getElementById('videoSelectOverlay') };
            });
            if (st.prompt) await page.keyboard.press('Escape');
            if (st.done) { done = st.done; break; }
            await page.waitForTimeout(50);
        }
        const state = await page.evaluate(() => {
            const s = window.__lucid.state.session;
            const status = (document.getElementById('statusText') || {}).textContent || '';
            const rawWorkers = window.__rawWorkers;
            if (!s) return { status, session: null, rawWorkers };
            let groups = 0, with3d = 0;
            for (const gs of s.instanceGroups.values()) for (const g of gs) { groups++; if (g.points3d && g.points3d.length) with3d++; }
            return { status, session: s.name, lazy: !!s.lazyLoader, groups, with3d, tracks: s.tracks.length, rawWorkers };
        });
        return { done, ...state };
    });

    const fine = await openInApp('eager', project, 'project.slp');
    check(fine.groups === FRAMES && fine.with3d === FRAMES && fine.session === 'Damaged', 'File ▸ Open: the healthy project opens', fine);
    for (const t of TABLES) {
        const r = await openInApp('eager', flipped(project, byTable[t][0]), 'another.slp');
        check(r.done === true && /^SLP parse error/.test(r.status) && saysDamaged(r.status, t) && r.rawWorkers === 0 && r.groups === FRAMES && r.with3d === FRAMES,
            `File ▸ Open, a damaged chunk of /${t}: says so, no second reader, the open project untouched`,
            { status: r.status, rawWorkers: r.rawWorkers, groups: r.groups, with3d: r.with3d, crashed: r.crashed });
        if (r.groups !== FRAMES) await openInApp('eager', project, 'project.slp'); // for the next one
    }
    for (const t of TABLES) {
        const r = await openInApp('lazy', flipped(project, byTable[t][0]), 'lazy.slp');
        check(r.done === true && /^Lazy reopen error/.test(r.status) && saysDamaged(r.status, t) && r.session === null,
            `File ▸ Open (large project), a damaged chunk of /${t}: says so, nothing half-read opens`,
            { status: r.status, session: r.session, groups: r.groups, crashed: r.crashed });
    }
    const lazyAgain = await openInApp('lazy', project, 'lazy.slp');
    check(lazyAgain.lazy && lazyAgain.groups === FRAMES && lazyAgain.with3d === FRAMES, 'and the healthy file still opens', lazyAgain);

    // The per-camera loader's lazy reader.
    const camOpen = await guarded(async () => {
        await pick(flipped(project, byTable.pred_points[0]), 'cam.slp');
        return await page.evaluate(async () => {
            const { SioLazyLoader } = await import('/loading/sio-lazy-loader.js');
            try { await window.__inTime(new SioLazyLoader().open('cam', document.getElementById('__picked').files[0])); return 'opened'; }
            catch (e) { return e.failedRead ? e.message : 'other: ' + e.message; }
        });
    });
    check(saysDamaged(camOpen, 'pred_points'), 'per-camera loader: a damaged chunk of a camera\'s /pred_points stops the load', camOpen);

    // A plain file has no /session_data, so the raw worker could stand in, and would read
    // the same chunk: the first error stands.
    const fallback = await guarded(async () => {
        await pick(flipped(everything, { offset: everythingChunks.points[0][0], length: everythingChunks.points[0][1] }));
        return await page.evaluate(async () => {
            const fileio = await import('/import-export/file-io.js');
            window.__rawWorkers = 0;
            const out = {};
            try { const d = await window.__inTime(fileio.parseSlpWithRawFallback(document.getElementById('__picked').files[0], () => {})); out.opened = d.frames.length; }
            catch (e) { out.threw = String(e.message).slice(0, 160); }
            out.rawWorkers = window.__rawWorkers;
            return out;
        });
    });
    check(saysDamaged(fallback.threw, 'points') && fallback.rawWorkers === 0, 'parseSlpWithRawFallback: a plain file\'s damaged chunk: no second reader', fallback);

    // ================================================================= 6
    console.log('\n-- 6. the raw worker --');
    const rawParse = (buf, name, opts) => guarded(async () => {
        await pick(buf, name);
        return await page.evaluate(async (opts) => {
            const fileio = await import('/import-export/file-io.js');
            const out = {};
            try {
                const d = await window.__inTime(fileio.parseSlpH5(document.getElementById('__picked').files[0], null, opts));
                out.parsed = { frames: d.frames.length, tracks: d.tracks.length, videos: d.videos.length, columnar: d.columnar ? d.columnar.nFrames : undefined };
            } catch (e) { out.threw = String(e.message).slice(0, 300); }
            return out;
        }, opts);
    });
    const rawHealthy = await rawParse(everything);
    check(rawHealthy.parsed && rawHealthy.parsed.frames === 60 && rawHealthy.parsed.tracks > 0, 'gzip-everything.slp, healthy', rawHealthy);
    const asItStands = (m) => !/\n\s*at /.test(m || '');
    for (const [table, list] of Object.entries(everythingChunks)) {
        const r = await rawParse(flipped(everything, { offset: list[0][0], length: list[0][1] }));
        check(saysDamaged(r.threw, table) && asItStands(r.threw) && !r.parsed,
            `a damaged chunk of /${table} (try/caught in there for "absent"): ends the parse in that error, with no stack in the status bar`, r);
    }
    const col = await rawParse(flipped(everything, { offset: everythingChunks.pred_points[0][0], length: everythingChunks.pred_points[0][1] }), undefined, { columnar: true });
    check(saysDamaged(col.threw, 'pred_points'), 'the per-camera parse (columnar) too', col);
    for (const t of ['frames', 'instances', 'points', 'pred_points']) {
        const r = await rawParse(flipped(project, byTable[t][0]));
        check(saysDamaged(r.threw, t) && !r.parsed, `a project's /${t}: the same`, r);
    }

    // Analysis .h5, both paths: its optional tables are try/caught too.
    const analysis = Buffer.from(await page.evaluate(() => {
        const h5 = window.h5wasm;
        const p = '/analysis-' + Math.random().toString(16).slice(2) + '.h5';
        const f = new h5.File(p, 'w');
        const gz = (name, data, shape, dtype, chunks) => f.create_dataset({ name, data, shape, dtype, chunks, compression: 'gzip', compression_opts: 1 });
        f.create_dataset({ name: 'node_names', data: ['a', 'b'] });
        f.create_dataset({ name: 'track_names', data: ['t0'] });
        f.create_dataset({ name: 'edge_inds', data: new Int32Array([0, 1]), shape: [1, 2], dtype: '<i' });
        // [n_tracks, 2, n_nodes, n_frames]
        gz('tracks', new Float64Array(2000).map((_, i) => i * 1.5), [1, 2, 2, 500], '<d', [1, 2, 2, 500]);
        gz('track_occupancy', new Int32Array(500).map((_, i) => (i % 7 ? i + 1 : 1)), [500, 1], '<i', [500, 1]);
        f.close();
        const out = h5.FS.readFile(p); h5.FS.unlink(p);
        return window.__b64(out);
    }), 'base64');
    const analysisChunks = zlibStreams(analysis);
    check(analysisChunks.length === 2, 'analysis .h5: its tables are gzipped', analysisChunks.length);
    const aHealthy = await rawParse(analysis, 'cam.analysis.h5');
    check(aHealthy.parsed && aHealthy.parsed.frames > 0, 'analysis .h5: healthy', aHealthy);
    const rawLazy = (buf) => guarded(async () => {
        await pick(buf, 'cam.analysis.h5');
        return await page.evaluate(async () => {
            const worker = new Worker(new URL('loading/slp-import-worker.js', document.baseURI), { type: 'module' });
            const out = await window.__inTime(new Promise((res) => {
                worker.onmessage = (e) => {
                    if (e.data.type === 'metadata') res({ nFrames: e.data.data.nFrames, occupancy: !!e.data.data.trackOccupancy });
                    else if (e.data.type === 'error') res({ threw: String(e.data.message).slice(0, 300) });
                };
                worker.onerror = (e) => res({ threw: 'worker died: ' + e.message });
                worker.postMessage({ type: 'open', file: document.getElementById('__picked').files[0] });
            })).catch(e => ({ threw: e.message }));
            worker.terminate();
            return out;
        });
    });
    const laHealthy = await rawLazy(analysis);
    check(laHealthy.nFrames === 500 && laHealthy.occupancy, 'analysis .h5, lazy: healthy', laHealthy);
    for (const c of analysisChunks) {
        const a = await rawParse(flipped(analysis, c), 'cam.analysis.h5');
        check(saysDamaged(a.threw) && asItStands(a.threw), `analysis .h5, a damaged chunk at ${c.offset}: an error`, a);
        const l = await rawLazy(flipped(analysis, c));
        check(saysDamaged(l.threw) && asItStands(l.threw), `analysis .h5, lazy, the same chunk: an error`, l);
    }

    // ================================================================= 7
    console.log('\n-- 7. HDF5 isn\'t stopped by this: the rest of the file reads --');
    const rest = await guarded(async () => {
        await pick(flipped(project, byTable.points[0]));
        return await page.evaluate(async () => {
            const { openH5Worker } = await import('/lib/sleap-io/index.browser.js');
            const h5 = await openH5Worker(document.getElementById('__picked').files[0], { h5wasmUrl: window.__h5wasmUrl });
            const attempt = async (fn) => { try { await fn(); return 'read'; } catch (e) { return (e.failedRead ? 'failed read: ' : 'other: ') + String(e.message).slice(0, 70); } };
            const out = {};
            out.before = await attempt(() => h5.getDatasetValue('frames'));
            out.damaged = await attempt(() => h5.getDatasetValue('points'));
            out.after = await attempt(() => h5.getDatasetValue('pred_points'));
            out.string = await attempt(() => h5.getDatasetValue('videos_json'));
            out.again = await attempt(() => h5.getDatasetValue('points'));
            out.close = await attempt(() => h5.close());
            return out;
        });
    });
    check(rest.before === 'read' && rest.after === 'read' && rest.string === 'read' && rest.close === 'read', 'the other tables read, and the file closes', rest);
    check(saysDamaged(rest.damaged, 'points') && rest.again === rest.damaged, 'the damaged one is the same error each time', rest);

    const pkg = fs.readFileSync(path.join(FIXTURES, 'slp-failed-read', 'cropped_format_2_3.pkg.slp'));
    const pkgChunks = zlibStreams(pkg);
    check(pkgChunks.length === 2, 'a .pkg.slp with a frame in two gzipped chunks', pkgChunks.length);
    const frameOf = (buf) => guarded(async () => {
        await pick(buf);
        return await page.evaluate(async () => {
            const labels = await window.SleapIO.readSlpStreaming(document.getElementById('__picked').files[0], { openVideos: false, lazyVideoMetadata: true, h5wasmUrl: window.__h5wasmUrl });
            const backend = labels.videos[0].backend;
            await backend.ensureLoaded();
            const frame = await backend.getFrameBuffer(backend.frameNumbers[0]);
            return { numbers: backend.frameNumbers.length, frame: frame ? frame.length : frame };
        });
    });
    const frameFine = await frameOf(pkg);
    check(frameFine.numbers > 0 && frameFine.frame > 0, 'its frame reads', frameFine);
    const frameBad = await frameOf(flipped(pkg, pkgChunks[0]));
    check(frameBad.numbers > 0 && frameBad.frame === null, 'a damaged chunk of it reads as no frame, never as what was in memory', frameBad);

    // ================================================================= 8
    console.log('\n-- 8. the other h5wasm instances --');
    // The bundle's main-thread reader (not the app's way in, but it is what `loadSlp` falls to:
    // `stream: 'download'` goes straight there).
    const mainRead = (buf) => guarded(() => page.evaluate(async (b) => {
        const SIO = await import('/lib/sleap-io/index.browser.js');
        const bytes = Uint8Array.from(atob(b), c => c.charCodeAt(0));
        const out = {};
        try { out.digest = window.__digest(await window.__inTime(SIO.loadSlp(bytes, { openVideos: false, h5: { stream: 'download' } }))); }
        catch (e) { out.threw = String(e && e.message || e).slice(0, 200); out.failedRead = !!(e && e.failedRead); }
        return out;
    }, buf.toString('base64')));
    const mainHealthy = await mainRead(everything);
    check(!!mainHealthy.digest, 'main thread: gzip-everything.slp, healthy', mainHealthy);
    for (const [table, list] of Object.entries(everythingChunks)) {
        const r = await mainRead(flipped(everything, { offset: list[0][0], length: list[0][1] }));
        check(saysDamaged(r.threw, table) && r.failedRead, `main thread, a damaged chunk of /${table}: an error`, r);
    }
    const mainAfter = await mainRead(everything);
    check(mainAfter.digest === mainHealthy.digest, 'and the module survives them all: the healthy file reads the same', mainAfter);
    // A hyperslab read of strings frees through reclaim_vlen_memory_count (h5wasm 0.10.3), and
    // a table of strings, unlike the rest, has h5wasm freeing what it finds in the buffer.
    const sliceOf = (buf) => guarded(() => page.evaluate(async (b) => {
        const H5 = await import('h5wasm');
        const p = '/slice-' + Math.random().toString(16).slice(2) + '.slp';
        H5.FS.writeFile(p, Uint8Array.from(atob(b), c => c.charCodeAt(0)));
        const f = new H5.File(p, 'r');
        const out = {};
        try { out.row = String(f.get('tracks_json').slice([[0, 1]])[0]).slice(0, 40); }
        catch (e) { out.threw = String(e && e.message || e).slice(0, 200); }
        f.close();
        H5.FS.unlink(p);
        return out;
    }, buf.toString('base64')));
    const sliceFine = await sliceOf(everything);
    check(!!sliceFine.row, 'main thread: a hyperslab of a table of strings, healthy', sliceFine);
    const sliceBad = await sliceOf(flipped(everything, { offset: everythingChunks.tracks_json[0][0], length: everythingChunks.tracks_json[0][1] }));
    check(saysDamaged(sliceBad.threw, 'tracks_json'), 'a damaged chunk of it: an error, and nothing freed from the unfilled buffer', sliceBad);
    const sliceAfter = await sliceOf(everything);
    check(sliceAfter.row === sliceFine.row, 'and the module survives: the healthy file reads the same', sliceAfter);

    // Load 3D Points: the app's own read on the main thread.
    const pts = Buffer.from(await page.evaluate(() => {
        const h5 = window.h5wasm;
        const p = '/pts-' + Math.random().toString(16).slice(2) + '.h5';
        const f = new h5.File(p, 'w');
        const N = 400;
        f.create_dataset({ name: 'points_3d', data: new Float64Array(N * 6).map((_, i) => i * 0.5), shape: [N, 1, 2, 3], dtype: '<d', chunks: [N, 1, 2, 3], compression: 'gzip', compression_opts: 1 });
        f.create_dataset({ name: 'frame_indices', data: new Float64Array(N).map((_, i) => i) });
        f.create_dataset({ name: 'track_names', data: ['t0'] });
        f.create_dataset({ name: 'node_names', data: ['a', 'b'] });
        f.close();
        const out = h5.FS.readFile(p); h5.FS.unlink(p);
        return window.__b64(out);
    }), 'base64');
    const parsePts = (buf) => guarded(() => page.evaluate(async (b) => {
        const fileio = await import('/import-export/file-io.js');
        const bytes = Uint8Array.from(atob(b), c => c.charCodeAt(0));
        try { const d = await fileio.parsePoints3dH5(bytes.buffer); return { frames: d.points3d.size, sample: d.points3d.get(1).get(0)[0] }; }
        catch (e) { return { threw: String(e.message).slice(0, 200) }; }
    }, buf.toString('base64')));
    const ptsFine = await parsePts(pts);
    check(ptsFine.frames === 400 && ptsFine.sample[0] === 3, 'Load 3D Points: healthy', ptsFine);
    const ptsBad = await parsePts(flipped(pts, zlibStreams(pts)[0]));
    check(saysDamaged(ptsBad.threw, 'points_3d'), 'Load 3D Points: a damaged chunk is an error, not 3D points from what was in memory', ptsBad);

    // The worker that serves a .pkg.slp's frames.
    const frameWorker = (buf) => guarded(async () => {
        await pick(buf, 'mice.pkg.slp');
        return await page.evaluate(async () => {
            const worker = new Worker(new URL('loading/frame-worker.js', document.baseURI), { type: 'module' });
            const next = (types) => window.__inTime(new Promise((res) => {
                const on = (e) => { if (types.includes(e.data.type)) { worker.removeEventListener('message', on); res(e.data); } };
                worker.addEventListener('message', on);
            }));
            const out = {};
            try {
                const ready = next(['ready', 'error']);
                out.ready = (await ready).type;
                worker.postMessage({ type: 'loadFile', file: document.getElementById('__picked').files[0] });
                const loaded = await next(['loaded', 'error']);
                out.loaded = loaded.type === 'loaded' ? loaded.videos.length : loaded.error;
                worker.postMessage({ type: 'getFrame', videoKey: 'video0', embeddedIdx: 0 });
                const frame = await next(['frame', 'error']);
                out.frame = frame.error || (frame.type === 'frame' ? 'a frame' : 'nothing');
            } catch (e) { out.threw = String(e.message); }
            worker.terminate();
            return out;
        });
    });
    const fwFine = await frameWorker(pkg);
    check(fwFine.loaded === 1 && fwFine.frame === 'a frame', 'frame worker: healthy', fwFine);
    const fwBad = await frameWorker(flipped(pkg, pkgChunks[0]));
    check(saysDamaged(fwBad.frame, 'video0/video'), 'frame worker: a damaged chunk is an error', fwBad);
    // The app gets that answer from its decoder: the labels open, and nothing waits for a frame.
    const pkgOpen = await openInApp('eager', flipped(pkg, pkgChunks[0]), 'mice.pkg.slp');
    check(pkgOpen.done === true && /^SLP loaded/.test(pkgOpen.status), 'File ▸ Open of that file: the labels open, and nothing waits for the frame', pkgOpen);

    check(pageErrors.length === 0, 'no page errors', pageErrors.slice(0, 3));
} catch (e) {
    console.error(e);
    fails++;
} finally {
    if (browser) await browser.close();
    server.close();
    fs.rmSync(tmp, { recursive: true, force: true });
}
console.log(fails ? `\nFAIL (${fails})` : '\nPASS');
process.exit(fails ? 1 : 0);
