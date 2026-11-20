/**
 * slp-failed-write.mjs — a save whose write fails is an error, never a success with the
 * file cut short or wrong.
 *
 * h5wasm drops what HDF5 says about a write, a close or a flush. And what HDF5 returns
 * doesn't cover it all: what `create_dataset` and `create_attribute` return is their last
 * cleanup's status, and `set_dataset_data`'s leaves out the H5Dclose that writes the
 * chunks it has cached. Measured on the code before `LUCID local patch (failed status)`:
 *   - h5wasm's heap (2 GiB at most, on the main thread for a save) nearly full, nothing
 *     injected: 67 of 135 saves resolved with a file holding wrong values, e.g. 71,997
 *     values of /instances, 93,894 bytes where a healthy save writes 173,752.
 *   - One device write failing (ENOSPC): 36 of 120 resolved with wrong values. Every write
 *     from the file's close on failing: resolved, sink closed, a file with no tables in it.
 *   - File ▸ Save with a real file handle: "Saved", over the user's file.
 *
 *  0. The healthy saves, and that HDF5 has nothing to report on them.
 *  1. h5wasm's heap fills up (real, nothing injected), on every save route.
 *  2. Every write from the moment the file closes fails, both ways a device write can
 *     fail: an errno, as a full disk gives, and a RangeError, which is what MEMFS itself
 *     throws when it can't grow a file (this one was always reported).
 *  3. Every write of a save fails in turn.
 *  4. File ▸ Save, with a real FileSystemFileHandle (OPFS), a project opened eagerly and
 *     lazily, a real storage quota, and a multi-session save.
 *  5. The app's own writer on the main thread: the 3D points export.
 *
 * Run: node tests/e2e/slp-failed-write.mjs   (starts its own server, on a free port)
 */
import { chromium } from 'playwright';
import fs from 'node:fs';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'lucid-failed-write-'));

let fails = 0;
const check = (c, m, extra) => {
    console.log((c ? '  ✓ ' : '  ✗ ') + m + (c || extra === undefined ? '' : '  ' + JSON.stringify(extra).slice(0, 600)));
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

const FAILED_WRITE = /^Could not finish writing the file: HDF5 reported a failed write/;
const ROUTES = ['saveSlpToBytes', 'saveSlpToBytes, a lazy Labels', 'openSlpWriter, close', 'openSlpWriter, writeToSink', 'saveAnalysisH5ToBytes'];

let browser;
try {
    browser = await chromium.launch();
    const context = await browser.newContext({ acceptDownloads: false });
    const page = await context.newPage();
    const pageErrors = [];
    const hdf5Complaints = [];
    page.on('pageerror', e => pageErrors.push(String(e).slice(0, 300)));
    page.on('dialog', d => d.accept());
    page.on('console', m => { if (/HDF5-DIAG/.test(m.text())) hdf5Complaints.push(m.text().slice(0, 120)); });
    const setUpPage = async () => {
    await page.goto(`${ORIGIN}/index.html`, { waitUntil: 'domcontentloaded', timeout: 180000 });
    await page.waitForFunction(() => window.SleapIO && window.h5wasm && window.__lucid, null, { timeout: 180000 });
    await page.evaluate(async () => {
        const input = document.createElement('input');
        input.type = 'file'; input.id = '__picked'; input.multiple = true; input.style.display = 'none';
        document.body.appendChild(input);
        const SIO = window.SleapIO;
        const SIOAll = await import('/lib/sleap-io/index.browser.js');
        const h5wasmUrl = new URL('lib/h5wasm/h5wasm.iife.js', document.baseURI).href;
        window.__h5wasmUrl = h5wasmUrl;
        // The h5wasm instance the bundle saves with (the importmap's), not window.h5wasm.
        const H5 = await import('h5wasm');
        await H5.ready;
        window.H5 = H5;
        // Enough rows that the 2D tables are chunked and gzipped.
        const N = 9000;
        const makeLabels = () => {
            const skeleton = new SIO.Skeleton({ nodes: [new SIO.Node('a'), new SIO.Node('b')] });
            const video = new SIO.Video({ filename: 'x.mp4', backendMetadata: { filename: 'x.mp4', type: 'MediaVideo', shape: [N, 4, 4, 1] } });
            const lfs = [];
            for (let i = 0; i < N; i++) {
                lfs.push(new SIO.LabeledFrame({ video, frameIdx: i, instances: [new SIO.PredictedInstance({ points: [[i + 0.5, i + 1.5], [i + 2.5, i + 3.5]], skeleton, score: 0.5 })] }));
            }
            return new SIO.Labels({ labeledFrames: lfs, videos: [video], skeletons: [skeleton] });
        };
        // What a save wrote, read by the OTHER h5wasm instance: every table, and whether it opens.
        const TABLES = ['frames', 'instances', 'points', 'pred_points', 'videos_json', 'tracks_json', 'sessions_json', 'identities_json', 'suggestions_json',
            'session_data/frame_groups', 'session_data/instance_groups', 'session_data/instance_group_members', 'session_data/points_3d',
            'tracks', 'track_occupancy', 'node_names', 'track_names', 'points_3d', 'frame_indices'];
        const tablesIn = (u8) => {
            const h5 = window.h5wasm;
            const p = '/r-' + Math.random().toString(16).slice(2);
            h5.FS.writeFile(p, u8);
            const quiet = console.error; console.error = () => {};
            try {
                const f = new h5.File(p, 'r');
                const out = {};
                for (const n of TABLES) { const d = f.get(n); if (d) out[n] = d.value; }
                f.close();
                return out;
            } catch (e) { return { cantOpen: String(e && e.message || e).slice(0, 120) }; }
            finally { console.error = quiet; try { h5.FS.unlink(p); } catch (e) { /* gone */ } }
        };
        window.__tablesIn = tablesIn;
        const differs = (a, b) => {
            if (b.cantOpen) return { cantOpen: b.cantOpen };
            const out = {};
            for (const k of new Set([...Object.keys(a), ...Object.keys(b)])) {
                const x = a[k], y = b[k];
                if (x == null || y == null) { if (x !== y) out[k] = 'missing'; continue; }
                if (ArrayBuffer.isView(x)) {
                    let d = 0; const n = Math.min(x.length, y.length);
                    for (let i = 0; i < n; i++) if (!(x[i] === y[i] || (x[i] !== x[i] && y[i] !== y[i]))) d++;
                    if (d || x.length !== y.length) out[k] = { values: x.length, other: y.length, different: d };
                } else if (JSON.stringify(x) !== JSON.stringify(y)) out[k] = 'text differs';
            }
            return out;
        };
        window.__differs = differs;
        const healthyLabels = makeLabels();
        const healthyBytes = await SIO.saveSlpToBytes(healthyLabels);
        const lazyLabels = await SIO.readSlpStreaming(new File([healthyBytes], 'healthy.slp'), { lazy: true, openVideos: false, h5wasmUrl });
        const writer = async (labels) => {
            const w = await SIO.openSlpWriter({ videos: labels.videos, skeletons: labels.skeletons, tracks: [] });
            try { w.appendFrames(labels.labeledFrames); } catch (e) { w.dispose(); throw e; }
            return w;
        };
        const runs = {
            'saveSlpToBytes': () => SIO.saveSlpToBytes(makeLabels()),
            'saveSlpToBytes, a lazy Labels': () => SIO.saveSlpToBytes(lazyLabels),
            'openSlpWriter, close': async () => {
                const w = await writer(makeLabels());
                try { return w.close(); } catch (e) { w.dispose(); throw e; }
            },
            'openSlpWriter, writeToSink': async () => {
                const w = await writer(makeLabels());
                const parts = [];
                let closed = false;
                try {
                    await w.writeToSink({ write: async (c) => { parts.push(c); }, close: async () => { closed = true; } }, {});
                } catch (e) {
                    w.dispose();
                    if (parts.length || closed) throw new Error('the sink got something: ' + e.message);
                    throw e;
                }
                if (!closed) throw new Error('resolved without closing the sink');
                const out = new Uint8Array(parts.reduce((a, c) => a + c.length, 0));
                let o = 0;
                for (const c of parts) { out.set(c, o); o += c.length; }
                return out;
            },
            'saveAnalysisH5ToBytes': () => SIOAll.saveAnalysisH5ToBytes(makeLabels()),
        };
        window.__refs = {};
        window.__recordRefs = async () => {
            const sizes = {};
            for (const [name, run] of Object.entries(runs)) {
                const bytes = await run();
                window.__refs[name] = tablesIn(bytes);
                sizes[name] = { size: bytes.length, tables: Object.keys(window.__refs[name]).length };
            }
            return sizes;
        };
        /** Leftovers of a save in h5wasm's file system. */
        const leftovers = () => {
            const FS = H5.FS;
            return { files: FS.readdir('/tmp').filter(n => n !== '.' && n !== '..'), open: FS.streams.filter(s => s && s.path && s.path.startsWith('/tmp/')).length };
        };
        window.__leftovers = leftovers;
        /** One save: what came of it, against that route's healthy save. */
        window.__trial = async (route, arm) => {
            window.__arm(arm || {});
            const out = {};
            const openBefore = leftovers().open;
            const quiet = console.error; console.error = () => {};
            try {
                const bytes = await runs[route]();
                out.size = bytes.length;
                out.diff = differs(window.__refs[route], tablesIn(bytes));
            } catch (e) { out.threw = String(e && e.message || e).slice(0, 160); }
            console.error = quiet;
            out.fired = window.__fail.fired;
            out.writes = window.__fail.count;
            window.__arm({});
            out.left = leftovers();
            out.leftOpen = out.left.open - openBefore;
            return out;
        };
        // ---- device failures: below HDF5, on the file a save is writing ----
        const FS = H5.FS;
        const write = FS.write;
        window.__fail = { nth: 0, count: 0, kind: 'errno', fired: 0, atClose: false, closing: false };
        FS.write = function (stream) {
            const c = window.__fail;
            if (stream && stream.path && stream.path.startsWith('/tmp/')) {
                c.count++;
                if (c.count === c.nth || (c.atClose && c.closing)) {
                    c.fired++;
                    if (c.kind === 'errno') throw new FS.ErrnoError(28); // ENOSPC
                    throw new RangeError('Array buffer allocation failed'); // what MEMFS throws
                }
            }
            return write.apply(this, arguments);
        };
        const proto = H5.File.prototype, close = proto.close;
        proto.close = function () {
            if (this.mode !== 'r') window.__fail.closing = true;
            try { return close.call(this); } finally { window.__fail.closing = false; }
        };
        window.__arm = (o) => { Object.assign(window.__fail, { nth: 0, count: 0, kind: 'errno', fired: 0, atClose: false, closing: false }, o); };
        // ---- h5wasm's heap: fill it, leaving `leave` bytes in one place, then save ----
        window.__oom = async (route, leave) => {
            const M = H5.Module;
            const reserve = M._malloc(leave);
            const blocks = [];
            for (let size = 1 << 28; size >= 1024; size >>= 1) {
                for (;;) { const p = M._malloc(size); if (!p) break; blocks.push(p); }
            }
            M._free(reserve);
            const out = { leave, heap: M.HEAPU8.length };
            const quiet = console.error; console.error = () => {};
            let bytes = null;
            try { bytes = await runs[route](); } catch (e) { out.threw = String(e && e.message || e).slice(0, 160); }
            console.error = quiet;
            for (const p of blocks) M._free(p);
            if (bytes) { out.size = bytes.length; out.diff = differs(window.__refs[route], tablesIn(bytes)); }
            out.left = leftovers();
            return out;
        };
    });
    };
    await setUpPage();

    // ================================================================= 0
    console.log('\n-- 0. the healthy saves --');
    const refs = await page.evaluate(() => window.__recordRefs());
    for (const route of ROUTES) check(refs[route] && refs[route].tables >= 3, `${route}: writes a file that opens`, refs[route]);
    check(hdf5Complaints.length === 0, 'HDF5 has nothing to report on a healthy save', hdf5Complaints.slice(0, 3));
    for (const route of ROUTES) {
        const r = await page.evaluate((route) => window.__trial(route), route);
        check(!r.threw && Object.keys(r.diff).length === 0, `${route}: the same values again`, r);
    }
    check(JSON.stringify(await page.evaluate(() => window.__leftovers())) === JSON.stringify({ files: [], open: 0 }), 'no temporary file left behind');

    // ================================================================= 1
    console.log('\n-- 1. h5wasm\'s heap fills up, nothing injected --');
    // The sizes where it shows: below them the data buffer can't be allocated (thrown in
    // JS, always reported), above them there is room.
    const LEAVES = [];
    for (let x = 1024 * 1024; x < 3.6 * 1024 * 1024; x = Math.ceil(x * 1.05)) LEAVES.push(x);
    for (const route of ROUTES) {
        const out = { threw: 0, same: 0, wrong: [], left: [] };
        for (const leave of LEAVES) {
            const r = await page.evaluate(([route, leave]) => window.__oom(route, leave), [route, leave]);
            if (r.threw) out.threw++;
            else if (Object.keys(r.diff).length) out.wrong.push({ leave, size: r.size, diff: r.diff });
            else out.same++;
            if (r.left.files.length) out.left.push({ leave, files: r.left.files });
        }
        check(out.wrong.length === 0, `${route}: ${LEAVES.length} saves on a nearly full heap: ${out.threw} errors, ${out.same} whole, none wrong`,
            { threw: out.threw, same: out.same, wrong: out.wrong.length, first: out.wrong.slice(0, 2) });
        check(out.left.length === 0, `${route}: and no temporary file left behind`, out.left.slice(0, 2));
        const after = await page.evaluate((route) => window.__trial(route), route);
        check(!after.threw && Object.keys(after.diff).length === 0, `${route}: the next save is whole`, after);
        if (route !== 'saveAnalysisH5ToBytes') check(out.threw > 0, `${route}: precondition: the heap really did run out`, out.threw);
    }
    // A string that doesn't fit in what's left: embind's glue doesn't check its malloc, so
    // unpatched it wrote the string from address 0, over HDF5's static data, and later calls
    // of that module could fail in any way (`LUCID local patch (full heap)`).
    for (const which of ['H5', 'h5wasm']) {
        const r = await page.evaluate((which) => {
            const h5 = window[which], M = h5.Module;
            const p = '/low-' + Math.random().toString(16).slice(2) + '.h5';
            const f = new h5.File(p, 'w');
            const blocks = [];
            for (let size = 1 << 28; size >= 16; size >>= 1) {
                for (;;) { const a = M._malloc(size); if (!a) break; blocks.push(a); }
            }
            const before = M.HEAPU8.slice(0, 1 << 16);
            const out = {};
            try { M.create_group(f.file_id, '/' + 'J'.repeat(5000), false); } catch (e) { out.threw = String(e && e.message || e).slice(0, 120); }
            const after = M.HEAPU8.subarray(0, 1 << 16);
            out.changed = 0;
            for (let i = 0; i < before.length; i++) if (before[i] !== after[i]) out.changed++;
            after.set(before); // the rest of the page shouldn't pay for a build that wrote there
            for (const a of blocks) M._free(a);
            f.close();
            h5.FS.unlink(p);
            return out;
        }, which);
        check(/^Out of memory in h5wasm/.test(r.threw || '') && r.changed === 0,
            `${which === 'H5' ? 'hdf5_util.js' : 'h5wasm.iife.js'}: a string that doesn't fit in the heap is an error, and low memory stays as it was`, r);
    }
    // Whatever that did to the module, the rest of this test starts from a fresh one.
    await setUpPage();
    await page.evaluate(() => window.__recordRefs());

    // ================================================================= 2
    console.log('\n-- 2. every write from the moment the file closes fails --');
    for (const route of ROUTES) {
        for (const kind of ['errno', 'range']) {
            const r = await page.evaluate(([route, kind]) => window.__trial(route, { atClose: true, kind }), [route, kind]);
            const said = kind === 'errno' ? FAILED_WRITE.test(r.threw || '') : /Array buffer allocation failed/.test(r.threw || '');
            check(r.fired > 0 && said, `${route}, ${kind}: an error, and nothing to a sink`, r);
            check(r.left.files.length === 0, `${route}, ${kind}: no temporary file left behind`, r.left);
            // A RangeError stops HDF5 mid-call, and its descriptor stays open until a reload.
            if (kind === 'errno') check(r.leftOpen === 0, `${route}, ${kind}: and none left open`, r);
            const after = await page.evaluate((route) => window.__trial(route), route);
            check(!after.threw && Object.keys(after.diff).length === 0, `${route}, ${kind}: the next save is whole`, after);
        }
    }

    // ================================================================= 3
    console.log('\n-- 3. every write of a save fails in turn --');
    for (const route of ROUTES) {
        for (const kind of ['errno', 'range']) {
            const healthy = await page.evaluate((route) => window.__trial(route), route);
            const wrong = [];
            let threw = 0, whole = 0;
            for (let nth = 1; nth <= healthy.writes; nth++) {
                const r = await page.evaluate(([route, nth, kind]) => window.__trial(route, { nth, kind }), [route, nth, kind]);
                if (r.threw) threw++;
                else if (Object.keys(r.diff).length) wrong.push({ nth, size: r.size, diff: r.diff });
                else whole++;
            }
            check(healthy.writes > 3 && wrong.length === 0, `${route}, ${kind}: ${healthy.writes} writes failed one at a time: ${threw} errors, ${whole} whole, none wrong`,
                { wrong: wrong.length, first: wrong.slice(0, 2) });
        }
    }

    // ================================================================= 4
    console.log('\n-- 4. File ▸ Save --');
    await page.evaluate(async () => {
        const pd = await import('/pose/pose-data.js');
        const fileio = await import('/import-export/file-io.js');
        const AS = await import('/ui/app-state.js');
        const { Skeleton, Camera, Instance, InstanceGroup, FrameGroup, Session } = pd;
        window.__buildProject = async () => {
            const FRAMES = 300, camNames = ['cam0', 'cam1'], nodeNames = ['n0', 'n1'];
            const M = [[900, 0, 256], [0, 900, 256], [0, 0, 1]];
            const cameras = camNames.map((n, i) => new Camera(n, M, [0, 0, 0, 0, 0], [0, i * 0.3, 0], [i * 10, 0, 0], [512, 512]));
            const session = new Session(cameras, new Skeleton('skeleton', nodeNames, [[0, 1]]), ['track_0'], 'FailedWrite');
            for (let f = 0; f < FRAMES; f++) {
                const fg = new FrameGroup(f);
                session.addFrameGroup(fg);
                const g = new InstanceGroup(f + 1, 0);
                camNames.forEach((cn, ci) => {
                    const inst = new Instance(nodeNames.map((_, k) => [100 + f + ci + k, 200 + f + ci * 2 + k]), 0, f % 2 ? 'predicted' : 'user', 0.9);
                    fg.addInstance(cn, inst);
                    g.addInstance(cn, inst);
                });
                g.points3d = new Float64Array(6).map((_, i) => f + i * 0.5);
                session.instanceGroups.set(f, [g]);
            }
            const views = camNames.map(n => ({ name: n, videoWidth: 512, videoHeight: 512, frameCount: FRAMES }));
            const videoFiles = camNames.map(n => ({ name: n, assignedCamera: n, videoPath: n + '.mp4' }));
            AS.state.sessions = [session]; AS.state.session = session; AS.state.activeSessionIdx = 0;
            AS.state.views = views; AS.state.videoFiles = videoFiles;
            const labels = fileio.buildSlpLabelsAllViews(session, views, videoFiles);
            AS.state.sessions = []; AS.state.session = null; AS.state.views = []; AS.state.videoFiles = [];
            return window.__b64(await window.SleapIO.saveSlpToBytes(labels));
        };
        window.__b64 = (bytes) => {
            let s = ''; const C = 0x8000;
            for (let o = 0; o < bytes.length; o += C) s += String.fromCharCode.apply(null, bytes.subarray(o, o + C));
            return btoa(s);
        };
        // A real file handle, in the origin's private file system: the same createWritable,
        // the same swap file, the same close that replaces the file.
        const root = await navigator.storage.getDirectory();
        window.__original = Uint8Array.from('the file the user already has'.split('').map(c => c.charCodeAt(0)));
        window.__file = async (name) => {
            const handle = await root.getFileHandle(name, { create: true });
            const w = await handle.createWritable();
            await w.write(window.__original);
            await w.close();
            return handle;
        };
        window.__bytesOf = async (handle) => new Uint8Array(await (await handle.getFile()).arrayBuffer());
        window.__usage = async () => (await navigator.storage.estimate()).usage;
    });
    const project = Buffer.from(await page.evaluate(() => window.__buildProject()), 'base64');

    const openInApp = async (fn, name) => {
        const dir = fs.mkdtempSync(path.join(tmp, 'p-'));
        const p = path.join(dir, name);
        fs.writeFileSync(p, project);
        await page.setInputFiles('#__picked', p);
        await page.evaluate((fn) => {
            window.__loadDone = false;
            (async () => {
                const f = document.getElementById('__picked').files[0];
                if (fn === 'eager') await (await import('/import-export/slp-import.js')).handleLoadSlpFile(f);
                else await (await import('/loading/session-loader.js')).handleLoadProjectSlpLazy(f);
            })().then(() => { window.__loadDone = true; }, e => { window.__loadDone = 'ERR ' + (e && e.stack || e); });
        }, fn);
        for (let i = 0; i < 2400; i++) {
            const st = await page.evaluate(() => {
                const b = [...document.querySelectorAll('button')].find(x => x.offsetParent && /^(Skip|Later|Cancel|Not now)$/i.test((x.textContent || '').trim()));
                if (b) { b.click(); return { clicked: true }; }
                return { done: window.__loadDone, prompt: !!document.getElementById('videoSelectOverlay') };
            });
            if (st.prompt) await page.keyboard.press('Escape');
            if (st.done) return st.done;
            await page.waitForTimeout(50);
        }
        return 'timeout';
    };
    /** Point the app at a real handle holding the original, ready for File ▸ Save. */
    const readyToSave = (name) => page.evaluate(async (name) => {
        const AS = await import('/ui/app-state.js');
        const cams = (AS.state.session.cameras || []).map(c => c.name);
        if (!AS.state.views.length) AS.state.views = cams.map(n => ({ name: n, videoWidth: 512, videoHeight: 512, frameCount: 300 }));
        if (!AS.state.videoFiles.length) AS.state.videoFiles = cams.map(n => ({ name: n, assignedCamera: n, videoPath: n + '.mp4' }));
        window.__handle = await window.__file(name);
        AS.state.slpFileHandle = window.__handle;
        (await import('/import-export/save-load.js')).markDirty();
    }, name);
    /** Press File ▸ Save, under `arm`; what the status says and what the file on disk holds. */
    const fileSave = (arm) => page.evaluate(async (arm) => {
        const sl = await import('/import-export/save-load.js');
        const AS = await import('/ui/app-state.js');
        window.__arm(arm || {});
        const proto = FileSystemWritableFileStream.prototype, abort = proto.abort;
        let aborted = 0;
        proto.abort = function () { aborted++; return abort.apply(this, arguments); };
        try { await sl.quickSave(); } finally { proto.abort = abort; }
        const fired = window.__fail.fired;
        window.__arm({});
        const bytes = await window.__bytesOf(window.__handle);
        const untouched = bytes.length === window.__original.length && bytes.every((b, i) => b === window.__original[i]);
        const tables = untouched ? null : window.__tablesIn(bytes);
        return { status: (document.getElementById('statusText') || {}).textContent, dirty: AS.state.isDirty, untouched, fired, aborted, size: bytes.length,
            opens: !!tables && !tables.cantOpen, frames: tables && tables.frames ? tables.frames.length : null, pred: tables && tables.pred_points ? tables.pred_points.length : null };
    }, arm);

    for (const [how, fn] of [['opened eagerly: saveSlpToBytes, bytes to the handle', 'eager'], ['opened lazily: the streaming writer, through the sink', 'lazy']]) {
        const done = await openInApp(fn, fn + '.slp');
        check(done === true, `${how}: opens`, done);
        await readyToSave(fn + '-quick.slp');
        const bad = await fileSave({ atClose: true });
        check(bad.fired > 0 && /^Save failed: Could not finish writing the file/.test(bad.status), `${how}: a write that fails while the file closes: "Save failed"`, bad);
        check(bad.untouched, `${how}: the file on disk is untouched`, bad);
        check(bad.dirty === true, `${how}: the project still says it has unsaved changes`, bad);
        const good = await fileSave();
        check(/^Saved/.test(good.status) && good.opens && good.frames > 0 && !good.dirty, `${how}: the same Save works once the writes do`, good);
    }

    // A real storage quota, nothing injected: the swap file can't grow.
    const cdp = await context.newCDPSession(page);
    for (const [how, fn] of [['opened eagerly', 'eager'], ['opened lazily', 'lazy']]) {
        const done = await openInApp(fn, fn + '-quota.slp');
        check(done === true, `quota, ${how}: opens`, done);
        await readyToSave(fn + '-quota-quick.slp');
        const before = await page.evaluate(() => window.__usage());
        await cdp.send('Storage.overrideQuotaForOrigin', { origin: ORIGIN, quotaSize: before + 2000 });
        const bad = await fileSave();
        const after = await page.evaluate(() => window.__usage());
        await cdp.send('Storage.overrideQuotaForOrigin', { origin: ORIGIN });
        check(/^Save failed/.test(bad.status) && bad.untouched, `quota, ${how}: "Save failed", the file on disk untouched`, bad);
        check(bad.aborted === 1 && after <= before, `quota, ${how}: the swap file is discarded, so a full disk isn't left full`, { before, after, aborted: bad.aborted });
        const good = await fileSave();
        check(/^Saved/.test(good.status) && good.opens, `quota, ${how}: the same Save works with room`, good);
    }

    // A multi-session save, whose second pass closes the one file.
    const twoLazySessions = async () => {
        const made = await page.evaluate(async () => {
            const pd = await import('/pose/pose-data.js');
            const fileio = await import('/import-export/file-io.js');
            const { Skeleton, Camera, Instance, FrameGroup, Session } = pd;
            const out = {};
            for (const [name, F] of [['camA', 30], ['camB', 20]]) {
                const cam = new Camera(name, [[900, 0, 128], [0, 900, 128], [0, 0, 1]], [0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0], [256, 256]);
                const s = new Session([cam], new Skeleton('skeleton', ['n0', 'n1'], [[0, 1]]), ['t0'], name);
                for (let f = 0; f < F; f++) {
                    const fg = new FrameGroup(f); s.addFrameGroup(fg);
                    fg.addInstance(name, new Instance([[50 + f, 60], [70 + f, 80]], 0, 'predicted', 0.9));
                }
                out[name] = window.__b64(await window.SleapIO.saveSlpToBytes(fileio.buildSlpLabels(s, name, false, { videoWidth: 256, videoHeight: 256, frameCount: F, videoPath: name + '.mp4' })));
            }
            return out;
        });
        const dir = fs.mkdtempSync(path.join(tmp, 'multi-'));
        const paths = {};
        for (const [name, b] of Object.entries(made)) {
            paths[name] = path.join(dir, name + '.slp');
            fs.writeFileSync(paths[name], Buffer.from(b, 'base64'));
        }
        await page.setInputFiles('#__picked', [paths.camA, paths.camB]);
        await page.evaluate(async () => {
            const pd = await import('/pose/pose-data.js');
            const AS = await import('/ui/app-state.js');
            const { SioLazyLoader } = await import('/loading/sio-lazy-loader.js');
            const { Skeleton, Camera, Session } = pd;
            const picked = [...document.getElementById('__picked').files];
            const sessions = [];
            for (const name of ['camA', 'camB']) {
                const cam = new Camera(name, [[900, 0, 128], [0, 900, 128], [0, 0, 1]], [0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0], [256, 256]);
                const s = new Session([cam], new Skeleton('skeleton', ['n0', 'n1'], [[0, 1]]), ['t0'], name);
                const loader = new SioLazyLoader();
                await loader.open(name, picked.find(f => f.name === name + '.slp'));
                s.lazyLoader = loader;
                sessions.push(s);
            }
            AS.state.sessions = sessions; AS.state.session = sessions[0]; AS.state.activeSessionIdx = 0;
            AS.state.views = ['camA', 'camB'].map(n => ({ name: n, videoWidth: 256, videoHeight: 256, frameCount: 30 }));
            AS.state.videoFiles = ['camA', 'camB'].map(n => ({ name: n, assignedCamera: n, videoPath: n + '.mp4' }));
            AS.state.slpFileHandle = null;
        });
    };
    const multiSave = (arm) => page.evaluate(async (arm) => {
        const sl = await import('/import-export/save-load.js');
        const orig = URL.createObjectURL;
        let blob = null;
        URL.createObjectURL = (b) => { if (b.type === 'application/x-hdf5') blob = b; return orig.call(URL, b); };
        window.__arm(arm || {});
        try { await sl.saveProjectSlp(); } finally { URL.createObjectURL = orig; }
        const fired = window.__fail.fired;
        window.__arm({});
        return { status: (document.getElementById('statusText') || {}).textContent, wrote: !!blob, fired };
    }, arm);
    await twoLazySessions();
    const m = await multiSave({ atClose: true });
    check(m.fired > 0 && !m.wrote && /^Save failed: Could not finish writing the file/.test(m.status), 'a multi-session save: a write that fails while the file closes: "Save failed", no file', m);
    const m2 = await multiSave();
    check(!m2.wrote && /Cannot save/.test(m2.status), 'and the next Save refuses, as after any multi-session save', m2);

    // ================================================================= 5
    console.log('\n-- 5. the app\'s own writer: the 3D points export --');
    await page.evaluate(async () => {
        const fileio = await import('/import-export/file-io.js');
        const pd = await import('/pose/pose-data.js');
        const { Skeleton, Camera, Instance, InstanceGroup, FrameGroup, Session } = pd;
        const cam = new Camera('cam0', [[900, 0, 128], [0, 900, 128], [0, 0, 1]], [0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0], [256, 256]);
        const session = new Session([cam], new Skeleton('skeleton', ['n0', 'n1'], [[0, 1]]), ['t0'], 'export');
        for (let f = 0; f < 50; f++) {
            const fg = new FrameGroup(f); session.addFrameGroup(fg);
            const g = new InstanceGroup(f + 1, 0);
            const inst = new Instance([[50 + f, 60], [70 + f, 80]], 0, 'predicted', 0.9);
            fg.addInstance('cam0', inst); g.addInstance('cam0', inst);
            g.points3d = new Float64Array(6).map((_, i) => f + i * 0.5);
            session.instanceGroups.set(f, [g]);
        }
        window.__exportSession = session;
        // Once, so the app has set up its h5wasm (window.h5wasm, which this export writes
        // with) before the failures go in underneath.
        await fileio.buildPoints3dH5(session);
        const FS = window.h5wasm.FS;
        const write = FS.write;
        window.__exportFail = { nth: 0, count: 0 };
        FS.write = function (stream) {
            const c = window.__exportFail;
            if (stream && stream.path && stream.path.endsWith('export_points3d.h5') && ++c.count === c.nth) throw new FS.ErrnoError(28);
            return write.apply(this, arguments);
        };
    });
    const exportWith = (nth) => page.evaluate(async (nth) => {
        const fileio = await import('/import-export/file-io.js');
        window.__exportFail = { nth, count: 0 };
        const quiet = console.error; console.error = () => {};
        const out = {};
        try {
            const blob = await fileio.buildPoints3dH5(window.__exportSession);
            out.tables = Object.fromEntries(Object.entries(window.__tablesIn(new Uint8Array(await blob.arrayBuffer()))).map(([k, v]) => [k, typeof v === 'string' ? v : Array.from(v).reduce((a, x) => a + (x === x ? Number(x) : 0), 0)]));
        } catch (e) { out.threw = String(e.message).slice(0, 160); }
        console.error = quiet;
        out.count = window.__exportFail.count;
        return out;
    }, nth);
    const expHealthy = await exportWith(0);
    check(expHealthy.count > 3 && expHealthy.tables && expHealthy.tables.points_3d > 0, 'healthy', expHealthy);
    const expWrong = [];
    let expThrew = 0;
    for (let nth = 1; nth <= expHealthy.count; nth++) {
        const r = await exportWith(nth);
        if (r.threw) expThrew++;
        else if (JSON.stringify(r.tables) !== JSON.stringify(expHealthy.tables)) expWrong.push({ nth, tables: r.tables });
    }
    check(expWrong.length === 0, `${expHealthy.count} writes failed one at a time: ${expThrew} errors, none wrong`, expWrong.slice(0, 2));

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
