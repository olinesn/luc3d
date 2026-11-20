/**
 * slp-embedded-images.mjs — a `.slp` with images inside it opens and shows them.
 *
 * `handleLoadSlpFile` failed on every such file: h5wasm returns the videos' int64
 * width, height and frame numbers as BigInt, and a canvas or frame arithmetic
 * throws on one ("SLP load error"). Behind that error were more, each of which this
 * test goes through with real files:
 *
 *  1. A real SLEAP package (variable-length rows): opens, and the canvas holds that
 *     file's exact image.
 *  2. Fixed-width rows, with video groups numbered out of list order: `getFrame`
 *     read one number where it needed a row, and the group's number decided the
 *     camera, which put the images on the wrong one. Here every image has its own
 *     pixel value, so each camera and frame is checked exactly. A frame with no
 *     image shows nothing, not the last picture under this frame's poses.
 *  3. Two sessions: every camera used to appear in every session, and switching
 *     away and back lost the views. A session opens on a frame with a picture. A
 *     switch seeks only to the frame it lands on, and an older seek that finishes
 *     late does not paint over a newer one (the switch used to seek to the old
 *     session's frame number first, which could leave that picture up).
 *  4. One embedded and one external video: the external one's camera has no view,
 *     and a labeled frame past the last image can still be reached.
 *  5. Images and no labels, with no size attributes (read from an image), a video
 *     with no images (no size to read, and no hang waiting for one), and two videos
 *     with no source name, on a camera each. A source video that is an image sequence,
 *     whose name is a list ("dName.replace is not a function"), through both readers.
 *  6. A file saved by the bundled JS writer opens the same way, though that writer
 *     runs all a video's images together in one row of bytes, and an older one
 *     without `frame_sizes` to say where each ends.
 *  7. A frame that cannot be read gives no picture; it does not hang the load.
 *  8. Save keeps the images. On real file handles (the browser's private storage, where a
 *     handle behaves as one on disk does): the menus make the opened file the save target,
 *     Save writes over it without asking, and the project goes on showing its images though
 *     the File they were read from no longer reads. Reopened, the file has the same images
 *     on the same cameras at the same frames and the edited label. Every layout, several
 *     sessions, two files merged, Save As, a save that fails (the file untouched), images
 *     in no session (not a save target), a download, the streaming writer. Python sleap-io
 *     0.9.5 reads what was saved, when `uv` is there to run it (`compare_saved.py`).
 *  9. File ▸ Open above 200 MB: a package stays on this path, which is the only one
 *     that shows its images.
 * 10. Load SLP with a project open adds the file's videos, on their own cameras.
 * 11. Opening another project closes the previous one's decoders and worker, as do
 *     File ▸ New, deleting the last session and the lazy route. Going to another
 *     session keeps the decoders and drops only their decoded frames.
 * 12. Playing through the frames neither stops nor throws. Thumbnails have a picture.
 * 13. Colour images: red and blue the right way round, as Python reads them.
 * 14. Uncompressed pixels (format "hdf5").
 *
 * The fixtures are in `tests/fixtures/slp-embedded/` (`make_fixture.py` says which is
 * what). Unit test of the rules: `tests/test-slp-package-reader.mjs`.
 *
 * Run: node tests/e2e/slp-embedded-images.mjs   (spawns its own http.server; PORT=…)
 */
import { chromium } from 'playwright';
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8433);
const FIXTURES = path.join(repoRoot, 'tests', 'fixtures', 'slp-embedded');
const fixture = (name) => path.join(FIXTURES, name);

let fails = 0;
const check = (c, m, extra) => { console.log((c ? '  ✓ ' : '  ✗ ') + m + (c || extra === undefined ? '' : ' — ' + JSON.stringify(extra))); if (!c) fails++; };
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
for (let i = 0; i < 60; i++) {
    try { if ((await fetch(`http://localhost:${PORT}/index.html`)).ok) break; } catch (e) { /* not up yet */ }
    await new Promise(r => setTimeout(r, 200));
}

const browser = await chromium.launch();

async function freshPage() {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    page.on('pageerror', e => { console.log('  [pageerror]', String(e).slice(0, 300)); fails++; });
    page.on('console', m => { if (m.type() === 'error' && /FATAL/.test(m.text())) { console.log('  [console]', m.text().slice(0, 300)); fails++; } });
    page.on('dialog', d => d.accept());
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state && window.SleapIO, null, { timeout: 60000 });
    await page.evaluate(() => {
        const input = document.createElement('input');
        input.type = 'file'; input.id = '__pick'; input.style.display = 'none';
        document.body.appendChild(input);
        window.__pickedFile = () => document.getElementById('__pick').files[0];
        const status = () => (document.getElementById('statusText') || document.getElementById('status')).textContent;
        const vc = async () => (await import('/ui/app-state.js')).videoController;
        // What each camera shows at a frame: its pixel value, or null for an empty canvas.
        window.__show = async (frame) => {
            if (frame !== undefined) await (await vc()).seekToFrame(frame);
            const out = {};
            for (const v of window.__lucid.state.views) {
                const d = v.ctx.getImageData(0, 0, 1, 1).data;
                out[v.name] = d[3] === 0 ? null : d[0];
            }
            return out;
        };
        window.__summary = () => {
            const s = window.__lucid.state;
            return {
                status: status(),
                active: s.activeSessionIdx,
                sessions: s.sessions.length,
                current: s.currentFrame,
                total: s.totalFrames,
                cameras: s.session ? s.session.cameras.map(c => c.name) : [],
                labeled: s.session ? s.session.frameIndices : [],
                views: s.views.map(v => v.name),
                videoFiles: s.videoFiles.length,
                workers: (s.frameWorkers || []).length,
                lazy: !!(s.session && s.session.lazyLoader),
            };
        };
    });
    return page;
}

/**
 * What each camera shows once the page has painted what it was doing: a session
 * switch returns before its first frame is on the canvas. Gives up after 5 s.
 */
async function settlesOn(page, want) {
    let got = null;
    for (let i = 0; i < 100; i++) {
        got = await page.evaluate(() => window.__show());
        if (eq(got, want)) break;
        await page.waitForTimeout(50);
    }
    return got;
}

/** Open a fixture through `handleLoadSlpFile`, as File ▸ Load SLP does. */
async function open(page, name) {
    await page.setInputFiles('#__pick', fixture(name));
    await page.evaluate(async () => {
        const { handleLoadSlpFile } = await import('/import-export/slp-import.js');
        // A load stuck waiting for a frame must fail this test, not hang it.
        await Promise.race([
            handleLoadSlpFile(window.__pickedFile()),
            new Promise(res => setTimeout(res, 60000)),
        ]);
    });
    // Views settle a frame after the load's own seek.
    await page.waitForTimeout(100);
    return page.evaluate(() => window.__summary());
}

try {
    console.log('\n1. A real SLEAP package (variable-length rows)');
    {
        const page = await freshPage();
        const s = await open(page, 'minimal_instance.pkg.slp');
        check(/^SLP loaded/.test(s.status), 'loads', s.status);
        check(eq(s.views, ['centered_pair_low_quality']), 'one view, named for its source video', s.views);
        check(eq(s.labeled, [0]) && s.current === 0, 'on its labeled frame', s);
        const shown = await page.evaluate(() => {
            const v = window.__lucid.state.views[0];
            const d = v.ctx.getImageData(0, 0, v.canvas.width, v.canvas.height).data;
            let sum = 0;
            for (let i = 0; i < d.length; i += 4) sum += d[i];
            return {
                sizes: [v.videoWidth, v.videoHeight, v.canvas.width, v.canvas.height],
                types: [typeof v.videoWidth, typeof v.videoHeight],
                frameNumbers: v.decoder._videoInfo.frameNumbers.map(n => typeof n),
                frames: v.decoder.samples.length,
                sum,
                center: d[(192 * 384 + 192) * 4],
                poses: window.__lucid.state.session.frameGroups.get(0).getUnlinkedInstances(v.name).length,
            };
        });
        check(eq(shown.sizes, [384, 384, 384, 384]), 'the file\'s size, on the view and the canvas', shown.sizes);
        check(eq(shown.types, ['number', 'number']) && eq(shown.frameNumbers, ['number']), 'plain numbers, not BigInt', shown);
        // From Pillow's decode of the same PNG.
        check(shown.sum === 754996 && shown.center === 12, 'the canvas holds that exact image', shown);
        check(shown.poses === 2, 'with its two poses', shown.poses);
        await page.close();
    }

    console.log('\n2. Fixed-width rows, video groups out of list order');
    {
        const page = await freshPage();
        const s = await open(page, 'out_of_order_embedded.pkg.slp');
        check(/^SLP loaded/.test(s.status), 'loads', s.status);
        check(eq(s.views, ['clip_a', 'clip_b', 'clip_c']), 'views in list order', s.views);
        check(s.total === 12, 'as many frames as the last image', s.total);
        check(s.current === 5, 'opens on the first labeled frame', s.current);
        // `video2` is clip_a, `video0` clip_b, `video1` clip_c: each frame's value says whose it is.
        const expect = [
            [5, { clip_a: 20, clip_b: null, clip_c: null }],
            [3, { clip_a: null, clip_b: null, clip_c: 60 }],
            [7, { clip_a: null, clip_b: 40, clip_c: null }],
            [9, { clip_a: null, clip_b: 41, clip_c: null }],
            [11, { clip_a: 21, clip_b: null, clip_c: null }],
            [6, { clip_a: null, clip_b: null, clip_c: null }],   // right after pictures: nothing stale
            [5, { clip_a: 20, clip_b: null, clip_c: null }],     // and again, from the cache
        ];
        for (const [frame, want] of expect) {
            const got = await page.evaluate(f => window.__show(f), frame);
            check(eq(got, want), `frame ${frame}: each camera shows its own image`, got);
        }
        const poses = await page.evaluate(() => {
            const sess = window.__lucid.state.session;
            const on = (f) => ['clip_a', 'clip_b', 'clip_c'].map(c => sess.frameGroups.get(f).getUnlinkedInstances(c).map(u => u.instance.type));
            return { f5: on(5), f7: on(7) };
        });
        check(eq(poses.f5, [['user'], [], []]), 'frame 5\'s pose is on the camera with frame 5\'s image', poses.f5);
        check(eq(poses.f7, [[], ['predicted'], []]), 'frame 7\'s pose is on the camera with frame 7\'s image', poses.f7);
        await page.close();
    }

    console.log('\n3. Two sessions');
    {
        const page = await freshPage();
        const s = await open(page, 'two_sessions.pkg.slp');
        check(/^SLP loaded/.test(s.status), 'loads', s.status);
        check(s.sessions === 2 && s.active === 0, 'two sessions, the first active', s);
        check(eq(s.views, ['north', 'south']), 'only this session\'s cameras have views', s.views);
        check(s.videoFiles === 4, 'all four videos are kept', s.videoFiles);
        check(s.workers === 1, 'one worker reads them all', s.workers);
        const indices = await page.evaluate(() => window.__lucid.state.sessions.map(x => x.videoFileIndices));
        check(eq(indices, [[0, 1], [2, 3]]), 'each session owns its videos', indices);
        // Frame 1 is labeled but has no image.
        check(eq(s.labeled, [1, 8]), 'session 1 is labeled on frames 1 and 8', s.labeled);
        check(s.current === 8, 'opens on its first labeled frame with an image', s.current);
        let want = { north: 31, south: 32 };
        let shown = await settlesOn(page, want);
        check(eq(shown, want), 'shows that frame', shown);
        check(eq(await page.evaluate(() => window.__show(3)), { north: 30, south: null }), 'and an unlabeled frame with an image');

        const toSession = (i) => page.evaluate(async (idx) => {
            await (await import('/ui/sessions-panes.js')).switchSession(idx);
            return window.__summary();
        }, i);
        await page.evaluate(() => { window.__firstSession = window.__lucid.state.views.map(v => v.decoder); });
        let t = await toSession(1);
        check(eq(t.views, ['east', 'west']), 'switching shows the other session\'s cameras', t.views);
        const left = await page.evaluate(() => window.__firstSession.map(d => [d.cache.size, d._closed]));
        check(eq(left, [[0, false], [0, false]]), 'the session left keeps its decoders, without their decoded frames', left);
        check(t.current === 6, 'on a frame with a picture', t.current);
        want = { east: 50, west: 51 };
        shown = await settlesOn(page, want);
        check(eq(shown, want), 'its own images', shown);
        check(eq(await page.evaluate(() => window.__show(12)), { east: null, west: 52 }), 'at another of its frames');
        await page.evaluate(async () => {
            const { VideoController } = await import('/loading/video.js');
            const seek = VideoController.prototype.seekToFrame;
            window.__seeks = [];
            window.__unwrapSeek = () => { VideoController.prototype.seekToFrame = seek; };
            VideoController.prototype.seekToFrame = function (f) { window.__seeks.push(f); return seek.call(this, f); };
        });
        t = await toSession(0);
        const seeks = await page.evaluate(() => { window.__unwrapSeek(); return window.__seeks; });
        check(seeks.length > 0 && seeks.every(f => f === 3), 'a switch only ever seeks to the frame it lands on', seeks);
        check(eq(t.views, ['north', 'south']), 'switching back brings the first session\'s views back', t.views);
        check(t.current === 3, 'on the frame it was left on', t.current);
        want = { north: 30, south: null };
        shown = await settlesOn(page, want);
        check(eq(shown, want), 'with its images', shown);
        t = await toSession(1);
        want = { east: null, west: 52 };
        shown = await settlesOn(page, want);
        check(eq(shown, want), 'and the second again', shown);

        const removed = await page.evaluate(async () => {
            (await import('/ui/sessions-panes.js')).removeSession(1);
            return window.__summary();
        });
        await page.waitForTimeout(100);
        check(eq(removed.views, ['north', 'south']) && removed.sessions === 1, 'deleting the active session shows the remaining one', removed);
        await page.evaluate(() => window.__show(8));
        want = { north: 31, south: 32 };
        shown = await settlesOn(page, want);
        check(eq(shown, want), 'with its images', shown);

        // A seek that finishes after a newer one: frame 3 asked for first and held
        // back, frame 8 (cached) asked for second and painted.
        const late = await page.evaluate(async () => {
            const { videoController: vc } = await import('/ui/app-state.js');
            const views = window.__lucid.state.views;
            const release = [];
            const getFrame = views.map(v => v.decoder.getFrame);
            views.forEach((v, i) => {
                v.decoder.getFrame = (f) => f === 3
                    ? new Promise(res => release.push(res)).then(() => getFrame[i].call(v.decoder, f))
                    : getFrame[i].call(v.decoder, f);
            });
            const older = vc.seekToFrame(3);
            await vc.seekToFrame(8);
            release.forEach(res => res());
            await older;
            views.forEach((v, i) => { v.decoder.getFrame = getFrame[i]; });
            return { shown: await window.__show(), current: window.__lucid.state.currentFrame };
        });
        check(late.current === 8 && eq(late.shown, { north: 31, south: 32 }), 'an older seek finishing late leaves the newer picture up', late);
        await page.close();
    }

    console.log('\n4. One embedded video, one external');
    {
        const page = await freshPage();
        const s = await open(page, 'mixed_external.pkg.slp');
        check(/^SLP loaded/.test(s.status), 'loads', s.status);
        check(eq(s.cameras, ['inside', 'outside']), 'both cameras', s.cameras);
        check(eq(s.views, ['inside']), 'a view for the embedded one', s.views);
        check(eq(await page.evaluate(() => window.__show()), { inside: 70 }), 'its image');
        check(s.total === 5, 'the labeled frame past the last image counts', s.total);
        await page.evaluate(() => window.__show(4));
        const at4 = await page.evaluate(() => ({ cur: window.__lucid.state.currentFrame, n: window.__lucid.state.session.frameGroups.get(4).getUnlinkedInstances('outside').length }));
        check(at4.cur === 4 && at4.n === 1, 'and can be reached, with its pose', at4);
        await page.close();
    }

    console.log('\n5. Images and no labels, no size attributes');
    {
        const page = await freshPage();
        const s = await open(page, 'images_only.pkg.slp');
        check(/^SLP loaded/.test(s.status), 'loads', s.status);
        check(s.labeled.length === 0 && s.current === 10, 'with no labels, opens on its first image', s);
        check(eq(s.views, ['images_only.pkg', 'images_only.pkg_2']), 'two videos with no source name get a camera each', s.views);
        const size = await page.evaluate(() => window.__lucid.state.views.map(v => [v.videoWidth, v.videoHeight]));
        check(eq(size[0], [2, 2]), 'the size is read from an image', size);
        check(size[1][0] > 0 && size[1][1] > 0, 'and a video with no images gets a size anyway, without hanging', size);
        const view = s.views[0];
        check((await page.evaluate(() => window.__show()))[view] === 80, 'its image (uint8 rows)');
        check((await page.evaluate(() => window.__show(20)))[view] === 81, 'and the next');
        const seq = await open(page, 'image_sequence.pkg.slp');
        check(/^SLP loaded/.test(seq.status), 'a source video that is an image sequence loads', seq.status);
        check(eq(seq.views, ['frame_00000']), 'and takes the name of its first image', seq.views);
        check(eq(await page.evaluate(() => window.__show()), { frame_00000: 95 }), 'with its image');
        await page.setInputFiles('#__pick', fixture('image_sequence.pkg.slp'));
        const raw = await page.evaluate(async () => {
            const { parseSlpH5 } = await import('/import-export/file-io.js');
            const data = await parseSlpH5(window.__pickedFile());
            return data.videos.map(v => [v.sourceFilename, v.embedded, v.dataset]);
        });
        check(eq(raw, [['seq/frame_00000.png', true, 'video0/video']]), 'the raw worker reads the same name', raw);
        await page.close();
    }

    console.log('\n6. A file saved by the bundled JS writer');
    {
        const page = await freshPage();
        await page.setInputFiles('#__pick', fixture('out_of_order_embedded.pkg.slp'));
        const before = await page.evaluate(async () => {
            const SIO = await import('/lib/sleap-io/index.browser.js');
            const labels = await SIO.loadSlp(await window.__pickedFile().arrayBuffer(), { openVideos: true });
            const bytes = await SIO.saveSlpToBytes(labels, { embed: true });
            window.__jsWritten = new File([bytes], 'js_written.pkg.slp');
            return { bytes: bytes.length, videos: labels.videos.length };
        });
        check(before.bytes > 0 && before.videos === 3, 'the JS writer saves it', before);
        await page.evaluate(async () => {
            const { handleLoadSlpFile } = await import('/import-export/slp-import.js');
            await handleLoadSlpFile(window.__jsWritten);
        });
        await page.waitForTimeout(100);
        const s = await page.evaluate(() => window.__summary());
        check(/^SLP loaded/.test(s.status), 'loads', s.status);
        const shown = {};
        for (const f of [5, 11, 7, 9, 3]) shown[f] = await page.evaluate(x => window.__show(x), f);
        // Each image goes with the camera the file says; the writer's choice of names is its own.
        const seen = [20, 21, 40, 41, 60].map(v => Object.values(shown).some(row => Object.values(row).includes(v)));
        check(seen.every(Boolean), 'every image shows, at its own frame', shown);
        const byCamera = (values) => {
            const owners = new Set();
            for (const row of Object.values(shown)) for (const [cam, v] of Object.entries(row)) if (values.includes(v)) owners.add(cam);
            return owners.size;
        };
        check(byCamera([20, 21]) === 1 && byCamera([40, 41]) === 1, 'each video\'s images stay on one camera', shown);
        await page.close();

        const old = await freshPage();
        const f = await open(old, 'flat_no_sizes.pkg.slp');
        check(/^SLP loaded/.test(f.status), 'one written with no frame sizes loads', f.status);
        const name = f.views[0];
        check(f.current === 2 && (await old.evaluate(() => window.__show()))[name] === 90, 'and shows its first image', f.current);
        check((await old.evaluate(() => window.__show(9)))[name] === 92, 'its last');
        check((await old.evaluate(() => window.__show(5)))[name] === 91, 'and the one between');
        check((await old.evaluate(() => window.__show(6)))[name] === null, 'and nothing where it has none');
        await old.close();
    }

    console.log('\n7. A frame that cannot be read');
    {
        const page = await freshPage();
        const logs = [];
        page.on('console', m => logs.push(m.text()));
        await open(page, 'out_of_order_embedded.pkg.slp');
        const r = await page.evaluate(async () => {
            const v = window.__lucid.state.views[0];
            // A frame the decoder believes in and the file does not have.
            v.decoder._displayToEmbedded.set(6, 99);
            const answer = await Promise.race([
                v.decoder.getFrame(6),
                new Promise(res => setTimeout(() => res('timed out'), 5000)),
            ]);
            // Nothing below would ever return: that seek waits on the same frame.
            if (answer !== null) return { answer, shown: {}, pending: v.decoder._pending.size, still: {} };
            const shown = await window.__show(6);
            return { answer, shown, pending: v.decoder._pending.size, still: await window.__show(5) };
        });
        check(r.answer === null, 'the decoder answers null, it does not wait forever', r.answer);
        check(logs.some(l => /Could not read embedded frame 99 of video2/.test(l)), 'and logs why', logs.slice(-3));
        check(r.pending === 0, 'and has nothing left waiting', r.pending);
        check(r.shown.clip_a === null, 'that frame shows nothing', r.shown);
        check(r.still.clip_a === 20, 'and the other frames still show', r.still);
        await page.close();
    }

    console.log('\n8. Save keeps the images');
    {
        const savedDir = fs.mkdtempSync(path.join(os.tmpdir(), 'lucid-embedded-save-'));
        const pythonChecks = [];   // [what, source fixture or '-', saved path, extra args]

        // A page with real file handles on the fixtures, and what to read them back with.
        const handlePage = async () => {
            const page = await freshPage();
            await page.evaluate(async () => {
                const root = await navigator.storage.getDirectory();
                for await (const name of root.keys()) await root.removeEntry(name);
                // A real handle holding a fixture, as if picked in the file dialog.
                window.__onDisk = async (name, fixture) => {
                    const handle = await root.getFileHandle(name, { create: true });
                    const writable = await handle.createWritable();
                    await writable.write(await (await fetch('/tests/fixtures/slp-embedded/' + fixture)).arrayBuffer());
                    await writable.close();
                    return handle;
                };
                window.__entries = async () => { const out = []; for await (const k of root.keys()) out.push(k); return out.sort(); };
                window.__bytesOf = async (handle) => new Uint8Array(await (await handle.getFile()).arrayBuffer());
                window.__save = async () => {
                    const { quickSave } = await import('/import-export/save-load.js');
                    await quickSave();
                    return { status: document.getElementById('statusText').textContent, dirty: window.__lucid.state.isDirty };
                };
                window.__reopen = async (handle) => {
                    const { handleLoadSlpFile } = await import('/import-export/slp-import.js');
                    await handleLoadSlpFile(await handle.getFile());
                    await new Promise(r => setTimeout(r, 100));
                };
                // What each camera shows at each frame, read from the file, not the decoded frames.
                window.__showFresh = async (frames) => {
                    const out = {};
                    for (const f of frames) {
                        window.__lucid.state.views.forEach(v => v.decoder.releaseFrames());
                        out[f] = await window.__show(f);
                    }
                    return out;
                };
                window.__poseAt = (frame, camera) => {
                    const fg = window.__lucid.state.session.frameGroups.get(frame);
                    const linked = fg.getInstances(camera);
                    const inst = linked.length ? linked[0] : fg.getUnlinkedInstances(camera)[0].instance;
                    return inst.toPointsArray()[0];
                };
                window.__editAt = async (frame, camera, x, y) => {
                    const fg = window.__lucid.state.session.frameGroups.get(frame);
                    const linked = fg.getInstances(camera);
                    const inst = linked.length ? linked[0] : fg.getUnlinkedInstances(camera)[0].instance;
                    inst.setPoint(0, x, y);
                    (await import('/import-export/save-load.js')).markDirty();
                };
            });
            return page;
        };
        // A saved file on the real disk, for Python.
        const keep = async (page, handleExpr, name, source, extra) => {
            const b64 = await page.evaluate(async (expr) => {
                const file = await (await eval(expr)).getFile();
                return await new Promise(res => { const r = new FileReader(); r.onload = () => res(r.result.split(',')[1]); r.readAsDataURL(file); });
            }, handleExpr);
            const to = path.join(savedDir, name);
            fs.writeFileSync(to, Buffer.from(b64, 'base64'));
            pythonChecks.push([name, source, to, extra || []]);
        };

        console.log('  -- saving over the opened file');
        {
            const page = await handlePage();
            await page.evaluate(async () => {
                window.__opened = await window.__onDisk('opened.pkg.slp', 'out_of_order_embedded.pkg.slp');
                window.__asked = 0;
                window.showOpenFilePicker = async () => [window.__opened];
                window.showSaveFilePicker = async () => { window.__asked++; throw new Error('should not ask'); };
            });
            for (const menu of ['menuLoadSlp', 'menuLoadProject']) {
                await page.evaluate(() => {
                    window.__lucid.state.session = null;
                    window.__lucid.state.sessions = [];
                    window.__lucid.state.slpFileHandle = null;
                });
                await page.evaluate((id) => document.getElementById(id).click(), menu);
                await page.waitForFunction(() => {
                    const s = window.__lucid.state;
                    return s.session && s.views.length === 3 && s.slpFileHandle;
                }, null, { timeout: 15000 }).catch(() => {});
                const opened = await page.evaluate(() => window.__lucid.state.slpFileHandle === window.__opened);
                check(opened, `${menu}: the file with the images is the save target`);
            }
            await page.waitForTimeout(100);
            const before = await page.evaluate(() => window.__showFresh([3, 5, 7, 9, 11, 6]));
            await page.evaluate(() => window.__editAt(5, 'clip_a', 1.5, 1.25));

            const first = await page.evaluate(async () => {
                const file = await window.__opened.getFile();
                const oldDecoder = window.__lucid.state.views[0].decoder;
                const oldWorker = oldDecoder._worker;
                const oldTerminate = oldWorker.terminate.bind(oldWorker);
                let stopped = false;
                oldWorker.terminate = () => { stopped = true; oldTerminate(); };
                const saved = await window.__save();
                let oldFileReads = true;
                try { await file.slice(0, 16).arrayBuffer(); } catch (e) { oldFileReads = e.name; }
                const s = window.__lucid.state;
                return {
                    ...saved, asked: window.__asked, oldFileReads, stopped,
                    handleKept: s.slpFileHandle === window.__opened,
                    sameDecoder: s.views[0].decoder === oldDecoder,
                    workers: s.frameWorkers.length,
                    bytes: (await window.__bytesOf(window.__opened)).length,
                    entries: await window.__entries(),
                    saving: s.isSaving,
                    card: !document.getElementById('loadingOverlay') || document.getElementById('loadingOverlay').classList.contains('hidden'),
                };
            });
            check(first.asked === 0 && first.handleKept, 'Save writes to it without asking', first);
            check(/^Saved \(/.test(first.status) && !/Labels only/.test(first.status), 'the status says saved, and nothing about labels only', first.status);
            check(first.dirty === false, 'and the project counts as saved', first);
            check(first.oldFileReads !== true, 'the File the images were read from no longer reads, as the file was replaced', first.oldFileReads);
            check(first.stopped && first.workers === 1 && first.sameDecoder, 'the views keep their decoders, on a new worker reading the new file', first);
            check(eq(first.entries, ['opened.pkg.slp']), 'no temporary file is left', first.entries);
            check(!first.saving && first.card, 'the save is over, and its card gone', first);
            check(eq(await page.evaluate(() => window.__showFresh([3, 5, 7, 9, 11, 6])), before), 'the project goes on showing the same images, from the new file');

            await page.evaluate(() => window.__editAt(5, 'clip_a', 1.75, 1.25));
            const second = await page.evaluate(() => window.__save());
            check(/^Saved \(/.test(second.status), 'a second Save works, with the images now read from the file saved over', second.status);
            check(eq(await page.evaluate(() => window.__showFresh([3, 5, 7, 9, 11, 6])), before), 'and still shows them');
            await keep(page, 'window.__opened', 'over_itself.slp', 'out_of_order_embedded.pkg.slp', ['--edited', '5,1.75,1.25']);

            await page.evaluate(() => window.__reopen(window.__opened));
            const s = await page.evaluate(() => window.__summary());
            check(/^SLP loaded/.test(s.status) && eq(s.views, ['clip_a', 'clip_b', 'clip_c']), 'reopened: the same cameras', s);
            check(s.current === 5 && s.total === 12, 'on the same frame, with as many frames', s);
            check(eq(await page.evaluate(() => window.__showFresh([3, 5, 7, 9, 11, 6])), before), 'the same images on the same cameras at the same frames');
            check(eq(await page.evaluate(() => window.__poseAt(5, 'clip_a')), [1.75, 1.25]), 'and the edited label');
            await page.close();
        }

        console.log('  -- several sessions');
        {
            const page = await handlePage();
            await page.evaluate(async () => {
                window.__target = await window.__onDisk('two.pkg.slp', 'two_sessions.pkg.slp');
                await window.__reopen(window.__target);
                window.__lucid.state.slpFileHandle = window.__target;
                await (await import('/ui/sessions-panes.js')).switchSession(1);
            });
            await settlesOn(page, { east: 50, west: 51 });
            const saved = await page.evaluate(() => window.__save());
            check(/^Saved \(/.test(saved.status), 'saves with the second session on screen', saved.status);
            const toSession = (i) => page.evaluate(async (idx) => { await (await import('/ui/sessions-panes.js')).switchSession(idx); }, i);
            check(eq(await page.evaluate(() => window.__showFresh([6, 12])), { 6: { east: 50, west: 51 }, 12: { east: null, west: 52 } }), 'the session on screen still shows its images');
            await toSession(0);
            await settlesOn(page, { north: 31, south: 32 });
            check(eq(await page.evaluate(() => window.__showFresh([3, 8])), { 3: { north: 30, south: null }, 8: { north: 31, south: 32 } }), 'and the session off screen when saved shows its own, cameras named as they were');
            await page.evaluate(() => window.__reopen(window.__target));
            const s = await page.evaluate(() => window.__summary());
            check(s.sessions === 2 && eq(s.views, ['north', 'south']), 'reopened: two sessions', s);
            check(eq(await page.evaluate(() => window.__lucid.state.sessions.map(x => x.videoFileIndices)), [[0, 1], [2, 3]]), 'each with its own videos');
            check(eq(await page.evaluate(() => window.__showFresh([3, 8])), { 3: { north: 30, south: null }, 8: { north: 31, south: 32 } }), 'the first session\'s images');
            await toSession(1);
            await settlesOn(page, { east: 50, west: 51 });
            check(eq(await page.evaluate(() => window.__showFresh([6, 12])), { 6: { east: 50, west: 51 }, 12: { east: null, west: 52 } }), 'the second session\'s images');
            await keep(page, 'window.__target', 'two_sessions.slp', 'two_sessions.pkg.slp');
            await page.close();

            // A real rig: every session's cameras have the same names.
            const same = await handlePage();
            const toSameSession = (i) => same.evaluate(async (idx) => { await (await import('/ui/sessions-panes.js')).switchSession(idx); }, i);
            const first = { 1: { north: 110, south: 111 } };
            const second = { 1: { north: 120, south: 121 }, 2: { north: 122, south: null } };
            await same.evaluate(async () => {
                window.__target = await window.__onDisk('same.pkg.slp', 'same_camera_names.pkg.slp');
                await window.__reopen(window.__target);
                window.__lucid.state.slpFileHandle = window.__target;
                await window.__save();
                await window.__reopen(window.__target);
            });
            check(eq(await same.evaluate(() => window.__showFresh([1])), first), 'sessions with the same camera names: each keeps its own images, the first');
            await toSameSession(1);
            await settlesOn(same, second[1]);
            check(eq(await same.evaluate(() => window.__showFresh([1, 2])), second), 'and the second');
            await keep(same, 'window.__target', 'same_camera_names.slp', 'same_camera_names.pkg.slp');
            await same.close();
        }

        console.log('  -- every layout');
        const layouts = [
            ['minimal_instance.pkg.slp', [0], true],
            ['mixed_external.pkg.slp', [2, 4], true],
            ['images_only.pkg.slp', [10, 20, 15], true],
            ['flat_no_sizes.pkg.slp', [2, 5, 9, 6], false],   // Python can't read the source's layout
            ['image_sequence.pkg.slp', [1], null],            // nor a source that is a list of images
            ['colors.pkg.slp', [0], true],
            ['colors_old_format.pkg.slp', [0], true],
            ['raw_pixels.pkg.slp', [4], true],
        ];
        for (const [fixture, frames, python] of layouts) {
            const page = await handlePage();
            const r = await page.evaluate(async ([fixture, frames]) => {
                window.__target = await window.__onDisk('saved.pkg.slp', fixture);
                await window.__reopen(window.__target);
                window.__lucid.state.slpFileHandle = window.__target;
                const read = () => ({
                    views: window.__lucid.state.views.map(v => v.name),
                    sizes: window.__lucid.state.views.map(v => [v.videoWidth, v.videoHeight]),
                });
                const pixel = async (f) => {
                    await window.__show(f);
                    window.__lucid.state.views.forEach(v => v.decoder.releaseFrames());
                    const out = {};
                    await window.__show(f);
                    for (const v of window.__lucid.state.views) out[v.name] = Array.from(v.ctx.getImageData(0, 0, 1, 1).data);
                    return out;
                };
                const before = { ...read(), pixels: {} };
                for (const f of frames) before.pixels[f] = await pixel(f);
                const saved = await window.__save();
                await window.__reopen(window.__target);
                const after = { ...read(), pixels: {} };
                for (const f of frames) after.pixels[f] = await pixel(f);
                return { saved, before, after, status: document.getElementById('statusText').textContent };
            }, [fixture, frames]);
            check(/^Saved \(/.test(r.saved.status) && /^SLP loaded/.test(r.status), `${fixture}: saves and reopens`, r);
            check(eq(r.after.views, r.before.views) && eq(r.after.sizes, r.before.sizes), `${fixture}: same cameras, same sizes`, r);
            check(eq(r.after.pixels, r.before.pixels), `${fixture}: same pixels, red and blue included, same empty frames`, r);
            if (python === true) await keep(page, 'window.__target', 'layout_' + fixture, fixture);
            if (python === false) await keep(page, 'window.__target', 'layout_' + fixture, '-');
            await page.close();
        }

        console.log('  -- two files in one project');
        {
            const page = await handlePage();
            await page.setInputFiles('#__pick', fixture('out_of_order_embedded.pkg.slp'));
            await page.evaluate(async () => {
                window.__target = await window.__onDisk('first.pkg.slp', 'two_sessions.pkg.slp');
                await window.__reopen(window.__target);
                window.__lucid.state.slpFileHandle = window.__target;
                await (await import('/ui/sessions-panes.js')).removeSession(1);
            });
            await page.waitForTimeout(100);
            const chooser = page.waitForEvent('filechooser');
            await page.evaluate(() => { window.__adding = import('/import-export/slp-import.js').then(m => m.handleAddSlp()); });
            await (await chooser).setFiles(fixture('out_of_order_embedded.pkg.slp'));
            await page.evaluate(() => window.__adding);
            await page.waitForTimeout(100);
            const added = await page.evaluate(async () => {
                const { paneManager } = await import('/ui/app-state.js');
                for (const name of ['clip_a', 'clip_b', 'clip_c']) paneManager.addVideoPanel(name);
                await new Promise(r => setTimeout(r, 200));
                return window.__summary();
            });
            check(added.workers === 2 && eq(added.views, ['north', 'south', 'clip_a', 'clip_b', 'clip_c']), 'a file added to the project: two sources', added);
            const frames = [3, 5, 7, 8, 11];
            const before = await page.evaluate((f) => window.__showFresh(f), frames);
            const saved = await page.evaluate(async () => ({ ...(await window.__save()), workers: window.__lucid.state.frameWorkers.length }));
            check(/^Saved \(/.test(saved.status) && saved.workers === 1, 'saved: one file holds them all, and one worker reads it', saved);
            check(eq(await page.evaluate((f) => window.__showFresh(f), frames), before), 'every camera shows the same');
            await page.evaluate(() => window.__reopen(window.__target));
            const s = await page.evaluate(async () => {
                const { paneManager } = await import('/ui/app-state.js');
                for (const name of ['clip_a', 'clip_b', 'clip_c']) if (!paneManager.hasPanel || !paneManager.hasPanel(name)) paneManager.addVideoPanel(name);
                await new Promise(r => setTimeout(r, 200));
                return window.__summary();
            });
            check(eq(s.cameras, ['north', 'south', 'clip_a', 'clip_b', 'clip_c']), 'reopened: all five cameras', s);
            check(eq(await page.evaluate((f) => window.__showFresh(f), frames), before), 'with their images');
            await page.close();
        }

        console.log('  -- Save As');
        {
            const page = await handlePage();
            const r = await page.evaluate(async () => {
                window.__opened = await window.__onDisk('opened.pkg.slp', 'out_of_order_embedded.pkg.slp');
                window.__copy = await window.__onDisk('copy.pkg.slp', 'minimal_instance.pkg.slp');
                await window.__reopen(window.__opened);
                window.__lucid.state.slpFileHandle = window.__opened;
                const original = await window.__bytesOf(window.__opened);
                let suggested = null;
                window.showSaveFilePicker = async (o) => { suggested = o.suggestedName; return window.__copy; };
                await window.__editAt(5, 'clip_a', 1.5, 1.25);
                const { saveAs } = await import('/import-export/save-load.js');
                await saveAs();
                const afterOne = await window.__bytesOf(window.__opened);
                const same = original.length === afterOne.length && original.every((b, i) => b === afterOne[i]);
                const status = document.getElementById('statusText').textContent;
                // The project no longer needs the file it came from: replace it.
                const w = await window.__opened.createWritable();
                await w.write(new Uint8Array(10));
                await w.close();
                window.__lucid.state.views.forEach(v => v.decoder.releaseFrames());
                return { same, status, suggested, handle: window.__lucid.state.slpFileHandle === window.__copy, shown: await window.__showFresh([5, 7]) };
            });
            check(r.same, 'the opened file is left as it was', r);
            check(r.handle && /^Saved \(/.test(r.status), 'the chosen file is the save target now', r);
            check(/\.pkg\.slp$/.test(r.suggested), 'it is offered with SLEAP\'s name for a file holding its images', r.suggested);
            check(eq(r.shown, { 5: { clip_a: 20, clip_b: null, clip_c: null }, 7: { clip_a: null, clip_b: 40, clip_c: null } }), 'and the project reads its images from there: the opened file can go', r.shown);
            const again = await page.evaluate(() => window.__save());
            check(/^Saved \(/.test(again.status), 'so it can save again, with the opened file gone', again.status);
            await page.evaluate(() => window.__reopen(window.__copy));
            check(eq(await page.evaluate(() => window.__poseAt(5, 'clip_a')), [1.5, 1.25]), 'with the edited label', 0);
            await page.close();
        }

        console.log('  -- a save that fails');
        {
            const page = await handlePage();
            const r = await page.evaluate(async () => {
                window.__source = await window.__onDisk('source.pkg.slp', 'out_of_order_embedded.pkg.slp');
                window.__target = await window.__onDisk('target.pkg.slp', 'minimal_instance.pkg.slp');
                await window.__reopen(window.__source);
                window.__lucid.state.slpFileHandle = window.__target;
                const before = await window.__bytesOf(window.__target);
                await window.__editAt(5, 'clip_a', 1.5, 1.25);
                // The file holding the images changes after it was opened.
                const w = await window.__source.createWritable();
                await w.write(new Uint8Array(1000));
                await w.close();
                let aborted = false;
                const open = window.__target.createWritable.bind(window.__target);
                window.__target.createWritable = async () => {
                    const writable = await open();
                    const abort = writable.abort.bind(writable);
                    writable.abort = () => { aborted = true; return abort(); };
                    return writable;
                };
                const saved = await window.__save();
                const after = await window.__bytesOf(window.__target);
                return {
                    ...saved,
                    aborted,
                    same: before.length === after.length && before.every((b, i) => b === after[i]),
                    entries: await window.__entries(),
                    saving: window.__lucid.state.isSaving,
                    hidden: document.getElementById('loadingOverlay').classList.contains('hidden'),
                };
            });
            check(/^Save failed/.test(r.status) && /could not read the images in source\.pkg\.slp/.test(r.status), 'it fails, and says which file it could not read', r.status);
            check(r.aborted && r.same, 'the write in progress is dropped, and the file it was saving to is untouched', r);
            check(r.dirty === true, 'the project still counts as unsaved', r);
            check(eq(r.entries, ['source.pkg.slp', 'target.pkg.slp']) && !r.saving && r.hidden, 'nothing is left over, and the app is free again', r);
            await page.close();

            // No room for the temporary file: the write fails, and HDF5 does not hang.
            const full = await handlePage();
            const cdp = await full.context().newCDPSession(full);
            await full.evaluate(async () => {
                window.__target = await window.__onDisk('target.pkg.slp', 'out_of_order_embedded.pkg.slp');
                await window.__reopen(window.__target);
                window.__lucid.state.slpFileHandle = window.__target;
                window.__original = await window.__bytesOf(window.__target);
            });
            const usage = await full.evaluate(async () => (await navigator.storage.estimate()).usage);
            await cdp.send('Storage.overrideQuotaForOrigin', { origin: new URL(full.url()).origin, quotaSize: usage + 4096 });
            const q = await full.evaluate(async () => {
                window.__lucidSaveScratch = 'disk';
                const saved = await Promise.race([window.__save(), new Promise(res => setTimeout(() => res({ status: 'timed out' }), 45000))]);
                window.__lucidSaveScratch = null;
                const after = await window.__bytesOf(window.__target);
                return { ...saved, same: after.length === window.__original.length && after.every((b, i) => b === window.__original[i]) };
            });
            check(/^Save failed/.test(q.status), 'with no room for its temporary file, it fails rather than hangs', q.status);
            check(q.same, 'and the file saved over is untouched', q);
            await cdp.send('Storage.overrideQuotaForOrigin', { origin: new URL(full.url()).origin });
            const retry = await full.evaluate(() => window.__save());
            check(/^Saved \(/.test(retry.status), 'with room again, the same save works', retry.status);
            await full.close();

            // Told it cannot read back the file it saved: said, and the save is still done.
            const noBack = await handlePage();
            const n = await noBack.evaluate(async () => {
                const real = await window.__onDisk('real.pkg.slp', 'out_of_order_embedded.pkg.slp');
                await window.__reopen(real);
                const getFile = real.getFile.bind(real);
                const target = await window.__onDisk('target.pkg.slp', 'minimal_instance.pkg.slp');
                const wrapped = { name: target.name, kind: 'file', createWritable: () => target.createWritable(), getFile: async () => { throw new Error('gone'); } };
                window.__lucid.state.slpFileHandle = wrapped;
                const saved = await window.__save();
                const shown = await window.__show(5);
                return { ...saved, shown, wrote: (await window.__bytesOf(target)).length, getFile: typeof getFile };
            });
            check(/^Saved \(/.test(n.status) && /could not be read back/.test(n.status), 'a saved file it can\'t read back: saved, and it says so', n.status);
            check(n.dirty === false && n.shown.clip_a === 20, 'and the views go on from the file they had', n);
            await noBack.close();
        }

        console.log('  -- images no session holds');
        {
            const page = await handlePage();
            const r = await page.evaluate(async () => {
                window.__opened = await window.__onDisk('opened.pkg.slp', 'video_in_no_session.pkg.slp');
                window.__newFile = await window.__onDisk('new.pkg.slp', 'minimal_instance.pkg.slp');
                window.__asked = 0;
                window.showOpenFilePicker = async () => [window.__opened];
                window.showSaveFilePicker = async () => { window.__asked++; return window.__newFile; };
                document.getElementById('menuLoadSlp').click();
                const s = window.__lucid.state;
                for (let i = 0; i < 300 && !(s.session && s.views.length === 2); i++) await new Promise(res => setTimeout(res, 50));
                await new Promise(res => setTimeout(res, 300));
                const handle = s.slpFileHandle;
                const original = await window.__bytesOf(window.__opened);
                const saved = await window.__save();
                const after = await window.__bytesOf(window.__opened);
                return {
                    handle: handle ? handle.name : null, leftOut: s.embeddedImagesLeftOut, asked: window.__asked, ...saved,
                    same: original.length === after.length && original.every((b, i) => b === after[i]),
                    shown: await window.__show(1),
                };
            });
            check(r.leftOut === 1 && r.handle === null, 'a video in no session: that file is not the save target', r);
            check(r.asked === 1 && r.same, 'Save asks where, and it is left as it was', r);
            check(/Not in this file: 1 video\(s\)/.test(r.status), 'the status says what is not in the new file', r.status);
            check(eq(r.shown, { north: 100, south: 101 }), 'the project keeps its own images', r.shown);
            await keep(page, 'window.__newFile', 'no_session.slp', '-');
            await page.close();
        }

        console.log('  -- a file that does not open');
        {
            const page = await handlePage();
            const r = await page.evaluate(async () => {
                const root = await navigator.storage.getDirectory();
                const broken = await root.getFileHandle('broken.slp', { create: true });
                const w = await broken.createWritable();
                await w.write(new Uint8Array(4096).fill(7));
                await w.close();
                window.showOpenFilePicker = async () => [broken];
                document.getElementById('menuLoadSlp').click();
                for (let i = 0; i < 100 && !/error|failed/i.test(document.getElementById('statusText').textContent); i++) await new Promise(res => setTimeout(res, 50));
                await new Promise(res => setTimeout(res, 100));
                return { handle: !!window.__lucid.state.slpFileHandle, status: document.getElementById('statusText').textContent };
            });
            check(!r.handle, 'a file that cannot be read is not the save target', r);

            // One that reads, and fails part way through opening, which logs that it did.
            page.removeAllListeners('console');
            const partWay = await page.evaluate(async () => {
                const handle = await window.__onDisk('fails.pkg.slp', 'out_of_order_embedded.pkg.slp');
                window.showOpenFilePicker = async () => [handle];
                const RealWorker = window.Worker;
                window.Worker = function (url, options) {
                    if (String(url).includes('frame-worker')) throw new Error('no frame worker');
                    return new RealWorker(url, options);
                };
                document.getElementById('menuLoadSlp').click();
                const status = () => document.getElementById('statusText').textContent;
                for (let i = 0; i < 200 && !/SLP load error/.test(status()); i++) await new Promise(res => setTimeout(res, 50));
                await new Promise(res => setTimeout(res, 100));
                window.Worker = RealWorker;
                return { handle: !!window.__lucid.state.slpFileHandle, status: status() };
            });
            check(/SLP load error/.test(partWay.status) && !partWay.handle, 'nor is one that fails part way through opening', partWay);
            await page.close();
        }

        console.log('  -- the download route, and a temporary file in memory');
        {
            const page = await handlePage();
            await page.evaluate(async () => {
                window.__opened = await window.__onDisk('opened.pkg.slp', 'out_of_order_embedded.pkg.slp');
                await window.__reopen(window.__opened);
                window.__lucidSaveScratch = 'memory';
                window.showSaveFilePicker = undefined;
                window.__lucid.state.slpFileHandle = null;
                window.__worker = window.__lucid.state.views[0].decoder._worker;
            });
            const before = await page.evaluate(() => window.__showFresh([5, 7, 3]));
            const downloading = page.waitForEvent('download');
            await page.evaluate(async () => { (await import('/import-export/save-load.js')).saveProjectSlp(); });
            const download = await downloading;
            const to = path.join(savedDir, 'downloaded.slp');
            await download.saveAs(to);
            check(/\.pkg\.slp$/.test(download.suggestedFilename()), 'a download keeps SLEAP\'s name for it', download.suggestedFilename());
            await page.waitForFunction(() => !window.__lucid.state.isSaving);
            const s = await page.evaluate(() => ({ status: document.getElementById('statusText').textContent, same: window.__lucid.state.views[0].decoder._worker === window.__worker }));
            check(/^Project saved as SLP/.test(s.status) && s.same, 'with no file handle: downloaded, and the project still reads the file it opened', s);
            check(eq(await page.evaluate(() => window.__showFresh([5, 7, 3])), before), 'which it goes on showing');
            await page.setInputFiles('#__pick', to);
            await page.evaluate(async () => {
                const { handleLoadSlpFile } = await import('/import-export/slp-import.js');
                await handleLoadSlpFile(window.__pickedFile());
                await new Promise(r => setTimeout(r, 100));
            });
            check(eq(await page.evaluate(() => window.__showFresh([5, 7, 3])), before), 'the downloaded file, built in memory, has the images');
            pythonChecks.push(['downloaded', 'out_of_order_embedded.pkg.slp', to, []]);
            await page.close();
        }

        console.log('  -- the progress card');
        {
            const page = await handlePage();
            const states = await page.evaluate(async () => {
                window.__target = await window.__onDisk('t.pkg.slp', 'two_sessions.pkg.slp');
                await window.__reopen(window.__target);
                window.__lucid.state.slpFileHandle = window.__target;
                const seen = [];
                const text = (id) => { const e = document.getElementById(id); return e && !e.classList.contains('hidden') ? e.textContent : ''; };
                const observer = new MutationObserver(() => {
                    const now = text('loadingStatus') + ' | ' + text('loadingSubstatus');
                    if (seen[seen.length - 1] !== now) seen.push(now);
                });
                observer.observe(document.getElementById('loadingOverlay'), { subtree: true, childList: true, characterData: true, attributes: true });
                await window.__save();
                observer.disconnect();
                return seen;
            });
            check(states.some(t => /^Writing the file… \| Step 1 of 3/.test(t)), 'the labels are step 1 of 3', states);
            check(states.some(t => /^Copying images: 6\/6 images… \| Step 2 of 3/.test(t)), 'the images are step 2, counted', states);
            check(states.some(t => /^Finishing the file… \| Step 3 of 3/.test(t) || /^Writing to disk: .* \| Step 3 of 3/.test(t)), 'and finishing the file step 3', states);
            await page.close();
        }

        console.log('  -- the streaming writer');
        {
            const page = await handlePage();
            const r = await page.evaluate(async () => {
                window.__target = await window.__onDisk('stream.pkg.slp', 'out_of_order_embedded.pkg.slp');
                await window.__reopen(window.__target);
                const s = window.__lucid.state;
                s.slpFileHandle = window.__target;
                // Saved once, the file has a session saying which camera each video is on.
                await window.__save();
                await window.__reopen(window.__target);
                s.slpFileHandle = window.__target;
                // A session whose 2D comes from a store, with cameras whose images are embedded.
                const lazyMod = await import('/loading/sio-lazy-loader.js');
                const SioLazyLoader = lazyMod.SioLazyLoader || lazyMod.default;
                const loader = new SioLazyLoader();
                await loader.openProjectSlp(await window.__target.getFile());
                s.session.lazyLoader = loader;
                const shownBefore = await window.__showFresh([5, 7, 3]);
                let streamed = false;
                const log = console.log;
                console.log = (...a) => { if (/streaming writer/.test(String(a[0]))) streamed = true; log(...a); };
                const saved = await window.__save();
                console.log = log;
                const after = await window.__showFresh([5, 7, 3]);
                return { ...saved, shownBefore, after, streamed };
            });
            check(r.streamed, 'a session with a store goes through it', r);
            check(/^Saved \(/.test(r.status), 'saves through it', r);
            check(eq(r.after, r.shownBefore), 'and the project goes on showing its images', r);
            await page.evaluate(() => window.__reopen(window.__target));
            check(eq(await page.evaluate(() => window.__showFresh([5, 7, 3])), r.shownBefore), 'reopened: the images are in the file');
            await keep(page, 'window.__target', 'streamed.slp', 'out_of_order_embedded.pkg.slp');
            await page.close();
        }

        console.log('  -- Python sleap-io 0.9.5 reads what was saved');
        const uv = spawnSync('uv', ['--version'], { encoding: 'utf8' });
        if (uv.status !== 0) {
            console.log('  (skipped: no uv to run Python with)');
        } else {
            for (const [what, source, saved, extra] of pythonChecks) {
                const sourcePath = source === '-' ? '-' : fixture(source);
                const run = spawnSync('uv', ['run', '--no-project', '--python', '3.12', '--with', 'sleap-io==0.9.5', '--with', 'h5py', '--with', 'numpy',
                    'python', '-I', path.join(FIXTURES, 'compare_saved.py'), sourcePath, saved, ...extra], { encoding: 'utf8', timeout: 180000 });
                let out = null;
                try { out = JSON.parse(run.stdout.trim().split('\n').pop()); } catch (e) { /* reported below */ }
                check(out && out.ok, `${what}: opens, with the same images and poses`, out || (run.stderr || '').slice(-600));
                if (what === 'layout_flat_no_sizes.pkg.slp' && out) check(eq(out.holds, [{ 2: 90, 5: 91, 9: 92 }]), 'a layout Python could not read in the source, it can now', out.holds);
                if (what === 'no_session.slp' && out) check(eq(out.holds, [{ 1: 100 }, { 1: 101 }]), 'the session\'s two videos', out.holds);
            }
        }
        fs.rmSync(savedDir, { recursive: true, force: true });
    }

    console.log('\n9. File ▸ Open above 200 MB');
    {
        const page = await freshPage();
        await page.setInputFiles('#__pick', fixture('two_sessions.pkg.slp'));
        const r = await page.evaluate(async () => {
            const big = window.__pickedFile();
            Object.defineProperty(big, 'size', { get: () => 300 * 1024 * 1024 });
            const { handleLoadProject } = await import('/import-export/save-load.js');
            // The lazy route would stop here, asking for the videos.
            const opened = await Promise.race([
                handleLoadProject(big),
                new Promise(res => setTimeout(() => res(false), 30000)),
            ]);
            return { opened: opened === false ? 'false' : 'ok', ...window.__summary() };
        });
        await page.waitForTimeout(100);
        check(r.opened === 'ok' && !r.lazy, 'a package stays on the path that shows its images', r);
        check(r.sessions === 2 && eq(r.views, ['north', 'south']), 'with every session and its views', r);
        if (r.views.length) check(eq(await page.evaluate(() => window.__show()), { north: 31, south: 32 }), 'and its images');
        const none = await page.evaluate(async () => {
            const { slpHasEmbeddedImages } = await import('/import-export/slp-import.js');
            const bytes = await (await fetch('/tests/fixtures/slp-compound/sleap-compound-small.slp')).arrayBuffer();
            return {
                plain: await slpHasEmbeddedImages(new File([bytes], 'plain.slp')),
                broken: await slpHasEmbeddedImages(new File([new Uint8Array(64)], 'broken.slp')),
            };
        });
        check(none.plain === false, 'a file without images still takes the lazy path', none);
        check(none.broken === false, 'and one that cannot be read is left to that path to report', none);
        await page.close();
    }

    console.log('\n10. Load SLP with a project open');
    {
        const page = await freshPage();
        await open(page, 'out_of_order_embedded.pkg.slp');
        const chooser = page.waitForEvent('filechooser');
        await page.evaluate(() => { window.__adding = import('/import-export/slp-import.js').then(m => m.handleAddSlp()); });
        await (await chooser).setFiles(fixture('out_of_order_embedded.pkg.slp'));
        await page.evaluate(() => window.__adding);
        await page.waitForTimeout(100);
        const s = await page.evaluate(() => window.__summary());
        check(eq(s.views, ['clip_a', 'clip_b', 'clip_c', 'clip_a_2', 'clip_b_2', 'clip_c_2']), 'adds its videos, renamed beside the open ones', s.views);
        check(s.workers === 2, 'each file has its worker', s.workers);
        // As the view strip does when one is clicked: an added view starts undocked.
        await page.evaluate(async () => {
            const { paneManager } = await import('/ui/app-state.js');
            for (const name of ['clip_a_2', 'clip_b_2', 'clip_c_2']) paneManager.addVideoPanel(name);
        });
        await page.waitForTimeout(200);
        check(eq(await page.evaluate(() => window.__show(7)), { clip_a: null, clip_b: 40, clip_c: null, clip_a_2: null, clip_b_2: 40, clip_c_2: null }), 'each on its own camera');
        check(eq(await page.evaluate(() => window.__show(11)), { clip_a: 21, clip_b: null, clip_c: null, clip_a_2: 21, clip_b_2: null, clip_c_2: null }), 'at every frame');
        await page.close();
    }

    console.log('\n11. Opening another project');
    {
        const page = await freshPage();
        await page.evaluate(() => {
            window.__terminated = 0;
            const t = Worker.prototype.terminate;
            Worker.prototype.terminate = function () { window.__terminated++; return t.call(this); };
        });
        await open(page, 'out_of_order_embedded.pkg.slp');
        await page.evaluate(() => { window.__oldDecoder = window.__lucid.state.views[0].decoder; window.__terminated = 0; });
        const s = await open(page, 'minimal_instance.pkg.slp');
        const r = await page.evaluate(() => ({ closed: window.__oldDecoder._closed, cached: window.__oldDecoder.cache.size, terminated: window.__terminated }));
        check(r.closed && r.cached === 0, 'the previous decoders are closed, without their frames', r);
        check(r.terminated >= 1, 'its worker is stopped', r);
        check(s.workers === 1 && eq(s.views, ['centered_pair_low_quality']), 'only the new project is left', s);

        const closesOn = async (how, replace) => {
            await open(page, 'out_of_order_embedded.pkg.slp');
            await page.evaluate(() => { window.__oldDecoder = window.__lucid.state.views[0].decoder; window.__terminated = 0; });
            await page.evaluate(replace);
            await page.waitForFunction(() => window.__oldDecoder._closed, null, { timeout: 30000 }).catch(() => {});
            const c = await page.evaluate(() => ({
                closed: window.__oldDecoder._closed,
                terminated: window.__terminated,
                workers: window.__lucid.state.frameWorkers.length,
            }));
            check(c.closed && c.terminated >= 1 && c.workers === 0, `${how}: its images are closed too`, c);
        };
        await closesOn('File ▸ New', async () => {
            (await import('/import-export/save-load.js')).newProject(true);
        });
        await closesOn('deleting the last session', async () => {
            (await import('/ui/sessions-panes.js')).removeSession(0);
        });
        await closesOn('the lazy route', async () => {
            const bytes = await (await fetch('/tests/fixtures/slp-compound/sleap-compound-small.slp')).arrayBuffer();
            // Not awaited: it goes on to ask for the videos, which nothing here answers.
            import('/loading/session-loader.js').then(m => m.handleLoadProjectSlpLazy(new File([bytes], 'plain.slp')));
        });
        await page.close();
    }

    console.log('\n12. Playing, and the view strip');
    {
        const page = await freshPage();
        const s = await open(page, 'out_of_order_embedded.pkg.slp');
        check(s.current === 5, 'starts on frame 5', s.current);
        const played = await page.evaluate(async () => {
            const vc = (await import('/ui/app-state.js')).videoController;
            const state = window.__lucid.state;
            const seen = new Set();
            vc.startPlayback();
            const t0 = performance.now();
            while (performance.now() - t0 < 4000 && state.isPlaying) {
                seen.add(state.currentFrame);
                await new Promise(r => requestAnimationFrame(r));
            }
            const stillPlaying = state.isPlaying;
            if (stillPlaying) vc.stopPlayback();
            const shown = await window.__show();
            // Stopping partway: the same loop on the longest fixture here is not long
            // enough to catch, so stop it from a known place on the way.
            vc.seekToFrame(0);
            await new Promise(r => setTimeout(r, 50));
            vc.startPlayback();
            await new Promise(r => setTimeout(r, 100));
            vc.stopPlayback();
            const stoppedAt = state.currentFrame;
            await new Promise(r => setTimeout(r, 200));
            return { seen: Array.from(seen), stillPlaying, last: stoppedAt === undefined ? null : shown, stoppedAt, after: state.currentFrame, isPlaying: state.isPlaying };
        });
        check(played.seen.length > 2 && !played.stillPlaying, 'playing moves through the frames, and stops at the end', played);
        check(played.last && played.last.clip_a === 21, 'it ends on the last frame\'s image', played.last);
        check(played.after === played.stoppedAt && !played.isPlaying, 'stopping stops it where it is', played);
        // The real button: it pauses through a different path from the stop above.
        await page.evaluate(() => window.__show(0));
        await page.waitForFunction(() => !window.__lucid.state.isPlaying);
        await page.click('#btnPlay');
        await page.waitForFunction(() => window.__lucid.state.isPlaying, null, { timeout: 5000 });
        await page.click('#btnPlay', { timeout: 5000 }).catch(() => {});
        await page.waitForFunction(() => !window.__lucid.state.isPlaying, null, { timeout: 5000 });
        const paused = await page.evaluate(async () => {
            const f = window.__lucid.state.currentFrame;
            await new Promise(r => setTimeout(r, 200));
            return { before: f, after: window.__lucid.state.currentFrame };
        });
        check(paused.before === paused.after, 'the Play button starts it and pauses it', paused);
        const thumbs = await page.evaluate(() => Array.from(document.querySelectorAll('.view-strip-item')).map(item => {
            const c = item.querySelector('.strip-thumb canvas');
            const d = c.getContext('2d').getImageData(Math.floor(c.width / 2), Math.floor(c.height / 2), 1, 1).data;
            return [item.getAttribute('data-view-name'), d[3] === 0 ? null : d[0]];
        }));
        check(eq(thumbs, [['clip_a', 20], ['clip_b', 40], ['clip_c', 60]]), 'each thumbnail shows that camera\'s first image', thumbs);
        await page.close();
    }

    console.log('\n13. Colour images');
    {
        const page = await freshPage();
        const colors = () => page.evaluate(async () => {
            const out = {};
            for (const v of window.__lucid.state.views) out[v.name] = Array.from(v.ctx.getImageData(0, 0, 1, 1).data).slice(0, 3);
            return out;
        });
        // Every PNG holds (10, 20, 30): `make_fixture.py` checks Python's reading of them.
        await open(page, 'colors.pkg.slp');
        check(eq(await colors(), { stored_bgr: [30, 20, 10], stored_rgb: [10, 20, 30] }), 'a video stored BGR is swapped, one stored RGB is not', await colors());
        const old = await open(page, 'colors_old_format.pkg.slp');
        const shown = await colors();
        check(eq(shown[old.views[0]], [30, 20, 10]), 'a format 1.2 file saying nothing is BGR', shown);
        await page.close();
    }

    console.log('\n14. Uncompressed pixels');
    {
        const page = await freshPage();
        const s = await open(page, 'raw_pixels.pkg.slp');
        check(/^SLP loaded/.test(s.status) && s.current === 4, 'loads, on its image', s);
        const px = await page.evaluate(() => Object.fromEntries(window.__lucid.state.views.map(v => [v.name, {
            size: [v.videoWidth, v.videoHeight],
            first: Array.from(v.ctx.getImageData(0, 0, 1, 1).data),
            last: Array.from(v.ctx.getImageData(1, 1, 1, 1).data),
        }])));
        check(eq(px.grey.size, [2, 2]), 'its size comes from the dataset', px.grey);
        check(eq(px.grey.first, [77, 77, 77, 255]), 'a grey video shows its pixels', px.grey);
        check(eq(px.grey.last, [0, 0, 0, 255]), 'and a dark last pixel is a pixel, not padding', px.grey);
        check(eq(px.colour.first, [10, 20, 30, 255]), 'a colour one shows as stored, as Python reads it', px.colour);
        await page.close();
    }
} finally {
    await browser.close();
    server.kill();
}

console.log(fails ? `\nFAIL — ${fails} failure(s)` : '\nPASS');
process.exit(fails ? 1 : 0);
