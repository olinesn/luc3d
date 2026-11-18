/**
 * lazy-open-plain-slp.mjs — a `.slp` with NO recording session (a plain SLEAP
 * prediction file) over 200 MB opens through the lazy route with one camera per
 * video and every pose, and a save loses nothing.
 *
 * File ▸ Open sends every `.slp` over 200 MB to `handleLoadProjectSlpLazy`, which
 * was written for LUC3D's own project files. On a plain file it failed with
 * "Lazy reopen error: Cannot read properties of null (reading 'frameGroups')",
 * and past that error `SioLazyLoader.openProjectSlp` had no camera-to-video map,
 * so the project would have been empty. The eager route is not the answer for a
 * big file: one real camera's predictions take 1.5 GB of the renderer's ~4 GB
 * JS heap there, and eight crash the tab (MODULES.md `loading/session-loader.js`).
 *
 * The fixtures are built in the page with sleap-io.js's typed API, with no
 * session: a user instance with an occluded node and a missing node, predictions
 * with their own per-node scores, trackless instances, frames that do not start
 * at 0, and a video with no labeled frames.
 *
 * Cases:
 *   A. File ▸ Open's route, the file made to look large: the lazy path. Same
 *      cameras as the eager path, every pose there, the same poses frame for
 *      frame (points, occlusion, type, track, score), most frames not resident,
 *      landing on the first labeled frame.
 *      The Tracks timeline shows each track over its whole run on both paths.
 *   B. Real videos attach: one view per video, including the one with no frames.
 *   C. File ▸ Save, with nothing visited, then after visiting every frame,
 *      then after an edit: the file holds every pose by value (x, y, visible,
 *      score, type, track, video, frame), and the edit. An export per camera
 *      holds that camera's poses and video alone.
 *   D. That save reopens through both loaders with the same poses, and saves
 *      the same again.
 *   E. Two videos with one stem (campy's `<camera>/0.mp4`): a camera each, on
 *      both paths, and each keeps its own poses through a save.
 *   F. Files the lazy store cannot hold go to the eager path instead, with a
 *      warning: two rows for one (video, frame), a session linking no camera to
 *      a video, two skeletons, and embedded images.
 *   G. A calibration loaded afterwards, which is how such a file gets 3D. Its
 *      cameras are named differently from the videos: with videos attached and
 *      without, the poses follow onto the new names, and a save keeps them all.
 *      A calibration that names none of them: a save refuses, and says why,
 *      rather than write a file with no poses over the one that has them.
 *
 * Run: node tests/e2e/lazy-open-plain-slp.mjs
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8249);
const FRAMES = Number(process.env.FRAMES || 2400);

let fails = 0;
const check = (c, m, extra) => {
    console.log((c ? '  ✓ ' : '  ✗ ') + m + (extra !== undefined ? '  ' + JSON.stringify(extra) : ''));
    if (!c) fails++;
};

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1200));

const fixtureDir = fs.mkdtempSync(path.join(os.tmpdir(), 'lazy-plain-'));
const fixture = (name) => path.join(fixtureDir, name + '.slp');
const VIDEOS = ['barcode-60', 'fragmented-60', 'bframes-test'];

let browser;
try {
    browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();
    const errs = [];
    page.on('pageerror', e => errs.push('pageerror: ' + String(e).slice(0, 300)));
    page.on('console', m => { if (m.type() === 'error') errs.push('console.error: ' + m.text().slice(0, 300)); });

    const sinks = new Map();
    await page.exposeFunction('__tstWrite', (name, b64) => {
        if (!sinks.has(name)) sinks.set(name, fs.openSync(fixture(name), 'w'));
        fs.writeSync(sinks.get(name), Buffer.from(b64, 'base64'));
    });

    await page.goto(`http://localhost:${PORT}/index.html`, { waitUntil: 'domcontentloaded', timeout: 120000 });
    await page.waitForFunction(() => window.SleapIO && window.h5wasm && window.__lucid, null, { timeout: 120000 });

    await page.evaluate(() => {
        window.__tstSend = async (name, bytes) => {
            let s = ''; const C = 0x8000;
            for (let o = 0; o < bytes.length; o += C) s += String.fromCharCode.apply(null, bytes.subarray(o, o + C));
            await window.__tstWrite(name, btoa(s));
        };
    });
    const closeFixtures = () => {
        for (const fd of sinks.values()) fs.closeSync(fd);
        sinks.clear();
    };

    // ---- fixtures, and what each holds ----
    const made = await page.evaluate(async ({ FRAMES, VIDEOS }) => {
        const S = window.SleapIO;
        const send = window.__tstSend;
        const nodes = ['n0', 'n1', 'n2'].map(n => new S.Node(n));
        const skeleton = new S.Skeleton({ nodes, edges: [new S.Edge(nodes[0], nodes[1]), new S.Edge(nodes[1], nodes[2])], name: 'plain' });
        const tracks = [new S.Track('a'), new S.Track('b')];
        const videoOf = (fn) => {
            const v = new S.Video({ filename: fn, backendMetadata: { type: 'MediaVideo', shape: [FRAMES + 60, 512, 640, 1], filename: fn }, openBackend: false });
            v.shape = [FRAMES + 60, 512, 640, 1];
            return v;
        };
        const pred = (v, f, k) => new S.PredictedInstance({
            points: [0, 1, 2].map(j => ({
                xy: [100 * v + 10 * k + j + 0.25, f / 8 + j + 0.5],
                // Every 7th frame the model calls node 1 not visible, at a real position.
                visible: !(f % 7 === 0 && j === 1),
                complete: true,
                score: 0.5 + 0.01 * j + 0.1 * k,
            })),
            skeleton, track: k < 2 ? tracks[k] : null, score: 0.75 + 0.1 * k,
        });
        // A hand label: node 1 occluded at a real position, node 2 missing.
        const user = (v, f) => new S.Instance({
            points: [
                { xy: [300 + v, 40 + f / 8], visible: true, complete: true },
                { xy: [310 + v, 50 + f / 8], visible: false, complete: true },
                { xy: [NaN, NaN], visible: false, complete: false },
            ],
            skeleton, track: tracks[0],
        });
        const isUserFrame = (f) => f % 400 === 5;
        // Video v's frames: FIRST + v * 7 .. on, video 2 has none.
        const FIRST = 5;
        const framesOf = (videos, nVideosWithFrames, predictionsOnly) => {
            const lfs = [];
            for (let v = 0; v < nVideosWithFrames; v++) {
                for (let f = FIRST + v * 7; f < FIRST + v * 7 + FRAMES; f++) {
                    const insts = [pred(v, f, 0), pred(v, f, 1)];
                    if (f % 3 === 0) insts.push(pred(v, f, 2));   // trackless
                    if (isUserFrame(f) && !predictionsOnly) insts.push(user(v, f));
                    lfs.push(new S.LabeledFrame({ video: videos[v], frameIdx: f, instances: insts }));
                }
            }
            return lfs;
        };
        const save = (lfs, videos) => S.saveSlpToBytes(new S.Labels({
            labeledFrames: lfs, videos, skeletons: [skeleton], tracks, sessions: [], provenance: {},
        }));

        const v3 = VIDEOS.map(n => videoOf('/data/recording/' + n + '.mp4'));
        await send('plain', await save(framesOf(v3, 2), v3));

        // Nothing but predictions, as a real prediction file holds.
        const vp = ['/data/one.mp4', '/data/two.mp4'].map(videoOf);
        await send('predonly', await save(framesOf(vp, 2, true).filter(lf => lf.frameIdx < 200), vp));

        // Two videos with one stem, as campy names them.
        const same = ['Camera0_topB/0.mp4', 'Camera1_topC/0.mp4'].map(videoOf);
        await send('samestem', await save(framesOf(same, 2).filter(lf => lf.frameIdx < 60), same));

        // Videos named as campy-and-SLEAP names them, for the calibration case.
        const longNames = ['Camera0_topB', 'Camera1_topC'].map(c => videoOf('/data/[20260713]-s#0066-' + c + '-05mice.mp4'));
        await send('longnames', await save(framesOf(longNames, 2).filter(lf => lf.frameIdx < 60), longNames));

        // Two rows for one (video, frame): the second row's instance is at x = 900.
        const dv = [videoOf('/data/dup.mp4')];
        const dupLfs = framesOf(dv, 1).filter(lf => lf.frameIdx < 40);
        dupLfs.push(new S.LabeledFrame({
            video: dv[0], frameIdx: 10,
            instances: [new S.PredictedInstance({
                points: [0, 1, 2].map(j => ({ xy: [900 + j, 900 + j], visible: true, complete: true, score: 0.9 })),
                skeleton, track: tracks[1], score: 0.9,
            })],
        }));
        await send('duprow', await save(dupLfs, dv));

        // Two skeletons.
        const sv = [videoOf('/data/twoskel.mp4')];
        const otherNodes = ['m0', 'm1', 'm2'].map(n => new S.Node(n));
        const other = new S.Skeleton({ nodes: otherNodes, edges: [new S.Edge(otherNodes[0], otherNodes[1])], name: 'other' });
        await send('twoskel', await S.saveSlpToBytes(new S.Labels({
            labeledFrames: [
                new S.LabeledFrame({ video: sv[0], frameIdx: 3, instances: [pred(0, 3, 0)] }),
                new S.LabeledFrame({ video: sv[0], frameIdx: 4, instances: [new S.PredictedInstance({
                    points: [0, 1, 2].map(j => ({ xy: [500 + j, 500 + j], visible: true, complete: true, score: 0.9 })),
                    skeleton: other, score: 0.9,
                })] }),
            ],
            videos: sv, skeletons: [skeleton, other], tracks, sessions: [], provenance: {},
        })));

        // Images in the file (the reader goes by the file name "."; the images
        // themselves are not what this case is about).
        const ev = new S.Video({ filename: '.', backendMetadata: { filename: '.', dataset: 'video0/video', type: 'HDF5Video' }, openBackend: false });
        await send('embedded', await save(framesOf([ev], 1).filter(lf => lf.frameIdx < 12), [ev]));

        // A session with a calibration that links no camera to a video.
        const cams = ['left', 'right'].map((name, i) => new S.Camera({
            name, matrix: [[900, 0, 320], [0, 900, 256], [0, 0, 1]], rvec: [0, 0.3 * i, 0], tvec: [0, 0, 1000],
            distortions: [0, 0, 0, 0, 0], size: [640, 512],
        }));
        const nv = ['/data/left.mp4', '/data/right.mp4'].map(videoOf);
        await send('nomap', await S.saveSlpToBytes(new S.Labels({
            labeledFrames: framesOf(nv, 2).filter(lf => lf.frameIdx < 30), videos: nv, skeletons: [skeleton], tracks,
            sessions: [new S.RecordingSession({ cameraGroup: new S.CameraGroup({ cameras: cams }) })], provenance: {},
        })));
        return { FIRST };
    }, { FRAMES, VIDEOS });
    closeFixtures();

    // Every file input the test uses, and the helpers the steps share.
    await page.evaluate(() => {
        const inp = document.createElement('input');
        inp.type = 'file'; inp.id = '__tstPick'; inp.style.cssText = 'position:fixed;left:-9999px';
        document.body.appendChild(inp);
        const vids = document.createElement('input');
        vids.type = 'file'; vids.id = '__tstVideos'; vids.multiple = true; vids.style.cssText = 'position:fixed;left:-9999px';
        document.body.appendChild(vids);

        /** Canonical text of one pose: everything a save must keep, by value. */
        const fmt = (n) => (Number.isFinite(n) ? String(Math.round(n * 1e4) / 1e4) : 'NaN');
        /** Every pose in a `.slp`'s bytes, one sorted line per instance. */
        window.__posesOf = async (bytes) => {
            const labels = await window.SleapIO.readSlpStreaming(new File([bytes], 'x.slp'), {
                openVideos: false, rawSessions: true,
                h5wasmUrl: new URL('lib/h5wasm/h5wasm.iife.js', document.baseURI).href,
            });
            const lines = [];
            const perVideo = {};
            for (const lf of labels.labeledFrames) {
                // By the video's place in the list: the eager path's save names
                // a video with none attached after its camera, not its old path.
                const vi = labels.videos.indexOf(lf.video);
                for (const inst of lf.instances) {
                    const isPred = inst instanceof window.SleapIO.PredictedInstance;
                    const pts = inst.points.map(p => [
                        fmt(p.xy[0]), fmt(p.xy[1]), p.visible ? 'v' : 'h', isPred ? fmt(p.score) : '',
                    ].join(','));
                    lines.push([vi, lf.frameIdx, isPred ? 'pred' : 'user', inst.track ? inst.track.name : '-',
                        isPred ? fmt(inst.score) : '', pts.join(' ')].join('|'));
                    perVideo[vi] = (perVideo[vi] || 0) + 1;
                }
            }
            lines.sort();
            return {
                lines, perVideo,
                videos: labels.videos.map(v => ({ filename: String(v.filename), shape: v.shape ? Array.from(v.shape) : null })),
                tracks: labels.tracks.map(t => t.name),
            };
        };
        /** File ▸ Save's bytes, through `quickSave` and a stand-in for the picked file. */
        window.__saveBytes = async () => {
            const { state } = window.__lucid;
            const chunks = [];
            state.slpFileHandle = {
                createWritable: async () => ({
                    write: async (c) => { chunks.push(new Uint8Array(c.buffer ? c.buffer.slice(c.byteOffset, c.byteOffset + c.byteLength) : c)); },
                    close: async () => {},
                    abort: async () => {},
                }),
            };
            await (await import('/import-export/save-load.js')).quickSave();
            state.slpFileHandle = null;
            const status = document.getElementById('statusText').textContent;
            if (!/^Saved/.test(status)) throw new Error('save: ' + status);
            const out = new Uint8Array(chunks.reduce((n, c) => n + c.length, 0));
            let o = 0;
            for (const c of chunks) { out.set(c, o); o += c.length; }
            return out;
        };
    });

    /**
     * Load the picked fixture: `lazy` calls that loader directly; `route` is
     * File ▸ Open's own route with the file made to look large (the app decides
     * on `file.size`; the reader only gets the real bytes); `eager` is File ▸
     * Load SLP. Dismisses the video prompt, as Esc or Skip would.
     */
    async function load(name, how) {
        await page.setInputFiles('#__tstPick', fixture(name));
        await page.evaluate((how) => {
            window.__tstLoad = { done: false, err: null };
            (async () => {
                try {
                    const f = document.getElementById('__tstPick').files[0];
                    if (how === 'lazy') await (await import('/loading/session-loader.js')).handleLoadProjectSlpLazy(f);
                    else if (how === 'route') {
                        Object.defineProperty(f, 'size', { value: 300 * 1024 * 1024 });
                        await (await import('/import-export/save-load.js')).handleLoadProject(f);
                    } else await (await import('/import-export/slp-import.js')).handleLoadSlpFile(f);
                    window.__tstLoad.done = true;
                } catch (e) { window.__tstLoad.err = String(e && e.stack || e).slice(0, 500); }
            })();
        }, how);
        for (let i = 0; i < 600; i++) {
            const s = await page.evaluate(() => {
                if (document.getElementById('videoSelectOverlay')) {
                    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
                    return 'esc';
                }
                return window.__tstLoad.done ? 'done' : (window.__tstLoad.err ? 'err' : 'wait');
            });
            if (s === 'done' || s === 'err') break;
            await new Promise(r => setTimeout(r, 100));
        }
        return await page.evaluate(() => ({
            err: window.__tstLoad.err,
            status: document.getElementById('statusText').textContent,
        }));
    }

    /** What the open session holds, without visiting a frame. */
    const summary = () => page.evaluate(async () => {
        const { state } = window.__lucid;
        const s = state.session;
        if (!s) return null;
        const pc = await import('/pose/placeholder-cameras.js');
        const perCam = {};
        let total = 0;
        if (s.lazyLoader) {
            s.lazyLoader.forEachInstanceRow((cam) => { perCam[cam] = (perCam[cam] || 0) + 1; total++; });
        } else {
            for (const [, fg] of s.frameGroups) {
                for (const [cam, l] of fg.instances) { perCam[cam] = (perCam[cam] || 0) + l.length; total += l.length; }
                for (const [cam, l] of fg.unlinkedInstances) { perCam[cam] = (perCam[cam] || 0) + l.length; total += l.length; }
            }
        }
        return {
            lazy: !!s.lazyLoader, cameras: s.cameras.map(c => c.name), perCam, total,
            calibrated: pc.hasRealCalibration(s), resident: s.frameGroups.size, tracks: s.tracks,
            totalFrames: state.totalFrames, currentFrame: state.currentFrame,
            groups: s.instanceGroups.size,
        };
    });

    /** The frames' instances as the app holds them, hydrating the lazy ones. */
    const framesAsLoaded = (frames) => page.evaluate(async (frames) => {
        const { state } = window.__lucid;
        const s = state.session;
        const tri = await import('/pose/triangulation.js');
        const out = {};
        for (const f of frames) {
            if (s.lazyLoader) await tri.ensureLazyFrameData(f);
            const fg = s.frameGroups.get(f);
            const row = {};
            if (fg) {
                for (const cam of s.cameras.map(c => c.name)) {
                    const insts = fg.getInstances(cam).concat(fg.getUnlinkedInstances(cam).map(u => u.instance));
                    if (insts.length === 0) continue;
                    row[cam] = insts.map(inst => {
                        const pts = [];
                        for (let k = 0; k < inst.numNodes; k++) {
                            pts.push(inst.hasPoint(k)
                                ? [inst.getX(k), inst.getY(k), inst.isOccluded(k) ? 'occ' : 'ok', inst.nulledNodes && inst.nulledNodes.has(k) ? 'nulled' : '']
                                : null);
                        }
                        return { type: inst.type, track: inst.trackIdx, score: inst.score, pts };
                    }).sort((a, b) => JSON.stringify(a) < JSON.stringify(b) ? -1 : 1);
                }
            }
            out[f] = row;
        }
        return out;
    }, frames);

    /** Read a fixture's poses as what the steps after it compare with. */
    const readSource = async (name) => {
        await page.setInputFiles('#__tstPick', fixture(name));
        return await page.evaluate(async () => {
            window.__source = await window.__posesOf(await document.getElementById('__tstPick').files[0].arrayBuffer());
            return { perVideo: window.__source.perVideo, n: window.__source.lines.length, videos: window.__source.videos,
                at900: window.__source.lines.filter(l => /\|900,900,v/.test(l)).length };
        });
    };
    const src = await readSource('plain');

    // ---------------------------------------------------------------- A
    console.log('\nA. File ▸ Open\'s route, a file with no session');
    let r = await load('plain', 'route');
    check(!r.err && !/error/i.test(r.status), 'opened', r);
    const lazySum = await summary();
    check(lazySum && lazySum.lazy, 'took the lazy path');
    check(JSON.stringify(lazySum.cameras) === JSON.stringify(VIDEOS), 'one camera per video, named from the videos', lazySum.cameras);
    check(!lazySum.calibrated, 'placeholder cameras: 3D stays off');
    check(lazySum.total === src.n, 'every pose is in the project', { project: lazySum.total, file: src.n });
    check(lazySum.resident < FRAMES / 4, 'most frames not resident', { resident: lazySum.resident });
    check(lazySum.totalFrames === made.FIRST + 7 + FRAMES, 'the frame span covers both videos', lazySum.totalFrames);
    check(lazySum.currentFrame === made.FIRST, 'lands on the first labeled frame, as the eager path does', lazySum.currentFrame);
    check(JSON.stringify(lazySum.tracks) === JSON.stringify(['a', 'b']), 'tracks', lazySum.tracks);

    const probe = [];
    for (let f = made.FIRST; f < made.FIRST + FRAMES + 7; f += 37) probe.push(f);
    probe.push(405, 805, 1205, FRAMES + made.FIRST - 1, FRAMES + made.FIRST + 6, 12);
    const lazyFrames = await framesAsLoaded(probe);

    r = await load('plain', 'eager');
    check(!r.err && !/error/i.test(r.status), 'the eager path opens it too', r);
    const eagerSum = await summary();
    check(!eagerSum.lazy, 'took the eager path');
    check(JSON.stringify(eagerSum.cameras) === JSON.stringify(lazySum.cameras), 'the same cameras on both paths', eagerSum.cameras);
    check(eagerSum.total === lazySum.total && JSON.stringify(eagerSum.perCam) === JSON.stringify(lazySum.perCam),
        'the same poses per camera on both paths', { eager: eagerSum.perCam, lazy: lazySum.perCam });
    check(eagerSum.totalFrames <= lazySum.totalFrames, 'frame span no longer than the lazy one', { eager: eagerSum.totalFrames, lazy: lazySum.totalFrames });
    const timelineRows = () => page.evaluate(() => {
        const s = window.__lucid.state.session, t = window.__lucid.timeline;
        const savedUp = s._uploadedCameras; s._uploadedCameras = undefined;
        t.setData(s); t.setDisplayMode('tracks');
        s._uploadedCameras = savedUp;
        return t._trackSegments.filter(r => r.treeRole !== 'empty' && r.segments && r.segments.length)
            .map(r => r.cameraName + '|' + r.trackName + '|' + r.segments.map(x => x.start + '-' + x.end).join(','));
    });
    /** The status bar's whole-project counts for one camera, once they have settled. */
    const counters = (cam) => page.evaluate(async (cam) => {
        const { state, interactionManager } = window.__lucid;
        if (interactionManager) interactionManager.lastInteractedView = cam;
        const rendering = await import('/ui/rendering.js');
        const read = () => ['statusLabeledFrames', 'statusInstances'].map(id => document.getElementById(id).textContent).join(' / ');
        let last = null;
        for (let i = 0; i < 60; i++) {
            rendering.updateFrameCounters();
            await new Promise(r => setTimeout(r, 100));
            const now = read();
            if (now === last && !/: 0\b/.test(now.split(' / ')[0])) break;
            last = now;
        }
        return read();
    }, cam);
    const eagerRows = await timelineRows();
    const eagerCounters = await counters(VIDEOS[0]);
    const eagerFrames = await framesAsLoaded(probe);
    const frameDiffs = probe.filter(f => JSON.stringify(lazyFrames[f]) !== JSON.stringify(eagerFrames[f]));
    check(frameDiffs.length === 0, `the same instances frame by frame on both paths (${probe.length} frames: points, occlusion, type, track, score)`,
        frameDiffs.length ? { frames: frameDiffs.slice(0, 5), lazy: lazyFrames[frameDiffs[0]], eager: eagerFrames[frameDiffs[0]] } : undefined);

    await load('plain', 'lazy');
    const lazyRows = await timelineRows();
    check(lazyRows.length > 0 && JSON.stringify(lazyRows) === JSON.stringify(eagerRows),
        'the Tracks timeline shows each track over its whole run, as on the eager path', { lazy: lazyRows.slice(0, 2), eager: eagerRows.slice(0, 2) });
    const lazyCounters = await counters(VIDEOS[0]);
    // These count hand labels, nearly all of them on frames not resident.
    check(/Labeled Frames: [1-9]/.test(lazyCounters) && lazyCounters === eagerCounters,
        'the status bar counts the whole project, as on the eager path', { lazy: lazyCounters, eager: eagerCounters });

    // ---------------------------------------------------------------- B
    console.log('\nB. Real videos attach: one view per video');
    await load('plain', 'lazy');
    await page.setInputFiles('#__tstVideos', VIDEOS.map(n => path.join(repoRoot, 'tests', 'fixtures', n, n + '.mp4')));
    const attached = await page.evaluate(async () => {
        const { state } = window.__lucid;
        const sl = await import('/loading/session-loader.js');
        const files = Array.from(document.getElementById('__tstVideos').files);
        const ok = await sl.attachVideosForLazyReopen(state.session, state.session.lazyLoader, files);
        return { ok, views: state.views.map(v => v.name), assigned: state.videoFiles.map(v => [v.name, v.assignedCamera]) };
    });
    check(attached.ok, 'videos attached');
    check(JSON.stringify(attached.views.slice().sort()) === JSON.stringify(VIDEOS.slice().sort()), 'one view per video, the one with no frames too', attached);
    check(attached.assigned.every(([n, c]) => n === c), 'each video on its own camera', attached.assigned);

    // ---------------------------------------------------------------- C
    console.log('\nC. File ▸ Save');
    const saveAndCompare = async (label, want) => {
        const res = await page.evaluate(async () => {
            const bytes = await window.__saveBytes();
            window.__lastSave = bytes;
            const got = await window.__posesOf(bytes);
            return {
                n: got.lines.length, perVideo: got.perVideo, videos: got.videos, tracks: got.tracks,
                sameAsSource: (() => {
                    const a = got.lines, b = window.__source.lines;
                    if (a.length !== b.length) return { lengths: [a.length, b.length] };
                    let diff = 0, first = null;
                    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) { diff++; if (first == null) first = [a[i], b[i]]; }
                    return diff === 0 ? null : { diff, first };
                })(),
            };
        });
        check(res.n === want.n, label + ': every pose saved', { saved: res.n, want: want.n });
        check(JSON.stringify(res.perVideo) === JSON.stringify(src.perVideo), label + ': each on its own video', res.perVideo);
        check(res.sameAsSource === null, label + ': the same poses, by value', res.sameAsSource);
        return res;
    };

    await load('plain', 'lazy');
    const c1 = await saveAndCompare('nothing visited', { n: src.n });
    const srcVideos = src.videos.map(v => v.filename);
    check(JSON.stringify(c1.videos.map(v => v.filename)) === JSON.stringify(srcVideos), 'every video keeps its file name, the one with no frames too', c1.videos.map(v => v.filename));
    check(JSON.stringify(c1.videos.map(v => v.shape)) === JSON.stringify(src.videos.map(v => v.shape)), 'and its shape', c1.videos.map(v => v.shape));
    await page.evaluate(() => { window.__savedOnce = window.__lastSave; });

    await load('plain', 'lazy');
    const everyFrame = [];
    for (let f = made.FIRST; f < made.FIRST + FRAMES + 7; f++) everyFrame.push(f);
    await page.evaluate(async (frames) => {
        const tri = await import('/pose/triangulation.js');
        for (const f of frames) await tri.ensureLazyFrameData(f);
    }, everyFrame);
    await saveAndCompare('every frame visited', { n: src.n });

    await load('plain', 'lazy');
    const edited = await page.evaluate(async (F) => {
        const { state } = window.__lucid;
        const tri = await import('/pose/triangulation.js');
        await tri.ensureLazyFrameData(F);
        const fg = state.session.frameGroups.get(F);
        const ul = fg.getUnlinkedInstances('barcode-60').find(u => u.instance.type === 'user');
        ul.instance.setPoint(0, 123.5, 77.25);
        return true;
    }, 405);
    check(edited, 'moved a node of a hand label on frame 405');
    const editRes = await page.evaluate(async () => {
        const bytes = await window.__saveBytes();
        const got = await window.__posesOf(bytes);
        const onlyInSource = window.__source.lines.filter(l => !got.lines.includes(l));
        const onlyInSave = got.lines.filter(l => !window.__source.lines.includes(l));
        return { n: got.lines.length, onlyInSource, onlyInSave };
    });
    check(editRes.n === src.n, 'still every pose', editRes.n);
    check(editRes.onlyInSave.length === 1 && /^0\|405\|user\|a\|\|123\.5,77\.25,v/.test(editRes.onlyInSave[0]),
        'the edit is saved', editRes.onlyInSave);
    check(editRes.onlyInSource.length === 1 && /^0\|405\|user\|/.test(editRes.onlyInSource[0]),
        'and it is the only pose that changed', editRes.onlyInSource);

    // File ▸ Export, one file per camera: with a hand label resident, and with
    // nothing but predictions, as a real prediction file holds.
    const exportPerCamera = (cams) => page.evaluate(async (cams) => {
        const { state } = window.__lucid;
        const fileio = await import('/import-export/file-io.js');
        const out = {};
        for (const cam of cams) {
            const blob = await fileio.exportSlpClientSide(state.session, cam, false, { name: cam }, cam + '.slp',
                { user: true, predicted: true, reprojected: false });
            const got = await window.__posesOf(await blob.arrayBuffer());
            out[cam] = { n: got.lines.length, videos: got.videos.length };
        }
        return out;
    }, cams);
    await load('plain', 'lazy');
    const byCam = await exportPerCamera(VIDEOS.slice(0, 2));
    check(byCam[VIDEOS[0]].n === src.perVideo[0] && byCam[VIDEOS[1]].n === src.perVideo[1],
        'an export per camera holds that camera\'s poses, no other\'s', { byCam, want: src.perVideo });
    check(byCam[VIDEOS[0]].videos === 1 && byCam[VIDEOS[1]].videos === 1, 'and that camera\'s video alone', byCam);
    const predSrc = await readSource('predonly');
    await load('predonly', 'lazy');
    const predByCam = await exportPerCamera(['one', 'two']);
    check(predByCam.one.n === predSrc.perVideo[0] && predByCam.two.n === predSrc.perVideo[1] && predByCam.one.videos === 1 && predByCam.two.videos === 1,
        'predictions only: the same', { predByCam, want: predSrc.perVideo });
    await readSource('plain');

    // ---------------------------------------------------------------- D
    console.log('\nD. That save reopens, through both loaders, and saves the same again');
    await page.evaluate(() => window.__tstSend('plain-saved', window.__savedOnce));
    closeFixtures();
    for (const how of ['lazy', 'eager']) {
        await load('plain-saved', how);
        const s = await summary();
        check(s && s.lazy === (how === 'lazy'), how + ': opened on that path');
        check(s && JSON.stringify(s.cameras) === JSON.stringify(VIDEOS), how + ': the same cameras', s && s.cameras);
        check(s && s.total === src.n, how + ': every pose', s && s.total);
        if (how === 'lazy') {
            const reopenedRows = await timelineRows();
            check(JSON.stringify(reopenedRows) === JSON.stringify(eagerRows), how + ': the Tracks timeline still shows whole runs, with nothing grouped', reopenedRows.slice(0, 2));
        }
        await saveAndCompare(how + ', saved again', { n: src.n });
    }

    // ---------------------------------------------------------------- E
    console.log('\nE. Two videos with one stem');
    const sameSrc = await readSource('samestem');
    for (const how of ['lazy', 'eager']) {
        await load('samestem', how);
        const s = await summary();
        check(s && JSON.stringify(s.cameras) === JSON.stringify(['Camera0_topB', 'Camera1_topC']), how + ': a camera each, named from the folders', s && s.cameras);
        check(s && s.total === sameSrc.n && Object.keys(s.perCam).length === 2, how + ': every pose, on two cameras', s && s.perCam);
        const res = await page.evaluate(async () => {
            const got = await window.__posesOf(await window.__saveBytes());
            return { perVideo: got.perVideo, n: got.lines.length, same: JSON.stringify(got.lines) === JSON.stringify(window.__source.lines) };
        });
        check(res.n === sameSrc.n && JSON.stringify(res.perVideo) === JSON.stringify(sameSrc.perVideo), how + ': a save keeps each on its own video', res.perVideo);
        check(res.same, how + ': the same poses, by value');
    }

    // ---------------------------------------------------------------- F
    console.log('\nF. What the lazy store cannot hold goes to the eager path, with a warning');
    const dupSrc = await readSource('duprow');
    check(dupSrc.at900 === 1, 'the fixture really holds two rows for one frame', dupSrc);
    r = await load('duprow', 'route');
    const dupSum = await summary();
    check(dupSum && !dupSum.lazy, 'two rows for one (video, frame): the eager path', dupSum && dupSum.lazy);
    check(dupSum && dupSum.total === dupSrc.n, 'every pose, both rows', dupSum && dupSum.total);
    check(/in full, not on demand: it lists the same video frame twice/.test(r.status), 'the status says so', r.status);
    const dupSave = await page.evaluate(async () => {
        const got = await window.__posesOf(await window.__saveBytes());
        return { n: got.lines.length, at900: got.lines.filter(l => /\|900,900,v/.test(l)).length };
    });
    check(dupSave.n === dupSrc.n && dupSave.at900 === 1, 'a save keeps both rows\' poses', dupSave);

    r = await load('nomap', 'route');
    const noMap = await summary();
    check(noMap && !noMap.lazy, 'a session that links no camera to a video: the eager path', noMap && noMap.lazy);
    check(noMap && JSON.stringify(noMap.cameras) === JSON.stringify(['left', 'right']), 'with the session\'s cameras', noMap && noMap.cameras);
    check(noMap && noMap.total > 0, 'and its poses', noMap && noMap.total);
    check(/in full, not on demand: its recording session/.test(r.status), 'the status says so', r.status);

    r = await load('twoskel', 'route');
    const twoSkel = await summary();
    check(twoSkel && !twoSkel.lazy && twoSkel.total === 1, 'two skeletons: the eager path, showing the pose on the session\'s skeleton', twoSkel);
    check(/in full, not on demand: it does not have exactly one skeleton/.test(r.status), 'the status says so', r.status);
    check(/1 pose on a camera's video is on another skeleton \("other"\)/.test(r.status), 'and that the other pose is not shown', r.status);
    const twoSave = await page.evaluate(async () => (await window.__posesOf(await window.__saveBytes())).lines);
    check(twoSave.length === 2 && twoSave.some(l => /\|500,500,v.* 502,502,v/.test(l)), 'a save keeps both, the other with all its nodes', twoSave);

    // When the eager path itself fails, its error stands.
    await page.evaluate(() => {
        const real = window.SleapIO.readSlpStreaming, RealWorker = window.Worker;
        window.__tstRestore = () => { window.SleapIO.readSlpStreaming = real; window.Worker = RealWorker; };
        window.SleapIO.readSlpStreaming = function (file, opts) {
            if (opts && opts.lazy) return real.call(this, file, opts);
            // The eager read, and then its raw worker fallback.
            window.Worker = function () { throw new Error('simulated read failure'); };
            return Promise.reject(new Error('simulated read failure'));
        };
    });
    r = await load('duprow', 'route');
    await page.evaluate(() => window.__tstRestore());
    check(/simulated read failure/.test(r.status) && !/in full/.test(r.status), 'a failed eager load keeps its own error, not "opened"', r.status);

    await page.evaluate(() => { window.__lucid.state.session = null; });
    r = await load('embedded', 'route');
    const emb = await summary();
    check(!emb || !emb.lazy, 'images in the file: never the lazy path, which would show none and save over them', emb && emb.lazy);
    check(!/Lazy reopen error/.test(r.status), 'and not the lazy path\'s error', r.status);

    // ---------------------------------------------------------------- G
    console.log('\nG. A calibration loaded afterwards');
    const CAL = path.join(fixtureDir, 'calibration.toml');
    const calOf = (names) => names.map((name, i) => `[cam_${i}]\nname = "${name}"\nsize = [ 640, 512,]\n` +
        'matrix = [ [ 900.0, 0.0, 320.0,], [ 0.0, 900.0, 256.0,], [ 0.0, 0.0, 1.0,],]\ndistortions = [ 0.0, 0.0, 0.0, 0.0, 0.0,]\n' +
        `rotation = [ 0.0, ${0.3 * (i + 1)}, 0.0,]\ntranslation = [ 0.0, 0.0, 1000.0,]\n`).join('\n');
    const loadCalibration = async (names) => {
        fs.writeFileSync(CAL, calOf(names));
        // The chooser needs the evaluate's user gesture, which an import can outlast.
        await page.evaluate(async () => { window.__tstLoader = await import('/loading/session-loader.js'); });
        const chooser = page.waitForEvent('filechooser', { timeout: 120000 });
        await page.evaluate(() => { window.__tstCal = window.__tstLoader.handleLoadCalibration(); });
        await (await chooser).setFiles(CAL);
        await page.evaluate(() => window.__tstCal);
    };
    const longSrc = await readSource('longnames');
    const WANT = ['Camera0_topB', 'Camera1_topC'];

    for (const how of ['lazy', 'eager']) {
        await load('longnames', how);
        await loadCalibration(WANT);
        const s = await summary();
        check(s && s.lazy === (how === 'lazy') && s.calibrated && JSON.stringify(s.cameras) === JSON.stringify(WANT), how + ', no video attached: the calibration\'s cameras', s && s.cameras);
        check(s && s.total === longSrc.n && Object.keys(s.perCam).sort().join() === WANT.join(), how + ': every pose followed onto them', s && s.perCam);
        const res = await page.evaluate(async () => {
            const got = await window.__posesOf(await window.__saveBytes());
            return { n: got.lines.length, perVideo: got.perVideo, same: JSON.stringify(got.lines) === JSON.stringify(window.__source.lines) };
        });
        check(res.n === longSrc.n && JSON.stringify(res.perVideo) === JSON.stringify(longSrc.perVideo), how + ': a save keeps each on its own video', res);
        check(res.same, how + ': the same poses, by value');
    }

    await load('longnames', 'lazy');
    const beforeRename = await framesAsLoaded([20]);
    await page.setInputFiles('#__tstVideos', [path.join(repoRoot, 'tests', 'fixtures', 'barcode-60', 'barcode-60.mp4')]);
    const withVideo = await page.evaluate(async () => {
        // A video picked for the first camera, named as the file refers to it.
        const { state } = window.__lucid;
        const sl = await import('/loading/session-loader.js');
        const f = document.getElementById('__tstVideos').files[0];
        const named = new File([f], state.session.cameras[0].name + '.mp4', { type: f.type });
        await sl.attachVideosForLazyReopen(state.session, state.session.lazyLoader, [named]);
        return { views: state.views.map(v => v.name) };
    });
    check(withVideo.views.length === 1, 'a video attached to the first camera', withVideo);
    await loadCalibration(WANT);
    const renamed = await summary();
    check(renamed && renamed.lazy && renamed.total === longSrc.n && Object.keys(renamed.perCam).sort().join() === WANT.join(), 'with a video attached: every pose followed too', renamed && renamed.perCam);
    const renamedViews = await page.evaluate(() => window.__lucid.state.views.map(v => v.name));
    check(JSON.stringify(renamedViews) === JSON.stringify(['Camera0_topB']), 'and the view', renamedViews);
    const afterRename = await framesAsLoaded([20]);
    const rowsOf = (byFrame) => Object.values(byFrame[20]).map(rows => JSON.stringify(rows));
    check(JSON.stringify(rowsOf(afterRename)) === JSON.stringify(rowsOf(beforeRename)) && Object.keys(afterRename[20]).sort().join() === WANT.join(),
        'a frame hydrates under the new names, with the same poses', Object.keys(afterRename[20]));
    const elsewhere = await framesAsLoaded([55]);
    check(Object.keys(elsewhere[55]).sort().join() === WANT.join(), 'a frame never resident hydrates under them too', Object.keys(elsewhere[55]));
    const res = await page.evaluate(async () => {
        const got = await window.__posesOf(await window.__saveBytes());
        return { n: got.lines.length, perVideo: got.perVideo, same: JSON.stringify(got.lines) === JSON.stringify(window.__source.lines) };
    });
    check(res.n === longSrc.n && res.same, 'a save keeps them all, by value', res);

    // The multi-session save reopens the file for its second pass, and the file
    // has the cameras' first names.
    const multi = await page.evaluate(async () => {
        const { state } = window.__lucid;
        const sl = await import('/import-export/save-load.js');
        const bytes = await sl.saveAllSessionsStreaming([state.session], {});
        const got = await window.__posesOf(bytes);
        return { n: got.lines.length, perVideo: got.perVideo, same: JSON.stringify(got.lines) === JSON.stringify(window.__source.lines) };
    });
    check(multi.n === longSrc.n && multi.same, 'so does the multi-session save, which reads the file again', multi);

    await load('longnames', 'lazy');
    await loadCalibration(['left', 'right']);
    const kept = await page.evaluate(async () => {
        const got = await window.__posesOf(await window.__saveBytes());
        // The calibration's cameras come first in the file's list, so each video moves by two.
        const moved = window.__source.lines.map(l => (Number(l.split('|')[0]) + 2) + l.slice(l.indexOf('|'))).sort();
        return {
            same: JSON.stringify(got.lines) === JSON.stringify(moved),
            videos: got.videos.slice(2).map(v => v.filename), source: window.__source.videos.map(v => v.filename),
        };
    });
    check(kept.same, 'a calibration naming none of the cameras: a save keeps every pose, each on its own video', kept);
    check(JSON.stringify(kept.videos) === JSON.stringify(kept.source), 'and lists those videos after its cameras', kept);

    // The console errors of a failed read are this test's own doing.
    const unexpected = errs.filter(e => !/Parse failed/.test(e));
    check(unexpected.length === 0, 'no page/console errors', unexpected.length ? unexpected.slice(0, 3) : undefined);
    await browser.close();
} finally {
    server.kill();
    fs.rmSync(fixtureDir, { recursive: true, force: true });
}

console.log(fails === 0 ? '\nPASS' : `\nFAIL (${fails})`);
process.exit(fails === 0 ? 0 : 1);
