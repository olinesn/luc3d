/**
 * lazy-session-stays-saveable.mjs — File ▸ Save and a session switch leave a lazy
 * session whole, so no save can write a file that is missing it with no word.
 *
 * A lazy session keeps its poses in a columnar store behind `session.lazyLoader`, and
 * its groups and 3D (Track All, Triangulate All) nowhere but `session.instanceGroups`.
 * Three things used to close the loader and empty the groups:
 *
 *  - the multi-session Save, which then left every session with neither. The second
 *    Save in one sitting wrote every camera with no poses, over the first if in place;
 *  - leaving a lazy session (`switchSession`), with nothing that ever reopened it;
 *  - so the multi-session folder load too, whose last step is `switchSession(0)`:
 *    the last session's poses were gone before the user saw it.
 *
 * Every section goes through the real folder loader, the real Track All and
 * Triangulate All, and the real Save (`saveProjectSlp`, or `quickSave` for a save in
 * place). The folders live in the origin private file system, so the `File`s are real
 * disk snapshots and the save handles are real ones. Each save is read with the app's
 * own reader and with Python sleap-io, and both must find every pose and every group,
 * or find a refusal that wrote nothing.
 *
 *  A. Two lazy sessions: the last session survives the load; Save over and over, with
 *     switches and an unsaved edit in between.
 *  B. One lazy session, one eager: the combined writer sees only frames in memory, so
 *     Save refuses, with the same words before and after a switch, and writes nothing,
 *     until every frame is in memory and it can write the whole project.
 *  C. Saving in place over a file a session was opened from: that `File` can no longer
 *     be read, and no save needs it.
 *  D. A session evicted on purpose (`commitSessionForMultiSessionSave` without
 *     `keepLoaded`): it wrote the same tables as a save that keeps the sessions, store
 *     edits included, and the app's Save then refuses instead of writing it empty.
 *  E. The same camera names in both sessions, as real recordings have.
 *
 * Needs `uv` for Python (skipped without it).
 *
 * Run: node tests/e2e/lazy-session-stays-saveable.mjs   (spawns its own http.server; PORT= to
 * move it, ONLY=AC for those sections)
 */
import { chromium } from 'playwright';
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { answerTrackSetup } from './fixtures/track-setup.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8743);
const PY_SPEC = process.env.SLEAP_IO_SPEC || 'sleap-io==0.9.5';

let fails = 0;
const check = (c, m, got) => {
    console.log((c ? '  ✓ ' : '  ✗ ') + m + (!c && got !== undefined ? ' — got ' + JSON.stringify(got).slice(0, 600) : ''));
    if (!c) fails++;
    stallGuard.refresh();
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const diff = (got, want) => {
    const g = new Set(got), w = new Set(want);
    return { got: got.length, want: want.length, missing: want.filter(x => !g.has(x)).slice(0, 3), extra: got.filter(x => !w.has(x)).slice(0, 3) };
};

// One line per pose (video|frame|P or U|track|x,y;x,y) and per group
// (s<session>|frame|cam=x|cam=x|x,y,z of node 0). Both readers print these.
const PY_SAME_TABLES = `
import json, sys
import h5py

def tables(path):
    out = {}
    with h5py.File(path, "r") as f:
        # As text, so that a NaN equals a NaN.
        f.visititems(lambda name, o: out.__setitem__(name, json.dumps(o[()].tolist(), default=str)) if isinstance(o, h5py.Dataset) else None)
    return out

a, b = tables(sys.argv[1]), tables(sys.argv[2])
print(json.dumps({"names": sorted(a), "different": sorted(n for n in set(a) | set(b) if a.get(n) != b.get(n))}))
`;

const PY_READ = `
import json, sys
import numpy as np
import sleap_io as sio

def num(v):
    return "nan" if v != v else f"{float(v):.3f}"

def pose_row(video, frame_idx, inst):
    cells = []
    for k in range(len(inst.points)):
        x, y = inst.points["xy"][k]
        cells.append("-" if x != x or y != y else f"{num(x)},{num(y)}")
    kind = "P" if isinstance(inst, sio.PredictedInstance) else "U"
    return "|".join([video, str(frame_idx), kind, inst.track.name if inst.track else "", ";".join(cells)])

def read(path):
    try:
        labels = sio.load_slp(path, open_videos=False)
    except Exception as e:
        return {"error": type(e).__name__ + ": " + str(e)[:200]}
    groups = []
    for si, s in enumerate(labels.sessions):
        for f, fg in s.frame_groups.items():
            for ig in fg.instance_groups:
                members = sorted(c.name + "=" + num(i.points["xy"][0][0]) for c, i in ig.instance_by_camera.items())
                p3 = "" if ig.instance_3d is None else ",".join(num(v) for v in np.asarray(ig.instance_3d.points)[0][:3])
                groups.append("|".join(["s" + str(si), str(f)] + members + [p3]))
    return {
        "error": None,
        "sessions": len(labels.sessions),
        "poses": sorted(pose_row(lf.video.filename, lf.frame_idx, i) for lf in labels.labeled_frames for i in lf.instances),
        "groups": sorted(groups),
    }

print(json.dumps({p: read(p) for p in sys.argv[1:]}))
`;

function uv(script, args) {
    const r = spawnSync('uv', ['run', '--no-project', '--quiet', '--python', '3.12', '--with', PY_SPEC, '--with', 'numpy', '--with', 'h5py', 'python', '-I', '-c', script, ...args],
        { encoding: 'utf8', maxBuffer: 256 * 1024 * 1024 });
    if (r.status !== 0) throw new Error('python failed: ' + (r.stderr || '').trim().split('\n').slice(-6).join(' | '));
    stallGuard.refresh();
    return r.stdout;
}

if (spawnSync('uv', ['--version'], { encoding: 'utf8' }).status !== 0) {
    console.log('  - skipped: no uv');
    process.exit(0);
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'lazy-stays-saveable-'));
const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
for (let i = 0; i < 240; i++) {
    try { if ((await fetch(`http://localhost:${PORT}/index.html`)).ok) break; } catch { /* not yet */ }
    await new Promise(r => setTimeout(r, 250));
}
const browser = await chromium.launch();
// `page.evaluate` has no timeout, so a step waiting on a click nobody makes would sit
// silent for good. Closing the browser fails that step.
const stallGuard = setTimeout(() => {
    console.log('\nStuck: no check for 8 minutes');
    browser.close();
}, 8 * 60 * 1000);
stallGuard.unref();

/** A page in its own context, so its private file system starts empty. */
async function newAppPage() {
    const context = await browser.newContext();
    const page = await context.newPage();
    page.on('pageerror', e => { console.log('  [pageerror]', String(e).slice(0, 300)); fails++; });
    await answerTrackSetup(page, 2);
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state && window.SleapIO, null, { timeout: 180000 });
    await page.evaluate(installHelpers);
    return page;
}

/**
 * Runs in the page: the fixture and the steps every section shares.
 *
 * Two animals in front of two cameras, with real geometry so that the real Track All
 * groups them. Each session's animals stand elsewhere, so a pose or a group that lands
 * on the wrong session cannot pass for the right one. Coordinates are rounded to 1/8
 * pixel, which 32 bits hold exactly.
 */
function installHelpers() {
    const NF = 6, NODES = ['nose', 'tail'];
    const K = [[600, 0, 320], [0, 600, 240], [0, 0, 1]];
    const ROT = i => [0, 0.3 * i, 0], TRANS = i => [20 * i, 0, 0];
    const round8 = v => Math.round(v * 8) / 8;
    const truth3d = (s, a, f, k) => [(a ? 12 : -12) + 0.2 * f + k + 3 * s, 5 + k, 50 + k];

    const modules = async () => ({
        AS: await import('/ui/app-state.js'),
        pd: await import('/pose/pose-data.js'),
        tri: await import('/pose/triangulation.js'),
        tracker: await import('/pose/tracker.js'),
        loader: await import('/loading/session-loader.js'),
        panes: await import('/ui/sessions-panes.js'),
        saveLoad: await import('/import-export/save-load.js'),
        fileio: await import('/import-export/file-io.js'),
        slpimp: await import('/import-export/slp-import.js'),
    });

    // The Python and app lines for what the fixture holds, computed from the fixture
    // alone: [{ camera, frame, track, nodes: ['x,y', ...], x0 }] per session.
    async function truth(spec) {
        const { pd } = await modules();
        return spec.map((sess, s) => {
            const rows = [];
            sess.cams.forEach((name, ci) => {
                const cam = new pd.Camera(name, K, [0, 0, 0, 0, 0], ROT(ci), TRANS(ci), [640, 480]);
                for (let f = 0; f < (sess.frames || NF); f++) {
                    for (let a = 0; a < 2; a++) {
                        const pts = NODES.map((_, k) => cam.project(truth3d(s, a, f, k)).map(round8));
                        rows.push({ camera: name, ci, frame: f, animal: a, track: 't' + a, pts, p3: truth3d(s, a, f, 0) });
                    }
                }
            });
            return rows;
        });
    }

    async function cameraSlpBytes(rows, camName) {
        const SIO = window.SleapIO;
        const skeleton = new SIO.Skeleton({ name: 'sk', nodes: NODES });
        const video = new SIO.Video({ filename: camName + '.mp4', shape: [60, 480, 640, 1] });
        const tracks = [new SIO.Track('t0'), new SIO.Track('t1')];
        const lfs = [];
        const ownRows = rows.filter(r => r.camera === camName);
        for (const f of new Set(ownRows.map(r => r.frame))) {
            const instances = ownRows.filter(r => r.frame === f).map(r => new SIO.PredictedInstance({
                points: r.pts, skeleton, track: tracks[r.animal], score: 0.9,
            }));
            lfs.push(new SIO.LabeledFrame({ video, frameIdx: f, instances }));
        }
        return SIO.saveSlpToBytes(new SIO.Labels({ skeletons: [skeleton], videos: [video], tracks, labeledFrames: lfs }));
    }

    async function writeFile(dir, name, bytes, extendTo) {
        const h = await dir.getFileHandle(name, { create: true });
        const w = await h.createWritable();
        await w.write(bytes);
        // `shouldUseLazySlp` reads only `file.size`, and HDF5 ignores trailing bytes,
        // so a longer file takes the lazy route. One lazy camera sends its whole
        // session down it.
        if (extendTo) await w.truncate(extendTo);
        await w.close();
        return h;
    }

    // <parent>/<session>/{calibration.toml, <camera>/{<camera>_v1.slp, <camera>.mp4}}
    async function buildFolder(spec) {
        const { tri } = await modules();
        const rowsBySession = await truth(spec);
        const video = new Uint8Array(await (await fetch('/tests/fixtures/barcode-60/barcode-60.mp4')).arrayBuffer());
        const root = await navigator.storage.getDirectory();
        const parent = await root.getDirectoryHandle('parent', { create: true });
        for (let s = 0; s < spec.length; s++) {
            const dir = await parent.getDirectoryHandle(spec[s].name, { create: true });
            let toml = '';
            spec[s].cams.forEach((cn, i) => {
                toml += '[cam_' + i + ']\nname = "' + cn + '"\nsize = [640, 480]\n'
                    + 'matrix = [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]]\n'
                    + 'distortions = [0.0, 0.0, 0.0, 0.0, 0.0]\n'
                    + 'rotation = [0.0, ' + ROT(i)[1] + ', 0.0]\ntranslation = [' + TRANS(i)[0] + ', 0.0, 0.0]\n\n';
            });
            await writeFile(dir, 'calibration.toml', new TextEncoder().encode(toml));
            for (let ci = 0; ci < spec[s].cams.length; ci++) {
                const cn = spec[s].cams[ci];
                const camDir = await dir.getDirectoryHandle(cn, { create: true });
                const bytes = await cameraSlpBytes(rowsBySession[s], cn);
                const big = spec[s].lazy && ci === 0;
                await writeFile(camDir, cn + '_v1.slp', bytes, big ? tri.LAZY_SLP_THRESHOLD + (1 << 20) : 0);
                await writeFile(camDir, cn + '.mp4', video);
            }
        }
        return parent;
    }

    // The real Load Multi-Session Folder, with the picker pointed at the fixture and
    // the skeleton question skipped.
    async function loadFolder(spec) {
        const { AS, loader } = await modules();
        const parent = await buildFolder(spec);
        window.showDirectoryPicker = async () => parent;
        const skipModals = new MutationObserver(() => {
            for (const id of ['msSkelSkip', 'btnCalibNoticeOk']) {
                const b = document.getElementById(id);
                if (b) b.click();
            }
            // The question a switch used to ask, which nobody here would answer.
            const discard = document.getElementById('swDiscard');
            if (discard) { window.__switchAsked = true; discard.click(); }
        });
        skipModals.observe(document, { childList: true, subtree: true });
        await loader.handleLoadMultiSession();
        // The note and the prompt come after `handleLoadMultiSession` returns.
        await new Promise(r => setTimeout(r, 300));
        return AS.state.sessions.length;
    }

    async function trackAndTriangulate() {
        const { tracker, tri } = await modules();
        await tracker.trackAll();
        await tri.triangulateAllFrames('dlt');
        return status();
    }

    // What the project holds right now, not what a file says.
    async function live() {
        const { AS } = await modules();
        return AS.state.sessions.map(s => {
            let rows = 0, groups = 0, with3d = 0;
            if (s.lazyLoader && s.lazyLoader.forEachInstanceRow) s.lazyLoader.forEachInstanceRow(() => { rows++; });
            for (const [, gs] of s.instanceGroups) for (const g of gs) { groups++; if (g.points3d && g.points3d.length) with3d++; }
            return { name: s.name, lazy: !!s.lazyLoader, rows, groups, with3d, resident: s.frameGroups.size };
        });
    }

    // What the app has for each camera of the active session on `frame`, hydrated
    // the way navigating to that frame hydrates it.
    async function frameShows(frame) {
        const { AS, tri } = await modules();
        const s = AS.state.session;
        await tri.ensureLazyFrameData(frame);
        const fg = s.frameGroups.get(frame);
        const perCam = {};
        for (const c of s.cameras) {
            const xs = [];
            if (fg) {
                for (const inst of fg.instances.get(c.name) || []) xs.push(inst.getX(0));
                for (const ul of fg.unlinkedInstances.get(c.name) || []) xs.push(ul.instance.getX(0));
            }
            perCam[c.name] = xs.sort((a, b) => a - b);
        }
        return { session: s.name, groups: (s.instanceGroups.get(frame) || []).length, perCam };
    }

    function status() {
        return (document.getElementById('statusText') || {}).textContent || '';
    }

    // File ▸ Save with no file handle: the whole file as a download. `window.__saved`
    // holds what would be downloaded; no download means nothing was written.
    async function saveProject() {
        const { saveLoad } = await modules();
        const orig = URL.createObjectURL;
        let blob = null;
        URL.createObjectURL = b => { blob = b; return orig.call(URL, b); };
        try { await saveLoad.saveProjectSlp(); } finally { URL.createObjectURL = orig; }
        window.__saved = blob ? new Uint8Array(await blob.arrayBuffer()) : null;
        return { wrote: !!blob, status: status() };
    }

    // Save in place, with `handle` as the file picked earlier: `window.__saved` is
    // what the file holds afterwards. Counts how the save ended the write.
    async function quickSaveTo(handle) {
        const { AS, saveLoad } = await modules();
        AS.state.slpFileHandle = handle;
        const ended = { closed: 0, aborted: 0 };
        const orig = handle.createWritable;
        handle.createWritable = async function () {
            const w = await orig.apply(handle, arguments);
            const close = w.close.bind(w), abort = w.abort.bind(w);
            w.close = () => { ended.closed++; return close(); };
            w.abort = r => { ended.aborted++; return abort(r); };
            return w;
        };
        try { await saveLoad.quickSave(); } finally { delete handle.createWritable; }
        const file = await handle.getFile();
        window.__saved = new Uint8Array(await file.arrayBuffer());
        return { ended, status: status(), size: file.size, dirty: AS.state.isDirty };
    }

    // What the app's own reader says `window.__saved` holds, in Python's terms.
    async function readSaved() {
        const { pd, fileio, slpimp } = await modules();
        const slp = await fileio.parseSlpViaSleapIO(new File([window.__saved], 'saved.slp'), () => {});
        const num = v => (v === v ? Number(v).toFixed(3) : 'nan');
        const poses = [];
        for (const f of slp.frames) {
            for (const d of f.instances) {
                const nodes = d.points.map(p => (p ? num(p[0]) + ',' + num(p[1]) : '-')).join(';');
                const track = d.trackIdx >= 0 ? slp.tracks[d.trackIdx] : '';
                poses.push([slp.videos[f.videoIdx].filename, f.frameIdx, d.type === 'predicted' ? 'P' : 'U', track, nodes].join('|'));
            }
        }
        const groups = [];
        for (let si = 0; si < (slp.sessions || []).length; si++) {
            const s = slp.sessions[si];
            if (!s._typedSession) continue;
            const keys = Object.keys(s.calibration || {}).filter(k => k !== 'metadata');
            const fresh = new pd.Session(
                keys.map(k => { const c = s.calibration[k]; return new pd.Camera(c.name || k, c.matrix, c.distortions, c.rotation, c.translation, c.size); }),
                new pd.Skeleton(slp.skeleton.name, slp.skeleton.nodes, slp.skeleton.edges), slp.tracks, 'read');
            await slpimp.reconstructInstanceGroupsFromSession(fresh, s._typedSession, s, slp.skeleton.nodes, {});
            for (const [f, gs] of fresh.instanceGroups) {
                for (const g of gs) {
                    const members = [...g.instances].map(([c, i]) => c + '=' + num(i.getX(0))).sort();
                    const p3 = g.points3d && g.points3d.length ? Array.from(g.points3d).slice(0, 3).map(num).join(',') : '';
                    groups.push(['s' + si, f].concat(members, [p3]).join('|'));
                }
            }
        }
        return { sessions: (slp.sessions || []).length, poses: poses.sort(), groups: groups.sort() };
    }

    function savedB64() {
        const u = window.__saved;
        let s = '';
        for (let i = 0; i < u.length; i += 32768) s += String.fromCharCode.apply(null, u.subarray(i, i + 32768));
        return btoa(s);
    }

    window.__t = { modules, truth, loadFolder, trackAndTriangulate, live, frameShows, status, saveProject, quickSaveTo, readSaved, savedB64 };
}

// ---- what a save must hold ----------------------------------------------------

const num = v => Number(v).toFixed(3);

/** Pose lines and group keys for the fixture; `videoOf(session, camera)` names a pose's video. */
function wantedFrom(rowsBySession, videoOf) {
    const poses = [], groups = new Map();
    rowsBySession.forEach((rows, s) => {
        for (const r of rows) {
            const nodes = r.pts.map(p => num(p[0]) + ',' + num(p[1])).join(';');
            poses.push([videoOf(s, r.camera), r.frame, 'P', r.track, nodes].join('|'));
            const key = 's' + s + '|' + r.frame + '|' + r.animal;
            if (!groups.has(key)) groups.set(key, { s, frame: r.frame, members: [], p3: r.p3 });
            groups.get(key).members.push(r.camera + '=' + num(r.pts[0][0]));
        }
    });
    return {
        poses: poses.sort(),
        groups: [...groups.values()].map(g => ({ key: ['s' + g.s, g.frame].concat(g.members.sort()).join('|'), p3: g.p3 })),
    };
}

/** A group line without its 3D, and its 3D as numbers. */
function splitGroup(line) {
    const at = line.lastIndexOf('|');
    return { key: line.slice(0, at), p3: line.slice(at + 1).split(',').map(Number) };
}

function compare(what, got, want) {
    if (got.error) { check(false, what + ': opens', got.error); return; }
    const poseKey = want.anyVideo ? (l => l.slice(l.indexOf('|') + 1)) : (l => l);
    const gotPoses = got.poses.map(poseKey).sort(), wantPoses = want.poses.map(poseKey).sort();
    check(eq(gotPoses, wantPoses), what + ': every pose (' + wantPoses.length + ')', diff(gotPoses, wantPoses));
    const lines = got.groups.map(splitGroup);
    const wantKeys = want.groups.map(g => g.key).sort();
    check(eq(lines.map(l => l.key).sort(), wantKeys), what + ': every group, on its own poses (' + wantKeys.length + ')',
        diff(lines.map(l => l.key).sort(), wantKeys));
    const p3Of = new Map(want.groups.map(g => [g.key, g.p3]));
    const off = lines.filter(l => {
        const t = p3Of.get(l.key);
        return t && !(l.p3.length === 3 && l.p3.every((v, i) => Math.abs(v - t[i]) < 0.25));
    });
    check(off.length === 0, what + ': every group\'s 3D where the animal is', off.slice(0, 2));
}

let savedSeq = 0;
const pythonWork = [];   // [{ file, what, want }]: one Python run reads every save
async function checkSaved(page, what, want) {
    const read = await page.evaluate(() => window.__t.readSaved());
    compare(what + ', app reader', read, want());
    const file = path.join(tmp, 'saved-' + (++savedSeq) + '.slp');
    fs.writeFileSync(file, Buffer.from(await page.evaluate(() => window.__t.savedB64()), 'base64'));
    pythonWork.push({ file, what: what + ', Python', want: want() });
}

const TWO_LAZY = [
    { name: 'sessA', cams: ['a_cam0', 'a_cam1'], lazy: true },
    { name: 'sessB', cams: ['b_cam0', 'b_cam1'], lazy: true },
];
// Long enough that a seek leaves some of the lazy session's frames out of memory.
const LAZY_THEN_EAGER = [
    { name: 'sessA', cams: ['a_cam0', 'a_cam1'], lazy: true, frames: 60 },
    { name: 'sessB', cams: ['b_cam0', 'b_cam1'], lazy: false },
];
const SAME_CAMERA_NAMES = [
    { name: 'sessA', cams: ['cam0', 'cam1'], lazy: true },
    { name: 'sessB', cams: ['cam0', 'cam1'], lazy: true },
];
const videoPath = spec => (s, cam) => spec[s].name + '/' + cam + '/' + cam + '.mp4';

/** File ▸ Load SLP in a new tab, answering the questions about videos; what each session then holds. */
async function openInNewTab(b64) {
    const page = await newAppPage();
    try {
        await page.evaluate((data) => {
            const bytes = Uint8Array.from(atob(data), c => c.charCodeAt(0));
            window.__opened = false;
            (async () => {
                const slpimp = await import('/import-export/slp-import.js');
                await slpimp.handleLoadSlpFile(new File([bytes], 'project.slp'));
            })().then(() => { window.__opened = true; }, e => { window.__opened = 'ERR ' + ((e && e.stack) || e); });
        }, b64);
        let opened = false;
        for (let i = 0; i < 3000 && !opened; i++) {
            const st = await page.evaluate(() => {
                const b = [...document.querySelectorAll('button')].find(x => x.offsetParent && /^(Skip|Later|Cancel|Not now|Continue)$/i.test((x.textContent || '').trim()));
                if (b) { b.click(); return { opened: false }; }
                return { opened: window.__opened, prompt: !!document.getElementById('videoSelectOverlay') };
            });
            if (st.prompt) await page.keyboard.press('Escape');
            opened = st.opened;
            if (!opened) await page.waitForTimeout(100);
        }
        if (opened !== true) throw new Error('the open did not finish: ' + opened);
        return await page.evaluate(() => window.__lucid.state.sessions.map(s => {
            let poses = 0, groups = 0, with3d = 0;
            for (const [, fg] of s.frameGroups) {
                for (const [, l] of fg.instances) poses += l.length;
                for (const [, l] of fg.unlinkedInstances) poses += l.length;
            }
            for (const [, gs] of s.instanceGroups) for (const g of gs) { groups++; if (g.points3d && g.points3d.length) with3d++; }
            return { name: s.name, poses, groups, with3d };
        }));
    } finally {
        await page.context().close().catch(() => {});
    }
}

/** One failure must not stop the other sections from reporting. */
async function section(title, run) {
    if (process.env.ONLY && !process.env.ONLY.includes(title[0])) return;
    console.log('\n' + title);
    const page = await newAppPage();
    try {
        await run(page);
    } catch (e) {
        check(false, 'ran to its end', String((e && e.message) || e).slice(0, 400));
    } finally {
        await page.context().close().catch(() => {});
    }
}

try {
    await section('A. Two lazy sessions', async (page) => {
        const fullRows = 2 * 6 * 2;   // cameras x frames x animals
        const nSessions = await page.evaluate(spec => window.__t.loadFolder(spec), TWO_LAZY);
        check(nSessions === 2, 'both sessions loaded', nSessions);
        let state = await page.evaluate(() => window.__t.live());
        check(state.every(s => s.lazy && s.rows === fullRows),
            'after the folder load every session still has its loader and every pose, the last one too', state);

        await page.evaluate(() => window.__t.trackAndTriangulate());
        await page.evaluate(async () => { const { panes } = await window.__t.modules(); await panes.switchSession(1); });
        await page.evaluate(() => window.__t.trackAndTriangulate());
        state = await page.evaluate(() => window.__t.live());
        check(state.every(s => s.lazy && s.rows === fullRows && s.groups === 12 && s.with3d === 12),
            'Track All and Triangulate All ran on each, one after the other: 12 groups with 3D each', state);

        const rows = await page.evaluate(spec => window.__t.truth(spec), TWO_LAZY);
        const want = wantedFrom(rows, videoPath(TWO_LAZY));
        const stillWhole = async (what) => {
            const now = await page.evaluate(() => window.__t.live());
            check(now.every(s => s.lazy && s.rows === fullRows && s.groups === 12 && s.with3d === 12),
                what + ': every session is as it was', now);
        };

        let saved = await page.evaluate(() => window.__t.saveProject());
        check(saved.wrote, 'first Save wrote a file', saved);
        await checkSaved(page, 'first Save', () => want);
        await stillWhole('after the first Save');

        const onScreen = await page.evaluate(() => window.__t.frameShows(2));
        const wantShown = {};
        for (const r of rows[1].filter(x => x.frame === 2)) (wantShown[r.camera] = wantShown[r.camera] || []).push(r.pts[0][0]);
        for (const k in wantShown) wantShown[k].sort((a, b) => a - b);
        check(onScreen.groups === 2 && eq(onScreen.perCam, wantShown),
            'after the first Save the session on screen still shows its poses and groups', onScreen);

        saved = await page.evaluate(() => window.__t.saveProject());
        check(saved.wrote, 'second Save wrote a file', saved);
        await checkSaved(page, 'second Save', () => want);

        // Away and back, with no save prompt in the way (nothing is lost by switching).
        const edit = await page.evaluate(async () => {
            const { AS, panes, saveLoad, tri } = await window.__t.modules();
            await panes.switchSession(0);
            const s = AS.state.session;
            await tri.ensureLazyFrameData(3);
            const g = (s.instanceGroups.get(3) || [])[0];
            const [cam, inst] = [...g.instances][0];
            const before = inst.getX(0), y = inst.getY(0);
            inst.setPoint(0, before + 7.5, y - 3.25);
            inst.type = 'user';
            inst.modified = true;
            g.markDirty();
            saveLoad.markDirty();
            const dirtyBefore = AS.state.isDirty && s.isDirty;
            await panes.switchSession(1);
            await panes.switchSession(0);
            const resident = (s.instanceGroups.get(3) || [])[0].instances.get(cam);
            return { cam, before, y, x: resident.getX(0), dirtyBefore, dirtyAfter: AS.state.isDirty && s.isDirty, asked: !!window.__switchAsked };
        });
        check(!edit.asked, 'no switch asked to save first: nothing is lost by switching');
        check(edit.dirtyBefore && edit.dirtyAfter, 'the edit is still unsaved after going away and back', edit);
        check(edit.x === edit.before + 7.5, 'and it is still there', edit);

        const withEdit = () => {
            const was = wantedFrom(rows, videoPath(TWO_LAZY));
            const row = rows[0].find(r => r.camera === edit.cam && r.frame === 3 && r.pts[0][0] === edit.before);
            const line = r => [videoPath(TWO_LAZY)(0, r.camera), r.frame, 'P', r.track, r.pts.map(p => num(p[0]) + ',' + num(p[1])).join(';')].join('|');
            const moved = [videoPath(TWO_LAZY)(0, edit.cam), 3, 'U', row.track,
                num(edit.before + 7.5) + ',' + num(edit.y - 3.25) + ';' + num(row.pts[1][0]) + ',' + num(row.pts[1][1])].join('|');
            was.poses = was.poses.filter(l => l !== line(row)).concat([moved]).sort();
            const oldMember = edit.cam + '=' + num(edit.before), newMember = edit.cam + '=' + num(edit.before + 7.5);
            for (const g of was.groups) {
                const parts = g.key.split('|');
                if (parts[0] === 's0' && parts[1] === '3' && parts.includes(oldMember)) {
                    const m = parts.slice(2).map(p => (p === oldMember ? newMember : p)).sort();
                    g.key = [parts[0], parts[1]].concat(m).join('|');
                }
            }
            return was;
        };
        saved = await page.evaluate(() => window.__t.saveProject());
        check(saved.wrote, 'Save after the switches wrote a file', saved);
        await checkSaved(page, 'Save after an edit and switches', withEdit);
        await page.evaluate(async () => { const { panes } = await window.__t.modules(); await panes.switchSession(1); });
        saved = await page.evaluate(() => window.__t.saveProject());
        check(saved.wrote, 'one more Save, from the other session, wrote a file', saved);
        await checkSaved(page, 'Save from the other session', withEdit);

        // The whole way back: the same file through the app's own open, in a new tab.
        const b64 = await page.evaluate(() => window.__t.savedB64());
        const reopened = await openInNewTab(b64);
        check(eq(reopened, [
            { name: 'sessA', poses: 24, groups: 12, with3d: 12 },
            { name: 'sessB', poses: 24, groups: 12, with3d: 12 },
        ]), 'opened again, both sessions are there with every pose and group', reopened);

    });

    await section('B. One lazy session, one eager', async (page) => {
        await page.evaluate(spec => window.__t.loadFolder(spec), LAZY_THEN_EAGER);
        await page.evaluate(() => window.__t.trackAndTriangulate());
        await page.evaluate(async () => { const { panes } = await window.__t.modules(); await panes.switchSession(1); });
        await page.evaluate(() => window.__t.trackAndTriangulate());
        await page.evaluate(async () => { const { panes } = await window.__t.modules(); await panes.switchSession(0); });
        const before = await page.evaluate(() => window.__t.live());
        check(before[0].lazy && before[0].rows === 240 && before[0].groups === 120 && before[0].with3d === 120 && before[0].resident < 60,
            'the lazy session has every pose and group, with some frames out of memory', before[0]);
        check(!before[1].lazy && before[1].groups === 12 && before[1].with3d === 12, 'the eager session is tracked too', before[1]);

        await page.evaluate(async () => {
            const root = await navigator.storage.getDirectory();
            const h = await root.getFileHandle('project.slp', { create: true });
            const w = await h.createWritable();
            await w.write(new TextEncoder().encode('earlier save'));
            await w.close();
            window.__target = h;
        });

        const refused = async (what) => {
            const now0 = await page.evaluate(() => window.__t.live());
            const saved = await page.evaluate(() => window.__t.saveProject());
            check(!saved.wrote, what + ': Save wrote nothing', saved);
            check(/Cannot save/.test(saved.status) && /sessA/.test(saved.status), what + ': and said why, naming the lazy session', saved.status);
            // The picker can empty the file it is given, so a refusal comes first.
            const picker = await page.evaluate(async () => {
                const { AS, saveLoad } = await window.__t.modules();
                let asked = 0;
                window.showSaveFilePicker = async () => { asked++; throw new DOMException('no', 'AbortError'); };
                AS.state.slpFileHandle = null;
                await saveLoad.quickSave();
                return { asked, status: window.__t.status() };
            });
            check(picker.asked === 0 && /Cannot save/.test(picker.status), what + ': Save with no file yet refuses before the save dialog', picker);
            const saveAs = await page.evaluate(async () => {
                const { AS, saveLoad } = await window.__t.modules();
                let asked = 0;
                window.showSaveFilePicker = async () => { asked++; throw new DOMException('no', 'AbortError'); };
                AS.state.slpFileHandle = window.__target;
                await saveLoad.saveAs();
                return { asked, kept: AS.state.slpFileHandle === window.__target, status: window.__t.status() };
            });
            check(saveAs.asked === 0 && saveAs.kept && /Cannot save/.test(saveAs.status), what + ': so does Save As, and it forgets no file', saveAs);
            const inPlace = await page.evaluate(async () => {
                const { saveLoad } = await window.__t.modules();
                saveLoad.markDirty();
                const r = await window.__t.quickSaveTo(window.__target);
                r.text = new TextDecoder().decode(window.__saved);
                return r;
            });
            check(inPlace.text === 'earlier save', what + ': a save in place left the file as it was', inPlace);
            check(inPlace.ended.closed === 0, what + ': and never finished a write over it', inPlace.ended);
            check(inPlace.dirty, what + ': the project still reads as unsaved', inPlace);
            check(/Cannot save/.test(inPlace.status), what + ': and said why', inPlace.status);
            const now = await page.evaluate(() => window.__t.live());
            check(eq(now, now0), what + ': the project is as it was', now);
        };

        await refused('On the lazy session');
        await page.evaluate(async () => { const { panes } = await window.__t.modules(); await panes.switchSession(1); });
        await refused('After leaving the lazy session');

        const shown = await page.evaluate(async () => {
            const { panes } = await window.__t.modules();
            await panes.switchSession(0);
            return window.__t.frameShows(45);
        });
        check(shown.session === 'sessA' && shown.groups === 2 && shown.perCam.a_cam0.length === 2 && shown.perCam.a_cam1.length === 2,
            'back on the lazy session, its poses and groups are there', shown);

        // With every frame of it in memory the combined writer can hold it all, so the
        // save goes ahead. It has to be the whole project, also from the other session.
        const whole = await page.evaluate(async () => {
            const { AS, tri, panes } = await window.__t.modules();
            for (let f = 0; f < 60; f++) await tri.ensureLazyFrameData(f);
            const resident = AS.state.session.frameGroups.size;
            await panes.switchSession(1);
            return { resident, saved: await window.__t.saveProject() };
        });
        check(whole.resident >= 60 && whole.saved.wrote, 'with every frame in memory, and from the other session, Save wrote a file', whole);
        const rows = await page.evaluate(spec => window.__t.truth(spec), LAZY_THEN_EAGER);
        await checkSaved(page, 'Save of the mixed project', () => wantedFrom(rows, videoPath(LAZY_THEN_EAGER)));
    });

    await section('C. Saving in place over a file a session was opened from', async (page) => {
        await page.evaluate(spec => window.__t.loadFolder(spec), TWO_LAZY);
        await page.evaluate(() => window.__t.trackAndTriangulate());
        const rows = await page.evaluate(spec => window.__t.truth(spec), TWO_LAZY);
        const want = wantedFrom(rows, videoPath(TWO_LAZY));

        // Session A's groups are the only ones: B was never tracked.
        const wantA = () => ({ poses: want.poses, groups: want.groups.filter(g => g.key.startsWith('s0|')) });
        const inPlace = () => page.evaluate(async () => {
            const parent = await (await navigator.storage.getDirectory()).getDirectoryHandle('parent');
            const camDir = await (await parent.getDirectoryHandle('sessA')).getDirectoryHandle('a_cam0');
            return window.__t.quickSaveTo(await camDir.getFileHandle('a_cam0_v1.slp'));
        });

        let r = await inPlace();
        check(r.ended.closed === 1 && /^Saved/.test(r.status), 'first save over a source file', r);
        await checkSaved(page, 'First save in place', wantA);

        const stale = await page.evaluate(async () => {
            const { AS } = await window.__t.modules();
            const loader = AS.state.sessions[0].lazyLoader;
            const file = loader && loader.sourceFiles.get('a_cam0');
            if (!file) return { noLoader: true };
            try { await file.slice(0, 8).arrayBuffer(); return { readable: true }; }
            catch (e) { return { name: e.name }; }
        });
        check(!!stale.name, 'the File that session was opened from can no longer be read, so this tests that case', stale);

        r = await inPlace();
        check(r.ended.closed === 1 && /^Saved/.test(r.status), 'second save in place', r);
        await checkSaved(page, 'Second save in place', wantA);

        const after = await page.evaluate(async () => {
            const { panes } = await window.__t.modules();
            await panes.switchSession(1);
            await panes.switchSession(0);
            return window.__t.live();
        });
        check(after.every(s => s.lazy && s.rows === 24), 'switching away and back still has every pose', after);
        r = await inPlace();
        check(r.ended.closed === 1 && /^Saved/.test(r.status), 'third save in place, after the switches', r);
        await checkSaved(page, 'Third save in place', wantA);

        // A save that fails part-way gives up its write: the last good save stays.
        const failed = await page.evaluate(async () => {
            const parent = await (await navigator.storage.getDirectory()).getDirectoryHandle('parent');
            const camDir = await (await parent.getDirectoryHandle('sessA')).getDirectoryHandle('a_cam0');
            const handle = await camDir.getFileHandle('a_cam0_v1.slp');
            const before = new Uint8Array(await (await handle.getFile()).arrayBuffer());
            const real = window.SleapIO;
            window.SleapIO = Object.assign({}, real, { openSlpWriter: async () => { throw new Error('disk full'); } });
            let out;
            try { out = await window.__t.quickSaveTo(handle); } finally { window.SleapIO = real; }
            out.same = before.length === window.__saved.length && before.every((b, i) => b === window.__saved[i]);
            return out;
        });
        check(failed.same && failed.ended.closed === 0 && failed.ended.aborted === 1 && /disk full/.test(failed.status),
            'a save that fails part-way gives up its write and leaves the file as it was', failed);
        r = await inPlace();
        check(r.ended.closed === 1 && /^Saved/.test(r.status), 'the next save in place works', r);
        await checkSaved(page, 'Save in place after a failed one', wantA);
    });

    let sameTables = null;
    await section('D. A session evicted on purpose', async (page) => {
        await page.evaluate(spec => window.__t.loadFolder(spec), TWO_LAZY);
        await page.evaluate(() => window.__t.trackAndTriangulate());
        const rows = await page.evaluate(spec => window.__t.truth(spec), TWO_LAZY);

        // Edits that live only in the store, which the files on disk do not have:
        // session B loses one row and has its tracks swapped and renamed.
        const edited = await page.evaluate(async () => {
            const { AS } = await window.__t.modules();
            const B = AS.state.sessions[1];
            const del = B.lazyLoader.deleteInstanceRows((cam, f, offset) => cam === 'b_cam0' && f === 2 && offset === 0);
            const remap = B.lazyLoader.remapTracksFromIdentity(['animal_x', 'animal_y'], (cam, f, old) => (old === 0 ? 1 : old === 1 ? 0 : -1));
            B.tracks = ['animal_x', 'animal_y'];
            return { deleted: del.deleted, changed: remap.changed, errorRows: remap.errorRows };
        });
        check(edited.deleted === 1 && edited.changed > 0 && edited.errorRows === 0, 'session B edited in its store only', edited);
        const editedRows = rows.map((list, s) => (s === 0 ? list : list
            .filter(r => !(r.camera === 'b_cam0' && r.frame === 2 && r.animal === 0))
            .map(r => Object.assign({}, r, { track: r.animal === 0 ? 'animal_y' : 'animal_x' }))));
        const want = wantedFrom(editedRows, videoPath(TWO_LAZY));
        const wantA = () => ({ poses: want.poses, groups: want.groups.filter(g => g.key.startsWith('s0|')) });

        let saved = await page.evaluate(() => window.__t.saveProject());
        check(saved.wrote, 'File ▸ Save, which keeps the sessions, wrote a file', saved);
        await checkSaved(page, 'Save that keeps the sessions', wantA);
        const keptFile = pythonWork[pythonWork.length - 1].file;

        const evicted = await page.evaluate(async () => {
            const { AS, saveLoad } = await window.__t.modules();
            window.__saved = await saveLoad.saveAllSessionsStreaming(AS.state.sessions);
            return AS.state.sessions.map(s => ({ loader: !!s.lazyLoader, groups: s.instanceGroups.size }));
        });
        check(evicted.every(s => !s.loader && s.groups === 0), 'the sessions really were evicted', evicted);
        await checkSaved(page, 'The evicting writer, on the same project', wantA);
        sameTables = { a: keptFile, b: pythonWork[pythonWork.length - 1].file };

        saved = await page.evaluate(() => window.__t.saveProject());
        check(!saved.wrote, 'Save afterwards wrote nothing', saved);
        check(/Cannot save/.test(saved.status) && /sessA/.test(saved.status) && /sessB/.test(saved.status),
            'and said why, naming the sessions', saved.status);
        const inPlace = await page.evaluate(async () => {
            const root = await navigator.storage.getDirectory();
            const h = await root.getFileHandle('project.slp', { create: true });
            const w = await h.createWritable();
            await w.write(new TextEncoder().encode('earlier save'));
            await w.close();
            const r = await window.__t.quickSaveTo(h);
            r.text = new TextDecoder().decode(window.__saved);
            return r;
        });
        check(inPlace.text === 'earlier save' && inPlace.ended.closed === 0, 'a save in place left the file as it was', inPlace);
        const json = await page.evaluate(async () => {
            const { saveLoad } = await window.__t.modules();
            const orig = URL.createObjectURL;
            let wrote = false;
            URL.createObjectURL = b => { wrote = true; return orig.call(URL, b); };
            try { saveLoad.saveProject(); } finally { URL.createObjectURL = orig; }
            return { wrote, status: window.__t.status() };
        });
        check(!json.wrote && /Cannot save/.test(json.status), 'and so does the JSON project save', json);
    });

    await section('E. Sessions whose cameras have the same names, as real recordings do', async (page) => {
        await page.evaluate(spec => window.__t.loadFolder(spec), SAME_CAMERA_NAMES);
        const state0 = await page.evaluate(() => window.__t.live());
        check(state0.every(s => s.lazy && s.rows === 24), 'after the folder load both sessions have every pose', state0);
        await page.evaluate(() => window.__t.trackAndTriangulate());
        await page.evaluate(async () => { const { panes } = await window.__t.modules(); await panes.switchSession(1); });
        await page.evaluate(() => window.__t.trackAndTriangulate());
        const rows = await page.evaluate(spec => window.__t.truth(spec), SAME_CAMERA_NAMES);
        const want = () => Object.assign(wantedFrom(rows, () => ''), { anyVideo: true });

        for (const what of ['First Save', 'Second Save']) {
            const saved = await page.evaluate(() => window.__t.saveProject());
            check(saved.wrote, what + ' wrote a file', saved);
            await checkSaved(page, what + ' with the same camera names', want);
        }
        const shown = await page.evaluate(async () => {
            const { panes } = await window.__t.modules();
            await panes.switchSession(0);
            return window.__t.frameShows(2);
        });
        const wantShown = {};
        for (const r of rows[0].filter(x => x.frame === 2)) (wantShown[r.camera] = wantShown[r.camera] || []).push(r.pts[0][0]);
        for (const k in wantShown) wantShown[k].sort((a, b) => a - b);
        check(shown.session === 'sessA' && eq(shown.perCam, wantShown), 'each session shows its own poses', shown);
    });

    // ---- Python reads every save that wrote a file --------------------------------
    console.log('\nPython sleap-io ' + PY_SPEC);
    const readBack = JSON.parse(uv(PY_READ, pythonWork.map(w => w.file)).trim().split('\n').pop());
    for (const w of pythonWork) compare(w.what, readBack[w.file], w.want);
    if (sameTables) {
        const t = JSON.parse(uv(PY_SAME_TABLES, [sameTables.a, sameTables.b]).trim().split('\n').pop());
        check(t.names.length > 5 && t.different.length === 0,
            'keeping the sessions and evicting them wrote the same value in every table (' + t.names.length + ')', t.different);
    }
} finally {
    clearTimeout(stallGuard);
    await browser.close().catch(() => {});
    server.kill('SIGTERM');
    fs.rmSync(tmp, { recursive: true, force: true });
}

console.log(fails ? `\nFAIL (${fails})` : '\nPASS');
process.exit(fails ? 1 : 0);
