/**
 * eager-save-keeps-suggestions.mjs — a file's suggestions (`suggestions_json`, SLEAP's
 * frames to label next) stay on their own videos and frames through an eager open and
 * save. `tests/e2e/lazy-save-keeps-suggestions.mjs` is the same promise on the lazy route.
 *
 * `buildSlpBytes` used to build its `SIO.Labels` with no suggestions and no eager loader
 * kept them, so every save dropped them with no word. Handing the reader's along would
 * be worse: each points at the reader's `Video`, and the writer finds it in the header by
 * object. The header holds fresh camera videos, and a video it cannot find is written as
 * -1, which Python reads as the LAST video in the list: a suggestion on the wrong video,
 * no error. So this reads each save's raw rows as well as what Python makes of them.
 *
 * Real Python sleap-io writes the files and reads every save. A suggestion is named by
 * where it is: `cam:<camera>` on a camera's video, else `file:<filename>`.
 *
 *  A. Every layout through every eager route, two saves in a sitting, then the first
 *     save reopened through that route and saved again.
 *  B. A group in the file stays as the file holds it. A suggestion listed twice stays twice.
 *  C. Files with no session: a video no pose is on, and three videos with one file name.
 *  D. Per-camera folders, a single session and a parent folder of two.
 *  E. File ▸ Add SLP and "Add session folder": two files, whose video ids mean different
 *     videos, in one project.
 *  F. Edits: a corrected frame, an attached video, a renamed camera, a camera gone, a video
 *     moved to another session, a session deleted.
 *  G. A lazy session on the eager writer's path (a project mixing lazy and eager).
 *  H. A file with no suggestions gets none, and no extra video.
 *  I. Files with their images inside: the suggestions stay on those videos' entries.
 *
 * Needs `uv` for Python (skipped without it).
 *
 * Run: node tests/e2e/eager-save-keeps-suggestions.mjs   (spawns its own http.server; PORT= to
 * move it; SECTIONS=A,D, and in A ROUTES=file,folder and FILES=first,one, to run part of it)
 */
import { chromium } from 'playwright';
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8468);
const PY_SPEC = process.env.SLEAP_IO_SPEC || 'sleap-io==0.9.5';
const SECTIONS = process.env.SECTIONS ? process.env.SECTIONS.split(',') : null;
const runs = s => !SECTIONS || SECTIONS.includes(s);
const only = (list, env) => (process.env[env] ? list.filter(x => process.env[env].split(',').includes(x)) : list);

let fails = 0;
const check = (c, m, got) => {
    console.log((c ? '  ✓ ' : '  ✗ ') + m + (!c && got !== undefined ? ' — got ' + JSON.stringify(got) : ''));
    if (!c) fails++;
    stallGuard.refresh();
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const diff = (got, want) => {
    const g = new Set(got), w = new Set(want);
    return { missing: want.filter(x => !g.has(x)).slice(0, 4), extra: got.filter(x => !w.has(x)).slice(0, 4) };
};

// `listed` holds what a filename cannot show: each raw row's video index is one the list
// holds (a -1 reads in Python as the last video).
const PY_READ = `
import json, sys
import h5py
import numpy as np
import sleap_io as sio

def num(v):
    return "nan" if v != v else f"{float(v):.3f}"

def read(path):
    try:
        labels = sio.load_slp(path, open_videos=False)
        sio.load_slp(path)  # the default call opens the videos too
    except Exception as e:
        return {"error": type(e).__name__ + ": " + str(e)[:160]}
    with h5py.File(path, "r") as f:
        raw = [json.loads(x) for x in f["suggestions_json"][()]] if "suggestions_json" in f else []
    ids = [int(r["video"]) for r in raw]
    in_list = all(0 <= i < len(labels.videos) for i in ids)
    sessions, camera_of, session_of = [], {}, {}
    for si, s in enumerate(labels.sessions):
        cams = {}
        for c in s.camera_group.cameras:
            v = s.get_video(c)
            if v is not None:
                cams[c.name] = v.filename
                camera_of[id(v)] = c.name
                session_of[id(v)] = si
        sessions.append(cams)
    def where(v):
        return "cam:" + camera_of[id(v)] if id(v) in camera_of else "file:" + v.filename
    def pose_cells(inst):
        cells = []
        for k in range(len(inst.points)):
            x, y = inst.points["xy"][k]
            cells.append("-" if x != x or y != y else f"{num(x)},{num(y)}" + ("" if inst.points["visible"][k] else "~"))
        return "|".join(["P" if isinstance(inst, sio.PredictedInstance) else "U", inst.track.name if inst.track else "", ";".join(cells)])
    return {
        "error": None,
        "videos": [v.filename for v in labels.videos],
        "suggestions": [where(s.video) + "|" + str(s.frame_idx) + "|" + json.dumps(s.metadata.get("group")) for s in labels.suggestions],
        "byFile": [s.video.filename + "|" + str(s.frame_idx) + "|" + json.dumps(s.metadata.get("group")) for s in labels.suggestions],
        "onSession": [session_of.get(id(s.video), -1) for s in labels.suggestions],
        "listed": in_list,
        "rawIds": ids,
        "rawGroups": [r.get("group", "absent") for r in raw],
        "poses": sorted(where(lf.video) + "|" + str(lf.frame_idx) + "|" + pose_cells(i) for lf in labels.labeled_frames for i in lf.instances),
        "posesByFile": sorted(lf.video.filename + "|" + str(lf.frame_idx) + "|" + pose_cells(i) for lf in labels.labeled_frames for i in lf.instances),
        "sessions": sessions,
    }

print(json.dumps({p: read(p) for p in sys.argv[1:]}))
`;

const PY_WRITE = `
import json, os, shutil, sys
import h5py
import numpy as np
import sleap_io as sio
from sleap_io.model.camera import Instance3D
from sleap_io.model.suggestions import SuggestionFrame

out, packages = sys.argv[1], sys.argv[2]
sk = sio.Skeleton(nodes=["nose", "tail"], edges=[("nose", "tail")])
T1, T2 = sio.Track(name="mouse1"), sio.Track(name="mouse2")

def video(name):
    v = sio.Video(filename=name, open_backend=False)
    v.backend_metadata["shape"] = (400, 64, 64, 1)
    return v

def frames_on(v, base, n=3):
    return [sio.LabeledFrame(video=v, frame_idx=f, instances=[
        sio.PredictedInstance.from_numpy(np.array([[base + 10.0 * i + f, 10.0], [base + 20.0 * i + f, 20.0]]), skeleton=sk, track=t,
                                         score=0.9, point_scores=np.array([0.5, 0.8]))
        for i, t in enumerate((T1, T2))]) for f in range(n)]

# The raw worker stands in for the typed reader on what it can read in full, so its routes get
# each layout again, under "raw_", with no grouping or 3D (those are in /session_data).
PREFIX, GROUPED = "", True

def session(pairs):
    cams = [sio.Camera(name=n, rvec=np.array([0.0, 0.1 * i, 0.0]), tvec=np.array([10.0 * i, 0.0, 0.0]),
                       matrix=np.array([[500.0, 0, 32], [0, 500.0, 32], [0, 0, 1]]), dist=np.zeros(5), size=(64, 64))
            for i, (n, _, _) in enumerate(pairs)]
    s = sio.RecordingSession(camera_group=sio.CameraGroup(cameras=cams))
    for c, (_, v, _) in zip(cams, pairs):
        if v is not None:
            s.add_video(v, c)
    with_frames = [(c, {lf.frame_idx: lf for lf in fs}) for c, (_, _, fs) in zip(cams, pairs) if fs]
    for f in sorted(with_frames[0][1]) if with_frames and GROUPED else []:
        igs = [sio.InstanceGroup(instance_by_camera={c: by[f].instances[i] for c, by in with_frames}, score=0.5,
                                 instance_3d=Instance3D(points=np.array([[1.0 + i + f, 2.0, 3.0], [4.0, 5.0, 6.0 + i]]), skeleton=sk))
               for i in range(2)]
        s.frame_groups[f] = sio.FrameGroup(frame_idx=f, instance_groups=igs, labeled_frame_by_camera={c: by[f] for c, by in with_frames})
    return s

def sug(v, *frames):
    return [SuggestionFrame(video=v, frame_idx=f) for f in frames]

def save(name, frames, videos, sessions, suggestions):
    sio.Labels(labeled_frames=frames, videos=videos, skeletons=[sk], tracks=[T1, T2], sessions=sessions,
               suggestions=suggestions).save(os.path.join(out, PREFIX + name + ".slp"), embed=False)

def two_cams():
    a, b = video("camA.mp4"), video("camB.mp4")
    fa, fb = frames_on(a, 100), frames_on(b, 200)
    return a, b, fa, fb, session([("camA", a, fa), ("camB", b, fb)])

def layouts():
    # Only the cameras' videos. One suggestion on a frame with poses, the rest on frames with none.
    a, b, fa, fb, s = two_cams()
    save("cameras", fa + fb, [a, b], [s], sug(a, 10, 0) + sug(b, 20) + sug(a, 30))

    # A video no session names, listed last: an id past a one-video-per-camera header.
    a, b, fa, fb, s = two_cams()
    x = video("outside.mp4")
    save("last", fa + fb + frames_on(x, 500), [a, b, x], [s], sug(x, 40) + sug(a, 11) + sug(b, 21) + sug(x, 1))

    # Listed first: each id means another video in the save, whose order is [a, b, x].
    a, b, fa, fb, s = two_cams()
    x = video("outside.mp4")
    save("first", frames_on(x, 500) + fa + fb, [x, a, b], [s], sug(a, 12) + sug(x, 41) + sug(b, 22))

    # Two videos with no poses and no camera: only suggestions name them, and not in the
    # order the file lists them.
    a, b, fa, fb, s = two_cams()
    x, y, z = video("outside.mp4"), video("no_poses_y.mp4"), video("no_poses_z.mp4")
    save("only_suggested", fa + fb + frames_on(x, 500), [a, b, z, y, x], [s], sug(y, 50) + sug(a, 13) + sug(z, 51) + sug(x, 42) + sug(y, 52))

    # Two sessions and a video in neither. The other session's second camera has no poses.
    a, b, fa, fb, s1 = two_cams()
    c, d = video("camC.mp4"), video("camD.mp4")
    fc = frames_on(c, 300)
    s2 = session([("camC", c, fc), ("camD", d, [])])
    x = video("outside.mp4")
    save("two_sessions", frames_on(x, 500) + fa + fb + fc, [x, a, b, c, d], [s1, s2], sug(c, 60) + sug(a, 14) + sug(d, 61) + sug(x, 43) + sug(b, 23))

    # Cameras with no poses at all: their videos are in the header, on fresh objects.
    a, b, x = video("camA.mp4"), video("camB.mp4"), video("outside.mp4")
    save("all_outside", frames_on(x, 500), [a, b, x], [session([("camA", a, []), ("camB", b, [])])], sug(a, 15) + sug(x, 44))

    # A session that maps no video.
    a, b, x = video("camA.mp4"), video("camB.mp4"), video("third.mp4")
    save("no_map", frames_on(a, 100) + frames_on(b, 200) + frames_on(x, 500), [a, b, x], [session([("camA", None, []), ("camB", None, [])])], sug(a, 16) + sug(x, 45))

    # The same file name in three folders, as campy names every video.
    a, b, x = video("rec/cam0/0.mp4"), video("rec/cam1/0.mp4"), video("rec/other/0.mp4")
    fa, fb = frames_on(a, 100), frames_on(b, 200)
    save("same_name", frames_on(x, 500) + fa + fb, [x, a, b], [session([("camA", a, fa), ("camB", b, fb)])], sug(b, 24) + sug(x, 46) + sug(a, 17))

    # Exactly one.
    a, b, fa, fb, s = two_cams()
    save("one", fa + fb, [a, b], [s], sug(b, 25))

    # No suggestions: the control.
    a, b, fa, fb, s = two_cams()
    x = video("outside.mp4")
    save("none", fa + fb + frames_on(x, 500), [a, b, x], [s], [])
    a, b, fa, fb, s = two_cams()
    save("none_plain", fa + fb, [a, b], [s], [])

    # Groups as SLEAP files hold them: Python's writer keeps a group it read but writes
    # no other shape, so the rows are rewritten here. [x, a, b]: a is "1", b is "2", x is "0".
    a, b, fa, fb, s = two_cams()
    x = video("outside.mp4")
    save("groups", frames_on(x, 500) + fa + fb, [x, a, b], [s], [])
    rows = [
        {"video": "1", "frame_idx": 70, "group": 3},
        {"video": "2", "frame_idx": 71, "group": "night"},
        {"video": "0", "frame_idx": 72, "group": 0},
        {"video": "1", "frame_idx": 73, "group": None},
        {"video": "2", "frame_idx": 74},
        {"video": "2", "frame_idx": 75, "group": 1},
        {"video": "2", "frame_idx": 75, "group": 1},
    ]
    with h5py.File(os.path.join(out, PREFIX + "groups.slp"), "r+") as f:
        del f["suggestions_json"]
        f.create_dataset("suggestions_json", data=[json.dumps(r).encode() for r in rows])

layouts()
PREFIX, GROUPED = "raw_", False
layouts()
PREFIX, GROUPED = "", True

# No session at all: every video is a camera on File ▸ Open. A video no pose is on.
a, b = video("camA.mp4"), video("camB.mp4")
sio.Labels(labeled_frames=frames_on(a, 100), videos=[a, b], skeletons=[sk], tracks=[T1, T2],
           suggestions=sug(b, 26) + sug(a, 18)).save(os.path.join(out, "plain.slp"), embed=False)
# No session, and three videos with one file name.
a, b, x = video("rec/cam0/0.mp4"), video("rec/cam1/0.mp4"), video("rec/other/0.mp4")
sio.Labels(labeled_frames=frames_on(a, 100), videos=[a, b, x], skeletons=[sk], tracks=[T1, T2],
           suggestions=sug(x, 47) + sug(a, 19) + sug(b, 27)).save(os.path.join(out, "plain_same_name.slp"), embed=False)

# Per-camera files: frames on the camera's video and another; only on the second
# listed; a third video with only suggestions.
a, x, y = video("camA.mp4"), video("camA_second.mp4"), video("camA_no_poses.mp4")
sio.Labels(labeled_frames=frames_on(a, 100) + frames_on(x, 500), videos=[a, x, y], skeletons=[sk], tracks=[T1, T2],
           suggestions=sug(x, 80) + sug(a, 81) + sug(y, 82)).save(os.path.join(out, "camA_file.slp"), embed=False)
unused, b = video("camB_unused.mp4"), video("camB.mp4")
sio.Labels(labeled_frames=frames_on(b, 200), videos=[unused, b], skeletons=[sk], tracks=[T1, T2],
           suggestions=sug(b, 83) + sug(unused, 84)).save(os.path.join(out, "camB_file.slp"), embed=False)

# A second session, as per-camera files and as a project.
c, y = video("camC.mp4"), video("camC_second.mp4")
sio.Labels(labeled_frames=frames_on(c, 300) + frames_on(y, 600), videos=[c, y], skeletons=[sk], tracks=[T1, T2],
           suggestions=sug(c, 90) + sug(y, 91)).save(os.path.join(out, "camC_file.slp"), embed=False)
d = video("camD.mp4")
sio.Labels(labeled_frames=frames_on(d, 400), videos=[d], skeletons=[sk], tracks=[T1, T2],
           suggestions=sug(d, 92)).save(os.path.join(out, "camD_file.slp"), embed=False)
c, d = video("camC.mp4"), video("camD.mp4")
fc, fd = frames_on(c, 300), frames_on(d, 400)
y, n = video("second_outside.mp4"), video("second_no_poses.mp4")
save("second_project", frames_on(y, 600) + fc + fd, [y, n, c, d], [session([("camC", c, fc), ("camD", d, fd)])],
     sug(d, 93) + sug(n, 94) + sug(y, 95) + sug(c, 96))

# Images inside the file: the package fixtures, with a suggestion on each video.
for name in ("out_of_order_embedded", "mixed_external", "two_sessions"):
    path = os.path.join(out, "images_" + name + ".slp")
    shutil.copy(os.path.join(packages, name + ".pkg.slp"), path)
    with h5py.File(path, "r+") as f:
        rows = [{"video": str(i), "frame_idx": 5 + i, "group": i} for i in reversed(range(len(f["videos_json"])))]
        if "suggestions_json" in f:
            del f["suggestions_json"]
        f.create_dataset("suggestions_json", data=[json.dumps(r).encode() for r in rows])
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

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'eager-suggestions-'));
const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
for (let i = 0; i < 60; i++) {
    try { if ((await fetch(`http://localhost:${PORT}/index.html`)).ok) break; } catch { /* not yet */ }
    await new Promise(r => setTimeout(r, 250));
}
const browser = await chromium.launch();
// `page.evaluate` has no timeout, so a step waiting on a click nobody makes would sit
// silent for good. Closing the browser fails that step.
const stallGuard = setTimeout(() => {
    console.log('\nStuck: no check for 5 minutes');
    browser.close();
}, 5 * 60 * 1000);
stallGuard.unref();

async function newAppPage() {
    const page = await browser.newPage();
    page.on('pageerror', e => { console.log('  [pageerror]', String(e).slice(0, 300)); fails++; });
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state && window.SleapIO, null, { timeout: 60000 });
    return page;
}

const toB64 = bytes => Buffer.from(bytes).toString('base64');
async function putBytes(page, name, bytes) {
    await page.evaluate(([n, b64]) => { window[n] = Uint8Array.from(atob(b64), c => c.charCodeAt(0)); }, [name, toB64(bytes)]);
}

/** Answer the prompts a load puts up, until `window.__done` is set. */
async function settle(page, what) {
    let done = false;
    // By the clock, not a count of tries: at a high machine load each try takes seconds.
    for (const end = Date.now() + 240000; Date.now() < end && !done;) {
        const s = await page.evaluate(() => {
            const shown = [...document.querySelectorAll('button')].filter(x => x.offsetParent);
            // These folders hold no videos, so a load waits on the Missing Video Files popup.
            const b = shown.find(x => (x.textContent || '').trim() === 'Continue')
                || shown.find(x => /^(Skip|Later|Cancel|Not now)$/i.test((x.textContent || '').trim()));
            if (b) { b.click(); return { done: false }; }
            return { done: window.__done, prompt: !!document.getElementById('videoSelectOverlay') };
        });
        if (s.prompt) await page.keyboard.press('Escape');
        done = s.done;
        if (!done) await page.waitForTimeout(100);
    }
    if (done !== true) throw new Error(what + ': did not finish: ' + done);
}

// Ends in `undefined`: a promise as the last value would have this wait on the load, and the
// load waits on `settle` for its prompts.
const startLoad = (page, fn) => page.evaluate(`window.__done = false; (${fn})().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); }); undefined`);

function pickFolderOnce(page, bytes) {
    const dir = fs.mkdtempSync(path.join(tmp, 'folder-'));
    fs.writeFileSync(path.join(dir, 'project.slp'), bytes);
    page.once('filechooser', fc => fc.setFiles(dir));
}

// The typed reader failing sends these two routes to the raw worker, which reads the file itself.
const typedReaderOff = page => page.evaluate(() => {
    window.__typedReader = window.SleapIO.readSlpStreaming;
    window.SleapIO.readSlpStreaming = async () => { throw new Error('typed reader off, for this test'); };
});
const typedReaderOn = page => page.evaluate(() => { window.SleapIO.readSlpStreaming = window.__typedReader; });

const openFile = `async () => { await (await import('/import-export/slp-import.js')).handleLoadSlpFile(new File([window.__file], 'project.slp')); }`;
const openProject = `async () => { await (await import('/import-export/save-load.js')).handleLoadProject(new File([window.__file], 'project.slp')); }`;
const openFolder = `async () => { await (await import('/loading/session-loader.js')).handleLoadSessionFolderSingleSlp(); }`;
const openLazy = `async () => { await (await import('/loading/session-loader.js')).handleLoadProjectSlpLazy(new File([window.__file], 'project.slp')); }`;

const ROUTES = {
    file: async (page) => startLoad(page, openFile),
    project: async (page) => startLoad(page, openProject),
    folder: async (page, bytes) => { pickFolderOnce(page, bytes); return startLoad(page, openFolder); },
    rawFile: async (page) => { await typedReaderOff(page); return startLoad(page, openFile); },
    rawFolder: async (page, bytes) => { await typedReaderOff(page); pickFolderOnce(page, bytes); return startLoad(page, openFolder); },
    lazy: async (page) => startLoad(page, openLazy),
};

async function open(route, bytes) {
    const page = await newAppPage();
    await putBytes(page, '__file', bytes);
    await ROUTES[route](page, bytes);
    await settle(page, route + ' load');
    if (route.startsWith('raw')) await typedReaderOn(page);
    return page;
}

// A save holds what only the typed reader reads in full (`/session_data`), so the saves of
// the raw worker's routes reopen through the typed reader.
const reopenRoute = route => ({ rawFile: 'file', rawFolder: 'folder' }[route] || route);

async function addFile(page, how, name, bytes) {
    if (how === 'addSlp') {
        page.once('filechooser', fc => fc.setFiles(path.join(tmp, name + '.slp')));
        await startLoad(page, `async () => { await (await import('/import-export/slp-import.js')).handleAddSlp(); }`);
    } else {
        pickFolderOnce(page, bytes);
        await startLoad(page, openFolder);
    }
    await settle(page, how);
}

/** File ▸ Save, the real route. Returns the bytes; the page stays open for another save. */
async function save(page) {
    await page.evaluate(async () => {
        const sl = await import('/import-export/save-load.js');
        const orig = URL.createObjectURL;
        let blob = null;
        URL.createObjectURL = b => { blob = b; return orig.call(URL, b); };
        try { await sl.saveProjectSlp(); } finally { URL.createObjectURL = orig; }
        if (!blob) throw new Error('saveProjectSlp produced no file: ' + (document.getElementById('statusText') || {}).textContent);
        window.__saved = new Uint8Array(await blob.arrayBuffer());
    });
    const b64 = await page.evaluate(() => {
        const u = window.__saved;
        let s = '';
        for (let i = 0; i < u.length; i += 32768) s += String.fromCharCode.apply(null, u.subarray(i, i + 32768));
        return btoa(s);
    });
    return Buffer.from(b64, 'base64');
}

let savedSeq = 0;
const pythonWork = [];   // [{ file, what, want }] checked together: one Python run reads every save
function expectSave(bytes, what, want) {
    const file = path.join(tmp, 'saved-' + (++savedSeq) + '.slp');
    fs.writeFileSync(file, bytes);
    pythonWork.push({ file, what, want });
    stallGuard.refresh();
}

/** Only what `want` names is checked. Each row's video index being in the list always is. */
function compare(what, got, want) {
    if (got.error) { check(false, what + ': opens', got.error); return; }
    check(got.listed === true, what + ': each points at a video the list holds', { ids: got.rawIds, videos: got.videos });
    if (want.suggestions) check(eq(got.suggestions, want.suggestions), what + ': every suggestion on its own video and frame', { got: got.suggestions, want: want.suggestions });
    if (want.byFile) check(eq(got.byFile, want.byFile), what + ': every suggestion on its own video and frame, by file name', { got: got.byFile, want: want.byFile });
    if (want.onSession) check(eq(got.onSession, want.onSession), what + ': on the session its camera is in', got.onSession);
    if (want.rawGroups) check(eq(got.rawGroups, want.rawGroups), what + ': groups as the file held them', got.rawGroups);
    if (want.poses) check(eq(got.poses, [...want.poses].sort()), what + ': every pose on its own video', diff(got.poses || [], want.poses));
    if (want.posesByFile) check(eq(got.posesByFile, [...want.posesByFile].sort()), what + ': every pose on its own video, by file name', diff(got.posesByFile || [], want.posesByFile));
    if (want.sessions) check(eq(got.sessions, want.sessions), what + ': cameras and their videos', got.sessions);
    if (want.videos) check(eq(got.videos, want.videos), what + ': the video list', got.videos);
    if (want.onlyOnce) check(new Set(got.videos).size === got.videos.length, what + ': one entry per video', got.videos);
}

// What each layout's header lists after an eager save: every camera of an open session
// (session by session, each followed by the videos with poses it keeps), then those only a
// suggestion names.
const LAYOUTS = {
    cameras: { videos: ['camA.mp4', 'camB.mp4'] },
    one: { videos: ['camA.mp4', 'camB.mp4'] },
    last: { videos: ['camA.mp4', 'camB.mp4', 'outside.mp4'] },
    first: { videos: ['camA.mp4', 'camB.mp4', 'outside.mp4'] },
    only_suggested: { videos: ['camA.mp4', 'camB.mp4', 'outside.mp4', 'no_poses_y.mp4', 'no_poses_z.mp4'] },
    two_sessions: { videos: ['camA.mp4', 'camB.mp4', 'outside.mp4', 'camC.mp4', 'camD.mp4'] },
    all_outside: { videos: ['camA.mp4', 'camB.mp4', 'outside.mp4'] },
    no_map: { byFile: true, videos: ['camA.mp4', 'camB.mp4', 'third.mp4'] },
    same_name: { videos: ['rec/cam0/0.mp4', 'rec/cam1/0.mp4', 'rec/other/0.mp4'] },
};
const firstSessionOnly = { folder: true, rawFolder: true };
/** The copy of a layout a route reads: see "raw_" in the Python above. */
const sourceFor = (name, route) => (route.startsWith('raw') ? 'raw_' + name : name);

/** `where` lines of `source`, as a route that opens only the file's first session writes them. */
function closedSessionsAsFiles(source, lines) {
    const closed = new Map(source.sessions.slice(1).flatMap(cams => Object.entries(cams)));
    return lines.map((r) => {
        const m = /^cam:([^|]*)\|/.exec(r);
        return m && closed.has(m[1]) ? 'file:' + closed.get(m[1]) + r.slice(m[0].length - 1) : r;
    });
}

function wantFor(source, name, route) {
    const layout = LAYOUTS[name];
    const want = { videos: layout.videos };
    if (layout.byFile) return { ...want, byFile: source.byFile, posesByFile: source.posesByFile };
    const lines = firstSessionOnly[route] ? closedSessionsAsFiles.bind(null, source) : x => x;
    want.suggestions = lines(source.suggestions);
    want.poses = lines(source.poses);
    want.sessions = firstSessionOnly[route] ? source.sessions.slice(0, 1) : source.sessions;
    return want;
}

try {
    uv(PY_WRITE, [tmp, path.join(repoRoot, 'tests', 'fixtures', 'slp-embedded')]);
    const eachLayout = [...Object.keys(LAYOUTS), 'none', 'none_plain', 'groups'];
    const names = [...eachLayout, ...eachLayout.map(n => 'raw_' + n), 'plain', 'plain_same_name', 'camA_file', 'camB_file', 'camC_file', 'camD_file', 'second_project', 'images_two_sessions'];
    const rawSources = JSON.parse(uv(PY_READ, names.map(n => path.join(tmp, n + '.slp'))).trim().split('\n').pop());
    const sources = {};
    for (const n of names) sources[n] = rawSources[path.join(tmp, n + '.slp')];
    const readFile = n => fs.readFileSync(path.join(tmp, n + '.slp'));

    console.log('Sources, as Python reads them:');
    check(sources.last.suggestions.length === 4 && sources.first.suggestions[0].startsWith('cam:camA|12|'), 'the files hold suggestions, each on its video', sources.first.suggestions);
    check(sources.none.suggestions.length === 0, 'the control holds none');
    check(sources.groups.rawGroups.length === 7, 'the group file is as written', sources.groups.rawGroups);

    if (runs('A')) {
        console.log('\nA. Every layout, every eager route, two saves in a sitting, then reopened and saved');
        for (const route of only(['file', 'project', 'folder', 'rawFile', 'rawFolder'], 'ROUTES')) {
            for (const name of only(Object.keys(LAYOUTS), 'FILES')) {
                const want = wantFor(sources[sourceFor(name, route)], name, route);
                const page = await open(route, readFile(sourceFor(name, route)));
                const first = await save(page);
                expectSave(first, `${route}, ${name}, save 1`, want);
                expectSave(await save(page), `${route}, ${name}, save 2 in the same sitting`, want);
                await page.close();
                const again = await open(reopenRoute(route), first);
                expectSave(await save(again), `${route}, ${name}, reopened and saved`, want);
                await again.close();
            }
        }
    }

    if (runs('B')) {
        console.log('\nB. Groups, and a suggestion listed twice');
        // In [x, a, b] of the source; the save lists [a, b, x].
        const want = {
            suggestions: [
                'cam:camA|70|3', 'cam:camB|71|"night"', 'file:outside.mp4|72|0', 'cam:camA|73|0', 'cam:camB|74|0', 'cam:camB|75|1', 'cam:camB|75|1',
            ],
            rawGroups: [3, 'night', 0, 0, 0, 1, 1],
            videos: ['camA.mp4', 'camB.mp4', 'outside.mp4'],
        };
        for (const route of ['file', 'folder', 'rawFile']) {
            const page = await open(route, readFile(sourceFor('groups', route)));
            const first = await save(page);
            expectSave(first, `groups, ${route}, save 1`, want);
            await page.close();
            const again = await open(reopenRoute(route), first);
            expectSave(await save(again), `groups, ${route}, reopened and saved`, want);
            await again.close();
        }
    }

    if (runs('C')) {
        console.log('\nC. Files with no session');
        for (const route of ['file', 'folder', 'rawFile']) {
            const page = await open(route, readFile('plain'));
            const first = await save(page);
            const want = { byFile: sources.plain.byFile, posesByFile: sources.plain.posesByFile };
            expectSave(first, `no session, ${route}, save 1`, want);
            await page.close();
            const again = await open(reopenRoute(route), first);
            expectSave(await save(again), `no session, ${route}, reopened and saved`, want);
            await again.close();
        }
        {
            // Each video is a camera, named by its folder.
            const page = await open('file', readFile('plain_same_name'));
            const first = await save(page);
            const want = {
                suggestions: ['cam:other|47|0', 'cam:cam0|19|0', 'cam:cam1|27|0'],
                videos: ['rec/cam0/0.mp4', 'rec/cam1/0.mp4', 'rec/other/0.mp4'],
            };
            expectSave(first, 'no session, one file name, save 1', want);
            await page.close();
            const again = await open('file', first);
            expectSave(await save(again), 'no session, one file name, reopened and saved', want);
            await again.close();
        }
    }

    if (runs('D')) {
        console.log('\nD. Per-camera folders');
        const camAB = {
            suggestions: ['file:camA_second.mp4|80|0', 'cam:camA|81|0', 'file:camA_no_poses.mp4|82|0', 'cam:camB|83|0', 'file:camB_unused.mp4|84|0'],
            videos: ['camA.mp4', 'camB.mp4', 'camA_second.mp4', 'camA_no_poses.mp4', 'camB_unused.mp4'],
        };
        {
            const page = await newAppPage();
            await putBytes(page, '__camA', readFile('camA_file'));
            await putBytes(page, '__camB', readFile('camB_file'));
            await startLoad(page, `async () => {
                const loader = await import('/loading/session-loader.js');
                const mkFile = (bytes, rel) => {
                    const f = new File([bytes], rel.split('/').pop());
                    Object.defineProperty(f, 'webkitRelativePath', { value: rel });
                    return f;
                };
                await loader.handleLoadSessionFolderPerCamera([
                    mkFile(window.__camA, 'root/camA/camA.slp'), mkFile(window.__camB, 'root/camB/camB.slp'),
                ], true);
            }`);
            await settle(page, 'per-camera load');
            check(await page.evaluate(() => !window.__lucid.state.session.lazyLoader), 'the folder opened eagerly');
            const first = await save(page);
            expectSave(first, 'per-camera, save 1', camAB);
            expectSave(await save(page), 'per-camera, save 2 in the same sitting', camAB);
            await page.close();
            const again = await open('file', first);
            expectSave(await save(again), 'per-camera, saved again as a project', camAB);
            await again.close();
        }
        {
            const page = await newAppPage();
            for (const n of ['camA', 'camB', 'camC', 'camD']) await putBytes(page, '__' + n, readFile(n + '_file'));
            await startLoad(page, `async () => {
                const loader = await import('/loading/session-loader.js');
                const dir = (name, entries) => ({
                    kind: 'directory', name,
                    [Symbol.asyncIterator]: async function* () { for (const e of entries) yield [e.name, e]; },
                });
                const file = (name, bytes) => ({ kind: 'file', name, getFile: async () => new File([bytes], name) });
                const cam = n => dir(n, [file(n + '.slp', window['__' + n])]);
                window.showDirectoryPicker = async () => dir('parent', [
                    dir('sessA', [cam('camA'), cam('camB')]), dir('sessB', [cam('camC'), cam('camD')]),
                ]);
                await loader.handleLoadMultiSession();
            }`);
            await settle(page, 'multi-session load');
            check(await page.evaluate(() => window.__lucid.state.sessions.length === 2 && window.__lucid.state.sessions.every(s => !s.lazyLoader)), 'two sessions, both eager');
            const want = {
                suggestions: [...camAB.suggestions, 'cam:camC|90|0', 'file:camC_second.mp4|91|0', 'cam:camD|92|0'],
                videos: ['camA.mp4', 'camB.mp4', 'camA_second.mp4', 'camC.mp4', 'camD.mp4', 'camC_second.mp4', 'camA_no_poses.mp4', 'camB_unused.mp4'],
                onSession: [-1, 0, -1, 0, -1, 1, -1, 1],
            };
            const first = await save(page);
            expectSave(first, 'a parent folder of two sessions, save 1', want);
            expectSave(await save(page), 'a parent folder of two sessions, save 2 in the same sitting', want);
            await page.close();
            // As a project, the first session holds every video with poses and no camera.
            const again = await open('file', first);
            expectSave(await save(again), 'a parent folder of two sessions, saved again as a project', {
                ...want,
                videos: ['camA.mp4', 'camB.mp4', 'camA_second.mp4', 'camC_second.mp4', 'camC.mp4', 'camD.mp4', 'camA_no_poses.mp4', 'camB_unused.mp4'],
            });
            await again.close();
        }
    }

    if (runs('E')) {
        console.log('\nE. Two files in one project');
        const firstRows = sources.first.suggestions;
        const secondRows = sources.second_project.suggestions;
        {
            const page = await open('file', readFile('first'));
            await addFile(page, 'addSlp', 'second_project');
            const want = { suggestions: [...firstRows, ...secondRows] };
            expectSave(await save(page), 'File ▸ Add SLP, save 1', want);
            const bytes = await save(page);
            expectSave(bytes, 'File ▸ Add SLP, save 2 in the same sitting', want);
            await page.close();
            const again = await open('file', bytes);
            expectSave(await save(again), 'File ▸ Add SLP, reopened and saved', want);
            await again.close();
        }
        {
            // The camera names are taken, so the added file's get a suffix.
            const page = await open('file', readFile('cameras'));
            await addFile(page, 'addSlp', 'cameras');
            const renamed = sources.cameras.suggestions.map(r => r.replace(/^cam:(cam[AB])\|/, 'cam:$1_2|'));
            expectSave(await save(page), 'File ▸ Add SLP, the same cameras', { suggestions: [...sources.cameras.suggestions, ...renamed] });
            await page.close();
        }
        {
            // The same id means another video in each: a place found by id would merge them.
            const page = await open('folder', readFile('first'));
            await addFile(page, 'folder', 'second_project', readFile('second_project'));
            const want = {
                suggestions: [...firstRows, ...secondRows],
                videos: ['camA.mp4', 'camB.mp4', 'outside.mp4', 'camC.mp4', 'camD.mp4', 'second_outside.mp4', 'second_no_poses.mp4'],
            };
            const first = await save(page);
            expectSave(first, 'a session folder added to an open project, save 1', want);
            await page.close();
            const again = await open('file', first);
            expectSave(await save(again), 'a session folder added to an open project, reopened and saved', {
                ...want,
                videos: ['camA.mp4', 'camB.mp4', 'outside.mp4', 'second_outside.mp4', 'camC.mp4', 'camD.mp4', 'second_no_poses.mp4'],
            });
            await again.close();
        }
        {
            const page = await open('lazy', readFile('first'));
            check(await page.evaluate(() => !!window.__lucid.state.session.lazyLoader), 'the project opened lazily');
            await addFile(page, 'addSlp', 'second_project');
            expectSave(await save(page), 'File ▸ Add SLP on a lazy project', { suggestions: [...firstRows, ...secondRows] });
            await page.close();
        }
    }

    if (runs('F')) {
        console.log('\nF. Edits');
        {
            const page = await open('file', readFile('first'));
            await page.evaluate(async () => {
                const { state } = await import('/ui/app-state.js');
                const group = state.session.instanceGroups.get(1)[0];
                const inst = group.instances.get('camA');
                inst.setPoint(0, 999, 999);
                inst.type = 'user';
                inst.modified = true;
                group.markDirty();
            });
            expectSave(await save(page), 'a corrected frame', { suggestions: sources.first.suggestions });
            await page.close();
        }
        {
            const page = await open('file', readFile('first'));
            await page.evaluate(async () => {
                const { state } = await import('/ui/app-state.js');
                state.videoFiles = [{ name: 'camA', assignedCamera: 'camA', videoPath: '/moved/here/camA_new.mp4' }];
            });
            expectSave(await save(page), 'a video attached under another name', {
                suggestions: sources.first.suggestions,
                byFile: sources.first.byFile.map(r => r.replace(/^camA\.mp4\|/, '/moved/here/camA_new.mp4|')),
                videos: ['/moved/here/camA_new.mp4', 'camB.mp4', 'outside.mp4'],
            });
            await page.close();
        }
        {
            const page = await open('file', readFile('first'));
            await page.evaluate(async () => {
                const { state } = await import('/ui/app-state.js');
                state.session.cameras[0].name = 'top';
                state.session.renameCameraInAllData('camA', 'top');
            });
            expectSave(await save(page), 'a renamed camera', {
                suggestions: sources.first.suggestions.map(r => r.replace(/^cam:camA\|/, 'cam:top|')),
            });
            await page.close();
        }
        {
            // A calibration without this camera: its video stays in the file for its suggestions.
            // (camB is what stays: on camA alone, at the origin, a save calls the session
            // uncalibrated and brings the other camera back as a placeholder.)
            const page = await open('file', readFile('first'));
            await page.evaluate(async () => {
                const { state } = await import('/ui/app-state.js');
                state.session.cameras = state.session.cameras.filter(c => c.name !== 'camA');
            });
            expectSave(await save(page), 'a camera gone', {
                suggestions: sources.first.suggestions.map(r => r.replace(/^cam:camA\|/, 'file:camA.mp4|')),
                videos: ['camB.mp4', 'outside.mp4', 'camA.mp4'],
            });
            await page.close();
        }
        {
            const page = await open('file', readFile('two_sessions'));
            await page.evaluate(async () => {
                const { state } = await import('/ui/app-state.js');
                const panes = await import('/ui/sessions-panes.js');
                state.activeSessionIdx = 0;
                state.session = state.sessions[0];
                const moved = panes.showMoveVideoModal(['camB'], 0, 1);
                [...document.querySelectorAll('button')].find(b => b.offsetParent && b.textContent.trim() === 'Continue').click();
                await moved;
            });
            expectSave(await save(page), 'a video moved to another session', {
                suggestions: sources.two_sessions.suggestions,
                onSession: [1, 0, 1, -1, 1],
            });
            await page.close();
        }
        {
            const page = await open('file', readFile('two_sessions'));
            page.on('dialog', d => d.accept());
            await page.evaluate(async () => {
                const panes = await import('/ui/sessions-panes.js');
                panes.removeSession(1);
            });
            expectSave(await save(page), 'a session deleted', {
                suggestions: sources.two_sessions.suggestions.filter(r => !/^cam:cam[CD]\|/.test(r)),
                videos: ['camA.mp4', 'camB.mp4', 'outside.mp4'],
            });
            await page.close();
        }
        {
            // A save does not copy embedded images, so such a video is replaced by its source
            // video, and two records can share one. No fixture has that, so this calls the
            // writer's step directly.
            const page = await newAppPage();
            const r = await page.evaluate(async () => {
                const { suggestionsForSave } = await import('/import-export/kept-suggestions.js');
                const SIO = window.SleapIO;
                const source = new SIO.Video({ filename: 'source.mp4', openBackend: false });
                const embedded = () => ({ typed: new SIO.Video({ filename: 'pkg.slp', embedded: true, sourceVideo: source, openBackend: false }) });
                const records = [embedded(), embedded()];
                const row = (rec, f) => ({ camera: null, video: { id: 0, record: rec, typed: rec.typed }, frameIdx: f, group: 0, order: f });
                const allVideos = [];
                const frames = suggestionsForSave([{
                    session: { cameras: [], keptSuggestions: [row(records[0], 1), row(records[1], 2)] }, cameraVideos: [],
                }], allVideos);
                return { videos: allVideos.map(v => v.filename), on: frames.map(f => allVideos.indexOf(f.video)) };
            });
            check(eq(r, { videos: ['source.mp4'], on: [0, 0] }), 'two embedded records of one source video: one entry', r);
            await page.close();
        }
    }

    if (runs('G')) {
        console.log('\nG. A lazy session on the eager writer\'s path');
        for (const name of ['first', 'only_suggested']) {
            const page = await open('lazy', readFile(name));
            await page.evaluate(async () => {
                const tri = await import('/pose/triangulation.js');
                for (const f of [0, 1, 2]) await tri.ensureLazyFrameData(f);
            });
            await addFile(page, 'folder', 'second_project', readFile('second_project'));
            check(await page.evaluate(() => {
                const s = window.__lucid.state.sessions;
                return s.length === 2 && !!s[0].lazyLoader && !s[1].lazyLoader;
            }), name + ': one lazy session, one eager');
            const onlySuggested = LAYOUTS[name].videos.filter(v => v.startsWith('no_poses'));
            const want = {
                suggestions: [...sources[name].suggestions, ...sources.second_project.suggestions],
                videos: ['camA.mp4', 'camB.mp4', 'outside.mp4', 'camC.mp4', 'camD.mp4', 'second_outside.mp4', ...onlySuggested, 'second_no_poses.mp4'],
            };
            const first = await save(page);
            expectSave(first, name + ', mixed, save 1', want);
            expectSave(await save(page), name + ', mixed, save 2 in the same sitting', want);
            await page.close();
            const again = await open('file', first);
            expectSave(await save(again), name + ', mixed, reopened and saved', {
                ...want,
                videos: ['camA.mp4', 'camB.mp4', 'outside.mp4', 'second_outside.mp4', 'camC.mp4', 'camD.mp4', ...onlySuggested, 'second_no_poses.mp4'],
            });
            await again.close();
        }
    }

    if (runs('H')) {
        console.log('\nH. No suggestions: none, and no extra video');
        for (const route of ['file', 'project', 'folder', 'rawFile']) {
            for (const [name, videos] of [['none', ['camA.mp4', 'camB.mp4', 'outside.mp4']], ['none_plain', ['camA.mp4', 'camB.mp4']]]) {
                const source = sourceFor(name, route);
                const page = await open(route, readFile(source));
                expectSave(await save(page), `${route}, ${name}`, { suggestions: [], videos, poses: sources[source].poses, sessions: sources[source].sessions });
                await page.close();
            }
        }
    }

    if (runs('I')) {
        console.log('\nI. Files with their images inside');
        const inFile = {
            out_of_order_embedded: { suggestions: ['cam:clip_c|7|2', 'cam:clip_b|6|1', 'cam:clip_a|5|0'], videos: ['clip_a.mp4', 'clip_b.mp4', 'clip_c.mp4'] },
            mixed_external: { suggestions: ['cam:outside|6|1', 'cam:inside|5|0'], videos: ['inside.mp4', 'outside.mp4'] },
            two_sessions: { suggestions: sources.images_two_sessions.suggestions },
        };
        for (const [name, want] of Object.entries(inFile)) {
            const page = await open('file', readFile('images_' + name));
            const first = await save(page);
            expectSave(first, `images inside, ${name}, save 1`, want);
            expectSave(await save(page), `images inside, ${name}, save 2 in the same sitting`, want);
            await page.close();
            const again = await open('file', first);
            expectSave(await save(again), `images inside, ${name}, reopened and saved`, want);
            await again.close();
        }
        // The loader opens the first session only. The others' videos have poses, and images
        // a save does not copy, so the suggestions and the poses share a placeholder's entry.
        const page = await open('folder', readFile('images_two_sessions'));
        const first = await save(page);
        const want = {
            suggestions: ['file:unknown_video_3.mp4|8|3', 'file:unknown_video_2.mp4|7|2', 'cam:south|6|1', 'cam:north|5|0'],
            onlyOnce: true,
        };
        expectSave(first, 'images inside, the first session of two, save 1', want);
        await page.close();
        const again = await open('folder', first);
        expectSave(await save(again), 'images inside, the first session of two, reopened and saved', want);
        await again.close();
    }

    console.log('\nPython reads every save:');
    if (pythonWork.length > 0) {
        const readBack = JSON.parse(uv(PY_READ, pythonWork.map(w => w.file)).trim().split('\n').pop());
        for (const w of pythonWork) compare(w.what, readBack[w.file], w.want);
    }
} finally {
    await browser.close();
    server.kill();
    fs.rmSync(tmp, { recursive: true, force: true });
}
console.log(fails ? `\n${fails} check(s) FAILED` : '\nAll checks passed');
process.exit(fails ? 1 : 0);
