/**
 * lazy-keeps-other-sessions.mjs — the sessions a lazy open does not open stay in the
 * file: their cameras, groups, 3D, calibration, identities and every key LUC3D does not
 * read. The promise in the header of `import-export/unopened-sessions.js`.
 *
 * File ▸ Open sends every `.slp` over 200 MB to `handleLoadProjectSlpLazy`, which opens
 * the first recording session. A save used to keep the others' poses but write only the
 * first session, so the rest were gone from the file for good.
 *
 * Real Python sleap-io writes the files. Each goes through the lazy route, File ▸ Save,
 * the lazy route again and File ▸ Save again. Every save is read with the app's own
 * reader and with Python, and both must report what the source holds.
 *
 *  A. Every layout, two saves (see `FILES`): frames in any order, an open session with
 *     no frame, a (video, frame) listed twice, every kind of thing a session can hold,
 *     identities on poses, the same as an old file holds it inline, a video two sessions
 *     share. A first session with no camera or no video map, where the route differs.
 *  B. The real route: File ▸ Open on a file over 200 MB.
 *  C. Edits on the open session: a corrected frame goes first in the file (every other
 *     frame's position moves), a deleted pose compacts the store, a deleted track
 *     rewrites the track list, Track All and Triangulate All. Saving twice in one
 *     sitting. A renamed identity.
 *  D. A video two sessions share: a corrected or deleted pose on it.
 *  E. A LUC3D project: everything LUC3D keeps per session survives lazy saves and comes
 *     back when the file is opened eagerly. Project-scoped planes follow the open session.
 *  F. Per-session provenance.
 *  G. A project of two lazy files (two passes), and a project mixing lazy and eager.
 *  H. What is resident after the load: columns, not the reader's objects.
 *
 * Needs `uv` for Python (skipped without it).
 *
 * Run: node tests/e2e/lazy-keeps-other-sessions.mjs   (spawns its own http.server; PORT= to move it)
 *      KEEP_DIR=<dir> also writes every save there (for scripts/validate_slp_sleap_compat.py).
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
const PORT = Number(process.env.PORT || 8467);
const PY_SPEC = process.env.SLEAP_IO_SPEC || 'sleap-io==0.9.5';

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

// One line per pose: video|frame|U or P|track|nodes|scores. Each video's x values are its
// own, so a group member that lands on another video's pose cannot pass for its own.
// One line per group: session|frame|camera=x of its pose...|3D. The rich form adds what
// only a closed session keeps as the file had it (LUC3D rewrites the open session's own).
const PY_COMMON = `
import json, os, sys
import numpy as np
import sleap_io as sio

def num(v):
    return "nan" if v != v else f"{float(v):.3f}"

def pose_row(video, frame_idx, inst):
    predicted = isinstance(inst, sio.PredictedInstance)
    cells = []
    for k in range(len(inst.points)):
        x, y = inst.points["xy"][k]
        cells.append("-" if x != x or y != y else f"{num(x)},{num(y)}" + ("" if inst.points["visible"][k] else "~"))
    scores = [num(inst.score)] + [num(s) for s in inst.points["score"]] if predicted else []
    return "|".join([video, str(frame_idx), "P" if predicted else "U", inst.track.name if inst.track else "", ";".join(cells), ",".join(scores)])

def camera_row(c):
    return {
        "name": c.name,
        "rvec": [round(float(v), 6) for v in c.rvec], "tvec": [round(float(v), 6) for v in c.tvec],
        "matrix": [[round(float(v), 6) for v in r] for r in c.matrix], "dist": [round(float(v), 6) for v in c.dist],
        "size": list(c.size) if c.size is not None else None, "metadata": c.metadata,
    }

def read(path):
    try:
        labels = sio.load_slp(path, open_videos=False)
        sio.load_slp(path)  # the default call opens the videos too
    except Exception as e:
        return {"error": type(e).__name__ + ": " + str(e)[:160]}
    poses = sorted(pose_row(lf.video.filename, lf.frame_idx, i) for lf in labels.labeled_frames for i in lf.instances)
    pose_identities = sorted(lf.video.filename + "|" + str(lf.frame_idx) + "|" + num(i.points["xy"][0][0]) + "|" + i.identity.name
                             for lf in labels.labeled_frames for i in lf.instances if getattr(i, "identity", None) is not None)
    sessions, groups, rich, details = [], [], [], []
    for si, s in enumerate(labels.sessions):
        cams = {}
        for c in s.camera_group.cameras:
            v = s.get_video(c)
            if v is not None:
                cams[c.name] = v.filename
        sessions.append(cams)
        details.append({
            "cameras": [camera_row(c) for c in s.camera_group.cameras],
            "cameraGroupMetadata": s.camera_group.metadata,
            "metadata": s.metadata,
        })
        for f, fg in s.frame_groups.items():
            for ig in fg.instance_groups:
                members = sorted(c.name + "=" + num(i.points["xy"][0][0]) for c, i in ig.instance_by_camera.items())
                i3d = ig.instance_3d
                p3 = "" if i3d is None else ",".join(num(v) for v in np.asarray(i3d.points)[0][:3])
                groups.append("|".join(["s" + str(si), str(f)] + members + [p3]))
                every3d = "" if i3d is None else ";".join(",".join(num(v) for v in row) for row in np.asarray(i3d.points))
                point_scores = getattr(i3d, "point_scores", None)
                rich.append("|".join(["s" + str(si), str(f)] + members + [
                    every3d,
                    "score=" + ("" if ig.score is None else num(ig.score)),
                    "s3d=" + ("" if i3d is None or i3d.score is None else num(i3d.score)),
                    "ps=" + ("" if point_scores is None else ",".join(num(v) for v in point_scores)),
                    "id=" + ("" if ig.identity is None else ig.identity.name),
                    "ig=" + json.dumps(ig.metadata, sort_keys=True),
                    "fg=" + json.dumps(fg.metadata, sort_keys=True),
                ]))
    # Which sessions' groups name one catalog entry: what says they are one animal.
    shared = {}
    for si, s in enumerate(labels.sessions):
        for fg in s.frame_groups.values():
            for ig in fg.instance_groups:
                if ig.identity is not None:
                    shared.setdefault(str(labels.identities.index(ig.identity)), set()).add(si)
    return {
        "error": None, "videos": [v.filename for v in labels.videos], "tracks": [t.name for t in labels.tracks],
        "videoShapes": {v.filename: list(v.backend_metadata.get("shape") or []) for v in labels.videos},
        "identities": [i.name for i in labels.identities],
        "identityOwners": sorted(sorted(v) for v in shared.values()),
        "poseIdentities": pose_identities,
        "provenance": {k: v for k, v in labels.provenance.items() if k not in ("source", "exported_at", "sleap_version", "filename")},
        "sessions": sessions, "details": details, "poses": poses, "groups": sorted(groups), "rich": sorted(rich),
    }
`;

const PY_READ = PY_COMMON + `
print(json.dumps({p: read(p) for p in sys.argv[1:]}))
`;

const PY_WRITE = PY_COMMON + `
from sleap_io.model.instance import Instance3D, PredictedInstance3D
from sleap_io.io.slp import session_to_dict
import h5py, shutil

out = sys.argv[1]
sk = sio.Skeleton(nodes=["nose", "tail"], edges=[("nose", "tail")])
T1, T2 = sio.Track(name="mouse1"), sio.Track(name="mouse2")

def video(name):
    v = sio.Video(filename=name, open_backend=False)
    v.backend_metadata["shape"] = (400, 64, 64, 1)
    return v

def frames_on(v, base, n=3, first=0):
    return [sio.LabeledFrame(video=v, frame_idx=f, instances=[
        sio.PredictedInstance.from_numpy(np.array([[base + 10.0 * i + f, 10.0], [base + 20.0 * i + f, 20.0]]), skeleton=sk, track=t,
                                         score=0.9, point_scores=np.array([0.5 + 0.1 * i, 0.8]))
        for i, t in enumerate((T1, T2))]) for f in range(first, first + n)]

def cameras(names, shift=0.0):
    return [sio.Camera(name=n, rvec=np.array([0.0, 0.1 * i + shift, 0.0]), tvec=np.array([10.0 * i + shift, 0.0, 0.0]),
                       matrix=np.array([[500.0 + shift, 0, 32], [0, 500.0, 32], [0, 0, 1]]), dist=np.zeros(5), size=(64, 64))
            for i, n in enumerate(names)]

def session(pairs, shift=0.0):
    cams = cameras([p[0] for p in pairs], shift)
    s = sio.RecordingSession(camera_group=sio.CameraGroup(cameras=cams))
    for c, (_, v, _) in zip(cams, pairs):
        if v is not None:
            s.add_video(v, c)
    # Every frame of the session is grouped across its cameras, with 3D.
    with_frames = [(c, {lf.frame_idx: lf for lf in fs}) for c, (_, _, fs) in zip(cams, pairs) if fs]
    for f in sorted(with_frames[0][1]) if with_frames else []:
        igs = [sio.InstanceGroup(instance_by_camera={c: by[f].instances[i] for c, by in with_frames}, score=0.5,
                                 instance_3d=Instance3D(points=np.array([[1.0 + i + f + shift, 2.0, 3.0], [4.0, 5.0, 6.0 + i]]), skeleton=sk))
               for i in range(2)]
        s.frame_groups[f] = sio.FrameGroup(frame_idx=f, instance_groups=igs, labeled_frame_by_camera={c: by[f] for c, by in with_frames})
    return s

def save(name, frames, videos, sessions, **kw):
    sio.Labels(labeled_frames=frames, videos=videos, skeletons=[sk], tracks=[T1, T2], sessions=sessions, **kw).save(os.path.join(out, name), embed=False)

def two_sessions_parts():
    a, b, c, d, x = video("camA.mp4"), video("camB.mp4"), video("camC.mp4"), video("camD.mp4"), video("outside.mp4")
    fa, fb, fc, fd, fx = frames_on(a, 100), frames_on(b, 200), frames_on(c, 300), frames_on(d, 400), frames_on(x, 500)
    return a, b, c, d, x, fa, fb, fc, fd, fx

# The file the report names: two sessions and a video in neither, that one first.
a, b, c, d, x, fa, fb, fc, fd, fx = two_sessions_parts()
save("two_sessions.slp", fx + fa + fb + fc + fd, [x, a, b, c, d],
     [session([("camA", a, fa), ("camB", b, fb)]), session([("camC", c, fc), ("camD", d, fd)], 7.0)])

# Interleaved frames, and the closed session's frames before the open one's.
a, b, c, d, x, fa, fb, fc, fd, fx = two_sessions_parts()
save("closed_first.slp", fc + fa + fd + fb, [c, a, d, b],
     [session([("camA", a, fa), ("camB", b, fb)]), session([("camC", c, fc), ("camD", d, fd)], 7.0)])

# The open session has cameras and no frame, or no camera at all: all the file's rows
# are another session's, and a save has no camera row to count them from.
a, b, c, d, x, fa, fb, fc, fd, fx = two_sessions_parts()
save("first_empty.slp", fc + fd, [a, b, c, d],
     [session([("camA", a, []), ("camB", b, [])]), session([("camC", c, fc), ("camD", d, fd)], 7.0)])
a, b, c, d, x, fa, fb, fc, fd, fx = two_sessions_parts()
save("first_no_cameras.slp", fc + fd, [c, d],
     [sio.RecordingSession(camera_group=sio.CameraGroup(cameras=[])), session([("camC", c, fc), ("camD", d, fd)], 7.0)])
a, b, c, d, x, fa, fb, fc, fd, fx = two_sessions_parts()
save("first_no_map.slp", fa + fb + fc + fd, [a, b, c, d],
     [session([("camA", None, []), ("camB", None, [])]), session([("camC", c, fc), ("camD", d, fd)], 7.0)])

# One (video, frame) listed twice, before every closed session's rows: a save that drops
# or miscounts it sends every later group member to the wrong pose.
a, b, c, d, x, fa, fb, fc, fd, fx = two_sessions_parts()
save("dup_rows.slp", fa + frames_on(a, 700, n=1) + fb + fc + fd, [a, b, c, d],
     [session([("camA", a, fa), ("camB", b, fb)]), session([("camC", c, fc), ("camD", d, fd)], 7.0)])

# Every kind of thing a session can hold, in three sessions.
def rich_parts():
    ida = sio.Identity(name="A")
    idb = sio.Identity(name="B")
    idc = sio.Identity(name="C", metadata={"color": "#123456", "strain": "C57"})
    a, b, c, d, e, unused = video("camA.mp4"), video("camB.mp4"), video("camC.mp4"), video("camD.mp4"), video("camE.mp4"), video("no_frames.mp4")
    fa, fb, fc, fd = frames_on(a, 100), frames_on(b, 200), frames_on(c, 300), frames_on(d, 400)
    fe = frames_on(e, 600, n=2, first=20)
    # The open session: its groups name A and B.
    s1 = session([("camA", a, fa), ("camB", b, fb)])
    for fg in s1.frame_groups.values():
        fg.instance_groups[0].identity = ida
        fg.instance_groups[1].identity = idb
    # The second: A and B the other way round, extra keys everywhere, a camera with no
    # video, a video with no frames, every score.
    s2 = session([("camC", c, fc), ("camD", d, fd), ("camX", None, []), ("camY", unused, [])], 7.0)
    s2.camera_group.metadata["rig"] = "cage5"
    s2.camera_group.cameras[0].metadata["serial"] = "24153710"
    s2.metadata["experiment"] = {"name": "x1", "day": 3}
    for fg in s2.frame_groups.values():
        fg.metadata["note"] = "frame " + str(fg.frame_idx)
        fg.instance_groups[0].identity = idb
        fg.instance_groups[1].identity = ida
        fg.instance_groups[0]._score = 0.25
        fg.instance_groups[0].metadata["who"] = "é ☃"
        fg.instance_groups[1].instance_3d.score = 0.75
    # One group has no 3D, one has a node with no position.
    s2.frame_groups[1].instance_groups[0]._instance_3d = None
    s2.frame_groups[2].instance_groups[1]._instance_3d = Instance3D(points=np.array([[9.0, np.nan, 3.0], [np.nan, np.nan, np.nan]]), skeleton=sk, score=0.5)
    # The third: predicted 3D, a group on one camera, identity only it names, frames far on.
    s3 = sio.RecordingSession(camera_group=sio.CameraGroup(cameras=cameras(["camE"], 3.0)))
    s3.add_video(e, s3.camera_group.cameras[0])
    cam = s3.camera_group.cameras[0]
    for lf in fe:
        ig = sio.InstanceGroup(instance_by_camera={cam: lf.instances[1]}, identity=idc,
                               instance_3d=PredictedInstance3D(points=np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]), skeleton=sk, score=0.6, point_scores=np.array([0.1, 0.2])))
        s3.frame_groups[lf.frame_idx] = sio.FrameGroup(frame_idx=lf.frame_idx, instance_groups=[ig], labeled_frame_by_camera={cam: lf})
    # The fourth: cameras and a video, no group at all.
    w = video("camW.mp4")
    s4 = sio.RecordingSession(camera_group=sio.CameraGroup(cameras=cameras(["camW"], 5.0)))
    s4.add_video(w, s4.camera_group.cameras[0])
    return fe + fd + fa + fc + fb + frames_on(w, 800), [a, b, c, d, e, unused, w], [s1, s2, s3, s4], [ida, idb, idc]

frames, videos, sessions, identities = rich_parts()
save("rich.slp", frames, videos, sessions, identities=identities)

# The same file with identities on poses too, an open session's and a closed one's. They
# are matched by name, so a closed group on C must not end up on a second C.
frames, videos, sessions, identities = rich_parts()
by_name = {i.name: i for i in identities}
for lf in frames:
    if lf.video.filename == "camA.mp4" and lf.frame_idx == 0:
        lf.instances[0].identity = by_name["A"]
    if lf.video.filename == "camD.mp4" and lf.frame_idx == 1:
        lf.instances[1].identity = by_name["A"]
    if lf.video.filename == "camE.mp4" and lf.frame_idx == 20:
        lf.instances[0].identity = by_name["C"]
save("pose_identities.slp", frames, videos, sessions, identities=identities)

# The same file the way a writer before /session_data left it: everything inline in sessions_json.
frames, videos, sessions, identities = rich_parts()
labels = sio.Labels(labeled_frames=frames, videos=videos, skeletons=[sk], tracks=[T1, T2], sessions=sessions, identities=identities)
labels.save(os.path.join(out, "legacy.slp"), embed=False)
with h5py.File(os.path.join(out, "legacy.slp"), "r+") as f:
    video_to_idx = {v: i for i, v in enumerate(labels.videos)}
    lf_to_idx = {lf: i for i, lf in enumerate(labels.labeled_frames)}
    # Python's reader takes NaN as null there.
    def no_nan(v):
        if isinstance(v, dict):
            return {str(k): no_nan(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [no_nan(x) for x in v]
        return None if isinstance(v, float) and v != v else v
    legacy = [json.dumps(no_nan(session_to_dict(s, video_to_idx, lf_to_idx, identities=labels.identities))) for s in labels.sessions]
    del f["session_data"]
    del f["sessions_json"]
    f.create_dataset("sessions_json", data=np.array([x.encode() for x in legacy], dtype=object), dtype=h5py.special_dtype(vlen=bytes))
    f["metadata"].attrs["format_id"] = 2.7

# Two sessions naming one video.
a, b, d = video("camA.mp4"), video("camB.mp4"), video("camD.mp4")
fa, fb, fd = frames_on(a, 100), frames_on(b, 200), frames_on(d, 400)
s1 = session([("camA", a, fa), ("camB", b, fb)])
s2 = session([("camC", b, fb), ("camD", d, fd)], 7.0)
save("shared_video.slp", fa + fb + fd, [a, b, d], [s1, s2])

# Per-session provenance, as a LUC3D save of a project from two files writes it.
a, b, c, d, x, fa, fb, fc, fd, fx = two_sessions_parts()
s1, s2 = session([("camA", a, fa), ("camB", b, fb)]), session([("camC", c, fc), ("camD", d, fd)], 7.0)
s1.metadata["metadata"] = {"lucid": {"sessionName": "one"}}
s2.metadata["metadata"] = {"lucid": {"sessionName": "two"}}
save("provenance.slp", fa + fb + fc + fd, [a, b, c, d], [s1, s2],
     provenance={"recording": "first.mp4", "operator": "campy", "lucid_sessions": {"two": {"recording": "second.mp4"}}})

# One session: nothing to keep aside.
a, b = video("camA.mp4"), video("camB.mp4")
fa, fb = frames_on(a, 100), frames_on(b, 200)
save("one_session.slp", fa + fb, [a, b], [session([("camA", a, fa), ("camB", b, fb)])])

# A second lazy session for section G.
for name, base in (("camF", 900), ("camG", 950)):
    v = video(name + ".mp4")
    sio.Labels(labeled_frames=frames_on(v, base), videos=[v], skeletons=[sk], tracks=[T1, T2]).save(os.path.join(out, name + "_alone.slp"), embed=False)
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

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'lazy-keeps-'));
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

const LOADERS = {
    eager: async (page) => page.evaluate(() => { window.__done = false; (async () => {
        await (await import('/import-export/slp-import.js')).handleLoadSlpFile(new File([window.__file], 'project.slp'));
    })().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); }); }),
    lazy: async (page) => page.evaluate(() => { window.__done = false; (async () => {
        await (await import('/loading/session-loader.js')).handleLoadProjectSlpLazy(new File([window.__file], 'project.slp'));
    })().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); }); }),
    // File ▸ Open. HDF5 ignores trailing bytes, so padding makes a small file "over 200 MB".
    fileOpen: async (page) => page.evaluate(() => { window.__done = false; (async () => {
        const pad = new Uint8Array(200 * 1024 * 1024 + 1);
        await (await import('/import-export/save-load.js')).handleLoadProject(new File([window.__file, pad], 'project.slp'));
    })().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); }); }),
};

/** Answer the video prompts a load puts up, until `window.__done` is set. */
async function settle(page, what) {
    let done = false;
    for (let i = 0; i < 400 && !done; i++) {
        const s = await page.evaluate(() => {
            const b = [...document.querySelectorAll('button')].find(x => x.offsetParent && /^(Skip|Later|Cancel|Not now)$/i.test((x.textContent || '').trim()));
            if (b) { b.click(); return { done: false }; }
            return { done: window.__done, prompt: !!document.getElementById('videoSelectOverlay') };
        });
        if (s.prompt) await page.keyboard.press('Escape');
        done = s.done;
        if (!done) await page.waitForTimeout(100);
    }
    if (done !== true) throw new Error(what + ': did not finish: ' + done);
}

async function open(loader, bytes) {
    const page = await newAppPage();
    await putBytes(page, '__file', bytes);
    await LOADERS[loader](page);
    await settle(page, loader + ' load');
    return page;
}

/** Runs in the page: File ▸ Save, the real route, into `window.__saved`. */
async function saveInPage() {
    const sl = await import('/import-export/save-load.js');
    const orig = URL.createObjectURL;
    let blob = null;
    URL.createObjectURL = b => { blob = b; return orig.call(URL, b); };
    try { await sl.saveProjectSlp(); } finally { URL.createObjectURL = orig; }
    if (!blob) throw new Error('saveProjectSlp produced no file: ' + (document.getElementById('statusText') || {}).textContent);
    window.__saved = new Uint8Array(await blob.arrayBuffer());
}

/** Runs in the page: what the app's own reader says `window.__saved` holds, in Python's terms. */
async function readInPage() {
    const fileio = await import('/import-export/file-io.js');
    const slpimp = await import('/import-export/slp-import.js');
    const { Session, Camera, Skeleton } = await import('/pose/pose-data.js');
    const slp = await fileio.parseSlpViaSleapIO(new File([window.__saved], 'saved.slp'), () => {});
    const num = v => (v === v ? Number(v).toFixed(3) : 'nan');
    const poses = [];
    for (const f of slp.frames) {
        for (const d of f.instances) {
            const nodes = d.points.map((p, k) => (p ? num(p[0]) + ',' + num(p[1]) + (d.occluded[k] ? '~' : '') : '-')).join(';');
            const scores = d.type === 'predicted' ? [num(d.score)].concat(Array.from(d.pointScores || [], num)).join(',') : '';
            const track = d.trackIdx >= 0 ? slp.tracks[d.trackIdx] : '';
            poses.push([slp.videos[f.videoIdx].filename, f.frameIdx, d.type === 'predicted' ? 'P' : 'U', track, nodes, scores].join('|'));
        }
    }
    const sessions = [], groups = [];
    for (let si = 0; si < (slp.sessions || []).length; si++) {
        const s = slp.sessions[si];
        const keys = Object.keys(s.calibration || {}).filter(k => k !== 'metadata');
        const names = keys.map(k => s.calibration[k].name || k);
        const cams = {};
        for (const k in s.camcorder_to_video_idx_map || {}) {
            const name = names[parseInt(k, 10)] !== undefined ? names[parseInt(k, 10)] : k;
            cams[name] = slp.videos[s.camcorder_to_video_idx_map[k]].filename;
        }
        sessions.push(cams);
        if (!s._typedSession) continue;
        const fresh = new Session(
            keys.map(k => { const c = s.calibration[k]; return new Camera(c.name || k, c.matrix, c.distortions, c.rotation, c.translation, c.size); }),
            new Skeleton(slp.skeleton.name, slp.skeleton.nodes, slp.skeleton.edges), slp.tracks, 'read');
        await slpimp.reconstructInstanceGroupsFromSession(fresh, s._typedSession, s, slp.skeleton.nodes, {});
        for (const [f, gs] of fresh.instanceGroups) {
            for (const g of gs) {
                const members = [...g.instances].map(([c, i]) => c + '=' + num(i.getX(0))).sort();
                const p3 = g.points3d && g.points3d.length ? Array.from(g.points3d).slice(0, 3).map(num).join(',') : '';
                groups.push(['s' + si, f].concat(members, [p3]).join('|'));
            }
        }
    }
    return {
        videos: slp.videos.map(v => v.filename), tracks: slp.tracks,
        identities: (slp.identities || []).map(i => i.name),
        sessions, poses: poses.sort(), groups: groups.sort(),
    };
}

/** Save the open project, close its page; keep the bytes and the app's own read of them. */
async function saveAndRead(page, keepOpen) {
    await page.evaluate(saveInPage);
    const read = await page.evaluate(readInPage);
    const b64 = await page.evaluate(() => {
        const u = window.__saved;
        let s = '';
        for (let i = 0; i < u.length; i += 32768) s += String.fromCharCode.apply(null, u.subarray(i, i + 32768));
        return btoa(s);
    });
    if (!keepOpen) await page.close();
    return { bytes: Buffer.from(b64, 'base64'), read };
}
const statusOf = page => page.evaluate(() => (document.getElementById('statusText') || {}).textContent || '');

let savedSeq = 0;
const pythonWork = [];   // [{ file, what, want }] checked together: one Python run reads every save
function savedForPython(bytes, what, want) {
    const file = path.join(tmp, 'saved-' + (++savedSeq) + '.slp');
    fs.writeFileSync(file, bytes);
    if (process.env.KEEP_DIR) fs.copyFileSync(file, path.join(process.env.KEEP_DIR, path.basename(file)));
    pythonWork.push({ file, what, want });
    return file;
}

/** Both readers against one expectation. Only the keys `want` has are compared. */
function compare(what, got, want) {
    if (got.error) { check(false, what + ': opens', got.error); return; }
    if (want.poses) check(eq(got.poses, want.poses), what + ': every pose on its own video', diff(got.poses || [], want.poses));
    if (want.groups) check(eq(got.groups, want.groups), what + ': groups point at the right poses, with their 3D', diff(got.groups || [], want.groups));
    if (want.sessions) check(eq(got.sessions, want.sessions), what + ': each session\'s cameras and their videos', got.sessions);
    if (want.rich && got.rich) {
        const closed = (got.rich || []).filter(r => !r.startsWith('s0|'));
        check(eq(closed, want.rich), what + ': a closed session\'s scores, identities, metadata and every 3D point', diff(closed, want.rich));
    }
    if (want.details && got.details) {
        check(eq(got.details.slice(1), want.details), what + ': a closed session\'s calibration, extra keys and metadata', got.details.slice(1));
        check(eq(got.details[0].cameras.map(c => [c.name, c.rvec, c.tvec, c.matrix, c.dist, c.size]), want.firstCameras), what + ': the open session\'s calibration', got.details[0].cameras);
    }
    if (want.closedVideoShapes && got.videoShapes) {
        const shapes = {};
        for (const name in want.closedVideoShapes) shapes[name] = got.videoShapes[name];
        check(eq(shapes, want.closedVideoShapes), what + ': a closed session\'s videos keep their shape', shapes);
    }
    if (want.identities && got.identities) check(eq(got.identities, want.identities), what + ': the identity list', got.identities);
    if (want.identityOwners && got.identityOwners) check(eq(got.identityOwners, want.identityOwners), what + ': sessions that shared an identity still do', got.identityOwners);
    if (want.poseIdentities && got.poseIdentities) check(eq(got.poseIdentities, want.poseIdentities), what + ': identities on poses', diff(got.poseIdentities, want.poseIdentities));
    if (want.provenance && got.provenance) check(eq(got.provenance, want.provenance), what + ': provenance', got.provenance);
}

function checkSave(what, saved, want) {
    compare(what + ', app reader', saved.read, want);
    return savedForPython(saved.bytes, what + ', Python', want);
}

const FILES = ['two_sessions', 'closed_first', 'first_empty', 'dup_rows', 'rich', 'pose_identities', 'legacy', 'shared_video'];

/** What every save of `name` must hold, from Python's read of the source. */
function expectedFor(source, extra) {
    return Object.assign({
        poses: source.poses,
        groups: source.groups,
        sessions: source.sessions,
        rich: source.rich.filter(r => !r.startsWith('s0|')),
        details: source.details.slice(1),
        firstCameras: source.details[0].cameras.map(c => [c.name, c.rvec, c.tvec, c.matrix, c.dist, c.size]),
        identities: source.identities,
        identityOwners: source.identityOwners,
        poseIdentities: source.poseIdentities,
        closedVideoShapes: Object.fromEntries(source.sessions.slice(1).flatMap(s => Object.values(s)).map(name => [name, source.videoShapes[name]])),
    }, extra);
}

try {
    uv(PY_WRITE, [tmp]);
    const sourcePaths = [...FILES, 'first_no_cameras', 'first_no_map', 'provenance', 'one_session', 'camF_alone', 'camG_alone'].map(n => path.join(tmp, n + '.slp'));
    const sources = {};
    const rawSources = JSON.parse(uv(PY_READ, sourcePaths).trim().split('\n').pop());
    for (const p of sourcePaths) sources[path.basename(p, '.slp')] = rawSources[p];
    const readFile = n => fs.readFileSync(path.join(tmp, n + '.slp'));

    console.log('Sources, as Python reads them:');
    check(eq(sources.two_sessions.sessions, [{ camA: 'camA.mp4', camB: 'camB.mp4' }, { camC: 'camC.mp4', camD: 'camD.mp4' }]), 'two_sessions.slp has two sessions', sources.two_sessions.sessions);
    check(sources.rich.sessions.length === 4 && sources.rich.identities.length === 3, 'rich.slp has four sessions and three identities');
    check(eq(sources.rich.identityOwners, [[0, 1], [0, 1], [2]]), 'in rich.slp the first two sessions share two identities', sources.rich.identityOwners);
    check(sources.rich.rich.some(r => /ps=0\.100,0\.200/.test(r)) && sources.rich.rich.some(r => /score=0\.250/.test(r)), 'rich.slp holds predicted 3D and group scores');
    check(sources.rich.details[1].cameras[0].metadata.serial === '24153710' && sources.rich.details[1].metadata.experiment.day === 3, 'rich.slp holds keys LUC3D does not read');
    check(eq(sources.legacy.rich, sources.rich.rich) && eq(sources.legacy.details, sources.rich.details), 'legacy.slp holds the same, inline', sources.legacy.error || diff(sources.legacy.rich, sources.rich.rich));
    check(eq(sources.shared_video.sessions[1].camC, 'camB.mp4'), 'shared_video.slp names one video in both sessions');

    console.log('\nA. Every layout, the lazy route, two saves');
    for (const name of FILES) {
        let bytes = readFile(name);
        const want = expectedFor(sources[name]);
        let firstVideos = null;
        for (let gen = 1; gen <= 2; gen++) {
            const page = await open('lazy', bytes);
            const status = await statusOf(page);
            const what = `${name}, save ${gen}`;
            if (gen === 1) {
                const n = sources[name].sessions.length;
                check(new RegExp('Only the first of ' + n + ' sessions in this file is open: saving keeps the others').test(status), what + ': the status says what a save does', status);
            }
            const open1 = await page.evaluate(() => {
                const s = window.__lucid.state;
                return { sessions: s.sessions.length, lazy: !!s.session.lazyLoader, cameras: s.session.cameras.map(c => c.name) };
            });
            check(open1.sessions === 1 && open1.lazy, what + ': one session is open, lazily', open1);
            const saved = await saveAndRead(page);
            checkSave(what, saved, want);
            if (gen === 1) firstVideos = saved.read.videos;
            else check(eq(saved.read.videos, firstVideos), what + ': the video list did not move', [saved.read.videos, firstVideos]);
            bytes = saved.bytes;
        }
    }

    console.log('\nA2. A first session that says little about its videos');
    for (const [name, route] of [['first_no_cameras', 'lazy'], ['first_no_map', 'eager']]) {
        // No camera at all: it reads like a plain file, one camera per video, and those
        // videos are the other session's too. Cameras but no video map: the file goes the
        // older way, which reads every session. Either way nothing of the other session goes.
        let bytes = readFile(name);
        const src = sources[name];
        for (let gen = 1; gen <= 2; gen++) {
            const page = await open('lazy', bytes);
            const lazy = await page.evaluate(() => !!window.__lucid.state.session.lazyLoader);
            const saved = await saveAndRead(page);
            const what = `${name}, save ${gen}`;
            // A save gives the session cameras once, so the second open is an ordinary one.
            if (gen === 1) check(lazy === (route === 'lazy'), `${what}: opened ${route === 'lazy' ? 'lazily' : 'in full'}`, lazy);
            check(eq(saved.read.poses, src.poses), `${what}: every pose`, diff(saved.read.poses, src.poses));
            const closedGroups = list => list.filter(g => /camC=/.test(g)).map(g => g.replace(/^s\d+\|/, ''));
            check(eq(closedGroups(saved.read.groups), closedGroups(src.groups)), `${what}: the other session's groups`, diff(closedGroups(saved.read.groups), closedGroups(src.groups)));
            check(saved.read.sessions.some(s => s.camC === 'camC.mp4' && s.camD === 'camD.mp4'), `${what}: the other session's cameras`, saved.read.sessions);
            savedForPython(saved.bytes, `${what}, Python`, { poses: src.poses });
            bytes = saved.bytes;
        }
    }

    console.log('\nB. File ▸ Open on a file over 200 MB');
    {
        const page = await open('fileOpen', readFile('two_sessions'));
        const lazy = await page.evaluate(() => !!window.__lucid.state.session.lazyLoader);
        check(lazy, 'it took the lazy route');
        const saved = await saveAndRead(page);
        checkSave('File ▸ Open', saved, expectedFor(sources.two_sessions));
    }

    console.log('\nC. Edits on the open session');
    {
        // A corrected frame goes first in the file and its store row is skipped, so every
        // other row moves: the closed session's members must follow them.
        for (const name of ['two_sessions', 'closed_first', 'rich']) {
            const page = await open('lazy', readFile(name));
            await page.evaluate(async () => {
                const tri = await import('/pose/triangulation.js');
                const { state } = await import('/ui/app-state.js');
                await tri.ensureLazyFrameData(1);
                const group = state.session.instanceGroups.get(1)[0];
                const inst = group.instances.get('camA');
                inst.setPoint(0, 999, 999);
                inst.type = 'user';
                inst.modified = true;
                group.markDirty();
            });
            let saved = await saveAndRead(page);
            const moved = 'camA.mp4|1|U|mouse1|999.000,999.000;101.000,20.000|';
            const poses = sources[name].poses.map(r => (r.startsWith('camA.mp4|1|P|mouse1|') ? moved : r)).sort();
            const groups = sources[name].groups.map(g => g.replace(/^(s0\|1\|)camA=101\.000/, '$1camA=999.000')).sort();
            const want = expectedFor(sources[name], { poses, groups });
            checkSave(name + ', a corrected frame', saved, want);
            saved = await saveAndRead(await open('lazy', saved.bytes));
            checkSave(name + ', a corrected frame, saved again', saved, want);
        }
    }
    {
        // Delete a pose: the store compacts its instance rows; the closed session's offsets stay.
        const page = await open('lazy', readFile('closed_first'));
        await page.evaluate(async () => {
            const tri = await import('/pose/triangulation.js');
            const { state } = await import('/ui/app-state.js');
            const del = await import('/ui/custom-delete-ops.js');
            await tri.ensureLazyFrameData(0);
            const group = state.session.instanceGroups.get(0)[0];
            del.deleteTargetsFromStore(state.session, del.groupMemberTargets(0, group));
            state.session.removeInstanceGroup(0, group);
        });
        const saved = await saveAndRead(page);
        const poses = sources.closed_first.poses.filter(r => !/^cam[AB]\.mp4\|0\|P\|mouse1\|/.test(r));
        checkSave('a deleted pose', saved, expectedFor(sources.closed_first, { poses, groups: sources.closed_first.groups.filter(g => !/^s0\|0\|camA=100/.test(g)) }));
    }
    {
        // Delete a track: only the open session's rows lose it.
        const page = await open('lazy', readFile('two_sessions'));
        await page.evaluate(async () => {
            const ops = await import('/ui/track-identity-ops.js');
            const { state } = await import('/ui/app-state.js');
            ops.deleteTrackAt(state.session, state.session.tracks.indexOf('mouse1'));
        });
        const saved = await saveAndRead(page);
        const poses = sources.two_sessions.poses.map(r => (/^cam[AB]\.mp4\|/.test(r) ? r.replace('|P|mouse1|', '|P||') : r)).sort();
        // A group on the deleted track dissolves in the open session; the closed one's stay.
        const groups = sources.two_sessions.groups.filter(g => !g.startsWith('s0|') || /camA=11\d/.test(g));
        checkSave('a deleted track', saved, expectedFor(sources.two_sessions, { poses, groups }));
    }
    {
        // Two saves in one sitting: a save must not spend what the next one needs.
        const page = await open('lazy', readFile('rich'));
        const first = await saveAndRead(page, true);
        const second = await saveAndRead(page);
        const want = expectedFor(sources.rich);
        checkSave('saved once', first, want);
        checkSave('saved again without reopening', second, want);
    }
    {
        // Renaming an identity in the open session leaves the closed session's name alone.
        const page = await open('lazy', readFile('rich'));
        await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            state.session.identities[0].name = 'renamed';
        });
        const saved = await saveAndRead(page);
        check(eq(saved.read.identities.slice().sort(), ['A', 'B', 'C', 'renamed']), 'a renamed identity stays in the open session', saved.read.identities);
        const want = expectedFor(sources.rich);
        savedForPython(saved.bytes, 'a renamed identity, Python', { poses: want.poses, groups: want.groups, rich: want.rich, details: want.details, firstCameras: want.firstCameras });
    }

    {
        // Track All and Triangulate All on the open session: it can re-track and re-solve
        // everything of its own, and the closed session must not notice.
        const page = await newAppPage();
        await answerTrackSetup(page, 2);
        await putBytes(page, '__file', readFile('rich'));
        await LOADERS.lazy(page);
        await settle(page, 'lazy load');
        const result = await page.evaluate(async () => {
            const tracker = await import('/pose/tracker.js');
            const tri = await import('/pose/triangulation.js');
            const { state } = await import('/ui/app-state.js');
            await tracker.trackAll();
            await tri.triangulateAllFrames();
            return { groups: state.session.instanceGroups.size, status: document.getElementById('statusText').textContent };
        });
        const saved = await saveAndRead(page);
        const src = sources.rich;
        check(result.groups > 0, 'Track All left the open session with groups', result);
        const onClosed = list => list.filter(r => !/^cam[AB]\.mp4\|/.test(r));
        check(eq(onClosed(saved.read.poses), onClosed(src.poses)), 'the closed sessions\' poses are untouched', diff(onClosed(saved.read.poses), onClosed(src.poses)));
        check(eq(saved.read.groups.filter(g => !g.startsWith('s0|')), src.groups.filter(g => !g.startsWith('s0|'))), 'the closed sessions\' groups are untouched');
        check(eq(saved.read.sessions, src.sessions), 'every session keeps its cameras', saved.read.sessions);
        savedForPython(saved.bytes, 'after Track All, Python', { groups: saved.read.groups, sessions: src.sessions, rich: src.rich.filter(r => !r.startsWith('s0|')), details: src.details.slice(1), firstCameras: src.details[0].cameras.map(c => [c.name, c.rvec, c.tvec, c.matrix, c.dist, c.size]) });
    }

    console.log('\nD. A video two sessions share');
    {
        // A correction on a shared frame: the closed session's member follows it into the overlay.
        const page = await open('lazy', readFile('shared_video'));
        await page.evaluate(async () => {
            const tri = await import('/pose/triangulation.js');
            const { state } = await import('/ui/app-state.js');
            await tri.ensureLazyFrameData(1);
            const group = state.session.instanceGroups.get(1)[1];
            const inst = group.instances.get('camB');
            inst.setPoint(0, 888, 888);
            inst.type = 'user';
            inst.modified = true;
            group.markDirty();
        });
        const saved = await saveAndRead(page);
        const src = sources.shared_video;
        const poses = src.poses.map(r => (r.startsWith('camB.mp4|1|P|mouse2|') ? 'camB.mp4|1|U|mouse2|888.000,888.000;221.000,20.000|' : r)).sort();
        const groups = src.groups.map(g => g.replace('camB=211.000', 'camB=888.000').replace('camC=211.000', 'camC=888.000')).sort();
        const rich = src.rich.filter(r => !r.startsWith('s0|')).map(g => g.replace('camC=211.000', 'camC=888.000')).sort();
        checkSave('a corrected pose on a shared video', saved, expectedFor(src, { poses, groups, rich }));
    }
    {
        // A delete on a shared frame: the member on the deleted pose goes, the group stays,
        // and the other pose's offset follows it down.
        const page = await open('lazy', readFile('shared_video'));
        await page.evaluate(async () => {
            const tri = await import('/pose/triangulation.js');
            const { state } = await import('/ui/app-state.js');
            const del = await import('/ui/custom-delete-ops.js');
            await tri.ensureLazyFrameData(0);
            const group = state.session.instanceGroups.get(0)[0];
            del.deleteTargetsFromStore(state.session, del.groupMemberTargets(0, group));
            state.session.removeInstanceGroup(0, group);
        });
        const saved = await saveAndRead(page);
        const src = sources.shared_video;
        const poses = src.poses.filter(r => !/^cam[AB]\.mp4\|0\|P\|mouse1\|/.test(r));
        const groups = src.groups.filter(g => !/^s0\|0\|camA=100/.test(g)).map(g => g.replace('camC=200.000|', '')).sort();
        const rich = src.rich.filter(r => !r.startsWith('s0|')).map(g => g.replace('camC=200.000|', '')).sort();
        checkSave('a deleted pose on a shared video', saved, expectedFor(src, { poses, groups, rich }));
    }

    console.log('\nE. A LUC3D project');
    {
        // An eager open reads both sessions; give each what LUC3D keeps per session.
        const page = await open('eager', readFile('two_sessions'));
        await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            const second = state.sessions[1];
            second._hiddenCameras = new Set(['camD']);
            second.addIdentity('alpha', '#112233');
            second.addIdentity('beta', '#445566');
            const frame0 = second.instanceGroups.get(0);
            frame0[0].identityId = second.identities[0].id;
            frame0[1].identityId = second.identities[1].id;
            const TS = await import('/pose/temporal-smoothing.js');
            TS.smoothSession(second, { smoothGaussianWindow: 3, smoothMedianWindow: 0, smoothMaxGap: 5, smoothMinSegment: 1 });
            window.__smoothed = {};
            const SM = await import('/import-export/smoothing-metadata.js');
            for (const [f, groups] of second.instanceGroups) {
                groups.forEach((g, i) => {
                    window.__smoothed[f + ':' + i] = { p3: Array.from(g.points3d), src: SM.smoothingSourceFor(g) ? Array.from(SM.smoothingSourceFor(g)) : null };
                });
            }
        });
        const smoothedBefore = await page.evaluate(() => window.__smoothed);
        check(Object.values(smoothedBefore).some(g => g.src), 'the second session has smoothed 3D');
        const lucid = await saveAndRead(page);
        const lucidRead = lucid.read;
        check(lucidRead.sessions.length === 2, 'the project file has both sessions', lucidRead.sessions);
        let bytes = lucid.bytes;
        for (let gen = 1; gen <= 2; gen++) {
            const saved = await saveAndRead(await open('lazy', bytes));
            check(eq(saved.read.poses, lucidRead.poses), `project, save ${gen}: every pose`, diff(saved.read.poses, lucidRead.poses));
            check(eq(saved.read.groups, lucidRead.groups), `project, save ${gen}: every group`, diff(saved.read.groups, lucidRead.groups));
            check(eq(saved.read.sessions, lucidRead.sessions), `project, save ${gen}: cameras and videos`, saved.read.sessions);
            savedForPython(saved.bytes, `project, save ${gen}, Python`, { poses: lucidRead.poses, sessions: lucidRead.sessions });
            bytes = saved.bytes;
        }
        // An eager open sees the second session as it was.
        const back = await open('eager', bytes);
        const second = await back.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            const s = state.sessions[1];
            const frame0 = s.instanceGroups.get(0) || [];
            const SM = await import('/import-export/smoothing-metadata.js');
            const smoothed = {};
            for (const [f, groups] of s.instanceGroups) {
                groups.forEach((g, i) => {
                    smoothed[f + ':' + i] = { p3: Array.from(g.points3d), src: SM.smoothingSourceFor(g) ? Array.from(SM.smoothingSourceFor(g)) : null };
                });
            }
            return {
                smoothed, temporalSmoothing: s.temporalSmoothing || null,
                sessions: state.sessions.length, cameras: s.cameras.map(c => c.name), name: s.name,
                hidden: Array.from(s._hiddenCameras || []),
                identities: s.identities.map(i => [i.name, i.color]),
                groupIdentities: frame0.map(g => g.identityId),
                tracks: s.tracks,
            };
        });
        await back.close();
        check(second.sessions === 2, 'opened eagerly, the file has both sessions', second.sessions);
        check(eq(second.cameras, ['camC', 'camD']), 'the second session has its cameras', second.cameras);
        check(eq(second.hidden, ['camD']), 'the second session keeps its Visibility settings', second.hidden);
        check(eq(second.identities, [['alpha', '#112233'], ['beta', '#445566']]), 'the second session keeps its identities', second.identities);
        check(eq(second.groupIdentities, [0, 1]), 'its groups keep them', second.groupIdentities);
        check(eq(second.smoothed, smoothedBefore), 'its smoothed 3D and what it was smoothed from, bit for bit', second.smoothed);
    }
    {
        // Planes are one state written into every session: deleting them in the open
        // session must not bring them back from the closed one's copy.
        const page = await open('eager', readFile('two_sessions'));
        await page.evaluate(async () => {
            const P = await import('/ui/plane-definition.js');
            const model = P.planeState.model;
            const floor = model.createPlane ? model.createPlane('floor') : model.addPlane('floor');
            for (const n of ['a', 'b', 'c']) model.createNodeInPlane(n, floor);
        });
        const withPlanes = await saveAndRead(page);
        const lazyPage = await open('lazy', withPlanes.bytes);
        const before = await lazyPage.evaluate(async () => {
            const P = await import('/ui/plane-definition.js');
            const n = P.planeState.model.planes.length;
            const ids = P.planeState.model.planes.map(p => p.id);
            for (const id of ids) P.planeState.model.deletePlane(id);
            for (const node of Array.from(P.planeState.model.pool.nodes ? P.planeState.model.pool.nodes.values() : [])) P.planeState.model.pool.removeNode && P.planeState.model.pool.removeNode(node.id);
            return { n, after: P.planeState.model.planes.length, pool: P.planeState.model.pool.size };
        });
        check(before.n === 1 && before.after === 0 && before.pool === 0, 'the open session has the plane, then none', before);
        const gone = await saveAndRead(lazyPage);
        const eagerPage = await open('eager', gone.bytes);
        const planes = await eagerPage.evaluate(async () => {
            const P = await import('/ui/plane-definition.js');
            return { sessions: window.__lucid.state.sessions.length, planes: P.planeState.model.planes.length, pool: P.planeState.model.pool.size };
        });
        await eagerPage.close();
        check(planes.sessions === 2 && planes.planes === 0 && planes.pool === 0, 'a deleted plane stays deleted when the file is opened eagerly', planes);
    }

    console.log('\nF. Per-session provenance');
    {
        const src = sources.provenance;
        check(eq(src.provenance, { recording: 'first.mp4', operator: 'campy', lucid_sessions: { two: { recording: 'second.mp4' } } }), 'the source holds it', src.provenance);
        let bytes = readFile('provenance');
        for (let gen = 1; gen <= 2; gen++) {
            const saved = await saveAndRead(await open('lazy', bytes));
            savedForPython(saved.bytes, `provenance, save ${gen}, Python`, { sessions: src.sessions, provenance: { recording: 'first.mp4', operator: 'campy', lucid_sessions: { two: { recording: 'second.mp4' } } } });
            bytes = saved.bytes;
        }
        const page = await open('eager', bytes);
        const provs = await page.evaluate(() => window.__lucid.state.sessions.map(s => s.fileProvenance));
        await page.close();
        check(eq(provs, [{ recording: 'first.mp4', operator: 'campy' }, { recording: 'second.mp4' }]), 'each session has its own when the file is opened eagerly', provs);
    }

    console.log('\nG. A project of two lazy files, and one mixing lazy and eager');
    {
        const page = await open('lazy', readFile('rich'));
        await putBytes(page, '__camF', readFile('camF_alone'));
        await putBytes(page, '__camG', readFile('camG_alone'));
        const err = await page.evaluate(async () => {
            const pd = await import('/pose/pose-data.js');
            const { SioLazyLoader } = await import('/loading/sio-lazy-loader.js');
            const AS = await import('/ui/app-state.js');
            const sl = await import('/import-export/save-load.js');
            const { Camera, Instance, InstanceGroup, FrameGroup, Session } = pd;
            const names = ['camF', 'camG'];
            const mtx = [[500, 0, 32], [0, 500, 32], [0, 0, 1]];
            const cams = names.map((n, i) => new Camera(n, mtx, [0, 0, 0, 0, 0], [0, 0.1 * i, 0], [10 * i, 0, 0], [64, 64]));
            const second = new Session(cams, AS.state.session.skeleton, ['mouse1', 'mouse2'], 'second');
            const loader = new SioLazyLoader();
            await loader.open('camF', new File([window.__camF], 'camF.slp'));
            await loader.open('camG', new File([window.__camG], 'camG.slp'));
            second.lazyLoader = loader;
            for (let f = 0; f < 3; f++) {
                second.addFrameGroup(new FrameGroup(f));
                const gs = [];
                for (let i = 0; i < 2; i++) {
                    const g = new InstanceGroup(f * 10 + i, null);
                    names.forEach(n => {
                        const inst = new Instance([[1, 1], [2, 2]], i, 'predicted', 0.9);
                        inst._rawInstIndex = i;
                        g.addInstance(n, inst);
                    });
                    g.points3d = new Float64Array([1 + i + f, 2, 3, 4, 5, 6 + i]);
                    gs.push(g);
                }
                second.instanceGroups.set(f, gs);
            }
            AS.state.sessions.push(second);
            try {
                window.__saved = await sl.saveAllSessionsStreaming(AS.state.sessions.slice());
                return null;
            } catch (e) { return String(e && e.stack || e); }
        });
        check(!err, 'the two-pass save runs', err);
        if (!err) {
            const read = await page.evaluate(readInPage);
            const b64 = await page.evaluate(() => { const u = window.__saved; let s = ''; for (let i = 0; i < u.length; i += 32768) s += String.fromCharCode.apply(null, u.subarray(i, i + 32768)); return btoa(s); });
            const secondPoses = [...sources.camF_alone.poses, ...sources.camG_alone.poses];
            const secondGroups = [];
            for (let f = 0; f < 3; f++) {
                for (let i = 0; i < 2; i++) {
                    secondGroups.push(['s4', f, 'camF=' + (900 + 10 * i + f).toFixed(3), 'camG=' + (950 + 10 * i + f).toFixed(3), [1 + i + f, 2, 3].map(v => v.toFixed(3)).join(',')].join('|'));
                }
            }
            const src = sources.rich;
            const want = {
                poses: [...src.poses, ...secondPoses].sort(),
                groups: [...src.groups, ...secondGroups].sort(),
                sessions: [...src.sessions, { camF: 'camF.mp4', camG: 'camG.mp4' }],
                rich: undefined,
            };
            checkSave('two lazy files', { bytes: Buffer.from(b64, 'base64'), read }, want);
        }
        await page.close();
    }
    {
        // The eager writer's path: every lazy frame resident, one eager session beside it.
        const page = await open('lazy', readFile('rich'));
        await page.evaluate(async () => {
            const pd = await import('/pose/pose-data.js');
            const tri = await import('/pose/triangulation.js');
            const { state } = await import('/ui/app-state.js');
            for (const f of [0, 1, 2, 20, 21]) await tri.ensureLazyFrameData(f);
            const cam = new pd.Camera('camZ', [[500, 0, 32], [0, 500, 32], [0, 0, 1]], [0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0], [64, 64]);
            const eager = new pd.Session([cam], state.session.skeleton, ['mouse1'], 'eager');
            const fg = new pd.FrameGroup(0);
            eager.addFrameGroup(fg);
            fg.addInstance('camZ', new pd.Instance([[1000, 10], [1001, 20]], 0, 'user', 0));
            state.sessions.push(eager);
        });
        const saved = await saveAndRead(page);
        const src = sources.rich;
        const closedSessions = src.sessions.slice(1);
        check(eq(saved.read.sessions.slice(1, 4), closedSessions), 'the closed sessions keep their cameras', saved.read.sessions);
        const closedGroups = src.groups.filter(g => !g.startsWith('s0|'));
        check(eq(saved.read.groups.filter(g => !g.startsWith('s0|') && !g.startsWith('s4|')), closedGroups), 'the closed sessions keep their groups', diff(saved.read.groups, closedGroups));
        savedForPython(saved.bytes, 'mixed project, Python', { poses: saved.read.poses, groups: saved.read.groups, sessions: saved.read.sessions, rich: src.rich.filter(r => !r.startsWith('s0|')).map(r => r) });
    }

    console.log('\nH. What is resident after the load');
    {
        const page = await open('lazy', readFile('rich'));
        const shape = await page.evaluate(() => {
            const s = window.__lucid.state.session;
            const kept = s.unopenedSessions;
            return {
                readerSessions: s.lazyLoader._projectLabels.sessions.length,
                kept: kept.sessions.length,
                typed: kept.sessions.every(r => ArrayBuffer.isView(r.memberRow) && ArrayBuffer.isView(r.points3d) && ArrayBuffer.isView(r.identity)),
            };
        });
        await page.close();
        check(shape.readerSessions === 1, 'the reader\'s graphs of the other sessions are gone', shape);
        check(shape.kept === 3 && shape.typed, 'three sessions are kept, as typed columns', shape);
    }
    {
        // A file with one session: nothing kept, nothing said.
        const page = await open('lazy', readFile('one_session'));
        const got = await page.evaluate(() => ({ kept: window.__lucid.state.session.unopenedSessions, status: document.getElementById('statusText').textContent }));
        await page.close();
        check(got.kept === null && !/Only the first/.test(got.status), 'a file with one session keeps nothing aside', got);
    }

    console.log('\nPython reads every save:');
    const readBack = JSON.parse(uv(PY_READ, pythonWork.map(w => w.file)).trim().split('\n').pop());
    for (const w of pythonWork) compare(w.what, readBack[w.file], w.want);
} finally {
    await browser.close();
    server.kill();
    fs.rmSync(tmp, { recursive: true, force: true });
}
console.log(fails ? `\n${fails} check(s) FAILED` : '\nAll checks passed');
process.exit(fails ? 1 : 0);
