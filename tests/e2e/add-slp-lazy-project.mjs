/**
 * add-slp-lazy-project.mjs — File ▸ Add SLP on a lazy project reaches the file.
 *
 * A lazy project's poses live in its columnar store, and a save writes the store plus the
 * user's edits on resident frames. Add SLP used to put its poses in `session.frameGroups`
 * only: it said "3 frames merged", the save's pruning dropped them with every other
 * prediction-only frame, and the file had none. It also replaced those frames' groups, 3D
 * included. They go in the store now (`SioLazyLoader.prepareAddedRows`/`commitAddedRows`,
 * `loading/lazy-store-rows.js`), so they hydrate, evict and save like any other camera's.
 *
 * Real Python sleap-io writes the files. Every save is read with the app's own reader and
 * with Python, and both must report every added pose on its camera's video, every original
 * pose and group unchanged, and any pose on a video no camera has on its own video.
 *
 *  A. The whole path, on both store layouts: add, play until eviction runs, scrub back,
 *     save twice, reopen lazily and eagerly, save again.
 *  B. A file whose cameras and videos have the project's names and whose nodes are in
 *     another order, added on top of another add.
 *  C. Other files: no tracks, no frame on any camera, a second skeleton, an analysis .h5, a
 *     file with images inside; and a large plain file as the project.
 *  D. Edits on added poses: delete, correct, delete a track, group by hand.
 *  E. Track All: the status's promise that it groups them, and a sweep's window release.
 *  F. A project opened from per-camera files.
 *  G. A lazy multi-session save (two passes), and a project mixing lazy and eager.
 *  H. What must be refused (a file too new to read, a store too old or with no room for
 *     rows), and a failure half way: nothing changes.
 *
 * Needs `uv` for Python (skipped without it).
 *
 * Run: node tests/e2e/add-slp-lazy-project.mjs   (spawns its own http.server; PORT= to move it,
 *      ONLY=DE to run some sections)
 */
import { chromium } from 'playwright';
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8531);
const PY_SPEC = process.env.SLEAP_IO_SPEC || 'sleap-io==0.9.5';
const wants = section => !process.env.ONLY || process.env.ONLY.includes(section);

let fails = 0;
const check = (c, m, got) => {
    console.log((c ? '  ✓ ' : '  ✗ ') + m + (!c && got !== undefined ? ' — got ' + JSON.stringify(got) : ''));
    if (!c) fails++;
    stallGuard.refresh();
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
/** The first few lines two sorted tables disagree on, for the failure message. */
const diff = (got, want) => {
    const g = new Set(got), w = new Set(want);
    return { missing: want.filter(x => !g.has(x)).slice(0, 4), extra: got.filter(x => !w.has(x)).slice(0, 4), counts: [got.length, want.length] };
};
const sortedKeys = o => Object.fromEntries(Object.entries(o).sort((a, b) => (a[0] < b[0] ? -1 : 1)));

// One line per pose: video|frame|U or P|track|nodes|scores. A node is `x,y`, `x,y~` when
// it is placed but not visible, `-` when it has no position. Each video's x values are its
// own, so a pose that moved cannot pass for the one it replaced. `cameraPoses` is the same
// with the camera a session gives that video, because two videos can share a filename.
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

def read(path):
    try:
        labels = sio.load_slp(path, open_videos=False)
        sio.load_slp(path)  # the default call opens the videos too
    except Exception as e:
        return {"error": type(e).__name__ + ": " + str(e)[:160]}
    poses = sorted(pose_row(lf.video.filename, lf.frame_idx, i) for lf in labels.labeled_frames for i in lf.instances)
    sessions, groups, tvecs, tracking = [], [], {}, {}
    camera_of = {}
    for si, s in enumerate(labels.sessions):
        cams = {}
        for c in s.camera_group.cameras:
            tvecs[c.name] = float(np.asarray(c.tvec)[0])
            v = s.get_video(c)
            if v is not None:
                cams[c.name] = v.filename
                camera_of[id(v)] = c.name
        sessions.append(cams)
        for f, fg in s.frame_groups.items():
            for ig in fg.instance_groups:
                members = sorted(c.name + "=" + num(i.points["xy"][0][0]) for c, i in ig.instance_by_camera.items())
                p3 = "" if ig.instance_3d is None else ",".join(num(v) for v in np.asarray(ig.instance_3d.points)[0][:3])
                groups.append("|".join(["s" + str(si), str(f)] + members + [p3]))
    camera_poses = []
    model_labels = {}
    for lf in labels.labeled_frames:
        cam = camera_of.get(id(lf.video))
        for i in lf.instances:
            if i.identity is not None or i.category is not None:
                model_labels[(cam or lf.video.filename) + "|" + str(lf.frame_idx) + "|" + num(i.points["xy"][0][0])] = "|".join([
                    (i.identity.name + "=" + num(i.identity_score)) if i.identity is not None else "",
                    (i.category.name + "=" + num(i.category_score)) if i.category is not None else "",
                ])
            if cam is None:
                continue
            camera_poses.append(pose_row(cam, lf.frame_idx, i))
            if isinstance(i, sio.PredictedInstance):
                ts = i.tracking_score
                tracking[cam + "|" + str(lf.frame_idx)] = None if ts is None or ts != ts else round(float(ts), 3)
    return {
        "error": None, "videos": [v.filename for v in labels.videos], "tracks": [t.name for t in labels.tracks],
        "skeletons": [[n.name for n in sk.nodes] for sk in labels.skeletons], "tvecs": tvecs, "tracking": tracking, "modelLabels": model_labels,
        "sessions": sessions, "poses": poses, "cameraPoses": sorted(camera_poses), "groups": sorted(groups),
    }
`;

const PY_READ = PY_COMMON + `
print(json.dumps({p: read(p) for p in sys.argv[1:]}))
`;

const NFRAMES = 2000;
const ADDED_FRAMES = [0, 1, 2, 500, 1990, 1991, 2100];

const PY_WRITE = PY_COMMON + `
from sleap_io.model.camera import Instance3D
import h5py, shutil

out = sys.argv[1]
minimal_pkg = sys.argv[2]
N = ${NFRAMES}
ADDED_FRAMES = ${JSON.stringify(ADDED_FRAMES)}
sk = sio.Skeleton(nodes=["nose", "tail"], edges=[("nose", "tail")])
T1, T2, T3, T9 = sio.Track(name="mouse1"), sio.Track(name="mouse2"), sio.Track(name="only_outside"), sio.Track(name="mouse9")

def video(name):
    v = sio.Video(filename=name, open_backend=False)
    v.backend_metadata["shape"] = (400, 64, 64, 1)
    return v

def frames_on(v, base, n=3, tracks=(T1, T2), skeleton=sk):
    return [sio.LabeledFrame(video=v, frame_idx=f, instances=[
        sio.PredictedInstance.from_numpy(np.array([[base + 10.0 * i + f, 10.0], [base + 20.0 * i + f, 20.0]]), skeleton=skeleton, track=t,
                                         score=0.9, point_scores=np.array([0.5 + 0.1 * i, 0.8]))
        for i, t in enumerate(tracks)]) for f in range(n)]

def camera(name, i, tvec=None):
    return sio.Camera(name=name, rvec=np.array([0.0, 0.1 * i, 0.0]), tvec=np.array([tvec if tvec is not None else 10.0 * i, 0.0, 0.0]),
                      matrix=np.array([[500.0, 0, 32], [0, 500.0, 32], [0, 0, 1]]), dist=np.zeros(5), size=(64, 64))

def session(pairs, groups=True, tvec_base=None):
    cams = [camera(n, i, None if tvec_base is None else tvec_base + i) for i, (n, _, _) in enumerate(pairs)]
    s = sio.RecordingSession(camera_group=sio.CameraGroup(cameras=cams))
    for c, (_, v, _) in zip(cams, pairs):
        if v is not None:
            s.add_video(v, c)
    if not groups:
        return s
    # Every frame of the session is grouped across its cameras, with 3D.
    with_frames = [(c, {lf.frame_idx: lf for lf in fs}) for c, (_, _, fs) in zip(cams, pairs) if fs]
    for f in sorted(with_frames[0][1]) if with_frames else []:
        igs = [sio.InstanceGroup(instance_by_camera={c: by[f].instances[i] for c, by in with_frames}, score=0.5,
                                 instance_3d=Instance3D(points=np.array([[1.0 + i + f, 2.0, 3.0], [4.0, 5.0, 6.0 + i]]), skeleton=sk))
               for i in range(2)]
        s.frame_groups[f] = sio.FrameGroup(frame_idx=f, instance_groups=igs, labeled_frame_by_camera={c: by[f] for c, by in with_frames})
    return s

def save(name, frames, videos, sessions=(), tracks=(T1, T2, T3), skeletons=(sk,)):
    sio.Labels(labeled_frames=list(frames), videos=list(videos), skeletons=list(skeletons), tracks=list(tracks), sessions=list(sessions)).save(os.path.join(out, name), embed=False)

# ---- The projects: every frame grouped with 3D, past the residency cap.
# Native ids equal the save's header indices, so the store goes to the writer as it is.
a, b = video("camA.mp4"), video("camB.mp4")
fa, fb = frames_on(a, 10000, N), frames_on(b, 20000, N)
save("project_plain.slp", fa + fb, [a, b], [session([("camA", a, fa), ("camB", b, fb)])])
# A video no camera has, listed first: the ids move on save, and its rows must stay apart.
a, b, x = video("camA.mp4"), video("camB.mp4"), video("project_outside.mp4")
fa, fb = frames_on(a, 10000, N), frames_on(b, 20000, N)
save("project.slp", frames_on(x, 90000, 3, (T1, T3)) + fa + fb, [x, a, b], [session([("camA", a, fa), ("camB", b, fb)])])
a, b = video("camA.mp4"), video("camB.mp4")
fa, fb = frames_on(a, 10000, 6), frames_on(b, 20000, 6)
save("project_small.slp", fa + fb, [a, b], [session([("camA", a, fa), ("camB", b, fb)])])

# ---- The main file: two cameras, a video neither has, every kind of pose.
FEMALE, MOUSE = sio.Identity(name="female"), sio.Category(name="mouse")
def added_frames(v, base):
    out_frames = []
    for f in ADDED_FRAMES:
        first = sio.PredictedInstance.from_numpy(np.array([[base + f, 10.0], [base + f + 1, 20.0]]), skeleton=sk, track=T1,
                                                 score=0.9, point_scores=np.array([0.5, 0.6]), tracking_score=0.25)
        second = sio.PredictedInstance.from_numpy(np.array([[base + f + 2, 11.0], [base + f + 3, 21.0]]), skeleton=sk, track=T9,
                                                  score=0.8, point_scores=np.array([0.7, 0.75]), tracking_score=0.25)
        if f == 0:
            first.points["visible"][1] = False   # placed but not visible
        if f == 1:
            # A hand label: one node placed but not visible, one with no place.
            second = sio.Instance.from_numpy(np.array([[base + f + 2, 11.0], [np.nan, np.nan]]), skeleton=sk, track=T9)
            second.points["visible"][0] = False
        if f == 2:
            first.track = None
        if f == 500:
            # What a model says about a detection, which a save must not lose.
            first.identity, first.identity_score = FEMALE, 0.75
            first.category, first.category_score = MOUSE, 0.5
        out_frames.append(sio.LabeledFrame(video=v, frame_idx=f, instances=[first, second]))
    return out_frames

cx, cy, o = video("camX.mp4"), video("camY.mp4"), video("added_outside.mp4")
s = session([("camX", cx, []), ("camY", cy, [])], groups=False, tvec_base=70.0)
sio.Labels(labeled_frames=frames_on(o, 50000, 3, (T1, T3)) + added_frames(cx, 30000) + added_frames(cy, 40000), videos=[o, cx, cy], skeletons=[sk], tracks=[T1, T3, T9],
           sessions=[s], identities=[FEMALE]).save(os.path.join(out, "added.slp"), embed=False)

# ---- Other files.
# A file newer than this LUC3D reads: only its format_id says so.
shutil.copy(os.path.join(out, "added.slp"), os.path.join(out, "added_too_new.slp"))
with h5py.File(os.path.join(out, "added_too_new.slp"), "r+") as f:
    f["metadata"].attrs["format_id"] = 3.0
# The project's own camera and video names, nodes in another order.
sk_rev = sio.Skeleton(nodes=["tail", "nose"], edges=[("tail", "nose")])
a2, b2 = video("camA.mp4"), video("camB.mp4")
s = session([("camA", a2, []), ("camB", b2, [])], groups=False, tvec_base=80.0)
save("added_same.slp", frames_on(a2, 60000, 2, (T1,), sk_rev) + frames_on(b2, 61000, 2, (T1,), sk_rev), [a2, b2], [s], tracks=(T1,), skeletons=(sk_rev,))
# No session, no tracks.
p = video("plain_cam.mp4")
save("added_plain.slp", frames_on(p, 62000, 2, (None,)), [p], tracks=())
# Cameras with no frames, every frame on a video neither has.
cp, cq, o2 = video("camP.mp4"), video("camQ.mp4"), video("all_outside.mp4")
s = session([("camP", cp, []), ("camQ", cq, [])], groups=False, tvec_base=90.0)
save("added_all_outside.slp", frames_on(o2, 63000, 2, (T1,)), [cp, cq, o2], [s], tracks=(T1,))
# A second skeleton: on a camera's poses, and on a video no camera has.
sk2 = sio.Skeleton(nodes=["a", "b", "c"], edges=[("a", "b")], name="other")
def frames_on_sk2(v, base):
    return [sio.LabeledFrame(video=v, frame_idx=f, instances=[
        sio.PredictedInstance.from_numpy(np.array([[base + f, 10.0], [base + f + 5, 20.0], [base + f + 9, 30.0]]), skeleton=sk2, track=T1,
                                         score=0.9, point_scores=np.array([0.5, 0.6, 0.7]))]) for f in range(3)]
c2, o3 = video("camR.mp4"), video("second_skeleton.mp4")
s = session([("camR", c2, [])], groups=False, tvec_base=95.0)
save("added_sk2.slp", frames_on(c2, 64000, 2, (T1,)) + frames_on_sk2(o3, 65000), [c2, o3], [s], tracks=(T1,), skeletons=(sk, sk2))
# Smaller than the small project, for the mixed one.
c3, o4 = video("camS.mp4"), video("small_outside.mp4")
s = session([("camS", c3, [])], groups=False, tvec_base=96.0)
save("added_small.slp", frames_on(c3, 66000, 2, (T1,)) + frames_on(o4, 67000, 2, (T1,)), [c3, o4], [s], tracks=(T1,))

# An analysis .h5: tracks is [n_tracks, 2, n_nodes, n_frames].
tracks = np.array([[[[68000.0 + f for f in range(3)], [68500.0 + f for f in range(3)]],
                    [[10.0] * 3, [20.0] * 3]],
                   [[[68100.0 + f for f in range(3)], [np.nan] * 3],
                    [[11.0] * 3, [np.nan] * 3]]])
with h5py.File(os.path.join(out, "h5cam.analysis.h5"), "w") as h:
    h["tracks"] = tracks
    h["node_names"] = [b"nose", b"tail"]
    h["track_names"] = [b"mouse1", b"h5_only"]
    h["instance_scores"] = np.array([[0.9, 0.9, 0.9], [0.7, 0.7, 0.7]])

# A large plain file: no session at all, so one placeholder camera per video.
a0, a1 = video("a0.mp4"), video("a1.mp4")
save("project_no_session.slp", frames_on(a0, 10000, 6) + frames_on(a1, 20000, 6), [a0, a1])
# A project a file with images inside can be added to: the same nodes.
pkg_nodes = [n.name for n in sio.load_slp(minimal_pkg, open_videos=False).skeletons[0].nodes]
sk_pkg = sio.Skeleton(nodes=pkg_nodes, edges=[(pkg_nodes[0], pkg_nodes[1])], name="pkg")
m = video("camM.mp4")
sio.Labels(labeled_frames=[sio.LabeledFrame(video=m, frame_idx=f, instances=[
    sio.PredictedInstance.from_numpy(np.array([[73000.0 + f + 10.0 * k, 10.0] for k in range(len(pkg_nodes))]), skeleton=sk_pkg, track=T1, score=0.9,
                                     point_scores=np.full(len(pkg_nodes), 0.8))]) for f in range(3)],
           videos=[m], skeletons=[sk_pkg], tracks=[T1], sessions=[session([("camM", m, [])], groups=False)]).save(os.path.join(out, "project_for_pkg.slp"), embed=False)

# Per-camera files, for section F.
for name, base in (("camA", 10000), ("camB", 20000)):
    v = video(name + ".mp4")
    sio.Labels(labeled_frames=frames_on(v, base, 6), videos=[v], skeletons=[sk], tracks=[T1, T2]).save(os.path.join(out, name + "_alone.slp"), embed=False)
for name, base in (("camC", 71000), ("camD", 72000)):
    v = video(name + ".mp4")
    sio.Labels(labeled_frames=frames_on(v, base, 3), videos=[v], skeletons=[sk], tracks=[T1, T2]).save(os.path.join(out, name + "_alone.slp"), embed=False)

# ---- Poses Track All can really group: animals in 3D, seen by calibrated cameras.
def real_camera(name, angle, tx):
    return sio.Camera(name=name, rvec=np.array([0.0, angle, 0.0]), tvec=np.array([tx, 0.0, 1000.0]),
                      matrix=np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]]), dist=np.zeros(5), size=(640, 480))

def project_point(cam, p):
    c, s_ = np.cos(cam.rvec[1]), np.sin(cam.rvec[1])
    R = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
    q = R @ np.asarray(p, float) + np.asarray(cam.tvec, float)
    return [round(800.0 * q[0] / q[2] + 320.0, 1), round(800.0 * q[1] / q[2] + 240.0, 1)]   # a save keeps f32

def real_frames(v, cam, n=30):
    return [sio.LabeledFrame(video=v, frame_idx=f, instances=[
        sio.PredictedInstance.from_numpy(np.array([project_point(cam, [-150.0 + 300.0 * animal + 2.0 * f, 30.0 * node - 15.0, 40.0 * node])
                                                   for node in range(2)]), skeleton=sk, score=0.9, point_scores=np.array([0.9, 0.9]))
        for animal in range(2)]) for f in range(n)]

cams = [real_camera("camA", -0.5, -50.0), real_camera("camB", 0.0, 0.0), real_camera("camC", 0.5, 50.0)]
va, vb, vc = video("camA.mp4"), video("camB.mp4"), video("camC.mp4")
s = sio.RecordingSession(camera_group=sio.CameraGroup(cameras=cams[:2]))
s.add_video(va, cams[0]); s.add_video(vb, cams[1])
sio.Labels(labeled_frames=real_frames(va, cams[0]) + real_frames(vb, cams[1]), videos=[va, vb], skeletons=[sk], tracks=[], sessions=[s]).save(os.path.join(out, "project_3d.slp"), embed=False)
s = sio.RecordingSession(camera_group=sio.CameraGroup(cameras=[cams[2]]))
s.add_video(vc, cams[2])
sio.Labels(labeled_frames=real_frames(vc, cams[2]), videos=[vc], skeletons=[sk], tracks=[], sessions=[s]).save(os.path.join(out, "added_3d.slp"), embed=False)
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

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'add-slp-lazy-'));
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

/** Answer the video prompts a step puts up, until `window.__done` is set. */
async function settle(page, what) {
    let done = false;
    for (let i = 0; i < 600 && !done; i++) {
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
    await page.evaluate((how) => { window.__done = false; (async () => {
        if (how === 'lazy') await (await import('/loading/session-loader.js')).handleLoadProjectSlpLazy(new File([window.__file], 'project.slp'));
        else await (await import('/import-export/slp-import.js')).handleLoadSlpFile(new File([window.__file], 'project.slp'));
    })().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); }); }, loader);
    await settle(page, loader + ' load');
    return page;
}

/** File ▸ Add SLP, through the real file picker. */
async function addSlp(page, file) {
    page.once('filechooser', fc => fc.setFiles(file));
    await page.evaluate(() => { window.__done = false; (async () => {
        await (await import('/import-export/slp-import.js')).handleAddSlp();
    })().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); }); });
    await settle(page, 'Add SLP ' + path.basename(file));
}

const statusOf = page => page.evaluate(() => (document.getElementById('statusText') || {}).textContent || '');

/** What the playback loader does (`onPlaybackStateChange`), frame `from` to `to`. */
function play(page, from, to, step) {
    return page.evaluate(async ([from, to, step]) => {
        const tri = await import('/pose/triangulation.js');
        const res = await import('/pose/lazy-residency.js');
        const { state } = await import('/ui/app-state.js');
        const report = { evicted: 0, passes: 0, blocked: {}, resident: 0 };
        for (let cur = from; cur <= to; cur += step) {
            state.currentFrame = cur;
            await tri.batchLoadLazyFrames(cur, res.LAZY_PLAYBACK_AHEAD);
            const pass = tri.evictLazyFrames(cur);
            if (!pass) continue;
            report.passes++;
            report.evicted += pass.evicted;
            for (const k in pass.blocked) report.blocked[k] = (report.blocked[k] || 0) + pass.blocked[k];
        }
        report.resident = state.session.frameGroups.size;
        return report;
    }, [from, to, step]);
}

/** One frame as the app holds it, hydrating it if it has to. */
function shownFrame(page, f) {
    return page.evaluate(async (f) => {
        const tri = await import('/pose/triangulation.js');
        const { state } = await import('/ui/app-state.js');
        const s = state.session;
        await tri.ensureLazyFrameData(f);
        const fg = s.frameGroups.get(f);
        const shown = {};
        const row = (cam, inst) => {
            const nodes = [];
            // To 3 places: a LUC3D save keeps points and scores as f32.
            const r = v => (v === v ? Number(v.toFixed(3)) : null);
            for (let k = 0; k < inst.numNodes; k++) nodes.push(inst.hasPoint(k) ? [r(inst.getX(k)), r(inst.getY(k)), r(s.predictedNodeScore(cam, f, inst, k))] : null);
            return [inst.type, inst.trackIdx == null ? null : s.tracks[inst.trackIdx], r(inst.score), JSON.stringify(nodes)];
        };
        for (const c of s.cameras) {
            shown[c.name] = [...fg.getInstances(c.name), ...fg.getUnlinkedInstances(c.name).map(u => u.instance)].map(i => row(c.name, i));
        }
        const groups = (s.instanceGroups.get(f) || []).map(g => [[...g.instances.keys()].sort(), Array.from(g.points3d || []).slice(0, 3)]);
        return { shown, groups };
    }, f);
}

const residentFrames = (page, frames) => page.evaluate(async (frames) => {
    const { state } = await import('/ui/app-state.js');
    return frames.filter(f => state.session.frameGroups.has(f));
}, frames);

/** A digest of every store's rows, table by table, to prove none moved. */
function storeRows(page, known) {
    return page.evaluate(async (known) => {
        const { state } = await import('/ui/app-state.js');
        const loader = state.session.lazyLoader;
        const storesOf = new Map();
        if (loader._projectLabels) storesOf.set(loader._projectLabels._lazyDataStore, 'project');
        for (const [cam, labels] of loader.labelsByCam) if (!storesOf.has(labels._lazyDataStore)) storesOf.set(labels._lazyDataStore, cam);
        const out = {};
        for (const [store, name] of storesOf) {
            if (known && !known[name]) continue;
            const tables = {};
            for (const t of ['framesData', 'instancesData', 'pointsData', 'predPointsData']) {
                const cols = {};
                for (const k of Object.keys(store[t])) {
                    const col = store[t][k];
                    const n = known ? known[name][t][k].rows : col.length;
                    let h = 2166136261;
                    for (let i = 0; i < n; i++) { h ^= Math.round(Number(col[i]) * 1000) | 0; h = Math.imul(h, 16777619); }
                    cols[k] = { rows: n, hash: h, length: col.length };
                }
                tables[t] = cols;
            }
            out[name] = tables;
        }
        const rows = {};
        for (const [cam, map] of loader.frameRowByCam) rows[cam] = [map.size, ...Array.from(map.entries()).slice(0, 4).flat()];
        return { out, rows };
    }, known);
}
/** Same rows, same values, whatever was added after them. */
function sameRows(before, after) {
    const strip = o => JSON.parse(JSON.stringify(o, (k, v) => (k === 'length' ? undefined : v)));
    return Object.keys(before.out).every(name => Object.keys(before.out[name]).every(t =>
        Object.keys(before.out[name][t]).every(k => eq(strip(before.out[name][t][k]), strip(after.out[name][t][k])))));
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
    const poses = [], cameraPoses = [];
    const cameraOfVideo = new Map();
    const sessions = [], groups = [];
    for (let si = 0; si < (slp.sessions || []).length; si++) {
        const s = slp.sessions[si];
        const keys = Object.keys(s.calibration || {}).filter(k => k !== 'metadata');
        const names = keys.map(k => s.calibration[k].name || k);
        const cams = {};
        for (const k in s.camcorder_to_video_idx_map || {}) {
            const name = names[parseInt(k, 10)] !== undefined ? names[parseInt(k, 10)] : k;
            cams[name] = slp.videos[s.camcorder_to_video_idx_map[k]].filename;
            cameraOfVideo.set(s.camcorder_to_video_idx_map[k], name);
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
    for (const f of slp.frames) {
        for (const d of f.instances) {
            const nodes = d.points.map((p, k) => (p ? num(p[0]) + ',' + num(p[1]) + (d.occluded[k] ? '~' : '') : '-')).join(';');
            const scores = d.type === 'predicted' ? [num(d.score)].concat(Array.from(d.pointScores || [], num)).join(',') : '';
            const track = d.trackIdx >= 0 ? slp.tracks[d.trackIdx] : '';
            const rest = [f.frameIdx, d.type === 'predicted' ? 'P' : 'U', track, nodes, scores].join('|');
            poses.push(slp.videos[f.videoIdx].filename + '|' + rest);
            if (cameraOfVideo.has(f.videoIdx)) cameraPoses.push(cameraOfVideo.get(f.videoIdx) + '|' + rest);
        }
    }
    return {
        videos: slp.videos.map(v => v.filename), tracks: slp.tracks,
        sessions, poses: poses.sort(), cameraPoses: cameraPoses.sort(), groups: groups.sort(),
    };
}

async function savedBytes(page) {
    const b64 = await page.evaluate(() => {
        const u = window.__saved;
        let s = '';
        for (let i = 0; i < u.length; i += 32768) s += String.fromCharCode.apply(null, u.subarray(i, i + 32768));
        return btoa(s);
    });
    return Buffer.from(b64, 'base64');
}

/** Save the open project; keep the bytes and the app's own read of them. */
async function saveAndRead(page) {
    await page.evaluate(saveInPage);
    const read = await page.evaluate(readInPage);
    return { bytes: await savedBytes(page), read };
}

let savedSeq = 0;
const pythonWork = [];   // [{ file, what, want }] checked together: one Python run reads every save
function savedForPython(bytes, what, want) {
    const file = path.join(tmp, 'saved-' + (++savedSeq) + '.slp');
    fs.writeFileSync(file, bytes);
    pythonWork.push({ file, what, want });
}

/** Both readers against one expectation. */
function compare(what, got, want) {
    if (got.error) { check(false, what + ': opens', got.error); return; }
    check(eq(got.poses, want.poses), what + ': every pose on its own video', diff(got.poses || [], want.poses));
    if (want.cameraPoses) check(eq(got.cameraPoses, want.cameraPoses), what + ': every pose on its camera', diff(got.cameraPoses || [], want.cameraPoses));
    if (want.groups) check(eq(got.groups, want.groups), what + ': groups point at the right poses', diff(got.groups || [], want.groups));
    if (want.sessions) check(eq((got.sessions || []).map(sortedKeys), want.sessions.map(sortedKeys)), what + ': cameras and their videos', got.sessions);
    if (want.tracks) check(eq(got.tracks, want.tracks), what + ': track list', got.tracks);
    if (want.tvecs) check(Object.entries(want.tvecs).every(([c, t]) => got.tvecs[c] === t), what + ': added cameras keep their calibration', got.tvecs);
    if (want.tracking) check(Object.entries(want.tracking).every(([k, t]) => got.tracking[k] === t), what + ': a row\'s tracking score comes along', got.tracking);
    if (want.skeletons) check(eq(got.skeletons, want.skeletons), what + ': skeletons', got.skeletons);
    if (want.modelLabels) check(eq(sortedKeys(got.modelLabels || {}), sortedKeys(want.modelLabels)), what + ': what a model says about a detection', got.modelLabels);
}

function checkSave(what, saved, want) {
    compare(what + ', app reader', saved.read, Object.assign({}, want, { tvecs: undefined, tracking: undefined, skeletons: undefined, modelLabels: undefined }));
    savedForPython(saved.bytes, what + ', Python', want);
}

/** A file's rows as they should read on `cameras` (video name -> camera name). */
const onCameras = (rows, cameras) => rows.flatMap(r => {
    const i = r.indexOf('|');
    const cam = cameras[r.slice(0, i)];
    return cam ? [cam + r.slice(i)] : [];
});
const rowsOn = (rows, videos) => rows.filter(r => videos.includes(r.slice(0, r.indexOf('|'))));
const merged = (...lists) => [].concat(...lists).sort();

try {
    const minimalPkg = path.join(repoRoot, 'tests', 'fixtures', 'slp-embedded', 'minimal_instance.pkg.slp');
    uv(PY_WRITE, [tmp, minimalPkg]);
    const NAMES = ['project', 'project_plain', 'project_small', 'project_no_session', 'project_for_pkg', 'added', 'added_same', 'added_plain', 'added_all_outside', 'added_sk2', 'added_small', 'camA_alone', 'camB_alone', 'camC_alone', 'camD_alone', 'project_3d', 'added_3d'];
    const sourcePaths = NAMES.map(n => path.join(tmp, n + '.slp'));
    const rawSources = JSON.parse(uv(PY_READ, sourcePaths.concat(minimalPkg)).trim().split('\n').pop());
    const sources = {};
    for (const p of sourcePaths) sources[path.basename(p, '.slp')] = rawSources[p];
    sources.minimal_pkg = rawSources[minimalPkg];
    const readFile = n => fs.readFileSync(path.join(tmp, n + '.slp'));
    const fileOf = n => path.join(tmp, n);

    const ADDED_CAMS = { 'camX.mp4': 'camX', 'camY.mp4': 'camY' };
    const PROJECT_CAMS = { 'camA.mp4': 'camA', 'camB.mp4': 'camB' };

    console.log('Sources, as Python reads them:');
    check(sources.project.videos[0] === 'project_outside.mp4', 'project.slp lists the other video first', sources.project.videos);
    check(sources.project.groups.length === NFRAMES * 2, 'every project frame has two groups with 3D', sources.project.groups.length);
    check(sources.added.poses.some(r => r.includes('~')), 'added.slp has a node placed but not visible');
    check(sources.added.poses.some(r => r.includes('|U|')), 'added.slp has a hand label');
    check(sources.added.poses.some(r => r.includes('|P||')), 'added.slp has a pose with no track');
    check(sources.added.poses.some(r => /\|U\|[^|]*\|[^|]*~;-\|/.test(r)), 'added.slp has a hand label with a hidden node and an empty one');
    const modelLabels = sources.added.modelLabels;
    check(eq(Object.keys(modelLabels).sort(), ['camX|500|30500.000', 'camY|500|40500.000']) && modelLabels['camX|500|30500.000'] === 'female=0.750|mouse=0.500',
        'added.slp has what a model says about a detection on each camera', modelLabels);

    console.log('\nA. The whole path');
    for (const name of wants('A') ? ['project', 'project_plain'] : []) {
        const source = sources[name];
        console.log(' ' + name);
        const page = await open('lazy', readFile(name));
        const before = await storeRows(page);
        const firstBefore = await shownFrame(page, 0);

        await addSlp(page, fileOf('added.slp'));
        const status = await statusOf(page);
        check(/Added SLP: 2 camera\(s\), 7 frames merged/.test(status), 'the status counts the frames', status);
        check(/not grouped yet/.test(status), 'and says they are not grouped', status);
        check(/not shown; saving keeps/.test(status), 'and says what is not shown', status);
        const dirty = await page.evaluate(async () => (await import('/ui/app-state.js')).state.isDirty);
        check(dirty === true, 'the project now has unsaved changes');

        const after = await storeRows(page, before.out);
        check(sameRows(before, after), 'no store row moved or changed');
        for (const cam of Object.keys(before.rows)) check(eq(before.rows[cam], after.rows[cam]), cam + ': its frames are on the same rows', after.rows[cam]);
        check(after.rows.camX && after.rows.camX[0] === 7 && after.rows.camY[0] === 7, 'each added camera has its 7 frames', after.rows);

        // What the app shows, before and after the frame leaves memory.
        const f0 = await shownFrame(page, 0);
        check(f0.shown.camX.length === 2 && f0.shown.camY.length === 2, 'the frame on screen shows the new cameras', f0.shown);
        check(eq(f0.groups, firstBefore.groups) && f0.groups.length === 2, 'its groups and their 3D are still there', f0.groups);
        check(eq(f0.shown.camA, firstBefore.shown.camA) && eq(f0.shown.camB, firstBefore.shown.camB), 'its own cameras show what they did');
        const shown = {};
        for (const f of [0, 2, 500]) shown[f] = await shownFrame(page, f);

        const played = await play(page, 0, NFRAMES, 100);
        check(played.evicted > 0, 'playing far enough runs eviction', played);
        check(!played.blocked.camera, 'no frame is held for a camera the store does not back', played.blocked);
        check(played.blocked.user >= 1, 'a hand label pins its frame', played.blocked);
        check(eq(await residentFrames(page, [0, 2, 500]), []), 'frames holding added predictions left memory');
        check(eq(await residentFrames(page, [1]), [1]), 'the one with a hand label did not');

        const again = {};
        for (const f of [0, 2, 500]) again[f] = await shownFrame(page, f);
        check(eq(again, shown), 'each comes back from the store as it was', { want: shown[0], got: again[0] });
        const played2 = await play(page, 0, NFRAMES, 100);
        check(played2.evicted > 0 && eq(await residentFrames(page, [0, 2, 500]), []), 'and leaves again');

        const want = {
            poses: merged(source.poses, sources.added.poses),
            cameraPoses: merged(onCameras(source.poses, PROJECT_CAMS), onCameras(sources.added.poses, ADDED_CAMS)),
            groups: source.groups,
            sessions: [Object.assign({}, source.sessions[0], { camX: 'camX.mp4', camY: 'camY.mp4' })],
            tvecs: { camX: 70, camY: 71 },
            tracking: { 'camX|500': 0.25, 'camY|2100': 0.25 },
            skeletons: [['nose', 'tail']],
            modelLabels,
        };
        const save1 = await saveAndRead(page);
        checkSave(name + ', save 1', save1, want);
        const save2 = await saveAndRead(page);
        checkSave(name + ', save 2, same page', save2, want);
        const stillShown = await shownFrame(page, 0);
        check(eq(stillShown, shown[0]), 'the project is unchanged by saving', stillShown);
        await page.close();

        const lazyAgain = await open('lazy', save2.bytes);
        check(eq((await shownFrame(lazyAgain, 500)).shown.camX, shown[500].shown.camX), 'reopened lazily, frame 500 shows what it did');
        const save3 = await saveAndRead(lazyAgain);
        checkSave(name + ', save 3, reopened lazily', save3, want);
        await lazyAgain.close();

        const eagerAgain = await open('eager', save3.bytes);
        const save4 = await saveAndRead(eagerAgain);
        checkSave(name + ', save 4, reopened eagerly', save4, Object.assign({}, want, { tracking: undefined }));
        await eagerAgain.close();
    }

    console.log('\nB. The project\'s own names, another node order, on top of another add');
    if (wants('B')) {
        const source = sources.project;
        const page = await open('lazy', readFile('project'));
        await addSlp(page, fileOf('added.slp'));
        await addSlp(page, fileOf('added_same.slp'));
        const status = await statusOf(page);
        check(/Added SLP: 2 camera\(s\), 2 frames merged/.test(status), 'the second add counts its own frames', status);
        // The file lists tail, nose: the project nose, tail.
        const swapNodes = r => {
            const c = r.split('|');
            c[4] = c[4].split(';').reverse().join(';');
            const s = c[5].split(',');
            c[5] = [s[0], s[2], s[1]].join(',');
            return c.join('|');
        };
        const same = sources.added_same.poses.map(swapNodes);
        const want = {
            poses: merged(source.poses, sources.added.poses, same),
            cameraPoses: merged(onCameras(source.poses, PROJECT_CAMS), onCameras(sources.added.poses, ADDED_CAMS),
                same.filter(r => r.includes('|P|mouse1|60')).map(r => 'camA_2' + r.slice(r.indexOf('|'))),
                same.filter(r => r.includes('|P|mouse1|61')).map(r => 'camB_2' + r.slice(r.indexOf('|')))),
            groups: source.groups,
            sessions: [{ camA: 'camA.mp4', camB: 'camB.mp4', camX: 'camX.mp4', camY: 'camY.mp4', camA_2: 'camA.mp4', camB_2: 'camB.mp4' }],
            skeletons: [['nose', 'tail']],
            tvecs: { camA_2: 80, camB_2: 81 },
        };
        const shown = await shownFrame(page, 1);
        check(eq(shown.shown.camA_2.map(r => JSON.parse(r[3]).map(n => n[0])), [[60001, 60001]]), 'each node shows on the right skeleton node', shown.shown.camA_2);
        checkSave('two adds, save 1', await saveAndRead(page), want);
        const save2 = await saveAndRead(page);
        checkSave('two adds, save 2', save2, want);
        await page.close();
        const again = await open('lazy', save2.bytes);
        checkSave('two adds, reopened', await saveAndRead(again), want);
        await again.close();
    }

    console.log('\nC. Files with less');
    if (wants('C')) {
        // No session and no tracks: a camera named after its video, and no track made up.
        const page = await open('lazy', readFile('project_small'));
        const tracksBefore = await page.evaluate(async () => (await import('/ui/app-state.js')).state.session.tracks.slice());
        await addSlp(page, fileOf('added_plain.slp'));
        const tracksAfter = await page.evaluate(async () => (await import('/ui/app-state.js')).state.session.tracks.slice());
        check(eq(tracksAfter, tracksBefore), 'a file with no tracks adds none', tracksAfter);
        const want = {
            poses: merged(sources.project_small.poses, sources.added_plain.poses),
            groups: sources.project_small.groups,
            sessions: [{ camA: 'camA.mp4', camB: 'camB.mp4', plain_cam: 'plain_cam.mp4' }],
            tracks: sources.project_small.tracks,
        };
        const saved = await saveAndRead(page);
        checkSave('no session, no tracks', saved, want);
        checkSave('no session, no tracks, save 2', await saveAndRead(page), want);
        await page.close();
    }
    if (wants('C')) {
        // No frame on any camera: the cameras stay, with their videos' names, and the frames stay on their own.
        const page = await open('lazy', readFile('project_small'));
        await addSlp(page, fileOf('added_all_outside.slp'));
        const status = await statusOf(page);
        check(/2 camera\(s\), 0 frames merged/.test(status) && /not shown; saving keeps/.test(status), 'the status says none merged, and what is not shown', status);
        const want = {
            poses: merged(sources.project_small.poses, sources.added_all_outside.poses),
            groups: sources.project_small.groups,
            sessions: [{ camA: 'camA.mp4', camB: 'camB.mp4', camP: 'camP.mp4', camQ: 'camQ.mp4' }],
            tvecs: { camP: 90, camQ: 91 },
        };
        checkSave('every frame on another video', await saveAndRead(page), want);
        checkSave('every frame on another video, save 2', await saveAndRead(page), want);
        await page.close();
    }
    if (wants('C')) {
        // A second skeleton: its poses stay on it.
        const page = await open('lazy', readFile('project_small'));
        await addSlp(page, fileOf('added_sk2.slp'));
        const saved = await saveAndRead(page);
        // The app's reader reads every pose on the first skeleton, so only Python can tell.
        savedForPython(saved.bytes, 'a second skeleton, Python', {
            poses: merged(sources.project_small.poses, sources.added_sk2.poses),
            groups: sources.project_small.groups,
            skeletons: [['nose', 'tail'], ['a', 'b', 'c']],
        });
        await page.close();
    }
    if (wants('C')) {
        // An analysis .h5, parsed eagerly: every node takes its instance's score.
        const page = await open('lazy', readFile('project_small'));
        await addSlp(page, fileOf('h5cam.analysis.h5'));
        const h5Rows = [];
        for (let f = 0; f < 3; f++) {
            h5Rows.push(`h5cam|${f}|P|mouse1|${68000 + f}.000,10.000;${68500 + f}.000,20.000|0.900,0.900,0.900`);
            h5Rows.push(`h5cam|${f}|P|h5_only|${68100 + f}.000,11.000;-|0.700,0.700,0.700`);
        }
        const want = {
            poses: merged(sources.project_small.poses, h5Rows),
            groups: sources.project_small.groups,
            sessions: [{ camA: 'camA.mp4', camB: 'camB.mp4', 'h5cam': 'h5cam' }],
        };
        const saved = await saveAndRead(page);
        checkSave('an analysis .h5', saved, want);
        checkSave('an analysis .h5, save 2', await saveAndRead(page), want);
        await page.close();
    }

    if (wants('C')) {
        // A plain file over 200 MB opens on this route too: one placeholder camera per video.
        const page = await open('lazy', readFile('project_no_session'));
        const before = await page.evaluate(async () => (await import('/ui/app-state.js')).state.session.cameras.map(c => c.name));
        check(eq(before, ['a0', 'a1']), 'a plain file opens with a camera per video', before);
        await addSlp(page, fileOf('added.slp'));
        const want = {
            poses: merged(sources.project_no_session.poses, sources.added.poses),
            cameraPoses: merged(onCameras(sources.project_no_session.poses, { 'a0.mp4': 'a0', 'a1.mp4': 'a1' }), onCameras(sources.added.poses, ADDED_CAMS)),
            sessions: [{ a0: 'a0.mp4', a1: 'a1.mp4', camX: 'camX.mp4', camY: 'camY.mp4' }],
            modelLabels,
        };
        checkSave('a plain file, save 1', await saveAndRead(page), want);
        const save2 = await saveAndRead(page);
        checkSave('a plain file, save 2', save2, want);
        await page.close();
        const again = await open('lazy', save2.bytes);
        checkSave('a plain file, reopened', await saveAndRead(again), want);
        await again.close();
    }
    if (wants('C')) {
        // A file with images inside: its images show, its poses go on the camera its
        // video becomes, and a save writes no images, so that video becomes its source.
        const page = await open('lazy', readFile('project_for_pkg'));
        const beforeRows = await storeRows(page);
        await addSlp(page, minimalPkg);
        check(sameRows(beforeRows, await storeRows(page, beforeRows.out)), 'no store row moved');
        const added = await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            const s = state.session;
            const cam = s.cameras[s.cameras.length - 1].name;
            const view = state.views.find(v => v.name === cam);
            return { cam, frames: view && view.decoder ? view.decoder.samples.length : 0, status: (document.getElementById('statusText') || {}).textContent };
        });
        check(added.cam !== 'camM' && added.frames > 0, 'its view shows its images', added);
        const pkgPoses = sources.minimal_pkg.poses.map(r => added.cam + r.slice(r.indexOf('|')));
        const saved = await saveAndRead(page);
        const want = {
            cameraPoses: merged(onCameras(sources.project_for_pkg.poses, { 'camM.mp4': 'camM' }), pkgPoses),
        };
        check(eq(saved.read.cameraPoses, want.cameraPoses), 'every pose is on its camera, app reader', diff(saved.read.cameraPoses, want.cameraPoses));
        savedForPython(saved.bytes, 'a file with images inside, Python', { poses: saved.read.poses, cameraPoses: want.cameraPoses });
        const second = await saveAndRead(page);
        savedForPython(second.bytes, 'a file with images inside, save 2, Python', { poses: saved.read.poses, cameraPoses: want.cameraPoses });
        await page.close();
    }

    console.log('\nD. Edits on added poses');
    const projectAdded = () => merged(sources.project.poses, sources.added.poses);
    const editSessions = [Object.assign({}, sources.project.sessions[0], { camX: 'camX.mp4', camY: 'camY.mp4' })];
    if (wants('D')) {
        // Delete one: the store compacts, and every row after it keeps its pose.
        const page = await open('lazy', readFile('project'));
        await addSlp(page, fileOf('added.slp'));
        await page.evaluate(async () => {
            const tri = await import('/pose/triangulation.js');
            const del = await import('/ui/custom-delete-ops.js');
            const { state } = await import('/ui/app-state.js');
            await tri.ensureLazyFrameData(500);
            const fg = state.session.frameGroups.get(500);
            const ul = fg.getUnlinkedInstances('camX')[0];
            del.deleteTargetsFromStore(state.session, [del.unlinkedTarget(500, ul)]);
            fg.unlinkedInstances.get('camX').splice(0, 1);
        });
        const gone = /^camX\.mp4\|500\|P\|mouse1\|/;
        const want = { poses: projectAdded().filter(r => !gone.test(r)), groups: sources.project.groups, sessions: editSessions };
        checkSave('a deleted added pose', await saveAndRead(page), want);
        checkSave('a deleted added pose, save 2', await saveAndRead(page), want);
        await page.close();
    }
    if (wants('D')) {
        // Correct one: it goes first in the file and its store row is skipped, so the
        // file-wide group refs must still line up.
        const page = await open('lazy', readFile('project'));
        await addSlp(page, fileOf('added.slp'));
        await page.evaluate(async () => {
            const tri = await import('/pose/triangulation.js');
            const { state } = await import('/ui/app-state.js');
            await tri.ensureLazyFrameData(2);
            const inst = state.session.frameGroups.get(2).getUnlinkedInstances('camY')[0].instance;
            inst.setPoint(0, 999, 999);
            inst.type = 'user';
            inst.modified = true;
        });
        const old = /^camY\.mp4\|2\|P\|\|/;
        const want = {
            poses: merged(projectAdded().filter(r => !old.test(r)), ['camY.mp4|2|U||999.000,999.000;40003.000,20.000|']),
            groups: sources.project.groups,
            sessions: editSessions,
        };
        checkSave('a corrected added pose', await saveAndRead(page), want);
        const save2 = await saveAndRead(page);
        checkSave('a corrected added pose, save 2', save2, want);
        await page.close();
        const again = await open('lazy', save2.bytes);
        checkSave('a corrected added pose, reopened', await saveAndRead(again), want);
        await again.close();
    }
    if (wants('D')) {
        // Delete a track: every camera's rows lose it, the other videos' keep their names.
        const page = await open('lazy', readFile('project'));
        await addSlp(page, fileOf('added.slp'));
        await page.evaluate(async () => {
            const ops = await import('/ui/track-identity-ops.js');
            const { state } = await import('/ui/app-state.js');
            ops.deleteTrackAt(state.session, state.session.tracks.indexOf('mouse1'));
        });
        const other = ['project_outside.mp4', 'added_outside.mp4'];
        const lose = r => (other.includes(r.slice(0, r.indexOf('|'))) ? r : r.replace('|P|mouse1|', '|P||').replace('|U|mouse1|', '|U||'));
        // A group on the deleted track dissolves; mouse2's stay.
        const kept = sources.project.groups.filter(g => { const c = g.split('|'); return Number(c[2].split('=')[1]) === 10010 + Number(c[1]); });
        const want = { poses: projectAdded().map(lose).sort(), groups: kept, sessions: editSessions };
        checkSave('a deleted track', await saveAndRead(page), want);
        checkSave('a deleted track, save 2', await saveAndRead(page), want);
        await page.close();
    }
    if (wants('D')) {
        // Group two added poses by hand: a group's refs reach added rows, and the frame's
        // other groups stay.
        const page = await open('lazy', readFile('project'));
        await addSlp(page, fileOf('added.slp'));
        await page.evaluate(async () => {
            const pd = await import('/pose/pose-data.js');
            const tri = await import('/pose/triangulation.js');
            const { state } = await import('/ui/app-state.js');
            await tri.ensureLazyFrameData(500);
            const s = state.session;
            const fg = s.frameGroups.get(500);
            const g = new pd.InstanceGroup(987654, null);
            for (const cam of ['camX', 'camY']) {
                const ul = fg.unlinkedInstances.get(cam).shift();
                g.addInstance(cam, ul.instance);
                fg.addInstance(cam, ul.instance);
            }
            g.points3d = new Float64Array([7, 8, 9, 10, 11, 12]);
            s.instanceGroups.get(500).push(g);
        });
        const want = {
            poses: projectAdded(),
            groups: merged(sources.project.groups, ['s0|500|camX=30500.000|camY=40500.000|7.000,8.000,9.000']),
            sessions: editSessions,
        };
        checkSave('a group of added poses', await saveAndRead(page), want);
        const save2 = await saveAndRead(page);
        checkSave('a group of added poses, save 2', save2, want);
        await page.close();
        const again = await open('lazy', save2.bytes);
        const played = await play(again, 0, NFRAMES, 100);
        check(played.evicted > 0, 'reopened, it plays and evicts', played);
        checkSave('a group of added poses, reopened and played', await saveAndRead(again), want);
        await again.close();
    }

    console.log('\nE. Track All');
    if (wants('E')) {
        // A sweep's window release drops every prediction-only frame.
        const page = await open('lazy', readFile('project'));
        await addSlp(page, fileOf('added.slp'));
        const swept = await page.evaluate(async () => {
            const tri = await import('/pose/triangulation.js');
            const { state } = await import('/ui/app-state.js');
            let camX = 0;
            await tri.sweepLazyFrameWindows(state.session, (f, fg) => {
                if (fg) camX += fg.getInstances('camX').length + fg.getUnlinkedInstances('camX').length;
            }, { window: 300 });
            return { camX, resident: state.session.frameGroups.has(500) };
        });
        check(swept.camX === 14, 'a sweep reads every added pose', swept);
        check(swept.resident === false, 'and releases their frames', swept);
        checkSave('after a sweep', await saveAndRead(page), { poses: projectAdded(), groups: sources.project.groups, sessions: editSessions });
        await page.close();
    }
    if (wants('E')) {
        // The status's promise, on poses that really do project from 3D.
        const page = await open('lazy', readFile('project_3d'));
        await addSlp(page, fileOf('added_3d.slp'));
        const result = await page.evaluate(async () => {
            const tk = await import('/pose/tracker.js');
            const settings = await import('/ui/settings.js');
            const { state } = await import('/ui/app-state.js');
            const prev = settings.getTrackingThresholds();
            settings.setTrackingThresholds(Object.assign({}, prev, { autoSwitchCheck: 0, autoImageSwitchCheck: 0 }));
            tk.setTrackerNumAnimals(2);   // or the first-use dialog waits for an answer
            try {
                await tk.trackAll();
                await (await import('/pose/triangulation.js')).triangulateAllFrames('dlt');
            } catch (e) { return { err: String(e && e.stack || e).slice(0, 400) }; } finally { settings.setTrackingThresholds(prev); }
            const s = state.session;
            let groups = 0, withC = 0, worst = 0;
            for (const [f, gs] of s.instanceGroups) {
                for (const g of gs) {
                    groups++;
                    if (g.instances.has('camC')) withC++;
                    // Node 0 of animal a is at (-150 + 300 a + 2 f, -15, 0).
                    const [x, y, z] = g.points3d;
                    const a = x > 0 ? 1 : 0;
                    worst = Math.max(worst, Math.abs(x - (-150 + 300 * a + 2 * f)), Math.abs(y + 15), Math.abs(z));
                }
            }
            return { groups, withC, worst };
        });
        check(!result.err, 'Track All and Triangulate All run', result.err);
        check(result.groups === 60 && result.withC === 60, 'Track All groups the added camera with the others', result);
        check(result.worst < 2, 'and the 3D solves to where the animals are', result);
        const saved = await saveAndRead(page);
        const noTrack = r => { const c = r.split('|'); c[3] = ''; return c.join('|'); };
        const wantPoses = merged(sources.project_3d.poses, sources.added_3d.poses).map(noTrack).sort();
        check(eq(saved.read.poses.map(noTrack).sort(), wantPoses), 'every pose is still where it was, app reader', diff(saved.read.poses.map(noTrack).sort(), wantPoses));
        check(saved.read.groups.length === 60 && saved.read.groups.every(g => /camA=.*camB=.*camC=/.test(g)), 'and the save holds the groups, app reader', saved.read.groups.slice(0, 2));
        savedForPython(saved.bytes, 'after Track All and Triangulate All, Python', { poses: saved.read.poses, groups: saved.read.groups });
        await page.close();
    }

    console.log('\nF. A project opened from per-camera files');
    if (wants('F')) {
        const page = await newAppPage();
        await putBytes(page, '__camA', readFile('camA_alone'));
        await putBytes(page, '__camB', readFile('camB_alone'));
        // The folder goes lazy on a file's size alone, and HDF5 ignores trailing bytes.
        const loading = page.evaluate(async () => {
            const loader = await import('/loading/session-loader.js');
            const tri = await import('/pose/triangulation.js');
            const pad = [new Uint8Array(tri.LAZY_SLP_THRESHOLD + (1 << 20))];
            const mkFile = (bytes, rel) => {
                const f = new File([bytes].concat(pad), rel.split('/').pop());
                Object.defineProperty(f, 'webkitRelativePath', { value: rel });
                return f;
            };
            await loader.handleLoadSessionFolderPerCamera([mkFile(window.__camA, 'root/camA/camA.slp'), mkFile(window.__camB, 'root/camB/camB.slp')], true);
        });
        // These folders hold no videos, so the load waits on the Missing Video Files popup.
        await page.getByRole('button', { name: 'Continue', exact: true }).click();
        await loading;
        const before = await storeRows(page);
        await addSlp(page, fileOf('added.slp'));
        const addStatus = await statusOf(page);
        check(/Added SLP: 2 camera\(s\), 7 frames merged/.test(addStatus), 'the status counts the frames', addStatus);
        const after = await storeRows(page, before.out);
        check(sameRows(before, after), 'the cameras\' own stores keep every row');
        const kinds = await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            const l = state.session.lazyLoader;
            return { shared: !!l._sharedStore, stores: new Set(l.labelsByCam.values()).size, outside: l.outsideFrameCounts() };
        });
        check(kinds.shared === false && kinds.stores === 4, 'each added camera has a store of its own', kinds);
        check(kinds.outside.frames === 3 && kinds.outside.videos === 1, 'the loader counts the other video\'s frames', kinds.outside);
        const shown = await shownFrame(page, 500);
        check(shown.shown.camX && shown.shown.camX.length === 2, 'an added frame past the project\'s shows', shown.shown);
        const poses = merged(sources.camA_alone.poses, sources.camB_alone.poses, sources.added.poses);
        const want = { poses, sessions: [{ camA: 'camA.mp4', camB: 'camB.mp4', camX: 'camX.mp4', camY: 'camY.mp4' }], tvecs: { camX: 70 } };
        checkSave('per-camera, save 1', await saveAndRead(page), want);
        const save2 = await saveAndRead(page);
        checkSave('per-camera, save 2', save2, want);
        await page.close();
        const again = await open('lazy', save2.bytes);
        checkSave('per-camera, reopened', await saveAndRead(again), want);
        await again.close();
    }
    if (wants('F')) {
        // A per-camera project, no frame on any added camera: the first one's store holds the other video.
        const page = await newAppPage();
        await putBytes(page, '__camA', readFile('camA_alone'));
        await putBytes(page, '__camB', readFile('camB_alone'));
        const loading = page.evaluate(async () => {
            const loader = await import('/loading/session-loader.js');
            const tri = await import('/pose/triangulation.js');
            const pad = [new Uint8Array(tri.LAZY_SLP_THRESHOLD + (1 << 20))];
            const mkFile = (bytes, rel) => {
                const f = new File([bytes].concat(pad), rel.split('/').pop());
                Object.defineProperty(f, 'webkitRelativePath', { value: rel });
                return f;
            };
            await loader.handleLoadSessionFolderPerCamera([mkFile(window.__camA, 'root/camA/camA.slp'), mkFile(window.__camB, 'root/camB/camB.slp')], true);
        });
        await page.getByRole('button', { name: 'Continue', exact: true }).click();
        await loading;
        await addSlp(page, fileOf('added_all_outside.slp'));
        const want = {
            poses: merged(sources.camA_alone.poses, sources.camB_alone.poses, sources.added_all_outside.poses),
            sessions: [{ camA: 'camA.mp4', camB: 'camB.mp4', camP: 'camP.mp4', camQ: 'camQ.mp4' }],
        };
        checkSave('per-camera, every frame on another video', await saveAndRead(page), want);
        checkSave('per-camera, every frame on another video, save 2', await saveAndRead(page), want);
        await page.close();
    }

    console.log('\nG. Other save routes');
    if (wants('G')) {
        // Two lazy sessions: pass 2 re-reads each from its files, which do not hold added rows.
        const page = await open('lazy', readFile('project'));
        await addSlp(page, fileOf('added.slp'));
        await putBytes(page, '__camC', readFile('camC_alone'));
        await putBytes(page, '__camD', readFile('camD_alone'));
        const err = await page.evaluate(async () => {
            const pd = await import('/pose/pose-data.js');
            const { SioLazyLoader } = await import('/loading/sio-lazy-loader.js');
            const AS = await import('/ui/app-state.js');
            const sl = await import('/import-export/save-load.js');
            const names = ['camC', 'camD'];
            const mtx = [[500, 0, 32], [0, 500, 32], [0, 0, 1]];
            const cams = names.map((n, i) => new pd.Camera(n, mtx, [0, 0, 0, 0, 0], [0, 0.1 * i, 0], [10 * i, 0, 0], [64, 64]));
            const second = new pd.Session(cams, AS.state.session.skeleton, ['mouse1', 'mouse2'], 'second');
            const loader = new SioLazyLoader();
            await loader.open('camC', new File([window.__camC], 'camC.slp'));
            await loader.open('camD', new File([window.__camD], 'camD.slp'));
            second.lazyLoader = loader;
            AS.state.sessions.push(second);
            try {
                window.__saved = await sl.saveAllSessionsStreaming(AS.state.sessions.slice());
                return null;
            } catch (e) { return String(e && e.stack || e); }
        });
        check(!err, 'the two-pass save runs', err);
        if (!err) {
            const read = await page.evaluate(readInPage);
            const bytes = await savedBytes(page);
            checkSave('a lazy multi-session save', { bytes, read }, {
                poses: merged(sources.project.poses, sources.added.poses, sources.camC_alone.poses, sources.camD_alone.poses),
                groups: sources.project.groups,
                sessions: [Object.assign({}, sources.project.sessions[0], { camX: 'camX.mp4', camY: 'camY.mp4' }), { camC: 'camC.mp4', camD: 'camD.mp4' }],
            });
        }
        await page.close();
    }
    if (wants('G')) {
        // A project mixing lazy and eager goes through the eager writer, every lazy frame resident.
        const page = await open('lazy', readFile('project_small'));
        await addSlp(page, fileOf('added_small.slp'));
        await page.evaluate(async () => {
            const pd = await import('/pose/pose-data.js');
            const tri = await import('/pose/triangulation.js');
            const { state } = await import('/ui/app-state.js');
            for (let f = 0; f < 6; f++) await tri.ensureLazyFrameData(f);
            const cam = new pd.Camera('camE', [[500, 0, 32], [0, 500, 32], [0, 0, 1]], [0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0], [64, 64]);
            const eager = new pd.Session([cam], state.session.skeleton, ['mouse1'], 'eager');
            const fg = new pd.FrameGroup(0);
            eager.addFrameGroup(fg);
            fg.addInstance('camE', new pd.Instance([[900, 10], [901, 20]], 0, 'user', 0));
            state.sessions.push(eager);
        });
        const saved = await saveAndRead(page);
        const added = rowsOn(sources.added_small.poses, ['small_outside.mp4']);
        check(added.every(r => saved.read.poses.includes(r)), 'the added other video keeps its poses', diff(saved.read.poses, added));
        // Per-node scores aside: this writer gives a lazy row's nodes its instance score,
        // the project's own rows too.
        const noScores = r => r.slice(0, r.lastIndexOf('|'));
        const onCam = onCameras(sources.added_small.poses, { 'camS.mp4': 'camS' }).map(noScores);
        const savedOnCam = saved.read.cameraPoses.map(noScores);
        check(onCam.every(r => savedOnCam.includes(r)), 'the added camera keeps its poses', diff(savedOnCam, onCam));
        savedForPython(saved.bytes, 'mixed project, Python', { poses: saved.read.poses, groups: saved.read.groups });
        await page.close();
    }

    console.log('\nH. What must be refused, and a failure half way');
    if (wants('H')) {
        const page = await open('lazy', readFile('project_small'));
        const shape = () => page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            const s = state.session;
            return { cams: s.cameras.map(c => c.name), tracks: s.tracks.slice(), loaderCams: Array.from(s.lazyLoader.labelsByCam.keys()), dirty: state.isDirty };
        });
        const before = await shape();
        const beforeRows = await storeRows(page);

        // Out of memory (or anything else) while finding room: nothing has changed yet.
        page.once('filechooser', fc => fc.setFiles(fileOf('added.slp')));
        await page.evaluate(() => { window.__done = false; (async () => {
            const { state } = await import('/ui/app-state.js');
            const real = state.session.lazyLoader.prepareAddedRows;
            state.session.lazyLoader.prepareAddedRows = () => { throw new RangeError('Array buffer allocation failed'); };
            await (await import('/import-export/slp-import.js')).handleAddSlp();
            state.session.lazyLoader.prepareAddedRows = real;
        })().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); }); });
        await settle(page, 'Add SLP that fails');
        check(/Add SLP error: Array buffer allocation failed/.test(await statusOf(page)), 'running out of memory is reported', await statusOf(page));
        check(eq(await shape(), before), 'and changes nothing', await shape());
        check(sameRows(beforeRows, await storeRows(page, beforeRows.out)), 'not a row');

        // A file too new to read: the reader refuses it before anything changes.
        await addSlp(page, fileOf('added_too_new.slp'));
        const newer = await statusOf(page);
        check(/uses SLP format 3\.0, which is newer than this version of LUC3D/.test(newer), 'a file too new to read is refused, and says why', newer);
        check(eq(await shape(), before), 'and changes nothing', await shape());
        check(sameRows(beforeRows, await storeRows(page, beforeRows.out)), 'not a row');

        // A store too old to hold today's coordinates.
        await page.evaluate(async () => { (await import('/ui/app-state.js')).state.session.lazyLoader._projectLabels._lazyDataStore.formatId = 1.0; });
        await page.evaluate(() => { window.__done = false; (async () => {
            await (await import('/import-export/slp-import.js')).handleAddSlp();
        })().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); }); });
        await settle(page, 'Add SLP on an old store');
        const oldStatus = await statusOf(page);
        check(/^Nothing added: .*very old SLEAP/.test(oldStatus), 'an old store is refused, and says why', oldStatus);
        check(eq(await shape(), before), 'and nothing changes', await shape());
        await page.evaluate(async () => { (await import('/ui/app-state.js')).state.session.lazyLoader._projectLabels._lazyDataStore.formatId = 1.4; });

        // A loader with no store (analysis .h5 per camera).
        const real = await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            window.__realLoader = state.session.lazyLoader;
            state.session.lazyLoader = { isSync: false, workers: new Map(), nFrames: 6 };
            window.__done = false;
            await (await import('/import-export/slp-import.js')).handleAddSlp();
            const status = (document.getElementById('statusText') || {}).textContent;
            state.session.lazyLoader = window.__realLoader;
            return status;
        });
        check(/^Nothing added: .*analysis \.h5/.test(real), 'a loader with no store is refused, and says why', real);
        await page.close();
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
