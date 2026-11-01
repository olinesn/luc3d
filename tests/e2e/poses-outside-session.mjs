/**
 * poses-outside-session.mjs — poses on a video no recording session names stay on
 * that video, however the file is opened and however often it is saved. The promise
 * in the header of `import-export/outside-frames.js`.
 *
 * A lazy save used to write the whole columnar store under a header listing one video
 * per camera, so those frames' rows pointed past the end of the list (Python's
 * `load_slp` raises IndexError) or, when the video's id equalled a camera's new index,
 * replaced that camera's poses (first write wins). An eager save dropped them. The
 * same went for the other sessions' frames of a multi-session file, because the lazy
 * loader and the single-`.slp` folder loader open only the first session.
 *
 * Real Python sleap-io writes the files. Each goes through a real loader, File ▸ Save,
 * the same loader again and File ▸ Save again. Every save is read with the app's own
 * reader and with Python, and both must report exactly the source's poses and groups.
 *
 *  A. Every layout through every loader (see `FILES`).
 *  B. Edits on a lazy project: a corrected frame (the store row it replaces is skipped,
 *     so the file-wide group refs must still line up), and a deleted track (outside
 *     rows keep their track name).
 *  C. A lazy multi-session save (two passes): the header holds another session's
 *     videos after this one's, including a project with no frame on any of its cameras
 *     and a per-camera session whose file holds another video.
 *  D. Edits on an eager project: skeleton node removed / added, track deleted.
 *  E. File ▸ Add SLP: a video no camera has is not merged into a camera's poses.
 *  F. A file an older LUC3D already damaged: frames on a video the list does not hold.
 *  G. A per-camera folder whose files hold frames on another video, or only on their
 *     second listed video: both routes (the folder goes lazy over 150 MB), then a
 *     deleted track on the lazy one.
 *  H. A second skeleton on a lazy save: the header lists what the rows use.
 *  I. A lazy session on the eager writer's path (a project mixing lazy and eager).
 *
 * Needs `uv` for Python (skipped without it).
 *
 * Run: node tests/e2e/poses-outside-session.mjs   (spawns its own http.server; PORT= to move it)
 */
import { chromium } from 'playwright';
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8462);
const PY_SPEC = process.env.SLEAP_IO_SPEC || 'sleap-io==0.9.5';

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
    return { missing: want.filter(x => !g.has(x)).slice(0, 4), extra: got.filter(x => !w.has(x)).slice(0, 4) };
};

// One line per pose: video|frame|U or P|track|nodes|scores. A node is `x,y`, `x,y~` when
// it is placed but not visible, `-` when it has no position. Each video's x values are
// its own, so a pose that moved cannot pass for the one it replaced.
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
    sessions, groups = [], []
    for si, s in enumerate(labels.sessions):
        cams = {}
        for c in s.camera_group.cameras:
            v = s.get_video(c)
            if v is not None:
                cams[c.name] = v.filename
        sessions.append(cams)
        for f, fg in s.frame_groups.items():
            for ig in fg.instance_groups:
                members = sorted(c.name + "=" + num(i.points["xy"][0][0]) for c, i in ig.instance_by_camera.items())
                p3 = "" if ig.instance_3d is None else ",".join(num(v) for v in np.asarray(ig.instance_3d.points)[0][:3])
                groups.append("|".join(["s" + str(si), str(f)] + members + [p3]))
    return {
        "error": None, "videos": [v.filename for v in labels.videos], "tracks": [t.name for t in labels.tracks],
        "skeleton": [n.name for n in labels.skeletons[0].nodes],
        "sessions": sessions, "poses": poses, "groups": sorted(groups),
    }
`;

const PY_READ = PY_COMMON + `
print(json.dumps({p: read(p) for p in sys.argv[1:]}))
`;

const PY_WRITE = PY_COMMON + `
from sleap_io.model.camera import Instance3D
import h5py, shutil

out = sys.argv[1]
sk = sio.Skeleton(nodes=["nose", "tail"], edges=[("nose", "tail")])
T1, T2, T3 = sio.Track(name="mouse1"), sio.Track(name="mouse2"), sio.Track(name="only_outside")

def video(name):
    v = sio.Video(filename=name, open_backend=False)
    v.backend_metadata["shape"] = (400, 64, 64, 1)
    return v

def frames_on(v, base, n=3, tracks=(T1, T2)):
    return [sio.LabeledFrame(video=v, frame_idx=f, instances=[
        sio.PredictedInstance.from_numpy(np.array([[base + 10.0 * i + f, 10.0], [base + 20.0 * i + f, 20.0]]), skeleton=sk, track=t,
                                         score=0.9, point_scores=np.array([0.5 + 0.1 * i, 0.8]))
        for i, t in enumerate(tracks)]) for f in range(n)]

def session(pairs):
    cams = [sio.Camera(name=n, rvec=np.array([0.0, 0.1 * i, 0.0]), tvec=np.array([10.0 * i, 0.0, 0.0]),
                       matrix=np.array([[500.0, 0, 32], [0, 500.0, 32], [0, 0, 1]]), dist=np.zeros(5), size=(64, 64))
            for i, (n, _, _) in enumerate(pairs)]
    s = sio.RecordingSession(camera_group=sio.CameraGroup(cameras=cams))
    for c, (_, v, _) in zip(cams, pairs):
        if v is not None:
            s.add_video(v, c)
    # Every frame of the session is grouped across its cameras, with 3D.
    with_frames = [(c, {lf.frame_idx: lf for lf in fs}) for c, (_, _, fs) in zip(cams, pairs) if fs]
    for f in sorted(with_frames[0][1]) if with_frames else []:
        igs = [sio.InstanceGroup(instance_by_camera={c: by[f].instances[i] for c, by in with_frames}, score=0.5,
                                 instance_3d=Instance3D(points=np.array([[1.0 + i + f, 2.0, 3.0], [4.0, 5.0, 6.0 + i]]), skeleton=sk))
               for i in range(2)]
        s.frame_groups[f] = sio.FrameGroup(frame_idx=f, instance_groups=igs, labeled_frame_by_camera={c: by[f] for c, by in with_frames})
    return s

def save(name, frames, videos, sessions):
    sio.Labels(labeled_frames=frames, videos=videos, skeletons=[sk], tracks=[T1, T2, T3], sessions=sessions).save(os.path.join(out, name), embed=False)

def two_cams():
    a, b = video("camA.mp4"), video("camB.mp4")
    fa, fb = frames_on(a, 100), frames_on(b, 200)
    return a, b, fa, fb, session([("camA", a, fa), ("camB", b, fb)])

# No video outside: the control.
a, b, fa, fb, s = two_cams()
save("plain.slp", fa + fb, [a, b], [s])
# The other video last: its id is past the end of a header with one video per camera.
a, b, fa, fb, s = two_cams()
x = video("outside.mp4")
save("last.slp", fa + fb + frames_on(x, 500, tracks=(T1, T3)), [a, b, x], [s])
# The other video first, and its frames first: its id is camera 0's new index.
a, b, fa, fb, s = two_cams()
x = video("outside.mp4")
save("first.slp", frames_on(x, 500, tracks=(T1, T3)) + fa + fb, [x, a, b], [s])
# One at each end, frames interleaved.
a, b, fa, fb, s = two_cams()
x, y = video("outside1.mp4"), video("outside2.mp4")
save("both_ends.slp", frames_on(x, 500) + fa + frames_on(y, 700) + fb, [x, a, b, y], [s])
# Its name contains a camera's: only a loader matching by filename could confuse them.
a, b, fa, fb, s = two_cams()
x = video("camA_extra.mp4")
save("name_like.slp", fa + fb + frames_on(x, 500), [a, b, x], [s])
# Two sessions and a video in neither. The lazy and folder loaders open only the first.
a, b, fa, fb, s1 = two_cams()
c, d = video("camC.mp4"), video("camD.mp4")
fc, fd = frames_on(c, 300), frames_on(d, 400)
s2 = session([("camC", c, fc), ("camD", d, fd)])
x = video("outside.mp4")
save("two_sessions.slp", frames_on(x, 500) + fa + fb + fc + fd, [x, a, b, c, d], [s1, s2])
# A session that maps no video, and more videos than cameras.
a, b, x = video("camA.mp4"), video("camB.mp4"), video("third.mp4")
save("no_map.slp", frames_on(a, 100) + frames_on(b, 200) + frames_on(x, 500), [a, b, x], [session([("camA", None, []), ("camB", None, [])])])
# Every frame outside: no camera has a row.
a, b, x = video("camA.mp4"), video("camB.mp4"), video("outside.mp4")
save("all_outside.slp", frames_on(x, 500), [a, b, x], [session([("camA", a, []), ("camB", b, [])])])
# Every kind of pose outside: user, trackless, a node with no position, a node placed
# but not visible, per-node scores. The same frame number as a camera's frame.
a, b, fa, fb, s = two_cams()
x = video("outside.mp4")
hidden = sio.Instance.from_numpy(np.array([[601.0, 11.0], [602.0, 12.0]]), skeleton=sk, track=T3)
hidden.points["visible"][0] = False
frame = sio.LabeledFrame(video=x, frame_idx=1, instances=[
    sio.Instance.from_numpy(np.array([[600.0, 10.0], [np.nan, np.nan]]), skeleton=sk),
    hidden,
    sio.PredictedInstance.from_numpy(np.array([[610.0, 15.0], [620.0, 25.0]]), skeleton=sk, score=0.25, point_scores=np.array([0.1, 0.2])),
])
save("kinds.slp", fa + fb + [frame], [a, b, x], [s])

# Per-camera files for the second session of section C.
for name, base in (("camC", 300), ("camD", 400)):
    v = video(name + ".mp4")
    sio.Labels(labeled_frames=frames_on(v, base), videos=[v], skeletons=[sk], tracks=[T1, T2]).save(os.path.join(out, name + "_alone.slp"), embed=False)
c, y = video("camC.mp4"), video("camC_second.mp4")
sio.Labels(labeled_frames=frames_on(c, 300) + frames_on(y, 600), videos=[c, y], skeletons=[sk], tracks=[T1, T2]).save(os.path.join(out, "camC_two_videos.slp"), embed=False)

# Per-camera files for section G: frames on two videos; frames only on the second listed.
a, x = video("camA.mp4"), video("camA_second.mp4")
sio.Labels(labeled_frames=frames_on(a, 100) + frames_on(x, 500), videos=[a, x], skeletons=[sk], tracks=[T1, T2]).save(os.path.join(out, "camA_two_videos.slp"), embed=False)
unused, b = video("camB_unused.mp4"), video("camB.mp4")
sio.Labels(labeled_frames=frames_on(b, 200), videos=[unused, b], skeletons=[sk], tracks=[T1, T2]).save(os.path.join(out, "camB_second_video.slp"), embed=False)

# A second skeleton: on the other video's poses; on a camera's.
sk2 = sio.Skeleton(nodes=["a", "b", "c"], edges=[("a", "b")], name="other")
def frames_on_sk2(v, base):
    return [sio.LabeledFrame(video=v, frame_idx=f, instances=[
        sio.PredictedInstance.from_numpy(np.array([[base + f, 10.0], [base + f + 5, 20.0], [base + f + 9, 30.0]]), skeleton=sk2, track=T1,
                                         score=0.9, point_scores=np.array([0.5, 0.6, 0.7]))]) for f in range(3)]
a, b, x = video("camA.mp4"), video("camB.mp4"), video("outside.mp4")
sio.Labels(labeled_frames=frames_on(a, 100) + frames_on(b, 200) + frames_on_sk2(x, 500), videos=[a, b, x], skeletons=[sk, sk2], tracks=[T1, T2],
           sessions=[session([("camA", a, []), ("camB", b, [])])]).save(os.path.join(out, "skeleton_outside.slp"), embed=False)
a, b = video("camA.mp4"), video("camB.mp4")
sio.Labels(labeled_frames=frames_on(a, 100) + frames_on_sk2(b, 200), videos=[a, b], skeletons=[sk, sk2], tracks=[T1, T2],
           sessions=[session([("camA", a, []), ("camB", b, [])])]).save(os.path.join(out, "skeleton_camera.slp"), embed=False)

# What an older LUC3D's lazy save left: last.slp with the other video gone from the list.
shutil.copy(os.path.join(out, "last.slp"), os.path.join(out, "damaged.slp"))
with h5py.File(os.path.join(out, "damaged.slp"), "r+") as f:
    kept = f["videos_json"][:2]
    del f["videos_json"]
    f.create_dataset("videos_json", data=kept)
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

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'poses-outside-'));
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
    folder: async (page, bytes) => {
        const dir = fs.mkdtempSync(path.join(tmp, 'folder-'));
        fs.writeFileSync(path.join(dir, 'project.slp'), bytes);
        page.once('filechooser', fc => fc.setFiles(dir));
        return page.evaluate(() => { window.__done = false; (async () => {
            await (await import('/loading/session-loader.js')).handleLoadSessionFolderSingleSlp();
        })().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); }); });
    },
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
    await LOADERS[loader](page, bytes);
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
        videos: slp.videos.map(v => v.filename), tracks: slp.tracks, skeleton: slp.skeleton.nodes,
        sessions, poses: poses.sort(), groups: groups.sort(),
    };
}

/** Save the open project, close its page; keep the bytes and the app's own read of them. */
async function saveAndRead(page) {
    await page.evaluate(saveInPage);
    const read = await page.evaluate(readInPage);
    const b64 = await page.evaluate(() => {
        const u = window.__saved;
        let s = '';
        for (let i = 0; i < u.length; i += 32768) s += String.fromCharCode.apply(null, u.subarray(i, i + 32768));
        return btoa(s);
    });
    await page.close();
    return { bytes: Buffer.from(b64, 'base64'), read };
}
const statusOf = page => page.evaluate(() => (document.getElementById('statusText') || {}).textContent || '');

let savedSeq = 0;
const pythonWork = [];   // [{ file, what, want }] checked together: one Python run reads every save
function savedForPython(bytes, what, want) {
    const file = path.join(tmp, 'saved-' + (++savedSeq) + '.slp');
    fs.writeFileSync(file, bytes);
    pythonWork.push({ file, what, want });
}

/** Both readers against one expectation: { poses, groups, sessions, videosOnly, tracks, skeleton }. */
function compare(what, got, want) {
    if (got.error) { check(false, what + ': opens', got.error); return; }
    check(eq(got.poses, want.poses), what + ': every pose on its own video', diff(got.poses || [], want.poses));
    if (want.groups) check(eq(got.groups, want.groups), what + ': groups point at the right poses', diff(got.groups || [], want.groups));
    if (want.sessions) check(eq(got.sessions, want.sessions), what + ': cameras and their videos', got.sessions);
    if (want.tracks) check(eq(got.tracks, want.tracks), what + ': track list', got.tracks);
    if (want.skeleton) check(eq(got.skeleton, want.skeleton), what + ': skeleton', got.skeleton);
}

function checkSave(what, saved, want) {
    compare(what + ', app reader', saved.read, want);
    savedForPython(saved.bytes, what + ', Python', want);
}

const FILES = ['plain', 'last', 'first', 'both_ends', 'name_like', 'two_sessions', 'no_map', 'all_outside', 'kinds'];
const firstSessionOnly = { lazy: true, folder: true, eager: false };

function expectedFor(source, name, loader) {
    const want = { poses: source.poses };
    if (name === 'no_map') return want;   // a session that maps no video: only the poses are defined
    want.groups = firstSessionOnly[loader] ? source.groups.filter(g => g.startsWith('s0|')) : source.groups;
    want.sessions = firstSessionOnly[loader] ? source.sessions.slice(0, 1) : source.sessions;
    return want;
}

try {
    uv(PY_WRITE, [tmp]);
    const sourcePaths = [...FILES, 'damaged', 'camC_alone', 'camC_two_videos', 'camD_alone', 'camA_two_videos', 'camB_second_video', 'skeleton_outside', 'skeleton_camera']
        .map(n => path.join(tmp, n + '.slp'));
    const sources = {};
    const rawSources = JSON.parse(uv(PY_READ, sourcePaths).trim().split('\n').pop());
    for (const p of sourcePaths) sources[path.basename(p, '.slp')] = rawSources[p];
    const readFile = n => fs.readFileSync(path.join(tmp, n + '.slp'));

    console.log('Sources, as Python reads them:');
    check(sources.first.videos[0] === 'outside.mp4' && sources.first.poses.some(r => r.startsWith('outside.mp4|')),
        'first.slp lists the other video first, with poses');
    check(/^IndexError/.test(sources.damaged.error || ''), 'damaged.slp is what Python cannot open', sources.damaged.error);

    console.log('\nA. Every layout, every loader, two saves');
    for (const name of FILES) {
        for (const loader of Object.keys(LOADERS)) {
            let bytes = readFile(name);
            const want = expectedFor(sources[name], name, loader);
            for (let gen = 1; gen <= 2; gen++) {
                const page = await open(loader, bytes);
                const status = await statusOf(page);
                const what = `${name}, ${loader}, save ${gen}`;
                if (gen === 1) {
                    if (name === 'plain') {
                        check(!/not shown/.test(status), what + ': nothing to say in the status', status);
                    } else if (name === 'two_sessions' && firstSessionOnly[loader]) {
                        check(/Only the first of 2 sessions/.test(status), what + ': the status says only the first session is open', status);
                    } else {
                        check(/not shown; saving keeps/.test(status), what + ': the status says what is not shown', status);
                    }
                }
                const saved = await saveAndRead(page);
                checkSave(what, saved, want);
                bytes = saved.bytes;
            }
        }
    }

    console.log('\nB. Edits on a lazy project');
    {
        // A corrected frame goes first in the file and its store row is skipped.
        const page = await open('lazy', readFile('first'));
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
        const saved = await saveAndRead(page);
        const moved = 'camA.mp4|1|U|mouse1|999.000,999.000;101.000,20.000|';
        const poses = sources.first.poses.map(r => (r.startsWith('camA.mp4|1|P|mouse1|') ? moved : r)).sort();
        const groups = sources.first.groups.map(g => g.replace('s0|1|camA=101.000', 's0|1|camA=999.000')).sort();
        checkSave('a corrected frame', saved, { poses, groups, sessions: sources.first.sessions });
    }
    {
        // Delete a track: the session's rows lose it, the outside rows keep their names.
        const page = await open('lazy', readFile('first'));
        const occupancy = await page.evaluate(async () => {
            const ops = await import('/ui/track-identity-ops.js');
            const { state } = await import('/ui/app-state.js');
            ops.deleteTrackAt(state.session, state.session.tracks.indexOf('mouse1'));
            return { nTracks: Array.from(state.session.lazyLoader.trackOccupancy.values()).map(o => o.nTracks), session: state.session.tracks.length };
        });
        check(occupancy.nTracks.every(n => n === occupancy.session), 'the timeline counts only the session\'s tracks', occupancy);
        const saved = await saveAndRead(page);
        const poses = sources.first.poses.map(r => (r.startsWith('outside.mp4|') ? r : r.replace('|P|mouse1|', '|P||'))).sort();
        // A group on the deleted track dissolves; mouse2's stay.
        checkSave('a deleted track', saved, { poses, groups: sources.first.groups.filter(g => /camA=11\d/.test(g)), sessions: sources.first.sessions });
    }
    for (const addTrack of [false, true]) {
        // The deleted track's name stays, past the session's list, for the outside rows.
        // A rename and a new track change only the session's list, and a save lists them.
        const page = await open('lazy', readFile('first'));
        await page.evaluate(async (addTrack) => {
            const ops = await import('/ui/track-identity-ops.js');
            const tri = await import('/pose/triangulation.js');
            const { state } = await import('/ui/app-state.js');
            const s = state.session;
            ops.deleteTrackAt(s, s.tracks.indexOf('mouse1'));
            s.tracks[s.tracks.indexOf('mouse2')] = 'renamed';
            if (!addTrack) return;
            s.tracks.push('added');
            await tri.ensureLazyFrameData(1);
            const group = s.instanceGroups.get(1)[0];
            const inst = group.instances.get('camA');
            inst.setPoint(0, 999, 999);
            inst.type = 'user';
            inst.trackIdx = s.tracks.indexOf('added');
            inst.modified = true;
            group.markDirty();
        }, addTrack);
        const saved = await saveAndRead(page);
        const poses = sources.first.poses.map(r => {
            if (r.startsWith('outside.mp4|')) return r;
            if (addTrack && r.startsWith('camA.mp4|1|P|mouse2|')) return 'camA.mp4|1|U|added|999.000,999.000;121.000,20.000|';
            return r.replace('|P|mouse1|', '|P||').replace('|P|mouse2|', '|P|renamed|');
        }).sort();
        const tracks = addTrack ? ['renamed', 'only_outside', 'added', 'mouse1'] : ['renamed', 'only_outside', 'mouse1'];
        checkSave(addTrack ? 'a deleted track, a rename and a new track' : 'a deleted track and a rename', saved, { poses, tracks });
    }
    {
        // A track the outside rows do not use goes, one they use stays.
        const page = await open('lazy', readFile('last'));
        await page.evaluate(async () => {
            const ops = await import('/ui/track-identity-ops.js');
            const { state } = await import('/ui/app-state.js');
            ops.deleteTrackAt(state.session, state.session.tracks.indexOf('mouse2'));
        });
        const saved = await saveAndRead(page);
        check(eq(saved.read.tracks, ['mouse1', 'only_outside']), 'an unused deleted track does not come back', saved.read.tracks);
    }

    {
        // Delete a camera's pose: the store compacts, and the outside rows after it
        // must keep their own poses.
        const page = await open('lazy', readFile('last'));
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
        const poses = sources.last.poses.filter(r => !/^cam[AB]\.mp4\|0\|P\|mouse1\|/.test(r));
        checkSave('a deleted pose', saved, { poses, groups: sources.last.groups.filter(g => !/^s0\|0\|camA=100/.test(g)), sessions: sources.last.sessions });
    }

    console.log('\nC. A lazy multi-session save');
    // File ▸ Save keeps every session loaded; the evicting writer reads the files back.
    for (const keepLoaded of [true, false]) for (const [name, camCFile, whatFiles] of [
        ['first', 'camC_alone', 'first + a second session'],
        ['all_outside', 'camC_alone', 'all_outside + a second session'],
        ['first', 'camC_two_videos', 'first + a second session whose file holds another video'],
    ]) {
        const what = whatFiles + (keepLoaded ? '' : ', evicting writer');
        const page = await open('lazy', readFile(name));
        await putBytes(page, '__camC', readFile(camCFile));
        await putBytes(page, '__camD', readFile('camD_alone'));
        const err = await page.evaluate(async (keepLoaded) => {
            const pd = await import('/pose/pose-data.js');
            const { SioLazyLoader } = await import('/loading/sio-lazy-loader.js');
            const AS = await import('/ui/app-state.js');
            const sl = await import('/import-export/save-load.js');
            const { Camera, Instance, InstanceGroup, FrameGroup, Session, Skeleton } = pd;
            const names = ['camC', 'camD'];
            const mtx = [[500, 0, 32], [0, 500, 32], [0, 0, 1]];
            const cams = names.map((n, i) => new Camera(n, mtx, [0, 0, 0, 0, 0], [0, 0.1 * i, 0], [10 * i, 0, 0], [64, 64]));
            const second = new Session(cams, AS.state.session.skeleton, ['mouse1', 'mouse2'], 'second');
            const loader = new SioLazyLoader();
            await loader.open('camC', new File([window.__camC], 'camC.slp'));
            await loader.open('camD', new File([window.__camD], 'camD.slp'));
            second.lazyLoader = loader;
            for (let f = 0; f < 3; f++) {
                second.addFrameGroup(new FrameGroup(f));
                const gs = [];
                for (let i = 0; i < 2; i++) {
                    const g = new InstanceGroup(f * 10 + i, null);
                    names.forEach((n, ci) => {
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
                window.__saved = await sl.saveAllSessionsStreaming(AS.state.sessions.slice(), { keepLoaded });
                return null;
            } catch (e) { return String(e && e.stack || e); }
        }, keepLoaded);
        check(!err, what + ': the two-pass save runs', err);
        if (err) { await page.close(); continue; }
        const read = await page.evaluate(readInPage);
        const b64 = await page.evaluate(() => { const u = window.__saved; let s = ''; for (let i = 0; i < u.length; i += 32768) s += String.fromCharCode.apply(null, u.subarray(i, i + 32768)); return btoa(s); });
        await page.close();
        const secondPoses = [...sources[camCFile].poses, ...sources.camD_alone.poses];
        const secondGroups = [];
        for (let f = 0; f < 3; f++) {
            for (let i = 0; i < 2; i++) {
                secondGroups.push(['s1', f, 'camC=' + (300 + 10 * i + f).toFixed(3), 'camD=' + (400 + 10 * i + f).toFixed(3), [1 + i + f, 2, 3].map(v => v.toFixed(3)).join(',')].join('|'));
            }
        }
        const source = sources[name];
        const want = {
            poses: [...source.poses, ...secondPoses].sort(),
            groups: [...source.groups, ...secondGroups].sort(),
            sessions: [...source.sessions, { camC: 'camC.mp4', camD: 'camD.mp4' }],
        };
        // The other session's cameras save under their own video names.
        checkSave(what, { bytes: Buffer.from(b64, 'base64'), read }, want);
    }

    console.log('\nD. Edits on an eager project');
    {
        // Remove a node: every pose loses it, the kept ones too.
        const page = await open('eager', readFile('last'));
        const tally = await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            const { countSkeletonEditImpact } = await import('/pose/skeleton-edit-impact.js');
            const t = countSkeletonEditImpact([state.session]);
            state.session.skeleton.removeNode(0);
            state.session.propagateNodeRemoved(0);
            return t.instances;
        });
        check(tally === 18, 'the skeleton dialog counts the kept poses too (12 + 6)', tally);
        const saved = await saveAndRead(page);
        const keepTail = r => { const c = r.split('|'); c[4] = c[4].split(';')[1]; c[5] = c[5] ? c[5].split(',').filter((_, i) => i !== 1).join(',') : ''; return c.join('|'); };
        checkSave('a removed node', saved, { poses: sources.last.poses.map(keepTail).sort(), skeleton: ['tail'] });
    }
    {
        // Add a node: the kept poses get an empty slot, nothing made up.
        const page = await open('eager', readFile('last'));
        await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            state.session.skeleton.addNode('ear');
            state.session.propagateNodeAdded({ hidden: true });
        });
        const saved = await saveAndRead(page);
        const outside = saved.read.poses.filter(r => r.startsWith('outside.mp4|'));
        check(outside.length === 6 && outside.every(r => r.split('|')[4].split(';').length === 3 && r.split('|')[4].endsWith(';-')),
            'a kept pose has an empty slot for the new node', outside.slice(0, 2));
        savedForPython(saved.bytes, 'an added node, Python', { poses: saved.read.poses, skeleton: ['nose', 'tail', 'ear'] });
    }
    {
        // Delete a track from the session: a kept pose keeps its name.
        const page = await open('eager', readFile('last'));
        await page.evaluate(async () => {
            const ops = await import('/ui/track-identity-ops.js');
            const { state } = await import('/ui/app-state.js');
            ops.deleteTrackAt(state.session, state.session.tracks.indexOf('mouse1'));
        });
        const saved = await saveAndRead(page);
        const poses = sources.last.poses.map(r => (r.startsWith('outside.mp4|') ? r : r.replace('|P|mouse1|', '|P||'))).sort();
        checkSave('a deleted track, eager', saved, { poses, sessions: sources.last.sessions });
    }

    console.log('\nE. File ▸ Add SLP');
    {
        const page = await open('eager', readFile('plain'));
        page.once('filechooser', fc => fc.setFiles(path.join(tmp, 'last.slp')));
        await page.evaluate(() => { window.__done = false; (async () => {
            await (await import('/import-export/slp-import.js')).handleAddSlp();
        })().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); }); });
        await settle(page, 'Add SLP');
        const status = await statusOf(page);
        check(/not shown; saving keeps/.test(status), 'the status says what is not shown', status);
        const saved = await saveAndRead(page);
        const outside = sources.last.poses.filter(r => r.startsWith('outside.mp4|'));
        check(eq(saved.read.poses.filter(r => r.startsWith('outside.mp4|')), outside), 'the other video keeps its poses', diff(saved.read.poses, outside));
        const onCamA = saved.read.poses.filter(r => /^camA(_2)?\.mp4\|/.test(r));
        check(onCamA.every(r => !/\|5\d\d\.000,/.test(r)), 'none of them were merged into a camera', onCamA.slice(0, 3));
        savedForPython(saved.bytes, 'Add SLP, Python', { poses: saved.read.poses });
    }
    {
        // A session with no video map: a video past the last camera has no camera.
        const page = await open('eager', readFile('plain'));
        page.once('filechooser', fc => fc.setFiles(path.join(tmp, 'no_map.slp')));
        await page.evaluate(() => { window.__done = false; (async () => {
            await (await import('/import-export/slp-import.js')).handleAddSlp();
        })().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); }); });
        await settle(page, 'Add SLP, no map');
        const saved = await saveAndRead(page);
        const third = sources.no_map.poses.filter(r => r.startsWith('third.mp4|'));
        check(eq(saved.read.poses.filter(r => r.startsWith('third.mp4|')), third), 'the third video stays apart', diff(saved.read.poses, third));
    }

    console.log('\nF. A file an older LUC3D damaged');
    {
        const page = await open('lazy', readFile('damaged'));
        const saved = await saveAndRead(page);
        const rename = r => r.replace(/^outside\.mp4\|/, 'unknown_video_2.mp4|');
        const want = { poses: sources.last.poses.map(rename).sort(), groups: sources.last.groups, sessions: sources.last.sessions };
        checkSave('damaged, lazy', saved, want);
    }

    console.log('\nG. A per-camera folder whose files hold other videos');
    const openFolder = async (route, camA = readFile('camA_two_videos'), camB = readFile('camB_second_video')) => {
        const page = await newAppPage();
        await putBytes(page, '__camA', camA);
        await putBytes(page, '__camB', camB);
        // The folder goes lazy on a file's size alone, and HDF5 ignores trailing bytes.
        const loading = page.evaluate(async (route) => {
            const loader = await import('/loading/session-loader.js');
            const tri = await import('/pose/triangulation.js');
            const pad = route === 'lazy' ? [new Uint8Array(tri.LAZY_SLP_THRESHOLD + (1 << 20))] : [];
            const mkFile = (bytes, rel) => {
                const f = new File([bytes].concat(pad), rel.split('/').pop());
                Object.defineProperty(f, 'webkitRelativePath', { value: rel });
                return f;
            };
            await loader.handleLoadSessionFolderPerCamera([
                mkFile(window.__camA, 'root/camA/camA.slp'), mkFile(window.__camB, 'root/camB/camB.slp'),
            ], true);
            return (document.getElementById('statusText') || {}).textContent;
        }, route);
        // These folders hold no videos, so the load waits on the Missing Video Files popup.
        await page.getByRole('button', { name: 'Continue', exact: true }).click();
        return { page, status: await loading };
    };
    for (const route of ['eager', 'lazy']) {
        const { page, status } = await openFolder(route);
        check(/3 labeled frames on 1 video in no session are not shown/.test(status), route + ': the status says what is not shown', status);
        const shown = await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            const tri = await import('/pose/triangulation.js');
            await tri.ensureLazyFrameData(1);
            const fg = state.session.frameGroups.get(1);
            const xs = cam => [...fg.getInstances(cam), ...fg.getUnlinkedInstances(cam).map(u => u.instance)].map(i => i.getX(0));
            return { camA: xs('camA'), camB: xs('camB') };
        });
        check(eq(shown, { camA: [101, 111], camB: [201, 211] }), route + ': each camera shows its own video\'s poses', shown);
        let poses = [
            ...sources.camA_two_videos.poses,
            ...sources.camB_second_video.poses,
        ].sort();
        let groups;
        if (route === 'lazy') {
            // Group frames 1 and 2, then correct camA on frame 1: its store row is
            // skipped, the other video's row on that frame number is not, and the
            // refs of frame 2 must still land.
            await page.evaluate(async () => {
                const pd = await import('/pose/pose-data.js');
                const { state } = await import('/ui/app-state.js');
                const tri = await import('/pose/triangulation.js');
                const s = state.session;
                for (const f of [1, 2]) {
                    const g = new pd.InstanceGroup(f, null);
                    for (const [cam, x] of [['camA', 100 + f], ['camB', 200 + f]]) {
                        const inst = new pd.Instance([[x, 10], [x, 20]], 0, 'predicted', 0.9);
                        inst._rawInstIndex = 0;
                        g.addInstance(cam, inst);
                    }
                    g.points3d = new Float64Array([f, 2, 3, 4, 5, 6]);
                    s.instanceGroups.set(f, [g]);
                    s.frameGroups.delete(f);
                    await tri.ensureLazyFrameData(f);
                }
                const inst = s.instanceGroups.get(1)[0].instances.get('camA');
                inst.setPoint(0, 999, 999);
                inst.type = 'user';
                inst.modified = true;
            });
            poses = poses.map(r => (r.startsWith('camA.mp4|1|P|mouse1|') ? 'camA.mp4|1|U|mouse1|999.000,999.000;101.000,20.000|' : r)).sort();
            groups = ['s0|1|camA=999.000|camB=201.000|1.000,2.000,3.000', 's0|2|camA=102.000|camB=202.000|2.000,2.000,3.000'];
        }
        const saved = await saveAndRead(page);
        checkSave('per-camera, ' + route, saved, { poses, groups, sessions: [{ camA: 'camA.mp4', camB: 'camB.mp4' }] });
        // And as a project from then on.
        const again = await saveAndRead(await open('lazy', saved.bytes));
        checkSave('per-camera, ' + route + ', saved again as a project', again, { poses, groups, sessions: [{ camA: 'camA.mp4', camB: 'camB.mp4' }] });
    }
    {
        // Delete a track: the cameras' rows lose it, the other video's keep their names.
        const { page } = await openFolder('lazy');
        await page.evaluate(async () => {
            const ops = await import('/ui/track-identity-ops.js');
            const { state } = await import('/ui/app-state.js');
            ops.deleteTrackAt(state.session, state.session.tracks.indexOf('mouse1'));
        });
        const saved = await saveAndRead(page);
        const poses = [...sources.camA_two_videos.poses, ...sources.camB_second_video.poses]
            .map(r => (r.startsWith('camA_second.mp4|') ? r : r.replace('|P|mouse1|', '|P||'))).sort();
        checkSave('per-camera, a deleted track', saved, { poses, sessions: [{ camA: 'camA.mp4', camB: 'camB.mp4' }] });
    }
    {
        // A later calibration renames a camera. Its video is still the second in its file.
        const { page } = await openFolder('lazy');
        await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            state.session.renameCameraInAllData('camB', 'renamedB');
            state.session.cameras.find(c => c.name === 'camB').name = 'renamedB';
        });
        const saved = await saveAndRead(page);
        const poses = [...sources.camA_two_videos.poses, ...sources.camB_second_video.poses].sort();
        checkSave('per-camera, a renamed camera', saved, { poses, sessions: [{ camA: 'camA.mp4', renamedB: 'camB.mp4' }] });
    }

    console.log('\nH. A second skeleton, lazy');
    for (const name of ['skeleton_outside', 'skeleton_camera']) {
        let bytes = readFile(name);
        for (let gen = 1; gen <= 2; gen++) {
            const saved = await saveAndRead(await open('lazy', bytes));
            // The app's reader reads every pose on the first skeleton, so only Python can tell.
            savedForPython(saved.bytes, `${name}, save ${gen}, Python`, { poses: sources[name].poses, skeleton: ['nose', 'tail'] });
            bytes = saved.bytes;
        }
    }

    console.log('\nI. A project mixing lazy and eager sessions, every lazy frame resident');
    const openMixed = async (bytes) => {
        const page = await open('lazy', bytes);
        await page.evaluate(async () => {
            const pd = await import('/pose/pose-data.js');
            const tri = await import('/pose/triangulation.js');
            const { state } = await import('/ui/app-state.js');
            for (const f of [0, 1, 2]) await tri.ensureLazyFrameData(f);
            const cam = new pd.Camera('camE', [[500, 0, 32], [0, 500, 32], [0, 0, 1]], [0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0], [64, 64]);
            const eager = new pd.Session([cam], state.session.skeleton, ['mouse1'], 'eager');
            const fg = new pd.FrameGroup(0);
            eager.addFrameGroup(fg);
            fg.addInstance('camE', new pd.Instance([[900, 10], [901, 20]], 0, 'user', 0));
            state.sessions.push(eager);
        });
        return page;
    };
    {
        const saved = await saveAndRead(await openMixed(readFile('first')));
        const outside = sources.first.poses.filter(r => r.startsWith('outside.mp4|'));
        check(eq(saved.read.poses.filter(r => r.startsWith('outside.mp4|')), outside), 'the lazy store\'s other video keeps its poses', diff(saved.read.poses, outside));
        savedForPython(saved.bytes, 'mixed project, Python', { poses: saved.read.poses, groups: saved.read.groups });
    }

    console.log('\nJ. Tracker scores and model identities on these poses');
    const h5wasmUrl = `http://localhost:${PORT}/lib/h5wasm/h5wasm.iife.js`;
    const labelsPage = await newAppPage();
    // A model's output: what a detection carries, on every detection. Its first x says which.
    const withLabels = async (name) => {
        await putBytes(labelsPage, '__raw', readFile(name));
        const b64 = await labelsPage.evaluate(async (h5wasmUrl) => {
            const SIO = window.SleapIO;
            const typed = await SIO.readSlpStreaming(new File([window.__raw], 'x.slp'), { openVideos: false, h5wasmUrl });
            const identities = ['female', 'male'].map(n => new SIO.Identity({ name: n }));
            typed.identities.push(...identities);
            for (const lf of typed.labeledFrames) {
                for (const inst of lf.instances) {
                    const x = inst._xy[0];
                    inst.trackingScore = x / 10000;
                    inst.identity = identities[Math.floor(x / 10) % 2];
                    inst.identityScore = x / 20000;
                    inst.identityEmbedding = { vector: [x, x + 1] };
                    inst.category = 'mouse';
                    inst.categoryScore = x / 40000;
                    inst.categoryEmbedding = { vector: [x + 2] };
                }
            }
            const u = await SIO.saveSlpToBytes(typed);
            let s = '';
            for (let i = 0; i < u.length; i += 32768) s += String.fromCharCode.apply(null, u.subarray(i, i + 32768));
            return btoa(s);
        }, h5wasmUrl);
        return Buffer.from(b64, 'base64');
    };
    const labelsIn = async (bytes) => {
        await putBytes(labelsPage, '__raw', bytes);
        return labelsPage.evaluate(async (h5wasmUrl) => {
            const labels = await window.SleapIO.readSlpStreaming(new File([window.__raw], 'x.slp'), { openVideos: false, h5wasmUrl });
            const n = v => (v == null ? '' : Number(v).toFixed(6));
            const rows = [];
            for (const lf of labels.labeledFrames) {
                for (const i of lf.instances) {
                    rows.push([lf.video.filename, lf.frameIdx, i._xy[0], n(i.trackingScore),
                        i.identity ? i.identity.name : '', n(i.identityScore), i.identityEmbedding ? i.identityEmbedding.vector.join() : '',
                        i.category || '', n(i.categoryScore), i.categoryEmbedding ? i.categoryEmbedding.vector.join() : ''].join('|'));
                }
            }
            return { rows: rows.sort(), catalog: labels.identities.map(id => id.name).sort() };
        }, h5wasmUrl);
    };
    const checkLabels = async (what, saved, want) => {
        const got = await labelsIn(saved.bytes);
        check(eq(got.rows, want.rows), what + ': every detection keeps what it carried', diff(got.rows, want.rows));
        check(want.catalog.every(n => got.catalog.includes(n)), what + ': the identity catalog holds them', got.catalog);
    };
    for (const name of ['last', 'first', 'all_outside']) {
        const labelled = await withLabels(name);
        const want = await labelsIn(labelled);
        check(want.rows.length === sources[name].poses.length && want.rows.every(r => /\|(female|male)\|/.test(r)), name + ': every detection has labels to keep');
        for (const loader of Object.keys(LOADERS)) {
            let bytes = labelled;
            for (let gen = 1; gen <= 2; gen++) {
                const saved = await saveAndRead(await open(loader, bytes));
                const what = `${name}, ${loader}, save ${gen}`;
                await checkLabels(what, saved, want);
                savedForPython(saved.bytes, what + ', with labels, Python', { poses: sources[name].poses });
                bytes = saved.bytes;
            }
        }
    }
    {
        const labelled = await withLabels('first');
        const want = await labelsIn(labelled);
        const saved = await saveAndRead(await openMixed(labelled));
        const outside = r => r.startsWith('outside.mp4|');
        const got = await labelsIn(saved.bytes);
        check(eq(got.rows.filter(outside), want.rows.filter(outside)), 'mixed project: the lazy store\'s other video keeps what it carried', diff(got.rows.filter(outside), want.rows.filter(outside)));
        check(want.catalog.every(n => got.catalog.includes(n)), 'mixed project: the identity catalog holds them', got.catalog);
    }
    for (const route of ['eager', 'lazy']) {
        // A camera file with identities takes the typed reader, even on the eager route.
        const camA = await withLabels('camA_two_videos'), camB = await withLabels('camB_second_video');
        const wantA = await labelsIn(camA), wantB = await labelsIn(camB);
        const want = { rows: [...wantA.rows, ...wantB.rows].sort(), catalog: wantA.catalog };
        const { page, status } = await openFolder(route, camA, camB);
        check(/3 labeled frames on 1 video in no session are not shown/.test(status), route + ' per-camera, with labels: the status says what is not shown', status);
        const shown = await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            const tri = await import('/pose/triangulation.js');
            await tri.ensureLazyFrameData(1);
            const fg = state.session.frameGroups.get(1);
            const xs = cam => [...fg.getInstances(cam), ...fg.getUnlinkedInstances(cam).map(u => u.instance)].map(i => i.getX(0));
            return { camA: xs('camA'), camB: xs('camB') };
        });
        check(eq(shown, { camA: [101, 111], camB: [201, 211] }), route + ' per-camera, with labels: each camera shows its own video\'s poses', shown);
        const saved = await saveAndRead(page);
        await checkLabels(route + ' per-camera', saved, want);
        savedForPython(saved.bytes, route + ' per-camera, with labels, Python', { poses: [...sources.camA_two_videos.poses, ...sources.camB_second_video.poses].sort() });
    }
    await labelsPage.close();

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
