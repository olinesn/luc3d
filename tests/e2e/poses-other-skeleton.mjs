/**
 * poses-other-skeleton.mjs — a pose on a skeleton its session does not show keeps its
 * own skeleton and its own nodes, however the file is opened and however often it is
 * saved, or the load refuses and changes nothing. The promise in the header of
 * `import-export/outside-frames.js`.
 *
 * An eager load used to read every pose by the file's first skeleton: a pose on the
 * second had its values put under the first's node names by position and its extra
 * nodes dropped, on any video, with no warning, and a save then wrote it that way.
 *
 * Real Python sleap-io writes the files and reads every save. Only Python can tell:
 * it reads each pose on the skeleton the file says it is on.
 *
 *  A. Every layout through the eager routes, two saves: File ▸ Open, the single-`.slp`
 *     folder, the raw worker (the fallback when the typed reader throws).
 *  B. A camera's frame holding poses of both skeletons, with groups on the session's.
 *  C. From the eager save to the lazy route and back: what one wrote, the other keeps.
 *  D. Edits around a kept pose: node removed / added, track deleted, camera renamed, a
 *     corrected pose beside it, every shown pose on its frame deleted.
 *  E. File ▸ Add SLP.
 *  F. A per-camera folder: the raw worker's columnar route, and the typed reader's for a
 *     file whose detections carry labels.
 *  G. A group linking a pose on another skeleton: the load refuses, nothing changes.
 *  H. A LUC3D project whose sessions are on different skeletons: each shows its own.
 *  I. A project mixing lazy and eager sessions.
 *  J. A suggestion on a video kept poses are on: one entry of the list for both, through
 *     the routes, a renamed camera, a camera gone, a video moved to another session, File ▸
 *     Add SLP and a per-camera folder.
 *
 * Needs `uv` for Python (skipped without it).
 *
 * Run: node tests/e2e/poses-other-skeleton.mjs   (spawns its own http.server; PORT= to move it)
 */
import { chromium } from 'playwright';
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8463);
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

// One line per pose: video|frame|U or P|track|skeleton|node=value;...|scores. A value is
// `x,y`, `x,y~` when placed but not visible, `-` when it has no position. A skeleton with
// no name reads as LUC3D writes it. `detail` adds what a kept pose must also keep, for the
// poses on the skeletons these fixtures keep aside (`KEPT`).
const PY_COMMON = `
import json, os, sys
import numpy as np
import sleap_io as sio

KEPT = {"other", "third", "reversed", "edged"}

def num(v):
    return "nan" if v != v else f"{float(v):.3f}"

def pose_row(video, frame_idx, inst):
    sk = inst.skeleton
    predicted = isinstance(inst, sio.PredictedInstance)
    cells = []
    for k, name in enumerate(sk.node_names):
        x, y = inst.points["xy"][k]
        cells.append(name + "=" + ("-" if x != x or y != y else f"{num(x)},{num(y)}" + ("" if inst.points["visible"][k] else "~")))
    scores = [num(inst.score)] + [num(s) for s in inst.points["score"]] if predicted else []
    return "|".join([video, str(frame_idx), "P" if predicted else "U", inst.track.name if inst.track else "", sk.name or "skeleton", ";".join(cells), ",".join(scores)])

def read(path):
    try:
        labels = sio.load_slp(path, open_videos=False)
        sio.load_slp(path)  # the default call opens the videos too
    except Exception as e:
        return {"error": type(e).__name__ + ": " + str(e)[:160]}
    poses, detail = [], []
    row_of = {}
    for lf in labels.labeled_frames:
        for i in lf.instances:
            row_of[id(i)] = pose_row(lf.video.filename, lf.frame_idx, i)
    for lf in labels.labeled_frames:
        for i in lf.instances:
            poses.append(row_of[id(i)])
            if i.skeleton.name in KEPT:
                source = i.from_predicted if isinstance(i, sio.Instance) and not isinstance(i, sio.PredictedInstance) else None
                # No tracking score: NaN in Python's files, 0 in what sleap-io.js reads.
                tracking = i.tracking_score if i.tracking_score is not None and i.tracking_score == i.tracking_score else 0.0
                detail.append("|".join([row_of[id(i)], "tracking=" + num(tracking),
                                        "complete=" + "".join("1" if c else "0" for c in i.points["complete"]),
                                        "from=" + (row_of.get(id(source), "?") if source is not None else ""),
                                        "identity=" + (i.identity.name if i.identity is not None else "")]))
    keys = [(lf.video.filename, lf.frame_idx) for lf in labels.labeled_frames]
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
                members = sorted(c.name + "=" + num(i.points["xy"][0][0]) + "/" + (i.skeleton.name or "skeleton") for c, i in ig.instance_by_camera.items())
                p3 = "" if ig.instance_3d is None else ",".join(num(v) for v in np.asarray(ig.instance_3d.points)[0][:3])
                groups.append("|".join(["s" + str(si), str(f)] + members + [p3]))
    return {
        "error": None, "videos": [v.filename for v in labels.videos], "tracks": sorted(t.name for t in labels.tracks),
        "skeletons": [[s.name or "skeleton", s.node_names, sorted(list(e) for e in s.edge_names), sorted(sorted(p) for p in s.symmetry_names)] for s in labels.skeletons],
        "identities": [i.name for i in labels.identities],
        "sessions": sessions, "poses": sorted(poses), "detail": sorted(detail), "groups": sorted(groups),
        "repeatedFrames": len(keys) - len(set(keys)),
        "suggestions": [s.video.filename + "|" + str(s.frame_idx) for s in labels.suggestions],
    }
`;

const PY_READ = PY_COMMON + `
print(json.dumps({p: read(p) for p in sys.argv[1:]}))
`;

const PY_WRITE = PY_COMMON + `
from sleap_io.model.camera import Instance3D

out = sys.argv[1]
sk = sio.Skeleton(nodes=["nose", "tail"], edges=[("nose", "tail")])
sk2 = sio.Skeleton(nodes=["a", "b", "c"], edges=[("a", "b")], name="other")
T1, T2, T3 = sio.Track(name="mouse1"), sio.Track(name="mouse2"), sio.Track(name="only_kept")

def video(name):
    v = sio.Video(filename=name, open_backend=False)
    v.backend_metadata["shape"] = (400, 64, 64, 1)
    return v

def on_sk(base, f, i=0, t=T1, skeleton=sk):
    return sio.PredictedInstance.from_numpy(np.array([[base + 10.0 * i + f, 10.0], [base + 20.0 * i + f, 20.0]]), skeleton=skeleton, track=t,
                                            score=0.9, point_scores=np.array([0.5 + 0.1 * i, 0.8]))

def frames_on(v, base, n=3, tracks=(T1, T2), skeleton=sk):
    return [sio.LabeledFrame(video=v, frame_idx=f, instances=[on_sk(base, f, i, t, skeleton) for i, t in enumerate(tracks)]) for f in range(n)]

def on_sk2(base, f, skeleton=sk2, t=T1):
    n = len(skeleton.node_names)
    return sio.PredictedInstance.from_numpy(np.array([[base + f + 5.0 * k, 10.0 * (k + 1)] for k in range(n)]), skeleton=skeleton, track=t,
                                            score=0.9, point_scores=np.array([0.5 + 0.1 * k for k in range(n)]))

def frames_on_sk2(v, base, skeleton=sk2):
    return [sio.LabeledFrame(video=v, frame_idx=f, instances=[on_sk2(base, f, skeleton)]) for f in range(3)]

def session(pairs, groups=None):
    cams = [sio.Camera(name=n, rvec=np.array([0.0, 0.1 * i, 0.0]), tvec=np.array([10.0 * i, 0.0, 0.0]),
                       matrix=np.array([[500.0, 0, 32], [0, 500.0, 32], [0, 0, 1]]), dist=np.zeros(5), size=(64, 64))
            for i, (n, _) in enumerate(pairs)]
    s = sio.RecordingSession(camera_group=sio.CameraGroup(cameras=cams))
    for c, (_, v) in zip(cams, pairs):
        if v is not None:
            s.add_video(v, c)
    # groups: one per camera, a list of (labeled frame, the instances to group).
    if groups:
        for f in range(len(groups[0])):
            by = {c: g[f] for c, g in zip(cams, groups)}
            igs = [sio.InstanceGroup(instance_by_camera={c: by[c][1][i] for c in cams}, score=0.5,
                                     instance_3d=Instance3D(points=np.array([[1.0 + i + f, 2.0, 3.0], [4.0, 5.0, 6.0 + i]]), skeleton=sk))
                   for i in range(len(by[cams[0]][1]))]
            s.frame_groups[f] = sio.FrameGroup(frame_idx=f, instance_groups=igs, labeled_frame_by_camera={c: by[c][0] for c in cams})
    return s

def save(name, frames, videos, skeletons, sessions=(), tracks=(T1, T2), **kw):
    sio.Labels(labeled_frames=frames, videos=videos, skeletons=list(skeletons), tracks=list(tracks), sessions=list(sessions), **kw).save(os.path.join(out, name + ".slp"), embed=False)

# The two files the task names: every pose of a video on the second skeleton.
a, b, x = video("camA.mp4"), video("camB.mp4"), video("outside.mp4")
save("skeleton_outside", frames_on(a, 100) + frames_on(b, 200) + frames_on_sk2(x, 500), [a, b, x], [sk, sk2], [session([("camA", a), ("camB", b)])])
a, b = video("camA.mp4"), video("camB.mp4")
save("skeleton_camera", frames_on(a, 100) + frames_on_sk2(b, 200), [a, b], [sk, sk2], [session([("camA", a), ("camB", b)])])

# One frame holding poses of both, each way round, with groups on the session's.
a, b = video("camA.mp4"), video("camB.mp4")
fa, fb, ga, gb = [], [], [], []
for f in range(3):
    mine_a = [on_sk(100, f, i, t) for i, t in enumerate((T1, T2))]
    mine_b = [on_sk(200, f, i, t) for i, t in enumerate((T1, T2))]
    la = sio.LabeledFrame(video=a, frame_idx=f, instances=[on_sk2(600, f)] + mine_a)  # the other skeleton first
    lb = sio.LabeledFrame(video=b, frame_idx=f, instances=mine_b + [on_sk2(700, f)])
    fa.append(la); fb.append(lb); ga.append((la, mine_a)); gb.append((lb, mine_b))
save("skeleton_mixed", fa + fb, [a, b], [sk, sk2], [session([("camA", a), ("camB", b)], [ga, gb])])

# Every kind of pose kept: a user instance with a node with no position, a node placed but
# not visible, a flag LUC3D does not read; a tracking score, a from_predicted link, an
# identity, a track only kept poses use, a symmetry on the skeleton; trackless too.
sk_sym = sio.Skeleton(nodes=["a", "b", "c"], edges=[("a", "b")], symmetries=[("a", "c")], name="other")
a, b = video("camA.mp4"), video("camB.mp4")
alpha = sio.Identity(name="alpha")
predicted = on_sk2(800, 1, sk_sym, T3)
predicted.tracking_score = 0.75
predicted.identity = alpha
corrected = sio.Instance.from_numpy(np.array([[810.0, 10.0], [np.nan, np.nan], [812.0, 12.0]]), skeleton=sk_sym)
corrected.points["visible"][0] = False
corrected.points["complete"][0] = True
corrected.from_predicted = predicted
corrected.tracking_score = 0.25
fb = frames_on(b, 200)
fb[1].instances.extend([predicted, corrected])
save("skeleton_kinds", frames_on(a, 100) + fb, [a, b], [sk, sk_sym], [session([("camA", a), ("camB", b)])], tracks=(T1, T2, T3), identities=[alpha])

# The same nodes in another order: read by position, nose and tail would swap.
sk_rev = sio.Skeleton(nodes=["tail", "nose"], edges=[("tail", "nose")], name="reversed")
a, b = video("camA.mp4"), video("camB.mp4")
save("skeleton_reversed", frames_on(a, 100) + frames_on(b, 200, skeleton=sk_rev), [a, b], [sk, sk_rev], [session([("camA", a), ("camB", b)])])

# The same nodes in the same order but other edges: a skeleton of its own.
sk_edged = sio.Skeleton(nodes=["nose", "tail"], edges=[("tail", "nose")], name="edged")
a, b = video("camA.mp4"), video("camB.mp4")
save("skeleton_edged", frames_on(a, 100) + frames_on(b, 200, skeleton=sk_edged), [a, b], [sk, sk_edged], [session([("camA", a), ("camB", b)])])

# A duplicate, differing only by name (legacy files carry them): its poses line up, so they show.
sk_dup = sio.Skeleton(nodes=["nose", "tail"], edges=[("nose", "tail")], name="twin")
a, b = video("camA.mp4"), video("camB.mp4")
save("skeleton_twin", frames_on(a, 100) + frames_on(b, 200, skeleton=sk_dup), [a, b], [sk, sk_dup], [session([("camA", a), ("camB", b)])])

# An unused skeleton at the head of the list: the poses are all on the second.
sk_old = sio.Skeleton(nodes=["x", "y", "z", "w"], name="stale")
sk_real = sio.Skeleton(nodes=["nose", "tail", "ear"], edges=[("nose", "tail")], name="real")
a, b = video("camA.mp4"), video("camB.mp4")
save("skeleton_stale", frames_on_sk2(a, 100, sk_real) + frames_on_sk2(b, 200, sk_real), [a, b], [sk_old, sk_real], [session([("camA", a), ("camB", b)])])

# No recording session: every video is a camera.
a, b = video("camA.mp4"), video("camB.mp4")
save("skeleton_no_session", frames_on(a, 100) + frames_on_sk2(b, 200), [a, b], [sk, sk2])

# Three skeletons, two of them kept, on one frame.
sk3 = sio.Skeleton(nodes=["p", "q"], name="third")
a, b = video("camA.mp4"), video("camB.mp4")
fb = frames_on(b, 200)
fb[0].instances.extend([on_sk2(900, 0), on_sk2(950, 0, sk3)])
save("skeleton_three", frames_on(a, 100) + fb, [a, b], [sk, sk2, sk3], [session([("camA", a), ("camB", b)])])

# Two sessions: a kept pose on the second one's camera stays on that camera's video.
a, b, c, d = video("camA.mp4"), video("camB.mp4"), video("camC.mp4"), video("camD.mp4")
save("skeleton_two_sessions", frames_on(a, 100) + frames_on(b, 200) + frames_on(c, 300) + frames_on_sk2(d, 400), [a, b, c, d], [sk, sk2],
     [session([("camA", a), ("camB", b)]), session([("camC", c), ("camD", d)])])

# The control: one skeleton, nothing to keep.
a, b = video("camA.mp4"), video("camB.mp4")
save("one_skeleton", frames_on(a, 100) + frames_on(b, 200), [a, b], [sk], [session([("camA", a), ("camB", b)])])

# A group linking a pose on another skeleton: LUC3D cannot hold it.
a, b = video("camA.mp4"), video("camB.mp4")
mine = [[on_sk(100, f, 0, T1)] for f in range(3)]
theirs = [[on_sk2(200, f)] for f in range(3)]
fa = [sio.LabeledFrame(video=a, frame_idx=f, instances=mine[f]) for f in range(3)]
fb = [sio.LabeledFrame(video=b, frame_idx=f, instances=theirs[f]) for f in range(3)]
save("skeleton_group", fa + fb, [a, b], [sk, sk2], [session([("camA", a), ("camB", b)], [[(fa[f], mine[f]) for f in range(3)], [(fb[f], theirs[f]) for f in range(3)]])])

# skeleton_mixed with its groups the way older files hold them, as positions in a frame's
# instance list (the raw worker reads only these): what they point at moves once the
# other skeleton's poses are set aside.
import h5py, shutil
shutil.copy(os.path.join(out, "skeleton_mixed.slp"), os.path.join(out, "skeleton_legacy_groups.slp"))
with h5py.File(os.path.join(out, "skeleton_legacy_groups.slp"), "r+") as f:
    if "session_data" in f:
        del f["session_data"]
    dicts = []
    for fi in range(3):
        dicts.append({"frame_idx": fi, "instance_groups": [
            {"camcorder_to_lf_and_inst_idx_map": {"0": [fi, 1 + i], "1": [3 + fi, i]}} for i in range(2)]})
    sessions = [json.loads(v) for v in f["sessions_json"][()]]
    sessions[0]["frame_group_dicts"] = dicts
    del f["sessions_json"]
    f.create_dataset("sessions_json", data=[json.dumps(x) for x in sessions])

# Per-camera files for section F: both skeletons on the camera's video, and another video.
a, y = video("camA.mp4"), video("camA_second.mp4")
fa = frames_on(a, 100)
fa[0].instances.append(on_sk2(600, 0))
sio.Labels(labeled_frames=fa + frames_on_sk2(y, 500), videos=[a, y], skeletons=[sk, sk2], tracks=[T1, T2]).save(os.path.join(out, "camA_two_skeletons.slp"), embed=False)
# The same with an identity on a detection: only the typed reader has those.
a, y = video("camA.mp4"), video("camA_second.mp4")
beta = sio.Identity(name="beta")
fa = frames_on(a, 100)
labeled = on_sk2(600, 0)
labeled.identity = beta
fa[0].instances.append(labeled)
sio.Labels(labeled_frames=fa + frames_on_sk2(y, 500), videos=[a, y], skeletons=[sk, sk2], tracks=[T1, T2], identities=[beta]).save(os.path.join(out, "camA_labeled.slp"), embed=False)
b = video("camB.mp4")
sio.Labels(labeled_frames=frames_on(b, 200), videos=[b], skeletons=[sk], tracks=[T1, T2]).save(os.path.join(out, "camB_alone.slp"), embed=False)

# Suggestions on the videos kept poses are on, for section J. All of camA's poses are kept,
# because camA is the camera J can lose: on camB alone a save still calls the session
# calibrated, while camA alone, at the origin, would not.
from sleap_io.model.suggestions import SuggestionFrame
a, b, x = video("camA.mp4"), video("camB.mp4"), video("outside.mp4")
save("skeleton_suggested", frames_on_sk2(a, 100) + frames_on(b, 200) + frames_on_sk2(x, 500), [a, b, x], [sk, sk2], [session([("camA", a), ("camB", b)])],
     suggestions=[SuggestionFrame(video=x, frame_idx=40), SuggestionFrame(video=b, frame_idx=41), SuggestionFrame(video=a, frame_idx=42)])
# Two sessions, so a camera holding a kept pose and a suggestion can move between them.
a, b, c, d = video("camA.mp4"), video("camB.mp4"), video("camC.mp4"), video("camD.mp4")
fb = frames_on(b, 200)
fb[0].instances.append(on_sk2(600, 0))
save("skeleton_suggested_two", frames_on(a, 100) + fb + frames_on(c, 300) + frames_on(d, 400), [a, b, c, d], [sk, sk2],
     [session([("camA", a), ("camB", b)]), session([("camC", c), ("camD", d)])], suggestions=[SuggestionFrame(video=b, frame_idx=41)])
# A camera's own file: both skeletons on its video, another video with only the second.
a, y = video("camA.mp4"), video("camA_second.mp4")
fa = frames_on(a, 100)
fa[0].instances.append(on_sk2(600, 0))
sio.Labels(labeled_frames=fa + frames_on_sk2(y, 500), videos=[a, y], skeletons=[sk, sk2], tracks=[T1, T2],
           suggestions=[SuggestionFrame(video=y, frame_idx=43), SuggestionFrame(video=a, frame_idx=44)]).save(os.path.join(out, "camA_suggested.slp"), embed=False)
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

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'poses-other-skeleton-'));
const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
for (let i = 0; i < 60; i++) {
    try { if ((await fetch(`http://localhost:${PORT}/index.html`)).ok) break; } catch { /* not yet */ }
    await new Promise(r => setTimeout(r, 250));
}
const browser = await chromium.launch();
// `page.evaluate` has no timeout, so a step waiting on a click nobody makes would sit
// silent for good. Closing the browser fails that step.
const stallGuard = setTimeout(() => {
    console.log('\nStuck: no step finished for 5 minutes');
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
    eager: page => page.evaluate(() => { window.__done = false; (async () => {
        await (await import('/import-export/slp-import.js')).handleLoadSlpFile(new File([window.__file], 'project.slp'));
    })().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); }); }),
    lazy: page => page.evaluate(() => { window.__done = false; (async () => {
        await (await import('/loading/session-loader.js')).handleLoadProjectSlpLazy(new File([window.__file], 'project.slp'));
    })().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); }); }),
    // The typed reader throws (a file it cannot read), so the load goes to the raw worker.
    raw: page => page.evaluate(() => { window.__done = false; (async () => {
        window.SleapIO.readSlpStreaming = async () => { throw new Error('forced to the raw worker'); };
        await (await import('/import-export/slp-import.js')).handleLoadSlpFile(new File([window.__file], 'project.slp'));
    })().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); }); }),
    // Stands in for the folder picker: this test is about what the load does with the file.
    folder: page => page.evaluate(() => { window.__done = false; (async () => {
            const file = new File([window.__file], 'project.slp');
            Object.defineProperty(file, 'webkitRelativePath', { value: 'root/project.slp' });
            const click = HTMLInputElement.prototype.click;
            HTMLInputElement.prototype.click = function () {
                if (!this.webkitdirectory) return click.call(this);
                HTMLInputElement.prototype.click = click;
                Object.defineProperty(this, 'files', { value: [file] });
                this.dispatchEvent(new Event('change'));
            };
            await (await import('/loading/session-loader.js')).handleLoadSessionFolderSingleSlp();
    })().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); }); }),
};

/** Answer the video prompts a load puts up, until `window.__done` is set. Patient: a loaded machine is slow, and the stall guard catches a real hang. */
async function settle(page, what) {
    let done = false;
    for (let i = 0; i < 2000 && !done; i++) {
        const s = await page.evaluate(() => {
            const b = [...document.querySelectorAll('button')].find(x => x.offsetParent && /^(Skip|Later|Cancel|Not now)$/i.test((x.textContent || '').trim()));
            if (b) { b.click(); return { done: false }; }
            return { done: window.__done, prompt: !!document.getElementById('videoSelectOverlay') };
        });
        if (s.prompt) await page.keyboard.press('Escape');
        done = s.done;
        if (!done) await page.waitForTimeout(100);
    }
    if (done !== true) throw new Error(what + ': did not finish: ' + done + ', status: ' + await statusOf(page));
}

async function open(loader, bytes, page) {
    page = page || await newAppPage();
    await putBytes(page, '__file', bytes);
    await LOADERS[loader](page, bytes);
    await settle(page, loader + ' load');
    stallGuard.refresh();
    return page;
}

/** File ▸ Save, the real route; the file's bytes. Closes the page. */
async function save(page) {
    const b64 = await page.evaluate(async () => {
        const sl = await import('/import-export/save-load.js');
        const orig = URL.createObjectURL;
        let blob = null;
        URL.createObjectURL = b => { blob = b; return orig.call(URL, b); };
        try { await sl.saveProjectSlp(); } finally { URL.createObjectURL = orig; }
        if (!blob) throw new Error('saveProjectSlp produced no file: ' + (document.getElementById('statusText') || {}).textContent);
        const u = new Uint8Array(await blob.arrayBuffer());
        let s = '';
        for (let i = 0; i < u.length; i += 32768) s += String.fromCharCode.apply(null, u.subarray(i, i + 32768));
        return btoa(s);
    });
    await page.close();
    stallGuard.refresh();
    return Buffer.from(b64, 'base64');
}

const statusOf = page => page.evaluate(() => (document.getElementById('statusText') || {}).textContent || '');

let savedSeq = 0;
const pythonWork = [];   // [{ file, what, want }] checked together: one Python run reads every save
function savedForPython(bytes, what, want) {
    const file = path.join(tmp, 'saved-' + (++savedSeq) + '.slp');
    fs.writeFileSync(file, bytes);
    pythonWork.push({ file, what, want });
}

/**
 * Python's read of a save against what it must hold; `sessions`, `groups`, `skeletons`,
 * `videos`, `tracks` when given. `onVideo` limits the poses compared to that video's.
 */
function compare(what, got, want) {
    if (got.error) { check(false, what + ': opens', got.error); return; }
    const gotPoses = want.onVideo ? got.poses.filter(r => r.startsWith(want.onVideo + '|')) : got.poses;
    check(eq(gotPoses, want.poses), what + ': every pose on its own skeleton, nodes and video', diff(gotPoses, want.poses));
    check(got.repeatedFrames === 0, what + ': one row per video and frame', got.repeatedFrames);
    if (want.detail) check(eq(got.detail, want.detail), what + ': what a kept pose also carries', diff(got.detail || [], want.detail));
    if (want.groups) check(eq(got.groups, want.groups), what + ': groups point at the right poses', diff(got.groups || [], want.groups));
    if (want.sessions) check(eq(got.sessions, want.sessions), what + ': cameras and their videos', got.sessions);
    if (want.videos) check(eq(got.videos, want.videos), what + ': video list', got.videos);
    if (want.tracks) check(eq(got.tracks, want.tracks), what + ': track list', got.tracks);
    if (want.skeletons) check(eq(got.skeletons, want.skeletons), what + ': skeleton list', got.skeletons);
    if (want.identities) check(eq(got.identities, want.identities), what + ': identity list', got.identities);
    if (want.suggestions) check(eq(got.suggestions, want.suggestions), what + ': every suggestion on its own video and frame', got.suggestions);
}

/** The app's own view of the open project, session by session. */
const shownInApp = page => page.evaluate(async () => {
    const { state } = await import('/ui/app-state.js');
    return state.sessions.map(s => {
        const cams = {};
        for (const [f, fg] of s.frameGroups) {
            for (const cam of new Set([...fg.instances.keys(), ...fg.unlinkedInstances.keys()])) {
                for (const inst of [...fg.getInstances(cam), ...fg.getUnlinkedInstances(cam).map(u => u.instance)]) {
                    (cams[cam] = cams[cam] || []).push([f, inst.numNodes, inst.getX(0)]);
                }
            }
        }
        for (const cam in cams) cams[cam].sort((a, b) => a[0] - b[0] || a[2] - b[2]);
        return { nodes: s.skeleton.nodes, cams };
    });
});

// What a save lists, from the source's list: the session's first, then the kept ones.
const FIXTURES = {
    skeleton_outside: { note: /3 labeled frames on 1 video in no session are not shown; saving keeps them/, videos: ['camA.mp4', 'camB.mp4', 'outside.mp4'] },
    skeleton_camera: { note: /3 poses on a camera's video are on another skeleton \("other"\) and are not shown; saving keeps them on that skeleton/, videos: ['camA.mp4', 'camB.mp4'] },
    skeleton_kinds: { note: /2 poses on a camera's video are on another skeleton \("other"\)/, videos: ['camA.mp4', 'camB.mp4'] },
    skeleton_reversed: { note: /6 poses on a camera's video are on another skeleton \("reversed"\)/, videos: ['camA.mp4', 'camB.mp4'] },
    skeleton_edged: { note: /6 poses on a camera's video are on another skeleton \("edged"\)/, videos: ['camA.mp4', 'camB.mp4'] },
    skeleton_three: { note: /2 poses on a camera's video are on another skeleton \("other", "third"\)|2 poses on a camera's video are on another skeleton \("third", "other"\)/, videos: ['camA.mp4', 'camB.mp4'] },
    skeleton_no_session: { note: /3 poses on a camera's video are on another skeleton \("other"\)/, videos: ['camA.mp4', 'camB.mp4'] },
};

try {
    uv(PY_WRITE, [tmp]);
    const NAMES = [...Object.keys(FIXTURES), 'skeleton_mixed', 'skeleton_twin', 'skeleton_stale', 'skeleton_two_sessions', 'one_skeleton', 'skeleton_group', 'camA_two_skeletons', 'camA_labeled', 'camB_alone', 'skeleton_suggested', 'skeleton_suggested_two', 'camA_suggested'];
    const sourcePaths = NAMES.map(n => path.join(tmp, n + '.slp'));
    const sources = {};
    const rawSources = JSON.parse(uv(PY_READ, sourcePaths).trim().split('\n').pop());
    for (const p of sourcePaths) sources[path.basename(p, '.slp')] = rawSources[p];
    const readFile = n => fs.readFileSync(path.join(tmp, n + '.slp'));

    console.log('Sources, as Python reads them:');
    check(sources.skeleton_camera.poses.some(r => r.includes('|other|a=200.000,10.000;b=205.000,20.000;c=210.000,30.000|')), 'skeleton_camera.slp holds poses on the second skeleton');
    check(sources.skeleton_kinds.detail.some(r => r.includes('tracking=0.750') && r.includes('identity=alpha')), 'skeleton_kinds.slp holds a tracking score and an identity', sources.skeleton_kinds.detail);
    check(sources.skeleton_kinds.detail.some(r => r.includes('complete=') && /\|from=camB\.mp4/.test(r)), 'skeleton_kinds.slp holds a from_predicted link', sources.skeleton_kinds.detail);
    check(sources.skeleton_group.groups.length === 3 && sources.skeleton_group.groups[0].includes('/other'), 'skeleton_group.slp groups a pose on the second skeleton', sources.skeleton_group.groups);

    console.log('\nA. Every layout through the eager routes, two saves');
    for (const name of Object.keys(FIXTURES)) {
        for (const loader of ['eager', 'folder', 'raw']) {
            // A file with no session has no calibration for the folder route to read.
            if (name === 'skeleton_no_session' && loader === 'folder') continue;
            // The raw worker does not stand in for a file with labels on its detections: below.
            if (name === 'skeleton_kinds' && loader === 'raw') continue;
            let bytes = readFile(name);
            for (let gen = 1; gen <= 2; gen++) {
                const page = await open(loader, bytes);
                const status = await statusOf(page);
                const what = `${name}, ${loader}, save ${gen}`;
                check(FIXTURES[name].note.test(status), what + ': the status says what is not shown', status);
                if (gen === 1) {
                    const shown = await shownInApp(page);
                    const shownNodes = new Set(Object.values(shown[0].cams).flat().map(r => r[1]));
                    check(shownNodes.size === 1 && shownNodes.has(2), what + ': every pose shown has the session\'s nodes', [...shownNodes]);
                }
                bytes = await save(page);
                const source = sources[name];
                savedForPython(bytes, what, {
                    poses: source.poses, detail: source.detail, videos: FIXTURES[name].videos,
                    sessions: name === 'skeleton_no_session' ? undefined : source.sessions,
                    tracks: source.tracks, skeletons: source.skeletons, identities: source.identities,
                });
            }
        }
    }
    {
        // It would lose them, so it does not open the file: nothing changes.
        const page = await open('raw', readFile('skeleton_kinds'));
        const status = await statusOf(page);
        check(/forced to the raw worker/.test(status), 'skeleton_kinds, raw: not opened, as the raw worker would lose its labels', status);
        check(await page.evaluate(() => window.__lucid.state.sessions.length) === 0, 'skeleton_kinds, raw: nothing opened');
        await page.close();
    }
    for (const loader of ['eager', 'folder', 'raw']) {
        // The control: nothing to say and nothing to keep.
        const page = await open(loader, readFile('one_skeleton'));
        const status = await statusOf(page);
        check(!/not shown/.test(status), `one_skeleton, ${loader}: nothing to say in the status`, status);
        savedForPython(await save(page), `one_skeleton, ${loader}`, { poses: sources.one_skeleton.poses, skeletons: sources.one_skeleton.skeletons });
    }
    {
        // A duplicate shows, on the session's skeleton: its node names do not change.
        const onSession = sources.skeleton_twin.poses.map(r => r.replace('|twin|', '|skeleton|')).sort();
        for (const loader of ['eager', 'folder', 'raw']) {
            const page = await open(loader, readFile('skeleton_twin'));
            const status = await statusOf(page);
            check(!/not shown/.test(status), `skeleton_twin, ${loader}: nothing hidden`, status);
            const shown = await shownInApp(page);
            check(shown[0].cams.camB && shown[0].cams.camB.length === 6, `skeleton_twin, ${loader}: the duplicate's poses show`, shown[0].cams.camB);
            savedForPython(await save(page), `skeleton_twin, ${loader}`, { poses: onSession, skeletons: [sources.skeleton_twin.skeletons[0]] });
        }
    }
    {
        // An unused skeleton at the head of the list does not hide the poses.
        for (const loader of ['eager', 'folder', 'raw']) {
            let bytes = readFile('skeleton_stale');
            for (let gen = 1; gen <= 2; gen++) {
                const page = await open(loader, bytes);
                const status = await statusOf(page);
                check(!/not shown/.test(status), `skeleton_stale, ${loader}, save ${gen}: nothing hidden`, status);
                const shown = await shownInApp(page);
                check(eq(shown[0].nodes, ['nose', 'tail', 'ear']), `skeleton_stale, ${loader}, save ${gen}: the session shows the skeleton the poses are on`, shown[0].nodes);
                check(shown[0].cams.camB && shown[0].cams.camB.every(r => r[1] === 3), `skeleton_stale, ${loader}, save ${gen}: every node of them`, shown[0].cams.camB);
                bytes = await save(page);
                savedForPython(bytes, `skeleton_stale, ${loader}, save ${gen}`, { poses: sources.skeleton_stale.poses, skeletons: [sources.skeleton_stale.skeletons[1]] });
            }
        }
    }
    {
        // Two sessions: each pose on the second session's camera stays on its video.
        for (const loader of ['eager', 'raw']) {
            let bytes = readFile('skeleton_two_sessions');
            for (let gen = 1; gen <= 2; gen++) {
                const page = await open(loader, bytes);
                const status = await statusOf(page);
                check(/3 poses on a camera's video are on another skeleton/.test(status), `two sessions, ${loader}, save ${gen}: the status counts every session`, status);
                const kept = await page.evaluate(async () => {
                    const { state } = await import('/ui/app-state.js');
                    return state.sessions.map(s => s.outsideFrames ? s.outsideFrames.videos.map(v => v.camera) : null);
                });
                check(eq(kept, [null, ['camD']]), `two sessions, ${loader}, save ${gen}: held by the session reading that video`, kept);
                bytes = await save(page);
                savedForPython(bytes, `two sessions, ${loader}, save ${gen}`, {
                    poses: sources.skeleton_two_sessions.poses, videos: ['camA.mp4', 'camB.mp4', 'camC.mp4', 'camD.mp4'],
                    sessions: sources.skeleton_two_sessions.sessions, skeletons: sources.skeleton_two_sessions.skeletons,
                });
            }
        }
        // A folder opens the first session: the second's cameras are outside it.
        let bytes = readFile('skeleton_two_sessions');
        for (let gen = 1; gen <= 2; gen++) {
            bytes = await save(await open('folder', bytes));
            savedForPython(bytes, `two sessions, folder, save ${gen}`, { poses: sources.skeleton_two_sessions.poses, skeletons: sources.skeleton_two_sessions.skeletons });
        }
    }

    {
        // Save twice with the project still open: each save starts from what is held.
        const page = await open('eager', readFile('skeleton_mixed'));
        const sizes = await page.evaluate(async () => {
            const sl = await import('/import-export/save-load.js');
            const { state } = await import('/ui/app-state.js');
            const orig = URL.createObjectURL;
            let blob = null;
            URL.createObjectURL = b => { blob = b; return orig.call(URL, b); };
            const held = () => (state.session.outsideFrames ? state.session.outsideFrames.otherSkeletonFrames : []).map(f => f.instances.length);
            const before = held();
            try { await sl.saveProjectSlp(); } finally { URL.createObjectURL = orig; }
            return { before, after: held() };
        });
        check(sizes.before.length === 6 && eq(sizes.before, sizes.after), 'a save does not change what is held', sizes);
        savedForPython(await save(page), 'mixed, second save of one open project', {
            poses: sources.skeleton_mixed.poses, detail: sources.skeleton_mixed.detail, groups: sources.skeleton_mixed.groups,
        });
    }

    console.log('\nB. Poses of both skeletons on one camera frame, groups on the session\'s');
    for (const loader of ['eager', 'folder']) {
        let bytes = readFile('skeleton_mixed');
        for (let gen = 1; gen <= 2; gen++) {
            const page = await open(loader, bytes);
            const what = `mixed, ${loader}, save ${gen}`;
            const status = await statusOf(page);
            check(/6 poses on a camera's video are on another skeleton/.test(status), what + ': the status says what is not shown', status);
            const shown = await shownInApp(page);
            check(shown[0].cams.camA.length === 6 && shown[0].cams.camA.every(r => r[1] === 2 && r[2] < 600), what + ': a camera shows only the session\'s poses', shown[0].cams.camA);
            bytes = await save(page);
            savedForPython(bytes, what, {
                poses: sources.skeleton_mixed.poses, detail: sources.skeleton_mixed.detail, groups: sources.skeleton_mixed.groups,
                sessions: sources.skeleton_mixed.sessions, videos: ['camA.mp4', 'camB.mp4'], skeletons: sources.skeleton_mixed.skeletons,
            });
        }
    }

    console.log('\nC. From one route to the other');
    // Not skeleton_kinds: its hand-labeled pose on the second skeleton makes a lazy save
    // rewrite that resident frame by position, whoever wrote the file (not fixed yet).
    for (const name of ['skeleton_outside', 'skeleton_camera', 'skeleton_mixed', 'skeleton_three']) {
        const source = sources[name];
        const lazyOfEager = await save(await open('lazy', await save(await open('eager', readFile(name)))));
        savedForPython(lazyOfEager, `${name}: eager save, then lazy save`, { poses: source.poses, groups: source.groups, skeletons: source.skeletons });
        const eagerOfLazy = await save(await open('eager', await save(await open('lazy', readFile(name)))));
        savedForPython(eagerOfLazy, `${name}: lazy save, then eager save`, { poses: source.poses, groups: source.groups, skeletons: source.skeletons });
    }

    console.log('\nD. Edits around a kept pose');
    {
        // Remove a node of the session's skeleton: the kept poses are on another.
        const page = await open('eager', readFile('skeleton_camera'));
        const tally = await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            const { countSkeletonEditImpact } = await import('/pose/skeleton-edit-impact.js');
            const t = countSkeletonEditImpact([state.session]);
            state.session.skeleton.removeNode(0);
            state.session.propagateNodeRemoved(0);
            return t.instances;
        });
        check(tally === 6, 'the skeleton dialog counts only the session\'s poses', tally);
        const bytes = await save(page);
        const poses = sources.skeleton_camera.poses.map(r => {
            if (r.includes('|other|')) return r;
            const c = r.split('|');
            c[5] = c[5].split(';')[1];
            c[6] = c[6].split(',').filter((_, i) => i !== 1).join(',');
            return c.join('|');
        }).sort();
        savedForPython(bytes, 'a removed node', { poses, detail: sources.skeleton_camera.detail });
    }
    {
        // Add a node: the kept poses keep theirs, nothing made up.
        const page = await open('eager', readFile('skeleton_camera'));
        await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            state.session.skeleton.addNode('ear');
            state.session.propagateNodeAdded({ hidden: true });
        });
        const bytes = await save(page);
        const kept = sources.skeleton_camera.poses.filter(r => r.includes('|other|'));
        const onCamA = sources.skeleton_camera.poses.filter(r => !r.includes('|other|')).map(r => {
            const c = r.split('|');
            c[5] += ';ear=-';
            c[6] = c[6].split(',').concat([c[6].split(',')[0]]).join(',');
            return c.join('|');
        });
        savedForPython(bytes, 'an added node', { poses: [...kept, ...onCamA].sort(), detail: sources.skeleton_camera.detail });
    }
    {
        // Rename a node of the session: only the session's poses take it.
        const page = await open('eager', readFile('skeleton_reversed'));
        await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            state.session.skeleton.nodes[0] = 'snout';
        });
        const bytes = await save(page);
        const poses = sources.skeleton_reversed.poses.map(r => (r.includes('|reversed|') ? r : r.replace('nose=', 'snout='))).sort();
        savedForPython(bytes, 'a renamed node', { poses, detail: sources.skeleton_reversed.detail });
    }
    {
        // Delete a track from the session: a kept pose keeps its name.
        const page = await open('eager', readFile('skeleton_camera'));
        await page.evaluate(async () => {
            const ops = await import('/ui/track-identity-ops.js');
            const { state } = await import('/ui/app-state.js');
            ops.deleteTrackAt(state.session, state.session.tracks.indexOf('mouse1'));
        });
        const bytes = await save(page);
        const poses = sources.skeleton_camera.poses.map(r => (r.includes('|other|') ? r : r.replace('|P|mouse1|', '|P||'))).sort();
        savedForPython(bytes, 'a deleted track', { poses, detail: sources.skeleton_camera.detail, tracks: ['mouse1', 'mouse2'] });
    }
    {
        // Rename a camera: its kept poses stay with it, on the one video it writes, under
        // the file's own name for it. Left behind, they would list that video a second time.
        const page = await open('eager', readFile('skeleton_camera'));
        await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            state.session.cameras.find(c => c.name === 'camB').name = 'camZ';
            state.session.renameCameraInAllData('camB', 'camZ');
        });
        const bytes = await save(page);
        savedForPython(bytes, 'a renamed camera', { poses: sources.skeleton_camera.poses, videos: ['camA.mp4', 'camB.mp4'], sessions: [{ camA: 'camA.mp4', camZ: 'camB.mp4' }] });
    }
    {
        // Lose a camera: its kept poses stay on the file's own record of that video.
        const page = await open('eager', readFile('skeleton_camera'));
        await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            state.session.cameras = state.session.cameras.filter(c => c.name !== 'camB');
        });
        const bytes = await save(page);
        savedForPython(bytes, 'a removed camera', { poses: sources.skeleton_camera.poses, videos: ['camA.mp4', 'camB.mp4'], sessions: [{ camA: 'camA.mp4' }] });
    }
    {
        // Correct a shown pose on a frame holding a kept one: one row, both poses, groups right.
        const page = await open('eager', readFile('skeleton_mixed'));
        await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            const group = state.session.instanceGroups.get(1)[0];
            const inst = group.instances.get('camA');
            inst.setPoint(0, 999, 999);
            inst.type = 'user';
            inst.modified = true;
            group.markDirty();
        });
        const bytes = await save(page);
        const poses = sources.skeleton_mixed.poses.map(r => (r.startsWith('camA.mp4|1|P|mouse1|skeleton|') ? 'camA.mp4|1|U|mouse1|skeleton|nose=999.000,999.000;tail=101.000,20.000|' : r)).sort();
        const groups = sources.skeleton_mixed.groups.map(g => g.replace('s0|1|camA=101.000', 's0|1|camA=999.000')).sort();
        savedForPython(bytes, 'a corrected pose beside a kept one', { poses, groups, detail: sources.skeleton_mixed.detail });
    }
    {
        // Delete every pose the session shows on a frame: the kept one stays.
        const page = await open('eager', readFile('skeleton_mixed'));
        await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            const s = state.session;
            for (const g of [...s.instanceGroups.get(0)]) s.removeInstanceGroup(0, g);
        });
        const bytes = await save(page);
        const poses = sources.skeleton_mixed.poses.filter(r => !/^cam[AB]\.mp4\|0\|P\|mouse\d\|skeleton\|/.test(r));
        const groups = sources.skeleton_mixed.groups.filter(g => !g.startsWith('s0|0|'));
        savedForPython(bytes, 'a frame with only kept poses left', { poses, groups, detail: sources.skeleton_mixed.detail });
    }

    console.log('\nE. File ▸ Add SLP');
    {
        const page = await open('eager', readFile('one_skeleton'));
        page.once('filechooser', fc => fc.setFiles(path.join(tmp, 'skeleton_outside.slp')));
        await page.evaluate(() => { window.__done = false; (async () => {
            await (await import('/import-export/slp-import.js')).handleAddSlp();
        })().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); }); });
        await settle(page, 'Add SLP');
        const status = await statusOf(page);
        check(/not shown; saving keeps/.test(status), 'the status says what is not shown', status);
        const bytes = await save(page);
        const outside = sources.skeleton_outside.poses.filter(r => r.startsWith('outside.mp4|'));
        const fromFile = sources.skeleton_outside.poses.filter(r => !r.startsWith('outside.mp4|')).map(r => r.replace(/^cam([AB])\.mp4\|/, 'cam$1_2.mp4|'));
        savedForPython(bytes, 'Add SLP, kept on an outside video', { poses: [...sources.one_skeleton.poses, ...fromFile, ...outside].sort(), detail: sources.skeleton_outside.detail });
    }
    {
        const page = await open('eager', readFile('one_skeleton'));
        page.once('filechooser', fc => fc.setFiles(path.join(tmp, 'skeleton_mixed.slp')));
        await page.evaluate(() => { window.__done = false; (async () => {
            await (await import('/import-export/slp-import.js')).handleAddSlp();
        })().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); }); });
        await settle(page, 'Add SLP, mixed');
        const status = await statusOf(page);
        check(/6 poses on a camera's video are on another skeleton/.test(status), 'the status counts the poses on the added cameras', status);
        const bytes = await save(page);
        const renamed = sources.skeleton_mixed.poses.map(r => r.replace(/^cam([AB])\.mp4\|/, 'cam$1_2.mp4|'));
        savedForPython(bytes, 'Add SLP, kept on an added camera', { poses: [...sources.one_skeleton.poses, ...renamed].sort(), skeletons: sources.skeleton_mixed.skeletons });
    }

    console.log('\nF. A per-camera folder, through the raw worker');
    const openFolder = async (route, camA = 'camA_two_skeletons') => {
        const page = await newAppPage();
        await putBytes(page, '__camA', readFile(camA));
        await putBytes(page, '__camB', readFile('camB_alone'));
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
    {
        const poses = [...sources.camA_two_skeletons.poses, ...sources.camB_alone.poses].sort();
        const { page, status } = await openFolder('eager');
        check(/3 labeled frames on 1 video in no session are not shown/.test(status), 'the status says what is on another video', status);
        check(/1 pose on a camera's video is on another skeleton \("other"\) and is not shown; saving keeps it on that skeleton/.test(status), 'the status says what is on another skeleton', status);
        const shown = await shownInApp(page);
        check(shown[0].cams.camA.every(r => r[1] === 2 && r[2] < 200), 'a camera shows only the session\'s poses', shown[0].cams.camA);
        const saved = await save(page);
        savedForPython(saved, 'per-camera, eager', { poses, detail: sources.camA_two_skeletons.detail, sessions: [{ camA: 'camA.mp4', camB: 'camB.mp4' }], skeletons: sources.camA_two_skeletons.skeletons });
        const again = await save(await open('eager', saved));
        savedForPython(again, 'per-camera, eager, saved again as a project', { poses, detail: sources.camA_two_skeletons.detail, sessions: [{ camA: 'camA.mp4', camB: 'camB.mp4' }], skeletons: sources.camA_two_skeletons.skeletons });
    }
    {
        // A file with labels on its detections takes the typed reader: the same rule, and the label stays.
        const poses = [...sources.camA_labeled.poses, ...sources.camB_alone.poses].sort();
        const { page, status } = await openFolder('eager', 'camA_labeled');
        check(/1 pose on a camera's video is on another skeleton \("other"\)/.test(status), 'typed route: the status says what is on another skeleton', status);
        const shown = await shownInApp(page);
        check(shown[0].cams.camA.every(r => r[1] === 2 && r[2] < 200), 'typed route: a camera shows only the session\'s poses', shown[0].cams.camA);
        check(sources.camA_labeled.detail.some(r => r.includes('identity=beta')), 'the source holds an identity on a kept pose', sources.camA_labeled.detail);
        savedForPython(await save(page), 'per-camera, typed route', { poses, detail: sources.camA_labeled.detail, sessions: [{ camA: 'camA.mp4', camB: 'camB.mp4' }], skeletons: sources.camA_labeled.skeletons });
    }
    {
        // The lazy route's save, for the same poses.
        const poses = [...sources.camA_two_skeletons.poses, ...sources.camB_alone.poses].sort();
        const { page } = await openFolder('lazy');
        savedForPython(await save(page), 'per-camera, lazy', { poses, detail: sources.camA_two_skeletons.detail, sessions: [{ camA: 'camA.mp4', camB: 'camB.mp4' }] });
    }

    console.log('\nG. A group linking a pose on another skeleton: the load refuses');
    {
        // Nothing open: nothing opens, and the status says why.
        for (const loader of ['eager', 'folder']) {
            const page = await open(loader, readFile('skeleton_group'));
            const status = await statusOf(page);
            check(/Not opened: 3 groups in this file link poses on different skeletons/.test(status), `${loader}: the status says why`, status);
            const held = await page.evaluate(async () => {
                const { state } = await import('/ui/app-state.js');
                return { sessions: state.sessions.length, session: !!state.session };
            });
            check(eq(held, { sessions: 0, session: false }), `${loader}: nothing opened`, held);
            await page.close();
        }
    }
    {
        // A project open: it stays, untouched, and saves as before.
        for (const loader of ['eager', 'folder']) {
            const page = await open('eager', readFile('one_skeleton'));
            const before = await page.evaluate(async () => {
                const { state } = await import('/ui/app-state.js');
                return { name: state.session.name, frames: state.session.numFrames, dirty: !!state.isDirty, sessions: state.sessions.length, views: state.views.length };
            });
            await putBytes(page, '__file', readFile('skeleton_group'));
            await LOADERS[loader](page, readFile('skeleton_group'));
            await settle(page, 'refused load');
            const status = await statusOf(page);
            check(/Not opened: 3 groups/.test(status), `${loader}, project open: the status says why`, status);
            const after = await page.evaluate(async () => {
                const { state } = await import('/ui/app-state.js');
                return { name: state.session.name, frames: state.session.numFrames, dirty: !!state.isDirty, sessions: state.sessions.length, views: state.views.length };
            });
            check(eq(before, after), `${loader}, project open: the project is as it was`, { before, after });
            savedForPython(await save(page), `${loader}, project open: its save`, { poses: sources.one_skeleton.poses, skeletons: sources.one_skeleton.skeletons });
        }
    }
    {
        // Through the real menus. File ▸ Open records the file as the save target once it
        // has opened: a refused one must not become it, or the next Save writes the open
        // project over the file that was refused.
        const menu = async (page, id, bytesGlobal, name) => {
            await page.evaluate(([id, g, name]) => {
                const bytes = window[g];
                window.showOpenFilePicker = async () => [{ __tag: name, kind: 'file', name, getFile: async () => new File([bytes], name) }];
                document.getElementById('statusText').textContent = 'waiting';
                document.getElementById(id).click();
            }, [id, bytesGlobal, name]);
            for (let i = 0; i < 240; i++) {
                const s = await page.evaluate(() => {
                    const b = [...document.querySelectorAll('button')].find(x => x.offsetParent && /^(Skip|Later|Cancel|Not now)$/i.test((x.textContent || '').trim()));
                    if (b) { b.click(); return { clicked: true }; }
                    return { status: document.getElementById('statusText').textContent, prompt: !!document.getElementById('videoSelectOverlay') };
                });
                if (s.prompt) await page.keyboard.press('Escape');
                if (/^SLP loaded|^Not opened/.test(s.status || '')) {
                    // The handle is recorded once the loader's promise resolves, just after its status.
                    await page.waitForTimeout(300);
                    return { status: s.status, handle: await page.evaluate(() => { const h = window.__lucid.state.slpFileHandle; return h ? h.__tag : null; }) };
                }
                await page.waitForTimeout(250);
            }
            return { status: 'timeout', handle: null };
        };
        const page = await newAppPage();
        await putBytes(page, '__refused', readFile('skeleton_group'));
        await putBytes(page, '__project', readFile('one_skeleton'));
        let r = await menu(page, 'menuLoadSlp', '__refused', 'refused.slp');
        check(/^Not opened/.test(r.status) && r.handle === null, 'File ▸ Load SLP, refused: not the save target', r);
        r = await menu(page, 'menuLoadProject', '__refused', 'refused.slp');
        check(/^Not opened/.test(r.status) && r.handle === null, 'File ▸ Open, refused: not the save target', r);
        r = await menu(page, 'menuLoadProject', '__project', 'project.slp');
        check(/^SLP loaded/.test(r.status) && r.handle === 'project.slp', 'File ▸ Open of a file that opens: the save target', r);
        r = await menu(page, 'menuLoadProject', '__refused', 'refused.slp');
        check(/^Not opened/.test(r.status) && r.handle === 'project.slp', 'File ▸ Open, refused over a project: the next Save still goes to that project\'s file', r);
        await page.close();
    }
    {
        // The raw worker has no typed session to check, and these groups go by position.
        const page = await open('raw', readFile('skeleton_legacy_groups'));
        const status = await statusOf(page);
        check(/Not opened: 6 groups/.test(status), 'raw worker, groups beside another skeleton\'s poses: refuses', status);
        await page.close();
        // Groups in `/session_data` are out of its reach, so it does not stand in at all.
        const raw = await open('raw', readFile('skeleton_mixed'));
        check(/forced to the raw worker/.test(await statusOf(raw)), 'raw worker, a file holding its groups in /session_data: not opened', await statusOf(raw));
        await raw.close();
    }
    {
        // The lazy route keeps such a file's groups in its store: nothing refused, nothing lost.
        const bytes = await save(await open('lazy', readFile('skeleton_group')));
        savedForPython(bytes, 'lazy route, a group across skeletons', { poses: sources.skeleton_group.poses, groups: sources.skeleton_group.groups, skeletons: sources.skeleton_group.skeletons });
    }

    console.log('\nH. A LUC3D project whose sessions are on different skeletons');
    for (const [label, nameA, nameB, nodesB, edgesB] of [
        ['same name', 'skeleton', 'skeleton', ['a', 'b', 'c'], [[0, 1]]],
        ['different names', 'skA', 'skB', ['a', 'b', 'c'], [[0, 1]]],
        ['one node more', 'skA', 'skB', ['nose', 'tail', 'ear'], [[0, 1]]],
        ['other edges', 'skA', 'skB', ['nose', 'tail'], [[1, 0]]],
    ]) {
        const page = await newAppPage();
        const built = await page.evaluate(async ([nameA, nameB, nodesB, edgesB]) => {
            const pd = await import('/pose/pose-data.js');
            const { state } = await import('/ui/app-state.js');
            const cam = n => new pd.Camera(n, [[500, 0, 32], [0, 500, 32], [0, 0, 1]], [0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0], [64, 64]);
            const mk = (camName, skel, name, base) => {
                const s = new pd.Session([cam(camName)], skel, ['mouse1'], name);
                for (let f = 0; f < 3; f++) {
                    const fg = new pd.FrameGroup(f);
                    s.addFrameGroup(fg);
                    const pts = skel.nodes.map((_, k) => [base + f + k, 10 + k]);
                    const inst = new pd.Instance(pts, 0, 'user', 0);
                    if (f === 0) {
                        // Each session's own groups must not read as crossing skeletons.
                        const g = new pd.InstanceGroup(f, null);
                        g.addInstance(camName, inst);
                        g.points3d = new Float64Array(skel.nodes.length * 3).fill(base);
                        fg.addInstance(camName, inst);
                        s.instanceGroups.set(f, [g]);
                    } else {
                        fg.addUnlinkedInstance(camName, new pd.UnlinkedInstance(inst, camName));
                    }
                }
                return s;
            };
            const a = mk('camA', new pd.Skeleton(nameA, ['nose', 'tail'], [[0, 1]]), 'A', 100);
            const b = mk('camB', new pd.Skeleton(nameB, nodesB, edgesB), 'B', 200);
            state.sessions = [a, b];
            state.session = a;
            state.activeSessionIdx = 0;
            return true;
        }, [nameA, nameB, nodesB, edgesB]);
        check(built, label + ': built');
        const first = await save(page);
        const page2 = await open('eager', first);
        const status = await statusOf(page2);
        check(!/not shown/.test(status), label + ': nothing hidden', status);
        const shown = await shownInApp(page2);
        check(shown[1] && shown[1].cams.camB && shown[1].cams.camB.every(r => r[1] === nodesB.length) && shown[1].cams.camB.length === 3, label + ': the second session shows every node of its own poses', shown[1] && shown[1].cams);
        check(shown[0] && shown[0].cams.camA && shown[0].cams.camA.every(r => r[1] === 2), label + ': the first session shows its own', shown[0] && shown[0].cams);
        const second = await save(page2);
        const expected = [];
        for (let f = 0; f < 3; f++) {
            expected.push(`camA.mp4|${f}|U|mouse1|${nameA}|nose=${(100 + f).toFixed(3)},10.000;tail=${(101 + f).toFixed(3)},11.000|`);
            expected.push(`camB.mp4|${f}|U|mouse1|${nameB}|` + nodesB.map((n, k) => `${n}=${(200 + f + k).toFixed(3)},${(10 + k).toFixed(3)}`).join(';') + '|');
        }
        const groups = [`s0|0|camA=100.000/${nameA}|100.000,100.000,100.000`, `s1|0|camB=200.000/${nameB}|200.000,200.000,200.000`];
        savedForPython(first, label + ', save 1', { poses: expected.sort(), groups });
        savedForPython(second, label + ', save 2', { poses: expected, groups });
    }

    console.log('\nI. A project mixing lazy and eager sessions');
    {
        const page = await open('lazy', readFile('skeleton_outside'));
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
        // Only that video's: a lazy camera's poses on this writer lose their per-node scores, as they always have.
        const outside = sources.skeleton_outside.poses.filter(r => r.startsWith('outside.mp4|'));
        savedForPython(await save(page), 'mixed project', { onVideo: 'outside.mp4', poses: outside, detail: sources.skeleton_outside.detail });
    }

    console.log('\nJ. A suggestion on a video kept poses are on');
    {
        const source = sources.skeleton_suggested;
        check(eq(source.suggestions, ['outside.mp4|40', 'camB.mp4|41', 'camA.mp4|42']), 'skeleton_suggested.slp holds a suggestion on each of its videos', source.suggestions);
        const held = { poses: source.poses, detail: source.detail, suggestions: source.suggestions };
        for (const loader of ['eager', 'folder', 'raw']) {
            let bytes = readFile('skeleton_suggested');
            for (let gen = 1; gen <= 2; gen++) {
                const page = await open(loader, bytes);
                bytes = await save(page);
                savedForPython(bytes, `suggested, ${loader}, save ${gen}`, { ...held, videos: ['camA.mp4', 'camB.mp4', 'outside.mp4'], sessions: source.sessions });
            }
        }
        {
            // A renamed camera keeps the one entry for its video, poses and suggestion.
            const page = await open('eager', readFile('skeleton_suggested'));
            await page.evaluate(async () => {
                const { state } = await import('/ui/app-state.js');
                state.session.cameras.find(c => c.name === 'camA').name = 'top';
                state.session.renameCameraInAllData('camA', 'top');
            });
            savedForPython(await save(page), 'suggested, a renamed camera', { ...held, videos: ['camA.mp4', 'camB.mp4', 'outside.mp4'], sessions: [{ top: 'camA.mp4', camB: 'camB.mp4' }] });
        }
        {
            // A camera gone: its kept poses and its suggestion share the file's own record.
            const page = await open('eager', readFile('skeleton_suggested'));
            await page.evaluate(async () => {
                const { state } = await import('/ui/app-state.js');
                state.session.cameras = state.session.cameras.filter(c => c.name !== 'camA');
            });
            savedForPython(await save(page), 'suggested, a camera gone', { ...held, videos: ['camB.mp4', 'camA.mp4', 'outside.mp4'], sessions: [{ camB: 'camB.mp4' }] });
        }
    }
    {
        // A video moved to another session takes its kept pose along, as it does its suggestion:
        // one row for the frame, in the session the video is now in. A calibrated session
        // keeps its camera, so it lists that video too, with no poses, whatever the poses are.
        const source = sources.skeleton_suggested_two;
        const page = await open('eager', readFile('skeleton_suggested_two'));
        await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            const panes = await import('/ui/sessions-panes.js');
            state.activeSessionIdx = 0;
            state.session = state.sessions[0];
            const moved = panes.showMoveVideoModal(['camB'], 0, 1);
            [...document.querySelectorAll('button')].find(b => b.offsetParent && b.textContent.trim() === 'Continue').click();
            await moved;
        });
        const bytes = await save(page);
        const movedWant = { poses: source.poses, detail: source.detail, suggestions: source.suggestions, sessions: [{ camA: 'camA.mp4', camB: 'camB.mp4' }, { camC: 'camC.mp4', camD: 'camD.mp4', camB: 'camB.mp4' }], videos: ['camA.mp4', 'camB.mp4', 'camC.mp4', 'camD.mp4', 'camB.mp4'] };
        savedForPython(bytes, 'suggested, a video moved to another session', movedWant);
        savedForPython(await save(await open('eager', bytes)), 'suggested, a video moved to another session, saved again', movedWant);
    }
    {
        // File ▸ Add SLP: the added file's ids mean other videos than the open project's.
        const source = sources.skeleton_suggested;
        const page = await open('eager', readFile('one_skeleton'));
        page.once('filechooser', fc => fc.setFiles(path.join(tmp, 'skeleton_suggested.slp')));
        await page.evaluate(() => { window.__done = false; (async () => {
            await (await import('/import-export/slp-import.js')).handleAddSlp();
        })().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); }); });
        await settle(page, 'Add SLP, suggested');
        const renamed = r => r.replace(/^cam([AB])\.mp4\|/, 'cam$1_2.mp4|');
        savedForPython(await save(page), 'suggested, Add SLP', {
            poses: [...sources.one_skeleton.poses, ...source.poses.map(renamed)].sort(), suggestions: source.suggestions.map(renamed),
            videos: ['camA.mp4', 'camB.mp4', 'camA_2.mp4', 'camB_2.mp4', 'outside.mp4'],
        });
    }
    {
        const { page } = await openFolder('eager', 'camA_suggested');
        savedForPython(await save(page), 'suggested, per-camera', {
            poses: [...sources.camA_suggested.poses, ...sources.camB_alone.poses].sort(), detail: sources.camA_suggested.detail,
            suggestions: ['camA_second.mp4|43', 'camA.mp4|44'], videos: ['camA.mp4', 'camB.mp4', 'camA_second.mp4'], sessions: [{ camA: 'camA.mp4', camB: 'camB.mp4' }],
        });
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
