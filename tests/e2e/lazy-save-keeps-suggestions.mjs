/**
 * lazy-save-keeps-suggestions.mjs — a file's suggestions (`suggestions_json`, SLEAP's
 * frames to label next) stay on their own videos and frames through a lazy save.
 *
 * The streaming writer used to receive no suggestions, so every save of a lazily
 * opened `.slp` dropped them with no word. They point at the reader's `Video` objects
 * and the writer finds each by object in a header holding fresh objects for the
 * session's cameras, so a suggestion passed through as read writes video -1, which
 * Python reads as the LAST video in the list: a suggestion on the wrong video, no error.
 * So this reads each save's raw rows as well as what Python makes of them.
 *
 * Real Python sleap-io writes the files and reads every save.
 *
 *  A. Every layout through the lazy route: two saves in one sitting, then the first
 *     save reopened and saved again.
 *  B. A group in the file stays as the file holds it. A suggestion listed twice stays twice.
 *  C. Per-camera lazy files: a suggestion on the camera's video, on another video
 *     with poses, and on one with none.
 *  D. The two-pass save (`saveAllSessionsStreaming`), whose header lists a later
 *     session's videos at an offset and whose pass 2 re-reads every file.
 *  E. A correction moves frames, not suggestions. An attached video renames a camera's
 *     entry, and its suggestions follow.
 *  F. A file with no suggestions gets none, and no extra video.
 *
 * Needs `uv` for Python (skipped without it).
 *
 * Run: node tests/e2e/lazy-save-keeps-suggestions.mjs   (spawns its own http.server; PORT= to move it)
 */
import { chromium } from 'playwright';
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

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

// One line per suggestion: video filename|frame|group, in file order. `listed` and
// `onCamera` hold what a filename cannot show: each raw row's video index is one the list
// holds (a -1 reads in Python as the last video), and which camera's video it is on.
const PY_READ = `
import json, sys
import h5py
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
    return "|".join([video, str(frame_idx), "P" if predicted else "U", inst.track.name if inst.track else "", ";".join(cells)])

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
    sessions, groups = [], []
    camera_videos = {}
    for si, s in enumerate(labels.sessions):
        cams = {}
        for c in s.camera_group.cameras:
            v = s.get_video(c)
            if v is not None:
                cams[c.name] = v.filename
                camera_videos[id(v)] = c.name
        sessions.append(cams)
        for f, fg in s.frame_groups.items():
            for ig in fg.instance_groups:
                members = sorted(c.name + "=" + num(i.points["xy"][0][0]) for c, i in ig.instance_by_camera.items())
                groups.append("|".join(["s" + str(si), str(f)] + members))
    return {
        "error": None,
        "videos": [v.filename for v in labels.videos],
        "suggestions": [s.video.filename + "|" + str(s.frame_idx) + "|" + json.dumps(s.metadata.get("group")) for s in labels.suggestions],
        "onCamera": [camera_videos.get(id(s.video), "") for s in labels.suggestions],
        "listed": in_list,
        "rawGroups": [r.get("group", "absent") for r in raw],
        "poses": sorted(pose_row(lf.video.filename, lf.frame_idx, i) for lf in labels.labeled_frames for i in lf.instances),
        "sessions": sessions,
        "groups": sorted(groups),
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

out = sys.argv[1]
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

def session(pairs):
    cams = [sio.Camera(name=n, rvec=np.array([0.0, 0.1 * i, 0.0]), tvec=np.array([10.0 * i, 0.0, 0.0]),
                       matrix=np.array([[500.0, 0, 32], [0, 500.0, 32], [0, 0, 1]]), dist=np.zeros(5), size=(64, 64))
            for i, (n, _, _) in enumerate(pairs)]
    s = sio.RecordingSession(camera_group=sio.CameraGroup(cameras=cams))
    for c, (_, v, _) in zip(cams, pairs):
        if v is not None:
            s.add_video(v, c)
    with_frames = [(c, {lf.frame_idx: lf for lf in fs}) for c, (_, _, fs) in zip(cams, pairs) if fs]
    for f in sorted(with_frames[0][1]) if with_frames else []:
        igs = [sio.InstanceGroup(instance_by_camera={c: by[f].instances[i] for c, by in with_frames}, score=0.5,
                                 instance_3d=Instance3D(points=np.array([[1.0 + i + f, 2.0, 3.0], [4.0, 5.0, 6.0 + i]]), skeleton=sk))
               for i in range(2)]
        s.frame_groups[f] = sio.FrameGroup(frame_idx=f, instance_groups=igs, labeled_frame_by_camera={c: by[f] for c, by in with_frames})
    return s

def sug(v, *frames):
    return [SuggestionFrame(video=v, frame_idx=f) for f in frames]

def save(name, frames, videos, sessions, suggestions):
    sio.Labels(labeled_frames=frames, videos=videos, skeletons=[sk], tracks=[T1, T2], sessions=sessions,
               suggestions=suggestions).save(os.path.join(out, name + ".slp"), embed=False)

def two_cams():
    a, b = video("camA.mp4"), video("camB.mp4")
    fa, fb = frames_on(a, 100), frames_on(b, 200)
    return a, b, fa, fb, session([("camA", a, fa), ("camB", b, fb)])

# Only the cameras' videos. One suggestion on a frame with poses, the rest on frames with none.
a, b, fa, fb, s = two_cams()
save("cameras", fa + fb, [a, b], [s], sug(a, 10, 0) + sug(b, 20) + sug(a, 30))

# A video no session names, listed last: a store id past a one-video-per-camera header.
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
with h5py.File(os.path.join(out, "groups.slp"), "r+") as f:
    del f["suggestions_json"]
    f.create_dataset("suggestions_json", data=[json.dumps(r).encode() for r in rows])

# Per-camera files: frames on the camera's video and another; only on the second
# listed; a third video with only suggestions.
a, x, y = video("camA.mp4"), video("camA_second.mp4"), video("camA_no_poses.mp4")
sio.Labels(labeled_frames=frames_on(a, 100) + frames_on(x, 500), videos=[a, x, y], skeletons=[sk], tracks=[T1, T2],
           suggestions=sug(x, 80) + sug(a, 81) + sug(y, 82)).save(os.path.join(out, "camA_file.slp"), embed=False)
unused, b = video("camB_unused.mp4"), video("camB.mp4")
sio.Labels(labeled_frames=frames_on(b, 200), videos=[unused, b], skeletons=[sk], tracks=[T1, T2],
           suggestions=sug(b, 83) + sug(unused, 84)).save(os.path.join(out, "camB_file.slp"), embed=False)

# A second session for section D, as per-camera files and as a project.
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

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'lazy-suggestions-'));
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

/** Answer the video prompts a load puts up, until `window.__done` is set. */
async function settle(page, what) {
    let done = false;
    // By the clock, not a count of tries: at a high machine load each try takes seconds.
    for (const end = Date.now() + 240000; Date.now() < end && !done;) {
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

/** The lazy route: `handleLoadProjectSlpLazy`, as File ▸ Open takes any `.slp` over 200 MB. */
async function openLazy(bytes) {
    const page = await newAppPage();
    await putBytes(page, '__file', bytes);
    await page.evaluate(() => { window.__done = false; (async () => {
        await (await import('/loading/session-loader.js')).handleLoadProjectSlpLazy(new File([window.__file], 'project.slp'));
    })().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); }); });
    await settle(page, 'lazy load');
    return page;
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
    return savedBytes(page);
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

let savedSeq = 0;
const pythonWork = [];   // [{ file, what, want }] checked together: one Python run reads every save
function expectSave(bytes, what, want) {
    const file = path.join(tmp, 'saved-' + (++savedSeq) + '.slp');
    fs.writeFileSync(file, bytes);
    pythonWork.push({ file, what, want });
    stallGuard.refresh();
}

/**
 * Only what `want` names is checked. `onCamera` is one camera name per suggestion, '' for
 * none, null for a closed session's video: a loader that opens only a file's first session
 * makes that session's cameras no cameras of the save, or, if it keeps the session, keeps
 * them. `sessions` and `groups` likewise name the open session's when `want.openSessionOnly`.
 */
function compare(what, got, want) {
    if (got.error) { check(false, what + ': opens', got.error); return; }
    if (want.suggestions) check(eq(got.suggestions, want.suggestions), what + ': every suggestion on its own video and frame', { got: got.suggestions, want: want.suggestions });
    check(got.listed === true, what + ': each points at a video the list holds', got.videos);
    if (want.onCamera) {
        const cams = got.onCamera.map((c, i) => (want.onCamera[i] === null ? null : c));
        check(eq(cams, want.onCamera), what + ': a camera\'s suggestion is on that camera\'s video', got.onCamera);
    }
    if (want.openSessionOnly) {
        got = { ...got, sessions: (got.sessions || []).slice(0, 1), groups: (got.groups || []).filter(g => g.startsWith('s0|')) };
    }
    if (want.rawGroups) check(eq(got.rawGroups, want.rawGroups), what + ': groups as the file held them', got.rawGroups);
    if (want.poses) check(eq(got.poses, want.poses), what + ': every pose on its own video', diff(got.poses || [], want.poses));
    if (want.groups) check(eq(got.groups, want.groups), what + ': groups point at the right poses', diff(got.groups || [], want.groups));
    if (want.sessions) check(eq(got.sessions, want.sessions), what + ': cameras and their videos', got.sessions);
    if (want.videos) check(eq(got.videos, want.videos), what + ': the video list', got.videos);
}

/** The camera each of `source`'s suggestions is on, as `compare` wants it. */
function onCameras(source) {
    const camOf = new Map(Object.entries(source.sessions[0]).map(([cam, file]) => [file, cam]));
    const closed = new Set(source.sessions.slice(1).flatMap(cams => Object.values(cams)));
    return source.suggestions.map((r) => {
        const file = r.split('|')[0];
        return camOf.has(file) ? camOf.get(file) : (closed.has(file) ? null : '');
    });
}

// What each layout's header should list after a lazy save: the session's cameras, then
// videos with poses, then those only a suggestion names.
const FILES = {
    cameras: { videos: ['camA.mp4', 'camB.mp4'] },
    one: { videos: ['camA.mp4', 'camB.mp4'] },
    last: { videos: ['camA.mp4', 'camB.mp4', 'outside.mp4'] },
    first: { videos: ['camA.mp4', 'camB.mp4', 'outside.mp4'] },
    only_suggested: { videos: ['camA.mp4', 'camB.mp4', 'outside.mp4', 'no_poses_y.mp4', 'no_poses_z.mp4'] },
    two_sessions: { videos: ['camA.mp4', 'camB.mp4', 'outside.mp4', 'camC.mp4', 'camD.mp4'], firstSessionOnly: true },
    all_outside: { videos: ['camA.mp4', 'camB.mp4', 'outside.mp4'] },
    no_map: { poseOnly: true },
    same_name: { videos: ['rec/cam0/0.mp4', 'rec/cam1/0.mp4', 'rec/other/0.mp4'] },
};

try {
    uv(PY_WRITE, [tmp]);
    const names = [...Object.keys(FILES), 'none', 'none_plain', 'groups', 'camA_file', 'camB_file', 'camC_file', 'camD_file', 'second_project'];
    const rawSources = JSON.parse(uv(PY_READ, names.map(n => path.join(tmp, n + '.slp'))).trim().split('\n').pop());
    const sources = {};
    for (const n of names) sources[n] = rawSources[path.join(tmp, n + '.slp')];
    const readFile = n => fs.readFileSync(path.join(tmp, n + '.slp'));

    console.log('Sources, as Python reads them:');
    check(sources.last.suggestions.length === 4 && sources.first.suggestions[0].startsWith('camA.mp4|12|'), 'the files hold suggestions, each on its video', sources.first.suggestions);
    check(sources.none.suggestions.length === 0, 'the control holds none');
    check(sources.groups.rawGroups.length === 7, 'the group file is as written', sources.groups.rawGroups);

    console.log('\nA. Every layout, the lazy route, two saves in a sitting, then reopened and saved');
    for (const [name, layout] of Object.entries(FILES)) {
        const source = sources[name];
        const want = { suggestions: source.suggestions, poses: source.poses };
        if (!layout.poseOnly) {
            want.videos = layout.videos;
            want.openSessionOnly = !!layout.firstSessionOnly;
            want.sessions = layout.firstSessionOnly ? source.sessions.slice(0, 1) : source.sessions;
            want.groups = layout.firstSessionOnly ? source.groups.filter(g => g.startsWith('s0|')) : source.groups;
            want.onCamera = onCameras(source);
        }
        const page = await openLazy(readFile(name));
        const first = await save(page);
        expectSave(first, `${name}, save 1`, want);
        expectSave(await save(page), `${name}, save 2 in the same sitting`, want);
        await page.close();
        const again = await openLazy(first);
        expectSave(await save(again), `${name}, reopened and saved`, want);
        await again.close();
    }

    console.log('\nB. Groups, and a suggestion listed twice');
    {
        // In [x, a, b] of the source; the save lists [a, b, x].
        const want = {
            suggestions: [
                'camA.mp4|70|3', 'camB.mp4|71|"night"', 'outside.mp4|72|0', 'camA.mp4|73|0', 'camB.mp4|74|0', 'camB.mp4|75|1', 'camB.mp4|75|1',
            ],
            rawGroups: [3, 'night', 0, 0, 0, 1, 1],
        };
        const page = await openLazy(readFile('groups'));
        const first = await save(page);
        expectSave(first, 'groups, save 1', want);
        await page.close();
        const again = await openLazy(first);
        expectSave(await save(again), 'groups, reopened and saved', want);
        await again.close();
    }

    console.log('\nC. Per-camera lazy files');
    {
        const page = await newAppPage();
        await putBytes(page, '__camA', readFile('camA_file'));
        await putBytes(page, '__camB', readFile('camB_file'));
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
            await loader.handleLoadSessionFolderPerCamera([
                mkFile(window.__camA, 'root/camA/camA.slp'), mkFile(window.__camB, 'root/camB/camB.slp'),
            ], true);
            return !!window.__lucid.state.session.lazyLoader;
        });
        // These folders hold no videos, so the load waits on the Missing Video Files popup.
        await page.getByRole('button', { name: 'Continue', exact: true }).click();
        check(await loading, 'the folder opened lazily');
        const want = {
            suggestions: [...sources.camA_file.suggestions, ...sources.camB_file.suggestions],
            onCamera: ['', 'camA', '', 'camB', ''],
            poses: [...sources.camA_file.poses, ...sources.camB_file.poses].sort(),
            videos: ['camA.mp4', 'camB.mp4', 'camA_second.mp4', 'camA_no_poses.mp4', 'camB_unused.mp4'],
        };
        const first = await save(page);
        expectSave(first, 'per-camera, save 1', want);
        expectSave(await save(page), 'per-camera, save 2 in the same sitting', want);
        await page.close();
        const again = await openLazy(first);
        expectSave(await save(again), 'per-camera, saved again as a project', want);
        await again.close();
    }

    console.log('\nD. The two-pass save');
    for (const [firstFile, second, what] of [
        ['first', 'project', 'two project files'],
        ['two_sessions', 'project', 'a two-session file + a project file'],
        ['only_suggested', 'cameras', 'a project file + per-camera files'],
    ]) {
        const page = await openLazy(readFile(firstFile));
        await putBytes(page, '__second', readFile('second_project'));
        await putBytes(page, '__camC', readFile('camC_file'));
        await putBytes(page, '__camD', readFile('camD_file'));
        const err = await page.evaluate(async (second) => {
            const pd = await import('/pose/pose-data.js');
            const { SioLazyLoader } = await import('/loading/sio-lazy-loader.js');
            const AS = await import('/ui/app-state.js');
            const sl = await import('/import-export/save-load.js');
            const { Camera, Instance, InstanceGroup, FrameGroup, Session } = pd;
            const names = ['camC', 'camD'];
            const mtx = [[500, 0, 32], [0, 500, 32], [0, 0, 1]];
            const cams = names.map((n, i) => new Camera(n, mtx, [0, 0, 0, 0, 0], [0, 0.1 * i, 0], [10 * i, 0, 0], [64, 64]));
            const s2 = new Session(cams, AS.state.session.skeleton, ['mouse1', 'mouse2'], 'second');
            const loader = new SioLazyLoader();
            if (second === 'project') {
                await loader.openProjectSlp(new File([window.__second], 'second.slp'));
            } else {
                await loader.open('camC', new File([window.__camC], 'camC.slp'));
                await loader.open('camD', new File([window.__camD], 'camD.slp'));
            }
            s2.lazyLoader = loader;
            for (let f = 0; f < 3; f++) {
                s2.addFrameGroup(new FrameGroup(f));
                const gs = [];
                for (let i = 0; i < 2; i++) {
                    const g = new InstanceGroup(f * 10 + i, null);
                    names.forEach((n) => {
                        const inst = new Instance([[1, 1], [2, 2]], i, 'predicted', 0.9);
                        inst._rawInstIndex = i;
                        g.addInstance(n, inst);
                    });
                    g.points3d = new Float64Array([1 + i + f, 2, 3, 4, 5, 6 + i]);
                    gs.push(g);
                }
                s2.instanceGroups.set(f, gs);
            }
            AS.state.sessions.push(s2);
            try {
                const orig = URL.createObjectURL;
                let blob = null;
                URL.createObjectURL = b => { blob = b; return orig.call(URL, b); };
                try { await sl.saveProjectSlp(); } finally { URL.createObjectURL = orig; }
                if (!blob) return 'saveProjectSlp produced no file: ' + (document.getElementById('statusText') || {}).textContent;
                window.__saved = new Uint8Array(await blob.arrayBuffer());
                return null;
            } catch (e) { return String(e && e.stack || e); }
        }, second);
        check(!err, what + ': the two-pass save runs', err);
        if (err) { await page.close(); continue; }
        const bytes = await savedBytes(page);
        await page.close();
        const secondSource = second === 'project'
            ? { suggestions: sources.second_project.suggestions, poses: sources.second_project.poses }
            : { suggestions: [...sources.camC_file.suggestions, ...sources.camD_file.suggestions], poses: [...sources.camC_file.poses, ...sources.camD_file.poses] };
        const want = {
            suggestions: [...sources[firstFile].suggestions, ...secondSource.suggestions],
            poses: [...sources[firstFile].poses, ...secondSource.poses].sort(),
        };
        const secondCams = { 'camC.mp4': 'camC', 'camD.mp4': 'camD' };
        expectSave(bytes, what + ', the two-pass save', {
            ...want,
            onCamera: [...onCameras(sources[firstFile]), ...secondSource.suggestions.map(r => secondCams[r.split('|')[0]] || '')],
        });
        // Its next life: the first session opens, the rest of the file comes along.
        const again = await openLazy(bytes);
        expectSave(await save(again), what + ', reopened and saved', want);
        await again.close();
    }

    console.log('\nE. Edits: a correction, and a video attached under another name');
    {
        const page = await openLazy(readFile('first'));
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
        const bytes = await save(page);
        await page.close();
        expectSave(bytes, 'a corrected frame', { suggestions: sources.first.suggestions });
    }
    {
        // By the video's id, not its name: the header calls camera A's video something else.
        const page = await openLazy(readFile('first'));
        await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            state.videoFiles = [{ name: 'camA', assignedCamera: 'camA', videoPath: '/moved/here/camA_new.mp4' }];
        });
        const bytes = await save(page);
        await page.close();
        expectSave(bytes, 'a video attached under another name', {
            suggestions: sources.first.suggestions.map(r => r.replace(/^camA\.mp4\|/, '/moved/here/camA_new.mp4|')),
            onCamera: onCameras(sources.first),
            videos: ['/moved/here/camA_new.mp4', 'camB.mp4', 'outside.mp4'],
        });
    }

    console.log('\nF. No suggestions: none, and no extra video');
    for (const [name, videos] of [['none', ['camA.mp4', 'camB.mp4', 'outside.mp4']], ['none_plain', ['camA.mp4', 'camB.mp4']]]) {
        const page = await openLazy(readFile(name));
        const bytes = await save(page);
        await page.close();
        expectSave(bytes, name, { suggestions: [], videos, poses: sources[name].poses, groups: sources[name].groups, sessions: sources[name].sessions });
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
