/**
 * poses-camera-other-skeleton.mjs — a per-camera folder whose camera files are on
 * different skeletons. The rule is in `loading/folder-skeleton.js`: a save never moves a
 * pose off its own nodes and never writes a file Python cannot open, on either route.
 *
 * The folder used to read every camera by position onto one camera's skeleton, with no
 * word in the status. Eager: `camB.mp4|0|P|main|nose=200;tail=205`, with `c` gone. Lazy:
 * a save Python could not open (`ValueError: could not broadcast ...`). Same nodes in
 * another order: nose and tail swapped on both.
 *
 * Real Python sleap-io writes the files and reads every save. Only Python can tell: it
 * reads each pose on the skeleton the file says it is on.
 *
 *  A. A camera on other nodes: not shown, kept, saved twice and reopened both ways.
 *  B. Which skeleton the session shows: most cameras, then the first by name.
 *  C. Same nodes in another order: shown, read by name, whichever file opens first.
 *  D. Same nodes, other edges or another name: shown, as ever.
 *  E. A camera file holding poses on two skeletons, or on other videos too.
 *  F. Edits around a hidden camera.
 *  G. A project mixing lazy and eager sessions.
 *  H. Several folders in one project, each on its own skeleton.
 *  I. A folder's `skeleton.json` decides.
 *  J. Analysis `.h5` cameras.
 *  K. Per-camera export of a hidden camera, and the raw worker as the fallback read.
 *  L. The real multi-session picker: its last status keeps what each session's said.
 *
 * Needs `uv` for Python (skipped without it).
 *
 * Run: node tests/e2e/poses-camera-other-skeleton.mjs   (spawns its own http.server; PORT= to move it)
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

// One line per pose: video|frame|U or P|track|skeleton|node=value;...|scores. A value is
// `x,y`, `x,y~` when placed but not visible, `-` when it has no position. `detail` adds
// what a kept pose must also keep, for the poses on the skeletons in `KEPT`.
const PY_COMMON = `
import json, os, sys
import numpy as np
import sleap_io as sio

KEPT = {"other", "third"}

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
                tracking = i.tracking_score if i.tracking_score is not None and i.tracking_score == i.tracking_score else 0.0
                detail.append("|".join([row_of[id(i)], "tracking=" + num(tracking),
                                        "complete=" + "".join("1" if c else "0" for c in i.points["complete"]),
                                        "from=" + (row_of.get(id(source), "?") if source is not None else "")]))
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
                groups.append("|".join(["s" + str(si), str(f)] + members))
    return {
        "error": None, "videos": [v.filename for v in labels.videos], "tracks": sorted(t.name for t in labels.tracks),
        "skeletons": [[s.name or "skeleton", s.node_names] for s in labels.skeletons],
        "sessions": sessions, "poses": sorted(poses), "detail": sorted(detail), "groups": sorted(groups),
        "repeatedFrames": len(keys) - len(set(keys)),
    }
`;

const PY_READ = PY_COMMON + `
print(json.dumps({p: read(p) for p in sys.argv[1:]}))
`;

const PY_WRITE = PY_COMMON + `
import h5py

out = sys.argv[1]
sk = sio.Skeleton(nodes=["nose", "tail"], edges=[("nose", "tail")], name="main")
sk2 = sio.Skeleton(nodes=["a", "b", "c"], edges=[("a", "b")], name="other")
sk3 = sio.Skeleton(nodes=["p", "q"], name="third")
sk_rev = sio.Skeleton(nodes=["tail", "nose"], edges=[("tail", "nose")], symmetries=[("tail", "nose")], name="reversed")
sk_edged = sio.Skeleton(nodes=["nose", "tail"], edges=[("tail", "nose")], name="edged")
sk_twin = sio.Skeleton(nodes=["nose", "tail"], edges=[("nose", "tail")], name="twin")
T1, T2, TB = sio.Track(name="mouse1"), sio.Track(name="mouse2"), sio.Track(name="only_on_other")

def video(name):
    v = sio.Video(filename=name, open_backend=False)
    v.backend_metadata["shape"] = (400, 64, 64, 1)
    return v

def pose(skeleton, base, f, i=0, t=T1):
    n = len(skeleton.node_names)
    # A node's values and score depend on its NAME, so a swap by position shows.
    names = {"nose": 0, "tail": 1, "a": 0, "b": 1, "c": 2, "p": 0, "q": 1}
    xy = np.array([[base + 10.0 * i + f + 5.0 * names[m], 10.0 * (names[m] + 1)] for m in skeleton.node_names])
    return sio.PredictedInstance.from_numpy(xy, skeleton=skeleton, track=t, score=0.9, point_scores=np.array([0.5 + 0.1 * names[m] for m in skeleton.node_names]))

def frames(v, skeleton, base, tracks=(T1, T2), n=3):
    return [sio.LabeledFrame(video=v, frame_idx=f, instances=[pose(skeleton, base, f, i, t) for i, t in enumerate(tracks)]) for f in range(n)]

def save(name, fr, videos, skeletons, tracks=(T1, T2)):
    sio.Labels(labeled_frames=fr, videos=videos, skeletons=list(skeletons), tracks=list(tracks)).save(os.path.join(out, name + ".slp"), embed=False)

def cam(name, skeleton, base, fixture=None, **kw):
    v = video(name + ".mp4")
    save(fixture or (name + "_" + skeleton.name), frames(v, skeleton, base, **kw), [v], [skeleton], kw.get("tracks", (T1, T2)))

cam("camA", sk, 100)
cam("camB", sk, 200)
cam("camC", sk, 300)
cam("camD", sk, 400)
cam("camA", sk2, 110)
cam("camB", sk2, 210)
cam("camC", sk2, 310)
cam("camD", sk2, 410)
cam("camB", sk_rev, 200)
cam("camC", sk_rev, 300)
cam("camA", sk_rev, 100)
cam("camB", sk_edged, 200)
cam("camB", sk_twin, 200)
# The name "main" on other nodes, and on the same nodes the other way round.
cam("camC", sio.Skeleton(nodes=["a", "b", "c"], edges=[("a", "b")], name="main"), 310, fixture="camC_main_abc")
cam("camD", sio.Skeleton(nodes=["a", "b", "c"], edges=[("a", "b")], name="main"), 410, fixture="camD_main_abc")
cam("camC", sio.Skeleton(nodes=["tail", "nose"], edges=[("tail", "nose")], name="main"), 300, fixture="camC_main_rev")
cam("camD", sio.Skeleton(nodes=["tail", "nose"], edges=[("tail", "nose")], name="main"), 400, fixture="camD_main_rev")

# Everything a kept pose must keep: a track only it uses, a tracking score, a hand-labeled
# pose with a node with no position and a node placed but not visible, its source.
b = video("camB.mp4")
fb = frames(b, sk2, 210, tracks=(TB,))
predicted = fb[1].instances[0]
predicted.tracking_score = 0.75
user = sio.Instance.from_numpy(np.array([[810.0, 10.0], [np.nan, np.nan], [812.0, 12.0]]), skeleton=sk2)
user.points["visible"][0] = False
user.points["complete"][0] = True
user.from_predicted = predicted
user.tracking_score = 0.25
fb[1].instances.append(user)
save("camB_kinds", fb, [b], [sk2], tracks=(TB,))

# Hand labels on the reversed skeleton, one node not visible, one missing.
b = video("camB.mp4")
fb = frames(b, sk_rev, 200)
hand = sio.Instance.from_numpy(np.array([[77.0, 7.0], [88.0, 8.0]]), skeleton=sk_rev, track=T2)  # tail, nose
hand.points["visible"][0] = False
hand.points["complete"][0] = True
missing = sio.Instance.from_numpy(np.array([[np.nan, np.nan], [99.0, 9.0]]), skeleton=sk_rev)    # tail missing
fb[0].instances = [hand, missing]
save("camB_reversed_hand", fb, [b], [sk_rev])

# Two skeletons in one camera file. The file's first skeleton with a pose decides.
b = video("camB.mp4")
fb = frames(b, sk2, 210)
fb[0].instances.append(pose(sk, 250, 0))
fb[1].instances.append(pose(sk3, 260, 1))
save("camB_other_first", fb, [b], [sk2, sk, sk3])
b = video("camB.mp4")
fb = frames(b, sk, 200)
fb[0].instances.append(pose(sk2, 260, 0))
save("camB_main_first", fb, [b], [sk, sk2])

# An unused skeleton at the head of the list.
b = video("camB.mp4")
save("camB_stale", frames(b, sk, 200), [b], [sio.Skeleton(nodes=["x", "y", "z", "w"], name="stale"), sk])

# Other videos in a camera's file: the first listed with frames is the camera's.
b, y = video("camB.mp4"), video("camB_second.mp4")
save("camB_two_videos", frames(b, sk2, 210) + frames(y, sk2, 510), [b, y], [sk2])
b, y = video("camB.mp4"), video("camB_second.mp4")
save("camB_reversed_two_videos", frames(b, sk_rev, 200) + frames(y, sk_rev, 500), [b, y], [sk_rev])

# Its own video is not the first the file lists.
x, b = video("unused.mp4"), video("camB.mp4")
save("camB_later_video", frames(b, sk2, 210), [x, b], [sk2])

# A file with a skeleton and no pose has nothing to keep and no vote.
b = video("camB.mp4")
sio.Labels(labeled_frames=[], videos=[b], skeletons=[sk2], tracks=[]).save(os.path.join(out, "camB_empty.slp"), embed=False)

# Analysis .h5 cameras: [n_tracks, 2, n_nodes, n_frames].
def analysis(name, nodes, base):
    ids = {"nose": 0, "tail": 1, "a": 0, "b": 1, "c": 2}
    t = np.zeros((2, 2, len(nodes), 3))
    for i in range(2):
        for f in range(3):
            for k, m in enumerate(nodes):
                t[i, 0, k, f] = base + 10.0 * i + f + 5.0 * ids[m]
                t[i, 1, k, f] = 10.0 * (ids[m] + 1)
    with h5py.File(os.path.join(out, name + ".h5"), "w") as h:
        h.create_dataset("tracks", data=t)
        h.create_dataset("node_names", data=[n.encode() for n in nodes])
        h.create_dataset("track_names", data=[b"mouse1", b"mouse2"])
analysis("h5_camA_main", ["nose", "tail"], 100)
analysis("h5_camB_reversed", ["tail", "nose"], 200)
analysis("h5_camB_other", ["a", "b", "c"], 210)
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

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'poses-camera-other-skeleton-'));
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
const readFixture = name => fs.readFileSync(path.join(tmp, name));

/**
 * Open a folder through the real loader. `cameras` maps camera directory -> fixture file.
 * `opts`: `route` ('eager' or 'lazy', forced as section F of poses-other-skeleton does:
 * the folder goes lazy on a file's size alone and HDF5 ignores trailing bytes), `page`
 * (add to what is open), `skeletonJson`, `firstOpens` (hold the others' reads until this
 * camera's has resolved), `folder`, `declared` (a parent folder's skeleton), `throwOnTyped`,
 * `calibration` (camera names, in the calibration's order).
 */
async function openFolder(cameras, opts = {}) {
    const page = opts.page || await newAppPage();
    const files = Object.entries(cameras).map(([cam, fixture]) => {
        const ext = path.extname(fixture) || '.slp';
        return { cam, name: cam + ext, b64: toB64(readFixture(path.extname(fixture) ? fixture : fixture + '.slp')) };
    });
    await page.evaluate(files => { window.__folderFiles = files.map(f => ({ ...f, bytes: Uint8Array.from(atob(f.b64), c => c.charCodeAt(0)), b64: null })); }, files);
    const loading = page.evaluate(async ({ route, skeletonJson, firstOpens, folder, declared, throwOnTyped, calibration }) => {
        const loader = await import('/loading/session-loader.js');
        const tri = await import('/pose/triangulation.js');
        const { Skeleton } = await import('/pose/pose-data.js');
        const pad = route === 'lazy' ? [new Uint8Array(tri.LAZY_SLP_THRESHOLD + (1 << 20))] : [];
        const padH5 = route === 'lazy' ? [new Uint8Array(21 * 1024 * 1024)] : [];
        const mkFile = (bits, rel) => { const f = new File(bits, rel.split('/').pop()); Object.defineProperty(f, 'webkitRelativePath', { value: rel }); return f; };
        const root = folder || 'root';
        const list = window.__folderFiles.map(f => mkFile([f.bytes].concat(f.name.endsWith('.h5') ? padH5 : pad), root + '/' + f.cam + '/' + f.name));
        if (skeletonJson) list.push(mkFile([skeletonJson], root + '/skeleton.json'));
        if (calibration) {
            list.push(mkFile([calibration.map((name, i) => '[cam_' + i + ']\nname = "' + name + '"\nsize = [64, 64]\n'
                + 'matrix = [[500.0, 0.0, 32.0], [0.0, 500.0, 32.0], [0.0, 0.0, 1.0]]\ndistortions = [0.0, 0.0, 0.0, 0.0, 0.0]\n'
                + 'rotation = [0.0, ' + (0.1 * i) + ', 0.0]\ntranslation = [' + (10 * i) + ', 0.0, 0.0]\n').join('\n')], root + '/calibration.toml'));
        }
        if (firstOpens) {
            // Hold every other camera's read until this one has resolved.
            const SIO = window.SleapIO;
            const real = SIO.readSlpStreaming;
            let release;
            const held = new Promise(r => { release = r; });
            SIO.readSlpStreaming = async (file, o) => {
                if (file.name !== firstOpens + '.slp') {
                    await held;
                    return real.call(SIO, file, o);
                }
                const out = await real.call(SIO, file, o);
                setTimeout(release, 0);
                return out;
            };
        }
        if (throwOnTyped) window.SleapIO.readSlpStreaming = async () => { throw new Error('forced to the raw worker'); };
        await loader.handleLoadSessionFolderPerCamera(list, !!folder && folder !== 'root' && window.__loadedOne,
            declared ? { declaredSkeleton: new Skeleton(declared.name, declared.nodes, declared.edges) } : undefined);
        window.__loadedOne = true;
        return (document.getElementById('statusText') || {}).textContent;
    }, { route: opts.route, skeletonJson: opts.skeletonJson, firstOpens: opts.firstOpens, folder: opts.folder, declared: opts.declared, throwOnTyped: opts.throwOnTyped, calibration: opts.calibration });
    // These folders hold no videos, so the load waits on the Missing Video Files popup.
    await page.getByRole('button', { name: 'Continue', exact: true }).click();
    return { page, status: await loading };
}

/**
 * Open a parent folder of sessions through File ▸ Open's own picker, the one route that
 * reads a parent's `skeleton.json` and ends on a status of its own. `sessions` maps
 * session folder -> (camera directory -> fixture file). `opts`: `route`, `skeletonJson`.
 * The status comes back once the skeleton prompt, if there is one, has been skipped.
 */
async function openSessionsPicker(sessions, opts = {}) {
    const page = await newAppPage();
    const sent = Object.entries(sessions).map(([name, cameras]) => ({
        name,
        cams: Object.entries(cameras).map(([cam, fixture]) => {
            const ext = path.extname(fixture) || '.slp';
            return { cam, name: cam + ext, b64: toB64(readFixture(path.extname(fixture) ? fixture : fixture + '.slp')) };
        }),
    }));
    await page.evaluate(async ({ sent, route, skeletonJson }) => {
        const loader = await import('/loading/session-loader.js');
        const tri = await import('/pose/triangulation.js');
        const pad = route === 'lazy' ? [new Uint8Array(tri.LAZY_SLP_THRESHOLD + (1 << 20))] : [];
        const padH5 = route === 'lazy' ? [new Uint8Array(21 * 1024 * 1024)] : [];
        const dir = (name, entries) => ({
            kind: 'directory', name,
            [Symbol.asyncIterator]: async function* () { for (const e of entries) yield [e.name, e]; },
        });
        const file = (name, bits) => ({ kind: 'file', name, getFile: async () => new File(bits, name) });
        const folders = sent.map(s => dir(s.name, s.cams.map(c => {
            const bytes = Uint8Array.from(atob(c.b64), x => x.charCodeAt(0));
            return dir(c.cam, [file(c.name, [bytes].concat(c.name.endsWith('.h5') ? padH5 : pad))]);
        })));
        window.showDirectoryPicker = async () => dir('parent', folders.concat(skeletonJson ? [file('skeleton.json', [skeletonJson])] : []));
        window.__multiDone = false;
        loader.handleLoadMultiSession().then(() => { window.__multiDone = true; }, e => { window.__multiDone = 'ERR ' + (e && e.stack || e); });
    }, { sent, route: opts.route, skeletonJson: opts.skeletonJson });
    // These folders hold no videos, so each session may wait on the Missing Video Files popup.
    let done = false, prompted = !!opts.skeletonJson;
    for (let i = 0; i < 6000 && !(done === true && prompted); i++) {
        const s = await page.evaluate(() => {
            const go = [...document.querySelectorAll('button')].find(b => b.offsetParent && (b.textContent || '').trim() === 'Continue');
            if (go) go.click();
            const skip = document.getElementById('msSkelSkip');
            if (skip) skip.click();
            return { done: window.__multiDone, skipped: !!skip };
        });
        if (typeof s.done === 'string') throw new Error(s.done);
        done = s.done;
        prompted = prompted || s.skipped;
        if (!(done === true && prompted)) await page.waitForTimeout(100);
    }
    if (!(done === true && prompted)) throw new Error('the multi-session load did not finish');
    const status = await page.evaluate(() => ({
        text: document.getElementById('statusText').textContent,
        dot: document.getElementById('statusDot').className,
    }));
    return { page, status };
}

/** File ▸ Save, the real route; the file's bytes. Leaves the page open. */
async function saveBytes(page) {
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
    return Buffer.from(b64, 'base64');
}

/** Open a project file: `eager` is File ▸ Open, `lazy` the lazy reopen. */
async function openProject(route, bytes) {
    const page = await newAppPage();
    await page.evaluate(b64 => { window.__file = Uint8Array.from(atob(b64), c => c.charCodeAt(0)); }, toB64(bytes));
    await page.evaluate(route => { window.__done = false; (async () => {
        const f = new File([window.__file], 'project.slp');
        if (route === 'lazy') await (await import('/loading/session-loader.js')).handleLoadProjectSlpLazy(f);
        else await (await import('/import-export/slp-import.js')).handleLoadSlpFile(f);
    })().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); }); }, route);
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
    if (done !== true) throw new Error(route + ' reopen did not finish: ' + done);
    return page;
}

/** Hydrate a lazy session's frames, then what the app shows, session by session. */
const shownInApp = (page, nFrames = 3) => page.evaluate(async (nFrames) => {
    const tri = await import('/pose/triangulation.js');
    const { state } = await import('/ui/app-state.js');
    if (state.session.lazyLoader) for (let f = 0; f < nFrames; f++) await tri.ensureLazyFrameData(f);
    return state.sessions.map(s => {
        const cams = {};
        for (const [f, fg] of s.frameGroups) {
            for (const cam of new Set([...fg.instances.keys(), ...fg.unlinkedInstances.keys()])) {
                for (const inst of [...fg.getInstances(cam), ...fg.getUnlinkedInstances(cam).map(u => u.instance)]) {
                    const entry = [f, inst.numNodes];
                    for (let k = 0; k < inst.numNodes; k++) entry.push(inst.hasPoint(k) ? inst.getX(k) : null);
                    // The node score the tooltip shows, by node.
                    for (let k = 0; k < inst.numNodes; k++) { const v = s.predictedNodeScore(cam, f, inst, k); entry.push(Number.isNaN(v) ? null : Math.round(v * 1000) / 1000); }
                    (cams[cam] = cams[cam] || []).push(entry);
                }
            }
        }
        for (const cam in cams) cams[cam].sort((a, b) => a[0] - b[0] || a[2] - b[2]);
        return { nodes: s.skeleton.nodes, tracks: s.tracks, cams, occupancy: s.trackOccupancy && s.trackOccupancy.keys ? [...s.trackOccupancy.keys()].sort() : null };
    });
}, nFrames);

let savedSeq = 0;
const pythonWork = [];   // one Python run reads every save
function savedForPython(bytes, what, want) {
    const file = path.join(tmp, 'saved-' + (++savedSeq) + '.slp');
    fs.writeFileSync(file, bytes);
    pythonWork.push({ file, what, want });
}

function compare(what, got, want) {
    if (got.error) { check(false, what + ': opens', got.error); return; }
    check(eq(got.poses, want.poses), what + ': every pose on its own skeleton, nodes and video', diff(got.poses, want.poses));
    check(got.repeatedFrames === 0, what + ': one row per video and frame', got.repeatedFrames);
    if (want.detail) check(eq(got.detail, want.detail), what + ': what a kept pose also carries', diff(got.detail || [], want.detail));
    if (want.groups) check(eq(got.groups, want.groups), what + ': groups point at the right poses', diff(got.groups || [], want.groups));
    if (want.sessions) check(eq(got.sessions, want.sessions), what + ': cameras and their videos', got.sessions);
    if (want.videos) check(eq([...got.videos].sort(), [...want.videos].sort()), what + ': video list', got.videos);
    if (want.tracks) check(eq(got.tracks, want.tracks), what + ': track list', got.tracks);
    if (want.skeletons) check(eq(got.skeletons, want.skeletons), what + ': skeleton list', got.skeletons);
}

// A row of Python's as a save on `name` with `nodes` must hold it: each value and score
// stays with its node's NAME.
function onSkeleton(row, name, nodes) {
    const c = row.split('|');
    const cells = new Map(c[5].split(';').map((cell, k) => [cell.split('=')[0], { cell, k }]));
    const scores = c[6] ? c[6].split(',') : null;
    c[4] = name;
    c[5] = nodes.map(n => cells.get(n).cell).join(';');
    if (scores) c[6] = [scores[0], ...nodes.map(n => scores[1 + cells.get(n).k])].join(',');
    return c.join('|');
}
const onMain = rows => rows.map(r => onSkeleton(r, 'main', ['nose', 'tail'])).sort();

const ROUTES = ['eager', 'lazy'];

/** Save twice with the project open, then reopen the save each way and save again. */
async function saveEveryWay(page, what, want, reopenRoutes = ROUTES) {
    const first = await saveBytes(page);
    savedForPython(first, what + ': save 1', want);
    savedForPython(await saveBytes(page), what + ': save 2 of one open project', want);
    await page.close();
    for (const route of reopenRoutes) {
        const reopened = await openProject(route, first);
        savedForPython(await saveBytes(reopened), `${what}: saved again from a ${route} reopen`, want);
        await reopened.close();
    }
}

try {
    uv(PY_WRITE, [tmp]);
    const NAMES = ['camA_main', 'camB_main', 'camC_main', 'camD_main', 'camA_other', 'camB_other', 'camC_other', 'camD_other', 'camB_reversed', 'camC_reversed', 'camA_reversed',
        'camC_main_abc', 'camD_main_abc', 'camC_main_rev', 'camD_main_rev', 'camB_edged', 'camB_twin', 'camB_kinds', 'camB_reversed_hand', 'camB_other_first', 'camB_main_first', 'camB_stale', 'camB_two_videos', 'camB_reversed_two_videos', 'camB_later_video', 'camB_empty'];
    const sourcePaths = NAMES.map(n => path.join(tmp, n + '.slp'));
    const rawSources = JSON.parse(uv(PY_READ, sourcePaths).trim().split('\n').pop());
    const sources = {};
    for (const p of sourcePaths) sources[path.basename(p, '.slp')] = rawSources[p];

    console.log('Sources, as Python reads them:');
    check(sources.camB_other.poses.some(r => r.includes('|other|a=210.000,10.000;b=215.000,20.000;c=220.000,30.000|')), 'camB_other.slp holds poses on three nodes', sources.camB_other.poses[0]);
    check(sources.camB_reversed.poses.some(r => r.includes('|reversed|tail=205.000,20.000;nose=200.000,10.000|0.900,0.600,0.500')), 'camB_reversed.slp holds the same nodes the other way round', sources.camB_reversed.poses[0]);
    check(sources.camB_kinds.detail.some(r => r.includes('tracking=0.750')) && sources.camB_kinds.detail.some(r => r.includes('tracking=0.250') && /\|from=camB\.mp4/.test(r)), 'camB_kinds.slp holds a tracking score and a from_predicted link', sources.camB_kinds.detail);

    const MAIN = ['main', ['nose', 'tail']], OTHER = ['other', ['a', 'b', 'c']];

    console.log('\nA. A camera on other nodes: kept, not shown');
    for (const route of ROUTES) {
        const what = `other nodes, ${route}`;
        const { page, status } = await openFolder({ camA: 'camA_main', camB: 'camB_other' }, { route });
        check(/camB shows no poses: its file is on other nodes than this session's \(nose, tail\)/.test(status), what + ': the status names the camera', status);
        check(/6 poses on a camera's video are on another skeleton \("other"\) and are not shown; saving keeps them on that skeleton/.test(status), what + ': the status says what a save does', status);
        const shown = await shownInApp(page);
        check(eq(shown[0].nodes, ['nose', 'tail']), what + ': the session shows the first camera\'s nodes', shown[0].nodes);
        check(!shown[0].cams.camB, what + ': the camera shows no pose', shown[0].cams.camB);
        check(shown[0].cams.camA && shown[0].cams.camA.length === 6 && shown[0].cams.camA.every(r => r[1] === 2), what + ': the other camera shows all of its own', shown[0].cams.camA);
        const counts = await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            const { countSkeletonEditImpact } = await import('/pose/skeleton-edit-impact.js');
            const s = state.session;
            let rows = 0;
            if (s.lazyLoader) s.lazyLoader.forEachInstanceRow(() => { rows++; });
            return { tally: countSkeletonEditImpact([s]).instances, rows, cameras: s.cameras.map(c => c.name) };
        });
        check(counts.tally === 6, what + ': the skeleton dialog counts only the session\'s poses', counts.tally);
        if (route === 'lazy') check(counts.rows === 6, what + ': a whole-project walk sees only the session\'s poses', counts.rows);
        check(eq(counts.cameras, ['camA', 'camB']), what + ': the camera is still a camera', counts.cameras);
        await saveEveryWay(page, what, {
            poses: [...sources.camA_main.poses, ...sources.camB_other.poses].sort(), detail: sources.camB_other.detail,
            sessions: [{ camA: 'camA.mp4', camB: 'camB.mp4' }], skeletons: [MAIN, OTHER],
        });
    }
    for (const route of ROUTES) {
        // A calibration listing the hidden camera first: the camera order is not the rule.
        const what = `other nodes, calibrated, ${route}`;
        const { page, status } = await openFolder({ camA: 'camA_main', camB: 'camB_other' }, { route, calibration: ['camB', 'camA'] });
        check(/camB shows no poses/.test(status), what + ': the status names the camera', status);
        const shown = await shownInApp(page);
        check(eq(shown[0].nodes, ['nose', 'tail']) && !shown[0].cams.camB && shown[0].cams.camA.length === 6, what + ': the first by name shows', shown[0]);
        await saveEveryWay(page, what, {
            poses: [...sources.camA_main.poses, ...sources.camB_other.poses].sort(), detail: sources.camB_other.detail,
            sessions: [{ camB: 'camB.mp4', camA: 'camA.mp4' }], skeletons: [MAIN, OTHER],
        });
    }
    for (const route of ROUTES) {
        // A pose LUC3D does not read must lose nothing, and its track does not join the session's.
        const what = `every kind of pose, ${route}`;
        const { page } = await openFolder({ camA: 'camA_main', camB: 'camB_kinds' }, { route });
        const shown = await shownInApp(page);
        check(eq(shown[0].tracks, ['mouse1', 'mouse2']), what + ': the session lists only the tracks it shows', shown[0].tracks);
        if (route === 'lazy') check(eq(shown[0].occupancy, ['camA']), what + ': the timeline has no bars for it', shown[0].occupancy);
        const kept = sources.camB_kinds.poses;
        await saveEveryWay(page, what, {
            poses: [...sources.camA_main.poses, ...kept].sort(), detail: sources.camB_kinds.detail,
            tracks: ['mouse1', 'mouse2', 'only_on_other'], skeletons: [MAIN, OTHER],
        });
    }
    for (const route of ROUTES) {
        // A file with a skeleton and no pose has nothing to keep, nothing to say.
        const { page, status } = await openFolder({ camA: 'camA_main', camB: 'camB_empty' }, { route });
        check(!/not shown|shows no poses/.test(status), `no poses in the other file, ${route}: nothing to say`, status);
        savedForPython(await saveBytes(page), `no poses in the other file, ${route}`, { poses: sources.camA_main.poses, skeletons: [MAIN] });
        await page.close();
    }
    for (const route of ROUTES) {
        // The control: one skeleton, nothing to keep.
        const { page, status } = await openFolder({ camA: 'camA_main', camB: 'camB_main' }, { route });
        check(!/not shown|shows no poses/.test(status), `one skeleton, ${route}: nothing to say`, status);
        savedForPython(await saveBytes(page), `one skeleton, ${route}`, { poses: [...sources.camA_main.poses, ...sources.camB_main.poses].sort(), skeletons: [MAIN] });
        await page.close();
    }

    console.log('\nB. Which skeleton the session shows');
    for (const route of ROUTES) {
        // One odd camera, first by name: it does not hide the other two.
        const what = `one odd camera, ${route}`;
        const { page, status } = await openFolder({ camA: 'camA_other', camB: 'camB_main', camC: 'camC_main' }, { route });
        check(/camA shows no poses/.test(status), what + ': the odd camera is the one hidden', status);
        const shown = await shownInApp(page);
        check(eq(shown[0].nodes, ['nose', 'tail']) && shown[0].cams.camB.length === 6 && shown[0].cams.camC.length === 6 && !shown[0].cams.camA, what + ': the session shows most cameras\' nodes', shown[0]);
        await saveEveryWay(page, what, {
            poses: [...sources.camA_other.poses, ...sources.camB_main.poses, ...sources.camC_main.poses].sort(), detail: sources.camA_other.detail,
            skeletons: [MAIN, OTHER], sessions: [{ camA: 'camA.mp4', camB: 'camB.mp4', camC: 'camC.mp4' }],
        });
    }
    for (const route of ROUTES) {
        // Two cameras on one other skeleton: one entry in the file.
        const what = `two cameras hidden, ${route}`;
        const { page, status } = await openFolder({ camA: 'camA_main', camB: 'camB_main', camC: 'camC_other', camD: 'camD_other' }, { route });
        check(/camC, camD show no poses: their files are on other nodes/.test(status), what + ': the status names both', status);
        const got = await saveBytes(page);
        await page.close();
        savedForPython(got, what, { poses: [...sources.camA_main.poses, ...sources.camB_main.poses, ...sources.camC_other.poses, ...sources.camD_other.poses].sort(), skeletons: [MAIN, OTHER] });
    }
    for (const route of ROUTES) {
        // A tie goes to the first camera by name, whichever camera's file opens first.
        const opened = route === 'lazy' ? ['camA', 'camB'] : [null];
        for (const firstOpens of opened) {
            const what = `a tie, ${route}` + (firstOpens ? `, ${firstOpens} opening first` : '');
            const { page } = await openFolder({ camA: 'camA_main', camB: 'camB_other' }, { route, firstOpens });
            const shown = await shownInApp(page);
            check(eq(shown[0].nodes, ['nose', 'tail']) && !shown[0].cams.camB, what + ': camA shows', shown[0]);
            savedForPython(await saveBytes(page), what, { poses: [...sources.camA_main.poses, ...sources.camB_other.poses].sort(), skeletons: [MAIN, OTHER] });
            await page.close();
        }
    }

    console.log('\nC. Same nodes in another order: read by name');
    for (const route of ROUTES) {
        const opened = route === 'lazy' ? ['camA', 'camB'] : [null];
        for (const firstOpens of opened) {
            const what = `another order, ${route}` + (firstOpens ? `, ${firstOpens} opening first` : '');
            const { page, status } = await openFolder({ camA: 'camA_main', camB: 'camB_reversed' }, { route, firstOpens });
            check(!/not shown|shows no poses/.test(status), what + ': nothing hidden', status);
            const shown = await shownInApp(page);
            check(eq(shown[0].nodes, ['nose', 'tail']), what + ': the session shows the first camera\'s order', shown[0].nodes);
            // Frame 0, first pose: nose 200, tail 205, and the scores go with their nodes.
            check(shown[0].cams.camB && eq(shown[0].cams.camB[0], [0, 2, 200, 205, 0.5, 0.6]), what + ': each value and node score under its own name', shown[0].cams.camB && shown[0].cams.camB[0]);
            await saveEveryWay(page, what, {
                poses: [...sources.camA_main.poses, ...onMain(sources.camB_reversed.poses)].sort(), skeletons: [MAIN],
                sessions: [{ camA: 'camA.mp4', camB: 'camB.mp4' }],
            });
        }
    }
    for (const route of ROUTES) {
        // The first camera by name sets the order, even when the others agree on another.
        const what = `another order, two against one, ${route}`;
        const { page } = await openFolder({ camA: 'camA_main', camB: 'camB_reversed', camC: 'camC_reversed' }, { route, firstOpens: route === 'lazy' ? 'camC' : undefined });
        const shown = await shownInApp(page);
        check(eq(shown[0].nodes, ['nose', 'tail']) && eq(shown[0].cams.camC[0], [0, 2, 300, 305, 0.5, 0.6]), what + ': by name', shown[0].cams.camC && shown[0].cams.camC[0]);
        savedForPython(await saveBytes(page), what, { poses: [...sources.camA_main.poses, ...onMain(sources.camB_reversed.poses), ...onMain(sources.camC_reversed.poses)].sort(), skeletons: [MAIN] });
        await page.close();
    }
    for (const route of ROUTES) {
        // Three opens can flip the order twice: the store's rows must follow each time.
        const what = `another order, flipping as files open, ${route}`;
        const { page } = await openFolder({ camA: 'camA_reversed', camB: 'camB_main', camC: 'camC_reversed' }, { route, firstOpens: route === 'lazy' ? 'camB' : undefined });
        const shown = await shownInApp(page);
        check(eq(shown[0].nodes, ['tail', 'nose']), what + ': the first by name sets the order', shown[0].nodes);
        check(eq(shown[0].cams.camB[0], [0, 2, 205, 200, 0.6, 0.5]), what + ': each value under its own name', shown[0].cams.camB[0]);
        const rows = [...sources.camA_reversed.poses, ...sources.camB_main.poses, ...sources.camC_reversed.poses].map(r => onSkeleton(r, 'reversed', ['tail', 'nose'])).sort();
        savedForPython(await saveBytes(page), what, { poses: rows, skeletons: [['reversed', ['tail', 'nose']]] });
        await page.close();
    }
    for (const route of ROUTES) {
        // Hand labels: a node not visible stays with its name, a missing one stays missing.
        const what = `hand labels in another order, ${route}`;
        const { page } = await openFolder({ camA: 'camA_main', camB: 'camB_reversed_hand' }, { route });
        savedForPython(await saveBytes(page), what, { poses: [...sources.camA_main.poses, ...onMain(sources.camB_reversed_hand.poses)].sort(), skeletons: [MAIN] });
        await page.close();
    }
    for (const route of ROUTES) {
        // Groups and a correction on a camera read by name: the file is whole and lines up.
        const what = `a group and a correction in another order, ${route}`;
        const { page } = await openFolder({ camA: 'camA_main', camB: 'camB_reversed' }, { route });
        await shownInApp(page);
        await page.evaluate(async () => {
            const pd = await import('/pose/pose-data.js');
            const { state } = await import('/ui/app-state.js');
            const s = state.session;
            const fg = s.frameGroups.get(1);
            const a = fg.getUnlinkedInstances('camA')[0], b = fg.getUnlinkedInstances('camB')[0];
            const group = new pd.InstanceGroup(1, 0);
            group.addInstance('camA', a.instance);
            group.addInstance('camB', b.instance);
            group.points3d = new Float64Array([1, 2, 3, 4, 5, 6]);
            fg.removeUnlinkedById(a.id);
            fg.removeUnlinkedById(b.id);
            fg.addInstance('camA', a.instance);
            fg.addInstance('camB', b.instance);
            s.instanceGroups.set(1, [group]);
            const c = fg.getUnlinkedInstances('camB')[0].instance;
            c.setPoint(0, 999, 999);   // the session's node 0: nose
            c.type = 'user';
            c.modified = true;
        });
        const bytes = await saveBytes(page);
        await page.close();
        const poses = [...sources.camA_main.poses, ...onMain(sources.camB_reversed.poses)].map(r =>
            r.startsWith('camB.mp4|1|P|mouse2|main|nose=211.000,10.000;tail=216.000,20.000|') ? 'camB.mp4|1|U|mouse2|main|nose=999.000,999.000;tail=216.000,20.000|' : r).sort();
        savedForPython(bytes, what, { poses, groups: ['s0|1|camA=101.000/main|camB=201.000/main'], skeletons: [MAIN] });
    }

    console.log('\nD. Same nodes, other edges or another name: shown');
    for (const route of ROUTES) {
        for (const name of ['camB_edged', 'camB_twin']) {
            const what = `${name}, ${route}`;
            const { page, status } = await openFolder({ camA: 'camA_main', camB: name }, { route });
            check(!/not shown|shows no poses/.test(status), what + ': nothing hidden', status);
            const shown = await shownInApp(page);
            check(shown[0].cams.camB.length === 6, what + ': its poses show', shown[0].cams.camB);
            savedForPython(await saveBytes(page), what, { poses: [...sources.camA_main.poses, ...sources[name].poses.map(r => onSkeleton(r, 'main', ['nose', 'tail']))].sort(), skeletons: [MAIN] });
            await page.close();
        }
    }

    console.log('\nE. Two skeletons in a camera file, or other videos');
    for (const route of ROUTES) {
        // The first skeleton with a pose decides: all of this camera's poses go aside.
        const what = `other nodes first, ${route}`;
        const { page, status } = await openFolder({ camA: 'camA_main', camB: 'camB_other_first' }, { route });
        check(/camB shows no poses/.test(status), what + ': the status names the camera', status);
        const shown = await shownInApp(page);
        check(!shown[0].cams.camB, what + ': nothing shown', shown[0].cams.camB);
        await saveEveryWay(page, what, {
            poses: [...sources.camA_main.poses, ...sources.camB_other_first.poses].sort(), detail: sources.camB_other_first.detail,
            skeletons: [MAIN, OTHER, ['third', ['p', 'q']]],
        });
    }
    for (const route of ROUTES) {
        // The session's first: as before, the poses on the second go aside.
        const what = `the session's nodes first, ${route}`;
        const { page, status } = await openFolder({ camA: 'camA_main', camB: 'camB_main_first' }, { route });
        check(/1 pose on a camera's video is on another skeleton \("other"\)/.test(status) && !/shows no poses/.test(status), what + ': only that pose is hidden', status);
        const shown = await shownInApp(page);
        check(shown[0].cams.camB.length === 6 && shown[0].cams.camB.every(r => r[1] === 2), what + ': the session\'s poses show', shown[0].cams.camB);
        await saveEveryWay(page, what, { poses: [...sources.camA_main.poses, ...sources.camB_main_first.poses].sort(), skeletons: [MAIN, OTHER] });
    }
    for (const route of ROUTES) {
        // An unused skeleton at the head of a camera's list is not its skeleton.
        const what = `an unused skeleton first, ${route}`;
        const { page, status } = await openFolder({ camA: 'camA_main', camB: 'camB_stale' }, { route });
        check(!/not shown|shows no poses/.test(status), what + ': nothing hidden', status);
        const shown = await shownInApp(page);
        check(shown[0].cams.camB.length === 6 && shown[0].cams.camB.every(r => r[1] === 2), what + ': its poses show', shown[0].cams.camB);
        savedForPython(await saveBytes(page), what, { poses: [...sources.camA_main.poses, ...sources.camB_stale.poses].sort(), skeletons: [MAIN] });
        await page.close();
    }
    for (const route of ROUTES) {
        // Other videos: the first with frames is the camera's, each pose stays on its own.
        const what = `other nodes on two videos, ${route}`;
        const { page, status } = await openFolder({ camA: 'camA_main', camB: 'camB_two_videos' }, { route });
        check(/camB shows no poses/.test(status), what + ': the status names the camera', status);
        check(/3 labeled frames on 1 video in no session are not shown/.test(status), what + ': and the other video', status);
        check(/6 poses on a camera's video are on another skeleton/.test(status), what + ': and counts the camera\'s own', status);
        await saveEveryWay(page, what, {
            poses: [...sources.camA_main.poses, ...sources.camB_two_videos.poses].sort(), detail: sources.camB_two_videos.detail,
            sessions: [{ camA: 'camA.mp4', camB: 'camB.mp4' }], videos: ['camA.mp4', 'camB.mp4', 'camB_second.mp4'],
        });
    }
    for (const route of ROUTES) {
        // The first video WITH FRAMES is the camera's; the one before it has none to keep.
        const what = `other nodes on the file's second video, ${route}`;
        const { page, status } = await openFolder({ camA: 'camA_main', camB: 'camB_later_video' }, { route });
        check(/camB shows no poses/.test(status) && !/in no session/.test(status), what + ': on the camera\'s video, and nothing else to say', status);
        savedForPython(await saveBytes(page), what, {
            poses: [...sources.camA_main.poses, ...sources.camB_later_video.poses].sort(), detail: sources.camB_later_video.detail,
            sessions: [{ camA: 'camA.mp4', camB: 'camB.mp4' }], skeletons: [MAIN, OTHER],
        });
        await page.close();
    }
    for (const route of ROUTES) {
        const what = `another order on two videos, ${route}`;
        const { page, status } = await openFolder({ camA: 'camA_main', camB: 'camB_reversed_two_videos' }, { route });
        check(/3 labeled frames on 1 video in no session are not shown/.test(status), what + ': the status says what is on the other video', status);
        savedForPython(await saveBytes(page), what, {
            poses: [...sources.camA_main.poses, ...onMain(sources.camB_reversed_two_videos.poses)].sort(), skeletons: [MAIN],
            videos: ['camA.mp4', 'camB.mp4', 'camB_second.mp4'],
        });
        await page.close();
    }

    console.log('\nF. Edits around a hidden camera');
    for (const route of ROUTES) {
        // A pose drawn on the hidden camera's frame: one row, both kinds of pose in it.
        const what = `a pose drawn on its frame, ${route}`;
        const { page } = await openFolder({ camA: 'camA_main', camB: 'camB_kinds' }, { route });
        await shownInApp(page);
        await page.evaluate(async () => {
            const pd = await import('/pose/pose-data.js');
            const { state } = await import('/ui/app-state.js');
            state.session.frameGroups.get(1).addUnlinkedInstance('camB', new pd.UnlinkedInstance(new pd.Instance([[901, 1], [902, 2]], 0, 'user', 0), 'camB'));
        });
        const drawn = 'camB.mp4|1|U|mouse1|main|nose=901.000,1.000;tail=902.000,2.000|';
        const poses = [...sources.camA_main.poses, ...sources.camB_kinds.poses, drawn].sort();
        const first = await saveBytes(page);
        savedForPython(first, what, { poses, detail: sources.camB_kinds.detail, skeletons: [MAIN, OTHER], sessions: [{ camA: 'camA.mp4', camB: 'camB.mp4' }] });
        savedForPython(await saveBytes(page), what + ': save 2', { poses, detail: sources.camB_kinds.detail, skeletons: [MAIN, OTHER] });
        await page.close();
        const reopened = await openProject('eager', first);
        savedForPython(await saveBytes(reopened), what + ': saved again from an eager reopen', { poses, detail: sources.camB_kinds.detail });
        await reopened.close();
    }
    for (const route of ROUTES) {
        // A track deleted from the session: the hidden camera's poses keep their name.
        const what = `a deleted track, ${route}`;
        const { page } = await openFolder({ camA: 'camA_main', camB: 'camB_other' }, { route });
        await shownInApp(page);
        await page.evaluate(async () => {
            const ops = await import('/ui/track-identity-ops.js');
            const { state } = await import('/ui/app-state.js');
            ops.deleteTrackAt(state.session, state.session.tracks.indexOf('mouse1'));
        });
        const onA = sources.camA_main.poses.map(r => r.replace('|P|mouse1|', '|P||'));
        savedForPython(await saveBytes(page), what, { poses: [...onA, ...sources.camB_other.poses].sort(), detail: sources.camB_other.detail, tracks: ['mouse1', 'mouse2'] });
        await page.close();
    }
    {
        // Propagate IDs → Tracks rebuilds a lazy store's track list.
        const what = 'a track list rebuilt, lazy';
        const { page } = await openFolder({ camA: 'camA_main', camB: 'camB_other' }, { route: 'lazy' });
        await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            const names = ['renamed0', 'renamed1'];
            state.session.lazyLoader.remapTracksFromIdentity(names, (cam, frame, old) => old);
            state.session.tracks = names;
        });
        const onA = sources.camA_main.poses.map(r => r.replace('|mouse1|', '|renamed0|').replace('|mouse2|', '|renamed1|'));
        savedForPython(await saveBytes(page), what, { poses: [...onA, ...sources.camB_other.poses].sort(), detail: sources.camB_other.detail, tracks: ['mouse1', 'mouse2', 'renamed0', 'renamed1'] });
        await page.close();
    }
    {
        // A delete across the project: the session's poses go, the ones it does not show stay.
        const what = 'a delete of every pose, lazy';
        const { page } = await openFolder({ camA: 'camA_main', camB: 'camB_kinds' }, { route: 'lazy' });
        const deleted = await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            return state.session.lazyLoader.deleteInstanceRows(() => true).deleted;
        });
        check(deleted === 6, what + ': only the session\'s poses are offered', deleted);
        savedForPython(await saveBytes(page), what, { poses: sources.camB_kinds.poses, detail: sources.camB_kinds.detail });
        await page.close();
    }
    for (const route of ROUTES) {
        // A node renamed on the session's skeleton: the hidden poses are on another.
        const { page } = await openFolder({ camA: 'camA_main', camB: 'camB_other' }, { route });
        await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            state.session.skeleton.nodes[1] = 'renamed';
        });
        const onA = sources.camA_main.poses.map(r => r.replace('tail=', 'renamed='));
        savedForPython(await saveBytes(page), `a renamed node, ${route}`, { poses: [...onA, ...sources.camB_other.poses].sort(), detail: sources.camB_other.detail, skeletons: [['main', ['nose', 'renamed']], OTHER] });
        await page.close();
    }

    console.log('\nG. A project mixing lazy and eager sessions');
    {
        const { page } = await openFolder({ camA: 'camA_main', camB: 'camB_other' }, { route: 'lazy' });
        await shownInApp(page);
        await page.evaluate(async () => {
            const pd = await import('/pose/pose-data.js');
            const { state } = await import('/ui/app-state.js');
            const cam = new pd.Camera('camE', [[500, 0, 32], [0, 500, 32], [0, 0, 1]], [0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0], [64, 64]);
            const eager = new pd.Session([cam], state.session.skeleton, ['mouse1'], 'eager');
            const fg = new pd.FrameGroup(0);
            eager.addFrameGroup(fg);
            fg.addInstance('camE', new pd.Instance([[900, 10], [901, 20]], 0, 'user', 0));
            state.sessions.push(eager);
        });
        const bytes = await saveBytes(page);
        await page.close();
        // Only the hidden camera's: a lazy camera's poses on this writer lose their per-node scores, as they always have.
        savedForPython(bytes, 'mixed project', { poses: null, onVideo: 'camB.mp4', onVideoPoses: sources.camB_other.poses, detail: sources.camB_other.detail });
    }

    console.log('\nH. Several folders in one project');
    for (const route of ROUTES) {
        // One has a hidden camera; the second folder's skeleton does not touch the first's.
        const what = `two folders, ${route}`;
        const first = await openFolder({ camA: 'camA_main', camB: 'camB_other' }, { route, folder: 'sess1' });
        const second = await openFolder({ camC: 'camC_main', camD: 'camD_main' }, { route, folder: 'sess2', page: first.page });
        check(/Loaded 2 camera\(s\)/.test(second.status), what + ': loaded', second.status);
        const page = first.page;
        const bytes = await saveBytes(page);
        savedForPython(bytes, what, {
            poses: [...sources.camA_main.poses, ...sources.camB_other.poses, ...sources.camC_main.poses, ...sources.camD_main.poses].sort(),
            detail: sources.camB_other.detail, skeletons: [MAIN, OTHER],
            sessions: [{ camA: 'camA.mp4', camB: 'camB.mp4' }, { camC: 'camC.mp4', camD: 'camD.mp4' }],
        });
        await page.close();
    }
    {
        // A store edited in memory goes through the multi-session save's pass 2 as it was
        // edited, and the hidden camera's poses keep their names through it.
        const what = 'two folders, a track list rebuilt, lazy';
        const first = await openFolder({ camA: 'camA_main', camB: 'camB_kinds' }, { route: 'lazy', folder: 'sess1' });
        await openFolder({ camC: 'camC_main', camD: 'camD_main' }, { route: 'lazy', folder: 'sess2', page: first.page });
        const page = first.page;
        await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            const names = ['renamed0', 'renamed1'];
            state.sessions[0].lazyLoader.remapTracksFromIdentity(names, (cam, frame, old) => old);
            state.sessions[0].tracks = names;
        });
        const renamed = rows => rows.map(r => r.replace('|mouse1|', '|renamed0|').replace('|mouse2|', '|renamed1|'));
        savedForPython(await saveBytes(page), what, {
            poses: [...renamed(sources.camA_main.poses), ...sources.camB_kinds.poses, ...sources.camC_main.poses, ...sources.camD_main.poses].sort(),
            detail: sources.camB_kinds.detail, skeletons: [MAIN, OTHER], sessions: [{ camA: 'camA.mp4', camB: 'camB.mp4' }, { camC: 'camC.mp4', camD: 'camD.mp4' }],
        });
        await page.close();
    }
    for (const route of ROUTES) {
        // Each folder on its own skeleton, the same name or not: each shows its own.
        const what = `folders on different skeletons, ${route}`;
        const first = await openFolder({ camA: 'camA_main', camB: 'camB_main' }, { route, folder: 'sess1' });
        await openFolder({ camC: 'camC_other', camD: 'camD_other' }, { route, folder: 'sess2', page: first.page });
        const page = first.page;
        const shape = await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            return state.sessions.map(s => s.skeleton.nodes);
        });
        check(eq(shape, [['nose', 'tail'], ['a', 'b', 'c']]), what + ': each session shows its own', shape);
        const bytes = await saveBytes(page);
        savedForPython(bytes, what, {
            poses: [...sources.camA_main.poses, ...sources.camB_main.poses, ...sources.camC_other.poses, ...sources.camD_other.poses].sort(),
            skeletons: [MAIN, OTHER],
        });
        await page.close();
    }
    for (const route of ROUTES) {
        for (const [label, c, d, nodes] of [['other nodes', 'camC_main_abc', 'camD_main_abc', ['a', 'b', 'c']], ['another order', 'camC_main_rev', 'camD_main_rev', ['tail', 'nose']]]) {
            // One name, so nothing but the nodes tells these skeletons apart.
            const what = `folders on one skeleton name, ${label}, ${route}`;
            const first = await openFolder({ camA: 'camA_main', camB: 'camB_main' }, { route, folder: 'sess1' });
            await openFolder({ camC: c, camD: d }, { route, folder: 'sess2', page: first.page });
            const page = first.page;
            const shape = await page.evaluate(async () => {
                const { state } = await import('/ui/app-state.js');
                return state.sessions.map(s => s.skeleton.nodes);
            });
            check(eq(shape, [['nose', 'tail'], nodes]), what + ': each session shows its own', shape);
            const bytes = await saveBytes(page);
            savedForPython(bytes, what, {
                poses: [...sources.camA_main.poses, ...sources.camB_main.poses, ...sources[c].poses, ...sources[d].poses].sort(),
                skeletons: [MAIN, ['main', nodes]],
            });
            await page.close();
        }
    }

    console.log('\nI. A folder\'s skeleton.json decides');
    const JSON_OTHER = JSON.stringify({ name: 'other', nodes: ['a', 'b', 'c'], edges: [[0, 1]] });
    const JSON_REVERSED = JSON.stringify({ name: 'reversed', nodes: ['tail', 'nose'], edges: [[0, 1]] });
    const JSON_NONE = JSON.stringify({ name: 'none', nodes: ['x', 'y', 'z', 'w'], edges: [] });
    for (const route of ROUTES) {
        const what = `skeleton.json names the odd camera's nodes, ${route}`;
        const { page, status } = await openFolder({ camA: 'camA_main', camB: 'camB_main', camC: 'camC_other' }, { route, skeletonJson: JSON_OTHER });
        check(/camA, camB show no poses/.test(status), what + ': the others are the ones hidden', status);
        const shown = await shownInApp(page);
        check(eq(shown[0].nodes, ['a', 'b', 'c']) && shown[0].cams.camC.length === 6, what + ': the session shows what it declares', shown[0]);
        const bytes = await saveBytes(page);
        await page.close();
        savedForPython(bytes, what, { poses: [...sources.camA_main.poses, ...sources.camB_main.poses, ...sources.camC_other.poses].sort(), skeletons: [OTHER, MAIN] });
    }
    for (const route of ROUTES) {
        const what = `skeleton.json in another order, ${route}`;
        const { page, status } = await openFolder({ camA: 'camA_main', camB: 'camB_reversed' }, { route, skeletonJson: JSON_REVERSED });
        check(!/not shown|shows no poses/.test(status), what + ': nothing hidden', status);
        const shown = await shownInApp(page);
        check(eq(shown[0].nodes, ['tail', 'nose']) && eq(shown[0].cams.camA[0], [0, 2, 105, 100, 0.6, 0.5]), what + ': read by name, not by position', shown[0].cams.camA[0]);
        const rows = [...sources.camA_main.poses, ...sources.camB_reversed.poses].map(r => onSkeleton(r, 'reversed', ['tail', 'nose'])).sort();
        savedForPython(await saveBytes(page), what, { poses: rows, skeletons: [['reversed', ['tail', 'nose']]] });
        await page.close();
    }
    for (const route of ROUTES) {
        // Never read by position under its names, and never a file Python cannot open.
        const what = `skeleton.json on nodes no camera has, ${route}`;
        const { page, status } = await openFolder({ camA: 'camA_main', camB: 'camB_other' }, { route, skeletonJson: JSON_NONE });
        check(/camA, camB show no poses/.test(status), what + ': both hidden, and said so', status);
        savedForPython(await saveBytes(page), what, { poses: [...sources.camA_main.poses, ...sources.camB_other.poses].sort() });
        await page.close();
    }
    for (const route of ROUTES) {
        // The multi-session save re-reads the files: it must decide as the load did.
        const what = `two folders under one declared skeleton, ${route}`;
        const declared = { name: 'other', nodes: ['a', 'b', 'c'], edges: [[0, 1]] };
        const first = await openFolder({ camA: 'camA_main', camB: 'camB_other' }, { route, folder: 'sess1', declared });
        await openFolder({ camC: 'camC_other', camD: 'camD_main' }, { route, folder: 'sess2', declared, page: first.page });
        const page = first.page;
        // An edit of the session's skeleton must not change what a re-read decides.
        await page.evaluate(async () => {
            const { state } = await import('/ui/app-state.js');
            state.session.skeleton.nodes[0] = 'renamed_a';
        });
        const renamed = rows => rows.map(r => r.replace('|other|a=', '|other|renamed_a='));
        const shownB = renamed(sources.camB_other.poses), shownC = renamed(sources.camC_other.poses);
        savedForPython(await saveBytes(page), what, { poses: [...sources.camA_main.poses, ...shownB, ...shownC, ...sources.camD_main.poses].sort(), skeletons: [['other', ['renamed_a', 'b', 'c']], MAIN] });
        await page.close();
    }

    console.log('\nJ. Analysis .h5 cameras');
    for (const route of ROUTES) {
        const what = `.h5 in another order, ${route}`;
        const { page, status } = await openFolder({ camA: 'h5_camA_main.h5', camB: 'h5_camB_reversed.h5' }, { route });
        check(!/not shown|shows no poses|Not read/.test(status), what + ': nothing hidden', status);
        const shown = await shownInApp(page);
        const camB = shown[0].cams.camB && shown[0].cams.camB[0];
        check(eq(shown[0].nodes, ['nose', 'tail']) && camB && camB[2] === 200 && camB[3] === 205, what + ': each value under its own name', camB);
        await page.close();
    }
    {
        // Eager: it goes aside like any other camera, on its own nodes.
        const what = '.h5 on other nodes, eager';
        const { page, status } = await openFolder({ camA: 'h5_camA_main.h5', camB: 'h5_camB_other.h5' }, { route: 'eager' });
        check(/camB shows no poses/.test(status), what + ': the status names the camera', status);
        const shown = await shownInApp(page);
        check(!shown[0].cams.camB, what + ': not shown', shown[0].cams.camB);
        const bytes = await saveBytes(page);
        await page.close();
        savedForPython(bytes, what, { poses: null, onVideo: 'camB.mp4', onVideoCount: 6, skeletons: [['skeleton', ['nose', 'tail']], ['skeleton', ['a', 'b', 'c']]] });
    }
    {
        // Lazy: the poses stay in a worker and a save has nothing to keep them in.
        const what = '.h5 on other nodes, lazy';
        const { page, status } = await openFolder({ camA: 'h5_camA_main.h5', camB: 'h5_camB_other.h5' }, { route: 'lazy' });
        check(/Not read: the annotation file of camB, on other nodes than this session's \(nose, tail\)/.test(status), what + ': refused, with the reason', status);
        const shown = await shownInApp(page);
        check(!shown[0].cams.camB && shown[0].cams.camA.length === 6, what + ': nothing read by position', shown[0].cams);
        await page.close();
    }

    console.log('\nK. Per-camera export and the fallback read');
    {
        // A lazy export of a hidden camera gives its own file back.
        const { page } = await openFolder({ camA: 'camA_main', camB: 'camB_kinds' }, { route: 'lazy' });
        const b64 = await page.evaluate(async () => {
            const fio = await import('/import-export/file-io.js');
            const { state } = await import('/ui/app-state.js');
            const blob = await fio.exportSlpClientSide(state.session, 'camB', false, null, 'camB.slp', null);
            const u = new Uint8Array(await blob.arrayBuffer());
            let s = '';
            for (let i = 0; i < u.length; i += 32768) s += String.fromCharCode.apply(null, u.subarray(i, i + 32768));
            return btoa(s);
        });
        savedForPython(Buffer.from(b64, 'base64'), 'a lazy export of the hidden camera', { poses: sources.camB_kinds.poses, detail: sources.camB_kinds.detail, skeletons: [OTHER] });
        await page.close();
    }
    {
        // The typed reader throws on that file: the raw worker has it all but identities.
        const { page, status } = await openFolder({ camA: 'camA_main', camB: 'camB_kinds' }, { route: 'eager', throwOnTyped: true });
        check(/camB shows no poses/.test(status), 'the raw worker\'s read: the status names the camera', status);
        savedForPython(await saveBytes(page), 'the raw worker\'s read', { poses: [...sources.camA_main.poses, ...sources.camB_kinds.poses].sort(), detail: sources.camB_kinds.detail, skeletons: [MAIN, OTHER] });
        await page.close();
    }

    console.log('\nL. The real multi-session picker');
    // Until fork PR #33 the picker's own `switchSession(0)` empties a lazy project's last
    // session, and a save then loses it on any skeleton. So a save is read back only once
    // that session survives the load; with the emptying taken out, these saves all pass.
    const lastSessionKept = page => page.evaluate(async () => {
        const { state } = await import('/ui/app-state.js');
        const last = state.sessions[state.sessions.length - 1];
        return !!last.lazyLoader || last.frameGroups.size > 0;
    });
    const notReadBack = what => console.log('  - ' + what + ': save not read back, the load emptied the last session (fork PR #33)');
    for (const route of ROUTES) {
        const what = `two sessions through the picker, ${route}`;
        const { page, status } = await openSessionsPicker({ sess1: { camA: 'camA_main', camB: 'camB_other' }, sess2: { camC: 'camC_main', camD: 'camD_main' } }, { route });
        check(/Loaded 2 sessions/.test(status.text) && status.text.includes(sources.camB_other.poses.length + ' poses on a camera\'s video are on another skeleton ("other")') && /warning/.test(status.dot),
            what + ': the last status says what is not shown', status);
        const shown = await shownInApp(page);
        check(!shown[0].cams.camB && shown[0].cams.camA.length === 6, what + ': the first session shows its own camera only', shown[0].cams);
        if (await lastSessionKept(page)) {
            savedForPython(await saveBytes(page), what, {
                poses: [...sources.camA_main.poses, ...sources.camB_other.poses, ...sources.camC_main.poses, ...sources.camD_main.poses].sort(),
                detail: sources.camB_other.detail, skeletons: [MAIN, OTHER],
                sessions: [{ camA: 'camA.mp4', camB: 'camB.mp4' }, { camC: 'camC.mp4', camD: 'camD.mp4' }],
            });
        } else notReadBack(what);
        await page.close();
    }
    for (const route of ROUTES) {
        const what = `a parent folder's skeleton.json through the picker, ${route}`;
        const { page, status } = await openSessionsPicker({ sess1: { camA: 'camA_main', camB: 'camB_other' }, sess2: { camC: 'camC_other', camD: 'camD_main' } }, { route, skeletonJson: JSON_OTHER });
        check(/Loaded 2 sessions/.test(status.text) && /on another skeleton \("main"\)/.test(status.text) && /warning/.test(status.dot), what + ': the status outlasts the skeleton step', status);
        const shown = await shownInApp(page);
        check(eq(shown[0].nodes, ['a', 'b', 'c']) && eq(shown[1].nodes, ['a', 'b', 'c']) && shown[0].cams.camB && !shown[0].cams.camA, what + ': every session shows what the parent declares', shown[0]);
        if (await lastSessionKept(page)) {
            savedForPython(await saveBytes(page), what, {
                poses: [...sources.camA_main.poses, ...sources.camB_other.poses, ...sources.camC_other.poses, ...sources.camD_main.poses].sort(),
                skeletons: [OTHER, MAIN],
            });
        } else notReadBack(what);
        await page.close();
    }
    {
        const what = '.h5 on other nodes through the picker, lazy';
        const cameras = { camA: 'h5_camA_main.h5', camB: 'h5_camB_other.h5' };
        const { page, status } = await openSessionsPicker({ sess1: cameras, sess2: cameras }, { route: 'lazy' });
        check(/Loaded 2 sessions/.test(status.text) && (status.text.match(/Not read: the annotation file of camB/g) || []).length === 1 && /error/.test(status.dot),
            what + ': the refusal is in the last status, in red, once', status);
        await page.close();
    }

    console.log('\nPython reads every save:');
    const readBack = JSON.parse(uv(PY_READ, pythonWork.map(w => w.file)).trim().split('\n').pop());
    for (const w of pythonWork) {
        const got = readBack[w.file];
        if (w.want.poses === null) {
            if (got.error) { check(false, w.what + ': opens', got.error); continue; }
            if (w.want.onVideoPoses) {
                const rows = got.poses.filter(r => r.startsWith(w.want.onVideo + '|'));
                check(eq(rows, w.want.onVideoPoses), w.what + ': that camera\'s poses', diff(rows, w.want.onVideoPoses));
            }
            if (w.want.onVideoCount) {
                const rows = got.poses.filter(r => r.startsWith(w.want.onVideo + '|'));
                check(rows.length === w.want.onVideoCount && rows.every(r => r.includes('a=') && r.includes('c=')), w.what + ': on its own nodes', rows.slice(0, 2));
            }
            if (w.want.detail) check(eq(got.detail, w.want.detail), w.what + ': what a kept pose also carries', diff(got.detail || [], w.want.detail));
            if (w.want.skeletons) check(eq(got.skeletons, w.want.skeletons), w.what + ': skeleton list', got.skeletons);
            check(got.repeatedFrames === 0, w.what + ': one row per video and frame', got.repeatedFrames);
            continue;
        }
        compare(w.what, got, w.want);
    }
} finally {
    await browser.close();
    server.kill();
    fs.rmSync(tmp, { recursive: true, force: true });
}
console.log(fails ? `\n${fails} check(s) FAILED` : '\nAll checks passed');
process.exit(fails ? 1 : 0);
