/**
 * track-all-camera-other-skeleton.mjs — the work a per-camera folder is for, when its
 * camera files disagree on the skeleton: Track All and Triangulate All, on both routes,
 * with one camera in another node order and one on other nodes.
 *
 * The poses are projections of known 3D points through the folder's calibration, so the
 * 3D says whether each node's values went to the right node: read by position, nose and
 * tail of the reordered camera would swap and the 3D would miss. The camera on other
 * nodes must sit out and lose nothing. Real Python reads every save.
 *
 * Needs `uv` for Python (skipped without it).
 *
 * Run: node tests/e2e/track-all-camera-other-skeleton.mjs   (spawns its own http.server; PORT= to move it)
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
const PORT = Number(process.env.PORT || 8471);
const PY_SPEC = process.env.SLEAP_IO_SPEC || 'sleap-io==0.9.5';
const NF = 12;

let fails = 0;
const check = (c, m, got) => {
    console.log((c ? '  ✓ ' : '  ✗ ') + m + (!c && got !== undefined ? ' — got ' + JSON.stringify(got) : ''));
    if (!c) fails++;
    stallGuard.refresh();
};

const PY_READ = `
import json, sys
import numpy as np
import sleap_io as sio
def num(v):
    return "nan" if v != v else f"{float(v):.2f}"
out = {}
for p in sys.argv[1:]:
    try:
        labels = sio.load_slp(p, open_videos=False)
    except Exception as e:
        out[p] = {"error": type(e).__name__ + ": " + str(e)[:160]}
        continue
    poses = {}
    for lf in labels.labeled_frames:
        for i in lf.instances:
            key = lf.video.filename + "|" + (i.skeleton.name or "skeleton") + "|" + ",".join(i.skeleton.node_names)
            poses[key] = poses.get(key, 0) + 1
    groups = []
    s = labels.sessions[0]
    for f, fg in s.frame_groups.items():
        for ig in fg.instance_groups:
            members = {c.name: {n: [float(v) for v in i.points["xy"][k]] for k, n in enumerate(i.skeleton.node_names)} for c, i in ig.instance_by_camera.items()}
            groups.append({"frame": int(f), "members": members, "points3d": None if ig.instance_3d is None else np.asarray(ig.instance_3d.points).tolist()})
    out[p] = {"error": None, "poses": poses, "groups": groups, "skeletons": [[s.name or "skeleton", s.node_names] for s in labels.skeletons]}
print(json.dumps(out))
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

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'track-all-camera-other-skeleton-'));
const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
for (let i = 0; i < 60; i++) {
    try { if ((await fetch(`http://localhost:${PORT}/index.html`)).ok) break; } catch { /* not yet */ }
    await new Promise(r => setTimeout(r, 250));
}
const browser = await chromium.launch();
const stallGuard = setTimeout(() => {
    console.log('\nStuck: no check for 10 minutes');
    browser.close();
}, 10 * 60 * 1000);
stallGuard.unref();

// Three shown cameras and one on other nodes, all around two animals.
const CAMERAS = ['camA', 'camB', 'camC', 'camD'];
const animalAt = (animal, f) => (animal === 0
    ? { nose: [-30 + f * 0.5, 0, 200], tail: [-22 + f * 0.5, 4, 200] }
    : { nose: [30 - f * 0.5, 20, 240], tail: [38 - f * 0.5, 24, 240] });

async function newAppPage() {
    const page = await browser.newPage();
    page.on('pageerror', e => { console.log('  [pageerror]', String(e).slice(0, 300)); fails++; });
    await answerTrackSetup(page, '');   // the first-use dialog's animal count -> auto-detect
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state && window.SleapIO, null, { timeout: 120000 });
    return page;
}

/** The camera files, projected in the page by the app's own `Camera`. */
async function writeFixtures() {
    const page = await newAppPage();
    const files = await page.evaluate(async ({ CAMERAS, NF, A, B }) => {
        const pd = await import('/pose/pose-data.js');
        const SIO = window.SleapIO;
        const animal = (i, f) => { const s = i === 0 ? A : B; return { nose: s.nose.map((v, k) => v + s.dNose[k] * f), tail: s.tail.map((v, k) => v + s.dTail[k] * f) }; };
        const b64 = u => { let s = ''; for (let i = 0; i < u.length; i += 32768) s += String.fromCharCode.apply(null, u.subarray(i, i + 32768)); return btoa(s); };
        const out = {};
        for (let ci = 0; ci < CAMERAS.length; ci++) {
            const name = CAMERAS[ci];
            const cam = new pd.Camera(name, [[500, 0, 32], [0, 500, 32], [0, 0, 1]], [0, 0, 0, 0, 0], [0, 0.1 * ci, 0], [10 * ci, 0, 0], [64, 64]);
            // camB in another node order; camD on other nodes, with the same animals' values.
            const nodes = name === 'camB' ? ['tail', 'nose'] : name === 'camD' ? ['a', 'b', 'c'] : ['nose', 'tail'];
            const skeleton = new SIO.Skeleton({ name: name === 'camD' ? 'other' : 'main', nodes });
            const video = new SIO.Video({ filename: name + '.mp4', shape: [NF, 64, 64, 1] });
            const tracks = [new SIO.Track('t0'), new SIO.Track('t1')];
            const lfs = [];
            for (let f = 0; f < NF; f++) {
                const instances = [0, 1].map(i => {
                    const a = animal(i, f);
                    const at = n => cam.project(n === 'tail' ? a.tail : a.nose);
                    const points = nodes.map(n => at(n === 'nose' || n === 'tail' ? n : 'nose'));
                    return new SIO.PredictedInstance({ points, skeleton, track: tracks[i], score: 0.9 });
                });
                lfs.push(new SIO.LabeledFrame({ video, frameIdx: f, instances }));
            }
            out[name] = b64(await SIO.saveSlpToBytes(new SIO.Labels({ skeletons: [skeleton], videos: [video], tracks, labeledFrames: lfs })));
        }
        return out;
    }, {
        CAMERAS, NF,
        A: { nose: animalAt(0, 0).nose, tail: animalAt(0, 0).tail, dNose: [0.5, 0, 0], dTail: [0.5, 0, 0] },
        B: { nose: animalAt(1, 0).nose, tail: animalAt(1, 0).tail, dNose: [-0.5, 0, 0], dTail: [-0.5, 0, 0] },
    });
    await page.close();
    return files;
}

async function openFolder(route, files) {
    const page = await newAppPage();
    await page.evaluate(files => { window.__cams = Object.fromEntries(Object.entries(files).map(([k, v]) => [k, Uint8Array.from(atob(v), c => c.charCodeAt(0))])); }, files);
    const loading = page.evaluate(async ({ route, CAMERAS }) => {
        const loader = await import('/loading/session-loader.js');
        const tri = await import('/pose/triangulation.js');
        const pad = route === 'lazy' ? [new Uint8Array(tri.LAZY_SLP_THRESHOLD + (1 << 20))] : [];
        const mkFile = (bits, rel) => { const f = new File(bits, rel.split('/').pop()); Object.defineProperty(f, 'webkitRelativePath', { value: rel }); return f; };
        const list = CAMERAS.map(c => mkFile([window.__cams[c]].concat(pad), 'root/' + c + '/' + c + '.slp'));
        list.push(mkFile([CAMERAS.map((name, i) => '[cam_' + i + ']\nname = "' + name + '"\nsize = [64, 64]\n'
            + 'matrix = [[500.0, 0.0, 32.0], [0.0, 500.0, 32.0], [0.0, 0.0, 1.0]]\ndistortions = [0.0, 0.0, 0.0, 0.0, 0.0]\n'
            + 'rotation = [0.0, ' + (0.1 * i) + ', 0.0]\ntranslation = [' + (10 * i) + ', 0.0, 0.0]\n').join('\n')], 'root/calibration.toml'));
        await loader.handleLoadSessionFolderPerCamera(list, true);
        return (document.getElementById('statusText') || {}).textContent;
    }, { route, CAMERAS });
    await page.getByRole('button', { name: 'Continue', exact: true }).click();
    return { page, status: await loading };
}

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

const close = (got, want, tol = 1) => got.every((v, k) => Math.abs(v - want[k]) < tol);

try {
    const files = await writeFixtures();
    const saves = [];
    for (const route of ['eager', 'lazy']) {
        console.log(`\n${route}`);
        const { page, status } = await openFolder(route, files);
        check(/camD shows no poses/.test(status), 'the status names the camera on other nodes', status);
        const run = await page.evaluate(async ({ NF }) => {
            const tracker = await import('/pose/tracker.js');
            const tri = await import('/pose/triangulation.js');
            const { state } = await import('/ui/app-state.js');
            await tracker.trackAll();
            await tri.triangulateAllFrames();
            const s = state.session;
            const groups = [];
            for (let f = 0; f < NF; f++) {
                for (const g of s.instanceGroups.get(f) || []) groups.push({ f, cams: [...g.instances.keys()].sort(), p: Array.from(g.points3d || []) });
            }
            return { nodes: s.skeleton.nodes, identities: s.identities.length, status: (document.getElementById('statusText') || {}).textContent, groups };
        }, { NF });
        // Noiseless projections: anything but 0.00 px means a node went to the wrong place.
        check(new RegExp('Triangulated ' + NF + ' frames via DLT \\(' + (2 * NF) + ' groups, avg error: 0\\.00px\\)').test(run.status), 'Triangulate All solves every group with no reprojection error', run.status);
        check(JSON.stringify(run.nodes) === '["nose","tail"]', 'the session shows the first camera\'s node order', run.nodes);
        check(run.identities === 2, 'two animals found', run.identities);
        check(run.groups.length === 2 * NF, 'every frame has two groups', run.groups.length);
        check(run.groups.every(g => JSON.stringify(g.cams) === '["camA","camB","camC"]'), 'each group links the three shown cameras, not the hidden one', run.groups[0] && run.groups[0].cams);
        // Which animal a group is does not matter: its nose and tail must be one animal's.
        const good = run.groups.filter(g => [0, 1].some(a => {
            const want = animalAt(a, g.f);
            return g.p.length === 6 && close(g.p.slice(0, 3), want.nose) && close(g.p.slice(3, 6), want.tail);
        }));
        check(good.length === run.groups.length, 'the 3D lands on each node of each animal, so the reordered camera was read by name', run.groups.find(g => !good.includes(g)));
        const first = await saveBytes(page);
        const second = await saveBytes(page);
        await page.close();
        for (const [label, bytes] of [['save 1', first], ['save 2', second]]) {
            const file = path.join(tmp, route + '-' + label.replace(' ', '') + '.slp');
            fs.writeFileSync(file, bytes);
            saves.push({ route, label, file });
        }
    }

    console.log('\nPython reads every save:');
    const read = JSON.parse(uv(PY_READ, saves.map(s => s.file)).trim().split('\n').pop());
    for (const { route, label, file } of saves) {
        const what = `${route}, ${label}`;
        const got = read[file];
        if (got.error) { check(false, what + ': opens', got.error); continue; }
        check(got.poses['camD.mp4|other|a,b,c'] === 2 * NF, what + ': the hidden camera\'s poses, on their own skeleton', got.poses);
        check(got.poses['camB.mp4|main|nose,tail'] === 2 * NF, what + ': the reordered camera\'s poses, on the session\'s', got.poses);
        check(got.groups.length === 2 * NF, what + ': every group', got.groups.length);
        // A group's 2D, as Python reads it by name, is the projection of its 3D's nodes.
        const good = got.groups.filter(g => [0, 1].some(a => {
            const want = animalAt(a, g.frame);
            const p = g.points3d;
            return p && close(p[0].slice(0, 3), want.nose) && close(p[1].slice(0, 3), want.tail);
        }));
        check(good.length === got.groups.length, what + ': each group\'s 3D is its animal\'s, node for node', got.groups.find(g => !good.includes(g)));
        const noses = got.groups.filter(g => {
            const m = g.members;
            return m.camA && m.camB && m.camC && !m.camD && 'nose' in m.camB && 'tail' in m.camB;
        });
        check(noses.length === got.groups.length, what + ': members are the three shown cameras, each on nose and tail', got.groups[0] && Object.keys(got.groups[0].members));
    }
} finally {
    await browser.close();
    server.kill();
    fs.rmSync(tmp, { recursive: true, force: true });
}
console.log(fails ? `\n${fails} check(s) FAILED` : '\nAll checks passed');
process.exit(fails ? 1 : 0);
