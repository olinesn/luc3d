/**
 * cli-scene.mjs — the "sample_session" input the command-line LUCID tests run
 * on: a recording folder laid out as Load Session Folder expects it,
 *
 *     <dir>/calibration.toml                  the demo rig (demo-data.js), 4 cameras
 *     <dir>/<cam>/<cam>.mp4                   sample_session's clip of that camera
 *     <dir>/<cam>/<cam>.predictions.slp       per-camera predictions
 *
 * with deterministic (seeded) predictions untidy enough that association has
 * real decisions to make: 3 animals random-walking through each other in
 * front of the four demo cameras, per-node dropouts, missed detections
 * (targets die and are re-born), false detections, a camera's raw track ids
 * that sometimes collide, and the per-camera slot order shuffled every frame.
 * Each camera numbers its tracks on its own, as per-camera SLEAP files do.
 *
 * Node only (writes the `.slp` files with sleap-io.js through the CLI's Node
 * platform). `writeCliScene(dir, opts)` returns { dir, cameras, frames, animals }.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { registerImportMap } from '../../../cli/node-resolve.mjs';
import { installNodePlatform } from '../../../cli/node-platform.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..', '..', '..');

function toml(cameras) {
    const arr = (a) => '[ ' + a.map(x => Array.isArray(x) ? arr(x) : String(x)).join(', ') + ',]';
    return cameras.map((c, i) => '[cam_' + i + ']\nname = "' + c.name + '"\nsize = ' + arr(c.size) +
        '\nmatrix = ' + arr(c.matrix) + '\ndistortions = ' + arr(c.dist) + '\nrotation = ' + arr(c.rvec) +
        '\ntranslation = ' + arr(c.tvec) + '\n').join('\n') + '\n[metadata]\n';
}

export async function writeCliScene(dir, opts) {
    opts = opts || {};
    const N = opts.frames || 600, NA = 3, seed = opts.seed || 7;
    registerImportMap();
    const { baseURI } = installNodePlatform();
    const SIO = await import(new URL('lib/sleap-io/index.browser.js', baseURI).href);
    const demo = await import('../../../demo-data.js');
    const cams = demo.createDemoCalibration();
    const skel = demo.createDemoSkeleton();
    const NN = skel.nodes.length;

    let s = seed >>> 0;
    const u = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
    const gauss = () => Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u());
    // Each animal: a rigid body (offsets around its centre, mm) on a bounded random walk.
    const shape = [], c = [], v = [];
    for (let a = 0; a < NA; a++) {
        shape.push(Array.from({ length: NN }, (_, k) => [(k - NN / 2) * 6 + (u() - 0.5) * 3, (u() - 0.5) * 6, (u() - 0.5) * 6]));
        c.push([(a - 1) * 30, (u() - 0.5) * 20, 0]); v.push([0, 0, 0]);
    }
    const W = 40;
    const obs = [];   // obs[f].get(cam) = [{ pts, track, score }]
    for (let f = 0; f < N; f++) {
        for (let a = 0; a < NA; a++) for (let d = 0; d < 3; d++) {
            v[a][d] = 0.9 * v[a][d] + 0.8 * gauss();
            c[a][d] += v[a][d];
            if (c[a][d] > W) { c[a][d] = 2 * W - c[a][d]; v[a][d] = -v[a][d]; }
            if (c[a][d] < -W) { c[a][d] = -2 * W - c[a][d]; v[a][d] = -v[a][d]; }
        }
        const perCam = new Map();
        for (const cam of cams) {
            const list = [];
            for (let a = 0; a < NA; a++) {
                if (u() < 0.04) continue;                                   // missed detection
                const pts = shape[a].map(o => {
                    const q = cam.distortPoint(cam.project([c[a][0] + o[0], c[a][1] + o[1], c[a][2] + o[2]]));
                    const keep = u() >= 0.08;                               // per-node dropout
                    return keep ? [q[0] + gauss() * 1.5, q[1] + gauss() * 1.5, 0.5 + 0.5 * u()] : null;
                });
                list.push({ pts, track: u() < 0.02 ? (a + 1) % NA : a, score: 0.5 + 0.5 * u() });
            }
            if (u() < 0.03) list.push({                                     // false detection
                pts: Array.from({ length: NN }, () => [320 + (u() - 0.5) * 300, 240 + (u() - 0.5) * 200, 0.2 * u()]),
                track: NA, score: 0.3 * u() });
            for (let i = list.length - 1; i > 0; i--) {                     // shuffle the slot order
                const j = Math.floor(u() * (i + 1)); const t = list[i]; list[i] = list[j]; list[j] = t;
            }
            perCam.set(cam.name, list);
        }
        obs.push(perCam);
    }

    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, 'calibration.toml'), toml(cams));
    const sioSkel = new SIO.Skeleton({ nodes: skel.nodes.map(n => new SIO.Node(n)),
        edges: [], name: skel.name || 'skeleton' });
    sioSkel.edges = (skel.edges || []).map(e => new SIO.Edge(sioSkel.nodes[e[0]], sioSkel.nodes[e[1]]));
    for (const cam of cams) {
        const camDir = path.join(dir, cam.name);
        fs.mkdirSync(camDir, { recursive: true });
        const clip = path.join(ROOT, 'sample_session', cam.name + '.mp4');
        if (fs.existsSync(clip)) fs.copyFileSync(clip, path.join(camDir, cam.name + '.mp4'));
        // This camera's own track numbering: its first-seen order of the scene's ids.
        const order = [];
        for (let f = 0; f < N; f++) for (const d of obs[f].get(cam.name)) if (order.indexOf(d.track) < 0) order.push(d.track);
        const tracks = order.map((_, i) => new SIO.Track('track_' + i));
        const video = new SIO.Video({ filename: cam.name + '.mp4', backendMetadata: { type: 'MediaVideo', shape: [21, 1024, 1280, 3] }, openBackend: false });
        video.shape = [21, 1024, 1280, 3];
        const lfs = [];
        for (let f = 0; f < N; f++) {
            const list = obs[f].get(cam.name);
            if (!list.length) continue;
            const insts = list.map(d => new SIO.PredictedInstance({
                points: d.pts.map(p => p ? [p[0], p[1], p[2]] : [NaN, NaN, NaN]),
                skeleton: sioSkel, track: tracks[order.indexOf(d.track)], score: d.score,
            }));
            lfs.push(new SIO.LabeledFrame({ video, frameIdx: f, instances: insts }));
        }
        const labels = new SIO.Labels({ labeledFrames: lfs, videos: [video], skeletons: [sioSkel], tracks });
        const bytes = await SIO.saveSlpToBytes(labels);
        fs.writeFileSync(path.join(camDir, cam.name + '.predictions.slp'), bytes);
    }
    return { dir, cameras: cams.map(c => c.name), frames: N, animals: NA };
}

// `node tests/e2e/fixtures/cli-scene.mjs <dir>` writes the folder.
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    const out = await writeCliScene(path.resolve(process.argv[2] || 'cli-scene'));
    console.log(JSON.stringify(out));
    process.exit(0);
}
