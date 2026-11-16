/**
 * test-cli-track.mjs — the command-line LUCID (cli/track.mjs) in plain Node,
 * end to end, on the sample_session scene (tests/e2e/fixtures/cli-scene.mjs:
 * sample_session's four cameras, seeded 3-animal predictions, one `.slp` per
 * camera). Its agreement with the app is tests/e2e/cli-parity.mjs; this checks
 * the CLI's own behaviour:
 *  1. per-camera files + calibration: 3 identities, every group triangulated
 *     (DLT), numAnimals saved, the input's video paths kept;
 *  2. the output reopens as a project: tracking it again (project mode)
 *     gives the same identities and groups;
 *  3. --no-triangulate leaves Track All's own 3D;
 *  4. --frames a-b tracks only a..b (0-based, inclusive);
 *  5. CAMERA=path names a camera; usage errors exit 2 with a message;
 *  6. the platform: DiskFile reads lazily and slices like a File, and the
 *     import map resolves the page's bare specifiers.
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { writeCliScene } from './e2e/fixtures/cli-scene.mjs';
import { digestSlp, firstDifference, compare3d } from './e2e/fixtures/slp-digest.mjs';
import { DiskFile } from '../cli/node-platform.mjs';
import { buildImportMap } from '../cli/node-resolve.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CLI = path.resolve(__dirname, '..', 'cli', 'track.mjs');
let passed = 0, failed = 0;
const check = (cond, msg, extra) => {
    if (cond) { passed++; console.log('  ok   ' + msg); }
    else { failed++; console.log('  FAIL ' + msg + (extra !== undefined ? '  ' + JSON.stringify(extra).slice(0, 400) : '')); }
};
const run = (args) => spawnSync(process.execPath, [CLI, ...args], { encoding: 'utf8' });

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'lucid-cli-test-'));
try {
    const sceneDir = path.join(tmp, 'rec');
    const scene = await writeCliScene(sceneDir, { frames: 400 });
    const cams = scene.cameras.map(c => path.join(sceneDir, c, c + '.predictions.slp'));
    const calib = path.join(sceneDir, 'calibration.toml');

    console.log('-- 1. per-camera files --');
    let r = run(['--slp', ...cams, '--calibration', calib, '--num-animals', '3', '--out', path.join(tmp, 'a.slp')]);
    check(r.status === 0, 'exit 0', r.stderr.slice(-400));
    check(/Track All: 3 identities/.test(r.stderr) && /Triangulate All \(DLT\)/.test(r.stderr), 'reports Track All and Triangulate All');
    const a = await digestSlp(path.join(tmp, 'a.slp'));
    check(a.identities.join(',') === 'id_0,id_1,id_2', '3 identities: ' + a.identities.join(','));
    check(a.groups > 0 && a.with3d === a.groups, 'every group has 3D (' + a.with3d + ' of ' + a.groups + ')');
    check(/numAnimals 3/.test(a.text), 'the animal count is saved with the project');
    check(/^ {2}G id_\d dlt /m.test(a.text), 'groups are DLT-triangulated (Group by Identity & Triangulate All)');
    check(/^session rec cameras back,mid,side,top /m.test(a.text), 'session named after the recording folder, cameras in calibration order');
    const SIO = globalThis.SleapIO;
    const labels = await SIO.readSlpStreaming(new DiskFile(path.join(tmp, 'a.slp')), { openVideos: false,
        h5wasmUrl: new URL('../lib/h5wasm/h5wasm.iife.js', import.meta.url).href });
    check(labels.videos.map(v => v.filename).join(',') === 'back.mp4,mid.mp4,side.mp4,top.mp4',
        'the videos the input referenced are kept: ' + labels.videos.map(v => v.filename).join(','));

    console.log('-- 2. the output reopens as a project --');
    r = run(['--slp', path.join(tmp, 'a.slp'), '--out', path.join(tmp, 'b.slp'), '--quiet']);
    check(r.status === 0, 'project mode, animal count from the file: exit 0', r.stderr.slice(-400));
    const b = await digestSlp(path.join(tmp, 'b.slp'));
    check(!firstDifference(a.text, b.text), 're-tracking the saved project gives the same identities and groups',
        firstDifference(a.text, b.text) || undefined);

    console.log('-- 3. --no-triangulate --');
    r = run(['--slp', ...cams, '--calibration', calib, '--num-animals', '3', '--no-triangulate', '--out', path.join(tmp, 'c.slp'), '--quiet']);
    const c = await digestSlp(path.join(tmp, 'c.slp'));
    check(r.status === 0 && !/Triangulate All/.test(r.stderr), 'stops after Track All');
    // Same groups, but Track All's own 3D (its normalized-coordinate solve), not Triangulate All's.
    // (A reopened file calls every group's 3D DLT: the method is not saved, only 'ba' is.)
    check(!firstDifference(a.text, c.text) && !compare3d(a, c, 0).ok, "the same groups, with Track All's 3D");

    console.log('-- 4. --frames --');
    r = run(['--slp', ...cams, '--calibration', calib, '--num-animals', '3', '--frames', '50-149', '--out', path.join(tmp, 'd.slp'), '--quiet']);
    const d = await digestSlp(path.join(tmp, 'd.slp'));
    const tracked = d.text.split('\n').filter((l, i, arr) => /^\d+$/.test(l) && /^ {2}G /.test(arr[i + 1] || '')).map(Number);
    check(r.status === 0 && tracked.length > 0 && tracked[0] === 50 && tracked[tracked.length - 1] === 149,
        'only frames 50..149 are grouped (0-based, inclusive): ' + tracked[0] + '..' + tracked[tracked.length - 1]);

    console.log('-- 5. camera names and usage errors --');
    const named = scene.cameras.map((cn, i) => cn + '=' + cams[i]);
    r = run(['--slp', ...named, '--calibration', calib, '--num-animals', '3', '--out', path.join(tmp, 'e.slp'), '--quiet']);
    const e = await digestSlp(path.join(tmp, 'e.slp'));
    check(r.status === 0 && !firstDifference(a.text, e.text), 'CAMERA=path gives the same project');
    const odd = path.join(tmp, 'odd.slp');
    fs.copyFileSync(cams[0], odd);
    r = run(['--slp', odd, cams[1], '--calibration', calib, '--num-animals', '3', '--out', path.join(tmp, 'x.slp')]);
    check(r.status === 2 && /cannot tell which camera/.test(r.stderr), 'an unmatched file name is refused: ' + r.stderr.trim().slice(0, 120));
    r = run(['--slp', ...cams, '--calibration', calib, '--out', path.join(tmp, 'x.slp')]);
    check(r.status === 2 && /--num-animals is required/.test(r.stderr), 'no animal count is refused');
    r = run(['--slp', ...cams, '--calibration', calib, '--num-animals', '3', '--frames', '9-3', '--out', path.join(tmp, 'x.slp')]);
    check(r.status === 2 && /end is before the start/.test(r.stderr), 'a reversed --frames is refused');
    r = run(['--help']);
    check(r.status === 0 && /FRAMES ARE 0-BASED/.test(r.stdout), '--help says frames are 0-based');

    console.log('-- 6. the Node platform --');
    const f = new DiskFile(calib);
    const text = fs.readFileSync(calib, 'utf8');
    check(f.size === Buffer.byteLength(text) && (await f.text()) === text, 'DiskFile: size and text');
    const sl = f.slice(5, 25);
    check(sl.size === 20 && (await sl.text()) === text.slice(5, 25) && sl instanceof File, 'DiskFile.slice: a File over the range');
    check((await f.slice(-10).text()) === text.slice(-10), 'DiskFile.slice: negative start counts from the end');
    const map = buildImportMap();
    check(['h5wasm', 'mediabunny', 'pako', 'yaml'].every(k => map.has(k)) && /lib\/h5wasm\/hdf5_hl\.js$/.test(map.get('h5wasm')),
        "the import map resolves index.html's bare specifiers");
} finally {
    fs.rmSync(tmp, { recursive: true, force: true });
}
console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
