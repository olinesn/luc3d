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
 *  6. --fps runs Track All's ID-switch check by body size and saves its
 *     checklist, without changing the tracking; --no-size-check, the settings'
 *     autoSwitchCheck 0 and a missing --fps skip it; a bad --fps is refused.
 *  7. the platform: DiskFile reads lazily and slices like a File, and the
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

    console.log('-- 6. ID-switch check by body size --');
    const sizeDir = path.join(tmp, 'rec-size');
    const sizeScene = await writeCliScene(sizeDir, { frames: 2400, sizeSkeleton: true, seed: 11 });
    const sizeCams = sizeScene.cameras.map(c => path.join(sizeDir, c, c + '.predictions.slp'));
    const sizeArgs = (...more) => ['--slp', ...sizeCams, '--calibration', path.join(sizeDir, 'calibration.toml'), '--num-animals', '3', ...more];
    const reviewOf = (d) => { const m = d.text.match(/^idSwitchReview (.*)$/m); return m ? JSON.parse(m[1]) : null; };
    const noReview = (d) => d.text.replace(/^(idSwitchReview|lucidKeys) .*\n/mg, '');
    r = run(sizeArgs('--fps', '30', '--out', path.join(tmp, 's1.slp')));
    const s1 = await digestSlp(path.join(tmp, 's1.slp')), rv = reviewOf(s1);
    check(r.status === 0 && /ID-switch check \(body size\): .* close encounters at 30 fps/.test(r.stderr),
        '--fps 30 runs the check: ' + ((r.stderr.match(/ID-switch check.*$/m) || [''])[0]));
    check(rv && rv.v === 1 && rv.checks.size && rv.checks.size.fps === 30 && rv.checks.size.fpsFromVideo === false &&
        rv.checks.size.encounters > 0 && /lucidKeys .*idSwitchReview/.test(s1.text),
        'its checklist is saved (metadata.lucid.idSwitchReview): ' + (rv ? rv.checks.size.encounters + ' encounters, ' + rv.checks.size.points.length + ' points' : 'none'));
    r = run(sizeArgs('--fps', '30', '--no-size-check', '--out', path.join(tmp, 's2.slp')));
    const s2 = await digestSlp(path.join(tmp, 's2.slp'));
    check(r.status === 0 && !/ID-switch check/.test(r.stderr) && !reviewOf(s2), '--no-size-check: no check, no checklist');
    check(!firstDifference(noReview(s1), noReview(s2)) && compare3d(s1, s2, 0).ok, 'the check changes nothing else in the file');
    fs.writeFileSync(path.join(tmp, 'nocheck.json'), JSON.stringify({ trackingThresholds: { autoSwitchCheck: 0 } }));
    r = run(sizeArgs('--fps', '30', '--settings', path.join(tmp, 'nocheck.json'), '--out', path.join(tmp, 's3.slp')));
    check(r.status === 0 && !/ID-switch check/.test(r.stderr) && !reviewOf(await digestSlp(path.join(tmp, 's3.slp'))),
        'settings autoSwitchCheck 0 (the Tracking Wizard\'s box unticked): no check');
    r = run(sizeArgs('--out', path.join(tmp, 's4.slp')));
    check(r.status === 0 && /body size\): skipped .*--fps/.test(r.stderr) && !reviewOf(await digestSlp(path.join(tmp, 's4.slp'))),
        'without --fps: skipped, and it says why');
    const shortDir = path.join(tmp, 'rec-short');
    const shortScene = await writeCliScene(shortDir, { frames: 900, sizeSkeleton: true, seed: 11 });
    r = run(['--slp', ...shortScene.cameras.map(c => path.join(shortDir, c, c + '.predictions.slp')), '--calibration',
        path.join(shortDir, 'calibration.toml'), '--num-animals', '3', '--fps', '30', '--out', path.join(tmp, 's5.slp')]);
    check(r.status === 0 && /body size\): skipped — needs at least 60 s of tracking/.test(r.stderr), 'a 30 s recording: skipped, as in the app');
    r = run(['--slp', ...cams, '--calibration', calib, '--num-animals', '3', '--fps', '30', '--out', path.join(tmp, 's6.slp')]);
    check(r.status === 0 && /body size\): skipped — The skeleton has too few of the size nodes/.test(r.stderr),
        'a skeleton without the size nodes: skipped, and it says why');
    r = run(sizeArgs('--fps', 'fast', '--out', path.join(tmp, 'x.slp')));
    check(r.status === 2 && /--fps must be a positive number/.test(r.stderr), 'a bad --fps is refused');

    console.log('-- 7. the Node platform --');
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
