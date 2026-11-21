/**
 * volume-fusion.mjs — File ▸ Load Volume 3D and Triangulate ▸ Fuse with volume, end to end.
 *
 * On the bundle-adjustment rig fixture (3 cameras with real k1 distortion, 2 animals, 8 nodes, 2 px noise;
 * `fixtures/ba-rig-fixture.js`), a .vol3d.h5 is written IN the page with h5wasm (the file layout
 * scripts/export_volume3d.py writes: node order reversed and an extra node, to exercise matching by name) and
 * loaded through the real handler (`handleLoadVolume3d`). The volume is the truth, sharper than the 2D:
 *   1. the File menu item exists; Triangulate's "Fuse with volume" checkbox is greyed out until a volume is
 *      loaded, then on;
 *   2. This frame (`triangulateCurrentFrame`): with fusion off, plain DLT; on, every group is fused
 *      (`group.volumeFused`) and lands nearer the truth;
 *   3. All frames on a LAZY session, through a forced 2-worker pool (the volume travels in each job): every
 *      group fused and nearer the truth than an independent plain DLT;
 *   4. the display fill (ui/rendering.js) keeps a fused group's stored 3D even when the reprojection threshold is
 *      on (a re-solve there would drop the volume);
 *   5. fusion off again: Triangulate All gives exactly the plain DLT (bit-identical), no group marked fused.
 * Run: node tests/e2e/volume-fusion.mjs
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import net from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
/** a port nothing listens on (a taken one would silently serve another checkout) */
const freePort = () => new Promise((res, rej) => {
  const s = net.createServer();
  s.once('error', rej);
  s.listen(0, () => { const p = s.address().port; s.close(() => res(p)); });
});
const PORT = Number(process.env.PORT) || await freePort();
const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
for (let i = 0; i < 120; i++) {
  try { if ((await fetch(`http://localhost:${PORT}/index.html`)).ok) break; } catch {}
  await new Promise(r => setTimeout(r, 250));
}

let fails = 0;
const check = (c, m) => { console.log((c ? '  ✓ ' : '  ✗ ') + m); if (!c) fails++; };

try {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  const errs = [];
  page.on('pageerror', e => errs.push(String(e)));
  page.on('console', m => {
    if (m.type() !== 'error' || /Failed to load resource|net::ERR|404/.test(m.text())) return;   // absent demo assets
    errs.push('console: ' + m.text());
  });
  await page.goto(`http://localhost:${PORT}/index.html`);
  await page.waitForFunction(() => window.__lucid && window.__lucid.state, null, { timeout: 30000 });

  const out = await page.evaluate(async () => {
    const AS = await import('/ui/app-state.js');
    const TRI = await import('/pose/triangulation.js');
    const RND = await import('/ui/rendering.js');
    const V3D = await import('/pose/volume3d.js');
    const VFILE = await import('/import-export/volume3d-file.js');
    const FIO = await import('/import-export/file-io.js');
    const SET = await import('/ui/settings.js');
    const { buildBaRigFixture } = await import('/tests/e2e/fixtures/ba-rig-fixture.js');
    const F = await buildBaRigFixture({ frames: 6 });
    const r = {};
    const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    /** mean node distance of a group's 3D to the nearest animal's truth at frame f */
    const errTruth = (pts, f) => Math.min.apply(null, F.truth[f].map(T => {
      let s = 0, n = 0;
      for (let k = 0; k < F.NNODE; k++) {
        const X = [pts[3 * k], pts[3 * k + 1], pts[3 * k + 2]];
        if (X.every(Number.isFinite)) { s += dist3(X, T[k]); n++; }
      }
      return n ? s / n : Infinity;
    }));
    const useSession = (s) => {
      AS.state.sessions = [s]; AS.state.activeSessionIdx = 0; AS.state.session = s;
      AS.state.totalFrames = F.NF; AS.state.currentFrame = 0; AS.state.views = [];
      AS.state.triangulationResults = new Map();
    };
    SET.updateTrackingThresholds({ reprojErrorThreshold: 0 });

    // ---- 1. menu, checkbox, a real file through the real handler
    r.menuItem = !!document.getElementById('menuLoadVolume3d');
    const box = document.getElementById('tbVolumeFusion');
    const sE = F.mkSession('VOL_EAGER', { eager: true });
    useSession(sE);
    V3D.setVolumeSettings({ enabled: true });
    document.getElementById('triangulateDropdown').dispatchEvent(new Event('mouseenter'));
    r.boxDisabledBefore = !!(box && box.disabled);

    const SD = 0.02;    // the volume: the truth, sharper than the 2D (2 px ~ 0.1 world units here)
    const names = ['extra'].concat(Array.from({ length: F.NNODE }, (_, i) => 'n' + (F.NNODE - 1 - i)));
    const J = names.length, R = F.NF * F.NTRACK;
    const frames = new Int32Array(R), means = new Float32Array(R * J * 3), covs = new Float32Array(R * J * 6);
    for (let f = 0; f < F.NF; f++) for (let t = 0; t < F.NTRACK; t++) {
      const row = f * F.NTRACK + t;
      frames[row] = f;
      names.forEach((n, j) => {
        const X = n === 'extra' ? [0, 0, 0] : F.truth[f][t][Number(n.slice(1))];
        means.set(X, (row * J + j) * 3);
        covs.set([SD * SD, 0, 0, SD * SD, 0, SD * SD], (row * J + j) * 6);
      });
    }
    const mod = await FIO.initH5wasm();
    const fname = '/e2e_volume_' + Date.now() + '.h5';
    const hf = new mod.File(fname, 'w');
    hf.create_dataset({ name: 'frame_indices', data: frames });
    hf.create_dataset({ name: 'means', data: means, shape: [R, J, 3] });
    hf.create_dataset({ name: 'covs', data: covs, shape: [R, J, 6] });
    hf.create_dataset({ name: 'node_names', data: names });
    hf.create_attribute('format', 'lucid-volume3d/1');
    hf.create_attribute('cov_scale', new Float64Array([1.0]));
    hf.close();
    const bytes = mod.FS.readFile(fname);
    mod.FS.unlink(fname);
    const data = await VFILE.handleLoadVolume3d(new File([bytes], 'e2e.vol3d.h5'));
    r.loaded = !!data;
    r.rows = data ? data.frameIndices.length : 0;
    r.boxOnAfter = !!(box && !box.disabled && box.checked);
    r.coverage = V3D.volumeNodeCoverage(sE);

    // ---- 2. This frame: plain, then fused
    V3D.setVolumeSettings({ enabled: false });
    TRI.triangulateCurrentFrame('dlt');
    const plain0 = (TRI.getInstanceGroupsForFrame(0) || []).map(g => Float64Array.from(g.points3d));
    V3D.setVolumeSettings({ enabled: true });
    TRI.triangulateCurrentFrame('dlt');
    const g0 = TRI.getInstanceGroupsForFrame(0) || [];
    r.frameFused = g0.map(g => !!g.volumeFused);
    r.frameErrPlain = plain0.map(p => errTruth(p, 0));
    r.frameErrFused = g0.map(g => errTruth(g.points3d, 0));

    // ---- 3. All frames, lazy session, through 2 workers
    window.LUCID_TRIANGULATION_WORKERS = 2;
    const sL = F.mkSession('VOL_LAZY');
    useSession(sL);
    V3D.setSessionVolume(sL, data);
    await TRI.triangulateAllFrames('dlt');
    r.released = sL.lazyLoader.released.length;
    r.allFused = []; r.allErrFused = []; r.allErrRef = [];
    for (let f = 0; f < F.NF; f++) {
      const gs = sL.instanceGroups.get(f) || [];
      gs.forEach(g => { r.allFused.push(!!g.volumeFused); r.allErrFused.push(errTruth(g.points3d, f)); });
      F.refSolve(f, 'dlt').forEach(x => r.allErrRef.push(errTruth(x.points3d, f)));
    }

    // ---- 4. the display fill keeps the fused 3D with the reprojection threshold on
    SET.updateTrackingThresholds({ reprojErrorThreshold: 15 });
    const before = (sL.instanceGroups.get(0) || []).map(g => Float64Array.from(g.points3d));
    (sL.instanceGroups.get(0) || []).forEach(g => { g.reprojections = null; if (g.reprojectedInstances) g.reprojectedInstances.clear(); });
    AS.state.triangulationResults = new Map();
    RND.drawAllOverlays(0);
    const shown = AS.state.triangulationResults.get(0) || [];
    r.fillShown = shown.length;
    r.fillKeeps = shown.every(x => {
      const i = (sL.instanceGroups.get(0) || []).indexOf(x.group);
      return i >= 0 && F.points3dDelta(x.points3d, before[i]) === 0;
    });
    SET.updateTrackingThresholds({ reprojErrorThreshold: 0 });

    // ---- 5. off again: exactly plain DLT
    V3D.setVolumeSettings({ enabled: false });
    await TRI.triangulateAllFrames('dlt');
    r.offFused = 0; r.offDelta = 0;
    for (let f = 0; f < F.NF; f++) {
      const gs = sL.instanceGroups.get(f) || [];
      const ref = F.refSolve(f, 'dlt').map(x => x.points3d);
      gs.forEach(g => {
        if (g.volumeFused) r.offFused++;
        r.offDelta = Math.max(r.offDelta, Math.min.apply(null, ref.map(p => F.points3dDelta(g.points3d, p))));
      });
    }
    V3D.setVolumeSettings({ enabled: true });
    delete window.LUCID_TRIANGULATION_WORKERS;
    return r;
  });

  const mean = a => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
  check(out.menuItem, 'File ▸ Load Volume 3D is in the menu');
  check(out.boxDisabledBefore, '"Fuse with volume" is greyed out before a volume is loaded');
  check(out.loaded && out.rows === 12, `the .vol3d.h5 loads through the real handler (${out.rows} rows)`);
  check(out.boxOnAfter, '"Fuse with volume" is on once loaded');
  check(out.coverage.matched === 8 && out.coverage.total === 8, `nodes matched by name: ${out.coverage.matched}/${out.coverage.total}`);
  check(out.frameFused.length === 2 && out.frameFused.every(Boolean), 'This frame: every group fused');
  console.log(`    This frame, mean node error to the truth: plain ${mean(out.frameErrPlain).toFixed(4)}, fused ${mean(out.frameErrFused).toFixed(4)}`);
  check(mean(out.frameErrFused) < 0.5 * mean(out.frameErrPlain), 'This frame: fused nearer the truth than plain DLT');
  check(out.released > 0, `All frames took the windowed (lazy) path (${out.released} window releases)`);
  check(out.allFused.length === 12 && out.allFused.every(Boolean), `All frames: every group fused (${out.allFused.filter(Boolean).length}/${out.allFused.length})`);
  console.log(`    All frames (2 workers), mean error: independent plain DLT ${mean(out.allErrRef).toFixed(4)}, fused ${mean(out.allErrFused).toFixed(4)}`);
  check(mean(out.allErrFused) < 0.5 * mean(out.allErrRef), 'All frames: fused nearer the truth than plain DLT');
  check(out.fillShown === 2 && out.fillKeeps, 'the display keeps the fused 3D with the reprojection threshold on');
  check(out.offFused === 0, 'fusion off: no group marked fused');
  check(out.offDelta === 0, `fusion off: Triangulate All equals plain DLT exactly (worst delta ${out.offDelta})`);
  check(errs.length === 0, 'no page errors' + (errs.length ? ': ' + errs.slice(0, 3).join(' | ') : ''));
  await browser.close();
} catch (e) {
  console.error(e);
  fails++;
} finally {
  server.kill('SIGTERM');
}
console.log(fails ? `\n${fails} check(s) failed` : '\nall checks passed');
process.exit(fails ? 1 : 0);
