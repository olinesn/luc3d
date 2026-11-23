/**
 * reference-mesh-load.mjs — File ▸ Load Reference Mesh, through the real file
 * chooser, into the real 3D viewport.
 *
 * A reference mesh is display-only geometry (e.g. a cage scan aligned to the
 * rig by the cage-scan tools) drawn in calibration world coordinates. Pinned:
 *
 *   1. A coloured PLY (quads) loads through File ▸ Load Reference Mesh: one
 *      Mesh in the viewport, indexed, its vertex colours used, see-through,
 *      listed under Visibility ▸ Reference Meshes; "Fit to scene" frames it.
 *   2. A binary STL loads through the section's own button; show, colour,
 *      opacity and remove act on the drawn object.
 *   3. A points-only PLY is drawn as points; a broken file is refused with a
 *      status and leaves nothing behind.
 *   4. It is display only: no project key, the project stays clean, and it
 *      survives the viewport being rebuilt (every project load does that).
 *   5. A 2,000,000-triangle binary STL loads with the progress card counting
 *      triangles and no long freeze of the page.
 *
 * Run: node reference-mesh-load.mjs   (serves the app itself)
 */
import { startServer, launchBrowser } from './lib/harness.mjs';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const server = await startServer(repoRoot);
const PORT = server.port;

let fails = 0;
const check = (c, m) => { console.log((c ? '  ✓ ' : '  ✗ ') + m); if (!c) fails++; };

// --- fixture files ------------------------------------------------------------
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'lucid-refmesh-'));

/** A coloured axis-aligned box as an ascii PLY with 6 quad faces. */
function boxPly(c, s) {
    const v = [];
    for (let k = 0; k < 8; k++) {
        v.push([c[0] + ((k & 1) ? s : -s), c[1] + ((k & 2) ? s : -s), c[2] + ((k & 4) ? s : -s),
            (k & 1) ? 255 : 0, (k & 2) ? 255 : 0, (k & 4) ? 255 : 0]);
    }
    const f = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]];
    return 'ply\nformat ascii 1.0\nelement vertex 8\nproperty float x\nproperty float y\nproperty float z\n' +
        'property uchar red\nproperty uchar green\nproperty uchar blue\nelement face 6\n' +
        'property list uchar int vertex_indices\nend_header\n' +
        v.map(r => r.join(' ')).join('\n') + '\n' + f.map(q => '4 ' + q.join(' ')).join('\n') + '\n';
}
/** Binary STL: `n` thin triangles marching along x. */
function binaryStl(n) {
    const buf = Buffer.alloc(84 + 50 * n);
    buf.write('test stl', 0);
    buf.writeUInt32LE(n, 80);
    for (let i = 0; i < n; i++) {
        const o = 84 + 50 * i + 12;
        const x = (i % 2000) * 0.5, y = Math.floor(i / 2000) * 0.5;
        const v = [x, y, 0, x + 0.4, y, 0, x, y + 0.4, 0.1];
        for (let k = 0; k < 9; k++) buf.writeFloatLE(v[k], o + 4 * k);
    }
    return buf;
}
const PLY_BOX = path.join(tmp, 'cage_scan.ply');
fs.writeFileSync(PLY_BOX, boxPly([3000, 0, 50], 40));
const STL_SMALL = path.join(tmp, 'hull.stl');
fs.writeFileSync(STL_SMALL, binaryStl(10));
const PLY_POINTS = path.join(tmp, 'points.ply');
fs.writeFileSync(PLY_POINTS, 'ply\nformat ascii 1.0\nelement vertex 3\nproperty double x\nproperty double y\n' +
    'property double z\nend_header\n0 0 0\n10 0 0\n0 10 0\n');
const BAD = path.join(tmp, 'broken.ply');
fs.writeFileSync(BAD, 'ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\n');
const BIG_N = 2000000;
const STL_BIG = path.join(tmp, 'big_scan.stl');
fs.writeFileSync(STL_BIG, binaryStl(BIG_N));

let browser;
try {
    browser = await launchBrowser();
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    page.on('pageerror', e => { console.log('  [pageerror]', String(e).slice(0, 300)); fails++; });
    const fits = [];
    page.on('console', m => {
        const t = m.text();
        const fm = /\[3D\] fitToScene: (\d+) pts/.exec(t);
        if (fm) fits.push(Number(fm[1]));
        if (m.type() !== 'error') return;
        if (/Failed to load resource|net::ERR|404/.test(t)) return;
        console.log('  [console.error]', t.slice(0, 300));
        fails++;
    });

    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state && window.SleapIO, null, { timeout: 30000 });

    // A calibrated session, so `setup3DViewport` really builds a viewport.
    await page.evaluate(async () => {
        const pd = await import('/pose/pose-data.js');
        const AS = await import('/ui/app-state.js');
        const { Skeleton, Camera, Instance, InstanceGroup, FrameGroup, Session } = pd;
        const K = [[1000, 0, 255.5], [0, 1000, 255.5], [0, 0, 1]];
        const cams = [
            new Camera('cam1', K, [0, 0, 0, 0, 0], [0.01, 0.02, 0.03], [0, 0, 500], [512, 512]),
            new Camera('cam2', K, [0, 0, 0, 0, 0], [0.04, 0.05, 0.06], [100, 0, 500], [512, 512]),
        ];
        const skel = new Skeleton('skeleton', ['nose', 'tail'], [[0, 1]]);
        const session = new Session(cams, skel, ['mouseA'], 'RefMeshTest');
        const fg = new FrameGroup(0);
        session.addFrameGroup(fg);
        const g = new InstanceGroup(1, 0);
        g.addInstance('cam1', new Instance([[10, 20], [30, 40]], 0, 'user', 1));
        g.addInstance('cam2', new Instance([[12, 22], [32, 42]], 0, 'user', 1));
        for (const [cn, inst] of g.instances) fg.addInstance(cn, inst);
        g.points3d = new Float64Array([0, 0, 10, 0, 0, 20]);
        session.instanceGroups.set(0, [g]);
        AS.state.sessions = [session];
        AS.state.activeSessionIdx = 0;
        AS.state.session = session;
        AS.state.totalFrames = 1;
        AS.state.currentFrame = 0;
        AS.state.isDirty = false;
        (await import('/pose/initialization.js')).setup3DViewport();
    });
    await page.waitForTimeout(600);

    const inspect = () => page.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        const R = await import('/ui/reference-meshes.js');
        const vp = AS.viewport3d;
        const group = vp && vp.scene.children.find(c => c.name === 'referenceMeshes');
        const objs = group ? group.children.map(o => ({
            name: o.name, type: o.type, visible: o.visible,
            positions: o.geometry.getAttribute('position').count,
            index: o.geometry.index ? o.geometry.index.count : null,
            hasColor: !!o.geometry.getAttribute('color'),
            hasNormal: !!o.geometry.getAttribute('normal'),
            vertexColors: !!o.material.vertexColors, transparent: o.material.transparent,
            opacity: o.material.opacity, color: '#' + o.material.color.getHexString(),
        })) : null;
        const rows = Array.from(document.querySelectorAll('#referenceMeshList .ref-mesh-row')).map(r => ({
            name: r.querySelector('.ref-mesh-name').textContent,
            count: r.querySelector('.ref-mesh-count').textContent,
            colorDisabled: r.querySelector('.ref-mesh-color').disabled,
        }));
        return {
            hasViewport: !!vp, objs, rows,
            entries: R.referenceMeshState.entries.map(e => ({ id: e.id, name: e.name, kind: e.mesh.kind,
                tri: e.mesh.triangleCount })),
            empty: document.getElementById('referenceMeshEmpty').style.display !== 'none',
            status: document.getElementById('statusText').textContent,
            dirty: AS.state.isDirty === true,
            overlayHidden: document.getElementById('loadingOverlay').classList.contains('hidden'),
        };
    });
    const pick = async (file, trigger) => {
        // Clear the status first: the wait below keys on the NEXT load's message.
        await page.evaluate(() => { document.getElementById('statusText').textContent = ''; });
        page.once('filechooser', async (fc) => { await fc.setFiles(file); });
        await page.evaluate(trigger);
        await page.waitForFunction(() =>
            document.getElementById('loadingOverlay').classList.contains('hidden') &&
            /Loaded reference mesh|Could not load/.test(document.getElementById('statusText').textContent),
        null, { timeout: 120000 });
    };

    let s = await inspect();
    check(s.hasViewport, 'the calibrated session has a 3D viewport');
    check(s.objs && s.objs.length === 0 && s.empty, 'no reference mesh at first; the section says "None loaded"');

    // ===================================================================
    console.log('\n-- 1. a coloured PLY through File ▸ Load Reference Mesh --');
    const fitsBefore = fits.length;
    await pick(PLY_BOX, () => { document.getElementById('menuLoadReferenceMesh').click(); });
    s = await inspect();
    check(/Loaded reference mesh "cage_scan.ply": 12 triangles/.test(s.status), `status: "${s.status}"`);
    const o1 = s.objs && s.objs[0];
    check(s.objs.length === 1 && o1.type === 'Mesh' && o1.name === 'referenceMesh_1', 'one Mesh in the viewport');
    check(o1 && o1.index === 36 && o1.positions === 8, `indexed: 8 vertices, 6 quads -> 36 indices (${o1 && o1.index})`);
    check(o1 && o1.hasColor && o1.vertexColors && o1.color === '#ffffff', 'the file\'s vertex colours are used');
    check(o1 && o1.hasNormal, 'normals computed for shading');
    check(o1 && o1.transparent && Math.abs(o1.opacity - 0.5) < 1e-9, 'see-through by default');
    check(s.rows.length === 1 && s.rows[0].name === 'cage_scan.ply' && s.rows[0].count === '12 tri' &&
        s.rows[0].colorDisabled && !s.empty, `listed under Visibility ▸ Reference Meshes (${JSON.stringify(s.rows)})`);
    const lastFit = fits.length > fitsBefore ? fits[fits.length - 1] : null;
    const fitNow = await page.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        const R = await import('/ui/reference-meshes.js');
        R.setReferenceMeshVisible(1, false);
        AS.viewport3d.fitToScene();
        R.setReferenceMeshVisible(1, true);
    }).then(() => fits[fits.length - 1]);
    check(lastFit !== null && lastFit === fitNow + 8,
        `loading fits the view to include the mesh's box (${lastFit} pts with it, ${fitNow} without)`);

    // ===================================================================
    console.log('\n-- 2. a binary STL through the section\'s button; the row controls --');
    await pick(STL_SMALL, () => { document.getElementById('btnLoadReferenceMesh').click(); });
    s = await inspect();
    check(s.objs.length === 2 && s.objs[1].index === null && s.objs[1].positions === 30 && !s.objs[1].hasColor,
        `an STL is a soup of 30 vertices, no colours (${JSON.stringify(s.objs[1])})`);
    check(s.objs[1].color === '#ce93d8', `the second mesh gets the next colour (${s.objs[1].color})`);
    const ctl = await page.evaluate(async () => {
        const rows = document.querySelectorAll('#referenceMeshList .ref-mesh-row');
        const row = rows[1];
        const fire = (el, t) => el.dispatchEvent(new Event(t, { bubbles: true }));
        const show = row.querySelector('.ref-mesh-show');
        show.checked = false; fire(show, 'change');
        const color = row.querySelector('.ref-mesh-color');
        color.value = '#123456'; fire(color, 'input');
        const slider = row.nextElementSibling.querySelector('.ref-mesh-opacity');
        slider.value = '100'; fire(slider, 'input');
        return true;
    });
    s = await inspect();
    check(ctl && s.objs[1].visible === false, 'the show box hides it');
    check(s.objs[1].color === '#123456', 'the colour picker recolours it');
    check(s.objs[1].opacity === 1 && s.objs[1].transparent === false, 'opacity 100% draws it opaque');
    await page.evaluate(() => document.querySelectorAll('#referenceMeshList .ref-mesh-remove')[1].click());
    s = await inspect();
    check(s.objs.length === 1 && s.rows.length === 1 && s.entries.length === 1, '× removes it from the scene and the list');

    // ===================================================================
    console.log('\n-- 3. points, and a broken file --');
    await pick(PLY_POINTS, () => { document.getElementById('menuLoadReferenceMesh').click(); });
    s = await inspect();
    check(s.objs.length === 2 && s.objs[1].type === 'Points' && s.objs[1].positions === 3 &&
        s.rows[1].count === '3 pts', 'a PLY without faces is drawn as points');
    await pick(BAD, () => { document.getElementById('menuLoadReferenceMesh').click(); });
    s = await inspect();
    check(/Could not load "broken.ply": .*end_header/.test(s.status), `a broken file is refused with a reason ("${s.status}")`);
    check(s.objs.length === 2 && s.entries.length === 2 && s.overlayHidden, 'and leaves nothing behind');

    // ===================================================================
    console.log('\n-- 4. display only --');
    const r4 = await page.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        const fileio = await import('/import-export/file-io.js');
        const R = await import('/ui/reference-meshes.js');
        const views = AS.state.session.cameras.map(c => ({ name: c.name, videoWidth: 512, videoHeight: 512, frameCount: 1 }));
        const vf = AS.state.session.cameras.map(c => ({ name: c.name, assignedCamera: c.name, videoPath: c.name + '.mp4' }));
        const keysOf = async () => {
            const bytes = await window.SleapIO.saveSlpToBytes(fileio.buildSlpLabelsAllViews(AS.state.session, views, vf));
            const parsed = await fileio.parseSlpViaSleapIO(new File([bytes], 'p.slp'), () => {});
            return Object.keys(parsed.sessions[0].metadata.lucid).sort();
        };
        const withMeshes = await keysOf();
        const saved = R.referenceMeshState.entries.slice();
        R.referenceMeshState.entries.length = 0;
        const without = await keysOf();
        R.referenceMeshState.entries.push(...saved);
        const dirty = AS.state.isDirty === true;
        // Rebuild the viewport, as every project load does.
        (await import('/pose/initialization.js')).setup3DViewport();
        await new Promise(r => setTimeout(r, 300));
        const g = AS.viewport3d.scene.children.find(c => c.name === 'referenceMeshes');
        return { withMeshes, without, dirty, rebuilt: g ? g.children.map(o => o.type) : null };
    });
    check(JSON.stringify(r4.withMeshes) === JSON.stringify(r4.without),
        `the saved file is the same with or without reference meshes (${r4.withMeshes.join(',')})`);
    check(!r4.dirty, 'loading reference meshes never marks the project unsaved');
    check(JSON.stringify(r4.rebuilt) === JSON.stringify(['Mesh', 'Points']),
        `a rebuilt 3D viewport still draws them (${JSON.stringify(r4.rebuilt)})`);

    // ===================================================================
    console.log('\n-- 5. a 2,000,000-triangle STL --');
    await page.evaluate(async () => {
        // Collapse the 3D view first: what is timed is reading the file. Headless
        // runners draw in software, where any 2M-triangle frame is slow.
        const AS = await import('/ui/app-state.js');
        AS.viewport3d.setVisible(false);
        window.__gapMax = 0; window.__gapLast = performance.now(); window.__statusSeen = [];
        window.__gapTimer = setInterval(() => {
            const now = performance.now();
            window.__gapMax = Math.max(window.__gapMax, now - window.__gapLast);
            window.__gapLast = now;
        }, 20);
        const st = document.getElementById('loadingStatus');
        window.__mo = new MutationObserver(() => window.__statusSeen.push(st.textContent));
        window.__mo.observe(st, { childList: true, characterData: true, subtree: true });
    });
    const t0 = Date.now();
    await pick(STL_BIG, () => { document.getElementById('menuLoadReferenceMesh').click(); });
    const ms = Date.now() - t0;
    const r5 = await page.evaluate(async () => {
        clearInterval(window.__gapTimer);
        window.__mo.disconnect();
        const R = await import('/ui/reference-meshes.js');
        const e = R.referenceMeshState.entries[R.referenceMeshState.entries.length - 1];
        return { gap: window.__gapMax, seen: window.__statusSeen, tri: e.mesh.triangleCount,
                 pos: e.mesh.positions.length, status: document.getElementById('statusText').textContent };
    });
    check(r5.tri === BIG_N && r5.pos === 9 * BIG_N, `all ${BIG_N.toLocaleString('en-US')} triangles loaded (${r5.status})`);
    check(r5.seen.some(t => /triangles…$/.test(t)), `the progress card counted triangles (${r5.seen.find(t => /triangles/.test(t))})`);
    check(r5.gap < 2500, `the page never froze for long: longest gap ${Math.round(r5.gap)} ms (whole load ${ms} ms)`);
} catch (err) {
    console.error('\nFATAL', err);
    fails++;
} finally {
    if (browser) await browser.close();
    server.kill();
    try { fs.rmSync(tmp, { recursive: true, force: true }); } catch { /* ignore */ }
}

console.log(fails ? `\n${fails} FAILED` : '\nPASSED');
process.exit(fails ? 1 : 0);
