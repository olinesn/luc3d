/**
 * behavior-kpms-roundtrip.mjs — the keypoint-MoSeq round trip in the real
 * app: Export for keypoint-MoSeq…, a results file, Import keypoint-MoSeq
 * Results…, the States section and timeline rows, promoting syllables to
 * predictions, then save -> reopen.
 *
 * The session is "synth" with animals id_1 and id_2 over 300 frames — the
 * recordings of tests/fixtures/kpms, whose results.h5 a real keypoint-MoSeq
 * 0.8.4 wrote — so the export's names and the fixture's agree.
 *
 *  1. Export (the browser without a folder picker: downloads): a pose table
 *     per animal in anipose's layout, every row a frame, z the height above
 *     the floor, and luc3d-kpms.yml naming both recordings with their frames.
 *     Esc closes the dialog.
 *  2. Import through the dialog's file chooser (results.h5 + the exported
 *     luc3d-kpms.yml): both recordings matched by the manifest; Import stores
 *     the syllables as runs that give back the fixture's syllables frame by
 *     frame; the timeline grows a "States" group, one row per animal.
 *  3. The States section: a row per syllable; a name typed in it lands in the
 *     layer; its button goes to its next run; labels show up as
 *     P(behavior | syllable); Promote makes the checked syllables "rear"
 *     predictions (source states:keypoint-MoSeq), listed under Predictions,
 *     out of date once the layer is imported again (not when a syllable is
 *     named).
 *  4. Save (eager writer) -> reopen through the importer's reader: the layer,
 *     its names and the promoted run come back exactly. Without states the
 *     key is not written.
 *  5. An ID-switch fix carries the states; Rename Identity renames them; a
 *     pass that hands names out afresh asks first on a session with states.
 *  6. Choose folder… on the folder exported to, after keypoint-MoSeq ran
 *     there: results.h5 and the CSV results folder are offered (never
 *     checkpoint.h5 or the pose tables), syll_info.csv names syllables, and
 *     the CSV results give the same syllables as results.h5.
 *
 * Run: node tests/e2e/behavior-kpms-roundtrip.mjs   (spawns its own http.server; PORT= to move it)
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8231);
const FIX = path.join(repoRoot, 'tests/fixtures/kpms');

let fails = 0;
const check = (c, m) => { console.log((c ? '  ✓ ' : '  ✗ ') + m); if (!c) fails++; };
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1200));
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'lucid-kpms-'));

let browser;
try {
    browser = await chromium.launch();
    const context = await browser.newContext({ acceptDownloads: true });
    const page = await context.newPage();
    const errs = [];
    page.on('pageerror', e => errs.push('pageerror: ' + String(e).slice(0, 300)));
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state, null, { timeout: 30000 });

    // A session like the fixture's: "synth", id_1 and id_2, 300 frames of 3D walking on a z = 0 floor.
    await page.evaluate(async () => {
        const pd = await import('/pose/pose-data.js');
        const AS = await import('/ui/app-state.js');
        const NODES = ['nose', 'head', 'ear L', 'ear R', 'spine', 'tail_base', 'tail_tip'];
        const ALONG = [40, 28, 26, 26, 5, -25, -60], SIDE = [0, 0, 7, -7, 0, 0, 0], H = [15, 18, 22, 22, 20, 12, 4];
        const T = 300;
        // Two cameras 1 m above the floor: the floor fit takes "up" from them.
        const K = [[600, 0, 320], [0, 600, 240], [0, 0, 1]];
        const cams = [[0, 0, -1000], [-500, 0, -1000]].map((t, i) => new pd.Camera('cam' + i, K, [0, 0, 0, 0, 0], [0, 0, 0], t, [640, 480]));
        const session = new pd.Session(cams, new pd.Skeleton('m', NODES, [[0, 1], [1, 2], [1, 3], [1, 4], [4, 5], [5, 6]]), [], 'synth');
        ['id_1', 'id_2'].forEach((nm) => session.addIdentity(nm));
        let gid = 1;
        for (let t = 0; t < T; t++) {
            session.instanceGroups.set(t, [0, 1].map((k) => {
                const g = new pd.InstanceGroup(gid++, session.identities[k].id);
                const ang = 0.02 * t + 2 * k, cx = 300 * k + 150 * Math.cos(ang), cy = 150 * Math.sin(ang), hd = ang + Math.PI / 2;
                const p = new Float64Array(NODES.length * 3);
                for (let n = 0; n < NODES.length; n++) {
                    p[3 * n] = cx + ALONG[n] * Math.cos(hd) - SIDE[n] * Math.sin(hd);
                    p[3 * n + 1] = cy + ALONG[n] * Math.sin(hd) + SIDE[n] * Math.cos(hd);
                    p[3 * n + 2] = H[n] + 2 * Math.sin(0.3 * t + n);
                }
                g.points3d = p;
                return g;
            }));
        }
        AS.state.sessions = [session]; AS.state.activeSessionIdx = 0; AS.state.session = session;
        AS.state.totalFrames = T; AS.state.currentFrame = 0; AS.state.views = []; AS.state.fps = 30;
        AS.timeline.setTotalFrames(T); AS.timeline.setData(session);
        (await import('/ui/info-panel.js')).updateInfoPanel();
        document.querySelector('.panel-tab[data-tab="tabBehavior"]').click();
    });

    console.log('\n-- 1. Export for keypoint-MoSeq (downloads) --');
    await page.evaluate(() => { window.showDirectoryPicker = undefined; });
    await page.evaluate(async () => (await import('/ui/behavior-kpms-modal.js')).showKpmsExportModal());
    check(await page.isVisible('.kpms-export-modal'), 'the dialog opens');
    await page.keyboard.press('Escape');
    check(!(await page.isVisible('.kpms-export-modal')), 'Esc closes it');
    await page.evaluate(async () => (await import('/ui/behavior-kpms-modal.js')).showKpmsExportModal());
    const bodyLine = await page.textContent('#kxBody');
    check(/models 6 of 7 nodes \(tail_tip left out/.test(bodyLine) && /heading from nose to tail_base/.test(bodyLine) && /ear L → ear_L/.test(bodyLine),
        `it says what keypoint-MoSeq will model (${bodyLine.slice(0, 160)}…)`);
    const downloads = [];
    page.on('download', d => downloads.push(d));
    await page.click('#kxGo');
    await page.waitForFunction(() => /Exported 2 pose tables/.test(document.getElementById('statusText').textContent), null, { timeout: 30000 });
    for (let i = 0; i < 50 && downloads.length < 3; i++) await new Promise(r => setTimeout(r, 100));
    const files = {};
    for (const d of downloads) {
        const p = path.join(tmp, d.suggestedFilename());
        await d.saveAs(p);
        files[d.suggestedFilename()] = fs.readFileSync(p, 'utf8');
    }
    check(eq(Object.keys(files).sort(), ['luc3d-kpms.yml', 'synth__id_1.csv', 'synth__id_2.csv']), `the files: ${Object.keys(files).sort().join(', ')}`);
    const csv = (files['synth__id_1.csv'] || '').trim().split('\n');
    const header = csv[0] || '';
    check([...header.matchAll(/(\w+)_x,\1_y,\1_z/g)].map(m => m[1]).join(',') === 'nose,head,ear_L,ear_R,spine,tail_base,tail_tip' && header.endsWith(',fnum'),
        'anipose columns keypoint-MoSeq\'s pattern finds, then fnum');
    check(csv.length === 301 && csv[300].endsWith(',299'), `a row per frame (${csv.length - 1})`);
    // Columns: 4 per node (x, y, z, score); the tail tip is node 6, the head node 1.
    const tailZ = csv.slice(1).map(l => Number(l.split(',')[26]));
    const headZ = csv.slice(1).map(l => Number(l.split(',')[6]));
    const zr = (v) => Math.min(...v).toFixed(1) + '–' + Math.max(...v).toFixed(1);
    // Built 2–6 (tail tip) and 16–20 (head) above z = 0; the fitted floor lies on the lowest points.
    check(tailZ.every(z => Math.abs(z) < 4) && headZ.every((z, i) => z - tailZ[i] > 9 && z - tailZ[i] < 19),
        `z is the height above the floor: tail tip ${zr(tailZ)}, head ${zr(headZ)}`);
    const yml = files['luc3d-kpms.yml'] || '';
    check(/^use_bodyparts: \["nose", "head", "ear_L", "ear_R", "spine", "tail_base"\]$/m.test(yml) &&
        /^anterior_bodyparts: \["nose"\]$/m.test(yml) && /^posterior_bodyparts: \["tail_base"\]$/m.test(yml) && /^fps: 30$/m.test(yml),
        'luc3d-kpms.yml: use_bodyparts without the tail tip, nose → tail base, 30 fps');
    check(/"synth__id_2": \{"session": "synth", "animal": "id_2", "first_frame": 0, "frames": 300\}/.test(yml), 'and each recording\'s session, animal and frames');
    fs.writeFileSync(path.join(tmp, 'luc3d-kpms.yml'), yml);

    console.log('\n-- 2. Import keypoint-MoSeq Results… (file chooser) --');
    await page.evaluate(async () => (await import('/ui/behavior-kpms-modal.js')).showKpmsImportModal());
    const [chooser] = await Promise.all([page.waitForEvent('filechooser'), page.click('#kiFiles')]);
    await chooser.setFiles([path.join(FIX, 'results.h5'), path.join(tmp, 'luc3d-kpms.yml')]);
    await page.waitForSelector('.kpms-recordings tbody tr', { timeout: 30000 });
    const table = await page.$$eval('.kpms-recordings tbody tr', trs => trs.map(tr => [tr.cells[0].textContent, tr.cells[1].textContent,
        tr.querySelector('select').selectedOptions[0].textContent, tr.querySelector('input').value, tr.className]));
    check(eq(table, [['synth__id_1', '300', 'synth · id_1', '1', ''], ['synth__id_2', '300', 'synth · id_2', '1', '']]),
        `both recordings matched to their animals, from frame 1 (${JSON.stringify(table)})`);
    check(/2 recordings into 1 session/.test(await page.textContent('#kiSummary')), 'the summary counts them');
    await page.click('#kiGo');
    await page.waitForFunction(() => /Imported keypoint-MoSeq syllables/.test(document.getElementById('statusText').textContent), null, { timeout: 30000 });
    check(!(await page.isVisible('.kpms-import-modal')), 'Import closes the dialog');
    // The fixture's syllables, frame by frame, from its CSV results.
    const expected = {};
    for (const a of ['id_1', 'id_2']) {
        expected[a] = fs.readFileSync(path.join(FIX, 'results', 'synth__' + a + '.csv'), 'utf8').trim().split('\n').slice(1).map(l => Number(l.split(',')[0]));
    }
    const imported = await page.evaluate(async () => {
        const S = await import('/pose/behavior-states.js');
        const AS = await import('/ui/app-state.js');
        const L = S.stateLayer(AS.state.session, 'keypoint-MoSeq');
        const out = {};
        for (const a of ['id_1', 'id_2']) { out[a] = []; for (let f = 0; f < 300; f++) out[a].push(S.stateAt(L, a, f)); }
        const rows = AS.timeline._trackSegments.filter(r => r._isBehavior).map(r => [r.cameraName, r.trackName, r.segments.length]);
        return { source: L && L.source, out, runs: Array.from(L.animals.values()).map(r => r.length / 3), rows, dirty: AS.state.isDirty };
    });
    check(imported.source === 'keypoint-MoSeq' && eq(imported.out, expected), 'the stored runs give back the fixture\'s syllables on every frame');
    check(eq(imported.rows.map(r => r.slice(0, 2)), [['States', 'id_1'], ['States', 'id_2']]) && eq(imported.rows.map(r => r[2]), imported.runs),
        `the timeline's "States" group: a row per animal, a segment per run (${JSON.stringify(imported.rows)})`);
    check(imported.dirty === true, 'the project is marked changed');

    console.log('\n-- 3. The States section --');
    await page.waitForSelector('.behavior-state-row', { timeout: 10000 });
    const syll = await page.$$eval('.behavior-state-row', rows => rows.map(r => Number(r.dataset.state)));
    const allStates = [...new Set(expected.id_1.concat(expected.id_2))].sort((a, b) => a - b);
    check(syll.length > 0 && syll.every(k => allStates.includes(k)), `a row per syllable (${syll.join(', ')} of ${allStates.join(', ')})`);
    const k0 = syll[0];
    await page.fill(`.behavior-state-row[data-state="${k0}"] input[data-act="name"]`, 'walk');
    await page.press(`.behavior-state-row[data-state="${k0}"] input[data-act="name"]`, 'Enter');
    await page.waitForFunction((k) => /Named syllable/.test(document.getElementById('statusText').textContent), k0, { timeout: 5000 });
    const named = await page.evaluate(async (k) => {
        const S = await import('/pose/behavior-states.js');
        return S.stateName(S.stateLayer((await import('/ui/app-state.js')).state.session, 'keypoint-MoSeq'), k);
    }, k0);
    check(named === 'walk', 'a name typed in the table lands in the layer');
    // The syllable's button: its next run after the playhead (any animal).
    const starts = [];
    for (const a of ['id_1', 'id_2']) expected[a].forEach((k, f) => { if (k === k0 && (f === 0 || expected[a][f - 1] !== k0)) starts.push(f); });
    const from = await page.evaluate(async () => (await import('/ui/app-state.js')).state.currentFrame);
    const wantNext = starts.filter(f => f > from).sort((a, b) => a - b)[0] ?? Math.min(...starts);
    await page.click(`.behavior-state-row[data-state="${k0}"] button[data-act="next"]`);
    await page.waitForFunction((f) => window.__lucid.state.currentFrame === f, wantNext, { timeout: 5000 }).catch(() => {});
    const landed = await page.evaluate(async () => (await import('/ui/app-state.js')).state.currentFrame);
    check(landed === wantNext && /Syllable \d+ · walk: id_\d, frames/.test(await page.textContent('#statusText')),
        `its button goes to its next run after the playhead (frame ${landed}, want ${wantNext})`);
    // Label "rear" on id_1's first 30 frames present and the next 30 "not".
    const rearId = await page.evaluate(async () => {
        const eth = await import('/pose/behavior-ethogram.js');
        const L = await import('/pose/behavior-labels.js');
        const AS = await import('/ui/app-state.js');
        const rear = eth.addBehavior({ name: 'rear', key: 'r' }).behavior;
        L.labelRange(AS.state.session, rear.id, 'id_1', '', 0, 29, true);
        L.labelRange(AS.state.session, rear.id, 'id_1', '', 30, 59, false);
        (await import('/ui/behavior-coding.js')).requestBehaviorRepaint({ ethogram: true, labels: true });
        return rear.id;
    });
    await page.waitForFunction(() => /rear \d+% \(\d+\/\d+\)/.test(document.querySelector('.behavior-states-list').textContent), null, { timeout: 5000 });
    // Expected P(rear | syllable) from the fixture's syllables.
    const want = new Map();
    expected.id_1.forEach((k, f) => {
        if (f >= 60) return;
        const c = want.get(k) || { p: 0, l: 0 };
        c.l++; if (f < 30) c.p++;
        want.set(k, c);
    });
    const cells = await page.$$eval('.behavior-state-row', rows => rows.map(r => [Number(r.dataset.state), r.querySelector('.behavior-state-labels').textContent]));
    check(cells.every(([k, txt]) => {
        const c = want.get(k);
        return c ? txt === 'rear ' + Math.round(100 * c.p / c.l) + '% (' + c.p + '/' + c.l + ')' : txt === 'no labels in it';
    }), `labels as P(rear | syllable) with the frames behind it (${cells.map(c => c.join(': ')).join('; ')})`);
    // Promote the syllables of id_1's first frames to "rear".
    const pick = [...new Set(expected.id_1.slice(0, 30))].filter(k => syll.includes(k));
    for (const k of pick) await page.check(`.behavior-state-row[data-state="${k}"] input[data-act="pick"]`);
    await page.selectOption('.behavior-states-promote select', rearId);
    await page.click('.behavior-states-promote button[data-act="promote"]');
    await page.waitForFunction(() => /Promoted syllable/.test(document.getElementById('statusText').textContent), null, { timeout: 5000 });
    const run = await page.evaluate(async (id) => {
        const P = await import('/pose/behavior-predictions.js');
        const S = await import('/pose/behavior-states.js');
        const AS = await import('/ui/app-state.js');
        const r = P.behaviorRunOf(AS.state.session, id);
        const L = S.stateLayer(AS.state.session, 'keypoint-MoSeq');
        return { source: r.source, states: r.states, bouts: r.bouts.map(b => [b.animal, b.start, b.end]),
                 want: S.statesToBouts(L, r.states).map(b => [b.animal, b.start, b.end]),
                 listed: document.getElementById('behaviorPredictionsCount').textContent };
    }, rearId);
    check(run.source === 'states:keypoint-MoSeq' && eq(run.states, [...pick].sort((a, b) => a - b)) && run.bouts.length > 0 && eq(run.bouts, run.want),
        `Promote: the checked syllables' runs are "rear" predictions (${run.bouts.length} bouts, states ${run.states.join(', ')})`);
    check(Number(run.listed.replace(/,/g, '')) > 0, `listed under Predictions (${run.listed})`);
    // Promoted states go stale when their layer changes — not when a syllable is named.
    const stale = await page.evaluate(async () => {
        const S = await import('/pose/behavior-states.js');
        const AS = await import('/ui/app-state.js');
        const s = AS.state.session;
        const frame = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
        const banner = () => { const b = document.querySelector('#behaviorPredictionsBody .behavior-stale'); return b ? b.textContent : ''; };
        const saved = S.exportStateLayers(s);
        S.setStateName(s, 'keypoint-MoSeq', 0, 'zero'); await frame();
        const afterName = banner();
        const L = S.stateLayer(s, 'keypoint-MoSeq');
        const animals = {};
        L.animals.forEach((r, a) => { const c = Array.from(r); if (a === 'id_1' && c[1] > c[0]) c[1]--; animals[a] = c; });
        S.setStateLayer(s, { name: 'keypoint-MoSeq', source: 'keypoint-MoSeq', animals }); await frame();
        const afterImport = banner();
        const button = !!document.querySelector('#behaviorPredictionsBody .behavior-stale button[data-act="detect-all"]');
        S.importStateLayers(s, saved);
        (await import('/ui/behavior-coding.js')).requestBehaviorRepaint({ states: true, predictions: true }); await frame();
        return { afterName, afterImport, button, restored: banner() };
    });
    check(stale.afterName === '' && /the states rear came from changed \(promote them again under States\)/.test(stale.afterImport) && !stale.button,
        `naming a syllable leaves the promoted predictions as they were; another import of the layer marks them out of date (${stale.afterImport.slice(0, 120)})`);
    check(stale.restored === '', 'and the layer as it was makes them current again');

    console.log('\n-- 4. Save -> reopen --');
    const rt = await page.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        const fileio = await import('/import-export/file-io.js');
        const meta = await import('/import-export/behavior-metadata.js');
        const S = await import('/pose/behavior-states.js');
        const P = await import('/pose/behavior-predictions.js');
        const pd = await import('/pose/pose-data.js');
        const s = AS.state.session;
        const views = [{ name: 'cam0', videoWidth: 256, videoHeight: 256, frameCount: 300 }];
        const vfs = [{ name: 'cam0', assignedCamera: 'cam0', videoPath: 'cam0.mp4' }];
        AS.state.views = views; AS.state.videoFiles = vfs;
        const before = { states: S.exportStateLayers(s), runs: P.exportPredictionRuns(s) };
        const live = {};
        meta.writeBehaviorMetadata(live, s);
        const bytes = await window.SleapIO.saveSlpToBytes(fileio.buildSlpLabelsAllViews(s, views, vfs));
        const slp = await fileio.parseSlpViaSleapIO(new File([bytes], 'kpms.slp'), () => {});
        meta.resetBehaviorState();          // as every load does
        const lucid = slp.sessions[0].metadata.lucid;
        const fresh = new pd.Session([], s.skeleton, [], 'reopened');
        meta.readBehaviorMetadata(fresh, lucid);
        const after = { states: S.exportStateLayers(fresh), runs: P.exportPredictionRuns(fresh) };
        // And a session with no states writes no key.
        const bare = new pd.Session([], s.skeleton, [], 'bare');
        const l2 = {};
        meta.writeBehaviorMetadata(l2, bare);
        // The live session again, as before the "load", for the next checks.
        meta.resetBehaviorState();
        meta.readBehaviorMetadata(s, JSON.parse(JSON.stringify(live)));
        return { before, after, hasKey: 'behaviorStates' in lucid, bareKey: 'behaviorStates' in l2 };
    });
    check(rt.hasKey && eq(rt.after.states, rt.before.states) && rt.before.states.layers[0].stateNames[k0] === 'walk',
        'the layer — every run and the name — reads back exactly');
    check(eq(rt.after.runs, rt.before.runs) && rt.after.runs.runs.some(r => r.source === 'states:keypoint-MoSeq'), 'and the promoted predictions');
    check(!rt.bareKey, 'a session without states writes no behaviorStates key');

    console.log('\n-- 5. Identities --');
    const ident = await page.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        const S = await import('/pose/behavior-states.js');
        const meta = await import('/import-export/behavior-metadata.js');
        const s = AS.state.session;
        const L0 = S.stateLayer(s, 'keypoint-MoSeq');
        if (!L0) return { missing: true };
        const at = (L, a) => { const o = []; for (let f = 0; f < 300; f++) o.push(S.stateAt(L, a, f)); return o; };
        const a0 = at(L0, 'id_1'), b0 = at(L0, 'id_2');
        const differ = a0.some((k, f) => f >= 100 && f <= 149 && k !== b0[f]);
        s.swapIdentitiesInRange(100, 149, s.identities[0].id, s.identities[1].id);
        const L1 = S.stateLayer(s, 'keypoint-MoSeq');
        const a1 = at(L1, 'id_1');
        const swapped = a1.every((k, f) => k === (f >= 100 && f <= 149 ? b0[f] : a0[f]));
        s.swapIdentitiesInRange(100, 149, s.identities[0].id, s.identities[1].id);
        const undone = JSON.stringify(at(S.stateLayer(s, 'keypoint-MoSeq'), 'id_1')) === JSON.stringify(a0);
        return { swapped, undone, differ };
    });
    check(!ident.missing && ident.differ && ident.swapped && ident.undone,
        'an ID-switch fix swaps the two animals\' syllables on its frames, and its Undo swaps them back');
    const beforeRename = await page.evaluate(async () => {
        const S = await import('/pose/behavior-states.js');
        return Array.from(S.stateLayer((await import('/ui/app-state.js')).state.session, 'keypoint-MoSeq').animals.get('id_1'));
    });
    await page.evaluate(() => document.getElementById('menuRenameIdentity').click());
    await page.click('.rename-list >> text=id_1');
    await page.fill('#renameNameInput', 'Alice');
    await page.click('#renameApply');
    const renamed = await page.evaluate(async () => {
        const AS = await import('/ui/app-state.js');
        const S = await import('/pose/behavior-states.js');
        const L = S.stateLayer(AS.state.session, 'keypoint-MoSeq');
        await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
        return { animals: S.layerAnimals(L), runs: Array.from(L.animals.get('Alice') || []),
                 rows: AS.timeline._trackSegments.filter(r => r._isBehavior && r.cameraName === 'States').map(r => r.trackName) };
    });
    check(eq(renamed.animals, ['Alice', 'id_2']) && eq(renamed.runs, beforeRename), 'Rename Identity renames the animal\'s syllables');
    check(eq(renamed.rows, ['Alice', 'id_2']), `and its States row (${renamed.rows.join(', ')})`);
    // A pass that hands names out afresh asks first on a session with states, labels or not.
    const asked = page.evaluate(async () => {
        const pd = await import('/pose/pose-data.js');
        const S = await import('/pose/behavior-states.js');
        const C = await import('/ui/behavior-confirm.js');
        const s2 = new pd.Session([], new pd.Skeleton('m', ['n0'], []), [], 'states only');
        s2.addIdentity('id_9');
        S.setStateLayer(s2, { name: 'keypoint-MoSeq', animals: { id_9: [[0, 9, 1]] } }, { silent: true });
        return C.confirmRetrackWithBehaviorLabels(s2, 'Track All');
    });
    await page.waitForSelector('#behaviorRetrackConfirm', { timeout: 5000 });
    const askText = await page.textContent('#behaviorRetrackConfirm');
    await page.keyboard.press('Escape');
    check(/Behavioral states follow identity names/.test(askText) && /Animals with keypoint-MoSeq states\s*1/.test(askText) && (await asked) === false,
        'Track All on a session with states (no labels) asks first, and Esc cancels it');

    console.log('\n-- 6. Import from a folder (Choose folder…) --');
    // The folder you exported to, after keypoint-MoSeq ran there: a stand-in
    // directory handle (a real picker cannot be driven from a test).
    await page.evaluate(async (yml) => {
        const get = async (p) => new Uint8Array(await (await fetch('/tests/fixtures/kpms/' + p)).arrayBuffer());
        const file = (name, bytes, t) => new File([bytes], name, { lastModified: t });
        const dir = (name, entries) => ({ kind: 'directory', name, values: async function* () { yield* entries; } });
        const fh = (f) => ({ kind: 'file', name: f.name, getFile: async () => f });
        const model = dir('2026_10_30-00_25_57', [
            fh(file('checkpoint.h5', new Uint8Array(4096).fill(7), 3)),
            fh(file('results.h5', await get('results.h5'), 2)),
            fh(file('syll_info.csv', 'syllable,label,short_description,movie_path\n0,walk,,\n2,"rear, high",,\n', 2)),
            dir('results', [fh(file('synth__id_1.csv', await get('results/synth__id_1.csv'), 1)), fh(file('synth__id_2.csv', await get('results/synth__id_2.csv'), 1))]),
        ]);
        const root = dir('export', [
            fh(file('luc3d-kpms.yml', yml, 1)),
            fh(file('synth__id_1.csv', 'nose_x,nose_y,nose_z,nose_score,fnum\n1,2,3,1.0,0\n', 1)),
            dir('kpms_project', [fh(file('config.yml', 'x: 1\n', 1)), model]),
        ]);
        window.showDirectoryPicker = async () => root;
        (await import('/ui/behavior-kpms-modal.js')).showKpmsImportModal();
    }, yml);
    await page.click('#kiFolder');
    await page.waitForSelector('.kpms-recordings tbody tr', { timeout: 30000 });
    const sources = await page.$$eval('#kiSource option', (o) => o.map((x) => x.textContent));
    check(eq(sources, ['kpms_project/2026_10_30-00_25_57/results.h5', 'kpms_project/2026_10_30-00_25_57/results/ (2 CSV files)']),
        `the folder offers results.h5 (newest first) and the results folder — not checkpoint.h5, nor the pose tables (${sources.join(' | ')})`);
    check(/syll_info\.csv names 2 syllables/.test(await page.textContent('#kiBody')), 'and reads syllable names from syll_info.csv beside them');
    // id_1 was renamed Alice (§5): its recording no longer matches, so the dialog asks.
    const first = await page.$eval('.kpms-recordings tbody tr', (tr) => [tr.className, tr.querySelector('select').value]);
    check(first[0] === 'kpms-unmatched' && first[1] === '-1', `a recording whose animal was renamed is unmatched: Skip until chosen (${first.join(', ')})`);
    await page.$eval('.kpms-recordings', (t) => { t.dataset.old = '1'; });
    await page.selectOption('#kiSource', '1');
    await page.waitForFunction(() => { const t = document.querySelector('.kpms-recordings'); return t && !t.dataset.old; }, null, { timeout: 30000 });
    const alice = await page.$eval('.kpms-recordings tbody tr select', (sel) => Array.from(sel.options).find((o) => /Alice/.test(o.textContent)).value);
    await page.selectOption('.kpms-recordings tbody tr select >> nth=0', alice);
    await page.waitForFunction(() => /2 recordings into 1 session/.test((document.getElementById('kiSummary') || {}).textContent || ''), null, { timeout: 30000 });
    await page.fill('#kiLayer', 'from CSV');
    await page.click('#kiGo');
    await page.waitForFunction(() => /Imported keypoint-MoSeq syllables "from CSV"/.test(document.getElementById('statusText').textContent), null, { timeout: 30000 });
    const viaCsv = await page.evaluate(async () => {
        const S = await import('/pose/behavior-states.js');
        const s = (await import('/ui/app-state.js')).state.session;
        const a = S.stateLayer(s, 'from CSV'), b = S.stateLayer(s, 'keypoint-MoSeq');
        const runs = (L, n) => Array.from(L.animals.get(n) || []);
        return { same: JSON.stringify(runs(a, 'Alice')) === JSON.stringify(runs(b, 'Alice')) && JSON.stringify(runs(a, 'id_2')) === JSON.stringify(runs(b, 'id_2')),
                 alice: runs(a, 'Alice').length, names: [S.stateName(a, 0), S.stateName(a, 2)], shown: document.querySelector('.behavior-states-head select').value };
    });
    check(viaCsv.same && viaCsv.alice > 0, 'the CSV results give the same syllables as results.h5');
    check(eq(viaCsv.names, ['walk', 'rear, high']), `names from syll_info.csv (${viaCsv.names.join(' / ')})`);
    check(viaCsv.shown === 'from CSV', 'and the States section shows the layer just imported');

    check(errs.length === 0, `no page errors (${errs.join(' | ').slice(0, 300)})`);
    console.log(fails === 0 ? '\nPASS' : `\nFAIL — ${fails} check(s)`);
    await browser.close();
    browser = null;
    process.exitCode = fails === 0 ? 0 : 1;
} finally {
    if (browser) await browser.close().catch(() => {});
    server.kill();
    fs.rmSync(tmp, { recursive: true, force: true });
}
