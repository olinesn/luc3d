/**
 * behavior-import-export.mjs — Behavior ▸ Export Labels… and Import Labels…
 * in the real app: the menus, the dialogs, real downloads, the file chooser.
 *
 * The file layouts and the frame/second arithmetic are pinned by the Node test
 * `tests/test-behavior-io.mjs`; this pins the app around them:
 *
 *  1. Export ▸ Bouts downloads one CSV named after the session, a row per bout.
 *  2. The dialog remembers its choices, and Esc closes it.
 *  3. Export ▸ Per-frame table writes one file per labelled animal into the
 *     picked folder (showDirectoryPicker, stubbed), a row per frame.
 *  4. Export ▸ BORIS tabular events, then Import of that file into the emptied
 *     session: the dialog maps each subject to its animal, says how many bouts
 *     it labels, and Import brings every present bout back frame for frame;
 *     the Behavior tab's Undo takes the whole import back in one step.
 *  5. Import of a BORIS project whose subjects match no animal: they start as
 *     Skip, the summary follows the selects, and Import adds the project's
 *     behaviors (exclusions as a group, social from modifiers) and its labels.
 *  6. Edit Ethogram ▸ Import takes a BORIS project's behaviors.
 *
 * Run: node tests/e2e/behavior-import-export.mjs   (spawns its own http.server; PORT= to move it)
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8195);

let fails = 0;
const check = (c, m) => { console.log((c ? '  ✓ ' : '  ✗ ') + m); if (!c) fails++; };
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1200));

let browser;
try {
    browser = await chromium.launch();
    const context = await browser.newContext({ acceptDownloads: true, viewport: { width: 1440, height: 900 } });
    const page = await context.newPage();
    const errs = [];
    page.on('pageerror', e => errs.push('pageerror: ' + String(e).slice(0, 300)));
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state, null, { timeout: 20000 });

    // A camera-less session of three animals, an ethogram, and labels of every kind.
    await page.evaluate(async () => {
        const pd = await import('/pose/pose-data.js');
        const AS = await import('/ui/app-state.js');
        const eth = await import('/pose/behavior-ethogram.js');
        const L = await import('/pose/behavior-labels.js');
        const T = 600;
        const session = new pd.Session([], new pd.Skeleton('m', ['Nose', 'Tail'], [[0, 1]]), [], 'BehaviorIO');
        for (let i = 0; i < 3; i++) session.addIdentity('id_' + i);
        AS.state.sessions = [session]; AS.state.activeSessionIdx = 0; AS.state.session = session;
        AS.state.totalFrames = T; AS.state.currentFrame = 0; AS.state.views = []; AS.state.fps = 30;
        AS.timeline.setTotalFrames(T); AS.timeline.setData(session);
        const b = {
            rear: eth.addBehavior({ name: 'rear', key: 'r' }).behavior,
            groom: eth.addBehavior({ name: 'groom', key: 'g', group: 'posture' }).behavior,
            walk: eth.addBehavior({ name: 'walk', key: 'w', group: 'posture' }).behavior,
            jump: eth.addBehavior({ name: 'jump', key: 'j', kind: 'point' }).behavior,
            chase: eth.addBehavior({ name: 'chase', key: 'c', social: true }).behavior,
            sniff: eth.addBehavior({ name: 'sniff', key: 'n', social: true, mutual: true }).behavior,
        };
        L.labelRange(session, b.rear.id, 'id_0', '', 10, 19, true, { note: 'tall, "very"' });
        L.labelRange(session, b.rear.id, 'id_0', '', 20, 29, false);
        L.labelRange(session, b.groom.id, 'id_0', '', 40, 49, true);
        L.labelRange(session, b.walk.id, 'id_0', '', 45, 60, true);
        L.labelRange(session, b.jump.id, 'id_1', '', 70, 70, true);
        L.labelRange(session, b.chase.id, 'id_0', 'id_1', 100, 110, true);
        L.labelRange(session, b.chase.id, 'id_0', 'id_2', 105, 120, true);
        L.labelRange(session, b.sniff.id, 'id_2', 'id_0', 200, 210, true);
        (await import('/ui/behavior-coding.js')).requestBehaviorRepaint({ ethogram: true, labels: true });
        (await import('/ui/info-panel.js')).updateInfoPanel();
    });
    const present = () => page.evaluate(async () => {
        const L = await import('/pose/behavior-labels.js'); const AS = await import('/ui/app-state.js');
        return L.listBouts(AS.state.session, { present: true }).map(b => [b.behaviorId, b.animal, b.target, b.start, b.end].join(' ')).sort();
    });
    const allRows = () => page.evaluate(async () => {
        const L = await import('/pose/behavior-labels.js'); const AS = await import('/ui/app-state.js');
        return L.exportBoutRows(AS.state.session).length;
    });
    const menu = async (id) => {
        await page.click('.menu-item[data-menu="behavior"]');
        await page.click('#' + id);
    };
    const status = () => page.textContent('#statusText');
    const original = await present();

    console.log('\n-- 1. Export ▸ Bouts --');
    await menu('menuBehaviorExport');
    await page.waitForSelector('.behavior-export-modal');
    await page.check('input[name="bxTable"][value="bouts"]');
    await page.selectOption('#bxSep', ',');
    let dl = page.waitForEvent('download');
    await page.click('#bxGo');
    let d = await dl;
    const boutsCsv = fs.readFileSync(await d.path(), 'utf8');
    const lines = boutsCsv.trim().split('\r\n');
    check(d.suggestedFilename() === 'BehaviorIO_behavior-bouts.csv', 'one CSV named after the session (' + d.suggestedFilename() + ')');
    check(lines[0].startsWith('session,behavior,kind,social,animal,target,present,start_frame,end_frame') && lines.length - 1 === await allRows(),
        `a row per bout, present and "not" (${lines.length - 1})`);
    check(boutsCsv.includes('"tall, ""very"""'), 'notes are quoted');
    check(!(await page.isVisible('.behavior-export-modal')) && /Exported BehaviorIO_behavior-bouts\.csv/.test(await status()), 'the dialog closes and the status says what was written');

    console.log('\n-- 2. Remembered choices, Esc --');
    await menu('menuBehaviorExport');
    await page.waitForSelector('.behavior-export-modal');
    check(await page.isChecked('input[name="bxTable"][value="bouts"]') && await page.inputValue('#bxSep') === ',', 'the last table and separator are chosen again');
    await page.keyboard.press('Escape');
    check(!(await page.isVisible('.behavior-export-modal')), 'Esc closes the dialog');

    console.log('\n-- 3. Export ▸ Per-frame table --');
    await page.evaluate(() => {
        window.__written = {};
        window.showDirectoryPicker = async () => ({
            name: 'labels',
            getFileHandle: async (name) => ({
                createWritable: async () => {
                    let text = '';
                    return { write: async (t) => { text += t; }, close: async () => { window.__written[name] = text; } };
                },
            }),
        });
    });
    await menu('menuBehaviorExport');
    await page.waitForSelector('.behavior-export-modal');
    await page.check('input[name="bxTable"][value="frames"]');
    await page.selectOption('#bxSep', 'tab');
    await page.click('#bxGo');
    await page.waitForFunction(() => Object.keys(window.__written).length >= 3, null, { timeout: 10000 });
    const written = await page.evaluate(() => window.__written);
    check(eq(Object.keys(written).sort(), ['BehaviorIO_id_0_behavior-frames.tsv', 'BehaviorIO_id_1_behavior-frames.tsv', 'BehaviorIO_id_2_behavior-frames.tsv']),
        'one TSV per labelled animal, into the picked folder');
    const t0 = written['BehaviorIO_id_0_behavior-frames.tsv'].trim().split('\r\n').map(l => l.split('\t'));
    check(t0.length === 601 && eq(t0[0].slice(0, 4), ['frame', 'time_s', 'rear', 'groom']), 'a row per frame of the session (600) under the header');
    check(t0[11][2] === '1' && t0[21][2] === '0' && t0[31][2] === '', 'rear for id_0: 1, then "not" (0), then unlabelled (blank)');
    check(/Exported 3 per-frame tables to labels/.test(await status()), 'the status names the folder');

    console.log('\n-- 4. BORIS tabular events, out and back in --');
    await menu('menuBehaviorExport');
    await page.waitForSelector('.behavior-export-modal');
    await page.check('input[name="bxTable"][value="boris-tabular"]');
    await page.selectOption('#bxSep', ',');
    dl = page.waitForEvent('download');
    await page.click('#bxGo');
    d = await dl;
    const borisCsv = fs.readFileSync(await d.path(), 'utf8');
    check(d.suggestedFilename() === 'BehaviorIO_boris-tabular.csv' && borisCsv.startsWith('Observation id,Observation date,'), 'a BORIS tabular CSV');
    await page.evaluate(async () => {
        const L = await import('/pose/behavior-labels.js'); const AS = await import('/ui/app-state.js');
        L.resetBehaviorLabels(AS.state.session);
        (await import('/ui/behavior-coding.js')).requestBehaviorRepaint({ labels: true });
    });
    check((await present()).length === 0, '(the session emptied)');
    let fc = page.waitForEvent('filechooser');
    await menu('menuBehaviorImport');
    await (await fc).setFiles({ name: 'BehaviorIO_boris-tabular.csv', mimeType: 'text/csv', buffer: Buffer.from(borisCsv) });
    await page.waitForSelector('.behavior-import-modal');
    check(/BORIS tabular events/.test(await page.textContent('.behavior-io-file')), 'the dialog names the file and its kind');
    const mapped = await page.$$eval('#biSubjects select', ss => ss.map(s => s.closest('tr').cells[0].textContent + '=' + s.value));
    check(eq(mapped.sort(), ['id_0=id_0', 'id_1=id_1']), 'each subject maps to the animal of its name (id_2 is only ever a target: ' + mapped.join(', ') + ')');
    check(/^7 bouts will be labelled\.$/.test(await page.textContent('#biSummary')), 'the summary counts the bouts (' + await page.textContent('#biSummary') + ')');
    await page.click('#biGo');
    check(eq(await present(), original), 'every present bout is back, frame for frame');
    check(/Imported BehaviorIO_boris-tabular\.csv: 7 bouts labelled/.test(await status()), 'the status says so');
    check(await page.evaluate(async () => (await import('/ui/app-state.js')).state.isDirty), 'the project is dirty');
    await page.click('.panel-tab[data-tab="tabBehavior"]');
    await page.click('#behaviorCodingBody button[data-act="undo"]');
    check((await present()).length === 0, 'one Undo takes the whole import back');

    console.log('\n-- 5. A BORIS project with unknown subjects --');
    const project = {
        project_name: 'p', project_format_version: '7.0',
        subjects_conf: { '0': { key: '', name: 'mouse A', description: '' }, '1': { key: '', name: 'mouse B', description: '' } },
        behaviors_conf: {
            '0': { type: 'State event', key: 'h', code: 'huddle', category: 'rest', excluded: 'explore', color: '', description: '', modifiers: '' },
            '1': { type: 'State event', key: 'e', code: 'explore', category: 'rest', excluded: 'huddle', color: '', description: '', modifiers: '' },
            '2': { type: 'State event', key: 'f', code: 'follow', category: '', excluded: '', color: '', description: '', modifiers: '' },
        },
        observations: {
            BehaviorIO: {
                type: 'MEDIA', file: { '1': ['cam.mp4'] }, 'time offset': 0, media_info: { fps: { 'cam.mp4': 30 } },
                events: [[1, 'mouse A', 'huddle', '', ''], [2, 'mouse A', 'huddle', '', ''], [3, 'mouse A', 'follow', 'mouse B', ''],
                         [4, 'mouse A', 'follow', 'mouse B', ''], [5, 'mouse B', 'explore', '', ''], [6, 'mouse B', 'explore', '', '']],
            },
        },
    };
    fc = page.waitForEvent('filechooser');
    await menu('menuBehaviorImport');
    await (await fc).setFiles({ name: 'p.boris', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(project)) });
    await page.waitForSelector('.behavior-import-modal');
    check(await page.isChecked('#biAdd') && /Add its 3 new behaviors/.test(await page.textContent('.behavior-import-modal')), 'offers to add its 3 behaviors');
    check(/exclusive group: rest \(huddle, explore\)/.test(await page.textContent('#biGroups')), 'and says the exclusions become a group');
    const skipAll = await page.$$eval('#biSubjects select', ss => ss.map(s => s.value));
    check(skipAll.every(v => v === ''), 'unknown subjects start as Skip');
    check(/Nothing to label\. Skipped: 3 bouts of skipped subjects\./.test(await page.textContent('#biSummary')), 'the summary says why nothing would be labelled');
    const rows = await page.$$eval('#biSubjects tr', trs => trs.map(tr => tr.cells[0].textContent));
    await page.selectOption(`#biSubjects tr:nth-child(${rows.indexOf('mouse A') + 1}) select`, 'id_0');
    await page.selectOption(`#biSubjects tr:nth-child(${rows.indexOf('mouse B') + 1}) select`, 'id_1');
    check(/^3 bouts will be labelled\.$/.test(await page.textContent('#biSummary')), 'mapping them updates the summary');
    await page.click('#biGo');
    const after = await page.evaluate(async () => {
        const eth = await import('/pose/behavior-ethogram.js'); const L = await import('/pose/behavior-labels.js'); const AS = await import('/ui/app-state.js');
        const byName = (n) => eth.behaviorByName(n);
        return {
            groups: [byName('huddle').group, byName('explore').group], social: byName('follow').social,
            bouts: L.listBouts(AS.state.session, { present: true })
                .filter(b => ['huddle', 'explore', 'follow'].includes(eth.behaviorById(b.behaviorId).name))
                .map(b => [eth.behaviorById(b.behaviorId).name, b.animal, b.target, b.start, b.end, b.source].join(' ')),
        };
    });
    check(eq(after.groups, ['rest', 'rest']) && after.social, 'the behaviors are added: huddle/explore exclusive, follow social');
    check(eq(after.bouts, ['huddle id_0  30 59 i', 'follow id_0 id_1 90 119 i', 'explore id_1  150 179 i']),
        'and the labels land on the mapped animals, the target from the modifier (' + after.bouts.join('; ') + ')');

    console.log('\n-- 6. Edit Ethogram ▸ Import of a BORIS project --');
    await page.click('#behaviorEthogramBtn');
    await page.waitForSelector('.behavior-ethogram-modal');
    fc = page.waitForEvent('filechooser');
    await page.click('#behaviorEdImport');
    const p2 = { behaviors_conf: { '0': { type: 'Point event', code: 'bite', key: 'b', excluded: '' } }, subjects_conf: {}, observations: {} };
    await (await fc).setFiles({ name: 'p2.boris', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(p2)) });
    await page.waitForFunction(() => /Added 1 behavior from p2\.boris/.test(document.getElementById('behaviorEdMsg').textContent), null, { timeout: 5000 });
    check(await page.evaluate(async () => { const b = (await import('/pose/behavior-ethogram.js')).behaviorByName('bite'); return !!b && b.kind === 'point' && b.key === 'b'; }),
        'the editor adds bite (a point, key b)');
    await page.keyboard.press('Escape');

    check(errs.length === 0, 'no page errors' + (errs.length ? ': ' + errs.join(' | ') : ''));
} catch (err) {
    console.error(err);
    fails++;
} finally {
    if (browser) await browser.close();
    server.kill();
}
console.log(fails ? `\nFAIL (${fails})` : '\nPASS');
process.exit(fails ? 1 : 0);
