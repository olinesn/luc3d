/**
 * slp-format-30.mjs — a `.slp` in SLP format 3.0 opens in LUC3D and saves without loss.
 *
 * Python sleap-io 0.10 saves 3.0 by default: every table is a group of columns. The vendored
 * reader turns those into the columns the 2.x tables have (`LUCID local patch (slp 3.0)`), so
 * a save writes 2.x. `tests/fixtures/slp-format-30/`: `luc3d_project_*` is a real project
 * trimmed to 12 frames in both layouts (3D groups, ethogram, states, bouts, predictions), and
 * `py010_*` are files Python release/0.10 wrote itself, in pairs. See `make_project_fixture.py`.
 *
 *  1. Both loaders open the 3.0 project the same as its 2.x twin, in digests and in values
 *     counted by hand.
 *  2. A save from it keeps every row by value, 3.0 column by 2.x row, and equals a save of
 *     the twin. Reopening that save gives the same project.
 *  2b. A node moved and instances deleted on a lazy 3.0 store save as on its twin.
 *  3. Every reader: the raw worker hands a 3.0 file to the typed reader, the per-camera route,
 *     File ▸ Add SLP and the single-.slp folder route read it.
 *  4. Files Python wrote: twins read the same, frames land on the right video, what a save
 *     can't keep is said on the console, images show.
 *  5. The byte count of the Open card works on columns, and a sliced read equals a whole one.
 *  6. lzf is refused everywhere with one clear message, in every layout, and that refusal
 *     changes nothing and is never a save target. So is an unknown `required_features` entry.
 *  7. 3.0 tables that contradict each other are refused, naming the table, with no retry
 *     through the raw worker.
 *  8. A file holding what a save leaves out is never overwritten: Save asks where, and the
 *     load's status says why. A real project is saved straight over its file.
 *
 * Run: node tests/e2e/slp-format-30.mjs   (spawns its own http.server; PORT=…)
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { digestSession } from './fixtures/session-digest.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8393);
const FIXTURES = '/tests/fixtures/slp-format-30/';

let fails = 0;
const check = (c, m, extra) => { console.log((c ? '  ✓ ' : '  ✗ ') + m + (c || extra === undefined ? '' : ' — ' + extra)); if (!c) fails++; };
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const lzfRefusal = (name, table, filter = 'lzf') => `${name}: /${table} is compressed with "${filter}", which LUC3D cannot read. Re-save the file from Python sleap-io with compression="gzip" or without compression.`;

const server = spawn('python3', ['-m', 'http.server', String(PORT), '--bind', '127.0.0.1'], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1200));

// What two loads of one project are compared on: all of it but the file's own name and time.
function comparable(digest) {
    const { status, heapGB, provenanceKeys, session, ...rest } = digest;
    return rest;
}
function differences(a, b, where = '') {
    if (eq(a, b)) return [];
    if (a && b && typeof a === 'object' && typeof b === 'object' && !Array.isArray(a) && !Array.isArray(b)) {
        return [...new Set([...Object.keys(a), ...Object.keys(b)])].flatMap(k => differences(a[k], b[k], where ? where + '.' + k : k));
    }
    return [where + ': ' + String(JSON.stringify(a)).slice(0, 120) + ' vs ' + String(JSON.stringify(b)).slice(0, 120)];
}

let browser;
try {
    browser = await chromium.launch({ headless: true });

    async function freshPage() {
        const context = await browser.newContext({ acceptDownloads: true, viewport: { width: 1440, height: 900 } });
        const page = await context.newPage();
        page.consoleLines = [];
        page.on('console', m => page.consoleLines.push(m.text()));
        page.on('pageerror', e => { console.log('  [pageerror]', String(e).slice(0, 300)); fails++; });
        page.on('dialog', d => d.accept());
        await page.goto(`http://127.0.0.1:${PORT}/index.html`, { waitUntil: 'domcontentloaded', timeout: 120000 });
        await page.waitForFunction(() => window.SleapIO && window.h5wasm && window.__lucid && window.__lucid.state, null, { timeout: 120000 });
        await page.evaluate((FIXTURES) => {
            window.__fixture = async (name) => new Uint8Array(await (await fetch(FIXTURES + name)).arrayBuffer());
            window.__status = () => ({
                text: document.getElementById('statusText').textContent,
                dot: document.getElementById('statusDot').className,
            });
            // What a refused load must leave exactly as it found.
            window.__snapshot = async () => {
                const AS = await import('/ui/app-state.js');
                const s = AS.state.session;
                let instances = 0;
                if (s) {
                    for (const [, fg] of s.frameGroups) {
                        for (const [, xs] of fg.instances) instances += xs.length;
                        for (const [, xs] of fg.unlinkedInstances) instances += xs.length;
                    }
                }
                return {
                    sessions: AS.state.sessions.map(x => x.name),
                    active: s ? s.name : null,
                    frameGroups: s ? s.frameGroups.size : null,
                    groups: s ? s.instanceGroups.size : null,
                    instances,
                    lazy: !!(s && s.lazyLoader),
                    views: AS.state.views.length,
                };
            };

            // A file's tables, in either layout, as { path: { column: values } } plus its
            // plain datasets. 3.0 column names become the 2.x ones.
            let n = 0;
            window.__readFile = async (bytes) => {
                const h5 = window.h5wasm;
                await h5.ready;
                const p = '/compare-' + (n++) + '.slp';
                h5.FS.writeFile(p, bytes);
                const f = new h5.File(p, 'r');
                const number = (v) => (typeof v === 'bigint' ? Number(v) : Number(v));
                const asNumbers = (a) => Float64Array.from(a, number);
                const columns = (path) => {
                    const node = f.get(path);
                    if (!node) return null;
                    const out = {};
                    if (node.type === 'Group') {
                        for (const k of node.keys()) {
                            const c = f.get(path + '/' + k);
                            if (c.type === 'Dataset' && c.shape.length === 1) out[k] = asNumbers(c.value);
                        }
                        return out;
                    }
                    const attr = node.attrs.field_names;
                    const names = JSON.parse(attr.value);
                    const flat = node.value, [rows, cols] = node.shape;
                    names.forEach((name, c) => {
                        const col = new Float64Array(rows);
                        for (let r = 0; r < rows; r++) col[r] = number(flat[r * cols + c]);
                        out[name] = col;
                    });
                    return out;
                };
                const strings = (path) => {
                    const d = f.get(path);
                    return d ? Array.from(d.value, v => (typeof v === 'string' ? v : new TextDecoder().decode(v))) : null;
                };
                const out = {
                    format: Number(f.get('metadata').attrs.format_id.value),
                    keys: f.keys(),
                    frames: columns('frames'),
                    instances: columns('instances'),
                    points: columns('points'),
                    pred_points: columns('pred_points'),
                    frame_groups: columns('session_data/frame_groups'),
                    instance_groups: columns('session_data/instance_groups'),
                    members: columns('session_data/instance_group_members'),
                    points_3d: f.get('session_data/points_3d') ? Array.from(f.get('session_data/points_3d').value) : null,
                    identityLinks: f.get('identity/links') ? columns('identity/links') : null,
                    categoryLinks: f.get('categories/links') ? columns('categories/links') : null,
                    embeddings: f.get('embeddings/vectors') ? Array.from(f.get('embeddings/vectors').value) : null,
                    group_meta: strings('session_data/instance_group_meta'),
                    tracks: strings('tracks_json'),
                    events: {},
                    eventStrings: {},
                };
                for (const k of ['video', 'start_frame', 'end_frame', 'type', 'subject_kind', 'target_kind', 'subject_idx', 'target_idx', 'is_predicted', 'score', 'meta_owner']) {
                    out.events[k] = asNumbers(f.get('events/' + k).value);
                }
                for (const k of ['name', 'source', 'meta_key', 'meta_val']) out.eventStrings[k] = strings('events/' + k);
                f.close();
                h5.FS.unlink(p);
                return out;
            };
            // What a save of a file Python wrote holds, for comparing two saves.
            window.__summary = (f) => {
                const rows = (t) => (t ? Object.keys(t).sort().map(k => [k, Array.from(t[k])]) : []);
                return {
                    frames: Array.from(f.frames.frame_idx), video: Array.from(f.frames.video), x: Array.from(f.pred_points.x), xu: Array.from(f.points.x),
                    v: Array.from(f.points.visible), p3: f.points_3d, score: Array.from(f.instances.score),
                    identityLinks: rows(f.identityLinks).filter(([k]) => k === 'owner_id' || k === 'identity_idx'),
                    categoryLinks: rows(f.categoryLinks).filter(([k]) => k === 'owner_id' || k === 'category_idx'),
                    embeddings: f.embeddings || [],
                };
            };
            const same = (u, v) => u === v || (Number.isNaN(u) && Number.isNaN(v));
            // Every saved row against the source's columns. 2D points and their scores are
            // stored 32-bit on purpose, so the source's doubles are cast first.
            window.__compareRows = (source, saved) => {
                const report = {};
                const table = (name, renames, round) => {
                    const src = source[name], dst = saved[name];
                    const rows = Object.values(dst)[0].length;
                    const r = { rows, differ: {} };
                    for (const col of Object.keys(dst)) {
                        const from = renames[col] === undefined ? col : renames[col];
                        for (let i = 0; i < rows; i++) {
                            const u = from === null ? i : src[from][i];
                            if (!same(round.includes(col) ? Math.fround(u) : u, dst[col][i])) r.differ[col] = (r.differ[col] || 0) + 1;
                        }
                    }
                    const srcRows = Object.values(src).reduce((m, c) => Math.max(m, c.length), 0);
                    if (srcRows !== rows) r.differ.rowCount = srcRows;
                    report[name] = r;
                };
                table('frames', { frame_id: null }, []);
                table('instances', { instance_id: null }, []);
                table('pred_points', { complete: 'verified' }, ['x', 'y', 'score']);
                table('frame_groups', {}, []);
                table('instance_groups', {}, []);
                table('members', {}, []);
                report.points_3d = source.points_3d.length === saved.points_3d.length && source.points_3d.every((v, i) => same(v, saved.points_3d[i]));
                report.group_meta = JSON.stringify(source.group_meta) === JSON.stringify(saved.group_meta);
                report.tracks = JSON.stringify(source.tracks) === JSON.stringify(saved.tracks);
                // Bouts as rows, in any order: trimming the fixture left some that start
                // together, and a save writes those in an order of its own.
                const bouts = (f) => {
                    const keys = Object.keys(f.events).filter(k => k !== 'meta_owner');
                    const rows = [];
                    for (let i = 0; i < f.events.video.length; i++) {
                        rows.push(JSON.stringify([...keys.map(k => String(f.events[k][i])), f.eventStrings.name[i], f.eventStrings.source[i]]));
                    }
                    return rows.sort();
                };
                report.events = JSON.stringify(bouts(source)) === JSON.stringify(bouts(saved));
                report.eventStrings = JSON.stringify(source.eventStrings.meta_val) === JSON.stringify(saved.eventStrings.meta_val);
                return report;
            };
        }, FIXTURES);
        return page;
    }

    /** Run a loader on page-global bytes, dismissing the prompts it raises. */
    async function run(page, code, globalName, fileName, size) {
        await page.evaluate(([code, g, name, size]) => {
            window.__done = false;
            window.__returned = undefined;
            const f = new File([window[g]], name);
            if (size) Object.defineProperty(f, 'size', { value: size });
            (new Function('f', 'return (async () => { ' + code + ' })()'))(f)
                .then(r => { window.__returned = r; window.__done = true; }, e => { window.__done = 'THREW ' + (e && e.stack || e); });
        }, [code, globalName, fileName, size || 0]);
        for (let i = 0; i < 480; i++) {
            const s = await page.evaluate(() => {
                const b = [...document.querySelectorAll('button')].find(x => x.offsetParent && /^(Skip|Later|Cancel|Not now)/i.test((x.textContent || '').trim()));
                if (b) { b.click(); return { clicked: true }; }
                return { done: window.__done, prompt: !!document.getElementById('videoSelectOverlay') };
            });
            if (s.prompt) await page.keyboard.press('Escape');
            if (s.done) return { done: s.done, returned: await page.evaluate(() => window.__returned) };
            await page.waitForTimeout(250);
        }
        return { done: 'timeout' };
    }
    /** File ▸ Add SLP, with its picker answered and any prompts it raises dismissed. */
    async function addSlp(page, file) {
        page.once('filechooser', fc => fc.setFiles(file));
        await page.evaluate(() => {
            window.__added = false;
            import('/import-export/slp-import.js').then(m => m.handleAddSlp()).then(() => { window.__added = true; }, e => { window.__added = 'THREW ' + e; });
        });
        for (let i = 0; i < 240; i++) {
            const s = await page.evaluate(() => {
                const b = [...document.querySelectorAll('button')].find(x => x.offsetParent && /^(Skip|Later|Not now)/i.test((x.textContent || '').trim()));
                if (b) { b.click(); return {}; }
                return { done: window.__added, prompt: !!document.getElementById('videoSelectOverlay') };
            });
            if (s.prompt) await page.keyboard.press('Escape');
            if (s.done) return s.done;
            await page.waitForTimeout(250);
        }
        return 'timeout';
    }
    const EAGER = "return (await import('/import-export/slp-import.js')).handleLoadSlpFile(f);";
    const LAZY = "return (await import('/loading/session-loader.js')).handleLoadProjectSlpLazy(f);";
    const ROUTED = "return (await import('/import-export/save-load.js')).handleLoadProject(f);";
    const OVER_200_MB = 250 * 1024 * 1024;
    const OVER_150_MB = 180 * 1024 * 1024;
    const LOADERS = [['handleLoadSlpFile', EAGER], ['handleLoadProjectSlpLazy', LAZY]];

    /** Open a fixture on a fresh page: `{ page, r, digest }`. */
    async function openFixture(name, code) {
        const page = await freshPage();
        await page.evaluate(async (name) => { window.__bytes = await window.__fixture(name); }, name);
        const r = await run(page, code, '__bytes', name);
        const digest = await page.evaluate(digestSession);
        return { page, r, digest };
    }

    /** The app's File ▸ Save, as bytes in the page (`window.__saved`). */
    async function save(page, tmpName) {
        const [dl] = await Promise.all([
            page.waitForEvent('download', { timeout: 120000 }),
            page.evaluate(async () => (await import('/import-export/save-load.js')).saveProjectSlp()),
        ]);
        const p = path.join(repoRoot, '.' + tmpName + '-' + PORT + '.tmp');
        await dl.saveAs(p);
        const b64 = fs.readFileSync(p).toString('base64');
        fs.rmSync(p, { force: true });
        await page.evaluate((b64) => {
            const bin = atob(b64);
            const out = new Uint8Array(bin.length);
            for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
            window.__saved = out;
        }, b64);
    }

    /** A per-camera folder: each camera's `.slp` is `bytesOf[camera]`. What each holds, and the status. */
    const PER_CAMERA = async ([fixtures, size]) => {
        const CAMS = ['cam1', 'cam2'];
        const mp4 = await (await fetch('/tests/fixtures/barcode-60/barcode-60.mp4')).blob();
        let toml = '';
        CAMS.forEach((cn, i) => {
            toml += '[cam_' + i + ']\nname = "' + cn + '"\nsize = [320, 240]\n'
                + 'matrix = [[600.0, 0.0, 160.0], [0.0, 600.0, 120.0], [0.0, 0.0, 1.0]]\n'
                + 'distortions = [0.0, 0.0, 0.0, 0.0, 0.0]\n'
                + 'rotation = [0.0, ' + (0.2 * i) + ', 0.0]\ntranslation = [' + (15 * i) + ', 0.0, 0.0]\n\n';
        });
        const mk = (bits, rel, sz) => {
            const f = new File(bits, rel.split('/').pop(), { type: 'application/octet-stream' });
            Object.defineProperty(f, 'webkitRelativePath', { value: rel });
            if (sz) Object.defineProperty(f, 'size', { value: sz });
            return f;
        };
        const files = [mk([toml], 'sess/calibration.toml')];
        for (let i = 0; i < CAMS.length; i++) {
            const c = CAMS[i];
            files.push(mk([mp4], 'sess/' + c + '/' + c + '.mp4'));
            files.push(mk([await window.__fixture(fixtures[i])], 'sess/' + c + '/' + c + '.slp', size));
        }
        const loader = await import('/loading/session-loader.js');
        await loader.handleLoadSessionFolderPerCamera(files, false);
        const AS = await import('/ui/app-state.js');
        const s = AS.state.session;
        const per = { cam1: 0, cam2: 0 };
        if (s.lazyLoader && s.lazyLoader.labelsByCam) {
            for (const [cam, labels] of s.lazyLoader.labelsByCam) per[cam] = labels._lazyDataStore.framesData.frame_id.length;
        } else {
            for (const [, fg] of s.frameGroups) {
                for (const [cn, xs] of fg.instances) per[cn] = (per[cn] || 0) + xs.length;
                for (const [cn, xs] of fg.unlinkedInstances) per[cn] = (per[cn] || 0) + xs.length;
            }
        }
        return { per, lazy: !!s.lazyLoader, views: AS.state.views.length, status: window.__status() };
    };

    // =================================================================
    console.log('\n-- 1. both loaders open the 3.0 project the same as its 2.x twin --');
    const twins = {};
    for (const [label, code] of LOADERS) {
        const a = await openFixture('luc3d_project_30.slp', code);
        const b = await openFixture('luc3d_project_2x.slp', code);
        check(a.r.done === true && !a.page.consoleLines.some(l => /falling back/i.test(l)), label + ': opens it, with no retry through the raw worker', JSON.stringify(a.r));
        const d = differences(comparable(a.digest), comparable(b.digest));
        check(d.length === 0, '  every 3D group, 2D pose, identity, bout, prediction and setting is the twin\'s', d.slice(0, 6).join(' | '));
        const s = a.digest;
        check(s.groups === 48 && s.with3d === 48 && s.members === 320 && s.points3d === 720, '  48 groups, all with 3D, 320 members, 720 3D points', JSON.stringify([s.groups, s.with3d, s.members, s.points3d]));
        check(s.identities.length === 4 && s.tracks === 1641 && s.numAnimals === 4, '  4 identities, 1,641 tracks, the animal count', JSON.stringify([s.identities.length, s.tracks, s.numAnimals]));
        check(s.behaviors.length === 9 && s.bouts.length === 3 && s.predicted.length === 6 && s.runs.length > 0, '  the ethogram, 3 bouts, 6 predictions and their runs', JSON.stringify([s.behaviors.length, s.bouts.length, s.predicted.length, s.runs.length]));
        check(s.states.length === 1 && s.states[0].name === 'keypoint-MoSeq' && s.states[0].animals === 4, '  the states layer', JSON.stringify(s.states));
        check(s.bouts.some(b => b[7] === 'note 0'), '  a bout keeps its note', JSON.stringify(s.bouts));
        if (code === LAZY) {
            check(s.lazy && s.pose.tables['pred_points.x'][0] === 4815 && s.pose.tables['instances.track'][0] === 321 && s.pose.tables['frames.video'][0] === 96,
                '  every row is in the store', JSON.stringify(s.pose.tables['pred_points.x']));
            check(s.pose.hydrated.resident === 12 && s.pose.hydrated.instances === 321 && eq(s.pose.hydrated.blockers, {}), '  every frame hydrates, and could be evicted again: it matches the store', JSON.stringify(s.pose.hydrated));
            // A store column in the heap would undo the memory work (`reopen-store-columns-typed`).
            const plain = await a.page.evaluate(async () => {
                const AS = await import('/ui/app-state.js');
                const store = Array.from(AS.state.session.lazyLoader.labelsByCam.values())[0]._lazyDataStore;
                const ids = store.instancesData.instance_id;
                return {
                    plain: ['framesData', 'instancesData', 'pointsData', 'predPointsData'].flatMap(t => Object.entries(store[t]).filter(([, c]) => !ArrayBuffer.isView(c)).map(([k]) => t + '.' + k)),
                    idsAreRows: ids.length === 321 && ids.every((v, i) => v === i),
                };
            });
            check(plain.plain.length === 0, '  and no store column is a plain array', JSON.stringify(plain));
            check(plain.idsAreRows, '  and each row number stands in for the id column 3.0 leaves out', JSON.stringify(plain));
        }
        twins[label] = a;
        await b.page.context().close();
    }
    {
        // File ▸ Open sends a big file to the lazy route, after a look for embedded images.
        const big = await freshPage();
        await big.evaluate(async () => { window.__bytes = await window.__fixture('luc3d_project_30.slp'); });
        const r = await run(big, ROUTED, '__bytes', 'big.slp', OVER_200_MB);
        const d = await big.evaluate(digestSession);
        const want = comparable(twins.handleLoadProjectSlpLazy.digest);
        const diff = differences(comparable(d), want);
        check(r.done === true && d.lazy && diff.length === 0, 'handleLoadProject, above 200 MB, takes the lazy route to the same project', diff.slice(0, 4).join(' | '));
        await big.context().close();
    }

    // =================================================================
    console.log('\n-- 2. a save from a 3.0 file keeps every row by value --');
    for (const [label, code] of LOADERS) {
        const { page, digest } = twins[label];
        await save(page, 'slp-format-30-save');
        const report = await page.evaluate(async () => {
            const source = await window.__readFile(await window.__fixture('luc3d_project_30.slp'));
            const saved = await window.__readFile(window.__saved);
            const twin = await window.__readFile(await window.__fixture('luc3d_project_2x.slp'));
            return {
                format: saved.format,
                keys: saved.keys,
                source: window.__compareRows(source, saved),
                sourceIsTwin: window.__compareRows(source, twin),
            };
        });
        check(report.format < 3, label + ': the save is 2.x, what SLEAP and older tools read', report.format);
        for (const t of ['frames', 'instances', 'pred_points', 'frame_groups', 'instance_groups', 'members']) {
            const r = report.source[t];
            check(Object.keys(r.differ).length === 0 && r.rows > 0, `  /${t}: all ${r.rows} rows are the source's`, JSON.stringify(r.differ));
        }
        check(report.source.points_3d === true, '  every 3D point, bit for bit');
        check(report.source.group_meta === true && report.source.tracks === true, '  group metadata and tracks');
        check(report.source.events === true && report.source.eventStrings === true, '  every bout, with its notes');
        check(report.keys.includes('identity') && report.keys.includes('event_types'), '  the identities and the catalog', JSON.stringify(report.keys));
        // The comparison itself: the fixture's own twin, in the other layout, passes it too.
        check(Object.values(report.sourceIsTwin.pred_points.differ).length === 0, '  (the twin of the source passes the same comparison)');

        // And it reopens as the same project.
        const again = await run(page, code, '__saved', 'luc3d_project_30.slp');
        const reopened = await page.evaluate(digestSession);
        const d = differences(comparable(digest), comparable(reopened));
        check(again.done === true && d.length === 0, '  reopening the save gives the same project', d.slice(0, 6).join(' | '));
        await page.context().close();
    }

    // =================================================================
    console.log('\n-- 2b. edits on a lazy 3.0 store save the same as on its 2.x twin --');
    {
        const edited = {};
        for (const name of ['luc3d_project_30.slp', 'luc3d_project_2x.slp']) {
            const page = await freshPage();
            await page.evaluate(async (name) => { window.__bytes = await window.__fixture(name); }, name);
            await run(page, LAZY, '__bytes', 'open.slp');
            const r = await page.evaluate(async () => {
                const st = window.__lucid.state;
                const s = st.session;
                const im = window.__lucid.interactionManager;
                const tri = await import('/pose/triangulation.js');
                // A node moved on frame 0.
                await tri.ensureLazyFrameData(0);
                const g0 = s.instanceGroups.get(0)[0];
                const [cam, inst] = g0.instances.entries().next().value;
                inst.setPoint(0, inst.getX(0) + 7.5, inst.getY(0) - 3.25);
                inst.type = 'user';
                inst.modified = true;
                g0.markDirty();
                // The Delete key, on one view of a group and on a whole group.
                for (const [f, whole] of [[5, false], [7, true]]) {
                    await tri.ensureLazyFrameData(f);
                    st.currentFrame = f;
                    im.clearSelection();
                    const g = s.instanceGroups.get(f)[0];
                    im.select(g, -1);
                    im.lastInteractedView = g.instances.keys().next().value;
                    im._deleteSelected(whole);
                }
                let rows = 0;
                s.lazyLoader.forEachInstanceRow(() => { rows++; });
                return { cam, rows, groups5: s.instanceGroups.get(5).length, groups7: (s.instanceGroups.get(7) || []).length };
            });
            await save(page, 'slp-format-30-edit');
            r.saved = await page.evaluate(async () => {
                const f = await window.__readFile(window.__saved);
                return {
                    rows: f.instances.instance_type.length, userPoints: Array.from(f.points.x), type: Array.from(f.instances.instance_type),
                    start: Array.from(f.frames.instance_id_start), members: Array.from(f.members.inst), lf: Array.from(f.members.lf), p3: f.points_3d.length,
                    igStart: Array.from(f.instance_groups.member_start),
                };
            });
            edited[name] = r;
            await page.context().close();
        }
        const e30 = edited['luc3d_project_30.slp'], e2x = edited['luc3d_project_2x.slp'];
        const removed = 321 - e30.rows;
        check(removed >= 2 && e30.saved.rows === e30.rows, 'a move and two deletes reach the store and the save', JSON.stringify([e30.rows, e30.saved.rows]));
        check(e30.saved.userPoints.length === 15 && e30.saved.userPoints.filter(Number.isFinite).length > 0, 'the moved pose is a user instance in the save', JSON.stringify(e30.saved.userPoints));
        check(eq(e30, e2x), 'and all of it is the same as on the 2.x twin', JSON.stringify(differences(e30, e2x)).slice(0, 400));
    }

    // =================================================================
    console.log('\n-- 3. every reader --');
    {
        const page = await freshPage();
        const r = await page.evaluate(async () => {
            const fio = await import('/import-export/file-io.js');
            const { SioLazyLoader } = await import('/loading/sio-lazy-loader.js');
            const sio = await import('/lib/sleap-io/index.browser.js');
            const bytes = await window.__fixture('luc3d_project_30.slp');
            const file = new File([bytes], 'mice.slp');
            const h5wasmUrl = new URL('lib/h5wasm/h5wasm.iife.js', document.baseURI).href;
            const typed = await fio.parseSlpViaSleapIO(file);
            const raw = await fio.parseSlpH5(file);
            const columnar = await fio.parseSlpH5(file, null, { columnar: true });
            const fallback = await fio.parseSlpWithRawFallback(file);
            const perCam = await new SioLazyLoader().open('cam', file);
            const project = await new SioLazyLoader().openProjectSlp(file);
            const main = await sio.loadSlp(bytes, { openVideos: false, h5: { stream: 'download', filenameHint: 'mice.slp', h5wasmUrl } }).then(() => 'read', e => [e.name, e.message]);
            const mainLazy = await sio.loadSlp(bytes, { openVideos: false, lazy: true, h5: { stream: 'download', filenameHint: 'mice.slp', h5wasmUrl } }).then(() => 'read', e => [e.name, e.message]);
            const picked = await sio.loadSlp(file, { openVideos: false, h5: { h5wasmUrl } });
            const probe = await fio.checkSlpFileFormat(file).then(() => 'passes', e => e.message);
            return {
                typed: typed.frames.length, raw: raw.frames.length, rawShape: Object.keys(raw).sort().join() === Object.keys(typed).sort().join(),
                columnar: columnar.frames.length, columnarKey: !!columnar.columnar, fallback: fallback.frames.length,
                perCam: perCam.nFrames, project: project.nFrames, main, mainLazy, picked: picked.labeledFrames.length, probe,
                projectSession: !!project.typedSession,
            };
        });
        check(r.typed === 96 && r.raw === 96 && r.fallback === 96, 'the typed reader, the raw worker (parseSlpH5) and the fallback route read it', JSON.stringify(r));
        check(r.rawShape, '  the raw worker\'s result is the typed reader\'s, in shape');
        check(r.columnar === 96 && !r.columnarKey, '  the columnar option falls back to `frames`, which its one caller reads', JSON.stringify([r.columnar, r.columnarKey]));
        check(r.perCam > 0 && r.project > 0 && r.projectSession, 'SioLazyLoader.open and openProjectSlp read it', JSON.stringify([r.perCam, r.project, r.projectSession]));
        check(r.probe === 'passes', 'the format probe lets it through', r.probe);
        check(r.picked === 96, 'loadSlp(File) reads it through the worker', r.picked);
        const onlyStreaming = 'mice.slp uses SLP format 3.0, which only readSlpStreaming reads.';
        check(eq(r.main, ['UnsupportedSlpFormatError', onlyStreaming]) && eq(r.mainLazy, ['UnsupportedSlpFormatError', onlyStreaming]),
            'the main-thread readers, which LUC3D does not use, refuse it rather than read 0 frames', JSON.stringify([r.main, r.mainLazy]));
        await page.context().close();

        // Per-camera folders, on both routes.
        for (const [label, size] of [['small files (eager route)', 0], ['large files (lazy route)', OVER_150_MB]]) {
            const page2 = await freshPage();
            const cams = await page2.evaluate(PER_CAMERA, [['py_written_30_columnar.slp', 'py_written_30_columnar.slp'], size]);
            check(cams.lazy === !!size && cams.per.cam1 === 1 && cams.per.cam2 === 1 && !/could not be read/.test(cams.status.text),
                'per-camera folder, ' + label + ': each camera has its data', JSON.stringify(cams));
            await page2.context().close();

            const page3 = await freshPage();
            const refused = await page3.evaluate(PER_CAMERA, [['py_written_30_events_lzf.slp', 'py_written_30_columnar.slp'], size]);
            check(refused.views === 2 && refused.per.cam1 === 0 && refused.per.cam2 === 1 && refused.status.text.includes('1 annotation file(s) could not be read: cam1 (cam1.slp)')
                && refused.status.text.includes(lzfRefusal('cam1.slp', 'events/name')), '  an lzf camera is a view and no data, and the status says why', JSON.stringify(refused));
            await page3.context().close();
        }

        // File ▸ Add SLP goes through a picker. The file has to be on the open skeleton.
        const addFile = path.join(repoRoot, 'tests/fixtures/slp-format-30/py010_30_twin_events.slp');
        const pageAdd = await freshPage();
        await pageAdd.evaluate(async () => { window.__bytes = await window.__fixture('py010_2x_twin_events.slp'); });
        await run(pageAdd, EAGER, '__bytes', 'open.slp');
        const addedBefore = await pageAdd.evaluate(() => window.__snapshot());
        const addedDone = await addSlp(pageAdd, addFile);
        const added = await pageAdd.evaluate(() => window.__snapshot());
        const addedStatus = await pageAdd.evaluate(() => window.__status());
        check(addedDone === true && /error/.test(addedStatus.dot) === false && added.instances > addedBefore.instances, 'handleAddSlp adds a 3.0 file to an open project', JSON.stringify([addedDone, addedBefore, added, addedStatus]));
        await pageAdd.context().close();

        // A lazily opened 3.0 project takes another file's poses as store rows, the same as
        // a 2.x one, and a save has both (`/frames/is_negative` stays one column long enough).
        const addedSaves = {};
        for (const base of ['py010_30_twin_events.slp', 'py010_2x_twin_events.slp']) {
            const page4 = await freshPage();
            await page4.evaluate(async (base) => { window.__bytes = await window.__fixture(base); }, base);
            await run(page4, LAZY, '__bytes', 'open.slp');
            const lazyAdded = await addSlp(page4, addFile);
            const st = await page4.evaluate(async () => {
                const AS = await import('/ui/app-state.js');
                const store = Array.from(AS.state.session.lazyLoader.labelsByCam.values())[0]._lazyDataStore;
                const lengths = Object.values(store.framesData).map(c => c.length);
                return { status: window.__status(), cameras: AS.state.session.cameras.map(c => c.name), even: new Set(lengths).size === 1, rows: lengths[0], videos: store.videos.length };
            });
            st.done = lazyAdded;
            await save(page4, 'slp-format-30-add');
            addedSaves[base] = await page4.evaluate(async () => {
                const f = await window.__readFile(window.__saved);
                return { frames: Array.from(f.frames.frame_idx), video: Array.from(f.frames.video), track: Array.from(f.instances.track), x: Array.from(f.points.x), px: Array.from(f.pred_points.x), p3: f.points_3d };
            });
            addedSaves[base].st = st;
            await page4.context().close();
        }
        const add30 = addedSaves['py010_30_twin_events.slp'], add2x = addedSaves['py010_2x_twin_events.slp'];
        check(add30.st.done === true && add30.st.even && add30.st.rows > 5 && !/error/.test(add30.st.status.dot), 'Add SLP on a lazy 3.0 project: the store grows evenly', JSON.stringify(add30.st));
        check(add30.frames.length > 5 && add30.x.length > 6, '  a save has the added poses', JSON.stringify([add30.frames, add30.x]));
        const { st: _s1, ...a30 } = add30, { st: _s2, ...a2x } = add2x;
        check(eq(a30, a2x), '  and is the same as for the 2.x project', JSON.stringify(differences(a30, a2x)).slice(0, 300));

        // The single-.slp folder route, with a video next to it.
        const page5 = await freshPage();
        const recDir = fs.mkdtempSync(path.join(os.tmpdir(), 'lucid-slp3-'));
        try {
            fs.mkdirSync(path.join(recDir, 'videos'));
            fs.copyFileSync(path.join(repoRoot, 'tests/fixtures/slp-format-30/luc3d_project_30.slp'), path.join(recDir, 'mice.slp'));
            page5.once('filechooser', fc => fc.setFiles(recDir));
            await page5.evaluate(async () => {
                window.__folderDone = false;
                (await import('/loading/session-loader.js')).handleLoadSessionFolderSingleSlp().then(() => { window.__folderDone = true; }, e => { window.__folderDone = 'THREW ' + e; });
            });
            for (let i = 0; i < 240; i++) {
                const s = await page5.evaluate(() => {
                    const b = [...document.querySelectorAll('button')].find(x => x.offsetParent && /^(Skip|Later|Cancel|Not now|Continue)/i.test((x.textContent || '').trim()));
                    if (b) { b.click(); return {}; }
                    return { done: window.__folderDone, prompt: !!document.getElementById('videoSelectOverlay') };
                });
                if (s.prompt) await page5.keyboard.press('Escape');
                if (s.done) break;
                await page5.waitForTimeout(250);
            }
            const folder = await page5.evaluate(digestSession);
            check(folder.groups === 48 && folder.with3d === 48 && folder.bouts.length === 3, 'handleLoadSessionFolderSingleSlp reads it with its 3D and bouts', JSON.stringify([folder.groups, folder.with3d, folder.bouts && folder.bouts.length, folder.status]));
        } finally {
            fs.rmSync(recDir, { recursive: true, force: true });
        }
        await page5.context().close();

        // Set Env reads each camera's file through the raw worker's door.
        const page6 = await freshPage();
        const env = await page6.evaluate(async () => {
            const fio = await import('/import-export/file-io.js');
            const data = await fio.parseSlpH5(new File([await window.__fixture('py_written_30_columnar.slp')], 'cam0.externals.slp'));
            return { frames: data.frames.length, instances: data.frames[0].instances.length, frameIdx: data.frames[0].frameIdx, points: data.frames[0].instances[0].points, skeleton: data.skeleton.nodes };
        });
        check(eq(env, { frames: 1, instances: 1, frameIdx: 0, points: [[1, 2], [3, 4]], skeleton: ['a', 'b'] }), 'the environment step gets frames, poses and a skeleton from it', JSON.stringify(env));
        await page6.context().close();
    }

    // =================================================================
    console.log('\n-- 4. files Python wrote itself --');
    {
        for (const [label, code] of LOADERS) {
            const a = await openFixture('py010_30_twin_events.slp', code);
            const b = await openFixture('py010_2x_twin_events.slp', code);
            // Python's 2.x file holds instance scores as 32-bit, and 3.0 as 64-bit, so the
            // scores differ past float precision there. All else is the same.
            const ignore = /^(memberScoresDigest|pose\.digest|pose\.hydrated\.digest|pose\.tables\.instances\.score)/;
            const d = differences(comparable(a.digest), comparable(b.digest)).filter(l => !ignore.test(l));
            check(a.r.done === true && d.length === 0, label + ': Python\'s twin reads the same as its 2.x file', d.slice(0, 6).join(' | '));
            check(a.digest.groups === 3 && a.digest.with3d === 2 && a.digest.bouts.length === 3 && a.digest.predicted.length === 2, '  its groups, 3D, bouts and predictions', JSON.stringify([a.digest.groups, a.digest.with3d, a.digest.bouts.length, a.digest.predicted.length]));
            const warned = a.page.consoleLines.filter(l => /\[sleap-io\]/.test(l));
            check(warned.some(l => /\/points has 1 unlabeled point/.test(l)), '  the console says what a save stores differently: an unlabeled point', JSON.stringify(warned));
            check(warned.some(l => /\/instances holds metadata/.test(l)) && warned.some(l => /\/frames holds metadata/.test(l)), '  and what it does not keep: per-record metadata', JSON.stringify(warned));
            // What a save of each writes is the same project.
            await save(a.page, 'slp-format-30-a');
            const savedA = await a.page.evaluate(async () => {
                const f = await window.__readFile(window.__saved);
                return window.__summary(f);
            });
            await save(b.page, 'slp-format-30-b');
            const savedB = await b.page.evaluate(async () => {
                const f = await window.__readFile(window.__saved);
                return window.__summary(f);
            });
            const { score: scoreA, ...restA } = savedA;
            const { score: scoreB, ...restB } = savedB;
            check(eq(restA, restB), '  a save from each holds the same poses and 3D', JSON.stringify(differences(restA, restB)).slice(0, 300));
            check(scoreA.length === scoreB.length && scoreA.every((v, i) => Math.fround(v) === Math.fround(scoreB[i])), '  and the same scores, to 32 bits', JSON.stringify([scoreA, scoreB]));
            check(restA.identityLinks.length === 2 && restA.categoryLinks.length > 0 && restA.embeddings.length > 0, '  and the identities and categories on detections', JSON.stringify([restA.identityLinks, restA.categoryLinks, restA.embeddings]));
            await a.page.context().close();
            await b.page.context().close();
        }

        // Sparse videos and a negative frame: `video` is the list index in 3.0, and the
        // guess from group names must not run (buildVideoIdMap).
        const video = await (async () => {
            const page = await freshPage();
            const r = await page.evaluate(async () => {
                const sio = await import('/lib/sleap-io/index.browser.js');
                const h5wasmUrl = new URL('lib/h5wasm/h5wasm.iife.js', document.baseURI).href;
                const twin = new File([await window.__fixture('py010_30_twin_events.slp')], 'twin.slp');
                const labels = await sio.readSlpStreaming(twin, { openVideos: false, h5wasmUrl });
                const lazy = await sio.readSlpStreaming(twin, { openVideos: false, lazy: true, h5wasmUrl });
                const frameOf = (lf) => [labels.videos.indexOf(lf.video), lf.frameIdx, lf.instances.length];
                const lazyFrames = [];
                for (let i = 0; i < lazy._lazyFrameList.length; i++) {
                    const lf = lazy._lazyFrameList.at(i);
                    lazyFrames.push([lazy.videos.indexOf(lf.video), lf.frameIdx, lf.instances.length]);
                }
                const frameRows = labels.labeledFrames.map(frameOf);
                const inst = labels.labeledFrames[0].instances;
                return { eager: frameRows, lazy: lazyFrames, fromPredicted: inst.some(x => x.fromPredicted && x.fromPredicted === inst[0]), provenance: labels.provenance, trackingScore: inst[0].trackingScore };
            });
            await page.context().close();
            return r;
        })();
        const want = [[0, 5, 2], [1, 7, 1], [1, 9, 0], [0, 11, 0], [0, 13, 1]];
        check(eq(video.eager, want) && eq(video.lazy, want), 'each frame is on its own video, in the eager and the lazy store', JSON.stringify(video));
        check(video.fromPredicted, 'a user instance keeps its link to the prediction it came from');
        check(video.trackingScore === 0.75, 'and a tracking score is read', video.trackingScore);
        check(video.provenance && Object.keys(video.provenance).length > 0, 'provenance comes from /provenance_json', JSON.stringify(video.provenance));

        // `/frames/video` is the list index: with only one of two videos used, and group
        // names in another order, the 2.x guess from those names would pick the wrong one.
        const listIndex = await (async () => {
            const page = await freshPage();
            const r = await page.evaluate(async () => {
                const sio = await import('/lib/sleap-io/index.browser.js');
                const h5wasmUrl = new URL('lib/h5wasm/h5wasm.iife.js', document.baseURI).href;
                const file = new File([await window.__fixture('py_written_30_video_index.slp')], 'index.slp');
                const eager = await sio.readSlpStreaming(file, { openVideos: false, h5wasmUrl });
                const lazy = await sio.readSlpStreaming(file, { openVideos: false, lazy: true, h5wasmUrl });
                return { eager: eager.labeledFrames[0].video.filename, lazy: lazy._lazyFrameList.at(0).video.filename };
            });
            await page.context().close();
            return r;
        })();
        check(listIndex.eager === 'clip_b.mp4' && listIndex.lazy === 'clip_b.mp4', '`video` is the list index, whatever the videos\' group names say', JSON.stringify(listIndex));

        // Only the required columns: the rest read as Python's defaults.
        const required = await (async () => {
            const page = await freshPage();
            const r = await page.evaluate(async () => {
                const sio = await import('/lib/sleap-io/index.browser.js');
                const h5wasmUrl = new URL('lib/h5wasm/h5wasm.iife.js', document.baseURI).href;
                const file = new File([await window.__fixture('py_written_30_only_required.slp')], 'required.slp');
                const labels = await sio.readSlpStreaming(file, { openVideos: false, lazy: true, h5wasmUrl });
                const store = labels._lazyDataStore;
                const col = (c) => Array.from(c);
                const inst = labels._lazyFrameList.at(0).instances[0];
                return {
                    track: col(store.instancesData.track), from: col(store.instancesData.from_predicted), visible: col(store.pointsData.visible),
                    complete: col(store.pointsData.complete), x: col(store.pointsData.x), kinds: [store.pointsData.visible.constructor.name, store.instancesData.track.constructor.name],
                    instanceTrack: inst.track, trackingScore: inst.trackingScore, bouts: labels.events.length,
                };
            });
            await page.context().close();
            return r;
        })();
        check(eq(required.track, [-1]) && eq(required.from, [-1]) && eq(required.visible, [0, 0]) && eq(required.complete, [0, 0]) && eq(required.x, [1, 3]) && required.instanceTrack === null && required.trackingScore === 0,
            'a table with only its required columns reads with the defaults', JSON.stringify(required));
        check(required.kinds[0] === 'Uint8Array' && required.kinds[1] === 'Float64Array', '  flags left out take one byte each', JSON.stringify(required.kinds));

        // /provenance_json wins, as Python's `read_provenance`, which drops the attribute's
        // copy once it would not fit there.
        const provenance = await (async () => {
            const page = await freshPage();
            const r = await page.evaluate(async () => {
                const sio = await import('/lib/sleap-io/index.browser.js');
                const h5wasmUrl = new URL('lib/h5wasm/h5wasm.iife.js', document.baseURI).href;
                const h5 = window.h5wasm;
                await h5.ready;
                const p = '/provenance.slp';
                h5.FS.writeFile(p, await window.__fixture('py010_30_twin_events.slp'));
                const f = new h5.File(p, 'a');
                const meta = f.get('metadata');
                const json = JSON.parse(meta.attrs.json.value);
                const fromDataset = Object.keys(JSON.parse(f.get('provenance_json').value)).sort();
                delete json.provenance;
                meta.delete_attribute('json');
                meta.create_attribute('json', JSON.stringify(json));
                f.close();
                const file = new File([new Uint8Array(h5.FS.readFile(p))], 'provenance.slp');
                h5.FS.unlink(p);
                const labels = await sio.readSlpStreaming(file, { openVideos: false, h5wasmUrl });
                return { fromDataset, read: Object.keys(labels.provenance).sort() };
            });
            await page.context().close();
            return r;
        })();
        check(provenance.fromDataset.length > 0 && eq(provenance.read, provenance.fromDataset), 'provenance only in /provenance_json is read', JSON.stringify(provenance));

        const sparse = await openFixture('py010_30_sparse_events.slp', LAZY);
        check(sparse.r.done === true && sparse.digest.pose.tables['points.x'][0] === 4, 'a sparse file with a negative frame opens with all its points', JSON.stringify(sparse.digest.pose.tables['points.x']));
        await sparse.page.context().close();

        // Images in a 3.0 package show, the same as in the 2.x one.
        const pictures = {};
        for (const name of ['py010_30_embedded_events.pkg.slp', 'py010_2x_embedded_events.pkg.slp']) {
            const { page, r, digest } = await openFixture(name, EAGER);
            const info = await page.evaluate(async () => {
                const AS = await import('/ui/app-state.js');
                const view = AS.state.views[0];
                const canvas = view && view.canvas;
                return { views: AS.state.views.length, frameWorkers: (AS.state.frameWorkers || []).length, canvas: canvas ? [canvas.width, canvas.height] : null };
            });
            pictures[name] = { r, info, status: digest.status };
            await page.context().close();
        }
        const p30 = pictures['py010_30_embedded_events.pkg.slp'], p2x = pictures['py010_2x_embedded_events.pkg.slp'];
        check(p30.r.done === true && p30.info.views >= 1 && eq(p30.info, p2x.info), 'a 3.0 package opens with its images, as its 2.x twin does', JSON.stringify([p30, p2x]));
    }

    // =================================================================
    console.log('\n-- 5. the Open card counts bytes of columns, and a sliced read equals a whole one --');
    {
        const page = await freshPage();
        const r = await page.evaluate(async () => {
            const sio = await import('/lib/sleap-io/index.browser.js');
            const h5wasmUrl = new URL('lib/h5wasm/h5wasm.iife.js', document.baseURI).href;
            const file = new File([await window.__fixture('luc3d_project_30.slp')], 'mice.slp');
            const read = async (elems) => {
                window.__lucidReadSliceElems = elems;
                const counts = [];
                const labels = await sio.readSlpStreaming(file, { lazy: true, openVideos: false, h5wasmUrl, onReadBytes: (d, t) => { counts.push([d, t]); } });
                window.__lucidReadSliceElems = undefined;
                return { counts, store: labels._lazyDataStore };
            };
            const whole = await read(undefined), sliced = await read(37);
            const diffs = [];
            for (const t of ['framesData', 'instancesData', 'predPointsData', 'pointsData']) {
                for (const k of Object.keys(whole.store[t])) {
                    const u = whole.store[t][k], v = sliced.store[t][k];
                    if (u.constructor !== v.constructor || u.length !== v.length) diffs.push(t + '.' + k + ' shape');
                    else for (let i = 0; i < u.length; i++) if (!(u[i] === v[i] || (Number.isNaN(u[i]) && Number.isNaN(v[i])))) { diffs.push(t + '.' + k); break; }
                }
            }
            const c = sliced.counts;
            return {
                updates: c.length, first: c[0], last: c[c.length - 1], rising: c.every((x, i) => !i || x[0] >= c[i - 1][0]), wholeUpdates: whole.counts.length, diffs,
                // What the table holds, from the file: 4,815 rows of x, y, score (8 bytes) and three flags.
                types: { x: whole.store.predPointsData.x.constructor.name, visible: whole.store.predPointsData.visible.constructor.name, track: whole.store.instancesData.track.constructor.name, type: whole.store.instancesData.instance_type.constructor.name },
            };
        });
        check(r.first[0] === 0 && r.first[1] > 4815 * 27 && r.last[0] === r.last[1] && r.rising, 'the count goes from 0 to the whole size', JSON.stringify([r.first, r.last, r.rising]));
        check(r.updates > r.wholeUpdates && r.updates > 100, 'and more often with smaller slices', JSON.stringify([r.updates, r.wholeUpdates]));
        check(r.diffs.length === 0, 'a sliced read gives every column the same values and type', JSON.stringify(r.diffs));
        check(r.types.x === 'Float64Array' && r.types.visible !== 'Array' && r.types.track === 'Float64Array', 'coordinates stay doubles, 64-bit ids become doubles, flags stay small', JSON.stringify(r.types));
        await page.context().close();
    }

    // =================================================================
    console.log('\n-- 6. lzf is refused everywhere, with one clear message --');
    {
        const cases = [
            ['py010_edge_30_lzf.slp', 'Python 3.0, every table a group of lzf columns'],
            ['py010_edge_2x_lzf.slp', 'Python 2.x, compound tables in lzf'],
            ['luc3d_project_2x_lzf.slp', 'LUC3D\'s flat /pred_points in lzf, which a big project reads in slices', 'pred_points'],
            ['py_written_30_events_lzf.slp', 'only one string column of /events in lzf'],
        ];
        const page = await freshPage();
        for (const [name, what, onlyTable] of cases) {
            const r = await page.evaluate(async (name) => {
                const fio = await import('/import-export/file-io.js');
                const { SioLazyLoader } = await import('/loading/sio-lazy-loader.js');
                const { LazyFrameLoader } = await import('/pose/triangulation.js');
                const sio = await import('/lib/sleap-io/index.browser.js');
                const bytes = await window.__fixture(name);
                const file = new File([bytes], name);
                const h5wasmUrl = new URL('lib/h5wasm/h5wasm.iife.js', document.baseURI).href;
                const outcome = async (fn) => {
                    try { await fn(); return 'read'; } catch (e) { return [e.name, e.message]; }
                };
                let counted = 0;
                return {
                    'typed reader': await outcome(() => fio.parseSlpViaSleapIO(file)),
                    'raw worker': await outcome(() => fio.parseSlpH5(file)),
                    'raw worker, columnar': await outcome(() => fio.parseSlpH5(file, null, { columnar: true })),
                    'the fallback route': await outcome(() => fio.parseSlpWithRawFallback(file)),
                    'the format probe': await outcome(() => fio.checkSlpFileFormat(file)),
                    'per-camera lazy': await outcome(() => new SioLazyLoader().open('cam', file)),
                    'project lazy, reading tables in slices': await outcome(() => new SioLazyLoader().openProjectSlp(file, null, () => { counted++; })),
                    'an analysis .h5, on its lazy route': await outcome(() => new LazyFrameLoader().open('cam', new File([bytes], name))),
                    'main thread': await outcome(() => sio.loadSlp(bytes, { openVideos: false, h5: { stream: 'download', filenameHint: name, h5wasmUrl } })),
                };
            }, name);
            const first = r['typed reader'];
            const table = /: \/([^ ]+) is compressed/.exec(first[1] || '');
            check(table && first[0] === 'UnsupportedSlpFormatError' && first[1] === lzfRefusal(name, onlyTable || table[1]), what + ': the typed reader refuses it', JSON.stringify(first));
            for (const [where, o] of Object.entries(r)) {
                const main = where === 'main thread' && name.includes('30');
                if (main) check(o[0] === 'UnsupportedSlpFormatError', '  ' + where + ' refuses', JSON.stringify(o));
                else if (where.startsWith('an analysis')) check(o !== 'read' && o[1].startsWith(first[1]), '  ' + where + ', in the same words', JSON.stringify(o));
                else check(o[0] === 'UnsupportedSlpFormatError' && o[1] === first[1], '  ' + where + ', in the same words', JSON.stringify(o));
            }
        }
        check(!page.consoleLines.some(l => /falling back/i.test(l)), 'none of them retried through the raw worker');
        await page.context().close();

        // Nothing changes on a refusal: both loaders, with a project open.
        const page2 = await freshPage();
        await page2.evaluate(async () => {
            window.__project = await window.__fixture('luc3d_project_2x.slp');
            window.__lzf = await window.__fixture('py010_edge_30_lzf.slp');
        });
        const opened = await run(page2, EAGER, '__project', 'open-project.slp');
        const before = await page2.evaluate(() => window.__snapshot());
        check(opened.done === true && before.frameGroups === 12, 'a project is open', JSON.stringify(before));
        for (const [label, code, size] of [
            ['handleLoadSlpFile', EAGER],
            ['handleLoadProjectSlpLazy (which frees the open project first)', LAZY],
            ['handleLoadProject', ROUTED],
            ['handleLoadProject, above 200 MB', ROUTED, OVER_200_MB],
        ]) {
            const r = await run(page2, code, '__lzf', 'lzf.slp', size);
            const st = await page2.evaluate(() => window.__status());
            const after = await page2.evaluate(() => window.__snapshot());
            check(r.done === true && r.returned === false, label + ': tells its caller the file did not open', JSON.stringify(r));
            check(/^lzf\.slp: \/[^ ]+ is compressed with "lzf", which LUC3D cannot read\./.test(st.text) && /error/.test(st.dot), '  the status is the refusal, in red', JSON.stringify(st));
            check(eq(after, before), '  the open project is untouched', JSON.stringify(after));
        }

        // Not a save target.
        const handle = await page2.evaluate(async () => {
            const bytes = window.__lzf;
            window.showOpenFilePicker = async () => [{ __tag: 'lzf.slp', kind: 'file', name: 'lzf.slp', getFile: async () => new File([bytes], 'lzf.slp') }];
            document.getElementById('statusText').textContent = 'waiting';
            document.getElementById('menuLoadProject').click();
            for (let i = 0; i < 200 && !/cannot read/.test(document.getElementById('statusText').textContent); i++) await new Promise(r => setTimeout(r, 100));
            await new Promise(r => setTimeout(r, 300));
            const h = window.__lucid.state.slpFileHandle;
            return { status: document.getElementById('statusText').textContent, tag: h ? h.__tag : null };
        });
        check(/cannot read/.test(handle.status) && handle.tag !== 'lzf.slp', 'File ▸ Open of it records no save target', JSON.stringify(handle));
        await page2.context().close();

        // An unknown `required_features` entry is refused like a newer format, on both readers.
        const page3 = await freshPage();
        const f = await page3.evaluate(async () => {
            const fio = await import('/import-export/file-io.js');
            const h5 = window.h5wasm;
            await h5.ready;
            const p = '/features.slp';
            h5.FS.writeFile(p, await window.__fixture('py_written_30_columnar.slp'));
            const file = new h5.File(p, 'a');
            file.get('metadata').create_attribute('required_features', ['zeta', 'alpha']);
            file.close();
            const bytes = new Uint8Array(h5.FS.readFile(p));
            h5.FS.unlink(p);
            const picked = new File([bytes], 'features.slp');
            const outcome = async (fn) => { try { await fn(); return 'read'; } catch (e) { return [e.name, e.message]; } };
            return { typed: await outcome(() => fio.parseSlpViaSleapIO(picked)), raw: await outcome(() => fio.parseSlpH5(picked)), probe: await outcome(() => fio.checkSlpFileFormat(picked)) };
        });
        const features = 'features.slp requires SLP features this version of LUC3D doesn\'t support (alpha, zeta). Update LUC3D to read it, or re-save the file from Python sleap-io with `legacy_format=True`.';
        check(eq(f.typed, ['UnsupportedSlpFormatError', features]) && eq(f.raw, f.typed) && eq(f.probe, f.typed), 'required_features: refused on every reader, in the same words', JSON.stringify(f));
        await page3.context().close();
    }

    // =================================================================
    console.log('\n-- 7. 3.0 tables that contradict each other are refused --');
    {
        const page = await freshPage();
        await page.evaluate(async () => {
            // A copy of a file with one table changed, through h5wasm.
            let n = 0;
            window.__broken = async (edit, name = 'luc3d_project_30.slp') => {
                const bytes = await window.__fixture(name);
                const h5 = window.h5wasm;
                await h5.ready;
                const p = '/broken-' + (n++) + '.slp';
                h5.FS.writeFile(p, bytes);
                const f = new h5.File(p, 'a');
                edit(f);
                f.close();
                const out = new Uint8Array(h5.FS.readFile(p));
                h5.FS.unlink(p);
                return out;
            };
        });
        const outcomes = async (bytes) => {
            const fio = await import('/import-export/file-io.js');
            const file = new File([bytes], 'broken.slp');
            const outcome = async (fn) => { try { await fn(); return 'read'; } catch (e) { return String(e.message).split('\n')[0]; } };
            return { fallback: await outcome(() => fio.parseSlpWithRawFallback(file)), typed: await outcome(() => fio.parseSlpViaSleapIO(file)), raw: await outcome(() => fio.parseSlpH5(file)) };
        };
        // h5wasm can't delete a dataset, so these two are files (make_project_fixture.py).
        for (const [name, what, want] of [
            ['py_written_30_missing_column.slp', 'a missing column', /\/frames is missing column\(s\) frame_idx/],
            ['py_written_30_uneven_columns.slp', 'columns of different lengths', /\/points columns have different lengths/],
        ]) {
            const r = await page.evaluate(async ([name, code]) => {
                const outcomes = new Function('return ' + code)();
                return outcomes(await window.__fixture(name));
            }, [name, outcomes.toString()]);
            check(want.test(r.fallback) && want.test(r.typed) && want.test(r.raw), what + ': refused, naming the table', JSON.stringify(r).slice(0, 400));
        }
        const broken = [
            ['/frames ranges past the instances', "const c = f.get('frames/instance_id_end'); c.write_slice([[95, 96]], new BigInt64Array([9999n]));", /inconsistent tables: .*\/frames instance ranges end at 9999/],
            ['a point range past /pred_points', "const c = f.get('instances/point_id_end'); c.write_slice([[320, 321]], new BigInt64Array([999999n]));", /inconsistent tables: .*point ranges fall outside \/pred_points/],
            ['a frame on no video', "const c = f.get('frames/video'); c.write_slice([[0, 1]], new BigInt64Array([77n]));", /inconsistent tables: .*\/frames video is not one of the 8 videos/],
            ['a member on no frame', "const c = f.get('session_data/instance_group_members/lf'); c.write_slice([[0, 1]], new BigInt64Array([99999n]));", /inconsistent tables: .*instance_group_members lf is not one of the 96 frames/],
            ['3D past /points_3d', "const c = f.get('session_data/instance_groups/pts3d_end'); c.write_slice([[47, 48]], new BigInt64Array([99999n]));", /inconsistent tables: .*3D ranges fall outside points_3d/],
        ];
        for (const [what, edit, want] of broken) {
            const r = await page.evaluate(async ([edit, code]) => {
                const outcomes = new Function('return ' + code)();
                return outcomes(await window.__broken(new Function('f', edit)));
            }, [edit, outcomes.toString()]);
            check(want.test(r.fallback) && want.test(r.typed) && want.test(r.raw), what + ': refused, naming the table, with no raw read in its place', JSON.stringify(r).slice(0, 400));
        }
        // A plain file, with nothing else to keep it from the raw worker's door.
        const plain = await page.evaluate(async ([code]) => {
            const outcomes = new Function('return ' + code)();
            return outcomes(await window.__broken((f) => {
                f.get('frames/instance_id_end').write_slice([[1, 2]], new BigInt64Array([9999n]));
            }, 'py010_30_sparse_events.slp'));
        }, [outcomes.toString()]);
        const plainRefused = /inconsistent tables: .*\/frames instance ranges end at 9999/;
        check(plainRefused.test(plain.fallback) && plainRefused.test(plain.typed), 'a plain file with no sessions: refused too', JSON.stringify(plain).slice(0, 300));
        check(!page.consoleLines.some(l => /falling back/i.test(l)), 'none of them retried through the raw worker');
        await page.context().close();
    }

    // =================================================================
    console.log('\n-- 8. a file holding what a save leaves out is never overwritten --');
    {
        // File ▸ Open with its pickers stubbed, then a plain Save, recording where it went.
        const openThenSave = async (name, size) => {
            const page = await freshPage();
            await page.evaluate(async ([name, size]) => {
                const bytes = await window.__fixture(name);
                const writes = window.__writes = { opened: 0, picked: 0, pickerCalls: 0 };
                const handle = (tag) => ({
                    __tag: tag, kind: 'file', name: tag + '.slp',
                    getFile: async () => {
                        const f = new File([bytes], tag + '.slp');
                        if (size) Object.defineProperty(f, 'size', { value: size });
                        return f;
                    },
                    createWritable: async () => ({
                        write: async () => { writes[tag]++; },
                        close: async () => {},
                        abort: async () => {},
                    }),
                });
                window.showOpenFilePicker = async () => [handle('opened')];
                window.showSaveFilePicker = async () => { writes.pickerCalls++; return handle('picked'); };
                document.getElementById('statusText').textContent = 'waiting';
                document.getElementById('menuLoadProject').click();
            }, [name, size || 0]);
            for (let i = 0; i < 480; i++) {
                const s = await page.evaluate(() => {
                    const b = [...document.querySelectorAll('button')].find(x => x.offsetParent && /^(Skip|Later|Not now)/i.test((x.textContent || '').trim()));
                    if (b) { b.click(); return {}; }
                    return {
                        prompt: !!document.getElementById('videoSelectOverlay'),
                        done: !!window.__lucid.state.slpFileHandle && /^(SLP loaded|Reopened|Opened)/.test(document.getElementById('statusText').textContent),
                    };
                });
                if (s.prompt) await page.keyboard.press('Escape');
                if (s.done) break;
                await page.waitForTimeout(250);
            }
            const r = await page.evaluate(async () => {
                const status = window.__status();
                const lazy = !!window.__lucid.state.session.lazyLoader;
                const save = (await import('/import-export/save-load.js')).quickSave;
                await save();
                const first = { ...window.__writes };
                const handleAfter = window.__lucid.state.slpFileHandle && window.__lucid.state.slpFileHandle.__tag;
                await save();
                return { status, lazy, first, second: { ...window.__writes }, handleAfter, saved: window.__status().text };
            });
            await page.context().close();
            return r;
        };
        const real = await openThenSave('luc3d_project_30.slp');
        check(!/A save leaves out/.test(real.status.text) && /success/.test(real.status.dot), 'a real project opens with nothing to warn about', JSON.stringify(real.status));
        check(real.first.pickerCalls === 0 && real.first.opened > 0 && real.first.picked === 0, 'and Save goes straight over its file', JSON.stringify(real.first));

        const python = await openThenSave('py010_30_twin_events.slp');
        check(/A save leaves out this file's ROIs, masks, boxes, centroids, label images, negative frames, per-row metadata, so Save asks where to save and this file stays as it is\.$/.test(python.status.text) && /warning/.test(python.status.dot),
            'a file with ROIs, masks and the rest says what a save leaves out', JSON.stringify(python.status));
        check(python.first.pickerCalls === 1 && python.first.opened === 0 && python.first.picked > 0, 'Save asks where, and the opened file is not touched', JSON.stringify(python.first));
        check(python.handleAfter === 'picked', 'the new file is now the save target', python.handleAfter);
        check(python.second.pickerCalls === 1 && python.second.opened === 0 && python.second.picked > python.first.picked, 'and the next Save goes there without asking', JSON.stringify(python.second));

        const big = await openThenSave('py010_30_twin_events.slp', OVER_200_MB);
        check(big.lazy && /^Reopened .* A save leaves out this file's ROIs/.test(big.status.text) && /warning/.test(big.status.dot), 'the lazy route says so too', JSON.stringify([big.lazy, big.status]));
        check(big.first.pickerCalls === 1 && big.first.opened === 0 && big.first.picked > 0, '  and Save asks where', JSON.stringify(big.first));
        const bigReal = await openThenSave('luc3d_project_30.slp', OVER_200_MB);
        check(bigReal.lazy && !/A save leaves out/.test(bigReal.status.text) && bigReal.first.pickerCalls === 0 && bigReal.first.opened > 0, 'a real project on the lazy route is saved straight over its file', JSON.stringify(bigReal).slice(0, 300));

        const twin2x = await openThenSave('py010_2x_twin_events.slp');
        check(/A save leaves out this file's ROIs/.test(twin2x.status.text) && twin2x.first.pickerCalls === 1 && twin2x.first.opened === 0, 'the same for its 2.x twin, with all but the per-row metadata', JSON.stringify(twin2x).slice(0, 400));
        check(!/per-row metadata/.test(twin2x.status.text), '  which 2.x has no place for', twin2x.status.text);
    }
} catch (e) {
    console.log('  ✗ threw:', e && e.stack || e);
    fails++;
} finally {
    if (browser) await browser.close();
    server.kill();
}

console.log(fails === 0 ? '\nPASS' : `\nFAIL (${fails})`);
process.exit(fails === 0 ? 0 : 1);
