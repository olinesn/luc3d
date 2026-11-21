/**
 * behavior-refused-file.mjs — a `.slp` too new to read leaves the open project's
 * behavior work alone.
 *
 * `slp-newer-format.mjs` holds the refusal itself, on every route. This is the part
 * it cannot see: every project loader calls `resetBehaviorState()` on its way to a new
 * project, and the ethogram is a module singleton, so a refusal that came after that
 * call would cost the user their ethogram, labels, predictions and states, all of which
 * can be unsaved.
 *
 * It opens a project on each route (eager, lazy), adds all four, and sends it the real
 * 3.0 file through every loader that can reach it. Each must refuse, keep the same
 * session object and leave the saved form of the behavior state exactly as it was. A
 * readable file then does replace them, so these checks can fail.
 *
 * Run: node tests/e2e/behavior-refused-file.mjs   (spawns its own http.server; PORT=…)
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8392);
const FIXTURE = path.join(repoRoot, 'tests', 'fixtures', 'slp-format-30', 'py_written_30_columnar.slp');
const VIDEO = path.join(repoRoot, 'tests', 'fixtures', 'barcode-60', 'barcode-60.mp4');
const tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'lucid-refused-behavior-'));

let fails = 0;
const check = (c, m, extra) => { console.log((c ? '  ✓ ' : '  ✗ ') + m + (c || extra === undefined ? '' : ' — ' + extra)); if (!c) fails++; };

const REFUSED = /^mice\.slp uses SLP format 3\.0, which is newer than this version of LUC3D supports/;
const EAGER = "return (await import('/import-export/slp-import.js')).handleLoadSlpFile(f);";
const LAZY = "return (await import('/loading/session-loader.js')).handleLoadProjectSlpLazy(f);";
const ROUTED = "return (await import('/import-export/save-load.js')).handleLoadProject(f);";
const OVER_200_MB = 250 * 1024 * 1024;

const server = spawn('python3', ['-m', 'http.server', String(PORT), '--bind', '127.0.0.1'], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1200));

let browser;
try {
    browser = await chromium.launch({ headless: true });

    async function freshPage() {
        const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
        page.on('pageerror', e => { console.log('  [pageerror]', String(e).slice(0, 300)); fails++; });
        page.on('dialog', d => d.accept());
        await page.goto(`http://127.0.0.1:${PORT}/index.html`, { waitUntil: 'domcontentloaded', timeout: 120000 });
        await page.waitForFunction(() => window.SleapIO && window.h5wasm && window.__lucid && window.__lucid.state, null, { timeout: 120000 });
        await page.evaluate(() => {
            // A project .slp the way LUC3D writes one: 3 frames, grouped, with sessions.
            window.__projectBytes = async (name, cams) => {
                const pd = await import('/pose/pose-data.js');
                const fileio = await import('/import-export/file-io.js');
                const M = [[900, 0, 256], [0, 900, 256], [0, 0, 1]];
                const cameras = cams.map((c, i) => new pd.Camera(c, M, [0, 0, 0, 0, 0], [0, 0.3 * i, 0], [-12 * i, 0, 2 * i], [512, 512]));
                const s = new pd.Session(cameras, new pd.Skeleton('skeleton', ['a', 'b'], [[0, 1]]), ['track_0'], name);
                for (let f = 0; f < 3; f++) {
                    const fg = new pd.FrameGroup(f);
                    s.addFrameGroup(fg);
                    const g = new pd.InstanceGroup(f + 1, 0);
                    cams.forEach((cn, ci) => {
                        const inst = new pd.Instance([[100 + f + ci, 120], [140 + f, 160 + ci]], 0, 'user', 1);
                        fg.addInstance(cn, inst);
                        g.addInstance(cn, inst);
                    });
                    s.instanceGroups.set(f, [g]);
                }
                const views = cams.map(c => ({ name: c, videoWidth: 512, videoHeight: 512, frameCount: 3 }));
                const vfs = cams.map(c => ({ name: c, assignedCamera: c, videoPath: c + '.mp4' }));
                return await window.SleapIO.saveSlpToBytes(fileio.buildSlpLabelsAllViews(s, views, vfs));
            };
            // An ethogram, a label, a predicted bout and a states layer, none of them saved.
            window.__addBehaviorWork = async () => {
                const AS = await import('/ui/app-state.js');
                const eth = await import('/pose/behavior-ethogram.js');
                const L = await import('/pose/behavior-labels.js');
                const P = await import('/pose/behavior-predictions.js');
                const S = await import('/pose/behavior-states.js');
                const s = AS.state.session;
                const rear = eth.addBehavior({ name: 'rear', key: 'r' }).behavior;
                const chase = eth.addBehavior({ name: 'chase', key: 'c', social: true }).behavior;
                L.labelRange(s, rear.id, 'mouseA', '', 0, 1, true, { note: 'not saved yet' });
                P.setBehaviorRun(s, { behaviorId: chase.id, cfg: 'c', fps: 30, mode: '3d',
                    bouts: [{ animal: 'mouseA', target: 'mouseB', start: 0, end: 2, score: 0.7 }] });
                S.setStateLayer(s, { name: 'keypoint-MoSeq', source: 'keypoint-MoSeq', animals: { mouseA: [[0, 2, 1]] } });
                window.__openSession = s;
            };
            // The behavior state as a project file would save it.
            window.__behaviorPrint = async () => {
                const AS = await import('/ui/app-state.js');
                const meta = await import('/import-export/behavior-metadata.js');
                const s = AS.state.session;
                if (!s) return null;
                const lucid = {};
                meta.writeBehaviorMetadata(lucid, s);
                return JSON.stringify(lucid);
            };
            window.__sameSession = async () => (await import('/ui/app-state.js')).state.session === window.__openSession;
            window.__status = () => document.getElementById('statusText').textContent;
        });
        return page;
    }

    /** Run a loader on page-global bytes, dismissing the prompts it raises. `size` overrides `f.size`. */
    async function run(page, code, globalName, fileName, size) {
        await page.evaluate(([code, g, name, size]) => {
            window.__done = false;
            const f = new File([window[g]], name);
            if (size) Object.defineProperty(f, 'size', { value: size });
            (new Function('f', 'return (async () => { ' + code + ' })()'))(f)
                .then(() => { window.__done = true; }, e => { window.__done = 'THREW ' + (e && e.stack || e); });
        }, [code, globalName, fileName, size || 0]);
        for (let i = 0; i < 240; i++) {
            const s = await page.evaluate(() => {
                const b = [...document.querySelectorAll('button')].find(x => x.offsetParent && /^(Skip|Later|Cancel|Not now)$/i.test((x.textContent || '').trim()));
                if (b) { b.click(); return { clicked: true }; }
                return { done: window.__done, prompt: !!document.getElementById('videoSelectOverlay') };
            });
            if (s.prompt) await page.keyboard.press('Escape');
            if (s.done) return s.done;
            await page.waitForTimeout(250);
        }
        return 'timeout';
    }

    const newFile = path.join(tmpRoot, 'mice.slp');
    fs.copyFileSync(FIXTURE, newFile);
    const recDir = path.join(tmpRoot, 'rec');
    fs.mkdirSync(path.join(recDir, 'videos'), { recursive: true });
    fs.copyFileSync(newFile, path.join(recDir, 'mice.slp'));
    for (const cam of ['cam0', 'cam1']) fs.copyFileSync(VIDEO, path.join(recDir, 'videos', cam + '.mp4'));

    for (const [kind, openCode] of [['eager', EAGER], ['lazy', LAZY]]) {
        console.log(`\n-- a ${kind} project with behavior work open --`);
        const page = await freshPage();
        await page.evaluate(async (fixture) => {
            window.__new = new Uint8Array(await (await fetch(fixture)).arrayBuffer());
            // Both built before the ethogram exists, because a save writes the ethogram it finds.
            window.__project = await window.__projectBytes('open-project', ['cam0', 'cam1']);
            window.__other = await window.__projectBytes('other', ['cam0', 'cam1']);
        }, '/' + path.relative(repoRoot, FIXTURE).split(path.sep).join('/'));

        const opened = await run(page, openCode, '__project', 'open-project.slp');
        await page.evaluate(() => window.__addBehaviorWork());
        const before = await page.evaluate(() => window.__behaviorPrint());
        const saved = JSON.parse(before);
        check(opened === true && saved.ethogram && saved.behaviorLabels && saved.behaviorPredictions && saved.behaviorStates,
            'it has an ethogram, a label, a predicted bout and a states layer', String(opened) + ' ' + before);

        const unchanged = async (label) => {
            const st = await page.evaluate(() => window.__status());
            check(REFUSED.test(st), label + ': refused', st);
            check(await page.evaluate(() => window.__sameSession()), '  the same session stays open');
            const after = await page.evaluate(() => window.__behaviorPrint());
            check(after === before, '  its behavior work is exactly as it was', after);
        };

        for (const [label, code, size] of [
            ['handleLoadSlpFile (File ▸ Load SLP)', EAGER],
            ['handleLoadProjectSlpLazy', LAZY],
            ['handleLoadProject (File ▸ Open)', ROUTED],
            ['handleLoadProject, above 200 MB (the lazy route frees the project first)', ROUTED, OVER_200_MB],
        ]) {
            await page.evaluate(() => { document.getElementById('statusText').textContent = 'before'; });
            const r = await run(page, code, '__new', 'mice.slp', size);
            check(r === true, label + ': finished', String(r));
            await unchanged(label);
        }

        page.once('filechooser', fc => fc.setFiles(newFile));
        await page.evaluate(async () => { await (await import('/import-export/slp-import.js')).handleAddSlp(); });
        await unchanged('handleAddSlp');

        page.once('filechooser', fc => fc.setFiles(recDir));
        await page.evaluate(async () => { await (await import('/loading/session-loader.js')).handleLoadSessionFolderSingleSlp(); });
        await unchanged('handleLoadSessionFolderSingleSlp');

        const replaced = await run(page, EAGER, '__other', 'other.slp');
        const after = JSON.parse(await page.evaluate(() => window.__behaviorPrint()));
        check(replaced === true && !after.ethogram && !after.behaviorLabels && !after.behaviorPredictions && !after.behaviorStates,
            'a readable file does replace them', JSON.stringify(after));
        await page.close();
    }
} catch (e) {
    console.log('  ✗ threw:', e && e.stack || e);
    fails++;
} finally {
    if (browser) await browser.close();
    server.kill();
    fs.rmSync(tmpRoot, { recursive: true, force: true });
}

console.log(fails === 0 ? '\nPASS' : `\nFAIL (${fails})`);
process.exit(fails === 0 ? 0 : 1);
