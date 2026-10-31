/**
 * save-keeps-detection-labels.mjs — a LUC3D save keeps what a file says about a detection
 * that LUC3D itself does not use: the tracker's score, and from a model an identity and
 * category with their scores and embeddings, plus the identity catalog's metadata.
 *
 * A save rebuilds every detection from LUC3D's model, so whatever the model does not hold
 * used to be wiped: 100 of 100 identities and 1,662 of 1,662 tracker scores on real
 * sleap-nn output (import-export/detection-labels.js).
 *
 * Every route that opens a file and saves it: Load SLP with no session, with groups (a
 * member is rebuilt from the typed instance), the lazy reopen (streaming writer, patch
 * `(detection labels)`), and a per-camera folder, eager and lazy. A file with none of
 * these labels stays the same (the control, plus tests/e2e/save-golden-digest.mjs).
 *
 * Run: node tests/e2e/save-keeps-detection-labels.mjs
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8279);
const FRAMES = 12, CAMS = 2, ANIMALS = 3, NODES = 3;

let fails = 0;
const check = (c, m, extra) => {
    console.log((c ? '  ✓ ' : '  ✗ ') + m + (!c && extra !== undefined ? '  ' + JSON.stringify(extra).slice(0, 900) : ''));
    if (!c) fails++;
};

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1200));

let browser;
try {
    browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();
    const errs = [];
    page.on('pageerror', e => errs.push('pageerror: ' + String(e).slice(0, 300)));
    await page.goto(`http://localhost:${PORT}/index.html`, { waitUntil: 'domcontentloaded', timeout: 120000 });
    await page.waitForFunction(() => window.SleapIO && window.h5wasm && window.__lucid, null, { timeout: 120000 });

    await page.evaluate(async ({ FRAMES, CAMS, ANIMALS, NODES }) => {
        const SIO = window.SleapIO;
        const pd = await import('/pose/pose-data.js');
        const fileio = await import('/import-export/file-io.js');
        const nodeNames = Array.from({ length: NODES }, (_, i) => 'n' + i);
        const readTyped = (bytes) => SIO.readSlpStreaming(new File([bytes], 'x.slp'), {
            openVideos: false,
            h5wasmUrl: new URL('lib/h5wasm/h5wasm.iife.js', document.baseURI).href,
        });

        // The labels every detection should keep. Its first x says which it is; the
        // middle animal has an identity and category, the last only a tracker's score.
        window.__labelsFor = (x) => {
            const animal = Math.floor((x % 1000) / 100);
            const out = { trackingScore: x / 10000 };
            if (animal === 2) return out;
            return Object.assign(out, {
                identity: ['female', 'male'][animal],
                identityScore: x / 20000,
                identityEmbedding: [x, x + 1],
                category: 'mouse',
                categoryScore: x / 40000,
                categoryEmbedding: [x + 2],
            });
        };
        const label = (inst, identities) => {
            const want = window.__labelsFor(inst._xy[0]);
            inst.trackingScore = want.trackingScore;
            if (!want.identity) return;
            inst.identity = identities.find((i) => i.name === want.identity);
            inst.identityScore = want.identityScore;
            inst.identityEmbedding = { vector: want.identityEmbedding };
            inst.category = want.category;
            inst.categoryScore = want.categoryScore;
            inst.categoryEmbedding = { vector: want.categoryEmbedding };
        };
        const catalog = () => ['female', 'male'].map((name) =>
            new SIO.Identity({ name, metadata: { strain: name + '_strain' } }));

        // A plain model output: one camera, no session.
        const plain = (cam, withLabels) => {
            const skeleton = new SIO.Skeleton({ name: 'sk', nodes: nodeNames });
            const video = new SIO.Video({ filename: `cam${cam}.mp4`, shape: [FRAMES, 64, 64, 1] });
            const tracks = Array.from({ length: ANIMALS }, (_, a) => new SIO.Track('track_' + a));
            const identities = withLabels ? catalog() : [];
            const labeledFrames = [];
            for (let f = 0; f < FRAMES; f++) {
                const instances = [];
                for (let a = 0; a < ANIMALS; a++) {
                    const x = 1000 * cam + 100 * a + f;
                    const inst = new SIO.PredictedInstance({
                        points: nodeNames.map((_, k) => [x + k, x + k + 0.5]), skeleton, track: tracks[a], score: 0.9,
                    });
                    if (withLabels) label(inst, identities);
                    instances.push(inst);
                }
                labeledFrames.push(new SIO.LabeledFrame({ video, frameIdx: f, instances }));
            }
            return SIO.saveSlpToBytes(new SIO.Labels({ skeletons: [skeleton], videos: [video], tracks, identities, labeledFrames }));
        };
        window.__plain = plain;

        // A LUC3D project with groups, then labelled the way another tool would.
        const { Skeleton, Camera, Instance, InstanceGroup, FrameGroup, Session } = pd;
        const camNames = Array.from({ length: CAMS }, (_, i) => 'cam' + i);
        const M = [[900, 0, 256], [0, 900, 256], [0, 0, 1]];
        const cameras = camNames.map((n, i) => new Camera(n, M, [0, 0, 0, 0, 0], [0, i * 0.3, 0], [i * 10, 0, 0], [512, 512]));
        const skeleton = new Skeleton('skeleton', nodeNames, nodeNames.slice(1).map((_, i) => [i, i + 1]));
        const tracks = Array.from({ length: ANIMALS }, (_, i) => 'track_' + i);
        const session = new Session(cameras, skeleton, tracks, 'Labels');
        session.identities = Array.from({ length: ANIMALS }, (_, i) => new pd.Identity(i, 'id_' + i));
        for (let f = 0; f < FRAMES; f++) {
            const fg = new FrameGroup(f);
            session.addFrameGroup(fg);
            const gs = [];
            for (let a = 0; a < ANIMALS; a++) {
                const g = new InstanceGroup(f * 10 + a + 1, a);
                camNames.forEach((cn, ci) => {
                    const x = 1000 * ci + 100 * a + f;
                    const inst = new Instance(nodeNames.map((_, k) => [x + k, x + k + 0.5]), a, 'predicted', 0.9);
                    fg.addInstance(cn, inst);
                    // The last animal's last frame stays out of any group: an unlinked detection.
                    if (a === ANIMALS - 1 && f === FRAMES - 1) return;
                    g.addInstance(cn, inst);
                    session.setFrameIdentity(f, cn, a, a);
                });
                if (!g.instances.size) continue;
                g.points3d = new Float64Array(NODES * 3).map((_, i) => f + a * 40 + i * 0.5);
                gs.push(g);
            }
            session.instanceGroups.set(f, gs);
        }
        const views = camNames.map(n => ({ name: n, videoWidth: 512, videoHeight: 512 }));
        const videoFiles = camNames.map(n => ({ name: n, assignedCamera: n, videoPath: n + '.mp4' }));
        const project = await SIO.saveSlpToBytes(fileio.buildSlpLabelsAllViews(session, views, videoFiles));
        const typed = await readTyped(project);
        const added = catalog();
        for (const id of added) typed.identities.push(id);
        for (const lf of typed.labeledFrames) for (const inst of lf.instances) label(inst, added);
        window.__project = await SIO.saveSlpToBytes(typed);
        window.__unlabelled = project;

        // `cam|frame|x` → what a saved file says, through sleap-io.
        window.__dumpLabels = async (bytes) => {
            const labels = await readTyped(bytes);
            const dets = {};
            for (const lf of labels.labeledFrames) {
                for (const inst of lf.instances) {
                    const d = { trackingScore: Number(inst.trackingScore.toFixed(6)) };
                    if (inst.identity) {
                        d.identity = inst.identity.name;
                        d.identityScore = Number(inst.identityScore.toFixed(6));
                    }
                    if (inst.identityEmbedding) d.identityEmbedding = inst.identityEmbedding.vector;
                    if (inst.category) {
                        d.category = inst.category;
                        d.categoryScore = Number(inst.categoryScore.toFixed(6));
                    }
                    if (inst.categoryEmbedding) d.categoryEmbedding = inst.categoryEmbedding.vector;
                    dets[inst._xy[0]] = d;
                }
            }
            const groups = [];
            for (const session of labels.sessions) {
                for (const [f, fg] of session.frameGroups) {
                    fg.instanceGroups.forEach((g) => groups.push(`${f}:${g.identity ? g.identity.name : ''}:${g.instanceByCamera.size}`));
                }
            }
            return {
                dets,
                groups: groups.sort(),
                catalog: labels.identities.map((i) => [i.name, i.metadata && i.metadata.strain]),
            };
        };
        window.__wantFor = (x) => {
            const w = window.__labelsFor(x);
            const out = { trackingScore: Number(w.trackingScore.toFixed(6)) };
            if (w.identity) {
                out.identity = w.identity;
                out.identityScore = Number(w.identityScore.toFixed(6));
                out.identityEmbedding = w.identityEmbedding;
                out.category = w.category;
                out.categoryScore = Number(w.categoryScore.toFixed(6));
                out.categoryEmbedding = w.categoryEmbedding;
            }
            return out;
        };
        window.__format = async (bytes) => {
            const h5 = await import('h5wasm');
            const m = await h5.ready;
            const p = '/tmp/__labels_' + Math.random().toString(16).slice(2) + '.slp';
            m.FS.writeFile(p, bytes);
            const f = new h5.File(p, 'r');
            try {
                return { format: Number(f.get('metadata').attrs.format_id.value), links: !!f.get('identity/links'), categories: !!f.get('categories/links'), embeddings: !!f.get('embeddings') };
            } finally { f.close(); m.FS.unlink(p); }
        };
        window.__saveEager = async () => {
            const AS = await import('/ui/app-state.js');
            const s = AS.state.session;
            window.__saved = await SIO.saveSlpToBytes(fileio.buildSlpLabelsAllViews(s, AS.state.views, AS.state.videoFiles));
        };
        window.__saveLazy = async () => {
            const saveLoad = await import('/import-export/save-load.js');
            const AS = await import('/ui/app-state.js');
            window.__saved = await saveLoad.saveAllSessionsStreaming(AS.state.sessions);
        };
    }, { FRAMES, CAMS, ANIMALS, NODES });

    async function waitLoad(click) {
        for (let i = 0; i < 400; i++) {
            const s = await page.evaluate((click) => {
                if (click) {
                    const b = [...document.querySelectorAll('button')].find(x => /Skip|Later/i.test(x.textContent || ''));
                    if (b) b.click();
                }
                return { done: window.__loadDone, prompt: !!document.getElementById('videoSelectOverlay') };
            }, click);
            if (s.prompt) await page.keyboard.press('Escape');
            if (s.done) return s.done;
            await page.waitForTimeout(100);
        }
        return 'timeout';
    }
    async function loadSlp(global) {
        await page.evaluate(async (g) => {
            const imp = await import('/import-export/slp-import.js');
            window.__loadDone = false;
            imp.handleLoadSlpFile(new File([window[g]], 'in.slp'))
                .then(() => { window.__loadDone = true; }, (e) => { window.__loadDone = 'ERR ' + (e && e.stack || e); });
        }, global);
        return waitLoad(false);
    }
    async function loadLazy(global) {
        await page.evaluate((g) => {
            window.__loadDone = false;
            (async () => {
                try {
                    const sl = await import('/loading/session-loader.js');
                    await sl.handleLoadProjectSlpLazy(new File([window[g]], 'in.slp'));
                    window.__loadDone = true;
                } catch (e) { window.__loadDone = 'ERR ' + String(e && e.stack || e).slice(0, 500); }
            })();
        }, global);
        return waitLoad(true);
    }
    /** A per-camera folder of plain model outputs. `lazy` forces the lazy reader. */
    async function loadPerCamera(lazy) {
        await page.evaluate(async ({ CAMS, lazy }) => {
            const loader = await import('/loading/session-loader.js');
            const tri = await import('/pose/triangulation.js');
            const toml = (name, i) => `[cam_${i}]\nname = "${name}"\nsize = [ 64, 64,]\nmatrix = [ [ 900.0, 0.0, 32.0,], [ 0.0, 900.0, 32.0,], [ 0.0, 0.0, 1.0,],]\ndistortions = [ 0.0, 0.0, 0.0, 0.0, 0.0,]\nrotation = [ 0.0, ${i * 0.3}, 0.0,]\ntranslation = [ ${i * 10}.0, 0.0, 0.0,]\n`;
            const AS = await import('/ui/app-state.js');
            AS.state.sessions = []; AS.state.activeSessionIdx = 0;
            AS.state.videoFiles = []; AS.state.session = null; AS.state.views = [];
            const files = [];
            const named = (f, rel) => { Object.defineProperty(f, 'webkitRelativePath', { value: rel }); return f; };
            let calib = '';
            for (let c = 0; c < CAMS; c++) {
                calib += toml('cam' + c, c);
                // Videos are deferred; a camera folder only has to have one.
                files.push(named(new File([new Uint8Array(16)], `cam${c}.mp4`, { type: 'video/mp4' }), `sess/cam${c}/cam${c}.mp4`));
                const bytes = await window.__plain(c, true);
                // Zero padding past the threshold sends a real file down the lazy path.
                const parts = lazy ? [bytes, new Uint8Array(tri.LAZY_SLP_THRESHOLD + (1 << 20))] : [bytes];
                files.push(named(new File(parts, 'labels.slp'), `sess/cam${c}/labels.slp`));
            }
            files.push(named(new File([calib], 'calibration.toml'), 'sess/calibration.toml'));
            window.__loadDone = false;
            loader.handleLoadSessionFolderPerCamera(files, true)
                .then(() => { window.__loadDone = true; }, (e) => { window.__loadDone = 'ERR ' + (e && e.stack || e); });
        }, { CAMS, lazy });
        return waitLoad(true);
    }

    /** One route: every detection, the catalog and the format of what LUC3D then saved. */
    async function route(name, load, save, opts) {
        console.log(`\n-- ${name} --`);
        const done = await load();
        check(done === true, 'loads', done);
        const info = await page.evaluate(() => {
            const s = window.__lucid.state.session;
            return { lazy: !!(s && s.lazyLoader), detections: s ? s.frameGroups.size : 0, cameras: s ? s.cameras.length : 0 };
        });
        check(info.lazy === !!opts.lazy, `the route is ${opts.lazy ? 'lazy' : 'eager'}`, info);
        await page.evaluate((s) => (s === 'lazy' ? window.__saveLazy() : window.__saveEager()), save);
        const r = await page.evaluate(async () => {
            const dump = await window.__dumpLabels(window.__saved);
            const wrong = [];
            const xs = Object.keys(dump.dets).map(Number);
            for (const x of xs) {
                const got = JSON.stringify(dump.dets[x]);
                const want = JSON.stringify(window.__wantFor(x));
                if (got !== want) wrong.push([x, dump.dets[x], window.__wantFor(x)]);
            }
            return { n: xs.length, wrong: wrong.slice(0, 3), wrongCount: wrong.length, catalog: dump.catalog, groups: dump.groups, format: await window.__format(window.__saved) };
        });
        check(r.n === opts.detections, `every detection saved (${r.n})`, r.n);
        check(r.wrongCount === 0, 'each keeps its tracker score, identity, category, scores and embeddings', r.wrong);
        const catalog = new Map(r.catalog);
        check(catalog.get('female') === 'female_strain' && catalog.get('male') === 'male_strain', 'the catalog keeps its metadata', r.catalog);
        check(r.format.links && r.format.categories && r.format.embeddings && r.format.format >= 2.7, 'the file says what it holds', r.format);
        return r;
    }

    const perCam = FRAMES * CAMS * ANIMALS;
    await page.evaluate(async () => { window.__plainFile = await window.__plain(0, true); });
    await route('Load SLP, no session', () => loadSlp('__plainFile'), 'eager', { detections: FRAMES * ANIMALS });
    const eager = await route('Load SLP, with groups', () => loadSlp('__project'), 'eager', { detections: perCam });
    const lazy = await route('Lazy reopen, with groups', () => loadLazy('__project'), 'lazy', { lazy: true, detections: perCam });
    const wantGroups = FRAMES * ANIMALS - 1;
    check(eager.groups.length === wantGroups && JSON.stringify(eager.groups) === JSON.stringify(lazy.groups),
        'both routes keep the groups, and their own identities beside the detections', { eager: eager.groups.length, lazy: lazy.groups.length });
    check(eager.catalog.map((c) => c[0]).join() === 'id_0,id_1,id_2,female,male', "the session's identities come first in the catalog", eager.catalog);

    await route('Per-camera folder, eager', () => loadPerCamera(false), 'eager', { detections: perCam });
    await route('Per-camera folder, lazy', () => loadPerCamera(true), 'lazy', { lazy: true, detections: perCam });

    // A lazy project's labels are keyed by instance row, which a delete moves.
    console.log('\n-- Lazy reopen, a detection deleted, save --');
    check((await loadLazy('__project')) === true, 'loads');
    const deleted = await page.evaluate(async () => {
        const s = window.__lucid.state.session;
        const x0 = 1000 * 0 + 100 * 0 + 3;
        const r = s.lazyLoader.deleteInstanceRows((cam, frame, offset) => cam === 'cam0' && frame === 3 && offset === 0);
        await window.__saveLazy();
        const dump = await window.__dumpLabels(window.__saved);
        const wrong = Object.keys(dump.dets).map(Number)
            .filter((x) => JSON.stringify(dump.dets[x]) !== JSON.stringify(window.__wantFor(x)));
        return { deleted: r.deleted, n: Object.keys(dump.dets).length, wrong: wrong.slice(0, 5), gone: !(x0 in dump.dets) };
    });
    check(deleted.deleted === 1 && deleted.gone, 'that detection is gone', deleted);
    check(deleted.n === perCam - 1 && deleted.wrong.length === 0, 'every other keeps its own labels', deleted);

    // An edit makes LUC3D rewrite the whole camera-frame in place of its store rows.
    console.log('\n-- Lazy reopen, a camera-frame rewritten by an edit, save --');
    check((await loadLazy('__project')) === true, 'loads');
    const edited = await page.evaluate(async () => {
        const tri = await import('/pose/triangulation.js');
        const s = window.__lucid.state.session;
        await tri.ensureLazyFrameData(4);
        const group = s.instanceGroups.get(4)[0];
        group.instances.get('cam1').type = 'user';
        await window.__saveLazy();
        const dump = await window.__dumpLabels(window.__saved);
        const xs = Object.keys(dump.dets).map(Number);
        const wrong = xs.filter((x) => JSON.stringify(dump.dets[x]) !== JSON.stringify(window.__wantFor(x)));
        return { n: xs.length, wrong: wrong.slice(0, 5).map((x) => [x, dump.dets[x], window.__wantFor(x)]) };
    });
    check(edited.n === perCam && edited.wrong.length === 0, 'its detections keep their labels', edited);

    console.log('\n-- a file with none of these --');
    const none = await page.evaluate(async () => {
        const dump = await window.__dumpLabels(window.__unlabelled);
        return { format: await window.__format(window.__unlabelled), any: Object.values(dump.dets).some((d) => d.identity || d.category || d.trackingScore) };
    });
    check(!none.any && !none.format.links && !none.format.categories && !none.format.embeddings, 'a project with none writes none', none);
    for (const [name, load, save] of [['eager', () => loadSlp('__unlabelled'), 'eager'], ['lazy', () => loadLazy('__unlabelled'), 'lazy']]) {
        check((await load()) === true, `${name}: loads`);
        await page.evaluate((s) => (s === 'lazy' ? window.__saveLazy() : window.__saveEager()), save);
        const f = await page.evaluate(() => window.__format(window.__saved));
        check(!f.links && !f.categories && !f.embeddings, `${name}: and saves none`, f);
    }

    check(errs.length === 0, 'no page errors', errs);
} catch (e) {
    console.error(e);
    fails++;
} finally {
    if (browser) await browser.close();
    server.kill();
}
console.log(fails ? `\n${fails} check(s) failed` : '\nAll checks passed');
process.exit(fails ? 1 : 0);
