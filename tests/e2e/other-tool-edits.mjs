/**
 * other-tool-edits.mjs — a project another tool edited opens as that tool left it.
 *
 * A LUC3D project keeps its own copies of the track and identity lists in
 * `metadata.lucid`, and each group's identity. sleap-app and Python edit only the
 * standard slots and carry those copies along unread, so trusting them showed the old
 * names and animals, stripped the track from every detection of a renamed track, and
 * on the next save wrote the other tool's work away (import-export/file-edits.js).
 *
 * Each edit is made the way those tools make it: read the file with sleap-io, change the
 * standard slots, save. Then the project is reopened through both loaders
 * (handleLoadSlpFile, handleLoadProjectSlpLazy), checked, saved by LUC3D through that
 * loader's writer, and read back with sleap-io, so the edit must also survive a LUC3D save.
 *
 * Run: node tests/e2e/other-tool-edits.mjs
 */
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8267);
const FRAMES = 40, CAMS = 2, ANIMALS = 3, NODES = 3;

let fails = 0;
const check = (c, m, extra) => {
    console.log((c ? '  ✓ ' : '  ✗ ') + m + (!c && extra !== undefined ? '  ' + JSON.stringify(extra).slice(0, 800) : ''));
    if (!c) fails++;
};
const eq = (a, b, m) => check(JSON.stringify(a) === JSON.stringify(b), m, { got: a, want: b });

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
await new Promise(r => setTimeout(r, 1200));

let browser;
try {
    browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();
    const errs = [];
    const logs = [];
    page.on('pageerror', e => errs.push('pageerror: ' + String(e).slice(0, 300)));
    page.on('console', m => { if (/edited outside LUC3D/.test(m.text())) logs.push(m.text()); });
    await page.goto(`http://localhost:${PORT}/index.html`, { waitUntil: 'domcontentloaded', timeout: 120000 });
    await page.waitForFunction(() => window.SleapIO && window.h5wasm && window.__lucid, null, { timeout: 120000 });

    // Fixture: every detection's first x says which it is, whatever its position.
    await page.evaluate(async ({ FRAMES, CAMS, ANIMALS, NODES }) => {
        const pd = await import('/pose/pose-data.js');
        const fileio = await import('/import-export/file-io.js');
        const { Skeleton, Camera, Instance, InstanceGroup, FrameGroup, Session } = pd;
        const camNames = Array.from({ length: CAMS }, (_, i) => 'cam' + i);
        const nodeNames = Array.from({ length: NODES }, (_, i) => 'n' + i);
        const M = [[900, 0, 256], [0, 900, 256], [0, 0, 1]];
        const cameras = camNames.map((n, i) => new Camera(n, M, [0, 0, 0, 0, 0], [0, i * 0.3, 0], [i * 10, 0, 0], [512, 512]));
        const skeleton = new Skeleton('skeleton', nodeNames, nodeNames.slice(1).map((_, i) => [i, i + 1]));
        const tracks = Array.from({ length: ANIMALS }, (_, i) => 'track_' + i);
        const session = new Session(cameras, skeleton, tracks, 'Edits');
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
                    g.addInstance(cn, inst);
                    session.setFrameIdentity(f, cn, a, a);
                });
                g.points3d = new Float64Array(NODES * 3).map((_, i) => f + a * 40 + i * 0.5);
                gs.push(g);
            }
            session.instanceGroups.set(f, gs);
        }
        // A single-view fix: not in any standard slot, and must outlive others' edits.
        session.setFrameIdentity(5, 'cam0', 0, 2);
        const views = camNames.map(n => ({ name: n, videoWidth: 512, videoHeight: 512 }));
        const videoFiles = camNames.map(n => ({ name: n, assignedCamera: n, videoPath: n + '.mp4' }));
        window.__base = await window.SleapIO.saveSlpToBytes(fileio.buildSlpLabelsAllViews(session, views, videoFiles));

        const readTyped = (bytes) => window.SleapIO.readSlpStreaming(new File([bytes], 'x.slp'), {
            openVideos: false,
            h5wasmUrl: new URL('lib/h5wasm/h5wasm.iife.js', document.baseURI).href,
        });
        window.__typed = readTyped;

        // What other tools do. Each takes a freshly read file and edits it in place.
        const groupsOf = (labels) => labels.sessions[0].frameGroups;
        const everyInstance = (labels, fn) => {
            for (const lf of labels.labeledFrames) for (const inst of lf.instances) fn(inst, lf);
        };
        window.__edits = {
            none() {},
            renameTrack(labels) { labels.tracks[1].name = 'renamed'; },
            newTrack(labels) {
                const t = new window.SleapIO.Track('added');
                labels.tracks.push(t);
                const lf = labels.labeledFrames.find((x) => x.frameIdx === 7 && labels.videos.indexOf(x.video) === 1);
                lf.instances[2].track = t;
            },
            deleteTrack(labels) {
                const gone = labels.tracks[0];
                everyInstance(labels, (inst) => { if (inst.track === gone) inst.track = null; });
                labels.tracks.splice(0, 1);
            },
            renameIdentity(labels) { labels.identities[0].name = 'mouse_A'; },
            swapIdentities(labels) {
                const [a, b] = labels.identities;
                for (const [f, fg] of groupsOf(labels)) {
                    if (f < 20) continue;
                    for (const g of fg.instanceGroups) {
                        if (g.identity === a) g.identity = b;
                        else if (g.identity === b) g.identity = a;
                    }
                }
            },
            newIdentity(labels) {
                const fresh = new window.SleapIO.Identity({ name: 'fresh', color: '#00ff00' });
                labels.identities.push(fresh);
                groupsOf(labels).get(3).instanceGroups[1].identity = fresh;
            },
            clearIdentity(labels) { groupsOf(labels).get(3).instanceGroups[1].identity = undefined; },
        };
        window.__edited = async (name) => {
            const labels = await readTyped(window.__base);
            window.__edits[name](labels);
            window.__file = await window.SleapIO.saveSlpToBytes(labels);
        };

        // What a reopened LUC3D session holds, group by group.
        window.__dump = () => {
            const s = window.__lucid.state.session;
            const members = {};
            for (const [f, gs] of s.instanceGroups) {
                gs.forEach((g, gi) => {
                    const id = g.identityId >= 0 && s.identities[g.identityId] ? s.identities[g.identityId].name : null;
                    for (const [cn, inst] of g.instances) {
                        const track = inst.trackIdx != null && inst.trackIdx >= 0 ? s.tracks[inst.trackIdx] : null;
                        const perDetection = inst.trackIdx != null && inst.trackIdx >= 0 && s.hasFrameIdentity(f, cn, inst.trackIdx)
                            ? (s.identities[s.getFrameIdentityValue(f, cn, inst.trackIdx)] || {}).name : null;
                        members[`${f}/${gi}/${cn}`] = [id, track, perDetection];
                    }
                });
            }
            return {
                lazy: !!s.lazyLoader,
                tracks: s.tracks.slice(),
                identities: s.identities.map((i) => i.name),
                members,
            };
        };
        // The same from a file, through sleap-io: only the standard slots.
        window.__dumpFile = async (bytes) => {
            const labels = await readTyped(bytes);
            const members = {};
            for (const [f, fg] of labels.sessions[0].frameGroups) {
                fg.instanceGroups.forEach((g, gi) => {
                    for (const [cam, inst] of g.instanceByCamera) {
                        members[`${f}/${gi}/${cam.name}`] = [g.identity ? g.identity.name : null, inst.track ? inst.track.name : null];
                    }
                });
            }
            return { tracks: labels.tracks.map((t) => t.name), identities: labels.identities.map((i) => i.name), members };
        };
        window.__saveEager = async () => {
            const AS = await import('/ui/app-state.js');
            const s = AS.state.session;
            window.__saved = await window.SleapIO.saveSlpToBytes(fileio.buildSlpLabelsAllViews(s, AS.state.views, AS.state.videoFiles));
        };
        window.__saveLazy = async () => {
            const saveLoad = await import('/import-export/save-load.js');
            const AS = await import('/ui/app-state.js');
            window.__saved = await saveLoad.saveAllSessionsStreaming(AS.state.sessions);
        };
    }, { FRAMES, CAMS, ANIMALS, NODES });

    async function loadEager() {
        await page.evaluate(async () => {
            const imp = await import('/import-export/slp-import.js');
            window.__loadDone = false;
            imp.handleLoadSlpFile(new File([window.__file], 'edited.slp'))
                .then(() => { window.__loadDone = true; }, (e) => { window.__loadDone = 'ERR ' + (e && e.stack || e); });
        });
        for (let i = 0; i < 400; i++) {
            const s = await page.evaluate(() => ({ done: window.__loadDone, prompt: !!document.getElementById('videoSelectOverlay') }));
            if (s.prompt) await page.keyboard.press('Escape');
            if (s.done) return s.done;
            await page.waitForTimeout(100);
        }
        return 'timeout';
    }
    async function loadLazy() {
        await page.evaluate(() => {
            window.__loadDone = false;
            (async () => {
                try {
                    const sl = await import('/loading/session-loader.js');
                    await sl.handleLoadProjectSlpLazy(new File([window.__file], 'edited.slp'));
                    window.__loadDone = true;
                } catch (e) { window.__loadDone = 'ERR ' + String(e && e.stack || e).slice(0, 500); }
            })();
        });
        for (let i = 0; i < 400; i++) {
            const s = await page.evaluate(() => {
                const b = [...document.querySelectorAll('button')].find(x => /Skip|Later/i.test(x.textContent || ''));
                if (b) b.click();
                return window.__loadDone;
            });
            if (s) return s;
            await page.waitForTimeout(100);
        }
        return 'timeout';
    }

    /** The untouched project: `f/g/cam` → [group identity, track, per-detection identity]. */
    function untouched() {
        const out = {};
        for (let f = 0; f < FRAMES; f++) {
            for (let a = 0; a < ANIMALS; a++) {
                for (let c = 0; c < CAMS; c++) out[`${f}/${a}/cam${c}`] = [`id_${a}`, `track_${a}`, `id_${a}`];
            }
        }
        out['5/0/cam0'][2] = 'id_2'; // the single-view fix
        return out;
    }

    /**
     * One edit, both loaders: `expect` adjusts the untouched project, in LUC3D's terms.
     * Each route is then saved by LUC3D and read back through the standard slots.
     */
    async function scenario(name, expect) {
        console.log(`\n-- ${name} --`);
        await page.evaluate((n) => window.__edited(n), name);
        for (const route of ['eager', 'lazy']) {
            logs.length = 0;
            const done = route === 'eager' ? await loadEager() : await loadLazy();
            check(done === true, `${route}: loads`, done);
            const got = await page.evaluate(() => window.__dump());
            eq(got.lazy, route === 'lazy', `${route}: the route is what it says`);
            const want = { tracks: ['track_0', 'track_1', 'track_2'], identities: ['id_0', 'id_1', 'id_2'], members: untouched() };
            expect(want, route);
            eq(got.tracks, want.tracks, `${route}: tracks`);
            eq(got.identities, want.identities, `${route}: identities`);
            const wrong = Object.keys(want.members).filter((k) => JSON.stringify(got.members[k]) !== JSON.stringify(want.members[k]));
            check(wrong.length === 0, `${route}: every group member (${Object.keys(want.members).length})`,
                wrong.slice(0, 4).map((k) => [k, got.members[k], want.members[k]]));
            // A lazy project's tracks are the file's own, so there is nothing to follow.
            eq(logs.length > 0, want.edited !== false && !(route === 'lazy' && want.lazySilent),
                `${route}: says when it followed the file`);

            await page.evaluate((r) => (r === 'eager' ? window.__saveEager() : window.__saveLazy()), route);
            const saved = await page.evaluate(() => window.__dumpFile(window.__saved));
            const standard = {};
            for (const [k, v] of Object.entries(want.members)) standard[k] = [v[0], v[1]];
            const wrongSaved = Object.keys(standard).filter((k) => JSON.stringify(saved.members[k]) !== JSON.stringify(standard[k]));
            check(wrongSaved.length === 0, `${route}: the edit survives a LUC3D save`,
                wrongSaved.slice(0, 4).map((k) => [k, saved.members[k], standard[k]]));
            for (const old of want.goneFromFile || []) {
                check(!saved.tracks.includes(old) && !saved.identities.includes(old), `${route}: ${old} is gone from the saved file`, saved);
            }
        }
    }

    await scenario('none', (w) => { w.edited = false; });

    await scenario('renameTrack', (w) => {
        w.tracks = ['track_0', 'renamed', 'track_2'];
        for (const v of Object.values(w.members)) if (v[1] === 'track_1') v[1] = 'renamed';
        w.goneFromFile = ['track_1'];
        w.lazySilent = true;
    });

    await scenario('newTrack', (w) => {
        w.tracks = ['track_0', 'track_1', 'track_2', 'added'];
        // The per-detection identity comes from the group, on its new track too.
        w.members['7/2/cam1'] = ['id_2', 'added', 'id_2'];
        w.lazySilent = true;
    });

    await scenario('deleteTrack', (w) => {
        w.tracks = ['track_1', 'track_2'];
        for (const v of Object.values(w.members)) {
            if (v[1] === 'track_0') { v[1] = null; v[2] = null; }
        }
        w.goneFromFile = ['track_0'];
        w.lazySilent = true;
    });

    await scenario('renameIdentity', (w) => {
        w.identities = ['mouse_A', 'id_1', 'id_2'];
        for (const v of Object.values(w.members)) {
            if (v[0] === 'id_0') v[0] = 'mouse_A';
            if (v[2] === 'id_0') v[2] = 'mouse_A';
        }
        w.goneFromFile = ['id_0'];
    });

    await scenario('swapIdentities', (w) => {
        for (const [k, v] of Object.entries(w.members)) {
            if (Number(k.split('/')[0]) < 20) continue;
            if (v[0] === 'id_0') { v[0] = 'id_1'; v[2] = 'id_1'; }
            else if (v[0] === 'id_1') { v[0] = 'id_0'; v[2] = 'id_0'; }
        }
    });

    await scenario('newIdentity', (w) => {
        w.identities = ['id_0', 'id_1', 'id_2', 'fresh'];
        for (const cam of ['cam0', 'cam1']) w.members[`3/1/${cam}`] = ['fresh', 'track_1', 'fresh'];
    });

    await scenario('clearIdentity', (w) => {
        for (const cam of ['cam0', 'cam1']) w.members[`3/1/${cam}`] = [null, 'track_1', null];
    });

    // The session-folder loader takes the file's tracks too, so its keys must follow them.
    console.log('\n-- deleted track, through Load Single Session Folder --');
    await page.evaluate(() => window.__edited('deleteTrack'));
    const folderDir = fs.mkdtempSync(path.join(os.tmpdir(), 'lucid-other-tool-edits-'));
    try {
        fs.mkdirSync(path.join(folderDir, 'videos'), { recursive: true });
        fs.writeFileSync(path.join(folderDir, 'project.slp'),
            Buffer.from(await page.evaluate(() => Array.from(window.__file))));
        const video = path.join(repoRoot, 'tests', 'fixtures', 'barcode-60', 'barcode-60.mp4');
        for (const cam of ['cam0', 'cam1']) fs.copyFileSync(video, path.join(folderDir, 'videos', cam + '.mp4'));
        page.once('filechooser', (fc) => fc.setFiles(folderDir));
        await page.evaluate(async () => {
            const AS = await import('/ui/app-state.js');
            AS.state.sessions = []; AS.state.activeSessionIdx = 0;
            AS.state.videoFiles = []; AS.state.session = null; AS.state.views = [];
            const m = await import('/loading/session-loader.js');
            await m.handleLoadSessionFolderSingleSlp();
        });
        const got = await page.evaluate(() => window.__dump());
        eq(got.tracks, ['track_1', 'track_2'], 'tracks');
        const wantMembers = untouched();
        for (const v of Object.values(wantMembers)) {
            if (v[1] === 'track_0') { v[1] = null; v[2] = null; }
        }
        const wrong = Object.keys(wantMembers).filter((k) => JSON.stringify(got.members[k]) !== JSON.stringify(wantMembers[k]));
        check(wrong.length === 0, 'every group member, on its own identity',
            wrong.slice(0, 4).map((k) => [k, got.members[k], wantMembers[k]]));
    } finally {
        fs.rmSync(folderDir, { recursive: true, force: true });
    }

    // LUC3D's own edits on a lazy project must not read as another tool's next time: they
    // reach the file's track list, not only LUC3D's copy.
    console.log('\n-- renamed and created in LUC3D, on a lazy project --');
    await page.evaluate(() => { window.__file = window.__base; });
    check((await loadLazy()) === true, 'lazy: loads');
    await page.evaluate(async () => {
        const s = window.__lucid.state.session;
        s.tracks[1] = 'renamed_here';
        s.tracks.push('created_here');
        await window.__saveLazy();
        window.__file = window.__saved;
    });
    const named = await page.evaluate(() => window.__dumpFile(window.__saved));
    eq(named.tracks, ['track_0', 'renamed_here', 'track_2', 'created_here'], "the file's track list has them");
    for (const route of ['eager', 'lazy']) {
        logs.length = 0;
        check((await (route === 'eager' ? loadEager() : loadLazy())) === true, `${route}: reopens`);
        const got = await page.evaluate(() => window.__dump());
        eq(got.tracks, ['track_0', 'renamed_here', 'track_2', 'created_here'], `${route}: tracks`);
        check(got.members['3/1/cam0'][1] === 'renamed_here' && got.members['3/1/cam0'][2] === 'id_1',
            `${route}: the detections follow the rename`, got.members['3/1/cam0']);
        eq(logs.length, 0, `${route}: and it does not read as another tool's edit`);
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
