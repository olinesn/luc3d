/**
 * suggested-frame-nav.mjs — Previous / Next suggested frame (Opt+Shift+← / →) steps
 * through the frames a `.slp`'s suggestions name, in the file's order, and the
 * seekbar ticks them (ui/suggested-frame-nav.js, pose/suggested-frames.js).
 *
 * The file is a predictions-only project, as a relabel scan writes: Opt+← / → (user-
 * labelled frames) has nothing to jump to. Its suggestions are ranked, not in frame
 * order, and each rank is listed once per camera.
 *
 *  A. File ▸ Open (the eager load), with the videos: the keys step in rank order,
 *     each frame once, and the status line says "Suggestion k of n". A suggestion
 *     on a video no camera of the session reads, or on the other session's cameras,
 *     is not one of the session's. The seekbar's thumb follows.
 *  B. Off a suggested frame, a step carries on from the one last stepped to: Next to
 *     the one after it, Previous back to it. Steps stop at both ends. → still steps
 *     one frame.
 *  C. The seekbar ticks exactly the session's suggested frames, and follows a
 *     session switch; each session remembers its own last suggestion. A suggestion
 *     the session cannot show (here: no video loaded for it) is named, not landed on.
 *  D. Stepping changes nothing in the project (not dirty), and a file without
 *     suggestions says so, moves nowhere, and clears the ticks.
 *  E. The two shortcuts are dispatched catalog entries with a handler, on keys no
 *     other dispatched action uses.
 *  F. The lazy load File ▸ Open takes for a large file, without videos (the
 *     video-less frame path): the same steps, and the keys step no frame of their own.
 *  G. A single-.slp session folder with its videos: the same steps.
 *
 * Real Python sleap-io writes the files. Needs `uv` (skipped without it).
 *
 * Run: node tests/e2e/suggested-frame-nav.mjs   (spawns its own http.server; PORT= to move it)
 */
import { chromium } from 'playwright';
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8479);
const PY_SPEC = process.env.SLEAP_IO_SPEC || 'sleap-io==0.9.5';

let fails = 0;
const check = (c, m, got) => {
    console.log((c ? '  ✓ ' : '  ✗ ') + m + (!c && got !== undefined ? ' — got ' + JSON.stringify(got) : ''));
    if (!c) fails++;
    stallGuard.refresh();
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// Two sessions (camA/camB, camC/camD) and a video no session names. Predictions only,
// no tracks. The suggestions, in rank order:
//   30 on camA and camB; 5 on camB and camA; 44 on the outside video; 12 on camC;
//   17 on camA; 5 on camA again; 42 on camB; 3 on camD.
// Session 1's suggested frames are therefore [30, 5, 17, 42], session 2's [12, 3].
const PY_WRITE = `
import os, sys
import numpy as np
import sleap_io as sio
from sleap_io.model.suggestions import SuggestionFrame

out = sys.argv[1]
sk = sio.Skeleton(nodes=["nose", "tail"], edges=[("nose", "tail")])

def video(name):
    # The 60-frame barcode clip each test video is a copy of.
    v = sio.Video(filename=name, open_backend=False)
    v.backend_metadata["shape"] = (60, 32, 256, 3)
    return v

def predictions(v, base):
    return [sio.LabeledFrame(video=v, frame_idx=f, instances=[
        sio.PredictedInstance.from_numpy(np.array([[base + f, 10.0], [base + f + 5.0, 20.0]]), skeleton=sk,
                                         score=0.9, point_scores=np.array([0.5, 0.8]))]) for f in range(50)]

def session(pairs):
    cams = [sio.Camera(name=n, rvec=np.array([0.0, 0.1 * i, 0.0]), tvec=np.array([10.0 * i, 0.0, 0.0]),
                       matrix=np.array([[500.0, 0, 32], [0, 500.0, 32], [0, 0, 1]]), dist=np.zeros(5), size=(64, 64))
            for i, (n, _) in enumerate(pairs)]
    s = sio.RecordingSession(camera_group=sio.CameraGroup(cameras=cams))
    for c, (_, v) in zip(cams, pairs):
        s.add_video(v, c)
    return s

def sug(v, f):
    return SuggestionFrame(video=v, frame_idx=f)

a, b, c, d, x = video("camA.mp4"), video("camB.mp4"), video("camC.mp4"), video("camD.mp4"), video("outside.mp4")
frames = predictions(a, 100) + predictions(b, 200) + predictions(c, 300) + predictions(d, 400) + predictions(x, 500)
s1, s2 = session([("camA", a), ("camB", b)]), session([("camC", c), ("camD", d)])
sio.Labels(labeled_frames=frames, videos=[a, b, c, d, x], skeletons=[sk], sessions=[s1, s2],
           suggestions=[sug(a, 30), sug(b, 30), sug(b, 5), sug(a, 5), sug(x, 44), sug(c, 12), sug(a, 17),
                        sug(a, 5), sug(b, 42), sug(d, 3)]).save(os.path.join(out, "ranked.slp"), embed=False)

a, b = video("camA.mp4"), video("camB.mp4")
sio.Labels(labeled_frames=predictions(a, 100) + predictions(b, 200), videos=[a, b], skeletons=[sk],
           sessions=[session([("camA", a), ("camB", b)])], suggestions=[]).save(os.path.join(out, "none.slp"), embed=False)

# A session folder: labels.slp + videos/<camera>.mp4 (the 60-frame barcode clip).
a, b = video("videos/camA.mp4"), video("videos/camB.mp4")
os.makedirs(os.path.join(out, "withvideo", "videos"))
sio.Labels(labeled_frames=predictions(a, 100) + predictions(b, 200), videos=[a, b], skeletons=[sk],
           sessions=[session([("camA", a), ("camB", b)])],
           suggestions=[sug(a, 30), sug(b, 30), sug(b, 5), sug(a, 5), sug(a, 17), sug(b, 42)]
           ).save(os.path.join(out, "withvideo", "labels.slp"), embed=False)
`;

if (spawnSync('uv', ['--version'], { encoding: 'utf8' }).status !== 0) {
    console.log('  - skipped: no uv');
    process.exit(0);
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'suggested-frame-nav-'));
const py = spawnSync('uv', ['run', '--no-project', '--quiet', '--python', '3.12', '--with', PY_SPEC, '--with', 'numpy', '--with', 'h5py',
    'python', '-I', '-c', PY_WRITE, tmp], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
if (py.status !== 0) {
    console.log('  ✗ python failed: ' + (py.stderr || '').trim().split('\n').slice(-6).join(' | '));
    process.exit(1);
}
const RANKED = fs.readFileSync(path.join(tmp, 'ranked.slp'));
const CLIP = path.join(repoRoot, 'tests', 'fixtures', 'barcode-60', 'barcode-60.mp4');
// The videos ranked.slp names, for File ▸ Open's video prompt; and the session folder's.
fs.mkdirSync(path.join(tmp, 'videos'));
for (const name of ['camA', 'camB', 'camC', 'camD', 'outside']) fs.copyFileSync(CLIP, path.join(tmp, 'videos', name + '.mp4'));
for (const cam of ['camA', 'camB']) fs.copyFileSync(CLIP, path.join(tmp, 'withvideo', 'videos', cam + '.mp4'));
const NONE = fs.readFileSync(path.join(tmp, 'none.slp'));

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
for (let i = 0; i < 60; i++) {
    try { if ((await fetch(`http://localhost:${PORT}/index.html`)).ok) break; } catch { /* not yet */ }
    await new Promise(r => setTimeout(r, 250));
}
const browser = await chromium.launch();
const stallGuard = setTimeout(() => {
    console.log('\nStuck: no check for 5 minutes');
    browser.close();
}, 5 * 60 * 1000);
stallGuard.unref();

async function newAppPage() {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    page.on('pageerror', e => { console.log('  [pageerror]', String(e).slice(0, 300)); fails++; });
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state && window.SleapIO, null, { timeout: 60000 });
    return page;
}

async function putBytes(page, name, bytes) {
    await page.evaluate(([n, b64]) => { window[n] = Uint8Array.from(atob(b64), ch => ch.charCodeAt(0)); }, [name, Buffer.from(bytes).toString('base64')]);
}

const ROUTES = {
    // File ▸ Open: a file this small takes the eager load.
    eager: (name) => `async () => { await (await import('/import-export/save-load.js')).handleLoadProject(new File([window.__file], ${JSON.stringify(name)})); }`,
    // The route File ▸ Open takes for a large file.
    lazy: (name) => `async () => { await (await import('/loading/session-loader.js')).handleLoadProjectSlpLazy(new File([window.__file], ${JSON.stringify(name)})); }`,
};

/**
 * Open a file by `route`. With `videoFolder`, the video prompt is answered with it
 * (Select Session Folder); other prompts are skipped.
 */
async function open(page, bytes, route, name, videoFolder) {
    await putBytes(page, '__file', bytes);
    if (videoFolder) page.once('filechooser', fc => fc.setFiles(videoFolder));
    await page.evaluate((src) => {
        window.__done = false;
        (0, eval)('(' + src + ')')().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); });
    }, ROUTES[route](name));
    let done = false, answered = false;
    for (const end = Date.now() + 240000; Date.now() < end && !done;) {
        const s = await page.evaluate(() => ({
            folder: !![...document.querySelectorAll('#videoSelectOverlay button')].find(b => /Select Session Folder/.test(b.textContent)),
            prompt: !!document.getElementById('videoSelectOverlay'),
            done: window.__done,
        }));
        if (s.folder && videoFolder && !answered) {
            answered = true;
            await page.click('#videoSelectOverlay button:has-text("Select Session Folder")');
        } else if (!videoFolder || answered) {
            const skipped = await page.evaluate(() => {
                const b = [...document.querySelectorAll('button')].find(x => x.offsetParent && /^(Skip|Later|Cancel|Not now)$/i.test((x.textContent || '').trim()));
                if (b) b.click();
                return !!b;
            });
            if (!skipped && s.prompt) await page.keyboard.press('Escape');
        }
        done = s.done;
        if (!done) await page.waitForTimeout(100);
    }
    if (done !== true) throw new Error(route + ' load of ' + name + ': did not finish: ' + done);
    // The two sessions' calibrations differ: the load ends with a note naming them. Esc closes it.
    for (let i = 0; i < 20; i++) {
        if (!(await page.evaluate(() => !!document.getElementById('calibNoticeOverlay')))) break;
        await page.keyboard.press('Escape');
        await page.waitForTimeout(100);
    }
    if (videoFolder) check(answered, `${route}: the video prompt was answered with the videos' folder`);
}

const view = (page) => page.evaluate(() => ({
    frame: window.__lucid.state.currentFrame,
    status: document.getElementById('statusText').textContent,
    ticks: [...document.querySelectorAll('#seekbarSuggestionMarks .seekbar-suggestion-mark')].map(m => Number(m.dataset.frame)).sort((p, q) => p - q),
    session: window.__lucid.state.sessions.indexOf(window.__lucid.state.session),
    dirty: !!window.__lucid.state.isDirty,
    thumb: parseFloat(document.getElementById('seekbarThumb').style.left),
    total: window.__lucid.state.totalFrames,
    views: window.__lucid.state.views.map(x => x.name + (x.decoder ? '' : ' (no decoder)')),
}));

/** Press a key on the page itself (nothing focused), wait for `frame` (a decode is async), then read where it went. */
async function stepTo(page, key, frame) {
    await page.evaluate(() => { if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur(); });
    await page.keyboard.press(key);
    await page.waitForFunction(f => window.__lucid.state.currentFrame === f, frame, { timeout: 20000 }).catch(() => {});
    await page.waitForTimeout(50);   // a stray extra step would show by now
    return view(page);
}

async function goTo(page, frame) {
    await page.evaluate(async (f) => { (await import('/pose/initialization.js')).navigateToFrame(f); }, frame);
    await page.waitForFunction(f => window.__lucid.state.currentFrame === f, frame, { timeout: 20000 }).catch(() => {});
}

const decoded = (page, n) => page.waitForFunction(k => window.__lucid.state.views.length === k && window.__lucid.state.views.every(x => x.decoder), n, { timeout: 60000 });

const NEXT = 'Alt+Shift+ArrowRight', PREV = 'Alt+Shift+ArrowLeft';

/** Sections A and B on session 1 ([30, 5, 17, 42]), from frame 0. `video`: the thumb check. */
async function stepSession1(page, tag, video) {
    await goTo(page, 0);
    let v;
    const want = [[30, 'Suggestion 1 of 4: frame 31'], [5, 'Suggestion 2 of 4: frame 6'],
        [17, 'Suggestion 3 of 4: frame 18'], [42, 'Suggestion 4 of 4: frame 43']];
    for (const [frame, status] of want) {
        v = await stepTo(page, NEXT, frame);
        check(v.frame === frame && v.status === status, `${tag}: Next → frame ${frame}, "${status}"`, [v.frame, v.status]);
    }
    if (video) check(Math.abs(v.thumb - 100 * 42 / (v.total - 1)) < 0.01, `${tag}: the seekbar thumb is on frame 42`, [v.thumb, v.total]);
    v = await stepTo(page, NEXT, 42);
    check(v.frame === 42 && v.status === 'No next suggestion: 4 of 4 is the last', `${tag}: Next at the last stays there and says so`, [v.frame, v.status]);
    v = await stepTo(page, PREV, 17);
    check(v.frame === 17 && v.status === 'Suggestion 3 of 4: frame 18', `${tag}: Previous → rank 3`, [v.frame, v.status]);
    await stepTo(page, PREV, 5);
    await stepTo(page, PREV, 30);
    v = await stepTo(page, PREV, 30);
    check(v.frame === 30 && v.status === 'No previous suggestion: 1 of 4 is the first', `${tag}: Previous at the first stays there and says so`, [v.frame, v.status]);

    // B. Off a suggested frame: carry on from the one last stepped to.
    await stepTo(page, NEXT, 5);                                   // rank 2
    v = await stepTo(page, 'ArrowRight', 6);
    check(v.frame === 6, `${tag}: → still steps one frame`, v.frame);
    v = await stepTo(page, NEXT, 17);
    check(v.frame === 17 && v.status === 'Suggestion 3 of 4: frame 18', `${tag}: off rank 2, Next → rank 3`, [v.frame, v.status]);
    await goTo(page, 20);
    v = await stepTo(page, PREV, 17);
    check(v.frame === 17 && v.status === 'Suggestion 3 of 4: frame 18', `${tag}: off rank 3, Previous → back to rank 3`, [v.frame, v.status]);
}

console.log('\nA–E. File ▸ Open with the videos (eager)');
{
    const page = await newAppPage();
    await open(page, RANKED, 'eager', 'ranked.slp', path.join(tmp, 'videos'));
    const lazy = await page.evaluate(() => !!window.__lucid.state.session.lazyLoader);
    check(!lazy, 'eager: opened eagerly', lazy);
    await decoded(page, 2);
    let v = await view(page);
    check(v.session === 0 && v.total === 60 && eq(v.ticks, [5, 17, 30, 42]), "eager: 60 frames from the video, and the seekbar ticks session 1's suggested frames", v);
    const dirtyBefore = v.dirty;
    await stepSession1(page, 'eager', true);

    // C. The other session: its own frames and ticks, and each session its own last suggestion.
    await page.evaluate(async () => { await (await import('/ui/sessions-panes.js')).switchSession(1); });
    await page.waitForFunction(() => window.__lucid.state.sessions.indexOf(window.__lucid.state.session) === 1, null, { timeout: 30000 }).catch(() => {});
    await decoded(page, 2).catch(() => {});
    await goTo(page, 0);
    v = await view(page);
    check(v.session === 1 && eq(v.ticks, [3, 12]), "eager: after a switch the seekbar ticks session 2's frames", [v.session, v.ticks, v.views, v.total]);
    // This load opened only session 1's videos, so session 2 has no frames to show:
    // a step names its suggestion and stays put rather than land on another frame.
    v = await stepTo(page, NEXT, 0);
    check(v.frame === 0 && v.status === 'Suggestion 1 of 2 is frame 13, but this session has no frames to show',
        "eager: session 2's first suggestion is camC's 12, which it cannot show", [v.frame, v.status, v.views, v.total]);
    v = await stepTo(page, NEXT, 0);
    check(v.frame === 0 && v.status === 'Suggestion 2 of 2 is frame 4, but this session has no frames to show', "eager: then camD's 3", [v.frame, v.status]);
    await page.evaluate(async () => { await (await import('/ui/sessions-panes.js')).switchSession(0); });
    await page.waitForFunction(() => window.__lucid.state.sessions.indexOf(window.__lucid.state.session) === 0, null, { timeout: 30000 }).catch(() => {});
    await decoded(page, 2).catch(() => {});
    await goTo(page, 0);
    v = await view(page);
    check(v.session === 0 && eq(v.ticks, [5, 17, 30, 42]), 'eager: back on session 1, its ticks again', [v.session, v.ticks]);
    // Session 1 last stepped to rank 3 (17); session 2's steps did not move that.
    v = await stepTo(page, NEXT, 42);
    check(v.frame === 42 && v.status === 'Suggestion 4 of 4: frame 43', 'eager: session 1 kept its own last suggestion (Next → rank 4)', [v.frame, v.status]);

    // D. Stepping is not an edit.
    v = await view(page);
    check(v.dirty === dirtyBefore, `eager: stepping leaves the project's dirty flag as it was (${dirtyBefore})`, v.dirty);

    // E. The catalog.
    const cat = await page.evaluate(async () => {
        const s = await import('/ui/settings.js');
        const acts = s.getActions();
        const mine = acts.filter(a => a.id === 'prevSuggestedFrame' || a.id === 'nextSuggestedFrame');
        const keys = new Set(mine.map(a => a.binding));
        return {
            mine: mine.map(a => [a.id, a.label, a.category, a.binding, a.dispatched]),
            clash: acts.filter(a => a.dispatched && !mine.includes(a) && keys.has(a.binding)).map(a => a.id),
        };
    });
    check(eq(cat.mine, [['prevSuggestedFrame', 'Previous suggested frame', 'Navigation', 'Alt+Shift+ArrowLeft', true],
        ['nextSuggestedFrame', 'Next suggested frame', 'Navigation', 'Alt+Shift+ArrowRight', true]]),
        'eager: both shortcuts are dispatched Navigation entries', cat.mine);
    check(cat.clash.length === 0, 'eager: no other dispatched action uses their keys', cat.clash);

    // D. A file without suggestions: says so, moves nowhere, and the ticks go.
    await open(page, NONE, 'eager', 'none.slp', path.join(tmp, 'videos'));
    await decoded(page, 2);
    await goTo(page, 7);
    v = await view(page);
    check(v.ticks.length === 0, 'eager: a file without suggestions clears the ticks', v.ticks);
    v = await stepTo(page, NEXT, 7);
    check(v.frame === 7 && v.status === 'No suggested frames in this session', 'eager: Next there says so and stays', [v.frame, v.status]);
    await page.close();
}

console.log('\nF. The lazy load, without videos (video-less frame stepping)');
{
    const page = await newAppPage();
    await open(page, RANKED, 'lazy', 'ranked.slp', null);
    const lazy = await page.evaluate(() => !!window.__lucid.state.session.lazyLoader);
    check(lazy, 'lazy: opened lazily', lazy);
    const v = await view(page);
    check(v.session === 0 && eq(v.ticks, [5, 17, 30, 42]), "lazy: the seekbar ticks session 1's suggested frames", v);
    // Exact frames also show the keys step no frame of their own on this path.
    await stepSession1(page, 'lazy', false);
    await page.close();
}

console.log('\nG. A single-.slp session folder with its videos');
{
    const page = await newAppPage();
    page.once('filechooser', fc => fc.setFiles(path.join(tmp, 'withvideo')));
    await page.evaluate(async () => { await (await import('/loading/session-loader.js')).handleLoadSessionFolderSingleSlp(); });
    await decoded(page, 2);
    const v = await view(page);
    check(v.total === 60 && eq(v.ticks, [5, 17, 30, 42]), 'folder: the seekbar ticks the suggested frames', v);
    await stepSession1(page, 'folder', true);
    await page.close();
}

clearTimeout(stallGuard);
await browser.close();
server.kill();
fs.rmSync(tmp, { recursive: true, force: true });
console.log(fails === 0 ? '\nPASS suggested-frame-nav' : `\nFAIL suggested-frame-nav: ${fails} check(s) failed`);
process.exit(fails === 0 ? 0 : 1);
