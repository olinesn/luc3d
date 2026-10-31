/**
 * behavior-events-outside-session.mjs — a behavior event on a video the session does
 * not hold stays in the file, on that video, however the file is opened and however
 * often it is saved. The promise in the header of `import-export/behavior-events.js`.
 *
 * A `.slp` can list videos that no recording session names. The session writers write
 * one video per camera, and a bout normally goes to the session's first camera, so such
 * an event used to vanish on a save (`handleLoadSlpFile`, the session's video set left it
 * out) or move to the first camera (`handleLoadProjectSlpLazy` and the single-`.slp`
 * folder loader read every row of a one-session file).
 *
 * This drives the three real loaders and File ▸ Save (both writers), reads each save
 * back with the real reader, opens that again and saves it again:
 *
 *  A. One session, the other video last.
 *  B. One session, the other video FIRST in the file: its index is a camera's on save.
 *  C. Two sessions and a video in neither, with an event on no video: each session shows
 *     only its own, nothing is doubled. A loader that opens only the first session keeps
 *     the second session's events too, on their videos.
 *  D. A video in no session with no event on it: not written back, as before.
 *
 * Kinds of event on the other video: a bout, a "not" label (its type must keep saying
 * what it negates), a prediction with framewise scores, an event with no animal.
 * Real Python sleap-io reads the last saves when `uv` is here (skipped without it).
 *
 * Run: node tests/e2e/behavior-events-outside-session.mjs   (spawns its own http.server; PORT= to move it)
 */
import { chromium } from 'playwright';
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const PORT = Number(process.env.PORT || 8461);
const PY_SPEC = process.env.SLEAP_IO_SPEC || 'sleap-io==0.9.5';

let fails = 0;
const check = (c, m, got) => {
    console.log((c ? '  ✓ ' : '  ✗ ') + m + (!c && got !== undefined ? ' — got ' + JSON.stringify(got) : ''));
    if (!c) fails++;
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'behavior-outside-'));

/** Runs in the page: a `.slp` from the app's own library, with exactly these videos and sessions. */
async function buildFile(spec) {
    const SIO = window.SleapIO;
    const nodes = [new SIO.Node('nose'), new SIO.Node('tail')];
    const skeleton = new SIO.Skeleton({ nodes, edges: [new SIO.Edge(nodes[0], nodes[1])], name: 'skeleton' });
    const track = new SIO.Track('mouse1');
    const animals = { a: new SIO.Identity({ name: 'animal_A' }), b: new SIO.Identity({ name: 'animal_B' }) };
    const videoOf = {};
    const videos = spec.videos.map(name => {
        const shape = [400, 64, 64, 1];
        const v = new SIO.Video({ filename: name, backendMetadata: { type: 'MediaVideo', shape, filename: name }, openBackend: false });
        v.shape = shape;
        videoOf[name] = v;
        return v;
    });
    const frames = [];
    const sessions = spec.sessions.map((names, si) => {
        const cameras = names.map((n, ci) => new SIO.Camera({
            name: 's' + si + '_cam' + ci, rvec: [0, 0.1 * ci, 0], tvec: [10 * ci, 0, 0],
            matrix: [[500, 0, 32], [0, 500, 32], [0, 0, 1]], distortions: [0, 0, 0, 0, 0], size: [64, 64],
        }));
        const session = new SIO.RecordingSession({ cameraGroup: new SIO.CameraGroup({ cameras }) });
        names.forEach((n, ci) => {
            session.addVideo(videoOf[n], cameras[ci]);
            for (let f = 0; f < 2; f++) {
                const pt = (x, y) => ({ xy: [x, y], visible: true, complete: true, score: 0.9 });
                frames.push(new SIO.LabeledFrame({ video: videoOf[n], frameIdx: f, instances: [
                    new SIO.PredictedInstance({ points: [pt(10 + f, 10), pt(20, 20)], skeleton, track, score: 0.9 }),
                ] }));
            }
        });
        return session;
    });
    const types = {
        attack: new SIO.EventType({ name: 'attack', metadata: { directed: 'true' } }),
        rear: new SIO.EventType({ name: 'rear' }),
        'not rear': new SIO.EventType({ name: 'not rear', metadata: { negates: 'rear' } }),
        stimulus: new SIO.EventType({ name: 'stimulus' }),
    };
    const events = spec.events.map(e => {
        const options = {
            type: types[e.type], video: e.video ? videoOf[e.video] : null, startFrame: e.start, endFrame: e.end,
            subject: e.subject ? animals[e.subject] : null, target: e.target ? animals[e.target] : null,
            source: e.source || '', name: e.name || '', metadata: e.metadata || {},
        };
        return e.scores ? new SIO.PredictedEvent(Object.assign(options, { score: e.score, scores: e.scores })) : new SIO.UserEvent(options);
    });
    const labels = new SIO.Labels({
        labeledFrames: frames, videos, skeletons: [skeleton], tracks: [track], identities: [animals.a, animals.b],
        sessions, eventTypes: Object.values(types), events,
    });
    window.__built = await SIO.saveSlpToBytes(labels);
}

/** Runs in the page: what a file says, by video name. */
async function readFile() {
    const fileio = await import('/import-export/file-io.js');
    const slp = await fileio.parseSlpViaSleapIO(new File([window.__saved], 'saved.slp'), () => {});
    const videos = slp.videos.map(v => v.filename);
    const sessions = (slp.sessions || []).map(s => Object.values(s.camcorder_to_video_idx_map || {}).map(Number));
    const firstCams = sessions.map(s => s[0]);
    const inSession = new Set(sessions.flat());
    const place = (vi) => {
        const si = firstCams.indexOf(vi);
        return si >= 0 ? 'session ' + si + ' first camera' : videos[vi];
    };
    const round = (a) => (a ? Array.from(a).map(x => Math.round(x * 1000) / 1000) : null);
    const events = (slp.behaviorEvents ? slp.behaviorEvents.rows : []).map(r => [
        place(r.video), r.type, r.start, r.end, r.subject, r.target, r.predicted, r.source,
        Number.isFinite(r.score) ? r.score : null, round(r.scores), r.metadata, r.name,
    ]).sort((a, b) => a[2] - b[2]);
    return {
        videos, sessions, outside: videos.filter((n, i) => !inSession.has(i)), events,
        types: slp.behaviorEvents ? slp.behaviorEvents.types.map(t => ({ name: t.name, negates: t.metadata.negates || null })) : [],
        framesOffSession: slp.frames.filter(f => !inSession.has(f.videoIdx)).length,
    };
}

/** Runs in the page: what each open session shows. */
async function sessionsShown() {
    const AS = await import('/ui/app-state.js');
    const L = await import('/pose/behavior-labels.js');
    const P = await import('/pose/behavior-predictions.js');
    const eth = await import('/pose/behavior-ethogram.js');
    const nameOf = (id) => (eth.behaviorById(id) || {}).name;
    return AS.state.sessions.map(s => ({
        bouts: L.exportBoutRows(s).map(r => [nameOf(r[0]), r[3], r[4], r[5]]).sort((a, b) => a[1] - b[1]),
        predicted: P.predictedBouts(s).length,
    }));
}

/** File ▸ Save as the app does it without a file handle. */
async function saveInPage() {
    const sl = await import('/import-export/save-load.js');
    const orig = URL.createObjectURL;
    let blob = null;
    URL.createObjectURL = (b) => { blob = b; return orig.call(URL, b); };
    try { await sl.saveProjectSlp(); } finally { URL.createObjectURL = orig; }
    if (!blob) throw new Error('saveProjectSlp produced no file: ' + (document.getElementById('statusText') || {}).textContent);
    window.__saved = new Uint8Array(await blob.arrayBuffer());
}

async function putBytes(page, name, bytes) {
    await page.evaluate(([n, b64]) => {
        window[n] = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    }, [name, Buffer.from(bytes).toString('base64')]);
}
async function getBytes(page, name) {
    const b64 = await page.evaluate((n) => {
        const u = window[n];
        let s = '';
        for (let i = 0; i < u.length; i += 32768) s += String.fromCharCode.apply(null, u.subarray(i, i + 32768));
        return btoa(s);
    }, name);
    return Buffer.from(b64, 'base64');
}

const server = spawn('python3', ['-m', 'http.server', String(PORT)], { cwd: repoRoot, stdio: 'ignore' });
let browser;
const pageErrors = [];

async function newAppPage() {
    const page = await browser.newPage();
    page.on('pageerror', e => pageErrors.push(String(e).slice(0, 300)));
    await page.goto(`http://localhost:${PORT}/index.html`);
    await page.waitForFunction(() => window.__lucid && window.__lucid.state && window.SleapIO, null, { timeout: 30000 });
    return page;
}

/** The three loaders; each is called as the app calls it, and its prompts dismissed. */
const LOADERS = {
    'File ▸ Open': async (page) => {
        return page.evaluate(() => { window.__done = false; (async () => {
            await (await import('/import-export/slp-import.js')).handleLoadSlpFile(new File([window.__file], 'project.slp'));
        })().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); }); });
    },
    'large project (lazy)': async (page) => {
        return page.evaluate(() => { window.__done = false; (async () => {
            await (await import('/loading/session-loader.js')).handleLoadProjectSlpLazy(new File([window.__file], 'project.slp'));
        })().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); }); });
    },
    'single .slp folder': async (page, bytes) => {
        const dir = fs.mkdtempSync(path.join(tmp, 'folder-'));
        fs.writeFileSync(path.join(dir, 'project.slp'), bytes);
        page.once('filechooser', fc => fc.setFiles(dir));
        return page.evaluate(() => { window.__done = false; (async () => {
            await (await import('/loading/session-loader.js')).handleLoadSessionFolderSingleSlp();
        })().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); }); });
    },
};

/** Open `bytes` with a loader, see what it shows, save, read the save. */
async function openAndSave(loader, bytes) {
    const page = await newAppPage();
    await putBytes(page, '__file', bytes);
    await LOADERS[loader](page, bytes);
    let done = false;
    for (let i = 0; i < 240 && !done; i++) {
        const s = await page.evaluate(() => {
            const b = [...document.querySelectorAll('button')].find(x => x.offsetParent && /^(Skip|Later|Cancel|Not now)$/i.test((x.textContent || '').trim()));
            if (b) { b.click(); return { done: false }; }
            return { done: window.__done, prompt: !!document.getElementById('videoSelectOverlay') };
        });
        if (s.prompt) await page.keyboard.press('Escape');
        done = s.done;
        if (!done) await page.waitForTimeout(100);
    }
    if (done !== true) throw new Error(loader + ': load did not finish: ' + done);
    const shown = await page.evaluate(sessionsShown);
    await page.evaluate(saveInPage);
    const saved = await getBytes(page, '__saved');
    const read = await page.evaluate(readFile);
    await page.close();
    return { shown, saved, read };
}

const OWN = [
    { type: 'rear', video: 'a0.mp4', start: 10, end: 19, subject: 'a', source: 'manual' },
    { type: 'rear', video: 'a1.mp4', start: 30, end: 39, subject: 'a', source: 'manual' },
];
const OTHER = [
    { type: 'attack', video: 'other.mp4', start: 300, end: 301, subject: 'b', target: 'a', source: 'manual', name: 'first', metadata: { certainty: 'high' } },
    { type: 'not rear', video: 'other.mp4', start: 310, end: 311, subject: 'a', source: 'manual' },
    { type: 'rear', video: 'other.mp4', start: 320, end: 322, subject: 'a', source: 'classifier', score: 0.5, scores: [0.25, 0.5, 0.75] },
    { type: 'stimulus', video: 'other.mp4', start: 330, end: 335, source: 'daq', metadata: { hz: '20' } },
];
// How a file reads the OTHER events, wherever the video sits in its list.
const OTHER_READ = (where) => [
    [where, 'attack', 300, 301, 'animal_B', 'animal_A', false, 'manual', null, null, { certainty: 'high' }, 'first'],
    [where, 'not rear', 310, 311, 'animal_A', '', false, 'manual', null, null, {}, ''],
    [where, 'rear', 320, 322, 'animal_A', '', true, 'classifier', 0.5, [0.25, 0.5, 0.75], {}, ''],
    [where, 'stimulus', 330, 335, '', '', false, 'daq', null, null, { hz: '20' }, ''],
];
const ownRead = (first, spans, who = 'animal_A') => spans.map(([s, e]) => [first, 'rear', s, e, who, '', false, 'manual', null, null, {}, '']);
const negatesOf = (read, name) => (read.types.find(t => t.name === name) || {}).negates;

/** A–B: one session, and a video outside it. */
async function oneSession(loader, title, videos) {
    console.log(`\n-- ${title}, ${loader} --`);
    const page = await newAppPage();
    await page.evaluate(buildFile, { videos, sessions: [['a0.mp4', 'a1.mp4']], events: OWN.concat(OTHER) });
    const source = await getBytes(page, '__built');
    await page.close();

    const one = await openAndSave(loader, source);
    check(one.shown.length === 1 && eq(one.shown[0].bouts, [['rear', 10, 19, 1], ['rear', 30, 39, 1]]), 'the session shows its own bouts, none of the other video\'s', one.shown);
    check(one.shown[0].predicted === 0, 'and none of its predictions', one.shown);
    const expected = ownRead('session 0 first camera', [[10, 19], [30, 39]]).concat(OTHER_READ('other.mp4'));
    check(eq(one.read.events, expected), 'saved: every event, the other video\'s unchanged and on that video', one.read.events);
    check(eq(one.read.outside, ['other.mp4']) && one.read.sessions[0].length === 2, 'that video is in the file, and not in the session', one.read);
    check(negatesOf(one.read, 'not rear') === 'rear', 'a "not" still says what it negates', one.read.types);
    check(one.read.framesOffSession === 0, 'no pose landed on it', one.read.framesOffSession);

    const two = await openAndSave(loader, one.saved);
    check(eq(two.shown, one.shown), 'reopened, the session shows the same', two.shown);
    check(eq(two.read.events, expected) && eq(two.read.videos, one.read.videos), 'saved again: the same events on the same videos', two.read);
    check(negatesOf(two.read, 'not rear') === 'rear', 'a "not" still a "not"', two.read.types);
    return two.saved;
}

/** C: two sessions, a video in neither, an event on no video. */
async function twoSessions(loader, opensAll) {
    console.log(`\n-- C. two sessions and a video in neither, ${loader} --`);
    const OWN_B = [
        { type: 'rear', video: 'b0.mp4', start: 110, end: 119, subject: 'b', source: 'manual' },
        { type: 'rear', video: 'b1.mp4', start: 130, end: 139, subject: 'b', source: 'manual' },
    ];
    const NO_VIDEO = { type: 'rear', video: null, start: 500, end: 501, subject: 'a', source: 'manual' };
    const page = await newAppPage();
    await page.evaluate(buildFile, {
        videos: ['a0.mp4', 'a1.mp4', 'b0.mp4', 'b1.mp4', 'other.mp4'], sessions: [['a0.mp4', 'a1.mp4'], ['b0.mp4', 'b1.mp4']],
        events: OWN.concat(OWN_B, OTHER, [NO_VIDEO]),
    });
    const source = await getBytes(page, '__built');
    await page.close();

    const one = await openAndSave(loader, source);
    const firstShows = [['rear', 10, 19, 1], ['rear', 30, 39, 1], ['rear', 500, 501, 1]];
    const firstCam = 'session 0 first camera';
    const noVideoRead = ownRead(firstCam, [[500, 501]]);
    let expected;
    if (opensAll) {
        check(one.shown.length === 2 && eq(one.shown[0].bouts, firstShows), 'the first session: its own and the event with no video', one.shown);
        check(eq(one.shown[1].bouts, [['rear', 110, 119, 1], ['rear', 130, 139, 1]]), 'the second: only its own', one.shown);
        expected = ownRead(firstCam, [[10, 19], [30, 39]])
            .concat(ownRead('session 1 first camera', [[110, 119], [130, 139]], 'animal_B'), OTHER_READ('other.mp4'), noVideoRead);
        check(eq(one.read.outside, ['other.mp4']) && one.read.sessions.length === 2, 'saved: both sessions, and that video in neither', one.read);
    } else {
        check(one.shown.length === 1 && eq(one.shown[0].bouts, firstShows), 'this loader opens the first session: its own and the event with no video', one.shown);
        expected = ownRead(firstCam, [[10, 19], [30, 39]])
            .concat(ownRead('b0.mp4', [[110, 119]], 'animal_B'), ownRead('b1.mp4', [[130, 139]], 'animal_B'), OTHER_READ('other.mp4'), noVideoRead);
    }
    check(eq(one.read.events, expected), 'saved: no event lost or doubled, each where it belongs', one.read.events);
    const two = await openAndSave(loader, one.saved);
    check(eq(two.shown, one.shown), 'reopened, the same', two.shown);
    check(eq(two.read.events, expected), 'saved again: the same', two.read.events);
}

/** D: a video in no session with no event on it. */
async function noEventThere(loader) {
    console.log(`\n-- D. no event on the other video, ${loader} --`);
    const page = await newAppPage();
    await page.evaluate(buildFile, { videos: ['a0.mp4', 'a1.mp4', 'other.mp4'], sessions: [['a0.mp4', 'a1.mp4']], events: OWN });
    const source = await getBytes(page, '__built');
    await page.close();
    const one = await openAndSave(loader, source);
    check(one.read.videos.length === 2 && one.read.outside.length === 0, 'the save writes the session\'s cameras only, as before', one.read.videos);
}

function pythonReads(file) {
    const script = `
import json, sys, sleap_io as sio
labels = sio.load_slp(sys.argv[1], open_videos=False)
def who(p):
    return None if p is None else [type(p).__name__, p.name]
print(json.dumps({
    "videos": [v.filename for v in labels.videos],
    "types": {t.name: t.metadata.get("negates") for t in labels.event_types},
    "events": [[e.video.filename, e.type.name, e.start_frame, e.end_frame, who(e.subject), who(e.target), e.is_predicted] for e in labels.events],
}))
`;
    const r = spawnSync('uv', ['run', '--no-project', '--quiet', '--with', PY_SPEC, 'python', '-I', '-c', script, file], { encoding: 'utf8' });
    if (r.status !== 0) throw new Error('python failed: ' + r.stderr);
    return JSON.parse(r.stdout.trim().split('\n').pop());
}

try {
    await new Promise(r => setTimeout(r, 1200));
    browser = await chromium.launch();
    const finalSaves = {};
    for (const loader of Object.keys(LOADERS)) {
        finalSaves[loader] = await oneSession(loader, 'A. one session, the other video last', ['a0.mp4', 'a1.mp4', 'other.mp4']);
    }
    for (const loader of Object.keys(LOADERS)) {
        await oneSession(loader, 'B. one session, the other video first', ['other.mp4', 'a0.mp4', 'a1.mp4']);
    }
    await twoSessions('File ▸ Open', true);
    await twoSessions('large project (lazy)', false);
    await twoSessions('single .slp folder', false);
    await noEventThere('File ▸ Open');
    await noEventThere('large project (lazy)');

    console.log('\n-- real Python reads the saves --');
    if (spawnSync('uv', ['--version'], { encoding: 'utf8' }).status !== 0) {
        console.log('  - skipped: no uv');
    } else {
        for (const [loader, bytes] of Object.entries(finalSaves)) {
            const file = path.join(tmp, 'for-python.slp');
            fs.writeFileSync(file, bytes);
            const py = pythonReads(file);
            const there = py.events.filter(e => e[0] === 'other.mp4').map(e => e.slice(1));
            check(eq(there, [
                ['attack', 300, 301, ['Identity', 'animal_B'], ['Identity', 'animal_A'], false],
                ['not rear', 310, 311, ['Identity', 'animal_A'], null, false],
                ['rear', 320, 322, ['Identity', 'animal_A'], null, true],
                ['stimulus', 330, 335, null, null, false],
            ]), `${loader}: every event on its video, its animals by kind`, py.events);
            check(py.videos.includes('other.mp4') && py.types['not rear'] === 'rear', `${loader}: the video is listed; "not rear" negates rear`, py);
        }
    }

    check(pageErrors.length === 0, 'no page errors', pageErrors.slice(0, 3));
    console.log(fails === 0 ? '\nPASS' : `\nFAIL — ${fails} check(s)`);
    process.exitCode = fails === 0 ? 0 : 1;
} finally {
    if (browser) await browser.close().catch(() => {});
    server.kill();
    fs.rmSync(tmp, { recursive: true, force: true });
}
