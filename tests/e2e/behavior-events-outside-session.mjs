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
 *  E. Separate videos and no session. A save writes a session of placeholder cameras
 *     naming them all, and that changes nothing: every event stays on its own video through
 *     every later save, and one animal's adjacent bouts on two videos stay two bouts.
 *  F. An uncalibrated session whose cameras are not in the file's order, and a video in no
 *     session: only the first camera's events are labels, each other stays where it is.
 *  G. Two uncalibrated sessions: each event stays on its own session's camera.
 *  H. An uncalibrated session written by Python, through LUC3D, Python and LUC3D again.
 *  I. A video outside the open sessions holding poses AND bouts is ONE entry of the saved
 *     list, with both on it: last or first in the file, among videos with only poses or only
 *     bouts, every video named `0.mp4` (campy's name), another session's videos in a loader
 *     of one session, a camera the folder loader makes by filename, File ▸ Add SLP (a video of
 *     ANOTHER file at the same place in its list stays apart), a project mixing lazy and
 *     eager, a two-pass save, two saves in a row, and files real Python wrote.
 *  J. No session and a calibration file beside the `.slp`: the folder loader says which video
 *     is which camera, so its order decides the first camera, never the file's. Only that
 *     camera's events are labels, each other stays on its camera, a video matching no camera
 *     stays on its own entry. The save gives the file a calibrated session, whose rule (a bout
 *     on any of its cameras is the session's) applies from the next time it is opened.
 *
 * Kinds of event on the other video: a bout, a "not" label (its type must keep saying
 * what it negates), a prediction with framewise scores, an event with no animal.
 * Real Python sleap-io reads the last saves when `uv` is here (skipped without it).
 *
 * Run: node tests/e2e/behavior-events-outside-session.mjs   (spawns its own http.server; PORT= to move it)
 *      SECTIONS=G,I runs only those sections
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
const SECTIONS = process.env.SECTIONS ? process.env.SECTIONS.split(',') : null;
const runs = (section) => !SECTIONS || SECTIONS.includes(section);

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
        const filename = spec.filename || name;
        const v = new SIO.Video({ filename, backendMetadata: { type: 'MediaVideo', shape, filename }, openBackend: false });
        v.shape = shape;
        videoOf[name] = v;
        return v;
    });
    const frames = [];
    const addFrames = (video) => {
        // Each video's x values are its own, so a pose that moved cannot pass for another's.
        const base = 1000 * videos.indexOf(video);
        for (let f = 0; f < 2; f++) {
            const pt = (x, y) => ({ xy: [x, y], visible: true, complete: true, score: 0.9 });
            frames.push(new SIO.LabeledFrame({ video, frameIdx: f, instances: [
                new SIO.PredictedInstance({ points: [pt(base + 10 + f, 10), pt(20, 20)], skeleton, track, score: 0.9 }),
            ] }));
        }
    };
    const sessions = spec.sessions.map((names, si) => {
        const cameras = names.map((n, ci) => new SIO.Camera({
            name: 's' + si + '_cam' + ci, rvec: spec.uncalibrated ? [0, 0, 0] : [0, 0.1 * ci, 0], tvec: spec.uncalibrated ? [0, 0, 0] : [10 * ci, 0, 0],
            matrix: [[500, 0, 32], [0, 500, 32], [0, 0, 1]], distortions: [0, 0, 0, 0, 0], size: [64, 64],
        }));
        const session = new SIO.RecordingSession({ cameraGroup: new SIO.CameraGroup({ cameras }) });
        names.forEach((n, ci) => {
            session.addVideo(videoOf[n], cameras[ci]);
            addFrames(videoOf[n]);
        });
        return session;
    });
    if (spec.sessions.length === 0) videos.forEach(addFrames);
    (spec.posesOn || []).forEach(name => addFrames(videoOf[name]));
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
    const inSession = new Set(sessions.flat());
    const where = new Map();
    sessions.forEach((s, si) => s.forEach((vi, ci) => where.set(vi, 'session ' + si + (ci === 0 ? ' first camera' : ' camera ' + ci))));
    const place = (vi) => where.get(vi) || videos[vi];
    const round = (a) => (a ? Array.from(a).map(x => Math.round(x * 1000) / 1000) : null);
    const rows = slp.behaviorEvents ? slp.behaviorEvents.rows : [];
    const events = rows.map(r => [
        place(r.video), r.type, r.start, r.end, r.subject, r.target, r.predicted, r.source,
        Number.isFinite(r.score) ? r.score : null, round(r.scores), r.metadata, r.name,
    ]).sort((a, b) => a[2] - b[2]);
    const ascending = (a) => a.sort((x, y) => x - y);
    return {
        videos, sessions, outside: videos.filter((n, i) => !inSession.has(i)), events,
        types: slp.behaviorEvents ? slp.behaviorEvents.types.map(t => ({ name: t.name, negates: t.metadata.negates || null })) : [],
        framesOffSession: slp.frames.filter(f => !inSession.has(f.videoIdx)).length,
        framesOn: videos.map((n, i) => slp.frames.filter(f => f.videoIdx === i).length),
        // Each entry of the list: where it sits, its poses (by x) and its events (by first frame).
        entries: videos.map((n, i) => ({
            where: place(i),
            poses: ascending(slp.frames.filter(f => f.videoIdx === i).map(f => Math.round(f.instances[0].points[0][0]))),
            events: ascending(rows.filter(r => r.video === i).map(r => r.start)),
        })),
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
    'single .slp folder': async (page, bytes, calibration) => {
        const dir = fs.mkdtempSync(path.join(tmp, 'folder-'));
        fs.writeFileSync(path.join(dir, 'project.slp'), bytes);
        if (calibration) fs.writeFileSync(path.join(dir, 'calibration.toml'), calibration);
        page.once('filechooser', fc => fc.setFiles(dir));
        return page.evaluate(() => { window.__done = false; (async () => {
            await (await import('/loading/session-loader.js')).handleLoadSessionFolderSingleSlp();
        })().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); }); });
    },
};

/** Dismiss a load's prompts until it finishes. */
async function settleLoad(page, what) {
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
    if (done !== true) throw new Error(what + ': load did not finish: ' + done);
}

/** Open `bytes` with a loader, see what it shows, save, read the save. */
async function openAndSave(loader, bytes, calibration) {
    const page = await newAppPage();
    await putBytes(page, '__file', bytes);
    await LOADERS[loader](page, bytes, calibration);
    await settleLoad(page, loader);
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
    check(one.read.videos.length === 5, 'no copy of a video', one.read.videos);
    const two = await openAndSave(loader, one.saved);
    check(eq(two.shown, one.shown), 'reopened, the same', two.shown);
    check(eq(two.read.events, expected), 'saved again: the same', two.read.events);
    check(eq(two.read.videos, one.read.videos), 'on the same videos', two.read.videos);
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

const rearOn = (where, start, end, who = 'animal_A') => [where, 'rear', start, end, who, '', false, 'manual', null, null, {}, ''];

/** E: separate videos, no session. The first save writes a session of placeholder cameras. */
async function separateVideos(firstLoader, thenLoader) {
    console.log(`\n-- E. separate videos with no session, ${firstLoader} then ${thenLoader} --`);
    const page = await newAppPage();
    await page.evaluate(buildFile, {
        videos: ['one.mp4', 'two.mp4', 'three.mp4'], sessions: [],
        events: [
            { type: 'rear', video: 'one.mp4', start: 10, end: 19, subject: 'a', source: 'manual' },
            { type: 'rear', video: 'two.mp4', start: 20, end: 29, subject: 'a', source: 'manual' },   // the same animal, right after
        ].concat(OTHER.map(e => Object.assign({}, e, { video: 'three.mp4' }))),
    });
    const source = await getBytes(page, '__built');
    await page.close();

    const shows = [{ bouts: [['rear', 10, 19, 1]], predicted: 0 }];
    const expected = [rearOn('session 0 first camera', 10, 19), rearOn('session 0 camera 1', 20, 29)].concat(OTHER_READ('session 0 camera 2'));
    const one = await openAndSave(firstLoader, source);
    check(eq(one.shown, shows), 'it shows the first video\'s bouts', one.shown);
    check(eq(one.read.events, expected), 'saved: every event on its own video', one.read.events);
    check(one.read.videos.length === 3 && one.read.sessions.length === 1 && one.read.outside.length === 0, 'a session of the three videos, and no copy of one', one.read);
    check(eq(one.read.framesOn, [2, 2, 2]), 'each video keeps its poses', one.read.framesOn);

    let last = one;
    for (const step of ['opened again', 'and again']) {
        const next = await openAndSave(thenLoader, last.saved);
        check(eq(next.shown, shows), `${step}: it shows the same`, next.shown);
        check(eq(next.read.events, expected), `saved: nothing moved, nothing merged`, next.read.events);
        check(eq(next.read.videos, one.read.videos) && eq(next.read.sessions, one.read.sessions), 'the same videos, in the same session', next.read);
        check(eq(next.read.framesOn, [2, 2, 2]), 'and each its poses', next.read.framesOn);
        check(negatesOf(next.read, 'not rear') === 'rear', 'a "not" still a "not"', next.read.types);
        last = next;
    }
    return { saved: last.saved, videos: one.read.videos };
}

/** F: an uncalibrated session, its first camera not the file's first video, and a video in no session. */
async function uncalibratedShuffled(loader) {
    console.log(`\n-- F. an uncalibrated session in another order, ${loader} --`);
    const page = await newAppPage();
    await page.evaluate(buildFile, {
        videos: ['other.mp4', 'b.mp4', 'a.mp4'], sessions: [['a.mp4', 'b.mp4']], uncalibrated: true,
        events: [
            { type: 'rear', video: 'a.mp4', start: 10, end: 19, subject: 'a', source: 'manual' },
            { type: 'rear', video: 'b.mp4', start: 20, end: 29, subject: 'a', source: 'manual' },
        ].concat(OTHER),
    });
    const source = await getBytes(page, '__built');
    await page.close();

    const shows = [{ bouts: [['rear', 10, 19, 1]], predicted: 0 }];
    const expected = [rearOn('session 0 first camera', 10, 19), rearOn('session 0 camera 1', 20, 29)].concat(OTHER_READ('other.mp4'));
    const one = await openAndSave(loader, source);
    check(eq(one.shown, shows), 'it shows the first camera\'s bouts', one.shown);
    check(eq(one.read.events, expected), 'saved: the camera\'s on that camera, the other video\'s on that video', one.read.events);
    check(one.read.videos.length === 3 && eq(one.read.outside, ['other.mp4']), 'no copy of a camera\'s video', one.read);
    const two = await openAndSave(loader, one.saved);
    check(eq(two.shown, shows) && eq(two.read.events, expected), 'reopened and saved again: the same', two.read.events);
    check(eq(two.read.videos, one.read.videos), 'on the same videos', two.read.videos);
}

/** G: two uncalibrated sessions: a camera's place is in its session, not in the file. */
async function uncalibratedTwoSessions(loader, opensAll) {
    console.log(`\n-- G. two uncalibrated sessions, ${loader} --`);
    const page = await newAppPage();
    await page.evaluate(buildFile, {
        videos: ['a0.mp4', 'a1.mp4', 'b0.mp4', 'b1.mp4'], sessions: [['a0.mp4', 'a1.mp4'], ['b0.mp4', 'b1.mp4']], uncalibrated: true,
        events: OWN.concat([
            { type: 'rear', video: 'b0.mp4', start: 110, end: 119, subject: 'b', source: 'manual' },
            { type: 'rear', video: 'b1.mp4', start: 130, end: 139, subject: 'b', source: 'manual' },
        ]),
    });
    const source = await getBytes(page, '__built');
    await page.close();

    const one = await openAndSave(loader, source);
    const aRead = [rearOn('session 0 first camera', 10, 19), rearOn('session 0 camera 1', 30, 39)];
    const shows = opensAll
        ? [{ bouts: [['rear', 10, 19, 1]], predicted: 0 }, { bouts: [['rear', 110, 119, 1]], predicted: 0 }]
        : [{ bouts: [['rear', 10, 19, 1]], predicted: 0 }];
    const expected = opensAll
        ? aRead.concat([rearOn('session 1 first camera', 110, 119, 'animal_B'), rearOn('session 1 camera 1', 130, 139, 'animal_B')])
        : aRead.concat([rearOn('b0.mp4', 110, 119, 'animal_B'), rearOn('b1.mp4', 130, 139, 'animal_B')]);
    check(eq(one.shown, shows), 'each session shows its first camera\'s bouts', one.shown);
    check(eq(one.read.events, expected), 'saved: each on its own session\'s camera', one.read.events);
    check(one.read.videos.length === 4, 'no copy of a video', one.read.videos);
    check(eq(one.read.framesOn, [2, 2, 2, 2]), 'and each its poses', one.read.framesOn);
    const two = await openAndSave(loader, one.saved);
    check(eq(two.shown, shows) && eq(two.read.events, expected) && eq(two.read.videos, one.read.videos), 'reopened and saved again: the same', two.read);
}

/** H: a session Python wrote with no calibration: its cameras keep their default rotation and translation. */
function pythonUncalibrated(file) {
    const script = `
import sys, numpy as np, sleap_io as sio
skeleton = sio.Skeleton(nodes=["nose", "tail"], edges=[("nose", "tail")])
videos = [sio.Video(filename=n, open_backend=False) for n in ("one.mp4", "two.mp4")]
for v in videos:
    v.backend_metadata["shape"] = (50, 64, 64, 1)
t1 = sio.Track(name="mouse1")
frames = [sio.LabeledFrame(video=v, frame_idx=f, instances=[
    sio.PredictedInstance.from_numpy(np.array([[10.0, 10.0], [20.0, 20.0]]), skeleton=skeleton, track=t1, score=0.9, point_scores=np.array([0.9, 0.9])),
]) for v in videos for f in range(2)]
cameras = [sio.Camera(name=n) for n in ("camA", "camB")]
session = sio.RecordingSession(camera_group=sio.CameraGroup(cameras=cameras))
for v, c in zip(videos, cameras):
    session.add_video(v, c)
rear = sio.EventType(name="rear")
events = [
    sio.UserEvent(type=rear, video=videos[0], start_frame=10, end_frame=19, subject=t1, source="manual"),
    sio.UserEvent(type=rear, video=videos[1], start_frame=20, end_frame=29, subject=t1, source="manual"),
]
sio.Labels(labeled_frames=frames, videos=videos, skeletons=[skeleton], tracks=[t1], sessions=[session], event_types=[rear], events=events).save(sys.argv[1], embed=False)
`;
    const r = spawnSync('uv', ['run', '--no-project', '--quiet', '--with', PY_SPEC, 'python', '-I', '-c', script, file], { encoding: 'utf8' });
    if (r.status !== 0) throw new Error('python failed: ' + r.stderr);
}

/** Load a file in Python and save it again, as a Python script would. */
function pythonResaves(from, to) {
    const script = `
import sys, sleap_io as sio
sio.load_slp(sys.argv[1], open_videos=False).save(sys.argv[2], embed=False)
`;
    const r = spawnSync('uv', ['run', '--no-project', '--quiet', '--with', PY_SPEC, 'python', '-I', '-c', script, from, to], { encoding: 'utf8' });
    if (r.status !== 0) throw new Error('python failed: ' + r.stderr);
}

/** A file real Python writes: poses on every video, bouts as `spec.events` says, by the same x values as `buildFile`. */
function pythonFile(file, spec) {
    const script = `
import json, sys, numpy as np, sleap_io as sio
spec = json.loads(sys.argv[2])
skeleton = sio.Skeleton(nodes=["nose", "tail"], edges=[("nose", "tail")])
videos = [sio.Video(filename=n, open_backend=False) for n in spec["videos"]]
video_of = dict(zip(spec["videos"], videos))
for v in videos:
    v.backend_metadata["shape"] = (400, 64, 64, 1)
t1 = sio.Track(name="mouse1")
frames = [sio.LabeledFrame(video=v, frame_idx=f, instances=[
    sio.PredictedInstance.from_numpy(np.array([[1000.0 * i + 10 + f, 10.0], [20.0, 20.0]]), skeleton=skeleton, track=t1, score=0.9, point_scores=np.array([0.9, 0.9])),
]) for i, v in enumerate(videos) for f in range(2)]
sessions = []
for si, names in enumerate(spec["sessions"]):
    cameras = [sio.Camera(name=f"s{si}_cam{ci}", rvec=[0.0, 0.1 * ci, 0.0], tvec=[10.0 * ci, 0.0, 0.0], matrix=[[500.0, 0.0, 32.0], [0.0, 500.0, 32.0], [0.0, 0.0, 1.0]]) for ci in range(len(names))]
    session = sio.RecordingSession(camera_group=sio.CameraGroup(cameras=cameras))
    for n, c in zip(names, cameras):
        session.add_video(video_of[n], c)
    sessions.append(session)
rear = sio.EventType(name="rear")
events = [sio.UserEvent(type=rear, video=video_of[e["video"]], start_frame=e["start"], end_frame=e["end"], subject=t1, source=e["source"]) for e in spec["events"]]
sio.Labels(labeled_frames=frames, videos=videos, skeletons=[skeleton], tracks=[t1], sessions=sessions, event_types=[rear], events=events).save(sys.argv[1], embed=False)
`;
    const r = spawnSync('uv', ['run', '--no-project', '--quiet', '--with', PY_SPEC, 'python', '-I', '-c', script, file, JSON.stringify(spec)], { encoding: 'utf8' });
    if (r.status !== 0) throw new Error('python failed: ' + r.stderr);
    return fs.readFileSync(file);
}

/** Each entry of a file's video list as Python holds it: frames and events on THAT entry. */
function pythonEntries(file) {
    const script = `
import json, sys, sleap_io as sio
labels = sio.load_slp(sys.argv[1], open_videos=False)
print(json.dumps([{
    "filename": v.filename,
    "poses": sorted(round(float(lf.instances[0].points["xy"][0][0])) for lf in labels.labeled_frames if lf.video is v),
    "events": sorted(e.start_frame for e in labels.events if e.video is v),
} for v in labels.videos]))
`;
    const r = spawnSync('uv', ['run', '--no-project', '--quiet', '--with', PY_SPEC, 'python', '-I', '-c', script, file], { encoding: 'utf8' });
    if (r.status !== 0) throw new Error('python failed: ' + r.stderr);
    return JSON.parse(r.stdout.trim().split('\n').pop());
}

/** The poses `buildFile` gives the video at this place in the source's list. */
const posesOf = (i) => [1000 * i + 10, 1000 * i + 11];
const entry = (where, poses, events) => ({ where, poses, events });
const rearAt = (video, start) => ({ type: 'rear', video, start, end: start + 1, subject: 'a', source: 'manual' });

/** The saves section I hands to Python: [what, bytes, the app's read of them]. */
const forPython = [];

/** I: open `spec` with a loader, save, and do it again: `want` is the saved list, entry by entry. */
async function posesAndBouts(loader, title, spec, want) {
    const page = await newAppPage();
    await page.evaluate(buildFile, spec);
    const source = await getBytes(page, '__built');
    await page.close();
    await posesAndBoutsOf(loader, title, source, want);
}

/** I: the same for a file already made. */
async function posesAndBoutsOf(loader, title, source, want) {
    console.log(`\n-- I. ${title}, ${loader} --`);
    const one = await openAndSave(loader, source);
    check(eq(one.read.entries, want), 'one entry per video, its poses and bouts both on it', one.read.entries);
    const two = await openAndSave(loader, one.saved);
    check(eq(two.read.entries, want), 'saved again: the same', two.read.entries);
    check(eq(two.read.videos, one.read.videos), 'in the same order', two.read.videos);
    check(eq(two.read.events, one.read.events), 'and every event unchanged', two.read.events);
    forPython.push([`${title}, ${loader}`, two.saved, two.read]);
}

/** The source the rest of section I edits: one session, and a video outside it with both. */
const ONE_VIDEO_BOTH = { videos: ['a0.mp4', 'a1.mp4', 'other.mp4'], sessions: [['a0.mp4', 'a1.mp4']], posesOn: ['other.mp4'], events: OWN.concat(OTHER) };
const ONE_VIDEO_BOTH_WANT = [
    entry('session 0 first camera', posesOf(0), [10, 30]),
    entry('session 0 camera 1', posesOf(1), []),
    entry('other.mp4', posesOf(2), [300, 310, 320, 330]),
];

/** A calibration file beside a `.slp`, listing these cameras in this order. */
const calibrationFor = (names) => names.map((name, i) => `[cam_${i}]\nname = "${name}"\nsize = [64, 64]\n` +
    'matrix = [[500.0, 0.0, 32.0], [0.0, 500.0, 32.0], [0.0, 0.0, 1.0]]\ndistortions = [0.0, 0.0, 0.0, 0.0, 0.0]\n' +
    `rotation = [0.0, ${0.1 * i}, 0.0]\ntranslation = [${10 * i}.0, 0.0, 0.0]\n`).join('\n');

/** I: a calibration file beside the `.slp`: the folder loader makes a camera of a video no session names, by its filename. */
async function cameraByFilename() {
    console.log('\n-- I. the single .slp folder makes a camera of a video no session names --');
    const page = await newAppPage();
    await page.evaluate(buildFile, {
        videos: ['s0_cam0.mp4', 's0_cam1.mp4'], sessions: [['s0_cam0.mp4']], posesOn: ['s0_cam1.mp4'],
        events: [rearAt('s0_cam0.mp4', 10), rearAt('s0_cam1.mp4', 300)],
    });
    const source = await getBytes(page, '__built');
    await page.close();
    const calibration = calibrationFor(['s0_cam0', 's0_cam1']);

    const one = await openAndSave('single .slp folder', source, calibration);
    check(eq(one.read.entries, [entry('session 0 first camera', posesOf(0), [10]), entry('session 0 camera 1', posesOf(1), [300])]),
        'one entry per video, that camera\'s poses and bout both on it', one.read.entries);
    forPython.push(['a camera by its filename, single .slp folder', one.saved, one.read]);
    // The file's session names that camera now, and a bout on a camera of a calibrated session is the session's.
    const two = await openAndSave('single .slp folder', one.saved, calibration);
    check(eq(two.read.entries, [entry('session 0 first camera', posesOf(0), [10, 300]), entry('session 0 camera 1', posesOf(1), [])]),
        'saved again: still one entry per video, the bout now the session\'s', two.read.entries);
}

/** I: File ▸ Add SLP. The event's video and the added file's video are both at place 2 of their lists. */
async function addSlpSamePlace() {
    console.log('\n-- I. File ▸ Add SLP: a video of another file at the same place stays apart --');
    const page = await newAppPage();
    await page.evaluate(buildFile, { videos: ['c0.mp4', 'c1.mp4', 'added.mp4'], sessions: [['c0.mp4', 'c1.mp4']], posesOn: ['added.mp4'], events: [] });
    const added = path.join(tmp, 'added.slp');
    fs.writeFileSync(added, await getBytes(page, '__built'));
    await page.evaluate(buildFile, { videos: ['a0.mp4', 'a1.mp4', 'other.mp4'], sessions: [['a0.mp4', 'a1.mp4']], events: OWN.concat(OTHER) });
    await page.evaluate(() => { window.__file = window.__built; });
    await LOADERS['File ▸ Open'](page);
    await settleLoad(page, 'open');
    page.once('filechooser', fc => fc.setFiles(added));
    await page.evaluate(() => { window.__done = false; (async () => {
        await (await import('/import-export/slp-import.js')).handleAddSlp();
    })().then(() => { window.__done = true; }, e => { window.__done = 'ERR ' + (e && e.stack || e); }); });
    await settleLoad(page, 'Add SLP');
    await page.evaluate(saveInPage);
    const read = await page.evaluate(readFile);
    await page.close();
    const on = (name) => read.entries.filter(e => e.where === name);
    check(eq(on('other.mp4'), [entry('other.mp4', [], [300, 310, 320, 330])]), 'the bouts\' video: once, its bouts and no pose', read.entries);
    check(eq(on('added.mp4'), [entry('added.mp4', posesOf(2), [])]), 'the added file\'s video: once, its poses and no bout', read.entries);
}

/** I: a lazy session on the eager writer's path. */
async function mixedProject() {
    console.log('\n-- I. a project mixing lazy and eager sessions --');
    const page = await newAppPage();
    await page.evaluate(buildFile, ONE_VIDEO_BOTH);
    await page.evaluate(() => { window.__file = window.__built; });
    await LOADERS['large project (lazy)'](page);
    await settleLoad(page, 'open');
    await page.evaluate(async () => {
        const pd = await import('/pose/pose-data.js');
        const tri = await import('/pose/triangulation.js');
        const { state } = await import('/ui/app-state.js');
        for (const f of [0, 1]) await tri.ensureLazyFrameData(f);
        const cam = new pd.Camera('camE', [[500, 0, 32], [0, 500, 32], [0, 0, 1]], [0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0], [64, 64]);
        const eager = new pd.Session([cam], state.session.skeleton, ['mouse1'], 'eager');
        const fg = new pd.FrameGroup(0);
        eager.addFrameGroup(fg);
        fg.addInstance('camE', new pd.Instance([[900, 10], [901, 20]], 0, 'user', 0));
        state.sessions.push(eager);
    });
    await page.evaluate(saveInPage);
    const read = await page.evaluate(readFile);
    await page.close();
    const here = read.entries.filter(e => e.where === 'other.mp4');
    check(eq(here, [entry('other.mp4', posesOf(2), [300, 310, 320, 330])]), 'that video: once, its poses and bouts both on it', read.entries);
}

/** I: two lazy sessions go through the two-pass save. */
async function twoPassSave() {
    console.log('\n-- I. a two-pass save --');
    const page = await newAppPage();
    await page.evaluate(buildFile, { videos: ['camC.mp4'], sessions: [], events: [] });
    const camC = await getBytes(page, '__built');
    await page.evaluate(buildFile, ONE_VIDEO_BOTH);
    await page.evaluate(() => { window.__file = window.__built; });
    await LOADERS['large project (lazy)'](page);
    await settleLoad(page, 'open');
    await putBytes(page, '__camC', camC);
    const err = await page.evaluate(async () => {
        const pd = await import('/pose/pose-data.js');
        const { SioLazyLoader } = await import('/loading/sio-lazy-loader.js');
        const AS = await import('/ui/app-state.js');
        const sl = await import('/import-export/save-load.js');
        const cam = new pd.Camera('camC', [[500, 0, 32], [0, 500, 32], [0, 0, 1]], [0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0], [64, 64]);
        const second = new pd.Session([cam], AS.state.session.skeleton, ['mouse1'], 'second');
        second.lazyLoader = new SioLazyLoader();
        await second.lazyLoader.open('camC', new File([window.__camC], 'camC.slp'));
        AS.state.sessions.push(second);
        try {
            window.__saved = await sl.saveAllSessionsStreaming(AS.state.sessions.slice());
            return null;
        } catch (e) { return String(e && e.stack || e); }
    });
    check(!err, 'the two-pass save runs', err);
    const read = await page.evaluate(readFile);
    await page.close();
    const here = read.entries.filter(e => e.where === 'other.mp4');
    check(eq(here, [entry('other.mp4', posesOf(2), [300, 310, 320, 330])]), 'that video: once, its poses and bouts both on it', read.entries);
}

/** File ▸ Save twice with the project open: nothing a save leaves behind adds an entry or moves a bout. */
async function savedTwiceInARow(loader, spec = ONE_VIDEO_BOTH, want = ONE_VIDEO_BOTH_WANT, calibration = undefined) {
    console.log(`\n-- two saves in a row, ${loader} --`);
    const page = await newAppPage();
    await page.evaluate(buildFile, spec);
    await page.evaluate(() => { window.__file = window.__built; });
    await LOADERS[loader](page, await getBytes(page, '__built'), calibration);
    await settleLoad(page, loader);
    await page.evaluate(saveInPage);
    await page.evaluate(saveInPage);
    const read = await page.evaluate(readFile);
    await page.close();
    check(eq(read.entries, want), 'one entry per video, its poses and bouts both on it', read.entries);
}

const FIRST_CAMERA = 'session 0 first camera', SECOND_CAMERA = 'session 0 camera 1';

/**
 * J: a file with no session, opened by the folder loader with a calibration file beside it.
 * @param {string[]} cameraNames the calibration's cameras, in its order
 * @param {{shows: number[], videos: string[], saved: Array, reopenedShows: number[], reopenedSaved: Array}} want
 *   what it shows and what a save holds, entry by entry; then the same once that save is opened again
 */
async function calibrationBeside(title, spec, cameraNames, want) {
    console.log(`\n-- J. ${title} --`);
    const page = await newAppPage();
    await page.evaluate(buildFile, spec);
    const source = await getBytes(page, '__built');
    await page.close();
    const calibration = calibrationFor(cameraNames);
    const firstFrames = (shown) => shown[0].bouts.map(b => b[1]);

    const one = await openAndSave('single .slp folder', source, calibration);
    check(eq(firstFrames(one.shown), want.shows), 'it shows only the first camera\'s bouts', one.shown);
    check(eq(one.read.videos, want.videos), 'saved: the cameras in the calibration\'s order, then the other videos', one.read.videos);
    check(eq(one.read.entries, want.saved), 'each bout on its own video, with that video\'s poses', one.read.entries);
    forPython.push([`${title}, first save`, one.saved, one.read]);
    await savedTwiceInARow('single .slp folder', spec, want.saved, calibration);

    const two = await openAndSave('single .slp folder', one.saved, calibration);
    check(eq(firstFrames(two.shown), want.reopenedShows), 'opened again: a calibrated session\'s, whichever camera a bout is on', two.shown);
    check(eq(two.read.videos, want.videos), 'saved: the same list', two.read.videos);
    check(eq(two.read.entries, want.reopenedSaved), 'each video still one entry, with its poses', two.read.entries);
    forPython.push([`${title}, saved after opening it again`, two.saved, two.read]);
}

/** J: every kind of event, on the camera that is not the first and on no video. */
async function calibrationBesideEveryKind() {
    console.log('\n-- J. every kind of event, the calibration in another order than the videos --');
    const page = await newAppPage();
    await page.evaluate(buildFile, {
        videos: ['camA.mp4', 'camB.mp4'], sessions: [],
        events: [
            { type: 'rear', video: 'camB.mp4', start: 10, end: 19, subject: 'a', source: 'manual' },
            { type: 'rear', video: null, start: 500, end: 501, subject: 'a', source: 'manual' },
        ].concat(OTHER.map(e => Object.assign({}, e, { video: 'camA.mp4' }))),
    });
    const source = await getBytes(page, '__built');
    await page.close();
    const one = await openAndSave('single .slp folder', source, calibrationFor(['camB', 'camA']));
    check(eq(one.shown, [{ bouts: [['rear', 10, 19, 1], ['rear', 500, 501, 1]], predicted: 0 }]), 'it shows the first camera\'s bout and the event with no video', one.shown);
    check(eq(one.read.events, [rearOn(FIRST_CAMERA, 10, 19)].concat(OTHER_READ(SECOND_CAMERA), [rearOn(FIRST_CAMERA, 500, 501)])),
        'saved: the other camera\'s events unchanged and on that camera', one.read.events);
    check(negatesOf(one.read, 'not rear') === 'rear', 'a "not" still says what it negates', one.read.types);
}

/** Each event as [video, start], in frame order. */
const placed = (py) => py.events.map(e => [e[0], e[2]]).sort((a, b) => a[1] - b[1]);

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
    if (runs('A')) {
        for (const loader of Object.keys(LOADERS)) {
            finalSaves[loader] = await oneSession(loader, 'A. one session, the other video last', ['a0.mp4', 'a1.mp4', 'other.mp4']);
        }
    }
    if (runs('B')) {
        for (const loader of Object.keys(LOADERS)) {
            await oneSession(loader, 'B. one session, the other video first', ['other.mp4', 'a0.mp4', 'a1.mp4']);
        }
    }
    if (runs('C')) {
        await twoSessions('File ▸ Open', true);
        await twoSessions('large project (lazy)', false);
        await twoSessions('single .slp folder', false);
    }
    if (runs('D')) {
        await noEventThere('File ▸ Open');
        await noEventThere('large project (lazy)');
    }
    const separate = {};
    if (runs('E')) {
        separate['File ▸ Open'] = await separateVideos('File ▸ Open', 'File ▸ Open');
        separate['large project (lazy)'] = await separateVideos('File ▸ Open', 'large project (lazy)');
        separate['single .slp folder'] = await separateVideos('single .slp folder', 'single .slp folder');
    }
    if (runs('F')) {
        for (const loader of Object.keys(LOADERS)) await uncalibratedShuffled(loader);
    }
    if (runs('G')) {
        await uncalibratedTwoSessions('File ▸ Open', true);
        await uncalibratedTwoSessions('large project (lazy)', false);
        await uncalibratedTwoSessions('single .slp folder', false);
    }

    if (runs('I')) {
        for (const loader of Object.keys(LOADERS)) {
            await posesAndBouts(loader, 'the video last', ONE_VIDEO_BOTH, ONE_VIDEO_BOTH_WANT);
            await posesAndBouts(loader, 'the video first', Object.assign({}, ONE_VIDEO_BOTH, { videos: ['other.mp4', 'a0.mp4', 'a1.mp4'] }), [
                entry('session 0 first camera', posesOf(1), [10, 30]),
                entry('session 0 camera 1', posesOf(2), []),
                entry('other.mp4', posesOf(0), [300, 310, 320, 330]),
            ]);
            // The writers list the cameras, then the videos with poses, then those with only bouts.
            await posesAndBouts(loader, 'among videos with only poses or only bouts', {
                videos: ['a0.mp4', 'a1.mp4', 'both1.mp4', 'poses.mp4', 'bouts.mp4', 'both2.mp4'], sessions: [['a0.mp4', 'a1.mp4']],
                posesOn: ['both1.mp4', 'poses.mp4', 'both2.mp4'],
                events: OWN.concat([rearAt('both2.mp4', 500), rearAt('bouts.mp4', 400), rearAt('both1.mp4', 300)]),
            }, [
                entry('session 0 first camera', posesOf(0), [10, 30]),
                entry('session 0 camera 1', posesOf(1), []),
                entry('both1.mp4', posesOf(2), [300]),
                entry('poses.mp4', posesOf(3), []),
                entry('both2.mp4', posesOf(5), [500]),
                entry('bouts.mp4', [], [400]),
            ]);
            // campy names every video `0.mp4`: only the file's own record tells them apart.
            await posesAndBouts(loader, 'every video named 0.mp4', {
                videos: ['a0', 'a1', 'x', 'y'], filename: '0.mp4', sessions: [['a0', 'a1']], posesOn: ['x', 'y'],
                events: [rearAt('a0', 10), rearAt('a1', 30), rearAt('x', 300), rearAt('y', 400)],
            }, [
                entry('session 0 first camera', posesOf(0), [10, 30]),
                entry('session 0 camera 1', posesOf(1), []),
                entry('0.mp4', posesOf(2), [300]),
                entry('0.mp4', posesOf(3), [400]),
            ]);
        }
        // A multi-session file: a loader of one session holds the others' videos like any other.
        const twoSessionsBoth = {
            videos: ['a0.mp4', 'a1.mp4', 'b0.mp4', 'b1.mp4', 'other.mp4'], sessions: [['a0.mp4', 'a1.mp4'], ['b0.mp4', 'b1.mp4']], posesOn: ['other.mp4'],
            events: OWN.concat([rearAt('b0.mp4', 110), rearAt('b1.mp4', 130)], OTHER),
        };
        for (const loader of ['large project (lazy)', 'single .slp folder']) {
            await posesAndBouts(loader, 'two sessions, this loader opens the first', twoSessionsBoth, [
                entry('session 0 first camera', posesOf(0), [10, 30]),
                entry('session 0 camera 1', posesOf(1), []),
                entry('b0.mp4', posesOf(2), [110]),
                entry('b1.mp4', posesOf(3), [130]),
                entry('other.mp4', posesOf(4), [300, 310, 320, 330]),
            ]);
        }
        await posesAndBouts('File ▸ Open', 'two sessions, both open', twoSessionsBoth, [
            entry('session 0 first camera', posesOf(0), [10, 30]),
            entry('session 0 camera 1', posesOf(1), []),
            entry('other.mp4', posesOf(4), [300, 310, 320, 330]),
            entry('session 1 first camera', posesOf(2), [110, 130]),
            entry('session 1 camera 1', posesOf(3), []),
        ]);
        await cameraByFilename();
        await addSlpSamePlace();
        await mixedProject();
        await twoPassSave();
        for (const loader of Object.keys(LOADERS)) await savedTwiceInARow(loader);
    }
    if (runs('J')) {
        const twoCameras = { videos: ['camA.mp4', 'camB.mp4'], sessions: [], events: [rearAt('camA.mp4', 10), rearAt('camB.mp4', 300)] };
        await calibrationBeside('the calibration in another order than the videos', twoCameras, ['camB', 'camA'], {
            shows: [300], videos: ['camB.mp4', 'camA.mp4'],
            saved: [entry(FIRST_CAMERA, posesOf(1), [300]), entry(SECOND_CAMERA, posesOf(0), [10])],
            reopenedShows: [10, 300],
            reopenedSaved: [entry(FIRST_CAMERA, posesOf(1), [10, 300]), entry(SECOND_CAMERA, posesOf(0), [])],
        });
        await calibrationBeside('the calibration in the videos\' order', twoCameras, ['camA', 'camB'], {
            shows: [10], videos: ['camA.mp4', 'camB.mp4'],
            saved: [entry(FIRST_CAMERA, posesOf(0), [10]), entry(SECOND_CAMERA, posesOf(1), [300])],
            reopenedShows: [10, 300],
            reopenedSaved: [entry(FIRST_CAMERA, posesOf(0), [10, 300]), entry(SECOND_CAMERA, posesOf(1), [])],
        });
        // Its first video matches no camera, so the file's order is off even with the cameras in the videos' order.
        await calibrationBeside('a video matching no camera, first in the file', {
            videos: ['other.mp4', 'camA.mp4', 'camB.mp4'], sessions: [],
            events: [rearAt('other.mp4', 200), rearAt('camA.mp4', 10), rearAt('camB.mp4', 300)],
        }, ['camA', 'camB'], {
            shows: [10], videos: ['camA.mp4', 'camB.mp4', 'other.mp4'],
            saved: [entry(FIRST_CAMERA, posesOf(1), [10]), entry(SECOND_CAMERA, posesOf(2), [300]), entry('other.mp4', posesOf(0), [200])],
            reopenedShows: [10, 300],
            reopenedSaved: [entry(FIRST_CAMERA, posesOf(1), [10, 300]), entry(SECOND_CAMERA, posesOf(2), []), entry('other.mp4', posesOf(0), [200])],
        });
        await calibrationBeside('no video for the first camera', { videos: ['camB.mp4'], sessions: [], events: [rearAt('camB.mp4', 300)] }, ['camA', 'camB'], {
            shows: [], videos: ['camA.mp4', 'camB.mp4'],
            saved: [entry(FIRST_CAMERA, [], []), entry(SECOND_CAMERA, posesOf(0), [300])],
            reopenedShows: [300],
            reopenedSaved: [entry(FIRST_CAMERA, [], [300]), entry(SECOND_CAMERA, posesOf(0), [])],
        });
        // campy names every video `0.mp4`, so no video matches a camera. They stay outside the session, however often it is saved.
        const zeroes = [entry(FIRST_CAMERA, [], []), entry(SECOND_CAMERA, [], []), entry('0.mp4', posesOf(0), [10]), entry('0.mp4', posesOf(1), [300])];
        await calibrationBeside('every video named 0.mp4, so none matches a camera', {
            videos: ['x', 'y'], filename: '0.mp4', sessions: [], events: [rearAt('x', 10), rearAt('y', 300)],
        }, ['camA', 'camB'], {
            shows: [], videos: ['camA.mp4', 'camB.mp4', '0.mp4', '0.mp4'],
            saved: zeroes, reopenedShows: [], reopenedSaved: zeroes,
        });
        await calibrationBesideEveryKind();
    }

    console.log('\n-- real Python reads the saves --');
    if (spawnSync('uv', ['--version'], { encoding: 'utf8' }).status !== 0) {
        console.log('  - skipped: no uv');
    } else {
        if (runs('I')) {
            const pyFile = path.join(tmp, 'python-source.slp');
            const otherFirst = pythonFile(pyFile, {
                videos: ['other.mp4', 'a0.mp4', 'a1.mp4'], sessions: [['a0.mp4', 'a1.mp4']],
                events: [rearAt('a0.mp4', 10), rearAt('other.mp4', 300), rearAt('other.mp4', 310)],
            });
            for (const loader of ['File ▸ Open', 'large project (lazy)']) {
                await posesAndBoutsOf(loader, 'a file Python wrote', otherFirst, [
                    entry('session 0 first camera', posesOf(1), [10]),
                    entry('session 0 camera 1', posesOf(2), []),
                    entry('other.mp4', posesOf(0), [300, 310]),
                ]);
            }
            const twoSessionsFile = pythonFile(pyFile, {
                videos: ['a0.mp4', 'a1.mp4', 'b0.mp4', 'b1.mp4', 'other.mp4'], sessions: [['a0.mp4', 'a1.mp4'], ['b0.mp4', 'b1.mp4']],
                events: [rearAt('a0.mp4', 10), rearAt('b0.mp4', 110), rearAt('b1.mp4', 130), rearAt('other.mp4', 300)],
            });
            for (const loader of ['large project (lazy)', 'single .slp folder']) {
                await posesAndBoutsOf(loader, 'two sessions Python wrote, this loader opens the first', twoSessionsFile, [
                    entry('session 0 first camera', posesOf(0), [10]),
                    entry('session 0 camera 1', posesOf(1), []),
                    entry('b0.mp4', posesOf(2), [110]),
                    entry('b1.mp4', posesOf(3), [130]),
                    entry('other.mp4', posesOf(4), [300]),
                ]);
            }
        }
        for (const [what, saved, read] of forPython) {
            const file = path.join(tmp, 'both-for-python.slp');
            fs.writeFileSync(file, saved);
            const want = read.videos.map((filename, i) => ({ filename, poses: read.entries[i].poses, events: read.entries[i].events }));
            const py = pythonEntries(file);
            check(eq(py, want), `${what}: each video once, its poses and bouts on that entry`, py);
        }
        for (const [loader, { saved, videos }] of Object.entries(separate)) {
            const file = path.join(tmp, 'separate-for-python.slp');
            fs.writeFileSync(file, saved);
            const py = pythonReads(file);
            check(eq(placed(py), [[videos[0], 10], [videos[1], 20], [videos[2], 300], [videos[2], 310], [videos[2], 320], [videos[2], 330]]),
                `separate videos, ${loader}: every event on its video`, py.events);
        }

        if (runs('H')) console.log('\n-- H. an uncalibrated session from Python, through LUC3D, Python and LUC3D --');
        for (const loader of runs('H') ? Object.keys(LOADERS) : []) {
            const file = path.join(tmp, 'python-uncalibrated.slp');
            pythonUncalibrated(file);
            const shows = [{ bouts: [['rear', 10, 19, 1]], predicted: 0 }];
            const one = await openAndSave(loader, fs.readFileSync(file));
            check(eq(one.shown, shows), `${loader}: it shows the first camera's bout`, one.shown);
            const saved1 = path.join(tmp, 'python-uncalibrated-1.slp'), saved2 = path.join(tmp, 'python-uncalibrated-2.slp'), saved3 = path.join(tmp, 'python-uncalibrated-3.slp');
            fs.writeFileSync(saved1, one.saved);
            const py1 = pythonReads(saved1);
            const [first, second] = one.read.sessions[0].map(vi => one.read.videos[vi]);
            check(eq(placed(py1), [[first, 10], [second, 20]]), 'Python reads each bout on its camera\'s video', py1.events);
            pythonResaves(saved1, saved2);
            const two = await openAndSave(loader, fs.readFileSync(saved2));
            check(eq(two.shown, shows), 'after Python\'s save, the same', two.shown);
            fs.writeFileSync(saved3, two.saved);
            check(eq(placed(pythonReads(saved3)), [[first, 10], [second, 20]]), 'and after LUC3D\'s next one, each bout is still on its video', pythonReads(saved3).events);
        }

        console.log('\n-- real Python reads the other saves --');
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
