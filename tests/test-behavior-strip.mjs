/**
 * test-behavior-strip.mjs — the pure half of the bouts strip above the seekbar
 * (ui/behavior-strip-model.js): when it shows, its zoom, its lanes (from real
 * labels: a mutual pair stored once, shown for both animals), its window, the
 * film it moves instead of repainting, its ticks and its hover text. The DOM half
 * is covered by tests/e2e/behavior-strip.mjs.
 *
 * Run:  node tests/test-behavior-strip.mjs
 */
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const load = (p) => import(pathToFileURL(path.join(HERE, '..', p)).href);
const M = await load('ui/behavior-strip-model.js');
const L = await load('pose/behavior-labels.js');
const E = await load('pose/behavior-ethogram.js');

let passed = 0, failed = 0;
const failures = [];
function ok(c, m) {
    if (c) passed++;
    else { failed++; failures.push(m); console.error('  ✗ ' + m); }
}
function eq(a, e, m) {
    const sa = JSON.stringify(a), se = JSON.stringify(e);
    ok(sa === se, `${m} (expected ${se}, got ${sa})`);
}
function near(a, e, m, tol = 1e-9) { ok(Math.abs(a - e) <= tol, `${m} (expected ${e}, got ${a})`); }

const glyph = (b) => (b && b.mutual ? ' ↔ ' : ' → ');

// ---- when it shows ----------------------------------------------------------
{
    const base = { on: true, hasEthogram: true, totalFrames: 100, mode: true, hasLabels: false };
    ok(M.stripWanted(base), 'in the mode, with an ethogram, even before any label');
    ok(!M.stripWanted({ ...base, mode: false }), 'outside the mode with nothing to read: hidden');
    ok(M.stripWanted({ ...base, mode: false, hasLabels: true }), 'outside the mode once the session has labels');
    ok(!M.stripWanted({ ...base, hasEthogram: false, hasLabels: true }), 'never without an ethogram');
    ok(!M.stripWanted({ ...base, on: false, hasLabels: true }), 'the off choice wins, in the mode too');
    ok(!M.stripWanted({ ...base, totalFrames: 0 }), 'nothing to seek through: hidden');
}

// ---- zoom ---------------------------------------------------------------------
{
    eq(M.DEFAULT_HALF_SECONDS, 5, 'about ±5 s by default');
    ok(M.STRIP_ZOOM_STEPS.includes(M.DEFAULT_HALF_SECONDS), 'the default is a step');
    eq(M.normalizeHalfSeconds('10'), 10, 'a stored string reads as its step');
    eq(M.normalizeHalfSeconds(6), 5, 'something between steps snaps to the nearest');
    eq(M.normalizeHalfSeconds(100000), 120, 'too large: the largest');
    for (const bad of [null, undefined, '', 'abc', NaN, -3, 0, Infinity]) {
        eq(M.normalizeHalfSeconds(bad), 5, 'no usable number gives the default: ' + bad);
    }
    eq(M.zoomedHalfSeconds(5, 1), 3, 'zooming in shows less time');
    eq(M.zoomedHalfSeconds(5, -1), 10, 'zooming out shows more');
    eq(M.zoomedHalfSeconds(M.STRIP_ZOOM_STEPS[0], 1), M.STRIP_ZOOM_STEPS[0], 'stops at the closest');
    const last = M.STRIP_ZOOM_STEPS[M.STRIP_ZOOM_STEPS.length - 1];
    eq(M.zoomedHalfSeconds(last, -1), last, 'stops at the widest');
    eq(M.formatStripWindow(0.5), '±0.5 s', 'the readout');
}

// ---- lanes, from real labels ----------------------------------------------------
{
    E.resetEthogram();
    const rear = E.addBehavior({ name: 'rear', key: 'r' }).behavior;
    const attack = E.addBehavior({ name: 'attack', key: 'a', social: true }).behavior;
    const sniff = E.addBehavior({ name: 'sniff', key: 's', social: true, mutual: true }).behavior;
    const jump = E.addBehavior({ name: 'jump', key: 'j', kind: 'point' }).behavior;
    const walk = E.addBehavior({ name: 'walk', key: 'w' }).behavior;
    const session = {};
    L.labelRange(session, attack.id, 'id_1', 'id_2', 50, 60, true);
    L.labelRange(session, sniff.id, 'id_2', 'id_1', 70, 80, true);    // stored as (id_1, id_2)
    L.labelRange(session, rear.id, 'id_2', '', 10, 20, true);
    L.labelRange(session, rear.id, 'id_2', '', 21, 30, false);
    L.labelRange(session, jump.id, 'id_1', '', 5, 5, true);
    L.toggleState(session, walk.id, 'id_1', '', 90);
    const lanes = M.stripLanes(['id_1', 'id_2'], E.behaviors(), L.listBouts(session), L.runningStates(session), glyph);

    eq(lanes.map(l => l.animal + ':' + l.name),
        ['id_1:attack', 'id_1:sniff', 'id_1:jump', 'id_1:walk', 'id_2:rear', 'id_2:sniff'],
        'one lane per animal and behavior that has bouts; animals in order, then the ethogram order');
    eq(L.listBouts(session, { behaviorId: sniff.id }).length, 1, 'a mutual pair is stored once');
    const of = (animal, name) => lanes.find(l => l.animal === animal && l.name === name);
    eq(of('id_1', 'sniff').items.map(i => [i.start, i.end, i.partner]), [[70, 80, 'id_2']], 'and shown for one animal …');
    eq(of('id_2', 'sniff').items.map(i => [i.start, i.end, i.partner]), [[70, 80, 'id_1']], '… and for the other, each naming its partner');
    ok(!lanes.some(l => l.animal === 'id_2' && l.name === 'attack'), 'a directed bout is the actor\'s only');
    eq(of('id_1', 'attack').items[0].partner, 'id_2', 'and names its target');
    eq(of('id_1', 'attack').pair, ' → ', 'how a directed pair reads');
    eq(of('id_1', 'sniff').pair, ' ↔ ', 'how a mutual pair reads');
    eq(of('id_2', 'rear').items.map(i => i.present), [false, true], '"not" labels come first, so present bouts paint over them');
    ok(of('id_1', 'jump').point, 'a point behavior is known as one');
    ok(of('id_1', 'walk').items[0].running, 'a running state has a lane from its first frame');
    eq(of('id_1', 'rear'), undefined, 'no bouts, no lane');
    eq(of('id_1', 'attack').color, attack.color, 'in the ethogram color');

    const orphaned = M.stripLanes(['id_2'], E.behaviors(), L.listBouts(session), [], glyph);
    eq(orphaned[0].animal, 'id_2', 'known animals first');
    ok(orphaned.some(l => l.animal === 'id_1'), 'then a deleted identity\'s labels, which are still labels');
    eq(M.stripLanes(['id_1'], [], L.listBouts(session), [], glyph), [], 'a bout naming no ethogram behavior has no lane');
    eq(M.stripLanes([], E.behaviors(), [], [], glyph), [], 'nothing labelled: no lanes');
}

// ---- the window ----------------------------------------------------------------
{
    const s = M.stripScale(5, 60, 1000);
    eq([s.half, s.visible], [300, 601], '±5 s at 60 fps: 300 frames each side of the playhead\'s');
    near(s.pxPerFrame, 1000 / 601, 'every frame has the same slot');
    eq(M.stripScale(0.001, 30, 500).half, 1, 'always at least a frame each side');
    eq(M.stripScale(5, 29.97, 500).half, 150, 'whole frames at any rate');

    eq(M.frameAtX(500, 1000, s), 1000, 'the middle is the playhead\'s frame');
    near(M.xOfFrame(1000, 1000, s), 500, 'and the playhead\'s frame is in the middle');
    eq(M.frameAtX(0, 1000, s), 700, 'the left edge is the first frame on show');
    eq(M.frameAtX(999.999, 1000, s), 1300, 'the right edge the last');
    for (const f of [700, 701, 999, 1000, 1001, 1300]) {
        eq(M.frameAtX(M.xOfFrame(f, 1000, s), 1000, s), f, 'a click on the middle of a slot lands on its frame: ' + f);
    }
    const close = M.stripScale(0.5, 30, 620);       // 31 frames, 20 px each
    eq(M.frameAtX(310 + 9.9, 100, close), 100, 'within the playhead\'s slot');
    eq(M.frameAtX(310 + 10.1, 100, close), 101, 'just past it, the next frame');
}

// ---- the film: moved, not repainted ------------------------------------------------
{
    const s = M.stripScale(5, 60, 1000);
    const film = M.stripFilm(5000, s);
    eq(film.end - film.start + 1, 3 * s.visible, 'three windows of frames');
    near(film.width, 3000, 'three windows wide', 1e-6);
    near(M.filmSourceX(5000, film, s, 1), 1000, 'centred on its frame, the window starts one window in', 0.5);
    const short = M.stripFilm(5000, s, 1600);
    ok(short.width <= 1600 && short.width > 1500, 'a limit makes a shorter film (' + short.width + ')');
    near((short.start + short.end) / 2, 5000, 'still centred');
    ok(M.filmHolds(short, 5000, s), 'that still holds the window');
    const least = M.stripFilm(5000, s, 500);
    near(least.width, 1000, 'never less than the window itself', 1e-6);
    ok(M.filmHolds(film, 5000, s), 'it holds the window it was built for');
    ok(M.filmHolds(film, 5000 + s.visible, s), 'and a whole window of playback later');
    ok(!M.filmHolds(film, 5000 + s.visible + 1, s), 'but not one frame more: then it is rebuilt');
    ok(M.filmHolds(film, 5000 - s.visible, s) && !M.filmHolds(film, 5000 - s.visible - 1, s), 'backwards too');

    // The playhead stays on its frame anywhere along the film, for any film.
    const dpr = 2;
    for (const [center, frame] of [[5000, 5000], [5000, 5123], [5000, 4400], [100, 0], [100, 650]]) {
        const fm = M.stripFilm(center, s);
        const sx = M.filmSourceX(frame, fm, s, dpr);
        ok(Number.isInteger(sx), 'in whole device pixels, so edges stay sharp');
        const middle = (frame - fm.start + 0.5) * s.pxPerFrame - sx / dpr;
        near(middle, 500, `frame ${frame} sits under the playhead`, 0.5 / dpr);
    }
    const far = M.stripScale(120, 60, 1000);
    const ff = M.stripFilm(50000, far);
    eq(M.filmSourceX(50001, ff, far, 1), M.filmSourceX(50000, ff, far, 1), 'zoomed far out, a frame can move nothing');

    // Clipping and the narrowest bar.
    const r = M.filmRect(0, 1e9, film, s);
    near(r.x, 0, 'a long bout is clipped to the film');
    near(r.w, film.width, 'so the DOM never holds a huge bar');
    eq(M.filmRect(0, film.start - 1, film, s), null, 'off the film to the left');
    eq(M.filmRect(film.end + 1, film.end + 5, film, s), null, 'off the film to the right');
    const wide = M.stripScale(120, 60, 1000);       // far under a pixel a frame
    const wf = M.stripFilm(50000, wide);
    const one = M.filmRect(50000, 50000, wf, wide);
    eq(one.w, M.MIN_BAR_PX, 'a one-frame bout stays visible when zoomed out');
    near(one.x + one.w / 2, (50000 - wf.start + 0.5) * wide.pxPerFrame, 'around the middle of its frame');
}

// ---- bars -----------------------------------------------------------------------
{
    const lane = (items, extra) => Object.assign({ animal: 'a', behaviorId: 'b1', name: 'rear', color: '#123456', point: false, pair: ' → ', items }, extra);
    const item = (start, end, present = true, running = false, partner = '') => ({ start, end, present, running, partner });
    const s = M.stripScale(5, 60, 1000);
    const film = M.stripFilm(1000, s);

    const [a, n, r] = M.filmBars([lane([item(900, 950, false), item(960, 980, true), item(990, 990, true, true)])], film, s, 1050);
    eq(a.kind, 'absent', 'a "not" label');
    eq(n.kind, 'present', 'a present bout');
    eq(n.color, '#123456', 'in the lane\'s color');
    eq(r.kind, 'running', 'a running state');
    near(r.x, (990 - film.start) * s.pxPerFrame, 'starts where it started');
    near(r.w, (1051 - 990) * s.pxPerFrame, 'and grows to the end of the playhead\'s slot, as on the Timeline', 1e-6);
    near(M.runningBarWidth(r.item, 1050, film, s), r.w, 'the per-frame width matches what was built', 1e-6);
    near(M.runningBarWidth(r.item, 1060, film, s) - r.w, 10 * s.pxPerFrame, 'ten frames later, ten frames longer', 1e-6);
    eq(M.runningBarWidth(r.item, 500, film, s), Math.max(M.MIN_BAR_PX, s.pxPerFrame), 'the playhead behind its start: just its first frame');

    const [offLeft] = M.filmBars([lane([item(film.start - 500, film.start + 10, true)])], film, s, 1000);
    near(offLeft.x, 0, 'a bar reaching off the film starts at its edge');
    eq(M.filmBars([lane([item(film.end + 1, film.end + 9)])], film, s, 1000), [], 'a bout off the film has no bar');
    eq(M.filmBars([lane([item(10, 20)]), lane([item(1000, 1000)])], film, s, 1000)[0].lane, 1, 'each bar knows its lane, by position in the list');

    const pt = M.filmBars([lane([item(1000, 1000), item(1001, 1001)], { point: true })], M.stripFilm(1000, M.stripScale(0.5, 30, 620)), M.stripScale(0.5, 30, 620), 1000);
    eq(pt.length, 2, 'two point events on neighbouring frames stay two');
    eq(pt[0].w, M.POINT_BAR_PX, 'a point is a tick');
    ok(pt[0].x + pt[0].w < pt[1].x, 'with a gap between them');
    const notPoint = M.filmBars([lane([item(1000, 1005, false)], { point: true })], film, s, 1000)[0];
    near(notPoint.w, 6 * s.pxPerFrame, '"not" on a point behavior is still its whole range');

    const wide = M.stripScale(120, 60, 1000);
    const wf = M.stripFilm(50000, wide);
    const dense = [];
    for (let i = 0; i < 2000; i++) dense.push(item(50000 + i * 3, 50000 + i * 3));
    const merged = M.filmBars([lane(dense)], wf, wide, 50000);
    ok(merged.length < 10, 'zoomed out past a pixel a frame, bars that touch become one (' + merged.length + ')');
    near(merged[0].x + merged[0].w, (50000 + 1999 * 3 - wf.start + 0.5) * wide.pxPerFrame + M.MIN_BAR_PX / 2, 'reaching the last of them', 1e-6);
    const twoTargets = M.filmBars([lane([item(50000, 50000, true, false, 'x'), item(50000, 50000, true, false, 'y')])], wf, wide, 50000);
    eq(twoTargets.length, 2, 'but not bars naming different partners');
    const fine = M.filmBars([lane([item(1000, 1010), item(1011, 1020)])], film, s, 1000);
    eq(fine.length, 2, 'at a pixel a frame or more, every bout keeps its own bar');
}

// ---- outside the recording ---------------------------------------------------------
{
    const s = M.stripScale(5, 60, 1000);
    const start = M.stripFilm(0, s);
    const [o] = M.filmOutside(start, s, 1e6);
    near(o.x, 0, 'before the first frame');
    near(o.w, -start.start * s.pxPerFrame, 'up to frame 0');
    const end = M.stripFilm(5999, s);
    const [t] = M.filmOutside(end, s, 6000);
    near(t.x, (6000 - end.start) * s.pxPerFrame, 'from just after the last frame', 1e-6);
    near(t.x + t.w, end.width, 'to the end of the film', 1e-6);
    eq(M.filmOutside(M.stripFilm(5000, s), s, 1e6), [], 'none in the middle of a recording');
    eq(M.filmOutside(start, s, 1e6).length, 1, 'only the end it reaches');
}

// ---- ticks ----------------------------------------------------------------------------
{
    const s = M.stripScale(5, 60, 1000);
    const film = M.stripFilm(3000, s);
    const t = M.filmTicks(film, s, 60, 36000);
    eq(t.step, 1, 'about a second a label at ±5 s on 1000 px');
    ok(t.minorStep > 0 && t.minorStep < 1, 'with minor ticks between');
    const at41 = t.majors.find(m => m.label === '00:50');
    ok(!!at41, 'labelled in the ruler\'s format');
    near(at41.x - M.filmSourceX(3000, film, s, 1000) / 1000, 500, 'the 00:50 tick is under the playhead when the readout says 00:50.000', 1e-2);
    const start = M.filmTicks(M.stripFilm(0, s), s, 60, 36000);
    ok(start.majors[0].label === '00:00', 'the first label is the recording\'s start');
    const end = M.stripFilm(35999, s);
    const et = M.filmTicks(end, s, 60, 36000);
    ok(et.majors[et.majors.length - 1].x <= (35999 - end.start + 1) * s.pxPerFrame, 'nothing past the last frame');
    const close = M.stripScale(0.5, 60, 1000);
    const ct = M.filmTicks(M.stripFilm(3000, close), close, 60, 36000);
    ok(ct.step < 1, 'zoomed in, tenths');
    ok(/\.\d$/.test(ct.majors[0].label), 'with a decimal');
    const far = M.stripScale(120, 60, 1000);
    ok(M.filmTicks(M.stripFilm(3000, far), far, 60, 36000).step >= 10, 'zoomed out, longer steps');
    for (const sc of [close, s, far]) {
        const mt = M.filmTicks(M.stripFilm(3000, sc), sc, 60, 36000);
        for (let i = 1; i < mt.majors.length; i++) {
            ok(mt.majors[i].x - mt.majors[i - 1].x >= 60 - 1e-6, 'labels clear each other');
        }
    }
    eq(M.filmTicks(film, s, 0, 36000).majors, [], 'no rate, no times');
    eq(M.filmTicks(film, M.stripScale(5, 60, 0), 60, 36000).majors, [], 'no width, no ticks');
}

// ---- hover ----------------------------------------------------------------------------
{
    const lane = { animal: 'id_1', name: 'attack', pair: ' → ', items: [
        { start: 100, end: 200, present: false, running: false, partner: 'id_2' },
        { start: 150, end: 160, present: true, running: false, partner: 'id_2' },
        { start: 300, end: 300, present: true, running: true, partner: 'id_2' },
    ] };
    eq(M.boutAt(lane, 155, 0).present, true, 'the topmost: a present bout over a "not"');
    eq(M.boutAt(lane, 120, 0).present, false, 'the "not" where it shows');
    eq(M.boutAt(lane, 250, 400), null, 'nothing between bouts');
    eq(M.boutAt(lane, 201, 0), null, 'nothing just past a bout');
    eq(M.boutAt(lane, 201, 0, 2).present, false, 'a bar widened to stay visible counts as hit');
    ok(M.boutAt(lane, 350, 400).running, 'a running state reaches the playhead');
    eq(M.boutAt(lane, 350, 320), null, 'and no further');
    ok(M.boutAt(lane, 300, 200).running, 'with the playhead behind it, its first frame still shows');

    const bt = { start: 1199, end: 1259, present: true, running: false, partner: 'id_2' };
    eq(M.boutText(lane, bt, 60), 'attack · id_1 → id_2 · frames 1,200–1,260 (1.0 s)', 'frames read 1-based, with separators');
    eq(M.boutText(lane, { ...bt, present: false, partner: '' }, 60), 'not attack · id_1 · frames 1,200–1,260 (1.0 s)', 'a "not" label');
    eq(M.boutText(lane, { ...bt, end: bt.start }, 60), 'attack · id_1 → id_2 · frame 1,200', 'a one-frame bout');
    eq(M.boutText(lane, { ...bt, running: true }, 60), '● attack · id_1 → id_2 · since frame 1,200', 'a running state');
    eq(M.stripHoverText(1233, 60, null, null), 'Frame 1,234 · 00:20.550', 'the seekbar tooltip\'s words');
    eq(M.stripHoverText(1233, 60, lane, bt), 'Frame 1,234 · 00:20.550 — attack · id_1 → id_2 · frames 1,200–1,260 (1.0 s)', 'and the bout under the pointer');
}

if (failed) {
    console.error(`\n${failed} of ${passed + failed} failed:`);
    failures.forEach(m => console.error('  - ' + m));
    process.exit(1);
}
console.log(`behavior strip: ${passed} checks passed`);
