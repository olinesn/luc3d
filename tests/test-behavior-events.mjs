/**
 * test-behavior-events.mjs — behavior labels as sleap-io events
 * (`import-export/behavior-events.js`), the form sleap-app and Python sleap-io read.
 *
 * The file itself is the e2e `behavior-label-roundtrip` (both writers) and, across
 * apps, the e2e `behavior-cross-app`. This pins the mapping:
 *
 *   - a project with nothing labelled makes no events and no catalog;
 *   - every label field survives: animal, target, span, source, note, "not";
 *   - a "not" label has its own type, so counting a behavior never counts it;
 *   - predictions keep their source and score, their settings stay in `lucid`;
 *   - an event another tool wrote comes in: a track's name is an animal's name, a
 *     target makes a behavior social, a prediction makes a run that never goes stale;
 *   - an event on any camera of a session is the session's, and only that session's;
 *   - a frame-level event survives a load and save unchanged;
 *   - a file of separate videos keeps each video's events on that video;
 *   - a video in no recording session keeps its events, on that video, through saves;
 *   - a video the save already lists for its poses takes its events on that entry;
 *   - with several sessions each event goes to one of them;
 *   - a session with only placeholder cameras reads like a file with no session;
 *   - renamed elsewhere, a behavior keeps its detector.
 */

import { buildBehaviorEvents, fileEventsOf, keptVideoRecords, sessionEventScope, sessionVideoIdxs } from '../import-export/behavior-events.js';
import { readBehaviorMetadata, writeBehaviorSlpMetadata, resetBehaviorState } from '../import-export/behavior-metadata.js';
import { addBehavior, behaviors, behaviorByName, setBehaviorDetector, serializeEthogram } from '../pose/behavior-ethogram.js';
import { labelRange, noteBehaviorFps, exportBoutRows, behaviorFps } from '../pose/behavior-labels.js';
import { setBehaviorRun, predictedBouts, predictionStaleness } from '../pose/behavior-predictions.js';

let passed = 0, failed = 0;
const check = (cond, msg) => {
    if (cond) { passed++; console.log('  ok   ' + msg); }
    else { failed++; console.log('  FAIL ' + msg); }
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// The sleap-io classes the module is handed, in the shape the library has.
class EventType { constructor(o) { Object.assign(this, { name: '', description: '', metadata: {} }, o); } }
class BaseEvent {
    constructor(o) {
        Object.assign(this, { subject: null, target: null, name: '', source: '', metadata: {} }, o);
        this.type = typeof o.type === 'string' ? new EventType({ name: o.type }) : o.type;
    }
}
class UserEvent extends BaseEvent { get isPredicted() { return false; } }
class PredictedEvent extends BaseEvent { get isPredicted() { return true; } }
class Identity { constructor(o) { Object.assign(this, { name: '', metadata: {} }, o); } }
class Track { constructor(name) { this.name = name; } }
class Video { constructor(o) { Object.assign(this, { backendMetadata: {} }, o); } }
const SIO = { EventType, UserEvent, PredictedEvent, Identity, Track, Video };

/** What a file holds after a save: the plain form the readers get. */
function asFile(eventTypes, events, videos) {
    return fileEventsOf({ eventTypes, events, videos });
}

console.log('\n1. Nothing labelled, nothing written');
{
    resetBehaviorState();
    const out = buildBehaviorEvents(SIO, [{ session: { name: 's' }, video: {}, identities: [], tracks: [] }]);
    check(out.eventTypes.length === 0 && out.events.length === 0, 'no ethogram, no bouts: no catalog, no events');
    check(fileEventsOf({ eventTypes: [], events: [], videos: [] }) === null, 'a file without events reads as none');
    const lucid = {};
    writeBehaviorSlpMetadata(lucid, { name: 's' });
    check(Object.keys(lucid).length === 0, 'and no key beside them');
}

console.log('\n2. Labels survive');
{
    resetBehaviorState();
    const rear = addBehavior({ name: 'rear', key: 'r', description: 'forepaws up' }).behavior;
    const chase = addBehavior({ name: 'chase', key: 'c', social: true, group: 'social' }).behavior;
    const sniff = addBehavior({ name: 'sniff', social: true, mutual: true }).behavior;
    const bite = addBehavior({ name: 'bite', kind: 'point', social: true }).behavior;
    const s = { name: 'day1' };
    labelRange(s, rear.id, 'id_1', '', 10, 20, true, { note: 'tall' });
    labelRange(s, rear.id, 'id_1', '', 30, 40, false);
    labelRange(s, chase.id, 'id_1', 'id_2', 50, 60, true, { source: 'a' });
    labelRange(s, sniff.id, 'id_2', 'id_1', 70, 75, true);
    labelRange(s, bite.id, 'id_2', 'id_1', 80, 80, true);
    noteBehaviorFps(s, 50);
    const before = exportBoutRows(s);
    const eth = serializeEthogram();

    const video = { filename: 'cam0.mp4' };
    const id1 = new Identity({ name: 'id_1' }), id2 = new Identity({ name: 'id_2' });
    const out = buildBehaviorEvents(SIO, [{ session: s, video, identities: [id1, id2], tracks: [new Track('t0')] }]);
    check(eq(out.eventTypes.map(t => t.name), ['rear', 'chase', 'sniff', 'bite', 'not rear']), 'the catalog: the ethogram, then "not rear"');
    const rearT = out.eventTypes[0], chaseT = out.eventTypes[1], sniffT = out.eventTypes[2], biteT = out.eventTypes[3], notT = out.eventTypes[4];
    check(rearT.description === 'forepaws up' && rearT.metadata.key === 'r' && rearT.metadata.lucid_id === rear.id && /^#/.test(rearT.metadata.color), 'a type carries its description, key, id and color');
    check(chaseT.metadata.directed === 'true' && chaseT.metadata.group === 'social' && !chaseT.metadata.mutual, 'a social behavior is directed');
    check(sniffT.metadata.mutual === 'true' && !sniffT.metadata.directed, 'a mutual one says mutual');
    check(biteT.metadata.kind === 'point' && !rearT.metadata.kind, 'a point says so; a state writes nothing');
    check(notT.metadata.negates === 'rear', 'the "not" type names what it negates');
    check(out.events.length === 5 && out.events.every(e => e.video === video), 'one event per bout, on the given video');
    const rearE = out.events.find(e => e.type === rearT);
    check(rearE.startFrame === 10 && rearE.endFrame === 20 && rearE.subject === id1 && rearE.target === null && rearE.source === 'manual' && rearE.metadata.note === 'tall',
        'a bout: span, animal, source, note');
    check(out.events.filter(e => e.type === rearT).length === 1, 'counting "rear" does not count its "not" label');
    check(out.events.find(e => e.type === chaseT).source === 'accepted_prediction', 'an accepted prediction says so');

    const file = asFile(out.eventTypes, out.events, [video]);
    const lucid = {};
    writeBehaviorSlpMetadata(lucid, s);
    check(eq(lucid, { behaviorLabels: { v: 2, fps: 50 } }), 'beside the events only the frame rate is in lucid');
    resetBehaviorState();
    const back = { name: 'back' };
    const res = readBehaviorMetadata(back, JSON.parse(JSON.stringify(lucid)), file);
    check(res.kept === 5 && res.dropped === 0, 'all bouts read back');
    check(eq(exportBoutRows(back), before), 'exactly as they were, sources and notes too');
    check(eq(serializeEthogram(), eth), 'the ethogram too, ids included');
    check(behaviorFps(back) === 50, 'and the frame rate');
}

console.log('\n3. Predictions: bouts in events, settings in lucid');
{
    resetBehaviorState();
    const rear = addBehavior({ name: 'rear' }).behavior;
    const s = { name: 's' };
    setBehaviorRun(s, { behaviorId: rear.id, cfg: 'print', fps: 30, mode: '3d',
        bouts: [{ animal: 'id_1', target: '', start: 5, end: 9, score: 0.9126 }, { animal: 'id_2', target: '', start: 30, end: 33, score: NaN }] });
    const out = buildBehaviorEvents(SIO, [{ session: s, video: {}, identities: [new Identity({ name: 'id_1' }), new Identity({ name: 'id_2' })] }]);
    check(out.events.length === 2 && out.events.every(e => e.isPredicted && e.source === 'rule'), 'predicted events, with the run\'s source');
    check(out.events[0].score === 0.913 && out.events[1].score === null, 'and their scores (a missing one stays missing)');
    const lucid = {};
    writeBehaviorSlpMetadata(lucid, s);
    check(lucid.behaviorPredictions.v === 2 && lucid.behaviorPredictions.runs[0].cfg === 'print' && !lucid.behaviorPredictions.runs[0].bouts, 'the run\'s settings stay in lucid, without the bouts');
    resetBehaviorState();
    const back = { name: 'back' };
    readBehaviorMetadata(back, JSON.parse(JSON.stringify(lucid)), asFile(out.eventTypes, out.events, []));
    const rows = predictedBouts(back);
    check(rows.length === 2 && rows[0].start === 5 && rows[0].score === 0.913 && Number.isNaN(rows[1].score), 'they read back onto one run');
}

console.log('\n4. An event another tool wrote');
{
    resetBehaviorState();
    const video = {};
    const t1 = new Track('mouse1'), t2 = new Track('mouse2');
    const attack = new EventType({ name: 'attack', metadata: { color: '#e6194b' } });
    const rear = new EventType({ name: 'rear' });
    const events = [
        new UserEvent({ type: attack, video, startFrame: 100, endFrame: 140, subject: t1, target: t2, source: 'manual' }),
        new UserEvent({ type: rear, video, startFrame: 50, endFrame: 50, subject: t1 }),
        new PredictedEvent({ type: rear, video, startFrame: 7, endFrame: 9, subject: t2, score: 0.5, source: 'classifier', scores: Float32Array.from([1, 2, 3]) }),
    ];
    const s = { name: 's' };
    const res = readBehaviorMetadata(s, null, asFile([attack, rear], events, [video]));
    check(res.kept === 2, 'read with no metadata.lucid at all');
    check(behaviorByName('attack').social && !behaviorByName('rear').social, 'a target makes a behavior social');
    check(behaviorByName('attack').color === '#e6194b', 'its color comes along');
    const rows = exportBoutRows(s);
    check(rows.some(r => r[1] === 'mouse1' && r[2] === 'mouse2' && r[3] === 100), 'a track\'s name is the animal\'s name');
    check(rows.some(r => r[3] === 50 && r[6] === 'i'), 'no source reads as imported');
    const pred = predictedBouts(s);
    check(pred.length === 1 && pred[0].animal === 'mouse2' && pred[0].score === 0.5, 'a prediction becomes a run');
    check(!predictionStaleness(s, {}).settings.length, 'which no setting here can make stale');
    const out = buildBehaviorEvents(SIO, [{ session: s, video, tracks: [t1, t2] }]);
    const again = out.events.find(e => e.isPredicted);
    check(again.source === 'classifier' && eq(Array.from(again.scores), [1, 2, 3]), 'and it saves back as it came, framewise scores too');
    check(out.events.find(e => e.startFrame === 100).subject === t1, 'an animal with no identity is written as its track');
}

console.log('\n5. Sessions: any camera, only its own');
{
    resetBehaviorState();
    const rear = new EventType({ name: 'rear' });
    const v = [{}, {}, {}, {}];
    const a = new Identity({ name: 'a' });
    const events = [
        new UserEvent({ type: rear, video: v[0], startFrame: 1, endFrame: 2, subject: a }),
        new UserEvent({ type: rear, video: v[1], startFrame: 10, endFrame: 11, subject: a }),   // a second camera of session 0
        new UserEvent({ type: rear, video: v[2], startFrame: 20, endFrame: 21, subject: a }),   // session 1
    ];
    const file = asFile([rear], events, v);
    const sessions = [{ camcorder_to_video_idx_map: { 0: 0, 1: 1 } }, { camcorder_to_video_idx_map: { 0: 2, 1: 3 } }];
    const s0 = { name: 's0' }, s1 = { name: 's1' };
    readBehaviorMetadata(s0, {}, file, sessionEventScope(sessions, 4, 0));
    readBehaviorMetadata(s1, {}, file, sessionEventScope(sessions, 4, 1));
    check(eq(exportBoutRows(s0).map(r => r[3]), [1, 10]), 'an event on either camera is its session\'s');
    check(eq(exportBoutRows(s1).map(r => r[3]), [20]), 'and not the other session\'s');
    check(behaviors().length === 1, 'the catalog comes in once');
    check(sessionVideoIdxs({}) === null && sessionVideoIdxs(null) === null, 'a session may name no videos');
    const namesNone = sessionEventScope([{}], 4, 0);
    check(namesNone.videos.size === 4 && namesNone.keep.size === 0, 'and then it reads every row');
    const besideOther = sessionEventScope([{}, sessions[1]], 4, 0);
    check(eq([...besideOther.videos], [0, 1]) && besideOther.keep.size === 0, 'except another session\'s');
}

console.log('\n6. What labels cannot hold survives a save');
{
    resetBehaviorState();
    const light = new EventType({ name: 'light_on', description: 'stimulus' });
    const chase = new EventType({ name: 'chase', metadata: { directed: 'true' } });
    const video = {}, m = new Identity({ name: 'm' });
    const events = [
        new UserEvent({ type: light, video, startFrame: 200, endFrame: 250, source: 'daq', metadata: { hz: '20' } }),
        new UserEvent({ type: chase, video, startFrame: 300, endFrame: 310, subject: m }),      // directed, no target
    ];
    const s = { name: 's' };
    const res = readBehaviorMetadata(s, {}, asFile([light, chase], events, [video]));
    check(res.kept === 0 && exportBoutRows(s).length === 0, 'not labels');
    const out = buildBehaviorEvents(SIO, [{ session: s, video, identities: [m] }]);
    const lightE = out.events.find(e => e.type.name === 'light_on');
    check(lightE && lightE.subject === null && lightE.startFrame === 200 && lightE.endFrame === 250 && lightE.source === 'daq' && lightE.metadata.hz === '20',
        'a frame-level event is written back unchanged');
    const chaseE = out.events.find(e => e.type.name === 'chase');
    check(chaseE && chaseE.subject === m && chaseE.target === null, 'and so is a social event with no target');
}

console.log('\n6b. A file of separate videos, which LUC3D opens as cameras');
{
    resetBehaviorState();
    const rear = new EventType({ name: 'rear' });
    const v = [new Video({ filename: 'one.mp4' }), new Video({ filename: 'two.mp4' }), new Video({ filename: 'three.mp4' })];
    const a = new Track('a');
    const events = [
        new UserEvent({ type: rear, video: v[0], startFrame: 1, endFrame: 2, subject: a, source: 'manual' }),
        new UserEvent({ type: rear, video: v[2], startFrame: 1, endFrame: 2, subject: a, source: 'manual' }),
        new UserEvent({ type: rear, video: v[2], startFrame: 50, endFrame: 60, subject: a, source: 'manual' }),
    ];
    const scope = sessionEventScope([], 3, 0);
    check(eq([...scope.videos], [0]) && eq([...scope.keep], [1, 2]), 'no session, several videos: only the first is labels');
    check(sessionEventScope([], 1, 0).keep.size === 0, 'one video: all of it');
    const named = sessionEventScope([{ camcorder_to_video_idx_map: { 0: 0, 1: 1, 2: 2 } }], 3, 0);
    check(named.videos.size === 3 && named.keep.size === 0, 'a session says which videos go together');
    const s = { name: 's' };
    const res = readBehaviorMetadata(s, null, asFile([rear], events, v), scope);
    check(res.kept === 1 && eq(exportBoutRows(s).map(r => r[3]), [1]), 'the first video\'s bouts are the labels');
    const out = buildBehaviorEvents(SIO, [{ session: s, video: v[0], videos: v, tracks: [a] }]);
    check(out.events.filter(e => e.video === v[0]).length === 1 && out.events.filter(e => e.video === v[2]).length === 2, 'saved, each bout is still on its own video');
    const oneCamera = buildBehaviorEvents(SIO, [{ session: s, video: v[0], tracks: [a] }]);
    check(oneCamera.events.length === 1, 'a one-camera export leaves the others out');
    const noCameras = buildBehaviorEvents(SIO, [{ session: s, video: null, videos: [], tracks: [a] }]);
    check(noCameras.events.filter(e => e.video && e.video.filename === 'three.mp4').length === 2, 'with no camera to go back to, its own video is written back');
}

console.log('\n6c. A video in no recording session');
{
    resetBehaviorState();
    const rear = new EventType({ name: 'rear' });
    const attack = new EventType({ name: 'attack', metadata: { directed: 'true' } });
    const notRear = new EventType({ name: 'not rear', metadata: { negates: 'rear' } });
    const stimulus = new EventType({ name: 'stimulus' });
    const types = [attack, rear, notRear, stimulus];
    // The other video comes first in the file, so its index is also a camera's on save.
    const outside = new Video({ filename: 'outside.mp4', backendMetadata: { shape: [400, 64, 64, 1], fps: 30 } });
    const a0 = new Video({ filename: 'a0.mp4' }), a1 = new Video({ filename: 'a1.mp4' });
    const a = new Identity({ name: 'a' }), b = new Identity({ name: 'b' });
    const events = [
        new UserEvent({ type: rear, video: a0, startFrame: 10, endFrame: 19, subject: a, source: 'manual' }),
        new UserEvent({ type: rear, video: a1, startFrame: 30, endFrame: 39, subject: a, source: 'manual' }),
        new UserEvent({ type: attack, video: outside, startFrame: 300, endFrame: 301, subject: b, target: a, source: 'manual', name: 'first', metadata: { certainty: 'high' } }),
        new UserEvent({ type: notRear, video: outside, startFrame: 310, endFrame: 311, subject: a, source: 'manual' }),
        new PredictedEvent({ type: rear, video: outside, startFrame: 320, endFrame: 322, subject: a, score: 0.5, scores: Float32Array.from([0.25, 0.5, 0.75]), source: 'classifier' }),
        new UserEvent({ type: stimulus, video: outside, startFrame: 330, endFrame: 335, source: 'daq', metadata: { hz: '20' } }),
    ];
    const rawSessions = [{ camcorder_to_video_idx_map: { 0: 1, 1: 2 } }];
    const scope = sessionEventScope(rawSessions, 3, 0);
    check(eq([...scope.videos], [1, 2]) && eq([...scope.keep], [0]) && !scope.cameras, 'the session reads its cameras and only keeps the other video');
    const s = { name: 's' };
    const res = readBehaviorMetadata(s, null, asFile(types, events, [outside, a0, a1]), scope);
    check(res.kept === 2 && eq(exportBoutRows(s).map(r => r[3]), [10, 30]), 'the session shows only its own bouts');
    check(predictedBouts(s).length === 0, 'and none of that video\'s predictions');
    check(behaviorByName('attack').social, 'the type still comes into the ethogram, as what it is');

    const cams = [new Video({ filename: 'camA.mp4' }), new Video({ filename: 'camB.mp4' })];
    const saved = (session, videos) => buildBehaviorEvents(SIO, [{ session, video: videos[0], videos, identities: [a, b] }]);
    const out = saved(s, cams);
    const there = out.events.filter(e => !cams.includes(e.video));
    check(out.events.length === 6 && out.events.filter(e => e.video === cams[0]).length === 2, 'saved: no event lost, the session\'s on its first camera');
    check(there.length === 4 && new Set(there.map(e => e.video)).size === 1, 'the others on one video, none of them on a camera');
    const madeVideo = there[0].video;
    check(madeVideo.filename === 'outside.mp4' && eq(madeVideo.backendMetadata, outside.backendMetadata) && madeVideo.openBackend === false,
        'that video is written back as the file described it');
    const attackE = there.find(e => e.type.name === 'attack');
    check(attackE.startFrame === 300 && attackE.endFrame === 301 && attackE.subject === b && attackE.target === a && attackE.source === 'manual' && attackE.name === 'first' && attackE.metadata.certainty === 'high',
        'a bout: span, animals, source, name and metadata unchanged');
    const predE = there.find(e => e.isPredicted);
    check(predE.score === 0.5 && eq(Array.from(predE.scores), [0.25, 0.5, 0.75]) && predE.source === 'classifier', 'a prediction: score, framewise scores and source');
    check(there.find(e => e.type.name === 'stimulus').subject === null, 'an event with no animal');
    check(out.eventTypes.find(t => t.name === 'not rear').metadata.negates === 'rear', 'a "not" stays a "not", with no other bout to keep its type');
    check(buildBehaviorEvents(SIO, [{ session: s, video: cams[0], identities: [a, b] }]).events.length === 2, 'a one-camera export leaves that video out');

    // The save read back: the file now lists the video last, with the same result.
    resetBehaviorState();
    const again = { name: 'again' };
    const file2 = asFile(out.eventTypes, out.events, [cams[0], cams[1], madeVideo]);
    const res2 = readBehaviorMetadata(again, null, file2, sessionEventScope([{ camcorder_to_video_idx_map: { 0: 0, 1: 1 } }], 3, 0));
    check(res2.kept === 2 && eq(exportBoutRows(again).map(r => r[3]), [10, 30]), 'reopened, the session shows the same bouts');
    const out2 = saved(again, cams);
    const key = e => [e.video.filename, e.type.name, e.startFrame, e.endFrame, e.subject && e.subject.name, e.target && e.target.name, e.isPredicted, e.source];
    check(eq(out2.events.map(key), out.events.map(key)), 'and saves the same events on the same videos');
    check(out2.eventTypes.find(t => t.name === 'not rear').metadata.negates === 'rear', 'a "not" still a "not"');

    // A "not" with no animal is kept on the session's own video: the same catalog entry.
    resetBehaviorState();
    const alone = { name: 'alone' };
    readBehaviorMetadata(alone, null, asFile([rear, notRear], [new UserEvent({ type: notRear, video: a0, startFrame: 5, endFrame: 6 })], [a0]));
    const aloneOut = buildBehaviorEvents(SIO, [{ session: alone, video: a0, videos: [a0] }]);
    check(aloneOut.events[0].video === a0 && aloneOut.eventTypes.find(t => t.name === 'not rear').metadata.negates === 'rear', 'a "not" no label can hold stays a "not" too');

    // Images stored in the file do not come along: the source video goes back, and no dataset.
    resetBehaviorState();
    const embedded = new Video({ filename: 'labels.pkg.slp', hasEmbeddedImages: true, shape: [5, 64, 64, 1], sourceVideo: new Video({ filename: 'original.mp4' }),
        backendMetadata: { dataset: 'video2/video', shape: [5, 64, 64, 1] } });
    const em = { name: 'embedded' };
    readBehaviorMetadata(em, null, asFile([rear], [new UserEvent({ type: rear, video: embedded, startFrame: 1, endFrame: 2, subject: a })], [a0, embedded]),
        sessionEventScope([{ camcorder_to_video_idx_map: { 0: 0 } }], 2, 0));
    const emOut = buildBehaviorEvents(SIO, [{ session: em, video: a0, videos: [a0], identities: [a] }]);
    check(emOut.events[0].video.filename === 'original.mp4' && eq(emOut.events[0].video.backendMetadata, { shape: [5, 64, 64, 1] }), 'an embedded video goes back as its source, pointing at no images');
}

console.log('\n6c2. A video the save already lists for its poses');
{
    resetBehaviorState();
    const rear = new EventType({ name: 'rear' });
    const a = new Identity({ name: 'a' });
    // campy names every video 0.mp4, so a name says nothing.
    const v = ['0.mp4', '0.mp4', '0.mp4', '0.mp4'].map(filename => new Video({ filename }));
    const bout = (video, start) => new UserEvent({ type: rear, video, startFrame: start, endFrame: start + 1, subject: a, source: 'manual' });
    const file = asFile([rear], [bout(v[0], 10), bout(v[2], 300), bout(v[3], 400), bout(v[2], 310)], v);
    const s = { name: 's' };
    readBehaviorMetadata(s, {}, file, sessionEventScope([{ camcorder_to_video_idx_map: { 0: 0, 1: 1 } }], 4, 0));

    const records = keptVideoRecords(s);
    check(eq([...records.keys()], [2, 3]) && records.get(2) === file.videos[2] && records.get(3) === file.videos[3], 'a session knows which of the file\'s videos it keeps events on, by their place in the file');
    check(keptVideoRecords({ name: 'none' }).size === 0, 'a session with none knows of none');

    const cams = [new Video({ filename: 'camA.mp4' }), new Video({ filename: 'camB.mp4' })];
    const listed = new Video({ filename: '0.mp4' });
    const out = buildBehaviorEvents(SIO, [{ session: s, video: cams[0], videos: cams, identities: [a], listedVideos: new Map([[file.videos[2], listed]]) }]);
    check(eq(out.events.filter(e => e.video === listed).map(e => e.startFrame), [300, 310]), 'its events go on the entry the writer lists');
    const other = out.events.find(e => e.startFrame === 400).video;
    check(other !== listed && !cams.includes(other) && other.filename === '0.mp4', 'a video of the same name keeps an entry of its own');
    check(out.events.find(e => e.startFrame === 10).video === cams[0], 'and the session\'s bout stays on its first camera');
    check(eq(buildBehaviorEvents(SIO, [{ session: s, video: cams[0], identities: [a], listedVideos: new Map([[file.videos[2], listed]]) }]).events.map(e => e.startFrame), [10]),
        'a one-camera export still leaves them out');
}

console.log('\n6d. Several sessions: each event goes to one of them');
{
    const rear = new EventType({ name: 'rear' });
    const v = ['a0', 'a1', 'b0', 'b1', 'outside'].map(n => new Video({ filename: n + '.mp4' }));
    const a = new Identity({ name: 'a' });
    const bout = (video, start) => new UserEvent({ type: rear, video, startFrame: start, endFrame: start + 1, subject: a, source: 'manual' });
    const events = [bout(v[0], 10), bout(v[1], 20), bout(v[2], 110), bout(v[3], 120), bout(v[4], 300), bout(null, 500)];
    const rawSessions = [{ camcorder_to_video_idx_map: { 0: 0, 1: 1 } }, { camcorder_to_video_idx_map: { 0: 2, 1: 3 } }];
    const file = asFile([rear], events, v);

    resetBehaviorState();
    const s0 = { name: 's0' }, s1 = { name: 's1' };
    readBehaviorMetadata(s0, {}, file, sessionEventScope(rawSessions, 5, 0));
    readBehaviorMetadata(s1, {}, file, sessionEventScope(rawSessions, 5, 1));
    check(eq(exportBoutRows(s0).map(r => r[3]), [10, 20, 500]), 'the first session: its own, and the event with no video');
    check(eq(exportBoutRows(s1).map(r => r[3]), [110, 120]), 'the second: only its own');
    const camsOf = names => names.map(n => new Video({ filename: n }));
    const c0 = camsOf(['a0.mp4', 'a1.mp4']), c1 = camsOf(['b0.mp4', 'b1.mp4']);
    const out = buildBehaviorEvents(SIO, [
        { session: s0, video: c0[0], videos: c0, identities: [a] },
        { session: s1, video: c1[0], videos: c1, identities: [a] },
    ]);
    const starts = (name) => out.events.filter(e => e.video && e.video.filename === name).map(e => e.startFrame);
    check(out.events.length === 6, 'saved, no event is lost or doubled');
    check(eq(starts('a0.mp4'), [10, 20, 500]) && eq(starts('b0.mp4'), [110, 120]) && eq(starts('outside.mp4'), [300]), 'and each is where it belongs');

    // A loader that opens only the first session: nothing else would keep the second's.
    resetBehaviorState();
    const only = { name: 'only' };
    const scope = sessionEventScope(rawSessions, 5, 0, true);
    check(eq([...scope.videos], [0, 1]) && eq([...scope.keep], [2, 3, 4]), 'a loader of one session keeps every other video');
    readBehaviorMetadata(only, {}, file, scope);
    check(eq(exportBoutRows(only).map(r => r[3]), [10, 20, 500]), 'and shows only its session\'s');
    const c = camsOf(['a0.mp4', 'a1.mp4']);
    const outOnly = buildBehaviorEvents(SIO, [{ session: only, video: c[0], videos: c, identities: [a] }]);
    const onName = (name) => outOnly.events.filter(e => e.video && e.video.filename === name).map(e => e.startFrame);
    check(outOnly.events.length === 6 && eq(onName('b0.mp4'), [110]) && eq(onName('b1.mp4'), [120]) && eq(onName('outside.mp4'), [300]),
        'saved, the other session\'s bouts stay on their videos');
}

console.log('\n6e. A session with only placeholder cameras counts as no session');
{
    const placeholder = (name) => ({ name, rotation: [0, 0, 0], translation: [0, 0, 0], matrix: [[600, 0, 32], [0, 600, 32], [0, 0, 1]], distortions: [0, 0, 0, 0, 0], size: [64, 64] });
    const real = (name, i) => Object.assign(placeholder(name), { rotation: [0, 0.1 * i, 0], translation: [10 * i, 0, 0] });
    const uncalibrated = (map, n = Object.keys(map).length) => {
        const calibration = { metadata: {} };
        for (let i = 0; i < n; i++) calibration['cam_' + i] = placeholder('c' + i);
        return { calibration, camcorder_to_video_idx_map: map };
    };
    const rear = new EventType({ name: 'rear' });
    const a = new Identity({ name: 'a' });
    const bout = (video, start, end = start + 1) => new UserEvent({ type: rear, video, startFrame: start, endFrame: end, subject: a, source: 'manual' });
    const videoNamed = n => new Video({ filename: n + '.mp4' });

    // What a LUC3D save writes for a file of separate videos: the same result as the file before it.
    const v = ['one', 'two', 'three'].map(videoNamed);
    const events = [bout(v[0], 1), bout(v[1], 10), bout(v[2], 20)];
    const scope = sessionEventScope([uncalibrated({ 0: 0, 1: 1, 2: 2 })], 3, 0);
    check(eq([...scope.videos], [0]) && eq([...scope.keep], [1, 2]), 'only the first camera\'s events are labels');
    resetBehaviorState();
    const s = { name: 's' };
    readBehaviorMetadata(s, null, asFile([rear], events, v), scope);
    check(eq(exportBoutRows(s).map(r => r[3]), [1]), 'it shows those');
    const out = buildBehaviorEvents(SIO, [{ session: s, video: v[0], videos: v, identities: [a] }]);
    check(eq(out.events.map(e => [e.video.filename, e.startFrame]), [['one.mp4', 1], ['two.mp4', 10], ['three.mp4', 20]]), 'saved, each stays on its camera');
    check(out.events.every(e => v.includes(e.video)), 'those cameras\' own videos, not copies');

    // One animal's adjacent bouts on different videos stay two bouts.
    resetBehaviorState();
    const apart = { name: 'apart' };
    const adjacent = [bout(v[0], 10, 19), bout(v[1], 20, 29)];
    readBehaviorMetadata(apart, null, asFile([rear], adjacent, v), scope);
    const apartOut = buildBehaviorEvents(SIO, [{ session: apart, video: v[0], videos: v, identities: [a] }]);
    check(eq(apartOut.events.map(e => [e.video.filename, e.startFrame, e.endFrame]), [['one.mp4', 10, 19], ['two.mp4', 20, 29]]), 'one animal\'s adjacent bouts on two videos do not merge');

    // The first camera is the one the session says, wherever its video is in the file.
    const shuffled = ['outside', 'b', 'a'].map(videoNamed);
    const sc = sessionEventScope([uncalibrated({ 0: 2, 1: 1 })], 3, 0);
    check(eq([...sc.videos], [2]) && eq([...sc.keep], [0, 1]), 'the first camera may be any of the file\'s videos');
    check(sc.cameras && sc.cameras.get(1) === 1 && sc.cameras.get(2) === 0 && !sc.cameras.has(0), 'each camera by its place in the session, a video in no session by none');
    resetBehaviorState();
    const sh = { name: 'sh' };
    readBehaviorMetadata(sh, null, asFile([rear], [bout(shuffled[2], 1), bout(shuffled[1], 10), bout(shuffled[0], 20)], shuffled), sc);
    check(eq(exportBoutRows(sh).map(r => r[3]), [1]), 'its bouts are the labels');
    const cams = ['camA', 'camB'].map(videoNamed);
    const shOut = buildBehaviorEvents(SIO, [{ session: sh, video: cams[0], videos: cams, identities: [a] }]);
    const placed = shOut.events.map(e => [e.video.filename, e.startFrame]).sort((p, q) => p[1] - q[1]);
    check(eq(placed, [['camA.mp4', 1], ['camB.mp4', 10], ['outside.mp4', 20]]), 'saved: the camera\'s bout on that camera, the other video\'s on that video');

    // Keys may be camera names or calibration keys, as the loaders allow.
    const named = uncalibrated({}, 2);
    named.camcorder_to_video_idx_map = { c1: 0, cam_0: 1 };
    check(eq([...sessionEventScope([named], 2, 0).videos], [1]), 'a camera may be named by its name or its calibration key');

    // Two uncalibrated sessions: a position is in the session, never in the file.
    const w = ['a0', 'a1', 'b0', 'b1'].map(videoNamed);
    const two = [uncalibrated({ 0: 0, 1: 1 }), uncalibrated({ 0: 2, 1: 3 })];
    const file = asFile([rear], [bout(w[0], 1), bout(w[1], 10), bout(w[2], 100), bout(w[3], 110)], w);
    resetBehaviorState();
    const s0 = { name: 's0' }, s1 = { name: 's1' };
    readBehaviorMetadata(s0, {}, file, sessionEventScope(two, 4, 0));
    readBehaviorMetadata(s1, {}, file, sessionEventScope(two, 4, 1));
    check(eq(exportBoutRows(s0).map(r => r[3]), [1]) && eq(exportBoutRows(s1).map(r => r[3]), [100]), 'each session shows its first camera\'s');
    const c0 = ['a0', 'a1'].map(videoNamed), c1 = ['b0', 'b1'].map(videoNamed);
    const twoOut = buildBehaviorEvents(SIO, [
        { session: s0, video: c0[0], videos: c0, identities: [a] },
        { session: s1, video: c1[0], videos: c1, identities: [a] },
    ]);
    const onCams = c0.concat(c1);
    check(twoOut.events.length === 4 && twoOut.events.every(e => onCams.includes(e.video)), 'saved, each on a camera of its own session, with no copy of a video');
    check(twoOut.events.find(e => e.startFrame === 110).video === c1[1], 'the second session\'s second camera, though its number in the file is 3');

    // A loader that opens one session keeps the other's on their videos, as any other session's.
    const scopeOne = sessionEventScope(two, 4, 0, true);
    check(eq([...scopeOne.videos], [0]) && eq([...scopeOne.keep], [1, 2, 3]) && !scopeOne.cameras.has(2), 'a loader of one session keeps the other session\'s videos as they are');

    // Beside a calibrated session, each goes by its own cameras.
    const mixed = [{ calibration: { cam_0: real('r0', 0), cam_1: real('r1', 1) }, camcorder_to_video_idx_map: { 0: 0, 1: 1 } }, uncalibrated({ 0: 2, 1: 3 })];
    check(eq([...sessionEventScope(mixed, 4, 0).videos], [0, 1]), 'a calibrated session reads all its cameras');
    check(eq([...sessionEventScope(mixed, 4, 1).videos], [2]), 'an uncalibrated one beside it, its first');

    // What is a calibration.
    const scopeOf = (calibration) => sessionEventScope([{ calibration, camcorder_to_video_idx_map: { 0: 0, 1: 1 } }], 2, 0);
    check(scopeOf({ cam_0: placeholder('a'), cam_1: real('b', 1) }).videos.size === 2, 'one real camera is enough, as for `hasRealCalibration`');
    check(scopeOf({ cam_0: Object.assign(placeholder('a'), { rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] }), cam_1: placeholder('b') }).videos.size === 2, 'a rotation matrix is a calibration');
    check(scopeOf({ cam_0: { name: 'a' }, cam_1: { name: 'b' } }).videos.size === 1, 'cameras with no rotation or translation are placeholders, as the loaders read them');
    check(sessionEventScope([{ camcorder_to_video_idx_map: { 0: 0, 1: 1 } }], 2, 0).videos.size === 2, 'a session with no calibration entry at all still says which videos go together');
    check(sessionEventScope([{ calibration: { metadata: {} }, camcorder_to_video_idx_map: { 0: 0, 1: 1 } }], 2, 0).videos.size === 2, 'nor do its metadata make it a camera');
    check(eq([...sessionEventScope([uncalibrated({}, 2)], 2, 0).videos], [0, 1]), 'cameras naming no video: it reads what no other session names, as before');
}

console.log('\n6f. Cameras a loader matched to the file\'s videos itself');
{
    const rear = new EventType({ name: 'rear' });
    const a = new Identity({ name: 'a' });
    const bout = (video, start) => new UserEvent({ type: rear, video, startFrame: start, endFrame: start + 1, subject: a, source: 'manual' });
    const videoNamed = n => new Video({ filename: n + '.mp4' });
    const placedBy = (out) => out.events.map(e => [e.video.filename, e.startFrame]).sort((p, q) => p[1] - q[1]);

    // No session; a calibration listing camB then camA; the first video matches no camera.
    const v = ['other', 'camA', 'camB'].map(videoNamed);
    const events = [bout(v[0], 200), bout(v[1], 10), bout(v[2], 300), bout(null, 500)];
    const onCameras = new Map([[1, 1], [2, 0]]);
    const scope = sessionEventScope([], 3, 0, true, onCameras);
    check(eq([...scope.videos], [2]) && eq([...scope.keep], [0, 1]), 'no session: the first camera\'s video is the labels, wherever the file lists it');
    check(scope.cameras.get(1) === 1 && scope.cameras.get(2) === 0 && !scope.cameras.has(0), 'each by its camera\'s place, a video with no camera by none');
    check(scope.noVideo === true, 'the event with no video is still read');
    resetBehaviorState();
    const s = { name: 's' };
    readBehaviorMetadata(s, null, asFile([rear], events, v), scope);
    check(eq(exportBoutRows(s).map(r => r[3]), [300, 500]), 'it shows the first camera\'s and the one with no video');
    const cams = ['camB', 'camA'].map(videoNamed);
    const out = buildBehaviorEvents(SIO, [{ session: s, video: cams[0], videos: cams, identities: [a] }]);
    check(eq(placedBy(out), [['camA.mp4', 10], ['other.mp4', 200], ['camB.mp4', 300], ['camB.mp4', 500]]), 'saved: each on its own video');
    check(out.events.filter(e => e.startFrame !== 200).every(e => cams.includes(e.video)), 'on those cameras\' own videos, not copies');

    // The file's order is what a loader that says nothing gets, as before.
    const inFileOrder = sessionEventScope([], 3, 0, true);
    check(eq([...inFileOrder.videos], [0]) && eq([...inFileOrder.keep], [1, 2]) && inFileOrder.cameras.get(2) === 2, 'without it: the file\'s order');

    // No video matches a camera (campy names every video `0.mp4`): none is a label, each stays put.
    const none = sessionEventScope([], 2, 0, true, new Map());
    check(none.videos.size === 0 && eq([...none.keep], [0, 1]) && none.cameras.size === 0, 'no match: every video only kept, on no camera');
    const zeros = [new Video({ filename: '0.mp4' }), new Video({ filename: '0.mp4' })];
    resetBehaviorState();
    const n = { name: 'n' };
    const zeroFile = asFile([rear], [bout(zeros[0], 10), bout(zeros[1], 300)], zeros);
    readBehaviorMetadata(n, null, zeroFile, none);
    check(exportBoutRows(n).length === 0, 'and shows none');
    const listed = new Map(zeroFile.videos.map(rec => [rec, new Video({ filename: '0.mp4' })]));
    const nOut = buildBehaviorEvents(SIO, [{ session: n, video: cams[0], videos: cams, identities: [a], listedVideos: listed }]);
    check(nOut.events.length === 2 && nOut.events[0].video === listed.get(zeroFile.videos[0]) && nOut.events[1].video === listed.get(zeroFile.videos[1]),
        'saved: each on the entry its poses have');

    // With a session: only a kept video the session has no place for goes by the loader.
    const named = sessionEventScope([{ camcorder_to_video_idx_map: { 0: 0 } }], 3, 0, true, new Map([[0, 0], [1, 1]]));
    check(eq([...named.videos], [0]) && eq([...named.keep], [1, 2]), 'with a session: the session still says what it reads');
    check(named.cameras.get(1) === 1 && named.cameras.size === 1, 'and a video only the loader matched goes back on that camera');
    const unmatched = sessionEventScope([{ camcorder_to_video_idx_map: { 0: 0 } }], 3, 0, true, new Map([[0, 0]]));
    check(unmatched.cameras === null, 'a kept video it did not match stays on no camera');
}

console.log('\n7. Renamed elsewhere');
{
    resetBehaviorState();
    const rear = addBehavior({ name: 'rear' }).behavior;
    setBehaviorDetector(rear.id, { conditions: [{ feature: 'n0 height · mean ±0.1 s', op: '>', value: 0.5 }], minDuration: 0.2 });
    const s = { name: 's' };
    labelRange(s, rear.id, 'a', '', 1, 2, true);
    const out = buildBehaviorEvents(SIO, [{ session: s, video: {}, identities: [] }]);
    const lucid = {};
    writeBehaviorSlpMetadata(lucid, s);
    check(lucid.ethogram.v === 2 && eq(lucid.ethogram.behaviors.map(b => b.name), ['rear']) && !lucid.ethogram.behaviors[0].color, 'lucid keeps only the detector');
    out.eventTypes[0].name = 'rearing';                        // sleap-app renames it, metadata untouched
    resetBehaviorState();
    const back = { name: 'b' };
    readBehaviorMetadata(back, JSON.parse(JSON.stringify(lucid)), asFile(out.eventTypes, out.events, []));
    check(behaviors().length === 1 && behaviors()[0].name === 'rearing' && !!behaviors()[0].detector, 'the new name, the same detector');
    check(behaviors()[0].id === rear.id, 'and the same id');
}

console.log('\n8. A label read through the old form, and both together');
{
    resetBehaviorState();
    const lucid = {
        ethogram: { v: 1, behaviors: [{ id: 'b1', name: 'rear', color: '#ffffff', kind: 'state' }] },
        behaviorLabels: { v: 1, fps: 30, bouts: [['b1', 'a', '', 1, 2, 1, 'm', '']] },
    };
    const s = { name: 's' };
    check(readBehaviorMetadata(s, lucid).kept === 1 && behaviorFps(s) === 30, 'a file saved before events still opens');
    const groom = new EventType({ name: 'groom' });
    const added = new UserEvent({ type: groom, video: {}, startFrame: 5, endFrame: 6, subject: new Identity({ name: 'a' }) });
    resetBehaviorState();
    const both = { name: 'both' };
    check(readBehaviorMetadata(both, lucid, asFile([groom], [added], [])).kept === 2 && behaviors().length === 2, 'and what another tool added to it is read too');
}

console.log(`\n${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
