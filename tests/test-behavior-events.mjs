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
 *   - renamed elsewhere, a behavior keeps its detector.
 */

import { buildBehaviorEvents, fileEventsOf, sessionVideoIdxs } from '../import-export/behavior-events.js';
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
const SIO = { EventType, UserEvent, PredictedEvent, Identity, Track };

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
    const s0 = { name: 's0' }, s1 = { name: 's1' };
    readBehaviorMetadata(s0, {}, file, sessionVideoIdxs({ camcorder_to_video_idx_map: { 0: 0, 1: 1 } }));
    readBehaviorMetadata(s1, {}, file, sessionVideoIdxs({ camcorder_to_video_idx_map: { 0: 2, 1: 3 } }));
    check(eq(exportBoutRows(s0).map(r => r[3]), [1, 10]), 'an event on either camera is its session\'s');
    check(eq(exportBoutRows(s1).map(r => r[3]), [20]), 'and not the other session\'s');
    check(behaviors().length === 1, 'the catalog comes in once');
    check(sessionVideoIdxs({}) === null && sessionVideoIdxs(null) === null, 'a session naming no videos reads every row');
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
