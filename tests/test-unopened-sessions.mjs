/**
 * test-unopened-sessions.mjs — the rules of `import-export/unopened-sessions.js`, on
 * plain objects. The real file through the real reader and writer, read back with
 * Python, is `tests/e2e/lazy-keeps-other-sessions.mjs`.
 */
import {
    keepUnopenedSessions, keepUnopenedSessionsOfRead, readKeepsUnopenedSessions, seatMembersOnOpenVideos,
    instancePlacesForSave, unopenedSessionsForSave, identityEntriesOfOpenSession,
    renumberUnopenedMembers, hasMembersOnOpenVideos, withUnopenedSessions, unopenedSessionsNote,
} from '../import-export/unopened-sessions.js';

let passed = 0, failed = 0;
const check = (cond, msg, got) => {
    if (cond) { passed++; console.log('  ok   ' + msg); }
    else { failed++; console.log('  FAIL ' + msg + (got !== undefined ? ' — got ' + JSON.stringify(got) : '')); }
};
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// Stand-ins for the sleap-io.js classes the module builds: it reads nothing else off them.
class CameraGroup { constructor(o) { this.cameras = o.cameras || []; this.metadata = o.metadata || {}; } }
class FrameGroup { constructor(o) { Object.assign(this, o); this.metadata = o.metadata || {}; } }
class InstanceGroup {
    constructor(o) { this._instanceRefsByCamera = o.instanceRefsByCamera; this.score = o.score; this.identity = o.identity; this.instance3d = o.instance3d; this.metadata = o.metadata || {}; }
    get points() { return this.instance3d ? this.instance3d.points : undefined; }
}
class Instance3D { constructor(o) { Object.assign(this, o); } }
class PredictedInstance3D extends Instance3D {}
class RecordingSession {
    constructor(o) { this.cameraGroup = o.cameraGroup; this.frameGroupByFrameIdx = o.frameGroupByFrameIdx || new Map(); this.videoByCamera = new Map(); this.metadata = o.metadata || {}; }
    get frameGroups() { return this.frameGroupByFrameIdx; }
    addVideo(video, camera) { this.videoByCamera.set(camera, video); }
}
const SIO = { CameraGroup, FrameGroup, InstanceGroup, Instance3D, PredictedInstance3D, RecordingSession };

function fileSession(camNames, videos, frames) {
    const cameras = camNames.map(name => ({ name }));
    const s = new RecordingSession({ cameraGroup: new CameraGroup({ cameras, metadata: { rig: camNames[0] } }) });
    cameras.forEach((c, i) => { if (videos[i]) s.addVideo(videos[i], c); });
    for (const f of frames(cameras)) s.frameGroups.set(f.frameIdx, new FrameGroup(f));
    return s;
}

function fixture() {
    const videos = [{ filename: 'a.mp4' }, { filename: 'c.mp4' }, { filename: 'd.mp4' }, { filename: 'nothing.mp4' }];
    const identities = [{ name: 'A', metadata: {} }, { name: 'B', metadata: {} }, { name: 'C', metadata: { strain: 'x' } }];
    const open = fileSession(['a'], [videos[0]], () => []);
    const second = fileSession(['c', 'd', 'no_video', 'empty'], [videos[1], videos[2], null, videos[3]], ([c, d]) => [
        { frameIdx: 5, metadata: { note: 'five' }, instanceGroups: [
            new InstanceGroup({ instanceRefsByCamera: new Map([[c, [1, 0]], [d, [3, 1]]]), score: 0.25, identity: identities[1],
                instance3d: new Instance3D({ points: new Float64Array([1, 2, 3, 4, NaN, 6]), score: 0.5 }), metadata: { who: 'é ☃' } }),
            new InstanceGroup({ instanceRefsByCamera: new Map([[c, [1, 1]]]), identity: identities[2],
                instance3d: new PredictedInstance3D({ points: new Float64Array([7, 8, 9]), pointScores: new Float64Array([0.9]) }) }),
        ] },
        { frameIdx: 9, instanceGroups: [
            new InstanceGroup({ instanceRefsByCamera: new Map([[d, [4, 0]]]) }),
            (() => {
                const i3d = new Instance3D({ points: [[1, 1, 1], [null, 2, 2]] });   // boxed, as an old file reads
                i3d.lucidUnsmoothed = new Float64Array([0, 0, 0, 5, 5, 5]);
                return new InstanceGroup({ instanceRefsByCamera: new Map([[c, [2, 0]]]), instance3d: i3d });
            })(),
        ] },
        { frameIdx: 11, instanceGroups: [] },   // no writer makes one, and the writer leaves it out
    ]);
    const third = fileSession(['e'], [videos[1]], () => []);
    const labels = {
        sessions: [open, second, third], videos, identities,
        rawSessionsJson: [
            { metadata: { lucid: { sessionName: 'first', planes: [1], planeNodes: [2] } } },
            { calibration: { cam_0: { name: 'c', serial: 'S1' }, metadata: { rig: 'c' } }, camcorder_to_video_idx_map: { 0: 1 }, fg_start: 0, fg_end: 3,
              experiment: 'x1', metadata: { lucid: { sessionName: 'second', planes: [1], planeNodes: [2], planePlacements: { p: 1 }, hiddenCameras: ['c'] } } },
            { calibration: {}, frame_group_dicts: [], metadata: { lucid: { sessionName: 'third', planes: [99] } } },
        ],
        provenance: { recording: 'one', lucid_sessions: { second: { recording: 'two' } } },
        _lazyDataStore: { framesData: { video: new Float64Array([0, 1, 1, 2, 2]) }, videoIndexFor: id => id },
    };
    return labels;
}

function writerGraph(session) {
    return [...session.frameGroups.values()].map(fg => ({
        f: fg.frameIdx, m: fg.metadata,
        g: fg.instanceGroups.map(g => ({
            refs: g._instanceRefsByCamera ? [...g._instanceRefsByCamera].map(([c, r]) => [c.name, r[0], r[1]]) : null,
            score: g.score, id: g.identity ? g.identity.name : null, meta: g.metadata,
            pts: g.instance3d ? Array.from(g.instance3d.points) : null,
            ps: g.instance3d && g.instance3d.pointScores ? Array.from(g.instance3d.pointScores) : null,
            pred: g.instance3d instanceof PredictedInstance3D,
            s3d: g.instance3d ? g.instance3d.score : null,
            src: g.instance3d && g.instance3d.lucidUnsmoothed ? Array.from(g.instance3d.lucidUnsmoothed) : null,
        })),
    }));
}

const saveOpts = (over) => {
    const header = new Map();
    return Object.assign({
        SIO, identities: [],
        headerVideo: (id, video) => { if (!header.has(id)) header.set(id, { id, video }); return header.get(id); },
        outputRow: row => row + 100,
    }, over);
};

console.log('1. A file with one session');
{
    const one = fixture();
    one.sessions.length = 1;
    check(await keepUnopenedSessions(one, {}) === null, 'keeps nothing');
}

console.log('2. At load: columns, and the graphs go');
const labels = fixture();
const secondTyped = labels.sessions[1];
const kept = await keepUnopenedSessions(labels, { openVideoIds: new Set([0]) });
{
    check(labels.sessions.length === 1, 'the reader keeps only the first session');
    check(secondTyped.frameGroups.size === 1, 'the graph is drained (only the empty frame is left)', secondTyped.frameGroups.size);
    check(kept.sessions.length === 2, 'two sessions kept');
    const r = kept.sessions[0];
    check(['frameIdx', 'groupEnd', 'identity', 'score', 'flags', 'memberRow', 'memberInst', 'points3d'].every(k => ArrayBuffer.isView(r[k])), 'every per-row column is typed');
    check(eq(Array.from(r.frameIdx), [5, 9]), 'a frame with no group is left out, as the writer leaves it out', Array.from(r.frameIdx));
    check(r.name === 'second', 'named as the provenance keys it', r.name);
    check(eq(r.fileProvenance, { recording: 'two' }), 'its own provenance', r.fileProvenance);
    check(eq(kept.sessions[1].fileProvenance, { recording: 'one' }), 'a session with none of its own takes the file\'s', kept.sessions[1].fileProvenance);
    check(eq(r.videos.map(v => [v.camera, v.id]), [[0, 1], [1, 2], [3, 3]]), 'each camera with a video, by store id', r.videos.map(v => [v.camera, v.id]));
}

console.log('3. At save: every member goes where its row lands');
{
    const a = unopenedSessionsForSave(kept, saveOpts());
    const graph = writerGraph(a[0]);
    check(eq(graph[0].g[0].refs, [['c', 101, 0], ['d', 103, 1]]), 'rows are mapped, in-frame places kept', graph[0].g[0].refs);
    check(graph[0].g[0].score === 0.25 && graph[0].g[1].score === undefined, 'a group\'s score, or none');
    check(graph[0].g[0].s3d === 0.5 && graph[1].g[1].s3d === undefined, 'a 3D score, or none');
    check(eq(graph[0].m, { note: 'five' }) && eq(graph[1].m, {}), 'frame metadata');
    check(eq(graph[0].g[0].meta, { who: 'é ☃' }), 'group metadata, not ASCII', graph[0].g[0].meta);
    check(Number.isNaN(graph[0].g[0].pts[4]) && graph[0].g[0].pts[5] === 6 && graph[0].g[0].pts.length === 6, 'a missing 3D value stays missing');
    check(graph[0].g[1].pred && eq(graph[0].g[1].ps, [0.9]), 'predicted 3D with its scores', graph[0].g[1]);
    check(graph[1].g[0].pts === null, 'no 3D stays no 3D');
    check(eq(graph[1].g[1].pts.slice(0, 3), [1, 1, 1]) && Number.isNaN(graph[1].g[1].pts[3]), 'boxed 3D is flat now, null as NaN', graph[1].g[1].pts);
    check(eq(graph[1].g[1].src, [0, 0, 0, 5, 5, 5]), 'Temporal Smoothing\'s source', graph[1].g[1].src);
    check(eq(writerGraph(a[0]), graph), 'the frame groups can be walked twice');
    check(a[0].videoByCamera.size === 3, 'its cameras get their header videos', a[0].videoByCamera.size);

    // A second save with other numbers: the columns stayed in the file's numbering.
    const b = unopenedSessionsForSave(kept, saveOpts({ outputRow: row => row + 7 }));
    check(eq(writerGraph(b[0])[0].g[0].refs, [['c', 8, 0], ['d', 10, 1]]), 'the next save maps from the file\'s numbering again', writerGraph(b[0])[0].g[0].refs);

    // A row replaced by a correction, and one with nothing left.
    const c = unopenedSessionsForSave(kept, saveOpts({
        outputRow: row => (row === 1 || row === 3 ? undefined : row),
        movedRef: (row, inst) => (row === 1 && inst === 0 ? [55, 2] : null),
    }));
    const moved = writerGraph(c[0]);
    check(eq(moved[0].g[0].refs, [['c', 55, 2]]), 'a replaced row follows the correction; a member with nothing left goes', moved[0].g[0].refs);
    check(moved[0].g[1].refs === null && moved[0].g[1].pts !== null, 'a group with no member left keeps its 3D', moved[0].g[1]);
}

console.log('4. The session\'s own dict goes back as the file had it');
{
    const [s2, s3] = unopenedSessionsForSave(kept, saveOpts({ firstLucid: { planes: [1], planeNodes: [2] } }));
    const d = JSON.parse(s2.lucidFileDictJson);
    check(d.experiment === 'x1' && d.calibration.cam_0.serial === 'S1' && d.calibration.metadata.rig === 'c', 'keys LUC3D does not read', d);
    check(d.fg_start === undefined && d.fg_end === undefined && d.camcorder_to_video_idx_map === undefined && d.frame_group_dicts === undefined, 'not the keys this save fills in', d);
    check(eq(d.metadata.lucid.hiddenCameras, ['c']) && eq(d.metadata.lucid.planePlacements, { p: 1 }), 'per-session keys', d.metadata.lucid);
    check(eq(JSON.parse(s3.lucidFileDictJson).frame_group_dicts, undefined), 'an old file\'s inline groups are not copied');

    const edited = unopenedSessionsForSave(kept, saveOpts({ firstLucid: { planes: [1, 'edited'] } }));
    const dEdited = JSON.parse(edited[0].lucidFileDictJson).metadata.lucid;
    check(eq(dEdited.planes, [1, 'edited']) && dEdited.planeNodes === undefined, 'its copy of the project\'s planes follows the open session', dEdited);
    check(eq(dEdited.planePlacements, { p: 1 }), 'but not its own 2D placements', dEdited);
    check(eq(JSON.parse(edited[1].lucidFileDictJson).metadata.lucid.planes, [99]), 'planes that were never the first session\'s are left alone');
}

console.log('5. Identities: one animal stays one entry');
{
    const entries = kept.identities;
    const headerA = { name: 'A' }, headerB = { name: 'B' };
    const identities = [headerA, headerB];
    const headerOf = new Map([[0, headerA], [1, headerB]]);
    const identityOfEntry = identityEntriesOfOpenSession(kept, [{ id: 0, name: 'A' }, { id: 1, name: 'B' }], headerOf);
    const [s2] = unopenedSessionsForSave(kept, saveOpts({ identities, identityOfEntry }));
    const g = writerGraph(s2);
    check(s2 && [...s2.frameGroups.values()][0].instanceGroups[0].identity === headerB, 'an entry the open session lists: its header identity');
    check(identities.length === 3 && identities[2] === entries[2], 'an entry only a closed session names is added, whole', identities.map(i => i.name));
    check(g[0].g[1].id === 'C', 'and named', g[0].g[1].id);
    unopenedSessionsForSave(kept, saveOpts({ identities, identityOfEntry }));
    check(identities.length === 3, 'not twice');

    const renamed = [{ name: 'Mouse', id: 'x' }];
    const headerM = { name: 'Mouse' };
    const of = identityEntriesOfOpenSession(kept, [{ id: 1, name: 'Mouse' }], new Map([[1, headerM]]));
    const [s2b] = unopenedSessionsForSave(kept, saveOpts({ identities: renamed, identityOfEntry: of }));
    check([...s2b.frameGroups.values()][0].instanceGroups[0].identity === entries[1], 'renamed in the open session: the closed one keeps the name it had');
    const withMeta = { name: 'C', metadata: {} };
    identityEntriesOfOpenSession(kept, [{ id: 2, name: 'C' }], new Map([[2, withMeta]]));
    check(eq(withMeta.metadata, { strain: 'x' }), 'a shared entry keeps what else it held', withMeta);
}

console.log('6. A delete on a video the open session shares');
{
    const shared = fixture();
    const k = await keepUnopenedSessions(shared, { openVideoIds: new Set([1]) });   // c.mp4
    check(hasMembersOnOpenVideos(k), 'it is noticed');
    check(!hasMembersOnOpenVideos(kept), 'and not noticed when there is none');
    const r = k.sessions[0];
    check(eq(Array.from(r.membersOnOpenVideos), [0, 2, 4]), 'the members on rows of that video', Array.from(r.membersOnOpenVideos));
    // Rows 1 and 2 (c.mp4) hold instances 0-1 and 2-3. Delete instance 0.
    const framesData = { instance_id_start: new Float64Array([9, 0, 2, 4, 6]), instance_id_end: new Float64Array([9, 2, 4, 6, 8]) };
    const kill = new Uint8Array(9); kill[0] = 1;
    const survBefore = new Uint32Array(10);
    for (let i = 0; i < 9; i++) survBefore[i + 1] = survBefore[i] + (kill[i] ? 0 : 1);
    renumberUnopenedMembers(k, framesData, kill, survBefore);
    check(r.memberInst[0] === -1, 'a member on the deleted pose has nothing left', r.memberInst[0]);
    check(r.memberInst[2] === 0, 'the pose after it moved down', r.memberInst[2]);
    check(r.memberInst[4] === 0, 'another frame stays', r.memberInst[4]);
    check(r.memberInst[1] === 1, 'a member on another video is not touched', r.memberInst[1]);
    const g = writerGraph(unopenedSessionsForSave(k, saveOpts({ outputRow: row => row }))[0]);
    check(eq(g[0].g[0].refs, [['d', 3, 1]]) && eq(g[0].g[1].refs, [['c', 1, 0]]), 'the save follows', g[0]);
}

console.log('7. A file old enough to hold groups inline');
{
    const old = fixture();
    const fgRec = { frame_idx: 5, note: 'five', labeled_frame_by_camera: {}, instance_groups: [
        { camcorder_to_lf_and_inst_idx_map: {}, score: 0.25, points: [], identity_idx: 1, who: 'kept', metadata: { lucid: { identityId: 1 } } },
        { camcorder_to_lf_and_inst_idx_map: {}, also: [1, 2] },
    ] };
    old.rawSessionsJson[1].frame_group_dicts = [fgRec];
    old.sessions[1].frameGroups.get(5).metadata = {};
    old.sessions[1].frameGroups.get(5).instanceGroups[0].metadata = { lucid: { identityId: 1 } };
    old.sessions[1].frameGroups.get(5).instanceGroups[1].metadata = {};
    const k = await keepUnopenedSessions(old, {});
    const g = writerGraph(unopenedSessionsForSave(k, saveOpts())[0]);
    check(eq(g[0].m, { note: 'five' }), 'a frame\'s extra keys become its metadata', g[0].m);
    check(eq(g[0].g[0].meta, { who: 'kept', lucid: { identityId: 1 } }), 'a group\'s too, beside `metadata`', g[0].g[0].meta);
    check(eq(g[0].g[1].meta, { also: [1, 2] }), 'each group its own', g[0].g[1].meta);
}

console.log('8. Sparse video ids');
{
    const sparse = fixture();
    sparse._lazyDataStore = { framesData: { video: new Float64Array([0, 4, 7]) }, videoIndexFor: id => ({ 0: 0, 4: 1, 7: 2 }[id]) };
    const k = await keepUnopenedSessions(sparse, {});
    check(eq(k.sessions[0].videos.map(v => v.id), [4, 7, 3]), 'a video goes by its id in the frames table', k.sessions[0].videos.map(v => v.id));
}

console.log('9. Provenance order and the note');
{
    const open = { name: 'first', unopenedSessions: kept };
    const other = { name: 'other' };
    check(eq(withUnopenedSessions([open, other]).map(s => s.name), ['first', 'second', 'third', 'other']), 'each session then those it carries');
    check(eq(withUnopenedSessions([other]), [other]), 'a session with none');
    check(/Only the first of 3 sessions/.test(unopenedSessionsNote(3)), 'the status sentence');
}

// What `parseSlpViaSleapIO` gives for `fixture()`'s file: raw dicts plus the reader's objects.
function eagerRead(over) {
    const f = fixture();
    const typedIdentities = f.identities;
    const pose = x => ({ points: [[x, 1], [x, 2]] });
    return Object.assign({
        sessions: f.rawSessionsJson.map((raw, i) => Object.assign({}, raw, { _typedSession: f.sessions[i], _typedIdentities: typedIdentities })),
        videos: f.videos.map(v => ({ _typedVideo: v })),
        provenance: f.provenance,
        // Row 0 has no instance, so the read leaves it out.
        frames: [
            { fileRow: 1, videoIdx: 1, frameIdx: 5, instances: [pose(10), pose(20)] },
            { fileRow: 2, videoIdx: 1, frameIdx: 9, instances: [pose(30)] },
            { fileRow: 3, videoIdx: 2, frameIdx: 5, instances: [pose(40), pose(50)] },
            { fileRow: 4, videoIdx: 2, frameIdx: 9, instances: [pose(60)] },
        ],
    }, over);
}
class Pose {
    constructor(xy) { this.xy = xy; }
    get numNodes() { return this.xy.length; }
    hasPoint(n) { return !!this.xy[n]; }
    getX(n) { return this.xy[n][0]; }
    getY(n) { return this.xy[n][1]; }
}

console.log('10. An eager read');
{
    check(!readKeepsUnopenedSessions(eagerRead({ sessions: [{ _typedSession: {} }] })), 'one session: nothing to keep');
    check(!readKeepsUnopenedSessions(eagerRead({ sessions: [{ calibration: {} }, { calibration: {} }] })), 'the raw worker\'s read has no typed sessions');
    check(readKeepsUnopenedSessions(eagerRead()), 'the typed reader\'s has');

    const read = eagerRead();
    const k = await keepUnopenedSessionsOfRead(read, { videoIdByCam: new Map([['a', 0]]) });
    check(k.eager && k.eager.videoIdByCam.get('a') === 0 && k.sessions.length === 2, 'kept as an eager read');
    check(read.sessions.length === 3, 'its own session list stays', read.sessions.length);
    check(eq(k.sessions[0].videos.map(v => v.id), [1, 2, 3]), 'a video goes by its place in the list', k.sessions[0].videos.map(v => v.id));
    const [s2] = unopenedSessionsForSave(k, saveOpts({ firstLucid: { planes: [1, 'edited'] } }));
    const d = JSON.parse(s2.lucidFileDictJson);
    check(d._typedSession === undefined && d._typedIdentities === undefined, 'the reader\'s objects stay out of the saved dict', Object.keys(d));
    check(eq(d.metadata.lucid.planes, [1, 'edited']), 'planes that matched follow the open session', d.metadata.lucid.planes);

    const foreign = await keepUnopenedSessionsOfRead(eagerRead(), { videoIdByCam: new Map(), keepOwnPlanes: true });
    const [f2] = unopenedSessionsForSave(foreign, saveOpts({ firstLucid: { planes: ['the open project'] } }));
    check(eq(JSON.parse(f2.lucidFileDictJson).metadata.lucid.planes, [1]), 'planes of another project stay as the file has them');

    const lazy = await keepUnopenedSessions(fixture(), {});
    check(instancePlacesForSave(lazy) === null, 'a lazy open has no poses to place');
    check(instancePlacesForSave(k) === null, 'nor does an eager read with nothing shared');
}

console.log('11. An eager read: a member on a video the open session has too');
{
    const read = eagerRead();
    const k = await keepUnopenedSessionsOfRead(read, { videoIdByCam: new Map([['open', 1]]) });
    const r = k.sessions[0];
    check(eq(Array.from(r.membersOnOpenVideos), [0, 2, 4]), 'found by the file row\'s video', Array.from(r.membersOnOpenVideos));
    // The open session lists the frame its own way; row 2's pose was dropped as a copy of
    // one that has more nodes, and that one stands for it.
    const frame5 = [new Pose([[20.2, 1], [20, 2]]), new Pose([[10, 1], [10, 2]]), new Pose([[10.3, 1], [10.3, 2]])];
    const frame9 = [new Pose([[30, 1], null])];
    seatMembersOnOpenVideos(k, read.frames, frame => (frame.frameIdx === 5 ? frame5 : frame9));
    const seated = k.eager.posesOfRows;
    check(seated.get(1)[0] === frame5[1], 'the closest pose, not the first that is near', seated.get(1)[0] && seated.get(1)[0].xy);
    check(seated.get(1)[1] === frame5[0], 'each place its own', seated.get(1)[1] && seated.get(1)[1].xy);
    check(seated.get(2)[0] === frame9[0], 'a pose matching on the nodes both have stands for it');
    check(!seated.has(3), 'rows on another video are not looked at');
    const places = instancePlacesForSave(k);
    check(places.size === 3 && places.has(frame5[0]) && !places.has(frame5[2]), 'a save asks only for those poses', places.size);
    check([...places.values()].every(v => v === null), 'nothing placed yet');
    // Nothing near: the member has no pose, and a save drops it.
    const k2 = await keepUnopenedSessionsOfRead(eagerRead(), { videoIdByCam: new Map([['open', 1]]) });
    seatMembersOnOpenVideos(k2, read.frames, () => [new Pose([[500, 500], [500, 500]])]);
    check(k2.eager.posesOfRows.size === 0, 'a pose nothing matches is left unseated', k2.eager.posesOfRows.size);
}

console.log('12. An eager read: a node edit');
{
    const k = await keepUnopenedSessionsOfRead(eagerRead(), { videoIdByCam: new Map() });
    const r = k.sessions[0];
    const groupsOf = () => writerGraph(unopenedSessionsForSave(k, saveOpts())[0]).flatMap(fg => fg.g);
    const before = groupsOf();
    check(eq(before[0].pts.slice(0, 3), [1, 2, 3]) && before[1].pts.length === 3 && before[2].pts === null && before[3].pts.length === 6, 'as read', before.map(g => g.pts && g.pts.length));
    k.eager.addNode();
    const added = groupsOf();
    check(added[0].pts.length === 9 && eq(added[0].pts.slice(0, 3), [1, 2, 3]) && added[0].pts.slice(6).every(Number.isNaN), 'an added node is a NaN row at the end', added[0].pts);
    check(added[1].pts.length === 6 && eq(added[1].ps.slice(0, 1), [0.9]) && Number.isNaN(added[1].ps[1]), 'a predicted group\'s scores grow too', added[1].ps);
    check(added[2].pts === null, 'a group with no 3D gets none');
    check(added[3].pts.length === 9 && eq(added[3].src.slice(0, 6), [0, 0, 0, 5, 5, 5]) && added[3].src.slice(6).every(Number.isNaN), 'Temporal Smoothing\'s source too', added[3].src);

    k.eager.removeNode(0);
    const removed = groupsOf();
    check(removed[0].pts.length === 6 && removed[0].pts[0] === 4 && Number.isNaN(removed[0].pts[1]) && removed[0].pts[2] === 6, 'a removed node\'s row goes, the next takes its index', removed[0].pts);
    check(removed[3].src.length === 6 && eq(removed[3].src.slice(0, 3), [5, 5, 5]), 'from the source too', removed[3].src);
    check(eq(Array.from(r.memberRow), [1, 3, 1, 4, 2]), 'members are not touched', Array.from(r.memberRow));

    // A hidden-node list goes by node index.
    const withHidden = eagerRead();
    withHidden.sessions[1]._typedSession.frameGroups.get(5).instanceGroups[0].metadata = { lucid: { instanceMeta: { c: { nulledNodes: [0, 2], modified: true } } } };
    const h = await keepUnopenedSessionsOfRead(withHidden, { videoIdByCam: new Map() });
    h.eager.removeNode(0);
    const meta = writerGraph(unopenedSessionsForSave(h, saveOpts())[0])[0].g[0].meta;
    check(eq(meta.lucid.instanceMeta.c, { nulledNodes: [1], modified: true }), 'a hidden node\'s index follows', meta);

    // Its copy of the skeleton follows the open session's, when it was the same.
    const sk = { name: 'sk', nodes: ['a', 'b'], edges: [] };
    const same = eagerRead();
    same.sessions[0].metadata.lucid.skeleton = sk;
    same.sessions[1].metadata.lucid.skeleton = JSON.parse(JSON.stringify(sk));
    same.sessions[2].metadata.lucid.skeleton = { name: 'other', nodes: ['x'], edges: [] };
    const fk = await keepUnopenedSessionsOfRead(same, { videoIdByCam: new Map() });
    const saved = unopenedSessionsForSave(fk, saveOpts({ firstLucid: { skeleton: { name: 'sk', nodes: ['a', 'b', 'new'], edges: [] } } }));
    check(eq(JSON.parse(saved[0].lucidFileDictJson).metadata.lucid.skeleton.nodes, ['a', 'b', 'new']), 'a copy of the skeleton follows an edit');
    check(eq(JSON.parse(saved[1].lucidFileDictJson).metadata.lucid.skeleton.nodes, ['x']), 'another skeleton is left alone');
    const lazyDicts = fixture();
    lazyDicts.rawSessionsJson[0].metadata.lucid.skeleton = sk;
    lazyDicts.rawSessionsJson[1].metadata.lucid.skeleton = JSON.parse(JSON.stringify(sk));
    const lk = await keepUnopenedSessions(lazyDicts, {});
    const [ls] = unopenedSessionsForSave(lk, saveOpts({ firstLucid: { skeleton: { name: 'sk', nodes: ['a', 'b', 'new'], edges: [] } } }));
    check(eq(JSON.parse(ls.lucidFileDictJson).metadata.lucid.skeleton.nodes, ['a', 'b']), 'a lazy open keeps the file\'s: its poses are not re-shaped');
}

console.log(`\ntest-unopened-sessions: ${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
