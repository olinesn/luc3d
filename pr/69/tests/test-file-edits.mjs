/**
 * test-file-edits.mjs — import-export/file-edits.js: a project file another tool edited
 * reads what the file says; one nobody edited reads exactly as before.
 *
 * Run:  node tests/test-file-edits.mjs
 */
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');
const {
    reconcileSessionTracks, lazyTrackSlotMap, remapFrameIdentityTrackSlots,
    reconcileSessionIdentities, typedSessionGroups, applyMovedGroupIdentities,
} = await import(pathToFileURL(path.join(ROOT, 'import-export', 'file-edits.js')).href);

let passed = 0, failed = 0;
function ok(cond, msg) {
    if (cond) passed++;
    else { failed++; console.error('  ✗ ' + msg); }
}
const same = (a, b, msg) => ok(JSON.stringify(a) === JSON.stringify(b), `${msg}: ${JSON.stringify(a)} != ${JSON.stringify(b)}`);
function group(name) { console.log('\n• ' + name); }

group('reconcileSessionTracks: a file nobody edited');
{
    const r = reconcileSessionTracks(['a', 'b'], ['a', 'b'], new Set([0, 1]));
    same(r, { names: ['a', 'b'], slotMap: null, edited: false }, 'same lists');
    ok(!reconcileSessionTracks(['b', 'a'], ['a', 'b', 'c'], new Set([0, 1])).edited,
        "another session's names and another order are not an edit");
    ok(!reconcileSessionTracks(['a', 'b'], ['a', 'b', 'a', 'b'], new Set([0, 3])).edited,
        "a streamed file's repeated names are not an edit");
    ok(!reconcileSessionTracks(['a', 'empty'], ['a', 'empty'], new Set([0])).edited,
        'a track with no detections is not an edit');
    same(reconcileSessionTracks(['a'], ['a'], new Set([0, -1, 4294967295])).names, ['a'],
        'no track (-1, or read as unsigned) names nothing');
}

group('reconcileSessionTracks: edits');
{
    let r = reconcileSessionTracks(['t0', 't1', 't2'], ['t0', 'renamed', 't2'], new Set([0, 1, 2]));
    same(r, { names: ['t0', 'renamed', 't2'], slotMap: null, edited: true }, 'a rename keeps its slot');

    r = reconcileSessionTracks(['t0', 't1', 't2'], ['x', 't1', 'y'], new Set([0, 1, 2]));
    same(r.names, ['x', 't1', 'y'], 'two renames, in order');

    r = reconcileSessionTracks(['t0'], ['t0', 'new'], new Set([0, 1]));
    same(r, { names: ['t0', 'new'], slotMap: null, edited: true }, 'a new track goes at the end');

    r = reconcileSessionTracks(['t0'], ['t0', 'other_session'], new Set([0]));
    ok(!r.edited, "a new name this session's detections do not use stays out");

    r = reconcileSessionTracks(['t0', 'gone', 't2'], ['t0', 't2'], new Set([0, 1]));
    same(r, { names: ['t0', 't2'], slotMap: [0, -1, 1], edited: true }, 'a deleted track drops its slot');

    r = reconcileSessionTracks(['t0', 'gone', 't2'], ['t0', 't2', 'a', 'b'], new Set([0, 1, 2, 3]));
    same(r.names, ['t0', 'a', 't2', 'b'], 'more new names than vacated slots');

    r = reconcileSessionTracks(['t0'], [], new Set());
    same(r.names, [], 'every track deleted');
}

group('lazyTrackSlotMap');
{
    same(lazyTrackSlotMap(['a', 'b'], ['a', 'b']), null, 'same lists');
    same(lazyTrackSlotMap(['a', 'b'], ['a', 'renamed']), null, 'a rename stays put');
    same(lazyTrackSlotMap(['a', 'gone', 'c'], ['a', 'c']), [0, -1, 1], 'a deletion shifts the rest');
    same(lazyTrackSlotMap(['b', 'a'], ['a', 'b']), [1, 0], 'another order');
    same(lazyTrackSlotMap(['a', 'b'], ['a', 'b', 'new']), null, 'a new name moves nothing');
    same(lazyTrackSlotMap(['a', 'a', 'b'], ['a', 'a', 'b']), null, 'a repeated name stays in its slot');
    same(lazyTrackSlotMap(['b', 'a', 'a'], ['a', 'a', 'b']), [2, 1, 0], 'and moves only when it must');
}

group('remapFrameIdentityTrackSlots');
{
    const lucid = {
        frameIdentityMap: [['1:cam:a:0', 7], ['2:cam:1', 8], ['3:cam:2', 9], [12345, 3]],
        frameIdentityMapDropped: ['4:cam:1', '5:cam:2'],
    };
    remapFrameIdentityTrackSlots(lucid, [0, -1, 1]);
    same(lucid.frameIdentityMap, [['1:cam:a:0', 7], ['3:cam:1', 9], [12345, 3]], 'entries');
    same(lucid.frameIdentityMapDropped, ['5:cam:1'], 'dropped keys');
    const none = {};
    remapFrameIdentityTrackSlots(none, [0]);
    same(none, {}, 'a session with neither');
}

const G = (identityId, name) => ({
    metadata: identityId == null ? {} : { lucid: { identityId } },
    identity: name == null ? null : { name, color: name === 'colored' ? '#123456' : undefined },
});
const LIST = [{ name: 'id_0', color: '#ff0000' }, { name: 'id_1' }, { name: 'id_2' }];
const names = (r) => r.identities.map((i) => i.name);

group('reconcileSessionIdentities: a file nobody edited');
{
    ok(reconcileSessionIdentities(LIST, [G(0, 'id_0'), G(1, 'id_1'), G(-1, null)]) === null, 'agreeing');
    ok(reconcileSessionIdentities(LIST, []) === null, 'no groups');
    ok(reconcileSessionIdentities(LIST, [G(null, null)]) === null, 'a group with neither');
}

group('reconcileSessionIdentities: edits');
{
    let g = [G(0, 'mouse_A'), G(0, 'mouse_A'), G(1, 'id_1'), G(2, 'id_2')];
    let r = reconcileSessionIdentities(LIST, g);
    same(names(r), ['mouse_A', 'id_1', 'id_2'], 'a rename keeps its slot');
    same(r.identities[0].color, '#ff0000', 'and its colour');
    same(g.map(r.identityOf), [0, 0, 1, 2], 'ids stay');

    g = [G(0, 'id_1'), G(1, 'id_0'), G(0, 'id_0'), G(1, 'id_1')];
    r = reconcileSessionIdentities(LIST, g);
    same(names(r), ['id_0', 'id_1', 'id_2'], 'a swap leaves the list');
    same(g.map(r.identityOf), [1, 0, 0, 1], 'and moves the groups');

    g = [G(0, 'new'), G(0, 'id_0'), G(1, 'id_1')];
    r = reconcileSessionIdentities(LIST, g);
    same(names(r), ['id_0', 'id_1', 'id_2', 'new'], 'a new name beside one still in use is added');
    same(g.map(r.identityOf), [3, 0, 1], 'and takes its groups');

    g = [G(0, 'colored')];
    r = reconcileSessionIdentities([{ name: 'a' }, { name: 'b' }], [G(0, 'b'), ...g]);
    same(names(r), ['colored', 'b'], 'a vacated slot goes to the name its groups moved to');

    g = [G(-1, 'colored'), G(0, 'id_0')];
    r = reconcileSessionIdentities(LIST, g);
    same(r.identities[3], { name: 'colored', color: '#123456' }, 'a group new to LUC3D brings its colour');

    g = [G(0, null), G(1, 'id_1')];
    r = reconcileSessionIdentities(LIST, g);
    same(g.map(r.identityOf), [-1, 1], 'a cleared identity is cleared');

    g = [G(0, 'two'), G(0, 'two'), G(1, 'one'), G(0, 'one')];
    r = reconcileSessionIdentities([{ name: 'a' }, { name: 'b' }], g);
    same(names(r), ['two', 'one'], 'a bigger move takes the slot first');

    g = [G(5, 'id_0'), G(0, 'id_0')];
    r = reconcileSessionIdentities(LIST, g);
    same(g.map(r.identityOf), [0, 0], 'an id past the list goes by name');

    r = reconcileSessionIdentities([{ name: 'same' }, { name: 'same' }], [G(1, 'same'), G(0, 'x')]);
    same(r.identityOf(G(1, 'same')), 1, 'of two with one name, a group keeps its own');
}

group('typedSessionGroups + applyMovedGroupIdentities');
{
    const session = { frameGroups: new Map([[0, { instanceGroups: [1, 2] }], [1, {}], [2, { instanceGroups: [3] }]]) };
    const walked = typedSessionGroups(session);
    same([...walked], [1, 2, 3], 'walks every group');
    same([...walked], [1, 2, 3], 'and can walk again');

    const log = [];
    const target = {
        setFrameIdentity: (...a) => log.push(['set', ...a]),
        deleteFrameIdentity: (...a) => log.push(['del', ...a]),
    };
    applyMovedGroupIdentities(target, [
        [7, { identityId: 2, instances: new Map([['a', { trackIdx: 1 }], ['b', { trackIdx: null }], ['c', null]]) }],
        [8, { identityId: -1, instances: new Map([['a', { trackIdx: 0 }]]) }],
    ]);
    same(log, [['set', 7, 'a', 1, 2], ['del', 8, 'a', 0]], 'sets or clears each tracked member');
}

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
