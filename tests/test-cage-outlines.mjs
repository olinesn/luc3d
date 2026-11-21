/**
 * test-cage-outlines.mjs — the per-view cage outline model and its on-disk form.
 *
 * The cage-scan Python tools read `metadata.lucid.cageOutlines` byte for byte
 * (`{version: 1, views: {<view>: [{id, name, points}]}}`), so the format is
 * pinned here exactly, along with the two things a lenient reader must never
 * do: let a NaN become a vertex, or write anything for an empty set.
 */
import {
    CageOutlineSet, OutlineTrace, serializeCageOutlines, restoreCageOutlines,
    nearestOutlineVertex, nearestOutlineEdge, CAGE_OUTLINES_VERSION, MIN_OUTLINE_POINTS,
} from '../pose/cage-outlines.js';

let passed = 0, failed = 0;
function check(cond, msg) {
    if (cond) { passed++; } else { failed++; console.log('  FAIL: ' + msg); }
}
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// --- 1. Empty sets write nothing -------------------------------------------
check(serializeCageOutlines(null) === null, 'null set serializes to null');
check(serializeCageOutlines(undefined) === null, 'undefined set serializes to null');
check(serializeCageOutlines(new CageOutlineSet()) === null, 'an empty set serializes to null (no default written)');
{
    const s = new CageOutlineSet();
    s.addPolygon('cam0', [[0, 0], [1, 0], [0, 1]]);
    s.clearView('cam0');
    check(serializeCageOutlines(s) === null, 'a set emptied by clearView serializes to null');
    check(s.isEmpty(), 'and reports isEmpty');
}

// --- 2. The exact on-disk format ---------------------------------------------
{
    const s = new CageOutlineSet();
    const a = s.addPolygon('Camera0', [[10, 20], [110.5, 20.25], [60, 90]]);
    const b = s.addPolygon('Camera1', [[1, 1], [2, 1], [2, 2], [1, 2]], 'lid');
    check(a && a.id === 1 && b && b.id === 2, 'ids are minted from 1 across views');
    const out = serializeCageOutlines(s);
    const expected = {
        version: 1,
        views: {
            Camera0: [{ id: 1, name: 'cage', points: [[10, 20], [110.5, 20.25], [60, 90]] }],
            Camera1: [{ id: 2, name: 'lid', points: [[1, 1], [2, 1], [2, 2], [1, 2]] }],
        },
    };
    check(eq(out, expected), 'format is {version:1, views:{view:[{id,name,points}]}} (got ' + JSON.stringify(out) + ')');
    check(CAGE_OUTLINES_VERSION === 1, 'version constant is 1');
    check(out.views.Camera0[0].points.length === 3, 'the first point is not repeated (closed implicitly)');
    // JSON round trip, as the .slp carries it.
    const back = restoreCageOutlines(JSON.parse(JSON.stringify(out)));
    check(eq(serializeCageOutlines(back), expected), 'JSON round trip is exact');
    const c = back.addPolygon('Camera0', [[5, 5], [6, 5], [6, 6]]);
    check(c.id === 3, 'restore advances the id counter past the file\'s ids (got ' + c.id + ')');
    // A copy, not the caller's arrays.
    const pts = [[0, 0], [4, 0], [0, 4]];
    const d = s.addPolygon('Camera2', pts);
    pts[0][0] = 99;
    check(d.points[0][0] === 0, 'addPolygon copies its points');
}

// --- 3. NaN and malformed input never become vertices ------------------------
{
    const s = new CageOutlineSet();
    check(s.addPolygon('v', [[0, 0], [NaN, 1], [1, 1]]) === null, 'a NaN point refuses the polygon');
    check(s.addPolygon('v', [[0, 0], [Infinity, 1], [1, 1]]) === null, 'an Infinity point refuses it');
    check(s.addPolygon('v', [[0, 0], [1, 1]]) === null, 'fewer than 3 points is not an outline');
    check(s.addPolygon('', [[0, 0], [1, 0], [0, 1]]) === null, 'an empty view name is refused');
    check(s.isEmpty(), 'nothing was added');

    const garbage = [null, 7, 'x', [], {}, { views: null }, { views: [] }, { views: 'a' },
        { views: { v: 'nope' } }, { views: { v: [null, 3, { points: 'x' }] } }];
    check(garbage.every(g => restoreCageOutlines(g).isEmpty()), 'garbage restores an empty set');

    const mixed = restoreCageOutlines({
        version: 1,
        views: {
            ok: [{ id: 4, name: 'cage', points: [[0, 0], [1, 0], [0, 1]] }],
            nan: [{ id: 5, points: [[0, 0], [null, 0], [0, 1]] }],
            short: [{ id: 6, points: [[0, 0], [1, 0]] }],
            dup: [{ id: 4, points: [[2, 2], [3, 2], [2, 3]] }, { points: [[5, 5], [6, 5], [5, 6]] }],
        },
    });
    const ser = serializeCageOutlines(mixed);
    check(eq(Object.keys(ser.views), ['ok', 'dup']), 'invalid polygons are dropped (got ' + JSON.stringify(Object.keys(ser.views)) + ')');
    const ids = Object.values(ser.views).flat().map(p => p.id);
    check(new Set(ids).size === ids.length, 'a duplicate id is renumbered so ids stay unique (' + JSON.stringify(ids) + ')');
    check(ser.views.ok[0].id === 4, 'a valid id is kept');
    check(ser.views.dup[0].name === 'cage', 'a missing name reads as "cage"');
    check(eq(ser.views.dup.map(p => p.points[0]), [[2, 2], [5, 5]]), 'file order is kept within a view');
}

// --- 4. Vertex edits ----------------------------------------------------------
{
    const s = new CageOutlineSet();
    const p = s.addPolygon('v', [[0, 0], [10, 0], [10, 10], [0, 10]]);
    check(s.moveVertex('v', p.id, 2, 12, 11) && eq(p.points[2], [12, 11]), 'moveVertex moves');
    check(!s.moveVertex('v', p.id, 2, NaN, 0), 'moveVertex refuses NaN');
    check(eq(p.points[2], [12, 11]), 'and leaves the vertex alone');
    check(!s.moveVertex('v', p.id, 9, 1, 1), 'moveVertex refuses an index past the end');
    check(s.insertVertex('v', p.id, 3, -1, 5) && p.points.length === 5 && eq(p.points[4], [-1, 5]),
        'insertVertex after the last vertex lands on the closing edge');
    check(s.insertVertex('v', p.id, 0, 5, -1) && eq(p.points[1], [5, -1]), 'insertVertex after 0 lands at 1');
    check(s.deleteVertex('v', p.id, 1) && p.points.length === 5, 'deleteVertex deletes');
    s.deleteVertex('v', p.id, 0);
    s.deleteVertex('v', p.id, 0);
    check(p.points.length === MIN_OUTLINE_POINTS, 'down to 3');
    check(!s.deleteVertex('v', p.id, 0) && p.points.length === 3, 'a vertex is never deleted below 3');
    check(s.vertexCount('v') === 3, 'vertexCount counts');
    const r = s.replaceView('v', [[1, 1], [2, 1], [1, 2]]);
    check(r && s.polygons('v').length === 1 && s.polygons('v')[0] === r, 'replaceView swaps the view\'s outline');
    check(s.replaceView('v', [[1, 1]]) === null && s.polygons('v')[0] === r,
        'a refused replace keeps the previous outline');
    check(s.removePolygon('v', r.id) && s.polygons('v').length === 0 && s.viewNames().length === 0,
        'removePolygon removes, and an empty view disappears');
}

// --- 5. Hit testing -------------------------------------------------------------
{
    const s = new CageOutlineSet();
    const p = s.addPolygon('v', [[0, 0], [100, 0], [100, 100], [0, 100]]);
    const hv = nearestOutlineVertex(s, 'v', 98, 3, 5);
    check(hv && hv.polyId === p.id && hv.index === 1, 'nearest vertex found');
    check(nearestOutlineVertex(s, 'v', 50, 50, 5) === null, 'no vertex far away');
    check(nearestOutlineVertex(s, 'other', 0, 0, 5) === null, 'another view has none');
    const he = nearestOutlineEdge(s, 'v', 50, 102, 5);
    check(he && he.index === 2 && eq(he.point, [50, 100]), 'nearest edge is the bottom one, foot of the perpendicular');
    const hc = nearestOutlineEdge(s, 'v', -2, 40, 5);
    check(hc && hc.index === 3 && eq(hc.point, [0, 40]), 'the closing edge (last -> first) is hit-tested');
    check(nearestOutlineEdge(s, 'v', 50, 50, 5) === null, 'no edge in the middle');
}

// --- 6. Tracing -----------------------------------------------------------------
{
    const t = new OutlineTrace('v');
    check(!t.canClose(), 'an empty trace cannot close');
    t.add(0, 0); t.add(10, 0);
    check(!t.add(NaN, 1) && t.points.length === 2, 'a NaN click is not a vertex');
    check(!t.canClose(), 'two points cannot close');
    t.add(10, 10);
    check(t.canClose(), 'three can');
    check(t.nearFirst(1, 1, 3) && !t.nearFirst(5, 5, 3), 'nearFirst tests the first vertex');
    check(eq(t.undo(), [10, 10]) && t.points.length === 2, 'undo removes the last vertex');
    t.undo(); t.undo();
    check(t.undo() === null, 'undo on an empty trace is a no-op');
}

console.log(`\n${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
