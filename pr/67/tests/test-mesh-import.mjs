/**
 * test-mesh-import.mjs — STL and PLY reading for the 3D view's reference meshes.
 *
 * Every fixture is built here, byte by byte, so each claim has a known answer:
 * binary + ASCII STL (including a binary header that starts with "solid"),
 * PLY in all three encodings with colours, a quad face, a points-only scan,
 * and the malformed cases. Bad data must be dropped and counted, never drawn:
 * a NaN coordinate that reached the GPU would silently break the bounding
 * sphere and the "fit to scene".
 */
import {
    parseMesh, parseMeshAsync, detectMeshFormat, MeshParseError,
} from '../import-export/mesh-import.js';

let passed = 0, failed = 0;
function check(cond, msg) {
    if (cond) { passed++; } else { failed++; console.log('  FAIL: ' + msg); }
}
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const arr = (t) => Array.from(t);
function throwsParse(fn, re, msg) {
    try { fn(); check(false, msg + ' (did not throw)'); }
    catch (e) { check(e instanceof MeshParseError && (!re || re.test(e.message)), msg + ' (' + e.message + ')'); }
}
const enc = (s) => new TextEncoder().encode(s);

// --- fixtures ------------------------------------------------------------------

function binaryStl(tris, header) {
    const buf = new ArrayBuffer(84 + 50 * tris.length);
    const u8 = new Uint8Array(buf);
    u8.set(enc((header || 'binary stl').padEnd(80, ' ').slice(0, 80)), 0);
    const dv = new DataView(buf);
    dv.setUint32(80, tris.length, true);
    tris.forEach((t, i) => {
        const o = 84 + 50 * i;
        // stored normal deliberately zero: the reader recomputes it
        for (let k = 0; k < 9; k++) dv.setFloat32(o + 12 + 4 * k, t[k], true);
    });
    return u8;
}

/** Binary PLY writer for the tests: vertex x,y,z (vtype) + rgb uchar; faces list<ctype,itype>. */
function binaryPly(endian, verts, faces, opts) {
    const o = Object.assign({ vtype: 'float', ctype: 'uchar', itype: 'int', colors: true, extra: false }, opts || {});
    const le = endian === 'le';
    let head = 'ply\nformat binary_' + (le ? 'little' : 'big') + '_endian 1.0\ncomment test\n' +
        'element vertex ' + verts.length + '\n' +
        'property ' + o.vtype + ' x\nproperty ' + o.vtype + ' y\nproperty ' + o.vtype + ' z\n';
    if (o.colors) head += 'property uchar red\nproperty uchar green\nproperty uchar blue\n';
    head += 'element face ' + faces.length + '\nproperty list ' + o.ctype + ' ' + o.itype + ' vertex_indices\n';
    if (o.extra) head += 'element edge 1\nproperty int vertex1\nproperty int vertex2\n';
    head += 'end_header\n';
    const vsz = o.vtype === 'double' ? 8 : 4;
    const csz = o.ctype === 'uchar' ? 1 : 4;
    let size = enc(head).length + verts.length * (3 * vsz + (o.colors ? 3 : 0));
    for (const f of faces) size += csz + 4 * f.length;
    if (o.extra) size += 8;
    const buf = new ArrayBuffer(size);
    const u8 = new Uint8Array(buf);
    const hb = enc(head);
    u8.set(hb, 0);
    const dv = new DataView(buf);
    let off = hb.length;
    for (const v of verts) {
        for (let k = 0; k < 3; k++) {
            if (vsz === 8) dv.setFloat64(off, v[k], le); else dv.setFloat32(off, v[k], le);
            off += vsz;
        }
        if (o.colors) { for (let k = 0; k < 3; k++) dv.setUint8(off++, v[3 + k]); }
    }
    for (const f of faces) {
        if (csz === 1) dv.setUint8(off, f.length); else dv.setInt32(off, f.length, le);
        off += csz;
        for (const i of f) {
            if (o.itype === 'uint') dv.setUint32(off, i, le); else dv.setInt32(off, i, le);
            off += 4;
        }
    }
    if (o.extra) { dv.setInt32(off, 0, le); dv.setInt32(off + 4, 1, le); }
    return u8;
}

// A unit square in z=5 (two triangles), plus a triangle with a NaN vertex.
const SQUARE = [
    [0, 0, 5, 10, 0, 5, 10, 10, 5],
    [0, 0, 5, 10, 10, 5, 0, 10, 5],
];

// --- 1. binary STL -----------------------------------------------------------
{
    const r = parseMesh(binaryStl(SQUARE.concat([[0, 0, NaN, 1, 0, 0, 0, 1, 0]])));
    check(r.format === 'stl-binary', 'binary STL detected (' + r.format + ')');
    check(r.kind === 'mesh' && r.triangleCount === 2 && r.vertexCount === 6, 'two triangles kept, soup of 6 vertices');
    check(r.dropped.triangles === 1, 'the NaN triangle was dropped and counted');
    check(r.indices === null, 'a soup has no index');
    check(r.positions instanceof Float32Array && r.positions.length === 18, 'positions are a compact Float32Array');
    check(arr(r.positions).every(Number.isFinite), 'no non-finite coordinate reaches the arrays');
    check(eq(arr(r.normals.subarray(0, 3)), [0, 0, 1]), 'face normal recomputed (+Z for CCW from above)');
    check(eq(r.bbox, { min: [0, 0, 5], max: [10, 10, 5] }), 'bbox ' + JSON.stringify(r.bbox));
    // A binary file whose header starts with "solid" is still binary.
    const s = parseMesh(binaryStl(SQUARE, 'solid exported by a CAD tool'));
    check(s.format === 'stl-binary' && s.triangleCount === 2, 'a "solid" header with a binary-sized body is binary');
    throwsParse(() => parseMesh(binaryStl(SQUARE).slice(0, 84 + 60)), /not an STL|truncated/, 'a truncated binary STL is refused');
    throwsParse(() => parseMesh(binaryStl([[NaN, 0, 0, 1, 0, 0, 0, 1, 0]])), /finite/, 'an all-NaN STL is refused');
}

// --- 2. ASCII STL ------------------------------------------------------------
{
    const txt = 'solid sq\n' +
        ' facet normal 0 0 0\n  outer loop\n   vertex 0 0 5\n   vertex 10 0 5\n   vertex 10 10 5\n  endloop\n endfacet\n' +
        ' facet normal 0 0 1\r\n  outer loop\r\n   vertex 0 0 5\r\n   vertex 10 10 5\r\n   vertex 0 10 5\r\n  endloop\r\n endfacet\r\n' +
        ' facet normal 0 0 1\n  outer loop\n   vertex 0 0 nan\n   vertex 1 0 0\n   vertex 0 1 0\n  endloop\n endfacet\n' +
        ' facet normal 0 0 1\n  outer loop\n   vertex 20 0 0\n   vertex 30 0 0\n   vertex 30 10 0\n   vertex 20 10 0\n  endloop\n endfacet\n' +
        'endsolid sq\n';
    const r = parseMesh(enc(txt));
    check(r.format === 'stl-ascii', 'ASCII STL detected');
    check(r.triangleCount === 4, 'two triangles + a quad fan-triangulated into two (got ' + r.triangleCount + ')');
    check(r.dropped.triangles === 1, 'the nan facet dropped');
    check(arr(r.positions).every(Number.isFinite), 'no NaN reaches the arrays');
    throwsParse(() => parseMesh(enc('solid empty\nendsolid empty\n')), /no triangles/, 'an empty ASCII STL is refused');
    throwsParse(() => parseMesh(enc('solid x\nfacet normal 0 0 1\nouter loop\nvertex 1 2\n')), /fewer than 3/,
        'a short vertex line is refused');
}

// --- 3. PLY, three encodings ---------------------------------------------------
const VERTS = [
    [0, 0, 0, 255, 0, 0],
    [10, 0, 0, 0, 255, 0],
    [10, 10, 0, 0, 0, 255],
    [0, 10, 0, 10, 20, 30],
];
const FACES = [[0, 1, 2, 3]];   // one quad
function checkQuad(r, label) {
    check(r.kind === 'mesh' && r.vertexCount === 4 && r.triangleCount === 2, label + ': quad -> 2 triangles');
    check(r.indices instanceof Uint32Array && eq(arr(r.indices), [0, 1, 2, 0, 2, 3]), label + ': fan indices ' + arr(r.indices));
    check(r.colors instanceof Uint8Array && eq(arr(r.colors), [255, 0, 0, 0, 255, 0, 0, 0, 255, 10, 20, 30]), label + ': colours');
    check(eq(arr(r.positions), [0, 0, 0, 10, 0, 0, 10, 10, 0, 0, 10, 0]), label + ': positions');
    check(r.normals === null, label + ': no stored normals');
}
{
    const ascii = 'ply\r\nformat ascii 1.0\r\ncomment made by hand\r\nelement vertex 4\r\n' +
        'property float x\r\nproperty float y\r\nproperty float z\r\n' +
        'property uchar red\r\nproperty uchar green\r\nproperty uchar blue\r\n' +
        'element face 1\r\nproperty list uchar int vertex_indices\r\nend_header\r\n' +
        VERTS.map(v => v.join(' ')).join('\n') + '\n4 0 1 2 3\n';
    const ra = parseMesh(enc(ascii));
    check(ra.format === 'ply-ascii', 'ascii PLY detected');
    checkQuad(ra, 'ascii');
    const rl = parseMesh(binaryPly('le', VERTS, FACES));
    check(rl.format === 'ply-binary-le', 'little-endian PLY detected');
    checkQuad(rl, 'binary LE');
    const rb = parseMesh(binaryPly('be', VERTS, FACES, { vtype: 'double', ctype: 'int', itype: 'uint', extra: true }));
    check(rb.format === 'ply-binary-be', 'big-endian PLY detected');
    checkQuad(rb, 'binary BE (double xyz, int count, uint index, extra element)');
    // A negative control on endianness: the same bytes read the other way are not the same quad.
    const flipped = binaryPly('le', VERTS, FACES);
    const asBe = new Uint8Array(flipped);
    const hdrEnd = new TextDecoder().decode(asBe.slice(0, 400)).indexOf('end_header');
    const swapped = new TextEncoder().encode(new TextDecoder('latin1').decode(asBe.slice(0, hdrEnd))
        .replace('binary_little_endian', 'binary_big_endian   '));
    // Same header length (padded), so the body offsets do not move.
    const mangled = new Uint8Array(asBe);
    mangled.set(swapped, 0);
    let differs = true;
    try { const m = parseMesh(mangled); differs = !eq(arr(m.positions), arr(rl.positions)); } catch (e) { differs = true; }
    check(differs, 'negative control: reading little-endian bytes as big-endian does not give the same quad');
}

// --- 4. PLY with normals and float colours, bad faces dropped -------------------
{
    const ascii = 'ply\nformat ascii 1.0\nelement vertex 4\n' +
        'property double x\nproperty double y\nproperty double z\n' +
        'property float nx\nproperty float ny\nproperty float nz\n' +
        'property float red\nproperty float green\nproperty float blue\n' +
        'element face 4\nproperty list uchar uint vertex_index\nend_header\n' +
        '0 0 0 0 0 1 1 0 0\n1 0 0 0 0 1 0 1 0\n0 1 0 0 0 1 0 0 1\nnan 0 0 0 0 1 0.5 0.5 0.5\n' +
        '3 0 1 2\n3 0 1 9\n3 0 1 3\n2 0 1\n';
    const r = parseMesh(enc(ascii));
    check(r.triangleCount === 1, 'only the valid face survives (got ' + r.triangleCount + ')');
    check(r.dropped.triangles === 3, 'out-of-range, NaN-vertex and 2-vertex faces dropped and counted (' + r.dropped.triangles + ')');
    check(r.dropped.vertices === 1, 'the NaN vertex counted');
    check(r.normals instanceof Float32Array && eq(arr(r.normals.subarray(0, 3)), [0, 0, 1]), 'stored normals read');
    check(eq(arr(r.colors.subarray(0, 9)), [255, 0, 0, 0, 255, 0, 0, 0, 255]), 'float colours scaled to 0-255');
    check(arr(r.positions).every(Number.isFinite), 'the NaN vertex never reaches the positions');
    check(eq(r.bbox, { min: [0, 0, 0], max: [1, 1, 0] }), 'bbox ignores the bad vertex ' + JSON.stringify(r.bbox));
}

// --- 5. points-only PLY (a scan's point cloud) -----------------------------------
{
    const pts = [[1, 2, 3, 9, 9, 9], [NaN, 0, 0, 1, 1, 1], [4, 5, 6, 7, 7, 7]];
    const bytes = (() => {
        const head = 'ply\nformat binary_little_endian 1.0\nelement vertex 3\n' +
            'property float x\nproperty float y\nproperty float z\n' +
            'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n';
        const hb = enc(head);
        const buf = new ArrayBuffer(hb.length + 3 * 15);
        new Uint8Array(buf).set(hb, 0);
        const dv = new DataView(buf);
        let off = hb.length;
        for (const p of pts) {
            for (let k = 0; k < 3; k++) { dv.setFloat32(off, p[k], true); off += 4; }
            for (let k = 3; k < 6; k++) dv.setUint8(off++, p[k]);
        }
        return new Uint8Array(buf);
    })();
    const r = parseMesh(bytes);
    check(r.kind === 'points' && r.vertexCount === 2 && r.triangleCount === 0, 'vertices without faces come back as points');
    check(eq(arr(r.positions), [1, 2, 3, 4, 5, 6]), 'the NaN point is compacted away');
    check(eq(arr(r.colors), [9, 9, 9, 7, 7, 7]), 'colours follow their points');
    check(r.dropped.vertices === 1, 'and counted');
}

// --- 6. malformed files -------------------------------------------------------------
{
    throwsParse(() => parseMesh(enc('hello world, not a mesh at all, but long enough to have a header'.repeat(3))),
        /not an STL or PLY/, 'arbitrary bytes are refused');
    throwsParse(() => parseMesh(new Uint8Array(0)), /not an STL or PLY/, 'an empty file is refused');
    throwsParse(() => parseMesh(enc('ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\n')),
        /end_header/, 'a PLY without end_header is refused');
    throwsParse(() => parseMesh(enc('ply\nformat binary_middle_endian 1.0\nend_header\n')),
        /unknown PLY format/, 'an unknown PLY format is refused');
    throwsParse(() => parseMesh(enc('ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nend_header\n1 2\n')),
        /x, y, z/, 'a PLY without z is refused');
    throwsParse(() => parseMesh(binaryPly('le', VERTS, FACES).slice(0, -3)), /truncated/, 'a truncated binary PLY is refused');
    throwsParse(() => parseMesh(enc('ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n1 1 1\n')),
        /truncated/, 'a truncated ascii PLY is refused');
    throwsParse(() => parseMesh(enc('ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\nnan nan nan\n')),
        /finite/, 'a PLY whose only vertex is NaN is refused');
    throwsParse(() => detectMeshFormat(new Uint8Array(10), 'scan.obj'), /scan\.obj: not an STL or PLY/, 'the error names the file');
}

// --- 7. the async path runs the same parse and reports progress ---------------------
{
    const many = [];
    for (let i = 0; i < 25; i++) many.push([i, 0, 0, i + 1, 0, 0, i, 1, 0]);
    const bytes = binaryStl(many);
    const progress = [];
    let yields = 0;
    const r = await parseMeshAsync(bytes, { chunk: 10 }, (d, t, u) => progress.push([d, t, u]), async () => { yields++; });
    check(r.triangleCount === 25, 'async parse gives the same triangles');
    check(eq(progress, [[10, 25, 'triangles'], [20, 25, 'triangles']]), 'progress every chunk ' + JSON.stringify(progress));
    check(yields === 2, 'and yields between chunks');
    check(eq(arr(r.positions), arr(parseMesh(bytes).positions)), 'same bytes as the one-shot parse');
}

console.log(`\n${passed} passed, ${failed} failed`);
if (failed) process.exit(1);
