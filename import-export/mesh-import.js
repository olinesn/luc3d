// import-export/mesh-import.js — STL and PLY files, read into typed arrays for
// the 3D view's reference meshes (File ▸ Load Reference Mesh).
//
// DOM-free and import-free, so Node tests load it directly. The UI half (file
// picking, the progress card, the 3D objects) is `ui/reference-meshes.js`.
//
// What it reads:
//   * STL, binary and ASCII. Binary is recognised by its size (84 + 50·n bytes),
//     not by the header, because plenty of binary writers start the header with
//     "solid" too. Facet normals are recomputed from the vertices (the stored
//     ones are often zero), so a triangle soup comes back flat-shaded.
//   * PLY, ascii / binary_little_endian / binary_big_endian: vertex x, y, z
//     (any numeric type), optional nx/ny/nz and red/green/blue (uchar, or
//     float 0..1), faces as `list <count type> <index type>` named
//     vertex_indices or vertex_index, polygons fan-triangulated. Other elements
//     and properties are read past and ignored. A PLY with vertices and no
//     faces comes back as POINTS (a scan's point cloud).
//
// Everything lands in typed arrays (positions Float32Array, indices
// Uint32Array, colours Uint8Array): a 2M-triangle scan must not become millions
// of boxed JS arrays in the renderer's capped heap (CLAUDE.md, memory rule).
//
// Bad data is dropped and COUNTED, never drawn: a triangle with a non-finite
// vertex or an index out of range is skipped, a non-finite point is skipped.
// A file that cannot be read at all throws `MeshParseError` with a reason.
//
// The parse is a generator (`parseMeshSteps`), so the UI can yield to the
// page between chunks of a big file (`parseMeshAsync`) while tests and small
// files drain it in one go (`parseMesh`); both run the same code.

export class MeshParseError extends Error {
    constructor(message) { super(message); this.name = 'MeshParseError'; }
}

/** Default rows (triangles or vertices) between progress yields. */
export const MESH_PARSE_CHUNK = 250000;

function toBytes(input) {
    if (input instanceof Uint8Array) return input;
    if (input instanceof ArrayBuffer) return new Uint8Array(input);
    if (ArrayBuffer.isView(input)) return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
    throw new MeshParseError('not a file buffer');
}

function latin1(bytes, start, end) {
    let s = '';
    const stop = Math.min(end, bytes.length);
    for (let i = start; i < stop; i += 8192) {
        s += String.fromCharCode.apply(null, bytes.subarray(i, Math.min(i + 8192, stop)));
    }
    return s;
}

/**
 * Which format the bytes are. `name` (the file name) breaks no tie the bytes
 * can decide; it is used only to word the error for an unknown file.
 * @returns {'stl-binary'|'stl-ascii'|'ply'}
 */
export function detectMeshFormat(input, name) {
    const bytes = toBytes(input);
    const head = latin1(bytes, 0, 512);
    if (/^ply\r?\n/.test(head)) return 'ply';
    if (bytes.length >= 84) {
        const n = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(80, true);
        if (84 + 50 * n === bytes.length) return 'stl-binary';
    }
    if (/^\s*solid\b/.test(head)) return 'stl-ascii';
    if (bytes.length >= 84) {
        const n = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(80, true);
        // Trailing bytes after a complete binary body (some writers pad).
        if (n > 0 && 84 + 50 * n <= bytes.length) return 'stl-binary';
    }
    throw new MeshParseError((name ? name + ': ' : '') + 'not an STL or PLY file');
}

// --- growable typed arrays (the ASCII paths and fan triangulation) ---------

class Grow {
    constructor(Ctor, cap) { this.Ctor = Ctor; this.a = new Ctor(Math.max(16, cap | 0)); this.n = 0; }
    push3(x, y, z) {
        if (this.n + 3 > this.a.length) this._grow(this.n + 3);
        this.a[this.n++] = x; this.a[this.n++] = y; this.a[this.n++] = z;
    }
    _grow(min) {
        let cap = this.a.length * 2;
        while (cap < min) cap *= 2;
        const b = new this.Ctor(cap); b.set(this.a.subarray(0, this.n)); this.a = b;
    }
    result() { return this.n === this.a.length ? this.a : this.a.slice(0, this.n); }
}

function bboxOf(positions, count) {
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    let any = false;
    for (let i = 0; i < count; i++) {
        const x = positions[3 * i], y = positions[3 * i + 1], z = positions[3 * i + 2];
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
        any = true;
        if (x < min[0]) min[0] = x; if (x > max[0]) max[0] = x;
        if (y < min[1]) min[1] = y; if (y > max[1]) max[1] = y;
        if (z < min[2]) min[2] = z; if (z > max[2]) max[2] = z;
    }
    return any ? { min: min, max: max } : null;
}

function faceNormalInto(p, o, out) {
    const ax = p[o + 3] - p[o], ay = p[o + 4] - p[o + 1], az = p[o + 5] - p[o + 2];
    const bx = p[o + 6] - p[o], by = p[o + 7] - p[o + 1], bz = p[o + 8] - p[o + 2];
    let nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
    const len = Math.hypot(nx, ny, nz);
    if (len > 0) { nx /= len; ny /= len; nz /= len; } else { nx = 0; ny = 0; nz = 0; }
    for (let k = 0; k < 3; k++) { out[o + 3 * k] = nx; out[o + 3 * k + 1] = ny; out[o + 3 * k + 2] = nz; }
}

function finishSoup(format, positions, triCount, dropped) {
    // A triangle soup: 3 vertices per triangle, no index, one normal per face.
    const normals = new Float32Array(positions.length);
    for (let t = 0; t < triCount; t++) faceNormalInto(positions, 9 * t, normals);
    return {
        format: format, kind: 'mesh',
        positions: positions, indices: null, normals: normals, colors: null,
        vertexCount: 3 * triCount, triangleCount: triCount,
        bbox: bboxOf(positions, 3 * triCount),
        dropped: { triangles: dropped, vertices: 0 },
    };
}

// --- STL ---------------------------------------------------------------------

function* parseStlBinary(bytes, chunk) {
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const n = dv.getUint32(80, true);
    if (84 + 50 * n > bytes.length) throw new MeshParseError('truncated binary STL');
    if (n === 0) throw new MeshParseError('the STL has no triangles');
    let positions = new Float32Array(9 * n);
    // Normals in the same pass, so a big file never blocks in a second loop.
    let normals = new Float32Array(9 * n);
    let kept = 0, dropped = 0;
    for (let t = 0; t < n; t++) {
        const off = 84 + 50 * t + 12;  // skip the stored normal
        let ok = true;
        const o = 9 * kept;
        for (let k = 0; k < 9; k++) {
            const v = dv.getFloat32(off + 4 * k, true);
            if (!Number.isFinite(v)) ok = false;
            positions[o + k] = v;
        }
        if (ok) { faceNormalInto(positions, o, normals); kept++; } else dropped++;
        if ((t + 1) % chunk === 0) yield { done: t + 1, total: n, unit: 'triangles' };
    }
    if (!kept) throw new MeshParseError('no triangle in the STL has finite coordinates');
    if (kept < n) { positions = positions.slice(0, 9 * kept); normals = normals.slice(0, 9 * kept); }
    return {
        format: 'stl-binary', kind: 'mesh',
        positions: positions, indices: null, normals: normals, colors: null,
        vertexCount: 3 * kept, triangleCount: kept,
        bbox: bboxOf(positions, 3 * kept),
        dropped: { triangles: dropped, vertices: 0 },
    };
}

function* parseStlAscii(bytes, chunk) {
    const text = latin1(bytes, 0, bytes.length);
    const pos = new Grow(Float32Array, 9 * 1024);
    let facet = [];   // the current loop's vertices, flat
    let kept = 0, dropped = 0, lines = 0, inFacet = false;
    let i = 0;
    const total = text.length;
    while (i < total) {
        let nl = text.indexOf('\n', i);
        if (nl < 0) nl = total;
        const line = text.slice(i, nl).trim();
        i = nl + 1;
        if (!line) continue;
        if (line.startsWith('vertex')) {
            const parts = line.split(/\s+/);
            if (parts.length < 4) throw new MeshParseError('ASCII STL: a vertex line has fewer than 3 numbers');
            facet.push(Number(parts[1]), Number(parts[2]), Number(parts[3]));
        } else if (line.startsWith('facet')) {
            facet = []; inFacet = true;
        } else if (line.startsWith('endfacet')) {
            if (inFacet) {
                const nv = facet.length / 3;
                if (nv < 3) dropped++;
                for (let k = 1; k + 1 < nv; k++) {
                    const tri = [0, k, k + 1];
                    let ok = true;
                    for (const v of tri) {
                        if (!Number.isFinite(facet[3 * v]) || !Number.isFinite(facet[3 * v + 1]) ||
                            !Number.isFinite(facet[3 * v + 2])) ok = false;
                    }
                    if (!ok) { dropped++; continue; }
                    for (const v of tri) pos.push3(facet[3 * v], facet[3 * v + 1], facet[3 * v + 2]);
                    kept++;
                }
            }
            facet = []; inFacet = false;
        }
        if (++lines % chunk === 0) yield { done: i, total: total, unit: 'bytes' };
    }
    if (!kept) throw new MeshParseError('the ASCII STL has no triangles');
    return finishSoup('stl-ascii', pos.result(), kept, dropped);
}

// --- PLY -----------------------------------------------------------------------

const PLY_TYPES = {
    char: 'i1', int8: 'i1', uchar: 'u1', uint8: 'u1',
    short: 'i2', int16: 'i2', ushort: 'u2', uint16: 'u2',
    int: 'i4', int32: 'i4', uint: 'u4', uint32: 'u4',
    float: 'f4', float32: 'f4', double: 'f8', float64: 'f8',
};
const TYPE_SIZE = { i1: 1, u1: 1, i2: 2, u2: 2, i4: 4, u4: 4, f4: 4, f8: 8 };

function parsePlyHeader(bytes) {
    const head = latin1(bytes, 0, Math.min(bytes.length, 1 << 16));
    const m = /end_header[ \t]*(\r\n|\n|\r)/.exec(head);
    if (!m) throw new MeshParseError('PLY header has no end_header');
    const dataStart = m.index + m[0].length;
    const lines = head.slice(0, m.index).split(/\r?\n|\r/);
    let format = null;
    const elements = [];
    for (let li = 0; li < lines.length; li++) {
        const parts = lines[li].trim().split(/\s+/);
        const kw = parts[0];
        if (!kw || kw === 'ply' || kw === 'comment' || kw === 'obj_info') continue;
        if (kw === 'format') {
            format = parts[1];
            if (format !== 'ascii' && format !== 'binary_little_endian' && format !== 'binary_big_endian') {
                throw new MeshParseError('unknown PLY format "' + parts[1] + '"');
            }
        } else if (kw === 'element') {
            const count = Number(parts[2]);
            if (!Number.isInteger(count) || count < 0) throw new MeshParseError('PLY element "' + parts[1] + '" has a bad count');
            elements.push({ name: parts[1], count: count, props: [] });
        } else if (kw === 'property') {
            const el = elements[elements.length - 1];
            if (!el) throw new MeshParseError('PLY property before any element');
            if (parts[1] === 'list') {
                const ct = PLY_TYPES[parts[2]], it = PLY_TYPES[parts[3]];
                if (!ct || !it) throw new MeshParseError('PLY list property has an unknown type');
                el.props.push({ list: true, countType: ct, type: it, name: parts[4] });
            } else {
                const t = PLY_TYPES[parts[1]];
                if (!t) throw new MeshParseError('PLY property has unknown type "' + parts[1] + '"');
                el.props.push({ list: false, type: t, name: parts[2] });
            }
        }
    }
    if (!format) throw new MeshParseError('PLY header has no format line');
    return { format: format, elements: elements, dataStart: dataStart };
}

/** Sequential reader over the PLY body: binary (either endianness) or ascii. */
function makeReader(bytes, format, start) {
    if (format === 'ascii') {
        const text = latin1(bytes, start, bytes.length);
        let i = 0;
        const n = text.length;
        return {
            ascii: true,
            read: function () {
                while (i < n) {
                    const c = text.charCodeAt(i);
                    if (c === 32 || c === 9 || c === 10 || c === 13) i++; else break;
                }
                if (i >= n) throw new MeshParseError('PLY data is truncated');
                let j = i;
                while (j < n) {
                    const c = text.charCodeAt(j);
                    if (c === 32 || c === 9 || c === 10 || c === 13) break;
                    j++;
                }
                const v = Number(text.slice(i, j));
                i = j;
                return v;
            },
            position: function () { return i; },
            size: n,
        };
    }
    const le = format === 'binary_little_endian';
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    let off = start;
    const len = bytes.length;
    return {
        ascii: false,
        readType: function (t) {
            const sz = TYPE_SIZE[t];
            if (off + sz > len) throw new MeshParseError('PLY data is truncated');
            let v;
            switch (t) {
            case 'i1': v = dv.getInt8(off); break;
            case 'u1': v = dv.getUint8(off); break;
            case 'i2': v = dv.getInt16(off, le); break;
            case 'u2': v = dv.getUint16(off, le); break;
            case 'i4': v = dv.getInt32(off, le); break;
            case 'u4': v = dv.getUint32(off, le); break;
            case 'f4': v = dv.getFloat32(off, le); break;
            default: v = dv.getFloat64(off, le); break;
            }
            off += sz;
            return v;
        },
        position: function () { return off; },
        size: len,
    };
}

function readValue(rd, type) { return rd.ascii ? rd.read() : rd.readType(type); }

const COLOR_NAMES = {
    red: 0, green: 1, blue: 2, diffuse_red: 0, diffuse_green: 1, diffuse_blue: 2, r: 0, g: 1, b: 2,
};

function* parsePly(bytes, chunk) {
    const hdr = parsePlyHeader(bytes);
    const rd = makeReader(bytes, hdr.format, hdr.dataStart);
    const vEl = hdr.elements.find(function (e) { return e.name === 'vertex'; });
    if (!vEl) throw new MeshParseError('the PLY has no vertex element');
    const xyzIdx = ['x', 'y', 'z'].map(function (n) {
        return vEl.props.findIndex(function (p) { return !p.list && p.name === n; });
    });
    if (xyzIdx.some(function (k) { return k < 0; })) throw new MeshParseError('the PLY vertices have no x, y, z');
    const nrmIdx = ['nx', 'ny', 'nz'].map(function (n) {
        return vEl.props.findIndex(function (p) { return !p.list && p.name === n; });
    });
    const hasNormals = nrmIdx.every(function (k) { return k >= 0; });
    const colIdx = [-1, -1, -1];
    vEl.props.forEach(function (p, k) {
        if (!p.list && Object.prototype.hasOwnProperty.call(COLOR_NAMES, p.name) && colIdx[COLOR_NAMES[p.name]] < 0) {
            colIdx[COLOR_NAMES[p.name]] = k;
        }
    });
    const hasColors = colIdx.every(function (k) { return k >= 0; });

    const nV = vEl.count;
    const positions = new Float32Array(3 * nV);
    const normals = hasNormals ? new Float32Array(3 * nV) : null;
    const colors = hasColors ? new Uint8Array(3 * nV) : null;
    const finite = new Uint8Array(nV);
    let badVertices = 0;
    const tris = new Grow(Uint32Array, 3 * 1024);
    let triCount = 0, droppedFaces = 0;
    const totalRows = hdr.elements.reduce(function (n, e) { return n + e.count; }, 0);
    let rowsDone = 0;
    const row = new Float64Array(Math.max(1, vEl.props.length));

    for (const el of hdr.elements) {
        if (el === vEl) {
            for (let v = 0; v < nV; v++) {
                for (let k = 0; k < el.props.length; k++) {
                    const p = el.props[k];
                    if (p.list) {
                        const c = readValue(rd, p.countType);
                        for (let q = 0; q < c; q++) readValue(rd, p.type);
                        row[k] = NaN;
                    } else {
                        row[k] = readValue(rd, p.type);
                    }
                }
                const x = row[xyzIdx[0]], y = row[xyzIdx[1]], z = row[xyzIdx[2]];
                if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {
                    finite[v] = 1;
                    positions[3 * v] = x; positions[3 * v + 1] = y; positions[3 * v + 2] = z;
                } else {
                    badVertices++;   // left at 0, never referenced by a kept face
                }
                if (normals) {
                    for (let a = 0; a < 3; a++) {
                        const nval = row[nrmIdx[a]];
                        normals[3 * v + a] = Number.isFinite(nval) ? nval : 0;
                    }
                }
                if (colors) {
                    for (let a = 0; a < 3; a++) {
                        const p = el.props[colIdx[a]];
                        let c = row[colIdx[a]];
                        if (p.type === 'f4' || p.type === 'f8') c = c * 255;
                        else if (p.type === 'u2' || p.type === 'i2') c = c / 257;
                        colors[3 * v + a] = Number.isFinite(c) ? Math.max(0, Math.min(255, Math.round(c))) : 0;
                    }
                }
                if (++rowsDone % chunk === 0) yield { done: rowsDone, total: totalRows, unit: 'rows' };
            }
        } else if (el.name === 'face') {
            const li = el.props.findIndex(function (p) {
                return p.list && (p.name === 'vertex_indices' || p.name === 'vertex_index');
            });
            const idx = [];
            for (let f = 0; f < el.count; f++) {
                for (let k = 0; k < el.props.length; k++) {
                    const p = el.props[k];
                    if (!p.list) { readValue(rd, p.type); continue; }
                    const c = readValue(rd, p.countType);
                    if (!Number.isInteger(c) || c < 0) throw new MeshParseError('PLY face has a bad vertex count');
                    if (k !== li) { for (let q = 0; q < c; q++) readValue(rd, p.type); continue; }
                    idx.length = 0;
                    for (let q = 0; q < c; q++) idx.push(readValue(rd, p.type));
                    if (c < 3) { droppedFaces++; continue; }
                    let ok = true;
                    for (let q = 0; q < c; q++) {
                        const iv = idx[q];
                        if (!Number.isInteger(iv) || iv < 0 || iv >= nV || !finite[iv]) { ok = false; break; }
                    }
                    if (!ok) { droppedFaces++; continue; }
                    for (let q = 1; q + 1 < c; q++) { tris.push3(idx[0], idx[q], idx[q + 1]); triCount++; }
                }
                if (++rowsDone % chunk === 0) yield { done: rowsDone, total: totalRows, unit: 'rows' };
            }
        } else {
            // Any other element (edges, materials…): read past it.
            for (let r = 0; r < el.count; r++) {
                for (const p of el.props) {
                    if (p.list) {
                        const c = readValue(rd, p.countType);
                        for (let q = 0; q < c; q++) readValue(rd, p.type);
                    } else {
                        readValue(rd, p.type);
                    }
                }
                if (++rowsDone % chunk === 0) yield { done: rowsDone, total: totalRows, unit: 'rows' };
            }
        }
    }

    const format = hdr.format === 'ascii' ? 'ply-ascii'
        : hdr.format === 'binary_little_endian' ? 'ply-binary-le' : 'ply-binary-be';
    if (triCount > 0) {
        return {
            format: format, kind: 'mesh',
            positions: positions, indices: tris.result(), normals: normals, colors: colors,
            vertexCount: nV, triangleCount: triCount,
            bbox: bboxOf(positions, nV),
            dropped: { triangles: droppedFaces, vertices: badVertices },
        };
    }
    // Points: compact away the non-finite ones (nothing references them).
    let kept = nV - badVertices;
    if (!kept) throw new MeshParseError('no vertex in the PLY has finite coordinates');
    let pPos = positions, pCol = colors, pNrm = normals;
    if (badVertices) {
        pPos = new Float32Array(3 * kept);
        pCol = colors ? new Uint8Array(3 * kept) : null;
        pNrm = normals ? new Float32Array(3 * kept) : null;
        let w = 0;
        for (let v = 0; v < nV; v++) {
            if (!finite[v]) continue;
            for (let a = 0; a < 3; a++) {
                pPos[3 * w + a] = positions[3 * v + a];
                if (pCol) pCol[3 * w + a] = colors[3 * v + a];
                if (pNrm) pNrm[3 * w + a] = normals[3 * v + a];
            }
            w++;
        }
    }
    return {
        format: format, kind: 'points',
        positions: pPos, indices: null, normals: pNrm, colors: pCol,
        vertexCount: kept, triangleCount: 0,
        bbox: bboxOf(pPos, kept),
        dropped: { triangles: droppedFaces, vertices: badVertices },
    };
}

/**
 * The parse as a generator: yields `{done, total, unit}` between chunks and
 * returns the result. `opts.chunk` = rows between yields.
 */
export function* parseMeshSteps(input, opts) {
    const bytes = toBytes(input);
    const chunk = Math.max(1, (opts && opts.chunk) || MESH_PARSE_CHUNK);
    const fmt = detectMeshFormat(bytes, opts && opts.name);
    if (fmt === 'stl-binary') return yield* parseStlBinary(bytes, chunk);
    if (fmt === 'stl-ascii') return yield* parseStlAscii(bytes, chunk);
    return yield* parsePly(bytes, chunk);
}

/**
 * Parse an STL or PLY file in one go.
 * @param {ArrayBuffer|Uint8Array} input
 * @param {{name?:string, chunk?:number}} [opts]
 * @returns {{format:string, kind:'mesh'|'points', positions:Float32Array,
 *   indices:Uint32Array|null, normals:Float32Array|null, colors:Uint8Array|null,
 *   vertexCount:number, triangleCount:number,
 *   bbox:{min:number[], max:number[]}|null,
 *   dropped:{triangles:number, vertices:number}}}
 * @throws {MeshParseError}
 */
export function parseMesh(input, opts) {
    const it = parseMeshSteps(input, opts);
    let r = it.next();
    while (!r.done) r = it.next();
    return r.value;
}

/**
 * The same parse, awaiting `yieldFn()` (e.g. the page's paint) after each chunk
 * and reporting `onProgress(done, total, unit)`.
 */
export async function parseMeshAsync(input, opts, onProgress, yieldFn) {
    const it = parseMeshSteps(input, opts);
    let r = it.next();
    while (!r.done) {
        if (onProgress) onProgress(r.value.done, r.value.total, r.value.unit);
        if (yieldFn) await yieldFn();
        r = it.next();
    }
    return r.value;
}
