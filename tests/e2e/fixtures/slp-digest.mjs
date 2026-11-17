/**
 * slp-digest.mjs — what a tracked + triangulated LUCID project holds, as text,
 * read back with the app's own reader (import-export/slp-session-build.js) in
 * Node, every double printed exactly (`String(x)` round-trips).
 *
 * Per session: its name, cameras, skeleton, tracks, identities, numAnimals,
 * the `metadata.lucid` keys, the per-frame identity map, and frame by frame
 * every group (identity, method, members by camera) and every unlinked pose;
 * the groups' 3D comes separately, as numbers (`compare3d`). A pose is named by
 * its content (type, track, score and every keypoint), so two files agree
 * only if the same poses landed in the same groups with the same identities.
 *
 * `node tests/e2e/fixtures/slp-digest.mjs <file.slp> <out-prefix>` writes
 * `<out-prefix>.txt` (the text), `<out-prefix>.sha256` (its hash) and
 * `<out-prefix>.f64` (the 3D, little-endian doubles): to compare a cluster run
 * with a local one without moving the projects.
 *
 * Left out on purpose: video paths and shapes (the app writes the videos it
 * opened, the CLI the ones the input referenced) and the file provenance's
 * `exported_at`.
 */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { registerImportMap } from '../../../cli/node-resolve.mjs';
import { installNodePlatform, DiskFile } from '../../../cli/node-platform.mjs';

let ready = null;
async function platform() {
    if (!ready) ready = (async () => {
        registerImportMap();
        const { baseURI } = installNodePlatform();
        globalThis.SleapIO = await import(new URL('lib/sleap-io/index.browser.js', baseURI).href);
        const read = await import('../../../import-export/slp-read.js');
        read.setAppBaseURI(baseURI);
        return {
            build: await import('../../../import-export/slp-session-build.js'),
            pd: await import('../../../pose/pose-data.js'),
            planes: await import('../../../import-export/plane-metadata.js'),
        };
    })();
    return ready;
}

const num = (x) => Number.isNaN(x) ? 'NaN' : String(x);

function poseKey(inst) {
    const pts = [];
    for (let k = 0; k < inst.numNodes; k++) pts.push(inst.hasPoint(k) ? num(inst.getX(k)) + ',' + num(inst.getY(k)) : '-');
    return inst.type[0] + inst.trackIdx + '@' + num(inst.score) + '[' + pts.join(' ') + ']';
}

/**
 * @returns {Promise<{text: string, points3d: Float64Array, sessions: number, groups: number,
 *   with3d: number, identities: string[]}>}  `text` is everything but the 3D
 *   coordinates (each group line says how many it has); `points3d` is every
 *   group's 3D in the text's order.
 */
export async function digestSlp(file) {
    const { build, pd, planes } = await platform();
    const quiet = { log: console.log, warn: console.warn };
    console.log = console.warn = () => {};
    try {
        planes.resetPlaneState();
        const slpData = await build.parseSlpForImport(new DiskFile(file), () => {});
        const sk = slpData.skeleton || { name: 'skeleton', nodes: [], edges: [] };
        const skeleton = new pd.Skeleton(sk.name, sk.nodes, sk.edges);
        const out = [];
        let groups = 0, with3d = 0, idNames = [];
        const p3 = [];   // every group's 3D, in the text's order
        const nSess = (slpData.sessions && slpData.sessions.length) || 0;
        for (let si = 0; si < nSess; si++) {
            const { session: s } = await build.buildSessionFromSlpData(slpData, si, skeleton, {});
            const name = new Map(s.identities.map(i => [i.id, i.name]));
            idNames = idNames.concat(s.identities.map(i => i.name));
            out.push('session ' + s.name + ' cameras ' + s.cameras.map(c => c.name).join(',') +
                ' nodes ' + s.skeleton.nodes.join(','));
            out.push('tracks ' + s.tracks.join(','));
            out.push('identities ' + s.identities.map(i => i.name).join(','));
            out.push('numAnimals ' + s.numAnimals);
            // Which metadata.lucid keys the file carries (an ID-switch review, planes, ...).
            const meta = slpData.sessions[si] && slpData.sessions[si].metadata && slpData.sessions[si].metadata.lucid;
            out.push('lucidKeys ' + (meta ? Object.keys(meta).sort().join(',') : '-'));
            // The ID Switches checklist an automatic check left (Track All's body-size check).
            if (meta && meta.idSwitchReview) out.push('idSwitchReview ' + JSON.stringify(meta.idSwitchReview));
            const fim = [];
            for (const rec of s.frameIdentityEntries()) {
                fim.push(rec.frameIdx + ':' + rec.camName + ':' + rec.trackIdx + '=' + (rec.identityId === -1 ? '-1' : name.get(rec.identityId)));
            }
            fim.sort();
            out.push('frameIdentityMap ' + fim.length + ' ' + fim.join(';'));
            const frames = new Set([...s.frameGroups.keys(), ...s.instanceGroups.keys()]);
            for (const fi of [...frames].sort((a, b) => a - b)) {
                const gs = (s.instanceGroups.get(fi) || []).map(g => {
                    const mem = [];
                    g.instances.forEach((inst, cam) => mem.push(cam + ':' + poseKey(inst)));
                    mem.sort();
                    groups++;
                    if (g.points3d && pd.someValidPoint3d(g.points3d)) with3d++;
                    return ['G ' + name.get(g.identityId) + ' ' + (g.triangulationMethod || '-') + ' {' + mem.join(' ') + '} ' +
                        (g.points3d ? g.points3d.length + ' 3D' : 'no 3D'), g.points3d ? Array.from(g.points3d) : []];
                }).sort((x, y) => (x[0] < y[0] ? -1 : x[0] > y[0] ? 1 : 0));
                for (const g of gs) for (const v of g[1]) p3.push(v);
                const fg = s.frameGroups.get(fi);
                const un = [];
                if (fg) fg.unlinkedInstances.forEach((list, cam) => list.forEach(x => un.push(cam + ':' + poseKey(x.instance))));
                un.sort();
                out.push(fi + '\n  ' + gs.map(g => g[0]).join('\n  ') + '\n  U ' + un.join(' '));
            }
        }
        return { text: out.join('\n'), points3d: Float64Array.from(p3), sessions: nSess, groups, with3d, identities: idNames };
    } finally {
        console.log = quiet.log; console.warn = quiet.warn;
    }
}

/** The first line where two digests differ, for the failure message (null when equal). */
export function firstDifference(a, b) {
    if (a === b) return null;
    const x = a.split('\n'), y = b.split('\n');
    let i = 0;
    while (i < x.length && i < y.length && x[i] === y[i]) i++;
    return { line: i, a: String(x[i]).slice(0, 300), b: String(y[i]).slice(0, 300) };
}

/**
 * Compare two digests' 3D: how many values are bit-identical, and the largest
 * difference relative to max(1, |value|) (NaN must meet NaN).
 *
 * Why a tolerance at all: the same code on the same input gives the same
 * doubles only in the same JavaScript engine. Chrome and Node ship different
 * builds of V8 whose Math.atan2 / cos / sin / exp / log can differ in the last
 * bit (sqrt, pow and hypot agree), and the DLT solve's Jacobi fallback
 * (pose/triangulation-core.js) uses atan2, cos and sin. Measured on a real
 * 8-camera, 18,058-frame session: 46 of 53,419 groups differed, by about
 * 1e-13 relative; re-solving the app file's own groups in Node gave the CLI's
 * values and in Chrome the app's, so the difference is the engine's.
 */
export function compare3d(a, b, relTol) {
    const x = a.points3d, y = b.points3d;
    if (x.length !== y.length) return { ok: false, n: Math.max(x.length, y.length), exact: 0, maxRel: Infinity, lengths: [x.length, y.length] };
    let exact = 0, maxRel = 0, nanMismatch = 0;
    for (let i = 0; i < x.length; i++) {
        const p = x[i], q = y[i];
        if (Object.is(p, q)) { exact++; continue; }
        if (Number.isNaN(p) || Number.isNaN(q)) { nanMismatch++; continue; }
        const rel = Math.abs(p - q) / Math.max(1, Math.abs(p), Math.abs(q));
        if (rel > maxRel) maxRel = rel;
    }
    return { ok: nanMismatch === 0 && maxRel <= relTol, n: x.length, exact, maxRel, nanMismatch };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    const [file, prefix] = process.argv.slice(2);
    const d = await digestSlp(file);
    fs.writeFileSync(prefix + '.txt', d.text);
    fs.writeFileSync(prefix + '.sha256', crypto.createHash('sha256').update(d.text).digest('hex') + '\n');
    fs.writeFileSync(prefix + '.f64', Buffer.from(d.points3d.buffer));
    console.log(JSON.stringify({ groups: d.groups, with3d: d.with3d, identities: d.identities, values3d: d.points3d.length }));
    process.exit(0);
}
