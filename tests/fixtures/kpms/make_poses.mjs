/**
 * make_poses.mjs — synthetic poses in the files LUC3D exports for
 * keypoint-MoSeq (import-export/kpms-io.js): two animals of one session,
 * 300 frames each at 30 fps, as anipose CSVs plus luc3d-kpms.yml.
 *
 * Each animal walks a loop at changing speeds and now and then rears (the
 * nose and head rise) or grooms (the head dips to the body), so a short fit
 * has something to find. Used by make_fixture.sh.
 *
 *   node tests/fixtures/kpms/make_poses.mjs <out dir>
 */
import fs from 'node:fs';
import path from 'node:path';
import { poseCsv, kpmsManifest, recordingNames } from '../../../import-export/kpms-io.js';

const OUT = process.argv[2];
if (!OUT) { console.error('usage: node make_poses.mjs <out dir>'); process.exit(2); }
fs.mkdirSync(OUT, { recursive: true });

const NODES = ['nose', 'head', 'ear_L', 'ear_R', 'spine', 'tail_base', 'tail_tip'];
const EDGES = [[0, 1], [1, 2], [1, 3], [1, 4], [4, 5], [5, 6]];
const T = 300, FPS = 30, N = NODES.length;

// A small seeded generator, so the fixture can be made again exactly.
let seed = 12345;
const rand = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
const noise = (s) => (rand() - 0.5) * 2 * s;

function animal(phase) {
    const data = new Float32Array(T * N * 3);
    let x = 100 * Math.cos(phase), y = 100 * Math.sin(phase), heading = phase + Math.PI / 2;
    for (let t = 0; t < T; t++) {
        const mode = Math.floor((t + phase * 40) / 50) % 3;              // 0 walk, 1 rear, 2 groom
        const speed = mode === 0 ? 3 + 2 * Math.sin(t / 7) : 0.3;
        heading += mode === 0 ? 0.03 : 0;
        x += speed * Math.cos(heading); y += speed * Math.sin(heading);
        const c = Math.cos(heading), s = Math.sin(heading);
        // Body-frame offsets (forward, left, up), in mm.
        const rise = mode === 1 ? 25 * Math.sin(Math.PI * ((t % 50) / 50)) : 0;
        const dip = mode === 2 ? -8 * (1 + Math.sin(t / 2)) : 0;
        const body = [
            [40, 0, 15 + rise + dip], [28, 0, 18 + 0.8 * rise + dip], [26, 7, 22 + 0.8 * rise], [26, -7, 22 + 0.8 * rise],
            [5, 0, 20 + 0.3 * rise], [-25, 0, 12], [-60, 3 * Math.sin(t / 3), 4],
        ];
        for (let n = 0; n < N; n++) {
            const [f, l, u] = body[n];
            const o = (t * N + n) * 3;
            data[o] = x + f * c - l * s + noise(0.5);
            data[o + 1] = y + f * s + l * c + noise(0.5);
            data[o + 2] = u + noise(0.5);
        }
    }
    return data;
}

const animals = ['id_1', 'id_2'];
const names = recordingNames(animals.map((a) => ({ session: 'synth', animal: a })));
const recordings = [];
animals.forEach((a, i) => {
    const data = animal(1.3 * i + 0.2);
    // The tail tip goes missing for a stretch, as real tracking does.
    if (i === 1) for (let t = 120; t < 135; t++) data.fill(NaN, (t * N + 6) * 3, (t * N + 6) * 3 + 3);
    fs.writeFileSync(path.join(OUT, names[i] + '.csv'),
        poseCsv({ dim: 3, bodyparts: NODES, data, nodeCount: N, first: 0, last: T - 1, decimals: 3 }));
    recordings.push({ name: names[i], session: 'synth', animal: a, firstFrame: 0, frames: T });
});
fs.writeFileSync(path.join(OUT, 'luc3d-kpms.yml'), kpmsManifest({
    format: 'anipose', dim: 3, units: 'mm', coordinates: 'floor', fps: FPS,
    bodyparts: NODES, useBodyparts: NODES.slice(0, 6), anterior: ['nose'], posterior: ['tail_base'],
    skeleton: EDGES.map(([a, b]) => [NODES[a], NODES[b]]), recordings,
}));
console.log('wrote ' + recordings.length + ' recordings and luc3d-kpms.yml to ' + OUT);
