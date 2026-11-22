/**
 * discovery-world.js — a small world for Discover Behaviors, for tests that only need
 * it to run where they run it (a stamped build, offline): two animals over `T` frames at
 * 30 fps, switching between resting, walking and rearing every 1–5 s.
 *
 * Page-side and import-free: the caller passes the app's modules, so a stamped build
 * uses the instances its URLs name.
 *
 * @param {number} T frames
 * @param {{tensor: Object, features: Object, forest: Object}} m pose/behavior-tensor.js, -features.js, -forest.js
 * @returns {Object} a behavior context, as ui/behavior-detect.js builds one
 */
export function discoveryContext(T, m) {
    const nodeNames = ['Nose', 'Head', 'Neck', 'Trunk', 'PawL', 'PawR', 'TTI'];
    const edges = [[0, 1], [1, 2], [2, 3], [2, 4], [2, 5], [3, 6]];
    const ALONG = [50, 36, 20, 0, 18, 18, -40], SIDE = [0, 0, 0, 0, 12, -12, 0], H = [14, 24, 30, 30, 6, 6, 14];
    const rand = m.forest.seededRandom(3);
    const gauss = () => { let u = 0; while (!u) u = rand(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand()); };
    const N = nodeNames.length;
    const data = [0, 1].map((a) => {
        const arr = new Float32Array(T * N * 3);
        let cx = 3000 * a, cy = 0, hd = rand() * 6.28, f = 0;
        while (f < T) {
            const k = Math.floor(rand() * 3), end = Math.min(T, f + Math.round(30 * (1 + rand() * 4)));
            for (; f < end; f++) {
                if (k === 1) { hd += 0.02 * gauss(); cx += 8 * Math.cos(hd); cy += 8 * Math.sin(hd); }
                for (let n = 0; n < N; n++) {
                    let along = ALONG[n], z = H[n];
                    if (k === 1 && (n === 4 || n === 5)) { along += 9 * Math.sin(0.9 * f + (n === 5 ? Math.PI : 0)); z += 4; }
                    if (k === 2 && n !== 6) { z += 75 * (ALONG[n] + 40) / 90; along = -40 + (ALONG[n] + 40) * 0.45; }
                    const c = Math.cos(hd), s = Math.sin(hd), o = (f * N + n) * 3;
                    arr[o] = cx + along * c - SIDE[n] * s + gauss();
                    arr[o + 1] = cy + along * s + SIDE[n] * c + gauss();
                    arr[o + 2] = z + gauss();
                }
            }
        }
        return arr;
    });
    const input = { mode: '3d', dim: 3, camera: null, numFrames: T, nodeNames, edges, animals: ['id_0', 'id_1'], animalIds: [0, 1], data, frames: [T, T], skipped: 0 };
    const body = m.tensor.resolveBody(nodeNames, edges, null);
    const spec = m.tensor.featureSpec(input, body, m.tensor.floorFrame([0, 0, 0], [0, 0, 1]), 30);
    return { session: { name: 'world' }, input, body, spec, catalog: m.features.catalogFor(spec), fps: 30, broken: m.tensor.brokenPoseFrames(input, body, spec.bodyLength, 30) };
}
