/**
 * pose/volume-fusion.js — combine a volumetric 3D model's estimate with the triangulation of the 2D
 * detections, per node.
 *
 * A volumetric pose model (e.g. a 3D UNet over unprojected 2D confidence maps) gives, per animal and
 * node, a 3D mean and a covariance (the second moments of its softmax over the cube). It runs offline
 * on a GPU; `import-export/volume3d-file.js` reads its output and `pose/volume3d.js` hands one frame of
 * it to the solve as `options.volume`. `triangulateAndReproject` (pose/triangulation-core.js) then:
 *   1. matches the group to one of the frame's volume animals (`matchVolumeAnimal`: median node distance
 *      to the plain solve, under `matchMaxMm`);
 *   2. VOLUME-GATED TRIANGULATION (`gateObservations`): each view's 2D detection of node k enters the
 *      triangulation only if it lies within max(`gateFloorPx`, `gateK` sigma) of the volume's node
 *      projected into that view, sigma = the largest axis of the volume's covariance projected there.
 *      This is gating BEFORE triangulation, not a test of the triangulation's own reprojection;
 *   3. re-solves (DLT or Refined, and the reprojection-threshold rule, unchanged) from what passed;
 *   4. FUSION (`fuseWithVolume`): the product of Gaussians of each solved node (its covariance from its
 *      views' ray geometry and reprojection residuals, `dltCovariance`) and the volume's (covariance x
 *      `covScale`); a node the solve leaves empty takes the volume's mean;
 *   5. SKELETON CHECK (`checkSkeletonChains`, only when the volume file names chains, e.g. the tail and the
 *      head): where a fused chain has a bone more than `chainTol` (50 %) off its usual length, the FEWEST
 *      nodes are changed (to the plain solve's point, the volume's, or dropped) that make the chain
 *      plausible, never one the plain solve and the volume agree on; else the chain is left as it is.
 * The gate uses the volume's covariance as given; the fusion scales it by `covScale`, which the volume's
 * exporter measures (the volfuse study, 2026-11-21: a softmax spread is 17-25x too wide in variance).
 *
 * Import-free and DOM-free: the triangulation worker loads it through the core. The projection function
 * is passed in (`project(X, camera) -> [u, v] | null`, distorted pixels: the core's `reprojectPointCamera`).
 *
 * Volume layout (`vol`): { n: animals, nNodes, means: Float64Array(n * nNodes * 3),
 *   covs: Float64Array(n * nNodes * 6) as [xx, xy, xz, yy, yz, zz], covScale, gateFloorPx, gateK, fuse,
 *   matchMaxMm, chains?: [{ name, bones: Int32Array(2 * nb) node index pairs, nodes: Int32Array,
 *   medians: Float64Array(nb) usual bone lengths }], chainTol? }, node order = the session skeleton's
 *   (NaN = the volume has no such node).
 */

export const VOLUME_DEFAULTS = Object.freeze({
    gateFloorPx: 12, gateK: 2, covScale: 1, fuse: true, matchMaxMm: 30, sigma0Px: 2, pseudoDof: 4,
    chainTol: 0.5, chainAgreeMm: 10,
});

/** A chain's badness for a missing bone (the volfuse study's value: worse than any measured one). */
const MISSING_BONE = 10;
/** Fewest finite lengths a bone's usual length is measured from. */
const MIN_BONE_SAMPLES = 5;
/** The repair's penalty for a kept chain node no path of present bones links to the body (the chain's anchor). */
const ORPHAN = 1;
/** Most option combinations the repair tries for one chain (4 options per node: 6 nodes); larger chains are left. */
const MAX_REPAIR_COMBOS = 4096;

const JAC_EPS = 0.05;   // world units (mm) for the numerical projection Jacobian

/** Skeleton node index -> volume node index (-1 = none), by name. */
export function mapVolumeNodes(volumeNodeNames, skeletonNodeNames) {
    const at = new Map();
    volumeNodeNames.forEach(function (n, j) { at.set(String(n), j); });
    return Int32Array.from(skeletonNodeNames, function (n) { return at.has(String(n)) ? at.get(String(n)) : -1; });
}

/** Read animal a, node k: mean into m (3), covariance into S (9, row-major). -> whether finite. */
export function volumeNode(vol, a, k, m, S) {
    const i3 = (a * vol.nNodes + k) * 3, i6 = (a * vol.nNodes + k) * 6;
    m[0] = vol.means[i3]; m[1] = vol.means[i3 + 1]; m[2] = vol.means[i3 + 2];
    const c = vol.covs;
    S[0] = c[i6]; S[1] = c[i6 + 1]; S[2] = c[i6 + 2];
    S[3] = c[i6 + 1]; S[4] = c[i6 + 3]; S[5] = c[i6 + 4];
    S[6] = c[i6 + 2]; S[7] = c[i6 + 4]; S[8] = c[i6 + 5];
    for (let i = 0; i < 3; i++) if (!Number.isFinite(m[i])) return false;
    for (let i = 0; i < 9; i++) if (!Number.isFinite(S[i])) return false;
    return true;
}

/** The volume animal this group is: smallest median node distance to `points3d` (flat 3N), under
 *  `vol.matchMaxMm`. -> animal index or -1. */
export function matchVolumeAnimal(points3d, vol) {
    const N = Math.min(vol.nNodes, Math.floor(points3d.length / 3));
    const lim = vol.matchMaxMm != null ? vol.matchMaxMm : VOLUME_DEFAULTS.matchMaxMm;
    let best = -1, bestD = Infinity;
    const d = [];
    for (let a = 0; a < vol.n; a++) {
        d.length = 0;
        for (let k = 0; k < N; k++) {
            const x = points3d[3 * k], y = points3d[3 * k + 1], z = points3d[3 * k + 2];
            const i = (a * vol.nNodes + k) * 3;
            const dx = x - vol.means[i], dy = y - vol.means[i + 1], dz = z - vol.means[i + 2];
            const e = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (Number.isFinite(e)) d.push(e);
        }
        if (d.length < 3) continue;
        d.sort(function (p, q) { return p - q; });
        const med = d.length % 2 ? d[(d.length - 1) / 2] : 0.5 * (d[d.length / 2 - 1] + d[d.length / 2]);
        if (med < bestD) { bestD = med; best = a; }
    }
    return bestD < lim ? best : -1;
}

/** Numerical Jacobian of `project` at X for camera `cam` -> J (6, row-major 2x3) or null. */
function projJacobian(X, cam, project, J) {
    const P = [0, 0, 0];
    for (let i = 0; i < 3; i++) {
        P[0] = X[0]; P[1] = X[1]; P[2] = X[2];
        P[i] = X[i] + JAC_EPS;
        const a = project(P, cam);
        P[i] = X[i] - JAC_EPS;
        const b = project(P, cam);
        if (a == null || b == null) return null;
        J[i] = (a[0] - b[0]) / (2 * JAC_EPS);
        J[3 + i] = (a[1] - b[1]) / (2 * JAC_EPS);
    }
    return J;
}

/** Largest eigenvalue's square root of the 2x2 J S J^T (px). */
function projectedSigma(J, S) {
    // JS = J (2x3) * S (3x3)
    const JS = [0, 0, 0, 0, 0, 0];
    for (let r = 0; r < 2; r++) for (let c = 0; c < 3; c++) {
        JS[3 * r + c] = J[3 * r] * S[c] + J[3 * r + 1] * S[3 + c] + J[3 * r + 2] * S[6 + c];
    }
    const a = JS[0] * J[0] + JS[1] * J[1] + JS[2] * J[2];
    const b = JS[0] * J[3] + JS[1] * J[4] + JS[2] * J[5];
    const d = JS[3] * J[3] + JS[4] * J[4] + JS[5] * J[5];
    const tr = a + d, det = a * d - b * b;
    const lmax = 0.5 * tr + Math.sqrt(Math.max(0, 0.25 * tr * tr - det));
    return Math.sqrt(Math.max(0, lmax));
}

/**
 * Volume-gated triangulation's gate: null out (in place, in BOTH observation tables) every detection of
 * a node farther than max(gateFloorPx, gateK * sigma) from the volume's node projected into its view.
 * `obs[k][c]` / `obsRaw[k][c]`: the solve's undistorted / native-pixel observation tables;
 * `cams[c]`, `included[c]` as in the solve. -> how many detections were dropped.
 */
export function gateObservations(vol, a, obs, obsRaw, cams, included, project, nKeypoints) {
    const floor = vol.gateFloorPx != null ? vol.gateFloorPx : VOLUME_DEFAULTS.gateFloorPx;
    const kk = vol.gateK != null ? vol.gateK : VOLUME_DEFAULTS.gateK;
    const m = [0, 0, 0], S = new Array(9), J = new Array(6);
    let dropped = 0;
    const N = Math.min(nKeypoints, vol.nNodes);
    for (let k = 0; k < N; k++) {
        if (!volumeNode(vol, a, k, m, S)) continue;
        for (let c = 0; c < cams.length; c++) {
            if (!included[c] || obsRaw[k][c] == null) continue;
            const p = project(m, cams[c]);
            if (p == null) continue;
            const raw = obsRaw[k][c];
            const dist = Math.hypot(raw[0] - p[0], raw[1] - p[1]);
            let gate = floor;
            if (kk > 0 && projJacobian(m, cams[c], project, J)) gate = Math.max(floor, kk * projectedSigma(J, S));
            if (dist > gate) { obs[k][c] = null; obsRaw[k][c] = null; dropped++; }
        }
    }
    return dropped;
}

function inv3(M, out) {
    const a = M[0], b = M[1], c = M[2], d = M[3], e = M[4], f = M[5], g = M[6], h = M[7], k = M[8];
    const det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
    // relative singularity test: a tiny but well-conditioned covariance (mm^2 or smaller) must still invert
    const scale = Math.abs(a * e * k) + Math.abs(b * f * g) + Math.abs(c * d * h) +
                  Math.abs(c * e * g) + Math.abs(b * d * k) + Math.abs(a * f * h);
    if (!Number.isFinite(det) || !(Math.abs(det) > 1e-12 * scale)) return null;
    const s = 1 / det;
    out[0] = (e * k - f * h) * s; out[1] = (c * h - b * k) * s; out[2] = (b * f - c * e) * s;
    out[3] = (f * g - d * k) * s; out[4] = (a * k - c * g) * s; out[5] = (c * d - a * f) * s;
    out[6] = (d * h - e * g) * s; out[7] = (b * g - a * h) * s; out[8] = (a * e - b * d) * s;
    return out;
}

/**
 * Covariance of a solved point X from its views (Gauss-Newton): s^2 (sum_v J_v^T J_v)^-1 with
 * s^2 = (pseudoDof sigma0^2 + RSS) / (pseudoDof + 2n - 3): the 2D noise measured by the residuals, shrunk
 * toward sigma0 (2 or 3 views leave few degrees of freedom). The views are the node's observations the
 * solve still holds in `obs` (after the gate and the reprojection-threshold rule, which nulls `obs` only),
 * measured at their native pixels in `obsRaw`. -> S (9) or null (fewer than 2 views).
 */
export function dltCovariance(X, k, obs, obsRaw, cams, included, project, out) {
    const sigma0 = VOLUME_DEFAULTS.sigma0Px, k0 = VOLUME_DEFAULTS.pseudoDof;
    const info = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    const J = new Array(6);
    let rss = 0, n = 0;
    for (let c = 0; c < cams.length; c++) {
        if (!included[c] || obs[k][c] == null || obsRaw[k][c] == null) continue;
        const p = project(X, cams[c]);
        if (p == null || !projJacobian(X, cams[c], project, J)) continue;
        const raw = obsRaw[k][c];
        rss += (raw[0] - p[0]) * (raw[0] - p[0]) + (raw[1] - p[1]) * (raw[1] - p[1]);
        for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) info[3 * i + j] += J[i] * J[j] + J[3 + i] * J[3 + j];
        n++;
    }
    if (n < 2) return null;
    const s2 = (k0 * sigma0 * sigma0 + rss) / (k0 + Math.max(2 * n - 3, 0));
    for (let i = 0; i < 9; i++) info[i] += (i % 4 === 0) ? 1e-12 : 0;
    if (!inv3(info, out)) return null;
    for (let i = 0; i < 9; i++) out[i] *= s2;
    return out;
}

/** Product of two Gaussians (means 3, covariances 9) -> mean into `out` (3), or null if singular. */
export function productOfGaussians(Xa, Sa, Xb, Sb, out) {
    const Ia = inv3(Sa, new Array(9)), Ib = inv3(Sb, new Array(9));
    if (!Ia || !Ib) return null;
    const I = new Array(9);
    for (let i = 0; i < 9; i++) I[i] = Ia[i] + Ib[i];
    const S = inv3(I, new Array(9));
    if (!S) return null;
    const v = [0, 0, 0];
    for (let r = 0; r < 3; r++) {
        v[r] = Ia[3 * r] * Xa[0] + Ia[3 * r + 1] * Xa[1] + Ia[3 * r + 2] * Xa[2] +
               Ib[3 * r] * Xb[0] + Ib[3 * r + 1] * Xb[1] + Ib[3 * r + 2] * Xb[2];
    }
    for (let r = 0; r < 3; r++) out[r] = S[3 * r] * v[0] + S[3 * r + 1] * v[1] + S[3 * r + 2] * v[2];
    return out;
}

/**
 * Fuse the solve's points (flat 3N, IN PLACE) with volume animal `a`: a solved node becomes the product
 * of Gaussians of its solve (covariance `dltCovariance`) and the volume (covariance x covScale); a node the
 * solve left empty takes the volume's mean. `vol.fuse === false`: only the empty nodes are filled.
 * -> { fused, filled } node counts.
 */
export function fuseWithVolume(points3d, vol, a, obs, obsRaw, cams, included, project, nKeypoints) {
    const sc = vol.covScale != null ? vol.covScale : VOLUME_DEFAULTS.covScale;
    const doFuse = vol.fuse !== false;
    const m = [0, 0, 0], S = new Array(9), Sv = new Array(9), Sd = new Array(9), X = [0, 0, 0], out = [0, 0, 0];
    let fused = 0, filled = 0;
    const N = Math.min(nKeypoints, vol.nNodes, Math.floor(points3d.length / 3));
    for (let k = 0; k < N; k++) {
        if (!volumeNode(vol, a, k, m, S)) continue;
        X[0] = points3d[3 * k]; X[1] = points3d[3 * k + 1]; X[2] = points3d[3 * k + 2];
        const solved = Number.isFinite(X[0]) && Number.isFinite(X[1]) && Number.isFinite(X[2]);
        if (!solved) {
            points3d[3 * k] = m[0]; points3d[3 * k + 1] = m[1]; points3d[3 * k + 2] = m[2];
            filled++;
            continue;
        }
        if (!doFuse) continue;
        if (!dltCovariance(X, k, obs, obsRaw, cams, included, project, Sd)) continue;
        for (let i = 0; i < 9; i++) Sv[i] = S[i] * sc;
        if (!productOfGaussians(X, Sd, m, Sv, out)) continue;
        points3d[3 * k] = out[0]; points3d[3 * k + 1] = out[1]; points3d[3 * k + 2] = out[2];
        fused++;
    }
    return { fused: fused, filled: filled };
}

/**
 * Each bone's usual length: the median over the volume file's rows of the distance between its two nodes
 * (`bones`: Int32Array of FILE node index pairs; `means`: the file's (R, J, 3) means, flat). The volfuse
 * study measured them on the plain solve's reliable points instead; on July 13 the two agree within 8 %,
 * well inside the 50 % tolerance. -> Float64Array (NaN = fewer than MIN_BONE_SAMPLES finite lengths).
 */
export function chainBoneMedians(bones, means, J) {
    const nb = bones.length >> 1, R = Math.floor(means.length / (3 * J));
    const out = new Float64Array(nb).fill(NaN);
    const L = [];
    for (let i = 0; i < nb; i++) {
        L.length = 0;
        for (let r = 0; r < R; r++) {
            const p = (r * J + bones[2 * i]) * 3, q = (r * J + bones[2 * i + 1]) * 3;
            const dx = means[p] - means[q], dy = means[p + 1] - means[q + 1], dz = means[p + 2] - means[q + 2];
            const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (Number.isFinite(d)) L.push(d);
        }
        if (L.length < MIN_BONE_SAMPLES) continue;
        L.sort(function (x, y) { return x - y; });
        const n = L.length;
        out[i] = n % 2 ? L[(n - 1) / 2] : 0.5 * (L[n / 2 - 1] + L[n / 2]);
    }
    return out;
}

/** How far a chain's bones in `P` (flat, node k at off + 3k) are past the tolerance: the sum over bones of
 *  max(|L / usual - 1| - tol, 0), MISSING_BONE per bone with a missing end. 0 = plausible. */
export function chainBadness(P, off, chain, tol) {
    const b = chain.bones, med = chain.medians;
    let s = 0;
    for (let i = 0; i < med.length; i++) {
        const p = off + 3 * b[2 * i], q = off + 3 * b[2 * i + 1];
        const dx = P[p] - P[q], dy = P[p + 1] - P[q + 1], dz = P[p + 2] - P[q + 2];
        const L = Math.sqrt(dx * dx + dy * dy + dz * dz);
        s += Number.isFinite(L) ? Math.max(Math.abs(L / med[i] - 1) - tol, 0) : MISSING_BONE;
    }
    return s;
}

/**
 * How far a candidate repair of a chain (`P` flat, node k at off + 3k; NaN = dropped or missing) is from plausible:
 * the sum over bones with BOTH ends present of max(|L / usual - 1| - tol, 0), plus ORPHAN for each present chain
 * node that no path of present bones links to the anchor (a bone node outside `chain.nodes`, e.g. TTI or the Neck).
 * A dropped node's bones are not judged, but it cannot cut a wrong piece of the chain loose. 0 = plausible.
 */
export function chainRepairBadness(P, off, chain, tol) {
    const b = chain.bones, med = chain.medians, nb = med.length;
    const has = function (k) {
        const q = off + 3 * k;
        return Number.isFinite(P[q]) && Number.isFinite(P[q + 1]) && Number.isFinite(P[q + 2]);
    };
    const inChain = new Set(chain.nodes);
    const reached = new Set();
    for (let i = 0; i < 2 * nb; i++) if (!inChain.has(b[i]) && has(b[i])) reached.add(b[i]);
    let grew = true;
    while (grew) {
        grew = false;
        for (let i = 0; i < nb; i++) {
            const p = b[2 * i], q = b[2 * i + 1];
            if (!has(p) || !has(q) || reached.has(p) === reached.has(q)) continue;
            reached.add(reached.has(p) ? q : p);
            grew = true;
        }
    }
    let s = 0;
    for (let i = 0; i < nb; i++) {
        const p = b[2 * i], q = b[2 * i + 1];
        if (!has(p) || !has(q)) continue;
        const L = Math.hypot(P[off + 3 * p] - P[off + 3 * q], P[off + 3 * p + 1] - P[off + 3 * q + 1],
            P[off + 3 * p + 2] - P[off + 3 * q + 2]);
        s += Math.max(Math.abs(L / med[i] - 1) - tol, 0);
    }
    for (let i = 0; i < chain.nodes.length; i++) if (has(chain.nodes[i]) && !reached.has(chain.nodes[i])) s += ORPHAN;
    return s;
}

const KEEP = 0, FROM_SOLVE = 1, FROM_VOLUME = 2, DROP = 3;
function finite3(P, q) { return Number.isFinite(P[q]) && Number.isFinite(P[q + 1]) && Number.isFinite(P[q + 2]); }
function dist3(P, p, Q, q) { return Math.hypot(P[p] - Q[q], P[p + 1] - Q[q + 1], P[p + 2] - Q[q + 2]); }

/**
 * Skeleton check of the fused points (flat 3N, IN PLACE) against volume animal `a`'s chains (`vol.chains`).
 * A chain whose fused bones are implausible (`chainBadness` > 0) is REPAIRED with the fewest node changes that make
 * it plausible (`chainRepairBadness` 0): each chain node may keep its fused point, take the plain solve's or the
 * volume's, or be dropped (NaN, no 3D); ties go to fewer drops, then to the smallest move. A node the plain solve
 * and the volume put within `vol.chainAgreeMm` (10 mm) of each other is never changed. If no such repair makes the
 * chain plausible, it is left as it is.
 * Why the fewest nodes, not the whole chain: in pile-ups one node is often wrong while the rest are right, and
 * whichever source has a plausible whole chain may have it on another mouse (the volfuse study's July 13 frames
 * 105,416 and 105,372, judged by eye). Why the agreement guard: two sources agreeing is the best evidence a node is
 * right (fight frame 4,294: a short nose bone, but both sources put the nose in the same place).
 * `getPlain()` -> the plain solve (flat 3N) or null; called at most once, and only when a chain is off.
 * -> { chains, fromSolve, fromVolume, dropped }: chains repaired and nodes changed by kind.
 */
export function checkSkeletonChains(points3d, getPlain, vol, a) {
    const out = { chains: 0, fromSolve: 0, fromVolume: 0, dropped: 0 };
    const chains = vol.chains;
    if (!chains || !chains.length || a < 0) return out;
    const tol = vol.chainTol != null ? vol.chainTol : VOLUME_DEFAULTS.chainTol;
    const agree = vol.chainAgreeMm != null ? vol.chainAgreeMm : VOLUME_DEFAULTS.chainAgreeMm;
    const V = vol.means, voff = a * vol.nNodes * 3;
    let plain;
    for (let c = 0; c < chains.length; c++) {
        const ch = chains[c];
        if (!(chainBadness(points3d, 0, ch, tol) > 0)) continue;
        if (plain === undefined) plain = getPlain() || null;
        const K = ch.nodes.length, opts = [];
        let total = 1;
        for (let i = 0; i < K; i++) {
            const q = 3 * ch.nodes[i], qv = voff + q;
            const f = finite3(points3d, q), p = !!plain && finite3(plain, q), v = finite3(V, qv);
            if (p && v && dist3(plain, q, V, qv) <= agree) { opts.push([KEEP]); continue; }
            const o = [KEEP];
            if (p && !(f && dist3(plain, q, points3d, q) <= 1e-6)) o.push(FROM_SOLVE);
            if (v && !(f && dist3(V, qv, points3d, q) <= 1e-6)) o.push(FROM_VOLUME);
            if (f) o.push(DROP);
            opts.push(o);
            total *= o.length;
        }
        if (total <= 1 || total > MAX_REPAIR_COMBOS) continue;
        const work = Float64Array.from(points3d);
        const idx = new Int32Array(K);
        let best = null;
        for (let n = 0; n < total; n++) {
            let changed = 0, drops = 0, moved = 0;
            for (let i = 0; i < K; i++) {
                const q = 3 * ch.nodes[i], o = opts[i][idx[i]];
                const src = o === FROM_SOLVE ? plain : (o === FROM_VOLUME ? V : points3d);
                const sq = o === FROM_VOLUME ? voff + q : q;
                if (o === DROP) { work[q] = NaN; work[q + 1] = NaN; work[q + 2] = NaN; }
                else { work[q] = src[sq]; work[q + 1] = src[sq + 1]; work[q + 2] = src[sq + 2]; }
                if (o !== KEEP) changed++;
                if (o === DROP) drops++;
                else if (o !== KEEP && finite3(points3d, q)) moved += dist3(src, sq, points3d, q);
            }
            const bad = chainRepairBadness(work, 0, ch, tol);
            const better = !best || bad < best.bad - 1e-9 || (Math.abs(bad - best.bad) <= 1e-9 &&
                (changed < best.changed || (changed === best.changed &&
                    (drops < best.drops || (drops === best.drops && moved < best.moved)))));
            if (better) best = { bad: bad, changed: changed, drops: drops, moved: moved, pick: Int32Array.from(idx) };
            for (let i = 0; i < K; i++) { if (++idx[i] < opts[i].length) break; idx[i] = 0; }
        }
        if (!best || best.bad > 1e-9 || !best.changed) continue;   // cannot be made plausible: leave it
        for (let i = 0; i < K; i++) {
            const q = 3 * ch.nodes[i], o = opts[i][best.pick[i]];
            if (o === KEEP) continue;
            if (o === DROP) { points3d[q] = NaN; points3d[q + 1] = NaN; points3d[q + 2] = NaN; out.dropped++; continue; }
            const src = o === FROM_SOLVE ? plain : V, sq = o === FROM_VOLUME ? voff + q : q;
            points3d[q] = src[sq]; points3d[q + 1] = src[sq + 1]; points3d[q + 2] = src[sq + 2];
            if (o === FROM_SOLVE) out.fromSolve++; else out.fromVolume++;
        }
        out.chains++;
    }
    return out;
}
