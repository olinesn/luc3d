// ui/threshold-examples-3d.js — the 3D view in the threshold card's "In this
// session" panel (ui/threshold-examples-panel.js): the animal's 3D skeleton from the
// solve being tried, seen from a viewpoint that can be dragged, with the example
// node's 3D point (the projection) and each camera's line of sight through it.
//
// An orthographic drawing on a 2D canvas, Z up as in the 3D viewport: a small
// figure that repaints on every slider tick has no use for a WebGL context of its
// own. The framing comes from the caller and holds still, so what moves as a value
// is tried is the point, not the view.
//
// Depends on: ui/overlays.js (drawSkeleton, the Predictions drawing).

import { drawSkeleton } from './overlays.js';

const RAY = 'rgba(200, 200, 200, 0.45)', RAY_DROPPED = '#f39c12', GRID = 'rgba(255, 255, 255, 0.09)';
const POINT_FILL = 'rgba(102, 126, 234, 0.45)', FIRST_SOLVE = '#b5b5b5';

/** The middle of a flat [x,y,z,...] list's solved points, and how far the furthest is. */
export function frameFromPoints(flat) {
    const axes = [[], [], []];
    for (let i = 0; i + 2 < flat.length; i += 3) {
        if (Number.isFinite(flat[i]) && Number.isFinite(flat[i + 1]) && Number.isFinite(flat[i + 2])) {
            for (let a = 0; a < 3; a++) axes[a].push(flat[i + a]);
        }
    }
    if (!axes[0].length) return null;
    const centre = axes.map(function (v) { v.sort(function (p, q) { return p - q; }); return v[v.length >> 1]; });
    let radius = 0, floor = Infinity;
    for (let i = 0; i < axes[0].length; i++) floor = Math.min(floor, axes[2][i]);
    for (let i = 0; i + 2 < flat.length; i += 3) {
        if (!Number.isFinite(flat[i])) continue;
        radius = Math.max(radius, Math.hypot(flat[i] - centre[0], flat[i + 1] - centre[1], flat[i + 2] - centre[2]));
    }
    return { centre: centre, radius: Math.max(radius, 1e-6), floor: floor };
}

/** World to canvas, orthographic: turn about Z by `yaw`, then tip by `pitch`. */
export function makeProjector(frame, size, yaw, pitch) {
    const scale = 0.46 * size / frame.radius;
    const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    return function (p) {
        const dx = p[0] - frame.centre[0], dy = p[1] - frame.centre[1], dz = p[2] - frame.centre[2];
        const x = cy * dx - sy * dy, y = sy * dx + cy * dy;
        return { x: size / 2 + scale * x, y: size / 2 - scale * (dz * cp + y * sp) };
    };
}

/**
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} size  the canvas side
 * @param {Object} scene
 * @param {{centre: number[], radius: number, floor: number}} scene.frame
 * @param {Float64Array|number[]} scene.points  flat [x,y,z,...] of the solve being tried
 * @param {number} scene.node  the example's node
 * @param {?number[]} scene.first  the node's point in the first solve, if a camera was dropped
 * @param {Array<{origin: number[], dir: number[], dropped: boolean}>} scene.rays
 * @param {Object} scene.skeleton  with `.edges`
 * @param {Object} scene.style  `drawSkeleton` options, sizes in this canvas's pixels
 * @param {number} scene.yaw
 * @param {number} scene.pitch
 */
export function draw3dScene(ctx, size, scene) {
    const toCanvas = makeProjector(scene.frame, size, scene.yaw, scene.pitch);
    const f = scene.frame, R = f.radius;
    ctx.fillStyle = '#111';
    ctx.fillRect(0, 0, size, size);

    // The floor under the animal, to give the turn something to read against.
    ctx.strokeStyle = GRID; ctx.lineWidth = 1.5;
    for (let i = -4; i <= 4; i++) {
        const t = i * R / 4;
        [[[f.centre[0] + t, f.centre[1] - R], [f.centre[0] + t, f.centre[1] + R]],
            [[f.centre[0] - R, f.centre[1] + t], [f.centre[0] + R, f.centre[1] + t]]].forEach(function (seg) {
            const a = toCanvas([seg[0][0], seg[0][1], f.floor]), b = toCanvas([seg[1][0], seg[1][1], f.floor]);
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        });
    }

    const k = scene.node, at = [scene.points[3 * k], scene.points[3 * k + 1], scene.points[3 * k + 2]];
    const has = at.every(Number.isFinite);
    const anchor = has ? at : scene.first;

    // Each line of sight, past the point on both sides.
    if (anchor) {
        ctx.lineWidth = 2;
        ctx.setLineDash([8, 7]);
        scene.rays.forEach(function (ray) {
            const t0 = (anchor[0] - ray.origin[0]) * ray.dir[0] + (anchor[1] - ray.origin[1]) * ray.dir[1] + (anchor[2] - ray.origin[2]) * ray.dir[2];
            const along = function (t) {
                return toCanvas([ray.origin[0] + ray.dir[0] * t, ray.origin[1] + ray.dir[1] * t, ray.origin[2] + ray.dir[2] * t]);
            };
            const a = along(t0 - 3 * R), b = along(t0 + 3 * R);
            ctx.strokeStyle = ray.dropped ? RAY_DROPPED : RAY;
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        });
        ctx.setLineDash([]);
    }

    const count = Math.floor(scene.points.length / 3);
    drawSkeleton(ctx, {
        numNodes: count,
        nulledNodes: null,
        hasPoint: function (i) { return Number.isFinite(scene.points[3 * i]); },
        getX: function (i) { return toCanvas([scene.points[3 * i], scene.points[3 * i + 1], scene.points[3 * i + 2]]).x; },
        getY: function (i) { return toCanvas([scene.points[3 * i], scene.points[3 * i + 1], scene.points[3 * i + 2]]).y; },
    }, scene.skeleton, scene.style);

    if (scene.first && has) {
        const a = toCanvas(scene.first), b = toCanvas(at);
        ctx.strokeStyle = FIRST_SOLVE; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        ctx.beginPath(); ctx.arc(a.x, a.y, 9, 0, 2 * Math.PI); ctx.stroke();
    }
    if (has) {
        const p = toCanvas(at);
        ctx.fillStyle = POINT_FILL; ctx.strokeStyle = '#fff'; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(p.x, p.y, 11, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    }
}
