// pose/cage-outlines.js — the cage's outline (silhouette) traced in each camera
// view, and its on-disk form.
//
// A cage outline is a closed polygon drawn around the cage in ONE view: the
// area of that image the cage covers, outer edge only, no holes. Unlike a plane
// node it has no 3D identity — the vertices of two views' outlines are not the
// same physical points — so it is purely per-view 2D. Its consumer is outside
// the app: the cage-scan tools intersect the views' silhouettes through the
// calibration (a visual hull) and fit the cage's planes to them.
//
// SESSION-scoped, like `planePlacements`: a view belongs to a session. The set
// lives on `Session.cageOutlines` and is written under `metadata.lucid` by
// `import-export/plane-metadata.js` (so every reader that already reads the
// Define Planes state picks it up).
//
// DOM-free and import-free, so Node tests load it directly.
//
// ## On-disk format (read by the cage-scan Python tools — do not change)
//
//     metadata.lucid.cageOutlines = {
//       "version": 1,
//       "views": { "<viewName>": [ { "id": 1, "name": "cage", "points": [[x, y], ...] } ] }
//     }
//
// Points are in video pixels, the same coordinates as keypoints and plane-node
// 2D. A polygon is closed implicitly (the first point is not repeated) and is
// written only with at least 3 points. The union of a view's polygons is the
// cage's silhouette in that view.

/** Format version written under `cageOutlines.version`. */
export const CAGE_OUTLINES_VERSION = 1;

/** Default polygon name. */
export const DEFAULT_OUTLINE_NAME = 'cage';

/** A polygon needs at least this many vertices to be an outline. */
export const MIN_OUTLINE_POINTS = 3;

function finitePoint(p) {
    return Array.isArray(p) && p.length >= 2 &&
        typeof p[0] === 'number' && typeof p[1] === 'number' &&
        Number.isFinite(p[0]) && Number.isFinite(p[1]);
}

/**
 * One view's outlines, plus the id counter they share across views (ids are
 * unique within a set, never reused).
 */
export class CageOutlineSet {
    constructor() {
        /** @type {Map<string, Array<{id:number, name:string, points:number[][]}>>} */
        this.views = new Map();
        this._nextId = 1;
    }

    /** True when no view holds a polygon. */
    isEmpty() {
        for (const list of this.views.values()) if (list.length) return false;
        return true;
    }

    /** Polygons on a view (the live array; empty when none). */
    polygons(viewName) {
        return this.views.get(viewName) || [];
    }

    /** Total vertex count on a view. */
    vertexCount(viewName) {
        return this.polygons(viewName).reduce(function (n, p) { return n + p.points.length; }, 0);
    }

    /** Names of views holding at least one polygon, in insertion order. */
    viewNames() {
        const out = [];
        for (const [v, list] of this.views) if (list.length) out.push(v);
        return out;
    }

    getPolygon(viewName, id) {
        const list = this.polygons(viewName);
        for (let i = 0; i < list.length; i++) if (list[i].id === id) return list[i];
        return null;
    }

    /**
     * Add a closed polygon. Points are copied; non-finite points are refused
     * (the whole polygon, so a NaN can never become a vertex).
     * @returns {Object|null} the polygon, or null when it is not a valid outline.
     */
    addPolygon(viewName, points, name) {
        if (typeof viewName !== 'string' || !viewName) return null;
        if (!Array.isArray(points) || points.length < MIN_OUTLINE_POINTS) return null;
        if (!points.every(finitePoint)) return null;
        const poly = {
            id: this._nextId++,
            name: (typeof name === 'string' && name.trim()) ? name.trim() : DEFAULT_OUTLINE_NAME,
            points: points.map(function (p) { return [p[0], p[1]]; }),
        };
        if (!this.views.has(viewName)) this.views.set(viewName, []);
        this.views.get(viewName).push(poly);
        return poly;
    }

    /** Restore path: keep the file's id (and advance the counter past it). */
    _adoptPolygon(viewName, poly) {
        if (!this.views.has(viewName)) this.views.set(viewName, []);
        this.views.get(viewName).push(poly);
        if (poly.id >= this._nextId) this._nextId = poly.id + 1;
    }

    removePolygon(viewName, id) {
        const list = this.views.get(viewName);
        if (!list) return false;
        const i = list.findIndex(function (p) { return p.id === id; });
        if (i < 0) return false;
        list.splice(i, 1);
        if (!list.length) this.views.delete(viewName);
        return true;
    }

    /** Remove every polygon on a view. @returns {boolean} whether anything went. */
    clearView(viewName) {
        const had = this.polygons(viewName).length > 0;
        this.views.delete(viewName);
        return had;
    }

    /** Replace a view's outlines with one polygon (Retrace). */
    replaceView(viewName, points, name) {
        const prev = this.views.get(viewName);
        this.views.delete(viewName);
        const poly = this.addPolygon(viewName, points, name);
        if (!poly && prev) this.views.set(viewName, prev);
        return poly;
    }

    moveVertex(viewName, id, index, x, y) {
        const poly = this.getPolygon(viewName, id);
        if (!poly || index < 0 || index >= poly.points.length) return false;
        if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
        poly.points[index] = [x, y];
        return true;
    }

    /** Insert a vertex AFTER `index` (on the edge index → index+1). */
    insertVertex(viewName, id, index, x, y) {
        const poly = this.getPolygon(viewName, id);
        if (!poly || index < 0 || index >= poly.points.length) return false;
        if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
        poly.points.splice(index + 1, 0, [x, y]);
        return true;
    }

    /** Delete a vertex; refused when it would leave fewer than 3. */
    deleteVertex(viewName, id, index) {
        const poly = this.getPolygon(viewName, id);
        if (!poly || index < 0 || index >= poly.points.length) return false;
        if (poly.points.length <= MIN_OUTLINE_POINTS) return false;
        poly.points.splice(index, 1);
        return true;
    }
}

/**
 * The polygon vertex nearest `(x, y)` on a view, within `maxDist`.
 * @returns {{polyId:number, index:number, distance:number}|null}
 */
export function nearestOutlineVertex(set, viewName, x, y, maxDist) {
    if (!set) return null;
    let best = null;
    const polys = set.polygons(viewName);
    for (const poly of polys) {
        for (let i = 0; i < poly.points.length; i++) {
            const d = Math.hypot(poly.points[i][0] - x, poly.points[i][1] - y);
            if (d <= maxDist && (!best || d < best.distance)) {
                best = { polyId: poly.id, index: i, distance: d };
            }
        }
    }
    return best;
}

/**
 * The polygon EDGE nearest `(x, y)`, within `maxDist`. `index` is the edge's
 * first vertex (the edge runs index → index+1, wrapping); `point` is the
 * nearest point on it, which is where an inserted vertex goes.
 * @returns {{polyId:number, index:number, distance:number, point:number[]}|null}
 */
export function nearestOutlineEdge(set, viewName, x, y, maxDist) {
    if (!set) return null;
    let best = null;
    for (const poly of set.polygons(viewName)) {
        const n = poly.points.length;
        for (let i = 0; i < n; i++) {
            const a = poly.points[i], b = poly.points[(i + 1) % n];
            const dx = b[0] - a[0], dy = b[1] - a[1];
            const len2 = dx * dx + dy * dy;
            let t = len2 > 0 ? ((x - a[0]) * dx + (y - a[1]) * dy) / len2 : 0;
            t = Math.max(0, Math.min(1, t));
            const px = a[0] + t * dx, py = a[1] + t * dy;
            const d = Math.hypot(px - x, py - y);
            if (d <= maxDist && (!best || d < best.distance)) {
                best = { polyId: poly.id, index: i, distance: d, point: [px, py] };
            }
        }
    }
    return best;
}

/**
 * The on-disk form, or null when there is nothing to write (never a default).
 * Only polygons with at least 3 finite points are written.
 * @param {CageOutlineSet|null|undefined} set
 * @returns {{version:number, views:Object}|null}
 */
export function serializeCageOutlines(set) {
    if (!set || !(set.views instanceof Map)) return null;
    const views = {};
    let any = false;
    for (const [viewName, list] of set.views) {
        const polys = [];
        for (const poly of list) {
            if (!poly || !Array.isArray(poly.points)) continue;
            if (poly.points.length < MIN_OUTLINE_POINTS || !poly.points.every(finitePoint)) continue;
            polys.push({
                id: poly.id,
                name: poly.name || DEFAULT_OUTLINE_NAME,
                points: poly.points.map(function (p) { return [p[0], p[1]]; }),
            });
        }
        if (polys.length) { views[viewName] = polys; any = true; }
    }
    return any ? { version: CAGE_OUTLINES_VERSION, views: views } : null;
}

/**
 * Rebuild a set from `serializeCageOutlines` output. Tolerates absence and
 * garbage: a polygon with fewer than 3 valid points, or any non-finite point,
 * is dropped; a missing or duplicate id gets a fresh one (an outline id is not
 * a reference target anywhere else, so renumbering cannot misattach anything).
 * Always returns a set (empty when there is nothing).
 * @param {*} data
 * @returns {CageOutlineSet}
 */
export function restoreCageOutlines(data) {
    const set = new CageOutlineSet();
    if (!data || typeof data !== 'object' || !data.views || typeof data.views !== 'object' ||
        Array.isArray(data.views)) return set;
    const seen = new Set();
    const pending = [];
    for (const viewName of Object.keys(data.views)) {
        const list = data.views[viewName];
        if (!Array.isArray(list)) continue;
        for (const d of list) {
            if (!d || typeof d !== 'object' || !Array.isArray(d.points)) continue;
            if (d.points.length < MIN_OUTLINE_POINTS || !d.points.every(finitePoint)) continue;
            const poly = {
                id: (Number.isInteger(d.id) && d.id > 0 && !seen.has(d.id)) ? d.id : null,
                name: (typeof d.name === 'string' && d.name.trim()) ? d.name.trim() : DEFAULT_OUTLINE_NAME,
                points: d.points.map(function (p) { return [p[0], p[1]]; }),
            };
            if (poly.id !== null) seen.add(poly.id);
            pending.push([viewName, poly]);
        }
    }
    // Ids first (so a fresh id never collides with one later in the file).
    for (const [viewName, poly] of pending) if (poly.id !== null) set._adoptPolygon(viewName, poly);
    for (const [viewName, poly] of pending) {
        if (poly.id === null) { poly.id = set._nextId++; set._adoptPolygon(viewName, poly); }
    }
    // Keep each view's polygons in file order.
    if (pending.length) {
        const order = new Map(pending.map(function (e, i) { return [e[1], i]; }));
        for (const list of set.views.values()) list.sort(function (a, b) { return order.get(a) - order.get(b); });
    }
    return set;
}

/**
 * A view's outline being traced (not yet part of the set): an open polyline
 * the user closes. DOM-free so the gesture rules are testable.
 */
export class OutlineTrace {
    constructor(viewName) {
        this.viewName = viewName;
        /** @type {number[][]} */
        this.points = [];
    }
    add(x, y) {
        if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
        this.points.push([x, y]);
        return true;
    }
    undo() { return this.points.length ? this.points.pop() : null; }
    canClose() { return this.points.length >= MIN_OUTLINE_POINTS; }
    /** Is `(x, y)` within `r` of the first vertex (a click there closes it)? */
    nearFirst(x, y, r) {
        if (!this.points.length) return false;
        return Math.hypot(this.points[0][0] - x, this.points[0][1] - y) <= r;
    }
}
