/**
 * plane-state.js — the Define Planes and Set Origin state objects, without the
 * page: `planeState` (the plane model — node pool, planes, per-view 2D — plus
 * the editor's selection and display sizes) and `originState` (the Set Origin
 * wizard's step and the defined origin frame).
 *
 * They were declared in ui/plane-definition.js and ui/origin-definition.js,
 * which re-export them unchanged; they live here so
 * import-export/plane-metadata.js — and through it the command-line LUCID
 * (cli/track.mjs) — can read and write the project's planes without loading
 * the page. Plain objects, one instance each per page.
 *
 * Depends on: pose/plane-data.js
 */
import { PlaneModel } from '../pose/plane-data.js';

export const planeState = {
    /** @type {boolean} True while Defining Plane Mode is active. */
    active: false,
    /**
     * @type {PlaneModel} The whole plane annotation state: the global node
     * pool, the planes, and the per-view 2D. Project-scoped (app-session
     * memory) except the 2D, which is re-attached per session by
     * `planeModel()`.
     */
    model: new PlaneModel(),
    /** @type {number|null} `PlaneSkeleton.id` the editor is editing. */
    selectedPlaneId: null,
    /**
     * @type {number} Node radius in canvas px, SHARED by every plane. One
     * value rather than per-plane: these are reference geometry you size
     * once for legibility against your video, not per-plane styling.
     */
    nodeSize: 5,
    /** @type {number} Edge stroke width in canvas px, shared by every plane. */
    edgeWidth: 2,
    /**
     * @type {number} Plane-corner sphere size in the 3D scene, shared by every
     * plane. Separate from `nodeSize` (canvas px) because the two live in
     * unrelated units, and separate from the viewport's `skeletonNodeSize` so
     * sizing plane corners for legibility never resizes pose nodes.
     */
    nodeSize3d: 4,
    /**
     * @type {number|null} The pool node id the panel has SELECTED, ringed in
     * the 3D view. Transient — a way to find one corner among nine, cleared by
     * the next click anywhere else, so it is neither saved nor remembered.
     * Separate from `expandedNodes`: opening a node's coordinates is a request
     * to READ something, selecting it is a request to be SHOWN where it is.
     */
    selectedNodeId: null,
    /** @type {Set<number>} Plane ids whose placement list is expanded. */
    expanded: new Set(),
    /**
     * @type {Set<number>} Pool node ids whose coordinate panel is open. Not
     * persisted — which rows you left open is a property of looking at the
     * panel, not of the project, and restoring nine open nodes on load would
     * hand back the clutter the collapsed list exists to remove.
     */
    expandedNodes: new Set(),
};

export const originState = {
    /** @type {boolean} True while Set Origin Mode is active. */
    active: false,
    /** @type {'node'|'axis'|'confirm'} Wizard step; meaningless when inactive. */
    step: 'node',
    /** @type {number|null} Plane the picked node belongs to. */
    planeId: null,
    /** @type {number|null} Index of the picked node, in PLANE order. */
    nodeIdx: null,
    /** @type {number[]|null} The picked node, in old-world coordinates. */
    originPoint: null,
    /** @type {number[]|null} The fitted plane's normal at pick time. */
    normal: null,
    /** @type {number} Candidate-arrow length, scaled to the plane's extent. */
    arrowLength: 0,
    /** @type {'positive'|'negative'|null} Which arrow is chosen. */
    chosen: null,
    /**
     * @type {Object|null} The APPLIED frame from `buildOriginFrame`, or null
     * while the calibration's own frame is in force. Survives leaving the mode
     * — that is the point of the whole exercise.
     */
    frame: null,
};
