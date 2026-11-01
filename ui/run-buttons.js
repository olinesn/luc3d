/**
 * run-buttons.js — the toolbar's Track and Triangulate buttons turn blue
 * (`.active`) once that step has been run on the session and while its result
 * still stands (`pose/run-state.js`), as Smooth does while smoothing is
 * applied (`ui/smoothing-button.js`); Behavior while its predictions stand
 * (some predicted bout, not stale). Derived on every overlay draw
 * (`ui/rendering.js`), so a session switch or a result that goes away
 * (re-tracking replaces Triangulate's 3D) needs no bookkeeping here.
 *
 * Depends on: ui/app-state.js, ui/behavior-detect.js, pose/run-state.js
 */

import { state } from './app-state.js';
import { detectionEnv, ensurePredictionInputChecked } from './behavior-detect.js';
import { hasTrackResult, hasTriangulateResult, hasBehaviorResult } from '../pose/run-state.js';

/** Re-derive the buttons' blue state from the active session. */
export function syncRunButtons() {
    var session = state.session;
    var track = document.getElementById('tbTrackAll');
    if (track) track.classList.toggle('active', hasTrackResult(session));
    var tri = document.getElementById('tbTriangulateAll');
    if (tri) tri.classList.toggle('active', hasTriangulateResult(session));
    var beh = document.getElementById('tbBehavior');
    if (beh) {
        // Predictions read from a file take the poses as loaded as their
        // reference: this only schedules that print, once.
        ensurePredictionInputChecked(session);
        beh.classList.toggle('active', hasBehaviorResult(session, detectionEnv()));
    }
}
