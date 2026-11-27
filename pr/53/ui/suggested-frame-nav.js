// ui/suggested-frame-nav.js — Previous / Next suggested frame (Opt+Shift+← / →,
// `prevSuggestedFrame` / `nextSuggestedFrame` in ACTION_CATALOG, ui/settings.js):
// steps through the frames a `.slp`'s suggestions name, in the file's order, and
// keeps their faint ticks on the seekbar (ui/seekbar-markers.js).
//
// A frame is suggested when any camera of the active session has a suggestion on
// it. The rows come from import-export/kept-suggestions.js (`suggestedFrameRows`,
// eager and lazy sessions alike); their order and the step rule are
// pose/suggested-frames.js. The status line says "Suggestion k of n".
//
// Nothing here is saved: the suggestion last stepped to is remembered per session,
// in memory only, so a step from a frame next to it carries on from it.

import { state } from './app-state.js?v=f2b1ad4c5abb';
import { setStatus } from '../import-export/save-load.js?v=f2b1ad4c5abb';
import { suggestedFrameRows } from '../import-export/kept-suggestions.js?v=f2b1ad4c5abb';
import { suggestedFrameOrder, stepSuggestedFrame } from '../pose/suggested-frames.js?v=f2b1ad4c5abb';
import { formatFrameNumber } from './frame-readout.js?v=f2b1ad4c5abb';
import { setSeekbarSuggestedFrames } from './seekbar-markers.js?v=f2b1ad4c5abb';

var _lastStepped = new WeakMap();   // Session -> the suggested frame last stepped to

/** The session's suggested frames, 0-based, in the file's order, each once. */
export function suggestedFramesOf(session) {
    if (!session || !session.cameras) return [];
    return suggestedFrameOrder(suggestedFrameRows(session), session.cameras.map(function (c) { return c.name; }));
}

/**
 * Step to the previous (-1) or next (+1) suggested frame of the active session.
 *
 * @param {number} direction
 * @param {function(number)} goTo  shows a frame (ui/ui-wiring.js `jumpToFrame`)
 * @returns {{index: number, frame: number}|null} where it went, or null if it did not move
 */
export function stepToSuggestedFrame(direction, goTo) {
    var session = state.session;
    if (!session) return null;
    var frames = suggestedFramesOf(session);
    var step = stepSuggestedFrame(frames, state.currentFrame, direction, _lastStepped.get(session));
    if (!step) {
        setStatus('No suggested frames in this session', 'info');
        return null;
    }
    var n = formatFrameNumber(frames.length);
    if (step.frame == null) {
        setStatus(direction > 0 ? 'No next suggestion: ' + n + ' of ' + n + ' is the last'
            : 'No previous suggestion: 1 of ' + n + ' is the first', 'info');
        return null;
    }
    var k = formatFrameNumber(step.index + 1), shown = formatFrameNumber(step.frame + 1);
    // Remembered either way, so the next step carries on from it.
    _lastStepped.set(session, step.frame);
    // A frame the session cannot show (no video loaded, or a shorter one): say so
    // rather than land on another frame.
    var total = state.totalFrames || 0;
    if (!(step.frame < total)) {
        setStatus('Suggestion ' + k + ' of ' + n + ' is frame ' + shown + (total > 0
            ? ", past this session's last frame (" + formatFrameNumber(total) + ')'
            : ', but this session has no frames to show'), 'warning');
        return null;
    }
    goTo(step.frame);
    setStatus('Suggestion ' + k + ' of ' + n + ': frame ' + shown, 'info');
    return step;
}

/** Tick the active session's suggested frames on the seekbar (a no-op when they are the ones shown). */
export function refreshSuggestedFrameMarks() {
    setSeekbarSuggestedFrames(suggestedFramesOf(state.session), state.totalFrames);
}
