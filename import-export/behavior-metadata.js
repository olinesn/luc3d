// import-export/behavior-metadata.js — behavior labels in the project file:
// the ethogram and each session's labelled bouts, so a reopened project shows
// exactly what was coded.
//
// ## What is stored (all under `metadata.lucid`, only when non-empty)
//
//   ethogram        project-scoped: the same payload in EVERY session's dict
//                   (as `planeNodes` is), read from the first session ingested
//                   (`ingestEthogram`; later sessions of the same project map
//                   onto it by name, a no-op). See pose/behavior-ethogram.js.
//     { v: 1, behaviors: [{ id, name, color, kind, key?, social?, group?, description? }] }
//
//   behaviorLabels  per session: every bout, plus the frame rate they were coded
//                   at when known (frames are the truth; fps is not saved in
//                   the `.slp`, so exports need to know what seconds meant).
//     { v: 1, fps?, bouts: [[behaviorId, animal, target, start, end, present, source, note], …] }
//
//   behaviorPredictions  per session: each behavior's last detection run, its
//                   bouts and their scores (per-frame scores stay in memory),
//                   with a print of the detector and settings that made it, so
//                   a reopened project can tell they no longer match
//                   (pose/behavior-predictions.js). A classifier's run names
//                   its forest by hash and keeps the print of the labels it
//                   learned from (`labels`); the forest itself is not saved.
//     { v: 1, runs: [{ behavior, source: 'rule'|'model:<hash>'|'states:<layer>', cfg, labels?, states?,
//                      mode, camera?, fps?, bouts: [[animal, target, start, end, score], …] }] }
//
//   behaviorStates  per session: layers of behavioral states per animal — an
//                   imported keypoint-MoSeq model's syllables, as runs, and the
//                   names given to them (pose/behavior-states.js).
//     { v: 1, layers: [{ name, source: 'keypoint-MoSeq'|'discovery', stateNames: {state: name},
//                        animals: {name: [[start, end, state], …]} }] }
//
// The ethogram carries the detectors (`behaviors[].detector`), each
// classifier's post-processing (`behaviors[].post`) and the feature settings
// (`features`), each only once set (pose/behavior-ethogram.js).
// A project nobody labelled behaviors in writes none of these keys, so its bytes are
// unchanged (tests/e2e/save-golden-digest.mjs). A running state (Label
// Behaviors mode) is saved as ending at the playhead without being stopped.
//
// Reading REPLACES the session's bouts (`importBoutRows`) and never merges:
// every reader builds a fresh Session. A session file added to an open
// project (the single-SLP folder loader) brings its ethogram in by name, and
// its bouts are remapped onto the project's behavior ids.
//
// DOM-free: Node tests import it directly.
//
// Depends on: pose/behavior-ethogram.js, pose/behavior-labels.js, pose/behavior-predictions.js,
//             pose/behavior-states.js, pose/behavior-classifier.js

import { serializeEthogram, ingestEthogram, resetEthogram } from '../pose/behavior-ethogram.js';
import { exportBoutRows, importBoutRows, behaviorFps, noteBehaviorFps } from '../pose/behavior-labels.js';
import { exportPredictionRuns, importPredictionRuns } from '../pose/behavior-predictions.js';
import { resetBehaviorModels, noteLabelsChanged } from '../pose/behavior-classifier.js';
import { exportStateLayers, importStateLayers, resetBehaviorStates } from '../pose/behavior-states.js';

/** Every `metadata.lucid` key this module may write. */
export var BEHAVIOR_METADATA_KEYS = ['ethogram', 'behaviorLabels', 'behaviorPredictions', 'behaviorStates'];

/**
 * Write the ethogram and the session's bouts into `lucid`, each only when it
 * has something in it.
 * @param {Object} lucid the session's `metadata.lucid` dict (mutated)
 * @param {Session} session
 */
export function writeBehaviorMetadata(lucid, session) {
    if (!lucid) return;
    var eth = serializeEthogram();
    if (eth) lucid.ethogram = eth;
    var preds = exportPredictionRuns(session);
    if (preds) lucid.behaviorPredictions = preds;
    var states = exportStateLayers(session);
    if (states) lucid.behaviorStates = states;
    var rows = exportBoutRows(session);
    if (!rows.length) return;
    var out = { v: 1, bouts: rows };
    var fps = behaviorFps(session);
    if (fps > 0) out.fps = fps;
    lucid.behaviorLabels = out;
}

/**
 * Read the ethogram (into the project) and the session's bouts back.
 * Tolerates absence and garbage.
 * @param {Session} session
 * @param {Object} lucid
 * @returns {{kept:number, dropped:number}} bouts read and bouts skipped
 */
export function readBehaviorMetadata(session, lucid) {
    if (!session || !lucid || typeof lucid !== 'object') return { kept: 0, dropped: 0 };
    var idMap = lucid.ethogram ? ingestEthogram(lucid.ethogram) : new Map();
    // The session's behavioral states (keypoint-MoSeq syllables), replaced like its bouts.
    var sl = importStateLayers(session, lucid.behaviorStates);
    if (sl.dropped) {
        console.warn('[behavior] ' + sl.dropped + ' saved state run' + (sl.dropped === 1 ? '' : 's') + ' or layer' +
            (sl.dropped === 1 ? '' : 's') + ' in session "' + (session.name || '') + '" were malformed, and were skipped');
    }
    var pr = importPredictionRuns(session, lucid.behaviorPredictions, idMap);
    if (pr.dropped) {
        console.warn('[behavior] ' + pr.dropped + ' saved predicted bout' + (pr.dropped === 1 ? '' : 's') +
            ' in session "' + (session.name || '') + '" named no known behavior or were malformed, and were skipped');
    }
    var bl = lucid.behaviorLabels;
    noteLabelsChanged();
    if (!bl || typeof bl !== 'object' || bl.v !== 1 || !Array.isArray(bl.bouts)) return { kept: 0, dropped: 0 };
    var res = importBoutRows(session, bl.bouts, idMap);
    if (typeof bl.fps === 'number' && bl.fps > 0) noteBehaviorFps(session, bl.fps);
    if (res.dropped) {
        console.warn('[behavior] ' + res.dropped + ' saved bout' + (res.dropped === 1 ? '' : 's') +
            ' in session "' + (session.name || '') + '" named no known behavior or were malformed, and were skipped');
    }
    return res;
}

/**
 * Forget the project's ethogram, its trained classifiers and every session's
 * behavioral states. The first two are project-scoped and live on module
 * singletons, so they survive `state.sessions = []`: call this before every
 * load and on New Project, or the next project opens with this one's
 * behaviors and models. (Bouts live on their Session and go with it;
 * classifiers are never saved — the labels are, and retraining is seeded;
 * every reader reads each session's states back.)
 */
export function resetBehaviorState() {
    resetEthogram();
    resetBehaviorModels();
    resetBehaviorStates();
}
