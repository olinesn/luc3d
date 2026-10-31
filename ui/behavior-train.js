// ui/behavior-train.js — behavior classifiers in the app: training them on
// the project's labels (with the progress card), the active-learning loop
// (Next uncertain ▸ / Next disagreement ▸, then Accept / Reject / Skip),
// changing a classifier's threshold and durations (its bouts re-derived at
// once from the scores in memory), and classifier files (Export / Import).
//
// Training reads every session with labels: the active one from the cached
// context, the others built one at a time (pose/behavior-classifier.js holds
// one session's poses at a time). On a lazy multi-session project the
// sessions not shown have no poses resident; they train from the rows cached
// while they were shown, or are named as left out.
//
// The active-learning queue works on the active session's latest prediction
// of a behavior (its per-frame scores stay in memory after a run): uncertain
// stretches are unlabelled frames scoring near the threshold, most uncertain
// first; disagreements are labelled frames the predicted bouts contradict,
// longest first. Accept labels the stretch present, Reject labels it "not"
// (each one undo step), and both move on — the next training learns from them.
//
// Depends on: ui/app-state.js, ui/loading-overlay.js, ui/behavior-detect.js,
//             ui/behavior-coding.js, pose/behavior-ethogram.js,
//             pose/behavior-labels.js, pose/behavior-classifier.js,
//             pose/behavior-predictions.js, import-export/behavior-model-io.js,
//             loading/cold-read-guard.js, import-export/save-load.js

import { state } from './app-state.js';
import { showLoading, showLoadingProgress, hideLoading, yieldToPaint } from './loading-overlay.js';
import { buildBehaviorContext, cachedBehaviorContext, detectionEnv, detectionFps } from './behavior-detect.js';
import { requestBehaviorRepaint, setReviewFocus, goToBehaviorFrame, BOUT_LEAD_IN_SECONDS, behaviorUi } from './behavior-coding.js';
import { behaviors, behaviorById, setBehaviorPost, exclusivePeers } from '../pose/behavior-ethogram.js';
import { labelBouts } from '../pose/behavior-labels.js';
import {
    trainBehaviorClassifiers, setBehaviorModel, behaviorModelOf, behaviorModels, labelCounts, gateProblems,
    uncertainSegments, disagreementSegments, deriveBouts, modelBoutSettings,
} from '../pose/behavior-classifier.js';
import { behaviorRunOf, replaceRunBouts, runConfigPrint, runModelHash } from '../pose/behavior-predictions.js';
import { buildModelFile, parseModelFile, modelFileProblems, adoptModelFile, modelFileName, MODEL_FILE_SUFFIX } from '../import-export/behavior-model-io.js';
import { readBlobArrayBuffer } from '../loading/cold-read-guard.js';
import { setStatus, markDirty } from '../import-export/save-load.js';

function n(x) { return Number(x || 0).toLocaleString('en-US'); }

function sessionName(s, i) {
    return (s && s.name) || 'session ' + (i + 1);
}

// ----------------------------------------------------------------------------
// What can train
// ----------------------------------------------------------------------------

/**
 * Every behavior with its labels over the project (frames and bouts per
 * class, from the labels alone), what the gate still needs, and its model.
 * @returns {Array<{behavior, counts, gate: string[], model: ?Object}>}
 */
export function classifierStatus() {
    var sessions = state.sessions || [];
    return behaviors().map(function (b) {
        var counts = labelCounts(sessions, b.id);
        return { behavior: b, counts: counts, gate: gateProblems(counts), model: behaviorModelOf(b.id) };
    });
}

var _training = false;
var _lastReports = new Map();

/** Is a training run going on? */
export function isTraining() {
    return _training;
}

var _switchSession = null;

/** ui-wiring hands over how to switch sessions (the train dialog offers to open a left-out one). */
export function setClassifierSessionSwitcher(fn) {
    _switchSession = typeof fn === 'function' ? fn : null;
}

/**
 * Open a session by name, so its poses are read and the next training can
 * include it (a lazy project keeps only the shown session's poses).
 * @returns {Promise<boolean>} whether it is now the active session
 */
export async function openSessionByName(name) {
    var i = (state.sessions || []).findIndex(function (s, k) { return sessionName(s, k) === name; });
    if (i < 0 || !_switchSession) return false;
    await _switchSession(i);
    return state.activeSessionIdx === i;
}

/** The report of a behavior's last training in this page (null after a reload or an import). */
export function lastTrainingReport(behaviorId) {
    return _lastReports.get(behaviorId) || null;
}

var STAGE_LABEL = { read: 'Reading labelled sessions', rows: 'Measuring labelled frames', bins: 'Preparing', trees: 'Growing trees' };

/**
 * Train classifiers for some behaviors on every labelled session, with
 * cross-validation, and keep them as those behaviors' classifiers.
 * @param {string[]} behaviorIds
 * @returns {Promise<Array<{behaviorId, model?, report, error?}>|null>}
 */
export async function trainClassifiers(behaviorIds) {
    if (!state.session) { setStatus('Load a session first'); return null; }
    if (_training) return null;
    var ids = (behaviorIds || []).filter(function (id) { return !!behaviorById(id); });
    if (!ids.length) { setStatus('Choose the behaviors to train'); return null; }
    _training = true;
    var title = 'Train Classifiers';
    var names = ids.map(function (id) { return behaviorById(id).name; });
    try {
        showLoading('Reading poses…', { title: title });
        await yieldToPaint();
        var sessions = state.sessions && state.sessions.length ? state.sessions : [state.session];
        var sources = sessions.map(function (s, i) {
            return {
                session: s, name: sessionName(s, i),
                getCtx: function () { return s === state.session ? cachedBehaviorContext(s) : buildBehaviorContext(s); },
            };
        });
        var stageOrder = ['read', 'rows', 'bins', 'trees'];
        var res = await trainBehaviorClassifiers({
            behaviorIds: ids, sources: sources, reference: sessionName(state.session, sessions.indexOf(state.session)),
            allSessions: sessions, pause: yieldToPaint,
            onProgress: function (p) {
                var b = p.behaviorId ? behaviorById(p.behaviorId) : null;
                var step = stageOrder.indexOf(p.stage) + 1;
                showLoadingProgress(STAGE_LABEL[p.stage] + (b ? ' · ' + b.name : ''), p.done, p.total, {
                    title: title, unit: p.stage === 'trees' ? 'trees' : p.stage === 'read' ? 'sessions' : p.stage === 'bins' ? 'features' : 'rows',
                    step: step, steps: stageOrder.length, detail: names.join(', '),
                });
            },
        });
        var ok = [], bad = [];
        res.results.forEach(function (r) {
            var b = behaviorById(r.behaviorId);
            if (r.report) _lastReports.set(r.behaviorId, r.report);
            if (r.model) {
                setBehaviorModel(r.model);
                var cv = r.model.cv && r.model.cv.frame ? ' (F1 ' + fmt(r.model.cv.frame.f1) + ')' : '';
                ok.push(b.name + cv);
            } else bad.push(b.name + ': ' + (r.error || 'not trained'));
        });
        if (res.cancelled) setStatus('Training cancelled', 'warning');
        else {
            var left = [];
            res.results.forEach(function (r) { (r.report && r.report.leftOut || []).forEach(function (lo) { if (left.indexOf(lo.session) < 0) left.push(lo.session); }); });
            setStatus((ok.length ? 'Trained ' + ok.join(', ') : 'Nothing trained') + (bad.length ? ' · ' + bad.join(' · ') : '') +
                (left.length ? ' · Left out: ' + left.join(', ') : '') + (ok.length ? ' · Predict (Behavior ▸ All frames) to see what they find' : ''),
                bad.length || left.length ? 'warning' : 'success');
        }
        requestBehaviorRepaint({ predictions: true, ethogram: true });
        return res.results;
    } catch (e) {
        console.error('[behavior] training failed', e);
        setStatus('Training failed: ' + (e && e.message || e), 'error');
        return null;
    } finally {
        hideLoading();
        _training = false;
    }
}

/** A metric as two decimals, or "–". */
export function fmt(v) {
    return Number.isFinite(v) ? v.toFixed(2) : '–';
}

// ----------------------------------------------------------------------------
// Post-processing
// ----------------------------------------------------------------------------

/**
 * Change how a behavior's classifier scores become bouts, and re-derive the
 * bouts of every run that stands (its scores in memory, its settings
 * current) at once — with its exclusive group's, whose top-behavior choice
 * can change too. Runs without scores in memory go stale.
 * @param {string} behaviorId
 * @param {{threshold:number, minBout:number, maxGap:number}} post
 */
export function applyClassifierPost(behaviorId, post) {
    var b = behaviorById(behaviorId);
    if (!b) return;
    var members = [b].concat(exclusivePeers(b));
    var env = detectionEnv();
    var cfgOf = function (bb, s, run) {
        return runConfigPrint(bb, s, { fps: run.fps, mode: run.mode, camera: run.camera, floorKey: env.floorKey, source: run.source });
    };
    // The model runs that stand now, per session, before anything changes.
    var fresh = (state.sessions || []).map(function (s) {
        return members.map(function (m) {
            var run = behaviorRunOf(s, m.id);
            return run && runModelHash(run) && run.scores && run.cfg === cfgOf(m, s, run) ? run : null;
        });
    });
    setBehaviorPost(behaviorId, post);
    var redone = 0;
    (state.sessions || []).forEach(function (s, si) {
        var runs = fresh[si];
        if (!runs.some(Boolean)) return;
        var scoreMaps = {}, how = {};
        runs.forEach(function (run, k) {
            if (!run) return;
            scoreMaps[members[k].id] = run.scores;
            how[members[k].id] = modelBoutSettings(members[k], run.fps);
        });
        var bouts = deriveBouts(scoreMaps, how);
        runs.forEach(function (run, k) {
            if (!run) return;
            replaceRunBouts(s, members[k].id, bouts[members[k].id], cfgOf(members[k], s, run));
            redone++;
        });
    });
    markDirty();
    requestBehaviorRepaint({ ethogram: true, predictions: true });
    return redone;
}

// ----------------------------------------------------------------------------
// Active learning
// ----------------------------------------------------------------------------

var _al = { behaviorId: '', kind: '', current: null, skipped: new Set(), session: null };

function itemKey(it) {
    return [it.behaviorId, it.animal, it.target, it.start, it.end].join('|');
}

/** The active-learning state: `{behaviorId, kind, current}` (current: the stretch shown, or null). */
export function activeLearning() {
    if (_al.session !== state.session) { _al.current = null; _al.skipped = new Set(); _al.session = state.session; }
    return _al;
}

/** The behavior the queue works on. */
export function setActiveLearningBehavior(id) {
    if (_al.behaviorId === id) return;
    _al.behaviorId = id || '';
    _al.current = null;
    _al.skipped = new Set();
    clearLearningFocus();
    requestBehaviorRepaint({ predictions: true });
}

/** Drop the video box of the stretch shown (only if this queue set it). */
function clearLearningFocus() {
    var r = behaviorUi().review;
    if (r && r.from === 'learning') setReviewFocus(null);
}

function queue(kind) {
    var s = state.session;
    if (!s || !_al.behaviorId) return [];
    var list = kind === 'disagreement' ? disagreementSegments(s, _al.behaviorId) : uncertainSegments(s, _al.behaviorId);
    return list.filter(function (it) { return !_al.skipped.has(itemKey(it)); });
}

/**
 * Show the next stretch to label: the most uncertain unlabelled one, or the
 * longest disagreement between the labels and the predicted bouts.
 * @param {'uncertain'|'disagreement'} kind
 * @returns {Object|null} the stretch
 */
export function nextActiveItem(kind) {
    activeLearning();
    var s = state.session;
    if (!s) return null;
    if (!_al.behaviorId) { setStatus('Choose the behavior to work on'); return null; }
    var b = behaviorById(_al.behaviorId);
    var run = behaviorRunOf(s, _al.behaviorId);
    if (!run || (kind === 'uncertain' && !run.scores)) {
        setStatus((b ? b.name : 'This behavior') + ' has no prediction in memory on this session: predict it first (Behavior ▸ All frames)', 'warning');
        return null;
    }
    _al.kind = kind;
    var list = queue(kind);
    // Move past the one shown.
    if (_al.current) {
        var cur = itemKey(_al.current);
        list = list.filter(function (it) { return itemKey(it) !== cur; });
    }
    _al.current = list[0] || null;
    if (!_al.current) {
        clearLearningFocus();
        setStatus(kind === 'uncertain' ? 'No unlabelled stretch of ' + b.name + ' is near the threshold: train again, or label elsewhere'
            : 'The labels and the predicted bouts of ' + b.name + ' agree here', 'success');
        requestBehaviorRepaint({ predictions: true });
        return null;
    }
    showItem(_al.current);
    return _al.current;
}

function showItem(it) {
    setReviewFocus(state.session, it, 'learning');
    var fps = state.fps > 0 ? state.fps : 30;
    goToBehaviorFrame(Math.max(0, it.start - Math.round(BOUT_LEAD_IN_SECONDS * fps)));
    requestBehaviorRepaint({ predictions: true });
}

/**
 * Label the shown stretch — present (Accept) or "not" (Reject), one undo
 * step — and move on to the next of the same kind.
 */
export function labelActiveItem(present) {
    var it = _al.current, s = state.session;
    if (!it || !s) return null;
    labelBouts(s, [{ behaviorId: it.behaviorId, animal: it.animal, target: it.target, start: it.start, end: it.end, present: !!present }],
        { source: present ? 'a' : 'm' });
    var b = behaviorById(it.behaviorId);
    setStatus((present ? 'Labelled ' : 'Labelled "not" ') + (b ? b.name : '') + ' · ' + it.animal + (it.target ? ' → ' + it.target : '') +
        ' · frames ' + n(it.start + 1) + '–' + n(it.end + 1) + ' (Train again to learn from it)', 'success');
    return nextActiveItem(_al.kind || 'uncertain');
}

/** Leave the shown stretch unlabelled and move on (it is not offered again on this session). */
export function skipActiveItem() {
    if (_al.current) _al.skipped.add(itemKey(_al.current));
    return nextActiveItem(_al.kind || 'uncertain');
}

// ----------------------------------------------------------------------------
// Files
// ----------------------------------------------------------------------------

function downloadText(text, filename) {
    var blob = new Blob([text], { type: 'application/json;charset=utf-8' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
}

/**
 * Behavior ▸ Export Classifiers…: every trained (or imported) classifier,
 * or those of `behaviorIds`, as one `*.lucid-behavior.json`.
 */
export function exportClassifiers(behaviorIds) {
    var models = behaviorModels().filter(function (m) { return !behaviorIds || behaviorIds.indexOf(m.behaviorId) >= 0; })
        .filter(function (m) { return !!behaviorById(m.behaviorId); });
    if (!models.length) { setStatus('No classifier to export: train one first (Behavior ▸ Train Classifiers…)', 'warning'); return null; }
    var file = buildModelFile(models, { fps: detectionFps() });
    var name = modelFileName(models);
    downloadText(JSON.stringify(file), name);
    setStatus('Exported ' + models.length + ' classifier' + (models.length === 1 ? '' : 's') + ' to ' + name, 'success');
    return name;
}

/** Behavior ▸ Import Classifiers…: pick a file, check it fits this session, and use its classifiers. */
export function showImportClassifiers() {
    var input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json,' + MODEL_FILE_SUFFIX;
    input.hidden = true;
    document.body.appendChild(input);
    input.addEventListener('cancel', function () { input.remove(); });
    input.addEventListener('change', function () {
        var f = input.files && input.files[0];
        input.remove();
        if (f) importClassifierFile(f);
    });
    input.click();
}

/**
 * Read a classifier file and use its classifiers on this project, when they
 * fit the active session (same features).
 * @param {File} file
 * @returns {Promise<{imported:number}|{error:string}>}
 */
export async function importClassifierFile(file) {
    var session = state.session;
    if (!session) { setStatus('Load a session first'); return { error: 'no session' }; }
    var json;
    try {
        json = JSON.parse(new TextDecoder('utf-8').decode(new Uint8Array(await readBlobArrayBuffer(file))));
    } catch (e) {
        setStatus(file.name + ': not a classifier file (' + (e && e.message || e) + ')', 'error');
        return { error: 'unreadable' };
    }
    var parsed = parseModelFile(json);
    if (!parsed.models.length) {
        setStatus(file.name + ': ' + (parsed.problems.join('; ') || 'no classifier'), 'error');
        return { error: parsed.problems.join('; ') };
    }
    var ctx = cachedBehaviorContext(session);
    var fit = modelFileProblems(parsed, ctx);
    if (fit.blocking.length) {
        setStatus(file.name + ' does not fit this session: ' + fit.blocking.join('; '), 'error');
        return { error: fit.blocking.join('; ') };
    }
    var models = adoptModelFile(parsed);
    var clash = [];
    models.forEach(function (m) {
        var b = behaviorById(m.behaviorId);
        if (b && (!!b.social !== !!m.social || !!b.mutual !== !!m.mutual)) { clash.push(b.name); return; }
        setBehaviorModel(m);
    });
    markDirty();
    requestBehaviorRepaint({ ethogram: true, predictions: true });
    var used = models.length - clash.length;
    setStatus('Imported ' + used + ' classifier' + (used === 1 ? '' : 's') + ' from ' + file.name +
        (clash.length ? ' · Not used (social or mutual differs from the ethogram): ' + clash.join(', ') : '') +
        (parsed.problems.length ? ' · Skipped: ' + parsed.problems.join('; ') : '') +
        (fit.warnings.length ? ' · Cannot run on this session: ' + fit.warnings.join('; ') : '') + ' · Predict (Behavior ▸ All frames) to use them',
        clash.length || parsed.problems.length || fit.warnings.length ? 'warning' : 'success');
    return { imported: used };
}
