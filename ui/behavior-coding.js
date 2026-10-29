// ui/behavior-coding.js — what Label Behaviors mode (ui/behavior-mode.js),
// the Behavior tab (ui/behavior-panel.js) and the behavior timeline rows
// (ui/behavior-timeline.js) share: the coding state and the coding actions.
//
// Coding state (browser memory only, never saved):
//   focal     the animal a behavior key or button labels (an identity name;
//             digits 1–9 pick it in the mode, the tab's animal buttons too)
//   selected  the behavior the timeline rows show (null = all behaviors)
//   rangeIn / rangeOut   a range armed with `[` and `]`; a behavior then labels
//             the whole range instead of starting / stopping at the playhead
//   pending   a social behavior waiting for its target (a digit or a click on
//             the animal picks it; Esc cancels)
//
// Every edit goes through pose/behavior-labels.js, whose change listener
// (set up in ui/behavior-panel.js) marks the project dirty and repaints.
// UI-only changes (a new focal animal, a selection) repaint through
// `requestBehaviorRepaint` here.
//
// Depends on: ui/app-state.js, pose/behavior-ethogram.js,
//             pose/behavior-labels.js, import-export/save-load.js,
//             ui/id-switch-highlight.js (the node mask boxes share)

import { state } from './app-state.js';
import { behaviorById } from '../pose/behavior-ethogram.js';
import {
    labelRange, clearRange, toggleState, noteBehaviorFps, listBouts, activeAt, hasBehaviorData, runningStates,
} from '../pose/behavior-labels.js';
import { setStatus } from '../import-export/save-load.js';
import { idSwitchBoxNodeMask } from './id-switch-highlight.js';

var ui = { focal: null, selected: null, rangeIn: null, rangeOut: null, pending: null, mode: false };

/** The coding state (read-only for callers; use the setters). */
export function behaviorUi() {
    return ui;
}

// ----------------------------------------------------------------------------
// Repaint bus: the panel, the timeline rows, the overlays and the mode bar
// each register one repainter; requests within a frame coalesce.
// ----------------------------------------------------------------------------

var _repainters = [];
var _pendingInfo = null;

/** Register a repainter: `fn(info)` where info merges every request since the last paint. */
export function registerBehaviorRepaint(fn) {
    if (typeof fn === 'function') _repainters.push(fn);
}

/**
 * Ask for a repaint on the next animation frame.
 * @param {{ui?:boolean, labels?:boolean, ethogram?:boolean, frame?:boolean}} [info]
 */
export function requestBehaviorRepaint(info) {
    var first = !_pendingInfo;
    _pendingInfo = Object.assign(_pendingInfo || {}, info || {});
    if (!first) return;
    var run = function () {
        var i = _pendingInfo; _pendingInfo = null;
        for (var k = 0; k < _repainters.length; k++) {
            try { _repainters[k](i || {}); } catch (e) { console.error('[behavior] repaint failed', e); }
        }
    };
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(run); else run();
}

// ----------------------------------------------------------------------------
// Navigation (registered once by ui-wiring: this module cannot import
// pose/initialization.js)
// ----------------------------------------------------------------------------

var _navigate = null;

/** Register the viewer's `navigateToFrame`. */
export function setBehaviorNavigator(fn) { _navigate = fn; }

/** Seek to a frame. */
export function goToBehaviorFrame(frame) {
    if (_navigate && Number.isFinite(frame)) _navigate(Math.max(0, Math.round(frame)));
}

/** Seconds of lead-in when a bout is opened from the list, as an ID-switch row lands. */
export var BOUT_LEAD_IN_SECONDS = 1;

/** Show the Behavior tab (opening the info panel if it is collapsed). */
export function openBehaviorTab() {
    var wrap = document.getElementById('infoPanelWrapper');
    if (wrap && wrap.classList.contains('collapsed')) {
        var t = document.getElementById('infoPanelToggleBtn');
        if (t) t.click();
    }
    var tab = document.querySelector('.panel-tab[data-tab="tabBehavior"]');
    if (tab && !tab.classList.contains('active')) tab.click();
}

// ----------------------------------------------------------------------------
// Boxes on the video (ui/overlays.js `drawBehaviorBoxes`): around the animals
// engaged in each behavior on the frame — both animals of a social one — so
// who is labelled with what is visible while coding. A browser-local display
// choice ("On video" in the Behavior tab), on by default, never saved.
// ----------------------------------------------------------------------------

var ON_VIDEO_KEY = 'lucid.behaviorOnVideo';

/** Are behavior boxes drawn on the videos? */
export function behaviorBoxesVisible() {
    try { return localStorage.getItem(ON_VIDEO_KEY) !== '0'; } catch (e) { return true; }
}

export function setBehaviorBoxesVisible(on) {
    try { localStorage.setItem(ON_VIDEO_KEY, on ? '1' : '0'); } catch (e) { /* storage unavailable: session only */ }
    requestBehaviorRepaint({ labels: true });
}

/** Label Behaviors mode tells the boxes whether it is on (ui/behavior-mode.js). */
export function noteBehaviorMode(on) {
    ui.mode = !!on;
    requestBehaviorRepaint({ ui: true });
}

/**
 * The `behaviorBoxes` render option for one frame, or null when nothing would
 * be drawn: one item per behavior an animal (with its target) is doing there —
 * dashed while it is still being coded. With `opts.live` (the viewer, never an
 * exported video) also the coding state: in Label Behaviors mode a light box
 * with its number on the animal the keys label, and while a social behavior
 * waits for its target a dashed box on the animal that will be its actor.
 * @param {{live?: boolean}} [opts]
 * @returns {{items: Array<{animals:string[], color:string, dashed:boolean, light?:boolean, parts:Array}>, nodeMask: ?boolean[]}|null}
 */
export function behaviorBoxesAt(session, frame, opts) {
    if (!session || !Number.isFinite(frame) || !behaviorBoxesVisible()) return null;
    var live = !!(opts && opts.live) && session === state.session;
    var items = [];
    var who = function (name) { return [name, animalColor(session, name)]; };
    var focal = live && ui.mode ? focalAnimal(session) : null;
    if (focal) {
        var num = sessionAnimals(session).indexOf(focal) + 1;
        items.push({ animals: [focal], color: 'rgba(255, 255, 255, 0.75)', dashed: false, light: true,
                     parts: (num >= 1 && num <= 9 ? [[num + ' · ', '#ffffff']] : []).concat([who(focal)]) });
    }
    activeAt(session, frame).forEach(function (a) {
        var b = behaviorById(a.behaviorId);
        if (!b) return;
        var parts = [[(a.running ? '● ' : '') + b.name, b.color], [' · ', '#ffffff'], who(a.animal)];
        if (a.target) parts.push([pairGlyph(b), '#ffffff'], who(a.target));
        items.push({ animals: a.target ? [a.animal, a.target] : [a.animal], color: b.color, dashed: !!a.running, parts: parts });
    });
    var p = live ? ui.pending : null;
    var pb = p ? behaviorById(p.behaviorId) : null;
    if (pb) {
        items.push({ animals: [p.animal], color: pb.color, dashed: true,
                     parts: [[pb.name, pb.color], [' · ', '#ffffff'], who(p.animal), [pairGlyph(pb) + '? (press 1–9)', '#ffffff']] });
    }
    if (!items.length) return null;
    return { items: items, nodeMask: idSwitchBoxNodeMask(session.skeleton && session.skeleton.nodes) };
}

// ----------------------------------------------------------------------------
// Animals and selection
// ----------------------------------------------------------------------------

/** The session's animals: its identity names, in identity order (what digits 1–9 pick). */
export function sessionAnimals(session) {
    return session && session.identities ? session.identities.map(function (i) { return i.name; }) : [];
}

/** How a pair reads: "↔" for a mutual behavior (no actor and target), "→" otherwise. */
export function pairGlyph(behavior) {
    return behavior && behavior.mutual ? ' ↔ ' : ' → ';
}

/** The identity color for an animal name, or a neutral gray. */
export function animalColor(session, name) {
    var ids = (session && session.identities) || [];
    for (var i = 0; i < ids.length; i++) if (ids[i].name === name) return ids[i].color || '#9aa0a6';
    return '#9aa0a6';
}

/** The animal behavior keys label: the chosen one while it exists, else the first. */
export function focalAnimal(session) {
    var names = sessionAnimals(session);
    if (ui.focal && names.indexOf(ui.focal) >= 0) return ui.focal;
    return names[0] || null;
}

export function setFocalAnimal(name) {
    if (ui.focal === name) return;
    ui.focal = name;
    requestBehaviorRepaint({ ui: true });
}

/** The behavior the timeline rows show, or null for all of them. */
export function selectedBehaviorId() {
    return ui.selected && behaviorById(ui.selected) ? ui.selected : null;
}

export function selectBehavior(id) {
    ui.selected = id || null;
    requestBehaviorRepaint({ ui: true, labels: true });
}

// ----------------------------------------------------------------------------
// Armed range ([ and ])
// ----------------------------------------------------------------------------

export function setRangeIn(frame) {
    ui.rangeIn = frame;
    if (ui.rangeOut != null && ui.rangeOut < frame) ui.rangeOut = null;
    setStatus('Range starts at frame ' + (frame + 1).toLocaleString('en-US') + ' — ] sets its end');
    requestBehaviorRepaint({ ui: true });
}

export function setRangeOut(frame) {
    if (ui.rangeIn == null) ui.rangeIn = frame;
    ui.rangeOut = Math.max(frame, ui.rangeIn);
    setStatus('Range ' + (ui.rangeIn + 1).toLocaleString('en-US') + '–' + (ui.rangeOut + 1).toLocaleString('en-US') +
        ': a behavior key labels it (Shift + key: not that behavior)');
    requestBehaviorRepaint({ ui: true });
}

/** Arm a range from elsewhere (the timeline's Shift+drag menu). */
export function armRange(start, end) {
    ui.rangeIn = Math.min(start, end);
    ui.rangeOut = Math.max(start, end);
    requestBehaviorRepaint({ ui: true });
}

export function clearArmedRange() {
    if (ui.rangeIn == null && ui.rangeOut == null) return;
    ui.rangeIn = null; ui.rangeOut = null;
    requestBehaviorRepaint({ ui: true });
}

/** The armed range when both ends are set, else null. */
export function armedRange() {
    return ui.rangeIn != null && ui.rangeOut != null ? { start: ui.rangeIn, end: ui.rangeOut } : null;
}

// ----------------------------------------------------------------------------
// Coding
// ----------------------------------------------------------------------------

/**
 * The partner in the one running bout of `behaviorId` by `animal` (as actor,
 * or as either animal of a mutual pair), or null (none, or several).
 */
function runningTarget(session, behaviorId, animal) {
    var b = behaviorById(behaviorId);
    var mine = runningStates(session).filter(function (r) {
        return r.behaviorId === behaviorId && (r.animal === animal || (b && b.mutual && r.target === animal));
    });
    if (mine.length !== 1) return null;
    return mine[0].animal === animal ? mine[0].target : mine[0].animal;
}

/** With exactly two animals the target of a social behavior is the other one. */
function automaticTarget(session, animal) {
    var others = sessionAnimals(session).filter(function (n) { return n !== animal; });
    return others.length === 1 ? others[0] : null;
}

function nameOf(b) { return '"' + b.name + '"'; }

function apply(session, b, animal, target, range, absent, frame) {
    noteBehaviorFps(session, state.fps);
    var who = animal + (target ? pairGlyph(b) + target : '');
    if (range) {
        var r = labelRange(session, b.id, animal, target, range.start, range.end, !absent);
        if (!r) { setStatus('Could not label ' + nameOf(b) + ' for ' + who, 'warning'); return null; }
        clearArmedRange();
        setStatus((absent ? 'Not ' : '') + b.name + ' · ' + who + ' · frames ' + (r.start + 1).toLocaleString('en-US') +
            '–' + (r.end + 1).toLocaleString('en-US'), 'success');
        return r;
    }
    var res = toggleState(session, b.id, animal, target, frame);
    if (!res) { setStatus('Could not code ' + nameOf(b) + ' for ' + who, 'warning'); return null; }
    if (res.started) setStatus(b.name + ' started · ' + who + ' — press its key again to stop');
    else if (res.stopped) setStatus(b.name + ' · ' + who + ' · frames ' + (res.start + 1).toLocaleString('en-US') + '–' +
        (res.end + 1).toLocaleString('en-US'), 'success');
    else setStatus(b.name + ' · ' + who + ' · frame ' + (frame + 1).toLocaleString('en-US'), 'success');
    return res;
}

/**
 * Code a behavior, as its key or its button does: with an armed (or given)
 * range, label the range (absent with `opts.absent`); otherwise start/stop a
 * state, or code a point, at the playhead. Labels the focal animal unless
 * `opts.animal` says otherwise. A social behavior with no obvious target
 * waits for one (`pickBehaviorTarget`).
 *
 * @param {string} behaviorId
 * @param {{absent?:boolean, animal?:string, start?:number, end?:number}} [opts]
 */
export function codeBehavior(behaviorId, opts) {
    opts = opts || {};
    var session = state.session;
    var b = behaviorById(behaviorId);
    if (!session || !b) return null;
    var animal = opts.animal || focalAnimal(session);
    if (!animal) {
        setStatus('No animals to label yet: behaviors are labelled per identity (run Track All, or Tracks ▸ New Identity)', 'warning');
        return null;
    }
    var range = opts.start != null && opts.end != null
        ? { start: Math.min(opts.start, opts.end), end: Math.max(opts.start, opts.end) }
        : armedRange();
    var frame = state.currentFrame;
    if (b.social) {
        // The key that started a social state stops it: with no range armed,
        // a running bout of this behavior by this animal supplies its target.
        var target = opts.target || (!range && runningTarget(session, b.id, animal)) || automaticTarget(session, animal);
        if (!target) {
            ui.pending = { behaviorId: b.id, animal: animal, range: range, absent: !!opts.absent, frame: frame };
            setStatus('Target of ' + nameOf(b) + ' by ' + animal + '? Press its number (1–9) or pick it in the Behavior tab · Esc cancels');
            requestBehaviorRepaint({ ui: true });
            return null;
        }
        return apply(session, b, animal, target, range, !!opts.absent, frame);
    }
    return apply(session, b, animal, '', range, !!opts.absent, frame);
}

/** Answer a pending social behavior with its target. @returns {boolean} whether one was pending */
export function pickBehaviorTarget(name) {
    var p = ui.pending;
    if (!p) return false;
    var session = state.session;
    var b = behaviorById(p.behaviorId);
    if (!session || !b) { ui.pending = null; return true; }
    if (name === p.animal) { setStatus('An animal cannot be its own target — pick another', 'warning'); return true; }
    ui.pending = null;
    apply(session, b, p.animal, name, p.range, p.absent, p.frame);
    requestBehaviorRepaint({ ui: true });
    return true;
}

/** Drop a pending target question. @returns {boolean} whether one was pending */
export function cancelPendingTarget() {
    if (!ui.pending) return false;
    ui.pending = null;
    setStatus('Cancelled');
    requestBehaviorRepaint({ ui: true });
    return true;
}

/** Unlabel a range for one animal: every behavior, or just `behaviorId`. */
export function clearBehaviorRange(session, animal, start, end, behaviorId) {
    var n = 0;
    listBouts(session, { animal: animal }).forEach(function (bt) {
        if (behaviorId && bt.behaviorId !== behaviorId) return;
        if (bt.end < start || bt.start > end) return;
        if (clearRange(session, bt.behaviorId, bt.animal, bt.target, start, end)) n++;
    });
    return n;
}

/**
 * The next (dir 1) or previous (dir -1) present bout start for the focal
 * animal — of the selected behavior, or of any — from `frame`.
 * @returns {Object|null} the bout
 */
export function adjacentBout(session, frame, dir) {
    var animal = focalAnimal(session);
    var filter = { present: true };
    if (animal) filter.animal = animal;
    var sel = selectedBehaviorId();
    if (sel) filter.behaviorId = sel;
    var bouts = listBouts(session, filter);
    if (dir > 0) {
        for (var i = 0; i < bouts.length; i++) if (bouts[i].start > frame) return bouts[i];
    } else {
        for (var j = bouts.length - 1; j >= 0; j--) if (bouts[j].start < frame) return bouts[j];
    }
    return null;
}
