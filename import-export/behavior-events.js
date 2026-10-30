// import-export/behavior-events.js — behavior labels as sleap-io EVENTS, the form
// every tool that reads a `.slp` shares: Python sleap-io (`labels.events`, SLP
// format 2.6), sleap-io.js and sleap-app.
//
// ## What goes where in a `.slp`
//
//   /event_types   the ethogram: one entry per behavior, in ethogram order.
//                    name, description
//                    metadata: color, key, group, kind ('point'; a state writes
//                    nothing), directed ('true': has a target), mutual ('true':
//                    a pair with no actor and target), lucid_id (LUC3D's stable
//                    id, so a behavior renamed elsewhere keeps its detector)
//   /events        each labelled bout and each predicted bout, one row:
//                    type, start_frame, end_frame (both inclusive),
//                    video     the session's FIRST camera. A bout belongs to the
//                              whole session, so an event on ANY camera of a
//                              session is read as the session's.
//                    subject   the animal: the identity of that name, else the
//                              track of that name (matched by name on reading)
//                    target    a social behavior's other animal
//                    source    labels: 'manual', 'accepted_prediction', 'imported';
//                              predictions: the run's source ('rule', 'model:<hash>',
//                              'states:<layer>')
//                    score     a predicted bout's score
//                    metadata  note (a label's note, when it has one)
//
// An explicit "not" label is a bout of a SEPARATE type named 'not <behavior>' with
// metadata negates = '<behavior>', so a script counting `rear` never counts a "not"
// as a rear. Other tools see an ordinary type with a plain name.
//
// What only LUC3D reads stays in `metadata.lucid` (import-export/behavior-metadata.js):
// detectors, classifier settings, feature settings, each run's settings print, the
// labels' frame rate, keypoint-MoSeq states.
//
// An event the per-animal labels cannot hold (no subject, e.g. a stimulus epoch; a
// social behavior with no target) is kept per session (`_keptRows`) and written back
// unchanged, so reopening and saving a file never loses it. A label keeps only its
// note: other per-event metadata and names written by other tools are not carried
// through a LUC3D save.
//
// DOM-free: Node tests import it directly (they pass their own `SIO` classes).
//
// Depends on: pose/behavior-ethogram.js, pose/behavior-labels.js, pose/behavior-predictions.js

import { behaviors, behaviorById, normalizeBehaviorKey } from '../pose/behavior-ethogram.js';
import { exportBoutRows } from '../pose/behavior-labels.js';
import { exportPredictionEvents } from '../pose/behavior-predictions.js';

/** Sources of a labelled bout, as saved. */
var SOURCE_NAMES = { m: 'manual', a: 'accepted_prediction', i: 'imported' };
var SOURCE_CODES = { manual: 'm', accepted_prediction: 'a', imported: 'i' };

/** Frame-level rows read from a file, per session, written back on save. */
var _keptRows = new WeakMap();

/** Rows and catalog entries of types this session's file had but LUC3D cannot show. */
function keptRowsOf(session) {
    return (session && _keptRows.get(session)) || [];
}

/** The `/event_types` metadata for a behavior: only what differs from the defaults. */
function behaviorTypeMetadata(b) {
    var m = {};
    if (b.color) m.color = b.color;
    if (b.key) m.key = b.key;
    if (b.group) m.group = b.group;
    if (b.kind === 'point') m.kind = 'point';
    if (b.mutual) m.mutual = 'true';
    else if (b.social) m.directed = 'true';
    m.lucid_id = b.id;
    return m;
}

/** The name of a behavior's "not" type, clear of every behavior's own name. */
function notTypeName(b) {
    var name = 'not ' + b.name;
    var taken = behaviors().some(function (o) { return o.name === name; });
    return taken ? name + ' (not label)' : name;
}

/**
 * The events of some sessions and the ethogram's catalog, to hand to `new SIO.Labels`
 * or `openSlpWriter`. Writes nothing for a project with no ethogram and no bouts, so
 * its bytes do not move.
 *
 * @param {Object} SIO sleap-io classes: EventType, UserEvent, PredictedEvent, Identity
 * @param {Array<{session: Session, video: ?Object, identities: ?Object[], tracks: ?Object[]}>} items
 *   each session with its first camera's video and the catalogs its animals are
 *   looked up in. An animal in neither (a deleted identity's bouts) becomes a new
 *   Identity, added to `identities`.
 * @returns {{eventTypes: Object[], events: Object[]}}
 */
export function buildBehaviorEvents(SIO, items) {
    var typeByName = new Map();
    function typeOf(name, description, metadata) {
        var t = typeByName.get(name);
        if (!t) {
            t = new SIO.EventType({ name: name, description: description || '', metadata: metadata || {} });
            typeByName.set(name, t);
        }
        return t;
    }
    behaviors().forEach(function (b) {
        typeOf(b.name, b.description, behaviorTypeMetadata(b));
    });
    var events = [];
    items.forEach(function (item) {
        var session = item.session;
        var byName = new Map();
        (item.tracks || []).forEach(function (t) { if (!byName.has(t.name)) byName.set(t.name, t); });
        (item.identities || []).forEach(function (id) { byName.set(id.name, id); });   // an identity wins over a track
        function animal(name) {
            if (!name) return null;
            var p = byName.get(name);
            if (!p) {
                p = new SIO.Identity({ name: name });
                byName.set(name, p);
                if (item.identities) item.identities.push(p);
            }
            return p;
        }
        exportBoutRows(session).forEach(function (r) {
            var b = behaviorById(r[0]);
            if (!b) return;
            var present = r[5] === 1;
            var type = present ? typeOf(b.name) : typeOf(notTypeName(b), '', { negates: b.name });
            events.push(new SIO.UserEvent({
                type: type, video: item.video || null, startFrame: r[3], endFrame: r[4],
                subject: animal(r[1]), target: animal(r[2]),
                source: SOURCE_NAMES[r[6]] || 'manual',
                metadata: r[7] ? { note: r[7] } : {},
            }));
        });
        exportPredictionEvents(session).forEach(function (bout) {
            var b = behaviorById(bout.behaviorId);
            if (!b) return;
            var ext = bout.ext || {};
            events.push(new SIO.PredictedEvent({
                type: typeOf(b.name), video: item.video || null, startFrame: bout.start, endFrame: bout.end,
                subject: animal(bout.animal), target: animal(bout.target),
                score: Number.isFinite(bout.score) ? bout.score : null,
                scores: ext.scores || null,
                name: ext.name || '', source: ext.source || bout.source,
                metadata: ext.metadata || {},
            }));
        });
        keptRowsOf(session).forEach(function (row) {
            var options = {
                type: typeOf(row.type), video: item.video || null, startFrame: row.start, endFrame: row.end,
                subject: animal(row.subject), target: animal(row.target),
                name: row.name, source: row.source, metadata: Object.assign({}, row.metadata),
            };
            events.push(row.predicted
                ? new SIO.PredictedEvent(Object.assign(options, {
                    score: Number.isFinite(row.score) ? row.score : null, scores: row.scores || null,
                }))
                : new SIO.UserEvent(options));
        });
    });
    return { eventTypes: Array.from(typeByName.values()), events: events };
}

/**
 * A file's events as plain data, or null when it has none: what the readers hand to
 * `readBehaviorMetadata`.
 * @param {Object} labels a sleap-io `Labels`
 * @returns {?{types: Array, rows: Array}}
 */
export function fileEventsOf(labels) {
    var types = (labels && labels.eventTypes) || [], events = (labels && labels.events) || [];
    if (!types.length && !events.length) return null;
    var videoIdx = new Map();
    ((labels && labels.videos) || []).forEach(function (v, i) { videoIdx.set(v, i); });
    return {
        types: types.map(function (t) {
            return { name: t.name, description: t.description || '', metadata: Object.assign({}, t.metadata) };
        }),
        rows: events.map(function (e) {
            return {
                video: e.video && videoIdx.has(e.video) ? videoIdx.get(e.video) : -1,
                start: e.startFrame, end: e.endFrame, type: e.type.name,
                subject: e.subject ? e.subject.name : '', target: e.target ? e.target.name : '',
                predicted: !!e.isPredicted,
                score: e.score === null || e.score === undefined ? NaN : e.score,
                scores: e.scores || null,
                name: e.name || '', source: e.source || '', metadata: Object.assign({}, e.metadata),
            };
        }),
    };
}

/** The video indices a raw `sessions_json` entry names, or null when it names none. */
export function sessionVideoIdxs(rawSession) {
    var map = rawSession && rawSession.camcorder_to_video_idx_map;
    if (!map || typeof map !== 'object') return null;
    var out = new Set();
    Object.keys(map).forEach(function (k) {
        var v = Number(map[k]);
        if (Number.isInteger(v)) out.add(v);
    });
    return out.size ? out : null;
}

/** A catalog entry as `ingestEthogram` reads it. */
function typeToBehavior(t, id, extra, hasTarget) {
    var m = t.metadata || {};
    var e = {
        id: id, name: t.name,
        kind: m.kind === 'point' ? 'point' : 'state',
        social: m.directed === 'true' || m.mutual === 'true' || !!hasTarget,
        mutual: m.mutual === 'true',
        description: t.description || '',
    };
    if (m.color) e.color = m.color;
    var key = normalizeBehaviorKey(m.key);
    if (key) e.key = key;
    if (m.group) e.group = m.group;
    if (extra && extra.detector) e.detector = extra.detector;
    if (extra && extra.post) e.post = extra.post;
    return e;
}

/**
 * Turn a file's events, for one session, into what the behavior stores take: the
 * ethogram, label rows, prediction runs and the rows LUC3D keeps aside. All ids in
 * the result are the FILE's, the ids in `ethogram.behaviors`.
 *
 * @param {?{types: Array, rows: Array}} fileEvents `fileEventsOf`'s result
 * @param {Object} lucid the session's `metadata.lucid` (`ethogram` extras, run headers)
 * @param {?Set<number>} videoIdxs this session's videos; null reads every row
 * @returns {{ethogram: Object, labelRows: Array, runs: Array, kept: Array}}
 */
export function eventsToBehaviorData(fileEvents, lucid, videoIdxs) {
    var types = (fileEvents && fileEvents.types) || [];
    var allRows = (fileEvents && fileEvents.rows) || [];
    var extras = lucid && lucid.ethogram && lucid.ethogram.v === 2 ? lucid.ethogram : null;
    var extraById = new Map(), extraByName = new Map();
    ((extras && extras.behaviors) || []).forEach(function (x) {
        if (!x || typeof x !== 'object') return;
        if (typeof x.id === 'string') extraById.set(x.id, x);
        if (typeof x.name === 'string') extraByName.set(x.name, x);
    });

    // A type used with a target is social, whatever its metadata says.
    var withTarget = new Set();
    allRows.forEach(function (r) { if (r.target) withTarget.add(r.type); });

    var negates = new Map();   // the "not" type's name -> the behavior's name
    var idOfName = new Map();
    var socialNames = new Set();
    var behaviorsOut = [];
    function addBehaviorEntry(t, id, extra) {
        var e = typeToBehavior(t, id, extra, withTarget.has(t.name));
        if (e.social) socialNames.add(t.name);
        behaviorsOut.push(e);
        idOfName.set(t.name, id);
    }
    types.forEach(function (t) {
        if (t.metadata && t.metadata.negates) negates.set(t.name, t.metadata.negates);
    });
    types.forEach(function (t, i) {
        if (negates.has(t.name)) return;
        var saved = t.metadata && t.metadata.lucid_id;
        var id = /^b\d+$/.test(saved || '') ? saved : 'file' + i;
        addBehaviorEntry(t, id, extraById.get(saved) || extraByName.get(t.name));
    });
    // A "not" label of a behavior the catalog lost still needs that behavior.
    negates.forEach(function (name) {
        if (idOfName.has(name)) return;
        addBehaviorEntry({ name: name }, 'fileNot' + idOfName.size, extraByName.get(name));
    });
    // An event whose type is not in the catalog (should not happen) gets one too.
    allRows.forEach(function (r) {
        if (idOfName.has(r.type) || negates.has(r.type) || !r.type) return;
        addBehaviorEntry({ name: r.type }, 'fileRow' + idOfName.size, null);
    });

    var ethogram = { v: 1, behaviors: behaviorsOut };
    if (extras && extras.features) ethogram.features = extras.features;

    var labelRows = [], kept = [];
    var headers = lucid && lucid.behaviorPredictions && lucid.behaviorPredictions.v === 2 && Array.isArray(lucid.behaviorPredictions.runs)
        ? lucid.behaviorPredictions.runs : [];
    var runByBehavior = new Map();
    var runs = [];
    headers.forEach(function (h) {
        if (!h || typeof h.behavior !== 'string') return;
        var run = Object.assign({}, h, { bouts: [] });
        runByBehavior.set(h.behavior, run);
        runs.push(run);
    });

    allRows.forEach(function (r) {
        if (videoIdxs && r.video >= 0 && !videoIdxs.has(r.video)) return;
        var negated = negates.has(r.type);
        var behaviorName = negated ? negates.get(r.type) : r.type;
        var id = idOfName.get(behaviorName);
        var social = socialNames.has(behaviorName);
        if (!id || !r.subject || (social && (!r.target || r.target === r.subject))) { kept.push(r); return; }
        if (!r.predicted) {
            labelRows.push([id, r.subject, r.target || '', r.start, r.end, negated ? 0 : 1,
                SOURCE_CODES[r.source] || 'i', (r.metadata && r.metadata.note) || '']);
            return;
        }
        // A prediction: onto its behavior's run. A row no LUC3D run made (a
        // classifier elsewhere) makes a run of its own, source 'file', and keeps
        // what the file said so a save writes it back unchanged.
        var run = runByBehavior.get(id);
        if (!run) {
            run = { behavior: id, source: 'file', cfg: '', mode: '3d', bouts: [] };
            runByBehavior.set(id, run);
            runs.push(run);
        }
        var ext = null;
        if (r.source !== run.source || r.scores || r.name || (r.metadata && Object.keys(r.metadata).length)) {
            ext = { source: r.source, scores: r.scores || null, name: r.name || '', metadata: r.metadata || {} };
        }
        run.bouts.push([r.subject, r.target || '', r.start, r.end, Number.isFinite(r.score) ? r.score : null, ext]);
    });

    return {
        ethogram: ethogram, labelRows: labelRows, kept: kept,
        runs: runs.filter(function (run) { return run.bouts.length > 0; }),
    };
}

/** Remember the rows of a session LUC3D cannot show, to write back on save. */
export function setKeptEventRows(session, rows) {
    if (!session) return;
    if (rows && rows.length) _keptRows.set(session, rows);
    else _keptRows.delete(session);
}
