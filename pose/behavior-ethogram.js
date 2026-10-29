// pose/behavior-ethogram.js — the project's ethogram: the behaviors a user
// decided ahead of time to label (Behavior ▸ Edit Ethogram…).
//
// One ethogram per PROJECT, shared by every session, so it lives on a module
// singleton the way the plane model does; `resetEthogram()` runs before every
// load (import-export/behavior-metadata.js `resetBehaviorState`).
//
// A behavior is
//   { id: 'b1', name: 'rear', key: 'r', color: '#…', kind: 'state'|'point',
//     social: false, mutual: false, group: '', description: '' }
//
//   * `id` is stable for the life of the project file and is what labelled
//     bouts reference (pose/behavior-labels.js), so renaming a behavior never
//     touches a bout. Ids are minted `b<n>` and never reused.
//   * `kind`: a state has a start and an end (grooming); a point is one frame
//     (a jump, a bite).
//   * `social`: the behavior has a target animal (actor → target, "chases").
//   * `mutual` (social only): the pair has no actor and target ("nose-to-nose
//     sniffing", id_1 ↔ id_4) — one bout for the pair, whichever animal was
//     picked first (pose/behavior-labels.js stores the pair in name order).
//   * `group`: behaviors that share a non-empty group are mutually exclusive
//     for one animal — starting one stops the others, and labelling one over a
//     range clears the others' present labels there (Noldus Observer's
//     mutually exclusive groups, BORIS's exclusion matrix).
//   * `key`: one lower-case letter, coded in Label Behaviors mode
//     (ui/behavior-mode.js). Optional: a behavior without a key is coded from
//     the Behavior tab. Digits pick the animal in the mode, and `[ ] , .`, Space,
//     Esc and the arrows keep their mode meanings, so keys are letters only.
//   * `detector` (optional): the rule that detects it automatically
//     (pose/behavior-rules.js), `{conditions, minDuration, maxGap}`.
//
// The ethogram also carries the project's behavior FEATURE settings
// (`featureSettings`): the node roles, the excluded nodes and the floor the
// features use (pose/behavior-tensor.js) — null until someone changes them,
// and only what was changed is kept, so defaults are never written.
//
// DOM-free: Node tests import it directly.
//
// Depends on: pose/behavior-rules.js (`normalizeDetector`)

import { normalizeDetector } from './behavior-rules.js';

/** The behavior kinds. */
export var BEHAVIOR_KINDS = ['state', 'point'];

/**
 * Default colors, handed out in order to new behaviors: vivid, light enough
 * to read on the dark timeline and video (L* >= 62), and chosen one at a time to stay as far
 * as possible (CIEDE2000) from the ones before and from the gray of "not"
 * labels — closest pair 18.2 (tests/test-track-identity-palette.mjs §10).
 */
export var BEHAVIOR_COLORS = [
    '#f2a541', '#5fb3f9', '#e8749a', '#7fd17a', '#40f4f4', '#f4e840',
    '#f4704c', '#e8b8f4', '#a0a040', '#e858f4', '#40ac94', '#9488f4',
];

var NAME_MAX = 40;
var DESCRIPTION_MAX = 500;
var GROUP_MAX = 40;

var _model = { behaviors: [], nextId: 1, features: null };

/** Drop every behavior and the feature settings (new project, or before a load). */
export function resetEthogram() {
    _model = { behaviors: [], nextId: 1, features: null };
}

var FLOOR_SOURCES = ['origin', 'fit', 'plane'];

/**
 * Feature settings as kept: only the parts that differ from the defaults, or
 * null when nothing does.
 *   nodeRoles  {head?: node name, tailBase?: node name}
 *   exclude    node names left out of features (replaces the default list)
 *   floor      {source: 'origin'|'fit'|'plane', plane?: plane id}
 */
function cleanFeatureSettings(f) {
    if (!f || typeof f !== 'object') return null;
    var out = {};
    if (f.nodeRoles && typeof f.nodeRoles === 'object') {
        var nr = {};
        ['head', 'tailBase'].forEach(function (r) {
            if (typeof f.nodeRoles[r] === 'string' && f.nodeRoles[r]) nr[r] = f.nodeRoles[r];
        });
        if (Object.keys(nr).length) out.nodeRoles = nr;
    }
    if (Array.isArray(f.exclude)) {
        out.exclude = f.exclude.filter(function (n, i, a) { return typeof n === 'string' && n && a.indexOf(n) === i; });
    }
    if (f.floor && typeof f.floor === 'object' && FLOOR_SOURCES.indexOf(f.floor.source) >= 0) {
        var fl = { source: f.floor.source };
        if (fl.source === 'plane') {
            if (!Number.isFinite(f.floor.plane)) fl = null;
            else fl.plane = f.floor.plane;
        }
        if (fl) out.floor = fl;
    }
    return Object.keys(out).length ? out : null;
}

/** The project's feature settings (null = every default). Do not mutate. */
export function featureSettings() {
    return _model.features;
}

/**
 * Replace the feature settings. Pass only what differs from the defaults
 * (`{nodeRoles: {head: 'Nose'}}`); null or {} restores every default.
 */
export function setFeatureSettings(f) {
    _model.features = cleanFeatureSettings(f);
    return _model.features;
}

/**
 * Set (or with null, remove) a behavior's detector.
 * @returns {{behavior?: Object, error?: string}}
 */
export function setBehaviorDetector(id, detector) {
    var b = behaviorById(id);
    if (!b) return { error: 'No such behavior' };
    if (detector == null) { delete b.detector; return { behavior: b }; }
    var d = normalizeDetector(detector);
    if (!d) return { error: 'A detector needs at least one condition' };
    b.detector = d;
    return { behavior: b };
}

/** The behaviors, in ethogram order. Do not mutate; use the functions below. */
export function behaviors() {
    return _model.behaviors;
}

/** True when the project has no behaviors. */
export function ethogramIsEmpty() {
    return _model.behaviors.length === 0;
}

/** The behavior with `id`, or null. */
export function behaviorById(id) {
    for (var i = 0; i < _model.behaviors.length; i++) {
        if (_model.behaviors[i].id === id) return _model.behaviors[i];
    }
    return null;
}

/** The behavior coded by `key` (a letter, either case), or null. */
export function behaviorByKey(key) {
    if (typeof key !== 'string' || key.length !== 1) return null;
    var k = key.toLowerCase();
    for (var i = 0; i < _model.behaviors.length; i++) {
        if (_model.behaviors[i].key === k) return _model.behaviors[i];
    }
    return null;
}

/** The behavior named `name` (case-insensitive), or null. */
export function behaviorByName(name) {
    if (typeof name !== 'string') return null;
    var n = name.trim().toLowerCase();
    for (var i = 0; i < _model.behaviors.length; i++) {
        if (_model.behaviors[i].name.toLowerCase() === n) return _model.behaviors[i];
    }
    return null;
}

/** The other behaviors in `b`'s exclusive group (empty when it has none). */
export function exclusivePeers(b) {
    if (!b || !b.group) return [];
    return _model.behaviors.filter(function (o) { return o !== b && o.group === b.group; });
}

/** The color a new behavior gets: the first default not already in use. */
export function nextBehaviorColor() {
    var used = new Set(_model.behaviors.map(function (b) { return b.color; }));
    for (var i = 0; i < BEHAVIOR_COLORS.length; i++) {
        if (!used.has(BEHAVIOR_COLORS[i])) return BEHAVIOR_COLORS[i];
    }
    return BEHAVIOR_COLORS[_model.behaviors.length % BEHAVIOR_COLORS.length];
}

/** A key as stored: one lower-case letter, or '' for none. */
export function normalizeBehaviorKey(key) {
    return typeof key === 'string' && /^[a-z]$/i.test(key.trim()) ? key.trim().toLowerCase() : '';
}

/**
 * Why `fields` cannot be saved, or null when they can.
 * @param {Object} fields  { name, key, color, kind, social, group, description }
 * @param {string} [exceptId] the behavior being edited (its own name/key are not a clash)
 * @returns {string|null}
 */
export function validateBehaviorFields(fields, exceptId) {
    var name = typeof fields.name === 'string' ? fields.name.trim() : '';
    if (!name) return 'A behavior needs a name';
    if (name.length > NAME_MAX) return 'Names are at most ' + NAME_MAX + ' characters';
    var clash = behaviorByName(name);
    if (clash && clash.id !== exceptId) return 'There is already a behavior named "' + clash.name + '"';
    if (fields.key != null && fields.key !== '') {
        var key = normalizeBehaviorKey(fields.key);
        if (!key) return 'Keys are single letters (digits pick the animal)';
        var taken = behaviorByKey(key);
        if (taken && taken.id !== exceptId) return 'Key "' + key + '" already codes "' + taken.name + '"';
    }
    if (fields.kind != null && BEHAVIOR_KINDS.indexOf(fields.kind) < 0) return 'Unknown kind "' + fields.kind + '"';
    if (fields.color != null && !/^#[0-9a-f]{6}$/i.test(fields.color)) return 'Colors are #rrggbb';
    return null;
}

function clean(fields, base) {
    var out = Object.assign({}, base);
    if (fields.name != null) out.name = String(fields.name).trim();
    if (fields.key != null) out.key = normalizeBehaviorKey(fields.key);
    if (fields.color != null) out.color = String(fields.color).toLowerCase();
    if (fields.kind != null) out.kind = fields.kind === 'point' ? 'point' : 'state';
    if (fields.social != null) out.social = !!fields.social;
    if (fields.mutual != null) out.mutual = !!fields.mutual;
    if (out.mutual && fields.social == null && fields.mutual) out.social = true;   // mutual implies social
    if (!out.social) out.mutual = false;                                         // and needs it
    if (fields.group != null) out.group = String(fields.group).trim().slice(0, GROUP_MAX);
    if (fields.description != null) out.description = String(fields.description).slice(0, DESCRIPTION_MAX);
    return out;
}

/**
 * Add a behavior.
 * @returns {{behavior?: Object, error?: string}}
 */
export function addBehavior(fields) {
    fields = fields || {};
    var err = validateBehaviorFields(fields, null);
    if (err) return { error: err };
    var b = clean(fields, {
        id: 'b' + (_model.nextId++), name: '', key: '', color: nextBehaviorColor(),
        kind: 'state', social: false, mutual: false, group: '', description: '',
    });
    _model.behaviors.push(b);
    return { behavior: b };
}

/**
 * Change a behavior's fields (its id never changes).
 * @returns {{behavior?: Object, error?: string}}
 */
export function updateBehavior(id, fields) {
    var b = behaviorById(id);
    if (!b) return { error: 'No such behavior' };
    var merged = Object.assign({}, b, fields || {});
    var err = validateBehaviorFields(merged, id);
    if (err) return { error: err };
    Object.assign(b, clean(fields || {}, b));
    return { behavior: b };
}

/** Remove a behavior from the ethogram. Its bouts are the caller's to delete. */
export function removeBehavior(id) {
    var i = _model.behaviors.findIndex(function (b) { return b.id === id; });
    if (i < 0) return false;
    _model.behaviors.splice(i, 1);
    return true;
}

/** Move a behavior `delta` places in the ethogram order. */
export function moveBehavior(id, delta) {
    var list = _model.behaviors;
    var i = list.findIndex(function (b) { return b.id === id; });
    var j = i + delta;
    if (i < 0 || j < 0 || j >= list.length) return false;
    var tmp = list[i]; list[i] = list[j]; list[j] = tmp;
    return true;
}

/**
 * The `metadata.lucid.ethogram` payload, or null for an empty ethogram with
 * default feature settings (nothing is written).
 */
export function serializeEthogram() {
    if (!_model.behaviors.length && !_model.features) return null;
    var out = {
        v: 1,
        behaviors: _model.behaviors.map(function (b) {
            var o = { id: b.id, name: b.name, color: b.color, kind: b.kind };
            if (b.key) o.key = b.key;
            if (b.social) o.social = true;
            if (b.mutual) o.mutual = true;
            if (b.group) o.group = b.group;
            if (b.description) o.description = b.description;
            if (b.detector) o.detector = JSON.parse(JSON.stringify(b.detector));
            return o;
        }),
    };
    if (_model.features) out.features = JSON.parse(JSON.stringify(_model.features));
    return out;
}

function idNumber(id) {
    var m = /^b(\d+)$/.exec(id);
    return m ? parseInt(m[1], 10) : 0;
}

/**
 * Bring a saved ethogram into the project. Into an empty ethogram, its
 * behaviors keep their ids. Into a non-empty one (a later session of the same
 * project, or a session file added to an open project) they are matched BY
 * NAME: a known name maps to the project's behavior, an unknown one is added
 * under a fresh id. Malformed entries are skipped.
 *
 * @param {Object} payload `metadata.lucid.ethogram`
 * @returns {Map<string,string>} the file's behavior id -> the project's id,
 *   for remapping that file's bouts
 */
export function ingestEthogram(payload) {
    var idMap = new Map();
    if (!payload || typeof payload !== 'object' || payload.v !== 1 || !Array.isArray(payload.behaviors)) return idMap;
    var intoEmpty = _model.behaviors.length === 0;
    // Feature settings are project-scoped: the first session ingested (or a
    // project that has none yet) brings them; later ones map onto them.
    if (!_model.features && payload.features) _model.features = cleanFeatureSettings(payload.features);
    payload.behaviors.forEach(function (e) {
        if (!e || typeof e !== 'object' || typeof e.id !== 'string' || typeof e.name !== 'string' || !e.name.trim()) return;
        var known = behaviorByName(e.name);
        if (known) {
            idMap.set(e.id, known.id);
            // A detector comes in with an imported ethogram for a behavior that has none.
            var kd = !known.detector && normalizeDetector(e.detector);
            if (kd) known.detector = kd;
            return;
        }
        var fields = {
            name: e.name, kind: e.kind === 'point' ? 'point' : 'state', social: !!e.social, mutual: !!(e.social && e.mutual),
            group: typeof e.group === 'string' ? e.group : '',
            description: typeof e.description === 'string' ? e.description : '',
            color: typeof e.color === 'string' && /^#[0-9a-f]{6}$/i.test(e.color) ? e.color : nextBehaviorColor(),
        };
        var key = normalizeBehaviorKey(e.key);
        if (key && !behaviorByKey(key)) fields.key = key;            // a clashing key is dropped, not the behavior
        var sameId = intoEmpty && /^b\d+$/.test(e.id) && !behaviorById(e.id);
        var b = clean(fields, { id: sameId ? e.id : 'b' + (_model.nextId++), name: '', key: '', color: '',
            kind: 'state', social: false, mutual: false, group: '', description: '' });
        if (sameId) _model.nextId = Math.max(_model.nextId, idNumber(e.id) + 1);
        var det = normalizeDetector(e.detector);
        if (det) b.detector = det;
        _model.behaviors.push(b);
        idMap.set(e.id, b.id);
    });
    return idMap;
}
