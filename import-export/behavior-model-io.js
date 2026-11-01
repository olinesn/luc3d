// import-export/behavior-model-io.js — behavior classifier files
// (`*.lucid-behavior.json`): trained forests carried to another project.
//
// A file holds, for each behavior, its ethogram entry (name, kind, social,
// mutual, group, color, key, description, post-processing) and its model:
// the forest (base64 typed arrays), the names of the features it reads, its
// parameters, what it was trained on and how it cross-validated. Around them,
// what the features depend on: the feature-spec version, the windows and
// bands (seconds and Hz, so another frame rate converts), the skeleton's node
// names, the head and tail base, the nodes used, 2D or 3D.
//
// Reading checks the file (format, every array decodes, every forest index is
// in range, the stored hash matches the forest); applying it to a session
// checks that the session measures the same features (same feature-spec
// version, 2D/3D, roles, used nodes, and every feature a forest reads). An
// applied model is a classifier like a trained one, except that this
// project's labels did not train it: its predictions never go stale on label
// edits (until it is retrained here).
//
// DOM-free: Node tests import it directly.
//
// Depends on: pose/behavior-forest.js, pose/behavior-features.js,
//             pose/behavior-tensor.js, pose/behavior-ethogram.js,
//             pose/behavior-classifier.js

import {
    FOREST_VERSION, typedToBase64, base64ToTyped, forestHash, forestProblem,
} from '../pose/behavior-forest.js';
import { FEATURE_SPEC_VERSION, WINDOW_SECONDS, BANDS } from '../pose/behavior-features.js';
import { POSE_CHECK_VERSION } from '../pose/behavior-tensor.js';
import { behaviorById, ingestEthogram } from '../pose/behavior-ethogram.js';
import { modelProblems } from '../pose/behavior-classifier.js';

/** The `format` field of a model file. */
export var MODEL_FILE_FORMAT = 'lucid-behavior-model';
export var MODEL_FILE_VERSION = 1;
/** Model files end with this. */
export var MODEL_FILE_SUFFIX = '.lucid-behavior.json';

function round3(v) {
    return Number.isFinite(v) ? Math.round(v * 1000) / 1000 : null;
}

function cvSummary(cv) {
    if (!cv) return null;
    var out = { scheme: cv.scheme, folds: cv.folds || [] };
    if (cv.frame) out.frame = { precision: round3(cv.frame.precision), recall: round3(cv.frame.recall), f1: round3(cv.frame.f1) };
    if (cv.bout) out.bout = { precision: round3(cv.bout.precision), recall: round3(cv.bout.recall), f1: round3(cv.bout.f1) };
    if (cv.threshold != null) out.threshold = cv.threshold;
    if (cv.best) out.best = { threshold: cv.best.threshold, f1: round3(cv.best.f1) };
    if (cv.note) out.note = cv.note;
    return out;
}

/**
 * The file for some models.
 * @param {Object[]} models from pose/behavior-classifier.js
 * @param {{fps?: number}} [info]
 * @returns {Object} JSON-ready
 */
export function buildModelFile(models, info) {
    var first = models[0] || {};
    return {
        format: MODEL_FILE_FORMAT, v: MODEL_FILE_VERSION, created: new Date().toISOString(),
        featureSpecVersion: FEATURE_SPEC_VERSION, forestVersion: FOREST_VERSION, poseCheckVersion: POSE_CHECK_VERSION,
        windows: WINDOW_SECONDS.slice(), bands: BANDS.map(function (b) { return b.slice(); }),
        mode: first.mode || '3d', nodes: (first.nodes || []).slice(),
        roles: { head: first.body ? first.body.head : '', tailBase: first.body ? first.body.tailBase : '' },
        used: first.body ? first.body.used.slice() : [],
        fps: info && info.fps > 0 ? info.fps : undefined,
        behaviors: models.map(function (m) {
            var b = behaviorById(m.behaviorId) || {};
            var entry = { id: m.behaviorId, name: b.name || m.name, kind: b.kind || m.kind || 'state', color: b.color };
            if (b.key) entry.key = b.key;
            if (m.social) entry.social = true;
            if (m.mutual) entry.mutual = true;
            if (b.group) entry.group = b.group;
            if (b.description) entry.description = b.description;
            if (b.post) entry.post = Object.assign({}, b.post);
            return {
                entry: entry,
                model: {
                    hash: m.hash, features: m.features.slice(), params: Object.assign({}, m.params),
                    training: m.training ? { sessions: m.training.sessions, rows: m.training.rows, counts: m.training.counts } : null,
                    cv: cvSummary(m.cv),
                    importance: typedToBase64(m.importance),
                    forest: {
                        feat: typedToBase64(m.forest.feat), val: typedToBase64(m.forest.val), kid: typedToBase64(m.forest.kid),
                        miss: typedToBase64(m.forest.miss), roots: typedToBase64(m.forest.roots),
                    },
                },
            };
        }),
    };
}

/** A suggested file name: the behaviors' names, joined. */
export function modelFileName(models) {
    var base = models.map(function (m) { var b = behaviorById(m.behaviorId); return (b ? b.name : m.name) || 'behavior'; }).join('-')
        .replace(/[^\w.-]+/g, '_').slice(0, 60) || 'behaviors';
    return base + MODEL_FILE_SUFFIX;
}

/**
 * Read a model file (its parsed JSON).
 * @returns {{file?: Object, models: Object[], problems: string[]}} `models` are ready to apply
 *   (each `{entry, model}` with decoded arrays); problems are file-level (nothing usable)
 */
export function parseModelFile(json) {
    var problems = [];
    if (!json || typeof json !== 'object' || json.format !== MODEL_FILE_FORMAT) return { models: [], problems: ['Not a LUC3D behavior classifier file'] };
    if (json.v !== MODEL_FILE_VERSION) return { models: [], problems: ['Made by a newer version of LUC3D (file version ' + json.v + ')'] };
    if (!Array.isArray(json.behaviors) || !json.behaviors.length) return { models: [], problems: ['The file holds no classifier'] };
    var models = [];
    json.behaviors.forEach(function (item, i) {
        var e = item && item.entry, m = item && item.model;
        var label = e && typeof e.name === 'string' ? '"' + e.name + '"' : 'classifier ' + (i + 1);
        if (!e || typeof e.name !== 'string' || !e.name.trim() || !m || !Array.isArray(m.features) || !m.forest) {
            problems.push(label + ': malformed'); return;
        }
        var forest = {
            feat: base64ToTyped(m.forest.feat, Uint16Array), val: base64ToTyped(m.forest.val, Float32Array),
            kid: base64ToTyped(m.forest.kid, Int32Array), miss: base64ToTyped(m.forest.miss, Uint8Array),
            roots: base64ToTyped(m.forest.roots, Int32Array),
        };
        var why = forestProblem(forest, m.features.length);
        if (why) { problems.push(label + ': the forest is damaged (' + why + ')'); return; }
        var hash = forestHash(forest, m.features);
        if (typeof m.hash === 'string' && m.hash !== hash) { problems.push(label + ': the forest does not match its hash'); return; }
        var importance = base64ToTyped(m.importance, Float32Array);
        if (!importance || importance.length !== m.features.length) importance = new Float32Array(m.features.length);
        models.push({ entry: e, model: {
            name: e.name, kind: e.kind === 'point' ? 'point' : 'state', social: !!e.social, mutual: !!(e.social && e.mutual), group: e.group || '',
            hash: hash, features: m.features.slice(), forest: forest, importance: importance,
            params: m.params || {}, training: m.training || null, cv: m.cv || null,
            versions: { forest: json.forestVersion, features: json.featureSpecVersion, poseCheck: json.poseCheckVersion },
            mode: json.mode === '2d' ? '2d' : '3d', nodes: Array.isArray(json.nodes) ? json.nodes.slice() : [],
            body: { head: json.roles ? json.roles.head || '' : '', tailBase: json.roles ? json.roles.tailBase || '' : '',
                    used: Array.isArray(json.used) ? json.used.slice() : [] },
            labelPrint: null, imported: true,
        } });
    });
    return { file: json, models: models, problems: problems };
}

/**
 * Why a file's models cannot run on a session: `blocking` (another
 * feature-spec or forest version, 2D vs 3D, other node roles, other used
 * nodes or a missing node — the features would not mean the same) and
 * `warnings` (a model that cannot run here, e.g. a social one on a session
 * with one animal; it is still imported, for the other sessions).
 * @param {Object} parsed from `parseModelFile`
 * @param {Object} ctx a behavior context of the session (ui/behavior-detect.js `buildBehaviorContext`)
 * @returns {{blocking: string[], warnings: string[]}}
 */
export function modelFileProblems(parsed, ctx) {
    var out = [], warn = [];
    var f = parsed.file || {};
    if (f.featureSpecVersion !== FEATURE_SPEC_VERSION) out.push('Its features are from another version of LUC3D (feature spec ' + f.featureSpecVersion + ', here ' + FEATURE_SPEC_VERSION + ')');
    if (f.forestVersion !== FOREST_VERSION) out.push('Its forests are from another version of LUC3D');
    if (!ctx) return { blocking: out.concat(['No session with poses to check it against']), warnings: warn };
    var names = ctx.input.nodeNames;
    if ((f.mode === '2d' ? '2d' : '3d') !== ctx.input.mode) out.push('It was trained on ' + (f.mode === '2d' ? '2D' : '3D') + ' poses; this session has ' + ctx.input.mode.toUpperCase());
    var head = names[ctx.spec.roles.head] || '', tail = names[ctx.spec.roles.tailBase] || '';
    if (f.roles && (f.roles.head !== head || f.roles.tailBase !== tail)) {
        out.push('Its head / tail base are ' + (f.roles.head || '–') + ' / ' + (f.roles.tailBase || '–') + '; here ' + (head || '–') + ' / ' + (tail || '–') + ' (set them under Behavior ▸ Detection Settings ▸ Body)');
    }
    var missingNodes = (f.nodes || []).filter(function (n) { return names.indexOf(n) < 0; });
    if (missingNodes.length) out.push('This skeleton has no ' + missingNodes.slice(0, 3).join(', ') + (missingNodes.length > 3 ? ', …' : ''));
    var used = ctx.spec.used.map(function (i) { return names[i]; });
    var fu = (f.used || []).slice().sort().join('|'), hu = used.slice().sort().join('|');
    if (f.used && fu !== hu) out.push('It measured other nodes (' + (f.used || []).length + ' used, here ' + used.length + '): check the excluded nodes in Detection Settings');
    if (!out.length) {
        parsed.models.forEach(function (it) {
            var probs = modelProblems(it.model, ctx);
            if (probs.length) warn.push('"' + it.entry.name + '": ' + probs.join('; '));
        });
    }
    return { blocking: out, warnings: warn };
}

/**
 * Bring a file's behaviors into the ethogram (matched by name; unknown ones
 * are added with their color, key, group and post-processing) and return its
 * models keyed to this project's behavior ids.
 * @returns {Object[]} models with `behaviorId` set
 */
export function adoptModelFile(parsed) {
    var idMap = ingestEthogram({ v: 1, behaviors: parsed.models.map(function (it) { return it.entry; }) });
    return parsed.models.map(function (it) {
        var id = idMap.get(it.entry.id) || (behaviorById(it.entry.id) ? it.entry.id : null);
        return Object.assign({}, it.model, { behaviorId: id });
    }).filter(function (m) { return m.behaviorId; });
}
