/**
 * behavior-worker.js — module worker for pose/behavior-pool.js: computes the
 * behavior features of one chunk of frames and, for a detection run, scores
 * the rule detectors on it; for a classifier, extracts training rows, grows
 * trees, or scores the forests.
 *
 * It runs the same pure code as the main thread (`./behavior-features.js`,
 * `./behavior-rules.js`, `./behavior-forest.js`), so a chunk scored here
 * equals one scored inline. No bare imports anywhere in this graph (the
 * modules it loads are import-free but for each other).
 *
 * Messages in:
 *   { type: 'shared', key, value }   what a run's jobs share (the forests, the
 *                                    binned training matrix); kept until the
 *                                    next one (key null drops it)
 *   { type: 'job', id, kind: 'scores'|'features'|'rows'|'forest'|'trees', sharedKey,
 *     spec, chunk: {start, end, padStart, padEnd}, data: Float32Array[] (one per animal,
 *     [padStart, padEnd)), plan?, extra?, request?, trees? }
 * Messages out:
 *   { type: 'done', id, result }   (every typed array transferred)
 *   { type: 'error', id, message }
 */
import { computeFeatureChunk } from './behavior-features.js';
import { scoreDetectorsChunk } from './behavior-rules.js';
import { extractTrainingRows, scoreForestsChunk, trainTrees } from './behavior-forest.js';

var shared = { key: null, value: null };

/** Every typed array's buffer in a result, to transfer rather than copy. */
function transferables(value, out) {
    if (!value || typeof value !== 'object') return out;
    if (ArrayBuffer.isView(value)) { if (out.indexOf(value.buffer) < 0) out.push(value.buffer); return out; }
    var keys = Object.keys(value);
    for (var i = 0; i < keys.length; i++) transferables(value[keys[i]], out);
    return out;
}

self.onmessage = function (e) {
    var m = e.data;
    if (!m) return;
    if (m.type === 'shared') { shared = { key: m.key, value: m.key == null ? null : m.value }; return; }
    if (m.type !== 'job') return;
    try {
        if (m.sharedKey != null && m.sharedKey !== shared.key) throw new Error('missing shared data ' + m.sharedKey);
        var result;
        if (m.kind === 'trees') result = trainTrees(shared.value, m.trees);
        else {
            var chunk = { start: m.chunk.start, end: m.chunk.end, padStart: m.chunk.padStart, padEnd: m.chunk.padEnd, data: m.data };
            if (m.kind === 'scores') result = scoreDetectorsChunk(m.spec, chunk, m.plan, m.extra);
            else if (m.kind === 'rows') result = extractTrainingRows(m.spec, chunk, m.request);
            else if (m.kind === 'forest') result = scoreForestsChunk(m.spec, chunk, m.plan, shared.value);
            else result = computeFeatureChunk(m.spec, chunk, m.request);
        }
        self.postMessage({ type: 'done', id: m.id, result: result }, transferables(result, []));
    } catch (err) {
        self.postMessage({ type: 'error', id: m.id, message: String(err && err.stack || err) });
    }
};
