/**
 * behavior-discovery-worker.js — module worker for behavior discovery's steps
 * that read the whole sample: standardize + PCA, the UMAP layouts, HDBSCAN. Two
 * workers per run (the two layouts run side by side), each with its own heap:
 * pose/behavior-discovery.js terminates them at the end, which frees everything
 * UMAP held.
 *
 * It runs the same code the inline path does (`./behavior-embedding.js`,
 * `./behavior-umap.js`). No bare imports anywhere in this graph, and it is not part
 * of the app's startup graph (the umap-js bundle loads only here, or dynamically
 * on the inline path).
 *
 * Messages in:
 *   { id, type: 'reduce', rows, M, F, opts }
 *   { id, type: 'umap', knn: {indices, distances, n, k}, opts }
 *   { id, type: 'cluster', E, E2, n, d, opts }
 * Messages out:
 *   { id, type: 'progress', done, total }    (UMAP's epochs)
 *   { id, type: 'done', result }             (typed arrays transferred)
 *   { id, type: 'error', message }
 */
import { standardizeAndReduce, clusterLayouts } from './behavior-embedding.js';
import { prepareUmap } from './behavior-umap.js';

function transferables(value, out) {
    if (!value || typeof value !== 'object') return out;
    if (ArrayBuffer.isView(value)) { if (out.indexOf(value.buffer) < 0) out.push(value.buffer); return out; }
    if (Array.isArray(value)) return out;
    var keys = Object.keys(value);
    for (var i = 0; i < keys.length; i++) transferables(value[keys[i]], out);
    return out;
}

self.onmessage = function (e) {
    var m = e.data;
    if (!m) return;
    try {
        var result;
        if (m.type === 'reduce') result = standardizeAndReduce(m.rows, m.M, m.F, m.opts);
        else if (m.type === 'cluster') result = clusterLayouts(m.E, m.E2, m.n, m.d, m.opts);
        else if (m.type === 'umap') {
            var run = prepareUmap(m.knn, m.opts);
            for (var ep = 0; ep < run.epochs; ep++) {
                run.step();
                if (ep % 5 === 4) self.postMessage({ id: m.id, type: 'progress', done: ep + 1, total: run.epochs });
            }
            result = { embedding: run.embedding() };
        } else return;
        self.postMessage({ id: m.id, type: 'done', result: result }, transferables(result, []));
    } catch (err) {
        self.postMessage({ id: m.id, type: 'error', message: String(err && err.stack || err) });
    }
};
