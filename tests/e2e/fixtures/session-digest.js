/**
 * session-digest.js — what a session holds, for comparing two loads of one project.
 *
 * Run in the page (`page.evaluate(digestSession)`). Numbers go through SHA-256, in a fixed
 * order: every 3D group, every 2D pose, the identities, the ethogram, the states layers and
 * every bout and prediction. Used by `slp-format-30.mjs` and `_real-slp3.mjs`.
 */
export const digestSession = async () => {
    const AS = await import('/ui/app-state.js');
    const L = await import('/pose/behavior-labels.js');
    const P = await import('/pose/behavior-predictions.js');
    const S = await import('/pose/behavior-states.js');
    const eth = await import('/pose/behavior-ethogram.js');
    const planes = await import('/ui/plane-definition.js');
    const state = AS.state;
    const s = state.session;
    if (!s) return { status: document.getElementById('statusText').textContent, noSession: true };
    const hex = async (buf) => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', buf))).slice(0, 12).map(b => b.toString(16).padStart(2, '0')).join('');
    const text = (v) => new TextEncoder().encode(JSON.stringify(v));
    // An appendable run of doubles, hashed by blocks (a buffer per block, then the list). By
    // value: a NaN read from a 32-bit column has other bits than one read from a 64-bit column.
    function hasher() {
        let buf = new Float64Array(1 << 20), n = 0;
        const parts = [];
        const flush = async () => { if (n) { parts.push(await hex(buf.slice(0, n).buffer)); n = 0; } };
        return {
            async push(v) { if (n === buf.length) await flush(); buf[n++] = v === v ? v : NaN; },
            async pushAll(arr) { for (let i = 0; i < arr.length; i++) { if (n === buf.length) await flush(); const v = arr[i]; buf[n++] = v === v ? v : NaN; } },
            async done() { await flush(); return hex(text(parts)); },
        };
    }
    const camIndex = new Map(s.cameras.map((c, i) => [c.name, i]));
    const nameOf = (id) => (eth.behaviorById(id) || {}).name;

    // Every 3D group: frame, identity, members, method, its 3D points bit for bit. Not its
    // id: LUC3D makes one from the clock and a random number on every load.
    let groups = 0, with3d = 0, members = 0, points3d = 0;
    const gh = hasher(), gm = hasher(), gs = hasher(), g3 = hasher();
    const frames = Array.from(s.instanceGroups.keys()).sort((a, b) => a - b);
    for (const f of frames) {
        for (const g of s.instanceGroups.get(f)) {
            groups++;
            await gh.pushAll([f, g.identityId, g.instances.size, g.triangulationMethod === 'ba' ? 1 : 0, g.points3d ? g.points3d.length : -1]);
            for (const [cam, inst] of g.instances) {
                members++;
                await gm.pushAll([camIndex.get(cam), inst.trackIdx, inst.type === 'predicted' ? 1 : 0]);
                await gs.push(inst.score);
            }
            if (g.points3d) {
                with3d++;
                points3d += g.points3d.length / 3;
                await g3.pushAll(g.points3d);
            }
        }
    }

    // Every 2D pose. Lazy: the store's columns, as the rest of the app reads them. Eager:
    // every instance in every frame.
    const pose = {};
    const loader = s.lazyLoader;
    const seen = new Set();
    if (loader) {
        const stores = [];
        for (const labels of loader.labelsByCam.values()) {
            const store = labels && labels._lazyDataStore;
            if (store && !seen.has(store)) { seen.add(store); stores.push(store); }
        }
        pose.stores = stores.length;
        pose.tables = {};
        for (const [name, cols] of [
            ['frames', ['video', 'frame_idx', 'instance_id_start', 'instance_id_end']],
            ['instances', ['instance_type', 'frame_id', 'skeleton', 'track', 'from_predicted', 'point_id_start', 'point_id_end', 'score', 'tracking_score']],
            ['points', ['x', 'y', 'visible', 'complete']],
            ['pred_points', ['x', 'y', 'visible', 'complete', 'score']],
        ]) {
            const table = name === 'frames' ? 'framesData' : name === 'instances' ? 'instancesData' : name === 'points' ? 'pointsData' : 'predPointsData';
            for (const col of cols) {
                const h = hasher();
                let rows = 0;
                for (const store of stores) {
                    const c = store[table][col] || [];
                    rows += c.length;
                    // Flags and ids are compared as numbers, whatever typed array holds them.
                    await h.pushAll(c);
                }
                pose.tables[name + '.' + col] = [rows, await h.done()];
            }
        }
    } else {
        const h = hasher();
        let instances = 0, unlinked = 0, withFrame = 0;
        const frameIds = Array.from(s.frameGroups.keys()).sort((a, b) => a - b);
        const hs = hasher();
        const one = async (f, cam, inst, kind) => {
            await h.pushAll([f, camIndex.get(cam), kind, inst.trackIdx, inst.type === 'predicted' ? 1 : 0, inst.score]);
            await h.pushAll(inst._xy);
            for (let k = 0; k < inst.numNodes; k++) {
                await h.push(inst.isOccluded(k) ? 1 : 0);
                await hs.push(s.predictedNodeScore(cam, f, inst, k));
            }
        };
        for (const f of frameIds) {
            const fg = s.frameGroups.get(f);
            withFrame++;
            for (const [cam, list] of fg.instances) for (const inst of list) { instances++; await one(f, cam, inst, 0); }
            for (const [cam, list] of fg.unlinkedInstances) for (const u of list) { unlinked++; await one(f, cam, u.instance, 1); }
        }
        pose.frames = withFrame;
        pose.instances = instances;
        pose.unlinked = unlinked;
        pose.digest = await h.done();
        pose.scores = await hs.done();
    }

    // What a frame holds once on screen: on a lazy project, a sample of frames hydrated from
    // the store (poses, occlusion, each node's score), and whether each could be evicted
    // again, which compares it with the store (pose/lazy-residency.js).
    if (loader) {
        const T = await import('/pose/triangulation.js');
        const R = await import('/pose/lazy-residency.js');
        const step = Math.max(1, Math.floor(loader.nFrames / 40));
        for (let f = 0; f < loader.nFrames; f += step) await T.batchLoadLazyFrames(f, Math.min(5, step));
        const h = hasher(), hs = hasher();
        const blockers = {};
        const one = async (f, cam, inst, kind) => {
            await h.pushAll([f, camIndex.get(cam), kind, inst.trackIdx, inst.type === 'predicted' ? 1 : 0, inst.score]);
            await h.pushAll(inst._xy);
            for (let k = 0; k < inst.numNodes; k++) {
                await h.push(inst.isOccluded(k) ? 1 : 0);
                await hs.push(s.predictedNodeScore(cam, f, inst, k));
            }
        };
        let resident = 0, instances = 0;
        for (const f of Array.from(s.frameGroups.keys()).sort((a, b) => a - b)) {
            const fg = s.frameGroups.get(f);
            resident++;
            for (const [cam, list] of fg.instances) for (const inst of list) { instances++; await one(f, cam, inst, 0); }
            for (const [cam, list] of fg.unlinkedInstances) for (const u of list) { instances++; await one(f, cam, u.instance, 1); }
            const why = R.frameEvictionBlocker(s, f, fg, new Set());
            if (why) blockers[why] = (blockers[why] || 0) + 1;
        }
        pose.hydrated = { resident, instances, blockers, digest: await h.done(), scores: await hs.done() };
    }

    const fim = s.frameIdentityMap;
    const planeModel = planes.planeModel();
    const out = {
        status: document.getElementById('statusText').textContent,
        session: s.name,
        sessions: state.sessions.length,
        lazy: !!s.lazyLoader,
        totalFrames: state.totalFrames,
        fps: state.fps,
        cameras: s.cameras.map(c => ({ name: c.name, matrix: c.matrix, dist: c.dist, rvec: c.rvec, tvec: c.tvec, size: c.size })),
        tracks: s.tracks.length,
        tracksDigest: await hex(text(s.tracks)),
        skeleton: { name: s.skeleton.name, nodes: s.skeleton.nodes, edges: s.skeleton.edges },
        identities: s.identities.map(i => ({ id: i.id, name: i.name, color: i.color })),
        frameIdentity: fim ? fim.size : null,
        groups, with3d, members, points3d,
        groupsDigest: await gh.done(),
        membersDigest: await gm.done(),
        memberScoresDigest: await gs.done(),
        points3dDigest: await g3.done(),
        pose,
        behaviors: eth.behaviors().map(b => ({ name: b.name, key: b.key, social: !!b.social, mutual: !!b.mutual, kind: b.kind, color: b.color, detector: b.detector || null, post: b.post || null, description: b.description || null, group: b.group || null })),
        features: eth.featureSettings(),
        bouts: L.exportBoutRows(s).map(r => [nameOf(r[0]), ...r.slice(1)]),
        predicted: P.predictedBouts(s).map(b => [nameOf(b.behaviorId), b.animal, b.target, b.start, b.end, Number.isFinite(b.score) ? Math.round(b.score * 1000) / 1000 : null]),
        runs: (P.exportPredictionHeaders(s) || { runs: [] }).runs.map(r => ({ behavior: nameOf(r.behavior), source: r.source, cfg: r.cfg, fps: r.fps, mode: r.mode })),
        stale: P.predictionStaleness(s, (await import('/ui/behavior-detect.js')).detectionEnv()),
        statesDigest: await hex(text(S.exportStateLayers(s))),
        states: S.stateLayers(s).map(l => ({ name: l.name, source: l.source, animals: l.animals.size })),
        planes: planeModel.planes.length,
        planeNodes: planeModel.nodes ? planeModel.nodes.length : null,
        numAnimals: s.numAnimals === undefined ? null : s.numAnimals,
        idSwitchReview: s.idSwitchReview === undefined ? null : JSON.stringify(s.idSwitchReview).length,
        temporalSmoothing: s.temporalSmoothing === undefined ? null : s.temporalSmoothing,
        provenanceKeys: s.fileProvenance ? Object.keys(s.fileProvenance).sort() : null,
        fileVideos: s.fileVideos ? s.fileVideos.length : null,
        heapGB: performance.memory ? +(performance.memory.usedJSHeapSize / 1e9).toFixed(2) : null,
    };
    return out;
};
