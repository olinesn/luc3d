/**
 * test-track-triangulate-cores.mjs — the DOM-free pieces split out of the app
 * so the command-line LUCID runs the same code (pose/track-all-core.js,
 * pose/triangulate-all-core.js, ui/settings.js `useSettingsSnapshot`,
 * import-export/slp-session-build.js's per-camera helpers), loaded in plain
 * Node with no page:
 *  1. every split module imports without a page;
 *  2. `useSettingsSnapshot` replaces the settings in effect with the stored
 *     JSON's shape, clamped as stored settings are, and a run snapshots it;
 *  3. `triangulateAllRoute` picks the toolbar's pass;
 *  4. `cameraDirMatches` is the folder loader's camera rule;
 *  5. Track All's pass on a small scene: identities, groups with members'
 *     3D, and `computeMaxInstancesPerView`; then `clearTrackingStateInRange`
 *     clears only its frames;
 *  6. Group by Identity & Triangulate: `groupFrameByIdentity` regroups a
 *     frame by identity, leaves a frame it cannot group alone, and the
 *     resident sweep visits frames in order within `start`/`end`;
 *  7. `moveInstancesToUnlinked` moves every linked pose, in order.
 */
let passed = 0, failed = 0;
const check = (cond, msg, extra) => {
    if (cond) { passed++; console.log('  ok   ' + msg); }
    else { failed++; console.log('  FAIL ' + msg + (extra !== undefined ? '  ' + JSON.stringify(extra).slice(0, 300) : '')); }
};

// The pass logs per-frame warnings; keep the output readable.
const realWarn = console.warn;
console.warn = () => {};

console.log('-- 1. imports without a page --');
const mods = {};
for (const m of ['../pose/track-all-core.js', '../pose/triangulate-all-core.js', '../ui/plane-state.js',
    '../import-export/slp-session-build.js', '../import-export/slp-labels-build.js', '../import-export/slp-read.js',
    '../import-export/calibration-parse.js', '../import-export/plane-metadata.js', '../ui/settings.js']) {
    try { mods[m] = await import(m); check(true, m); } catch (e) { check(false, m + ': ' + e.message); }
}
const T = mods['../pose/track-all-core.js'], G = mods['../pose/triangulate-all-core.js'];
const S = mods['../ui/settings.js'], B = mods['../import-export/slp-session-build.js'];
const pd = await import('../pose/pose-data.js');
const { createGroupSolver } = await import('../pose/triangulation-pool.js');

console.log('-- 2. settings snapshot --');
S.useSettingsSnapshot({ triangulationMethod: 'ba', trackingThresholds: { corr3dWeight: 3, stale: 9999, bogus: 1 },
    nodeWeights: { nose: 0.5, head: 7 }, cameraWeights: { side: 0 } });
check(S.getDefaultTriangulationMethod() === 'ba', 'triangulationMethod read');
check(S.getTrackingThreshold('corr3dWeight') === 3, 'a threshold read');
check(S.getTrackingThreshold('stale') === 500, 'an out-of-range threshold is clamped (stale 9999 -> 500)');
check(S.getNodeWeight('head') === 1 && S.getNodeWeight('nose') === 0.5, 'node weights clamped to [0, 1]');
check(!S.isCameraTracked('side') && S.isCameraTracked('top'), 'camera inclusion read');
check(T.snapshotTrackingThresholds().corr3dWeight === 3 && T.thr('corr3dWeight') === 3, "a run's snapshot reads it");
S.useSettingsSnapshot(null);
check(S.getDefaultTriangulationMethod() === 'dlt' && S.getTrackingThreshold('corr3dWeight') === 6 && S.isCameraTracked('side'),
    'null restores the defaults');

console.log('-- 3. the Triangulate All route --');
const fakeSession = (n) => ({ identities: new Array(n).fill({}) });
check(G.triangulateAllRoute('ba', fakeSession(2)) === 'ba', 'Refined: triangulate every group');
check(G.triangulateAllRoute('dlt', fakeSession(2)) === 'identity', 'DLT with identities: group by identity');
check(G.triangulateAllRoute('dlt', fakeSession(0)) === 'dlt' && G.triangulateAllRoute('dlt', null) === 'dlt', 'DLT without: triangulate the groups');

console.log('-- 4. camera folder rule --');
check(B.cameraDirMatches('back', 'back') && B.cameraDirMatches('camback', 'back') && B.cameraDirMatches('cam1', '1'),
    'same name, cam + name, name without "cam"');
check(!B.cameraDirMatches('backl', 'back') && !B.cameraDirMatches('top', 'back'), 'no near misses');

console.log('-- 5. Track All pass --');
const demo = await import('../demo-data.js');
const cams = demo.createDemoCalibration();
const skel = demo.createDemoSkeleton();
function scene() {
    let s = 3;
    const u = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
    const sess = new pd.Session(cams, skel, ['track_0', 'track_1'], 'cores');
    for (let f = 0; f < 40; f++) {
        const centres = [[-30 + f * 0.5, 0, 0], [30 - f * 0.5, 10, 0]];
        const fg = new pd.FrameGroup(f);
        for (const cam of cams) {
            centres.forEach((c, a) => {
                const pts = skel.nodes.map((_, k) => {
                    const q = cam.project([c[0] + k * 5, c[1] + (u() - 0.5), c[2] + (u() - 0.5)]);
                    return [q[0] + (u() - 0.5), q[1] + (u() - 0.5)];
                });
                fg.addUnlinkedInstance(cam.name, new pd.UnlinkedInstance(new pd.Instance(pts, a, 'predicted', 0.9), cam.name));
            });
        }
        sess.addFrameGroup(fg);
    }
    return sess;
}
let sess = scene();
check(T.computeMaxInstancesPerView(sess) === 2, 'auto-detected 2 animals per view');
T.clearTrackingState(sess);
let run = T.createTrackerRun(sess, sess.cameras, 2, null);
let res = T.trackFramesInline(sess, run, sess.frameIndices);
check(res.numIdentities === 2 && sess.identities.map(i => i.name).join(',') === 'id_0,id_1', '2 identities: ' + sess.identities.map(i => i.name).join(','));
let nG = 0, all4 = 0, has3d = 0;
for (const [, gs] of sess.instanceGroups) for (const g of gs) {
    nG++; if (g.instances.size === 4) all4++; if (pd.someValidPoint3d(g.points3d)) has3d++;
}
check(nG === 80 && all4 === 80 && has3d === 80, 'a 4-camera group with 3D per animal per frame (' + nG + ')');
check(sess.frameGroups.get(0).unlinkedInstances.get('back').length === 0, 'grouped poses were promoted out of the unlinked pool');
T.clearTrackingStateInRange(sess, 10, 19);
check(!sess.instanceGroups.has(10) && !sess.instanceGroups.has(19) && sess.instanceGroups.has(9) && sess.instanceGroups.has(20),
    'clearTrackingStateInRange clears only frames 10..19');
const fim = [...sess.frameIdentityEntries()];
check(!fim.some(r => r.frameIdx >= 10 && r.frameIdx <= 19) && fim.some(r => r.frameIdx === 9), '...and their per-frame identities');

console.log('-- 6. Group by Identity & Triangulate --');
sess = scene();
T.clearTrackingState(sess);
run = T.createTrackerRun(sess, sess.cameras, 2, null);
T.trackFramesInline(sess, run, sess.frameIndices);
const solver = createGroupSolver(sess.cameras, { method: 'dlt', triangulateOnly: true, expectedGroups: 40 });
check(!solver.parallel, 'without a page the solver runs inline');
const visited = [];
const gr = await G.groupByIdentityAndTriangulateResident(sess, sess.cameras, 'dlt', solver, {});
check(gr.processedFrames === 40 && gr.totalGrouped === 80 && gr.solvedDlt === 80 && gr.reused3d === 0,
    'every frame regrouped by identity, 80 groups solved with DLT', { p: gr.processedFrames, g: gr.totalGrouped, d: gr.solvedDlt, r: gr.reused3d });
let dltGroups = 0;
for (const [, gs] of sess.instanceGroups) for (const g of gs) if (g.triangulationMethod === 'dlt' && pd.someValidPoint3d(g.points3d)) dltGroups++;
check(dltGroups === 80, 'each group holds DLT 3D');
// A second identical run adopts the 3D it already has (adoptPrior3d).
const again = await G.groupByIdentityAndTriangulateResident(sess, sess.cameras, 'dlt',
    createGroupSolver(sess.cameras, { method: 'dlt', triangulateOnly: true, expectedGroups: 40 }), {});
check(again.reused3d === 80 && again.solvedDlt === 0, 'a re-run keeps the identical groups\' 3D');
// A frame nothing can regroup is left alone.
const lone = new pd.Session(cams, skel, ['t'], 'lone');
const lfg = new pd.FrameGroup(0);
lfg.addUnlinkedInstance('back', new pd.UnlinkedInstance(new pd.Instance([[1, 1]], 0, 'predicted', 1), 'back'));
lone.addFrameGroup(lfg);
lone.addIdentity('id_0');
const lrun = G.createIdentityGroupingRun(lone, cams, 'dlt', createGroupSolver(cams, { method: 'dlt', triangulateOnly: true }));
check(G.groupFrameByIdentity(lone, 0, lfg, lrun) === undefined && lfg.unlinkedInstances.get('back').length === 1,
    'a frame with no identity on 2 cameras is left alone');
await G.sweepResidentFrames(sess, (fi) => { visited.push(fi); }, { start: 5, end: 9 });
check(visited.join(',') === '5,6,7,8,9', 'the resident sweep visits start..end in order: ' + visited.join(','));
check(JSON.stringify(G.framesToTriangulate(sess).slice(0, 3)) === '[0,1,2]' && G.framesToTriangulate(sess).length === 40,
    'framesToTriangulate: every frame, ascending');

console.log('-- 7. per-camera folder helpers --');
const s7 = new pd.Session(cams, skel, ['t'], 's7');
const fg7 = new pd.FrameGroup(0);
fg7.addInstance('back', new pd.Instance([[1, 2]], 0, 'predicted', 1));
fg7.addInstance('back', new pd.Instance([[3, 4]], 0, 'user', 1));
s7.addFrameGroup(fg7);
const n7 = B.moveInstancesToUnlinked(s7);
check(n7.user === 1 && n7.predicted === 1 && fg7.instances.get('back').length === 0 &&
    fg7.unlinkedInstances.get('back').map(x => x.instance.getX(0)).join(',') === '1,3', 'moveInstancesToUnlinked: all, in order');

console.warn = realWarn;
console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
