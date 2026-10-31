# LUCID — Label Unification and Correspondence in 3D

Multi-view pose annotation GUI. Vanilla JS ES modules served as static files — no build step.

## Architecture
`index.html` loads `app.js` (`<script type="module">`), which only imports `pose/initialization.js`. Per-module purpose/exports/dependents: **`MODULES.md`** (repo root).
- `pose/` — data model, cross-view tracking, DLT triangulation, plane/mesh/origin models, ID-switch checks, lazy residency, the tracker's detection-filter rule (`detection-filter-stats.js`, also what the Tracking Wizard counts with), behavior labels and detection, app init (pure logic, mostly DOM-free). Workers: Track All runs on one (`tracker-pool.js`); triangulation math is `triangulation-core.js`, solved in parallel by `triangulation-pool.js`; behavior features and rule detectors run in parallel on `behavior-pool.js`.
- `ui/` — rendering, interaction, panels, modals, timeline, 3D viewport, settings, exports.
- `loading/` — video decoding, session/SLP loading, file selection, web workers.
- `import-export/` — save/load, SLP import/merge, `metadata.lucid` serializers, mesh export.
- `demo-data.js` (synthetic data), `styles.css` (all styling).

## Local Development
```bash
python3 server.py 8080            # Windows: py server.py 8080 (or python3 -m http.server — no --offline)
# App: http://localhost:8080/   Tests: http://localhost:8080/tests/test-runner.html
python3 scripts/offline_deps.py install && python3 server.py --offline
```

## Deployment
`luc3d.sleap.ai` is the GitHub Pages custom domain (Cloudflare-proxied) for this repo; the `CNAME` at the `gh-pages` root makes it work. `.github/workflows/deploy.yml` publishes channels on `gh-pages`; `pr-preview.yml` owns `/pr/<n>/`. A deployed "build" is a copy of the repo at a ref.

Channels:
- `/` — newest **full release** only. `/stable/` — a one-file meta-refresh alias to root (deep links under it 404).
- `/latest/` — newest release incl. pre-releases. `/dev/` — every push to `main`. `/pr/<n>/` — previews, removed on PR close.
- Only a full release newer than the current one moves root, and the same run rewrites `/stable/`; channels only move forward. `workflow_dispatch` target `root` is the manual promote (writes root + alias).
- Releases are cut **by hand** (`gh release create vX.Y.Z --generate-notes`) — an Actions-created release would not fire `release: published`. Tags must match `vX.Y.Z` or `vX.Y.Z-N`.

Rules:
- **Sub-path safe:** keep every URL relative (importmap, imports, `document.baseURI` in tests). Never origin-root-relative (`/lib/...`) — it 404s on every channel but root. Alias redirect targets are relative too.
- **Pushing to `gh-pages`:** both workflows push under different concurrency groups, so any step that pushes must re-sync and retry in a loop (as `deploy.yml` / the preview cleanup do) and must **never force-push** (the preview deploy action runs with `force: false`). The cleanup triggers on `pull_request_target: closed` (plain `pull_request` doesn't run for conflicted PRs); it is safe only because it never checks out PR code.
- **Root is the only target that wipes.** Adding a channel means adding it to the keep-list (`find . -maxdepth 1 … ! -name` in `apply_targets`) or the next full release deletes it. Root is requested by an explicit `root: true` flag; keep the refusal of an empty/`.`/`/` path. `CNAME` is excluded from the wipe and recreated if missing. No `.nojekyll`.
- **Version stamp:** both workflows run `scripts/stamp-version.mjs <dir> <commit> <label>` on the staged copy (never the repo). It appends `?v=<commit>` to every same-site code reference (relative imports, `new URL('…js', …)` workers, `<script src>`, stylesheets, importmap entries, modulepreloads) so Cloudflare-cached modules can't mismatch a fresh `index.html`. It is all-or-nothing — the deploy fails if any relative module ref is left unstamped (it would load twice). Keep code references literal and relative: no computed `import(var)`, no root-absolute paths.
- The label fills the `lucid-release` / `lucid-commit` `<meta>` tags shown by `ui/version-label.js`: the tag when the ref is a release tag, else `dev`, or `PR #<n>`. The tags stay **empty in the repo**; the stamp fails if they are missing. The label never reaches the `.slp`.
- **Modulepreload:** stamping then runs `scripts/preload-modules.mjs`, inserting one `<link rel="modulepreload">` per module of the startup graph (static imports + importmap CDN entries) **right after the importmap** (Chrome ignores an importmap arriving after module loading starts). The list is derived from the stamped files, never hand-maintained; dynamic `import()` and workers are deliberately excluded. `tests/e2e/stamped-build.mjs` fails if the app requests a same-site module outside the block or any preload not exactly once — so a static import the walker can't see fails there.
- `yaml` loads as jsdelivr's single-file `yaml@2.8.0/+esm` (offline maps it to `lib/yaml/browser/index.js`). The head `<script>`s (mp4box, three, OrbitControls, h5wasm IIFE) are `defer` — they still run in order before any module script. Bundling was considered and rejected (adds a build step; deployed code would differ from tested source).
- Tests: `tests/test-stamp-version.mjs`, `tests/test-preload-modules.mjs`; `tests/e2e/stamped-build.mjs` stamps the working tree, loads the app and runs the browser suite on it — run it after changing imports, workers or `index.html`. Cold-start bench: `tests/e2e/_bench-cold-start.mjs`.

## Offline mode

Four deps load from jsdelivr by default (three.js 0.147.0, mp4box 0.5.2, dockview-core 6.6.1, yaml 2.8.0); everything else in `lib/` is committed. **`offline-deps.json` is the single source of truth** (versions, tarball URLs, integrity, file lists, CDN URL → local path). Its consumers:
- `scripts/offline_deps.py install|check|bundle|clean` — stdlib only; installs into gitignored `lib/<pkg>/` (each keeps a tracked `.gitignore` + `PROVENANCE.txt`); honors `LUCID_NPM_REGISTRY`. `check` fails if any `cdn.jsdelivr.net/npm/<pkg>@<ver>` in source disagrees with the manifest; `check --strict` also fails on any CDN URL outside `lib/` the manifest doesn't map — a deliberate exception goes in the manifest's `unmapped` block **with a reason**.
- `server.py --offline` (or `LUCID_OFFLINE=1`) rewrites mapped URLs in served `.html`/`.js`/`.mjs` (must cover `.js`: dockview is imported from `ui/sessions-panes.js` and `ui/overlay-export-modal.js`); refuses to start if a package is missing.
- `offline_deps.py bundle` stages a zip with URLs pre-rewritten plus the launchers in `scripts/offline-bundle/`.

Rules:
- Rewritten paths are **depth-relative** (`./lib/…`, `../lib/…` from `tests/`), never `/lib/…`.
- Node-only `.mjs` outside `lib/` is never rewritten (`is_rewritable`, shared by bundle and server) — `tests/test-stamp-version.mjs` holds CDN URLs as fixtures.
- `MP4BOX_CDN` (unpkg mp4box@0.5.4) in `lib/sleap-io/chunk-X76PRJK6.js` is intentionally unmapped: unreachable because `index.html`'s mp4box `<script>` always sets `globalThis.MP4Box` first. If that tag is ever removed, the unpkg fallback goes live.
- **The image ID-switch check is the one feature that does not work offline**: `TRANSFORMERS_URL` (`ui/image-embedder.js`) is `unmapped` because the model weights come from huggingface.co too. Offline it reports `MODEL_OFFLINE_REASON` and retries without reload via `importTransformers`. Test: `tests/e2e/image-check-offline.mjs`.
- Test: `tests/e2e/offline-server.mjs` (all off-origin requests aborted). Licenses in `lib/LICENSES.txt`.

## Dependencies

Loaded via `index.html` script tags / importmap (mirrored in `tests/test-runner.html`).
- **dockview-core pinned to 6.6.1 in THREE places** — `index.html` CSS, `ui/sessions-panes.js`, `ui/overlay-export-modal.js` (`check` enforces agreement). 7.x renamed `onUnhandledDragOverEvent`. `ui/overlay-export-modal.js` also uses dockview **internals** (sash resizing, feature-detected) and `styles.css` `#ovDock` its DOM; on a bump run BOTH `tests/e2e/overlay-export-sash-distribute.mjs` and `overlay-export-sash-tracking.mjs`.
- **transformers.js pinned to 4.3.0** + `onnx-community/dinov2-small`, fetched only when the opt-in image ID-switch check runs. The check is calibrated to this model's embeddings of `cutCrop`'s crops: bumping either, or changing the crop, needs recalibration (MODULES.md `pose/id-switch-check.js`). Without a GPU it runs the **fp32** model (never int8) in `ui/image-model-worker.js` (MODULES.md `ui/image-embedder.js`).
- **Video export = mediabunny via `ui/video-encode.js`, the one encoding seam.** Do NOT add a second muxer (`lib/mp4-muxer/` was removed). Re-verify the encoder contracts in `lib/mediabunny/PROVENANCE.txt` on any mediabunny bump. mediabunny **1.30.0** also drives sleap-io.js decode, so a bump must retest both decode and encode.
- **h5wasm 0.10.3, vendored** (`lib/h5wasm/`, ESM + IIFE); no CDN h5wasm on any path. Local patch `LUCID local patch (cold-read)` in BOTH `hdf5_util.js` and `h5wasm.iife.js`: WORKERFS re-reads any all-zero part of a read **plus the 4 KiB block on each side** and fills in only zero bytes (`lucidColdReadGuard`) (cloud files read mid-download return zero pages). Every h5wasm File reader goes through it (SLP worker, `frame-worker.js`, sleap-io.js's streaming worker via `importScripts` of the IIFE); reads that bypass WORKERFS (video, Load 3D Points) use `loading/cold-read-guard.js`. Test: `tests/test-workerfs-cold-read.mjs`. (These files are binary-ish: use `grep -a`.)
- **sleap-io.js 0.5.5**, vendored in `lib/sleap-io/` (`index.browser.js`, `chunk-X76PRJK6.js`, `chunk-H7G4PJNA.js`, `chunk-YS7Q6CO6.js`, `gdrive-6DDSPUUK.js`; `mediabunny-stub.js` is unused), built from source at `e8dbaef8c` via `scripts/revendor-sleap-io.sh` — re-vendor from the npm dist once 0.5.5 is published. Writes **SLP 2.8** (columnar `/session_data` for 3D + grouping, slim `sessions_json`). Its `pako`/`mediabunny` deps are vendored in `lib/`. Default video decode is its `MediaBunnyVideoBackend` (opt out: `LUCID_VIDEO_BACKEND='html5'`).
  - **Read**: `parseSlpViaSleapIO` (`import-export/file-io.js`) → `readSlpStreaming`; grouping via `reconstructInstanceGroupsFromSession` (`slp-import.js`). Raw worker `parseSlpH5` remains for analysis `.h5` and fallback (`parseSlpForImport` dispatches; that path uses `reconstructInstanceGroupsFromDicts`).
  - **Write**: `saveSlpToBytes` / streaming `openSlpWriter` over `buildSlpLabelsAllViews`'s typed graph. Interop gate: `scripts/validate_slp_sleap_compat.py`.
  - **Bump**: `npm pack @talmolab/sleap-io.js@<ver>`, copy `dist/index.browser.js` + its chunk closure (keep `gdrive` chunk + `mediabunny-stub.js`), drop orphaned chunks, check exports and importmap bare imports, re-apply patches, run the suite.

### sleap-io.js local patches
**Re-apply every one after any re-vendor (grep `LUCID local patch` with `grep -a`), and report upstream.** Byte-level guard for all write-side patches: `tests/e2e/save-golden-digest.mjs` **must not move**.
- `(#115)` `chunk-X76PRJK6.js` `MediaBunnyVideoBackend`: `sample.close()` after `toVideoFrame()` (VideoSample leak); sort `_frameTimes` by timestamp in `initialize()` because `packets()` yields DECODE order (B-frame videos returned wrong frames). Test: `tests/e2e/mediabunny-bframe-decode-order.mjs` (regenerate fixture video + PNGs together: `tests/fixtures/bframes-test/make_fixture.sh`).
- `(luc3d frame-index)` same `initialize()`: `packets(…, { metadataOnly: true })` (indexing skips payloads). Test: `tests/e2e/mediabunny-frame-index-metadata-only.mjs`.
- `(sleap-io.js#231)` writes the `instances` table as `"<d"` not `"<f8"` (h5wasm parses `"<f8"` as float32, corrupting `point_id_start/end` past 2^24). `points`/`pred_points` stay f32 on purpose. Test: dtype check in `tests/test-lazy-reopen.js`.
- `(luc3d #191)` (+ older `#185`, `#190`, `#185/#189`) `writeSessions` streams the `/session_data` tables through `LucidAppendTable` (8192-row buffer; a table that never overflows is written one-shot, keeping small files byte-identical; datasets are created on first flush, so an empty table such as `pred_points_3d` stays absent). The #185/#190 sinks are retained but unused there; `instanceGroupMemberRowsInto` must stay in sync with `instanceGroupMemberRows`. Test: `tests/e2e/save-session-3d-typed-sink.mjs` (asserts > 8192 frame groups so flush boundaries stay covered).
- `(luc3d #189)` `chunk-H7G4PJNA.js` `reconstructColumnarFrameGroups` emits a flat `Float64Array(3N)` per instance group (LUCID's `points3d` shape) instead of boxed rows — compacted, never a `subarray` view (the predicted table is stride 4 and a view pins the whole matrix); `Instance3D.nVisible` handles flat; writer accepts flat or boxed (`pushFlat`, `lucidCount3dRows`). Tests: `tests/test-slp-export-canonical.js`, `test-lazy-reopen.js`, `test-slp-streaming-write.js`.
- `(luc3d #193)` `normalizeStructData`/`normalizeStructDataset` build `Float64Array` columns (f64: ids exceed 2^24; convert `BigInt64Array` explicitly or the store silently comes back empty). Test: `tests/e2e/reopen-store-columns-typed.mjs`.
- `(luc3d eager-save)` `writeLabeledFrames` appends the 2D tables through `LucidAppendTable` (`push` takes 10 values); `from_predicted` resolved by a pre-pass; boxed `emitInstancePoints` is kept for `SlpStreamWriter`. Test: `tests/e2e/save-2d-tables-append.mjs`.
- `(save progress)` `SlpStreamWriter.appendStore`'s body moved unchanged into the generator `_lucidAppendStoreSteps` (yields after each 5,000-frame window); `appendStore` runs it straight through (same bytes), and the added `appendStoreAsync(store, opts, onWindow)` awaits `onWindow(done, total)` between windows so File ▸ Save's progress card can count the 2D poses (`streamSessionIntoWriter`). Without it the save still works and that step shows the spinner. Test: `tests/e2e/save-progress-card.mjs`.
- `(luc3d gzip)` gzip level 1 on the big tables: always on the streaming writer's four 2D tables (`createAppendableMatrixDataset`, always chunked); on the eager 2D tables and the three `/session_data` struct tables only once one overflows its 8192-row buffer (`LucidAppendTable` gzip `"chunked"`), so small files stay contiguous and byte-identical. Values/shapes/dtypes unchanged (any HDF5 reader). Real 8-camera project: 2D 1.49 GB → ~0.70 GB. Cost: save ~+0.23 µs per `pred_points` row (~15 s on 57.8M rows), load ~+4 s of inflate. Test: `tests/e2e/frame-identity-compact-roundtrip.mjs` (file layout).
- `(open progress)` `readFromStreamingFile` takes `onReadBytes(done, total)` (from `readSlpStreaming`'s `options.onReadBytes`): it first sizes the four big tables from their metadata (`lucidPlanTableReads`), then reads a flat 2-D table in row slices of 512K elements (4 MB) into one pre-sized typed array (`lucidReadRowsChunked`), counting each slice; a compound or vlen table is one whole read, counted at its end. Without the option nothing changes. Lets the Open project card count megabytes. `globalThis.__lucidReadSliceElems` shrinks slices for tests. Test: `tests/e2e/project-open-progress.mjs` (a sliced read equals a whole read, value for value).
- `(luc3d newer-format)` `readFromStreamingFile`, `readSlp` and `readSlpLazy` (chunk-X76PRJK6) throw `UnsupportedSlpFormatError` for `format_id` ≥ 3.0, and `loadSlp` doesn't retry it. The wording says LUC3D and matches `loading/slp-format-check.js`; upstream's guard (olinesn/sleap-io.js `fix/refuse-newer-slp-format`) says sleap-io.js, so keep this wording if a re-vendor brings that in. Test: `tests/e2e/slp-newer-format.mjs`.
- `(smoothing source)` `writeSessions` (chunk-X76PRJK6) appends `Instance3D.lucidUnsmoothed` (a smoothed group's 3D from before Temporal Smoothing) to `/session_data/unsmoothed_points_3d`, ROW-PARALLEL to `points_3d` (NaN rows for groups without one; padded to `points_3d`'s length; created only once a group has one, so never-smoothed files are byte-identical); both `/session_data` readers (`readSessionDataStreaming`, `readSessionDataEager`) return it and `reconstructColumnarFrameGroups` (chunk-H7G4PJNA) sets `instance3d.lucidUnsmoothed` (compacted; all-NaN rows = none). SLEAP ignores the dataset. Test: `tests/e2e/smoothing-save-reopen.mjs`.
- `(luc3d events)` Frame-spanning events, the same model and file layout as Python sleap-io (SLP 2.6): `Event`, `UserEvent`, `PredictedEvent`, `EventType`, `cloneEvent`, `Labels.events` / `eventTypes` / `_collectEvents`, the `/event_types` + `/events` groups, read by all three readers and written by the eager, lazy, streaming and merged writers. Copied from the build of `olinesn/sleap-io.js` `feature/events` (`src/model/event.ts`, `src/codecs/slp/events.ts`); drop the patch once a sleap-io.js release has it. No events and no types writes nothing, so other files stay byte-identical. Tests: e2e `behavior-label-roundtrip`, `behavior-cross-app`, `save-golden-digest`.
- **Not a patch, but uses bundle internals:** `makePooledSioInstances` (`import-export/file-io.js`) serves `SIO.Instance`'s private `_xy`/`_visible`/`_complete`/`_score`/`_n`/`_names`; after a re-vendor check they still exist and run `tests/test-sio-pooled-instances.js`.
- **Obsolete — do NOT re-apply** the old #134 `serializeInstanceGroup` patch (function no longer exists under SLP 2.8; `tests/e2e/save-no-inline-dup.mjs` guards the layout).

**Memory rule behind #185–#193 and eager-save:** the renderer's V8 heap is hard-capped near 4 GB; big per-row data must live in typed arrays (outside the cage) or be flushed incrementally — never one boxed JS array per keypoint/row.

## Project state in `metadata.lucid` (Visibility + Temporal Smoothing + Define Planes + Behavior labels)

Per-session state, written to each session's dict by exactly four modules:
- `import-export/visibility-metadata.js` (`VISIBILITY_METADATA_KEYS`): `videoBrightness`/`videoContrast`/`videoRotation`, `hiddenCameras`/`hiddenTracks`/`hiddenIdentities`, `idSwitchReview`, `numAnimals` (a count or `"auto"`; absent until asked, so a reopened session asks again; changing it marks the project dirty).
- `import-export/smoothing-metadata.js` (`SMOOTHING_METADATA_KEYS`): `temporalSmoothing` (last run's scope + windows), written only while something is smoothed. Each smoothed group's source 3D goes with the 3D instead (`/session_data/unsmoothed_points_3d`, the `(smoothing source)` patch); readers re-mark it with `adoptSmoothingSource` once `points3d` is final.
- `import-export/plane-metadata.js` (`PLANE_METADATA_KEYS`, mapping in DOM-free `pose/plane-serialization.js`). `planeNodes`/`planes`/`meshObjects`/`planeOrigin` are project-scoped: the same in every session, read from the first one ingested. `planePlacements` is per session and must not leak.
- `import-export/behavior-metadata.js` (`BEHAVIOR_METADATA_KEYS`): `ethogram` (project-scoped like the plane pool; a later session maps onto it by behavior name; it also carries each behavior's `detector` and the project's feature settings `features`, each only once set), `behaviorLabels` (the session's bouts), `behaviorPredictions` (the session's predicted bouts, only when there are any) and `behaviorStates` (the session's keypoint-MoSeq syllables, only when there are any). See "Behavior labels" below.

Writers: `file-io.js`, `slp-streaming-write.js`, `save-load.js`. Readers: `handleLoadSlpFile`, `handleLoadProjectSlpLazy`, `handleLoadSessionFolderSingleSlp` and the `save-load.js` reader. A new setting changes only its module. A new loader that builds a Session by hand must call ALL FOUR readers (visibility, smoothing, plane, behavior), or settings, smoothing, planes and behavior labels and predictions silently reset to defaults.

`frameIdentityMap` (one entry per grouped detection after Track All: 101 MB of JSON on an 8-camera project) is written by `import-export/frame-identity-metadata.js` as only the entries the instance groups do not imply, plus `frameIdentityMapFromGroups: true` and `frameIdentityMapDropped`; the two `.slp` readers that rebuild groups call `readFrameIdentityMetadata` AFTER doing so. Never larger than the full list (falls back to it, without the marker); the project JSON keeps the full list. `Session._derivedFrameIdentity` (save) and `stampFrameIdentitiesFromGroups` (load) must keep the same rule. Test: e2e `frame-identity-compact-roundtrip`.

The file-level `provenance` (outside `metadata.lucid`) is kept too, via `import-export/slp-provenance.js`, because campy-live-export stores the recording's session.json there. The three `.slp` readers set `session.fileProvenance`, and project saves write `buildSaveProvenance(sessions)`, never a fresh `{source, exported_at}`. Exports keep the fresh pair. Tests: `test-slp-provenance.mjs`, e2e `slp-provenance-roundtrip`. The per-camera and multi-session folder loaders are NOT readers: they ignore a root `.slp` and read camera-folder `.slp` files as 2D only.

Rules:
- **Never write defaults.** `tests/e2e/save-golden-digest.mjs` must not move.
- **Never touch other `metadata.lucid` keys** (keeps files SLEAP-readable).
- **Address plane points and membership by node ID, never by pool index**, and keep the file's IDs (`adoptNode`/`adoptPlane`).
- **Call `resetPlaneState()` and `resetBehaviorState()` before every load** (`readPlaneMetadata` only fills an empty model; the ethogram is a project-scoped singleton too). Exception: the single-SLP folder loader with a project open adds a session without resetting, and reads plane 2D only if `planeMetadataConflict` finds every node and plane matching by id and name (node IDs restart at 1 in every project).
- **Call `markDirty()` on every plane or node mutation** (not the browser-local size sliders).
- Global appearance prefs stay in `localStorage.visibilitySettings`, never the `.slp`.

Tests: `test-visibility-metadata.js`, `test-plane-serialization.mjs`, e2e `visibility-settings-roundtrip`, `plane-persistence-roundtrip`, `reopen-folder-session-settings`.

## Behavior labels (ethogram, bouts, Label Behaviors mode)

Manual behavior labelling: an ethogram per project (`pose/behavior-ethogram.js`), bouts per session (`pose/behavior-labels.js`), saved by `import-export/behavior-metadata.js` and `import-export/behavior-events.js`, labelled from Label Behaviors mode (`ui/behavior-mode.js`), the Behavior tab (`ui/behavior-panel.js`, last tab) and the timeline (`ui/behavior-timeline.js`), exported and imported as CSV/TSV and BORIS files (`import-export/behavior-io.js`).
- **In a `.slp`, labels and predicted bouts are sleap-io EVENTS** (`/event_types`, `/events`), the one form sleap-app and Python sleap-io also read and write (`import-export/behavior-events.js`, whose header lists every field and metadata key; change the vocabulary together with sleap-app's `src/lib/behaviorEvents.ts`). `metadata.lucid` keeps only what no other tool reads (`writeBehaviorSlpMetadata`: detectors, classifier settings, feature settings, run settings prints, the labels' fps, states). A project JSON keeps everything in its dict (`writeBehaviorMetadata`), and `readBehaviorMetadata(session, lucid, fileEvents, scope)` reads both forms, so a new `.slp` loader must pass `fileEventsOf(labels)` (or `slpData.behaviorEvents`) and `sessionEventScope(rawSessions, nVideos, sessionIdx, onlyThisSession)`, which gives each event to one session. A bout is one event on the session's FIRST camera, and an event on ANY camera of a calibrated session is that session's. A "not" label is its own type (`not rear`, metadata `negates: rear`), so counting `rear` never counts it. An event labels cannot hold (no subject; a social one with no target) is kept per session and written back unchanged. So is an event on a video that is not one of the session's cameras, and it stays on that video: a video in no recording session (the file's first session keeps it, and a save writes the video back from what the file said about it, since the writers write one video per camera), another session's video when a loader opens only one session, another video of a file with no recording session or of a session with only placeholder cameras, whose videos need not be cameras of one recording (a save gives a file with no session exactly that session, so the two must read alike; such an event remembers its camera's place in the session, never the file's video number) (e2e `behavior-events-outside-session`). A prediction another tool wrote makes a run with source `'file'`, which never goes stale and keeps what the file said (`bout.ext`).
- **Animals on one camera with no identities are its tracks** (`sessionAnimals`, `drawBehaviorBoxes`), so a plain SLEAP file can be labelled; on several cameras tracks are per camera, so identities are required.
- **Frames are the truth.** Anything in seconds converts with the session's fps; fps is not saved in the `.slp`, so labels and exports record the fps they used.
- **Animals are identity NAMES** (ids are renumbered on reopen, like `idSwitchReview`), so every identity change must carry the bouts: `Session.swapIdentitiesInRange` does through `onIdentitySwap` (`pose/pose-data.js`; ID-switch Fix/Undo and identity reassignment), Rename Identity calls `renameAnimal`. A pass that hands names out afresh asks first on a labelled session (`confirmRetrackWithBehaviorLabels`: `runTrackingPass`, Propagate Tracks → IDs). Delete Identity keeps the bouts as orphans (reassign or delete in Summary).
- **Unlabelled is not absent.** A bout is present or explicitly "not"; "Mark reviewed" is the only way unlabelled frames become "not".
- **Mutual behaviors** (social with no actor and target) store each pair ONCE, in name order: build every key through `keyOf` (never concatenate names), and anything per animal (lists, time budget, timeline rows, exclusive peers, running-state lookup) must count both animals of a mutual pair. Making a behavior mutual re-keys its bouts (`rekeyBehavior`; present wins).
- **Labels live on the session (a WeakMap), never on FrameGroups** — no eviction blocker, nothing to hydrate on lazy projects. Never mutate a stored interval (undo keeps list references).
- **Enter/leave Label Behaviors mode only via `toggleBehaviorMode`** (it closes running states). It and Defining Plane Mode exclude each other. Its capture-phase listener takes only behavior letters (Shift+letter only with a range armed), `1`–`9`, `[ ] , .` (Alt allowed: non-US layouts), `Mod+Z`/`Mod+Shift+Z` and Esc; everything else must fall through. Behavior keys are letters only (digits pick the animal).
- **Never write defaults:** no ethogram and no bouts write neither key. A running state is saved as ending at the cursor without being stopped.
- **A label edit never walks frames:** the timeline's Behavior rows come only from the provider and repaint via `refreshBehaviorRows()`; panel/timeline/overlay repaints go through `requestBehaviorRepaint` (one animation frame). Video boxes (around each behavior's animals, both for a social one) come in as `drawFrameOverlays`' `behaviorBoxes` option from `behaviorBoxesAt` (overlays.js stays import-free), take an animal's extent from its 2D detection AND reprojection (never gated on the User/Predictions toggles), and are browser-local on/off. The coding state (the mode's light numbered box on the selected animal, the actor awaiting a target, the dashed "behavior?" box on the predicted bout selected for review) is `{live: true}` only: never in an exported video.
- `switchSession` has no behavior hook (brace-walked tests inject a fixed name list); `updateBehaviorFrame` closes a left session's running states.
- Behavior colors: `BEHAVIOR_COLORS`, held by `tests/test-track-identity-palette.mjs` §10 — change the palette, never the thresholds.
- **Files in and out** (Behavior ▸ Import / Export Labels…): `import-export/behavior-io.js` (DOM-free) holds every layout, `ui/behavior-io-modal.js` the dialogs. Frame numbers in files count from 0; a bout's end time is `(end + 1) / fps` (BORIS's stop), so import is `start = round(t × fps)`, `end = round(t_stop × fps) − 1` — change both directions together (the Node test round-trips at 30, 29.97, 120 and 1000 fps). BORIS tables carry present bouts only (no "not" labels) and a social target as `Modifier #1`; their `Image index` is the bout's first and LAST frame, because SimBA reads it and labels START…STOP inclusive (the Node test reads the export the way SimBA does). An import writes only the active session, as ONE `labelBouts` edit (one undo step), and adds behaviors only through `ingestEthogram` (by name).

Tests: `test-behavior-labels.mjs`, `test-behavior-metadata.mjs`, `test-behavior-events.mjs`, `test-behavior-io.mjs`, e2e `behavior-labelling`, `behavior-label-roundtrip`, `behavior-import-export`, `behavior-cross-app`, `behavior-refused-file` (a refused `.slp` keeps the open project's behavior work); real recordings `_real-behavior-cross-app.mjs` (LUC3D → sleap-app → LUC3D with every label, pose and setting compared, each step timed, tab crashes reported: `DATASET=<folder> NANIMALS=5 OUT=<dir>`, `SINGLE=1` for one camera's folder).

### Behavior detection (features, rule detectors, predictions)

Rules detect behaviors from the poses; their predicted bouts are reviewed into labels. Input `pose/behavior-tensor.js`, features `pose/behavior-features.js` (pure), rules `pose/behavior-rules.js`, scores → bouts `pose/behavior-postprocess.js`, workers `pose/behavior-pool.js` + `behavior-worker.js`, predictions `pose/behavior-predictions.js` (shared by every later source: classifiers, keypoint-MoSeq, discovery), running `ui/behavior-detect.js`, settings `ui/behavior-rules-modal.js`, review in the Behavior tab's Predictions section.
- **Features never read absolute world coordinates**: everything is put in the FLOOR frame and reported only as heights, distances, angles and differences over time — never a horizontal coordinate or an absolute heading (the horizontal axes are built from `up` alone and are arbitrary). `test-behavior-features.mjs` §5 rotates and translates the world with its floor, on noisy data, with an explicit and a fitted floor; a new feature must pass it.
- **The floor**, in order: the plane chosen as the floor (Behavior settings; its normal turned toward the animals), the Set Origin frame (+Z up; the default when set), else `fitFloor` (all nodes' lower envelope, tilted level with the bodies' head–tail base axes — not the points' spread, which a tall cage fools; up decided by the cameras, else the heights' skew, else flatness). A choice that cannot be honoured falls back to the fit and says so. 2D has no floor and no heights.
- **Lengths are in body lengths** (each animal's session-median head–tail base distance), so thresholds read the same in mm, cm or px. Node roles are guessed from names; saved settings are by node NAME, only what differs from the guesses (`ethogram.features`, never defaults).
- **Animals are identity names; 3D comes from `session.instanceGroups` by `group.identityId`** (resident on lazy projects); without identified 3D, ONE camera's 2D — lazy rows straight from the store columns, nothing hydrated.
- **Chunks, never the matrix:** `computeFeatureChunk` computes only the requested features over a chunk padded by `featurePadFrames` (chunked = one pass, tested); the frames × features matrix is never built (the 4 GB heap rule). Big arrays are typed arrays.
- **Workers:** `new URL('pose/behavior-worker.js', document.baseURI)` stays literal (stamp-version); no bare imports anywhere in the worker graph (features, rules, postprocess import nothing else); kill switch `window.LUCID_BEHAVIOR_WORKER = 0` (N > 0 forces N); inline below `MIN_WORKLOAD`. Inline and pool scores must stay bit-identical (`behavior-rules.mjs` §1).
- **Broken tracking is skipped, never scored:** `brokenPoseFrames` (`pose/behavior-tensor.js`) marks an animal-frame broken when a used node is more than 1.2 BL from the body's middle (per-axis median of the used nodes) or the middle jumps more than 0.5 BL and faster than 20 BL/s, plus the frames a speed could span across a jump. Every slice of poses passes its masks (`sliceChunkData`'s `skip`: Detect, This frame, the settings' histograms), so those frames read as missing. A change to what it skips bumps `POSE_CHECK_VERSION` (in `runConfigPrint`, so older runs go stale). It catches only clearly impossible poses: a coherent but wrong 3D pose still scores.
- **Decisions are exact, scores are soft:** a frame where every condition holds scores ≥ 0.5, any other < 0.5, a missing feature NaN (unscored, a gap). Gaps are joined first, then a bout needs at least the shortest bout of DETECTED frames (bridged gaps do not count: on real recordings, counting the span let scattered flickers make bouts). Detector durations are seconds, converted with the session fps.
- **Predictions:** one run per behavior per session (a WeakMap, never on FrameGroups), from its rule or its classifier; saved as bouts only (`behaviorPredictions`, only when non-empty; per-frame scores stay in memory). Accept = `labelBouts` present, source `'a'`; Reject = "not", source `'m'`; each (and Accept all in range, which asks first) is ONE undo step. Review status is read from the labels, never stored.
- **Stale** = settings (`runConfigPrint`: detector + the RESOLVED body + the floor in effect (`floorKey`) + fps + 2D/3D) or poses (`predictionInputPrint`, an order-free hash of every coordinate with frame and identity). Edits only note a possible change (the `onMarkDirty` hook — keep hooks cheap); the re-print runs sliced between tasks after the edits stop. Never walk the poses on a draw: the blue button reads cached state. Label edits never make a rule run stale. Rename Identity renames predictions (`renamePredictedAnimal`); an identity swap makes them stale.
- **Toolbar:** Behavior is a `wireActionMenu` — a click (All frames) detects; This frame shows the readout (writes nothing); the menu also holds Label Behaviors mode (`#tbBehaviorMode`, lit while on; Mod+Shift+L and the Behavior menu still toggle it) and Settings…. Blue while predictions stand (`hasBehaviorResult`, `pose/run-state.js`).

Tests: `test-behavior-features.mjs` (§9 broken tracking), `test-behavior-rules.mjs`, `test-behavior-metadata.mjs` §5, e2e `behavior-rules` (planted rearing → exact bouts; inline = pool), `behavior-label-roundtrip` (both writers), `toolbar-hover-menus` §4–5; timing `_bench-behavior-detect.mjs` (180,000 frames × 5 animals × 15 nodes, every template: ~2 s); real recordings `_real-behavior-detect.mjs` (load, Track All, Triangulate All at a 25 px reprojection error threshold (`REPROJ=`, the Tracking Wizard's; without it large tracking errors read as behavior), then every stage timed with heap / renderer memory and long tasks, bout counts, and a contact sheet of bouts on the video).

### Behavior classifiers (forests from labels, active learning)

A behavior can be learned from its labels: one binary random forest per behavior (`pose/behavior-forest.js`, pure; the worker runs it), trained, cross-validated and run by `pose/behavior-classifier.js` (DOM-free), in the app by `ui/behavior-train.js` and Behavior ▸ Train Classifiers… (`ui/behavior-train-modal.js`); classifier files `import-export/behavior-model-io.js`. A behavior with a classifier that can run on the session is predicted by it INSTEAD of its detector (`detectOnContext`, the toolbar button); its run's source is `'model:<hash>'`.
- **Deterministic, so never saved:** forests live in memory only (`resetBehaviorState` drops them and the row cache); the `.slp` keeps the labels, each run's hash and its label print. Every tree is seeded from (seed, fold, index) alone, training rows are computed in a FIXED chunk layout (`TRAIN_CHUNK_FRAMES`), importance is summed in job order: the same labels and poses give the same forest — the same hash — inline or on any number of workers (`behavior-classifier.mjs` e2e §1, Node §4). Keep it that way.
- **Bins are thresholds:** values are 16-bit order-keeping codes (bfloat16 resolution), ≤ 255 quantile bins + bin 0 for missing; `finalizeForest` turns "bin ≤ b" into "value ≤ `codeUpperValue`(edge)", and raw scoring must equal binned scoring exactly (`test-behavior-forest.mjs` §1, §6). A change to codes, bins or the tree format bumps `FOREST_VERSION`.
- **Memory:** the training matrix is Uint8 features × rows, capped (`rowCap`: 40,000 rows and `ROW_BUDGET_CELLS`); above the cap each class keeps the frames whose seeded hash is under 2^-k, plus one frame per bout. Workers get it ONCE per run through the pool's `opts.shared` (no SharedArrayBuffer: each holds a copy, so `TRAIN_WORKERS` caps them and `releaseShared` frees them). The kernels compute one animal's columns at a time (`computeFeatureChunk`'s `only` + `memo`); never the frames × features matrix.
- **Samples:** labelled frames whose animal (and target) have at least half the used nodes and are not broken tracking; a mutual pair gives a row per direction and scores as their mean; an exclusive peer's present frames count as "not" where the behavior's own labels say nothing. Gate: 20 frames and 2 bouts per class (`gateProblems`).
- **Cross-validation is grouped, never random frames:** by session (≥ 2 with both classes), else by animal / pair, else 5 contiguous time blocks whose training leaves out the rows within `featurePadFrames` of the block. The kept out-of-fold scores (`cv.oof`) let the threshold dialog recompute P/R/F1 without retraining (`evaluateAt`).
- **Stale:** a classifier run also goes stale when the labels it learned from change (`trainingLabelPrint`, cached by `labelPrintOf`; label edits invalidate it through `onBehaviorChange`, loads through `noteLabelsChanged` — a new loader that reads labels must call `readBehaviorMetadata`, which does) and when a newer model replaces its own (`detectionEnv().labelPrintOf / modelHashOf`). A reopened project has no model: its runs stand while the labels are unchanged. A changed threshold / shortest bout / gap re-derives the bouts at once from the scores in memory (`applyClassifierPost` → `replaceRunBouts`); runs without scores go stale. Rule runs never read labels and stay as they were (exclusive groups apply among classifier runs only).
- **Never write defaults:** `behaviors[].post` only where it differs from `DEFAULT_POST` (0.5, 0, 0); no classifier writes nothing (`save-golden-digest.mjs` must not move).
- **Training rows are cached per session NAME and chunk** (a print of that chunk's poses, skipped frames and the feature spec): retraining after a few labels computes only their chunks, and a lazy project's other sessions train from rows cached while they were shown, or are named as left out. Budget 256 MB, least recently trained sessions first.
- **Active learning** (Classifiers section): Next uncertain ▸ = unlabelled stretches near the threshold, most total uncertainty first; Next disagreement ▸ = labelled frames the predicted bouts contradict; Accept / Reject are one `labelBouts` edit each. Its video box is `setReviewFocus(…, 'learning')`: the Predictions list clears only its own ('predictions').
- **Model files** carry the feature-spec version, roles, used nodes, windows (s) and bands (Hz); import refuses another feature spec, 2D/3D, roles or body (`modelFileProblems().blocking`), and the stored hash must match the forest.

Tests: `test-behavior-forest.mjs`, `test-behavior-classifier.mjs` (a planted kinematic behavior: cross-validated and held-out F1 > 0.9), e2e `behavior-classifier`; timing `_bench-behavior-classifier.mjs` (1 h × 5 animals × 15 nodes, 50 fps, 12 cores: training two behaviors on 6 labelled minutes 4.1 s, retraining from the row cache 0.3 s, predicting the hour 6.1 s, This frame 24 ms).

### Behavioral states (keypoint-MoSeq round trip)

keypoint-MoSeq runs in the user's own Python, through files: Behavior ▸ Export for keypoint-MoSeq… writes the poses and `luc3d-kpms.yml`, Behavior ▸ Import keypoint-MoSeq Results… brings its syllables back as a states layer (`ui/behavior-kpms-modal.js`; formats `import-export/kpms-io.js`; layers `pose/behavior-states.js`; Behavior tab States section and timeline rows `ui/behavior-states.js`).
- **Its formats only, never its code** (academic-only licence), as keypoint-MoSeq 0.8.4 reads and writes them. `tests/fixtures/kpms/` is a real fit's output (`make_fixture.sh <python>` remakes it from the calls LUC3D writes); a format change is checked against it.
- **Export:** a table per session × animal named `<session>__<animal>.csv` (`recordingName`: parts fold `__`, so it only ever joins them). 3D in anipose's layout (`{bp}_x,_y,_z,_score` per node, then `fnum`) — keypoint-MoSeq finds bodyparts with `(\w+)_x,\1_y,\1_z`, so node names are written `\w+` (`kpmsBodypartNames`; the manifest lists renames); 2D in DeepLabCut's single-animal layout. Coordinates in the FLOOR frame (z up: keypoint-MoSeq turns poses about z), from `buildBehaviorContext`, so the floor rules above apply. Score 0 = missing (empty coordinates) or broken tracking (coordinates kept). use_bodyparts = the features' used nodes; anterior / posterior = head / tail base.
- **`luc3d-kpms.yml`**: keypoint-MoSeq's settings as top-level keys, `recordings` (session, animal, `first_frame`, frames) and `calls`. Every value is a JSON-style flow value, because `parseKpmsManifest` reads only that (no YAML library); PyYAML reads it too. The calls must run as a plain script: they select matplotlib's Agg backend first (keypoint-MoSeq's progress plots call `plt.show()`, which blocks a script on macOS).
- **Import:** `results.h5` through `readBlobArrayBuffer` into h5wasm's MEMFS (`readResultsH5`); `syllable` is int64 (keypoint-MoSeq turns on 64-bit jax), so convert BigInt64Array (`syllableArray`). A picked folder offers only `results.h5` and the CSVs of folders named `results` — never `checkpoint.h5` or the pose tables. Row i is frame `first_frame + i`; recordings map by the manifest, then the name, then its parts, else the dialog asks.
- **Storage:** layers per session (a WeakMap; never on FrameGroups) of Int32Array runs `[start, end, state]` — never a per-frame array; saved as `behaviorStates` only when a layer has a run, and read back replacing the session's layers (never merged); `resetBehaviorState` drops every session's. Layers are generic (`source` `'keypoint-MoSeq'` | `'discovery'`): later sources add layers through `setStateLayer`.
- **Animals are identity names here too:** `onIdentitySwap` swaps runs on the swapped frames (`swapStateAnimalsInRange`), Rename Identity calls `renameStateAnimal`, and a pass that hands names out afresh asks first on a session with states as on a labelled one (`confirmRetrackWithBehaviorLabels`).
- **Never walk frames on repaint:** the timeline's "States" group comes from the `behaviorRows` provider (rows with `group: 'States'`), segments cached per layer version; `refreshBehaviorRows()` repaints. State k takes `STATE_COLORS[k]` (keypoint-MoSeq numbers syllables by use), the rest `OTHER_STATE_COLOR`; held by `test-track-identity-palette.mjs` §11 — change the palette, never the thresholds.
- **Enrichment** is P(behavior | state) = present / (present + "not") over labelled frames only, a frame counted once per animal and behavior (the time budget's rule); unlabelled frames are not "not".
- **Promote:** the checked states of a layer become ONE behavior's run (`source: 'states:<layer>'`, lasting non-social behaviors only), replacing a rule or classifier run after a confirm, reviewed in Predictions like any run. Its `cfg` is the layer's `layerPrint` (runs only) plus the states: re-importing or removing the layer makes it stale; naming a state or renaming an animal does not. A bout's score is its main state's labelled share (NaN without labels).

Tests: `test-kpms-io.mjs`, `test-track-identity-palette.mjs` §11, e2e `behavior-kpms-roundtrip` (export → a real fit's results → import → States → promote → save → reopen; ID-switch, rename), `behavior-label-roundtrip` (both writers carry states); real data `_real-kpms-roundtrip.mjs` (load, Track All, Triangulate All at 25 px, export a stretch, fit keypoint-MoSeq briefly in `KPMS_PY`, import, save → reopen).

## Lazy projects: resident frames are a window, and eviction must be lossless

On a lazy project `session.frameGroups` holds only hydrated frames. Playback evicts them via `evictLazyFrames` (`pose/triangulation.js`), with the rules in DOM-free `pose/lazy-residency.js`.
- **Evict a frame only if re-hydrating it rebuilds exactly what is resident** (`frameEvictionBlocker` compared with `SioLazyLoader.describeStoreFrame`). The blocker list is in MODULES.md `pose/lazy-residency.js`; a new kind of in-memory-only state must be added to it. A user instance always pins its frame, group member or not: the streaming save reads 2D edits from resident frames only.
- **Never evict from `batchLoadLazyFrames`.** `sweepLazyFrameWindows` holds residency for its whole run.
- **`LAZY_PLAYBACK_AHEAD` is both the loader lookahead and the protected window.** Never split them. `LAZY_KEEP_BEHIND` (512) must stay above `MAX_TRAIL_FRAMES` (500, `ui/trail-presets.js`), because trails draw resident frames only. A seek hydrates ahead, so the draw path fills the trail behind it (`ensureLazyTrailWindow`), and a non-resident frame ENDS a lazy trail (`trailWindowFrames`' `residentOnly`).
- When you drop a frame, also drop its derived reprojection caches.
- `session.instanceGroups` is never evicted. Members' 2D returns to the store when their frame leaves residency (`releaseFrameMembers2d`), but only when `member2dReleaseBlocker` allows it. Re-adopt by the member's current `_rawInstIndex`, never a cached row (store compaction renumbers released members).
- **Never write the shared `lazyPlaceholderXY` buffer.** Every in-place writer on `Instance` must call `_ownXY()` first.
- To read member 2D on a frame you do not hydrate, bracket the read with `hydrateFrameMembers2d`/`releaseFrameMembers2d` (as `frameCropGeometry` in `ui/image-embedder.js` does).

Tests: `test-lazy-residency.mjs`, e2e `lazy-playback-eviction`, `sequence-lazy-workflow`.

## `.slp` files newer than LUC3D can read

Python sleap-io 0.10 writes SLP format 3.0 by default (every table a group of columns). Both readers know 2.x only, and without a check read such a file as 0 frames with no error, under a success message. Rule and wording: `loading/slp-format-check.js`; the vendored copy is the `(luc3d newer-format)` patch.
- **Both readers check.** A fallback must rethrow a refusal, not retry it on the other reader (`isNewerSlpFormatError`).
- **A refused file changes nothing.** A loader that clears the open project before its read runs `checkSlpFileFormat` first, as `handleLoadProjectSlpLazy` does. Others read first and are fine.
- **A file that did not open must not become the save target.** Loaders resolve `false` for it; the menu records `state.slpFileHandle` only after that. Quick save writes straight to the handle.
- Folders: a refused camera is a view and no data, on both routes, and the status gives the reason (`slpFailuresStatus`).
- When the readers learn 3.0: raise the limit in both places and honor the `/metadata` `required_features` attribute, as Python does.
Tests: `tests/test-slp-format-check.mjs`, e2e `slp-newer-format`.

## Per-node prediction scores

`ui/node-tooltip.js` shows a node's name and, on a prediction, that node's score, with a stripe in its skeleton colour. Take the colour from `getGroupColor`/`unlinkedBaseColor` (`ui/overlays.js`); never compute it separately.
- **Eager instances:** scores go into shared `Float32Array` pool chunks (`Instance.setPointScores`), and the instance keeps only the integer `_ps`. Never use one array per instance.
- **Lazy instances:** the store is the only copy. `SioLazyLoader.predictedPointScore` looks the score up by `_rawInstIndex` and node NAME (a skeleton edit re-indexes resident instances, not the store). Never copy scores onto hydrated instances.
- **Ask `Session.predictedNodeScore` for a score.** It returns NaN unless the instance type is `'predicted'`.
- **Every writer writes each node's own score** (`_buildSioPoints`), and uses the instance score only for a node with none.
- SLEAP analysis `.h5` `point_scores` are dropped on purpose (see the NOTE in `loading/slp-import-worker.js` `parseAnalysisH5`).

Tests: `test-node-scores.mjs`, `test-node-score-export.js`, e2e `node-score-tooltip`.

## Frames on a video no session names

A `.slp` can hold frames on a video no session's camera map lists, and a loader that opens only a file's first session sees the other sessions' frames the same way. They are not shown. A save keeps them, each on its own video: never dropped, never moved onto a camera's video, never left pointing at a video the file does not list (Python raises IndexError). `import-export/outside-frames.js`.
- **Lazy:** the store already holds them. `buildSessionRefGraph` gives each such video a header place after the session's cameras (`outsideVideos`) and `streamSessionIntoWriter` maps every id the store uses (`_headerMapFor` refuses an id pass 1 did not see, never passes one through raw). The file-wide refs count every store row (`sharedOut`), a camera's or not. Code that rewrites `store.tracks` must keep these rows' track names (`_keepOutsideRowTracks`); row-map walks do not visit them.
- **Eager:** `session.outsideFrames` (`keepOutsideFrames`: ordinary `Instance`s, the file's track names), written by `buildSlpBytes` (`outsideFramesForSave`). Skeleton node edits re-shape them and `countSkeletonEditImpact` counts them; track edits do not touch them. A loader that skips frames must keep them: `videosClaimedBy` says which videos it read. Never wrap a video index round the cameras.
- **A camera is one video:** the first its file lists with frames (`SioLazyLoader.open`, `ownVideoOfColumnar`); that file's other videos are outside.
- The vendored writers file a frame whose video is not in the header under video 0 (`Math.max(0, indexOf)`), so its video must be in the header. `appendStore` writes each row's skeleton id too, so `buildSessionRefGraph` also lists every skeleton the rows use (`skeletonPairsFor`).
- **One video, one entry.** A behavior event kept on such a video (`sessionEventScope`) goes on the entry its poses have: both writers hand `buildBehaviorEvents` `listedVideos` (the file's video record, as `fileEventsOf` made it, -> the header's `SIO.Video`), so the file's `frames` and `events` name the same place. Each session's part of the header lists its cameras, then the videos with poses, then those with only events. Go by that record (eager: `keepOutsideFrames`' `eventRecord`; lazy: the video's place in the project file, `keptVideoRecords`), never by filename: campy names every video `0.mp4`, and a video File ▸ Add SLP brought is a different file's. When the single-`.slp` folder loader's filename fallback makes a camera of such a video, its kept events go on that camera (`eventScope.cameras`).
- A file with none saves byte-identically (`save-golden-digest.mjs` must not move). The load's final status says what is not shown (`outsideFramesNote`) and, for a loader that opens only the first session, that a save keeps the others' poses but not their cameras or 3D.

Test: `tests/e2e/poses-outside-session.mjs`.

## Triangulation must not depend on where the origin is

`triangulatePointDLT` (`pose/triangulation-core.js`) solves in a frame built from the cameras only (`dltNormalizingFrame`). Without it, the answer would depend on where the world origin is, and Track All's association (`pose/cross-view-tracker.js` `_retriangulate`) scores against these points.
- Build that frame from the cameras ONLY, never from observations or a first-pass answer.
- The tracker always uses DLT, never BA. Fix the estimator, not the setting.
- Test with noisy fixtures: noiseless rays pass on a broken build.
- New tracker cost terms must not read absolute world coordinates.
- `dltHomogeneousFlat`'s frame arguments are optional; leaving them out must be bit-identical to the raw rows (`tests/test-triangulation-kernels.mjs`).
- `smallestEigvec4Inverse` must iterate until it converges, with the Jacobi kernel as fallback. Never loosen `INVERSE_TOL` (1e-15); invariance tests fail. Before changing the solve, A/B Track All identities with `tests/e2e/_diag-trackall-ab.mjs`.
Test: `tests/e2e/triangulation-frame-invariant.mjs`.

## The defined origin moves the CALIBRATION, never the points

Set Origin (`ui/origin-definition.js`, `pose/origin-frame.js`) changes only what is drawn. Never re-bake points, cameras or plane nodes. Any node of a fitted plane can be the origin; UI text says "node", not "corner".
- **Export New Calibration** writes `REBASED_CALIBRATION_NAME` via `rebaseExtrinsics`. **`t_new` comes from `frame.origin`, NOT `frame.translation`**; using translation gives an almost-correct calibration. Set as New Calibration writes the same name with byte-identical TOML (same writer). Intrinsics and camera order stay unchanged.
- **`pickCalibrationFile`** (`loading/calibration-pick.js`; keep it import-free so Node tests can load it) prefers the rebased name, otherwise the lexicographically first. It returns the ignored names so the loader can warn. Never rely on scan order.

**Set as New Calibration** (`pose/origin-rebase.js`, `ui/origin-rebase.js`):
- 3D points, plane normals (rotation only), plane nodes (locked ones too) and cameras all move together (`p' = R·p + t`; cameras `R_cam·Rᵀ`, `R_cam·origin + t_cam`), so no pixel moves.
- Plan, then apply: `planOriginRebase` writes nothing; `applyOriginRebase` swaps in its result. A cancel or failed write leaves the project byte-identical. Never transform in place and invert.
- Call `showSaveFilePicker` synchronously inside the Continue click; user activation expires.
- Write the calibration to a NEW file, never over `calibration.toml`.
- Count every 2D instance in the project, not just `frameGroups`: `forEachInstanceRow` when lazy, members plus `unlinkedInstances` when eager.
- Multi-session: re-base a filtered sessions array, with no mode flag. The active session (`sessions.indexOf(state.session)`) cannot be deselected. Recount with `subsetRebaseTally`, never by re-walking the project.
- `autoSaveAfterRebase` calls `quickSave` only when `state.slpFileHandle` exists. Otherwise set a status asking for Save As (the picker would throw `SecurityError`). Success means `!state.isDirty`, not "no exception". Save before writing the summary status; `quickSave` overwrites it.
- The dialog's folds are never persisted (no `persistSectionState`).
- Afterwards the origin and `triangulationResults` are cleared. A saved file changes only `points_3d` and `sessions_json` (`tests/e2e/origin-rebase-slp-diff.mjs`).

The Danger Zone actions show only while `originState.frame` is set. Reset to Calibration Origin goes through `confirmClearOrigin`. `clearOrigin` itself stays unguarded.
Tests: `tests/test-origin-frame.mjs`, `tests/test-origin-rebase.mjs`, `tests/test-calibration-file-pick.mjs`, `tests/e2e/set-new-calibration.mjs`, `tests/e2e/origin-picks-any-node.mjs`.

## Multi-session calibration divergence

Per-session calibrations can silently disagree. `compareSessionCalibrations` (`pose/calibration-compare.js`, DOM-free) + `noteSessionCalibrationDivergence` (`ui/calibration-notice.js`):
- **Runs LAST**, as `promptImportSkeletonForAllSessions`'s `onDone` — never mid-load or stacked on another modal.
- **Silent unless `ok === false`.** A session with no calibration is "not compared", not a third calibration.
- **The modal lists only WHO differs**: session folder names per calibration. No prose/offset/remedies (e2e asserts absent); the measured kind/offset go to the console.
- A note: one dismiss, Esc closes, writes nothing.
Tests: `tests/test-calibration-compare.mjs`, `tests/e2e/multi-session-calibration-notice.mjs`.

## Plane-node pin states; Set Angle Between Two Planes

`PlaneNode.pin`: `'none' | 'plane-locked' | 'locked'` (`pose/plane-nodes.js`). `immutable` is an accessor for `pin === 'locked'`; keep it.
- **`plane-locked` is NOT an anchor** — never pass it to `fitPlaneConstrained`. `planeImmutableMask` (`ui/plane-definition.js`) and `planeNodeImmutability` (`pose/plane-data.js`) report `locked` only.
- `locked` is enforced by `PlaneNode.setPoint3d` refusing; `plane-locked` by `PlaneModel.constrainPoint3dForNode`. **Every 3D write of a node goes through it** (`writePoints3dForPlane`'s `constrain` hook) — incl. 3D drags and `applyPlaneFit`, which must store `plane.planeFit` BEFORE the constrained write.
- **The constraint is the intersection of ALL planes the node is in** (`planeLockForNode` rank 0–3; `planeIntersectionBasis` / `projectOntoPlaneIntersection` in `pose/plane-fit.js`; near-parallel planes dropped by `PLANE_INDEPENDENCE_TOL`). **No `pinPlaneId`** — a saved one is read and discarded.
- Project onto a USABLE fit: `usablePlaneFit` (stored fit, else derived from the plane's other solved corners, **excluding the node being constrained**; <3 other corners → inert). Setting a pin clears stored fits, so never require one. Angle dialog and 3D drag share it. Set Origin needs a STORED fit.
- Typed node `x/y/z` (`ui/plane-nodes-panel.js`) takes the 3D-drag path: **2D follows 3D** (every placed view rewritten to the reprojection); `locked` → inputs `disabled`; an untouched field keeps its full stored double, not the 4-dp display.
- Save: defaults omitted, ride `planeNodes` (`save-golden-digest.mjs` must not move); `locked` also writes `immutable: true`, `plane-locked` does not.

Set Angle (`ui/plane-angle.js`, math `pose/plane-angle.js`): angle is unsigned `acos(|n·n|)` (normal sign is arbitrary); smallest rotation about the shared edge; ROTATE the stored fit, never re-fit. Apply re-plans before committing, then sets moved nodes `plane-locked` (not downgrading `locked`); a moved node also in another plane is pulled into it too → `plane_locked_elsewhere` warning (may miss the target). Dialog is non-modal (no scrim) but freezes plane data while open via `isAngleModalOpen()` in all edit paths (`applyAngleModalLock`, 3D `editable:false`, `isPlaneDataLocked` + `handlePlaneDrop`). Offer usable planes, not just fitted.
Tests: `test-plane-constrained-fit`, `test-plane-nodes` §20b/§20c, `e2e/plane-lock-holds`, `test-plane-angle`, `e2e/plane-angle`.

## 3D Mesh Objects

Object = group of planes (`pose/mesh-object-3d.js`); shape derived, never stored (`pose/mesh-object-geometry.js`); table `ui/mesh-objects.js`. Shared nodes weld vertices.
- **Strictly additive.** `PlaneModel.deletePlane` must NOT cascade into objects; membership resolves lazily (`resolvePlaneIds`). Use `planeCount(model)`, not `planeIds.length`.
- Winding derived: `orientFacesCoherently`, then closed → `signedVolume`, open → +Z up (`areaVectorZ`); an all-vertical surface (`areaVectorZ` ≈ 0) keeps the coherent pass's result. **No flip-normals control**; old `flipNormals` read and discarded.
- Concave faces: ear-clip, non-strict point-in-triangle.
- Selection: `syncMeshObject3D` → `viewport3d.setMeshMembership` gets plane IDS only, never geometry (derived geometry is in the origin frame, the viewport in calibration world). Highlight is one fixed `MESH_MEMBER_COLOR`; payload has no colour field. Face fill depth-tested, outline/corners `depthTest:false`.
- World is Z-up. Export (`import-export/mesh-export.js`): STL verbatim; GLB converts `(x,y,z)→(x,z,-y)`, a rotation — **no winding flip**, welded, no normals. Both float32, scale 1 (calibration mm); the project file stays source of truth.
Tests: `test-mesh-object-3d`, `test-mesh-object-geometry`, `test-mesh-export`, `e2e/mesh-object-roundtrip`, `e2e/mesh-object-export`, `e2e/mesh-membership-highlight`.

## Skeleton edits re-shape every annotation

One skeleton object is shared by all sessions, and `Instance` stores 2D by node INDEX, so a node edit re-shapes the whole project. All five edits (add/remove/rename node, add/remove edge) go through `confirmSkeletonEdit` (`ui/skeleton-edit-warning.js`), counted by `pose/skeleton-edit-impact.js`.
- Count the WHOLE project: lazy → `lazyLoader.forEachInstanceRow`; eager → frame-group instances + `unlinkedInstances`. Never `frameGroups` alone.
- Settle the sessions BEFORE mutating (handed to `onConfirm`) and apply only to them (`adoptEditedSkeleton`, `ui/info-panel.js`), never `setProjectSkeleton`; sessions on a different-shaped skeleton are named, not counted or edited.
- Node edits clear `state.triangulationResults` and each session's copy (stale snapshots duplicate info-panel tables).
- 2D, `nulledNodes` and `InstanceGroup.points3d` move together (removal splices the 3D); drop cached reprojections; keep 3D pooled (`pooledPoints3d`, `NO_REPROJECTED_INSTANCES`). A new node on a USER instance is placed and nulled (so it stays clickable); predicted instances get an empty slot.
- Lazy projects with non-resident frames get a red warning: edits are resident-only and the store cannot express them.
- No annotations → no dialog (`skeletonEditNeedsConfirmation`). "Do not show again" is `localStorage.skeletonEditWarningOff`: checked before the tally, recorded only when the edit is applied, never in the `.slp`; a storage failure means warn, never skip; `#skeletonWarnOffNote` turns it back on.
- Every edit calls `markDirty()`. Keep the dialog to title + one sentence + counts.
Tests: `tests/test-skeleton-edit-impact.mjs`, `tests/test-pose-data.js`, `tests/e2e/skeleton-edit-warning.mjs`.

## campy recording folders (`session.json`)

campy writes `<recording>/session.json` (`campy-session/1`); campy-live writes `campy-live-session*.json` beside it. `loading/campy-session.js` (import-free) reads them; `adoptCampyRecording` (`loading/session-loader.js`) puts the record on `session.campyRecording`; the Session tab's Recording section is `ui/recording-info.js`.
- **A folder without session.json loads exactly as before.** Every campy branch is gated on the record.
- **Animal IDs are display only.** Never name identities or tracks with them: campy-live numbers animals as it first matches them.
- **Read, never written:** not in the `.slp` (`save-golden-digest.mjs` must not move); picking the folder again re-reads it.
- The frame rate is `config.effective.frameRate.value` over the video estimate (`updateTotalFrames`). Views follow campy's camera order, but the calibration's camera list keeps its order. Each camera loads the file at campy's `video` path (campy names every video `0.mp4`, so never name a view by video stem when campy names the camera).
- A calibration FILE is hashed (`checkCampyCalibration`, through `readBlobArrayBuffer`) and compared with campy-live's `calibration.sha256`. A status other than `complete`, a mismatch, or cameras in only one of the two files go on the load's final status (`withCampyWarnings`) and in the section. Never a modal.
- A folder prompt that keeps only videos must also pass the root-level manifests through (`campyManifestsAmong`).
Tests: `tests/test-campy-session.mjs`, `tests/e2e/campy-session-folder.mjs`.

## UI conventions

- **One scroller per panel/modal.** No inner `max-height` + `overflow-y: auto`; make a section collapsible (`<details>`) instead.
- **Modals close on `Esc`** (remove the listener on close); mid-operation, `Esc` cancels the operation. Example: `showExport3DVideoModal` in `ui/export-modals.js`.
- **Display taste is browser-local** (`localStorage`), never the `.slp`: section open/closed (`persistSectionState`, `ui/section-state.js`, restored once at setup, not on repaint; best-effort storage; no stored value keeps the markup default, e.g. Danger Zone collapsed), slider sizes, plane visibility toggles, warning opt-outs. `save-golden-digest.mjs` must not move.
- **Plane-mode toolbar lock:** `applyPlaneModeToolbarLock` in `ui/plane-toolbar-lock.js`. Add a pose-annotation button to `PLANE_LOCKED_TOOLBAR_IDS`; a hover-menu wrapper of actions only also to `PLANE_LOCKED_DROPDOWN_IDS`. Visibility controls and the Animals field stay live, so the Track / Triangulate / Smooth / Behavior menus (which hold settings) are never wrapper-locked: each run item in them is a `<button>` in the id list.
- **Toolbar layout:** Sessions, then manual labelling (Instance ▾, Group ▾), then proofreading in run order (Track ▾, Triangulate ▾, Smooth ▾ = Temporal Smoothing), then Behavior ▾ (detection; Label Behaviors mode in its menu), each kind in its own `.toolbar-group`. Keep labels short (the method and full names go in tooltips); the toolbar must fit a 1440px window (`toolbar-hover-menus.mjs` §4; it fits down to ~1,120 px).
- **Toolbar action menus:** Track, Triangulate, Smooth and Behavior share one shape (`wireActionMenu`, `ui/ui-wiring.js`): a click on the button runs all frames; the hover menu has This frame / All frames, the action's quick settings, then Settings…. Smooth also has a Remove smoothing / Restore smoothing toggle after All frames (`#tbSmoothToggle`, in `PLANE_LOCKED_TOOLBAR_IDS`): Remove keeps what it takes off; Restore puts it back with no recompute unless the settings or any 3D changed since (`removeSmoothing` / `restoreSmoothing`, `pose/temporal-smoothing.js`), and then smooths again. Each button is blue while its step's result stands: Smooth while anything is smoothed (`ui/smoothing-button.js`); Track and Triangulate once run in this session, until re-tracking or a re-solve replaces the result (`pose/run-state.js`, `ui/run-buttons.js`; a new Triangulate path must call `noteTriangulateRun`, a new Track path `noteTrackRun`); Behavior while its predictions stand (saved, so also after a reopen). A new action with a this-frame / all-frames choice follows it. Keyboard shortcuts and menu items for these actions are still NOT gated.
- **Enter/leave Define Planes only via `togglePlaneMode`** (View menu and `definePlanes`, `Mod+Shift+P`), which does the unwinding.
- **`showHotkeysHelp` stays at module scope** in `ui/ui-wiring.js` (called from both `setupMenus` and `setupUI`).
- **Info tips:** declare with `setInfoTip(el, text)` (`ui/info-tip.js`, delegated, removes `title`); never `setStatus`. One or two sentences. Keep every ⓘ a `<button>`; its capture-phase click `stopPropagation`s so it neither folds a `<summary>` nor clicks its row. Explain a section in its ⓘ (e.g. `PLANE_SECTION_INFO`), not a paragraph. Test: `tests/e2e/info-tip.mjs`.
- **Text is selectable/copyable.** `shouldIgnoreShortcut` (`ui/keyboard-target.js`) yields copy/cut (not paste) while a selection exists. Use `user-select: none` only on drag/click surfaces (toolbars, menu bar, tab bar, modal drag handles). Test: `tests/e2e/copy-panel-text.mjs`.
- **Plane visibility** (`ui/plane-visibility.js`): four toggles, default ON; `planeVisibility(modeActive)` overrides them in Defining Plane Mode. They decide whether a plane is drawn, never how (no fill override). In 3D they hide meshes, never filter `_planes`.
- **Plane Views column** (`annotatedViewStats`, `ui/plane-list-panel.js`) counts views with at least one hand-placed (not reprojected) corner, using the same test as `triangulatePlane`'s `usableViews`.
- **Plane node selection** goes by node ID, never position (`setSelectedPlaneNode`); one capture-phase document listener (`onDocumentClickForNodeSelection`) selects and clears; don't rebuild the Nodes table on select.
- **Define Planes panel** is Nodes then Planes; the roster row is the only plane selector. `tests/e2e/define-plane-mode.mjs` §2 asserts the section's child list.
- **Progress popups** use one card style (`ui/loading-overlay.js`); pinned by `tests/e2e/fixtures/progress-card.js`. A long main-thread step must PAINT its card before it holds the thread (await `yieldToPaint`), as File ▸ Save does for each of its steps (`createSaveProgress`, `import-export/save-load.js`).
- **Every progress bar lights the stretch it just gained**: call `noteProgress(fill, frac)` (`ui/progress-delta.js`) after each update of a new bar. Not for position bars (seekbar, playheads, range sliders). Test: `tests/e2e/progress-delta.mjs`.
- **Counts and frame numbers** shown to users use `n.toLocaleString('en-US')` or `formatFrameNumber` (`ui/frame-readout.js`). Not in attributes, keys or logs. Join frame ranges with `'; '`. Tests reading counts accept `[\d,]+`.
- **Track vs identity colours:** paint track colours only through `getTrackColor(trackIdx, colorByIdentity)` (`ui/overlays.js`): vivid `TRACK_COLORS` in Track mode, muted `UNIDENTIFIED_TRACK_COLORS` in Identity mode. Every track-colour painter (labels, timeline rows, Visibility swatches) passes the mode; the `onColorByChange` handler in `ui/ui-wiring.js` recolors them, so change `state.colorByIdentity` only via `setColorByIdentity` (`ui/color-by.js`). Don't change `IDENTITY_COLORS` 0-6. After adding any identity or fixed canvas colour, run `tests/test-track-identity-palette.mjs`; if it fails, change the palette, never the thresholds.

## Cloud-synced files: read a user's File through `loading/cold-read-guard.js`

In Google Drive for desktop (macOS File Provider), a Blob read of a not-yet-downloaded range can throw or return runs of zeros with no error. Never call `file.arrayBuffer()` / `slice().arrayBuffer()` on a user's file directly; use `readBlobRange` / `readBlobArrayBuffer` (retries thrown reads; re-reads all-zero 4 KiB blocks **plus the block on each side**, filling in only zero bytes, since a zero run can start or end part-way into a block — the same rule as h5wasm's `lucidColdReadGuard`).
- The frame-accurate decoder must use `MediaBunnyVideoBackend.fromRangeSource(guardedRangeSource(file))` (`_initMediabunny`, `loading/video.js`), not `fromBlob`.
- `guardedRangeSource` reads whole 1 MiB windows.
- mp4box's index scan reads through it too (`readChunk`).
- Not covered: `<video>` blob-URL playback and small JSON/TOML `.text()` reads. `.slp` reads go through h5wasm's own `cold-read` patch.
- Tests: `tests/test-cold-read-guard.mjs`, `tests/e2e/video-cold-read.mjs`.

## Native resolution (1:1)

View ▸ Native Resolution / `Shift+0` (`setNativeResolution`, `loading/video.js`) is a zoom scale (1 video px per device px). Wheel/`+`/`-` steps snap to it (`snapZoomToNative`). Two rules keep it exact:
- **Per-axis scale** (`zoom.scale` / `zoom.scaleY`, read via `_scaleY`): anything mapping screen ↔ video must use both.
- **Pan on the device-pixel grid** (`_alignNativeOffsets` in `applyZoom`).
Enter 1:1 only via `_enterNative` (incl. the 100% preset). Anything changing the canvas box or devicePixelRatio must end in `reapplyZoom`/`reapplyAllZoom`. `state.nativeResolution` is derived (every shown view at 1:1); while on, a view coming on screen joins at 1:1 (`refreshPaneInteractions`). Exact only at integer devicePixelRatio; don't promise exactness otherwise. Only a view held at 1:1 may read "100%". Tests: `tests/e2e/native-resolution-zoom.mjs` (+`HEADED=1`), `tests/test-video-controller.js`.

## Tests

Three populations, disjoint coverage; run all three:
```bash
node tests/e2e/run-unit-tests.mjs   # tests/*.js classic scripts, headless browser
node tests/run-mjs-tests.mjs        # tests/test-*.mjs, Node (optional substring filter)
node tests/e2e/<name>.mjs           # Playwright, one file per behavior
```
- `tests/*.js` run only if `tests/test-runner.html` has a `<script>` tag for them; `tests/test-classic-tests-registered.mjs` fails on an untagged one. No Node runner for these; don't add one.
- `tests/test-*.mjs` are auto-discovered. Put ESM tests nowhere else.
- `tests/e2e/_diag-*`, `_bench-*`, `_real-*` are investigation tools, not assertions.
- Notable e2e: `sequence-lazy-workflow.mjs` (lazy-project regression harness; catches resident-only bugs, incl. edits: interactive deletes must use `deleteTargetsFromStore`, cycle 5c); `video-encode-streaming.mjs` (only coverage of streaming video export; run it if you touch `ui/video-encode.js`); `overlay-export-modal.mjs` / `export-3d-video.mjs` (the video modals); `_real-roundtrip.mjs` (real-data acceptance, needs a large `.slp`).

**One visible-browser run at a time, machine-wide** (headed = `open -a`, Playwright `headless: false`, `HEADED=1`): parallel sessions steal focus and throttle each other. Wrap any headed run not already taking the lock: `node scripts/browser-lock.mjs --label="what" -- <command>` (lock: `/tmp/luc3d-browser.lock`). Headless runs need no lock. Release re-activates the Claude app; don't add your own `osascript … activate`.

## Python scripts

`scripts/json_to_slp.py`, `scripts/json_to_h5.py` (JSON export → `.slp`/HDF5; need h5py, numpy). `scripts/validate_slp_sleap_compat.py` asserts LUCID `.slp` files load in SLEAP (`--compare`, `--metadata-roundtrip`); run with `uv run --with sleap-io --with numpy --with h5py python …`.

## Maintenance

- **When modifying any module, update its `MODULES.md` entry** (purpose, exports, dependencies).
- **Every keyboard shortcut needs an `ACTION_CATALOG` entry in `ui/settings.js`** (`{ id, label, category, binding, editable, dispatched }`). `dispatched: true` = matched by `dispatchEvent()`, needs `setHandler(id, fn)` (wired in `ui/ui-wiring.js`), rebindable if `editable`. `dispatched: false` = own handler; keep `binding` in sync with it.

