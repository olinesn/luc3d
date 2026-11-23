// import-export/detection-labels.js — what a file says about a detection that LUC3D has
// no use for, and must not lose.
//
// A detection in a `.slp` can carry a tracker's score, and from a model an identity and a
// category with their scores and embeddings. LUC3D reads none of these: its animals are
// its own groups and tracks. But a save rebuilds every detection from LUC3D's model, so
// whatever the model does not hold is wiped from the file (100 of 100 identities and
// 1,662 of 1,662 tracker scores, on real sleap-nn output). So they ride along: read on the
// way in, untouched, and written back on the way out. Nothing in LUC3D acts on them.
//
// `Instance.trackingScore` and `Instance.fileLabels` default on the prototype, so a
// detection that carries none costs nothing. A lazy project's stay in its store, which
// its writer reads directly (`(detection labels)`, lib/sleap-io/chunk-X76PRJK6.js); a
// camera-frame rewritten by a user edit takes them from the store (`storeRowLabels`).
//
// No imports, so Node tests can load it.

/**
 * What a typed (sleap-io) detection carries besides its pose, track and scores.
 * @param {Object} typed - a typed Instance
 * @returns {?Object} null when it carries none
 */
export function typedFileLabels(typed) {
    if (!typed.identity && !typed.identityEmbedding && !typed.category && !typed.categoryEmbedding) return null;
    return {
        identity: typed.identity || null,
        identityScore: typed.identityScore != null ? typed.identityScore : null,
        identityEmbedding: typed.identityEmbedding || null,
        category: typed.category || null,
        categoryScore: typed.categoryScore != null ? typed.categoryScore : null,
        categoryEmbedding: typed.categoryEmbedding || null,
    };
}

/**
 * What a lazy store holds for one instance row, in the shape a LUC3D instance carries it.
 * A camera-frame LUC3D rewrites (a user edit) replaces the store's rows on save, so this
 * is how their detections keep theirs. The store is the only copy otherwise.
 * @param {Object} store - a LazyDataStore
 * @param {number} row - the instance's row in the store
 * @returns {{trackingScore: number, fileLabels: ?Object}}
 */
export function storeRowLabels(store, row) {
    var column = store.instancesData && store.instancesData.tracking_score;
    var link = store._instanceIdentityLinks && store._instanceIdentityLinks.get(row);
    var catLink = store._instanceCategoryLinks && store._instanceCategoryLinks.get(row);
    var identityEmbedding = store._instanceEmbeddings && store._instanceEmbeddings.get(row);
    var categoryEmbedding = store._instanceCategoryEmbeddings && store._instanceCategoryEmbeddings.get(row);
    var fileLabels = null;
    if (link || catLink || identityEmbedding || categoryEmbedding) {
        fileLabels = {
            identity: (link && store.identities && store.identities[link[0]]) || null,
            identityScore: link ? link[1] : null,
            identityEmbedding: identityEmbedding || null,
            category: (catLink && store.categoryCatalog && store.categoryCatalog[catLink[0]]) || null,
            categoryScore: catLink ? catLink[1] : null,
            categoryEmbedding: categoryEmbedding || null,
        };
    }
    return { trackingScore: column ? Number(column[row]) || 0 : 0, fileLabels: fileLabels };
}

/**
 * What a lazy store keeps outside its columns that is keyed by instance row: labels by
 * row, and annotations with a link to their instance. The vendored streaming reader fills
 * no annotations yet (olinesn/sleap-io.js#5 would), so those are only ready for it.
 */
export var STORE_ROW_KEYED_FIELDS = Object.freeze([
    '_instanceIdentityLinks', '_instanceEmbeddings', '_instanceCategoryLinks', '_instanceCategoryEmbeddings',
    '_roiByFrame', '_maskByFrame', '_bboxByFrame', '_centroidByFrame', '_labelImageByFrame',
    '_undistributedRois', '_undistributedMasks', '_undistributedBboxes', '_undistributedCentroids', '_undistributedLabelImages',
]);

/**
 * Move those after rows were deleted: the survivors follow the columns, and what belonged
 * to a deleted row goes with it (an annotation stays, unlinked).
 * @param {Object} store - a LazyDataStore, updated in place
 * @param {ArrayLike<number>} kill - per old row, nonzero if deleted
 * @param {ArrayLike<number>} survBefore - per old row, its new row
 */
export function rekeyStoreRows(store, kill, survBefore) {
    var newRow = function (row) {
        return row >= 0 && row < kill.length && !kill[row] ? survBefore[row] : null;
    };
    var relink = function (annotation) {
        if (annotation._instanceIdx != null && annotation._instanceIdx >= 0) {
            annotation._instanceIdx = newRow(annotation._instanceIdx);
        }
        if (annotation._objectInstanceIdxs) {
            for (var [labelId, row] of annotation._objectInstanceIdxs) {
                var moved = newRow(row);
                annotation._objectInstanceIdxs.set(labelId, moved === null ? -1 : moved);
            }
        }
    };
    for (var f = 0; f < STORE_ROW_KEYED_FIELDS.length; f++) {
        var field = STORE_ROW_KEYED_FIELDS[f];
        var held = store[field];
        if (!held) continue;
        if (Array.isArray(held)) {
            held.forEach(relink);
        } else if (field.indexOf('ByFrame') >= 0) {
            for (var list of held.values()) list.forEach(relink);
        } else if (held.size) {
            var rekeyed = new Map();
            for (var [oldRow, value] of held) {
                var now = newRow(oldRow);
                if (now !== null) rekeyed.set(now, value);
            }
            store[field] = rekeyed;
        }
    }
}

/**
 * Keep a typed detection's labels on the LUC3D instance built from it.
 * @param {Object} inst - a LUC3D Instance
 * @param {Object} typed - the typed Instance it was built from
 */
export function adoptTypedLabels(inst, typed) {
    if (typed.trackingScore) inst.trackingScore = typed.trackingScore;
    var labels = typedFileLabels(typed);
    if (labels) inst.fileLabels = labels;
}

/**
 * The file's identity metadata (a strain, say) for the identities of a session, by name.
 * LUC3D's own per-session list stores only names and colours.
 * @param {Array<{name:string}>} identities - LUC3D Identity objects, updated in place
 * @param {?Array<{name:string, metadata:?Object}>} catalog - the file's identities
 */
export function adoptIdentityMetadata(identities, catalog) {
    if (!identities || !catalog) return;
    var byName = new Map();
    for (var c = 0; c < catalog.length; c++) {
        var meta = catalog[c].metadata;
        if (meta && Object.keys(meta).length && !byName.has(catalog[c].name)) byName.set(catalog[c].name, meta);
    }
    for (var i = 0; i < identities.length; i++) {
        var found = byName.get(identities[i].name);
        if (found) identities[i].metadata = found;
    }
}

/**
 * Finds a detection's identity in a save's catalog, by name, adding it to the catalog if
 * the session's own list does not have it.
 * @param {Object} SIO - the sleap-io module
 * @param {Array<Object>} sioIdentities - the catalog being written, added to in place
 * @returns {function(Object): Object}
 */
export function identityCatalogFinder(SIO, sioIdentities) {
    var byName = new Map();
    for (var i = sioIdentities.length - 1; i >= 0; i--) byName.set(sioIdentities[i].name, sioIdentities[i]);
    return function (identity) {
        var found = byName.get(identity.name);
        if (!found) {
            found = new SIO.Identity({ name: identity.name, color: identity.color, metadata: identity.metadata });
            sioIdentities.push(found);
            byName.set(identity.name, found);
        }
        return found;
    };
}

/**
 * Write a LUC3D instance's labels onto the typed detection a save builds from it.
 * @param {Object} sioInst - the typed Instance being written
 * @param {Object} inst - the LUC3D Instance
 * @param {function(Object): Object} findIdentity - from {@link identityCatalogFinder}
 */
export function writeFileLabels(sioInst, inst, findIdentity) {
    if (inst.trackingScore) sioInst.trackingScore = inst.trackingScore;
    var labels = inst.fileLabels;
    if (!labels) return;
    if (labels.identity) {
        sioInst.identity = findIdentity(labels.identity);
        sioInst.identityScore = labels.identityScore;
    }
    if (labels.identityEmbedding) sioInst.identityEmbedding = labels.identityEmbedding;
    if (labels.category) {
        sioInst.category = labels.category;
        sioInst.categoryScore = labels.categoryScore;
    }
    if (labels.categoryEmbedding) sioInst.categoryEmbedding = labels.categoryEmbedding;
}
