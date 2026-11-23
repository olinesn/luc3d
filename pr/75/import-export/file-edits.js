// import-export/file-edits.js — seeing what another tool changed in a project file.
//
// Tracks and identities are per session in LUC3D and per file everywhere else, so a
// project file holds both forms: the standard slots every tool reads (`/tracks_json`,
// each detection's track, the `/identity` catalog, each group's `identity_idx`) and
// LUC3D's own copies in `metadata.lucid` (each session's `tracks` and `identities`, each
// group's `identityId`). Another tool (sleap-app, Python) edits the standard slots and
// carries the copies along unread. So the copies hold only while they agree with the
// standard slots. Trusting them after an edit shows the old names and the old animals
// and, on the next save, writes them over the other tool's work.
//
// Both functions leave a file nobody else edited exactly as it reads today, positions
// included: a session's list keeps the slots `frameIdentityMap` keys and group
// `identityId`s were written against.
//
// No imports, so Node tests can load it.

/**
 * A session's track list, given what the file now says.
 *
 * A name the file no longer has was renamed or deleted; a name this session's detections
 * use that the list lacks was renamed or added. The first kind take the second kind's
 * names in order (a rename keeps its slot, so keys written against it stay true), any
 * name left over is added at the end, and a slot with no new name is dropped.
 *
 * @param {string[]} privateNames - `metadata.lucid.tracks`
 * @param {string[]} fileNames - the file's track list
 * @param {?Set<number>} usedFileIdx - positions in `fileNames` this session's detections
 *   use; null for every name in the file
 * @returns {{names: string[], slotMap: ?number[], edited: boolean}} the list to use and,
 *   when a slot moved, each old slot's new position (-1 if dropped)
 */
export function reconcileSessionTracks(privateNames, fileNames, usedFileIdx) {
    var inFile = new Set(fileNames);
    var known = new Set(privateNames);
    var incoming = [];
    var seen = new Set();
    for (var i = 0; i < fileNames.length; i++) {
        var name = fileNames[i];
        if (known.has(name) || seen.has(name)) continue;
        if (usedFileIdx && !usedFileIdx.has(i)) continue;
        seen.add(name);
        incoming.push(name);
    }
    var vacated = 0;
    for (var p = 0; p < privateNames.length; p++) {
        if (!inFile.has(privateNames[p])) vacated++;
    }
    if (vacated === 0 && incoming.length === 0) {
        return { names: privateNames.slice(), slotMap: null, edited: false };
    }

    var names = [];
    var slotMap = new Array(privateNames.length);
    var moved = false;
    for (var s = 0; s < privateNames.length; s++) {
        var kept = inFile.has(privateNames[s])
            ? privateNames[s]
            : (incoming.length ? incoming.shift() : null);
        if (kept === null) {
            slotMap[s] = -1;
            moved = true;
            continue;
        }
        slotMap[s] = names.length;
        if (names.length !== s) moved = true;
        names.push(kept);
    }
    for (var a = 0; a < incoming.length; a++) names.push(incoming[a]);
    return { names: names, slotMap: moved ? slotMap : null, edited: true };
}

/**
 * Where each slot of a session's track list went in a list that is the file's own, as a
 * lazy project and the session-folder loader use. Null when every slot is still at its
 * own position.
 *
 * @param {string[]} privateNames - `metadata.lucid.tracks`
 * @param {string[]} fileNames - the file's track list
 * @returns {?number[]}
 */
export function lazyTrackSlotMap(privateNames, fileNames) {
    var r = reconcileSessionTracks(privateNames, fileNames, null);
    var slotMap = new Array(privateNames.length);
    var moved = false;
    for (var s = 0; s < privateNames.length; s++) {
        var at = r.slotMap ? r.slotMap[s] : s;
        var name = at < 0 ? null : r.names[at];
        // A name can repeat in a list: a slot that still holds it stays put.
        var pos = name === null ? -1 : (fileNames[s] === name ? s : fileNames.indexOf(name));
        slotMap[s] = pos;
        if (pos !== s) moved = true;
    }
    return moved ? slotMap : null;
}

/**
 * Move `frameIdentityMap`'s serialized keys (`"frameIdx:camName:trackIdx"`) to a track
 * list's new positions, dropping entries of a dropped slot. In place, on the session's
 * `metadata.lucid` dict, before it is read.
 *
 * @param {Object} lucid
 * @param {number[]} slotMap - from {@link reconcileSessionTracks}
 */
export function remapFrameIdentityTrackSlots(lucid, slotMap) {
    var remap = function (key) {
        if (typeof key !== 'string') return key;
        var cut = key.lastIndexOf(':');
        if (cut < 0) return key;
        var slot = parseInt(key.substring(cut + 1), 10);
        if (!(slot >= 0 && slot < slotMap.length)) return key;
        return slotMap[slot] < 0 ? null : key.substring(0, cut + 1) + slotMap[slot];
    };
    if (Array.isArray(lucid.frameIdentityMap)) {
        lucid.frameIdentityMap = lucid.frameIdentityMap
            .map(function (entry) { return [remap(entry[0]), entry[1]]; })
            .filter(function (entry) { return entry[0] !== null; });
    }
    if (Array.isArray(lucid.frameIdentityMapDropped)) {
        lucid.frameIdentityMapDropped = lucid.frameIdentityMapDropped
            .map(remap)
            .filter(function (key) { return key !== null; });
    }
}

/**
 * Every typed InstanceGroup of a typed RecordingSession, as something that can be walked
 * more than once.
 * @param {Object} typedSession
 * @returns {Iterable<Object>}
 */
export function typedSessionGroups(typedSession) {
    return {
        [Symbol.iterator]: function* () {
            for (var frameGroup of typedSession.frameGroups.values()) {
                yield* frameGroup.instanceGroups || [];
            }
        },
    };
}

/**
 * Point the per-detection identity map at a group's new identity, for the groups another
 * tool moved to a different animal. Call after the map is read: its entries still say
 * what LUC3D last wrote. Only those groups: an entry of any other group may be a
 * single-view fix the standard slots have no room for.
 *
 * @param {Object} session
 * @param {Array<[number, Object]>} movedGroups - `[frameIdx, group]` pairs
 */
export function applyMovedGroupIdentities(session, movedGroups) {
    for (var i = 0; i < movedGroups.length; i++) {
        var frameIdx = movedGroups[i][0];
        var group = movedGroups[i][1];
        for (var [camName, inst] of group.instances) {
            if (!inst || inst.trackIdx == null || inst.trackIdx < 0) continue;
            if (group.identityId >= 0) {
                session.setFrameIdentity(frameIdx, camName, inst.trackIdx, group.identityId);
            } else {
                session.deleteFrameIdentity(frameIdx, camName, inst.trackIdx);
            }
        }
    }
}

/**
 * A session's identities, given what its groups' standard slot now says. Null when every
 * group agrees with LUC3D's copy, which is the case for a file nobody else edited.
 *
 * After an edit the standard slot wins. A name the list lacks takes over the slot whose
 * groups moved to it most, if nothing uses that slot's own name any more (a rename);
 * otherwise it is added at the end.
 *
 * @param {Array<{name:string, color:?string}>} privateList - `metadata.lucid.identities`
 * @param {Iterable<Object>} groups - typed InstanceGroups, read for
 *   `metadata.lucid.identityId` and `identity`
 * @returns {?{identities: Array<{name:string, color:?string}>,
 *   identityOf: function(Object): number}}
 *   `identityOf(group)` is a group's id in `identities`, -1 for none
 */
export function reconcileSessionIdentities(privateList, groups) {
    var privateIdOf = function (group) {
        var lucid = group.metadata && group.metadata.lucid;
        return lucid && lucid.identityId != null ? lucid.identityId : -1;
    };
    var edited = false;
    var usedNames = new Set();
    var moves = new Map(); // file name -> Map(private id -> groups)
    for (var group of groups) {
        var id = privateIdOf(group);
        var fileName = group.identity ? group.identity.name : null;
        var privateName = id >= 0 && id < privateList.length ? privateList[id].name : null;
        if (fileName !== privateName) edited = true;
        if (fileName === null) continue;
        usedNames.add(fileName);
        if (id < 0) continue;
        var counts = moves.get(fileName);
        if (!counts) moves.set(fileName, counts = new Map());
        counts.set(id, (counts.get(id) || 0) + 1);
    }
    if (!edited) return null;

    var identities = privateList.map(function (entry) {
        return { name: entry.name, color: entry.color || null };
    });
    var known = new Set(identities.map(function (entry) { return entry.name; }));
    var unknown = Array.from(usedNames).filter(function (name) { return !known.has(name); });
    // Largest move first, so a rename takes its slot before a smaller claim on it.
    var claims = [];
    for (var u = 0; u < unknown.length; u++) {
        var claimed = moves.get(unknown[u]);
        if (!claimed) continue;
        for (var [slot, n] of claimed) {
            if (slot < identities.length && !usedNames.has(identities[slot].name)) {
                claims.push({ name: unknown[u], slot: slot, n: n });
            }
        }
    }
    claims.sort(function (a, b) { return b.n - a.n; });
    var taken = new Set();
    var placed = new Set();
    for (var c = 0; c < claims.length; c++) {
        if (taken.has(claims[c].slot) || placed.has(claims[c].name)) continue;
        identities[claims[c].slot].name = claims[c].name;
        taken.add(claims[c].slot);
        placed.add(claims[c].name);
    }
    var colorByName = new Map();
    for (var g of groups) {
        if (g.identity && g.identity.color && !colorByName.has(g.identity.name)) {
            colorByName.set(g.identity.name, g.identity.color);
        }
    }
    for (var w = 0; w < unknown.length; w++) {
        if (placed.has(unknown[w])) continue;
        identities.push({ name: unknown[w], color: colorByName.get(unknown[w]) || null });
    }

    var byName = new Map();
    for (var b = identities.length - 1; b >= 0; b--) byName.set(identities[b].name, b);
    return {
        identities: identities,
        identityOf: function (typedGroup) {
            if (!typedGroup.identity) return -1;
            var name = typedGroup.identity.name;
            var own = privateIdOf(typedGroup);
            if (own >= 0 && own < identities.length && identities[own].name === name) return own;
            return byName.has(name) ? byName.get(name) : -1;
        },
    };
}
