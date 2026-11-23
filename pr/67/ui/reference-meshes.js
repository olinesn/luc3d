// ui/reference-meshes.js — STL/PLY files drawn in the 3D view as reference
// geometry (File ▸ Load Reference Mesh, Visibility ▸ Reference Meshes).
//
// The use it was built for: a 3D scan of the cage, aligned into the rig's
// calibrated frame by the cage-scan tools, shown against the planes and the
// triangulated points so the two can be compared by eye. Coordinates are taken
// as calibration world coordinates (the frame `points3d` and plane-node `xyz`
// live in) and units as the calibration's — nothing is transformed.
//
// DISPLAY ONLY: never written to the `.slp`, never marks the project dirty,
// gone on reload. The entries outlive a rebuilt 3D viewport (every project
// load rebuilds it), because `setup3DViewport` calls `syncReferenceMeshes3D`.
//
// Parsing is `import-export/mesh-import.js` (typed arrays only). A big file is
// parsed in chunks with the progress card up, yielding to paint between them.

import { viewport3d } from './app-state.js?v=20102c35280e';
import { pickFiles } from '../import-export/file-io.js?v=20102c35280e';
import { readBlobArrayBuffer } from '../loading/cold-read-guard.js?v=20102c35280e';
import { parseMeshAsync } from '../import-export/mesh-import.js?v=20102c35280e';
import {
    showLoading, showLoadingProgress, hideLoading, yieldToPaint,
} from './loading-overlay.js?v=20102c35280e';
import { setStatus } from '../import-export/save-load.js?v=20102c35280e';

/** Colours handed to meshes without vertex colours, in turn. */
export const REFERENCE_MESH_COLORS = ['#b0bec5', '#ce93d8', '#80cbc4', '#ffcc80', '#90caf9'];
/** Starting opacity: see-through, so the planes and points inside stay visible. */
export const DEFAULT_REFERENCE_OPACITY = 0.5;

/** @type {{entries: Array<Object>, nextId: number}} */
export const referenceMeshState = { entries: [], nextId: 1 };

function fmt(n) { return Number(n).toLocaleString('en-US'); }

/**
 * Add a parsed mesh (`parseMesh` output) to the list and draw it.
 * @returns {Object} the entry `{id, name, mesh, visible, color, opacity}`
 */
export function addReferenceMesh(name, mesh) {
    var id = referenceMeshState.nextId++;
    var entry = {
        id: id,
        name: name || ('mesh ' + id),
        mesh: mesh,
        visible: true,
        color: REFERENCE_MESH_COLORS[(id - 1) % REFERENCE_MESH_COLORS.length],
        opacity: DEFAULT_REFERENCE_OPACITY,
    };
    referenceMeshState.entries.push(entry);
    syncReferenceMeshes3D();
    renderReferenceMeshList();
    return entry;
}

export function getReferenceMesh(id) {
    return referenceMeshState.entries.find(function (e) { return e.id === id; }) || null;
}

export function removeReferenceMesh(id) {
    var i = referenceMeshState.entries.findIndex(function (e) { return e.id === id; });
    if (i < 0) return false;
    referenceMeshState.entries.splice(i, 1);
    syncReferenceMeshes3D();
    renderReferenceMeshList();
    return true;
}

export function setReferenceMeshVisible(id, on) {
    var e = getReferenceMesh(id);
    if (!e) return;
    e.visible = !!on;
    syncReferenceMeshes3D();
}

export function setReferenceMeshColor(id, color) {
    var e = getReferenceMesh(id);
    if (!e || !/^#[0-9a-f]{6}$/i.test(color)) return;
    e.color = color;
    syncReferenceMeshes3D();
}

export function setReferenceMeshOpacity(id, opacity) {
    var e = getReferenceMesh(id);
    if (!e || !Number.isFinite(opacity)) return;
    e.opacity = Math.max(0.05, Math.min(1, opacity));
    syncReferenceMeshes3D();
}

/** Push the list into the live 3D viewport (no-op while it does not exist). */
export function syncReferenceMeshes3D() {
    if (viewport3d && typeof viewport3d.setReferenceMeshes === 'function') {
        viewport3d.setReferenceMeshes(referenceMeshState.entries);
    }
}

/**
 * Read, parse and add one file, with the progress card up.
 * @param {File} file
 * @returns {Promise<Object|null>} the entry, or null when it could not be read
 */
export async function loadReferenceMeshFile(file) {
    var title = 'Load Reference Mesh';
    try {
        showLoading('Reading ' + file.name + '…', { title: title });
        await yieldToPaint();
        // Not file.arrayBuffer(): a cloud file still downloading reads back
        // with zeroed pages (loading/cold-read-guard.js).
        var buf = await readBlobArrayBuffer(file);
        var mesh = await parseMeshAsync(buf, { name: file.name },
            function (done, total, unit) {
                showLoadingProgress('Reading ' + file.name, done, total, { title: title, unit: unit });
            },
            yieldToPaint);
        showLoading('Drawing ' + file.name + '…', { title: title });
        await yieldToPaint();
        var entry = addReferenceMesh(file.name, mesh);
        if (viewport3d && typeof viewport3d.fitToScene === 'function') viewport3d.fitToScene();
        hideLoading();
        var what = mesh.kind === 'points'
            ? fmt(mesh.vertexCount) + ' points'
            : fmt(mesh.triangleCount) + ' triangles';
        var dropped = mesh.dropped.triangles + mesh.dropped.vertices;
        setStatus('Loaded reference mesh "' + file.name + '": ' + what +
            (dropped ? ' (' + fmt(dropped) + ' invalid skipped)' : '') +
            (viewport3d ? '' : '. Open the 3D view to see it'), 'success');
        return entry;
    } catch (err) {
        hideLoading();
        setStatus('Could not load "' + file.name + '": ' + (err && err.message ? err.message : err), 'error');
        return null;
    }
}

/** File ▸ Load Reference Mesh: pick one or more STL/PLY files. */
export async function loadReferenceMeshInteractive() {
    var files = await pickFiles({ accept: '.stl,.ply', multiple: true });
    if (!files.length) { setStatus('No file selected', 'warning'); return []; }
    var out = [];
    for (var i = 0; i < files.length; i++) {
        var e = await loadReferenceMeshFile(files[i]);
        if (e) out.push(e);
    }
    return out;
}

// ============================================
// Visibility ▸ Reference Meshes
// ============================================

/** Rebuild the list rows from `referenceMeshState`. */
export function renderReferenceMeshList() {
    var list = document.getElementById('referenceMeshList');
    if (!list) return;
    var empty = document.getElementById('referenceMeshEmpty');
    list.innerHTML = '';
    var entries = referenceMeshState.entries;
    if (empty) empty.style.display = entries.length ? 'none' : '';
    entries.forEach(function (e) {
        var row = document.createElement('div');
        row.className = 'ref-mesh-row';
        row.dataset.meshId = String(e.id);

        var show = document.createElement('input');
        show.type = 'checkbox';
        show.className = 'ref-mesh-show';
        show.checked = e.visible;
        show.title = 'Show in the 3D view';
        show.addEventListener('change', function () { setReferenceMeshVisible(e.id, show.checked); });

        var color = document.createElement('input');
        color.type = 'color';
        color.className = 'ref-mesh-color';
        color.value = e.color;
        color.disabled = !!e.mesh.colors;
        color.title = e.mesh.colors ? 'Drawn in the file\'s own vertex colours' : 'Colour in the 3D view';
        color.addEventListener('input', function () { setReferenceMeshColor(e.id, color.value); });

        var name = document.createElement('span');
        name.className = 'ref-mesh-name';
        name.textContent = e.name;
        name.title = e.name;

        var count = document.createElement('span');
        count.className = 'ref-mesh-count';
        count.textContent = e.mesh.kind === 'points'
            ? fmt(e.mesh.vertexCount) + ' pts'
            : fmt(e.mesh.triangleCount) + ' tri';

        var del = document.createElement('button');
        del.type = 'button';
        del.className = 'panel-btn ref-mesh-remove';
        del.textContent = '×';
        del.title = 'Remove this reference mesh';
        del.addEventListener('click', function () { removeReferenceMesh(e.id); });

        row.appendChild(show);
        row.appendChild(color);
        row.appendChild(name);
        row.appendChild(count);
        row.appendChild(del);
        list.appendChild(row);

        var op = document.createElement('div');
        op.className = 'vis-slider-row ref-mesh-opacity-row';
        var opLabel = document.createElement('span');
        opLabel.textContent = 'Opacity';
        var slider = document.createElement('input');
        slider.type = 'range';
        slider.min = '5';
        slider.max = '100';
        slider.step = '5';
        slider.value = String(Math.round(e.opacity * 100));
        slider.className = 'ref-mesh-opacity';
        var val = document.createElement('span');
        val.className = 'vis-val';
        val.textContent = slider.value + '%';
        slider.addEventListener('input', function () {
            val.textContent = slider.value + '%';
            setReferenceMeshOpacity(e.id, Number(slider.value) / 100);
        });
        op.appendChild(opLabel);
        op.appendChild(slider);
        op.appendChild(val);
        list.appendChild(op);
    });
}

/** Wire the section's Load button and draw the (empty) list. Called once. */
export function setupReferenceMeshes() {
    var btn = document.getElementById('btnLoadReferenceMesh');
    if (btn) btn.addEventListener('click', function () { loadReferenceMeshInteractive(); });
    renderReferenceMeshList();
}
