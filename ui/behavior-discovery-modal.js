// ui/behavior-discovery-modal.js — Behavior ▸ Discover Behaviors…: what to cluster
// and how (ui/behavior-discovery.js runs it; the Behavior tab's States section shows
// the result, with its map).
//
// "All frames" clusters every animal's poses (B-SOiD); choosing a behavior with
// present labels clusters only those frames, into candidate sub-behaviors (A-SOiD's
// directed discovery). The settings are B-SOiD's and rarely need changing, so they
// are folded away, and remembered in this browser only. Esc or Cancel leave.
//
// Depends on: ui/app-state.js, ui/behavior-coding.js, ui/behavior-discovery.js,
//             ui/info-tip.js, ui/plane-definition.js (the ⓘ icon), ui/section-state.js,
//             pose/behavior-discovery.js, pose/behavior-ethogram.js, pose/behavior-labels.js,
//             pose/behavior-states.js, import-export/save-load.js

import { state } from './app-state.js';
import { sessionAnimals } from './behavior-coding.js';
import { runDiscovery, isDiscovering, DEFAULT_DISCOVERY_LAYER, splitLayerName } from './behavior-discovery.js';
import { setInfoTip } from './info-tip.js';
import { ICON_INFO } from './plane-definition.js';
import { persistSectionState } from './section-state.js';
import { MAX_BINS, NEIGHBORS, CLUSTER_SHARE } from '../pose/behavior-discovery.js';
import { behaviors, behaviorById } from '../pose/behavior-ethogram.js';
import { listBouts } from '../pose/behavior-labels.js';
import { stateLayer, cleanLayerName } from '../pose/behavior-states.js';
import { setStatus } from '../import-export/save-load.js';

var INFO = 'Groups each animal\'s poses into clusters with no labels, as B-SOiD does: features in 100 ms bins, PCA, UMAP, ' +
    'HDBSCAN, then a forest per cluster labels every frame. A cluster counts only if a second layout finds it again and it happens ' +
    'at several separate times. They are states to look at, name and promote: nothing says they are behaviors until you have watched them.';

var PREFS_KEY = 'lucid.behaviorDiscovery';
var SECTIONS_KEY = 'lucid.behaviorDiscoverySections';

function esc(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function n(x) { return Number(x || 0).toLocaleString('en-US'); }
function plural(k, one, many) { return n(k) + ' ' + (k === 1 ? one : (many || one + 's')); }

function loadPrefs() {
    try { return JSON.parse(localStorage.getItem(PREFS_KEY) || '{}') || {}; } catch (e) { return {}; }
}

function savePrefs(p) {
    try { localStorage.setItem(PREFS_KEY, JSON.stringify(p)); } catch (e) { /* best effort */ }
}

function sessionsOf() {
    return (state.sessions && state.sessions.length) ? state.sessions : (state.session ? [state.session] : []);
}

/** The behaviors a split can start from: those with present labels here, with their frames. */
function splittable(sessions) {
    return behaviors().map(function (b) {
        var frames = 0, bouts = 0;
        sessions.forEach(function (s) {
            listBouts(s, { behaviorId: b.id, present: true }).forEach(function (bt) { frames += bt.end - bt.start + 1; bouts++; });
        });
        return { behavior: b, frames: frames, bouts: bouts };
    }).filter(function (x) { return x.frames > 0; });
}

function isOpen() {
    return !!document.querySelector('.behavior-discovery-modal');
}

/**
 * Behavior ▸ Discover Behaviors….
 * @param {{split?: string}} [opts] start on splitting this behavior
 */
export function showBehaviorDiscoveryModal(opts) {
    if (isOpen() || isDiscovering()) return;
    var session = state.session;
    if (!session) { setStatus('No session loaded', 'warning'); return; }
    if (!sessionAnimals(session).length) { setStatus('No animals to cluster: run Track All, or add identities', 'warning'); return; }
    var sessions = sessionsOf();
    var prefs = loadPrefs();
    var splits = splittable(sessions);
    var overlay = document.createElement('div');
    overlay.className = 'multi-frame-modal-overlay behavior-modal-overlay';
    var modal = document.createElement('div');
    modal.className = 'multi-frame-modal behavior-io-modal behavior-discovery-modal';
    modal.setAttribute('role', 'dialog');
    modal.setAttribute('aria-label', 'Discover behaviors');
    var share = Array.isArray(prefs.share) && prefs.share.length === 2 ? prefs.share : CLUSTER_SHARE;
    modal.innerHTML = '<h3>Discover behaviors <button type="button" class="plane-section-info" id="bdiInfo" aria-label="About discovering behaviors"></button></h3>' +
        '<div class="behavior-io-opts behavior-discovery-opts">' +
        '<label>Cluster <select id="bdiWhat"><option value="">all frames</option>' + splits.map(function (x) {
            return '<option value="' + esc(x.behavior.id) + '"' + (opts && opts.split === x.behavior.id ? ' selected' : '') + '>the "' + esc(x.behavior.name) +
                '" frames, to split it (' + plural(x.bouts, 'bout') + ')</option>';
        }).join('') + '</select></label>' +
        (sessions.length > 1 ? '<label>In <select id="bdiScope"><option value="one">this session</option><option value="all"' + (prefs.scope === 'all' ? ' selected' : '') + '>all ' +
            n(sessions.length) + ' sessions</option></select></label>' : '') +
        '<label>Save as <input type="text" id="bdiName" maxlength="60" value="' + esc(DEFAULT_DISCOVERY_LAYER) + '"></label></div>' +
        '<p class="behavior-io-warn" id="bdiReplaces" hidden></p>' +
        '<details class="info-section info-collapsible" id="bdiSettingsSection"><summary><h3>Settings</h3></summary>' +
        '<div class="behavior-io-opts behavior-discovery-opts">' +
        '<label title="At most this many 100 ms bins are clustered, across sessions and animals; the forests then label every frame">Bins at most ' +
        '<input type="number" id="bdiBins" min="100" max="100000" step="1000" value="' + (prefs.maxBins > 0 ? prefs.maxBins : MAX_BINS) + '"></label>' +
        '<label title="UMAP\'s n_neighbors">Neighbors <input type="number" id="bdiNeighbors" min="5" max="200" step="5" value="' + (prefs.neighbors > 0 ? prefs.neighbors : NEIGHBORS) + '"></label>' +
        '<label title="HDBSCAN\'s smallest cluster, as a share of the bins: of the sizes in this range, the one giving the most clusters is used">Smallest cluster ' +
        '<input type="number" id="bdiShareLo" min="0.1" max="20" step="0.1" value="' + +(100 * share[0]).toFixed(2) + '"> to ' +
        '<input type="number" id="bdiShareHi" min="0.1" max="20" step="0.1" value="' + +(100 * share[1]).toFixed(2) + '"> % of the bins</label>' +
        '</div></details>' +
        '<div class="modal-actions"><button id="bdiCancel">Cancel</button><button class="primary" id="bdiGo">Discover</button></div>';
    overlay.appendChild(modal);
    document.body.appendChild(overlay);
    var info = modal.querySelector('#bdiInfo');
    info.innerHTML = ICON_INFO;
    setInfoTip(info, INFO);
    persistSectionState('bdiSettingsSection', SECTIONS_KEY);

    var whatEl = modal.querySelector('#bdiWhat'), nameEl = modal.querySelector('#bdiName'), scopeEl = modal.querySelector('#bdiScope');
    var nameTouched = false;
    var refresh = function () {
        var b = whatEl.value ? behaviorById(whatEl.value) : null;
        if (!nameTouched) nameEl.value = b ? cleanLayerName(splitLayerName(b.name)) : DEFAULT_DISCOVERY_LAYER;
        var layer = cleanLayerName(nameEl.value);
        var list = scopeEl && scopeEl.value === 'all' ? sessions : [session];
        var replaces = list.filter(function (s) { return layer && stateLayer(s, layer); }).length;
        var warn = modal.querySelector('#bdiReplaces');
        warn.hidden = !replaces;
        warn.textContent = replaces ? 'This replaces the states named "' + layer + '" in ' + plural(replaces, 'session') + '.' : '';
    };
    whatEl.addEventListener('change', refresh);
    if (scopeEl) scopeEl.addEventListener('change', refresh);
    nameEl.addEventListener('input', function () { nameTouched = true; refresh(); });
    refresh();

    var onKey = function (e) {
        if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); close(); }
    };
    function close() {
        document.removeEventListener('keydown', onKey, true);
        overlay.remove();
    }
    document.addEventListener('keydown', onKey, true);
    modal.querySelector('#bdiCancel').addEventListener('click', close);
    modal.querySelector('#bdiGo').addEventListener('click', function () {
        var layer = cleanLayerName(nameEl.value);
        if (!layer) { setStatus('Give the states a name', 'warning'); nameEl.focus(); return; }
        var maxBins = Number(modal.querySelector('#bdiBins').value), neighbors = Number(modal.querySelector('#bdiNeighbors').value);
        var lo = Number(modal.querySelector('#bdiShareLo').value) / 100, hi = Number(modal.querySelector('#bdiShareHi').value) / 100;
        var choice = {
            all: !!(scopeEl && scopeEl.value === 'all'), split: whatEl.value || null, layer: layer,
            maxBins: maxBins >= 100 ? Math.round(maxBins) : MAX_BINS, neighbors: neighbors >= 2 ? Math.round(neighbors) : NEIGHBORS,
            clusterShare: lo > 0 && hi >= lo ? [lo, hi] : CLUSTER_SHARE,
        };
        savePrefs({ scope: choice.all ? 'all' : 'one', maxBins: choice.maxBins, neighbors: choice.neighbors, share: choice.clusterShare });
        close();
        runDiscovery(choice);
    });
}
