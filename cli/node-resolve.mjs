/**
 * node-resolve.mjs — the page's import map, for Node.
 *
 * In the browser, `index.html`'s import map resolves the bare specifiers the
 * vendored libraries import (`h5wasm`, `mediabunny`, `pako`, `yaml`), and
 * `offline-deps.json` maps each CDN URL to its vendored copy under `lib/`.
 * This hook reads both, so Node resolves the same specifiers to the same files.
 *
 * `yaml` is a CDN dependency that `scripts/offline_deps.py install` vendors
 * into `lib/yaml/` (gitignored). sleap-io.js imports it statically but reads
 * `.slp` files without it (it parses YAML skeletons and training configs), so
 * when it is not installed a stub stands in that throws if anything calls it.
 * Any other CDN URL without a local copy resolves to a stub that throws on
 * import, naming the URL.
 *
 * Call `registerImportMap()` before importing any app module.
 */
import fs from 'node:fs';
import path from 'node:path';
import module from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');

/** Bare specifier / URL -> absolute file URL (or a data: stub). */
export function buildImportMap() {
    const map = new Map();
    const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
    const m = html.match(/<script type="importmap">([\s\S]*?)<\/script>/);
    const imports = m ? JSON.parse(m[1]).imports || {} : {};
    const manifest = JSON.parse(fs.readFileSync(path.join(ROOT, 'offline-deps.json'), 'utf8'));
    const cdnToLocal = new Map();
    for (const pkg of Object.values(manifest.packages || {})) {
        for (const [url, local] of Object.entries(pkg.replaces || {})) cdnToLocal.set(url, local);
    }
    const local = (rel) => {
        const abs = path.join(ROOT, rel);
        return fs.existsSync(abs) ? pathToFileURL(abs).href : null;
    };
    for (const [spec, target] of Object.entries(imports)) {
        let href = null;
        if (/^https?:/.test(target)) {
            href = cdnToLocal.has(target) ? local(cdnToLocal.get(target)) : null;
            if (!href) href = stub(spec, target);
        } else {
            href = local(target);
        }
        if (href) map.set(spec, href);
    }
    for (const [url, rel] of cdnToLocal) map.set(url, local(rel) || stub(url, url));
    return map;
}

// A module standing in for a CDN dependency that is not vendored: importing it
// works; using it throws, saying how to vendor it.
function stub(spec, url) {
    const msg = JSON.stringify(spec + ' (' + url + ') is not available offline; run `python3 scripts/offline_deps.py install` once');
    const src = 'const fail = () => { throw new Error(' + msg + '); };\n' +
        'const handler = { get: (t, k) => k === Symbol.toPrimitive ? undefined : fail };\n' +
        'export default new Proxy({}, handler);\n';
    return 'data:text/javascript,' + encodeURIComponent(src);
}

let registered = false;

/** Install the resolve hook (idempotent). */
export function registerImportMap() {
    if (registered) return;
    registered = true;
    const map = buildImportMap();
    const resolveHook = (specifier, context, nextResolve) => {
        const hit = map.get(specifier);
        if (hit) return { url: hit, shortCircuit: true, format: 'module' };
        return nextResolve(specifier, context);
    };
    if (typeof module.registerHooks === 'function') {
        module.registerHooks({ resolve: resolveHook });
    } else {
        throw new Error('LUCID CLI needs Node >= 22.15 (module.registerHooks); this is Node ' + process.version);
    }
}
