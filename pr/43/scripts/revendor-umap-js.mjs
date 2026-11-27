#!/usr/bin/env node
/**
 * revendor-umap-js.mjs — rebuild lib/umap-js/umap-js.esm.js from the npm package.
 *
 * umap-js ships a webpack UMD bundle (lib/umap-js.js). Its wrapper reads `this`,
 * which is undefined at the top of an ES module, so a module worker or a Node
 * test cannot import it. This script replaces ONLY that wrapper, leaving every
 * line of the bundle's code as it came, and stamps what it did at the top.
 *
 * Usage:
 *   npm pack umap-js@1.4.0 && tar xzf umap-js-1.4.0.tgz
 *   node scripts/revendor-umap-js.mjs package/lib/umap-js.js lib/umap-js/umap-js.esm.js
 */
import fs from 'node:fs';

const [src, dest] = process.argv.slice(2);
if (!src || !dest) {
    console.error('usage: revendor-umap-js.mjs <package>/lib/umap-js.js <dest>');
    process.exit(2);
}
const text = fs.readFileSync(src, 'utf8');
const HEAD = `(function webpackUniversalModuleDefinition(root, factory) {
	if(typeof exports === 'object' && typeof module === 'object')
		module.exports = factory();
	else if(typeof define === 'function' && define.amd)
		define([], factory);
	else if(typeof exports === 'object')
		exports["UMAP"] = factory();
	else
		root["UMAP"] = factory();
})(this, function() {
`;
const TAIL = `});`;
const body = text.replace(/\s+$/, '');
if (!body.startsWith(HEAD) || !body.endsWith(TAIL)) {
    console.error('The bundle\'s wrapper is not the one this script knows: read it before changing anything.');
    process.exit(1);
}
const inner = body.slice(HEAD.length, body.length - TAIL.length);
const out = `// umap-js 1.4.0 as an ES module: npm \`umap-js@1.4.0\`, lib/umap-js.js, with only its
// UMD wrapper replaced by this header and the export at the end (scripts/revendor-umap-js.mjs).
// See PROVENANCE.txt. Do not edit: LUCID keeps its changes in pose/behavior-umap.js.
const umapJsBundle = (function () {
${inner}
})();
export const UMAP = umapJsBundle.UMAP;
`;
fs.writeFileSync(dest, out);
console.log(`wrote ${dest}: ${out.length.toLocaleString('en-US')} characters`);
