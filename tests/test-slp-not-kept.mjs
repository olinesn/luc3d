/**
 * test-slp-not-kept.mjs — what the app does about a `.slp` holding what a save leaves out
 * (`import-export/slp-not-kept.js`). The real reader, loaders and Save are in
 * `tests/e2e/slp-format-30.mjs` §8. ESM, so `tests/run-mjs-tests.mjs` picks it up.
 */
import { holdsNotKept, clearNotKept, notKeptNote } from '../import-export/slp-not-kept.js';

let passed = 0, failed = 0;
const check = (cond, msg) => {
    if (cond) { passed++; console.log('  ok   ' + msg); }
    else { failed++; console.log('  FAIL ' + msg); }
};

console.log('\n1. Which sessions hold any');
{
    check(!holdsNotKept([]), 'no sessions');
    check(!holdsNotKept(null), 'nothing at all');
    check(!holdsNotKept([{}, { fileNotKept: [] }, { fileNotKept: null }, null]), 'a session read from any other file');
    check(holdsNotKept([{}, { fileNotKept: ['ROIs'] }]), 'one is enough');
}

console.log('\n2. What the status says');
{
    check(notKeptNote([{}]) === '', 'nothing to say');
    check(notKeptNote([{ fileNotKept: ['ROIs', 'masks'] }]) === "A save leaves out this file's ROIs, masks, so Save asks where to save and this file stays as it is", 'names them');
    check(notKeptNote([{ fileNotKept: ['ROIs'] }, { fileNotKept: ['ROIs', 'boxes'] }]).includes('ROIs, boxes,'), 'each once, across a file\'s sessions');
}

console.log('\n3. Once saved elsewhere');
{
    const sessions = [{ fileNotKept: ['ROIs'] }, { fileNotKept: [] }, null];
    clearNotKept(sessions);
    check(!holdsNotKept(sessions), 'the project\'s file holds none of it');
    clearNotKept(null);
    check(true, 'and no sessions is fine');
}

console.log(`\n${passed} passed, ${failed} failed`);
console.log(failed === 0 ? 'PASS' : 'FAIL');
process.exit(failed === 0 ? 0 : 1);
