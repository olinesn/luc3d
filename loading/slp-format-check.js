/**
 * slp-format-check.js — refuse a `.slp` file LUC3D cannot read.
 *
 * Two kinds of file, and without a check each reads as something it isn't, with no
 * error, under a success message:
 * - Too new: a `format_id` at or above the limit, or a `required_features` entry LUC3D
 *   doesn't understand. Its tables may not be where the readers look. The rule is
 *   Python's `_check_format_id`: a file with no `format_id` passes, as before.
 * - Compressed with a filter this HDF5 build doesn't have, such as Python's
 *   `compression="lzf"`. h5wasm logs the failure but still hands back the buffer it
 *   allocated, so the tables read as garbage numbers or empty frames.
 *
 * No imports, deliberately: the raw worker and Node tests load this. The vendored
 * reader cannot (`lib/` stays self-contained), so it carries its own copy of both
 * rules and the wording (`LUCID local patch (luc3d newer-format)`, `(slp 3.0)`), and
 * `tests/e2e/slp-newer-format.mjs` fails if the messages ever differ.
 */

/** A `/metadata` `format_id` at or above this uses a layout LUC3D can't read. */
export const SLP_FORMAT_ID_LIMIT = 4.0;

/** `required_features` entries LUC3D understands. */
const SUPPORTED_FEATURES = new Set();

/** HDF5's own filters (deflate, shuffle, fletcher32, szip, nbit, scaleoffset) stop here. */
const LAST_BUILTIN_FILTER_ID = 6;

// The vendored reader's error class has this name too, so one test covers both.
const ERROR_NAME = 'UnsupportedSlpFormatError';

const UPDATE_ADVICE = ' Update LUC3D to read it, or re-save the file'
    + ' from Python sleap-io with `legacy_format=True`.';

function refusal(message, formatId) {
    var err = new Error(message);
    err.name = ERROR_NAME;
    err.formatId = formatId;
    return err;
}

/**
 * What a refused file tells the user.
 *
 * @param {number} formatId The file's `format_id`.
 * @param {string} [fileName] Its name, if known.
 * @returns {string}
 */
export function newerSlpFormatMessage(formatId, fileName) {
    var version = Number.isInteger(formatId) ? formatId.toFixed(1) : String(formatId);
    return (fileName || 'This file') + ' uses SLP format ' + version
        + ', which is newer than this version of LUC3D supports (max '
        + (SLP_FORMAT_ID_LIMIT - 1) + '.x).' + UPDATE_ADVICE;
}

/**
 * What a file asking for features LUC3D doesn't understand tells the user.
 *
 * @param {string[]} features The features, sorted.
 * @param {string} [fileName] Its name, if known.
 * @returns {string}
 */
export function requiredFeaturesMessage(features, fileName) {
    return (fileName || 'This file') + " requires SLP features this version of LUC3D doesn't support ("
        + features.join(', ') + ').' + UPDATE_ADVICE;
}

/**
 * What a file with a dataset LUC3D can't decompress tells the user.
 *
 * @param {string} path The dataset, no leading slash.
 * @param {string} filter The filter's name.
 * @param {string} [fileName] The file's name, if known.
 * @returns {string}
 */
export function unreadableFilterMessage(path, filter, fileName) {
    return (fileName || 'This file') + ': /' + path + ' is compressed with "' + filter
        + '", which LUC3D cannot read. Re-save the file from Python sleap-io with'
        + ' compression="gzip" or without compression.';
}

function requiredFeatures(attr) {
    if (attr === null || attr === undefined) return [];
    var value = attr.value !== undefined ? attr.value : attr;
    var names = Array.isArray(value) || (ArrayBuffer.isView(value) && !(value instanceof Uint8Array))
        ? Array.from(value) : [value];
    return names.map(function (name) {
        return name instanceof Uint8Array ? new TextDecoder().decode(name) : String(name);
    });
}

/**
 * Throw if a file's `/metadata` attributes say it is too new to read. Call this
 * before reading anything else from the file: a newer layout may not hold what
 * the other reads look for, and most of them turn a miss into an empty result.
 *
 * @param {?Object} metadataAttrs The attributes of `/metadata`, as h5wasm gives
 *   them (`{value}` wrappers) or bare; nothing at all passes.
 * @param {string} [fileName] The file's name, for the message.
 * @throws {Error} Named `UnsupportedSlpFormatError`, with `formatId`.
 */
export function checkSlpFormatAttrs(metadataAttrs, fileName) {
    if (!metadataAttrs) return;
    var attr = metadataAttrs.format_id;
    var formatId = attr === null || attr === undefined ? 1 : Number(attr.value !== undefined ? attr.value : attr);
    if (formatId >= SLP_FORMAT_ID_LIMIT) throw refusal(newerSlpFormatMessage(formatId, fileName), formatId);
    var unsupported = requiredFeatures(metadataAttrs.required_features)
        .filter(function (name) { return !SUPPORTED_FEATURES.has(name); })
        .sort();
    if (unsupported.length > 0) throw refusal(requiredFeaturesMessage(unsupported, fileName), formatId);
}

/**
 * The first dataset under an h5wasm group that is compressed with a filter this HDF5
 * build doesn't have.
 *
 * @param {Object} group An h5wasm `File` or `Group`.
 * @param {string} [prefix] The group's path.
 * @returns {?{path: string, filter: string}}
 */
export function findUnreadableFilter(group, prefix) {
    var names = group.keys();
    for (var i = 0; i < names.length; i++) {
        var path = prefix ? prefix + '/' + names[i] : names[i];
        var node = group.get(names[i]);
        if (!node) continue;
        if (typeof node.keys === 'function' && node.shape === undefined) {
            var inner = findUnreadableFilter(node, path);
            if (inner) return inner;
            continue;
        }
        var filters = null;
        try { filters = node.filters; } catch (e) { continue; } // nothing to go on
        if (!Array.isArray(filters)) continue;
        for (var k = 0; k < filters.length; k++) {
            if (filters[k].id > LAST_BUILTIN_FILTER_ID) {
                return { path: path, filter: filters[k].name || String(filters[k].id) };
            }
        }
    }
    return null;
}

/**
 * Throw if any dataset in a file can't be decompressed. One check of the whole file,
 * before anything is read, so no way of reading a table can skip it.
 *
 * @param {Object} file An h5wasm `File`.
 * @param {string} [fileName] The file's name, for the message.
 * @throws {Error} Named `UnsupportedSlpFormatError`.
 */
export function checkSlpFilters(file, fileName) {
    var found = findUnreadableFilter(file);
    if (found) throw refusal(unreadableFilterMessage(found.path, found.filter, fileName));
}

/**
 * Whether a failed read was one of the refusals above, from either reader. By name, so
 * it holds across the worker and the bundle.
 */
export function isNewerSlpFormatError(err) {
    return !!err && err.name === ERROR_NAME;
}

/**
 * Status-bar text for a failed `.slp` read. A refusal is left as it is: nothing
 * failed to parse, and the status line has no room to spare.
 *
 * @param {string} prefix What to call any other failure, e.g. `'SLP parse error'`.
 * @param {*} err What the read threw.
 * @returns {string}
 */
export function slpReadErrorStatus(prefix, err) {
    var message = (err && err.message) || String(err);
    return isNewerSlpFormatError(err) ? message : prefix + ': ' + message;
}
