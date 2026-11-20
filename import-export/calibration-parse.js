/**
 * calibration-parse.js — parse a calibration file (sleap-io / anipose TOML, or
 * LUCID's JSON) into `Camera`s. Moved unchanged from import-export/file-io.js,
 * which re-exports both parsers, so the command-line LUCID (cli/track.mjs)
 * reads a calibration with the app's own parser. DOM-free.
 *
 * Depends on: pose/pose-data.js
 */
import { Camera } from '../pose/pose-data.js';

/**
 * Parse a sleap-io format TOML calibration string into Camera objects.
 *
 * TOML format (per camera section):
 *   [cam_N]
 *   name = "back"
 *   size = [ 1280, 1024,]
 *   matrix = [ [ fx, 0.0, cx,], [ 0.0, fy, cy,], [ 0.0, 0.0, 1.0,],]
 *   distortions = [ k1, k2, p1, p2, k3,]
 *   rotation = [ rx, ry, rz,]
 *   translation = [ tx, ty, tz,]
 *
 * @param {string} text - TOML file content
 * @returns {Camera[]} Array of Camera objects
 */
export function parseCalibrationTOML(text) {
    const cameras = [];

    // Split into sections by [section_name] headers
    // Match lines like [cam_0], [cam_1], [metadata], etc.
    const sectionRegex = /^\[([^\]]+)\]\s*$/gm;
    const sections = [];
    let match;
    while ((match = sectionRegex.exec(text)) !== null) {
        sections.push({ name: match[1], start: match.index + match[0].length });
    }

    for (let i = 0; i < sections.length; i++) {
        const section = sections[i];
        // Skip non-camera sections (e.g. [metadata])
        if (!section.name.startsWith('cam_')) continue;

        const end = i + 1 < sections.length ? sections[i + 1].start : text.length;
        const body = text.substring(section.start, end);

        // Parse key-value pairs from the section body
        const props = parseTOMLSection(body);

        const name = props.name || section.name;
        const size = props.size || [640, 480];
        const matrix = props.matrix || [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        const dist = props.distortions || [0, 0, 0, 0, 0];
        const rvec = props.rotation || [0, 0, 0];
        const tvec = props.translation || [0, 0, 0];

        cameras.push(new Camera(name, matrix, dist, rvec, tvec, size));
    }

    return cameras;
}

/**
 * Parse key-value pairs from a TOML section body.
 * Handles strings, arrays, and nested arrays with trailing commas.
 *
 * @param {string} body - Section text (lines after [section_name])
 * @returns {Object} Key-value map
 */
function parseTOMLSection(body) {
    const result = {};
    const lines = body.split('\n');

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (!line || line.startsWith('#') || line.startsWith('[')) continue;

        const eqIdx = line.indexOf('=');
        if (eqIdx === -1) continue;

        const key = line.substring(0, eqIdx).trim();
        let value = line.substring(eqIdx + 1).trim();

        // Remove quotes for string values
        if (value.startsWith('"') && value.endsWith('"')) {
            result[key] = value.slice(1, -1);
            continue;
        }

        // Clean trailing commas inside arrays (TOML allows them, JSON doesn't)
        // Replace ",]" with "]" and ", ]" with "]"
        value = value.replace(/,\s*\]/g, ']');

        // Try to parse as JSON (arrays, numbers, booleans)
        try {
            result[key] = JSON.parse(value);
        } catch (e) {
            // Store as string if can't parse
            result[key] = value;
        }
    }

    return result;
}

// ============================================
// JSON calibration parser
// ============================================

/**
 * Parse a JSON calibration object into Camera objects.
 *
 * Expected format:
 *   { "cameras": [
 *       { "name": "back", "size": [w,h], "matrix": [[...]], "dist": [...],
 *         "rvec": [...], "tvec": [...] },
 *       ...
 *   ]}
 *
 * Or the export format from mv-gui:
 *   { "cameras": [
 *       { "name": "...", "matrix": [[...]], "dist": [...],
 *         "rvec": [...], "tvec": [...], "size": [...] },
 *       ...
 *   ]}
 *
 * @param {string} text - JSON file content
 * @returns {Camera[]} Array of Camera objects
 */
export function parseCalibrationJSON(text) {
    const data = JSON.parse(text);
    const cameras = [];

    const camArray = data.cameras || data;
    if (!Array.isArray(camArray)) {
        throw new Error('JSON calibration must contain a "cameras" array or be an array');
    }

    for (let i = 0; i < camArray.length; i++) {
        const c = camArray[i];
        const name = c.name || ('cam_' + i);
        const matrix = c.matrix || [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        const dist = c.dist || c.distortions || [0, 0, 0, 0, 0];
        const rvec = c.rvec || c.rotation || [0, 0, 0];
        const tvec = c.tvec || c.translation || [0, 0, 0];
        const size = c.size || [640, 480];

        cameras.push(new Camera(name, matrix, dist, rvec, tvec, size));
    }

    return cameras;
}
