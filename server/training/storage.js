import { readFileSync, writeFileSync, existsSync, mkdirSync, unlinkSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const DATA_DIR = join(__dirname, "../data/weights");

if (!existsSync(DATA_DIR)) mkdirSync(DATA_DIR, { recursive: true });

function pathFor(key) {
  return join(DATA_DIR, key.replace(/[^a-zA-Z0-9_\-]/g, "_") + ".json");
}

export function storageGet(key) {
  try {
    const p = pathFor(key);
    if (!existsSync(p)) return null;
    return readFileSync(p, "utf8");
  } catch { return null; }
}

export function storageSet(key, value) {
  writeFileSync(pathFor(key), value, "utf8");
}

export function storageRemove(key) {
  try { unlinkSync(pathFor(key)); } catch { /* already gone */ }
}
