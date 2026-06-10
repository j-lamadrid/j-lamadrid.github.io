const fs = require("fs");
const path = require("path");

const musicDir = path.join(__dirname, "..", "public", "music");
const manifestPath = path.join(musicDir, "tracks.json");

function titleFromFileName(fileName) {
  return path
    .basename(fileName, path.extname(fileName))
    .replace(/[-_]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

fs.mkdirSync(musicDir, { recursive: true });

const tracks = fs
  .readdirSync(musicDir)
  .filter((fileName) => path.extname(fileName).toLowerCase() === ".wav")
  .sort((a, b) => a.localeCompare(b))
  .map((fileName) => ({
    title: titleFromFileName(fileName),
    file: `/music/${fileName}`,
  }));

fs.writeFileSync(manifestPath, `${JSON.stringify(tracks, null, 2)}\n`);

console.log(`Generated music manifest with ${tracks.length} track(s).`);
