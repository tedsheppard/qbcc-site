// Puts the voiceover and background music on the explainer.
//   node scripts/workspace-video/mix.mjs <voiceover.mp3> <ending.mp3> <music.mp3>
// The voiceover was read in one take (ElevenLabs, "David - Australian Tech Pro
// & Storyteller"); each line is cut out of it at the pauses and placed at the
// start of its scene. The ending (privacy, storage in Australia, and "Put your
// firm's precedents to work") is a second take; the first take's last line is
// left out. The music sits well under the voice and dips further
// while he speaks. Reads explainer-silent.mp4 (record.mjs), writes explainer.mp4.
import { execFileSync } from "node:child_process";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const [vo, vo2, music] = process.argv.slice(2);
if (!vo || !vo2 || !music) { console.error("node mix.mjs <voiceover.mp3> <ending.mp3> <music.mp3>"); process.exit(1); }
const out = resolve(dirname(fileURLToPath(import.meta.url)), "../../site/assets/workspace");
const END = 88.4;

// [take, from, to] at the take's pauses, and where the line starts in the video
// (real seconds, after record.mjs slows the second scene).
const LINES = [
  [1, 0.0, 4.64, 0.5],     // Sopal Workspace. Your firm's adjudication submissions, in one place.
  [1, 5.3, 12.66, 5.9],    // Most firms keep years of submissions on a shared drive...
  [1, 13.15, 20.77, 14.5], // Drag in a folder or a zip file...
  [1, 21.26, 27.74, 24.4], // Search it the way you search case law...
  [1, 28.1, 35.59, 33.4],  // Or ask SopalAI a question in plain English...
  [1, 36.02, 46.26, 43.5], // When it's time to draft...
  [1, 46.65, 49.63, 58.5], // Then export it to Word, in your firm's own format.
  [2, 0.0, 5.24, 66.0],    // Your data is private to your firm...
  [2, 5.71, 14.89, 71.9],  // Documents are stored onshore in Australia...
  [2, 15.4, 18.99, 82.3],  // Put your firm's precedents to work, with Sopal Workspace.
];

const chains = LINES.map(([take, a, b, at], i) => {
  const from = Math.max(0, a - 0.04), to = b + 0.15;
  return `[${take}:a]atrim=start=${from}:end=${to},asetpts=PTS-STARTPTS,afade=t=in:d=0.03,afade=t=out:st=${(to - from - 0.08).toFixed(2)}:d=0.08,adelay=${Math.round(at * 1000)}:all=1[v${i}]`;
});
const filter = [
  ...chains,
  `${LINES.map((_, i) => `[v${i}]`).join("")}amix=inputs=${LINES.length}:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11,asplit=2[voice][key]`,
  `[3:a]atrim=0:${END},asetpts=PTS-STARTPTS,volume=0.12,afade=t=in:d=1.5,afade=t=out:st=${END - 3}:d=3[bed]`,
  `[bed][key]sidechaincompress=threshold=0.02:ratio=4:attack=30:release=500[ducked]`,
  `[voice][ducked]amix=inputs=2:normalize=0:duration=first,apad=whole_dur=${END}[a]`,
].join(";");

execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-i", join(out, "explainer-silent.mp4"), "-i", vo, "-i", vo2, "-i", music,
  "-filter_complex", filter, "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-t", String(END), "-movflags", "+faststart",
  join(out, "explainer.mp4")], { stdio: "inherit" });
console.log("Wrote", join(out, "explainer.mp4"));
