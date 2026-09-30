// Puts the voiceover and background music on the explainer.
//   node scripts/workspace-video/mix.mjs <voiceover.mp3> <music.mp3>
// The voiceover was read in one take (ElevenLabs, "David - Australian Tech Pro
// & Storyteller"); each line is cut out of it at the pauses and placed at the
// start of its scene. The closing line ("Sopal Workspace. Your first month is
// free.") is left out. The music sits well under the voice and dips further
// while he speaks. Reads explainer-silent.mp4 (record.mjs), writes explainer.mp4.
import { execFileSync } from "node:child_process";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const [vo, music] = process.argv.slice(2);
if (!vo || !music) { console.error("node mix.mjs <voiceover.mp3> <music.mp3>"); process.exit(1); }
const out = resolve(dirname(fileURLToPath(import.meta.url)), "../../site/assets/workspace");
const END = 74.9;

// [from, to] in the voiceover (at its pauses), and where it starts in the video.
const LINES = [
  [0.0, 4.64, 0.5],    // Sopal Workspace. Your firm's adjudication submissions, in one place.
  [5.3, 12.66, 5.9],   // Most firms keep years of submissions on a shared drive...
  [13.15, 20.77, 14.5], // Drag in a folder or a zip file...
  [21.26, 27.74, 24.4], // Search it the way you search case law...
  [28.1, 35.59, 33.4],  // Or ask SopalAI a question in plain English...
  [36.02, 46.26, 43.5], // When it's time to draft...
  [46.65, 49.63, 58.5], // Then export it to Word, in your firm's own format.
  [50.06, 54.96, 66.0], // Your library is private to your firm...
];

const chains = LINES.map(([a, b, at], i) =>
  `[1:a]atrim=start=${Math.max(0, a - 0.04)}:end=${b + 0.15},asetpts=PTS-STARTPTS,afade=t=in:d=0.03,afade=t=out:st=${(b + 0.15 - Math.max(0, a - 0.04) - 0.08).toFixed(2)}:d=0.08,adelay=${Math.round(at * 1000)}:all=1[v${i}]`);
const filter = [
  ...chains,
  `${LINES.map((_, i) => `[v${i}]`).join("")}amix=inputs=${LINES.length}:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11,asplit=2[voice][key]`,
  `[2:a]atrim=0:${END},asetpts=PTS-STARTPTS,volume=0.12,afade=t=in:d=1.5,afade=t=out:st=${END - 3}:d=3[bed]`,
  `[bed][key]sidechaincompress=threshold=0.02:ratio=4:attack=30:release=500[ducked]`,
  `[voice][ducked]amix=inputs=2:normalize=0:duration=first,apad=whole_dur=${END}[a]`,
].join(";");

execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-i", join(out, "explainer-silent.mp4"), "-i", vo, "-i", music,
  "-filter_complex", filter, "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-t", String(END), "-movflags", "+faststart",
  join(out, "explainer.mp4")], { stdio: "inherit" });
console.log("Wrote", join(out, "explainer.mp4"));
