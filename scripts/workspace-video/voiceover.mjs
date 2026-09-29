// Adds an ElevenLabs voiceover to site/assets/workspace/explainer.mp4.
//   ELEVENLABS_API_KEY=... node scripts/workspace-video/voiceover.mjs          voice every line, mix, mux
//   ELEVENLABS_API_KEY=... node scripts/workspace-video/voiceover.mjs --voices  list Australian male voices
// The key can also live in ~/.config/sopal/elevenlabs.key (one line). Choose a
// voice with ELEVENLABS_VOICE_ID; without one, the first Australian male voice
// in the account is used. Run record.mjs first: this reads the silent video
// it writes (explainer-silent.mp4) and writes explainer.mp4 with sound.
// Lines start at their scene's start and must end before the next line; a
// line that runs long is sped up a little (up to 12%), otherwise it stops
// with an error so the script or the timeline can be shortened.
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const assets = resolve(here, "../../site/assets/workspace");
const clips = join(here, "vo");
const keyFile = join(homedir(), ".config/sopal/elevenlabs.key");
const KEY = process.env.ELEVENLABS_API_KEY || (existsSync(keyFile) ? readFileSync(keyFile, "utf8").trim() : "");
if (!KEY) { console.error("No ElevenLabs key: set ELEVENLABS_API_KEY or write it to " + keyFile); process.exit(1); }

// [start second, text]. Scene starts are in explainer.html's render(t).
const LINES = [
  [0.6, "Sopal Workspace. Your firm's adjudication submissions, in one place."],
  [5.9, "Most firms keep years of submissions on a shared drive. Finding how you argued a point usually means opening them one by one."],
  [11.6, "Drag in a folder or a zip file. Each submission is read, and filed by type, parties, State, amount claimed and date."],
  [21.5, "Search it the way you search case law: Boolean terms, phrases and proximity, across everything the firm has filed."],
  [30.5, "Or ask SopalAI a question in plain English. Every answer cites the page it came from, and quotes are checked against that page."],
  [40.6, "When it's time to draft, SopalAI reads your firm's precedents first, then the Act, the case law and past adjudication decisions, and writes a properly numbered draft you can edit."],
  [55.6, "Then export it to Word, in your firm's own format."],
  [63.1, "Your library is private to your firm, and your documents are not used to train AI models."],
  [67.8, "Sopal Workspace. Your first month is free."],
];
const END = 72;

async function api(path, init = {}) {
  const res = await fetch("https://api.elevenlabs.io" + path, { ...init, headers: { "xi-api-key": KEY, ...(init.headers || {}) } });
  if (!res.ok) throw new Error(`${path}: ${res.status} ${await res.text()}`);
  return res;
}
async function australianMen() {
  const { voices } = await (await api("/v1/voices")).json();
  return voices.filter((v) => /austral/i.test(v.labels?.accent || "") && /^male$/i.test(v.labels?.gender || ""));
}

if (process.argv.includes("--voices")) {
  for (const v of await australianMen()) console.log(v.voice_id, v.name, JSON.stringify(v.labels), v.preview_url);
  process.exit(0);
}

let voice = process.env.ELEVENLABS_VOICE_ID;
if (!voice) {
  const found = await australianMen();
  if (!found.length) { console.error("No Australian male voice in this account. Add one from the Voice Library, or set ELEVENLABS_VOICE_ID."); process.exit(1); }
  voice = found[0].voice_id;
  console.log("Voice:", found[0].name, voice);
}

mkdirSync(clips, { recursive: true });
const dur = (f) => +execFileSync("ffprobe", ["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f]).toString();
const inputs = [];
for (let i = 0; i < LINES.length; i++) {
  const [start, text] = LINES[i];
  const raw = join(clips, `line${i + 1}.mp3`);
  const res = await api(`/v1/text-to-speech/${voice}?output_format=mp3_44100_128`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text, model_id: "eleven_multilingual_v2", voice_settings: { stability: 0.5, similarity_boost: 0.8, style: 0.15, use_speaker_boost: true } }),
  });
  writeFileSync(raw, Buffer.from(await res.arrayBuffer()));
  const slot = (LINES[i + 1]?.[0] ?? END) - start - 0.25;
  const d = dur(raw);
  let tempo = 1;
  if (d > slot) {
    tempo = d / slot;
    if (tempo > 1.12) throw new Error(`Line ${i + 1} is ${d.toFixed(1)}s for a ${slot.toFixed(1)}s slot. Shorten it: "${text}"`);
  }
  console.log(`line ${i + 1}: ${d.toFixed(1)}s in ${slot.toFixed(1)}s${tempo > 1 ? `, sped up ${(tempo * 100 - 100).toFixed(0)}%` : ""}`);
  inputs.push({ file: raw, start, tempo });
}

const silent = join(assets, "explainer-silent.mp4");
const args = ["-y", "-loglevel", "error", "-i", silent];
inputs.forEach((c) => args.push("-i", c.file));
const chains = inputs.map((c, i) => `[${i + 1}:a]${c.tempo > 1 ? `atempo=${c.tempo.toFixed(4)},` : ""}adelay=${Math.round(c.start * 1000)}:all=1[a${i}]`);
const mix = `${inputs.map((_, i) => `[a${i}]`).join("")}amix=inputs=${inputs.length}:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11[vo]`;
args.push("-filter_complex", [...chains, mix].join(";"), "-map", "0:v", "-map", "[vo]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-t", String(END), "-movflags", "+faststart", join(assets, "explainer.mp4"));
execFileSync("ffmpeg", args, { stdio: "inherit" });
console.log("Wrote", join(assets, "explainer.mp4"));
