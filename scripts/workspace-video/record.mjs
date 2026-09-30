// Renders explainer.html to site/assets/workspace/explainer-silent.mp4 (copied to
// explainer.mp4, which voiceover.mjs then replaces with a voiced cut) and a poster.
//   node scripts/workspace-video/record.mjs            full video, 30 fps
//   node scripts/workspace-video/record.mjs 5 20 60    stills at those seconds, to ./stills
// Uses Playwright's Chromium (from any project that has playwright installed:
// default ../sopal-docs; set PLAYWRIGHT_FROM to another) and ffmpeg on the PATH.
import { createRequire } from "node:module";
import { spawn } from "node:child_process";
import { copyFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const from = process.env.PLAYWRIGHT_FROM || resolve(here, "../../../sopal-docs");
const { chromium } = createRequire(join(from, "package.json"))("playwright");
const out = resolve(here, "../../site/assets/workspace");
const FPS = 30;
// --page=demo renders demo.html to demo.mp4 instead of the explainer.
const PAGE = (process.argv.find((a) => a.startsWith("--page=")) || "--page=explainer").slice(7);
const IS_EXPLAINER = PAGE === "explainer";
// The voiceover's second line is longer than the second scene, so that scene
// is played more slowly: real seconds 5.5 to 13.9 show video seconds 5.5 to 11,
// and everything after it starts 2.9 seconds later.
const SLOW = IS_EXPLAINER ? { from: 5.5, to: 11, extra: 2.9 } : { from: 1e9, to: 1e9, extra: 0 };
const videoTime = (t) => t < SLOW.from ? t
  : t < SLOW.to + SLOW.extra ? SLOW.from + (t - SLOW.from) * (SLOW.to - SLOW.from) / (SLOW.to - SLOW.from + SLOW.extra)
  : t - SLOW.extra;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
await page.goto(pathToFileURL(join(here, `${PAGE}.html`)).href + "?record=1");
await page.evaluate(() => document.fonts.ready);
await page.waitForTimeout(500);
const shot = async (t, type = "jpeg") => {
  await page.evaluate((t) => window.render(t), t);
  return page.screenshot({ type, quality: type === "jpeg" ? 93 : undefined, clip: { x: 0, y: 0, width: 1920, height: 1080 } });
};

const stills = process.argv.slice(2).filter((a) => !a.startsWith("--")).map(Number);
if (stills.length) {
  const dir = join(here, "stills");
  mkdirSync(dir, { recursive: true });
  for (const t of stills) writeFileSync(join(dir, `t${t}.jpg`), await shot(t));
} else {
  mkdirSync(out, { recursive: true });
  const duration = await page.evaluate(() => window.DURATION);
  const ff = spawn("ffmpeg", ["-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", String(FPS), "-c:v", "mjpeg", "-i", "-",
    "-c:v", "libx264", "-preset", "slow", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", join(out, `${PAGE}-silent.mp4`)], { stdio: ["pipe", "inherit", "inherit"] });
  const frames = Math.round((duration + SLOW.extra) * FPS);
  for (let i = 0; i < frames; i++) {
    const buf = await shot(videoTime(i / FPS));
    if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once("drain", r));
    if (i % 300 === 0) console.log(`${i}/${frames}`);
  }
  ff.stdin.end();
  await new Promise((r) => ff.on("close", r));
  writeFileSync(join(out, `${PAGE}-poster.jpg`), await shot(IS_EXPLAINER ? 53.9 : 46));
  // Until voiceover.mjs adds the voice, the page plays the silent cut.
  copyFileSync(join(out, `${PAGE}-silent.mp4`), join(out, `${PAGE}.mp4`));
}
await browser.close();
