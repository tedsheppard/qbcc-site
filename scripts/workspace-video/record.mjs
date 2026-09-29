// Renders explainer.html to site/assets/workspace/explainer.mp4 (and a poster).
//   node scripts/workspace-video/record.mjs            full video, 30 fps
//   node scripts/workspace-video/record.mjs 5 20 60    stills at those seconds, to ./stills
// Uses Playwright's Chromium (from any project that has playwright installed:
// default ../sopal-docs; set PLAYWRIGHT_FROM to another) and ffmpeg on the PATH.
import { createRequire } from "node:module";
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const from = process.env.PLAYWRIGHT_FROM || resolve(here, "../../../sopal-docs");
const { chromium } = createRequire(join(from, "package.json"))("playwright");
const out = resolve(here, "../../site/assets/workspace");
const FPS = 30;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
await page.goto(pathToFileURL(join(here, "explainer.html")).href + "?record=1");
await page.evaluate(() => document.fonts.ready);
await page.waitForTimeout(500);
const shot = async (t, type = "jpeg") => {
  await page.evaluate((t) => window.render(t), t);
  return page.screenshot({ type, quality: type === "jpeg" ? 93 : undefined, clip: { x: 0, y: 0, width: 1920, height: 1080 } });
};

const stills = process.argv.slice(2).map(Number);
if (stills.length) {
  const dir = join(here, "stills");
  mkdirSync(dir, { recursive: true });
  for (const t of stills) writeFileSync(join(dir, `t${t}.jpg`), await shot(t));
} else {
  mkdirSync(out, { recursive: true });
  const duration = await page.evaluate(() => window.DURATION);
  const ff = spawn("ffmpeg", ["-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", String(FPS), "-c:v", "mjpeg", "-i", "-",
    "-c:v", "libx264", "-preset", "slow", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", join(out, "explainer.mp4")], { stdio: ["pipe", "inherit", "inherit"] });
  const frames = Math.round(duration * FPS);
  for (let i = 0; i < frames; i++) {
    const buf = await shot(i / FPS);
    if (!ff.stdin.write(buf)) await new Promise((r) => ff.stdin.once("drain", r));
    if (i % 300 === 0) console.log(`${i}/${frames}`);
  }
  ff.stdin.end();
  await new Promise((r) => ff.on("close", r));
  writeFileSync(join(out, "explainer-poster.jpg"), await shot(53.9));
}
await browser.close();
