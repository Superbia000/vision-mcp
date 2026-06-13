/**
 * Vision-MCP v7: Video processing (ffmpeg splitting)
 */

import { spawn } from "child_process";
import { join } from "path";
import { tmpdir } from "os";
import { mkdirSync, unlinkSync, rmdirSync } from "fs";

function runProcess(
  cmd: string,
  args: string[],
  timeoutMs = 600_000
): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { windowsHide: true });
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error(`${cmd} timed out after ${timeoutMs}ms`));
    }, timeoutMs);
    child.stdout.on("data", (d) => { stdout += d.toString(); });
    child.stderr.on("data", (d) => { stderr += d.toString(); });
    child.on("error", (err) => {
      clearTimeout(timer);
      reject(err);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code === 0) resolve({ stdout, stderr });
      else reject(new Error(`${cmd} exited ${code}: ${stderr || stdout}`.trim()));
    });
  });
}

/** Get video duration in seconds using ffprobe */
export async function getVideoDurationSec(videoPath: string): Promise<number> {
  const { stdout } = await runProcess("ffprobe", [
    "-v", "error",
    "-show_entries", "format=duration",
    "-of", "default=noprint_wrappers=1:nokey=1",
    videoPath,
  ], 60_000);
  const duration = Number(stdout.trim());
  if (!Number.isFinite(duration) || duration <= 0) {
    throw new Error("ffprobe could not determine video duration");
  }
  return duration;
}

/** Split video into chunks using ffmpeg */
export async function splitVideoIntoChunks(
  videoPath: string,
  chunkDurationSec: number
): Promise<{ dir: string; chunks: string[] }> {
  const duration = await getVideoDurationSec(videoPath);
  const tmpDir = join(
    tmpdir(),
    `vision-video-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
  );
  mkdirSync(tmpDir, { recursive: true });
  const chunks: string[] = [];

  for (let start = 0, idx = 1; start < duration; start += chunkDurationSec, idx++) {
    const out = join(tmpDir, `chunk-${String(idx).padStart(4, "0")}.mp4`);
    await runProcess("ffmpeg", [
      "-hide_banner", "-loglevel", "error",
      "-ss", String(start),
      "-t", String(chunkDurationSec),
      "-i", videoPath,
      "-map", "0",
      "-c", "copy",
      "-avoid_negative_ts", "make_zero",
      out,
    ]);
    chunks.push(out);
  }
  return { dir: tmpDir, chunks };
}

/** Cleanup chunk files */
export function cleanupChunks(chunks: string[], dir: string): void {
  for (const chunkPath of chunks) {
    try { unlinkSync(chunkPath); } catch {}
  }
  try { rmdirSync(dir); } catch {}
}
