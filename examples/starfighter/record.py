#!/usr/bin/env python3
"""Record the Starfighter page with a headless Chromium and write assets/starfighter.{gif,mp4}.

    uv run cujev serve --model models/Qwen3.5-4B --port 8080     # in another terminal
    uv run --extra record examples/starfighter/record.py --takes 3 --seconds 40

Records N autoplayed takes, keeps the one with the fewest hull losses (then the
highest score), and converts it with imageio-ffmpeg's bundled ffmpeg.
"""
from __future__ import annotations

import argparse
import glob
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/examples/starfighter/index.html?autostart=1")
    ap.add_argument("--takes", type=int, default=3)
    ap.add_argument("--seconds", type=int, default=40)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "assets"))
    a = ap.parse_args()
    import imageio_ffmpeg
    from playwright.sync_api import sync_playwright

    tmp = Path(tempfile.mkdtemp(prefix="starfighter-"))
    best = None
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=swiftshader"])
        for take in range(a.takes):
            ctx = browser.new_context(viewport={"width": 1180, "height": 720},
                                      record_video_dir=str(tmp / f"take{take}"),
                                      record_video_size={"width": 1180, "height": 720})
            page = ctx.new_page()
            page.goto(a.url)
            deaths, top = 0, 0
            for _ in range(a.seconds):
                time.sleep(1)
                score = page.evaluate("parseInt(document.getElementById('score').textContent.slice(6))")
                if score < top - 5:  # the score only ever drops when the hull is lost and the game restarts
                    deaths += 1
                    top = score
                top = max(top, score)
            ctx.close()
            video = glob.glob(str(tmp / f"take{take}" / "*.webm"))[0]
            print(f"take {take}: score {top}, hull losses {deaths}")
            if best is None or (deaths, -top) < best[0]:
                best = ((deaths, -top), video)
        browser.close()
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    palette = ("fps=10,scale=780:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=112:stats_mode=diff[p];"
               "[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle")
    subprocess.run([ff, "-y", "-loglevel", "error", "-ss", "2", "-t", "22", "-i", best[1], "-vf", palette,
                    str(out / "starfighter.gif")], check=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-ss", "1.5", "-t", str(a.seconds), "-i", best[1],
                    "-c:v", "libx264", "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(out / "starfighter.mp4")], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {out / 'starfighter.gif'} and {out / 'starfighter.mp4'} from the best take")


if __name__ == "__main__":
    main()
