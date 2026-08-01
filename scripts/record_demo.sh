#!/usr/bin/env bash
# Record the dashboard demo: docs/assets/demo.mp4, demo.gif, and three stills.
#
# WHAT IT DOES
#   1. makes sure the demo container is up and healthy
#   2. launches a headless Chrome with a debugging port
#   3. runs scripts/record_demo.py, which positions the replay at a fixed point,
#      loads the dashboard, and screencasts it to numbered JPEG frames
#   4. encodes an H.264 mp4 and a palette-optimised looping GIF
#
# WHY THIS RUNS ON THE HOST AND NOT IN THE CONTAINER
#   The serving image is deliberately tiny — python:3.10-slim plus numpy,
#   onnxruntime, fastapi and uvicorn, and no PyTorch. Putting Chrome and ffmpeg
#   in it would add roughly 500 MB to an image whose entire argument is that the
#   deployment artefact is a 126 KiB ONNX graph. The recording is a publishing
#   step, not a deployment one, so it belongs outside.
#
#   That is a real constraint rather than a preference: this script needs a
#   Chrome build and an ffmpeg binary on the machine running it. Both are
#   checked for below and named if missing.
#
# REPRODUCIBILITY
#   The replay is positioned by --seek before the page loads, so two runs record
#   the same stretch of the same committed tapes. What is NOT identical between
#   runs is the frame timing: screencast frames arrive when the compositor
#   produces them. Each frame's true duration is written into an ffconcat
#   manifest and ffmpeg resamples from that, so the market plays at real speed
#   in both runs even though the frame boundaries differ.
#
# Usage:  scripts/record_demo.sh [seconds]

set -euo pipefail

SECONDS_TO_RECORD="${1:-23}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="$REPO_ROOT/docs/assets"
FRAMES="$REPO_ROOT/build/demo-frames"
URL="${TTS_URL:-http://localhost:8000}"
DEBUG_PORT="${TTS_CHROME_PORT:-9222}"

# GIF geometry. 960 px wide is the width at which the ladder's price column is
# still readable in a GitHub README at 100% zoom; 12 fps is the lowest rate at
# which the depth tape still reads as scrolling rather than stepping. Together
# with the palette settings below they are what keeps the file under 8 MB.
GIF_WIDTH=960
GIF_FPS=12

PYTHON="${PYTHON:-python}"
if [ -x "/d/miniconda/envs/py310/python.exe" ]; then
    PYTHON="/d/miniconda/envs/py310/python.exe"
fi

# ---------------------------------------------------------------- ffmpeg
# A system ffmpeg wins if there is one. Otherwise fall back to the binary
# imageio-ffmpeg ships, which lives inside the Python environment and needs no
# system install — see docs/INTERVIEW_NOTES_master.md on why this is a
# dev-only dependency rather than a project one.
if command -v ffmpeg >/dev/null 2>&1; then
    FFMPEG="ffmpeg"
else
    FFMPEG="$("$PYTHON" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || true)"
fi
if [ -z "${FFMPEG:-}" ] || { [ "$FFMPEG" != "ffmpeg" ] && [ ! -x "$FFMPEG" ]; }; then
    echo "No ffmpeg found. Install one of:"
    echo "    winget install Gyan.FFmpeg          # system-wide"
    echo "    $PYTHON -m pip install imageio-ffmpeg   # environment-local"
    exit 1
fi

# ---------------------------------------------------------------- chrome
CHROME="${TTS_CHROME:-}"
if [ -z "$CHROME" ]; then
    for candidate in \
        "/c/Program Files/Google/Chrome/Application/chrome.exe" \
        "/c/Program Files (x86)/Google/Chrome/Application/chrome.exe" \
        "/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" \
        "/usr/bin/google-chrome" \
        "/usr/bin/chromium" \
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"; do
        if [ -x "$candidate" ]; then CHROME="$candidate"; break; fi
    done
fi
if [ -z "$CHROME" ]; then
    echo "No Chrome or Edge found. Set TTS_CHROME to the browser binary."
    exit 1
fi

echo "=== 1/4  demo container ======================================="
if ! curl -sf --max-time 5 "$URL/health" >/dev/null 2>&1; then
    echo "starting it with docker compose"
    (cd "$REPO_ROOT" && docker compose up -d api)
    for _ in $(seq 1 30); do
        if curl -sf --max-time 5 "$URL/health" >/dev/null 2>&1; then break; fi
        sleep 2
    done
fi
curl -sf --max-time 5 "$URL/health" >/dev/null || { echo "$URL/health never answered"; exit 1; }
echo "healthy at $URL"

echo "=== 2/4  headless chrome ======================================"
PROFILE="$(mktemp -d)"
"$CHROME" --headless=new --disable-gpu --hide-scrollbars \
    --remote-debugging-port="$DEBUG_PORT" --user-data-dir="$PROFILE" \
    --window-size=1440,900 about:blank >/dev/null 2>&1 &
CHROME_PID=$!
# shellcheck disable=SC2064
trap "kill $CHROME_PID 2>/dev/null || true; rm -rf '$PROFILE'" EXIT
for _ in $(seq 1 20); do
    if curl -sf --max-time 2 "http://127.0.0.1:$DEBUG_PORT/json/version" >/dev/null 2>&1; then break; fi
    sleep 1
done

echo "=== 3/4  screencast ==========================================="
mkdir -p "$ASSETS"
"$PYTHON" "$REPO_ROOT/scripts/record_demo.py" \
    --url "$URL" --port "$DEBUG_PORT" --seconds "$SECONDS_TO_RECORD" \
    --frames "$FRAMES" --stills "$ASSETS"

echo "=== 4/4  encode ==============================================="

# H.264, constant 25 fps resampled from the manifest's real durations.
#   -safe 0        the manifest uses relative paths
#   -vsync cfr     hold each frame for its stated duration, then resample
#   -crf 20        visually lossless for flat UI colour at this size
#   -pix_fmt yuv420p   the only chroma layout every browser and player accepts
#   -movflags +faststart   metadata first, so it starts playing while loading
"$FFMPEG" -y -loglevel error \
    -f concat -safe 0 -i "$FRAMES/frames.ffconcat" \
    -vsync cfr -r 25 \
    -c:v libx264 -preset slow -crf 20 -pix_fmt yuv420p -movflags +faststart \
    "$ASSETS/demo.mp4"

# GIF in two passes, which is the only way to get an acceptable one.
#
#   PASS 1 — palettegen builds ONE 256-colour palette for the whole clip.
#     stats_mode=diff weights the palette toward pixels that CHANGE between
#     frames rather than toward the largest flat areas. This UI is mostly a
#     near-black background, so the default (full) spends the palette on
#     backgrounds and leaves the amber tape and cyan ribbon banded.
#
#   PASS 2 — paletteuse maps the frames onto it.
#     dither=bayer with bayer_scale=5 is chosen over the default
#     sierra2_4a error-diffusion dither on purpose: error diffusion produces a
#     different noise pattern per frame, so flat backgrounds shimmer between
#     frames and every frame becomes expensive to compress. Ordered Bayer noise
#     is static, which both looks calmer on a dark UI and shrinks the file by
#     roughly a third at the same visual quality.
#     diff_mode=rectangle limits the mapping to the changed region of each
#     frame, which keeps the unchanged panels bit-identical frame to frame and
#     lets the GIF encoder skip them entirely.
PALETTE="$FRAMES/palette.png"
"$FFMPEG" -y -loglevel error \
    -f concat -safe 0 -i "$FRAMES/frames.ffconcat" \
    -vf "fps=$GIF_FPS,scale=$GIF_WIDTH:-1:flags=lanczos,palettegen=stats_mode=diff" \
    "$PALETTE"
"$FFMPEG" -y -loglevel error \
    -f concat -safe 0 -i "$FRAMES/frames.ffconcat" -i "$PALETTE" \
    -lavfi "fps=$GIF_FPS,scale=$GIF_WIDTH:-1:flags=lanczos[frames];[frames][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
    -loop 0 \
    "$ASSETS/demo.gif"

echo
echo "wrote:"
ls -lh "$ASSETS/demo.mp4" "$ASSETS/demo.gif" "$ASSETS"/*.png | awk '{print "  " $5 "\t" $9}'

GIF_BYTES=$(stat -c %s "$ASSETS/demo.gif" 2>/dev/null || stat -f %z "$ASSETS/demo.gif")
if [ "$GIF_BYTES" -gt 8388608 ]; then
    echo
    echo "WARNING: demo.gif is $((GIF_BYTES / 1048576)) MB, over the 8 MB budget."
    echo "Lower GIF_FPS or GIF_WIDTH in this script and re-run the encode."
fi
