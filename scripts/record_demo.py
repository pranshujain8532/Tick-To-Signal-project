"""Record the running dashboard to a sequence of frames, and grab the stills.

WHAT
    Drives a headless Chrome over the DevTools Protocol: positions the replay at
    a fixed point, loads the dashboard, captures a screencast to numbered JPEG
    frames plus an `ffconcat` manifest carrying each frame's real duration, and
    optionally captures three cropped PNG stills. `scripts/record_demo.sh` turns
    the frames into `docs/assets/demo.mp4` and `docs/assets/demo.gif`.

WHY A SCREENCAST AND NOT A SCREEN RECORDER
    Screen recorders capture whatever is in front of them, which means the demo
    depends on the machine that made it: window chrome, notification popups,
    scaling, whatever else was on the desktop. `Page.startScreencast` captures
    the page's own compositor output at a known size, headless, with no desktop
    involved. The recording is therefore reproducible by anyone with Docker and
    a Chrome build, which is the point of committing the script at all.

WHY NOT A FIXED FRAME RATE
    Chrome emits screencast frames when the compositor produces them, and that
    rate is not constant — it dips while the page does a full tape repaint and
    while the encoder competes with the model for cores. Assuming a constant
    rate would make the finished video subtly wrong: the market would appear to
    speed up exactly where the dashboard was working hardest, which for a
    project about tail latency would be a self-flattering artefact.

    So each frame's arrival time is recorded and written into an `ffconcat`
    manifest as an explicit duration. ffmpeg resamples that to a constant output
    rate, and the result plays at real speed.

WHY JPEG AND NOT PNG FOR THE VIDEO FRAMES
    ~550 lossless frames of a 1440x900 dark UI is over 100 MB of temporary
    files, for pixels that are about to be quantised to 256 colours for a GIF
    and H.264-compressed for the video. Quality 92 is well above what either
    output preserves. The three STILLS are PNG, because those are the images
    that end up in the README at full fidelity.

DESIGN DECISION — the replay is positioned before the page is loaded.
    The demo has to show the boot sequence, which only happens on page load, and
    a live signal, which needs 599 anchors of warmup after any discontinuity. A
    seek is a discontinuity, so seeking after load would put a 2.4 s blackout
    into the first seconds of the recording.

    Instead this seeks first through its own control connection, waits for the
    engine to warm at that position, and only then loads the page. The
    recording opens on the boot sequence with real data already flowing behind
    it, and the position is fixed, so two runs record the same stretch of
    market.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import urllib.request
from pathlib import Path

import websockets

# The stills the README uses. Coordinates are in CSS pixels at 1440x900 and are
# tied to the grid in serving/dashboard/css/layout.css; if that grid changes,
# these move with it.
STILLS = {
    "hero": None,  # the whole viewport
    "depth-tape": {"x": 0, "y": 56, "width": 1080, "height": 360, "scale": 2},
    "pareto": {"x": 772, "y": 655, "width": 668, "height": 245, "scale": 2},
}


class DevTools:
    """A minimal DevTools Protocol client: request/response plus events.

    Deliberately not a library. The protocol is a JSON envelope over a
    websocket, this file uses six methods of it, and `websockets` is already a
    dependency because `data_engine/capture.py` speaks to Binance with it.
    """

    def __init__(self, socket: websockets.WebSocketClientProtocol) -> None:
        self._socket = socket
        self._next_id = 1
        self._pending: dict[int, asyncio.Future] = {}
        self._events: asyncio.Queue = asyncio.Queue()
        self._pump = asyncio.create_task(self._read_forever())

    async def _read_forever(self) -> None:
        async for raw in self._socket:
            message = json.loads(raw)
            identifier = message.get("id")
            if identifier is not None and identifier in self._pending:
                self._pending.pop(identifier).set_result(message.get("result", {}))
            elif "method" in message:
                await self._events.put(message)

    async def call(self, method: str, **params) -> dict:
        identifier = self._next_id
        self._next_id += 1
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[identifier] = future
        await self._socket.send(json.dumps({"id": identifier, "method": method, "params": params}))
        return await future

    async def next_event(self, method: str, timeout: float = 30.0) -> dict:
        while True:
            message = await asyncio.wait_for(self._events.get(), timeout=timeout)
            if message["method"] == method:
                return message["params"]

    async def evaluate(self, expression: str):
        result = await self.call("Runtime.evaluate", expression=expression, returnByValue=True)
        return result.get("result", {}).get("value")

    def close(self) -> None:
        self._pump.cancel()


def page_target(debug_port: int) -> str:
    """The websocket URL of the first page target in a running Chrome."""
    with urllib.request.urlopen(f"http://127.0.0.1:{debug_port}/json/list", timeout=10) as response:
        targets = json.loads(response.read())
    for target in targets:
        if target["type"] == "page":
            return target["webSocketDebuggerUrl"]
    raise SystemExit(f"no page target on port {debug_port}; is Chrome running with --headless?")


async def position_the_replay(base_url: str, seek: float, speed: float) -> None:
    """Seek and set speed through a control connection, then let the engine warm.

    Uses the same public websocket the dashboard uses. The feed is a single
    shared object — one producer serves every viewer — so a seek here moves the
    replay for the page that is about to be loaded.
    """
    stream_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    async with websockets.connect(f"{stream_url}/ws/stream") as socket:
        await socket.send(json.dumps({"cmd": "speed", "value": speed}))
        await socket.send(json.dumps({"cmd": "seek", "value": seek}))
        # Drain until a frame carries a signal: that is the engine reporting it
        # has its 599 anchors and the demo will not open on a blackout. Bounded,
        # because a demo that hangs waiting for a warm engine is worse than one
        # that records the warmup honestly.
        deadline = asyncio.get_running_loop().time() + 30.0
        while asyncio.get_running_loop().time() < deadline:
            message = json.loads(await socket.recv())
            if message.get("type") == "frame" and message.get("signal") is not None:
                return
    print("  warning: the engine did not warm within 30 s; recording anyway")


async def record(arguments: argparse.Namespace) -> None:
    frames_directory = Path(arguments.frames)
    frames_directory.mkdir(parents=True, exist_ok=True)
    for stale in frames_directory.glob("*.jpg"):
        stale.unlink()

    print(f"positioning replay at {arguments.seek:.3f} of the tape, speed {arguments.speed}x")
    await position_the_replay(arguments.url, arguments.seek, arguments.speed)

    async with websockets.connect(page_target(arguments.port), max_size=64 * 1024 * 1024) as socket:
        devtools = DevTools(socket)
        await devtools.call("Page.enable")
        await devtools.call("Runtime.enable")
        await devtools.call("Network.enable")
        # The recording must show what is on disk now, not what Chrome cached
        # during a previous run.
        await devtools.call("Network.setCacheDisabled", cacheDisabled=True)
        await devtools.call(
            "Emulation.setDeviceMetricsOverride",
            width=arguments.width,
            height=arguments.height,
            deviceScaleFactor=1,
            mobile=False,
        )

        await devtools.call("Page.navigate", url=arguments.url + "/")
        await devtools.call(
            "Page.startScreencast",
            format="jpeg",
            quality=arguments.quality,
            maxWidth=arguments.width,
            maxHeight=arguments.height,
            everyNthFrame=1,
        )

        print(f"recording {arguments.seconds:.0f} s ...")
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + arguments.seconds
        timestamps: list[float] = []
        index = 0

        while loop.time() < deadline:
            try:
                params = await devtools.next_event("Page.screencastFrame", timeout=5.0)
            except asyncio.TimeoutError:
                print("  warning: no screencast frame for 5 s")
                break
            # Acknowledged immediately: Chrome will not send the next frame
            # until this one is acknowledged, which is the same credit-of-one
            # backpressure the dashboard's own websocket uses.
            await devtools.call("Page.screencastFrameAck", sessionId=params["sessionId"])
            (frames_directory / f"frame_{index:05d}.jpg").write_bytes(
                base64.b64decode(params["data"])
            )
            timestamps.append(loop.time())
            index += 1

        await devtools.call("Page.stopScreencast")

        if index < 2:
            raise SystemExit("captured fewer than two frames; nothing to encode")

        write_manifest(frames_directory, timestamps)
        elapsed = timestamps[-1] - timestamps[0]
        print(f"  {index} frames over {elapsed:.1f} s = {index / elapsed:.1f} fps captured")

        if arguments.stills:
            await capture_stills(devtools, Path(arguments.stills))

        devtools.close()


def write_manifest(frames_directory: Path, timestamps: list[float]) -> None:
    """An ffconcat manifest with each frame's true on-screen duration.

    The last frame needs an explicit duration too — ffmpeg gives a trailing
    entry zero length otherwise and drops it — and the concat demuxer requires
    the final file to be repeated for its duration to be honoured.
    """
    lines = ["ffconcat version 1.0"]
    for index in range(len(timestamps)):
        if index + 1 < len(timestamps):
            duration = timestamps[index + 1] - timestamps[index]
        else:
            duration = timestamps[-1] - timestamps[-2]
        lines.append(f"file 'frame_{index:05d}.jpg'")
        lines.append(f"duration {duration:.6f}")
    lines.append(f"file 'frame_{len(timestamps) - 1:05d}.jpg'")
    (frames_directory / "frames.ffconcat").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def capture_stills(devtools: DevTools, directory: Path) -> None:
    """Three PNGs: the whole dashboard, the depth tape, the Pareto panel."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, clip in STILLS.items():
        parameters = {"format": "png"}
        if clip is not None:
            parameters["clip"] = {**clip, "scale": clip["scale"]}
        result = await devtools.call("Page.captureScreenshot", **parameters)
        target = directory / f"{name}.png"
        target.write_bytes(base64.b64decode(result["data"]))
        print(f"  still: {target}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--port", type=int, default=9222, help="Chrome remote debugging port")
    parser.add_argument("--seconds", type=float, default=23.0)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--quality", type=int, default=92)
    # A fraction of the total replay, so a run records the same market twice.
    parser.add_argument("--seek", type=float, default=0.34)
    parser.add_argument("--speed", type=float, default=10.0)
    parser.add_argument("--frames", default="build/demo-frames")
    parser.add_argument("--stills", default="", help="directory for the three PNG stills")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(record(parse_arguments()))
