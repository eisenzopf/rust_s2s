#!/usr/bin/env python3
"""Scripted s2s-server client: stream a WAV in real time, record the reply.

Connects to an s2s-server WebSocket (e.g. `s2s-kyutai-server`), starts a
session, sends the WAV (mono 24 kHz, 16-bit) in 80 ms frames on the real-time
clock, followed by `--tail` seconds of silence, and writes every audio frame
the server sends back to OUT.wav. Prints the transcript deltas, output frame
timing and level, and exits non-zero if the session failed or the reply was
silent or badly paced.

    pip install websockets numpy
    scripts/s2s-client.py question.wav reply.wav [--url ws://127.0.0.1:8080/ws] [--tail 8]
"""
import argparse
import asyncio
import json
import sys
import time
import wave

import numpy as np
import websockets

FRAME = 1920          # 80 ms at 24 kHz
TAG_S16 = 0x02        # binary audio frame tag (protocol.rs)


def read_wav(path):
    with wave.open(path) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (24000, 1, 2):
            sys.exit(f"{path}: need mono 24 kHz 16-bit PCM (e.g. sox in.wav -r 24000 -c 1 -b 16 out.wav)")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


async def run(args):
    pcm = np.concatenate([read_wav(args.input), np.zeros(int(args.tail * 24000), dtype="<i2")])
    n = len(pcm) // FRAME
    out, text, arrivals, errors = [], {}, [], []
    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.start", "mode": args.mode, "system_prompt": "",
                                  "tools": [], "output_format": "pcm_s16le"}))
        started = json.loads(await ws.recv())
        if started.get("type") != "session.started":
            sys.exit(f"session did not start: {started}")
        t0 = time.monotonic()

        async def receive():
            async for m in ws:
                if isinstance(m, bytes):
                    arrivals.append(time.monotonic() - t0)
                    out.append(np.frombuffer(m[1:], dtype="<i2"))
                    continue
                msg = json.loads(m)
                kind = msg.get("type", "")
                if kind.startswith("transcript.") and kind.endswith(".delta"):
                    who = kind.split(".")[1]
                    text[who] = text.get(who, "") + msg["text"]
                elif kind == "error":
                    errors.append(msg)

        rx = asyncio.create_task(receive())
        for i in range(n):
            await ws.send(bytes([TAG_S16]) + pcm[i * FRAME:(i + 1) * FRAME].tobytes())
            await asyncio.sleep(max(0.0, t0 + (i + 1) * 0.08 - time.monotonic()))
        await asyncio.sleep(0.5)
        await ws.send(json.dumps({"type": "session.end"}))
        try:
            await asyncio.wait_for(rx, 3)
        except (asyncio.TimeoutError, websockets.ConnectionClosed):
            rx.cancel()

    audio = np.concatenate(out) if out else np.zeros(0, dtype="<i2")
    with wave.open(args.output, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(audio.tobytes())
    gaps = np.diff(arrivals) * 1000 if len(arrivals) > 1 else np.array([0.0])
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)) / 32768) if len(audio) else 0.0
    print(f"sent {n} frames ({n * 0.08:.1f} s); received {len(out)} ({len(audio) / 24000:.1f} s) -> {args.output}")
    print(f"output frame gaps: median {np.median(gaps):.1f} ms, p95 {np.percentile(gaps, 95):.1f} ms, max {gaps.max():.1f} ms")
    print(f"output RMS {rms:.4f}")
    for who, s in text.items():
        print(f"{who} text: {s.strip()}")
    problems = [f"server error: {e}" for e in errors]
    if len(out) < 0.9 * n:
        problems.append(f"only {len(out)} of {n} frames came back")
    if args.mode == "s2s" and rms < 1e-3:
        problems.append("reply is silent")
    if np.percentile(gaps, 95) > 120:
        problems.append("output pacing p95 above 120 ms (server not keeping real time?)")
    for p in problems:
        print("FAIL:", p)
    return 1 if problems else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", help="mono 24 kHz 16-bit WAV to send")
    ap.add_argument("output", help="WAV to write the reply audio to")
    ap.add_argument("--url", default="ws://127.0.0.1:8080/ws")
    ap.add_argument("--mode", default="s2s", choices=["s2s", "stt"])
    ap.add_argument("--tail", type=float, default=8.0, help="seconds of silence after the input")
    sys.exit(asyncio.run(run(ap.parse_args())))


if __name__ == "__main__":
    main()
