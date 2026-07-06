# spatial-audio

Real-time 2D positional stereo mixer for Python games. Can handle 50-100 simultanous sounds without jittering, lowpass filtering and volume falloff based on distance, pitch and timescale shifting for slow-mo and panning based on direction. Pure Python + numba; no compiled extension of its own.

## Install

```bash
pip install spatial-audio
```

System libraries required by the audio backends (not pip-installable):

- **PortAudio** — backs PyAudio. Debian/Ubuntu: `apt install portaudio19-dev`; macOS: `brew install portaudio`. Windows wheels bundle it.
- **libsndfile** — backs soundfile; bundled in the `soundfile>=0.12` wheels on all three platforms.

## Quickstart

```python
import time
from spatialaudio import AudioMixer

mixer = AudioMixer()
mixer.start()

mixer.play("audio/ui_click.wav")

src = mixer.play_positional("audio/explosion.wav", (500.0, 0.0), max_dist=3000.0)

for x in range(0, 3000, 100):
    mixer.set_listener((float(x), 0.0))
    time.sleep(0.05)

time.sleep(2.0)
mixer.stop_stream()
```

`set_listener(pos)` is push-based: call it every frame with the current camera/listener center. Positions are any length-2 iterable — tuple, list, ndarray, `pygame.Vector2` all unpack.

## Lazy loading

Files are not preloaded. The first `play(path)` / `play_positional(path, ...)` / `load(path)` decodes the file from disk once and caches the decoded PCM array in memory, keyed by path. Every later play of the same path skips disk I/O and decoding — it instantiates a fresh `AudioSource` from the cached buffer, so concurrent overlapping plays of one file are independent and cheap.

Consequences:

- No warm-up call needed; first play of each asset pays a one-time decode cost (audible as a hitch only for large files on the audio thread's first touch — decode happens on your calling thread, not the callback, so it does not glitch playback).
- The cache is unbounded and never evicted. For a fixed asset set this is ideal; if you stream thousands of distinct one-shot files, memory grows monotonically.
- Pass an `AudioSource` instead of a path to bypass the cache entirely (e.g. procedurally generated buffers).

## Positional model

2D only. Per source, per audio callback, against the **nearest** listener:

- **Volume** = `base_volume × volume_curve(dist, max_dist)`. Default `volume_falloff(3.0)` → `1 − (d/max)³`, clamped to 0 at `d ≥ max_dist`.
- **Low-pass** cutoff = `lowpass_curve(dist, max_dist)`. Default `lowpass_sweep()` holds 20 kHz to `max_dist/3`, log-sweeps to 100 Hz by `2·max_dist/3`, flat beyond. Pass `lowpass_curve=None` to disable.
- **Pan** = `clip(lateral_offset / pan_width, −1, 1)`, constant-power. Larger `pan_width` = less aggressive stereo spread; raise it for loud/wide sources.

Curves are plain callables `f(dist, max_dist) -> float`, evaluated in Python outside the JIT loop, so custom curves cost nothing structurally:

```python
from spatialaudio import volume_falloff, lowpass_sweep

mixer.play_positional(
    "audio/engine_hum.wav", (0.0, 0.0),
    max_dist=8000.0,
    loop=True,
    pan_width=4000.0,
    volume_curve=volume_falloff(1.5),
    lowpass_curve=lowpass_sweep(hi_hz=18000, lo_hz=300),
)
```

## Split-screen

Pass every camera center; each source attenuates/pans against whichever listener is closest:

```python
mixer.set_listeners([cam_a_center, cam_b_center])
```

## Global controls

- `mixer.time_scale` — playback-rate multiplier applied to all sources (`0.3` = slow-mo, `1.0` = normal). Set per frame.
- `mixer.master_volume` — linear gain into the mix.
- `mixer.limiter_target` — peak ceiling for the output limiter (default `0.7`); set `None` to disable.
- Per-source `pitch` — playback-rate multiplier stacked on `time_scale`. A random value between 0.9-1.1 produces a nice effect.

## API

| Call | Effect |
|---|---|
| `AudioMixer(sample_rate=44100, chunk_size=1024, master_volume=1.0, limiter_target=0.7)` | Construct; primes JIT kernels. |
| `start()` / `stop_stream()` | Open / close the PortAudio stream. |
| `set_listener(pos)` / `set_listeners(list)` | Set listener(s); call every frame. |
| `play(audio, *, volume=1, pan=0, loop=False, pitch=1)` | Non-positional playback. |
| `play_positional(audio, pos, *, max_dist=6000, volume=1, loop=False, pitch=1, pan_width=2000, volume_curve=…, lowpass_curve=…)` | Positional playback. |
| `load(path) -> AudioSource` | Decode + cache without playing. |
| `stop(source)` / `stop_all()` | Remove source(s) from the mix. |
| `set_chunk_size(n)` | Reopen the stream with a new buffer size. |

`audio` is a path (cached) or an `AudioSource` (used directly). All play methods return the `AudioSource`.

## Threading

Playback runs in the PortAudio callback thread. Your game thread calls `set_listener`, `play*`, `stop*` freely; source-list mutations are lock-guarded with O(1) critical sections, so the callback never blocks meaningfully.

## Requirements

Python ≥ 3.10 · numpy ≥ 1.24 · numba ≥ 0.60 · soundfile ≥ 0.12 · PyAudio ≥ 0.2.12

numba is the binding constraint on the numpy version; it caps numpy via its own metadata, so no numpy upper pin is declared here.

## License

MIT