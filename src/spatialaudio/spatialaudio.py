import math
import time
import logging
import threading

import numpy as np
import soundfile as sf
import pyaudio
import numba as nb

__version__ = "0.1.0"

__all__ = [
    "AudioMixer",
    "AudioSource",
    "volume_falloff",
    "lowpass_sweep",
    "DEFAULT_VOLUME_CURVE",
    "DEFAULT_LOWPASS_CURVE",
]

logger = logging.getLogger(__name__)

_SQRT2_2 = math.sqrt(2.0) / 2.0


def volume_falloff(exponent=3.0):
    """Return a volume curve f(dist, max_dist) -> gain in [0, 1]."""
    def curve(dist, max_dist):
        if max_dist <= 0.0 or dist >= max_dist:
            return 0.0
        return max(0.0, 1.0 - (dist / max_dist) ** exponent)
    return curve


def lowpass_sweep(hi_hz=20000.0, lo_hz=100.0, start=1.0 / 3.0, end=2.0 / 3.0):
    """Return a cutoff curve f(dist, max_dist) -> cutoff Hz (<=0 bypasses)."""
    def curve(dist, max_dist):
        near = start * max_dist
        far = end * max_dist
        if dist <= near:
            return hi_hz
        if dist >= far or far <= near:
            return lo_hz
        t = (dist - near) / (far - near)
        return hi_hz * (lo_hz / hi_hz) ** t
    return curve


DEFAULT_VOLUME_CURVE = volume_falloff()
DEFAULT_LOWPASS_CURVE = lowpass_sweep()


@nb.njit(fastmath=True, cache=True)
def _render_chunk(data, position, frame_count, speed, loop, prev0, prev1, cutoff, fs):
    out = np.zeros((frame_count, 2), dtype=np.float32)
    length = data.shape[0]
    if length == 0:
        return out, np.float32(position), prev0, prev1

    if cutoff > 0.0:
        omega = np.float32(2.0 * np.pi * cutoff)
        alpha = omega / (omega + np.float32(fs))
    else:
        alpha = np.float32(0.0)

    pos = np.float32(position)
    speed = np.float32(speed)

    for i in range(frame_count):
        if pos >= length:
            if loop:
                pos = pos % length
            else:
                break

        i0 = int(pos)
        i1 = i0 + 1
        if i1 >= length:
            i1 = 0 if loop else length - 1
        frac = np.float32(pos - i0)

        s0 = data[i0]
        s1 = data[i1]
        a = s0[0] * (1.0 - frac) + s1[0] * frac
        b = s0[1] * (1.0 - frac) + s1[1] * frac

        if alpha > 0.0:
            prev0 = prev0 + alpha * (a - prev0)
            prev1 = prev1 + alpha * (b - prev1)
            a = prev0
            b = prev1

        out[i, 0] = a
        out[i, 1] = b
        pos += speed

    return out, pos, prev0, prev1


@nb.njit(fastmath=True, cache=True)
def _mix_chunk(mixed, chunk, volume, left_gain, right_gain):
    vlg = volume * left_gain
    vrg = volume * right_gain
    for i in range(chunk.shape[0]):
        mixed[i, 0] += chunk[i, 0] * vlg
        mixed[i, 1] += chunk[i, 1] * vrg
    return mixed


@nb.njit(fastmath=True, cache=True)
def _auto_gain(signal, target_level):
    peak = 0.0
    for i in range(signal.shape[0]):
        a0 = abs(signal[i, 0])
        a1 = abs(signal[i, 1])
        if a0 > peak:
            peak = a0
        if a1 > peak:
            peak = a1
    if peak > target_level and peak > 0.0:
        scale = target_level / peak
        for i in range(signal.shape[0]):
            signal[i, 0] *= scale
            signal[i, 1] *= scale
    return signal


@nb.njit(fastmath=True, cache=True)
def _resample_1ch(data, new_len):
    out = np.empty(new_len, dtype=data.dtype)
    old_len = data.shape[0]
    if old_len == 0:
        return out
    ratio = (old_len - 1) / (new_len - 1) if new_len > 1 else 1.0
    for i in range(new_len):
        pos = i * ratio
        i0 = int(pos)
        i1 = i0 + 1
        if i1 >= old_len:
            out[i] = data[old_len - 1]
        else:
            frac = pos - i0
            out[i] = data[i0] * (1.0 - frac) + data[i1] * frac
    return out


@nb.njit(fastmath=True, cache=True)
def _resample_2ch(data, new_len):
    out = np.empty((new_len, 2), dtype=data.dtype)
    old_len = data.shape[0]
    if old_len == 0:
        return out
    ratio = (old_len - 1) / (new_len - 1) if new_len > 1 else 1.0
    for i in range(new_len):
        pos = i * ratio
        i0 = int(pos)
        i1 = i0 + 1
        if i1 >= old_len:
            out[i, 0] = data[old_len - 1, 0]
            out[i, 1] = data[old_len - 1, 1]
        else:
            frac = pos - i0
            out[i, 0] = data[i0, 0] * (1.0 - frac) + data[i1, 0] * frac
            out[i, 1] = data[i0, 1] * (1.0 - frac) + data[i1, 1] * frac
    return out


def _resample_if_needed(data, original_rate, target_rate):
    if original_rate == target_rate:
        return data.astype(np.float32, copy=True)
    ratio = target_rate / original_rate
    new_length = max(1, int(round(len(data) * ratio)))
    data32 = data.astype(np.float32)
    if data.ndim == 1:
        return _resample_1ch(data32, new_length)
    return _resample_2ch(np.ascontiguousarray(data32[:, :2]), new_length)


def _as_xy(pos):
    x, y = pos
    return float(x), float(y)


class AudioSource:
    """A single playable stereo buffer with per-source playback state."""

    def __init__(self, data, sample_rate, mixer_sample_rate, *, pitch=1.0):
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        data = _resample_if_needed(data, sample_rate, mixer_sample_rate)

        if data.ndim == 1:
            data = np.column_stack([data, data])
        elif data.ndim == 2 and data.shape[1] == 1:
            data = np.column_stack([data[:, 0], data[:, 0]])
        elif data.ndim == 2 and data.shape[1] >= 2:
            data = data[:, :2]
        self.data = np.ascontiguousarray(data, dtype=np.float32)

        self.length = self.data.shape[0]
        self.position = np.float32(0.0)
        self.volume = 1.0
        self.pan = 0.0
        self.left_gain = np.float32(_SQRT2_2)
        self.right_gain = np.float32(_SQRT2_2)
        self.active = True
        self.loop = False
        self.pitch = float(pitch)

        self.cutoff = 0.0
        self.prev0 = np.float32(0.0)
        self.prev1 = np.float32(0.0)

        self.positional = False
        self.pos = None
        self.max_dist = 6000.0
        self.base_volume = 1.0
        self.pan_width = 2000.0
        self.volume_curve = DEFAULT_VOLUME_CURVE
        self.lowpass_curve = DEFAULT_LOWPASS_CURVE

    def set_pan(self, pan):
        self.pan = float(min(1.0, max(-1.0, pan)))
        self.left_gain = np.float32((1.0 - max(0.0, self.pan)) * _SQRT2_2)
        self.right_gain = np.float32((1.0 + min(0.0, self.pan)) * _SQRT2_2)

    def set_lowpass(self, cutoff_hz):
        self.cutoff = float(cutoff_hz) if cutoff_hz else 0.0

    def get_next_chunk(self, frame_count, fs, speed):
        chunk, pos, self.prev0, self.prev1 = _render_chunk(
            self.data, self.position, frame_count,
            np.float32(speed * self.pitch), self.loop,
            self.prev0, self.prev1, np.float32(self.cutoff), np.float32(fs),
        )
        self.position = pos
        if not self.loop and self.position >= self.length:
            self.active = False
        return chunk


class AudioMixer:
    """Real-time stereo mixer with optional 2D positional attenuation, panning
    and distance low-pass. Push listener position every frame with
    set_listener / set_listeners."""

    def __init__(self, sample_rate=44100, chunk_size=1024,
                 master_volume=1.0, limiter_target=0.7):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.master_volume = float(master_volume)
        self.limiter_target = limiter_target
        self.time_scale = 1.0

        self._sources = []
        self._listeners = [(0.0, 0.0)]
        self._cache = {}
        self._lock = threading.Lock()
        self._pa = None
        self._stream = None
        self.callback_time = 0.0

        self._prime()

    def _prime(self):
        dummy = np.zeros((max(8, self.chunk_size), 2), dtype=np.float32)
        dummy_1 = np.zeros(dummy.shape[0], dtype=np.float32)
        _render_chunk(dummy, np.float32(0.0), dummy.shape[0], np.float32(1.0),
                      False, np.float32(0.0), np.float32(0.0),
                      np.float32(0.0), np.float32(self.sample_rate))
        _mix_chunk(dummy.copy(), dummy, np.float32(1.0), np.float32(1.0), np.float32(1.0))
        _auto_gain(dummy.copy(), np.float32(0.7))
        _resample_1ch(dummy_1, 16)
        _resample_2ch(dummy, 16)

    def set_listener(self, pos):
        """Set a single listener position (any length-2 iterable)."""
        self._listeners = [_as_xy(pos)]

    def set_listeners(self, positions):
        """Set multiple listener positions; nearest one is used per source."""
        self._listeners = [_as_xy(p) for p in positions]
        if not self._listeners:
            self._listeners = [(0.0, 0.0)]

    def load(self, filename):
        """Load (and cache) a file into a new AudioSource."""
        if filename in self._cache:
            data, rate = self._cache[filename]
        else:
            data, rate = sf.read(filename, dtype="float32")
            self._cache[filename] = (data, rate)
        return AudioSource(data, rate, self.sample_rate)

    def _resolve(self, audio):
        return audio if isinstance(audio, AudioSource) else self.load(audio)

    def play(self, audio, *, volume=1.0, pan=0.0, loop=False, pitch=1.0):
        """Play a non-positional source."""
        source = self._resolve(audio)
        source.positional = False
        source.volume = float(volume)
        source.pitch = float(pitch)
        source.set_pan(pan)
        source.loop = bool(loop)
        source.position = np.float32(0.0)
        source.active = True
        with self._lock:
            self._sources.append(source)
        return source

    def play_positional(self, audio, pos, *, max_dist=6000.0, volume=1.0,
                        loop=False, pitch=1.0, pan_width=2000.0,
                        volume_curve=DEFAULT_VOLUME_CURVE,
                        lowpass_curve=DEFAULT_LOWPASS_CURVE):
        """Play a source whose gain, pan and low-pass track the nearest listener."""
        source = self._resolve(audio)
        source.positional = True
        source.pos = _as_xy(pos)
        source.max_dist = float(max_dist)
        source.base_volume = float(volume) * self.master_volume
        source.pan_width = float(pan_width)
        source.volume_curve = volume_curve
        source.lowpass_curve = lowpass_curve
        source.pitch = float(pitch)
        source.loop = bool(loop)
        source.position = np.float32(0.0)
        source.active = True
        with self._lock:
            self._sources.append(source)
        return source

    def stop(self, source):
        source.active = False
        with self._lock:
            try:
                self._sources.remove(source)
            except ValueError:
                pass

    def stop_all(self):
        with self._lock:
            self._sources.clear()

    def start(self):
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paFloat32, channels=2, rate=self.sample_rate,
            output=True, frames_per_buffer=self.chunk_size,
            stream_callback=self._callback,
        )
        self._stream.start_stream()

    def set_chunk_size(self, new_chunk_size):
        logger.info("chunk size %d -> %d", self.chunk_size, new_chunk_size)
        self.chunk_size = new_chunk_size
        if self._stream is not None:
            self._stream.close()
            self._stream = self._pa.open(
                format=pyaudio.paFloat32, channels=2, rate=self.sample_rate,
                output=True, frames_per_buffer=self.chunk_size,
                stream_callback=self._callback,
            )
            self._stream.start_stream()

    def stop_stream(self):
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    def _nearest_listener(self, x, y):
        best_x, best_y, best_d = 0.0, 0.0, float("inf")
        for lx, ly in self._listeners:
            d = math.hypot(x - lx, y - ly)
            if d < best_d:
                best_x, best_y, best_d = lx, ly, d
        return best_x, best_y, best_d

    def _callback(self, in_data, frame_count, time_info, status):
        t = time.perf_counter()
        mixed = np.zeros((frame_count, 2), dtype=np.float32)

        with self._lock:
            sources = list(self._sources)

        speed = self.time_scale
        finished = []

        for source in sources:
            if not source.active:
                finished.append(source)
                continue

            if source.positional and source.pos is not None:
                sx, sy = source.pos
                lx, ly, dist = self._nearest_listener(sx, sy)
                source.volume = source.base_volume * source.volume_curve(dist, source.max_dist)
                source.set_pan(max(-1.0, min(1.0, (sx - lx) / source.pan_width)))
                if source.lowpass_curve is not None:
                    source.set_lowpass(source.lowpass_curve(dist, source.max_dist))
                else:
                    source.cutoff = 0.0

            chunk = source.get_next_chunk(frame_count, self.sample_rate, speed)
            _mix_chunk(mixed, chunk,
                       np.float32(source.volume * self.master_volume),
                       source.left_gain, source.right_gain)

            if not source.active:
                finished.append(source)

        if self.limiter_target is not None:
            _auto_gain(mixed, np.float32(self.limiter_target))

        if finished:
            with self._lock:
                for source in finished:
                    try:
                        self._sources.remove(source)
                    except ValueError:
                        pass

        self.callback_time = 0.01 * (time.perf_counter() - t) + 0.99 * self.callback_time
        return (mixed.tobytes(), pyaudio.paContinue)


if __name__ == "__main__":
    mixer = AudioMixer()
    mixer.start()
    mixer.set_listener((0.0, 0.0))
    

    for x in range(0, 3000, 100):
        mixer.play_positional("audio/explosion1.wav", (0.0, 0.0), max_dist=3000.0)
        mixer.set_listener((float(x), 0.0))
        time.sleep(2)
        print(mixer.callback_time)

    time.sleep(2.0)
    mixer.stop_stream()