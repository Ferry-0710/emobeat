"""
gen_music.py — 本地曲库生成器 (免版权, 纯数字合成)
输出: assets/music/*.wav  +  js/data/library.json (含节奏/调性/标注情绪 ground truth)

合成思路:
  1. 事件表 (bar, beat, midi, dur, instrument, velocity)
  2. 加法合成音色 (谐波叠加 + ADSR + 一阶低通)
  3. 鼓组: kick 正弦扫频, snare 噪声+音调, hat 高通噪声
  4. 并联梳状混响 + tanh 软限幅 + 归一化
"""
import json, math, os
import numpy as np

SR = 22050
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "music")
LIB_JSON = os.path.join(os.path.dirname(__file__), "..", "js", "data", "library.json")

MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def env(n, a, d, s_level, r, sr=SR):
    """线性 ADSR 包络, n = 采样点数"""
    a_n = max(1, int(a * sr))
    d_n = max(1, int(d * sr))
    r_n = max(1, int(r * sr))
    s_n = max(0, n - a_n - d_n - r_n)
    out = np.zeros(n, dtype=np.float64)
    i = 0
    out[i:i + a_n] = np.linspace(0, 1, a_n)
    i += a_n
    out[i:i + d_n] = np.linspace(1, s_level, d_n)
    i += d_n
    out[i:i + s_n] = s_level
    i += s_n
    out[i:i + r_n] = np.linspace(s_level, 0, r_n)
    return out


def one_pole_lp(x, cutoff):
    """一阶 IIR 低通"""
    if cutoff >= SR / 2.5:
        return x
    a = math.exp(-2 * math.pi * cutoff / SR)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = (1 - a) * x[i] + a * acc
        y[i] = acc
    return y


def fast_lp(x, cutoff):
    """用 FFT 掩蔽实现的低通 (比逐点 IIR 快很多, 适合批量渲染)"""
    n = len(x)
    if n < 8:
        return x
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    # 二阶 Butterworth 幅度响应
    resp = 1.0 / np.sqrt(1.0 + (freqs / max(cutoff, 20.0)) ** 4)
    return np.fft.irfft(X * resp, n)


def synth_pluck(midi, dur, vel=1.0, bright=1.0):
    """拨弦/钢琴: 谐波指数衰减 + 高次谐波衰减更快"""
    n = int(dur * SR)
    if n <= 0:
        return np.zeros(0)
    f = mtof(midi)
    t = np.arange(n) / SR
    out = np.zeros(n)
    amps = [1.0, 0.55, 0.30, 0.18, 0.10, 0.06, 0.04]
    for k, a in enumerate(amps, start=1):
        if f * k > SR / 2.2:
            break
        tau = 0.85 / (k ** 0.55)
        out += a * np.exp(-t / tau) * np.sin(2 * np.pi * f * k * t + 0.4 * k)
    out *= env(n, 0.004, 0.02, 0.55, 0.16)
    out = fast_lp(out, 3200 * bright + 900)
    return out * vel * 0.35


def synth_pad(midi, dur, vel=1.0, vibrato=0.0, tremolo=0.0, cutoff=1800):
    """铺垫/弦乐: 多谐波锯齿堆叠, 慢起慢落"""
    n = int(dur * SR)
    if n <= 0:
        return np.zeros(0)
    t = np.arange(n) / SR
    f = mtof(midi)
    vib = 1.0 + vibrato * np.sin(2 * np.pi * 5.2 * t + midi)
    out = np.zeros(n)
    for det in (-0.006, 0.0, 0.007):
        ff = f * (1 + det) * vib
        for k in range(1, 11):
            if f * k > SR / 2.2:
                break
            out += (1.0 / k) * np.sin(2 * np.pi * ff * k * t + k * 0.7)
    out /= 3.0
    if tremolo > 0:
        out *= 1.0 - tremolo * 0.5 * (1 + np.sin(2 * np.pi * 7.5 * t))
    out *= env(n, min(0.5, dur * 0.3), 0.15, 0.8, min(0.6, dur * 0.35))
    out = fast_lp(out, cutoff)
    return out * vel * 0.16


def synth_bass(midi, dur, vel=1.0, drive=1.0):
    n = int(dur * SR)
    if n <= 0:
        return np.zeros(0)
    t = np.arange(n) / SR
    f = mtof(midi)
    out = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t) + 0.12 * np.sin(2 * np.pi * 3 * f * t)
    out *= np.exp(-t / 0.55)
    out *= env(n, 0.006, 0.05, 0.7, 0.08)
    out = np.tanh(out * 1.4 * drive)
    out = fast_lp(out, 700)
    return out * vel * 0.42


def synth_lead(midi, dur, vel=1.0, bright=1.0):
    n = int(dur * SR)
    if n <= 0:
        return np.zeros(0)
    t = np.arange(n) / SR
    f = mtof(midi)
    vib = 1.0 + 0.004 * np.sin(2 * np.pi * 5.8 * t)
    out = np.zeros(n)
    for k in range(1, 13):
        if f * k > SR / 2.2:
            break
        out += (1.0 / k) * np.sin(2 * np.pi * f * vib * k * t)
    out *= env(n, 0.02, 0.06, 0.6, 0.1)
    out = fast_lp(out, 2400 * bright + 800)
    return out * vel * 0.20


def synth_power(midi, dur, vel=1.0):
    """失真五度强力和弦 (angry)"""
    n = int(dur * SR)
    if n <= 0:
        return np.zeros(0)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for iv in (0, 7, 12):
        f = mtof(midi + iv)
        for k in range(1, 14):
            if f * k > SR / 2.2:
                break
            out += (1.0 / k) * np.sin(2 * np.pi * f * k * t + 0.3 * k)
    out = np.tanh(out * 3.2)
    out *= env(n, 0.004, 0.05, 0.75, 0.06)
    out = fast_lp(out, 4200)
    return out * vel * 0.20


def synth_bell(midi, dur, vel=1.0):
    """FM 钟琴 (surprised 用)"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = mtof(midi)
    mod = 2.2 * np.sin(2 * np.pi * f * 3.01 * t) * np.exp(-t / 0.25)
    out = np.sin(2 * np.pi * f * t + mod) * np.exp(-t / 0.45)
    out *= env(n, 0.002, 0.01, 0.5, 0.2)
    return out * vel * 0.26


def synth_kick(vel=1.0, dur=0.42, drive=1.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 105 * np.exp(-t / 0.055) + 42
    out = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.16)
    click = np.random.RandomState(7).randn(n) * np.exp(-t / 0.004) * 0.6
    out = np.tanh((out + click) * 1.8 * drive)
    return out * vel * 0.62


def synth_snare(vel=1.0, dur=0.22):
    n = int(dur * SR)
    t = np.arange(n) / SR
    rs = np.random.RandomState(11)
    noise = rs.randn(n) * np.exp(-t / 0.075)
    noise = noise - fast_lp(noise, 900)  # 高通近似
    tone = 0.4 * np.sin(2 * np.pi * 186 * t) * np.exp(-t / 0.05)
    return (noise * 0.75 + tone) * vel * 0.34


def synth_hat(vel=1.0, dur=0.06):
    n = int(dur * SR)
    t = np.arange(n) / SR
    rs = np.random.RandomState(3)
    noise = rs.randn(n) * np.exp(-t / 0.012)
    hp = noise - fast_lp(noise, 4200)
    return hp * vel * 0.10


def synth_tom(vel=1.0, dur=0.30):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 170 * np.exp(-t / 0.09) + 70
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.13) * vel * 0.32


def add(buf, x, start):
    start = int(start)
    if start < 0 or start >= len(buf):
        return
    end = min(len(buf), start + len(x))
    buf[start:end] += x[: end - start]


def reverb(x, wet=0.22):
    """并联梳状 + 全通近似简易混响"""
    delays = [int(0.0297 * SR), int(0.0371 * SR), int(0.0411 * SR), int(0.0437 * SR)]
    gains = [0.62, 0.56, 0.52, 0.48]
    out = np.zeros(len(x) + 12000)
    out[: len(x)] = x
    for d, g in zip(delays, gains):
        y = np.copy(x)
        for _ in range(6):
            y_next = np.zeros_like(y)
            y_next[d:] = y[:-d] * g
            y = fast_lp(y + y_next, 3200)
        out[: len(x)] += y * 0.5
    out = out[: len(x)]
    return (1 - wet) * x + wet * out


def scale_of(root, mode):
    ivs = MAJOR if mode == "major" else MINOR
    return [root + i for i in ivs]


def triad(root, mode, degree, octave_shift=0, seventh=False):
    """返回三和弦 (midi 列表), degree 0..6"""
    ivs = MAJOR if mode == "major" else MINOR
    notes = []
    for k in (0, 2, 4):
        idx = degree + k
        notes.append(root + ivs[idx % 7] + 12 * (idx // 7) + 12 * 2 + 12 * octave_shift)
    if seventh:
        idx = degree + 6
        notes.append(root + ivs[idx % 7] + 12 * (idx // 7) + 12 * 2 + 12 * octave_shift)
    return notes


def render(track):
    """把事件表渲染成波形"""
    bpm = track["bpm"]
    spb = 60.0 / bpm
    bars = track["bars"]
    total = bars * 4 * spb + 3.0
    buf = np.zeros(int(total * SR))
    rnd = np.random.RandomState(track["seed"])
    root, mode = track["root"], track["mode"]
    sc = scale_of(root, mode)

    for (bar, beat, midi, dur_beats, inst, vel) in track["events"]:
        st = bar * 4 * spb + beat * spb
        d = dur_beats * spb
        if inst == "pluck":
            x = synth_pluck(midi, d + 0.4, vel, track.get("bright", 1.0))
        elif inst == "pad":
            x = synth_pad(midi, d + 0.2, vel, track.get("vibrato", 0.0),
                          track.get("tremolo", 0.0), track.get("cutoff", 1900))
        elif inst == "bass":
            x = synth_bass(midi, d, vel, track.get("drive", 1.0))
        elif inst == "lead":
            x = synth_lead(midi, d, vel, track.get("bright", 1.0))
        elif inst == "power":
            x = synth_power(midi, d, vel)
        elif inst == "bell":
            x = synth_bell(midi, d + 0.3, vel)
        elif inst == "kick":
            x = synth_kick(vel, 0.42, track.get("drive", 1.0))
        elif inst == "snare":
            x = synth_snare(vel)
        elif inst == "hat":
            x = synth_hat(vel)
        elif inst == "tom":
            x = synth_tom(vel)
        else:
            continue
        add(buf, x, st * SR)

    buf = reverb(buf, track.get("wet", 0.2))
    # 淡入淡出
    fi = int(0.02 * SR)
    fo = int(1.2 * SR)
    buf[:fi] *= np.linspace(0, 1, fi)
    buf[-fo:] *= np.linspace(1, 0, fo)
    buf = np.tanh(buf * 1.1)
    peak = np.max(np.abs(buf)) or 1.0
    buf = buf / peak * 0.92
    return buf


def wav_write(path, x, sr=SR):
    x16 = np.clip(x, -1, 1)
    x16 = (x16 * 32767).astype("<i2")
    data = x16.tobytes()
    n = len(x16)
    with open(path, "wb") as f:
        f.write(b"RIFF")
        f.write((36 + len(data)).to_bytes(4, "little"))
        f.write(b"WAVEfmt ")
        f.write((16).to_bytes(4, "little"))
        f.write((1).to_bytes(2, "little"))
        f.write((1).to_bytes(2, "little"))
        f.write(sr.to_bytes(4, "little"))
        f.write((sr * 2).to_bytes(4, "little"))
        f.write((2).to_bytes(2, "little"))
        f.write((16).to_bytes(2, "little"))
        f.write(b"data")
        f.write(len(data).to_bytes(4, "little"))
        f.write(data)


# ---------------------------------------------------------------- 编曲工具
def bar_ev(ev, bar, beat, midi, dur, inst, vel=1.0):
    ev.append((bar, beat, midi, dur, inst, vel))


def arp_bar(ev, bar, notes, pattern, inst, vel=1.0, beats=4, octave=0):
    """pattern: 每拍细分次数"""
    step = beats / len(pattern)
    for i, idx in enumerate(pattern):
        bar_ev(ev, bar, i * step, notes[idx % len(notes)] + 12 * octave, step * 0.95, inst, vel)


def build_track(spec):
    """依据风格模板生成事件"""
    ev = []
    root, mode = spec["root"], spec["mode"]
    bpm = spec["bpm"]
    prog = spec["prog"]
    style = spec["style"]
    bars = spec["bars"]
    rnd = np.random.RandomState(spec["seed"])
    sc = scale_of(root, mode)

    for bar in range(bars):
        deg = prog[bar % len(prog)]
        ch = triad(root, mode, deg)
        section = "A" if bar < bars * 0.55 else "B"

        # ---- 和声
        if style in ("ballad", "ambient"):
            for m in ch[1:]:
                bar_ev(ev, bar, 0, m, 4.0, "pad", 0.9)
            bar_ev(ev, bar, 0, ch[0] + 12, 4.0, "pad", 0.75)
            for i, idx in enumerate([0, 2, 4, 2]):
                bar_ev(ev, bar, i, ch[idx % 3], 1.0, "pluck", 0.62)
            if section == "B" and bar % 2 == 0:
                mel = sc[(deg + 4 + rnd.randint(0, 3)) % 7] + 24
                bar_ev(ev, bar, 2, mel, 2.0, "pluck", 0.7)
        elif style == "bright":
            for m in ch:
                bar_ev(ev, bar, 0, m, 1.9, "pluck", 0.62)
                bar_ev(ev, bar, 2, m, 1.9, "pluck", 0.55)
            arp_bar(ev, bar, [ch[0] + 12, ch[1] + 12, ch[2] + 12, ch[0] + 24],
                    [0, 1, 2, 3, 2, 1, 2, 3], "pluck", 0.5, 4)
            # 主旋律
            for i in range(2):
                n = sc[(deg + 2 + rnd.randint(0, 4)) % 7] + 24
                bar_ev(ev, bar, i * 2, n, 1.4, "lead", 0.62)
        elif style == "rock":
            for m in ch:
                bar_ev(ev, bar, 0, m - 12, 1.9, "power", 0.85)
                bar_ev(ev, bar, 2, m - 12, 1.9, "power", 0.8)
            if section == "B":
                for m in ch:
                    bar_ev(ev, bar, 1, m - 12, 0.9, "power", 0.7)
                    bar_ev(ev, bar, 3, m - 12, 0.9, "power", 0.7)
        elif style == "tense":
            for m in ch[1:]:
                bar_ev(ev, bar, 0, m + 6, 4.0, "pad", 0.8)
            bar_ev(ev, bar, 0, ch[0] - 12, 4.0, "pad", 0.85)
            # 不协和半音簇 + 颤音
            for i in range(4):
                bar_ev(ev, bar, i, ch[0] + 12, 0.9, "pad", 0.4)
                bar_ev(ev, bar, i, ch[0] + 13, 0.9, "pad", 0.35)
        elif style == "odd":
            # 半音摩擦 (disgust / 怪异)
            # 注意: 低音必须稳定在主音上, 否则色度被摊平, 调性检测会退化成大调
            for i in range(4):
                bar_ev(ev, bar, i, ch[0] - 12, 0.9, "bass", 0.85)
            for m in ch:
                bar_ev(ev, bar, 0, m, 1.5, "power", 0.45)
            # 小二度摩擦保留原强度（维持粗糙度 → disgusted），只把低音固定到主音
            bar_ev(ev, bar, 2, m + 1, 1.5, "power", 0.38)
            if section == "B":
                bar_ev(ev, bar, 0, ch[0] + 19, 0.5, "bell", 0.5)
        elif style == "surprise":
            for i in range(3):
                if rnd.rand() < 0.7:
                    bar_ev(ev, bar, i * 1.5, ch[rnd.randint(0, 3)] + 12, 0.4, "bell", 0.85)
            for m in ch:
                bar_ev(ev, bar, 0, m, 0.5, "pluck", 0.5)
            if bar % 2 == 1:
                bar_ev(ev, bar, 3.5, ch[0] + 24, 0.5, "bell", 0.9)

        # ---- 低音
        if style in ("ballad", "ambient"):
            bar_ev(ev, bar, 0, ch[0] - 24, 4.0, "bass", 0.75)
        elif style == "bright":
            for i in range(4):
                bar_ev(ev, bar, i, ch[0] - 24 + (0 if i % 2 == 0 else 7), 0.9, "bass", 0.8)
        elif style == "rock":
            for i in range(8):
                bar_ev(ev, bar, i * 0.5, ch[0] - 24, 0.45, "bass", 0.85)
        elif style == "tense":
            for i in range(2):
                bar_ev(ev, bar, i * 2, ch[0] - 24, 1.8, "bass", 0.7)
        elif style == "odd":
            pass
        elif style == "surprise":
            for i in range(4):
                bar_ev(ev, bar, i, ch[0] - 24, 0.6, "bass", 0.8)

        # ---- 鼓
        if style == "rock":
            for i in range(4):
                bar_ev(ev, bar, i, 0, 0.2, "kick", 0.95)
            for i in (1, 3):
                bar_ev(ev, bar, i, 0, 0.2, "snare", 0.9)
            for i in range(8):
                bar_ev(ev, bar, i * 0.5, 0, 0.1, "hat", 0.55)
        elif style == "bright":
            for i in (0, 2):
                bar_ev(ev, bar, i, 0, 0.2, "kick", 0.9)
            for i in (1, 3):
                bar_ev(ev, bar, i, 0, 0.2, "snare", 0.7)
            for i in range(8):
                bar_ev(ev, bar, i * 0.5, 0, 0.1, "hat", 0.5)
        elif style == "surprise":
            for i in range(4):
                bar_ev(ev, bar, i, 0, 0.2, "kick", 0.8)
            if bar % 2 == 1:
                bar_ev(ev, bar, 2, 0, 0.2, "tom", 0.9)
                bar_ev(ev, bar, 3, 0, 0.2, "snare", 0.85)
            for i in range(4):
                bar_ev(ev, bar, i * 1.0, 0, 0.1, "hat", 0.45)
        elif style == "tense":
            for i in range(2):
                bar_ev(ev, bar, i * 2 + 0.5, 0, 0.2, "kick", 0.6)
            for i in range(4):
                bar_ev(ev, bar, i + 0.75, 0, 0.1, "snare", 0.35)
        elif style == "odd":
            for i in range(4):
                bar_ev(ev, bar, i, 0, 0.2, "kick", 0.7)
            for i in range(4):
                bar_ev(ev, bar, i * 1, 0, 0.1, "hat", 0.4)
        elif style in ("ballad", "ambient"):
            if style == "ballad" and bar % 2 == 0:
                bar_ev(ev, bar, 0, 0, 0.2, "kick", 0.55)

    spec["events"] = ev
    return spec


TRACKS = [
    # ---------- 高兴 happy
    dict(id="happy_sunlit", title="Sunlit Avenue", emotion="happy", bpm=126, root=57, mode="major",
         prog=[0, 4, 5, 0], style="bright", bars=16, seed=101, bright=1.15, wet=0.22,
         desc="明亮大调 + 四四拍律动, 高频密集"),
    dict(id="happy_bubblegum", title="Bubblegum Sprint", emotion="happy", bpm=142, root=62, mode="major",
         prog=[0, 3, 4, 0], style="bright", bars=16, seed=202, bright=1.3, wet=0.18,
         desc="高速琶音, 短促断奏"),
    # ---------- 悲伤 sad
    dict(id="sad_rain", title="Rain on Glass", emotion="sad", bpm=66, root=57, mode="minor",
         prog=[0, 3, 6, 0], style="ballad", bars=12, seed=303, bright=0.55, cutoff=1200, wet=0.34,
         desc="慢速小调, 稀疏钢琴"),
    dict(id="sad_platform", title="Empty Platform", emotion="sad", bpm=58, root=50, mode="minor",
         prog=[0, 3, 4, 0], style="ambient", bars=12, seed=404, bright=0.45, cutoff=900, wet=0.4,
         desc="极慢长音铺底, 低能量"),
    # ---------- 愤怒 angry
    dict(id="angry_iron", title="Iron Pulse", emotion="angry", bpm=154, root=52, mode="minor",
         prog=[0, 0, 5, 0], style="rock", bars=16, seed=505, bright=1.2, drive=1.6, wet=0.14,
         desc="失真强力和弦, 密集底鼓"),
    dict(id="angry_redline", title="Redline", emotion="angry", bpm=170, root=53, mode="minor",
         prog=[0, 6, 5, 0], style="rock", bars=16, seed=606, bright=1.35, drive=1.9, wet=0.12,
         desc="极速强拍, 高响度"),
    # ---------- 恐惧 fearful
    dict(id="fear_corridor", title="Cold Corridor", emotion="fearful", bpm=92, root=49, mode="minor",
         prog=[0, 1, 0, 0], style="tense", bars=12, seed=707, cutoff=800, vibrato=0.02, tremolo=0.9,
         wet=0.45, desc="半音簇颤音, 不规则节奏"),
    dict(id="fear_below", title="Something Below", emotion="fearful", bpm=96, root=45, mode="minor",
         prog=[0, 0, 1, 0], style="tense", bars=12, seed=808, cutoff=600, vibrato=0.03, tremolo=1.0,
         wet=0.5, desc="低频嗡鸣 + 突兀打击"),
    # ---------- 厌恶 disgusted
    dict(id="disgust_sour", title="Sour Machine", emotion="disgusted", bpm=104, root=55, mode="minor",
         prog=[0, 2, 1, 0], style="odd", bars=12, seed=909, cutoff=1500, wet=0.2,
         desc="半音下行, 失谐嗡响"),
    dict(id="disgust_grease", title="Grease Waltz", emotion="disgusted", bpm=96, root=51, mode="minor",
         prog=[0, 2, 1, 0], style="odd", bars=12, seed=111, cutoff=1300, wet=0.22,
         desc="怪异音程, 拖沓低音"),
    # ---------- 惊讶 surprised
    # 注: 原本这首标为 surprised, 但 major 调 + 中速合成后实测 V=0.74 (明亮快乐),
    # 与惊讶原型 V=0.62 偏离, 故按其真实听感归为 happy。
    dict(id="happy_bloom", title="Sudden Bloom", emotion="happy", bpm=156, root=60, mode="major",
         prog=[0, 5, 4, 0], style="surprise", bars=12, seed=222, bright=1.2, wet=0.24,
         desc="明亮钟琴 + 跳跃切分"),
    dict(id="surprise_bolt", title="Bolt from Blue", emotion="surprised", bpm=156, root=64, mode="major",
         prog=[0, 5, 3, 4], style="surprise", bars=12, seed=555, bright=1.35, wet=0.20,
         desc="高唤醒密集钟琴, 骤然加速"),
    dict(id="surprise_halo", title="Halo Flash", emotion="surprised", bpm=168, root=57, mode="major",
         prog=[0, 4, 1, 5], style="surprise", bars=12, seed=666, bright=1.30, wet=0.22,
         desc="极快切分, 明亮上行音型"),
    # ---------- 平静 neutral
    dict(id="neutral_lantern", title="Paper Lanterns", emotion="neutral", bpm=100, root=60, mode="major",
         prog=[1, 4, 0, 0], style="bright", bars=12, seed=333, bright=0.8, wet=0.26,
         desc="中速, 温和音色"),
    dict(id="neutral_still", title="Still Water", emotion="neutral", bpm=76, root=53, mode="major",
         prog=[0, 3, 0, 0], style="ambient", bars=12, seed=444, bright=0.6, cutoff=1400, wet=0.35,
         desc="慢速均匀, 低动态"),
]

META = []

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for spec in TRACKS:
        s = build_track(dict(spec))
        x = render(s)
        fn = s["id"] + ".wav"
        wav_write(os.path.join(OUT_DIR, fn), x)
        dur = len(x) / SR
        META.append(dict(
            id=s["id"], title=s["title"], artist="Local Synth Ensemble",
            emotion=s["emotion"], bpm=s["bpm"], root=NOTE_NAMES[s["root"] % 12],
            mode=s["mode"], bars=s["bars"], duration=round(dur, 2),
            file="assets/music/" + fn, desc=s["desc"], source="local",
        ))
        print(f"{fn:28s} {dur:5.1f}s {os.path.getsize(os.path.join(OUT_DIR,fn))/1024:7.1f}KB  {s['emotion']:10s} {s['bpm']}bpm {NOTE_NAMES[s['root']%12]} {s['mode']}")

    with open(LIB_JSON, "w", encoding="utf-8") as f:
        json.dump(META, f, ensure_ascii=False, indent=2)
    print("\nlibrary.json written:", len(META), "tracks")
