/* ============================================================================
 * melody-analyzer.js  —  旋律 / 曲调情绪特征提取
 * 纯 DSP 实现 (只依赖 Float32Array PCM), 浏览器与 Node 通用, 无任何网络依赖。
 * ----------------------------------------------------------------------------
 * 经典算法清单:
 *   1. Cooley–Tukey 基 2 快速傅里叶变换              O(N log N)
 *   2. Hann 窗 + 短时傅里叶变换 (STFT), 帧长 2048/帧移 512 (86.1 fps)
 *   3. 半波整流谱通量 (Spectral Flux) 起始点检测      Dixon, 2006
 *   4. onset 包络自相关 + 对数正态节拍先验 + 倍频修正  Ellis, 2007
 *   5. HPCP 谐波峰值色度 (逐帧归一化)                 Gómez, 2006
 *   6. Krumhansl–Schmuckler 调性轮廓相关 + 主音显著性  Krumhansl, 1990
 *   7. 谱平坦度 / 高频能量比 / 过零率 -> 粗糙度 (失真感)
 *   8. 色度熵 -> 和声复杂度 (半音化程度)
 *   9. Thayer–Russell 情感环状模型 (Valence × Arousal)
 * ==========================================================================*/
(function (root, factory) {
  const mod = factory();
  if (typeof module !== 'undefined' && module.exports) module.exports = mod;
  if (typeof window !== 'undefined') window.MelodyAnalyzer = mod;
})(this, function () {
  'use strict';

  const TARGET_SR = 22050;
  const FRAME = 2048;
  const HOP = 512;
  const EPS = 1e-10;
  const KEY_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const EMOTIONS = ['happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised', 'neutral'];

  /* ------------------------------------------------------------- FFT */
  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -2 * Math.PI / len;
      const wr = Math.cos(ang), wi = Math.sin(ang);
      const half = len >> 1;
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < half; k++) {
          const ur = re[i + k], ui = im[i + k];
          const vr = re[i + k + half] * cr - im[i + k + half] * ci;
          const vi = re[i + k + half] * ci + im[i + k + half] * cr;
          re[i + k] = ur + vr; im[i + k] = ui + vi;
          re[i + k + half] = ur - vr; im[i + k + half] = ui - vi;
          const ncr = cr * wr - ci * wi;
          ci = cr * wi + ci * wr; cr = ncr;
        }
      }
    }
  }

  function toMono(samples, ch, sr) {
    let mono;
    if (ch <= 1) mono = samples;
    else {
      const n = Math.floor(samples.length / ch);
      mono = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let c = 0; c < ch; c++) s += samples[i * ch + c];
        mono[i] = s / ch;
      }
    }
    if (Math.abs(sr - TARGET_SR) < 1) return { pcm: mono, sr: TARGET_SR };
    const ratio = sr / TARGET_SR;
    const outLen = Math.floor(mono.length / ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const p = i * ratio, i0 = Math.floor(p), f = p - i0;
      const a = mono[i0] || 0, b = mono[i0 + 1] || 0;
      out[i] = a + (b - a) * f;
    }
    return { pcm: out, sr: TARGET_SR };
  }

  /* --------------------------------------------- 频点 -> 音高类 映射表 */
  const BIN_PC = new Int8Array(FRAME / 2 + 1).fill(-1);
  const BIN_HZ = new Float64Array(FRAME / 2 + 1);
  const HF_START = Math.round(4000 / (TARGET_SR / FRAME));
  (function () {
    const binHz = TARGET_SR / FRAME;
    for (let k = 1; k < BIN_PC.length; k++) {
      const f = k * binHz;
      BIN_HZ[k] = f;
      if (f < 55 || f > 4500) continue;
      const midi = 69 + 12 * Math.log2(f / 440);
      BIN_PC[k] = ((Math.round(midi) % 12) + 12) % 12;
    }
  })();

  const KK_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
  const KK_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

  function corr(a, b) {
    const n = a.length;
    let ma = 0, mb = 0;
    for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    let num = 0, da = 0, db = 0;
    for (let i = 0; i < n; i++) {
      const x = a[i] - ma, y = b[i] - mb;
      num += x * y; da += x * x; db += y * y;
    }
    return num / (Math.sqrt(da * db) + EPS);
  }

  const clamp01 = x => x < 0 ? 0 : x > 1 ? 1 : x;
  const nrm = (x, lo, hi) => clamp01((x - lo) / (hi - lo));
  const median = arr => {
    if (!arr.length) return 0;
    const a = Float64Array.from(arr).sort();
    const m = a.length >> 1;
    return a.length % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
  };

  /* ------------------------------------------------ 音乐情绪原型 (四维) */
  // v: 效价(愉快度)  a: 唤醒度(激烈度)  r: 粗糙度(失真/噪感)  h: 和声复杂度(半音化)
  const MUSIC_PROTOTYPES = {
    happy:     { v: 0.80, a: 0.70, r: 0.55, h: 0.20 },
    sad:       { v: 0.20, a: 0.18, r: 0.25, h: 0.26 },
    angry:     { v: 0.18, a: 0.86, r: 0.62, h: 0.30 },
    fearful:   { v: 0.28, a: 0.50, r: 0.20, h: 0.55 },
    disgusted: { v: 0.22, a: 0.44, r: 0.36, h: 0.82 },
    surprised: { v: 0.62, a: 0.78, r: 0.35, h: 0.42 },
    neutral:   { v: 0.62, a: 0.40, r: 0.35, h: 0.22 }
  };
  const DIM_W = { v: 1.30, a: 1.15, r: 0.50 };   // 效价/唤醒 为主, 粗糙度为辅

  function analyzePcm(pcmIn, srIn, opts) {
    opts = opts || {};
    const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
    const t0 = now();

    const conv = toMono(pcmIn, opts.channels || 1, srIn || TARGET_SR);
    const pcm = conv.pcm, sr = conv.sr;
    const maxLen = Math.min(pcm.length, Math.floor(sr * 60));
    if (maxLen < FRAME * 3) throw new Error('音频过短，无法分析');

    const win = new Float64Array(FRAME);
    for (let i = 0; i < FRAME; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (FRAME - 1));

    const nFrames = Math.floor((maxLen - FRAME) / HOP) + 1;
    const bins = FRAME / 2 + 1;
    const normMag = 2 / FRAME;

    let prev = new Float64Array(bins);
    let mag = new Float64Array(bins);
    const onsetEnv = new Float64Array(nFrames);
    const chroma = new Float64Array(12);
    const bassChroma = new Float64Array(12);
    const re = new Float64Array(FRAME), im = new Float64Array(FRAME);

    const sRms = new Float64Array(nFrames);
    const zcrSeries = [];
    const flatSeries = [];
    const fluxSeries = [];
    const hfSeries = [];

    let centroidW = 0, centroidAcc = 0, rmsAcc = 0, rmsSqAcc = 0;

    for (let t = 0; t < nFrames; t++) {
      const off = t * HOP;
      let s2 = 0, zc = 0;
      for (let i = 0; i < FRAME; i++) {
        const s = pcm[off + i] || 0;
        s2 += s * s;
        if (i > 0 && s * (pcm[off + i - 1] || 0) < 0) zc++;
        re[i] = s * win[i]; im[i] = 0;
      }
      const rms = Math.sqrt(s2 / FRAME);
      sRms[t] = rms;
      fft(re, im);

      let magSum = 0, centNum = 0, flux = 0, logSum = 0, linSum = 0, nAct = 0, hfE = 0, totE = 0;
      for (let k = 1; k < bins; k++) {
        const m = Math.sqrt(re[k] * re[k] + im[k] * im[k]) * normMag;
        mag[k] = m;
        magSum += m;
        const f = BIN_HZ[k];
        centNum += f * m;
        totE += m;
        if (k >= HF_START) hfE += m;
        if (f < 5000) { linSum += m + EPS; logSum += Math.log(m + EPS); nAct++; }
        const d = m - prev[k];
        if (d > 0) flux += d;
      }

      // --- HPCP: 只取谱峰, 逐帧归一化后累加
      const hpcp = new Float64Array(12), hpcpBass = new Float64Array(12);
      let peakE = 0;
      for (let k = 2; k < bins - 1; k++) {
        const pc = BIN_PC[k];
        if (pc < 0) continue;
        const m = mag[k];
        if (m <= mag[k - 1] || m < mag[k + 1]) continue;
        if (m < 0.05 * (magSum / bins) * 12) continue;
        hpcp[pc] += m;
        if (BIN_HZ[k] < 250) hpcpBass[pc] += m;
        peakE += m;
      }
      const flat = nAct > 0 ? Math.exp(logSum / nAct) / (linSum / Math.max(1, nAct)) : 0;
      // 噪声帧降权: 软化的 1/(1+k·flat)。真实混音的谱平坦度常在 0.4~0.7,
      // 若用硬阈值会把所有帧判为噪声(色度归零), 因此这里用平滑衰减并设下限。
      const tonality = Math.max(0.18, 1 / (1 + 4.2 * flat));
      if (peakE > 1e-5) {
        let hMax = 0, hbMax = 0;
        for (let i = 0; i < 12; i++) {
          if (hpcp[i] > hMax) hMax = hpcp[i];
          if (hpcpBass[i] > hbMax) hbMax = hpcpBass[i];
        }
        for (let i = 0; i < 12; i++) {
          if (hMax > 0) chroma[i] += tonality * (hpcp[i] / hMax);
          if (hbMax > 0) bassChroma[i] += tonality * (hpcpBass[i] / hbMax);
        }
      }

      onsetEnv[t] = flux;
      if (rms > 0.006) {
        centroidAcc += (centNum / (magSum + EPS)) * rms;
        centroidW += rms;
        flatSeries.push(flat);
        fluxSeries.push(flux / (magSum + EPS));
        hfSeries.push(totE > 0 ? hfE / totE : 0);
      }
      zcrSeries.push(zc / FRAME);
      rmsAcc += rms; rmsSqAcc += rms * rms;
      const tmp = prev; prev = mag; mag = tmp;
    }

    const fps = sr / HOP;
    const durSec = maxLen / sr;

    /* ---------------- 1) 节拍 BPM ---------------- */
    let oeMean = 0;
    for (let i = 0; i < nFrames; i++) oeMean += onsetEnv[i];
    oeMean /= nFrames;
    const oe = new Float64Array(nFrames);
    for (let i = 1; i < nFrames - 1; i++) {
      const v = Math.max(0, onsetEnv[i] - oeMean);
      oe[i] = (Math.max(0, onsetEnv[i - 1] - oeMean) + 2 * v + Math.max(0, onsetEnv[i + 1] - oeMean)) / 4;
    }
    /* ------------- 起始点统计 (供节拍倍频判据与唤醒度使用) ------------- */
    let oeMax = 0, oeSum = 0;
    for (let i = 0; i < nFrames; i++) { if (oe[i] > oeMax) oeMax = oe[i]; oeSum += oe[i]; }
    const absThres = 0.28 * oeMax, relThres = 1.5 * oeSum / nFrames;
    let onsetCount = 0, strongCount = 0;
    for (let i = 2; i < nFrames - 2; i++) {
      if (oe[i] > relThres && oe[i] >= oe[i - 1] && oe[i] > oe[i + 1]) {
        onsetCount++;
        if (oe[i] > absThres) strongCount++;
      }
    }
    const onsetRate = onsetCount / durSec;
    const onsetDensity = strongCount / durSec;

    function acf(sig, lag) {
      let s = 0, c = 0;
      for (let i = 0; i + lag < sig.length; i++) { s += sig[i] * sig[i + lag]; c++; }
      return c > 0 ? s / c : 0;
    }
    const minLag = Math.max(2, Math.floor(fps * 60 / 220));
    const maxLag = Math.min(Math.ceil(fps * 60 / 40), Math.floor(nFrames / 2) - 1);
    const ac = new Float64Array(maxLag + 2);
    for (let lag = minLag; lag <= maxLag + 1; lag++) ac[lag] = acf(oe, lag);
    const prior = bpm => Math.exp(-0.5 * Math.pow(Math.log2(bpm / 115) / 1.05, 2));
    let bestLag = minLag, bestScore = -1e9;
    for (let lag = minLag; lag <= maxLag; lag++) {
      const bpm = 60 * fps / lag;
      const s = ac[lag] * prior(bpm);
      if (s > bestScore) { bestScore = s; bestLag = lag; }
    }
    // 倍频修正 (octave error 是自相关法最常见的失效模式)
    const cands = [];
    for (const mul of [0.5, 1, 2]) {
      const lg = Math.round(bestLag * mul);
      if (lg >= minLag && lg + 1 <= maxLag) {
        const b2 = 60 * fps / lg;
        cands.push({ lag: lg, acv: ac[lg], s: ac[lg] * prior(b2) * (b2 >= 80 && b2 <= 180 ? 1.10 : 1) });
      }
    }
    cands.sort((a, b) => b.s - a.s);
    if (cands.length && cands[0].acv > 0.46 * ac[bestLag]) bestLag = cands[0].lag;

    // 半速纠偏 (half-tempo error): 自相关在 2 拍周期上也会出峰, "每拍一击 + 密集八分音符"
    // 的织体容易被读成一半速度。判据: 慢读法 < 105 BPM, 而它的 2 倍读法落在 [110,190] 且
    // 自相关强度与慢读法相当 (>0.62 倍) —— 此时快读法才是真实拍点。
    // 抒情慢歌的八分音符层次相关值通常只有 0.1~0.2 倍, 因此不会被误翻倍。
    {
      const bpmNow = 60 * fps / bestLag;
      const fastLag = Math.round(bestLag / 2);
      if (bpmNow < 105 && fastLag >= minLag) {
        const bpmFast = 60 * fps / fastLag;
        if (bpmFast >= 110 && bpmFast <= 190 && ac[fastLag] > 0.62 * ac[bestLag]) {
          bestLag = fastLag;
        }
      }
    }

    let bpm = 60 * fps / bestLag;
    const y0 = ac[bestLag - 1], y1 = ac[bestLag], y2 = ac[bestLag + 1];
    const den = y0 - 2 * y1 + y2;
    if (Math.abs(den) > EPS) {
      const d = 0.5 * (y0 - y2) / den;
      if (Math.abs(d) < 1) bpm = 60 * fps / (bestLag + d);
    }
    let acMean = 0, cnt = 0;
    for (let l = minLag; l <= maxLag; l++) { acMean += ac[l]; cnt++; }
    acMean /= Math.max(1, cnt);
    const pulseClarity = clamp01((ac[bestLag] / (acMean + EPS) - 1) / 2.4);

    /* ---------------- 2) 调性 ---------------- */
    const cSum = Array.from(chroma).map(x => Math.sqrt(x));
    const cMax = Math.max.apply(null, cSum) + EPS;
    const chromaN = cSum.map(x => x / cMax);
    const bMax = Math.max.apply(null, bassChroma) + EPS;
    const bassN = Array.from(bassChroma).map(x => Math.sqrt(x) / Math.sqrt(bMax));

    // 主音显著性 = 该音级在色度中的相对能量 (音乐中主音通常最突出)
    let bestM = -9, bestm = -9, rootM = 0, rootm = 0;
    const keyTable = [];
    for (let r = 0; r < 12; r++) {
      const rot = [], rotB = [];
      for (let i = 0; i < 12; i++) { rot.push(chromaN[(i + r) % 12]); rotB.push(bassN[(i + r) % 12]); }
      const tonicBonus = 0.26 * chromaN[r] + 0.10 * rotB[0];
      const cM = corr(rot, KK_MAJOR) + tonicBonus;
      const cm = corr(rot, KK_MINOR) + tonicBonus;
      keyTable.push({ r, major: cM, minor: cm });
      if (cM > bestM) { bestM = cM; rootM = r; }
      if (cm > bestm) { bestm = cm; rootm = r; }
    }
    const isMajor = bestM >= bestm;
    const keyRoot = isMajor ? rootM : rootm;
    const keyScore = Math.max(bestM, bestm);
    // 三度音 (大三度 vs 小三度) 作为调式的直接证据
    const thirdMajor = chromaN[(keyRoot + 4) % 12];
    const thirdMinor = chromaN[(keyRoot + 3) % 12];
    const thirdBias = clamp01(0.5 + (thirdMajor - thirdMinor) * 2.0);
    const modeStrength = clamp01(0.55 * Math.abs(bestM - bestm) * 5 + 0.45 * Math.abs(thirdBias - 0.5) * 2);
    // 调式置信度: 大/小调轮廓得分差距越小越不可信, 用于收缩效价
    const modeConf = clamp01(0.55 + Math.abs(bestM - bestm) * 4.5 + Math.abs(thirdBias - 0.5) * 3);

    // 色度熵: 半音化 / 和声复杂度
    let cTot = 0;
    for (let i = 0; i < 12; i++) cTot += chromaN[i] + EPS;
    let ent = 0;
    for (let i = 0; i < 12; i++) {
      const p = (chromaN[i] + EPS) / cTot;
      ent -= p * Math.log(p);
    }
    const chromaEntropy = clamp01((ent / Math.log(12) - 0.72) / 0.28);

    /* ---------------- 3) 音色 ---------------- */
    const centroid = centroidW > 0 ? centroidAcc / centroidW : 0;
    const centroidNorm = clamp01(centroid / 4200);
    const rmsMean = rmsAcc / nFrames;
    const rmsSd = Math.sqrt(Math.max(0, rmsSqAcc / nFrames - rmsMean * rmsMean));
    const dynamicRange = clamp01(rmsSd / (rmsMean + EPS));
    const zcrMed = median(zcrSeries);
    const flatMed = median(flatSeries);
    const hfMed = median(hfSeries);
    const fluxMean = fluxSeries.length ? fluxSeries.reduce((a, b) => a + b, 0) / fluxSeries.length : 0;
    const loudnessDb = 20 * Math.log10(rmsMean + EPS);

    /* ---------------- 4) 起始点密度 (已在上方计算) ---------------- */

    /* ---------------- 5) 四维情绪特征 ---------------- */
    const roughRaw = clamp01(
      0.42 * nrm(hfMed, 0.02, 0.26)
      + 0.30 * nrm(zcrMed, 0.030, 0.120)
      + 0.18 * nrm(fluxMean, 0.090, 0.210)
      + 0.10 * nrm(flatMed, 0.05, 0.40)
    );
    const valence = clamp01(
      0.5
      + (isMajor ? 1 : -1) * (0.18 + 0.09 * modeStrength) * modeConf
      + 0.10 * (centroidNorm - 0.40) * 2
      - 0.10 * Math.max(0, roughRaw - 0.60)
      + 0.04 * (dynamicRange < 0.50 ? 1 : -0.5)
    );
    const arousal = clamp01(
      0.34 * nrm(bpm, 58, 172)
      + 0.20 * nrm(onsetDensity, 0.6, 6.5)
      + 0.16 * nrm(rmsMean, 0.15, 0.42)
      + 0.14 * nrm(fluxMean, 0.090, 0.210)
      + 0.16 * pulseClarity
    );

    /* ---------------- 6) 七类情绪 (加权欧氏距离 + softmax) -------- */
    const feat = { v: valence, a: arousal, r: roughRaw, h: chromaEntropy };
    const logits = {};
    let mx = -Infinity;
    EMOTIONS.forEach(e => {
      const p = MUSIC_PROTOTYPES[e];
      let d = 0;
      for (const k of ['v', 'a', 'r']) d += DIM_W[k] * Math.pow(feat[k] - p[k], 2);
      logits[e] = -d;
      if (-d > mx) mx = -d;
    });
    let sum = 0;
    const scores = {};
    EMOTIONS.forEach(e => { scores[e] = Math.exp((logits[e] - mx) / 0.05); sum += scores[e]; });
    EMOTIONS.forEach(e => { scores[e] /= sum; });
    let top = EMOTIONS[0];
    EMOTIONS.forEach(e => { if (scores[e] > scores[top]) top = e; });

    const t1 = now();
    const r3 = x => Math.round(x * 1000) / 1000;

    const out = {
      emotion: top,
      confidence: r3(scores[top]),
      scores: Object.fromEntries(EMOTIONS.map(e => [e, r3(scores[e])])),
      valence: r3(valence), arousal: r3(arousal), roughness: r3(roughRaw), harmonic: r3(chromaEntropy),
      bpm: Math.round(bpm * 10) / 10, pulseClarity: r3(pulseClarity),
      key: keyRoot, keyName: KEY_NAMES[keyRoot] + (isMajor ? ' major' : ' minor'),
      mode: isMajor ? 'major' : 'minor', modeStrength: r3(modeStrength), modeConf: r3(modeConf), keyScore: r3(keyScore),
      thirdMajor: r3(thirdMajor), thirdMinor: r3(thirdMinor),
      centroid: Math.round(centroid), centroidNorm: r3(centroidNorm),
      onsetRate: Math.round(onsetRate * 100) / 100, onsetDensity: Math.round(onsetDensity * 100) / 100,
      rms: r3(rmsMean), loudnessDb: Math.round(loudnessDb * 10) / 10, dynamicRange: r3(dynamicRange),
      zcr: r3(zcrMed), flatness: r3(flatMed), hfRatio: r3(hfMed), flux: r3(fluxMean),
      duration: Math.round(durSec * 10) / 10, analyzeMs: Math.round(t1 - t0), frames: nFrames,
      chroma: chromaN.map(r3), bassChroma: bassN.map(r3)
    };
    if (opts.debug) {
      out.keyTable = keyTable;
      out.debugInfo = { flatMed, frames: nFrames };
      // 节拍倍频判据的中间量 (调参用)
      (function () {
        const lf = Math.max(minLag, Math.round(bestLag / 2));
        const l2 = Math.min(maxLag, bestLag * 2);
        out.debugInfo.tempo = {
          bestLag, bpm: Math.round(60 * fps / bestLag * 10) / 10,
          acBest: +ac[bestLag].toFixed(4),
          fast: { lag: lf, bpm: Math.round(60 * fps / lf), ac: +ac[lf].toFixed(4) },
          slow: { lag: l2, bpm: Math.round(60 * fps / l2), ac: +ac[l2].toFixed(4) },
          onsetRate: +onsetRate.toFixed(2),
          needRateFast: +(0.72 * 2 * (60 * fps / lf) / 60).toFixed(2)
        };
      })();
    }
    return out;
  }

  return { analyzePcm, EMOTIONS, MUSIC_PROTOTYPES, KEY_NAMES, FRAME, HOP, TARGET_SR };
});
