/* ============================================================================
 * face-engine.js — 本地人脸检测 + 表情识别引擎 (face-api.js / TensorFlow.js)
 * ----------------------------------------------------------------------------
 * 算法链路:
 *   1) TinyFaceDetector  —— 基于 SSD-MobileNetV1 的轻量单阶段人脸检测器,
 *      输入缩放至 224×224, 单帧推理 ~10-20ms (CPU/WASM)
 *   2) FaceExpressionNet —— 8 层卷积 + 全连接, 输入 64×64 灰度人脸,
 *      输出 7 类表情概率 (FER2013 数据集训练)
 *   3) 时序 EMA 平滑 + 稳定性滞后 (hysteresis) 判决, 抑制单帧抖动
 * ==========================================================================*/
(function (root) {
  'use strict';

  const MODEL_URL = 'models';
  const LABELS = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised'];

  const state = {
    loaded: false,
    loading: false,
    error: null,
    backend: '',
    version: ''
  };

  /* ----------------------------------------------------------- 模型加载 */
  async function load(onProgress) {
    if (state.loaded) return true;
    if (state.loading) {
      while (state.loading) await new Promise(r => setTimeout(r, 120));
      return state.loaded;
    }
    state.loading = true;
    state.error = null;
    try {
      if (typeof faceapi === 'undefined') throw new Error('face-api.js 未加载');
      state.version = faceapi.version ? faceapi.version.tfjs || '' : '';
      onProgress && onProgress('加载人脸检测模型 (TinyFaceDetector)…', 15);
      await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
      onProgress && onProgress('加载表情分类模型 (FaceExpressionNet)…', 65);
      await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL);
      onProgress && onProgress('初始化推理后端…', 90);
      if (faceapi.tf) {
        try { state.backend = faceapi.tf.getBackend(); } catch (e) { state.backend = 'unknown'; }
        if (state.backend === 'webgl') {
          // 预热一次, 避免首帧卡顿
          const warm = document.createElement('canvas');
          warm.width = warm.height = 224;
          const ctx = warm.getContext('2d');
          ctx.fillStyle = '#888'; ctx.fillRect(0, 0, 224, 224);
          await faceapi.detectAllFaces(warm, detectorOptions(224));
        }
      }
      state.loaded = true;
      onProgress && onProgress('模型就绪', 100);
      return true;
    } catch (e) {
      state.error = e.message || String(e);
      throw e;
    } finally {
      state.loading = false;
    }
  }

  function detectorOptions(inputSize, scoreThreshold) {
    return new faceapi.TinyFaceDetectorOptions({
      inputSize: inputSize || 224,
      scoreThreshold: scoreThreshold == null ? 0.42 : scoreThreshold
    });
  }

  /**
   * 对任意图像源 (img / video / canvas) 执行一次检测 + 表情识别
   * @returns {Promise<{faces:Array, ms:number}>}
   */
  async function detect(source, opts) {
    opts = opts || {};
    const t0 = performance.now();
    const options = detectorOptions(opts.inputSize, opts.scoreThreshold);
    const res = await faceapi.detectAllFaces(source, options).withFaceExpressions();
    const ms = performance.now() - t0;

    const faces = (res || []).map((r, i) => {
      const exp = r.expressions || {};
      let dominant = 'neutral', best = -1;
      LABELS.forEach(k => {
        const v = exp[k] || 0;
        if (v > best) { best = v; dominant = k; }
      });
      return {
        index: i,
        box: { x: r.detection.box.x, y: r.detection.box.y, w: r.detection.box.width, h: r.detection.box.height },
        score: r.detection.score,
        expressions: LABELS.reduce((o, k) => { o[k] = exp[k] || 0; return o; }, {}),
        dominant,
        confidence: best
      };
    });
    // 取面积最大的人脸作为主目标
    faces.sort((a, b) => b.box.w * b.box.h - a.box.w * a.box.h);
    return { faces, ms, faceCount: faces.length };
  }

  /* ------------------------------------------- 时序平滑 + 稳定性判决器 */
  class EmotionSmoother {
    constructor(alpha) {
      this.alpha = alpha == null ? 0.35 : alpha;
      this.ema = null;
      this.stable = null;         // 稳定的情绪标签
      this.stableFrames = 0;      // 连续命中帧数
      this.frames = 0;
      this.history = [];          // 最近若干帧的概率向量 (用于趋势图)
      this.maxHistory = 60;
    }
    reset() {
      this.ema = null; this.stable = null; this.stableFrames = 0; this.frames = 0; this.history.length = 0;
    }
    push(expressions, faceScore, hold) {
      hold = hold || 3;
      this.frames++;
      const a = this.alpha;
      if (!this.ema) {
        this.ema = {};
        LABELS.forEach(k => { this.ema[k] = expressions[k] || 0; });
      } else {
        LABELS.forEach(k => {
          this.ema[k] = (1 - a) * this.ema[k] + a * (expressions[k] || 0);
        });
      }
      // 低质量人脸时衰减可信度
      const q = Math.min(1, Math.max(0, (faceScore - 0.3) / 0.5));

      let dom = 'neutral', best = -1, sum = 0;
      LABELS.forEach(k => {
        sum += this.ema[k];
        if (this.ema[k] > best) { best = this.ema[k]; dom = k; }
      });
      if (sum > 0) LABELS.forEach(k => { this.ema[k] /= sum; });

      if (dom === this.stable) this.stableFrames++;
      else {
        this.stableFrames = 1;
        // 需要连续 hold 帧才切换最终标签 (滞后判决)
        if (this.stable == null || this.stableFrames >= hold) this.stable = dom;
      }
      if (this.stable == null) this.stable = dom;

      const snapshot = { t: Date.now(), probs: Object.assign({}, this.ema), dom, q };
      this.history.push(snapshot);
      if (this.history.length > this.maxHistory) this.history.shift();

      return {
        probs: this.ema,
        dominant: dom,
        stable: this.stable,
        stableFrames: this.stableFrames,
        confidence: this.ema[this.stable] || 0,
        quality: q
      };
    }
  }

  /* ------------------------------------------------------- 叠加层绘制 */
  function drawOverlay(canvas, source, result, labelText, color) {
    if (!canvas || !source) return;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const dpr = Math.min(2, (window.devicePixelRatio || 1));
    const cw = Math.round(rect.width * dpr), ch = Math.round(rect.height * dpr);
    if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, cw, ch);
    if (!result || !result.faces || !result.faces.length) return;

    const vw = source.videoWidth || source.naturalWidth || source.width;
    const vh = source.videoHeight || source.naturalHeight || source.height;
    if (!vw || !vh) return;
    // object-fit: contain 的实际缩放与留白
    const scale = Math.min(cw / vw, ch / vh);
    const ox = (cw - vw * scale) / 2, oy = (ch - vh * scale) / 2;

    result.faces.forEach((f, i) => {
      const x = ox + f.box.x * scale;
      const y = oy + f.box.y * scale;
      const w = f.box.w * scale;
      const h = f.box.h * scale;
      const c = i === 0 ? (color || '#6366F1') : 'rgba(148,163,184,0.9)';
      ctx.lineWidth = Math.max(2, 2.2 * dpr);
      ctx.strokeStyle = c;
      const r = Math.min(10 * dpr, w / 4, h / 4);
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.arcTo(x + w, y, x + w, y + h, r);
      ctx.arcTo(x + w, y + h, x, y + h, r);
      ctx.arcTo(x, y + h, x, y, r);
      ctx.arcTo(x, y, x + w, y, r);
      ctx.closePath();
      ctx.stroke();

      const text = i === 0 ? (labelText || '') : ('Face ' + (i + 1));
      if (!text) return;
      const fs = 12 * dpr;
      ctx.font = '600 ' + fs + 'px system-ui, -apple-system, "PingFang SC", sans-serif';
      const tw = ctx.measureText(text).width;
      const tagH = fs + 8 * dpr;
      const tagW = Math.min(w, tw + 14 * dpr);
      const tagY = Math.max(0, y - tagH);
      ctx.fillStyle = c;
      ctx.beginPath();
      const rr = Math.min(6 * dpr, tagW / 2);
      ctx.moveTo(x + rr, tagY);
      ctx.arcTo(x + tagW, tagY, x + tagW, tagY + tagH, rr);
      ctx.arcTo(x + tagW, tagY + tagH, x, tagY + tagH, rr);
      ctx.arcTo(x, tagY + tagH, x, tagY, rr);
      ctx.arcTo(x, tagY, x + tagW, tagY, rr);
      ctx.closePath();
      ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, x + 7 * dpr, tagY + tagH / 2 + 0.5);
      ctx.textBaseline = 'alphabetic';
    });
  }

  root.FaceEngine = { load, detect, state, LABELS, EmotionSmoother, drawOverlay, detectorOptions };
})(window);
