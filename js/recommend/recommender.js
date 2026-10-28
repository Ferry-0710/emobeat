/* ============================================================================
 * recommender.js — 情绪 → 音乐 推荐引擎
 * ----------------------------------------------------------------------------
 * 核心思想: 把"人脸情绪"与"音乐情绪"放进同一个 Thayer 环状空间
 *          (Valence 效价 × Arousal 唤醒度 × Roughness 粗糙度),
 *          再用"内容相似度 + 情绪分布余弦相似度"加权打分,
 *          最后用 MMR (Maximal Marginal Relevance) 做多样性重排。
 *
 * 参考: Thayer 1989 情绪环状模型 / Russell 1980 circumplex
 *       Carbonell & Goldstein 1998 MMR
 * ==========================================================================*/
(function (root) {
  'use strict';

  const DIM_W = { v: 1.30, a: 1.15, r: 0.50 };   // 与音乐侧分类器保持一致
  const ORDER = ['happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised', 'neutral'];

  const STRATEGIES = {
    match: {
      key: 'match', cn: '情绪匹配', desc: '推荐与当前情绪同频的音乐 (共情策略)',
      transform: f => ({ v: f.v, a: f.a, r: f.r })
    },
    soothe: {
      key: 'soothe', cn: '舒缓调节', desc: '保持一定共鸣, 同时降低唤醒度、提升效价',
      transform: f => ({ v: 0.55 + 0.42 * f.v, a: 0.18 + 0.30 * f.a, r: Math.max(0.15, f.r - 0.18) })
    },
    energize: {
      key: 'energize', cn: '提振精神', desc: '向高唤醒、正向情绪迁移 (提神策略)',
      transform: f => ({ v: 0.62 + 0.36 * f.v, a: 0.62 + 0.32 * f.a, r: Math.min(0.95, f.r + 0.05) })
    }
  };

  /* ------------------------------------------------ 由表情概率求情绪坐标 */
  function faceVector(probs, EMO) {
    let sv = 0, sa = 0, sr = 0, sw = 0;
    ORDER.forEach(k => {
      const p = Math.max(0, probs[k] || 0);
      if (p <= 0) return;
      const w = Math.pow(p, 1.15);       // 幂次放大主情绪影响
      sv += w * EMO[k].face.v;
      sa += w * EMO[k].face.a;
      sr += w * EMO[k].face.r;
      sw += w;
    });
    if (sw <= 0) return { v: 0.5, a: 0.4, r: 0.2 };
    return { v: sv / sw, a: sa / sw, r: sr / sw };
  }

  // 注意: 人脸侧与音频侧的"粗糙度"量纲不同 (人脸 R 描述表情张力, 音频 R 描述失真/噪声),
  // 因此跨模态匹配只用 效价 V + 唤醒 A; 粗糙度仅参与音乐侧的内部情绪分类。
  function dist2(a, b) {
    return DIM_W.v * Math.pow((a.v || 0) - (b.v || 0), 2) +
           DIM_W.a * Math.pow((a.a || 0) - (b.a || 0), 2);
  }

  function cosine(a, b) {
    let dot = 0, na = 0, nb = 0;
    ORDER.forEach(k => {
      const x = a[k] || 0, y = b[k] || 0;
      dot += x * y; na += x * x; nb += y * y;
    });
    return dot / (Math.sqrt(na * nb) + 1e-9);
  }

  /* --------------------------------------------------------- 打分主函数 */
  /**
   * @param {object} ctx
   *  ctx.probs      - 7 类表情概率
   *  ctx.tracks     - 候选曲目 (需含 features: {valence, arousal, roughness, scores})
   *  ctx.emoData    - EMO_DATA
   *  ctx.strategy   - 'match' | 'soothe' | 'energize'
   *  ctx.recentIds  - 最近播放过的 id (用于去重/惩罚)
   *  ctx.lambda     - MMR 多样性权重 (0~1, 越大越看相关性)
   *  ctx.topN
   */
  function recommend(ctx) {
    const probs = ctx.probs || {};
    const EMO = ctx.emoData.EMOTIONS;
    const strategy = STRATEGIES[ctx.strategy] || STRATEGIES.match;
    const faceV = faceVector(probs, EMO);
    const target = strategy.transform(faceV);
    const lambda = ctx.lambda == null ? 0.78 : ctx.lambda;
    const recent = new Set(ctx.recentIds || []);

    const scored = (ctx.tracks || []).filter(t => t.features).map(t => {
      const f = {
        v: t.features.valence, a: t.features.arousal, r: t.features.roughness
      };
      const dScore = Math.exp(-dist2(target, f) / 0.16);        // 空间距离 -> 相似度
      const eScore = cosine(probs, t.features.scores || {});     // 情绪分布相似度
      let s = 0.58 * dScore + 0.32 * eScore;
      // 元数据缺失的在线曲目会在外部预置 features.confidence 低于 1
      const conf = t.features.confidence == null ? 1 : t.features.confidence;
      s *= 0.72 + 0.28 * conf;
      if (recent.has(t.id)) s *= 0.55;                           // 最近听过 -> 降权
      return {
        track: t, score: s, dScore, eScore,
        reasons: explain(t, target, faceV, strategy, EMO)
      };
    }).sort((a, b) => b.score - a.score);

    // ---- MMR 多样性重排
    const selected = [];
    const pool = scored.slice();
    const topN = ctx.topN || 6;
    while (selected.length < topN && pool.length) {
      let bestIdx = 0, bestVal = -Infinity;
      for (let i = 0; i < pool.length; i++) {
        let maxSim = 0;
        for (const s of selected) {
          const sim = 1 - Math.exp(-dist2(
            { v: pool[i].track.features.valence, a: pool[i].track.features.arousal, r: pool[i].track.features.roughness },
            { v: s.track.features.valence, a: s.track.features.arousal, r: s.track.features.roughness }
          ) / 0.16);
          if (sim > maxSim) maxSim = sim;
        }
        const val = lambda * pool[i].score - (1 - lambda) * maxSim;
        if (val > bestVal) { bestVal = val; bestIdx = i; }
      }
      selected.push(pool.splice(bestIdx, 1)[0]);
    }
    // 重排后按分数排序展示 (保留 MMR 的入选集合, 但展示顺序按相关度)
    selected.sort((a, b) => b.score - a.score);

    return {
      target, faceV, strategy: strategy.key, strategyName: strategy.cn,
      list: selected, poolSize: scored.length,
      topScore: selected.length ? selected[0].score : 0
    };
  }

  /* ------------------------------------------------------------ 推荐解释 */
  function explain(track, target, faceV, strategy, EMO) {
    const f = track.features;
    const parts = [];
    const key = EMO[track.emotion] ? EMO[track.emotion].cn : track.emotion;
    // 速度 / 调式 / 音色 三个最直观的线索
    const tempoTag = f.bpm >= 140 ? '快速' : f.bpm >= 100 ? '中速' : '慢速';
    const modeTag = f.mode === 'major' ? '大调' : '小调';
    const brightTag = f.centroid >= 2000 ? '明亮' : f.centroid >= 1300 ? '中性音色' : '暗色音色';
    parts.push(`${tempoTag} ${Math.round(f.bpm)} BPM、${modeTag}${f.keyName ? '(' + f.keyName + ')' : ''}、${brightTag}`);
    // 与目标坐标的偏差
    const dv = target.v - f.valence, da = target.a - f.arousal;
    if (Math.abs(dv) < 0.14 && Math.abs(da) < 0.14) parts.push('与目标情绪坐标高度重合');
    else {
      if (Math.abs(dv) >= 0.14) parts.push(dv > 0 ? '情绪色彩比目标更低沉' : '情绪色彩比目标更明亮');
      if (Math.abs(da) >= 0.14) parts.push(da > 0 ? '能量比目标更强' : '能量比目标更收敛');
    }
    if (strategy.key !== 'match') parts.push('属于「' + strategy.cn + '」迁移方向');
    return parts.join(' · ');
  }

  /* --------------------------------------------- 在线曲目: 元数据先验打分 */
  // 在线预览音频跨域无法读取 PCM, 因此按"检索意图 + 流派"给出先验特征
  function priorFeatures(emotion, genre) {
    const P = root.MelodyAnalyzer.MUSIC_PROTOTYPES[emotion] || { v: 0.5, a: 0.45, r: 0.3 };
    const scores = {};
    ORDER.forEach(k => { scores[k] = k === emotion ? 0.62 : 0.38 / 6; });
    const gm = /rock|metal|rap|hip|electronic|dance|house/i.test(genre || '');
    return {
      valence: P.v, arousal: P.a, roughness: gm ? Math.min(0.95, P.r + 0.12) : P.r,
      bpm: Math.round(60 + P.a * 110),
      keyName: '—', mode: P.v > 0.5 ? 'major' : 'minor',
      centroid: Math.round(900 + P.v * 1800),
      scores, confidence: 0.42, prior: true
    };
  }

  root.Recommender = { recommend, faceVector, STRATEGIES, priorFeatures, dist2, cosine };
})(window);
