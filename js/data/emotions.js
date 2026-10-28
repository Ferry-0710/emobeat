/* ============================================================================
 * emotions.js — 七类情绪的统一定义 (人脸侧 / 音乐侧共用)
 * ==========================================================================*/
(function (root) {
  const EMOTIONS = {
    happy: {
      key: 'happy', cn: '快乐', en: 'Happy', emoji: '😄', color: '#F59E0B', color2: '#FDE68A',
      // 人脸情绪在 Thayer 环状模型中的坐标 (效价, 唤醒度, 粗糙度)
      face: { v: 0.86, a: 0.66, r: 0.15 },
      // 曲库检索用的风格线索
      genres: ['流行 Pop', '放克 Funk', '电子舞曲 EDM'],
      query: 'happy upbeat pop feel good',
      tip: '大调、快节奏、明亮音色。适合继续保持愉悦，也可以来一首轻快的 Funk 让心情更有弹性。',
      faceDesc: '嘴角上扬、脸颊抬起、眼周出现"鱼尾纹"，是最容易识别的一类表情。'
    },
    sad: {
      key: 'sad', cn: '悲伤', en: 'Sad', emoji: '😢', color: '#3B82F6', color2: '#BFDBFE',
      face: { v: 0.10, a: 0.22, r: 0.18 },
      genres: ['抒情 Ballad', '钢琴独奏', '氛围 Ambient'],
      query: 'sad piano ballad emotional',
      tip: '小调、慢速、低亮度。悲伤时"共鸣"往往比"强颜欢笑"更有效：先用慢速小调接住情绪，再逐步过渡。',
      faceDesc: '嘴角下拉、内眉上扬并聚拢、上眼睑下垂，常伴随视线向下。'
    },
    angry: {
      key: 'angry', cn: '愤怒', en: 'Angry', emoji: '😠', color: '#EF4444', color2: '#FECACA',
      face: { v: 0.12, a: 0.88, r: 0.70 },
      genres: ['摇滚 Rock', '金属 Metal', '工业 Industrial'],
      query: 'hard rock aggressive energy',
      tip: '小调 + 强失真 + 高速底鼓。先用同频的高唤醒音乐"宣泄"，也可以选择舒缓策略快速降唤醒。',
      faceDesc: '眉毛压低并内聚、鼻翼扩张、下颌咬紧、嘴唇变薄。'
    },
    fearful: {
      key: 'fearful', cn: '恐惧', en: 'Fearful', emoji: '😨', color: '#8B5CF6', color2: '#DDD6FE',
      face: { v: 0.18, a: 0.72, r: 0.45 },
      genres: ['暗黑氛围 Dark Ambient', '悬疑配乐', '实验 Electronic'],
      query: 'dark ambient tense atmospheric',
      tip: '不协和音程、稀疏织体、不规则节奏。推荐先匹配"紧张感"再逐步降低唤醒度（音乐引导想象）。',
      faceDesc: '眼睛睁大露出眼白、眉毛整体上扬并向中间聚拢、嘴微张。'
    },
    disgusted: {
      key: 'disgusted', cn: '厌恶', en: 'Disgusted', emoji: '🤢', color: '#65A30D', color2: '#D9F99D',
      face: { v: 0.10, a: 0.55, r: 0.55 },
      genres: ['实验 Experimental', '噪音 Noise', '怪诞 Odd'],
      query: 'weird experimental industrial quirky',
      tip: '半音化、失谐、怪异音程。厌恶常与"排斥"相关，建议用中速干净音色逐步替换不适感。',
      faceDesc: '鼻梁皱起、上唇单侧抬起、眼睑眯起、头部后仰。'
    },
    surprised: {
      key: 'surprised', cn: '惊讶', en: 'Surprised', emoji: '😲', color: '#EC4899', color2: '#FBCFE8',
      face: { v: 0.55, a: 0.85, r: 0.30 },
      genres: ['管弦 Orchestral', 'Électro Swing', '鼓组打击'],
      query: 'energetic orchestral epic surprise',
      tip: '突强、切分音、钟琴式音色。惊讶是短时情绪，适合用节奏鲜明的曲子把注意力接住。',
      faceDesc: '眉毛高扬、眼睛睁大、下颌下垂呈"O"型嘴。'
    },
    neutral: {
      key: 'neutral', cn: '平静', en: 'Neutral', emoji: '😐', color: '#64748B', color2: '#E2E8F0',
      face: { v: 0.50, a: 0.40, r: 0.15 },
      genres: ['Lo-Fi', '轻爵士 Jazz', '新世纪 New Age'],
      query: 'lofi chill mellow instrumental',
      tip: '中速、温和音色、稳定律动。平静状态适合作为工作/学习背景，也可选择"提振"策略。',
      faceDesc: '面部肌肉放松、嘴唇闭合平直、眉毛处于静息位。'
    }
  };

  const ORDER = ['happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised', 'neutral'];

  // 演示用写实人脸: 每种情绪 3 张
  const FACE_SAMPLES = {};
  ORDER.forEach(e => {
    FACE_SAMPLES[e] = [1, 2, 3].map(i => ({
      emotion: e, index: i, src: 'assets/faces/' + e + '-' + i + '.jpg',
      label: EMOTIONS[e].cn + ' ' + i
    }));
  });

  const api = { EMOTIONS, ORDER, FACE_SAMPLES };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  window.EMO_DATA = api;
})(this);
