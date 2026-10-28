/* ============================================================================
 * app.js — Vue3 应用逻辑
 * ==========================================================================*/
(function () {
  'use strict';

  const EMO = window.EMO_DATA;
  const EMOTIONS = EMO.EMOTIONS;
  const ORDER = EMO.ORDER;
  const UNIFORM = ORDER.reduce((o, k) => { o[k] = 1 / ORDER.length; return o; }, {});

  const SOURCES = [
    {
      key: 'image', name: '图片', icon: '🖼', hint: '上传一张人脸照片',
      desc: '支持 JPG / PNG / WebP，也可以直接点击下方演示人脸素材。单帧识别，结果最稳定。'
    },
    {
      key: 'video', name: '视频', icon: '🎬', hint: '打开本地视频文件',
      desc: '支持 mp4 / webm / mov。程序逐帧检测，可观察同一人物情绪随剧情变化。'
    },
    {
      key: 'camera', name: '摄像头', icon: '📷', hint: '开启实时摄像头',
      desc: '需要授权摄像头权限。建议正对光源，脸部占画面 20% 以上，检测更稳定。'
    },
    {
      key: 'screen', name: '屏幕', icon: '🖥', hint: '共享屏幕 / 窗口',
      desc: '可捕获会议、播放器中的画面。共享整个屏幕比共享标签页更容易捕获视频画面。'
    }
  ];

  const App = {
    data() {
      return {
        view: 'studio',
        sourceList: SOURCES,
        source: 'image',
        precision: 224,
        alpha: 0.35,

        imgSrc: '',
        mediaReady: false,
        live: false,
        busy: false,
        dragging: false,

        model: { loaded: false, loading: true, progress: 0, text: '准备加载模型…', error: null, backend: '' },
        cur: { stable: 'neutral', dominant: 'neutral', stableFrames: 0, quality: 0 },
        faces: [],
        faceInfo: { count: 0, ms: 0, fps: 0, engine: '', usedSize: 0 },

        library: [],
        validation: null,
        rec: { list: [], target: { v: 0.5, a: 0.4, r: 0.2 }, faceV: { v: 0.5, a: 0.4, r: 0.2 }, poolSize: 0 },
        strategies: Object.keys(window.Recommender.STRATEGIES).map(k => window.Recommender.STRATEGIES[k]),
        strategy: 'match',
        recent: [],
        hasEmotion: false,
        emotionProbs: UNIFORM,
        lastResult: null,

        pickEmotion: 'all',
        showSamples: true,

        player: { track: null, playing: false, current: 0, duration: 0, volume: 0.85 },
        queue: [],

        online: { kw: '', loading: false, results: [], error: null },

        libAnalyzing: false,
        libProgress: 0,
        libMsg: '',
        libAcc: null,
        focusTrack: null,

        perf: { total: '—', audio: '—' },
        pitchNames: ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'],
        toast: ''
      };
    },

    computed: {
      emotions() { return ORDER; },
      activeSource() { return SOURCES.find(s => s.key === this.source) || SOURCES[0]; },
      isLiveMode() { return this.source !== 'image'; },
      probs() {
        return (this.hasEmotion && this.emotionProbs) ? this.emotionProbs : UNIFORM;
      },
      displayEmotion() { return this.cur.stable || 'neutral'; },
      confidence() {
        const p = this.probs[this.displayEmotion];
        return p == null ? 0 : p;
      },
      recReady() { return this.library.length > 0; },
      currentStrategy() { return window.Recommender.STRATEGIES[this.strategy]; },
      visibleSamples() {
        const all = [];
        ORDER.forEach(k => EMO.FACE_SAMPLES[k].forEach(s => all.push(s)));
        return this.pickEmotion === 'all' ? all : all.filter(s => s.emotion === this.pickEmotion);
      },
      seekPct() {
        const d = this.player.duration || 0;
        return d > 0 ? Math.min(100, (this.player.current / d) * 100) : 0;
      }
    },

    watch: {
      alpha() { this.resetSmoother(); },
      precision() { this.faceInfo.engine = 'TINY'; }
    },

    async mounted() {
      this.audio = this.$refs.audioEl;
      this.smoother = new window.FaceEngine.EmotionSmoother(this.alpha);
      this.emotionProbs = UNIFORM;

      await this.loadData();
      // 此处不再预先推荐: 还没有任何情绪结果时 probs 是均匀分布(各 1/7),
      // 加权出的目标坐标 (V/A≈0.34/0.61) 没有实际含义, 会推出与情绪无关的曲子。
      // 首个识别结果到达后 applySmoothed 会立即触发推荐, 期间界面显示空状态提示。
      this.loadModels();
      window.addEventListener('resize', this.redrawOverlay);
    },

    beforeUnmount() {
      window.removeEventListener('resize', this.redrawOverlay);
      this.stopLive();
    },

    methods: {
      /* ------------------------------------------------------- 基础工具 */
      emo(k) { return EMOTIONS[k] || EMOTIONS.neutral; },
      fmtTime(s) {
        s = Math.max(0, Math.floor(s || 0));
        return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0');
      },
      notify(msg) {
        this.toast = msg;
        clearTimeout(this._toastTimer);
        this._toastTimer = setTimeout(() => { this.toast = ''; }, 2600);
      },

      /* --------------------------------------------------------- 数据加载 */
      async loadData() {
        try {
          const res = await fetch('js/data/library.json');
          const list = await res.json();
          this.library = list.map(t => Object.assign({}, t, { source: 'local' }));
        } catch (e) {
          this.notify('曲库加载失败：' + e.message);
        }
        try {
          const v = await fetch('js/data/validation.json');
          this.validation = await v.json();
        } catch (e) { /* 可选文件 */ }
      },

      async loadModels() {
        this.model.loading = true;
        this.model.text = '加载本地模型…';
        try {
          await window.FaceEngine.load((text, pct) => {
            this.model.text = text;
            this.model.progress = pct;
          });
          this.model.loaded = true;
          this.model.backend = window.FaceEngine.state.backend || '';
          this.faceInfo.engine = 'TINY';
          if (!this.imgSrc && this.source === 'image') {
            // 首次进入自动载入一张演示人脸, 让推荐列表立刻有内容
            this.useSample(EMO.FACE_SAMPLES.happy[0]);
          }
        } catch (e) {
          this.model.error = e.message || String(e);
          this.model.text = '模型加载失败';
        } finally {
          this.model.loading = false;
        }
      },

      /* ------------------------------------------------------- 输入源切换 */
      setSource(key) {
        if (this.source === key) return;
        this.stopLive(true);
        this.source = key;
        this.hasEmotion = false;
        this.emotionProbs = UNIFORM;
        this.resetSmoother();
        this.faceInfo.count = 0;
        this.faceInfo.ms = 0;
        this.faceInfo.fps = 0;
        this.clearOverlay();
        if (key === 'image') {
          this.mediaReady = !!this.imgSrc;
        } else {
          this.imgSrc = '';
          this.mediaReady = false;
          const v = this.$refs.videoEl;
          if (v) { v.removeAttribute('src'); v.srcObject = null; v.load(); }
        }
      },

      setPrecision(p) {
        this.precision = p;
        if (this.source === 'image' && this.imgSrc) this.runImage();
      },

      resetSmoother() {
        this.smoother = new window.FaceEngine.EmotionSmoother(this.alpha);
      },

      /* ----------------------------------------------------------- 图片 */
      pickFile() { this.$refs.fileImage.click(); },

      onFilePick(e) {
        const f = e.target.files && e.target.files[0];
        if (f) this.loadImageFile(f);
        e.target.value = '';
      },

      loadImageFile(file) {
        if (!/^image\//.test(file.type)) { this.notify('请选择图片文件'); return; }
        const url = URL.createObjectURL(file);
        this.setSource('image');
        this.imgSrc = url;
        this.mediaReady = false;
        this.hasEmotion = false;
        this.resetSmoother();
      },

      async onDrop(e) {
        this.dragging = false;
        const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        if (!f) return;
        if (/^image\//.test(f.type)) this.loadImageFile(f);
        else if (/^video\//.test(f.type)) await this.loadVideoFile(f);
        else this.notify('只支持图片或视频文件');
      },

      stageClick() {
        if (this.source === 'image' && !this.imgSrc) this.pickFile();
      },

      useSample(s) {
        this.setSource('image');
        this.imgSrc = s.src;
        this.mediaReady = false;
        this.hasEmotion = false;
        this.resetSmoother();
      },

      onImageLoad() {
        this.mediaReady = true;
        this.$nextTick(() => this.runImage());
      },

      async runImage() {
        if (!this.model.loaded) { this.notify('模型尚未就绪，请稍候…'); return; }
        const img = this.$refs.imgEl;
        if (!img || !img.complete) return;
        try {
          const t0 = performance.now();
          let res = await window.FaceEngine.detect(img, { inputSize: this.precision });
          // 单帧识别允许重试: 证件照类竖幅人像在 160/224 下容易漏检, 自动升到 320 再试一次
          if (!res.faces.length && this.precision < 320) {
            res = await window.FaceEngine.detect(img, { inputSize: 320 });
            this.faceInfo.usedSize = res.faces.length ? 320 : this.precision;
          } else {
            this.faceInfo.usedSize = this.precision;
          }
          const total = Math.round(performance.now() - t0);
          this.faceInfo.ms = total;
          this.faceInfo.count = res.faces.length;
          this.perf.total = total;
          this.lastResult = res;
          if (!res.faces.length) {
            // 检不到脸时必须清空上一次的结果, 否则界面会停留在上一张照片的情绪上
            this.clearOverlay();
            this.hasEmotion = false;
            this.emotionProbs = UNIFORM;
            this.cur = { stable: null, dominant: null, stableFrames: 0, quality: 0 };
            this.notify('未检测到人脸，试试正脸、光线更好的照片');
            return;
          }
          const face = res.faces[0];
          const sm = this.smoother.push(face.expressions, face.score, 1);
          this.applySmoothed(sm);
          this.drawOverlay(res);
        } catch (e) {
          this.notify('识别失败：' + e.message);
        }
      },

      /* ------------------------------------------------ 视频 / 摄像头 / 屏幕 */
      async startLive() {
        if (!this.model.loaded) { this.notify('模型尚未就绪，请稍候…'); return; }
        if (this.source === 'video') { this.$refs.fileVideo.click(); return; }
        const video = this.$refs.videoEl;
        try {
          if (this.source === 'camera') {
            this.stream = await navigator.mediaDevices.getUserMedia({
              video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
              audio: false
            });
            video.srcObject = this.stream;
          } else if (this.source === 'screen') {
            this.stream = await navigator.mediaDevices.getDisplayMedia({
              video: { frameRate: { ideal: 15 } }, audio: false
            });
            video.srcObject = this.stream;
            this.stream.getVideoTracks().forEach(tr => {
              tr.addEventListener('ended', () => this.stopLive());
            });
          }
          await video.play().catch(() => {});
          this.mediaReady = true;
          this.live = true;
          this.resetSmoother();
          this.startLoop();
        } catch (e) {
          this.notify('无法打开视频源：' + (e.message || e.name));
          this.stopLive();
        }
      },

      onVideoPick(e) {
        const f = e.target.files && e.target.files[0];
        if (f) this.loadVideoFile(f);
        e.target.value = '';
      },

      async loadVideoFile(file) {
        if (!/^video\//.test(file.type)) { this.notify('请选择视频文件'); return; }
        // 拖拽载入时可能仍停留在「图片」源，此时 <video> 节点尚未渲染
        if (this.source === 'image' || !this.$refs.videoEl) {
          this.setSource('video');
          await this.$nextTick();
        }
        const video = this.$refs.videoEl;
        if (!video) { this.notify('视频播放器尚未就绪，请重试'); return; }
        video.srcObject = null;
        video.src = URL.createObjectURL(file);
        video.loop = true;
        this.mediaReady = true;
        this.live = true;
        this.resetSmoother();
        try { await video.play(); } catch (e) { /* 自动播放可能被阻止 */ }
        this.startLoop();
      },

      startLoop() {
        // 生成令牌: 新一轮循环会让旧循环在下一次 rAF 自动退出, 避免叠加
        const token = (this._loopToken || 0) + 1;
        this._loopToken = token;
        this._loopAlive = true;
        let lastT = 0;
        // 滚动 FPS: 统计最近 1 秒内完成的推理帧数
        this._fpsWin = { t0: performance.now(), n: 0 };
        const loop = async () => {
          if (!this._loopAlive || !this.live || this._loopToken !== token) return;
          const now = performance.now();
          // 帧级节流: 目标 ~22 FPS, 避免推理排队造成延迟累积
          if (!this.busy && now - lastT > 1000 / 22) {
            lastT = now;
            await this.detectFrame();
            const w = this._fpsWin;
            w.n++;
            if (now - w.t0 >= 1000) {
              this.faceInfo.fps = Math.round(w.n * 1000 / (now - w.t0));
              w.t0 = now; w.n = 0;
            }
          }
          requestAnimationFrame(loop);
        };
        requestAnimationFrame(loop);
      },

      stopLoop() { this._loopAlive = false; },

      async detectFrame() {
        const video = this.$refs.videoEl;
        if (!video || video.readyState < 2) return;
        this.busy = true;
        try {
          const res = await window.FaceEngine.detect(video, { inputSize: this.precision });
          this.faceInfo.ms = Math.round(res.ms);
          this.faceInfo.count = res.faces.length;
          this.perf.total = Math.round(res.ms);
          this.lastResult = res;
          if (res.faces.length) {
            const face = res.faces[0];
            const sm = this.smoother.push(face.expressions, face.score, 3);
            this.applySmoothed(sm);
          }
          this.drawOverlay(res);
        } catch (e) {
          // 忽略单帧错误 (例如视频切换瞬间)
        } finally {
          this.busy = false;
        }
      },

      stopLive(keepFrame) {
        this.stopLoop();
        if (this.stream) {
          this.stream.getTracks().forEach(t => t.stop());
          this.stream = null;
        }
        const video = this.$refs.videoEl;
        if (video && !keepFrame) {
          video.pause();
          if (this.source === 'camera' || this.source === 'screen') {
            video.srcObject = null;
            this.mediaReady = false;
          }
        }
        this.clearOverlay();
      },

      /* --------------------------------------------------------- 结果应用 */
      applySmoothed(sm) {
        const prev = this.cur.stable;
        const hadEmotion = this.hasEmotion;
        this.cur = {
          stable: sm.stable,
          dominant: sm.dominant,
          stableFrames: sm.stableFrames,
          quality: sm.quality
        };
        this.emotionProbs = Object.assign({}, sm.probs);
        this.hasEmotion = true;
        // 除了"情绪标签翻转", 从"无结果"→"有结果"也必须重算推荐:
        // 换图/重新识别后若标签恰好相同(例如连点两次同类演示图), 只比较 prev 会漏掉更新,
        // 列表将一直停留在首屏那次基于均匀概率(各 1/7, 对应 V/A≈0.34/0.61)的推荐上。
        if (prev !== sm.stable || !hadEmotion) this.scheduleRecommend(260);
      },

      scheduleRecommend(delay) {
        clearTimeout(this._recTimer);
        this._recTimer = setTimeout(() => this.doRecommend(), delay || 320);
      },

      doRecommend() {
        if (!this.library.length) return;
        const tracks = this.library.concat(
          this.online.results.filter(t => t.inPool && t.features)
        );
        const res = window.Recommender.recommend({
          probs: this.probs,
          tracks,
          emoData: EMO,
          strategy: this.strategy,
          recentIds: this.recent,
          topN: 6
        });
        this.rec = res;
        if (!this.queue.length) this.queue = res.list.map(r => r.track);
      },

      setStrategy(k) {
        this.strategy = k;
        this.doRecommend();
      },

      /* ----------------------------------------------------------- 绘制 */
      drawOverlay(res) {
        const canvas = this.$refs.overlayEl;
        const video = this.$refs.videoEl;
        const img = this.$refs.imgEl;
        const source = this.source === 'image' ? img : video;
        if (!canvas || !source) return;
        if (this.source === 'image' && (!img || !img.complete)) return;
        const result = res || this.lastResult || { faces: [] };
        const label = EMOTIONS[this.cur.stable] ? EMOTIONS[this.cur.stable].cn + ' ' +
          (this.confidence * 100).toFixed(0) + '%' : '';
        window.FaceEngine.drawOverlay(canvas, source, result, label,
          EMOTIONS[this.cur.stable] ? EMOTIONS[this.cur.stable].color : '#6366F1');
      },

      redrawOverlay() {
        if (this.hasEmotion) this.drawOverlay();
      },

      clearOverlay() {
        const canvas = this.$refs.overlayEl;
        if (canvas) {
          const ctx = canvas.getContext('2d');
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
      },

      /* --------------------------------------------------------- 在线检索 */
      async searchOnline() {
        const kw = (this.online.kw || '').trim() ||
          EMOTIONS[this.displayEmotion].query;
        this.online.loading = true;
        this.online.error = null;
        try {
          const url = 'https://itunes.apple.com/search?term=' + encodeURIComponent(kw) +
            '&media=music&limit=12&country=us';
          const res = await fetch(url);
          if (!res.ok) throw new Error('HTTP ' + res.status);
          const json = await res.json();
          const list = (json.results || []).filter(x => x.previewUrl).map(x => ({
            id: 'it-' + x.trackId,
            title: x.trackName,
            artist: x.artistName,
            artwork: (x.artworkUrl100 || '').replace('100x100bb', '200x200bb'),
            preview: x.previewUrl,
            genre: x.primaryGenreName || '',
            duration: 30,
            source: 'online',
            busy: false,
            inPool: false,
            features: null
          }));
          this.online.results = list;
          if (!list.length) this.notify('没有找到带试听的结果，换个关键词试试');
          else this.notify('已检索到 ' + list.length + ' 首（点击"分析"可本地分析音频情绪）');
        } catch (e) {
          this.online.error = e.message;
          this.online.results = [];
        } finally {
          this.online.loading = false;
        }
      },

      searchByEmotion() {
        this.online.kw = EMOTIONS[this.displayEmotion].query;
        this.searchOnline();
      },

      async analyzeOnline(t) {
        if (t.busy) return;
        t.busy = true;
        try {
          const buf = await this.fetchAndDecode(t.preview);
          const t0 = performance.now();
          const feat = window.MelodyAnalyzer.analyzePcm(buf.pcm, buf.sr, { channels: buf.ch });
          this.perf.audio = feat.analyzeMs;
          t.features = feat;
          t.analyzedMs = Math.round(performance.now() - t0);
          if (t.inPool) this.doRecommend();
          this.notify('「' + t.title + '」本地分析完成：' +
            EMOTIONS[feat.emotion].cn + ' · ' + Math.round(feat.bpm) + ' BPM');
        } catch (e) {
          this.notify('音频分析失败：' + (e.message || e));
        } finally {
          t.busy = false;
        }
      },

      async fetchAndDecode(url) {
        const res = await fetch(url);
        if (!res.ok) throw new Error('音频下载失败 HTTP ' + res.status);
        const arr = await res.arrayBuffer();
        const AC = window.AudioContext || window.webkitAudioContext;
        if (!this._ac) this._ac = new AC();
        const audioBuf = await this._ac.decodeAudioData(arr.slice(0));
        const ch = audioBuf.numberOfChannels;
        let pcm;
        if (ch === 1) pcm = audioBuf.getChannelData(0);
        else {
          const n = audioBuf.length;
          pcm = new Float32Array(n * ch);
          for (let c = 0; c < ch; c++) {
            const d = audioBuf.getChannelData(c);
            for (let i = 0; i < n; i++) pcm[i * ch + c] = d[i];
          }
        }
        return { pcm, sr: audioBuf.sampleRate, ch };
      },

      togglePool(t) {
        if (!t.features) {
          t.inPool = true;
          this.analyzeOnline(t).then(() => this.doRecommend());
          return;
        }
        t.inPool = !t.inPool;
        this.doRecommend();
      },

      /* ------------------------------------------------------- 本地曲库分析 */
      async analyzeOne(t) {
        this.focusTrack = t;
        if (t.features) return;
        await this.reanalyzeTrack(t, true);
      },

      async reanalyzeTrack(t) {
        const url = t.file;
        const buf = await this.fetchAndDecode(url);
        t.features = window.MelodyAnalyzer.analyzePcm(buf.pcm, buf.sr, { channels: buf.ch });
        this.perf.audio = t.features.analyzeMs;
        return t.features;
      },

      async reanalyzeLibrary() {
        if (this.libAnalyzing) return;
        this.libAnalyzing = true;
        this.libMsg = '';
        this.libProgress = 0;
        let ok = 0, totalMs = 0;
        try {
          for (const t of this.library) {
            try {
              const f = await this.reanalyzeTrack(t);
              t.liveFeatures = f;
              totalMs += f.analyzeMs;
              if (f.emotion === t.emotion) ok++;
            } catch (e) { /* 单曲失败不影响流程 */ }
            this.libProgress++;
            await new Promise(r => setTimeout(r, 0));
          }
          this.libMsg = '浏览器端重新分析完成：' + this.library.length + ' 首，总耗时 ' +
            Math.round(totalMs) + ' ms（平均 ' + Math.round(totalMs / this.library.length) +
            ' ms/首），情绪与标注一致 ' + ok + '/' + this.library.length +
            '（' + Math.round(ok / this.library.length * 100) + '%）。';
          this.doRecommend();
        } finally {
          this.libAnalyzing = false;
        }
      },

      /* ----------------------------------------------------------- 播放器 */
      play(track) {
        // 只接受形如曲目对象的入参, 避免把推荐条目包装对象误写进播放器状态
        if (!track || !track.id) return;
        const src = track.file || track.preview;
        if (!src) { this.notify('该曲目没有可播放地址'); return; }
        this.player.track = track;
        this.player.current = 0;
        this.player.duration = 0;
        this.audio.src = src;
        this.audio.volume = this.player.volume;
        this.audio.play().catch(e => this.notify('播放失败：' + e.message + '（在线音频可能被拦截）'));
        this.recent = [track.id].concat(this.recent.filter(x => x !== track.id)).slice(0, 10);
      },

      togglePlay() {
        if (!this.player.track) return;
        if (this.audio.paused) this.audio.play(); else this.audio.pause();
      },

      playNext() {
        const list = this.rec.list.length ? this.rec.list.map(r => r.track) : this.queue;
        if (!list.length) return;
        const idx = list.findIndex(t => this.player.track && t.id === this.player.track.id);
        this.play(list[(idx + 1) % list.length]);
      },

      playPrev() {
        const list = this.rec.list.length ? this.rec.list.map(r => r.track) : this.queue;
        if (!list.length) return;
        const idx = list.findIndex(t => this.player.track && t.id === this.player.track.id);
        this.play(list[(idx - 1 + list.length) % list.length]);
      },

      closePlayer() {
        this.audio.pause();
        this.audio.removeAttribute('src');
        this.player.track = null;
        this.player.playing = false;
      },

      onTime() {
        this.player.current = this.audio.currentTime || 0;
      },

      onMeta() {
        this.player.duration = this.audio.duration || 0;
      },

      seekTo(e) {
        const el = this.$refs.seekEl;
        if (!el || !this.player.duration) return;
        const rect = el.getBoundingClientRect();
        const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
        this.audio.currentTime = ratio * this.player.duration;
      },

      applyVolume() {
        if (this.audio) this.audio.volume = this.player.volume;
      }
    }
  };

  const app = Vue.createApp(App);
  const vm = app.mount('#app');
  // 调试句柄: 控制台可用 __app 查看状态/手动调用方法(也供自动化测试使用)
  window.__app = vm;
})();
