// 验证旋律情绪分析算法: 读取本地 WAV, 对比 library.json 中的标注
const fs = require('fs');
const path = require('path');
const A = require('../js/audio/melody-analyzer.js');

const ROOT = path.join(__dirname, '..');
const lib = JSON.parse(fs.readFileSync(path.join(ROOT, 'js/data/library.json'), 'utf8'));

function readWav(file) {
  const b = fs.readFileSync(file);
  let p = 12, fmt = null, data = null;
  while (p + 8 <= b.length) {
    const id = b.toString('ascii', p, p + 4);
    const size = b.readUInt32LE(p + 4);
    if (id === 'fmt ') {
      fmt = {
        ch: b.readUInt16LE(p + 10), sr: b.readUInt32LE(p + 12),
        bits: b.readUInt16LE(p + 22)
      };
    } else if (id === 'data') { data = b.subarray(p + 8, p + 8 + size); }
    p += 8 + size + (size % 2);
  }
  const n = data.length / 2;
  const f = new Float32Array(n);
  for (let i = 0; i < n; i++) f[i] = data.readInt16LE(i * 2) / 32768;
  return { pcm: f, sr: fmt.sr, ch: fmt.ch };
}

let ok = 0, tot = 0;
const rows = [];
for (const t of lib) {
  const { pcm, sr, ch } = readWav(path.join(ROOT, t.file));
  const r = A.analyzePcm(pcm, sr, { channels: ch });
  const hit = r.emotion === t.emotion;
  ok += hit ? 1 : 0; tot++;
  rows.push({
    id: t.id, truth: t.emotion, pred: r.emotion, conf: (r.confidence * 100).toFixed(1),
    v: r.valence.toFixed(2), a: r.arousal.toFixed(2), r: r.roughness.toFixed(2),
    bpm: r.bpm, bpmTruth: t.bpm, key: r.keyName, keyTruth: t.root + ' ' + t.mode,
    ms: r.analyzeMs, hit: hit ? 'OK' : 'MISS'
  });
}
console.table(rows);
console.log(`\n情绪分类: ${ok}/${tot} = ${(ok / tot * 100).toFixed(0)}%`);
const bpmErr = rows.map(r => Math.abs(r.bpm - r.bpmTruth) / r.bpmTruth);
console.log('BPM 平均相对误差:', (bpmErr.reduce((a, b) => a + b, 0) / bpmErr.length * 100).toFixed(1) + '%');
const keyOk = rows.filter(r => r.key.split(' ')[0] === r.keyTruth.split(' ')[0]).length;
console.log('主音命中:', keyOk + '/' + tot, ' 调式命中:', rows.filter(r => r.key.split(' ')[1] === r.keyTruth.split(' ')[1]).length + '/' + tot);
console.log('平均耗时:', (rows.reduce((a, b) => a + b.ms, 0) / rows.length).toFixed(0) + 'ms/首');
