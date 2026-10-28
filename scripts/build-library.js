// 预计算本地曲库的音频情绪特征, 写回 js/data/library.json
const fs=require('fs'),path=require('path'),A=require('../js/audio/melody-analyzer.js');
const ROOT=path.join(__dirname,'..');
const libPath=path.join(ROOT,'js/data/library.json');
const lib=JSON.parse(fs.readFileSync(libPath,'utf8'));
function readWav(f){const b=fs.readFileSync(f);let p=12,fmt,data;while(p+8<=b.length){const id=b.toString('ascii',p,p+4),s=b.readUInt32LE(p+4);if(id==='fmt ')fmt={ch:b.readUInt16LE(p+10),sr:b.readUInt32LE(p+12)};else if(id==='data')data=b.subarray(p+8,p+8+s);p+=8+s+(s%2);}const n=data.length/2,o=new Float32Array(n);for(let i=0;i<n;i++)o[i]=data.readInt16LE(i*2)/32768;return{pcm:o,sr:fmt.sr,ch:fmt.ch};}
const rows=[];let ok=0;
for(const t of lib){
  const {pcm,sr,ch}=readWav(path.join(ROOT,t.file));
  const r=A.analyzePcm(pcm,sr,{channels:ch});
  delete r.analyzeMs; delete r.frames;
  t.features=r;
  t.algorithmEmotion=r.emotion;
  const hit=r.emotion===t.emotion; ok+=hit?1:0;
  rows.push({id:t.id,truth:t.emotion,pred:r.emotion,conf:r.confidence,bpm:r.bpm,bpmTruth:t.bpm,key:r.keyName,hit});
}
fs.writeFileSync(libPath,JSON.stringify(lib,null,2),'utf8');
const bpmErr=rows.map(r=>Math.abs(r.bpm-r.bpmTruth)/r.bpmTruth);
const out={
  total:rows.length, emotionAcc:+(ok/rows.length).toFixed(3), ok,
  bpmMae:+(bpmErr.reduce((a,b)=>a+b,0)/bpmErr.length).toFixed(4),
  keyAcc:+(rows.filter(r=>r.key.split(' ')[1]===lib.find(l=>l.id===r.id).mode).length/rows.length).toFixed(3),
  rows, generatedAt:new Date().toISOString()
};
fs.writeFileSync(path.join(ROOT,'js/data/validation.json'),JSON.stringify(out,null,2),'utf8');
console.log('emotion acc',out.emotionAcc,'bpm mae',(out.bpmMae*100).toFixed(1)+'%','mode acc',out.keyAcc);
rows.forEach(r=>console.log(r.hit?' ✓':' ✗',r.id.padEnd(18),r.truth.padEnd(10),'->',r.pred.padEnd(10),r.conf.toFixed(2),r.bpm));
