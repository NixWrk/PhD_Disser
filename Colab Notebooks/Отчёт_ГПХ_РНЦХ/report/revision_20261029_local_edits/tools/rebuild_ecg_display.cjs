// Display-only reproduction of render_visuals.cjs:21-22, 79-105.
// Offline; no recording, filtering, detection, fitted result or common manifest is written.
const fs=require('fs'),path=require('path'),crypto=require('crypto'),assert=require('assert/strict');
const deps='C:/Users/ПАВЛУША/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {chromium}=require(path.join(deps,'playwright')),sharp=require(path.join(deps,'sharp'));
const revision=path.resolve(__dirname,'..'),root=path.resolve(revision,'../../../..');
const base=path.resolve(revision,'../revision_20261029_author_review');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const relative=p=>path.relative(root,p).replaceAll('\\','/');
function readJSON(s,pos){while(/[\s,]/.test(s[pos]))pos++;let start=pos,depth=0,str=false,escape=false;for(;pos<s.length;pos++){let c=s[pos];if(str){if(escape)escape=false;else if(c==='\\')escape=true;else if(c==='"')str=false;}else if(c==='"')str=true;else if(c==='['||c==='{')depth++;else if(c===']'||c==='}'){depth--;if(!depth)return[JSON.parse(s.slice(start,pos+1)),pos+1];}}throw Error('JSON boundary');}
function scenes(raw){let a=[],rx=/Plotly\.newPlot\(\s*("[^"\n]+")\s*,\s*/g,m;while(m=rx.exec(raw)){let[data,end]=readJSON(raw,rx.lastIndex),[layout,last]=readJSON(raw,end);a.push({div:JSON.parse(m[1]),data,layout});rx.lastIndex=last;}return a;}
function values(a){if(Array.isArray(a))return a;if(a.dtype==='f8'){let b=Buffer.from(a.bdata,'base64');return Array.from({length:b.length/8},(_,i)=>b.readDoubleLE(i*8));}throw Error('Unexpected ordinate encoding: '+JSON.stringify(a).slice(0,80));}
function minus(a){if(Array.isArray(a))return a.map(v=>-v);assert.equal(a.dtype,'f8');let b=Buffer.from(a.bdata,'base64');for(let i=0;i<b.length;i+=8)b.writeDoubleLE(-b.readDoubleLE(i),i);return{...a,bdata:b.toString('base64')};}
async function main(){
 const sc=path.join(root,'Colab Notebooks/scorpion_v2');
 const source=path.join(sc,fs.readdirSync(sc).find(x=>x.startsWith('33.40')&&x.endsWith('.html')));
 const raw=fs.readFileSync(source,'utf8'),ss=scenes(raw),s4=ss[3],s5=ss[4];
 const library=[...raw.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(x=>x[1]).find(x=>x.includes('plotly.js v')&&x.length>100000);
 assert(library);assert.deepEqual(s4.data[0].x,s5.data[0].x);
 let ecg=structuredClone(s4.data[1]),qrs=structuredClone(s4.data[2]),tt=structuredClone(s5.data[0]),side=structuredClone(s4.data[0]);
 [ecg,qrs].forEach(x=>{x.xaxis='x';x.yaxis='y';x.showlegend=true;});ecg.name='ЭКГ';qrs.name='QRS-кандидаты';
 [tt,side].forEach(x=>{x.xaxis='x2';x.yaxis='y2';x.showlegend=true;});tt.name='ТТРКГ 2×2';side.name='Боковая сборка 140 мм';tt.line={...(tt.line||{}),color:'#1769a7'};side.line={...(side.line||{}),color:'#b35806'};
 const original=[ecg,qrs,tt,side],display=structuredClone(original);
 for(let i of [0,1]){display[i].y=minus(original[i].y);assert.deepEqual(values(display[i].y),values(original[i].y).map(v=>-v));let a=structuredClone(display[i]);a.y=original[i].y;assert.deepEqual(a,original[i]);}
 assert.deepEqual(display.slice(2),original.slice(2));
 const l={xaxis:{domain:[0,1],anchor:'y',showticklabels:false},yaxis:{domain:[.57,1],title:'ЭКГ, мВ'},xaxis2:{domain:[0,1],anchor:'y2',title:'Время от первого QRS-кандидата, с'},yaxis2:{domain:[0,.43],title:'Пульсовой импеданс, мОм'},annotations:[]};
 l.width=1800;l.height=1000;l.font={family:'Arial',size:32,color:'#17212b'};l.title={text:'У1 · запись 001 · четыре сохранённых цикла',x:.5,font:{size:38}};l.margin={l:135,r:90,t:120,b:145};l.updatemenus=[];l.sliders=[];l.legend={font:{size:27},orientation:'h',y:-.16,x:0};
 for(const[k,v]of Object.entries(l))if(/^[xy]axis\d*$/.test(k)){v.tickfont={size:29};if(typeof v.title==='string')v.title={text:v.title};v.title={...(v.title||{}),font:{size:31}};v.automargin=true;}
 const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true,args:['--disable-gpu','--allow-file-access-from-files']});
 let record;
 try{
  const page=await browser.newPage({viewport:{width:1800,height:1100}});await page.route('**/*',r=>r.abort());await page.setContent('<html><body style="margin:0;background:white"><div id="plot"></div></body></html>');await page.addScriptTag({content:library});
  async function render(d){let r=await page.evaluate(async({d,l})=>{await Plotly.newPlot('plot',d,l,{displayModeBar:false,staticPlot:true});return await Plotly.toImage('plot',{format:'png',width:l.width,height:l.height,scale:1.7});},{d,l});return Buffer.from(r.split(',')[1],'base64');}
  const before=await render(original),after=await render(display),sourcePNG=path.join(base,'figures/V07_cycle_processing.png');
  const a=await sharp(before).removeAlpha().raw().toBuffer(),b=await sharp(sourcePNG).removeAlpha().raw().toBuffer();
  assert(a.equals(b),'Unmodified code must reproduce the source figure pixel for pixel');
  const md=await sharp(sourcePNG).metadata();
  // The lower scientific panel is tested after both display rows are changed.
  const box={left:230,top:920,width:2630,height:440};
  const lowerBefore=await sharp(before).extract(box).raw().toBuffer(),lowerAfter=await sharp(after).extract(box).raw().toBuffer();
  assert(lowerBefore.equals(lowerAfter),'Pulse panel moved or changed');
  const dest=path.join(revision,'figures/V07_cycle_processing_display_minus1.png');fs.writeFileSync(dest,after);
  record={old_asset:relative(sourcePNG),new_path:relative(dest),display_multiplier:-1,applied_to:['ECG displayed y','QRS marker displayed y'],source:{path:relative(source),sha256:sha(fs.readFileSync(source)),divs:[s4.div,s5.div],trace_indices:{ECG:[3,1],QRS:[3,2],TT:[4,0],lateral:[3,0]}},width:md.width,height:md.height,sha256:sha(after),checks:{baseline_pixels_equal:true,pulse_panel_pixels_equal:true,pulse_panel_xywh:box,all_x_unchanged:true,all_non_ecg_traces_unchanged:true,ECG_and_QRS_y_exactly_negated:true,source_file_unchanged:sha(fs.readFileSync(source))===sha(Buffer.from(raw))},QRS_time_s:values(original[1].x),QRS_original_mV:values(original[1].y),QRS_display_mV:values(display[1].y),source_plot_code:{path:relative(path.join(base,'tools/render_visuals.cjs')),sha256:sha(fs.readFileSync(path.join(base,'tools/render_visuals.cjs')))}};
 }finally{await browser.close();}
 fs.writeFileSync(path.join(__dirname,'ecg_display_check.json'),JSON.stringify(record,null,2));console.log(JSON.stringify(record.checks));
}
main().catch(e=>{console.error(e);process.exitCode=1;});
