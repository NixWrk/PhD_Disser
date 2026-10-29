// Render only saved Plotly scenes and protocol-based vector diagrams. Offline.
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const deps='C:/Users/ПАВЛУША/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {chromium}=require(path.join(deps,'playwright'));
const sharp=require(path.join(deps,'sharp'));
const rev=path.resolve(__dirname,'..'), root=path.resolve(rev,'../../../..'),out=path.join(rev,'figures');
const mp=path.join(out,'manifest.json');let manifest=JSON.parse(fs.readFileSync(mp,'utf8'));
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
const rel=p=>path.relative(root,p).replaceAll('\\','/');
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
function anon(s){for(const[a,b]of [['Георгия','У2'],['Георгий','У2'],['Георг','У2'],['Ярослава','У3'],['Ярослав','У3'],['Ника','У1'],['Ник','У1'],['nik','У1'],['georg','У2'],['yaroslav','У3'],['Н','У1'],['Г','У2'],['Я','У3']])s=s.replace(new RegExp('(?<![А-Яа-яЁёA-Za-z0-9_])'+a+'(?![А-Яа-яЁёA-Za-z0-9_])','g'),b);return s;}
function strings(v){if(typeof v==='string')return anon(v);if(Array.isArray(v))return v.map(strings);if(v&&typeof v==='object')return Object.fromEntries(Object.entries(v).map(([k,x])=>[k,k==='bdata'?x:strings(x)]));return v;}
function numeric(v,p='$',a=[]){if(typeof v==='number')a.push([p,v]);else if(Array.isArray(v))v.forEach((x,i)=>numeric(x,p+'.'+i,a));else if(v&&typeof v==='object')Object.entries(v).forEach(([k,x])=>numeric(x,p+'.'+k,a));return a;}
const text=(x,y,s,size=30,anchor='start',fill='#17212b')=>`<text x="${x}" y="${y}" font-size="${size}" text-anchor="${anchor}" fill="${fill}">${esc(s)}</text>`;
const line=(x,y,a,b,c='#34516a',w=3,extra='')=>`<line x1="${x}" y1="${y}" x2="${a}" y2="${b}" stroke="${c}" stroke-width="${w}" ${extra}/>`;
const box=(x,y,w,h,fill='#f3f6f8')=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${fill}" stroke="#8195a6" stroke-width="2"/>`;
const circle=(x,y,r,fill='#edf4f8',stroke='#34516a')=>`<circle cx="${x}" cy="${y}" r="${r}" fill="${fill}" stroke="${stroke}" stroke-width="3"/>`;
function svg(w,h,body){return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><defs><marker id="arr" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#34516a"/></marker></defs><rect width="100%" height="100%" fill="white"/><g font-family="Arial,DejaVu Sans,sans-serif">${body}</g></svg>`;}
async function saveSVG(id,slug,body,w,h){let p=path.join(out,`${id}_${slug}.svg`);fs.writeFileSync(p,svg(w,h,body));let png=p.replace(/svg$/,'png');await sharp(p).resize({width:2200}).png().toFile(png);return await fileRecord(png,{svg_path:rel(p)});}
async function fileRecord(p,extra={}){let m=await sharp(p).metadata();return{path:rel(p),width:m.width,height:m.height,unit:'px',sha256:sha(fs.readFileSync(p)),width_cm:16.5,height_cm:Math.min(11.5,16.5*m.height/m.width),...extra};}
function readJSON(s,pos){while(/[\s,]/.test(s[pos]))pos++;let start=pos,depth=0,str=false,escape=false;for(;pos<s.length;pos++){let c=s[pos];if(str){if(escape)escape=false;else if(c==='\\')escape=true;else if(c==='"')str=false;}else if(c==='"')str=true;else if(c==='['||c==='{')depth++;else if(c===']'||c==='}'){depth--;if(!depth)return[JSON.parse(s.slice(start,pos+1)),pos+1];}}throw Error('JSON boundary');}
function scenes(raw){let a=[],rx=/Plotly\.newPlot\(\s*("[^"\n]+")\s*,\s*/g,m;while(m=rx.exec(raw)){let[data,end]=readJSON(raw,rx.lastIndex),[layout,last]=readJSON(raw,end);a.push({div:JSON.parse(m[1]),data,layout});rx.lastIndex=last;}return a;}
async function main(){
 let b='';b+=text(750,65,'Схема получения исследовательских данных',40,'middle');
 const nodes=[['Индивидуальная КТ','Глубокий вдох с задержкой',60,140],['Сегментация анатомии','Маски и поверхности органов',790,140],['Импеданс и ЭКГ','Монтаж, протокол, номер записи',60,430],['Поясная ЭИТ','Отдельная регистрация напряжений',790,430]];
 for(const [a,c,x,y]of nodes){b+=box(x,y,650,180)+text(x+325,y+68,a,36,'middle')+text(x+325,y+124,c,30,'middle');}
 b+=line(710,230,780,230,'#34516a',4,'marker-end="url(#arr)"')+line(1115,320,400,420,'#34516a',4,'marker-end="url(#arr)"');
 b+=text(750,704,'Анатомическая опора и электрические наблюдения',34,'middle')+text(750,760,'Разные регистрации сохраняют собственные временные шкалы',29,'middle');
 let rec=await saveSVG('V01','materials_sequence',b,1500,820),e=manifest.figures.find(x=>x.id==='V01');
 e.assets.push({alias:'materials_sequence',...rec,caption:'Схема получения исследовательских данных: индивидуальная КТ и сегментация, запись импеданса и ЭКГ, отдельные поясные измерения ЭИТ. Разные регистрации сохраняют собственные временные шкалы.',source:{protocol:'AUTHOR_CONFIRMATIONS.md',blocks:['p0669–p0672','p0720–p0728']},restrictions:['Схема назначения данных; покадровая синхронность не предполагается.']});e.aliases.push('materials_sequence');e.alias_paths.materials_sequence=rec.path;

 b=text(750,56,'Схема процедуры и электродных подключений',39,'middle');b+=box(35,95,385,755,'#f7f8fa');
 b+=text(225,142,'Поза регистрации',31,'middle');
 b+=circle(225,345,51,'white');b+=`<path d="M155 420 Q225 390 295 420 L287 650 Q225 681 163 650 Z" fill="#edf2f5" stroke="#34516a" stroke-width="4"/>`;
 b+=`<path d="M157 429 Q106 385 105 306 L99 204 M293 429 Q344 385 345 306 L351 204 M188 662 L177 776 M262 662 L273 776" fill="none" stroke="#34516a" stroke-width="16" stroke-linecap="round"/>`;
 b+=text(225,815,'Вид сверху, без масштаба',25,'middle');
 b+=text(470,137,'Боковая сборка A–M–N–B',32);let xx=[520,690,1030,1200];
 b+=line(490,235,1270,235,'#91a2ad',5);
 xx.forEach((x,i)=>{b+=circle(x,235,13,i===0||i===3?'#b35035':'#247da1')+text(x,203,['A','M','N','B'][i],30,'middle');});
 b+=line(520,295,1200,295)+line(520,282,520,309)+line(1200,282,1200,309)+text(860,335,'Токовая база L; измерительная база L/2',28,'middle');
 b+=text(470,405,'ТТРКГ: группы контактов на каждой руке',31);
 for(let j=0;j<3;j++){let cy=495,base=560+j*310;b+=text(base,454,['1×1','2×2','1×4'][j],31,'middle');
  for(let k=0;k<2;k++){let cx=base+(k?64:-64);b+=circle(cx,cy,39,'white');let angles=j===2?(k?[-Math.PI/2,0,Math.PI/2,Math.PI]:[-Math.PI/2]):j?[-Math.PI/2,Math.PI/2]:[-Math.PI/2];
   angles.forEach(a=>b+=circle(cx+39*Math.cos(a),cy+39*Math.sin(a),8,k?'#247da1':'#b35035'));
  }b+=text(base,562,'ток / напряжение',24,'middle');}
 b+=text(470,640,'Перекрёстная боковая установка',31)+line(550,680,550,786)+line(499,733,601,733);
 [[550,680],[550,709],[550,757],[550,786],[499,733],[526,733],[574,733],[601,733]].forEach(([x,y])=>b+=circle(x,y,6,'#247da1'));
 b+=text(650,700,'Два канала — две боковые оси',28)+text(650,746,'В кресте ТТРКГ отсутствует',28);
 b+=text(750,901,'Лёжа на спине; руки вытянуты за голову, не сцеплены',32,'middle');
 rec=await saveSVG('V02','measurement_setup',b,1500,940);e=manifest.figures.find(x=>x.id==='V02');Object.assign(e,rec);e.aliases.push('materials_electrodes');e.alias_paths.materials_electrodes=e.path;
 e.restrictions.push('Для креста оба отведения боковые; расположение и площадь контактов на индивидуальной КТ этой схемой не задаются.');

 b=text(750,58,'Расчётные схемы полупространств',40,'middle');
 for(let k=0;k<2;k++){let ox=35+755*k;b+=text(ox+345,126,k?'Два слоя':'Однородная среда',35,'middle');
  b+=box(ox+20,305,665,420,k?'#f4f0e7':'#edf4f7');if(k)b+=box(ox+20,505,665,220,'#edf4f7');
  let ax=[ox+80,ox+225,ox+515,ox+660];ax.forEach((x,i)=>{b+=`<rect x="${x-14}" y="291" width="28" height="16" fill="${i===0||i===3?'#b35035':'#247da1'}"/>`+text(x,271,['A','M','N','B'][i],32,'middle');});
  b+=line(ax[0],190,ax[3],190)+line(ax[0],180,ax[0],202)+line(ax[3],180,ax[3],202)+text(ox+370,179,'L',30,'middle');
  b+=line(ax[1],225,ax[2],225)+text(ox+370,216,'L/2',28,'middle');
  b+=text(ox+350,k?408:500,k?'ρ₁':'ρ',44,'middle');if(k)b+=text(ox+350,625,'ρ₂',44,'middle')+line(ox+625,325,ox+625,485)+text(ox+595,420,'h',35,'middle');
 }
 b+=text(750,797,'Конечные площадки; плоская граница — допущение модели',31,'middle');
 rec=await saveSVG('V09','layered_model',b,1500,850);e=manifest.figures.find(x=>x.id==='V09');Object.assign(e,rec);e.aliases.push('models_layered_geometry');e.alias_paths.models_layered_geometry=e.path;

 b=text(750,65,'Схема аппроксимаций и преобразования одной опоры',37,'middle');
 const bean=`M-105 15 C-125 -80 -35 -110 20 -80 C110 -90 132 -15 74 55 C25 120 -68 98 -105 15 Z`;
 b+=`<g transform="translate(245 290)"><path d="${bean}" fill="#e8dce1" stroke="#763e54" stroke-width="4"/></g>`+circle(745,290,102,'#e4eff6')+`<ellipse cx="1240" cy="290" rx="135" ry="80" fill="#e9f1e6" stroke="#59764d" stroke-width="4"/>`;
 b+=text(245,448,'Опорная область',31,'middle')+text(745,448,'Равновеликая сфера',31,'middle')+text(1240,448,'Равновеликий эллипсоид',31,'middle');
 b+=text(750,536,'Условные проекции: анатомический масштаб не задан',29,'middle');
 b+=box(65,605,1370,220)+text(750,665,'x = c₀ + d + s(X − c₀)',40,'middle')+text(750,725,'V = s³V₀',40,'middle')+text(750,783,'Масштаб s меняет объём; перенос d меняет положение',29,'middle');
 rec=await saveSVG('V12','shape_transform_scheme',b,1500,875);Object.assign(manifest.figures.find(x=>x.id==='V12'),rec);

 const scdir=path.join(root,'Colab Notebooks/scorpion_v2'),h40=fs.readFileSync(path.join(scdir,fs.readdirSync(scdir).find(x=>x.startsWith('33.40')&&x.endsWith('.html'))),'utf8');
 const h40sc=scenes(h40);
 const library=[...h40.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(x=>x[1]).find(x=>x.includes('plotly.js v')&&x.length>100000);
 if(!library)throw Error('Bundled Plotly library absent');
 const browser=await chromium.launch({executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless:true,args:['--disable-gpu','--allow-file-access-from-files']});
 const page=await browser.newPage({viewport:{width:1800,height:1100}});await page.route('**/*',route=>route.abort());
 await page.setContent('<html><body style="margin:0;background:white"><div id="plot"></div></body></html>');await page.addScriptTag({content:library});
 async function plot(scene,dst,title,options={}){
  const orig=structuredClone(scene),data=options.indices?scene.data.filter((x,i)=>options.indices.includes(i)):scene.data;
  const nb=sha(JSON.stringify(numeric(data)));let d=strings(structuredClone(data)),l=strings(structuredClone(scene.layout));
  if(options.indices){d.forEach(x=>{x.visible=true;x.showlegend=true;});l.updatemenus=[];l.sliders=[];}
  if(sha(JSON.stringify(numeric(d)))!==nb)throw Error('Numerical trace data changed');
  l.width=1800;l.height=options.height||1000;l.font={...(l.font||{}),family:'Arial',size:32,color:'#17212b'};
  l.title={text:title,x:.5,font:{size:38}};l.margin={l:135,r:90,t:120,b:145};l.updatemenus=[];l.sliders=[];
  l.legend={...(l.legend||{}),font:{size:27},orientation:'h',y:-.16,x:0};
  for(const[k,v]of Object.entries(l)){if(/^[xy]axis\d*$/.test(k)&&v&&typeof v==='object'){v.tickfont={size:29};if(typeof v.title==='string')v.title={text:v.title};v.title={...(v.title||{}),font:{size:31}};v.automargin=true;}}
  (l.annotations||[]).forEach(x=>x.font={...(x.font||{}),size:30});
  if(options.threeD){l.margin={l:70,r:70,t:115,b:110};l.annotations=[];l.legend.y=-.03;l.scene=l.scene||{};l.scene.domain={x:[.04,.96],y:[.03,.93]};l.scene.aspectmode='data';l.scene.camera={eye:{x:-1.05,y:-1.15,z:.45},projection:{type:'orthographic'}};for(const [i,a] of ['xaxis','yaxis','zaxis'].entries()){l.scene[a]={...(l.scene[a]||{}),range:undefined,autorange:true,showticklabels:false,title:{text:['X, мм','Y, мм','Z, мм'][i],font:{size:24}},showspikes:false};}d.forEach(x=>{if(x.type==='mesh3d')x.opacity=x.name.includes('Кости')?.22:x.name.includes('Лёгкие')?.35:((x.name==='Тело'||/поверхност/i.test(x.name))?.2:.95);if(x.type==='scatter3d'&&x.marker)x.marker={...x.marker,size:8};});}
  let result=await page.evaluate(async({d,l})=>{await Plotly.newPlot('plot',d,l,{displayModeBar:false,staticPlot:true});return await Plotly.toImage('plot',{format:'png',width:l.width,height:l.height,scale:1.7});},{d,l});
  fs.writeFileSync(dst,Buffer.from(result.split(',')[1],'base64'));let m=await fileRecord(dst);
  return {...m,transformation:{method:'static rendering of frozen HTML Plotly scene',source_div:scene.div,numerical_trace_ledger_sha256:nb,numerical_trace_data_unchanged:true,selected_trace_indices:options.indices||null,layout_changes:'fonts, margins, static title and legend; no numerical recomputation'},source_div:scene.div};
 }
 for(const [id,number,title]of [['V06',1,'У1 · полная дыхательная проба и кандидатные окна'],['V08',2,'У1 · боковой импеданс и размер сборки'],['V10',16,'У1 · условные сопротивления при фиксированных h'],['V11',18,'У1 · прогноз исключённого размера по двум другим']]){
  e=manifest.figures.find(x=>x.id===id);let scene=h40sc[number-1];Object.assign(e,await plot(scene,path.join(root,e.path),title,{height:id==='V06'?1240:1000}));e.source.div=scene.div;
 }
 // Four stored cycles of an ordinary simultaneous TT/lateral pair, no excluded branch.
 let s4=h40sc[3],s5=h40sc[4];
 if(JSON.stringify(s4.data[0].x)!==JSON.stringify(s5.data[0].x))throw Error('Pulse time scales differ');
 let ecg=structuredClone(s4.data[1]),qrs=structuredClone(s4.data[2]),tt=structuredClone(s5.data[0]),side=structuredClone(s4.data[0]);
 [ecg,qrs].forEach(x=>{x.xaxis='x';x.yaxis='y';x.showlegend=true;});ecg.name='ЭКГ';qrs.name='QRS-кандидаты';
 [tt,side].forEach(x=>{x.xaxis='x2';x.yaxis='y2';x.showlegend=true;});tt.name='ТТРКГ 2×2';side.name='Боковая сборка 140 мм';tt.line={...(tt.line||{}),color:'#1769a7'};side.line={...(side.line||{}),color:'#b35806'};
 let cycle={div:s4.div+' + '+s5.div,data:[ecg,qrs,tt,side],layout:{xaxis:{domain:[0,1],anchor:'y',showticklabels:false},yaxis:{domain:[.57,1],title:'ЭКГ, мВ'},xaxis2:{domain:[0,1],anchor:'y2',title:'Время от первого QRS-кандидата, с'},yaxis2:{domain:[0,.43],title:'Пульсовой импеданс, мОм'},annotations:[]}};
 e=manifest.figures.find(x=>x.id==='V07');Object.assign(e,await plot(cycle,path.join(root,e.path),'У1 · запись 001 · четыре сохранённых цикла',{height:1000}));
 e.source={html:rel(path.join(scdir,fs.readdirSync(scdir).find(x=>x.startsWith('33.40')&&x.endsWith('.html')))),figures:[4,5],divs:[s4.div,s5.div],trace_indices:[[1,2,0],[0]],scope:'ordinary TT 2x2 and lateral 140 mm; same stored time grid'};
 e.caption='Четыре сохранённых кандидатных цикла обычной записи 001 У1: ЭКГ с QRS-кандидатами и пульсовые сигналы одновременно подключённых ТТРКГ 2×2 и боковой сборки 140 мм. Время отсчитано от первого QRS-кандидата, импеданс показан в мОм после полосы 0,7–12 Гц.';
 e.purpose='Показать выделенный временной блок обычной пары ТТРКГ и боковой сборки.';
 e.restrictions=['Границы задержки и QRS кандидатные; механические конечные фазы не назначены.','Использованы сохранённые массивы без новой фильтрации, поиска QRS или усреднения. Медианные формы на этом рисунке не представлены.'];
 // Real saved organ surfaces; omit all contacts and the body with model arms.
 const h2010path=path.join(root,'Colab Notebooks',fs.readdirSync(path.join(root,'Colab Notebooks')).find(x=>x.startsWith('20.10')&&x.endsWith('.html'))),r2010=fs.readFileSync(h2010path,'utf8'),s2010=scenes(r2010)[0];
 e=manifest.figures.find(x=>x.id==='V04');let contour={...e,alias:'ct_surface_control'};delete contour.assets;delete contour.alias_paths;
 e.assets.push(contour);e.aliases.push('ct_model_preparation');
 let p=path.join(out,'V04_anatomical_surfaces.png');Object.assign(e,await plot(s2010,p,'Сегментированные поверхности: лёгкие, сердце и кости',{indices:[1,2,3],threeD:true}));
 e.source={html:rel(h2010path),sha256:sha(fs.readFileSync(h2010path)),div:s2010.div,trace_indices:[1,2,3]};
 e.caption='Пример пространственного представления сегментированной анатомии из сохранённой модели C01: поверхности лёгких, сердца и костей. Показаны только сохранённые тканевые поверхности; тело, модельные руки и электродные контакты исключены из отображения.';
 e.purpose='Показать реальные пространственные поверхности органов отдельно от расчётных дополнений.';
 e.restrictions=['Историческое представление C01: его геометрия и электрическая постановка не подменяют канонические статические модели трёх участников.','Новая сегментация или FEM-расчёт не выполнялись; геометрические массивы выбранных поверхностей не изменены.'];
 e.alias_paths.ct_model_preparation=e.path;
 // A distinct actual computational surface/contact model for chapter 3.
 const h2020path=path.join(scdir,fs.readdirSync(scdir).find(x=>x.startsWith('20.20')&&x.endsWith('.html'))),r2020=fs.readFileSync(h2020path,'utf8'),s2020=scenes(r2020)[0];
 const active2020=s2020.data.map((x,i)=>[x,i]).filter(([x])=>x.visible!=='legendonly'&&x.visible!==false).map(([,i])=>i);
 p=path.join(out,'V04_fem_surface_contacts.png');rec=await plot(s2020,p,'Расчётная поверхность У2: боковая сборка и ТТРКГ',{indices:active2020,threeD:true});
 e.assets=e.assets.filter(x=>x.alias!=='models_ct_fem_preparation');
 e.assets.push({alias:'models_ct_fem_preparation',...rec,caption:'Сохранённая расчётная поверхность У2 с центрами боковых контактов и группами ТТРКГ. Искусственные продолжения рук являются элементами модели и отличаются от подтверждённой позы регистрации с руками за головой. Показана внешняя поверхность; контактные площадки и внутренние тетраэдры этой сценой не разрешены.',source:{html:rel(h2020path),div:s2020.div,trace_indices:active2020,sha256:sha(fs.readFileSync(h2020path))},restrictions:['Сохранённая расчётная геометрия; центры не являются независимым измерением наклейки.','Изображены только боковые и трансторакальные конфигурации; новая стадия обработки не добавлена.']});
 e.aliases.push('models_ct_fem_preparation');e.alias_paths.models_ct_fem_preparation=rec.path;
 await browser.close();
 // Read current chapter placeholders, map semantic aliases to explicit files.
 const aliasMap={materials_electrodes:'V02',ct_segmentation:'V03',models_layered_geometry:'V09',processing_cycles:'V07',models_geometry_comparison:'V13',models_finite_compensation:'V19',tissue_eit_baseline:'V14',tissue_pulse_projection:'V15',tissue_geometry_mixing:'V16',cardiac_phase_reversal:'V22',processing_breathing_protocol:'V06'};
 for(const[a,id]of Object.entries(aliasMap)){e=manifest.figures.find(x=>x.id===id);if(!e.aliases.includes(a))e.aliases.push(a);e.alias_paths[a]=e.path;}
 e=manifest.figures.find(x=>x.id==='V04');e.aliases.push('models_anatomy_contacts');e.alias_paths.models_anatomy_contacts=e.assets.find(x=>x.alias==='anatomical_model').path;
 e.assets.find(x=>x.alias==='anatomical_model').caption='Расчётная область У1: поверхности сердца, лёгких и тела с искусственным продолжением рук и площадками пояса ЭИТ. Продолжения рук и алгоритмическое положение пояса являются элементами модели; они не воспроизводят измеренную позу с руками за головой.';
 const cache=path.join(rev,'../revision_20261028/figures');
 for(const[alias,id,file]of [['models_step_quadrature','V13','reader_montage/ch07_04_volume_step_quadrature.png'],['models_electrical_factors','V17','reader/07_02_electrical_approximation.png']]){
  let source=path.join(cache,file),dest=path.join(out,`${id}_${alias}.png`);fs.copyFileSync(source,dest);e=manifest.figures.find(x=>x.id===id);rec=await fileRecord(dest);e.assets.push({alias,...rec,source:{cached_png:rel(source),sha256:sha(fs.readFileSync(source))},caption:alias==='models_step_quadrature'?'Объёмная чувствительность STL, G1 и L1 при изменении объёмного шага и числа квадратурных точек на одной сетке. Отдельно показаны две материальные опоры и три расчётных монтажа.':'Приращения ТТРКГ при одинаковых объёмных пробах разных форм сердца и отдельные сценарии эффективного сердечного сопротивления.',restrictions:['Сохранённый вычислительный результат; физиологическая точность не установлена.']});e.aliases.push(alias);e.alias_paths[alias]=rec.path;
 }
 for(const e of manifest.figures){e.width_cm=Math.min(16.5,11.5*e.width/e.height);e.height_cm=e.width_cm*e.height/e.width;if(e.id==='V03'){e.width_cm=16.5;e.height_cm=16.5*e.height/e.width;e.height_exception_cm=18;}for(const a of e.assets||[])if(a.width){a.width_cm=Math.min(16.5,11.5*a.width/a.height);a.height_cm=a.width_cm*a.height/a.width;}}
 manifest.main_aliases=[...Object.keys(aliasMap),'materials_sequence','ct_model_preparation','models_ct_fem_preparation','models_anatomy_contacts','models_step_quadrature','models_electrical_factors','photo_lateral_detail','photo_tt_wide','photo_belt'];
 manifest.source_availability_notes=['33.41 and 33.42 HTML files in this checkout are Git LFS pointers; not treated as readable figure sources.','ct_model_preparation and models_ct_fem_preparation intentionally resolve to distinct real HTML scenes.'];
 manifest.runtime.bundled_node=process.version;
 fs.writeFileSync(mp,JSON.stringify(manifest,null,2)+'\n');console.log('Saved vector diagrams, frozen Plotly exports and semantic aliases.');
}
main().catch(e=>{console.error(e);process.exitCode=1;});
