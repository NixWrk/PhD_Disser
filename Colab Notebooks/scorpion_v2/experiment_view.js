(() => {
const D=experimentPayload,recordControl=document.getElementById('experiment-record'),channelControl=document.getElementById('experiment-channel');
const ct=document.getElementById('experiment-ct'),signals=document.getElementById('experiment-signals');
const notch=document.getElementById('experiment-notch'),band=document.getElementById('experiment-band');
const channelVisible=[1,2].map(ch=>document.getElementById('experiment-show-channel-'+ch));
const leadColors={ttrkg:'#176aaf',lateral_horizontal:'#e47823',lateral_vertical:'#8154a4',unknown:'#667085'};
const leadNames={ttrkg:'ТТРКГ',lateral_horizontal:'Боковая горизонтальная',lateral_vertical:'Боковая вертикальная',unknown:'Неустановленное отведение'};
const sampling=document.getElementById('experiment-sampling');
let sampleRecord=null,sampleFields=null;
const phaseColors=['#d7ecd9','#d4e5ff','#f6ead0','#eadcf5'];
const charts=[
 {id:'bei-raw',title:'Общий импеданс BEI',unit:'BEI, Ом',kind:'bei_raw'},
 {id:'bei-slow',title:'Базовый импеданс',unit:'Базовый BEI, Ом',kind:'bei_slow'},
 {id:'bei-pulse',title:'Пульсовая полоса импеданса',unit:'Пульсовая полоса, мОм',kind:'bei_pulse_band'},
 {id:'ecg-overview',title:'ЭКГ: обзор всей записи',unit:'ЭКГ, мВ',kind:'ecg_selected'},
 {id:'ecg-detail',title:'ЭКГ: исходный и выбранный вариант, фрагмент 5 с',unit:'ЭКГ, мВ',compare:true},
 {id:'ecg-spectrum',title:'Спектральная плотность мощности ЭКГ: вся запись',unit:'S, мВ²/Гц',compare:true,spectral:true}
];
let ri=3,history=0,ready=false,queue=Promise.resolve();
D.records.forEach((r,i)=>recordControl.add(new Option(r.label,i)));recordControl.value=ri;
const xyz=p=>({x:p.map(v=>v[0]),y:p.map(v=>v[1]),z:p.map(v=>v[2])});
const filterKey=()=>`${Number(notch.checked)}${Number(band.checked)}`;
const lateralChannel=()=>{
 const rid=Number(D.records[ri].record_id.slice(-3));
 if(history===0)return rid>=1&&rid<=5?2:rid>=6&&rid<=11?1:0;
 return history<=2?history:history===3?(rid<=5?2:1):(rid<=5?1:2);
};
function physicalRole(ch){
 const r=D.records[ri],m=r.montage;
 if(history===0)return r.channel_assignment.physical_channel_roles[String(ch)];
 if(m.mapping==='unknown')return 'unknown';
 const lateral=lateralChannel();
 if(m.mapping==='cross')return ch===lateral?'lateral_horizontal':'lateral_vertical';
 if(m.mapping==='vertical')return ch===2?'lateral_vertical':'ttrkg';
 return ch===lateral?'lateral_horizontal':'ttrkg';
}
async function fullSamples(){
 const r=D.records[ri];if(sampleRecord===r.record_id)return;
 const p=r.full_samples,bytes=Uint8Array.from(atob(p.data),v=>v.charCodeAt(0));
 const raw=await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
 if(raw.byteLength!==p.rows*p.fields.length*8)throw Error('Размер массива отсчётов не совпадает');
 const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',raw)),v=>v.toString(16).padStart(2,'0')).join('');
 if(digest!==p.uncompressed_sha256)throw Error('Контрольная сумма отсчётов не совпадает');
 const a=new Float64Array(raw);sampleFields=Object.fromEntries(p.fields.map((name,i)=>[name,a.subarray(i*p.rows,(i+1)*p.rows)]));sampleRecord=r.record_id;
}
function visibilityControls(){
 const host=document.getElementById('experiment-visibility-controls');
 charts.forEach(chart=>{
  chart.plot=document.getElementById('experiment-'+chart.id);
  const field=document.createElement('fieldset'),legend=document.createElement('legend');legend.textContent=chart.title;field.append(legend);
  const checkbox=(name,label)=>{const wrapper=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.checked=true;input.dataset.visibility=name;
   wrapper.append(input,document.createTextNode(' '+label));wrapper.style.display='block';field.append(wrapper);return input;};
  chart.show=checkbox(chart.id+'-plot','Показывать график');chart.curves=[];
  for(const ch of [1,2])for(const selected of chart.compare?[false,true]:[true]){
   const label='Канал '+ch+(chart.compare?(selected?' · выбранные фильтры':' · исходная ЭКГ'):'');
   const input=checkbox(chart.id+'-'+ch+'-'+Number(selected),label);chart.curves.push({ch,selected,input});
  }
  host.append(field);
 });
}
function geometry(){
 const m=D.records[ri].montage,traces=[];
 const add=(p,lead,roles,name)=>traces.push({type:'scatter3d',mode:'markers+text',...xyz(p),name,text:roles,showlegend:!!p.length,visible:!!p.length,
  marker:{size:7,color:leadColors[lead]},textposition:'top center',hovertemplate:name+'<br>x=%{x:.2f}; y=%{y:.2f}; z=%{z:.2f} мм<extra></extra>'});
 add(m.lateral,'lateral_horizontal',['A','M','N','B'],'Боковая горизонтальная сборка');
 add(m.perpendicular,'lateral_vertical',['A′','M′','N′','B′'],'Боковая вертикальная сборка');
 const current=[],potential=[];m.arm.forEach((g,i)=>(i===0||i===3?current:potential).push(...g));
 add(current,'ttrkg',current.map(()=> 'I'),'Токовые контакты ТТРКГ');add(potential,'ttrkg',potential.map(()=> 'U'),'Измерительные контакты ТТРКГ');return traces;
}
function curveData(chart){
 const r=D.records[ri];
 return chart.curves.map(({ch,selected,input})=>{
  const key=selected?filterKey():'00',kind=chart.compare?(selected?'ecg_selected':'ecg_raw'):chart.kind;
  let x,y;
  if(chart.spectral){x=r.psd_frequency_hz;y=r.ecg_variants[key][ch-1].psd;}
  else if(chart.compare){const [a,b]=r.detail_indices;x=sampleFields.time_s.subarray(a,b);y=sampleFields['ecg_'+key+'_'+ch].subarray(a,b);}
  else if(sampling.value==='full'){x=sampleFields.time_s;y=sampleFields[kind==='ecg_selected'?'ecg_'+filterKey()+'_'+ch:kind+'_'+ch];}
  else if(kind==='ecg_selected'){const t=r.ecg_variants[filterKey()][ch-1];x=t.t;y=t.y;}
  else {const t=r.traces.find(t=>t.kind===kind&&t.channel===ch);x=t.t;y=t.y;}
  const name=(chart.compare?(selected?'Выбранные фильтры':'Исходная ЭКГ'):chart.title)+' · '+leadNames[physicalRole(ch)]+' · канал '+ch;
  return {type:chart.spectral?'scatter':'scattergl',mode:'lines',x,y,name,showlegend:false,visible:channelVisible[ch-1].checked&&input.checked,
   line:{color:leadColors[physicalRole(ch)],width:kind==='bei_slow'?2:1.2,dash:chart.compare&&selected?'dash':'solid'},
   hovertemplate:(chart.spectral?'f=%{x:.3f} Гц; S=%{y:.4g} мВ²/Гц':'t=%{x:.3f} с; %{y:.5g}')+'<extra>'+name+'</extra>'};
 });
}
function phaseDecor(window=null){
 const r=D.records[ri],b=r.breathing,shapes=[],annotations=[];
 b.phases.forEach((p,i)=>{
  const lo=window?Math.max(p.start,window[0]):p.start,hi=window?Math.min(p.end,window[1]):p.end;if(hi<=lo)return;
  shapes.push({type:'rect',xref:'x',yref:'paper',x0:lo,x1:hi,y0:0,y1:1,fillcolor:phaseColors[i],opacity:.48,line:{width:0},layer:'below'});
  annotations.push({xref:'x',yref:'paper',x:(lo+hi)/2,y:1,text:p.label.replace(' дыхание','<br>дыхание').replace('Задержка на','Задержка<br>на'),showarrow:false,yanchor:'bottom',font:{size:9,color:'#334155'}});
 });
 (b.boundary_uncertainty_ranges_samples||[]).forEach(v=>shapes.push({type:'rect',xref:'x',yref:'paper',x0:v[0]/r.fs,x1:v[1]/r.fs,y0:0,y1:1,fillcolor:'#777',opacity:.22,line:{width:0},layer:'below'}));
 (b.boundaries_samples||[]).forEach(n=>shapes.push({type:'line',xref:'x',yref:'paper',x0:n/r.fs,x1:n/r.fs,y0:0,y1:1,line:{color:'#555',width:1,dash:'dot'}}));
 return {shapes,annotations};
}
function text(){
 const r=D.records[ri],m=r.montage,channel=lateralChannel();
 document.getElementById('experiment-description').textContent=r.label+'; рабочая частота отсчётов '+r.fs+' Гц; последний отсчёт '+r.duration.toLocaleString('ru-RU',{maximumFractionDigits:3})+' с. Цвета отведений: ТТРКГ — синий; боковая горизонтальная — оранжевый; боковая вертикальная — фиолетовый. '+(sampling.value==='full'?'Отображаются все '+r.full_samples.rows+' отсчётов каждого сигнала.':'Быстрый обзор: сохранены экстремумы 900 временных интервалов. Увеличение не возвращает пропущенные отсчёты; для формы сигнала выберите все отсчёты.');
 charts.forEach(chart=>chart.curves.forEach(({ch,selected,input})=>{input.parentElement.style.color=leadColors[physicalRole(ch)];input.parentElement.lastChild.textContent=' '+leadNames[physicalRole(ch)]+' · канал '+ch+(chart.compare?(selected?' · выбранные фильтры':' · исходная ЭКГ'):'');}));
 const rid=Number(r.record_id.slice(-3));
 const roles=history===0?(rid===5?'005: ТТРКГ — канал 1; вертикальная боковая — канал 2. Подтверждение автора.':rid===6?'006: горизонтальная боковая — канал 1; сохранённая вертикальная — канал 2. Подтверждение автора.':(rid>=1&&rid<=4)||rid===7?'ТТРКГ — канал '+(3-channel)+'; боковая — канал '+channel+'. Подтверждение автора.':rid>=8&&rid<=11?'Продолжение схемы 007: боковая горизонтальная — канал 1; второй канал — '+(rid===11?'вертикальная ось креста.':'ТТРКГ.')+' Сохранение номера в этом блоке принято по описанию протокола.':'Номера ТЕПС/АЕПС не подтверждены; сигналы обозначены как неустановленные отведения.'):'Прежний диагностический сценарий; боковой канал '+channel+'.';
 document.getElementById('experiment-localisation').textContent=(m.mapping==='unknown'?'Монтаж не локализован на КТ. ':m.mapping==='test_horizontal'?'Тест: 1×1 на руках и горизонтальная боковая 140 мм. ':'Номинальная геометрия; координаты установки не измерены. ')+roles+((rid===1||rid===2)?' U сверху — сценарий с пока не подтверждённым углом.':'');
 document.getElementById('experiment-filter-description').textContent='Сейчас: '+({'00':'все фильтры отключены — исходная ЭКГ','10':'только режекторы 50/100/150 Гц','01':'только полоса 0,5–40 Гц','11':'режекторы 50/100/150 Гц, затем полоса 0,5–40 Гц'}[filterKey()])+'.';
 const phases=r.breathing.phases.map((p,i)=>`<span style="background:${phaseColors[i]};padding:4px 7px">${p.label}</span>`).join(' → ');
 document.getElementById('experiment-phase-description').innerHTML=(phases||'Дыхательные режимы не размечены.')+'<br>Кандидатная разметка, не принята автором. Серые полосы: неоднозначность границ; пунктир: выбранная кандидатная граница.';
}
async function update(){
 if(!ready)return;await fullSamples();text();const r=D.records[ri],g=geometry(),indices=g.map((_,i)=>1+D.tissues.length+i);
 await Plotly.restyle(ct,{x:g.map(t=>t.x),y:g.map(t=>t.y),z:g.map(t=>t.z),text:g.map(t=>t.text),marker:g.map(t=>t.marker),visible:g.map(t=>t.visible),showlegend:g.map(t=>t.showlegend)},indices);
 for(const chart of charts){
  const plot=chart.plot,traces=curveData(chart),changed=plot.dataset.record!==r.record_id,wasHidden=plot.style.display==='none';
  await Plotly.restyle(plot,{x:traces.map(t=>t.x),y:traces.map(t=>t.y),visible:traces.map(t=>t.visible),line:traces.map(t=>t.line),name:traces.map(t=>t.name),hovertemplate:traces.map(t=>t.hovertemplate)});
  if(changed){
   const range=chart.spectral?[0,r.fs/2]:chart.compare?[r.detail_t[0],r.detail_t.at(-1)]:[0,r.duration];
   await Plotly.relayout(plot,{...(chart.spectral?{}:phaseDecor(chart.compare?range:null)),'xaxis.range':range,'yaxis.autorange':true});
  }
  plot.style.display=chart.show.checked?'':'none';
  if(wasHidden&&chart.show.checked)await Plotly.Plots.resize(plot);
  plot.dataset.sampleCounts=JSON.stringify(traces.map(t=>t.x.length));plot.dataset.sampling=sampling.value;plot.dataset.leads=JSON.stringify(chart.curves.map(t=>physicalRole(t.ch)));
  plot.dataset.record=r.record_id;plot.dataset.visibleCurves=JSON.stringify(traces.map(t=>t.visible));plot.dataset.shown=String(chart.show.checked);plot.dataset.filters=filterKey();
 }
 ct.dataset.camera=JSON.stringify(ct._fullLayout.scene.camera);ct.dataset.record=r.record_id;ct.dataset.electrodeCounts=JSON.stringify(g.map(t=>t.x.length));ct.dataset.tissues=JSON.stringify(D.tissues.map(t=>t.name));ct.dataset.history=String(history);ct.dataset.lateralChannel=String(lateralChannel());
 signals.dataset.record=r.record_id;signals.dataset.phaseCount=r.breathing.phases.length;signals.dataset.filters=filterKey();
 const dd=curveData(charts[4]);charts[4].plot.dataset.rawEqualsSelected=String(dd[0].y.every((v,i)=>v===dd[1].y[i])&&dd[2].y.every((v,i)=>v===dd[3].y[i]));
 charts[5].plot.dataset.nyquistHz=r.fs/2;
}
async function init(){
 visibilityControls();await fullSamples();
 const mesh=(m,name,color,opacity)=>({type:'mesh3d',...xyz(m.vertices),i:m.faces.map(p=>p[0]),j:m.faces.map(p=>p[1]),k:m.faces.map(p=>p[2]),opacity,color,name,showlegend:true,hoverinfo:'skip'});
 const body=mesh(D,'Поверхность КТ и цилиндры рук','#b9b0a8',.16),tissues=D.tissues.map(t=>mesh(t,t.name,t.color,t.opacity)),config={responsive:true,displaylogo:false};
 await Plotly.newPlot(ct,[body,...tissues,...geometry()],{title:{text:'КТ: тканевые маски и номинальный монтаж'},height:790,scene:{aspectmode:'data',xaxis:{title:{text:'x LPS, мм'},range:[-430,430]},yaxis:{title:{text:'y LPS, мм'},range:[-180,180]},zaxis:{title:{text:'z LPS, мм'},range:[-380,0]},camera:{eye:{x:-1.3,y:-1.8,z:.9}},uirevision:'experiment-camera-fixed'},uirevision:'experiment-camera-fixed',legend:{orientation:'h',y:-.08},margin:{t:60,b:170,l:0,r:0}},config);
 for(const chart of charts)await Plotly.newPlot(chart.plot,curveData(chart),{title:{text:chart.title},height:chart.compare?390:290,
  xaxis:{title:{text:chart.spectral?'f, Гц':'t, с'}},yaxis:{title:{text:chart.unit},...(chart.spectral?{type:'log'}:{})},showlegend:false,margin:{t:80,b:55,l:80,r:15}},config);
 ct.on('plotly_relayout',()=>{ct.dataset.camera=JSON.stringify(ct._fullLayout.scene.camera);});ready=true;await update();
 const refresh=()=>{queue=queue.then(update).catch(e=>{document.getElementById('experiment-description').textContent='Ошибка представления: '+e.message;});};
 recordControl.onchange=()=>{ri=Number(recordControl.value);refresh();};channelControl.onchange=()=>{history=Number(channelControl.value);refresh();};notch.onchange=band.onchange=sampling.onchange=refresh;
 channelVisible.forEach(el=>el.onchange=refresh);charts.forEach(c=>{c.show.onchange=refresh;c.curves.forEach(t=>t.input.onchange=refresh);});
}
init().catch(e=>{document.getElementById('experiment-description').textContent='Ошибка представления: '+e.message;});
})();
