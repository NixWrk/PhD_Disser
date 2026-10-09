(() => {
const D=experimentPayload,recordControl=document.getElementById('experiment-record'),channelControl=document.getElementById('experiment-channel');
const ct=document.getElementById('experiment-ct'),signals=document.getElementById('experiment-signals'),detail=document.getElementById('experiment-ecg-detail'),spectrum=document.getElementById('experiment-ecg-spectrum');
const notch=document.getElementById('experiment-notch'),band=document.getElementById('experiment-band');
const colors=['#176aaf','#e47823'],phaseColors=['#d7ecd9','#d4e5ff','#f6ead0','#eadcf5'];
let ri=3,channel=1,ready=false,queue=Promise.resolve();
D.records.forEach((r,i)=>recordControl.add(new Option(r.label,i)));recordControl.value=ri;
const xyz=p=>({x:p.map(v=>v[0]),y:p.map(v=>v[1]),z:p.map(v=>v[2])});
const filterKey=()=>`${Number(notch.checked)}${Number(band.checked)}`;
function geometry(){
 const m=D.records[ri].montage,traces=[];
 const add=(p,ch,roles,name)=>traces.push({type:'scatter3d',mode:'markers+text',...xyz(p),name,text:roles,showlegend:!!p.length,visible:!!p.length,
  marker:{size:7,color:ch?colors[ch-1]:'#666'},textposition:'top center',hovertemplate:name+'<br>x=%{x:.2f}; y=%{y:.2f}; z=%{z:.2f} мм<extra></extra>'});
 add(m.lateral,['horizontal','cross'].includes(m.mapping)?channel:null,['A','M','N','B'],'Основная боковая сборка');
 add(m.perpendicular,m.mapping==='cross'?3-channel:null,['A′','M′','N′','B′'],'Перпендикулярная боковая сборка');
 const current=[],potential=[];m.arm.forEach((g,i)=>(i===0||i===3?current:potential).push(...g));
 const ch=m.mapping==='horizontal'?3-channel:null;
 add(current,ch,current.map(()=> 'I'),'Токовые контакты ТТРКГ');add(potential,ch,potential.map(()=> 'U'),'Измерительные контакты ТТРКГ');return traces;
}
function waveform(t,row=1,kind=t.kind){
 const suffix=row===1?'':String(row),labels={bei_raw:'Исходный BEI',bei_slow:'Базовый BEI (< 0,5 Гц)',bei_pulse_band:'Пульсовая полоса BEI (0,7–12 Гц)',ecg_raw:'Исходная ЭКГ',ecg_selected:'ЭКГ с выбранными фильтрами'},name=labels[kind]+' · канал '+t.channel;
 return {type:'scattergl',mode:'lines',x:t.t,y:t.y,name,legendgroup:String(t.channel),xaxis:'x'+suffix,yaxis:'y'+suffix,showlegend:row===1,
  line:{color:colors[t.channel-1],width:kind==='bei_slow'?2:1,dash:kind==='ecg_selected'?'dash':'solid'},hovertemplate:'t=%{x:.3f} с; %{y:.5g}<extra>'+name+'</extra>'};
}
function overview(){const r=D.records[ri];return [...r.traces.map(t=>waveform(t,t.row)),...r.ecg_variants[filterKey()].map(t=>waveform(t,4,'ecg_selected'))];}
function detailTraces(){const r=D.records[ri];return [1,2].flatMap(ch=>['00',filterKey()].map((key,i)=>waveform({channel:ch,t:r.detail_t,y:r.ecg_variants[key][ch-1].detail_y},1,i?'ecg_selected':'ecg_raw')));}
function spectrumTraces(){const r=D.records[ri];return [1,2].flatMap(ch=>['00',filterKey()].map((key,i)=>({type:'scatter',mode:'lines',x:r.psd_frequency_hz,y:r.ecg_variants[key][ch-1].psd,name:(i?'Выбранные фильтры':'Исходная ЭКГ')+' · канал '+ch,line:{color:colors[ch-1],width:1.2,dash:i?'dash':'solid'},hovertemplate:'f=%{x:.3f} Гц; S=%{y:.4g} мВ²/Гц<extra></extra>'})));}
function phaseDecor(rows=4,window=null){
 const r=D.records[ri],b=r.breathing,shapes=[],annotations=[];
 for(let row=1;row<=rows;row++){
  const suffix=row===1?'':String(row),xref='x'+suffix,yref='y'+suffix+' domain';
  b.phases.forEach((p,i)=>{
   const lo=window?Math.max(p.start,window[0]):p.start,hi=window?Math.min(p.end,window[1]):p.end;if(hi<=lo)return;
   shapes.push({type:'rect',xref,yref,x0:lo,x1:hi,y0:0,y1:1,fillcolor:phaseColors[i],opacity:.48,line:{width:0},layer:'below'});
   const label=p.label.replace(' дыхание','<br>дыхание').replace('Задержка на','Задержка<br>на');
   annotations.push({xref,yref,x:(lo+hi)/2,y:1,text:label,showarrow:false,yanchor:'bottom',font:{size:9,color:'#334155'}});
  });
  (b.boundary_uncertainty_ranges_samples||[]).forEach(v=>shapes.push({type:'rect',xref,yref,x0:v[0]/r.fs,x1:v[1]/r.fs,y0:0,y1:1,fillcolor:'#777',opacity:.22,line:{width:0},layer:'below'}));
  (b.boundaries_samples||[]).forEach(n=>shapes.push({type:'line',xref,yref,x0:n/r.fs,x1:n/r.fs,y0:0,y1:1,line:{color:'#555',width:1,dash:'dot'}}));
 }return {shapes,annotations};
}
function text(){
 const r=D.records[ri],m=r.montage;
 document.getElementById('experiment-description').textContent=r.label+'; рабочая частота отсчётов '+r.fs+' Гц; последний отсчёт '+r.duration.toLocaleString('ru-RU',{maximumFractionDigits:3})+' с. Синий: канал 1; оранжевый: канал 2.';
 document.getElementById('experiment-localisation').textContent=m.mapping==='unknown'?'Монтаж этой записи не локализован на КТ. Показаны зарегистрированные сигналы.':m.mapping==='unknown_vertical'?'Показан сценарий перпендикулярной сборки и ТТРКГ; соответствие каналам не установлено.':'Электроды показаны в номинальном сценарии; координаты в эксперименте не измерены. Основной боковой монтаж условно соответствует каналу '+channel+'.';
 document.getElementById('experiment-filter-description').textContent='Сейчас: '+({'00':'все фильтры отключены — исходная ЭКГ','10':'только режекторы 50/100/150 Гц','01':'только полоса 0,5–40 Гц','11':'режекторы 50/100/150 Гц, затем полоса 0,5–40 Гц'}[filterKey()])+'.';
 const phases=r.breathing.phases.map((p,i)=>`<span style="background:${phaseColors[i]};padding:4px 7px">${p.label}</span>`).join(' → ');
 document.getElementById('experiment-phase-description').innerHTML=(phases||'Дыхательные режимы не размечены.')+'<br>Кандидатная разметка, не принята автором. Серые полосы: неоднозначность границ; пунктир: выбранная кандидатная граница.';
}
async function update(){
 if(!ready)return;text();const r=D.records[ri],g=geometry(),w=overview(),dd=detailTraces(),ss=spectrumTraces();
 const indices=g.map((_,i)=>1+D.tissues.length+i);
 await Plotly.restyle(ct,{x:g.map(t=>t.x),y:g.map(t=>t.y),z:g.map(t=>t.z),text:g.map(t=>t.text),marker:g.map(t=>t.marker),visible:g.map(t=>t.visible),showlegend:g.map(t=>t.showlegend)},indices);
 await Plotly.restyle(signals,{x:w.map(t=>t.x),y:w.map(t=>t.y)});
 const layout={...phaseDecor()};for(let row=1;row<=4;row++){const suffix=row===1?'':row;layout['xaxis'+suffix+'.range']=[0,r.duration];layout['yaxis'+suffix+'.autorange']=true;}
 await Plotly.relayout(signals,layout);
 await Plotly.restyle(detail,{x:dd.map(t=>t.x),y:dd.map(t=>t.y)});
 await Plotly.relayout(detail,{...phaseDecor(1,[r.detail_t[0],r.detail_t.at(-1)]),'xaxis.range':[r.detail_t[0],r.detail_t.at(-1)],'yaxis.autorange':true});
 await Plotly.restyle(spectrum,{x:ss.map(t=>t.x),y:ss.map(t=>t.y)});await Plotly.relayout(spectrum,{'xaxis.range':[0,r.fs/2],'yaxis.autorange':true});
 ct.dataset.camera=JSON.stringify(ct._fullLayout.scene.camera);ct.dataset.record=r.record_id;ct.dataset.electrodeCounts=JSON.stringify(g.map(t=>t.x.length));ct.dataset.tissues=JSON.stringify(D.tissues.map(t=>t.name));
 signals.dataset.record=r.record_id;signals.dataset.phaseCount=r.breathing.phases.length;signals.dataset.filters=filterKey();
 detail.dataset.phaseCount=detail.layout.shapes.filter(t=>phaseColors.includes(t.fillcolor)).length;
 detail.dataset.rawEqualsSelected=String(dd[0].y.every((v,i)=>v===dd[1].y[i])&&dd[2].y.every((v,i)=>v===dd[3].y[i]));
 spectrum.dataset.filters=filterKey();spectrum.dataset.nyquistHz=r.fs/2;
}
async function init(){
 const mesh=(m,name,color,opacity)=>({type:'mesh3d',...xyz(m.vertices),i:m.faces.map(p=>p[0]),j:m.faces.map(p=>p[1]),k:m.faces.map(p=>p[2]),opacity,color,name,showlegend:true,hoverinfo:'skip'});
 const body=mesh(D,'Поверхность КТ и цилиндры рук','#b9b0a8',.16),tissues=D.tissues.map(t=>mesh(t,t.name,t.color,t.opacity)),config={responsive:true,displaylogo:false};
 await Plotly.newPlot(ct,[body,...tissues,...geometry()],{title:{text:'КТ: тканевые маски и номинальный монтаж'},height:790,scene:{aspectmode:'data',xaxis:{title:{text:'x LPS, мм'},range:[-430,430]},yaxis:{title:{text:'y LPS, мм'},range:[-180,180]},zaxis:{title:{text:'z LPS, мм'},range:[-380,0]},camera:{eye:{x:-1.3,y:-1.8,z:.9}},uirevision:'experiment-camera-fixed'},uirevision:'experiment-camera-fixed',legend:{orientation:'h',y:-.08},margin:{t:60,b:170,l:0,r:0}},config);
 await Plotly.newPlot(signals,overview(),{title:{text:'Общий, базовый и пульсовой импеданс; ЭКГ'},height:940,grid:{rows:4,columns:1,pattern:'independent',ygap:.24},xaxis:{},xaxis2:{matches:'x'},xaxis3:{matches:'x'},xaxis4:{title:{text:'t, с'}},yaxis:{title:{text:'Общий BEI, Ом'}},yaxis2:{title:{text:'Базовый BEI, Ом'}},yaxis3:{title:{text:'Пульсовая полоса, мОм'}},yaxis4:{title:{text:'ЭКГ, мВ'}},legend:{orientation:'h',y:-.08},margin:{t:90,b:100,l:80,r:15},...phaseDecor()},config);
 await Plotly.newPlot(detail,detailTraces(),{title:{text:'ЭКГ: исходный и выбранный вариант, фрагмент 5 с'},height:390,xaxis:{title:{text:'t, с'}},yaxis:{title:{text:'ЭКГ, мВ'}},legend:{orientation:'h',y:-.25},margin:{t:80,b:110,l:65,r:15}},config);
 await Plotly.newPlot(spectrum,spectrumTraces(),{title:{text:'Спектральная плотность мощности ЭКГ: вся запись'},height:390,xaxis:{title:{text:'f, Гц'}},yaxis:{title:{text:'S, мВ²/Гц'},type:'log'},legend:{orientation:'h',y:-.25},margin:{t:60,b:110,l:75,r:15}},config);
 ct.on('plotly_relayout',()=>{ct.dataset.camera=JSON.stringify(ct._fullLayout.scene.camera);});ready=true;await update();
 const refresh=()=>{queue=queue.then(update).catch(e=>{document.getElementById('experiment-description').textContent='Ошибка представления: '+e.message;});};
 recordControl.onchange=()=>{ri=Number(recordControl.value);refresh();};channelControl.onchange=()=>{channel=Number(channelControl.value);refresh();};notch.onchange=band.onchange=refresh;
}
init().catch(e=>{document.getElementById('experiment-description').textContent='Ошибка представления: '+e.message;});
})();
