(() => {
const D=experimentPayload,recordControl=document.getElementById('experiment-record'),channelControl=document.getElementById('experiment-channel');
const ct=document.getElementById('experiment-ct'),signals=document.getElementById('experiment-signals'),detail=document.getElementById('experiment-ecg-detail');
const colors=['#176aaf','#e47823'];let ri=3,channel=1,ready=false;
D.records.forEach((r,i)=>recordControl.add(new Option(r.label,i)));recordControl.value=ri;
const xyz=points=>({x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2])});
function geometry(){
 const m=D.records[ri].montage,traces=[];
 const add=(points,ch,roles,name)=>traces.push({type:'scatter3d',mode:'markers+text',...xyz(points),name,text:roles,showlegend:true,
  marker:{size:7,color:ch?colors[ch-1]:'#666',symbol:'circle'},textposition:'top center',hovertemplate:name+'<br>x=%{x:.2f}; y=%{y:.2f}; z=%{z:.2f} мм<extra></extra>'});
 add(m.lateral,m.mapping==='horizontal'||m.mapping==='cross'?channel:null,['A','M','N','B'],'Основная боковая сборка');
 add(m.perpendicular,m.mapping==='cross'?3-channel:null,['A′','M′','N′','B′'],'Перпендикулярная боковая сборка');
 const current=[],potential=[];m.arm.forEach((group,i)=>(i===0||i===3?current:potential).push(...group));
 const armChannel=m.mapping==='horizontal'?3-channel:null;
 add(current,armChannel,current.map(()=> 'I'),'Токовые контакты ТТРКГ');
 add(potential,armChannel,potential.map(()=> 'U'),'Измерительные контакты ТТРКГ');
 traces.forEach(t=>{if(t.x.length===0){t.showlegend=false;t.visible=false;}});return traces;
}
function traceLabel(t){
 const kind={bei_raw:'BEI, исходный',bei_slow:'BEI, ниже 0,5 Гц',bei_pulse_band:'BEI, 0,7–12 Гц',ecg_filtered:'ЭКГ, обработанная',ecg_raw:'ЭКГ, исходная'}[t.kind];
 return kind+' · канал '+t.channel;
}
function waveform(t,overview=true){
 const axis=overview&&t.row>1?String(t.row):'';
 return {type:'scattergl',mode:'lines',x:t.t,y:t.y,name:traceLabel(t),legendgroup:String(t.channel),
  xaxis:'x'+axis,yaxis:'y'+axis,showlegend:overview?t.row===1:true,
  line:{color:colors[t.channel-1],width:t.kind==='bei_slow'?2:1,dash:t.kind==='bei_slow'||t.kind==='ecg_filtered'&&!overview?'dash':'solid'},
  hovertemplate:'t=%{x:.3f} с; %{y:.5g}<extra>'+traceLabel(t)+'</extra>'};
}
function shapes(){
 const r=D.records[ri],b=r.breathing,result=[];
 for(let row=1;row<=3;row++){
 const suffix=row===1?'':String(row),xref='x'+suffix,yref='y'+suffix+' domain';
 (b.boundary_uncertainty_ranges_samples||[]).forEach(range=>result.push({type:'rect',xref,yref,x0:range[0]/r.fs,x1:range[1]/r.fs,y0:0,y1:1,fillcolor:'#aaa',opacity:.12,line:{width:0},layer:'below'}));
 (b.boundaries_samples||[]).forEach(n=>result.push({type:'line',xref,yref,x0:n/r.fs,x1:n/r.fs,y0:0,y1:1,line:{color:'#888',width:1,dash:'dot'}}));
 }return result;
}
function text(){
 const r=D.records[ri],m=r.montage;
 document.getElementById('experiment-description').textContent=r.label+'; рабочая частота отсчётов '+r.fs+' Гц; последний отсчёт '+r.duration.toLocaleString('ru-RU',{maximumFractionDigits:3})+' с. Синий: канал 1; оранжевый: канал 2.';
 document.getElementById('experiment-localisation').textContent=m.mapping==='unknown'?'Монтаж этой записи не локализован на КТ. Показаны зарегистрированные сигналы.':m.mapping==='unknown_vertical'?'Показан сценарий перпендикулярной сборки и ТТРКГ; соответствие каналам не установлено.':'Электроды показаны в номинальном сценарии; координаты в эксперименте не измерены. Основной боковой монтаж условно соответствует каналу '+channel+'.';
}
async function update(){
 if(!ready)return;text();
 const g=geometry(),w=D.records[ri].traces.map(t=>waveform(t)),dd=D.records[ri].detail.map(t=>waveform(t,false));
 await Plotly.restyle(ct,{x:g.map(t=>t.x),y:g.map(t=>t.y),z:g.map(t=>t.z),text:g.map(t=>t.text),marker:g.map(t=>t.marker),visible:g.map(t=>t.visible===false?false:true),showlegend:g.map(t=>t.showlegend)},[1,2,3,4]);
 await Plotly.restyle(signals,{x:w.map(t=>t.x),y:w.map(t=>t.y)});
 await Plotly.relayout(signals,{shapes:shapes(),'xaxis.range':[0,D.records[ri].duration],'xaxis2.range':[0,D.records[ri].duration],'xaxis3.range':[0,D.records[ri].duration],'yaxis.autorange':true,'yaxis2.autorange':true,'yaxis3.autorange':true});
 await Plotly.restyle(detail,{x:dd.map(t=>t.x),y:dd.map(t=>t.y)});
 await Plotly.relayout(detail,{'xaxis.range':[dd[0].x[0],dd[0].x[dd[0].x.length-1]],'yaxis.autorange':true});
 ct.dataset.camera=JSON.stringify(ct._fullLayout.scene.camera);
 ct.dataset.record=D.records[ri].record_id;
 signals.dataset.record=D.records[ri].record_id;
 ct.dataset.electrodeCounts=JSON.stringify(ct.data.slice(1).map(t=>t.x.length));
}
async function init(){
 const f=D.faces,body={type:'mesh3d',...xyz(D.vertices),i:f.map(p=>p[0]),j:f.map(p=>p[1]),k:f.map(p=>p[2]),opacity:.24,color:'#b9b0a8',name:'Поверхность КТ и цилиндры рук',showlegend:false,hoverinfo:'skip'};
 const config={responsive:true,displaylogo:false};
 await Plotly.newPlot(ct,[body,...geometry()],{title:{text:'КТ и номинальный монтаж'},height:710,scene:{aspectmode:'data',xaxis:{title:{text:'x LPS, мм'},range:[-430,430]},yaxis:{title:{text:'y LPS, мм'},range:[-180,180]},zaxis:{title:{text:'z LPS, мм'},range:[-380,0]},camera:{eye:{x:-1.3,y:-1.8,z:.9}},uirevision:'experiment-camera-fixed'},uirevision:'experiment-camera-fixed',legend:{orientation:'h',y:-.08},margin:{t:60,b:110,l:0,r:0}},config);
 await Plotly.newPlot(signals,D.records[ri].traces.map(t=>waveform(t)),{title:{text:'Импеданс и ЭКГ за всю запись'},height:710,grid:{rows:3,columns:1,pattern:'independent'},xaxis:{title:{text:'t, с'}},xaxis2:{title:{text:'t, с'},matches:'x'},xaxis3:{title:{text:'t, с'},matches:'x'},yaxis:{title:{text:'BEI, Ом'}},yaxis2:{title:{text:'Полосовой BEI, мОм'}},yaxis3:{title:{text:'ЭКГ, мВ'}},legend:{orientation:'h',y:-.18},margin:{t:60,b:130,l:65,r:15},shapes:shapes()},config);
 await Plotly.newPlot(detail,D.records[ri].detail.map(t=>waveform(t,false)),{title:{text:'Пятисекундный фрагмент исходной и обработанной ЭКГ'},height:380,xaxis:{title:{text:'t, с'}},yaxis:{title:{text:'ЭКГ, мВ'}},legend:{orientation:'h',y:-.25},margin:{t:60,b:110,l:65,r:15}},config);
 ct.on('plotly_relayout',()=>{ct.dataset.camera=JSON.stringify(ct._fullLayout.scene.camera);});
 ready=true;await update();recordControl.onchange=()=>{ri=Number(recordControl.value);update();};channelControl.onchange=()=>{channel=Number(channelControl.value);update();};
}
init().catch(e=>{document.getElementById('experiment-description').textContent='Ошибка представления: '+e.message;});
})();
