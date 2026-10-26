function geometry(){
 const m=D.records[ri].montage,traces=[];
 const add=(p,lead,roles,name)=>traces.push({type:'scatter3d',mode:'markers+text',...xyz(p),name,text:roles,
  showlegend:!!p.length,visible:!!p.length,marker:{size:7,color:leadColors[lead]},textposition:'top center',
  hovertemplate:name+'<br>x=%{x:.2f}; y=%{y:.2f}; z=%{z:.2f} мм<extra></extra>'});
 add(m.lateral,'lateral_horizontal',['A','M','N','B'],'Боковая сборка');
 add(m.inactive,'inactive',m.inactive.map(()=> '×'),'Неактивные контакты: физическое наличие — допущение');
 add(m.current,'ttrkg',m.current.map(()=> 'I'),'Токовые контакты ТТРКГ');
 add(m.potential,'ttrkg',m.potential.map(()=> 'U'),'Измерительные контакты ТТРКГ');
 add([],'tepc',[],'');return traces;
}
// SUBJECT_TEXT
function text(){
 const r=D.records[ri];
 document.getElementById('experiment-description').textContent=r.label+'; частота отсчётов '+r.fs+
  ' Гц; последний отсчёт '+r.duration.toLocaleString('ru-RU',{maximumFractionDigits:3})+' с. '+
  (sampling.value==='full'?'Показаны все '+r.full_samples.rows+' отсчётов каждого сигнала.':
   'Быстрый обзор сохраняет экстремумы 900 интервалов. Для всех точек выберите полный режим.');
 document.getElementById('experiment-localisation').textContent=r.montage.description;
 channelVisible.forEach((_,i)=>{document.getElementById('experiment-lead-label-'+(i+1)).textContent=
  leadNames[physicalRole(i+1)]+' на всех графиках';});
 charts.forEach(chart=>chart.curves.forEach(({ch,selected,input})=>{
  input.parentElement.style.color=leadColors[physicalRole(ch)];
  input.parentElement.lastChild.textContent=' '+leadNames[physicalRole(ch)]+(r.lead_notes[physicalRole(ch)]||'')+
   (chart.compare?(selected?' · выбранные фильтры':' · исходная ЭКГ'):'');
 }));
 document.getElementById('experiment-filter-description').textContent='Сейчас: '+
  ({'00':'исходная ЭКГ; оба этапа отключены','10':'только режекторы 50/100/150 Гц',
    '01':'только полоса 0,5–40 Гц','11':'режекторы 50/100/150 Гц, затем полоса 0,5–40 Гц'}[filterKey()])+'.';
 const colors=r.breathing.phases.length===1?['#d4e5ff']:phaseColors;
 const phases=r.breathing.phases.map((p,i)=>`<span style="background:${colors[i]};padding:4px 7px">${p.label}</span>`).join(' → ');
 document.getElementById('experiment-phase-description').innerHTML=phases+'<br>'+r.breathing.note;
}
