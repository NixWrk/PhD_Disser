(() => {
const D=beiReviewPayload,root=document.getElementById('bei-review'),control=n=>root.querySelector('[data-control="'+n+'"]');
const time=document.getElementById('bei-time'),levels=document.getElementById('bei-levels');
const phaseColors=['#d7ecd9','#d4e5ff','#f6ead0','#eadcf5','#eee'];
const colour=i=>'hsl('+Math.round(i*137.508%360)+',65%,38%)',fmt=n=>n===null?'—':Number(n).toLocaleString('ru-RU',{maximumFractionDigits:5});
let enabled=D.records.map(()=>true),queue=Promise.resolve();
D.modes.forEach((m,i)=>control('phase').add(new Option(D.mode_labels[i],m)));
D.records.forEach((r,i)=>{
 const label=document.createElement('label'),input=document.createElement('input');
 input.type='checkbox';input.checked=true;input.dataset.record=i;label.style.color=colour(i);
 label.append(input,document.createTextNode(' '+r.label));control('records').append(label);
 input.onchange=()=>{enabled[i]=input.checked;refresh();};
});
function selected(){
 const mode=control('phase').value,channel=control('channel').value,scope=control('analysis').checked?'analysis':'full',items=[];
 D.records.forEach((r,i)=>{if(enabled[i])r.channels.forEach(c=>{if(channel==='both'||String(c.channel)===channel){
  const parts=c.phases.filter(p=>(mode==='all'||p.mode===mode)&&p[scope]).map(p=>({...p,segment:p[scope]}));
  if(parts.length)items.push({r,i,channel:c.channel,parts});
 }});});return items;
}
function keepManualAxes(plot,layout,key){
 if(plot.dataset.viewKey===key&&plot.layout)for(const axis of ['xaxis','yaxis']){
  const old=plot.layout[axis];
  if(old&&old.autorange===false&&old.range){layout[axis].range=[...old.range];layout[axis].autorange=false;}
 }
 plot.dataset.viewKey=key;return layout;
}
async function update(){
 const items=selected(),mode=control('phase').value,centred=control('scale').value==='centred';
 const threshold=Number(control('threshold').value);
 if(!Number.isFinite(threshold)||threshold<0){control('status').textContent='Порог должен быть конечным неотрицательным числом в омах.';return;}
 const traces=[],summaries=[],rows=[],spread=control('spread').value;
 items.forEach(({r,i,channel,parts})=>{
  const name=r.label+' · BEI'+channel,x=[],y=[],custom=[],dx=[],dy=[],dt=[],medians=[],plus=[],minus=[],labels=[];
  const dash=channel===1?'solid':'dash';
  parts.forEach(p=>{
   const s=p.segment,offset=mode==='all'?0:p.start,baseline=centred?s.stats.median:0;
   s.t.forEach((t,j)=>{
    const raw=s.y[j],value=raw===null?null:raw-baseline;
    x.push(t-offset);y.push(value);custom.push([t,raw,p.label]);
    if(raw!==null&&s.stats.median!==null&&raw<s.stats.median-threshold){dx.push(t-offset);dy.push(value);dt.push([t,raw,p.label]);}
   });x.push(null);y.push(null);custom.push([null,null,'']);
   const a=s.stats;labels.push(p.label);medians.push(a.median);
   plus.push(a.median===null?0:spread==='range'?a.maximum-a.median:spread==='iqr'?a.q75-a.median:0);
   minus.push(a.median===null?0:spread==='range'?a.median-a.minimum:spread==='iqr'?a.median-a.q25:0);
   rows.push([r.label,'BEI'+channel,p.label,fmt(s.start)+'–'+fmt(s.end),fmt(a.median),fmt(a.minimum),fmt(a.maximum),fmt(a.q25),fmt(a.q75),a.below_zero,a.nonfinite,a.n]);
  });
  traces.push({type:'scatter',mode:'lines',x,y,customdata:custom,name,line:{color:colour(i),width:1.3,dash},connectgaps:false,showlegend:false,
    hovertemplate:'%{customdata[2]}<br>t записи=%{customdata[0]:.3f} с<br>Исходный BEI=%{customdata[1]:.6g} Ом<br>На графике=%{y:.6g} Ом<extra>'+name+'</extra>'});
  if(control('dips').checked)traces.push({type:'scatter',mode:'markers',x:dx,y:dy,customdata:dt,name:'Нисходящее отклонение · '+name,showlegend:false,
    marker:{color:'#ce1b35',size:6,symbol:'triangle-down'},hovertemplate:'%{customdata[2]}<br>t записи=%{customdata[0]:.3f} с<br>Исходный BEI=%{customdata[1]:.6g} Ом<extra>'+name+'</extra>'});
  summaries.push({type:'scatter',mode:'lines+markers',x:labels,y:medians,name,showlegend:false,line:{color:colour(i),dash},marker:{color:colour(i),size:7,symbol:channel===1?'circle':'diamond'},
    error_y:{type:'data',symmetric:false,array:plus,arrayminus:minus,visible:spread!=='none',color:colour(i),thickness:1,width:3},
    hovertemplate:'%{x}<br>Медиана=%{y:.6g} Ом<extra>'+name+'</extra>'});
 });
 const shapes=[{type:'line',xref:'paper',yref:'y',x0:0,x1:1,y0:0,y1:0,line:{color:'#444',width:1,dash:'dot'}}],annotations=[];
 const single=items.length&&new Set(items.map(x=>x.r.record_id)).size===1;
 if(single){
  const item=items[0],offset=mode==='all'?0:item.parts[0].start;
  item.parts.forEach(p=>{
   const lo=p.segment.start-offset,hi=p.segment.end-offset,colour=phaseColors[D.modes.indexOf(p.mode)];
   shapes.push({type:'rect',xref:'x',yref:'paper',x0:lo,x1:hi,y0:0,y1:1,fillcolor:colour,opacity:.5,line:{width:0},layer:'below'});
   annotations.push({x:(lo+hi)/2,y:1.04,xref:'x',yref:'paper',text:p.label,showarrow:false,font:{size:10}});
  });
  const left=item.parts[0].segment.start,right=item.parts.at(-1).segment.end;
  item.r.ambiguity_s.forEach(([lo,hi])=>{lo=Math.max(lo,left);hi=Math.min(hi,right);if(hi>lo)shapes.push({type:'rect',xref:'x',yref:'paper',x0:lo-offset,x1:hi-offset,y0:0,y1:1,fillcolor:'#777',opacity:.25,line:{width:0},layer:'below'});});
 }
 const base={template:{layout:{xaxis:{gridcolor:'#e3e6e8'},yaxis:{gridcolor:'#e3e6e8'}}},margin:{t:80,b:60,l:85,r:20},paper_bgcolor:'#fff',plot_bgcolor:'#fff',showlegend:false};
 const timeKey='bei-time-'+mode+'-'+control('scale').value+'-'+control('analysis').checked;
 const levelsKey='bei-levels-'+mode+'-'+control('analysis').checked;
 await Plotly.react(time,traces,keepManualAxes(time,{...base,height:570,title:{text:'Рисунок А, а. BEI по времени'},shapes,annotations,
  xaxis:{autorange:true,title:{text:mode==='all'?'Время от начала каждой записи, с':'Время от кандидатного начала этапа, с'}},
  yaxis:{autorange:true,title:{text:centred?'BEI − медиана этапа, Ом':'Исходный BEI, Ом'}},uirevision:timeKey},timeKey),{responsive:true,displaylogo:false});
 await Plotly.react(levels,summaries,keepManualAxes(levels,{...base,height:460,title:{text:'Рисунок А, б. Уровни BEI на этапах дыхания'},
  xaxis:{autorange:true,categoryorder:'array',categoryarray:D.mode_labels,title:{text:'Кандидатный этап дыхания'}},yaxis:{autorange:true,title:{text:'Исходный BEI, Ом'}},uirevision:levelsKey},levelsKey),{responsive:true,displaylogo:false});
 const headers=['Запись','Поле','Этап','Интервал записи, с','Медиана, Ом','Минимум, Ом','Максимум, Ом','Q25, Ом','Q75, Ом','Ниже нуля, отсчётов','Неконечных','Всего отсчётов'];
 const table=document.createElement('table');table.style.borderCollapse='collapse';
 const addRow=(values,tag)=>{const tr=document.createElement('tr');values.forEach(v=>{const td=document.createElement(tag);td.textContent=String(v);td.style.padding='5px';td.style.border='1px solid #ccc';tr.append(td);});table.append(tr);};
 addRow(headers,'th');rows.forEach(row=>addRow(row,'td'));control('table').replaceChildren(table);
 const key=document.createElement('table');key.style.borderCollapse='collapse';
 items.forEach(({r,i,channel})=>{
  const tr=document.createElement('tr'),label=document.createElement('td'),meaning=document.createElement('td');
  label.textContent=r.label+' · BEI'+channel;label.style.color=colour(i);
  meaning.textContent=(channel===1?'Сплошная линия, круглые метки':'Штриховая линия, ромбовидные метки')+'; слева '+(centred?'BEI за вычетом медианы каждого выбранного этапа':'исходный BEI')+'. Справа — медианы этапов; '+(spread==='range'?'усы показывают минимум и максимум':spread==='iqr'?'усы показывают квартили 25 и 75%':'усы отключены')+'.';
  for(const td of [label,meaning]){td.style.padding='6px';td.style.border='1px solid #ccc';tr.append(td);}key.append(tr);
 });
 const caption=document.createElement('p');caption.textContent='Пояснения каждой выбранной линии на обоих графиках. Красные треугольники — отображённые отсчёты ниже медианы этапа на выбранный порог; это кандидаты нисходящих отклонений, а не доказанные дефекты. Серая точечная горизонталь — ноль текущей шкалы.';
 control('line-key').replaceChildren(caption,key);
 control('status').textContent=items.length?'Показано полей: '+items.length+'; интервалов: '+rows.length+'. '+(mode==='all'?'Время каждой записи отсчитывается отдельно; записи не одновременны. ':'Этапы совмещены по кандидатному началу; длительности сохранены. ')+(control('analysis').checked?'Использованы обрезанные кандидатные интервалы расчёта уровней. ':'Показаны полные кандидатные этапы; их медианы могут отличаться от расчётных уровней 33.21.'):'Нет данных для выбранных записей, полей и интервалов.';
 control('phase-key').textContent=single?'Цветной фон — кандидатные этапы дыхания; серые полосы — неоднозначность границ из 11.20.':'При нескольких записях фон этапов не накладывается: их границы различаются. Для проверки границ оставьте одну запись.';
 time.dataset.fieldCount=String(items.length);time.dataset.intervalCount=String(rows.length);time.dataset.phase=mode;time.dataset.scale=control('scale').value;
 time.dataset.analysis=String(control('analysis').checked);time.dataset.threshold=String(threshold);time.dataset.enabledRecords=JSON.stringify(enabled);
 time.dataset.dips=String(control('dips').checked);
 levels.dataset.spread=spread;
}
function refresh(){queue=queue.then(update).catch(e=>{control('status').textContent='Ошибка просмотра BEI: '+e.message;});}
for(const n of ['phase','channel','scale','analysis','spread','dips'])control(n).onchange=refresh;
control('threshold').oninput=refresh;
function choose(predicate){enabled=D.records.map(predicate);root.querySelectorAll('input[data-record]').forEach(el=>{el.checked=enabled[Number(el.dataset.record)];});refresh();}
control('all').onclick=()=>choose(()=>true);control('main').onclick=()=>choose(r=>r.primary);control('none').onclick=()=>choose(()=>false);
control('reset').onclick=()=>{queue=queue.then(async()=>{await Plotly.relayout(time,{'xaxis.autorange':true,'yaxis.autorange':true});await Plotly.relayout(levels,{'xaxis.autorange':true,'yaxis.autorange':true});});};
refresh();
})();
