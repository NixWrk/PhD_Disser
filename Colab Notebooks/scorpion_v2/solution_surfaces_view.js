(() => {
const D=solutionPayload, colors=['#155da0','#ef8531','#3b9746','#b34444','#8154a4','#824c22','#bd55a0','#688c21','#2c9ba2','#6c65c2','#da446b','#8c7220','#555e6b'];
const select=document.getElementById('solution-case'), slider=document.getElementById('solution-h'), box=document.getElementById('solution-assemblies');
const plots=['solution-curves','solution-surfaces','solution-slice'].map(id=>document.getElementById(id));
let ci=0,hi=D.h.indexOf(20),enabled=D.cases[0].assemblies.map(()=>true),ready=false;
slider.value=hi;
D.cases.forEach((c,i)=>select.add(new Option(c.label,i)));
const fmt=v=>Number(v).toLocaleString('ru-RU',{maximumFractionDigits:4});
function curve(item,hindex){
 const k=D.sizes.indexOf(item.size_mm),y=D.cases[ci].observations[D.cases[ci].assemblies.indexOf(item)];
 const x=[],r=[];
 D.q.forEach((q,j)=>{const a=y/D.f[hindex][k][j],b=q*a,ok=a>=D.bounds[0][0]&&a<=D.bounds[0][1]&&b>=D.bounds[1][0]&&b<=D.bounds[1][1];x.push(ok?b:null);r.push(ok?a:null);});
 return {x,y:r};
}
function checks(){
 box.innerHTML='';D.cases[ci].assemblies.forEach((item,i)=>{
 const label=document.createElement('label'),input=document.createElement('input');
 input.type='checkbox';input.checked=enabled[i];input.dataset.assembly=i;
 label.style.color=colors[i];label.append(input,document.createTextNode(' '+item.label));box.append(label);
 input.onchange=()=>{enabled[i]=input.checked;update();};
 });
}
function pairs(all){
 const r=D.cases[ci].intersections.filter(p=>enabled[p[0]]&&enabled[p[1]]&&(all||p[4]===D.h[hi]));
 return {x:r.map(p=>p[3]),y:r.map(p=>p[2]),z:r.map(p=>p[4]),text:r.map(p=>D.cases[ci].assemblies[p[0]].label+' / '+D.cases[ci].assemblies[p[1]].label+'<br>Максимальное расхождение основного ряда: '+fmt(p[6])+'%')};
}
function data(){
 const cc=D.cases[ci],slice=[],curves=[],surfaces=[],hg=D.h.map((_,i)=>i).filter(i=>i%2===0||i===D.h.length-1),qg=D.q.map((_,i)=>i).filter(i=>i%4===0||i===D.q.length-1);
 cc.assemblies.forEach((item,i)=>{
 const c=curve(item,hi),base={name:item.label,showlegend:false,visible:enabled[i],hovertemplate:'ρ₂=%{x:.4g}; ρ₁=%{y:.4g} Ом·м<extra>'+item.label+'</extra>'};
 const line={color:colors[i],width:item.reference?4:2,dash:item.reference?'solid':'dash'};
 curves.push({...base,type:'scatter3d',mode:'lines',x:c.x,y:c.y,z:c.x.map(()=>D.h[hi]),line});
 slice.push({...base,type:'scatter',mode:'lines',x:c.x,y:c.y,line});
 const full=hg.map(j=>curve(item,j)),x=full.map(c=>qg.map(k=>c.x[k])),y=full.map(c=>qg.map(k=>c.y[k]));
 surfaces.push({...base,type:'surface',x,y,z:hg.map((j,n)=>qg.map((_,k)=>x[n][k]===null?null:D.h[j])),opacity:.38,colorscale:[[0,colors[i]],[1,colors[i]]],showscale:false,connectgaps:false,
 hovertemplate:'ρ₂=%{x:.4g}; ρ₁=%{y:.4g} Ом·м<br>h=%{z:.3g} мм<extra>'+item.label+'</extra>'});
 });
 const show=document.getElementById('solution-pairs').checked,p=pairs(false),pa=pairs(true),pm={size:4,color:'#222'};
 const base={name:'Попарные пересечения',showlegend:false,visible:show,mode:'markers',marker:pm,hovertemplate:'%{text}<br>ρ₂=%{x:.4g}; ρ₁=%{y:.4g} Ом·м<extra></extra>'};
 curves.push({...base,type:'scatter3d',...p});surfaces.push({...base,type:'scatter3d',...pa});slice.push({...base,type:'scatter',x:p.x,y:p.y,text:p.text});
 const fit=cc.fit.parameters,fp=[cc.profile_rho1[hi],cc.profile_rho2[hi]],mark={color:'#ce1b35',size:7,symbol:'diamond'},name='Совместный минимум пяти размеров';
 surfaces.push({type:'scatter3d',mode:'markers',x:[fit[1]],y:[fit[0]],z:[fit[2]],name,showlegend:false,marker:mark,hovertemplate:name+'<br>ρ₂=%{x:.4g}; ρ₁=%{y:.4g} Ом·м<br>h=%{z:.4g} мм<extra></extra>'});
 const prof={mode:'markers',x:[fp[1]],y:[fp[0]],marker:mark,name:'Сеточный минимум при выбранном h',showlegend:false};
 curves.push({...prof,type:'scatter3d',z:[D.h[hi]]});slice.push({...prof,type:'scatter'});
 // The slice plane provides a geometric link without implying admissible parameters everywhere.
 surfaces.push({type:'surface',x:[[D.bounds[1][0],D.bounds[1][1]],[D.bounds[1][0],D.bounds[1][1]]],y:[[D.bounds[0][0],D.bounds[0][0]],[D.bounds[0][1],D.bounds[0][1]]],z:[[D.h[hi],D.h[hi]],[D.h[hi],D.h[hi]]],opacity:.08,colorscale:[[0,'#aaa'],[1,'#aaa']],showscale:false,hoverinfo:'skip',name:'Плоскость выбранного h'});
 return [curves,surfaces,slice];
}
function layout(index){
 const axis=(title,range)=>({title:{text:title},range,autorange:false});
 const result={title:{text:['Рисунок 1. Кривые решений при заданном h','Рисунок 2. Поверхности решений и попарные пересечения','Рисунок 3. Срез поверхностей при заданном h'][index]},height:index===2?560:660,margin:{t:65,l:75,r:20,b:70},paper_bgcolor:'#fff',plot_bgcolor:'#fff',uirevision:'bounded-solutions-stable'};
 if(index<2)result.scene={xaxis:axis('ρ₂, Ом·м',D.bounds[1]),yaxis:axis('ρ₁, Ом·м',D.bounds[0]),zaxis:axis('h, мм',[.5,100]),aspectmode:'cube',uirevision:'bounded-solutions-camera',camera:{eye:{x:1.4,y:1.4,z:1}}};
 else {result.xaxis=axis('ρ₂, Ом·м',D.bounds[1]);result.yaxis=axis('ρ₁, Ом·м',D.bounds[0]);}
 return result;
}
async function update(){
 if(!ready)return;
 document.getElementById('solution-h-label').textContent=fmt(D.h[hi])+' мм';
 const v=data();
 // Only trace arrays change; scene.camera and axis ranges are untouched.
 for(let j=0;j<3;j++){
   const attrs={};for(const name of ['x','y','z','visible','text','name'])attrs[name]=v[j].map(t=>t[name]===undefined?null:t[name]);
   await Plotly.restyle(plots[j],attrs);
   plots[j].dataset.visibleAssemblies=JSON.stringify(plots[j].data.slice(0,13).map(t=>t.visible));
   plots[j].dataset.sliceH=String(D.h[hi]);
   if(j<2)plots[j].dataset.camera=JSON.stringify(plots[j]._fullLayout.scene.camera);
 }
 const cc=D.cases[ci],fit=cc.fit;
 document.getElementById('solution-status').textContent=cc.label+'; h = '+fmt(D.h[hi])+' мм. Совместная невязка пяти размеров: '+fmt(fit.residual_percent)+'%. Пересечения относятся только к указанным парам.';
}
async function init(){
 checks();const v=data();for(let j=0;j<3;j++)await Plotly.newPlot(plots[j],v[j],layout(j),{responsive:true,displaylogo:false});
 for(let j=0;j<2;j++)plots[j].on('plotly_relayout',()=>{plots[j].dataset.camera=JSON.stringify(plots[j]._fullLayout.scene.camera);});
 ready=true;await update();
 select.onchange=()=>{ci=Number(select.value);checks();update();};
 slider.onchange=()=>{hi=Number(slider.value);update();};
 slider.oninput=()=>{document.getElementById('solution-h-label').textContent=fmt(D.h[Number(slider.value)])+' мм';};
 document.getElementById('solution-pairs').onchange=update;
 document.getElementById('solution-all').onclick=()=>{enabled.fill(true);checks();update();};
 document.getElementById('solution-main').onclick=()=>{enabled=D.cases[ci].assemblies.map(a=>a.reference);checks();update();};
}
init().catch(e=>{document.getElementById('solution-status').textContent='Ошибка представления: '+e.message;});
})();
