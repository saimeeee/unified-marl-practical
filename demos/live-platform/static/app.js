'use strict';
const $=id=>document.getElementById(id);
const labels={warehouse:'库存补货',mpe:'MPE 协作',traffic:'交通信号',powergrid:'电网储能'};
const icons={warehouse:'▣',mpe:'◉',traffic:'╋',powergrid:'ϟ'};
const colors=['#27866d','#d56b37','#8b6aa5','#af8a32'];
let models=[],selected='warehouse',state=null,connected=false,busy=false;
let hiddenSessionId=null;
const retiredSessionIds=new Set();
const fmt=(v,d=2)=>Number(v).toLocaleString('zh-CN',{maximumFractionDigits:d});
function el(tag,cls,text){let n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;}
async function api(path,data){const response=await fetch(path,data?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}:{});const result=await response.json();if(!response.ok)throw Error(result.error||response.statusText);return result;}
function error(message){$('error').hidden=!message;$('error').textContent=message||'';}
function model(){return models.find(m=>m.id===$('model').value);}
function active(){return state && !['idle','completed','stopped','failed'].includes(state.status);}
function sameSelectedModel(next=state){return !!next?.model && next.model.id===$('model').value && next.model.scenario===selected;}
function visibleSession(next=state){return sameSelectedModel(next) && next.id!==hiddenSessionId;}
function clearView(loading=false){
  error('');
  $('empty').hidden=false;
  $('empty').querySelector('h3').textContent=loading?'正在加载模型与仿真环境':'让策略在场景中运行';
  $('empty').querySelector('p').textContent=loading?'正在准备新的独立回合，完成后可开始运行或单步观察。':'初始化后可逐步观察模型动作，或连续运行完整回合。';
  $('metrics').replaceChildren(...Array.from({length:4},()=>metricCard('等待运行','—','')));
  $('status').textContent=loading?'加载模型与环境…':'等待初始化';
  $('clock').textContent='—';
  $('progress').textContent='0 步';
  $('progress-bar').style.width='0%';
  $('source').textContent='每一帧由后端仿真状态生成';
  $('chart-title').textContent='运行指标走势';
  $('chart-min').textContent='—';
  $('chart-max').textContent='—';
  $('actions').replaceChildren(el('p','muted','模型执行后显示动作。'));
  $('action-count').textContent='0 个智能体';
  $('speed').value='1';
  $('export').href='#';
  $('export').classList.add('disabled');
  drawScene(null);
  drawChart([]);
}
function choose(scene,force=false){
  if((active()||busy)&&!force)return;
  if(!force)hiddenSessionId=state?.id??null;
  selected=scene;
  document.querySelectorAll('#scenes button').forEach(b=>b.classList.toggle('active',b.dataset.scene===scene));
  $('model').replaceChildren(...models.filter(m=>m.scenario===scene).map(m=>{let n=el('option','',m.algorithm);n.value=m.id;return n;}));
  details();
  clearView();
  controls();
}
function details(){const m=model();if(!m)return;$('scene-title').textContent=m.label;$('scene-detail').textContent=m.detail;$('model-training').textContent=m.training+' · 2026-10-28';$('note').textContent=m.note;}
function metricCard(label,value,unit){let card=el('div','metric');card.append(el('div','metric-label',label));let val=el('div','metric-value',value);val.append(el('small','',unit));card.append(val);return card;}
function controls(){
  const a=active(),matching=visibleSession()&&!busy,ready=connected&&matching&&state?.status==='ready';
  $('init').disabled=a||busy||!connected;
  for(let id of ['model','seed'])$(id).disabled=a||busy;
  document.querySelectorAll('#scenes button').forEach(b=>b.disabled=a||busy);
  $('play').disabled=!ready||state?.mode!=='paused';
  $('pause').disabled=!ready||state?.mode!=='running';
  $('step').disabled=!ready||state?.mode!=='paused';
  $('stop').disabled=!connected||!matching||!a||state?.mode==='stopping';
  $('speed').disabled=!connected||!matching||!a||state?.mode==='stopping';
  const exportable=matching&&!!state?.frame;
  $('export').classList.toggle('disabled',!exportable);
  $('export').href=exportable?'/api/export/'+state.id:'#';
}
function syncActiveSelection(next){
  if(next.model.scenario!==selected)choose(next.model.scenario,true);
  if($('model').value!==next.model.id){
    $('model').value=next.model.id;
    details();
    clearView();
  }
  $('seed').value=String(next.seed);
  const speed=String(next.speed);
  if(!Array.from($('speed').options).some(option=>option.value===speed)){
    const option=el('option','',speed+'×');option.value=speed;$('speed').append(option);
  }
  $('speed').value=speed;
  hiddenSessionId=null;
}
function receive(next){
  // A poll started before a new run or control request can arrive afterwards.
  if(next.id&&retiredSessionIds.has(next.id))return;
  if(state&&next.id===state.id&&next.seq<state.seq)return;
  if(busy&&next.id&&next.id===hiddenSessionId)return;
  const previousId=state?.id,newSession=!!next.id&&next.id!==previousId;
  if(newSession&&previousId)retiredSessionIds.add(previousId);
  state=next;connected=true;
  $('connection').textContent='● 服务已连接';$('connection').classList.remove('offline');
  if(newSession)clearView(next.status==='loading');
  if(next.model&&active())syncActiveSelection(next);
  controls();
  if(!visibleSession(next))return;
  const statuses={idle:'等待初始化',loading:'加载模型与环境…',finalizing:'正在保存结果…',ready:next.mode==='running'?'运行中':next.mode==='stopping'?'正在结束':'已暂停',completed:'回合完成',stopped:'已结束',failed:'运行失败'};
  $('status').textContent=statuses[next.status]||next.status;
  if(next.error)error(next.error);
  const f=next.frame;
  if(!f||f.scene!==selected){
    if(next.status==='failed'){
      $('empty').querySelector('h3').textContent='本次运行未能开始';
      $('empty').querySelector('p').textContent='请根据错误信息检查配置，再重新初始化场景。';
    }else if(next.status==='stopped'){
      $('empty').querySelector('h3').textContent='本次运行已结束';
      $('empty').querySelector('p').textContent='可重新初始化当前场景，或选择其他场景与模型。';
    }
    return;
  }
  $('empty').hidden=true;
  $('metrics').replaceChildren(...f.metrics.map(m=>metricCard(m.label,fmt(m.value,m.unit==='p.u.'?4:2),m.unit)));
  $('clock').textContent=f.clock;
  $('progress').textContent=f.scene==='traffic'?`${f.decision_step} / ${f.decision_limit} 次决策`:`${f.step} / ${f.limit} 步`;
  $('progress-bar').style.width=(100*(f.scene==='traffic'?f.decision_step/f.decision_limit:f.step/f.limit))+'%';
  $('source').textContent=({warehouse:'InventoryEnv · 库存与订单状态',mpe:'MPE · 原生粒子物理状态',traffic:'CityFlow · 原生逐秒车辆坐标',powergrid:'pandapower · 实际潮流解算'})[f.scene];
  $('chart-title').textContent=f.chart.label;
  drawScene(f);drawChart(next.series);
  $('actions').replaceChildren(...f.actions.map(a=>{let r=el('div','action-row');r.append(el('span','',a.agent),el('b','',a.action));return r;}));
  if(!f.actions.length)$('actions').append(el('p','muted','等待第一次模型决策。'));
  $('action-count').textContent=f.actions.length+' 个智能体';
}
async function command(op){
  if(!visibleSession()||!active()||busy)return;
  try{error('');receive(await api('/api/control',{id:state.id,op,speed:Number($('speed').value)}));}
  catch(e){error(e.message);}
}
$('init').onclick=async()=>{
  if(active()||busy)return;
  const request={model:$('model').value,seed:Number($('seed').value)};
  busy=true;hiddenSessionId=state?.id??null;clearView(true);controls();
  try{receive(await api('/api/start',request));}
  catch(e){clearView();error(e.message);}
  finally{busy=false;controls();}
};
function resetSelection(){hiddenSessionId=state?.id??null;details();clearView();controls();}
$('play').onclick=()=>command('run');$('pause').onclick=()=>command('pause');$('step').onclick=()=>command('step');$('stop').onclick=()=>command('stop');$('speed').onchange=()=>command('speed');$('model').onchange=resetSelection;$('seed').onchange=resetSelection;$('refresh-history').onclick=history;
async function history(){try{const rows=await api('/api/history');$('history').replaceChildren(...rows.map(r=>{let n=el('div','history-row');n.append(el('span','',r.model.label+' · '+r.model.algorithm),el('span','',new Date(r.started_at).toLocaleString('zh-CN')),el('span','','seed '+r.seed),el('span','',({completed:'回合完成',stopped:'手动结束',failed:'失败'})[r.status]||r.status));let a=el('a','','下载轨迹 ↓');a.href='/api/export/'+r.id;n.append(a);return n;}));if(!rows.length)$('history').append(el('div','history-row','尚无已结束的运行。'));}catch(e){error(e.message);}}
const canvas=$('scene'),ctx=canvas.getContext('2d'),W=1120,H=640;
function text(x,y,t,size=17,color='#514e49',align='left'){ctx.fillStyle=color;ctx.font=`${size}px "Microsoft YaHei",system-ui`;ctx.textAlign=align;ctx.fillText(t,x,y);ctx.textAlign='left';}
function line(points,color='#c4c0b6',width=2){ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();}
function circle(x,y,r,color,stroke){ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=2;ctx.stroke();}}
function box(x,y,w,h,fill,stroke){ctx.beginPath();ctx.roundRect(x,y,w,h,10);ctx.fillStyle=fill;ctx.fill();if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1;ctx.stroke();}}
function drawScene(f){ctx.clearRect(0,0,W,H);ctx.fillStyle='#faf9f5';ctx.fillRect(0,0,W,H);ctx.strokeStyle='#efede6';ctx.lineWidth=1;for(let x=0;x<W;x+=40){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}for(let y=0;y<H;y+=40){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}if(!f)return;({warehouse:drawWarehouse,mpe:drawMpe,traffic:drawTraffic,powergrid:drawGrid})[f.scene](f.view);}
function drawWarehouse(v){v.agents.forEach((a,i)=>{const x=35+(i%2)*550,y=28+Math.floor(i/2)*300,c=colors[i];box(x,y,510,278,'#ffffff','#e3dfd5');text(x+22,y+34,a.agent_id,20,c);text(x+487,y+33,'容量 '+fmt(a.capacity,0),13,'#7c776d','right');for(let row=0;row<3;row++)for(let col=0;col<10;col++){let filled=(row*10+col)/30<a.stock/a.capacity;box(x+23+col*34,y+61+row*37,25,26,filled?c:'#eeece5');}text(x+400,y+106,fmt(a.stock,0),42,c,'center');text(x+400,y+136,'当前库存 / 件',13,'#7d786d','center');line([[x+22,y+191],[x+487,y+191]],'#e7e3da',1);text(x+22,y+221,'本期需求 '+fmt(a.demand??a.last_demand,0),16);text(x+260,y+221,'本期销售 '+fmt(a.sales||0,0),16);text(x+22,y+254,'下单补货 '+fmt(a.executed_order||0,0),16,c);text(x+260,y+254,'在途 '+fmt(a.in_transit,0),16);});}
function transform(points,pad=65){const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);let scale=Math.min((W-pad*2)/Math.max(.1,xmax-xmin),(H-pad*2)/Math.max(.1,ymax-ymin));return p=>[W/2+(p[0]-(xmin+xmax)/2)*scale,H/2-(p[1]-(ymin+ymax)/2)*scale];}
function drawMpe(v){const bound=Math.max(1.3,...[...v.agents,...v.landmarks].flatMap(a=>a.position.map(Math.abs)))+.15,s=(H-120)/(2*bound),xy=p=>[W/2+p[0]*s,H/2-p[1]*s];text(30,35,'协作覆盖 · 中心距离 < 0.10',17);v.landmarks.forEach((a,i)=>{let [x,y]=xy(a.position),covered=v.nearest_landmark_distances[i]<.1;circle(x,y,.1*s,covered?'#d8eee2':'#ece8df',covered?'#27866d':'#999082');line([[x-8,y],[x+8,y]],'#9c9385',2);line([[x,y-8],[x,y+8]],'#9c9385',2);text(x,y+40,'地标 '+(i+1),14,'#787064','center');});v.agents.forEach((a,i)=>{let [x,y]=xy(a.position),nearest=v.landmarks.reduce((best,l)=>Math.hypot(l.position[0]-a.position[0],l.position[1]-a.position[1])<Math.hypot(best.position[0]-a.position[0],best.position[1]-a.position[1])?l:best);ctx.setLineDash([5,6]);line([[x,y],xy(nearest.position)],colors[i]+'60',1);ctx.setLineDash([]);circle(x,y,Math.max(12,a.radius*s),colors[i]+'22',colors[i]);circle(x,y,5,colors[i]);line([[x,y],[x+a.velocity[0]*s*.15,y-a.velocity[1]*s*.15]],colors[i],3);text(x,y-24,'A'+(i+1),16,colors[i],'center');});v.collision_pairs.forEach(([i,j])=>line([xy(v.agents[i].position),xy(v.agents[j].position)],'#ce6548',4));text(30,H-25,'箭头：当前速度方向    虚线：最近地标    红线：碰撞',14,'#81796d');}
function drawTraffic(v){let points=v.roads.flatMap(r=>r.points.map(p=>[p.x,p.y])),xy=transform(points,55);v.roads.forEach(r=>{const p=r.points.map(p=>xy([p.x,p.y]));line(p,'#d9d5ca',17);ctx.setLineDash([5,7]);line(p,'#ffffff',1);ctx.setLineDash([]);});v.intersections.filter(i=>!i.virtual).forEach((it,i)=>{let [x,y]=xy([it.point.x,it.point.y]);box(x-21,y-21,42,42,'#c9c4b8');let phase=v.phases[it.id];circle(x+26,y-26,5,phase===8?'#ce6548':'#27866d');text(x,y-42,`路口 ${i+1} · 相位 ${phase??'—'}`,13,'#655e54','center');});v.vehicles.forEach(a=>{let[x,y]=xy([a.x,a.y]);ctx.save();ctx.translate(x,y);ctx.rotate(-a.angle);ctx.fillStyle='#287f68';ctx.fillRect(-4,-2,8,4);ctx.restore();});text(24,30,'原生车辆位置 · 每个显示步推进 1 秒',16);text(24,H-20,'信号服务约束保持训练配置；车辆运动由 CityFlow 计算。',13,'#81796d');}
function drawGrid(v){const xy=transform(v.buses.map(b=>[b.x,b.y]),90),positions=Object.fromEntries(v.buses.map(b=>[b.id,xy([b.x,b.y])])),max=Math.max(.01,...v.line_power.map(Math.abs));v.lines.forEach((l,i)=>line([positions[l.source],positions[l.target]],'#b3b7a9',2+3*Math.abs(v.line_power[i])/max));v.buses.forEach((b,i)=>{let[x,y]=positions[b.id],value=v.voltage[i],c=value<.95||value>1.05?'#ce6548':'#27866d';circle(x,y,14,c+'1f',c);text(x,y+5,b.id+1,12,c,'center');text(x,y-23,value.toFixed(3),12,'#7c766c','center');});text(25,32,'33 节点配电网 · 节点颜色表示电压',16);circle(32,H-31,5,'#27866d');text(46,H-26,'0.95–1.05 p.u.',14);circle(222,H-31,5,'#ce6548');text(236,H-26,'电压越限',14);text(W-25,H-26,'线宽：有功功率绝对值',14,'#81796d','right');}
function drawChart(series){let c=$('chart'),g=c.getContext('2d'),w=c.width,h=c.height;g.clearRect(0,0,w,h);if(!series?.length)return;let values=series.map(r=>r.value),lo=Math.min(...values),hi=Math.max(...values),range=Math.max(1e-9,hi-lo);g.strokeStyle='#e8e4db';g.lineWidth=1;for(let y=45;y<h-10;y+=50){g.beginPath();g.moveTo(30,y);g.lineTo(w-25,y);g.stroke();}const p=values.map((v,i)=>[30+i*(w-55)/Math.max(1,values.length-1),h-25-(v-lo)/range*(h-70)]);if(hi===lo)p.forEach(a=>a[1]=h/2);g.beginPath();p.forEach(([x,y],i)=>i?g.lineTo(x,y):g.moveTo(x,y));g.strokeStyle='#27866d';g.lineWidth=3;g.stroke();g.lineTo(p[p.length-1][0],h-20);g.lineTo(30,h-20);g.closePath();const grad=g.createLinearGradient(0,20,0,h);grad.addColorStop(0,'#27866d25');grad.addColorStop(1,'#27866d00');g.fillStyle=grad;g.fill();$('chart-min').textContent='最低 '+fmt(lo);$('chart-max').textContent='最高 '+fmt(hi);}
async function poll(){let priorStatus='';for(;;){try{const s=await api('/api/state?after='+(state?.seq??-1));receive(s);if(s.status!==priorStatus&&['completed','stopped','failed'].includes(s.status))history();priorStatus=s.status;if(['idle','completed','stopped','failed'].includes(s.status))await new Promise(r=>setTimeout(r,1000));}catch(e){connected=false;$('connection').textContent='连接断开 · 正在重试';$('connection').classList.add('offline');controls();await new Promise(r=>setTimeout(r,2500));}}}
async function boot(){try{models=(await api('/api/catalog')).models;for(const scene of Object.keys(labels)){const b=el('button','');b.dataset.scene=scene;b.append(el('span','',icons[scene]),document.createTextNode(labels[scene]),el('em','','LIVE'));b.onclick=()=>choose(scene);$('scenes').append(b);}choose('warehouse');connected=true;controls();history();poll();}catch(e){error('服务连接失败：'+e.message);$('connection').textContent='服务未连接';}}
boot();
