'use strict';
const $=id=>document.getElementById(id), cameras=['front','top_camera','gripper'];
let episodes=[], episode=null, index=0, playing=false, playbackTime=0, lastTick=0, selection=0;
let yaw=0, pitch=0, zoom=1;
async function api(url){const r=await fetch(url);const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);return d;}
function stop(){playing=false;$('play').textContent='▶ 재생';cameras.forEach(c=>$(c).pause());}
function notice(text){$('notice').hidden=false;$('notice').textContent=text;}
function list(){const query=$('search').value.toLowerCase();$('episodes').replaceChildren();for(const item of episodes){if(!`${item.id} ${item.instruction||''}`.toLowerCase().includes(query))continue;const button=document.createElement('button');button.className='episode'+(episode?.id===item.id?' active':'');const title=document.createElement('strong');title.textContent=`Episode ${item.id}`;const desc=document.createElement('p');desc.textContent=item.error||item.instruction||'Instruction 없음';const stat=document.createElement('small');stat.textContent=item.error?'읽기 오류':`${item.samples.toLocaleString()} frames · ${item.duration.toFixed(1)} s`;button.append(title,desc,stat);button.onclick=()=>select(item.id);$('episodes').append(button);}}
async function refresh(){try{episodes=await api('/api/episodes');$('count').textContent=episodes.length;list();if(!episodes.length){stop();episode=null;$('detail').hidden=true;notice('아직 저장된 에피소드가 없습니다. collect_data.py에서 B로 녹화한 뒤 저장하고 목록을 새로고침하세요.');}else if(!episode)await select(episodes[0].id);}catch(e){notice(e.message);}}
async function select(id){const token=++selection;stop();episode=null;notice('에피소드와 EEF 궤적을 불러오는 중…');$('detail').hidden=true;try{const data=await api('/api/episodes/'+encodeURIComponent(id));if(token!==selection)return;episode=data;index=0;playbackTime=0;$('title').textContent=`Episode ${id}`;$('instruction').textContent=data.metadata.instruction||'Instruction이 저장되지 않았습니다.';$('stats').textContent=`${data.state.length.toLocaleString()} frames   /   ${(1/data.dt).toFixed(1)} Hz   /   ${(data.state.length*data.dt).toFixed(2)} s   /   ${Object.keys(data.videos).length} cameras`;
for(const c of cameras){const video=$(c), file=data.videos[c];video.pause();video.removeAttribute('src');video.parentElement.querySelector('.missing').hidden=!!file;if(file){video.src='/media/'+encodeURIComponent(id)+'/'+encodeURIComponent(file);video.onloadedmetadata=()=>{if(episode?.id===id)video.currentTime=playbackTime;};video.onerror=()=>{video.parentElement.querySelector('.missing').textContent='영상 로드 실패';video.parentElement.querySelector('.missing').hidden=false;};}video.load();}
$('timeline').max=data.state.length-1;$('notice').hidden=true;$('detail').hidden=false;fit();list();update(true);
}catch(e){if(token===selection)notice(e.message);}}
function fit(){yaw=0;pitch=0;zoom=1;draw();}
function update(seek=false){if(!episode)return;index=Math.max(0,Math.min(episode.state.length-1,index));$('timeline').value=index;$('frame').textContent=`${index+1} / ${episode.state.length}`;$('time').textContent=`${episode.time[index].toFixed(2)} s`;
for(const kind of ['state','action']){const row=$(kind+'Row');while(row.children.length>1)row.lastChild.remove();for(const value of episode[kind][index]){const td=document.createElement('td');td.textContent=value.toFixed(4);row.append(td);}$(kind+'XYZ').textContent=episode.trajectories[kind][index].map(v=>v.toFixed(4)).join(' / ');}
if(seek){playbackTime=episode.time[index];for(const c of cameras){const v=$(c);if(v.getAttribute('src'))v.currentTime=playbackTime;}}draw();}
function seek(i){stop();index=i;update(true);}
$('play').onclick=()=>{if(!episode)return;if(playing){stop();return;}if(index>=episode.state.length-1)seek(0);playing=true;lastTick=performance.now();$('play').textContent='Ⅱ 일시정지';for(const c of cameras){const v=$(c);if(v.getAttribute('src')){v.playbackRate=Number($('speed').value);v.play().catch(()=>{});}}};
$('timeline').oninput=e=>seek(Number(e.target.value));$('prev').onclick=()=>seek(index-1);$('next').onclick=()=>seek(index+1);$('speed').onchange=()=>cameras.forEach(c=>$(c).playbackRate=Number($('speed').value));$('refresh').onclick=refresh;$('search').oninput=list;$('resetView').onclick=()=>episode&&fit();$('showState').onchange=draw;$('showAction').onchange=draw;
function tick(now){if(playing&&episode){playbackTime+=(now-lastTick)/1000*Number($('speed').value);const duration=episode.time.at(-1)+episode.dt;if(playbackTime>=duration){index=episode.state.length-1;stop();update(true);}else{while(index<episode.time.length-1&&episode.time[index+1]<=playbackTime)index++;update();for(const c of cameras){const v=$(c);if(v.readyState>=2&&Math.abs(v.currentTime-playbackTime)>.18)v.currentTime=playbackTime;}}}lastTick=now;requestAnimationFrame(tick);}requestAnimationFrame(tick);
const canvas=$('plot');
let requested=null,busy=false,lastRender='',imageURL=null;
function draw(){
 if(!episode)return;
 const mode=$('showState').checked?'state':'action';
 const query=new URLSearchParams({episode:episode.id,frame:index,mode,yaw:yaw.toFixed(2),pitch:pitch.toFixed(2),zoom:zoom.toFixed(3)}).toString();
 if(query===lastRender)return;
 requested={query,id:episode.id,full:episode.full_replay};
 renderNext();
}
async function renderNext(){
 if(busy||!requested)return;
 busy=true;const job=requested;requested=null;lastRender=job.query;
 try{
  const response=await fetch('/api/render?'+job.query);
  if(!response.ok){const error=await response.json();throw Error(error.error);}
  const blob=await response.blob();
  if(episode?.id===job.id){
   const next=URL.createObjectURL(blob);canvas.src=next;if(imageURL)URL.revokeObjectURL(imageURL);imageURL=next;
   $('replayStatus').textContent=job.full?'로봇 모션 재생 · 작업 물체 숨김':'로봇 모션 재생 · 이전 데이터의 그리퍼 링크 자세는 근사';
  }
 }catch(e){$('replayStatus').textContent=e.message;}
 finally{busy=false;if(requested&&requested.query!==lastRender)renderNext();}
}
let drag=null;
canvas.onpointerdown=e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};
canvas.onpointermove=e=>{if(!drag)return;yaw-=(e.clientX-drag[0])*.4;pitch=Math.max(-70,Math.min(70,pitch+(e.clientY-drag[1])*.4));drag=[e.clientX,e.clientY];draw();};
canvas.onpointerup=canvas.onpointercancel=()=>drag=null;
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.3,Math.min(5,zoom*Math.exp(-e.deltaY*.001)));draw();},{passive:false});
refresh();
