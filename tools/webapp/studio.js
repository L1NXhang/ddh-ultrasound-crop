'use strict';
const $ = id => document.getElementById(id);
let token='', status=null, current=null, image=null, scale=1, mode='crop', active=null, anchor=null;
let windowBox=null, cleanBox=null, predictionWindow=null, busy=false, scope='pending', runMode='review';
let queue=[], lastBatchPrompt='', reviewPrompted=false, pollTimer=null;
let view={scale:1,x:0,y:0}, pan=null, spaceHeld=false;
const canvas=$('canvas'), ctx=canvas.getContext('2d');
const flagText={no_content:'未找到成像区域',area_too_small:'窗口可能偏小',area_too_large:'窗口可能包含界面内容',multiple_blocks:'检测到多个区域',touches_border:'成像内容靠近图片边缘',content_beyond_right:'请检查窗口右边界',content_beyond_left:'请检查窗口左边界',content_beyond_bottom:'请检查窗口下边界',content_beyond_top:'请检查窗口上边界'};
function visible(id,show){$(id).classList.toggle('hidden',!show)}
/* 夜间模式：跟随系统为默认；用户手动切换后记住选择 */
function applyMode(mode){document.documentElement.dataset.mode=mode;}
$('modeToggle').onclick=()=>{const mode=document.documentElement.dataset.mode==='dark'?'light':'dark';try{localStorage.setItem('ddh-mode',mode);}catch(error){}applyMode(mode);};
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',event=>{let stored=null;try{stored=localStorage.getItem('ddh-mode');}catch(error){}if(!stored)applyMode(event.matches?'dark':'light');});
function toast(message){$('toast').textContent=message;visible('toast',true);clearTimeout(toast.timer);toast.timer=setTimeout(()=>visible('toast',false),3500)}
async function api(path,body){
 const payload=reviewMode&&body&&body.id?{...body,task_token:current?.task_token}:body;
 const response=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-DDH-Token':token},body:JSON.stringify(payload)});
 const data=await response.json();if(!response.ok||data.error){if(reviewMode&&response.status===401)reviewNeedsLogin();throw Error(data.error||'请求失败');}return data;
}
function same(a,b){return JSON.stringify(a)===JSON.stringify(b)}
function stale(){return !same(windowBox,predictionWindow)}
function setMode(value){if(preview){preview=false;fit();}mode=value;for(const [id,name] of [['cropMode','crop'],['windowMode','window']]){$(id).classList.toggle('selected',value===name);$(id).setAttribute('aria-pressed',String(value===name));}draw();updateEditButtons();}
function setBusy(value){busy=value;for(const id of ['saveReview','repredict','skip','cropMode','windowMode','backHome'])$(id).disabled=value;updateEditButtons();}
async function chooseFolder(field){
 const button=field==='workPath'?$('chooseWork'):$('chooseSource');button.disabled=true;
 try{const initial=$(field).value||$('workPath').value;const result=typeof desktopRequest==='function'&&window.chrome?.webview?await desktopRequest('pickFolder',{initial}):await api('/api/folder',{initial});if(result.path)$(field).value=result.path;}
 catch(error){$('setupError').textContent=error.message+'，也可以直接输入完整路径。';}
 finally{button.disabled=false;}
}
function renderStatus(data){
 status=data;$('countTotal').textContent=data.total;$('countSaved').textContent=data.saved;$('countReviewed').textContent=data.reviewed;
 $('progressBar').style.width=(data.total?100*data.saved/data.total:0)+'%';
 $('outputPath').textContent=data.output||'尚未选择';$('outputPath').title=data.output||'';
 $('projectName').textContent=data.source?data.source.split(/[\\/]/).filter(Boolean).pop():'未打开文件夹';$('projectPath').textContent=data.source||'';
 $('device').textContent=data.device==='cuda'?'GPU':data.device==='cpu'?'CPU':data.device==='test'?'测试':'就绪';$('device').title=data.model?.device_note||(reviewMode?'预测由主机处理':data.device==='cuda'?'使用 CUDA 推理':'在本机处理');
 if(data.job.state==='running'){visible('batchPanel',true);$('batchFile').textContent=data.job.current||'正在加载模型…';$('batchNumbers').textContent=data.job.processed+' / '+data.job.total;$('batchBar').style.width=(data.job.total?100*data.job.processed/data.job.total:0)+'%';$('batchDetails').textContent='已保存 '+data.job.saved+' · 需人工处理 '+data.job.failed;}
 else{visible('batchPanel',false);const key=String(data.job.started||'');if(['completed','cancelled'].includes(data.job.state)&&key!==lastBatchPrompt){lastBatchPrompt=key;const parts=[data.saved+' 张已保存'];if(data.drafts)parts.push(data.drafts+' 张保留人工草稿');if(data.failed)parts.push(data.failed+' 张需人工处理');if(data.flagged)parts.push(data.flagged+' 张窗口需检查');showFinish(data.job.state==='cancelled'?'已停止':'裁剪完成',parts.join('，')+'。批量结果待审核。');refreshQueue().catch(error=>toast(error.message));}if(data.job.state==='error'&&key!==lastBatchPrompt){lastBatchPrompt=key;toast('批量处理停止：'+data.job.error);refreshQueue().catch(error=>toast(error.message));}}
 $('statAdjust').textContent=data.avg_adjust_px==null?'—':('±'+data.avg_adjust_px+' px');
}
async function poll(){try{renderStatus(await api('/api/status'));}catch(error){if(!busy)toast(reviewMode?'连接中断，请检查网络或联系负责人': '连接中断，请重新启动应用');}finally{if(!reviewMode||reviewAuthenticated)pollTimer=setTimeout(poll,1200);}}
async function refreshQueue(){
 queue=(await api('/api/queue?scope='+scope)).rows;const holder=$('queue');holder.replaceChildren();
 for(const row of queue){const button=document.createElement('button');button.className='queue-row'+(row.reviewed?' reviewed':'')+(row.error?' failed':'')+(row.draft?' draft':'')+(current?.id===row.id?' active':'');button.title=row.file+(row.draft?' · 有未确认草稿':'');const dot=document.createElement('i');dot.className='dot';const name=document.createElement('span');name.className='name';name.textContent=row.file;button.append(dot,name);button.onclick=()=>{if(!busy&&status?.job.state!=='running')loadImage(row.id).catch(error=>toast(error.message));};holder.append(button);}
 if(!queue.length){const empty=document.createElement('div');empty.className='muted empty-list';empty.textContent='当前队列没有图片';holder.append(empty);}
 return queue;
}
async function nextImage(){
 const rows=await refreshQueue();if(rows.length){await loadImage(rows[0].id);return;}
 clearCanvas();if(status?.configured&&!reviewPrompted){reviewPrompted=true;showFinish('审核完成',status.reviewed+' 张已确认保存。'+(status.skipped?'还有 '+status.skipped+' 张稍后处理。':''));}
}
async function loadImage(id){
 setBusy(true);if(!await flushDraft()){setBusy(false);return;}
 clearCanvas();current={id,file:queue.find(row=>row.id===id)?.file||''};setBusy(true);visible('loading',true);visible('emptyStage',false);
 try{current=await api('/api/predict?id='+encodeURIComponent(id));windowBox=current.current_window? [...current.current_window]:null;cleanBox=current.current_clean_box?[...current.current_clean_box]:null;predictionWindow=current.prediction_window?[...current.prediction_window]:null;
 const next=new Image();await new Promise((resolve,reject)=>{next.onload=resolve;next.onerror=()=>reject(Error('无法解码这张图片，可以稍后处理'));next.src='/api/image?id='+encodeURIComponent(id);});image=next;initEdits();setMode('crop');fit();updateInfo();await refreshQueue();}
 catch(error){toast(error.message);$('notice').textContent=error.message;}
 finally{setBusy(false);visible('loading',false);}
}
function clearCanvas(){preview=false;clearTimeout(draftTimer);draftLabel('');image=null;current=null;windowBox=null;cleanBox=null;ctx.clearRect(0,0,canvas.width,canvas.height);visible('emptyStage',true);canvas.classList.remove('previewing','panning');$('zoomBadge').classList.add('hidden');$('imageName').textContent='选择图片';$('imageMeta').textContent='';$('cropSize').textContent='—';$('coordinates').textContent='原图像素坐标';$('notice').textContent='';}
function fit(){if(!image)return;const stage=$('stage');$('zoomBadge').classList.toggle('hidden',!preview);
 if(preview&&cleanBox){const width=cleanBox[2]-cleanBox[0],height=cleanBox[3]-cleanBox[1];
  if(compare){const gap=12,half=(stage.clientWidth-gap)/2;
   view.scale=Math.max(.1,Math.min((half-28)/image.width,(stage.clientHeight-30)/image.height,1));
   view.x=(half-image.width*view.scale)/2;view.y=(stage.clientHeight-image.height*view.scale)/2;}
  else{view.scale=Math.max(.1,Math.min((stage.clientWidth-36)/width,(stage.clientHeight-30)/height,1));
   view.x=(stage.clientWidth-width*view.scale)/2;view.y=(stage.clientHeight-height*view.scale)/2;}
  applyView();draw();return;}
 const width=image.width,height=image.height;scale=Math.max(.01,Math.min((stage.clientWidth-36)/width,(stage.clientHeight-30)/height));canvas.width=Math.round(width*scale);canvas.height=Math.round(height*scale);canvas.style.left=(stage.clientWidth-canvas.width)/2+'px';canvas.style.top=(stage.clientHeight-canvas.height)/2+'px';draw();}
function applyView(){const stage=$('stage');
 if(compare){canvas.width=stage.clientWidth;canvas.height=stage.clientHeight;canvas.style.left='0px';canvas.style.top='0px';}
 else{const width=cleanBox[2]-cleanBox[0],height=cleanBox[3]-cleanBox[1];
  canvas.width=Math.max(1,Math.round(width*view.scale));canvas.height=Math.max(1,Math.round(height*view.scale));
  canvas.style.left=view.x+'px';canvas.style.top=view.y+'px';}
 canvas.classList.toggle('previewing',preview);
 $('zoomBadge').textContent=Math.round(view.scale*100)+'%';}
function zoomTo(next,cx,cy){const target=Math.max(.1,Math.min(4,next));if(target===view.scale)return;
 // 锚点缩放：光标下的内容点缩放后仍在光标下。对比模式按光标所在窗格取内容原点。
 let originX=view.x,originY=view.y;
 if(compare){const gap=12,half=($('stage').clientWidth-gap)/2;
  if(cx>half){originX=view.x+half+gap;originY=view.y+cleanBox[1]*view.scale;}}
 const ix=(cx-originX)/view.scale,iy=(cy-originY)/view.scale;
 if(compare){const gap=12,half=($('stage').clientWidth-gap)/2;
  if(cx>half){view.x=cx-ix*target-half-gap;view.y=cy-iy*target-cleanBox[1]*target;}
  else{view.x=cx-ix*target;view.y=cy-iy*target;}}
 else{view.x=cx-ix*target;view.y=cy-iy*target;}
 applyView();draw();}
function zoomBy(factor){const stage=$('stage');zoomTo(view.scale*factor,stage.clientWidth/(compare?4:2),stage.clientHeight/2);}
function box(rect,color,dashed){if(!rect)return;ctx.strokeStyle=color;ctx.lineWidth=2;ctx.setLineDash(dashed?[6,4]:[]);ctx.strokeRect(rect[0]*scale,rect[1]*scale,(rect[2]-rect[0])*scale,(rect[3]-rect[1])*scale);ctx.setLineDash([]);}
function draw(){
 ctx.clearRect(0,0,canvas.width,canvas.height);if(!image)return;
 if(preview&&compare&&cleanBox){ // 并排对比：左=原图，右=成品（画在它原来所在的解剖位置）
  const gap=12,half=(($('stage').clientWidth-gap)/2),cropW=cleanBox[2]-cleanBox[0],cropH=cleanBox[3]-cleanBox[1];
  ctx.drawImage(image,0,0,image.width,image.height,view.x,view.y,image.width*view.scale,image.height*view.scale);
  const rightX=half+gap,cropX=rightX+view.x,cropY=view.y+cleanBox[1]*view.scale;
  ctx.drawImage(image,cleanBox[0],cleanBox[1],cropW,cropH,cropX,cropY,cropW*view.scale,cropH*view.scale);
  ctx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue('--line')||'#888';ctx.lineWidth=1;
  ctx.strokeRect(.5,0.5,half-1,canvas.height-1);ctx.strokeRect(rightX+.5,.5,half-1,canvas.height-1);
  return;}
 if(preview&&cleanBox){const [l,t,r,b]=cleanBox;ctx.drawImage(image,l,t,r-l,b-t,0,0,canvas.width,canvas.height);return;}
 ctx.drawImage(image,0,0,canvas.width,canvas.height);box(windowBox,'#d88a73',true);
 if(cleanBox){const [l,t,r,b]=cleanBox;ctx.fillStyle='#00000080';ctx.fillRect(0,0,canvas.width,t*scale);ctx.fillRect(0,b*scale,canvas.width,canvas.height-b*scale);ctx.fillRect(0,t*scale,l*scale,(b-t)*scale);ctx.fillRect(r*scale,t*scale,canvas.width-r*scale,(b-t)*scale);box(cleanBox,'#7cbd9b',false);ctx.fillStyle='#7cbd9b';for(const [x,y] of [[l,t],[r,t],[l,b],[r,b],[(l+r)/2,t],[(l+r)/2,b],[l,(t+b)/2],[r,(t+b)/2]])ctx.fillRect(x*scale-3,y*scale-3,6,6);}
 if(mode==='window')box(windowBox,'#edb29f',false);
}
function updateInfo(){
 if(!current)return;$('imageName').textContent=current.file;$('imageMeta').textContent=current.original_size.join(' × ')+' px · '+(current.reviewed?'已审核':'待审核');
 $('coordinates').textContent='窗口 '+JSON.stringify(windowBox)+' · 裁剪 '+JSON.stringify(cleanBox);$('cropSize').textContent=cleanBox?(cleanBox[2]-cleanBox[0])+' × '+(cleanBox[3]-cleanBox[1])+' px':'—';
 const flags=(current.flags||[]).map(flag=>flagText[flag]||'请检查窗口边界');$('notice').title=flags.join('；');$('notice').textContent=preview?(compare?'对比模式 · 滚轮缩放 · 中键或空格+左键拖动 · 双击复位 · P 返回':'成品预览 · 滚轮缩放 · 中键或空格+左键拖动 · 双击复位 · P 返回'):stale()?'窗口已修改，请重新预测':!windowBox?'未找到窗口：按 W 手动框选':flags.length?flags.slice(0,2).join('；'):'拖动框的四边可微调';
}
function point(event){const rect=canvas.getBoundingClientRect();return [Math.round(Math.max(0,Math.min(image.width,(event.clientX-rect.left)/scale))),Math.round(Math.max(0,Math.min(image.height,(event.clientY-rect.top)/scale)))];}
canvas.addEventListener('pointerdown',event=>{
 if(!editable()||preview)return;const p=point(event),target=mode==='window'?windowBox:cleanBox;active=null;anchor=null;
 if(target){const near=(a,b)=>Math.abs(a-b)*scale<11;const insideX=p[0]>=target[0]-12/scale&&p[0]<=target[2]+12/scale;const insideY=p[1]>=target[1]-12/scale&&p[1]<=target[3]+12/scale;if(insideY&&near(p[0],target[0]))active=0;else if(insideY&&near(p[0],target[2]))active=2;else if(insideX&&near(p[1],target[1]))active=1;else if(insideX&&near(p[1],target[3]))active=3;}
 if(active===null&&mode==='window'){anchor=p;active='new';}canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener('pointermove',event=>{
 if(active===null||busy||!image)return;const p=point(event);
 if(active==='new'){windowBox=[Math.min(anchor[0],p[0]),Math.min(anchor[1],p[1]),Math.max(anchor[0],p[0]),Math.max(anchor[1],p[1])];}
 else{const target=[...(mode==='window'?windowBox:cleanBox)],limit=mode==='window'?[0,0,image.width,image.height]:(windowBox||[0,0,image.width,image.height]);const axis=active%2,value=p[axis];if(active<2)target[active]=Math.max(limit[active],Math.min(value,target[active+2]-2));else target[active]=Math.min(limit[active],Math.max(value,target[active-2]+2));if(mode==='window')windowBox=target;else cleanBox=target;}
 draw();updateInfo();
});
function finishDrag(){if(active!==null)changedGeometry();active=null;anchor=null;}
canvas.addEventListener('pointerup',finishDrag);
canvas.addEventListener('pointercancel',finishDrag);
/* 成品预览的查看交互：滚轮缩放（光标为锚点）、中键或空格+左键拖动、双击复位 */
$('stage').addEventListener('wheel',event=>{
 if(!preview||!image)return;event.preventDefault();
 const rect=$('stage').getBoundingClientRect();
 zoomTo(view.scale*(event.deltaY<0?1.15:1/1.15),event.clientX-rect.left,event.clientY-rect.top);
},{passive:false});
$('stage').addEventListener('pointerdown',event=>{
 if(!preview||!image||(event.button!==1&&!(event.button===0&&spaceHeld)))return;
 event.preventDefault();pan={x:event.clientX-view.x,y:event.clientY-view.y};
 canvas.classList.add('panning');try{$('stage').setPointerCapture(event.pointerId);}catch(error){}
});
$('stage').addEventListener('pointermove',event=>{if(!pan)return;view.x=event.clientX-pan.x;view.y=event.clientY-pan.y;applyView();});
function endPan(){if(pan===null)return;pan=null;canvas.classList.remove('panning');}
$('stage').addEventListener('pointerup',endPan);
$('stage').addEventListener('pointercancel',endPan);
$('stage').addEventListener('dblclick',()=>{if(preview&&image)fit();});
document.addEventListener('keydown',event=>{if(!preview||/INPUT|SELECT|TEXTAREA/.test(event.target.tagName)||event.target.isContentEditable||/BUTTON|SUMMARY|A/.test(event.target.tagName))return;if(event.key===' '){event.preventDefault();spaceHeld=true;}});
document.addEventListener('keyup',event=>{if(event.key===' ')spaceHeld=false;});
async function repredict(){if(!current||!windowBox||busy)return;setBusy(true);visible('loading',true);try{const result=await api('/api/repredict',{id:current.id,window:windowBox});predictionWindow=[...result.window];cleanBox=[...result.crop_box];current.latest_prediction=result;persistedGeometry='';changedGeometry();setMode('crop');fit();updateInfo();toast('已按当前窗口重新预测，可继续调整四边');}catch(error){toast(error.message);}finally{setBusy(false);visible('loading',false);}}
async function saveReview(){
 if(!current||busy)return;if(stale()){toast('请先按 R 重新预测');return;}if(!cleanBox){toast('请先定位窗口并重新预测');return;}
 setBusy(true);try{if(!await flushDraft())return;await api('/api/review',{id:current.id,window:windowBox,clean_box:cleanBox});markConfirmed();renderStatus(await api('/api/status'));toast('已保存');scope='pending';$('scope').value=scope;await nextImage();}catch(error){toast(error.message);}finally{setBusy(false);}
}
async function skip(){if(!current||busy)return;setBusy(true);try{if(!await flushDraft())return;await api('/api/skip',{id:current.id});scope='pending';$('scope').value=scope;renderStatus(await api('/api/status'));await nextImage();}catch(error){toast(error.message);}finally{setBusy(false);}}
function showFinish(title,text){$('finishTitle').textContent=title;$('finishText').textContent=text;visible('zipDownload',false);$('zipPath').textContent='';if(reviewMode)$('finishLater').textContent='返回工作台';visible('finish',true);updateEditButtons();}
async function makeArchive(){
 $('finishZip').disabled=true;try{const archive=await api('/api/archive',{});$('zipPath').textContent='已保存：'+archive.path;$('zipDownload').href='/api/archive/download?id='+archive.id;$('zipDownload').download=archive.name;$('zipDownload').onclick=typeof desktopRequest==='function'&&window.chrome?.webview?event=>{event.preventDefault();desktopRequest('saveArchive',{archive_id:archive.id}).then(result=>{if(result.path)toast('副本已保存：'+result.path);}).catch(error=>toast(error.message));}:null;$('zipDownload').textContent=window.chrome?.webview?'另存压缩包副本':'下载压缩包';visible('zipDownload',true);toast('ZIP 已保存到工作目录，可另外保存副本');}catch(error){toast(error.message);}finally{$('finishZip').disabled=false;}
}
async function startProject(){
 const source=$('sourcePath').value.trim(),work=$('workPath').value.trim();if(!source||!work){$('setupError').textContent='请选择图片文件夹和工作目录';return;}
 $('startProject').disabled=true;$('setupError').textContent='';
 try{if(!await flushDraft())return;renderStatus(await api('/api/project',{source,work}));reviewPrompted=false;visible('setup',false);refreshRecents();$('modeBadge').textContent=runMode==='auto'?'批量裁剪':'逐张审核';clearCanvas();
 if(runMode==='auto'){await api('/api/batch/start',{});renderStatus(await api('/api/status'));}else{scope='pending';$('scope').value=scope;await nextImage();}}
 catch(error){$('setupError').textContent=error.message;visible('setup',true);}finally{$('startProject').disabled=false;}
}
document.querySelectorAll('[name=runMode]').forEach(input=>input.onchange=()=>{runMode=input.value;document.querySelectorAll('.mode-option').forEach(label=>label.classList.toggle('active',label.querySelector('input').checked));$('startProject').textContent=runMode==='auto'?'开始裁剪 →':'开始审核 →';});
$('chooseWork').onclick=()=>chooseFolder('workPath');$('chooseSource').onclick=()=>chooseFolder('sourcePath');$('startProject').onclick=startProject;
async function openSetup(){if(busy)return;if(status?.job.state==='running'){toast('请先完成或停止当前批量任务');return;}setBusy(true);try{if(!await flushDraft())return;visible('finish',false);visible('setup',true);visible('cancelSetup',Boolean(status?.configured));await refreshRecents();}finally{setBusy(false);}}
$('newProject').onclick=openSetup;$('backHome').onclick=openSetup;
$('cropMode').onclick=()=>setMode('crop');$('windowMode').onclick=()=>setMode('window');$('repredict').onclick=repredict;$('saveReview').onclick=saveReview;$('skip').onclick=skip;
$('scope').onchange=()=>{scope=$('scope').value;refreshQueue().catch(error=>toast(error.message));};
$('finishLater').onclick=()=>{visible('finish',false);refreshQueue().catch(error=>toast(error.message));};$('finishZip').onclick=makeArchive;
$('archiveSide').onclick=()=>{if(!status?.saved&&!status?.failed){toast('先保存一张结果，再打包');return;}showFinish('保存压缩包',status.saved+' 张已保存，其中 '+status.reviewed+' 张已审核。');};
$('openOutput').onclick=()=>api('/api/output/open',{}).then(()=>toast('已在新窗口打开结果文件夹')).catch(error=>toast(error.message));$('stopBatch').onclick=()=>api('/api/batch/stop',{}).then(()=>toast('将在当前图片处理后停止')).catch(error=>toast(error.message));
/* ---------- 模型训练 ---------- */
let trainTimer=null;
function setTrainView(running){
 visible('trainForm',!running);visible('trainRunning',running);visible('trainStop',running);
 $('trainStart').textContent=running?'训练中…':'开始训练';$('trainStart').disabled=running;
}
async function pollTrain(){
 try{const job=await api('/api/train/status');
  setTrainView(job.running);
  if(job.running||job.state!=='idle'){$('trainLog').textContent=(job.lines||[]).join('\n');const box=$('trainLog');box.scrollTop=box.scrollHeight;}
  if(job.state==='completed'&&!pollTrain.reported){pollTrain.reported=true;setTrainView(false);$('trainState').textContent='训练完成';toast('训练完成，模型已保存到 '+(job.result?.output||'输出位置'));}
  if(job.state==='failed'&&!pollTrain.reported){pollTrain.reported=true;setTrainView(false);$('trainState').textContent='训练失败';$('trainError').textContent=job.error||'训练失败，请查看日志';}
 }catch(error){/* 关闭窗口后轮询失败则停止 */}
 finally{if(!$('trainPanel').classList.contains('hidden'))trainTimer=setTimeout(pollTrain,1500);}
}
function choosePath(field,mode){
 const button=$(field==='trainManifest'?'chooseTrainManifest':field==='trainRoot'?'chooseTrainRoot':'chooseTrainOutput');
 button.disabled=true;
 (async()=>{try{
   const isDesktop=typeof desktopRequest==='function'&&window.chrome?.webview;
   const result=isDesktop?await desktopRequest(mode==='file'?'pickFile':'pickFolder',{initial:$(field).value||''})
                         :await api(mode==='file'?'/api/pick-file':'/api/folder',{initial:$(field).value||''});
   if(result.path)$(field).value=result.path;
  }catch(error){$('trainError').textContent=error.message;}
  finally{button.disabled=false;}})();
}
$('trainOpen').onclick=async()=>{if(reviewMode){toast('协作审核模式下不开放训练');return;}
 visible('trainPanel',true);$('trainError').textContent='';pollTrain.reported=false;pollTrain();};
$('trainClose').onclick=()=>{visible('trainPanel',false);clearTimeout(trainTimer);};
$('trainStop').onclick=()=>api('/api/train/stop',{}).then(r=>toast(r.note||'正在停止训练')).catch(error=>toast(error.message));
$('trainStart').onclick=async()=>{
 $('trainError').textContent='';
 const spec={manifest:$('trainManifest').value.trim(),root:$('trainRoot').value.trim(),
             output:$('trainOutput').value.trim(),epochs:parseInt($('trainEpochs').value||'30',10)};
 if(!spec.manifest||!spec.root||!spec.output){$('trainError').textContent='请填写数据清单、图片根目录与输出位置';return;}
 try{await api('/api/train/start',spec);pollTrain.reported=false;setTrainView(true);toast('训练已开始');}
 catch(error){$('trainError').textContent=error.message;}};
$('chooseTrainManifest').onclick=()=>choosePath('trainManifest','file');
$('chooseTrainRoot').onclick=()=>choosePath('trainRoot','folder');
$('chooseTrainOutput').onclick=()=>choosePath('trainOutput','folder');
/* ---------- 推理组件（引导式安装）---------- */
let componentsTimer=null, componentsHinted=false;
async function pollComponents(){
 try{
  const status=await api('/api/components/status');
  $('componentsOpen').classList.toggle('hidden',!status.lock_available&&status.torch_ready);
  if(!status.torch_ready&&!componentsHinted){componentsHinted=true;openComponentsPanel(status);}
  if($('componentsPanel').classList.contains('hidden'))return;
  $('componentsBar').style.width=(status.progress&&status.progress.total?100*status.progress.done/status.progress.total:0)+'%';
  $('componentsFile').textContent=status.progress?`${status.progress.stage} · ${status.progress.file} · ${(status.progress.done/1e6).toFixed(0)}/${status.progress.total?(status.progress.total/1e6).toFixed(0):'?'} MB`:'';
  $('componentsLog').textContent=(status.lines||[]).join('\n');
  setComponentsView(status.running);
  if(!status.running&&pollComponents.lastRunning){pollComponents.lastRunning=false;
   if(status.state==='completed'){toast('组件安装完成，可以开始裁剪了');visible('componentsPanel',false);}
   else if(status.state==='failed'){$('componentsError').textContent=status.error||'安装失败，请查看日志';setComponentsView(false);}
  }
  pollComponents.lastRunning=status.running;
 }catch(error){/* 服务未就绪时忽略 */}
 finally{componentsTimer=setTimeout(pollComponents,1500);}
}
function setComponentsView(running){
 visible('componentsForm',!running);visible('componentsRunning',running);visible('componentsStop',running);
 $('componentsInstall').textContent=running?'安装中…':'安装组件';$('componentsInstall').disabled=running;
}
function openComponentsPanel(status){
 if(reviewMode){toast('协作审核模式请在主机上安装组件');return;}
 visible('componentsPanel',true);$('componentsError').textContent='';
 if(status){
  $('cpuSize').textContent='· 约 '+(status.variants?.cpu?.missing_bytes/1e6).toFixed(0)+' MB';
  $('cudaSize').textContent='· 约 '+(status.variants?.cuda?.missing_bytes/1e6).toFixed(0)+' MB';
  const suggest=status.suggested==='cuda'?'cuda':'cpu';
  const radio=document.querySelector(`[name=componentVariant][value=${suggest}]`);if(radio)radio.checked=true;
  if(!status.lock_available)$('componentsError').textContent=status.lock_error||'组件清单缺失：引导版安装包不完整，请重新下载';
 }
 pollComponents.lastRunning=false;setComponentsView(false);
}
$('componentsOpen').onclick=async()=>{openComponentsPanel(await api('/api/components/status'));};
$('componentsClose').onclick=()=>{visible('componentsPanel',false);};
$('componentsStop').onclick=()=>api('/api/components/stop',{}).catch(error=>toast(error.message));
$('componentsInstall').onclick=async()=>{
 $('componentsError').textContent='';
 const variant=document.querySelector('[name=componentVariant]:checked')?.value;
 if(!variant){$('componentsError').textContent='请先选择 CPU 或 CUDA 组件';return;}
 try{await api('/api/components/install',{variant});setComponentsView(true);toast('开始下载组件');}
 catch(error){$('componentsError').textContent=error.message;}};
document.addEventListener('keydown',event=>{
 if(/INPUT|SELECT|TEXTAREA/.test(event.target.tagName)||event.target.isContentEditable||!editable())return;
 const key=event.key.toLowerCase();
 if(event.ctrlKey||event.metaKey){
  if(key==='z'){event.preventDefault();undoEdit(event.shiftKey?1:-1);}
  else if(key==='y'){event.preventDefault();undoEdit(1);}
  else if(key==='s'){event.preventDefault();saveReview();}
  return;
 }
 if(event.altKey||/BUTTON|SUMMARY|A/.test(event.target.tagName))return;
 if(event.key==='Enter'){event.preventDefault();saveReview();}
 else if(key==='r')repredict();
 else if(key==='w')setMode('window');
 else if(key==='c')setMode('crop');
 else if(key==='k')skip();
 else if(key==='p')togglePreview();
 else if(preview&&(key==='='||key==='+')){event.preventDefault();zoomBy(1.2);}
 else if(preview&&key==='-'){event.preventDefault();zoomBy(1/1.2);}
 else if(event.key==='Escape'&&preview)togglePreview();
 else if(!preview&&['ArrowUp','ArrowDown'].includes(event.key)&&cleanBox){
  event.preventDefault();const delta=(event.key==='ArrowUp'?-1:1)*(event.shiftKey?10:1);
  cleanBox=[cleanBox[0],Math.max(windowBox[1],Math.min(cleanBox[3]-2,cleanBox[1]+delta)),cleanBox[2],cleanBox[3]];
  changedGeometry();draw();updateInfo();
 }
});
window.addEventListener('resize',fit);
async function boot(){if(reviewMode){await reviewBoot();return;}if(location.protocol==='file:'){$('setupError').textContent='请先双击工作区中的“启动裁剪工具.bat”，再打开 http://127.0.0.1:8770/';$('startProject').disabled=true;return;}try{const data=await api('/api/bootstrap');token=data.token;renderRecents(data.recent_projects||[]);$('workPath').value=data.status.work||data.default_work;const suggested=new URL(location.href).searchParams.get('source');$('sourcePath').value=suggested||data.status.source||'';renderStatus(data.status);if(data.status.configured&&!suggested){visible('setup',false);if(data.status.job.state==='running')await refreshQueue();else await nextImage();}pollComponents();pollTimer=setTimeout(poll,1200);}catch(error){$('setupError').textContent='连接失败：'+error.message+'。请重新启动裁剪工具。';}}
setupSessionControls();
setupCollaboration();
boot();
