'use strict';
const EDIT_HISTORY_LIMIT = 40;
const DRAFT_DELAY_MS = 350;
let editHistory = [], editIndex = -1, draftTimer = null, draftSaving = null;
let persistedGeometry = '', preview = false, draftHasPersisted = false;

function geometry() {
 return {window: windowBox ? [...windowBox] : null, clean_box: cleanBox ? [...cleanBox] : null};
}
function editable() {
 return current && image && !busy && ['setup','finish','batchPanel','reviewLogin','collaborationPanel'].every(id=>$(id).classList.contains('hidden'));
}
function draftLabel(text, failed=false) {
 $('draftState').textContent=text;
 $('draftState').classList.toggle('failed',failed);
}
function updateEditButtons() {
 const enabled=Boolean(editable());
 $('undoEdit').disabled=!enabled || editIndex<1;
 $('redoEdit').disabled=!enabled || editIndex>=editHistory.length-1;
 $('resetPrediction').disabled=!enabled || !current?.latest_prediction?.crop_box;
 $('previewCrop').disabled=!enabled || !cleanBox || stale();
 $('previewCrop').setAttribute('aria-pressed',String(preview));
 $('previewCrop').textContent=preview?'返回原图':'成品预览';
}
function initEdits() {
 clearTimeout(draftTimer);
 editHistory=[geometry()];editIndex=0;
 persistedGeometry=JSON.stringify(geometry());
 draftHasPersisted=Boolean(current.draft_restored);
 preview=false;
 draftLabel(current.draft_restored?'已恢复草稿':current.reviewed?'已确认':'');
 updateEditButtons();
}
function changedGeometry() {
 if(!current || !image)return;
 const value=geometry();
 if(!same(editHistory[editIndex],value)) {
  editHistory=[...editHistory.slice(0,editIndex+1),value].slice(-EDIT_HISTORY_LIMIT);
  editIndex=editHistory.length-1;
 }
 scheduleDraft();
 updateEditButtons();
}
function scheduleDraft() {
 clearTimeout(draftTimer);
 if(JSON.stringify(geometry())===persistedGeometry){draftLabel(draftHasPersisted?'草稿已保存':current.reviewed?'已确认':'');return;}
 draftLabel('正在保存草稿…');
 draftTimer=setTimeout(flushDraft,DRAFT_DELAY_MS);
}
async function flushDraft() {
 clearTimeout(draftTimer);
 if(draftSaving)await draftSaving;
 if(!current || !image || JSON.stringify(geometry())===persistedGeometry)return true;
 const id=current.id, snapshot=geometry();
 const valid=box=>box===null || box[0]<box[2] && box[1]<box[3];
 if(!valid(snapshot.window) || !valid(snapshot.clean_box)) {
  draftLabel('框选尚未完成',true);return false;
 }
 draftSaving=api('/api/draft',{id,...snapshot}).then(()=>{
  if(current?.id===id) {
   persistedGeometry=JSON.stringify(snapshot);
   draftHasPersisted=true;
   draftLabel(same(geometry(),snapshot)?'草稿已保存':'正在保存草稿…');
  }
  return true;
 }).catch(error=>{
  draftLabel('草稿保存失败',true);toast('草稿未能保存：'+error.message);return false;
 }).finally(()=>{draftSaving=null;});
 return draftSaving;
}
function markConfirmed() {
 clearTimeout(draftTimer);
 persistedGeometry=JSON.stringify(geometry());
 draftHasPersisted=false;
 draftLabel('已确认');
}
function restoreGeometry(value) {
 windowBox=value.window?[...value.window]:null;
 cleanBox=value.clean_box?[...value.clean_box]:null;
 if(preview && (!cleanBox || stale()))preview=false;
 fit();updateInfo();scheduleDraft();updateEditButtons();
}
function undoEdit(direction) {
 if(!editable())return;
 const next=editIndex+direction;
 if(next<0 || next>=editHistory.length)return;
 editIndex=next;restoreGeometry(editHistory[editIndex]);
}
function resetPrediction() {
 if(!editable() || !current.latest_prediction?.crop_box)return;
 const prediction=current.latest_prediction;
 windowBox=[...prediction.window];cleanBox=[...prediction.crop_box];
 preview=false;changedGeometry();fit();updateInfo();
}
function togglePreview() {
 if(!editable() || !cleanBox || stale())return;
 preview=!preview;fit();updateInfo();updateEditButtons();
}
function renderRecents(rows) {
 visible('recentTasks',rows.length>0);
 $('recentList').replaceChildren();
 for(const entry of rows) {
  const button=document.createElement('button');button.className='recent-row';
  const detail=document.createElement('span'),name=document.createElement('b'),path=document.createElement('small');
  name.textContent=entry.source.split(/[\\/]/).filter(Boolean).pop();
  path.textContent=entry.source;button.title=entry.source+'\n保存位置：'+entry.work;
  const progress=document.createElement('em');
  progress.textContent='已审核 '+(entry.reviewed||0)+' / '+entry.total+(entry.drafts?' · '+entry.drafts+' 草稿':'');
  detail.append(name,path,progress);const action=document.createElement('span');action.textContent='继续审核 →';
  button.append(detail,action);
  button.onclick=async()=>{
   if(busy || $('startProject').disabled)return;
   $('sourcePath').value=entry.source;$('workPath').value=entry.work;
   const radio=document.querySelector('[name=runMode][value=review]');radio.checked=true;radio.dispatchEvent(new Event('change'));
   await startProject();
  };
  $('recentList').append(button);
 }
}
async function refreshRecents() {
 try{renderRecents((await api('/api/recent')).rows);}
 catch(error){toast(error.message);}
}
window.prepareDesktopClose=async()=>{
 window.ddhCloseReady=null;
 const deadline=Date.now()+45000;
 while(busy && Date.now()<deadline)await new Promise(resolve=>setTimeout(resolve,100));
 if(busy){toast('当前操作尚未完成，请稍后关闭');window.ddhCloseReady=false;return;}
 setBusy(true);
 const ok=await flushDraft();
 window.ddhCloseReady=ok;
 if(!ok)setBusy(false);
};
function setupSessionControls() {
 $('undoEdit').onclick=()=>undoEdit(-1);
 $('redoEdit').onclick=()=>undoEdit(1);
 $('resetPrediction').onclick=resetPrediction;
 $('previewCrop').onclick=togglePreview;
 $('cancelSetup').onclick=()=>{visible('setup',false);updateEditButtons();};
 document.addEventListener('visibilitychange',()=>{if(document.visibilityState==='hidden')flushDraft();});
 window.addEventListener('pagehide',()=>{
  if(!current || !image || JSON.stringify(geometry())===persistedGeometry)return;
  fetch('/api/draft',{method:'POST',keepalive:true,headers:{'Content-Type':'application/json','X-DDH-Token':token},body:JSON.stringify({id:current.id,...geometry(),...(reviewMode?{task_token:current.task_token}:{})})}).catch(error=>console.error('Draft flush failed',error));
 });
}
