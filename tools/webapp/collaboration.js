'use strict';
let collaborationTimer=null;

function addCollaborationAccount(name=''){
 const rows=document.getElementById('collaborationAccounts');
 if(rows.children.length>=12)return;
 const row=document.createElement('div');row.className='collaboration-account';
 const user=document.createElement('input');user.placeholder='账号';user.value=name;user.maxLength=32;
 user.autocomplete='off';user.setAttribute('aria-label','审核账号');
 const password=document.createElement('input');password.type='password';password.placeholder='设置密码';password.minLength=10;password.maxLength=128;
 password.autocomplete='new-password';password.setAttribute('aria-label','审核密码');
 const remove=document.createElement('button');remove.className='quiet';remove.textContent='移除';
 remove.onclick=()=>{if(rows.children.length>1)row.remove();};
 row.append(user,password,remove);rows.append(row);
}
function renderCollaboration(data){
 visible('collaborationSettings',!data.active);visible('collaborationRunning',data.active);
 visible('collaborationStart',!data.active);visible('collaborationStop',data.active);
 $('collaborationOpen').textContent=data.active?'协作中':'协作审核';
 if(!data.active){
  const selected=$('collaborationListen').value;$('collaborationListen').replaceChildren(new Option('所有局域网 / 组网接口','0.0.0.0'));
  for(const address of data.addresses)$('collaborationListen').append(new Option(address,address));
  if([...$('collaborationListen').options].some(option=>option.value===selected))$('collaborationListen').value=selected;
  return;
 }
 $('collaborationLinks').replaceChildren();
 for(const url of data.urls.filter(value=>!value.includes('localhost'))){
  const row=document.createElement('div');row.className='collaboration-link';
  const value=document.createElement('code');value.textContent=url;
  const copy=document.createElement('button');copy.className='quiet';copy.textContent='复制';
  copy.onclick=async()=>{try{await navigator.clipboard.writeText(url);toast('地址已复制');}catch(error){toast('请选中地址复制');}};
  row.append(value,copy);$('collaborationLinks').append(row);
 }
 $('collaborationMembers').replaceChildren();
 for(const member of data.members){const row=document.createElement('div'),name=document.createElement('span'),progress=document.createElement('span');name.textContent=member.username;progress.textContent='已审核 '+member.reviewed+' / '+member.total;row.append(name,progress);$('collaborationMembers').append(row);}
}
async function refreshCollaboration(){
 clearTimeout(collaborationTimer);
 if($('collaborationPanel').classList.contains('hidden'))return;
 try{renderCollaboration(await api('/api/collaboration'));}catch(error){$('collaborationError').textContent=error.message;}
 finally{if(!$('collaborationPanel').classList.contains('hidden'))collaborationTimer=setTimeout(refreshCollaboration,2000);}
}
function setupCollaboration(){
 if(reviewMode)return;
 addCollaborationAccount('member1');addCollaborationAccount('member2');
 $('collaborationAdd').onclick=()=>addCollaborationAccount();
 $('collaborationOpen').onclick=async()=>{
  if(busy)return;
  if(!status?.configured){toast('请先打开需要分配的图片文件夹');return;}
  setBusy(true);
  try{if(!await flushDraft())return;visible('finish',false);visible('collaborationPanel',true);$('collaborationError').textContent='';await refreshCollaboration();}
  finally{setBusy(false);}
 };
 $('collaborationClose').onclick=()=>{visible('collaborationPanel',false);clearTimeout(collaborationTimer);updateEditButtons();};
 $('collaborationStart').onclick=async()=>{
  $('collaborationStart').disabled=true;$('collaborationError').textContent='';
  const accounts=[...$('collaborationAccounts').children].map(row=>({username:row.children[0].value.trim(),password:row.children[1].value}));
  try{
   const data=await api('/api/collaboration/start',{accounts,listen:$('collaborationListen').value,port:Number($('collaborationPort').value),public_url:$('collaborationPublic').value.trim()});
   for(const row of $('collaborationAccounts').children)row.children[1].value='';
   renderCollaboration(data);toast('协作审核已开启，请把地址和各自账号交给组员');
  }catch(error){$('collaborationError').textContent=error.message;}
  finally{$('collaborationStart').disabled=false;}
 };
 $('collaborationStop').onclick=async()=>{
  $('collaborationStop').disabled=true;
  try{renderCollaboration(await api('/api/collaboration/stop',{}));renderStatus(await api('/api/status'));await refreshQueue();toast('已停止协作，结果和草稿已保留');}
  catch(error){$('collaborationError').textContent=error.message;}
  finally{$('collaborationStop').disabled=false;}
 };
}
