'use strict';
const reviewMode=document.body.dataset.reviewOnly==='true';
let reviewAuthenticated=false;

function reviewNeedsLogin(){
 delete document.body.dataset.reviewAuthenticated;
 reviewAuthenticated=false;clearTimeout(pollTimer);pollTimer=null;
 visible('setup',false);visible('finish',false);visible('reviewLogin',true);
 $('reviewLogout').classList.add('hidden');
 updateEditButtons();
}
async function reviewConnect(){
 const data=await api('/api/bootstrap');token=data.token;reviewAuthenticated=true;
 document.body.dataset.reviewAuthenticated='true';
 visible('setup',false);visible('reviewLogin',false);visible('reviewLogout',true);
 $('environmentLabel').textContent='审核员 · '+data.status.username;
 $('modeBadge').textContent='协作审核';$('finishLater').textContent='返回工作台';
 renderStatus(data.status);reviewPrompted=false;scope='pending';$('scope').value=scope;
 await nextImage();clearTimeout(pollTimer);pollTimer=setTimeout(poll,1200);
}
async function reviewBoot(){
 document.title='DDH · 协作审核';visible('setup',false);
 try{await reviewConnect();}catch(error){reviewNeedsLogin();if(!error.message.includes('请登录'))$('reviewLoginError').textContent=error.message;}
}
if(reviewMode){
 document.getElementById('environmentLabel').textContent='协作审核';
 document.getElementById('setup').classList.add('hidden');
 document.getElementById('reviewLogin').classList.remove('hidden');
 document.getElementById('reviewLoginForm').onsubmit=async event=>{
  event.preventDefault();$('reviewLoginSubmit').disabled=true;$('reviewLoginError').textContent='';
  try{
   const info=await fetch('/api/login-info').then(response=>response.json());
   const response=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json','X-DDH-Login':info.nonce},body:JSON.stringify({username:$('reviewUsername').value.trim(),password:$('reviewPassword').value})});
   const data=await response.json();if(!response.ok)throw Error(data.error||'登录失败');
   $('reviewPassword').value='';await reviewConnect();
  }catch(error){$('reviewLoginError').textContent=error.message;}
  finally{$('reviewLoginSubmit').disabled=false;}
 };
 document.getElementById('reviewLogout').onclick=async()=>{
  if(busy)return;setBusy(true);
  try{if(!await flushDraft())return;await api('/api/logout',{});clearCanvas();reviewNeedsLogin();token='';}
  catch(error){toast(error.message);}finally{setBusy(false);}
 };
}
