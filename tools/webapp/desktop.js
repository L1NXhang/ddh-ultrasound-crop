'use strict';
const desktopHost=window.chrome?.webview;
const desktopRequests=new Map();
if(desktopHost){
 desktopHost.addEventListener('message',event=>{const data=event.data,request=desktopRequests.get(data.id);if(!request)return;desktopRequests.delete(data.id);clearTimeout(request.timer);data.error?request.reject(Error(data.error)):request.resolve(data.result);});
 document.getElementById('environmentLabel').textContent='桌面版';
 document.title='DDH 裁剪工具';
}
function desktopRequest(action,data){
 return new Promise((resolve,reject)=>{const id=crypto.randomUUID();const timer=setTimeout(()=>{desktopRequests.delete(id);reject(Error('目录选择超时，请重新选择'));},300000);desktopRequests.set(id,{resolve,reject,timer});desktopHost.postMessage({id,action,...data});});
}
