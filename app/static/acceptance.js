(() => {
 'use strict';mountAdminNav('acceptance');
 const $=id=>document.getElementById(id),page=document.body.dataset.page;
 let epoch=0;
 const el=(tag,text)=>{const n=document.createElement(tag);n.textContent=text;return n;};
 async function api(path,body){if(!$('token').value.trim())throw Error('请填写观测凭据');
  const current=epoch,response=await fetch(path,{method:body===undefined?'GET':'POST',cache:'no-store',headers:{Authorization:'Bearer '+$('token').value.trim(),'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
  const result=await response.json();if(current!==epoch)throw Error('凭据已清除');if(!response.ok)throw Error(typeof result.detail==='string'?result.detail:'读取失败，请重试');return result;}
 function clear(){epoch++;$('token').value='';$('versions').textContent='尚未加载';$('content').replaceChildren();$('job').textContent='';$('actions').hidden=true;}
 $('forget').onclick=clear;window.addEventListener('pagehide',clear);
 async function busy(fn){$('notice').textContent='';const buttons=[...document.querySelectorAll('button')].filter(n=>n.id!=='forget');buttons.forEach(n=>n.disabled=true);try{await fn();}catch(e){$('notice').textContent=e.message;}finally{buttons.forEach(n=>n.disabled=false);}}
 const pretty=data=>JSON.stringify(data,null,2);
 function card(item){const section=el('section','');section.append(el('h2',(item.title||'报告')+' · '+item.status));if(item.reason)section.append(el('p',item.reason));if(item.command)section.append(el('code',item.command));
  if(item.data){const details=el('details','');details.append(el('summary','查看同源报告'),el('pre',pretty(item.data)));section.append(details);}return section;}
 async function load(){$('content').replaceChildren();$('versions').textContent='加载中';
  const report=await api('/api/acceptance/'+page);
  if(page==='overview'){$('versions').replaceChildren(el('p','本机服务：'+report.health.status),el('pre',pretty(report.versions)));for(const item of report.items)$('content').append(card(item));$('actions').hidden=false;}
  else if(page==='data'){$('versions').textContent='与命令行产物同源';for(const item of report.items)$('content').append(card(item));}
  else if(page==='errors'){$('versions').replaceChildren(el('p','状态：'+report.status+' '+report.reason),el('pre',pretty(report.metadata)));if(!report.errors.length)$('content').append(el('p','没有可展示错例；请结合报告状态判断是否完成评估。'));for(const row of report.errors){const section=el('section','');section.append(el('h2','#'+row.index+' '+row.kind),el('p',row.text),el('p','漏标：'+row.missed.join('、')),el('p','多标：'+row.extra.join('、')));$('content').append(section);}}
  else{$('versions').textContent='状态：'+report.status;$('content').append(card(report));}
 }
 $('connect').onsubmit=e=>{e.preventDefault();busy(load);};
 $('smoke').onclick=()=>busy(async()=>{$('job').textContent=pretty(await api('/api/jobs/classifier-smoke',{}));});
 $('jobStatus').onclick=()=>busy(async()=>{$('job').textContent=pretty(await api('/api/jobs/classifier-smoke'));});
})();
