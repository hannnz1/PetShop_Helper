(() => {
 'use strict'; mountAdminNav('topics');
 const $=id=>document.getElementById(id), el=(tag,text)=>{const n=document.createElement(tag);n.textContent=text;return n;};
 const questions=document.body.dataset.page==='questions', label=new URLSearchParams(location.search).get('label');
 let epoch=0,offset=0,total=0;
 async function api(path,body){
   if(!$('token').value.trim())throw Error('请填写审核凭据');
   const current=epoch,response=await fetch(path,{method:body===undefined?'GET':'POST',cache:'no-store',
    headers:{Authorization:'Bearer '+$('token').value.trim(),'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
   const data=await response.json();if(current!==epoch)throw Error('凭据已清除');
   if(!response.ok)throw Error(typeof data.detail==='string'?data.detail:'加载失败，请重试');return data;
 }
 function clear(){epoch++;$('token').value='';$('summary').textContent='尚未加载';$('content').replaceChildren();if($('job'))$('job').textContent='';}
 $('forget').onclick=clear;window.addEventListener('pagehide',clear);
 async function busy(fn){$('notice').textContent='';const buttons=[...document.querySelectorAll('button')].filter(n=>n.id!=='forget');buttons.forEach(n=>n.disabled=true);
   try{await fn();}catch(e){$('notice').textContent=e.message;}finally{buttons.forEach(n=>n.disabled=false);if(questions){$('previous').disabled=offset===0;$('next').disabled=offset+20>=total;}}}
 async function load(){
   $('content').replaceChildren();$('summary').textContent='加载中';
   const data=await api(questions?'/api/topics/questions?label='+encodeURIComponent(label||'')+'&offset='+offset+'&limit=20':'/api/topics/distribution');
   if(questions){total=data.total;$('summary').textContent=`${data.label} · ${offset+(data.items.length?1:0)}–${offset+data.items.length} / ${total}`;
    if(!data.items.length)$('content').append(el('p','该主题暂无问题'));
    for(const row of data.items){const card=el('article','');card.append(el('h3','#'+row.id+' '+row.text),el('p',row.labels.join('、')+' · 来源：'+row.source),el('p','分类时间：'+row.classified_at),el('small','模型版本：'+row.model_version));
      card.append(el('p',row.review?'审核 #'+row.review.id+'：'+row.review.status:'暂无审核关联'));$('content').append(card);}
   }else{$('summary').replaceChildren(el('p',`已分类独立问题 ${data.question_count} · 未分类 ${data.unclassified_count}`),el('small','版本口径：'+(data.versions.map(v=>v.model_version+' ('+v.question_count+')').join('；')||'无数据')));
    const maximum=Math.max(1,...Object.values(data.class_counts));
    for(const [name,count]of Object.entries(data.class_counts)){const row=el('p',''),link=el('a',name+' · '+count);link.href='/topics/questions?label='+encodeURIComponent(name);const bar=document.createElement('meter');bar.min=0;bar.max=maximum;bar.value=count;bar.setAttribute('aria-label',name+'数量');row.append(link,document.createTextNode(' '),bar);$('content').append(row);}}
 }
 $('connect').onsubmit=e=>{e.preventDefault();offset=0;busy(load);};
 if(questions){$('previous').onclick=()=>{offset=Math.max(0,offset-20);busy(load);};$('next').onclick=()=>{if(offset+20<total){offset+=20;busy(load);}};}
 else $('classify').onclick=()=>busy(async()=>{$('job').textContent=JSON.stringify(await api('/api/jobs/classify-pool',{}),null,2);});
})();
