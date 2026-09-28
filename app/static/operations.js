/* Credentials and retry identifiers intentionally live only in this page. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const page = document.body.dataset.page;
  mountAdminNav(page);
  const token = $('token'), notice = $('notice');
  let epoch = 0, selected = null, offset = 0, total = 0;
  const requests = new Map();
  const el = (tag, text) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = String(text); return node; };
  async function api(path, body) {
    if (!token.value.trim()) throw Error('请填写当前页面所需的凭据。');
    const started = epoch;
    const response = await fetch(path, {method: body === undefined ? 'GET' : 'POST',
      headers: {'Authorization': 'Bearer ' + token.value.trim(), 'Content-Type': 'application/json'},
      body: body === undefined ? undefined : JSON.stringify(body), cache: 'no-store'});
    if (started !== epoch) throw Error('凭据已清除，请重新连接。');
    const result = await response.json();
    if (started !== epoch) throw Error('凭据已清除，请重新连接。');
    if (!response.ok) throw Error(typeof result.detail === 'string' ? result.detail : '请求失败，请核对输入后重试。');
    return result;
  }
  function clear() {
    epoch++; token.value = ''; requests.clear(); selected = null;
    if (page === 'review') { $('queue').replaceChildren(); $('detail').textContent = '请重新连接。'; $('decision').hidden = $('publishActions').hidden = true;
      for (const id of ['category','answer','reason','mergeTarget']) $(id).value = ''; $('paging').textContent = ''; $('vectorConfirm').hidden = true; }
    else for (const id of ['cost', 'trend', 'calibration', 'job']) $(id).textContent = '尚未加载';
  }
  $('forget').onclick = clear;
  window.addEventListener('pagehide', clear);
  async function busy(operation) {
    notice.textContent = '';
    const controls = [...document.querySelectorAll('button, fieldset')].filter(node => node.id !== 'forget');
    controls.forEach(node => node.disabled = true);
    try { await operation(); } catch (error) { notice.textContent = error.message || '操作失败，未确认成功，请重试。'; }
    finally { controls.forEach(node => node.disabled = false); }
  }
  function table(headers, rows) {
    const wrap = el('div'); wrap.className = 'scroll'; const grid = el('table');
    const head = el('tr'); headers.forEach(value => head.append(el('th', value))); grid.append(head);
    rows.forEach(values => { const row = el('tr'); values.forEach(value => row.append(el('td', value ?? '无数据'))); grid.append(row); });
    wrap.append(grid); return wrap;
  }
  async function mutation(path, payload) {
    const key = JSON.stringify([path, payload]);
    if (!requests.has(key)) requests.set(key, crypto.randomUUID());
    const result = await api(path, {...payload, request_id: requests.get(key)});
    requests.delete(key);
    return result;
  }
  if (page === 'review') {
    async function loadQueue() {
      const result = await api('/api/review/questions?offset=' + offset + '&limit=20');
      total = result.total; $('queue').replaceChildren();
      for (const item of result.items) {
        const button = el('button', '#' + item.id + ' ' + item.canonical_question + ' · ' + item.status);
        button.dataset.id = String(item.id);
        button.setAttribute('aria-pressed', String(item.id === selected));
        button.onclick = () => busy(() => loadDetail(item.id)); $('queue').append(button);
      }
      $('paging').textContent = `${offset + (result.items.length ? 1 : 0)}–${offset + result.items.length} / ${total}`;
    }
    async function loadDetail(id) {
      $('vectorConfirm').hidden = true;
      const detail = await api('/api/review/questions/' + id); selected = id;
      for (const button of $('queue').querySelectorAll('button')) button.setAttribute('aria-pressed', String(button.dataset.id === String(id)));
      const host = $('detail'); host.replaceChildren(el('h3', detail.canonical_question),
        el('p', `审核：${detail.status} · 发布：${detail.publication_status} · 向量：${detail.vector_status ?? '未入库'}`));
      host.append(el('h4', '模型草稿（仅供参考）'), el('p', detail.draft_answer || '无草稿'));
      if (detail.approved_answer) host.append(el('h4', '人工批准答案'), el('p', detail.approved_answer));
      if (detail.merged_into_id) host.append(el('p', '已合并到问题 #' + detail.merged_into_id));
      for (const [label, data] of [['已遮蔽的来源', detail.sources], ['审核记录', detail.actions]]) {
        const box = el('details'); box.append(el('summary', label), el('pre', JSON.stringify(data, null, 2))); host.append(box);
      }
      if (detail.sources_truncated || detail.actions_truncated) host.append(el('p', '详情仅展示最多 100 条来源和 100 条审核记录。'));
      $('category').value = detail.category || ''; $('answer').value = detail.approved_answer || '';
      $('reason').value = ''; $('mergeTarget').value = '';
      $('decision').hidden = !['pending_review', 'deferred'].includes(detail.status);
      $('publishActions').hidden = !['approved', 'approved_pending_vector'].includes(detail.status);
      $('vectorize').hidden = !detail.knowledge_chunk_id;
    }
    $('connect').onsubmit = event => { event.preventDefault(); busy(loadQueue); };
    $('previous').onclick = () => busy(async () => { offset = Math.max(0, offset - 20); await loadQueue(); });
    $('next').onclick = () => busy(async () => { if (offset + 20 < total) offset += 20; await loadQueue(); });
    $('decision').onsubmit = event => { event.preventDefault(); busy(async () => {
      const action = $('action').value, payload = {action};
      if (action === 'approve') { payload.category = $('category').value.trim(); payload.approved_answer = $('answer').value.trim(); }
      else if (action === 'merge') payload.merge_target_id = Number($('mergeTarget').value);
      else payload.reason = $('reason').value.trim();
      await mutation(`/api/review/questions/${selected}/decision`, payload);
      await loadDetail(selected); await loadQueue(); notice.textContent = '审核决定已保存。';
    }); };
    $('publish').onclick = () => busy(async () => {
      await mutation(`/api/review/questions/${selected}/publish`, {});
      await loadDetail(selected); await loadQueue(); notice.textContent = '人工批准答案已写入知识库，请查看向量状态。';
    });
    $('vectorize').onclick = () => { $('vectorConfirm').hidden = false; };
    $('vectorCancel').onclick = () => { $('vectorConfirm').hidden = true; };
    $('vectorConfirmSubmit').onclick = () => {
      $('vectorConfirm').hidden = true;
      busy(async () => {
        const result = await mutation(`/api/review/questions/${selected}/vectorize`, {confirm_all_pending: true});
        await loadDetail(selected); notice.textContent = `本次已向量化 ${result.vectorized} 条，请以详情状态为准。`;
      });
    };
  } else {
    const today = new Date(), localDate = d => `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
    $('from').value = localDate(new Date(today.getFullYear(), today.getMonth(), 1)); $('to').value = localDate(today);
    function showSection(id, section) {
      const host = $(id); host.replaceChildren();
      if (section.status !== 'available') { host.textContent = '读取失败：' + (section.error || '不可用'); return; }
      const data = section.data;
      if (id === 'cost') {
        const metrics = el('div'); metrics.className = 'metrics';
        for (const [label, value] of [['调用', data.calls], ['已测量 token', data.measured_tokens], ['平均 token / 可用调用', data.avg_tokens], ['缺失用量调用', data.unavailable_calls]]) {
          const box = el('div', label); box.append(el('strong', value === null ? '无数据' : typeof value === 'number' ? +value.toFixed(2) : value)); metrics.append(box);
        }
        host.append(metrics, table(['日期','意图','模型','调用','缺失','token','占比'], data.rows.map(row => [row.day,row.intent,row.model_name,row.calls,row.unavailable_calls,row.measured_tokens,row.token_share === null ? null : (row.token_share*100).toFixed(1)+'%'])));
      } else if (id === 'trend') {
        host.append(table(['开始时间','策略','状态','可比较','指标'], data.rows.map(row => [row.started_at,row.strategy,row.status,row.comparable ? '是' : '否',JSON.stringify(row.metrics)])));
        if (!data.rows.length) host.append(el('p', '当前评估集和策略没有历史批次。'));
      } else {
        host.append(el('p', `状态：${data.status} · 使用阈值：${data.threshold ?? '旧 Top-1 阈值'}`));
        if (data.counts) host.append(el('p', `可答 ${data.counts.answerable} 题，库外 ${data.counts.absent} 题`));
        if (data.scan) host.append(table(['阈值','可答通过率','库外放行率','Youden J'],data.scan.map(row=>[row.threshold,row.pass_rate,row.leak_rate,row.youden_j])));
      }
      if (section.note) { const note = el('p', section.note.text); note.className = 'muted'; note.dataset.reportId = section.note.report_id; host.append(note); }
    }
    $('connect').onsubmit = event => { event.preventDefault(); busy(async () => {
      for (const id of ['cost','trend','calibration']) $(id).textContent = '正在读取当前报告…';
      const params = new URLSearchParams({from:$('from').value,to:$('to').value,strategy:$('strategy').value});
      if ($('dataset').value.trim()) params.set('dataset_hash', $('dataset').value.trim());
      try { const report = await api('/api/observability/overview?' + params); for (const id of ['cost','trend','calibration']) showSection(id, report[id]); }
      catch (error) { for (const id of ['cost','trend','calibration']) $(id).textContent = '未取得当前报告。'; throw error; }
    }); };
    $('calibrate').onclick = () => {
      if (!confirm('重新校准会调用查询改写、嵌入和精排服务，可能产生费用。确认启动？')) return;
      busy(async () => { $('job').textContent = JSON.stringify(await api('/api/jobs/calibrate-confidence', {confirm:true}),null,2); });
    };
    $('jobStatus').onclick = () => busy(async () => { $('job').textContent = JSON.stringify(await api('/api/jobs/calibrate-confidence'),null,2); });
  }
})();
