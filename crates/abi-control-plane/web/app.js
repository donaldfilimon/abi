'use strict';
const element = id => document.getElementById(id);
function message(text) { element('message').textContent = text; }
function csrf() { const values = document.cookie.split(';').map(s => s.trim()); const value = values.find(s => s.startsWith('__Host-abi_csrf=') || s.startsWith('abi_csrf=')); return value ? value.slice(value.indexOf('=') + 1) : ''; }
async function api(path, method = 'GET', body) {
  const response = await fetch(path, { method, credentials: 'same-origin', headers: { 'Content-Type': 'application/json', 'X-CSRF-Token': csrf() }, body: body ? JSON.stringify(body) : undefined });
  if (!response.ok) throw new Error(response.status === 401 ? 'Your session expired or this action was refused. Sign in again.' : 'Request refused. Check the code, expiry, and node status.');
  return response.status === 204 ? null : response.json();
}
async function refresh() {
  try {
    const view = await api('/v1/account');
    element('signed-out').hidden = true; element('account').hidden = false;
    element('tenant').textContent = `Account ${view.account.account_id} · Workspace ${view.account.tenant_id}`;
    element('nodes').replaceChildren(); element('empty').hidden = view.nodes.length > 0;
    for (const node of view.nodes) {
      const item = document.createElement('li'); const text = document.createElement('p');
      text.textContent = `${node.node_id} — ${node.revoked ? 'Revoked' : Date.now() >= node.expires_at * 1000 ? 'Credential expired' : 'Enrolled'} · ${node.capabilities.join(', ')}`;
      item.append(text);
      if (!node.revoked) { const button = document.createElement('button'); button.className = 'secondary'; button.textContent = 'Revoke'; button.addEventListener('click', () => run(async () => { await api(`/v1/nodes/${encodeURIComponent(node.node_id)}/revoke`, 'POST'); await refresh(); })); item.append(button); }
      element('nodes').append(item);
    }
    message('Account connected. Enrollment status does not imply a worker is online.');
  } catch { element('signed-out').hidden = false; element('account').hidden = true; element('jobs').replaceChildren(); stopWatching(); message('Sign in to manage your private nodes.'); }
}
async function run(action) { try { await action(); } catch (error) { message(error.message); } }
element('pair').addEventListener('submit', event => { event.preventDefault(); run(async () => {
  const preview = await api('/v1/enrollments/claim', 'POST', { user_code: element('code').value.trim() });
  element('code').value = ''; element('preview').hidden = false; element('preview').textContent = JSON.stringify(preview, null, 2);
  message('Code consumed. Check this exact account, workspace, public key, and capabilities on your node before approving locally.');
}); });
element('refresh').addEventListener('click', () => run(refresh));
element('logout').addEventListener('click', () => run(async () => { await api('/v1/logout', 'POST'); await refresh(); }));
element('delete').addEventListener('click', () => run(async () => { if (window.confirm('Delete your account and revoke every enrolled node?')) { await api('/v1/account', 'DELETE'); element('preview').hidden = true; await refresh(); } }));
refresh();

const watchedJobs = new Map();
const durableJobs = new Map();
let watchGeneration = 0;
function stopWatching() { watchGeneration += 1; for (const token of watchedJobs.values()) token.active = false; watchedJobs.clear(); durableJobs.clear(); }
function showJob(receipt, frameCount = 0) {
  const id = receipt.binding.job_id;
  const durable = durableJobs.get(id);
  let item = document.getElementById(`job-${id}`);
  if (!item) { item = document.createElement('li'); item.id = `job-${id}`; element('jobs').prepend(item); }
  item.replaceChildren(); const label = document.createElement('p'); label.textContent = `${id} · ${receipt.state} · ${frameCount} encrypted frames`; item.append(label);
  const refresh = document.createElement('button'); refresh.className = 'secondary'; refresh.textContent = 'Download encrypted output';
  refresh.addEventListener('click', () => run(async () => {
    const generation = watchGeneration;
    const view = await api(durable ? `/v2/jobs/${encodeURIComponent(id)}/delivery?after=${durable.after}` : `/v1/jobs/${encodeURIComponent(id)}`); if (generation !== watchGeneration) return; showJob(view.receipt, view.frames?.length || view.packets?.length || 0);
    if (durable) { message(deliveryLabel(view)); durable.after = view.next < 16 ? view.next : 0; }
    if (view.result || view.frames?.length || view.packets?.length) {
      const url = URL.createObjectURL(new Blob([JSON.stringify(view)], { type: 'application/json' }));
      const link = document.createElement('a'); link.href = url; link.download = `abi-result-${id}.json`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
      message('Encrypted result downloaded. Open it with your local client and private reply key.');
    } else { message(`Job ${view.receipt.state}. ${view.receipt.state === 'completed' ? 'Encrypted output is no longer held by the relay.' : ''}`); }
  })); item.append(refresh);
  const watch = document.createElement('button'); watch.className = 'secondary'; watch.textContent = watchedJobs.has(id) ? 'Watching encrypted stream' : 'Watch encrypted stream';
  watch.addEventListener('click', () => {
    if (watchedJobs.has(id)) return;
    const token = { active: true }; watchedJobs.set(id, token);
    const poll = async () => {
      try {
        const view = await api(durable ? `/v2/jobs/${encodeURIComponent(id)}/delivery?after=0` : `/v1/jobs/${encodeURIComponent(id)}`);
        if (!token.active) return;
        showJob(view.receipt, view.frames?.length || view.packets?.length || 0);
        if (durable) message(deliveryLabel(view));
        if ((!durable && !['queued', 'offered', 'dispatched', 'cancel_requested'].includes(view.receipt.state)) || (durable && (Date.now() >= view.policy.policy.until_unix_s * 1000 || view.manifest?.terminal_digest && view.packets?.some(p => p.kind === 'terminal')))) { token.active = false; watchedJobs.delete(id); return; }
        setTimeout(poll, 1000);
      } catch (error) { token.active = false; watchedJobs.delete(id); message(error.message); }
    }; poll();
  }); item.append(watch);
  if (receipt.state === 'queued' || receipt.state === 'offered' || receipt.state === 'dispatched') {
    const cancel = document.createElement('button'); cancel.className = 'secondary'; cancel.textContent = 'Cancel';
    cancel.addEventListener('click', () => run(async () => { showJob(await api(`/v1/jobs/${encodeURIComponent(id)}/cancel`, 'POST')); message('Cancellation requested. Running work is cancelled only after the worker confirms it stopped.'); })); item.append(cancel);
  }
}
element('submit-job').addEventListener('submit', event => { event.preventDefault(); run(async () => {
  const generation = watchGeneration;
  const file = element('job-file').files[0]; if (!file || file.size > 384 * 1024) throw new Error('Choose a bounded encrypted submission JSON file.');
  const submission = JSON.parse(await file.text());
  if (generation !== watchGeneration) return;
  const durable = isDurableSubmission(submission);
  if (!durable && !isSealedSubmission(submission)) throw new Error('Choose a sealed job from the local client. Plaintext or unexpected fields will not be uploaded.');
  const receipt = await api(durable ? '/v2/jobs' : '/v1/jobs', 'POST', submission);
  if (generation !== watchGeneration) return;
  if (durable) durableJobs.set(receipt.binding.job_id, { after: 0 });
  showJob(receipt); element('job-file').value = ''; message('Encrypted job queued on your enrolled node.');
}); });

function isSealedSubmission(value, allowTools = false) {
  const object = (v, keys) => v && typeof v === 'object' && !Array.isArray(v) && Object.keys(v).length === keys.length && keys.every(k => Object.hasOwn(v, k));
  const bytes = (v, length) => Array.isArray(v) && v.length === length && v.every(n => Number.isInteger(n) && n >= 0 && n <= 255);
  const uuid = v => typeof v === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(v);
  if (!object(value, ['binding', 'payload']) || !object(value.payload, ['encapsulated_key', 'ciphertext'])) return false;
  const b = value.binding; const p = value.payload;
  if (!object(b, ['protocol', 'audience', 'job_id', 'account_id', 'tenant_id', 'node_id', 'capability', 'expires_at', 'submitted_at', 'max_result_bytes', 'reply_key'])) return false;
  return b.protocol === 'abi.relay-job.v1' && b.audience === location.origin && (b.capability === 'model.infer' || (allowTools && b.capability === 'tool.execute'))
    && [b.job_id, b.account_id, b.tenant_id, b.node_id].every(uuid)
    && [b.expires_at, b.submitted_at, b.max_result_bytes].every(Number.isSafeInteger)
    && b.max_result_bytes >= 4096 && b.max_result_bytes <= 65536
    && bytes(b.reply_key, 32) && bytes(p.encapsulated_key, 32)
    && Array.isArray(p.ciphertext) && p.ciphertext.length >= 16 && p.ciphertext.length <= 65552 && bytes(p.ciphertext, p.ciphertext.length);
}

function isDurableSubmission(value) {
  const exact = (v, keys) => v && typeof v === 'object' && !Array.isArray(v) && Object.keys(v).length === keys.length && keys.every(k => Object.hasOwn(v, k));
  const digest = v => typeof v === 'string' && /^[A-Za-z0-9_-]{43}$/.test(v);
  if (!exact(value, ['submission', 'delivery']) || !isSealedSubmission(value.submission, true) || !exact(value.delivery, ['until_unix_s', 'binding_digest', 'submission_digest'])) return false;
  const p = value.delivery; const b = value.submission.binding;
  return Number.isSafeInteger(p.until_unix_s) && p.until_unix_s >= b.submitted_at + 60 && p.until_unix_s <= b.submitted_at + 7 * 86400 && digest(p.binding_digest) && digest(p.submission_digest);
}
function isDeliveryAck(value) {
  const exact = (v, keys) => v && typeof v === 'object' && !Array.isArray(v) && Object.keys(v).length === keys.length && keys.every(k => Object.hasOwn(v, k));
  const digest = v => typeof v === 'string' && /^[A-Za-z0-9_-]{43}$/.test(v);
  return exact(value, ['job_id', 'ack']) && typeof value.job_id === 'string' && /^[0-9a-f-]{36}$/.test(value.job_id)
    && exact(value.ack, ['manifest_digest', 'contiguous', 'terminal_digest']) && digest(value.ack.manifest_digest)
    && Number.isInteger(value.ack.contiguous) && value.ack.contiguous >= 0 && value.ack.contiguous <= 16
    && (value.ack.terminal_digest === null || digest(value.ack.terminal_digest));
}
function deliveryLabel(page) {
  if (Date.now() >= page.policy.policy.until_unix_s * 1000) return 'Delivery retention expired.';
  if (page.cancelled) return 'Cancellation requested or acknowledged; only the verified saved prefix can be imported.';
  if (page.receipt.state === 'unresolved') return page.packets?.length ? 'Partial output; execution remains unresolved.' : 'Worker output unavailable; waiting for the enrolled worker to recover delivery.';
  if (page.receipt.state === 'completed') return page.packets?.some(p => p.kind === 'terminal') ? 'Encrypted completion available. Import locally, then upload its receipt acknowledgement.' : 'Recovering encrypted completion from your worker.';
  return 'Worker running or recovering delivery. Encrypted output is verified by your local client.';
}
element('ack-job').addEventListener('submit', event => { event.preventDefault(); run(async () => {
  const file = element('ack-file').files[0]; if (!file || file.size > 4096) throw new Error('Choose a bounded local receipt acknowledgement.');
  const value = JSON.parse(await file.text()); if (!isDeliveryAck(value)) throw new Error('Choose an acknowledgement exported by your local verified client.');
  await api(`/v2/jobs/${encodeURIComponent(value.job_id)}/delivery/ack`, 'POST', value.ack);
  element('ack-file').value = ''; message('Owning client receipt acknowledged. Worker ciphertext remains retained until its immutable delivery deadline.');
}); });
