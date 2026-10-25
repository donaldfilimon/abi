'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const tick = () => new Promise(resolve => setImmediate(resolve));
async function browser(views) {
  const nodes = new Map(); const created = []; const timers = []; const requests = [];
  function node() { return { textContent: '', children: [], listeners: {}, append(...items) { this.children.push(...items); }, prepend(...items) { this.children.unshift(...items); }, replaceChildren(...items) { this.children = items; }, addEventListener(name, handler) { this.listeners[name] = handler; } }; }
  const element = id => { if (!nodes.has(id)) nodes.set(id, node()); return nodes.get(id); };
  const context = vm.createContext({ document: { cookie: '', getElementById: element, createElement() { const n = node(); created.push(n); return n; } }, location: { origin: 'http://127.0.0.1:3000' }, setTimeout(fn) { timers.push(fn); }, fetch: async url => {
    requests.push(url); if (url === '/v1/account') return { ok: false, status: 401 };
    return { ok: true, status: 200, json: async () => views.shift() };
  } });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../app.js'), 'utf8'), context); await tick();
  vm.runInContext('showJob({binding:{job_id:"job"},state:"dispatched"})', context);
  const watch = created.find(n => n.textContent === 'Watch encrypted stream');
  return { context, element, created, timers, requests, watch };
}
test('watch polls encrypted frame counts through cancellation acknowledgement', async () => {
  const b = await browser([
    { receipt: { binding: { job_id: 'job' }, state: 'cancel_requested' }, frames: [{ payload: { ciphertext: [1, 2, 3] } }] },
    { receipt: { binding: { job_id: 'job' }, state: 'cancelled' }, frames: [{ payload: { ciphertext: [1, 2, 3] } }] },
  ]);
  b.watch.listeners.click(); await tick();
  assert.match(b.element('job-job').children[0].textContent, /cancel_requested.*1 encrypted frames/);
  assert.equal(b.timers.length, 1); b.timers.shift()(); await tick();
  assert.match(b.element('job-job').children[0].textContent, /cancelled.*1 encrypted frames/);
  assert.equal(b.timers.length, 0);
  assert.deepEqual(b.requests.filter(p => p.includes('/v1/jobs/')), ['/v1/jobs/job', '/v1/jobs/job']);
});
test('logout stops a pending watch response from restoring private job metadata', async () => {
  const b = await browser([{ receipt: { binding: { job_id: 'job' }, state: 'dispatched' }, frames: [] }]);
  b.watch.listeners.click(); vm.runInContext('stopWatching()', b.context); await tick();
  assert.equal(b.timers.length, 0);
});
