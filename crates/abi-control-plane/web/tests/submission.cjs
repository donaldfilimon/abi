'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
function browser() {
  const handlers = new Map(); const elements = new Map(); const requests = [];
  const element = id => { if (!elements.has(id)) elements.set(id, { hidden: false, textContent: '', value: '', files: [], replaceChildren() {}, addEventListener(event, handler) { handlers.set(`${id}:${event}`, handler); } }); return elements.get(id); };
  const context = vm.createContext({ document: { getElementById: element, cookie: '' }, location: { origin: 'http://127.0.0.1:3000' }, fetch: async (url, options) => { requests.push({ url, options }); return { ok: false, status: 401 }; } });
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../app.js'), 'utf8'), context);
  return { context, handlers, element, requests };
}
async function upload(value) {
  const b = browser(); b.element('job-file').files = [{ size: 1024, text: async () => JSON.stringify(value) }];
  b.handlers.get('submit-job:submit')({ preventDefault() {} });
  await new Promise(resolve => setImmediate(resolve)); return b;
}
test('plain prompt JSON never reaches the account service', async () => {
  const b = await upload({ prompt: 'private-browser-canary', tokens: 10 });
  assert.equal(b.requests.filter(r => r.options.method === 'POST').length, 0);
  assert.match(b.element('message').textContent, /will not be uploaded/);
});
function packet() {
  const id = 'a1b2c3d4-1234-4321-8765-a1b2c3d4e5f6';
  return { binding: { protocol: 'abi.relay-job.v1', audience: 'http://127.0.0.1:3000', job_id: id, account_id: id, tenant_id: id, node_id: id, capability: 'model.infer', expires_at: 200, submitted_at: 100, max_result_bytes: 65536, reply_key: Array(32).fill(1) }, payload: { encapsulated_key: Array(32).fill(2), ciphertext: Array(32).fill(3) } };
}
test('extra plaintext fields inside a sealed-looking packet are refused locally', async () => {
  const value = packet(); value.payload.prompt = 'private-nested-canary';
  const b = await upload(value); assert.equal(b.requests.filter(r => r.options.method === 'POST').length, 0);
});
test('a schema-shaped ciphertext artifact reaches the authenticated submit endpoint', async () => {
  const b = await upload(packet()); assert.equal(b.requests.filter(r => r.url === '/v1/jobs' && r.options.method === 'POST').length, 1);
});
