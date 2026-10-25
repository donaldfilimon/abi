'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
function browser() {
  const handlers = new Map(); const nodes = new Map(); const posts = [];
  const element = id => { if (!nodes.has(id)) nodes.set(id, { files: [], value: '', replaceChildren() {}, addEventListener(event, fn) { handlers.set(`${id}:${event}`, fn); } }); return nodes.get(id); };
  const context = vm.createContext({ document: { getElementById: element, cookie: '' }, location: { origin: 'http://127.0.0.1:3000' }, fetch: async (url, options) => { if (options.method === 'POST') posts.push({ url, body: JSON.parse(options.body) }); return { ok: false, status: 401 }; } });
  vm.runInContext(fs.readFileSync(`${__dirname}/../app.js`, 'utf8'), context);
  return { context, element, handlers, posts };
}
function submission() {
  const id = 'a1b2c3d4-1234-4321-8765-a1b2c3d4e5f6';
  return { submission: { binding: { protocol: 'abi.relay-job.v1', audience: 'http://127.0.0.1:3000', job_id: id, account_id: id, tenant_id: id, node_id: id, capability: 'model.infer', expires_at: 200, submitted_at: 100, max_result_bytes: 65536, reply_key: Array(32).fill(1) }, payload: { encapsulated_key: Array(32).fill(2), ciphertext: Array(32).fill(3) } }, delivery: { until_unix_s: 86500, binding_digest: 'a'.repeat(43), submission_digest: 'b'.repeat(43) } };
}
async function upload(value) { const b = browser(); b.element('job-file').files = [{ size: 1024, text: async () => JSON.stringify(value) }]; b.handlers.get('submit-job:submit')({ preventDefault() {} }); await new Promise(setImmediate); return b; }
test('durable wrapper uses v2 and excludes private keys and plaintext', async () => {
  assert.equal((await upload(submission())).posts[0].url, '/v2/jobs');
  for (const mutate of [v => v.reply_key_seed = [1], v => v.delivery.prompt = 'private', v => v.submission.payload.prompt = 'private', v => v.delivery.until_unix_s = 9999999]) { const v = submission(); mutate(v); assert.equal((await upload(v)).posts.length, 0); }
});
test('acknowledgement validates local artifact without browser decryption claims', async () => {
  const b = browser(); const ack = { job_id: submission().submission.binding.job_id, ack: { manifest_digest: 'a'.repeat(43), contiguous: 2, terminal_digest: null } };
  await new Promise(setImmediate); b.element('ack-file').files = [{ size: 1024, text: async () => JSON.stringify(ack) }]; b.handlers.get('ack-job:submit')({ preventDefault() {} }); await new Promise(setImmediate); assert.equal(b.posts.length, 1); assert.match(b.posts[0].url, /\/v2\/jobs\/.*\/delivery\/ack/);
  ack.ack.plaintext = 'private'; b.handlers.get('ack-job:submit')({ preventDefault() {} }); await new Promise(setImmediate); assert.equal(b.posts.length, 1);
});
test('delivery messages distinguish partial, recovering, unavailable and expiry', () => {
  const b = browser(); const page = { policy: { policy: { until_unix_s: 99999999999 } }, receipt: { state: 'unresolved' }, packets: [] };
  b.context.page = page; assert.match(vm.runInContext('deliveryLabel(page)', b.context), /unavailable/); page.packets = [{}]; assert.match(vm.runInContext('deliveryLabel(page)', b.context), /Partial/); page.receipt.state = 'completed'; assert.match(vm.runInContext('deliveryLabel(page)', b.context), /Recovering/); page.policy.policy.until_unix_s = 0; assert.match(vm.runInContext('deliveryLabel(page)', b.context), /expired/);
});

test('governed tool ciphertext requires durable wrapper and offers no decision authority', async () => {
  const value = submission(); value.submission.binding.capability = 'tool.execute';
  assert.equal((await upload(value)).posts[0].url, '/v2/jobs');
  assert.equal((await upload(value.submission)).posts.length, 0);
  value.approval = { decision: 'approved' };
  assert.equal((await upload(value)).posts.length, 0);
});
