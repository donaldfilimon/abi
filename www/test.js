function log(msg, type = 'info') {
    const logEl = document.getElementById('log');
    const color = type === 'error' ? '#fc8181' : type === 'success' ? '#68d391' : '#cbd5e0';
    logEl.innerHTML += `<div style="color:${color}">[${new Date().toLocaleTimeString()}] ${msg}</div>`;
    console.log(msg);
}

async function runTest() {
    log('Starting WASM verification...');
    const statusEl = document.getElementById('status');

    try {
        // Load the WASM module
        // Assuming abi.wasm is served at root or same directory
        const response = await fetch('abi.wasm');
        if (!response.ok) throw new Error(`Failed to fetch WASM: ${response.statusText}`);

        const bytes = await response.arrayBuffer();

        // Define imports expected by Zig wasm32-freestanding
        const imports = {
            env: {
                // Trap handler or other env needs? 
                // Currently abi_wasm.zig implementation is self-contained or relies on simple panic.
                // We might need to handle memory growing if not exported.
            }
        };

        const module = await WebAssembly.instantiate(bytes, imports);
        const exports = module.instance.exports;
        const memory = exports.memory;

        log('Module instantiated successfully.', 'success');
        log(`Exports found: ${Object.keys(exports).join(', ')}`);

        // Test ABI Init
        log('Calling abi_init()...');
        const initResult = exports.abi_init();
        if (initResult === 0) {
            log('abi_init success (0)', 'success');
        } else {
            throw new Error(`abi_init failed with code ${initResult}`);
        }

        // Test Version
        log('Reading version...');
        const verLen = exports.abi_version_len();
        log(`Version length: ${verLen}`);

        // Allocate buffer for version string in WASM memory
        // Since we don't have direct alloc/free exported yet (wait, I did export abi_alloc!)
        // Check abi_wasm.zig from earlier... yes abi_alloc/abi_free were exported.

        if (!exports.abi_alloc) throw new Error('abi_alloc not exported');

        const ptr = exports.abi_alloc(verLen);
        if (ptr === 0) throw new Error('abi_alloc returned 0 (null)');

        const written = exports.abi_version_get(ptr, verLen);
        log(`Written bytes: ${written}`);

        // Read string from memory
        const view = new Uint8Array(memory.buffer, ptr, written);
        const decoder = new TextDecoder();
        const versionStr = decoder.decode(view);
        log(`ABI Version: ${versionStr}`, 'success');

        exports.abi_free(ptr, verLen);
        log('Memory freed.');

        // Test Shutdown
        exports.abi_shutdown();
        log('Shutdown complete.');

        statusEl.textContent = 'Verification Passed!';
        statusEl.className = 'status success';
        statusEl.style.display = 'block';

    } catch (err) {
        log(err.message, 'error');
        statusEl.textContent = `Error: ${err.message}`;
        statusEl.className = 'status error';
        statusEl.style.display = 'block';
    }
}
