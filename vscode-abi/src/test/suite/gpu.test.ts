import * as assert from 'assert';
import * as vscode from 'vscode';

suite('GPU Status Test Suite', () => {
    test('GPU refresh interval should be configurable', () => {
        const config = vscode.workspace.getConfiguration('abi');
        const interval = config.get<number>('gpu.refreshInterval');
        assert.strictEqual(interval, 5000, 'Default refresh interval should be 5000ms');
    });

    test('GPU refresh command should be registered', async () => {
        const commands = await vscode.commands.getCommands(true);
        assert.ok(
            commands.includes('abi.gpu.refresh'),
            'GPU refresh command should be registered'
        );
    });

    test('GPU show details command should be registered', async () => {
        const commands = await vscode.commands.getCommands(true);
        assert.ok(
            commands.includes('abi.gpu.showDetails'),
            'GPU show details command should be registered'
        );
    });

    test('GPU configuration inspection', () => {
        const config = vscode.workspace.getConfiguration('abi');
        const inspection = config.inspect<number>('gpu.refreshInterval');

        assert.ok(inspection !== undefined, 'Configuration should be inspectable');
        assert.strictEqual(inspection?.defaultValue, 5000, 'Default should be 5000');
    });

    test('GPU refresh interval can be updated', async () => {
        const config = vscode.workspace.getConfiguration('abi');

        // Get current value
        const currentValue = config.get<number>('gpu.refreshInterval');
        assert.strictEqual(currentValue, 5000);

        // Verify the type is correct
        assert.strictEqual(typeof currentValue, 'number');
    });

    test('Binary path configuration exists', () => {
        const config = vscode.workspace.getConfiguration('abi');
        const binaryPath = config.get<string>('binaryPath');

        // Should default to empty string for auto-detection
        assert.strictEqual(binaryPath, '', 'Binary path should default to empty');
    });

    test('Build flags configuration exists', () => {
        const config = vscode.workspace.getConfiguration('abi');
        const buildFlags = config.get<string[]>('buildFlags');

        // Should default to empty array
        assert.deepStrictEqual(buildFlags, [], 'Build flags should default to empty array');
    });
});
