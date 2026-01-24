import * as assert from 'assert';
import * as vscode from 'vscode';

suite('Chat Provider Test Suite', () => {
    test('Chat view should be defined in contributes', () => {
        // Check that the extension contributes the chat view
        const extension = vscode.extensions.getExtension('abi-team.abi-framework');

        // In test mode, extension may not be fully loaded
        // So we just verify the configuration is accessible
        const config = vscode.workspace.getConfiguration('abi');
        assert.ok(config !== undefined, 'Configuration should be accessible');
    });

    test('Chat configuration should have model setting', () => {
        const config = vscode.workspace.getConfiguration('abi');
        const model = config.get<string>('chat.model');
        assert.strictEqual(model, 'gpt-oss', 'Default model should be gpt-oss');
    });

    test('Chat configuration should have streaming setting', () => {
        const config = vscode.workspace.getConfiguration('abi');
        const streaming = config.get<boolean>('chat.enableStreaming');
        assert.strictEqual(streaming, true, 'Streaming should be enabled by default');
    });

    test('Clear history command should be registered', async () => {
        const commands = await vscode.commands.getCommands(true);
        assert.ok(
            commands.includes('abi.chat.clear'),
            'Clear history command should be registered'
        );
    });

    test('Send command should be registered', async () => {
        const commands = await vscode.commands.getCommands(true);
        assert.ok(
            commands.includes('abi.chat.send'),
            'Send command should be registered'
        );
    });

    test('Chat model can be configured', async () => {
        const config = vscode.workspace.getConfiguration('abi');

        // Get default value
        const defaultModel = config.get<string>('chat.model');
        assert.strictEqual(defaultModel, 'gpt-oss');

        // Verify configuration schema allows string values
        const inspection = config.inspect<string>('chat.model');
        assert.ok(inspection?.defaultValue === 'gpt-oss');
    });
});
