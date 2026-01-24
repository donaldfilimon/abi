import * as assert from 'assert';
import * as vscode from 'vscode';

suite('StatusBar Test Suite', () => {
    test('Quick pick command is registered', async () => {
        const commands = await vscode.commands.getCommands(true);
        assert.ok(commands.includes('abi.showQuickPick'), 'showQuickPick command should be registered');
    });

    test('Diagnostics commands are registered', async () => {
        const commands = await vscode.commands.getCommands(true);
        assert.ok(commands.includes('abi.runDiagnostics'), 'runDiagnostics command should be registered');
        assert.ok(commands.includes('abi.clearDiagnostics'), 'clearDiagnostics command should be registered');
    });
});
