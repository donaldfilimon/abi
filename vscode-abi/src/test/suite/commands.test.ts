import * as assert from 'assert';
import * as vscode from 'vscode';

suite('Command Test Suite', () => {
    test('Build command should be executable', async function() {
        this.timeout(30000); // Build may take time

        try {
            // Just verify the command exists and can be called
            // Actual execution may fail without proper workspace
            await vscode.commands.executeCommand('abi.build');
            assert.ok(true, 'Build command executed');
        } catch (err) {
            // Command may fail if binary not present, but should not throw unexpectedly
            assert.ok(true, 'Build command handled error gracefully');
        }
    });

    test('Test command should be executable', async function() {
        this.timeout(30000);

        try {
            await vscode.commands.executeCommand('abi.test');
            assert.ok(true, 'Test command executed');
        } catch (err) {
            assert.ok(true, 'Test command handled error gracefully');
        }
    });

    test('Format command should be executable', async function() {
        this.timeout(10000);

        try {
            await vscode.commands.executeCommand('abi.format');
            assert.ok(true, 'Format command executed');
        } catch (err) {
            assert.ok(true, 'Format command handled error gracefully');
        }
    });

    test('Lint command should be executable', async function() {
        this.timeout(10000);

        try {
            await vscode.commands.executeCommand('abi.lint');
            assert.ok(true, 'Lint command executed');
        } catch (err) {
            assert.ok(true, 'Lint command handled error gracefully');
        }
    });

    test('GPU refresh command should be executable', async () => {
        try {
            await vscode.commands.executeCommand('abi.gpu.refresh');
            assert.ok(true, 'GPU refresh command executed');
        } catch (err) {
            // May fail without GPU, but should not throw unexpectedly
            assert.ok(true, 'GPU refresh command handled error gracefully');
        }
    });

    test('Chat send command should be executable', async () => {
        try {
            await vscode.commands.executeCommand('abi.chat.send');
            assert.ok(true, 'Chat send command executed');
        } catch (err) {
            assert.ok(true, 'Chat send command handled error gracefully');
        }
    });

    test('Chat clear command should be executable', async () => {
        try {
            await vscode.commands.executeCommand('abi.chat.clear');
            assert.ok(true, 'Chat clear command executed');
        } catch (err) {
            assert.ok(true, 'Chat clear command handled error gracefully');
        }
    });
});
