import * as vscode from 'vscode';
import { OutputChannelManager } from '../core/outputChannel';
import { ProcessRunner } from '../core/processRunner';

export interface TestOptions {
    filter?: string;
}

export async function buildCommand(outputChannel: OutputChannelManager): Promise<void> {
    const processRunner = new ProcessRunner(outputChannel);
    const config = vscode.workspace.getConfiguration('abi');
    const buildFlags = config.get<string[]>('buildFlags', []);

    outputChannel.clear();
    outputChannel.show();
    outputChannel.appendLine('=== Building ABI Framework ===\n');

    const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    if (!workspaceFolder) {
        vscode.window.showErrorMessage('No workspace folder found');
        return;
    }

    try {
        const result = await processRunner.run('zig', ['build', ...buildFlags], {
            cwd: workspaceFolder,
            showOutput: true
        });

        if (result.exitCode === 0) {
            vscode.window.showInformationMessage('Build completed successfully');
            outputChannel.appendLine('\n=== Build successful ===');
        } else {
            vscode.window.showErrorMessage('Build failed. Check output for details.');
            outputChannel.appendLine('\n=== Build failed ===');
        }
    } catch (error) {
        vscode.window.showErrorMessage(`Build error: ${error}`);
        outputChannel.appendLine(`\nError: ${error}`);
    }
}

export async function testCommand(outputChannel: OutputChannelManager, options?: TestOptions): Promise<void> {
    const processRunner = new ProcessRunner(outputChannel);

    outputChannel.clear();
    outputChannel.show();
    outputChannel.appendLine('=== Running ABI Tests ===\n');

    const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    if (!workspaceFolder) {
        vscode.window.showErrorMessage('No workspace folder found');
        return;
    }

    const testArgs = ['build', 'test', '--summary', 'all'];
    if (options?.filter) {
        outputChannel.appendLine(`Filter: ${options.filter}\n`);
    }

    try {
        const result = await processRunner.run('zig', testArgs, {
            cwd: workspaceFolder,
            showOutput: true
        });

        if (result.exitCode === 0) {
            vscode.window.showInformationMessage('All tests passed');
            outputChannel.appendLine('\n=== All tests passed ===');
        } else {
            vscode.window.showWarningMessage('Some tests failed. Check output for details.');
            outputChannel.appendLine('\n=== Some tests failed ===');
        }
    } catch (error) {
        vscode.window.showErrorMessage(`Test error: ${error}`);
        outputChannel.appendLine(`\nError: ${error}`);
    }
}

export async function formatCommand(outputChannel: OutputChannelManager): Promise<void> {
    const processRunner = new ProcessRunner(outputChannel);

    outputChannel.clear();
    outputChannel.show();
    outputChannel.appendLine('=== Formatting ABI Code ===\n');

    const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    if (!workspaceFolder) {
        vscode.window.showErrorMessage('No workspace folder found');
        return;
    }

    try {
        const result = await processRunner.run('zig', ['fmt', '.'], {
            cwd: workspaceFolder,
            showOutput: true
        });

        if (result.exitCode === 0) {
            vscode.window.showInformationMessage('Code formatted successfully');
            outputChannel.appendLine('\n=== Format complete ===');
        } else {
            vscode.window.showWarningMessage('Format completed with warnings');
        }
    } catch (error) {
        vscode.window.showErrorMessage(`Format error: ${error}`);
        outputChannel.appendLine(`\nError: ${error}`);
    }
}

export async function lintCommand(outputChannel: OutputChannelManager): Promise<void> {
    const processRunner = new ProcessRunner(outputChannel);

    outputChannel.clear();
    outputChannel.show();
    outputChannel.appendLine('=== Linting ABI Code ===\n');

    const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    if (!workspaceFolder) {
        vscode.window.showErrorMessage('No workspace folder found');
        return;
    }

    try {
        const result = await processRunner.run('zig', ['fmt', '--check', '.'], {
            cwd: workspaceFolder,
            showOutput: true
        });

        if (result.exitCode === 0) {
            vscode.window.showInformationMessage('Lint passed - no formatting issues');
            outputChannel.appendLine('\n=== Lint passed ===');
        } else {
            vscode.window.showWarningMessage('Lint found formatting issues. Run "ABI: Format Code" to fix.');
            outputChannel.appendLine('\n=== Lint found issues ===');
        }
    } catch (error) {
        vscode.window.showErrorMessage(`Lint error: ${error}`);
        outputChannel.appendLine(`\nError: ${error}`);
    }
}
