import * as vscode from 'vscode';
import { buildCommand, testCommand, formatCommand, lintCommand } from './commands';
import { ChatViewProvider } from './chat/chatProvider';
import { GpuStatusProvider } from './gpu/gpuStatusProvider';
import { GpuMonitor } from './gpu/gpuMonitor';
import { BinaryManager } from './core/binaryManager';
import { OutputChannelManager } from './core/outputChannel';
import { AbiTaskProvider } from './tasks/taskProvider';

export function activate(context: vscode.ExtensionContext) {
    console.log('ABI Framework extension activating...');

    // Initialize core services
    const binaryManager = new BinaryManager();
    const outputChannel = new OutputChannelManager();

    // Validate binary on activation
    binaryManager.validate().then(isValid => {
        if (!isValid) {
            vscode.window.showWarningMessage(
                'ABI binary not found. Build the project first.',
                'Build Now'
            ).then(selection => {
                if (selection === 'Build Now') {
                    vscode.commands.executeCommand('abi.build');
                }
            });
        }
    });

    // Register build/test commands
    context.subscriptions.push(
        vscode.commands.registerCommand('abi.build', () => buildCommand(outputChannel)),
        vscode.commands.registerCommand('abi.test', () => testCommand(outputChannel)),
        vscode.commands.registerCommand('abi.testFiltered', async () => {
            const filter = await vscode.window.showInputBox({
                prompt: 'Enter test filter pattern',
                placeHolder: 'e.g., "engine" or "database"'
            });
            if (filter) {
                await testCommand(outputChannel, { filter });
            }
        }),
        vscode.commands.registerCommand('abi.format', () => formatCommand(outputChannel)),
        vscode.commands.registerCommand('abi.lint', () => lintCommand(outputChannel))
    );

    // Register chat provider
    const chatProvider = new ChatViewProvider(context.extensionUri, binaryManager);
    context.subscriptions.push(
        vscode.window.registerWebviewViewProvider('abi.chatView', chatProvider, {
            webviewOptions: { retainContextWhenHidden: true }
        }),
        vscode.commands.registerCommand('abi.chat.send', () => {
            chatProvider.focus();
        }),
        vscode.commands.registerCommand('abi.chat.clear', () => {
            chatProvider.clearHistory();
        })
    );

    // Register GPU status provider
    const gpuMonitor = new GpuMonitor(binaryManager);
    const gpuStatusProvider = new GpuStatusProvider(gpuMonitor);
    context.subscriptions.push(
        vscode.window.registerTreeDataProvider('abi.gpuView', gpuStatusProvider),
        vscode.commands.registerCommand('abi.gpu.refresh', () => {
            gpuStatusProvider.refresh();
        }),
        vscode.commands.registerCommand('abi.gpu.showDetails', (item) => {
            gpuStatusProvider.showDetails(item);
        })
    );

    // Start GPU monitoring
    const refreshInterval = vscode.workspace.getConfiguration('abi').get<number>('gpu.refreshInterval', 5000);
    gpuMonitor.startPolling(refreshInterval);

    // Register task provider
    const taskProvider = new AbiTaskProvider(binaryManager);
    context.subscriptions.push(
        vscode.tasks.registerTaskProvider(AbiTaskProvider.TaskType, taskProvider)
    );

    // Clean up on deactivation
    context.subscriptions.push({
        dispose: () => gpuMonitor.dispose()
    });

    console.log('ABI Framework extension activated');
}

export function deactivate() {
    console.log('ABI Framework extension deactivating...');
}
