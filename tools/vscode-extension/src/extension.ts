import * as vscode from 'vscode';
import * as path from 'path';
import * as cp from 'child_process';

// Status bar item for ABI
let statusBarItem: vscode.StatusBarItem;
let outputChannel: vscode.OutputChannel;

/**
 * Activates the ABI extension
 */
export function activate(context: vscode.ExtensionContext): void {
    outputChannel = vscode.window.createOutputChannel('ABI');
    context.subscriptions.push(outputChannel);

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('abi.build', buildProject),
        vscode.commands.registerCommand('abi.test', runTests),
        vscode.commands.registerCommand('abi.testFile', testCurrentFile),
        vscode.commands.registerCommand('abi.run', runApplication),
        vscode.commands.registerCommand('abi.runWithArgs', runWithArguments),
        vscode.commands.registerCommand('abi.format', formatCode),
        vscode.commands.registerCommand('abi.listFeatures', listFeatures),
        vscode.commands.registerCommand('abi.showDocs', showDocumentation),
        vscode.commands.registerCommand('abi.generateDocs', generateDocs)
    );

    // Register task provider
    context.subscriptions.push(
        vscode.tasks.registerTaskProvider('abi', new ABITaskProvider())
    );

    // Create status bar item
    const config = vscode.workspace.getConfiguration('abi');
    if (config.get('showStatusBar', true)) {
        statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Left,
            100
        );
        statusBarItem.text = '$(circuit-board) ABI';
        statusBarItem.tooltip = 'ABI Framework - Click to build';
        statusBarItem.command = 'abi.build';
        statusBarItem.show();
        context.subscriptions.push(statusBarItem);
    }

    outputChannel.appendLine('ABI Development Tools extension activated');
}

/**
 * Deactivates the extension
 */
export function deactivate(): void {
    if (statusBarItem) {
        statusBarItem.dispose();
    }
}

/**
 * Gets the Zig executable path from configuration
 */
function getZigPath(): string {
    const config = vscode.workspace.getConfiguration('abi');
    return config.get('zigPath', 'zig');
}

/**
 * Gets the workspace root folder
 */
function getWorkspaceRoot(): string | undefined {
    const folders = vscode.workspace.workspaceFolders;
    if (folders && folders.length > 0) {
        return folders[0].uri.fsPath;
    }
    return undefined;
}

/**
 * Builds feature flags from configuration
 */
function getFeatureFlags(): string[] {
    const config = vscode.workspace.getConfiguration('abi');
    const flags: string[] = [];

    if (config.get('enableGpu', true)) {
        flags.push('-Denable-gpu=true');
    } else {
        flags.push('-Denable-gpu=false');
    }

    if (config.get('enableAi', true)) {
        flags.push('-Denable-ai=true');
    } else {
        flags.push('-Denable-ai=false');
    }

    if (config.get('enableDatabase', true)) {
        flags.push('-Denable-database=true');
    } else {
        flags.push('-Denable-database=false');
    }

    const gpuBackend = config.get('gpuBackend', 'auto');
    if (gpuBackend !== 'auto') {
        flags.push(`-Dgpu-backend=${gpuBackend}`);
    }

    const optimize = config.get('optimize', 'Debug');
    flags.push(`-Doptimize=${optimize}`);

    return flags;
}

/**
 * Executes a Zig command and shows output
 */
async function executeZigCommand(
    args: string[],
    title: string
): Promise<boolean> {
    const workspaceRoot = getWorkspaceRoot();
    if (!workspaceRoot) {
        vscode.window.showErrorMessage('No workspace folder open');
        return false;
    }

    const zigPath = getZigPath();
    const command = `${zigPath} ${args.join(' ')}`;

    outputChannel.clear();
    outputChannel.appendLine(`> ${command}`);
    outputChannel.appendLine('');
    outputChannel.show(true);

    if (statusBarItem) {
        statusBarItem.text = `$(sync~spin) ABI: ${title}...`;
    }

    return new Promise((resolve) => {
        const process = cp.spawn(zigPath, args, {
            cwd: workspaceRoot,
            shell: true
        });

        process.stdout?.on('data', (data: Buffer) => {
            outputChannel.append(data.toString());
        });

        process.stderr?.on('data', (data: Buffer) => {
            outputChannel.append(data.toString());
        });

        process.on('close', (code) => {
            if (statusBarItem) {
                statusBarItem.text = '$(circuit-board) ABI';
            }

            if (code === 0) {
                outputChannel.appendLine('');
                outputChannel.appendLine(`${title} completed successfully`);
                vscode.window.showInformationMessage(`ABI: ${title} completed`);
                resolve(true);
            } else {
                outputChannel.appendLine('');
                outputChannel.appendLine(`${title} failed with code ${code}`);
                vscode.window.showErrorMessage(`ABI: ${title} failed`);
                resolve(false);
            }
        });

        process.on('error', (err) => {
            if (statusBarItem) {
                statusBarItem.text = '$(circuit-board) ABI';
            }
            outputChannel.appendLine(`Error: ${err.message}`);
            vscode.window.showErrorMessage(`ABI: Failed to execute - ${err.message}`);
            resolve(false);
        });
    });
}

/**
 * Build the project
 */
async function buildProject(): Promise<void> {
    const flags = getFeatureFlags();
    await executeZigCommand(['build', ...flags], 'Build');
}

/**
 * Run all tests
 */
async function runTests(): Promise<void> {
    const flags = getFeatureFlags();
    await executeZigCommand(['build', 'test', '--summary', 'all', ...flags], 'Test');
}

/**
 * Test the current file
 */
async function testCurrentFile(): Promise<void> {
    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active editor');
        return;
    }

    const filePath = editor.document.uri.fsPath;
    if (!filePath.endsWith('.zig')) {
        vscode.window.showErrorMessage('Current file is not a Zig file');
        return;
    }

    const workspaceRoot = getWorkspaceRoot();
    if (!workspaceRoot) {
        vscode.window.showErrorMessage('No workspace folder open');
        return;
    }

    // Get relative path for display
    const relativePath = path.relative(workspaceRoot, filePath);

    await executeZigCommand(['test', filePath], `Test ${relativePath}`);
}

/**
 * Run the application
 */
async function runApplication(): Promise<void> {
    const flags = getFeatureFlags();
    await executeZigCommand(['build', 'run', ...flags], 'Run');
}

/**
 * Run with custom arguments
 */
async function runWithArguments(): Promise<void> {
    const args = await vscode.window.showInputBox({
        prompt: 'Enter command-line arguments',
        placeHolder: 'e.g., db stats, agent --persona architect'
    });

    if (args === undefined) {
        return; // User cancelled
    }

    const flags = getFeatureFlags();
    const runArgs = args ? ['--', ...args.split(' ')] : [];
    await executeZigCommand(['build', 'run', ...flags, ...runArgs], 'Run');
}

/**
 * Format Zig code
 */
async function formatCode(): Promise<void> {
    const editor = vscode.window.activeTextEditor;
    if (editor && editor.document.languageId === 'zig') {
        // Format current file
        await executeZigCommand(['fmt', editor.document.uri.fsPath], 'Format');
        // Reload the document
        await vscode.commands.executeCommand('workbench.action.files.revert');
    } else {
        // Format entire project
        await executeZigCommand(['fmt', '.'], 'Format All');
    }
}

/**
 * List available features
 */
async function listFeatures(): Promise<void> {
    const flags = getFeatureFlags();
    await executeZigCommand(['build', 'run', ...flags, '--', '--list-features'], 'List Features');
}

/**
 * Show documentation
 */
async function showDocumentation(): Promise<void> {
    const workspaceRoot = getWorkspaceRoot();
    if (!workspaceRoot) {
        vscode.window.showErrorMessage('No workspace folder open');
        return;
    }

    const readmePath = path.join(workspaceRoot, 'README.md');
    const claudePath = path.join(workspaceRoot, 'CLAUDE.md');

    const pick = await vscode.window.showQuickPick([
        { label: 'README.md', description: 'Project overview', path: readmePath },
        { label: 'CLAUDE.md', description: 'Development guide', path: claudePath },
        { label: 'API Reference', description: 'Generated API docs', path: path.join(workspaceRoot, 'docs', 'api', 'index.md') },
        { label: 'Troubleshooting', description: 'Common issues', path: path.join(workspaceRoot, 'docs', 'troubleshooting.md') }
    ], {
        placeHolder: 'Select documentation to view'
    });

    if (pick) {
        const doc = await vscode.workspace.openTextDocument(pick.path);
        await vscode.window.showTextDocument(doc);
    }
}

/**
 * Generate API documentation
 */
async function generateDocs(): Promise<void> {
    const flags = getFeatureFlags();
    await executeZigCommand(['build', 'gendocs', ...flags], 'Generate Docs');
}

/**
 * Task provider for ABI tasks
 */
class ABITaskProvider implements vscode.TaskProvider {
    private tasks: vscode.Task[] | undefined;

    provideTasks(): vscode.Task[] {
        if (!this.tasks) {
            this.tasks = this.getTasks();
        }
        return this.tasks;
    }

    resolveTask(task: vscode.Task): vscode.Task | undefined {
        const definition = task.definition as ABITaskDefinition;
        if (definition.task) {
            return this.getTask(definition);
        }
        return undefined;
    }

    private getTasks(): vscode.Task[] {
        const tasks: vscode.Task[] = [];
        const workspaceRoot = getWorkspaceRoot();

        if (!workspaceRoot) {
            return tasks;
        }

        // Build task
        tasks.push(this.createTask({ type: 'abi', task: 'build' }, 'Build', ['build']));

        // Test task
        tasks.push(this.createTask({ type: 'abi', task: 'test' }, 'Test', ['build', 'test', '--summary', 'all']));

        // Run task
        tasks.push(this.createTask({ type: 'abi', task: 'run' }, 'Run', ['build', 'run']));

        // Format task
        tasks.push(this.createTask({ type: 'abi', task: 'format' }, 'Format', ['fmt', '.']));

        // Generate docs task
        tasks.push(this.createTask({ type: 'abi', task: 'docs' }, 'Generate Docs', ['build', 'gendocs']));

        return tasks;
    }

    private getTask(definition: ABITaskDefinition): vscode.Task {
        const args: string[] = [];

        switch (definition.task) {
            case 'build':
                args.push('build', ...getFeatureFlags());
                break;
            case 'test':
                args.push('build', 'test', '--summary', 'all', ...getFeatureFlags());
                break;
            case 'run':
                args.push('build', 'run', ...getFeatureFlags());
                break;
            case 'format':
                args.push('fmt', definition.file || '.');
                break;
            case 'docs':
                args.push('build', 'gendocs', ...getFeatureFlags());
                break;
        }

        if (definition.args) {
            args.push(...definition.args);
        }

        return this.createTask(definition, definition.task, args);
    }

    private createTask(
        definition: ABITaskDefinition,
        name: string,
        args: string[]
    ): vscode.Task {
        const workspaceRoot = getWorkspaceRoot();
        const zigPath = getZigPath();

        const execution = new vscode.ShellExecution(zigPath, args, {
            cwd: workspaceRoot
        });

        const task = new vscode.Task(
            definition,
            vscode.TaskScope.Workspace,
            name,
            'ABI',
            execution,
            '$zig'
        );

        task.group = definition.task === 'build'
            ? vscode.TaskGroup.Build
            : definition.task === 'test'
                ? vscode.TaskGroup.Test
                : undefined;

        return task;
    }
}

interface ABITaskDefinition extends vscode.TaskDefinition {
    task: string;
    file?: string;
    args?: string[];
}
