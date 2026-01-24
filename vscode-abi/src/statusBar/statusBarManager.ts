import * as vscode from 'vscode';
import { BinaryManager } from '../core/binaryManager';

export type BuildStatus = 'idle' | 'building' | 'success' | 'error';

/**
 * Manages the status bar item for ABI Framework.
 */
export class StatusBarManager {
    private statusBarItem: vscode.StatusBarItem;
    private buildStatus: BuildStatus = 'idle';
    private lastBuildTime: Date | undefined;

    constructor(_binaryManager: BinaryManager) {
        // binaryManager reserved for future use (binary validation)
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Left,
            100
        );
        this.statusBarItem.command = 'abi.showQuickPick';
        this.updateStatusBar();
        this.statusBarItem.show();
    }

    /**
     * Set the current build status.
     */
    public setStatus(status: BuildStatus): void {
        this.buildStatus = status;
        if (status === 'success' || status === 'error') {
            this.lastBuildTime = new Date();
        }
        this.updateStatusBar();
    }

    /**
     * Update the status bar item based on current state.
     */
    private updateStatusBar(): void {
        switch (this.buildStatus) {
            case 'idle':
                this.statusBarItem.text = '$(circuit-board) ABI';
                this.statusBarItem.tooltip = 'ABI Framework - Click for actions';
                this.statusBarItem.backgroundColor = undefined;
                break;
            case 'building':
                this.statusBarItem.text = '$(loading~spin) ABI Building...';
                this.statusBarItem.tooltip = 'Build in progress...';
                this.statusBarItem.backgroundColor = undefined;
                break;
            case 'success':
                this.statusBarItem.text = '$(check) ABI';
                this.statusBarItem.tooltip = this.formatTooltip('Build successful');
                this.statusBarItem.backgroundColor = undefined;
                break;
            case 'error':
                this.statusBarItem.text = '$(error) ABI';
                this.statusBarItem.tooltip = this.formatTooltip('Build failed - click for details');
                this.statusBarItem.backgroundColor = new vscode.ThemeColor(
                    'statusBarItem.errorBackground'
                );
                break;
        }
    }

    /**
     * Format tooltip with timestamp.
     */
    private formatTooltip(message: string): string {
        if (this.lastBuildTime) {
            const timeStr = this.lastBuildTime.toLocaleTimeString();
            return `${message} at ${timeStr}`;
        }
        return message;
    }

    /**
     * Show quick pick menu with ABI actions.
     */
    public async showQuickPick(): Promise<void> {
        const items: vscode.QuickPickItem[] = [
            {
                label: '$(tools) Build',
                description: 'zig build',
                detail: 'Build the project'
            },
            {
                label: '$(beaker) Test',
                description: 'zig build test',
                detail: 'Run all tests'
            },
            {
                label: '$(filter) Test (Filtered)',
                description: 'zig build test --test-filter',
                detail: 'Run tests with a filter pattern'
            },
            {
                label: '$(symbol-file) Format',
                description: 'zig fmt',
                detail: 'Format all Zig files'
            },
            {
                label: '$(checklist) Lint',
                description: 'zig build lint',
                detail: 'Check code formatting'
            },
            {
                label: '$(server-process) GPU Status',
                description: 'View GPU information',
                detail: 'Show GPU backends and device status'
            },
            {
                label: '$(comment-discussion) Chat',
                description: 'Open AI chat',
                detail: 'Chat with the ABI assistant'
            },
        ];

        const selected = await vscode.window.showQuickPick(items, {
            placeHolder: 'Select an ABI action',
            matchOnDescription: true,
            matchOnDetail: true,
        });

        if (selected) {
            switch (selected.label) {
                case '$(tools) Build':
                    vscode.commands.executeCommand('abi.build');
                    break;
                case '$(beaker) Test':
                    vscode.commands.executeCommand('abi.test');
                    break;
                case '$(filter) Test (Filtered)':
                    vscode.commands.executeCommand('abi.testFiltered');
                    break;
                case '$(symbol-file) Format':
                    vscode.commands.executeCommand('abi.format');
                    break;
                case '$(checklist) Lint':
                    vscode.commands.executeCommand('abi.lint');
                    break;
                case '$(server-process) GPU Status':
                    vscode.commands.executeCommand('abi.gpu.refresh');
                    break;
                case '$(comment-discussion) Chat':
                    vscode.commands.executeCommand('abi.chat.send');
                    break;
            }
        }
    }

    /**
     * Dispose of resources.
     */
    public dispose(): void {
        this.statusBarItem.dispose();
    }
}
