import * as vscode from 'vscode';

export class OutputChannelManager {
    private channel: vscode.OutputChannel;

    constructor() {
        this.channel = vscode.window.createOutputChannel('ABI Framework');
    }

    show(): void {
        this.channel.show();
    }

    append(text: string): void {
        this.channel.append(text);
    }

    appendLine(text: string): void {
        this.channel.appendLine(text);
    }

    clear(): void {
        this.channel.clear();
    }

    dispose(): void {
        this.channel.dispose();
    }
}
