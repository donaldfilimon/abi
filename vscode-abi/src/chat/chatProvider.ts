import * as vscode from 'vscode';
import { BinaryManager } from '../core/binaryManager';
import { spawn } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

export interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
    timestamp: number;
}

export class ChatViewProvider implements vscode.WebviewViewProvider {
    private view?: vscode.WebviewView;
    private history: ChatMessage[] = [];
    private historyFile: string;

    constructor(
        private readonly extensionUri: vscode.Uri,
        private readonly binaryManager: BinaryManager
    ) {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
        this.historyFile = path.join(workspaceFolder ?? '', '.vscode', 'abi-chat-history.json');
        this.loadHistoryFromDisk();
    }

    resolveWebviewView(
        webviewView: vscode.WebviewView,
        _context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken
    ): void {
        this.view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [
                vscode.Uri.joinPath(this.extensionUri, 'media'),
                vscode.Uri.joinPath(this.extensionUri, 'dist')
            ]
        };

        webviewView.webview.html = this.getHtmlContent();

        webviewView.webview.onDidReceiveMessage(async (data) => {
            switch (data.type) {
                case 'sendMessage':
                    await this.handleUserMessage(data.message);
                    break;
                case 'clearHistory':
                    this.clearHistory();
                    this.view?.webview.postMessage({ type: 'historyCleared' });
                    break;
            }
        });

        this.loadHistory();
    }

    private async handleUserMessage(message: string): Promise<void> {
        if (!this.view) return;

        const userMessage: ChatMessage = { role: 'user', content: message, timestamp: Date.now() };
        this.history.push(userMessage);
        this.view.webview.postMessage({ type: 'addMessage', message: userMessage });
        this.view.webview.postMessage({ type: 'setTyping', isTyping: true });

        try {
            let binaryPath: string;
            try {
                binaryPath = this.binaryManager.getBinaryPath();
            } catch {
                // Fallback to mock response if binary not found
                const mockResponse = `[ABI Agent] I received your message: "${message.substring(0, 50)}..."`;
                const assistantMessage: ChatMessage = { role: 'assistant', content: mockResponse, timestamp: Date.now() };
                this.history.push(assistantMessage);
                this.view.webview.postMessage({ type: 'addMessage', message: assistantMessage });
                this.saveHistoryToDisk();
                return;
            }

            const model = vscode.workspace.getConfiguration('abi').get<string>('chat.model', 'gpt-oss');
            let response = '';

            await new Promise<void>((resolve, reject) => {
                const proc = spawn(binaryPath, ['agent', 'chat', '--message', message, '--model', model], {
                    shell: process.platform === 'win32'
                });

                proc.stdout.on('data', (data) => {
                    response += data.toString();
                    this.view?.webview.postMessage({ type: 'updateAssistantMessage', content: response });
                });

                proc.stderr.on('data', (data) => {
                    console.error('Agent stderr:', data.toString());
                });

                proc.on('close', (code) => {
                    if (code === 0) {
                        resolve();
                    } else {
                        reject(new Error(`Agent process exited with code ${code}`));
                    }
                });

                proc.on('error', reject);
            });

            const assistantMessage: ChatMessage = { role: 'assistant', content: response, timestamp: Date.now() };
            this.history.push(assistantMessage);
            this.view.webview.postMessage({ type: 'addMessage', message: assistantMessage });
            this.saveHistoryToDisk();
        } catch (error) {
            this.view.webview.postMessage({ type: 'error', message: `Error: ${error}` });
        } finally {
            this.view?.webview.postMessage({ type: 'setTyping', isTyping: false });
        }
    }

    private loadHistory(): void {
        this.view?.webview.postMessage({ type: 'loadHistory', messages: this.history });
    }

    clearHistory(): void {
        this.history = [];
        this.saveHistoryToDisk();
    }

    focus(): void {
        this.view?.show?.(true);
    }

    private loadHistoryFromDisk(): void {
        try {
            if (fs.existsSync(this.historyFile)) {
                const data = fs.readFileSync(this.historyFile, 'utf8');
                this.history = JSON.parse(data);
            }
        } catch (error) {
            console.error('Failed to load chat history:', error);
            this.history = [];
        }
    }

    private saveHistoryToDisk(): void {
        try {
            const dir = path.dirname(this.historyFile);
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
            }
            fs.writeFileSync(this.historyFile, JSON.stringify(this.history.slice(-100), null, 2));
        } catch (error) {
            console.error('Failed to save chat history:', error);
        }
    }

    private getHtmlContent(): string {
        // Using textContent for message rendering to prevent XSS
        return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ABI Chat</title>
    <style>
        body {
            padding: 10px;
            color: var(--vscode-foreground);
            font-family: var(--vscode-font-family);
            font-size: var(--vscode-font-size);
        }
        #chat-container { display: flex; flex-direction: column; height: 100vh; }
        #messages { flex: 1; overflow-y: auto; margin-bottom: 10px; padding: 10px; }
        .message { margin-bottom: 15px; padding: 10px; border-radius: 5px; }
        .message.user {
            background-color: var(--vscode-editor-selectionBackground);
            margin-left: 20px;
        }
        .message.assistant {
            background-color: var(--vscode-editor-inactiveSelectionBackground);
            margin-right: 20px;
        }
        .message-header { font-weight: bold; margin-bottom: 5px; font-size: 0.9em; opacity: 0.8; }
        #input-container { display: flex; gap: 5px; }
        #message-input {
            flex: 1;
            background: var(--vscode-input-background);
            color: var(--vscode-input-foreground);
            border: 1px solid var(--vscode-input-border);
            padding: 8px; border-radius: 3px;
        }
        button {
            background: var(--vscode-button-background);
            color: var(--vscode-button-foreground);
            border: none; padding: 8px 15px; cursor: pointer; border-radius: 3px;
        }
        button:hover { background: var(--vscode-button-hoverBackground); }
        .typing-indicator { font-style: italic; opacity: 0.7; padding: 10px; }
    </style>
</head>
<body>
    <div id="chat-container">
        <div id="messages"></div>
        <div class="typing-indicator" id="typing" style="display: none;">Assistant is typing...</div>
        <div id="input-container">
            <input type="text" id="message-input" placeholder="Ask the ABI agent..." />
            <button id="send-button">Send</button>
            <button id="clear-button">Clear</button>
        </div>
    </div>
    <script>
        const vscode = acquireVsCodeApi();
        const messagesDiv = document.getElementById('messages');
        const messageInput = document.getElementById('message-input');
        const typingIndicator = document.getElementById('typing');

        document.getElementById('send-button').addEventListener('click', sendMessage);
        document.getElementById('clear-button').addEventListener('click', clearMessages);
        messageInput.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });

        function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;
            vscode.postMessage({ type: 'sendMessage', message });
            messageInput.value = '';
        }

        function clearMessages() {
            vscode.postMessage({ type: 'clearHistory' });
        }

        function clearAllMessages() {
            // Remove all child nodes safely
            while (messagesDiv.firstChild) {
                messagesDiv.removeChild(messagesDiv.firstChild);
            }
        }

        function addMessage(msg) {
            const div = document.createElement('div');
            div.className = 'message ' + msg.role;
            const header = document.createElement('div');
            header.className = 'message-header';
            // Using textContent to prevent XSS
            header.textContent = msg.role === 'user' ? 'You' : 'ABI Agent';
            const content = document.createElement('div');
            content.textContent = msg.content;
            div.appendChild(header);
            div.appendChild(content);
            messagesDiv.appendChild(div);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        window.addEventListener('message', (event) => {
            const message = event.data;
            switch (message.type) {
                case 'addMessage':
                    addMessage(message.message);
                    break;
                case 'loadHistory':
                    clearAllMessages();
                    message.messages.forEach(addMessage);
                    break;
                case 'historyCleared':
                    clearAllMessages();
                    break;
                case 'setTyping':
                    typingIndicator.style.display = message.isTyping ? 'block' : 'none';
                    break;
                case 'updateAssistantMessage':
                    const last = messagesDiv.lastElementChild;
                    if (last && last.classList.contains('assistant') && last.lastElementChild) {
                        last.lastElementChild.textContent = message.content;
                    }
                    break;
            }
        });
    </script>
</body>
</html>`;
    }
}
