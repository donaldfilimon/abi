import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
  let disposable = vscode.commands.registerCommand('abi.helloWorld', () => {
    vscode.window.showInformationMessage('ABI extension: Hello World');
  });
  context.subscriptions.push(disposable);
}

export function deactivate() {}
