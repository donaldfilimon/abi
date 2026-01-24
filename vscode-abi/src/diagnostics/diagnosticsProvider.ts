import * as vscode from 'vscode';
import { spawn } from 'child_process';
import { BinaryManager } from '../core/binaryManager';

interface DiagnosticMatch {
    file: string;
    line: number;
    column: number;
    message: string;
    severity: 'error' | 'warning' | 'info';
}

interface ProcessResult {
    stdout: string;
    stderr: string;
    exitCode: number;
}

/**
 * Provides diagnostics (errors, warnings) from Zig compiler output.
 */
export class DiagnosticsProvider {
    private diagnosticCollection: vscode.DiagnosticCollection;
    private debounceTimer: NodeJS.Timeout | undefined;
    private readonly debounceMs = 1000;

    constructor(_binaryManager: BinaryManager) {
        // binaryManager reserved for future use (custom zig path)
        this.diagnosticCollection = vscode.languages.createDiagnosticCollection('abi');
    }

    /**
     * Schedule a diagnostic check (debounced).
     */
    public scheduleDiagnostics(): void {
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }
        this.debounceTimer = setTimeout(() => {
            this.runDiagnostics();
        }, this.debounceMs);
    }

    /**
     * Run a command and capture output.
     */
    private runCommand(command: string, args: string[], cwd: string): Promise<ProcessResult> {
        return new Promise((resolve) => {
            const proc = spawn(command, args, {
                cwd,
                shell: process.platform === 'win32'
            });

            let stdout = '';
            let stderr = '';

            proc.stdout.on('data', (data) => {
                stdout += data.toString();
            });

            proc.stderr.on('data', (data) => {
                stderr += data.toString();
            });

            proc.on('error', () => {
                resolve({ stdout, stderr, exitCode: -1 });
            });

            proc.on('close', (code) => {
                resolve({
                    stdout,
                    stderr,
                    exitCode: code ?? -1
                });
            });

            // Timeout after 60 seconds
            setTimeout(() => {
                proc.kill();
                resolve({ stdout, stderr, exitCode: -1 });
            }, 60000);
        });
    }

    /**
     * Run diagnostics immediately.
     */
    public async runDiagnostics(): Promise<void> {
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (!workspaceFolder) {
            return;
        }

        // Clear existing diagnostics
        this.diagnosticCollection.clear();

        // Run zig build check (compile-only, no linking)
        const result = await this.runCommand('zig', ['build', 'typecheck'], workspaceFolder.uri.fsPath);

        // Parse stderr for diagnostics (both success and failure cases)
        const diagnostics = this.parseOutput(result.stderr);
        this.updateDiagnostics(workspaceFolder.uri.fsPath, diagnostics);
    }

    /**
     * Clear all diagnostics.
     */
    public clear(): void {
        this.diagnosticCollection.clear();
    }

    /**
     * Parse Zig compiler output for diagnostic information.
     */
    private parseOutput(output: string): DiagnosticMatch[] {
        const diagnostics: DiagnosticMatch[] = [];
        const lines = output.split('\n');

        // Zig error format: file.zig:line:column: error/warning/note: message
        const errorPattern = /^(.+\.zig):(\d+):(\d+):\s*(error|warning|note):\s*(.+)$/;

        for (const line of lines) {
            const match = line.match(errorPattern);
            if (match) {
                const [, file, lineNum, column, severity, message] = match;
                diagnostics.push({
                    file,
                    line: parseInt(lineNum, 10),
                    column: parseInt(column, 10),
                    message,
                    severity: this.mapSeverity(severity),
                });
            }
        }

        return diagnostics;
    }

    /**
     * Map Zig severity to our diagnostic severity.
     */
    private mapSeverity(severity: string): 'error' | 'warning' | 'info' {
        switch (severity.toLowerCase()) {
            case 'error':
                return 'error';
            case 'warning':
                return 'warning';
            case 'note':
            default:
                return 'info';
        }
    }

    /**
     * Convert severity string to VS Code DiagnosticSeverity.
     */
    private getSeverity(severity: 'error' | 'warning' | 'info'): vscode.DiagnosticSeverity {
        switch (severity) {
            case 'error':
                return vscode.DiagnosticSeverity.Error;
            case 'warning':
                return vscode.DiagnosticSeverity.Warning;
            case 'info':
            default:
                return vscode.DiagnosticSeverity.Information;
        }
    }

    /**
     * Update the diagnostic collection with parsed diagnostics.
     */
    private updateDiagnostics(workspaceRoot: string, matches: DiagnosticMatch[]): void {
        // Group diagnostics by file
        const fileMap = new Map<string, vscode.Diagnostic[]>();

        for (const match of matches) {
            const filePath = vscode.Uri.file(
                match.file.startsWith('/') || match.file.includes(':')
                    ? match.file
                    : `${workspaceRoot}/${match.file}`
            );

            const range = new vscode.Range(
                Math.max(0, match.line - 1),
                Math.max(0, match.column - 1),
                Math.max(0, match.line - 1),
                Number.MAX_SAFE_INTEGER
            );

            const diagnostic = new vscode.Diagnostic(
                range,
                match.message,
                this.getSeverity(match.severity)
            );
            diagnostic.source = 'abi';

            const key = filePath.toString();
            const existing = fileMap.get(key) || [];
            existing.push(diagnostic);
            fileMap.set(key, existing);
        }

        // Update the collection
        for (const [fileUri, diagnostics] of fileMap) {
            this.diagnosticCollection.set(vscode.Uri.parse(fileUri), diagnostics);
        }
    }

    /**
     * Dispose of resources.
     */
    public dispose(): void {
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }
        this.diagnosticCollection.dispose();
    }
}
