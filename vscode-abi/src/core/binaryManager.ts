import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export class BinaryManager {
    private cachedPath: string | null = null;

    getBinaryPath(): string {
        if (this.cachedPath) {
            return this.cachedPath;
        }

        // Check user configuration first
        const configPath = vscode.workspace.getConfiguration('abi').get<string>('binaryPath');
        if (configPath && fs.existsSync(configPath)) {
            this.cachedPath = configPath;
            return configPath;
        }

        // Auto-detect based on workspace and platform
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
        if (!workspaceFolder) {
            throw new Error('No workspace folder found');
        }

        const isWindows = process.platform === 'win32';
        const binaryName = isWindows ? 'abi.exe' : 'abi';
        const binaryPath = path.join(workspaceFolder, 'zig-out', 'bin', binaryName);

        if (fs.existsSync(binaryPath)) {
            this.cachedPath = binaryPath;
            return binaryPath;
        }

        throw new Error(`ABI binary not found at ${binaryPath}. Run "zig build" first.`);
    }

    async validate(): Promise<boolean> {
        try {
            const binaryPath = this.getBinaryPath();
            await execAsync(`"${binaryPath}" --version`);
            return true;
        } catch (error) {
            return false;
        }
    }

    async getVersion(): Promise<string> {
        const binaryPath = this.getBinaryPath();
        const { stdout } = await execAsync(`"${binaryPath}" --version`);
        return stdout.trim();
    }

    invalidateCache(): void {
        this.cachedPath = null;
    }
}
