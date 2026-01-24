import * as vscode from 'vscode';
import { BinaryManager } from '../core/binaryManager';
import { spawn } from 'child_process';

export interface GpuDevice {
    name: string;
    index: number;
    memory?: {
        used: number;
        total: number;
    };
}

export interface GpuStatus {
    backend: string;
    devices: GpuDevice[];
    timestamp: number;
}

export class GpuMonitor {
    private _onStatusChange = new vscode.EventEmitter<GpuStatus>();
    readonly onStatusChange = this._onStatusChange.event;
    private refreshInterval: NodeJS.Timeout | null = null;
    private lastStatus: GpuStatus | null = null;

    constructor(private binaryManager: BinaryManager) {}

    async getStatus(): Promise<GpuStatus> {
        try {
            const binaryPath = this.binaryManager.getBinaryPath();
            const output = await this.runCommand(binaryPath, ['gpu', 'summary', '--json']);
            const status = this.parseOutput(output);
            this.lastStatus = status;
            return status;
        } catch (error) {
            // Return fallback status if binary not available
            return {
                backend: 'None',
                devices: [],
                timestamp: Date.now()
            };
        }
    }

    private runCommand(command: string, args: string[]): Promise<string> {
        return new Promise((resolve, reject) => {
            const proc = spawn(command, args, {
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

            proc.on('close', (code) => {
                if (code === 0) {
                    resolve(stdout);
                } else {
                    reject(new Error(`Command failed with code ${code}: ${stderr}`));
                }
            });

            proc.on('error', reject);
        });
    }

    private parseOutput(output: string): GpuStatus {
        try {
            const data = JSON.parse(output);
            return {
                backend: data.backend || 'Unknown',
                devices: (data.devices || []).map((d: Record<string, unknown>, i: number) => ({
                    name: (d.name as string) || `Device ${i}`,
                    index: i,
                    memory: d.memory ? {
                        used: (d.memory as { used?: number }).used || 0,
                        total: (d.memory as { total?: number }).total || 0
                    } : undefined
                })),
                timestamp: Date.now()
            };
        } catch {
            // Parse non-JSON output (fallback for text output)
            const lines = output.split('\n');
            const backend = lines.find(l => l.includes('Backend:'))?.split(':')[1]?.trim() || 'Unknown';

            return {
                backend,
                devices: [],
                timestamp: Date.now()
            };
        }
    }

    startPolling(intervalMs?: number): void {
        const interval = intervalMs ??
            vscode.workspace.getConfiguration('abi').get<number>('gpu.refreshInterval', 5000);

        this.stopPolling();

        this.refreshInterval = setInterval(async () => {
            try {
                const status = await this.getStatus();
                this._onStatusChange.fire(status);
            } catch (error) {
                console.error('GPU status poll failed:', error);
            }
        }, interval);
    }

    stopPolling(): void {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }

    getLastStatus(): GpuStatus | null {
        return this.lastStatus;
    }

    dispose(): void {
        this.stopPolling();
        this._onStatusChange.dispose();
    }
}
