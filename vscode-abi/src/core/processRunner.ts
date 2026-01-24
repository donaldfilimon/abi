import { spawn } from 'child_process';
import { OutputChannelManager } from './outputChannel';

export interface RunOptions {
    cwd?: string;
    env?: NodeJS.ProcessEnv;
    timeout?: number;
    showOutput?: boolean;
}

export interface ProcessResult {
    stdout: string;
    stderr: string;
    exitCode: number;
}

export class ProcessRunner {
    constructor(private outputChannel: OutputChannelManager) {}

    async run(command: string, args: string[], options: RunOptions = {}): Promise<ProcessResult> {
        return new Promise((resolve, reject) => {
            const proc = spawn(command, args, {
                cwd: options.cwd,
                env: { ...process.env, ...options.env },
                shell: process.platform === 'win32'
            });

            let stdout = '';
            let stderr = '';

            proc.stdout.on('data', (data) => {
                const text = data.toString();
                stdout += text;
                if (options.showOutput) {
                    this.outputChannel.append(text);
                }
            });

            proc.stderr.on('data', (data) => {
                const text = data.toString();
                stderr += text;
                if (options.showOutput) {
                    this.outputChannel.append(text);
                }
            });

            proc.on('error', (error) => {
                reject(new Error(`Failed to spawn process: ${error.message}`));
            });

            proc.on('close', (code) => {
                resolve({
                    stdout,
                    stderr,
                    exitCode: code ?? -1
                });
            });

            if (options.timeout) {
                setTimeout(() => {
                    proc.kill();
                    reject(new Error(`Process timed out after ${options.timeout}ms`));
                }, options.timeout);
            }
        });
    }

    async runStreaming(
        command: string,
        args: string[],
        onData: (data: string) => void,
        options: RunOptions = {}
    ): Promise<void> {
        return new Promise((resolve, reject) => {
            const proc = spawn(command, args, {
                cwd: options.cwd,
                env: { ...process.env, ...options.env },
                shell: process.platform === 'win32'
            });

            proc.stdout.on('data', (data) => {
                onData(data.toString());
            });

            proc.stderr.on('data', (data) => {
                onData(data.toString());
            });

            proc.on('error', (error) => {
                reject(error);
            });

            proc.on('close', (code) => {
                if (code === 0) {
                    resolve();
                } else {
                    reject(new Error(`Process exited with code ${code}`));
                }
            });
        });
    }
}
