import * as vscode from 'vscode';
import { BinaryManager } from '../core/binaryManager';

interface AbiTaskDefinition extends vscode.TaskDefinition {
    command: string;
    args?: string[];
}

export class AbiTaskProvider implements vscode.TaskProvider {
    static readonly TaskType = 'abi';
    private tasks: vscode.Task[] | undefined;

    constructor(private binaryManager: BinaryManager) {}

    provideTasks(): vscode.Task[] {
        if (!this.tasks) {
            this.tasks = this.getDefaultTasks();
        }
        return this.tasks;
    }

    resolveTask(task: vscode.Task): vscode.Task | undefined {
        const definition = task.definition as AbiTaskDefinition;
        if (definition.command) {
            return this.createTask(definition);
        }
        return undefined;
    }

    private getDefaultTasks(): vscode.Task[] {
        return [
            this.createTask({ type: AbiTaskProvider.TaskType, command: 'build' }),
            this.createTask({ type: AbiTaskProvider.TaskType, command: 'test' }),
            this.createTask({ type: AbiTaskProvider.TaskType, command: 'format' }),
            this.createTask({ type: AbiTaskProvider.TaskType, command: 'lint' }),
            this.createTask({ type: AbiTaskProvider.TaskType, command: 'benchmarks' }),
            this.createTask({ type: AbiTaskProvider.TaskType, command: 'gendocs' }),
            this.createTask({
                type: AbiTaskProvider.TaskType,
                command: 'run',
                args: ['--', 'system-info']
            }),
            this.createTask({
                type: AbiTaskProvider.TaskType,
                command: 'run',
                args: ['--', 'gpu', 'summary']
            }),
            this.createTask({
                type: AbiTaskProvider.TaskType,
                command: 'run',
                args: ['--', 'db', 'stats']
            }),
        ];
    }

    private createTask(definition: AbiTaskDefinition): vscode.Task {
        const args = ['build', definition.command];
        if (definition.args) {
            args.push(...definition.args);
        }

        const buildFlags = vscode.workspace.getConfiguration('abi').get<string[]>('buildFlags', []);
        if (buildFlags.length > 0) {
            args.push(...buildFlags);
        }

        const execution = new vscode.ShellExecution('zig', args);
        const problemMatcher = definition.command === 'build' || definition.command === 'test'
            ? '$zig'
            : undefined;

        const taskName = this.getTaskName(definition);
        const task = new vscode.Task(
            definition,
            vscode.TaskScope.Workspace,
            taskName,
            'abi',
            execution,
            problemMatcher
        );

        task.group = this.getTaskGroup(definition.command);
        task.presentationOptions = {
            reveal: vscode.TaskRevealKind.Always,
            panel: vscode.TaskPanelKind.Shared
        };

        return task;
    }

    private getTaskName(definition: AbiTaskDefinition): string {
        if (definition.args && definition.args.length > 0) {
            const subCommand = definition.args.filter(a => !a.startsWith('-')).join(' ');
            return `ABI: ${definition.command} ${subCommand}`.trim();
        }
        return `ABI: ${definition.command}`;
    }

    private getTaskGroup(command: string): vscode.TaskGroup | undefined {
        switch (command) {
            case 'build':
                return vscode.TaskGroup.Build;
            case 'test':
                return vscode.TaskGroup.Test;
            default:
                return undefined;
        }
    }

    refresh(): void {
        this.tasks = undefined;
    }
}
