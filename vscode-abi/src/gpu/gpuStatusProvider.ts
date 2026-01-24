import * as vscode from 'vscode';
import { GpuMonitor, GpuStatus } from './gpuMonitor';

export interface GpuItem {
    label: string;
    description?: string;
    tooltip?: string;
    children?: GpuItem[];
    iconPath?: vscode.ThemeIcon;
}

export class GpuStatusProvider implements vscode.TreeDataProvider<GpuItem> {
    private _onDidChangeTreeData = new vscode.EventEmitter<GpuItem | undefined | void>();
    readonly onDidChangeTreeData = this._onDidChangeTreeData.event;
    private status: GpuStatus | null = null;

    constructor(private gpuMonitor: GpuMonitor) {
        this.gpuMonitor.onStatusChange((status) => {
            this.status = status;
            this.refresh();
        });

        this.loadStatus();
    }

    private async loadStatus(): Promise<void> {
        try {
            this.status = await this.gpuMonitor.getStatus();
            this.refresh();
        } catch (error) {
            console.error('Failed to load GPU status:', error);
        }
    }

    refresh(): void {
        this._onDidChangeTreeData.fire();
    }

    getTreeItem(element: GpuItem): vscode.TreeItem {
        const treeItem = new vscode.TreeItem(
            element.label,
            element.children ? vscode.TreeItemCollapsibleState.Collapsed : vscode.TreeItemCollapsibleState.None
        );
        treeItem.description = element.description;
        treeItem.tooltip = element.tooltip;
        treeItem.iconPath = element.iconPath;
        return treeItem;
    }

    getChildren(element?: GpuItem): GpuItem[] {
        if (!this.status) {
            return [{
                label: 'Loading GPU status...',
                iconPath: new vscode.ThemeIcon('loading~spin')
            }];
        }

        if (!element) {
            return [
                {
                    label: 'Backend',
                    description: this.status.backend || 'None',
                    iconPath: new vscode.ThemeIcon('server-process'),
                    tooltip: `Active GPU backend: ${this.status.backend}`
                },
                {
                    label: 'Devices',
                    description: `${this.status.devices?.length ?? 0} device(s)`,
                    iconPath: new vscode.ThemeIcon('device-desktop'),
                    children: this.getDeviceItems()
                }
            ];
        }

        return element.children ?? [];
    }

    private getDeviceItems(): GpuItem[] {
        if (!this.status?.devices || this.status.devices.length === 0) {
            return [{
                label: 'No devices found',
                iconPath: new vscode.ThemeIcon('warning')
            }];
        }

        return this.status.devices.map((device, index) => ({
            label: device.name || `Device ${index}`,
            iconPath: new vscode.ThemeIcon('circuit-board'),
            children: [
                {
                    label: 'Memory',
                    description: this.formatMemory(device.memory?.used, device.memory?.total),
                    iconPath: new vscode.ThemeIcon('database'),
                    tooltip: `Memory usage: ${device.memory?.used ?? 0} / ${device.memory?.total ?? 0} MB`
                }
            ]
        }));
    }

    private formatMemory(used?: number, total?: number): string {
        if (used === undefined || total === undefined) {
            return 'N/A';
        }
        const usedGB = (used / 1024).toFixed(2);
        const totalGB = (total / 1024).toFixed(2);
        const percent = total > 0 ? ((used / total) * 100).toFixed(1) : '0';
        return `${usedGB} / ${totalGB} GB (${percent}%)`;
    }

    showDetails(item: GpuItem): void {
        vscode.window.showInformationMessage(
            `${item.label}: ${item.description ?? 'No details available'}`,
            { modal: false }
        );
    }
}
