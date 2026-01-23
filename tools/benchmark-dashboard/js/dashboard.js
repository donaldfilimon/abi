/**
 * ABI Benchmark Dashboard
 * Interactive benchmark visualization and comparison tool
 */

(function() {
    'use strict';

    // Dashboard state
    const state = {
        benchmarks: [],
        filteredBenchmarks: [],
        charts: {},
        currentPage: 1,
        itemsPerPage: 20,
        sortColumn: 'date',
        sortDirection: 'desc',
        theme: localStorage.getItem('dashboard-theme') || 'light'
    };

    // Color palette for charts
    const chartColors = {
        primary: 'rgba(59, 130, 246, 1)',
        primaryLight: 'rgba(59, 130, 246, 0.2)',
        secondary: 'rgba(16, 185, 129, 1)',
        secondaryLight: 'rgba(16, 185, 129, 0.2)',
        warning: 'rgba(245, 158, 11, 1)',
        warningLight: 'rgba(245, 158, 11, 0.2)',
        danger: 'rgba(239, 68, 68, 1)',
        dangerLight: 'rgba(239, 68, 68, 0.2)',
        purple: 'rgba(139, 92, 246, 1)',
        purpleLight: 'rgba(139, 92, 246, 0.2)',
        cyan: 'rgba(6, 182, 212, 1)',
        cyanLight: 'rgba(6, 182, 212, 0.2)'
    };

    const versionColors = [
        { bg: chartColors.primaryLight, border: chartColors.primary },
        { bg: chartColors.secondaryLight, border: chartColors.secondary },
        { bg: chartColors.warningLight, border: chartColors.warning },
        { bg: chartColors.purpleLight, border: chartColors.purple },
        { bg: chartColors.cyanLight, border: chartColors.cyan },
        { bg: chartColors.dangerLight, border: chartColors.danger }
    ];

    // Initialize dashboard
    async function init() {
        applyTheme(state.theme);
        setupEventListeners();
        await loadBenchmarkData();
        updateDashboard();
        updateGeneratedTime();
    }

    // Load benchmark data from JSON file
    async function loadBenchmarkData() {
        try {
            const response = await fetch('data/benchmarks.json');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            state.benchmarks = data.benchmarks || [];
            state.systemInfo = data.systemInfo || {};
            state.metadata = data.metadata || {};

            // Update version badge
            if (state.metadata.version) {
                document.getElementById('current-version').textContent = state.metadata.version;
            }

            populateVersionFilters();
            applyFilters();
        } catch (error) {
            console.error('Failed to load benchmark data:', error);
            showError('Failed to load benchmark data. Please ensure data/benchmarks.json exists.');
        }
    }

    // Populate version filter dropdowns
    function populateVersionFilters() {
        const versions = [...new Set(state.benchmarks.map(b => b.version))].sort().reverse();

        const versionFilter = document.getElementById('version-filter');
        const compareVersions = document.getElementById('compare-versions');

        // Clear existing options (except "All")
        versionFilter.innerHTML = '<option value="all">All Versions</option>';
        compareVersions.innerHTML = '';

        versions.forEach(version => {
            versionFilter.innerHTML += `<option value="${version}">${version}</option>`;
            compareVersions.innerHTML += `<option value="${version}">${version}</option>`;
        });
    }

    // Apply filters to benchmark data
    function applyFilters() {
        const typeFilter = document.getElementById('benchmark-type').value;
        const versionFilter = document.getElementById('version-filter').value;
        const dateRange = document.getElementById('date-range').value;
        const searchTerm = document.getElementById('search-input').value.toLowerCase();

        let filtered = [...state.benchmarks];

        // Filter by type
        if (typeFilter !== 'all') {
            filtered = filtered.filter(b => b.type === typeFilter);
        }

        // Filter by version
        if (versionFilter !== 'all') {
            filtered = filtered.filter(b => b.version === versionFilter);
        }

        // Filter by date range
        if (dateRange !== 'all') {
            const days = parseInt(dateRange);
            const cutoffDate = new Date();
            cutoffDate.setDate(cutoffDate.getDate() - days);
            filtered = filtered.filter(b => new Date(b.date) >= cutoffDate);
        }

        // Filter by search term
        if (searchTerm) {
            filtered = filtered.filter(b =>
                b.name.toLowerCase().includes(searchTerm) ||
                b.type.toLowerCase().includes(searchTerm) ||
                b.version.toLowerCase().includes(searchTerm)
            );
        }

        state.filteredBenchmarks = filtered;
        state.currentPage = 1;
    }

    // Update entire dashboard
    function updateDashboard() {
        updateStats();
        updateCharts();
        updateTable();
        updateSystemInfo();
    }

    // Update statistics cards
    function updateStats() {
        const benchmarks = state.filteredBenchmarks;

        document.getElementById('total-benchmarks').textContent = benchmarks.length;

        if (benchmarks.length > 0) {
            const avgTime = benchmarks.reduce((sum, b) => sum + (b.metrics.time || 0), 0) / benchmarks.length;
            const avgMemory = benchmarks.reduce((sum, b) => sum + (b.metrics.memory || 0), 0) / benchmarks.length;
            const avgThroughput = benchmarks.reduce((sum, b) => sum + (b.metrics.throughput || 0), 0) / benchmarks.length;

            document.getElementById('avg-time').textContent = formatNumber(avgTime, 2) + ' ms';
            document.getElementById('avg-memory').textContent = formatNumber(avgMemory, 2) + ' MB';
            document.getElementById('avg-throughput').textContent = formatNumber(avgThroughput, 0) + ' ops/s';
        } else {
            document.getElementById('avg-time').textContent = '0 ms';
            document.getElementById('avg-memory').textContent = '0 MB';
            document.getElementById('avg-throughput').textContent = '0 ops/s';
        }
    }

    // Update all charts
    function updateCharts() {
        updateTimeChart();
        updateMemoryChart();
        updateThroughputChart();
        updateComparisonChart();
    }

    // Get chart options based on theme
    function getChartOptions(title, yAxisLabel) {
        const isDark = state.theme === 'dark';
        const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
        const textColor = isDark ? '#b0b0c5' : '#666666';

        return {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    labels: {
                        color: textColor
                    }
                },
                title: {
                    display: false
                }
            },
            scales: {
                x: {
                    grid: {
                        color: gridColor
                    },
                    ticks: {
                        color: textColor
                    }
                },
                y: {
                    grid: {
                        color: gridColor
                    },
                    ticks: {
                        color: textColor
                    },
                    title: {
                        display: true,
                        text: yAxisLabel,
                        color: textColor
                    }
                }
            }
        };
    }

    // Update time trend chart
    function updateTimeChart() {
        const ctx = document.getElementById('time-chart');
        const data = prepareTimeSeriesData('time');

        if (state.charts.time) {
            state.charts.time.destroy();
        }

        state.charts.time = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: data.datasets.map((ds, i) => ({
                    ...ds,
                    borderColor: versionColors[i % versionColors.length].border,
                    backgroundColor: versionColors[i % versionColors.length].bg,
                    fill: true,
                    tension: 0.3
                }))
            },
            options: getChartOptions('Execution Time', 'Time (ms)')
        });
    }

    // Update memory trend chart
    function updateMemoryChart() {
        const ctx = document.getElementById('memory-chart');
        const data = prepareTimeSeriesData('memory');

        if (state.charts.memory) {
            state.charts.memory.destroy();
        }

        state.charts.memory = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: data.datasets.map((ds, i) => ({
                    ...ds,
                    borderColor: versionColors[i % versionColors.length].border,
                    backgroundColor: versionColors[i % versionColors.length].bg,
                    fill: true,
                    tension: 0.3
                }))
            },
            options: getChartOptions('Memory Usage', 'Memory (MB)')
        });
    }

    // Update throughput trend chart
    function updateThroughputChart() {
        const ctx = document.getElementById('throughput-chart');
        const data = prepareTimeSeriesData('throughput');

        if (state.charts.throughput) {
            state.charts.throughput.destroy();
        }

        state.charts.throughput = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.labels,
                datasets: data.datasets.map((ds, i) => ({
                    ...ds,
                    backgroundColor: versionColors[i % versionColors.length].border,
                    borderColor: versionColors[i % versionColors.length].border,
                    borderWidth: 1
                }))
            },
            options: getChartOptions('Throughput', 'Operations/second')
        });
    }

    // Update version comparison chart
    function updateComparisonChart() {
        const ctx = document.getElementById('comparison-chart');
        const selectedVersions = Array.from(document.getElementById('compare-versions').selectedOptions)
            .map(opt => opt.value);

        if (state.charts.comparison) {
            state.charts.comparison.destroy();
        }

        // If no versions selected, compare all unique versions
        const versionsToCompare = selectedVersions.length > 0
            ? selectedVersions
            : [...new Set(state.filteredBenchmarks.map(b => b.version))].slice(0, 5);

        const benchmarkNames = [...new Set(state.filteredBenchmarks.map(b => b.name))].slice(0, 10);

        const datasets = versionsToCompare.map((version, i) => {
            const versionData = benchmarkNames.map(name => {
                const benchmark = state.filteredBenchmarks.find(b => b.version === version && b.name === name);
                return benchmark ? benchmark.metrics.time : null;
            });

            return {
                label: version,
                data: versionData,
                backgroundColor: versionColors[i % versionColors.length].border,
                borderColor: versionColors[i % versionColors.length].border,
                borderWidth: 1
            };
        });

        state.charts.comparison = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: benchmarkNames,
                datasets: datasets
            },
            options: {
                ...getChartOptions('Version Comparison', 'Time (ms)'),
                indexAxis: 'y'
            }
        });
    }

    // Prepare time series data for charts
    function prepareTimeSeriesData(metric) {
        const versions = [...new Set(state.filteredBenchmarks.map(b => b.version))];
        const dates = [...new Set(state.filteredBenchmarks.map(b => b.date))].sort();

        const datasets = versions.map(version => {
            const data = dates.map(date => {
                const benchmarks = state.filteredBenchmarks.filter(b => b.version === version && b.date === date);
                if (benchmarks.length === 0) return null;
                return benchmarks.reduce((sum, b) => sum + (b.metrics[metric] || 0), 0) / benchmarks.length;
            });

            return {
                label: version,
                data: data
            };
        });

        return {
            labels: dates.map(d => formatDate(d)),
            datasets: datasets
        };
    }

    // Update benchmark table
    function updateTable() {
        const tbody = document.getElementById('benchmark-tbody');
        const sorted = sortBenchmarks([...state.filteredBenchmarks]);

        const startIndex = (state.currentPage - 1) * state.itemsPerPage;
        const endIndex = startIndex + state.itemsPerPage;
        const pageData = sorted.slice(startIndex, endIndex);

        tbody.innerHTML = pageData.map(b => `
            <tr>
                <td>${escapeHtml(b.name)}</td>
                <td>${escapeHtml(b.type)}</td>
                <td>${escapeHtml(b.version)}</td>
                <td>${formatDate(b.date)}</td>
                <td>${formatNumber(b.metrics.time, 2)}</td>
                <td>${formatNumber(b.metrics.memory, 2)}</td>
                <td>${formatNumber(b.metrics.throughput, 0)}</td>
                <td><span class="status-badge ${b.status}">${b.status}</span></td>
            </tr>
        `).join('');

        updatePagination(sorted.length);
        updateSortIndicators();
    }

    // Sort benchmarks
    function sortBenchmarks(benchmarks) {
        return benchmarks.sort((a, b) => {
            let aVal, bVal;

            switch (state.sortColumn) {
                case 'name':
                case 'type':
                case 'version':
                case 'status':
                    aVal = a[state.sortColumn].toLowerCase();
                    bVal = b[state.sortColumn].toLowerCase();
                    break;
                case 'date':
                    aVal = new Date(a.date);
                    bVal = new Date(b.date);
                    break;
                case 'time':
                    aVal = a.metrics.time;
                    bVal = b.metrics.time;
                    break;
                case 'memory':
                    aVal = a.metrics.memory;
                    bVal = b.metrics.memory;
                    break;
                case 'throughput':
                    aVal = a.metrics.throughput;
                    bVal = b.metrics.throughput;
                    break;
                default:
                    return 0;
            }

            if (aVal < bVal) return state.sortDirection === 'asc' ? -1 : 1;
            if (aVal > bVal) return state.sortDirection === 'asc' ? 1 : -1;
            return 0;
        });
    }

    // Update sort indicators in table headers
    function updateSortIndicators() {
        document.querySelectorAll('.benchmark-table th').forEach(th => {
            const indicator = th.querySelector('.sort-indicator');
            indicator.className = 'sort-indicator';
            if (th.dataset.sort === state.sortColumn) {
                indicator.classList.add(state.sortDirection);
            }
        });
    }

    // Update pagination controls
    function updatePagination(totalItems) {
        const pagination = document.getElementById('pagination');
        const totalPages = Math.ceil(totalItems / state.itemsPerPage);

        if (totalPages <= 1) {
            pagination.innerHTML = '';
            return;
        }

        let html = '';

        // Previous button
        html += `<button ${state.currentPage === 1 ? 'disabled' : ''} data-page="${state.currentPage - 1}">Prev</button>`;

        // Page numbers
        const startPage = Math.max(1, state.currentPage - 2);
        const endPage = Math.min(totalPages, state.currentPage + 2);

        if (startPage > 1) {
            html += `<button data-page="1">1</button>`;
            if (startPage > 2) html += `<span>...</span>`;
        }

        for (let i = startPage; i <= endPage; i++) {
            html += `<button class="${i === state.currentPage ? 'active' : ''}" data-page="${i}">${i}</button>`;
        }

        if (endPage < totalPages) {
            if (endPage < totalPages - 1) html += `<span>...</span>`;
            html += `<button data-page="${totalPages}">${totalPages}</button>`;
        }

        // Next button
        html += `<button ${state.currentPage === totalPages ? 'disabled' : ''} data-page="${state.currentPage + 1}">Next</button>`;

        pagination.innerHTML = html;

        // Add click handlers
        pagination.querySelectorAll('button').forEach(btn => {
            btn.addEventListener('click', () => {
                const page = parseInt(btn.dataset.page);
                if (page && page !== state.currentPage) {
                    state.currentPage = page;
                    updateTable();
                }
            });
        });
    }

    // Update system info section
    function updateSystemInfo() {
        const container = document.getElementById('system-info');
        const info = state.systemInfo || {};

        const items = [
            { label: 'OS', value: info.os || 'N/A' },
            { label: 'Architecture', value: info.arch || 'N/A' },
            { label: 'CPU', value: info.cpu || 'N/A' },
            { label: 'CPU Cores', value: info.cpuCores || 'N/A' },
            { label: 'Memory', value: info.memory || 'N/A' },
            { label: 'Zig Version', value: info.zigVersion || 'N/A' },
            { label: 'GPU', value: info.gpu || 'N/A' },
            { label: 'Build Mode', value: info.buildMode || 'N/A' }
        ];

        container.innerHTML = items.map(item => `
            <div class="system-info-item">
                <label>${item.label}</label>
                <span>${escapeHtml(item.value)}</span>
            </div>
        `).join('');
    }

    // Setup event listeners
    function setupEventListeners() {
        // Theme toggle
        document.getElementById('theme-toggle').addEventListener('click', () => {
            state.theme = state.theme === 'light' ? 'dark' : 'light';
            localStorage.setItem('dashboard-theme', state.theme);
            applyTheme(state.theme);
            updateCharts();
        });

        // Filters
        ['benchmark-type', 'version-filter', 'date-range'].forEach(id => {
            document.getElementById(id).addEventListener('change', () => {
                applyFilters();
                updateDashboard();
            });
        });

        // Version comparison
        document.getElementById('compare-versions').addEventListener('change', () => {
            updateComparisonChart();
        });

        // Search
        document.getElementById('search-input').addEventListener('input', debounce(() => {
            applyFilters();
            updateDashboard();
        }, 300));

        // Refresh
        document.getElementById('refresh-btn').addEventListener('click', async () => {
            await loadBenchmarkData();
            updateDashboard();
        });

        // Table sorting
        document.querySelectorAll('.benchmark-table th[data-sort]').forEach(th => {
            th.addEventListener('click', () => {
                const column = th.dataset.sort;
                if (state.sortColumn === column) {
                    state.sortDirection = state.sortDirection === 'asc' ? 'desc' : 'asc';
                } else {
                    state.sortColumn = column;
                    state.sortDirection = 'asc';
                }
                updateTable();
            });
        });

        // Export modal
        document.getElementById('export-btn').addEventListener('click', () => {
            document.getElementById('export-modal').classList.add('active');
        });

        document.getElementById('modal-close').addEventListener('click', closeModal);
        document.getElementById('cancel-export').addEventListener('click', closeModal);

        document.getElementById('confirm-export').addEventListener('click', () => {
            const format = document.querySelector('input[name="export-format"]:checked').value;
            const useFiltered = document.getElementById('export-filtered').checked;
            exportData(format, useFiltered);
            closeModal();
        });

        // Close modal on backdrop click
        document.getElementById('export-modal').addEventListener('click', (e) => {
            if (e.target.classList.contains('modal')) {
                closeModal();
            }
        });
    }

    // Apply theme
    function applyTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
    }

    // Close modal
    function closeModal() {
        document.getElementById('export-modal').classList.remove('active');
    }

    // Export data
    function exportData(format, useFiltered) {
        const data = useFiltered ? state.filteredBenchmarks : state.benchmarks;
        let content, filename, mimeType;

        switch (format) {
            case 'json':
                content = JSON.stringify({
                    benchmarks: data,
                    systemInfo: state.systemInfo,
                    metadata: state.metadata,
                    exportedAt: new Date().toISOString()
                }, null, 2);
                filename = 'benchmarks-export.json';
                mimeType = 'application/json';
                break;

            case 'csv':
                const headers = ['Name', 'Type', 'Version', 'Date', 'Time (ms)', 'Memory (MB)', 'Throughput (ops/s)', 'Status'];
                const rows = data.map(b => [
                    b.name,
                    b.type,
                    b.version,
                    b.date,
                    b.metrics.time,
                    b.metrics.memory,
                    b.metrics.throughput,
                    b.status
                ]);
                content = [headers, ...rows].map(row => row.map(cell => `"${cell}"`).join(',')).join('\n');
                filename = 'benchmarks-export.csv';
                mimeType = 'text/csv';
                break;

            case 'markdown':
                content = generateMarkdownReport(data);
                filename = 'benchmarks-report.md';
                mimeType = 'text/markdown';
                break;
        }

        downloadFile(content, filename, mimeType);
    }

    // Generate markdown report
    function generateMarkdownReport(data) {
        const lines = [
            '# ABI Benchmark Report',
            '',
            `Generated: ${new Date().toISOString()}`,
            '',
            '## Summary',
            '',
            `- Total Benchmarks: ${data.length}`,
            `- Passed: ${data.filter(b => b.status === 'passed').length}`,
            `- Failed: ${data.filter(b => b.status === 'failed').length}`,
            '',
            '## System Information',
            ''
        ];

        if (state.systemInfo) {
            Object.entries(state.systemInfo).forEach(([key, value]) => {
                lines.push(`- **${key}**: ${value}`);
            });
        }

        lines.push('', '## Benchmark Results', '', '| Name | Type | Version | Time (ms) | Memory (MB) | Throughput | Status |');
        lines.push('|------|------|---------|-----------|-------------|------------|--------|');

        data.forEach(b => {
            lines.push(`| ${b.name} | ${b.type} | ${b.version} | ${b.metrics.time.toFixed(2)} | ${b.metrics.memory.toFixed(2)} | ${b.metrics.throughput.toFixed(0)} | ${b.status} |`);
        });

        return lines.join('\n');
    }

    // Download file
    function downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // Update generated time
    function updateGeneratedTime() {
        document.getElementById('generated-time').textContent = new Date().toLocaleString();
    }

    // Show error message
    function showError(message) {
        const tbody = document.getElementById('benchmark-tbody');
        tbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--accent-danger);">${escapeHtml(message)}</td></tr>`;
    }

    // Utility functions
    function formatNumber(num, decimals = 0) {
        if (num == null || isNaN(num)) return '0';
        return num.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
    }

    function formatDate(dateStr) {
        const date = new Date(dateStr);
        return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
    }

    function escapeHtml(str) {
        if (!str) return '';
        const div = document.createElement('div');
        div.textContent = str;
        return div.innerHTML;
    }

    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
