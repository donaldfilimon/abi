# ABI Benchmark Dashboard
> **Last reviewed:** 2026-01-31

An interactive web-based dashboard for visualizing and comparing ABI framework benchmark results.

## Features

- **Performance Charts**: Line and bar charts showing execution time, memory usage, and throughput trends
- **Version Comparison**: Compare benchmark metrics across different versions side-by-side
- **Filtering**: Filter benchmarks by type (runtime, memory, throughput, GPU, database, network), version, and date range
- **Search**: Full-text search across benchmark names and types
- **Sortable Table**: Click column headers to sort benchmark results
- **Dark/Light Mode**: Toggle between dark and light themes
- **Export**: Export data as JSON, CSV, or Markdown report
- **Responsive Design**: Works on desktop and mobile devices

## Quick Start

1. **Open the dashboard**:
   ```bash
   # Using Python's built-in server
   cd tools/benchmark-dashboard
   python -m http.server 8080
   # Then open http://localhost:8080 in your browser

   # Or using Node.js
   npx serve tools/benchmark-dashboard

   # Or simply open index.html directly in your browser
   ```

2. **View benchmarks**: The dashboard loads data from `data/benchmarks.json`

3. **Filter results**: Use the dropdown filters to narrow down results by type, version, or date range

4. **Compare versions**: Select multiple versions in the "Compare Versions" dropdown to see side-by-side comparisons

5. **Export data**: Click "Export Data" to download results in your preferred format

## Data Format

The benchmark data is stored in `data/benchmarks.json`. Here's the structure:

```json
{
  "metadata": {
    "version": "v0.16.0",
    "generatedAt": "2026-01-23T10:00:00Z",
    "description": "ABI Framework Benchmark Results"
  },
  "systemInfo": {
    "os": "Linux Ubuntu 22.04",
    "arch": "x86_64",
    "cpu": "AMD Ryzen 9 5950X",
    "cpuCores": "16",
    "memory": "64 GB DDR4-3600",
    "zigVersion": "0.16.0",
    "gpu": "NVIDIA RTX 4090",
    "buildMode": "ReleaseFast"
  },
  "benchmarks": [
    {
      "name": "vector_add_1m",
      "type": "gpu",
      "version": "v0.16.0",
      "date": "2026-01-23",
      "metrics": {
        "time": 2.45,
        "memory": 128.5,
        "throughput": 408163265
      },
      "status": "passed",
      "tags": ["gpu", "compute", "vulkan"]
    }
  ]
}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `name` | Unique benchmark identifier |
| `type` | Category: `runtime`, `memory`, `throughput`, `gpu`, `database`, `network` |
| `version` | ABI framework version (e.g., "v0.16.0") |
| `date` | ISO date string (YYYY-MM-DD) |
| `metrics.time` | Execution time in milliseconds |
| `metrics.memory` | Peak memory usage in megabytes |
| `metrics.throughput` | Operations per second |
| `status` | `passed`, `failed`, or `running` |
| `tags` | Optional array of tags for categorization |

## Adding New Benchmark Results

### Manual Entry

Edit `data/benchmarks.json` directly and add new benchmark entries to the `benchmarks` array.

### Automated Generation

Create a script to output benchmark results in the JSON format above. Example Zig code:

```zig
const std = @import("std");

pub fn outputBenchmarkResult(
    writer: anytype,
    name: []const u8,
    benchmark_type: []const u8,
    time_ms: f64,
    memory_mb: f64,
    throughput: u64,
) !void {
    try std.json.stringify(.{
        .name = name,
        .type = benchmark_type,
        .version = @import("abi").version(),
        .date = std.time.timestamp(), // Format appropriately
        .metrics = .{
            .time = time_ms,
            .memory = memory_mb,
            .throughput = throughput,
        },
        .status = "passed",
    }, .{}, writer);
}
```

### CI/CD Integration

Add benchmark runs to your CI pipeline and append results:

```yaml
# Example GitHub Actions step
- name: Run Benchmarks
  run: |
    zig build benchmarks
    ./zig-out/bin/benchmarks --json >> tools/benchmark-dashboard/data/benchmarks.json

- name: Deploy Dashboard
  uses: peaceiris/actions-gh-pages@v3
  with:
    publish_dir: ./tools/benchmark-dashboard
```

## Customization

### Adding New Benchmark Types

1. Add the new type to the filter dropdown in `index.html`:
   ```html
   <option value="custom">Custom</option>
   ```

2. Add corresponding color in `js/dashboard.js` if desired

### Modifying Charts

Edit the chart configuration in `js/dashboard.js`. The dashboard uses [Chart.js](https://www.chartjs.org/) for visualizations.

### Styling

Modify `css/style.css` to customize the appearance. The dashboard uses CSS variables for theming:

```css
:root {
    --bg-primary: #f5f7fa;
    --accent-primary: #3b82f6;
    /* ... */
}

[data-theme="dark"] {
    --bg-primary: #0f0f1a;
    /* ... */
}
```

## Browser Support

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Dependencies

- [Chart.js 4.4.1](https://www.chartjs.org/) (loaded via CDN)

## License

Part of the ABI Framework. See the repository root for license information.
