async function loadAndRender() {
  const res = await fetch('data/sample_benchmarks.json');
  const data = await res.json();
  const labels = data.map(d => d.date);
  const p50 = data.map(d => d.p50);

  const ctx = document.getElementById('latencyChart').getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'p50 latency (ms)',
        data: p50,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
      }]
    }
  });
}

loadAndRender().catch(err => console.error(err));
