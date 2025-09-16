(function () {
  const trainButton = document.getElementById('train-button');
  const trainForm = document.getElementById('train-form');
  const trainStatus = document.getElementById('train-status');
  const metricsTable = document.getElementById('metrics-table');

  function formPayload() {
    const data = new FormData(trainForm);
    const payload = {};
    for (const [key, value] of data.entries()) {
      if (key === 'force_download') {
        payload[key] = true;
      } else if (['lookback_days', 'epochs'].includes(key)) {
        payload[key] = parseInt(value, 10);
      } else if (key === 'threshold') {
        payload[key] = parseFloat(value);
      } else {
        payload[key] = value;
      }
    }
    if (!payload.force_download) {
      payload.force_download = false;
    }
    payload.model = 'day_trading';
    return payload;
  }

  function setStatus(message, type = 'info') {
    trainStatus.className = `alert alert-${type}`;
    trainStatus.textContent = message;
    trainStatus.style.display = 'block';
  }

  function clearStatus() {
    trainStatus.style.display = 'none';
  }

  function updateMetricsTable(metrics) {
    const evaluation = metrics?.evaluation;
    if (!evaluation) {
      metricsTable.style.display = 'none';
      return;
    }
    metricsTable.style.display = '';
    const tbody = metricsTable.tBodies[0] || metricsTable.createTBody();
    tbody.innerHTML = '';
    Object.entries(evaluation).forEach(([key, value]) => {
      const row = tbody.insertRow();
      row.insertCell().textContent = key;
      const parsed = Number.parseFloat(value);
      row.insertCell().textContent = Number.isFinite(parsed) ? parsed.toFixed(4) : 'N/A';
    });
  }

  function buildHistoryConfig(initial) {
    const labels = initial.epochs || [];
    const datasets = Object.entries(initial.metrics || {}).map(([metric, values]) => ({
      label: metric,
      data: values,
      tension: 0.3,
      fill: false,
      borderWidth: 2,
    }));
    return { labels, datasets };
  }

  const historyCtx = document.getElementById('history-chart').getContext('2d');
  const initialHistory = window.__INITIAL_HISTORY__ || { epochs: [], metrics: {} };
  const historyConfig = buildHistoryConfig(initialHistory);
  const historyChart = new Chart(historyCtx, {
    type: 'line',
    data: {
      labels: historyConfig.labels,
      datasets: historyConfig.datasets,
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: 'bottom',
        },
      },
      scales: {
        x: {
          title: { display: true, text: 'Epoch' },
        },
        y: {
          beginAtZero: true,
        },
      },
    },
  });

  const priceCtx = document.getElementById('price-chart').getContext('2d');
  const probCtx = document.getElementById('prob-chart').getContext('2d');
  const priceChart = new Chart(priceCtx, {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Close price', data: [], borderColor: '#38bdf8', tension: 0.3 }] },
    options: {
      animation: false,
      scales: {
        x: { ticks: { display: false } },
        y: { title: { display: true, text: 'Price (USD)' } },
      },
    },
  });

  const probChart = new Chart(probCtx, {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Long probability', data: [], borderColor: '#facc15', tension: 0.3 }] },
    options: {
      animation: false,
      scales: {
        x: { title: { display: true, text: 'Time' } },
        y: { min: 0, max: 1, title: { display: true, text: 'Probability' } },
      },
    },
  });

  function updateHistoryChart(history) {
    const config = buildHistoryConfig(history);
    historyChart.data.labels = config.labels;
    historyChart.data.datasets = config.datasets;
    historyChart.update();
  }

  function historyFromPayload(history) {
    if (!Array.isArray(history) || history.length === 0) {
      return { epochs: [], metrics: {} };
    }
    const metricKeys = Object.keys(history[0]).filter((key) => key !== 'epoch');
    const metrics = Object.fromEntries(
      metricKeys.map((key) => [key, history.map((entry) => entry[key])])
    );
    return { epochs: history.map((entry) => entry.epoch), metrics };
  }

  function updateStreamCharts(stream) {
    if (!stream) return;
    priceChart.data.labels = stream.timestamps;
    priceChart.data.datasets[0].data = stream.prices;
    priceChart.update();

    probChart.data.labels = stream.timestamps;
    probChart.data.datasets[0].data = stream.probabilities;
    probChart.update();
  }

  async function pollStatus() {
    const response = await fetch('/day_trading/status');
    if (!response.ok) {
      return;
    }
    const payload = await response.json();
    if (payload.metrics) {
      updateMetricsTable(payload.metrics);
    }
    if (payload.history_plot) {
      updateHistoryChart(payload.history_plot);
    }
  }

  async function pollStream() {
    const response = await fetch('/day_trading/stream');
    if (!response.ok) {
      return;
    }
    const payload = await response.json();
    if (payload.stream) {
      updateStreamCharts(payload.stream);
    }
  }

  async function train() {
    trainButton.disabled = true;
    setStatus('Training in progress... this may take a couple of minutes depending on lookback size.');
    try {
      const response = await fetch('/day_trading/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formPayload()),
      });
      const payload = await response.json();
      if (!response.ok || payload.status !== 'success') {
        throw new Error(payload.message || 'Training failed');
      }
      updateMetricsTable({ evaluation: payload.evaluation });
      updateHistoryChart(historyFromPayload(payload.history));
      setStatus('Training completed successfully!', 'success');
      await pollStream();
    } catch (error) {
      console.error(error);
      setStatus(error.message, 'danger');
    } finally {
      trainButton.disabled = false;
      setTimeout(clearStatus, 5000);
    }
  }

  trainButton?.addEventListener('click', (event) => {
    event.preventDefault();
    train();
  });

  // Initialise tables and charts with data from server
  updateMetricsTable(window.__INITIAL_METRICS__);
  updateHistoryChart(initialHistory);
  pollStream();
  setInterval(pollStatus, 60000);
  setInterval(pollStream, 60000);
})();
