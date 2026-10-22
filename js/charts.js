/**
 * FedAgri Dynamic Charts Engine
 * Builds and updates real-time Chart.js visualizations
 */

const FedCharts = {
  convergenceChart: null,
  robustnessChart: null,
  paretoChart: null,
  partitionChart: null,

  commonOptions: {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 400 },
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#cbd5e1',
          font: { family: 'Inter', size: 12 },
          usePointStyle: true,
          boxWidth: 8
        }
      },
      tooltip: {
        backgroundColor: 'rgba(10, 20, 15, 0.95)',
        titleColor: '#f0fdf4',
        bodyColor: '#cbd5e1',
        borderColor: 'rgba(52, 211, 153, 0.3)',
        borderWidth: 1,
        padding: 10,
        cornerRadius: 8,
        displayColors: true
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.05)' },
        ticks: { color: '#94a3b8', font: { family: 'Inter', size: 11 } },
        title: { display: true, text: 'Federated Round', color: '#64748b', font: { size: 12 } }
      },
      y: {
        grid: { color: 'rgba(255, 255, 255, 0.05)' },
        ticks: { color: '#94a3b8', font: { family: 'Inter', size: 11 } },
        title: { display: true, text: 'Metric Value', color: '#64748b', font: { size: 12 } }
      }
    }
  },

  initConvergenceChart(canvasId = "chartConvergence") {
    const ctx = document.getElementById(canvasId);
    if (!ctx || typeof Chart === "undefined") return;

    if (this.convergenceChart) {
      this.convergenceChart.destroy();
    }

    this.convergenceChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Val Accuracy',
            data: [],
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            borderWidth: 2.5,
            fill: true,
            tension: 0.3,
            pointRadius: 3,
            pointHoverRadius: 6
          },
          {
            label: 'Val Macro-F1',
            data: [],
            borderColor: '#06b6d4',
            backgroundColor: 'transparent',
            borderWidth: 2,
            tension: 0.3,
            pointRadius: 3,
            pointHoverRadius: 6
          },
          {
            label: 'Train Loss',
            data: [],
            borderColor: '#f59e0b',
            backgroundColor: 'transparent',
            borderWidth: 2,
            borderDash: [4, 4],
            tension: 0.3,
            pointRadius: 2,
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        ...this.commonOptions,
        scales: {
          x: { ...this.commonOptions.scales.x },
          y: {
            ...this.commonOptions.scales.y,
            title: { display: true, text: 'Accuracy / F1 (0 - 1.0)', color: '#10b981' },
            min: 0,
            max: 1.05
          },
          y1: {
            position: 'right',
            grid: { drawOnChartArea: false },
            ticks: { color: '#f59e0b', font: { size: 10 } },
            title: { display: true, text: 'Training Loss', color: '#f59e0b' },
            min: 0
          }
        }
      }
    });
  },

  initRobustnessChart(canvasId = "chartRobustness") {
    const ctx = document.getElementById(canvasId);
    if (!ctx || typeof Chart === "undefined") return;

    if (this.robustnessChart) {
      this.robustnessChart.destroy();
    }

    this.robustnessChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          {
            label: 'Mean Client Acc',
            data: [],
            borderColor: '#34d399',
            borderWidth: 2,
            tension: 0.25,
            pointRadius: 3
          },
          {
            label: 'Worst Client Acc',
            data: [],
            borderColor: '#f43f5e',
            borderWidth: 2,
            tension: 0.25,
            pointRadius: 3
          },
          {
            label: 'Client Disparity (Std)',
            data: [],
            borderColor: '#a855f7',
            backgroundColor: 'rgba(168, 85, 247, 0.1)',
            borderWidth: 1.5,
            borderDash: [3, 3],
            fill: true,
            tension: 0.25,
            pointRadius: 2,
            yAxisID: 'y1'
          }
        ]
      },
      options: {
        ...this.commonOptions,
        scales: {
          x: { ...this.commonOptions.scales.x },
          y: {
            ...this.commonOptions.scales.y,
            title: { display: true, text: 'Client Accuracy (0 - 1.0)', color: '#34d399' },
            min: 0,
            max: 1.05
          },
          y1: {
            position: 'right',
            grid: { drawOnChartArea: false },
            ticks: { color: '#a855f7', font: { size: 10 } },
            title: { display: true, text: 'Std Dev (Client Drift)', color: '#a855f7' },
            min: 0
          }
        }
      }
    });
  },

  addRoundPoint(roundData) {
    const rnd = roundData.round;
    const valAcc = roundData.validation_accuracy;
    const valF1 = roundData.validation_macro_f1;
    const trainLoss = roundData.train_loss;

    const meanClient = roundData.mean_client_accuracy;
    const worstClient = roundData.worst_client_accuracy;
    const clientStd = roundData.client_accuracy_std;

    if (this.convergenceChart) {
      this.convergenceChart.data.labels.push(`R${rnd}`);
      this.convergenceChart.data.datasets[0].data.push(valAcc);
      this.convergenceChart.data.datasets[1].data.push(valF1);
      this.convergenceChart.data.datasets[2].data.push(trainLoss);
      this.convergenceChart.update();
    }

    if (this.robustnessChart) {
      this.robustnessChart.data.labels.push(`R${rnd}`);
      this.robustnessChart.data.datasets[0].data.push(meanClient);
      this.robustnessChart.data.datasets[1].data.push(worstClient);
      this.robustnessChart.data.datasets[2].data.push(clientStd);
      this.robustnessChart.update();
    }
  },

  resetCharts() {
    if (this.convergenceChart) {
      this.convergenceChart.data.labels = [];
      this.convergenceChart.data.datasets.forEach(ds => ds.data = []);
      this.convergenceChart.update();
    }
    if (this.robustnessChart) {
      this.robustnessChart.data.labels = [];
      this.robustnessChart.data.datasets.forEach(ds => ds.data = []);
      this.robustnessChart.update();
    }
  },

  loadFullHistory(historyList) {
    this.resetCharts();
    if (!historyList || !Array.isArray(historyList)) return;
    historyList.forEach(entry => this.addRoundPoint(entry));
  },

  renderParetoChart(canvasId, experiments) {
    const ctx = document.getElementById(canvasId);
    if (!ctx || typeof Chart === "undefined" || !experiments || experiments.length === 0) return;

    if (this.paretoChart) {
      this.paretoChart.destroy();
    }

    // Group into FedAvg and FedProx
    const fedavgPoints = [];
    const fedproxPoints = [];

    const sizeMap = {
      mobilenetv3: 5.94,
      resnet50: 89.97,
      efficientnet_b0: 15.47,
      convnext_tiny: 106.24
    };

    experiments.forEach(exp => {
      const mSize = sizeMap[exp.model.toLowerCase()] || 20;
      const rounds = exp.rounds_completed || 20;
      const commMB = Math.round(2 * 5 * rounds * mSize);
      const point = {
        x: commMB,
        y: (exp.val_macro_f1 || 0) * 100,
        model: exp.model,
        alpha: exp.alpha,
        f1: exp.val_macro_f1,
        comm: commMB
      };

      if (exp.algorithm.toLowerCase().includes("prox")) {
        fedproxPoints.push(point);
      } else {
        fedavgPoints.push(point);
      }
    });

    this.paretoChart = new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'FedProx (μ=0.01)',
            data: fedproxPoints,
            backgroundColor: '#10b981',
            borderColor: '#34d399',
            pointRadius: 6,
            pointHoverRadius: 8
          },
          {
            label: 'FedAvg',
            data: fedavgPoints,
            backgroundColor: '#06b6d4',
            borderColor: '#67e8f9',
            pointRadius: 6,
            pointHoverRadius: 8
          }
        ]
      },
      options: {
        ...this.commonOptions,
        plugins: {
          ...this.commonOptions.plugins,
          tooltip: {
            ...this.commonOptions.plugins.tooltip,
            callbacks: {
              label: (context) => {
                const pt = context.raw;
                return `${pt.model.toUpperCase()} (α=${pt.alpha}): ${pt.y.toFixed(1)}% F1 | Comm: ${pt.x.toLocaleString()} MB`;
              }
            }
          }
        },
        scales: {
          x: {
            ...this.commonOptions.scales.x,
            title: { display: true, text: 'Total Communication Volume (MB)', color: '#94a3b8' }
          },
          y: {
            ...this.commonOptions.scales.y,
            title: { display: true, text: 'Macro-F1 Score (%)', color: '#94a3b8' },
            min: 70,
            max: 102
          }
        }
      }
    });
  },

  renderPartitionChart(canvasId, clientCounts) {
    const ctx = document.getElementById(canvasId);
    if (!ctx || typeof Chart === "undefined" || !clientCounts) return;

    if (this.partitionChart) {
      this.partitionChart.destroy();
    }

    const labels = Object.keys(clientCounts).map(cid => `Client ${parseInt(cid) + 1}`);
    const data = Object.values(clientCounts);

    this.partitionChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Assigned Training Samples',
          data: data,
          backgroundColor: [
            'rgba(16, 185, 129, 0.75)',
            'rgba(6, 182, 212, 0.75)',
            'rgba(245, 158, 11, 0.75)',
            'rgba(168, 85, 247, 0.75)',
            'rgba(244, 63, 94, 0.75)'
          ],
          borderColor: [
            '#10b981', '#06b6d4', '#f59e0b', '#a855f7', '#f43f5e'
          ],
          borderWidth: 1.5,
          borderRadius: 6
        }]
      },
      options: {
        ...this.commonOptions,
        scales: {
          x: {
            ...this.commonOptions.scales.x,
            title: { display: true, text: 'Federated Client Node', color: '#94a3b8' }
          },
          y: {
            ...this.commonOptions.scales.y,
            title: { display: true, text: 'Number of Images', color: '#94a3b8' }
          }
        }
      }
    });
  }
};

window.FedCharts = FedCharts;
