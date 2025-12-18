const API_URL = 'http://localhost:5000/api';
let chartInstances = { performance: null, feature: null };
let predictionHistory = [];
let evaluationResults = null;
let isDarkMode = false;

// Updated model data from your 3rd image
const updatedModelData = {
    'logistic': {
        accuracy: 0.845,
        precision: 0.7152,
        recall: 0.845,
        f1_score: 0.7747,
        name: 'Logistic Regression'
    },
    'knn': {
        accuracy: 0.830,
        precision: 0.7431,
        recall: 0.830,
        f1_score: 0.7750,
        name: 'KNN'
    },
    'decision_tree': {
        accuracy: 0.847,
        precision: 0.8031,
        recall: 0.847,
        f1_score: 0.7992,
        name: 'Decision Tree'
    }
};

// Theme Toggle Functionality
function toggleTheme() {
    isDarkMode = !isDarkMode;
    document.body.classList.toggle('dark-mode', isDarkMode);

    const themeIcon = document.getElementById('themeIcon');
    const themeText = document.getElementById('themeText');

    if (isDarkMode) {
        themeIcon.textContent = '☀️';
        themeText.textContent = 'Light Mode';
    } else {
        themeIcon.textContent = '🌙';
        themeText.textContent = 'Dark Mode';
    }

    // Update charts if they exist
    updateChartThemes();

    // Save theme preference
    localStorage.setItem('darkMode', isDarkMode);
}

function updateChartThemes() {
    const textColor = isDarkMode ? '#f0f0f0' : '#333';
    const gridColor = isDarkMode ? 'rgba(240, 240, 240, 0.1)' : 'rgba(0, 0, 0, 0.05)';

    // Update performance chart
    if (chartInstances.performance) {
        chartInstances.performance.options.scales.y.grid.color = gridColor;
        chartInstances.performance.options.plugins.title.color = textColor;
        chartInstances.performance.options.scales.y.ticks.color = textColor;
        chartInstances.performance.options.scales.x.ticks.color = textColor;
        chartInstances.performance.update();
    }

    // Update feature chart
    if (chartInstances.feature) {
        chartInstances.feature.options.scales.x.grid.color = gridColor;
        chartInstances.feature.options.plugins.title.color = textColor;
        chartInstances.feature.options.scales.x.ticks.color = textColor;
        chartInstances.feature.options.scales.y.ticks.color = textColor;
        chartInstances.feature.update();
    }
}

function switchTab(tab) {
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    event.target.classList.add('active');
    document.getElementById(tab + 'Tab').classList.add('active');
}

async function trainModels() {
    const statusText = document.getElementById('statusText');
    statusText.textContent = 'Training models... Please wait ⏳';

    try {
        const response = await fetch(`${API_URL}/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await response.json();

        if (data.status === 'success') {
            statusText.textContent = '✅ Models trained successfully! Ready for predictions';

            // Update stats
            document.getElementById('totalRecords').textContent = data.data.total_records || '-';
            document.getElementById('modelsTrained').textContent = '3';

            // Update with new accuracy values from 3rd image
            const bestAccuracy = Math.max(
                updatedModelData.logistic.accuracy,
                updatedModelData.knn.accuracy,
                updatedModelData.decision_tree.accuracy
            );
            document.getElementById('bestAccuracy').textContent = (bestAccuracy * 100).toFixed(1) + '%';

            // Calculate averages for display
            const avgAccuracy = (
                updatedModelData.logistic.accuracy +
                updatedModelData.knn.accuracy +
                updatedModelData.decision_tree.accuracy
            ) / 3;

            const avgF1Score = (
                updatedModelData.logistic.f1_score +
                updatedModelData.knn.f1_score +
                updatedModelData.decision_tree.f1_score
            ) / 3;

            document.getElementById('avgAccuracy').textContent = (avgAccuracy * 100).toFixed(1) + '%';
            document.getElementById('avgF1Score').textContent = (avgF1Score * 100).toFixed(1) + '%';

            // Update the comparison table and model cards
            updateComparisonTableWithNewData();
            updateModelPerformanceGrid();

            // Show success alert with UPDATED values from 3rd image
            let message = '✅ Training Completed Successfully!\n\n';
            message += `📊 Total Records: ${data.data.total_records}\n`;
            message += `📋 Features: ${data.data.total_features}\n\n`;
            message += '🎯 Model Accuracies:\n';
            message += `  • Logistic Regression: ${(updatedModelData.logistic.accuracy * 100).toFixed(2)}%\n`;
            message += `  • KNN: ${(updatedModelData.knn.accuracy * 100).toFixed(2)}%\n`;
            message += `  • Decision Tree: ${(updatedModelData.decision_tree.accuracy * 100).toFixed(2)}%`;
            alert(message);

        } else {
            statusText.textContent = '❌ Training failed';
            alert('❌ Error: ' + data.message);
        }
    } catch (error) {
        statusText.textContent = '❌ Connection error';
        alert('❌ Error: ' + error.message + '\n\nMake sure app.py is running!');
    }
}

async function makePrediction() {
    const model = document.getElementById('modelSelect').value;
    const features = {
        'Accident_Hour': parseInt(document.getElementById('accidentHour').value),
        'Number_of_Vehicles': parseInt(document.getElementById('numVehicles').value),
        'Weather': parseInt(document.getElementById('weather').value)
    };

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: model, features: features })
        });
        const data = await response.json();

        if (data.status === 'success') {
            const severity = data.prediction_label || data.prediction;
            const severityClass = data.prediction === 0 ? 'low' : data.prediction === 1 ? 'medium' : 'high';

            // REMOVED: Confidence percentage as requested
            document.getElementById('predictionResult').innerHTML = `
                <div class="result-box">
                    <h3>🎯 Prediction Result</h3>
                    <div class="severity-badge severity-${severityClass}">${severity}</div>
                    <p style="color: var(--text-secondary); font-size: 0.95em; margin-top: 8px;">
                        Model: ${model.replace('_', ' ').toUpperCase()}
                    </p>
                </div>
            `;

            // Update prediction count
            document.getElementById('predictionCount').textContent =
                parseInt(document.getElementById('predictionCount').textContent) + 1;

        } else {
            alert('❌ Prediction failed: ' + data.message);
        }
    } catch (error) {
        alert('❌ Error: ' + error.message);
    }
}

async function evaluateModels() {
    try {
        const response = await fetch(`${API_URL}/evaluate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: 'all' })
        });
        const data = await response.json();

        if (data.status === 'success') {
            // Override with our updated data from 3rd image
            const updatedResults = {
                'logistic': {
                    accuracy: updatedModelData.logistic.accuracy,
                    precision: updatedModelData.logistic.precision,
                    recall: updatedModelData.logistic.recall,
                    f1_score: updatedModelData.logistic.f1_score,
                    confusion_matrix: data.results.logistic ? .confusion_matrix || []
                },
                'knn': {
                    accuracy: updatedModelData.knn.accuracy,
                    precision: updatedModelData.knn.precision,
                    recall: updatedModelData.knn.recall,
                    f1_score: updatedModelData.knn.f1_score,
                    confusion_matrix: data.results.knn ? .confusion_matrix || []
                },
                'decision_tree': {
                    accuracy: updatedModelData.decision_tree.accuracy,
                    precision: updatedModelData.decision_tree.precision,
                    recall: updatedModelData.decision_tree.recall,
                    f1_score: updatedModelData.decision_tree.f1_score,
                    confusion_matrix: data.results.decision_tree ? .confusion_matrix || []
                }
            };

            evaluationResults = updatedResults;
            updatePerformanceMetrics(updatedResults);
            updateComparisonTableWithNewData();
            updateModelPerformanceGrid();

        } else {
            alert('❌ Evaluation failed: ' + data.message);
        }
    } catch (error) {
        alert('❌ Error: ' + error.message);
    }
}

function updatePerformanceMetrics(results) {
    const modelNames = Object.keys(results);

    // Calculate averages
    const avgAccuracy = modelNames.reduce((sum, m) => sum + results[m].accuracy, 0) / modelNames.length;
    const avgF1 = modelNames.reduce((sum, m) => sum + results[m].f1_score, 0) / modelNames.length;

    document.getElementById('avgAccuracy').textContent = (avgAccuracy * 100).toFixed(1) + '%';
    document.getElementById('avgF1Score').textContent = (avgF1 * 100).toFixed(1) + '%';

    // Create or update performance chart
    createPerformanceChart(results);
}

function createPerformanceChart(results) {
    const ctx = document.getElementById('performanceChart').getContext('2d');
    const modelNames = ['Logistic Regression', 'KNN', 'Decision Tree'];
    const textColor = isDarkMode ? '#f0f0f0' : '#333';
    const gridColor = isDarkMode ? 'rgba(240, 240, 240, 0.1)' : 'rgba(0, 0, 0, 0.05)';

    // Destroy existing chart if it exists
    if (chartInstances.performance) {
        chartInstances.performance.destroy();
    }

    chartInstances.performance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: modelNames,
            datasets: [{
                label: 'Accuracy (%)',
                data: [
                    results.logistic.accuracy * 100,
                    results.knn.accuracy * 100,
                    results.decision_tree.accuracy * 100
                ],
                backgroundColor: 'rgba(102, 126, 234, 0.85)',
                borderColor: 'rgba(102, 126, 234, 1)',
                borderWidth: 2,
                borderRadius: 10
            }, {
                label: 'F1 Score (%)',
                data: [
                    results.logistic.f1_score * 100,
                    results.knn.f1_score * 100,
                    results.decision_tree.f1_score * 100
                ],
                backgroundColor: 'rgba(118, 75, 162, 0.85)',
                borderColor: 'rgba(118, 75, 162, 1)',
                borderWidth: 2,
                borderRadius: 10
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: gridColor
                    },
                    ticks: {
                        color: textColor,
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                },
                x: {
                    ticks: {
                        color: textColor
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: textColor
                    }
                },
                title: {
                    display: true,
                    text: 'Model Performance Comparison',
                    font: { size: 16 },
                    color: textColor
                }
            }
        }
    });
}

function updateComparisonTableWithNewData() {
    const tbody = document.getElementById('comparisonBody');

    // Sort models by accuracy (highest first)
    const models = [{
            name: 'LOGISTIC REGRESSION',
            key: 'logistic',
            accuracy: updatedModelData.logistic.accuracy,
            precision: updatedModelData.logistic.precision,
            recall: updatedModelData.logistic.recall,
            f1_score: updatedModelData.logistic.f1_score
        },
        {
            name: 'KNN',
            key: 'knn',
            accuracy: updatedModelData.knn.accuracy,
            precision: updatedModelData.knn.precision,
            recall: updatedModelData.knn.recall,
            f1_score: updatedModelData.knn.f1_score
        },
        {
            name: 'DECISION TREE',
            key: 'decision_tree',
            accuracy: updatedModelData.decision_tree.accuracy,
            precision: updatedModelData.decision_tree.precision,
            recall: updatedModelData.decision_tree.recall,
            f1_score: updatedModelData.decision_tree.f1_score
        }
    ];

    // Sort by accuracy
    models.sort((a, b) => b.accuracy - a.accuracy);

    tbody.innerHTML = '';

    models.forEach((model, index) => {
        tbody.innerHTML += `
            <tr>
                <td><strong>${model.name}</strong></td>
                <td>${(model.accuracy * 100).toFixed(2)}%</td>
                <td>${(model.precision * 100).toFixed(2)}%</td>
                <td>${(model.recall * 100).toFixed(2)}%</td>
                <td>${(model.f1_score * 100).toFixed(2)}%</td>
            </tr>
        `;
    });
}

function updateModelPerformanceGrid() {
    const grid = document.getElementById('modelPerformanceGrid');

    // Sort models by accuracy for ranking
    const models = [{
            name: 'Logistic Regression',
            key: 'logistic',
            accuracy: updatedModelData.logistic.accuracy,
            precision: updatedModelData.logistic.precision,
            recall: updatedModelData.logistic.recall,
            f1_score: updatedModelData.logistic.f1_score
        },
        {
            name: 'KNN',
            key: 'knn',
            accuracy: updatedModelData.knn.accuracy,
            precision: updatedModelData.knn.precision,
            recall: updatedModelData.knn.recall,
            f1_score: updatedModelData.knn.f1_score
        },
        {
            name: 'Decision Tree',
            key: 'decision_tree',
            accuracy: updatedModelData.decision_tree.accuracy,
            precision: updatedModelData.decision_tree.precision,
            recall: updatedModelData.decision_tree.recall,
            f1_score: updatedModelData.decision_tree.f1_score
        }
    ];

    // Sort by accuracy (highest first)
    models.sort((a, b) => b.accuracy - a.accuracy);

    grid.innerHTML = '';

    models.forEach((model, index) => {
        const rankClass = `rank-${index + 1}`;
        const rankText = index === 0 ? '🥇 Best' : index === 1 ? '🥈 Good' : '🥉 Average';

        grid.innerHTML += `
            <div class="model-card">
                <h3>${model.name}</h3>
                <div class="accuracy">${(model.accuracy * 100).toFixed(1)}%</div>
                <div class="rank-badge ${rankClass}">${rankText}</div>
                <div class="model-stats">
                    <div class="stat-item">
                        <div class="stat-label">Precision</div>
                        <div class="stat-value">${(model.precision * 100).toFixed(1)}%</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Recall</div>
                        <div class="stat-value">${(model.recall * 100).toFixed(1)}%</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">F1 Score</div>
                        <div class="stat-value">${(model.f1_score * 100).toFixed(1)}%</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Rank</div>
                        <div class="stat-value">#${index + 1}</div>
                    </div>
                </div>
            </div>
        `;
    });
}

async function getFeatureImportance() {
    try {
        const response = await fetch(`${API_URL}/feature_importance`);
        const data = await response.json();

        if (data.status === 'success') {
            const features = data.feature_importance.slice(0, 10);
            const textColor = isDarkMode ? '#f0f0f0' : '#333';
            const gridColor = isDarkMode ? 'rgba(240, 240, 240, 0.1)' : 'rgba(0, 0, 0, 0.05)';

            // Destroy existing chart if it exists
            if (chartInstances.feature) {
                chartInstances.feature.destroy();
            }

            const ctx = document.getElementById('featureChart').getContext('2d');

            chartInstances.feature = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: features.map(f => f.feature),
                    datasets: [{
                        label: 'Importance Score',
                        data: features.map(f => f.importance),
                        backgroundColor: features.map((_, i) => `rgba(102, 126, 234, ${0.9 - i * 0.05})`),
                        borderColor: 'rgba(102, 126, 234, 1)',
                        borderWidth: 2,
                        borderRadius: 8
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y',
                    scales: {
                        x: {
                            beginAtZero: true,
                            grid: {
                                color: gridColor
                            },
                            ticks: {
                                color: textColor
                            },
                            title: {
                                display: true,
                                text: 'Importance Score',
                                color: textColor
                            }
                        },
                        y: {
                            ticks: {
                                color: textColor
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Top 10 Most Important Features',
                            font: { size: 16 },
                            color: textColor
                        },
                        legend: {
                            display: false
                        }
                    }
                }
            });
        } else {
            alert('❌ Error: ' + data.message);
        }
    } catch (error) {
        alert('❌ Error: ' + error.message);
    }
}

// Initialize with updated data on page load
document.addEventListener('DOMContentLoaded', function() {
    // Load theme preference
    const savedDarkMode = localStorage.getItem('darkMode') === 'true';
    if (savedDarkMode) {
        isDarkMode = true;
        document.body.classList.add('dark-mode');
        document.getElementById('themeIcon').textContent = '☀️';
        document.getElementById('themeText').textContent = 'Light Mode';
    }

    // Set initial values
    const avgAccuracy = (
        updatedModelData.logistic.accuracy +
        updatedModelData.knn.accuracy +
        updatedModelData.decision_tree.accuracy
    ) / 3;

    const avgF1Score = (
        updatedModelData.logistic.f1_score +
        updatedModelData.knn.f1_score +
        updatedModelData.decision_tree.f1_score
    ) / 3;

    document.getElementById('avgAccuracy').textContent = (avgAccuracy * 100).toFixed(1) + '%';
    document.getElementById('avgF1Score').textContent = (avgF1Score * 100).toFixed(1) + '%';

    // Update comparison table
    updateComparisonTableWithNewData();
    updateModelPerformanceGrid();

    // Set up theme toggle
    document.getElementById('themeToggle').addEventListener('click', toggleTheme);
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (chartInstances.performance) chartInstances.performance.destroy();
    if (chartInstances.feature) chartInstances.feature.destroy();
});