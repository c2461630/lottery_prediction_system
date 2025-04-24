$(document).ready(function() {
    // 訓練模型
    $('#trainBtn').click(function() {
        $('#trainStatus').html('<div class="alert alert-info">正在訓練模型，請稍候...</div>');
        
        $.ajax({
            url: '/train',
            method: 'POST',
            contentType: 'application/json',
            success: function(response) {
                $('#trainStatus').html('<div class="alert alert-success">模型訓練成功！</div>');
                console.log(response);
            },
            error: function(error) {
                $('#trainStatus').html('<div class="alert alert-danger">模型訓練失敗：' + error.responseJSON.message + '</div>');
                console.error(error);
            }
        });
    });
    
    // 生成預測
    $('#predictBtn').click(function() {
        const modelName = $('#modelSelect').val();
        const numSets = $('#numSets').val();
        
        $('#predictions').html('<div class="col-12 text-center"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div></div>');
        
        $.ajax({
            url: '/predict',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                model_name: modelName,
                num_sets: parseInt(numSets)
            }),
            success: function(response) {
                $('#predictions').empty();
                
                response.predictions.forEach(function(predSet, setIndex) {
                    const setDiv = $('<div class="col-md-6 mb-3 prediction-set"></div>');
                    const card = $('<div class="card prediction-card"></div>');
                    const cardBody = $('<div class="card-body"></div>');
                    
                    cardBody.append('<h6>預測組 ' + (setIndex + 1) + '</h6>');
                    
                    predSet.forEach(function(numbers, rowIndex) {
                        const numbersDiv = $('<div class="mb-2 prediction-numbers"></div>');
                        
                        numbers.forEach(function(num) {
                            numbersDiv.append('<span class="number-ball">' + num + '</span>');
                        });
                        
                        cardBody.append(numbersDiv);
                    });
                    
                    card.append(cardBody);
                    setDiv.append(card);
                    $('#predictions').append(setDiv);
                });
            },
            error: function(error) {
                $('#predictions').html('<div class="col-12"><div class="alert alert-danger">預測失敗：' + error.responseJSON.message + '</div></div>');
                console.error(error);
            }
        });
    });
    
    // 使用最佳引數預測
    $('#predictWithBestParams').click(function() {
        const numSets = $('#numSets').val();
        
        $('#predictions').html('<div class="col-12 text-center"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div></div>');
        
        $.ajax({
            url: '/predict_with_best_params',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                num_sets: parseInt(numSets)
            }),
            success: function(response) {
                $('#predictions').empty();
                
                // 顯示使用的模型和命中率
                const modelInfo = $('<div class="col-12 mb-3"></div>');
                modelInfo.append('<div class="alert alert-info">使用模型: ' + response.model + ', 歷史命中率: ' + response.hit_rate + '%</div>');
                $('#predictions').append(modelInfo);
                
                response.predictions.forEach(function(predSet) {
                    const setDiv = $('<div class="col-md-6 mb-3 prediction-set"></div>');
                    const card = $('<div class="card prediction-card"></div>');
                    const cardBody = $('<div class="card-body"></div>');
                    
                    cardBody.append('<h6>預測組 ' + predSet.set_number + '</h6>');
                    
                    predSet.numbers.forEach(function(numbers) {
                        const numbersDiv = $('<div class="mb-2 prediction-numbers"></div>');
                        
                        numbers.forEach(function(num) {
                            numbersDiv.append('<span class="number-ball">' + num + '</span>');
                        });
                        
                        cardBody.append(numbersDiv);
                    });
                    
                    card.append(cardBody);
                    setDiv.append(card);
                    $('#predictions').append(setDiv);
                });
            },
            error: function(error) {
                $('#predictions').html('<div class="col-12"><div class="alert alert-danger">預測失敗：' + error.responseJSON.message + '</div></div>');
                console.error(error);
            }
        });
    });
    
    // 評估預測
    $('#evaluateBtn').click(function() {
        const actualNumbers = [];
        for (let i = 1; i <= 5; i++) {
            const num = parseInt($('#actualNum' + i).val());
            if (num >= 1 && num <= 49) {
                actualNumbers.push(num);
            }
        }
        
        if (actualNumbers.length !== 5) {
            $('#evaluationResults').html('<div class="alert alert-danger">請輸入5個有效的號碼（1-49）</div>');
            return;
        }
        
        // 獲取預測結果
        const predictions = [];
        $('.prediction-card').each(function() {
            const predSet = [];
            $(this).find('div.mb-2').each(function() {
                const numbers = [];
                $(this).find('.number-ball').each(function() {
                    numbers.push(parseInt($(this).text()));
                });
                predSet.push(numbers);
            });
            predictions.push(predSet);
        });
        
        if (predictions.length === 0) {
            $('#evaluationResults').html('<div class="alert alert-warning">請先生成預測結果</div>');
            return;
        }
        
        $('#evaluationResults').html('<div class="text-center"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div></div>');
        
        $.ajax({
            url: '/evaluate',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                predictions: predictions,
                actual_numbers: actualNumbers
            }),
            success: function(response) {
                let html = '<div class="row">';
                
                // 命中率結果
                html += '<div class="col-md-6"><h6>命中率結果</h6>';
                html += '<p>平均命中率: ' + (response.hit_results.avg_hit_rate * 100).toFixed(2) + '%</p>';
                html += '<p>命中3個或以上次數: ' + response.hit_results.hit_count_3_plus + '</p>';
                
                // 命中分佈
                html += '<h6>命中分佈</h6><ul>';
                for (const [hits, count] of Object.entries(response.hit_results.hit_distribution)) {
                    html += '<li>命中 ' + hits + ' 個: ' + count + ' 次</li>';
                }
                html += '</ul></div>';
                
                // 評估報告
                html += '<div class="col-md-6"><h6>評估報告</h6>';
                html += '<p>平均命中率: ' + (response.evaluation_report.avg_hit_rate * 100).toFixed(2) + '%</p>';
                html += '<p>命中3個或以上比例: ' + (response.evaluation_report.hit_count_3_plus_percentage * 100).toFixed(2) + '%</p>';
                html += '</div>';
                
                html += '</div>';
                
                $('#evaluationResults').html(html);
            },
            error: function(error) {
                $('#evaluationResults').html('<div class="alert alert-danger">評估失敗：' + error.responseJSON.message + '</div>');
                console.error(error);
            }
        });
    });
    
    // 最佳化引數
    $('#optimizeBtn').click(function() {
        const trials = parseInt($('#trials').val());
        
        // 獲取實際號碼
        const actualNumbers = [];
        for (let i = 1; i <= 5; i++) {
            const num = parseInt($('#actualNum' + i).val());
            if (num >= 1 && num <= 49) {
                actualNumbers.push(num);
            }
        }
        
        if (actualNumbers.length !== 5) {
            $('#evaluationResults').html('<div class="alert alert-danger">請輸入5個有效的號碼（1-49）</div>');
            return;
        }
        
        $('#evaluationResults').html('<div class="text-center"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div></div>');
        
        $.ajax({
            url: '/optimize',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                actual_numbers: actualNumbers,
                trials: trials
            }),
            success: function(response) {
                let html = '<div class="alert alert-success">';
                html += '<h6>最佳引數</h6>';
                html += '<p>最佳模型: ' + response.optimal_params.model_name + '</p>';
                html += '<p>命中率: ' + (response.optimal_params.hit_rate * 100).toFixed(2) + '%</p>';
                
                // 顯示最佳預測
                html += '<h6>最佳預測</h6>';
                response.optimal_params.predictions.forEach(function(predSet, setIndex) {
                    html += '<div class="mb-2">預測組 ' + (setIndex + 1) + ': ';
                    
                    predSet.forEach(function(numbers, rowIndex) {
                        html += '[';
                        numbers.forEach(function(num, numIndex) {
                            html += num;
                            if (numIndex < numbers.length - 1) {
                                html += ', ';
                            }
                        });
                        html += '] ';
                    });
                    
                    html += '</div>';
                });
                
                html += '</div>';
                
                $('#evaluationResults').html(html);
            },
            error: function(error) {
                $('#evaluationResults').html('<div class="alert alert-danger">最佳化失敗：' + error.responseJSON.message + '</div>');
                console.error(error);
            }
        });
    });
    
    // 尋找最佳參數按鈕點擊事件
    $('#startOptimizeBtn').click(function() {
        const modelName = $('#model_name').val();
        const nTrials = parseInt($('#n_trials').val());
        
        $('#optimizationProgress').html('<div class="alert alert-info">正在優化參數，請稍候...</div>');
        
        $.ajax({
            url: '/optimize',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                model_name: modelName,
                n_trials: nTrials
            }),
            success: function(response) {
                $('#optimizationProgress').html('<div class="alert alert-success">參數優化完成！</div>');
                
                let html = '<div class="card mt-3"><div class="card-header">最佳參數</div><div class="card-body">';
                html += `<p>模型: ${response.best_model}</p>`;
                html += `<p>最佳分數: ${response.best_score.toFixed(4)}</p>`;
                html += '<p>參數:</p><pre>' + JSON.stringify(response.best_params, null, 2) + '</pre>';
                html += '</div></div>';
                
                $('#optimizationResults').html(html);
            },
            error: function(error) {
                const errorMsg = error.responseJSON ? error.responseJSON.message : '未知錯誤';
                $('#optimizationProgress').html(`<div class="alert alert-danger">優化失敗: ${errorMsg}</div>`);
            }
        });
    });

    // 載入資料摘要
    $('#dataBtn').click(function() {
        $('#dataSummary').html('<div class="text-center"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div></div>');
        
        $.ajax({
            url: '/data',
            method: 'GET',
            success: function(response) {
                let html = '<div class="row">';
                
                // 基本統計資訊
                html += '<div class="col-md-6"><h6>基本統計資訊</h6>';
                html += '<p>總記錄數: ' + response.total_records + '</p>';
                
                html += '<table class="table table-sm table-striped">';
                html += '<thead><tr><th>號碼</th><th>最小值</th><th>最大值</th><th>平均值</th><th>中位數</th><th>標準差</th><th>最常見</th></tr></thead>';
                html += '<tbody>';
                
                for (const [col, stats] of Object.entries(response.summary)) {
                    html += '<tr>';
                    html += '<td>' + col + '</td>';
                    html += '<td>' + stats.min + '</td>';
                    html += '<td>' + stats.max + '</td>';
                    html += '<td>' + stats.mean.toFixed(2) + '</td>';
                    html += '<td>' + stats.median + '</td>';
                    html += '<td>' + stats.std.toFixed(2) + '</td>';
                    html += '<td>' + stats.most_frequent + '</td>';
                    html += '</tr>';
                }
                
                html += '</tbody></table></div>';
                
                // 最近記錄
                html += '<div class="col-md-6"><h6>最近記錄</h6>';
                html += '<table class="table table-sm table-striped">';
                html += '<thead><tr>';
                
                // 獲取列名
                const firstRecord = response.recent_records[0];
                for (const key in firstRecord) {
                    html += '<th>' + key + '</th>';
                }
                
                html += '</tr></thead><tbody>';
                
                // 新增記錄
                response.recent_records.forEach(function(record) {
                    html += '<tr>';
                    for (const key in record) {
                        html += '<td>' + record[key] + '</td>';
                    }
                    html += '</tr>';
                });
                
                html += '</tbody></table></div>';
                
                html += '</div>';
                
                $('#dataSummary').html(html);
            },
            error: function(error) {
                $('#dataSummary').html('<div class="alert alert-danger">載入資料失敗：' + error.responseJSON.message + '</div>');
                console.error(error);
            }
        });
    });
    
    // 高級分析功能
    $('#advancedAnalysisBtn').click(function() {
        $('#analysisResults').html('<div class="text-center"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div></div>');
        
        $.ajax({
            url: '/advanced_analysis',
            method: 'GET',
            success: function(response) {
                let html = '<div class="row">';
                
                // 相關性分析
                html += '<div class="col-md-6 mb-4"><div class="card"><div class="card-header">號碼相關性分析</div><div class="card-body">';
                html += '<h6>最強正相關</h6><ul>';
                response.correlations.strongest_positive.forEach(function(item) {
                    html += `<li>${item.pair}: ${item.correlation.toFixed(3)}</li>`;
                });
                html += '</ul>';
                
                html += '<h6>最強負相關</h6><ul>';
                response.correlations.strongest_negative.forEach(function(item) {
                    html += `<li>${item.pair}: ${item.correlation.toFixed(3)}</li>`;
                });
                html += '</ul></div></div></div>';
                // 週期性分析
                html += '<div class="col-md-6 mb-4"><div class="card"><div class="card-header">號碼週期性分析</div><div class="card-body">';
                html += '<h6>最強週期性號碼</h6><ul>';
                response.periodicity.forEach(function(item) {
                    html += `<li>號碼 ${item.number}: 變異係數 ${item.cv.toFixed(3)}</li>`;
                });
                html += '</ul></div></div></div>';
                
                // 趨勢分析
                html += '<div class="col-md-6 mb-4"><div class="card"><div class="card-header">號碼趨勢分析</div><div class="card-body">';
                html += '<h6>上升趨勢</h6><ul>';
                response.trends.rising.forEach(function(item) {
                    html += `<li>號碼 ${item.number}: +${item.trend.toFixed(3)}</li>`;
                });
                html += '</ul>';
                
                html += '<h6>下降趨勢</h6><ul>';
                response.trends.falling.forEach(function(item) {
                    html += `<li>號碼 ${item.number}: -${item.trend.toFixed(3)}</li>`;
                });
                html += '</ul></div></div></div>';
                
                html += '</div>'; // 結束 row
                
                $('#analysisResults').html(html);
            },
            error: function(error) {
                $('#analysisResults').html(`<div class="alert alert-danger">分析失敗: ${error.responseJSON?.message || '未知錯誤'}</div>`);
            }
        });
    });

    // 多樣性設置功能
    $('#diversitySettingsBtn').click(function() {
        $('#diversityModal').modal('show');
    });

    $('#saveDiversitySettings').click(function() {
        const diversityMethod = $('#diversityMethod').val();
        const diversityLevel = parseFloat($('#diversityLevel').val());
        
        // 保存設置到 localStorage
        localStorage.setItem('diversityMethod', diversityMethod);
        localStorage.setItem('diversityLevel', diversityLevel);
        
        $('#diversityModal').modal('hide');
        showAlert('多樣性設置已保存', 'success');
    });

    // 評估預測結果
    $('#evaluatePredictionsBtn').click(function() {
        const predictions = getPredictionsFromUI();
        const actualNumbers = getActualNumbersFromUI();
        
        if (!predictions.length || !actualNumbers.length) {
            showAlert('請先生成預測並輸入實際開獎號碼', 'warning');
            return;
        }
        
        $('#evaluationResults').html('<div class="text-center"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div></div>');
        
        $.ajax({
            url: '/evaluate',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                predictions: predictions,
                actual_numbers: actualNumbers
            }),
            success: function(response) {
                let html = '<div class="card"><div class="card-header">評估結果</div><div class="card-body">';
                
                // 命中率結果
                html += '<h5>命中率分析</h5>';
                html += `<p>平均命中率: ${(response.hit_results.avg_hit_rate * 100).toFixed(2)}%</p>`;
                html += `<p>3+號碼命中比例: ${(response.evaluation_report.hit_count_3_plus_percentage * 100).toFixed(2)}%</p>`;
                
                // 分布相似度
                html += '<h5>分布相似度</h5>';
                html += `<p>KL散度: ${response.evaluation_report.distribution_similarity.kl_divergence.toFixed(4)} (越小越相似)</p>`;
                html += `<p>KS統計量: ${response.evaluation_report.distribution_similarity.ks_statistic.toFixed(4)} (越小越相似)</p>`;
                html += `<p>餘弦相似度: ${response.evaluation_report.distribution_similarity.cosine_similarity.toFixed(4)} (越大越相似)</p>`;
                
                // 多樣性分數
                html += '<h5>多樣性分數</h5>';
                html += `<p>組內多樣性: ${response.evaluation_report.diversity_score.intra_set_diversity.toFixed(4)}</p>`;
                html += `<p>組間多樣性: ${response.evaluation_report.diversity_score.inter_set_diversity.toFixed(4)}</p>`;
                
                // 綜合評分
                html += '<h5>綜合評分</h5>';
                html += `<p>綜合分數: ${response.evaluation_report.composite_score.toFixed(4)}</p>`;
                
                html += '</div></div>';
                
                $('#evaluationResults').html(html);
            },
            error: function(error) {
                $('#evaluationResults').html(`<div class="alert alert-danger">評估失敗: ${error.responseJSON?.message || '未知錯誤'}</div>`);
            }
        });
    });

    // 從UI獲取預測結果
    function getPredictionsFromUI() {
        const predictions = [];
        $('.prediction-set').each(function() {
            const predSet = [];
            $(this).find('.prediction-numbers').each(function() {
                const numbersText = $(this).text();
                const numbers = numbersText.split(',').map(n => parseInt(n.trim())).filter(n => !isNaN(n));
                if (numbers.length > 0) {
                    predSet.push(numbers);
                }
            });
            if (predSet.length > 0) {
                predictions.push(predSet);
            }
        });
        return predictions;
    }

    // 從UI獲取實際開獎號碼
    function getActualNumbersFromUI() {
        const numbersText = $('#actualNumbers').val();
        return numbersText.split(',').map(n => parseInt(n.trim())).filter(n => !isNaN(n));
    }

    // 參數優化設置
    $('#optimizeSettingsBtn').click(function() {
        $('#optimizeModal').modal('show');
    });

    $('#startOptimization').click(function() {
        const trials = parseInt($('#optimizeTrials').val()) || 100;
        const actualNumbers = $('#optimizeActualNumbers').val().split(',').map(n => parseInt(n.trim())).filter(n => !isNaN(n));
        
        if (!actualNumbers.length) {
            showAlert('請輸入實際開獎號碼', 'warning');
            return;
        }
        
        $('#optimizeModal').modal('hide');
        $('#optimizationProgress').html('<div class="progress"><div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 0%"></div></div>');
        
        $.ajax({
            url: '/optimize',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                actual_numbers: actualNumbers,
                trials: trials
            }),
            success: function(response) {
                $('#optimizationProgress').html('');
                showAlert(`參數優化完成! 最佳模型: ${response.optimal_params.model_name}, 命中率: ${(response.optimal_params.hit_rate * 100).toFixed(2)}%`, 'success');
                
                // 顯示最佳參數
                let html = '<div class="card mt-3"><div class="card-header">最佳參數</div><div class="card-body">';
                html += `<p>模型: ${response.optimal_params.model_name}</p>`;
                html += `<p>命中率: ${(response.optimal_params.hit_rate * 100).toFixed(2)}%</p>`;
                html += '<p>參數:</p><pre>' + JSON.stringify(response.optimal_params.params, null, 2) + '</pre>';
                html += '</div></div>';
                
                $('#optimizationResults').html(html);
            },
            error: function(error) {
                $('#optimizationProgress').html('');
                showAlert(`參數優化失敗: ${error.responseJSON?.message || '未知錯誤'}`, 'danger');
            }
        });
    });

    // 初始化頁面
    $(document).ready(function() {
        // 載入多樣性設置
        const diversityMethod = localStorage.getItem('diversityMethod') || 'hybrid';
        const diversityLevel = localStorage.getItem('diversityLevel') || '0.2';
        
        $('#diversityMethod').val(diversityMethod);
        $('#diversityLevel').val(diversityLevel);
        
        // 載入數據摘要
        loadDataSummary();
    });

    // 載入數據摘要
    function loadDataSummary() {
        $.ajax({
            url: '/data',
            method: 'GET',
            success: function(response) {
                // 顯示基本統計信息
                $('#totalRecords').text(response.total_records);
                
                // 顯示熱門號碼
                let hotNumbersHtml = '';
                response.hot_numbers.forEach(function(num) {
                    hotNumbersHtml += `<span class="badge bg-danger me-1">${num}</span>`;
                });
                $('#hotNumbers').html(hotNumbersHtml);
                
                // 顯示冷門號碼
                let coldNumbersHtml = '';
                response.cold_numbers.forEach(function(num) {
                    coldNumbersHtml += `<span class="badge bg-info me-1">${num}</span>`;
                });
                $('#coldNumbers').html(coldNumbersHtml);
                
                // 顯示最近開獎結果
                let recentDrawsHtml = '<div class="table-responsive"><table class="table table-sm"><thead><tr><th>開獎號碼</th></tr></thead><tbody>';
                response.recent_draws.forEach(function(draw) {
                    recentDrawsHtml += '<tr><td>';
                    draw.forEach(function(num) {
                        recentDrawsHtml += `<span class="badge bg-primary me-1">${num}</span>`;
                    });
                    recentDrawsHtml += '</td></tr>';
                });
                recentDrawsHtml += '</tbody></table></div>';
                $('#recentDraws').html(recentDrawsHtml);
                
                // 顯示號碼頻率
                createFrequencyChart(response.number_frequencies);
            },
            error: function(error) {
                showAlert(`載入數據摘要失敗: ${error.responseJSON?.message || '未知錯誤'}`, 'danger');
            }
        });
    }

    // 創建頻率圖表
    function createFrequencyChart(frequencies) {
        const ctx = document.getElementById('frequencyChart').getContext('2d');
        
        // 準備數據
        const labels = frequencies.map(item => item.number);
        const data = frequencies.map(item => item.percentage);
        
        // 創建圖表
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: '出現頻率 (%)',
                    data: data,
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: '頻率 (%)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: '號碼'
                        }
                    }
                }
            }
        });
    }

    // 顯示提示訊息
    function showAlert(message, type) {
        const alertHtml = `<div class="alert alert-${type} alert-dismissible fade show" role="alert">
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        </div>`;
        
        $('#alertContainer').html(alertHtml);
        
        // 5秒後自動關閉
        setTimeout(function() {
            $('.alert').alert('close');
        }, 5000);
    }
});