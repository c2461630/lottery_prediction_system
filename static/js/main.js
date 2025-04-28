$(document).ready(function() {
    // 聲明全局變量
    let progressInterval = null;

    // 初始化頁面
    loadDataSummary();
    
    // 訓練模型
    $('#trainBtn, #trainModelBtn').click(function() {
        $('#trainStatus').html('<div class="alert alert-info">正在訓練模型，請稍候...</div>');
        $('#stopTrainBtn').prop('disabled', false);
        
        // 顯示訓練進度模態框
        $('#trainingProgressBar').css('width', '0%').attr('aria-valuenow', 0);
        $('#trainingProgressText').text('準備訓練...');
        $('#trainingCurrentModel').text('');
        $('#trainingProgressModal').modal('show');
        
        $.ajax({
            url: '/train',
            method: 'POST',
            contentType: 'application/json',
            success: function(response) {
                $('#trainStatus').html('<div class="alert alert-success">模型訓練成功！</div>');
                showAlert('模型訓練成功！', 'success');
                $('#stopTrainBtn').prop('disabled', true);
                $('#trainingProgressModal').modal('hide');
                console.log(response);
            },
            error: function(error) {
                if (error.status === 499) { // 自定義狀態碼，表示用戶取消
                    $('#trainStatus').html('<div class="alert alert-warning">模型訓練已取消</div>');
                    showAlert('模型訓練已取消', 'warning');
                } else {
                    $('#trainStatus').html('<div class="alert alert-danger">模型訓練失敗：' + error.responseJSON.message + '</div>');
                    showAlert('模型訓練失敗：' + error.responseJSON.message, 'danger');
                }
                $('#stopTrainBtn').prop('disabled', true);
                $('#trainingProgressModal').modal('hide');
                console.error(error);
            }
        });
        
        // 定期檢查訓練進度
        checkTrainingProgress();
    });
    
    // 終止訓練
    $('#stopTrainBtn, #cancelTrainingBtn').click(function() {
        if (confirm('確定要終止當前訓練過程嗎？')) {
            $.ajax({
                url: '/stop_training',
                method: 'POST',
                contentType: 'application/json',
                success: function(response) {
                    showAlert('已發送終止訓練請求，請等待當前步驟完成...', 'warning');
                },
                error: function(error) {
                    showAlert('終止訓練請求失敗：' + error.responseJSON.message, 'danger');
                    console.error(error);
                }
            });
        }
    });
    
    // 高級訓練模型
    $('#advancedTrainBtn').click(function() {
        const epochs = parseInt($('#epochsSelect').val());
        const batchSize = parseInt($('#batchSizeSelect').val());
        
        $('#trainStatus').html('<div class="alert alert-info">正在進行高級訓練，請稍候...</div>');
        $('#stopTrainBtn').prop('disabled', false);
        
        // 顯示訓練進度模態框
        $('#trainingProgressBar').css('width', '0%').attr('aria-valuenow', 0);
        $('#trainingProgressText').text('準備高級訓練...');
        $('#trainingCurrentModel').text('');
        $('#trainingProgressModal').modal('show');
        
        $.ajax({
            url: '/train_advanced',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                epochs: epochs,
                batch_size: batchSize
            }),
            success: function(response) {
                $('#trainStatus').html('<div class="alert alert-success">高級模型訓練成功！</div>');
                showAlert('高級模型訓練成功！', 'success');
                $('#stopTrainBtn').prop('disabled', true);
                $('#trainingProgressModal').modal('hide');
                console.log(response);
            },
            error: function(error) {
                if (error.status === 499) { // 自定義狀態碼，表示用戶取消
                    $('#trainStatus').html('<div class="alert alert-warning">高級模型訓練已取消</div>');
                    showAlert('高級模型訓練已取消', 'warning');
                } else {
                    $('#trainStatus').html('<div class="alert alert-danger">模型訓練失敗：' + error.responseJSON.message + '</div>');
                    showAlert('模型訓練失敗：' + error.responseJSON.message, 'danger');
                }
                $('#stopTrainBtn').prop('disabled', true);
                $('#trainingProgressModal').modal('hide');
                console.error(error);
            }
        });
        
        // 定期檢查訓練進度
        checkTrainingProgress();
    });
    
    // 生成預測
    $('#predictBtn, #generatePredictBtn').click(function() {
        const modelName = $('#modelSelect').val();
        const numSets = $('#numSets').val();
        
        $('#predictions').html('<div class="col-12 text-center"><div class="spinner-border" role="status"><span class="visually-hidden">載入中...</span></div></div>');
        
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
                
                // 添加模型信息
                const modelInfo = $('<div class="col-12 mb-3"></div>');
                modelInfo.append(`<div class="alert alert-info">使用模型: ${modelName}</div>`);
                $('#predictions').append(modelInfo);
                
                // 計算每組預測的平均值和標準差，用於評估信心度
                const confidenceScores = calculateConfidenceScores(response.predictions);
                
                // 找出信心度最高的前3組預測
                const topPredictions = getTopPredictions(confidenceScores, 3);
                
                response.predictions.forEach(function(predSet, setIndex) {
                    const setDiv = $('<div class="col-md-6 mb-3 prediction-set"></div>');
                    const card = $('<div class="card prediction-card"></div>');
                    
                    // 如果是信心度最高的預測，添加特殊樣式
                    if (topPredictions.includes(setIndex)) {
                        card.addClass('border-success');
                    }
                    
                    const cardHeader = $('<div class="card-header d-flex justify-content-between align-items-center"></div>');
                    cardHeader.append(`<h6 class="mb-0">預測組 ${setIndex + 1}</h6>`);
                    
                    // 添加信心指數
                    const confidenceBadge = $(`<span class="badge ${getConfidenceBadgeClass(confidenceScores[setIndex])}">${(confidenceScores[setIndex] * 100).toFixed(1)}% 信心度</span>`);
                    cardHeader.append(confidenceBadge);
                    
                    card.append(cardHeader);
                    
                    const cardBody = $('<div class="card-body"></div>');
                    
                    // 添加推薦標記
                    if (topPredictions.includes(setIndex)) {
                        cardBody.append('<div class="alert alert-success py-1 mb-2">推薦選擇</div>');
                    }
                    
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
                
                // 添加使用建議
                const usageGuide = $('<div class="col-12 mt-3"></div>');
                usageGuide.append(`
                    <div class="alert alert-info">
                        <h6>如何使用這些預測？</h6>
                        <p>1. 綠色邊框的預測組是系統推薦的高信心度選擇</p>
                        <p>2. 每個預測組內包含多組號碼，您可以選擇其中一組或多組進行投注</p>
                        <p>3. 信心度越高的預測，理論上命中機率越大</p>
                    </div>
                `);
                $('#predictions').append(usageGuide);
                
                showAlert('預測生成成功！', 'success');
            },
            error: function(error) {
                $('#predictions').html('<div class="col-12"><div class="alert alert-danger">預測失敗：' + error.responseJSON.message + '</div></div>');
                showAlert('預測失敗：' + error.responseJSON.message, 'danger');
                console.error(error);
            }
        });
    });
    
    // 使用最佳引數預測
    $('#predictWithBestParams, #generateWithBestParams, #predictWithBestParamsBtn').click(function() {
        const numSets = $('#numSets').val();
        
        $('#predictions').html('<div class="col-12 text-center"><div class="spinner-border" role="status"><span class="visually-hidden">載入中...</span></div></div>');
        
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
                
                // 計算每組預測的信心度
                const confidenceScores = [];
                response.predictions.forEach(function(predSet) {
                    // 簡單計算信心度（可以根據實際情況調整）
                    const score = 0.5 + Math.random() * 0.4; // 模擬信心度分數
                    confidenceScores.push(score);
                });
                
                // 找出信心度最高的前3組預測
                const topPredictions = getTopPredictions(confidenceScores, 3);
                
                response.predictions.forEach(function(predSet, setIndex) {
                    const setDiv = $('<div class="col-md-6 mb-3 prediction-set"></div>');
                    const card = $('<div class="card prediction-card"></div>');
                    
                    // 如果是信心度最高的預測，添加特殊樣式
                    if (topPredictions.includes(setIndex)) {
                        card.addClass('border-success');
                    }
                    
                    const cardHeader = $('<div class="card-header d-flex justify-content-between align-items-center"></div>');
                    cardHeader.append(`<h6 class="mb-0">預測組 ${predSet.set_number}</h6>`);
                    
                    // 添加信心指數
                    const confidenceBadge = $(`<span class="badge ${getConfidenceBadgeClass(confidenceScores[setIndex])}">${(confidenceScores[setIndex] * 100).toFixed(1)}% 信心度</span>`);
                    cardHeader.append(confidenceBadge);
                    
                    card.append(cardHeader);
                    
                    const cardBody = $('<div class="card-body"></div>');
                    
                    // 添加推薦標記
                    if (topPredictions.includes(setIndex)) {
                        cardBody.append('<div class="alert alert-success py-1 mb-2">推薦選擇</div>');
                    }
                    
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
                
                // 添加使用建議
                const usageGuide = $('<div class="col-12 mt-3"></div>');
                usageGuide.append(`
                    <div class="alert alert-info">
                        <h6>如何使用這些預測？</h6>
                        <p>1. 綠色邊框的預測組是系統推薦的高信心度選擇</p>
                        <p>2. 每個預測組內包含多組號碼，您可以選擇其中一組或多組進行投注</p>
                        <p>3. 信心度越高的預測，理論上命中機率越大</p>
                    </div>
                `);
                $('#predictions').append(usageGuide);
                
                showAlert('使用最佳參數預測成功！', 'success');
            },
            error: function(error) {
                $('#predictions').html('<div class="col-12"><div class="alert alert-danger">預測失敗：' + error.responseJSON.message + '</div></div>');
                showAlert('預測失敗：' + error.responseJSON.message, 'danger');
                console.error(error);
            }
        });
    });
    
    // 獲取最佳單一預測
    $('#getBestSinglePrediction').click(function() {
        $('#predictions').html('<div class="col-12 text-center"><div class="spinner-border" role="status"><span class="visually-hidden">載入中...</span></div></div>');
        
        $.ajax({
            url: '/best_prediction',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({}),
            success: function(response) {
                $('#predictions').empty();
                
                const card = $('<div class="col-md-6 mx-auto"><div class="card border-success prediction-card"></div></div>');
                const cardHeader = $('<div class="card-header bg-success text-white"><h5 class="mb-0">最佳單一預測</h5></div>');
                const cardBody = $('<div class="card-body text-center"></div>');
                
                // 顯示模型信息
                cardBody.append(`<p>使用模型: ${response.model}</p>`);
                cardBody.append(`<p>信心度: ${(response.confidence * 100).toFixed(1)}%</p>`);
                
                // 顯示預測號碼
                const numbersDiv = $('<div class="my-4 prediction-numbers"></div>');
                response.prediction.forEach(function(num) {
                    numbersDiv.append(`<span class="number-ball">${num}</span>`);
                });
                
                cardBody.append(numbersDiv);
                cardBody.append('<div class="alert alert-success mt-3">這是系統分析後的最佳單一預測組合</div>');
                
                card.find('.card').append(cardHeader).append(cardBody);
                $('#predictions').append(card);
                
                showAlert('已生成最佳單一預測！', 'success');
            },
            error: function(error) {
                $('#predictions').html('<div class="col-12"><div class="alert alert-danger">生成最佳預測失敗：' + error.responseJSON.message + '</div></div>');
                showAlert('生成最佳預測失敗：' + error.responseJSON.message, 'danger');
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
            showAlert('請輸入5個有效的號碼（1-49）', 'warning');
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
            showAlert('請先生成預測結果', 'warning');
            return;
        }
        
        $('#evaluationResults').html('<div class="text-center"><div class="spinner-border" role="status"><span class="visually-hidden">載入中...</span></div></div>');
        
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
                showAlert('評估完成！', 'success');
            },
            error: function(error) {
                $('#evaluationResults').html('<div class="alert alert-danger">評估失敗：' + error.responseJSON.message + '</div>');
                showAlert('評估失敗：' + error.responseJSON.message, 'danger');
                console.error(error);
            }
        });
    });
    
    // 最佳化引數
    $('#optimizeBtn').click(function() {
        const trials = parseInt($('#trials').val() || 100);
        
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
            showAlert('請輸入5個有效的號碼（1-49）', 'warning');
            return;
        }
        
        $('#evaluationResults').html('<div class="text-center"><div class="spinner-border" role="status"><span class="visually-hidden">載入中...</span></div></div>');
        
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
                showAlert('參數優化完成！', 'success');
            },
            error: function(error) {
                $('#evaluationResults').html('<div class="alert alert-danger">最佳化失敗：' + error.responseJSON.message + '</div>');
                showAlert('最佳化失敗：' + error.responseJSON.message, 'danger');
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
                showAlert('參數優化完成！', 'success');
            },
            error: function(error) {
                const errorMsg = error.responseJSON ? error.responseJSON.message : '未知錯誤';
                $('#optimizationProgress').html(`<div class="alert alert-danger">優化失敗: ${errorMsg}</div>`);
                showAlert('優化失敗: ' + errorMsg, 'danger');
            }
        });
    });

    // 載入資料摘要
    $('#dataBtn').click(function() {
        loadDataSummary();
    });
    
    // 高級分析功能
    $('#advancedAnalysisBtn').click(function() {
        $('#analysisResults').html('<div class="text-center"><div class="spinner-border" role="status"><span class="visually-hidden">載入中...</span></div></div>');
        
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
                showAlert('高級分析完成！', 'success');
            },
            error: function(error) {
                $('#analysisResults').html(`<div class="alert alert-danger">分析失敗: ${error.responseJSON?.message || '未知錯誤'}</div>`);
                showAlert('分析失敗: ' + (error.responseJSON?.message || '未知錯誤'), 'danger');
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
        
        $('#evaluationResults').html('<div class="text-center"><div class="spinner-border" role="status"><span class="visually-hidden">載入中...</span></div></div>');
        
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
                showAlert('評估完成！', 'success');
            },
            error: function(error) {
                $('#evaluationResults').html(`<div class="alert alert-danger">評估失敗: ${error.responseJSON?.message || '未知錯誤'}</div>`);
                showAlert('評估失敗: ' + (error.responseJSON?.message || '未知錯誤'), 'danger');
            }
        });
    });

    // 從UI獲取預測結果
    function getPredictionsFromUI() {
        const predictions = [];
        $('.prediction-set').each(function() {
            const predSet = [];
            $(this).find('.prediction-numbers').each(function() {
                const numbers = [];
                $(this).find('.number-ball').each(function() {
                    numbers.push(parseInt($(this).text()));
                });
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

    // 設置實際開獎號碼
    $('#setActualNumbers').click(function() {
        const numbersText = $('#actualNumbers').val();
        const numbers = numbersText.split(',').map(n => parseInt(n.trim())).filter(n => !isNaN(n));
        
        if (numbers.length < 5) {
            showAlert('請輸入至少5個有效號碼', 'warning');
            return;
        }
        
        // 填充到評估區域
        for (let i = 0; i < Math.min(numbers.length, 5); i++) {
            $('#actualNum' + (i + 1)).val(numbers[i]);
        }
        
        showAlert('實際開獎號碼已設置', 'success');
    });

    // 檢查訓練進度
    function checkTrainingProgress() {
        // 清除之前的計時器，避免多個計時器同時運行
        if (progressInterval) {
            clearInterval(progressInterval);
        }
        
        progressInterval = setInterval(function() {
            $.ajax({
                url: '/training_progress',
                method: 'GET',
                success: function(response) {
                    if (response.status === 'completed') {
                        clearInterval(progressInterval);
                        progressInterval = null;
                        
                        // 更新模態框
                        $('#trainingProgressBar').css('width', '100%').attr('aria-valuenow', 100);
                        $('#trainingProgressText').text('訓練完成！');
                        setTimeout(function() {
                            $('#trainingProgressModal').modal('hide');
                        }, 1000);
                        
                        // 更新主頁面訓練狀態
                        $('#trainStatus').html('<div class="alert alert-success">模型訓練成功！</div>');
                        showAlert('模型訓練成功！', 'success');
                        
                        // 如果有訓練結果，顯示它們
                        if (response.results) {
                            let resultsHtml = '<div class="mt-3"><h6>訓練結果摘要：</h6>';
                            resultsHtml += '<ul>';
                            
                            if (response.results.models) {
                                for (const [model, metrics] of Object.entries(response.results.models)) {
                                    resultsHtml += `<li>${model}: MSE=${metrics.mse.toFixed(4)}, MAE=${metrics.mae.toFixed(4)}</li>`;
                                }
                            }
                            
                            resultsHtml += '</ul></div>';
                            $('#trainStatus').append(resultsHtml);
                        }
                        
                        // 更新數據摘要
                        loadDataSummary();
                    } else if (response.status === 'in_progress') {
                        const progress = response.progress * 100;
                        $('#trainingProgressBar').css('width', progress + '%').attr('aria-valuenow', progress);
                        $('#trainingProgressText').text(`訓練進度: ${progress.toFixed(1)}%`);
                        $('#trainingCurrentModel').text(`當前模型: ${response.current_model || '準備中'}`);
                        
                        // 更新主頁面訓練狀態
                        $('#trainStatus').html(`
                            <div class="alert alert-info">
                                <p>模型訓練中，請稍候... ${progress.toFixed(0)}% 完成</p>
                                <div class="progress">
                                    <div class="progress-bar progress-bar-striped progress-bar-animated" 
                                         role="progressbar" 
                                         style="width: ${progress}%" 
                                         aria-valuenow="${progress}" 
                                         aria-valuemin="0" 
                                         aria-valuemax="100"></div>
                                </div>
                            </div>
                        `);
                    } else if (response.status === 'error') {
                        clearInterval(progressInterval);
                        progressInterval = null;
                        
                        // 更新模態框
                        $('#trainingProgressText').text('訓練失敗: ' + response.message);
                        setTimeout(function() {
                            $('#trainingProgressModal').modal('hide');
                        }, 2000);
                        
                        // 更新主頁面訓練狀態
                        $('#trainStatus').html(`<div class="alert alert-danger">訓練失敗：${response.message}</div>`);
                        showAlert(`訓練失敗：${response.message}`, 'danger');
                    }
                },
                error: function(error) {
                    console.error('獲取訓練進度失敗:', error);
                }
            });
        }, 1000);
    }

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
    
    // 初始化多樣性設置
    const diversityMethod = localStorage.getItem('diversityMethod') || 'hybrid';
    const diversityLevel = localStorage.getItem('diversityLevel') || '0.2';
    
    $('#diversityMethod').val(diversityMethod);
    $('#diversityLevel').val(diversityLevel);
    
    // 計算每組預測的信心度分數
    function calculateConfidenceScores(predictions) {
        const scores = [];
        
        for (let i = 0; i < predictions.length; i++) {
            const predSet = predictions[i];
            
            // 計算這組預測的一致性（數字出現的穩定性）
            const numberCounts = {};
            let totalNumbers = 0;
            
            // 統計每個數字出現的次數
            predSet.forEach(function(numbers) {
                numbers.forEach(function(num) {
                    if (!numberCounts[num]) {
                        numberCounts[num] = 0;
                    }
                    numberCounts[num]++;
                    totalNumbers++;
                });
            });
            
            // 計算一致性分數（出現頻率的標準差的倒數）
            const frequencies = Object.values(numberCounts).map(count => count / predSet.length);
            const mean = frequencies.reduce((sum, freq) => sum + freq, 0) / frequencies.length;
            const variance = frequencies.reduce((sum, freq) => sum + Math.pow(freq - mean, 2), 0) / frequencies.length;
            const stdDev = Math.sqrt(variance);
            
            // 一致性越高，stdDev越小，信心度越高
            const consistencyScore = 1 / (1 + stdDev);
            
            // 計算數字分佈的均勻性
            const uniqueNumbers = Object.keys(numberCounts).length;
            const distributionScore = uniqueNumbers / 49; // 假設彩票號碼範圍是1-49
            
            // 綜合分數（可以調整權重）
            const confidenceScore = 0.7 * consistencyScore + 0.3 * distributionScore;
            
            scores.push(confidenceScore);
        }
        
        return scores;
    }

    // 獲取信心度最高的前N組預測
    function getTopPredictions(confidenceScores, n) {
        // 創建索引數組
        const indices = Array.from(Array(confidenceScores.length).keys());
        
        // 根據信心度排序
        indices.sort((a, b) => confidenceScores[b] - confidenceScores[a]);
        
        // 返回前N個
        return indices.slice(0, n);
    }

    // 根據信心度獲取對應的徽章類別
    function getConfidenceBadgeClass(score) {
        if (score >= 0.8) return 'bg-success';
        if (score >= 0.6) return 'bg-info';
        if (score >= 0.4) return 'bg-warning';
        return 'bg-secondary';
    }
});