/**
 * Chess FEN Scanner - Client Application
 * Lightweight SPA for end users to scan chess positions and find games
 */

// Configuration (can be overridden by config.js)
const CONFIG = {
    API_BASE: window.APP_CONFIG?.API_BASE || 'http://localhost:5001/api/chess',
    CHESS_COM_SEARCH_URL: window.APP_CONFIG?.CHESS_COM_SEARCH_URL || 'https://www.chess.com/games/search',
    CHESS_COM_ANALYSIS_URL: 'https://www.chess.com/analysis',
    CHESS_COM_PLAY_URL: 'https://www.chess.com/analysis',
    CAPTURE_SIZE: 512,
    PIECE_UNICODE: {
        'wK': '♔', 'wQ': '♕', 'wR': '♖', 'wB': '♗', 'wN': '♘', 'wP': '♙',
        'bK': '♚', 'bQ': '♛', 'bR': '♜', 'bB': '♝', 'bN': '♞', 'bP': '♟',
        'empty': ''
    }
};

// State
const state = {
    imageBase64: null,
    warpedImageBase64: null,  // Warped board image for feedback
    resultFen: null,
    originalFen: null,
    board: null,
    originalBoard: null,
    corrections: {},
    cameraStream: null,
    facingMode: 'environment',
    editMode: false,
    selectedPiece: 'empty'
};

// DOM Elements
const elements = {};

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    cacheElements();
    bindEvents();
    loadModels();
});

function cacheElements() {
    elements.fileInput = document.getElementById('fileInput');
    elements.cameraBtn = document.getElementById('cameraBtn');
    elements.previewContainer = document.getElementById('previewContainer');
    elements.previewImage = document.getElementById('previewImage');
    elements.clearBtn = document.getElementById('clearBtn');
    elements.optionsPanel = document.getElementById('optionsPanel');
    elements.skipDetection = document.getElementById('skipDetection');
    elements.modelSelect = document.getElementById('modelSelect');
    elements.analyzeBtn = document.getElementById('analyzeBtn');
    elements.deepAnalyzeBtn = document.getElementById('deepAnalyzeBtn');
    
    elements.inputSection = document.getElementById('inputSection');
    elements.loadingSection = document.getElementById('loadingSection');
    elements.resultsSection = document.getElementById('resultsSection');
    elements.errorSection = document.getElementById('errorSection');
    elements.errorMessage = document.getElementById('errorMessage');
    
    elements.chessBoard = document.getElementById('chessBoard');
    elements.fenInput = document.getElementById('fenInput');
    elements.fenStatus = document.getElementById('fenStatus');
    elements.copyFenBtn = document.getElementById('copyFenBtn');
    elements.searchWhiteBtn = document.getElementById('searchWhiteBtn');
    elements.analyzeWhiteBtn = document.getElementById('analyzeWhiteBtn');
    elements.analyzeBlackBtn = document.getElementById('analyzeBlackBtn');
    elements.playWhiteBtn = document.getElementById('playWhiteBtn');
    elements.playBlackBtn = document.getElementById('playBlackBtn');
    elements.searchBlackBtn = document.getElementById('searchBlackBtn');
    elements.newScanBtn = document.getElementById('newScanBtn');
    elements.retryBtn = document.getElementById('retryBtn');
    
    elements.cameraModal = document.getElementById('cameraModal');
    elements.cameraVideo = document.getElementById('cameraVideo');
    elements.cameraStatus = document.getElementById('cameraStatus');
    elements.captureBtn = document.getElementById('captureBtn');
    elements.switchCameraBtn = document.getElementById('switchCameraBtn');
    elements.closeCameraBtn = document.getElementById('closeCameraBtn');
    
    elements.toast = document.getElementById('toast');
    
    // Board editing elements
    elements.editBoardBtn = document.getElementById('editBoardBtn');
    elements.resetBoardBtn = document.getElementById('resetBoardBtn');
    elements.piecePalette = document.getElementById('piecePalette');
    elements.correctionNotice = document.getElementById('correctionNotice');
    elements.correctionCount = document.getElementById('correctionCount');
    elements.submitCorrectionsBtn = document.getElementById('submitCorrectionsBtn');
    
    // Check for games elements
    elements.checkGamesBtn = document.getElementById('checkGamesBtn');
    elements.gamesResults = document.getElementById('gamesResults');
    elements.whiteGamesResult = document.getElementById('whiteGamesResult');
    elements.blackGamesResult = document.getElementById('blackGamesResult');
    
    // Deep analysis elements
    elements.deepAnalysisResults = document.getElementById('deepAnalysisResults');
    elements.consensusStatus = document.getElementById('consensusStatus');
    elements.gamesStatus = document.getElementById('gamesStatus');
    elements.modelResultsList = document.getElementById('modelResultsList');
    elements.disagreementsList = document.getElementById('disagreementsList');
}

function bindEvents() {
    // File input
    elements.fileInput.addEventListener('change', handleFileSelect);
    
    // Camera
    elements.cameraBtn.addEventListener('click', openCamera);
    elements.closeCameraBtn.addEventListener('click', closeCamera);
    elements.switchCameraBtn.addEventListener('click', switchCamera);
    elements.captureBtn.addEventListener('click', captureFrame);
    
    // Actions
    elements.clearBtn.addEventListener('click', clearImage);
    elements.analyzeBtn.addEventListener('click', analyzeBoard);
    elements.deepAnalyzeBtn.addEventListener('click', deepAnalyzeBoard);
    elements.copyFenBtn.addEventListener('click', copyFen);
    elements.newScanBtn.addEventListener('click', resetToInput);
    elements.retryBtn.addEventListener('click', resetToInput);
    
    // Board editing
    elements.editBoardBtn.addEventListener('click', toggleEditMode);
    elements.resetBoardBtn.addEventListener('click', resetBoard);
    elements.submitCorrectionsBtn.addEventListener('click', submitCorrections);
    
    // Check for games
    elements.checkGamesBtn.addEventListener('click', checkForGames);
    
    // Piece palette clicks
    elements.piecePalette.addEventListener('click', handlePaletteClick);
}

// ============================================
// Image Handling
// ============================================

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        const base64 = e.target.result.split(',')[1];
        setImage(base64);
    };
    reader.readAsDataURL(file);
}

function setImage(base64) {
    state.imageBase64 = base64;
    elements.previewImage.src = `data:image/jpeg;base64,${base64}`;
    elements.previewContainer.hidden = false;
    elements.optionsPanel.hidden = false;
    elements.analyzeBtn.disabled = false;
    elements.deepAnalyzeBtn.disabled = false;
}

function clearImage() {
    state.imageBase64 = null;
    elements.previewContainer.hidden = true;
    elements.optionsPanel.hidden = true;
    elements.analyzeBtn.disabled = true;
    elements.deepAnalyzeBtn.disabled = true;
    elements.fileInput.value = '';
}

// ============================================
// Camera
// ============================================

async function openCamera() {
    elements.cameraModal.hidden = false;
    elements.captureBtn.disabled = true;
    updateCameraStatus('Requesting camera access...', '');
    
    try {
        const constraints = {
            video: { facingMode: state.facingMode },
            audio: false
        };
        
        state.cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        elements.cameraVideo.srcObject = state.cameraStream;
        
        await new Promise((resolve, reject) => {
            const timeout = setTimeout(() => reject(new Error('Video timeout')), 8000);
            elements.cameraVideo.onloadedmetadata = () => {
                clearTimeout(timeout);
                elements.cameraVideo.play().then(resolve).catch(reject);
            };
        });
        
        // Wait for video to stabilize
        await new Promise(r => setTimeout(r, 500));
        
        elements.captureBtn.disabled = false;
        updateCameraStatus('Camera ready! Align the board and tap Capture.', 'success');
        
    } catch (error) {
        console.error('Camera error:', error);
        let message = error.message;
        if (error.name === 'NotAllowedError') {
            message = 'Camera permission denied. Please allow camera access.';
        } else if (error.name === 'NotFoundError') {
            message = 'No camera found on this device.';
        }
        updateCameraStatus(message, 'error');
    }
}

function closeCamera() {
    if (state.cameraStream) {
        state.cameraStream.getTracks().forEach(t => t.stop());
        state.cameraStream = null;
    }
    elements.cameraVideo.srcObject = null;
    elements.cameraModal.hidden = true;
}

async function switchCamera() {
    state.facingMode = state.facingMode === 'environment' ? 'user' : 'environment';
    closeCamera();
    await openCamera();
}

function captureFrame() {
    try {
        const video = elements.cameraVideo;
        if (!video || !video.videoWidth) {
            alert('Camera not ready');
            return;
        }
        
        updateCameraStatus('Capturing...', '');
        
        const vw = video.videoWidth;
        const vh = video.videoHeight;
        const size = Math.min(vw, vh) * 0.8;
        const sx = (vw - size) / 2;
        const sy = (vh - size) / 2;
        
        const canvas = document.createElement('canvas');
        canvas.width = CONFIG.CAPTURE_SIZE;
        canvas.height = CONFIG.CAPTURE_SIZE;
        const ctx = canvas.getContext('2d');
        
        // Draw cropped region
        ctx.drawImage(video, sx, sy, size, size, 0, 0, CONFIG.CAPTURE_SIZE, CONFIG.CAPTURE_SIZE);
        
        // No client-side preprocessing - all preprocessing is done server-side for consistency
        
        // Get base64
        const base64 = canvas.toDataURL('image/jpeg', 0.92).split(',')[1];
        
        closeCamera();
        setImage(base64);
    } catch (error) {
        alert('Capture error: ' + error.message);
        console.error('Capture error:', error);
    }
}

function updateCameraStatus(message, type) {
    elements.cameraStatus.textContent = message;
    elements.cameraStatus.className = 'camera-status' + (type ? ' ' + type : '');
}

// ============================================
// Image Preprocessing
// ============================================

function preprocessImage(ctx, width, height) {
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;
    
    // Auto-levels
    let minR = 255, maxR = 0, minG = 255, maxG = 0, minB = 255, maxB = 0;
    for (let i = 0; i < data.length; i += 4) {
        minR = Math.min(minR, data[i]);
        maxR = Math.max(maxR, data[i]);
        minG = Math.min(minG, data[i + 1]);
        maxG = Math.max(maxG, data[i + 1]);
        minB = Math.min(minB, data[i + 2]);
        maxB = Math.max(maxB, data[i + 2]);
    }
    
    const rangeR = maxR - minR || 1;
    const rangeG = maxG - minG || 1;
    const rangeB = maxB - minB || 1;
    
    for (let i = 0; i < data.length; i += 4) {
        data[i] = Math.round(((data[i] - minR) / rangeR) * 255);
        data[i + 1] = Math.round(((data[i + 1] - minG) / rangeG) * 255);
        data[i + 2] = Math.round(((data[i + 2] - minB) / rangeB) * 255);
    }
    
    // Increase contrast (1.3x)
    const factor = 1.3;
    for (let i = 0; i < data.length; i += 4) {
        data[i] = Math.min(255, Math.max(0, 128 + (data[i] - 128) * factor));
        data[i + 1] = Math.min(255, Math.max(0, 128 + (data[i + 1] - 128) * factor));
        data[i + 2] = Math.min(255, Math.max(0, 128 + (data[i + 2] - 128) * factor));
    }
    
    ctx.putImageData(imageData, 0, 0);
}

// ============================================
// API Calls
// ============================================

async function loadModels() {
    try {
        const response = await fetch(`${CONFIG.API_BASE}/models`);
        if (response.ok) {
            const data = await response.json();
            console.log('Models response:', data);
            
            // Handle both formats: { models: [...] } or direct array, and PascalCase
            const models = data.models || data.Models || data;
            
            if (Array.isArray(models)) {
                models.forEach(model => {
                    const option = document.createElement('option');
                    // Handle both camelCase and PascalCase
                    const name = model.name || model.Name;
                    const accuracy = model.accuracy || model.Accuracy || '';
                    option.value = name;
                    option.textContent = accuracy ? `${name} (${accuracy})` : name;
                    elements.modelSelect.appendChild(option);
                });
            }
        }
    } catch (error) {
        console.warn('Could not load models:', error);
    }
}

async function analyzeBoard() {
    if (!state.imageBase64) return;
    
    showSection('loading');
    
    try {
        const skipDetection = elements.skipDetection.checked;
        const model = elements.modelSelect.value;
        
        const queryParams = ['debug=true'];  // Always request debug to get warped image
        if (skipDetection) queryParams.push('skip_detection=true');
        if (model) queryParams.push(`model=${model}`);
        const query = '?' + queryParams.join('&');
        
        const url = `${CONFIG.API_BASE}/predict/base64${query}`;
        console.log('Calling API:', url);
        
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: state.imageBase64 })
        });
        
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            const text = await response.text();
            console.error('API error response:', text);
            throw new Error(`API error: ${response.status} - ${text.substring(0, 200)}`);
        }
        
        const result = await response.json();
        console.log('API result:', JSON.stringify(result, null, 2));
        
        // Check for FEN in both camelCase and PascalCase (Python vs .NET)
        const fen = result.fen || result.Fen;
        const board = result.board || result.Board;
        const success = result.success ?? result.Success;
        const error = result.error || result.Error;
        
        if (success && fen) {
            state.resultFen = fen;
            state.originalFen = fen;
            state.board = JSON.parse(JSON.stringify(board)); // Deep copy
            state.originalBoard = JSON.parse(JSON.stringify(board)); // Deep copy
            state.corrections = {};
            state.editMode = false;
            
            // Store confidences for highlighting uncertain predictions
            state.confidences = result.confidences || result.Confidences || null;
            
            // Store warped image for feedback (if available from debug response)
            const debugImages = result.debug_images || result.debugImages;
            if (debugImages && debugImages.warped) {
                state.warpedImageBase64 = debugImages.warped;
                console.log('Stored warped image for feedback');
            } else {
                state.warpedImageBase64 = null;
            }
            
            displayResults();
        } else {
            // Show the actual response for debugging
            throw new Error(error || 'Board detection failed. Try "skip detection" if image is already cropped. Response: ' + JSON.stringify(result).substring(0, 100));
        }
        
    } catch (error) {
        console.error('Analyze error:', error);
        // Show detailed error including if it's CORS
        let errorMsg = error.message;
        if (error.message === 'Failed to fetch') {
            errorMsg = 'Cannot connect to API. Check that the API is running at ' + CONFIG.API_BASE;
        }
        showError(errorMsg);
    }
}

async function deepAnalyzeBoard() {
    if (!state.imageBase64) return;
    
    showSection('loading');
    document.querySelector('#loadingSection p').textContent = 'Running deep analysis with all models...';
    
    try {
        const skipDetection = elements.skipDetection.checked;
        const query = skipDetection ? '?skipDetection=true' : '';
        
        const url = `${CONFIG.API_BASE}/analyze/deep${query}`;
        console.log('Calling Deep Analysis API:', url);
        
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: state.imageBase64 })
        });
        
        if (!response.ok) {
            const text = await response.text();
            throw new Error(`API error: ${response.status} - ${text.substring(0, 200)}`);
        }
        
        const result = await response.json();
        console.log('Deep Analysis result:', result);
        
        // Check for success
        const success = result.success ?? result.Success;
        const error = result.error || result.Error;
        
        if (!success) {
            throw new Error(error || 'Deep analysis failed');
        }
        
        // Use recommended FEN
        const fen = result.recommendedFen || result.RecommendedFen;
        
        if (fen) {
            state.resultFen = fen;
            state.originalFen = fen;
            
            // Get board from first successful model
            const modelResults = result.modelResults || result.ModelResults || [];
            const successfulModel = modelResults.find(m => m.success || m.Success);
            if (successfulModel) {
                const board = successfulModel.board || successfulModel.Board;
                if (board) {
                    state.board = JSON.parse(JSON.stringify(board));
                    state.originalBoard = JSON.parse(JSON.stringify(board));
                }
            }
            
            state.corrections = {};
            state.editMode = false;
            
            // Display results and deep analysis info
            displayResults();
            displayDeepAnalysisResults(result);
        } else {
            throw new Error('No FEN could be determined from any model');
        }
        
    } catch (error) {
        console.error('Deep analyze error:', error);
        let errorMsg = error.message;
        if (error.message === 'Failed to fetch') {
            errorMsg = 'Cannot connect to API. Check that the API is running at ' + CONFIG.API_BASE;
        }
        showError(errorMsg);
    }
}

function displayDeepAnalysisResults(result) {
    const consensus = result.consensus ?? result.Consensus;
    const gamesMessage = result.gamesMessage || result.GamesMessage;
    const gamesFoundWhite = result.gamesFoundWhite ?? result.GamesFoundWhite;
    const gamesFoundBlack = result.gamesFoundBlack ?? result.GamesFoundBlack;
    const modelResults = result.modelResults || result.ModelResults || [];
    const disagreements = result.disagreements || result.Disagreements || [];
    
    // Show the deep analysis results section
    elements.deepAnalysisResults.hidden = false;
    
    // Consensus status
    if (consensus) {
        elements.consensusStatus.innerHTML = '<span class="consensus-yes">✅ All models agree on the position!</span>';
        elements.consensusStatus.className = 'consensus-status success';
    } else {
        const uniqueFens = [...new Set(modelResults.filter(m => m.success || m.Success).map(m => m.fen || m.Fen))];
        elements.consensusStatus.innerHTML = `<span class="consensus-no">⚠️ Models disagree (${uniqueFens.length} different results)</span>`;
        elements.consensusStatus.className = 'consensus-status warning';
    }
    
    // Games status
    if (gamesMessage) {
        const hasGames = gamesFoundWhite || gamesFoundBlack;
        elements.gamesStatus.innerHTML = `<span class="${hasGames ? 'games-found' : 'games-not-found'}">${gamesMessage}</span>`;
        elements.gamesStatus.className = 'games-status ' + (hasGames ? 'success' : 'muted');
    } else {
        elements.gamesStatus.innerHTML = '';
    }
    
    // Model results list
    let modelsHtml = '<h4>Model Results:</h4><ul class="model-list">';
    for (const m of modelResults) {
        const success = m.success ?? m.Success;
        const model = m.model || m.Model;
        const accuracy = m.accuracy || m.Accuracy;
        const icon = success ? '✅' : '❌';
        modelsHtml += `<li>${icon} <strong>${model}</strong> (${accuracy})</li>`;
    }
    modelsHtml += '</ul>';
    elements.modelResultsList.innerHTML = modelsHtml;
    
    // Disagreements
    if (disagreements.length > 0) {
        let disHtml = '<h4>⚠️ Disagreements:</h4><ul class="disagreement-list">';
        for (const d of disagreements) {
            const square = d.square || d.Square;
            const preds = d.predictions || d.Predictions || {};
            const predStr = Object.entries(preds).map(([model, piece]) => `${model}: ${piece || 'empty'}`).join(', ');
            disHtml += `<li><strong>${square}</strong>: ${predStr}</li>`;
        }
        disHtml += '</ul>';
        elements.disagreementsList.innerHTML = disHtml;
    } else {
        elements.disagreementsList.innerHTML = '';
    }
}

// ============================================
// Results Display
// ============================================

function displayResults() {
    // Hide deep analysis results (will be shown by displayDeepAnalysisResults if needed)
    elements.deepAnalysisResults.hidden = true;
    
    // Show captured board image (warped image if available, otherwise original)
    const capturedBoardContainer = document.getElementById('capturedBoardContainer');
    const capturedBoardImage = document.getElementById('capturedBoardImage');
    if (state.warpedImageBase64) {
        capturedBoardImage.src = 'data:image/png;base64,' + state.warpedImageBase64;
        capturedBoardContainer.hidden = false;
    } else if (state.imageBase64) {
        capturedBoardImage.src = 'data:image/jpeg;base64,' + state.imageBase64;
        capturedBoardContainer.hidden = false;
    } else {
        capturedBoardContainer.hidden = true;
    }
    
    // Render board
    renderBoard();
    
    // Show FEN
    elements.fenInput.value = state.resultFen;
    
    // Validate FEN
    const validation = validateFen(state.resultFen);
    if (validation.isValid) {
        elements.fenStatus.textContent = '✓ Valid FEN';
        elements.fenStatus.className = 'fen-status valid';
    } else {
        elements.fenStatus.textContent = `⚠ ${validation.issues[0] || 'Invalid FEN'}`;
        elements.fenStatus.className = 'fen-status invalid';
    }
    
    // Set Chess.com search links
    const castling = inferCastlingRights();
    const whiteFen = `${state.resultFen} w ${castling} - 0 1`;
    const blackFen = `${state.resultFen} b ${castling} - 0 1`;
    
    elements.searchWhiteBtn.href = `${CONFIG.CHESS_COM_SEARCH_URL}?fen=${encodeURIComponent(whiteFen)}`;
    elements.searchBlackBtn.href = `${CONFIG.CHESS_COM_SEARCH_URL}?fen=${encodeURIComponent(blackFen)}`;
    
    // Set Chess.com analysis links
    elements.analyzeWhiteBtn.href = `${CONFIG.CHESS_COM_ANALYSIS_URL}?fen=${encodeURIComponent(whiteFen)}&flip=false&tab=analysis`;
    elements.analyzeBlackBtn.href = `${CONFIG.CHESS_COM_ANALYSIS_URL}?fen=${encodeURIComponent(blackFen)}&flip=true&tab=analysis`;
    
    // Set Chess.com play links
    elements.playWhiteBtn.href = `${CONFIG.CHESS_COM_PLAY_URL}?fen=${encodeURIComponent(whiteFen)}&flip=false&tab=play`;
    elements.playBlackBtn.href = `${CONFIG.CHESS_COM_PLAY_URL}?fen=${encodeURIComponent(blackFen)}&flip=true&tab=play`;
    
    // Reset games check results
    elements.gamesResults.hidden = true;
    elements.whiteGamesResult.innerHTML = '';
    elements.blackGamesResult.innerHTML = '';
    
    // Update edit UI
    updateEditUI();
    
    showSection('results');
}

function renderBoard() {
    // Clear existing content
    elements.chessBoard.innerHTML = '';
    elements.chessBoard.className = state.editMode ? 'chess-board editing' : 'chess-board';
    
    // Check if we have a valid board
    if (!state.board || !Array.isArray(state.board) || state.board.length !== 8) {
        console.warn('renderBoard: Invalid board state', state.board);
        return;
    }
    
    for (let row = 0; row < 8; row++) {
        const tr = document.createElement('tr');
        for (let col = 0; col < 8; col++) {
            const td = document.createElement('td');
            td.className = (row + col) % 2 === 0 ? 'light' : 'dark';
            td.dataset.row = row;
            td.dataset.col = col;
            
            const piece = state.board?.[row]?.[col];
            if (piece && piece !== 'empty') {
                td.textContent = CONFIG.PIECE_UNICODE[piece] || '';
            }
            
            // Highlight low-confidence predictions (< 90%)
            const confidence = state.confidences?.[row]?.[col];
            if (confidence !== undefined && confidence < 0.90 && piece !== 'empty') {
                td.classList.add('low-confidence');
                td.title = `Confidence: ${(confidence * 100).toFixed(0)}%`;
            }
            
            // Mark edited squares
            const key = `${row},${col}`;
            if (state.corrections[key]) {
                td.classList.add('edited');
            }
            
            // Click handler for editing
            if (state.editMode) {
                td.addEventListener('click', () => handleSquareClick(row, col));
            }
            
            tr.appendChild(td);
        }
        elements.chessBoard.appendChild(tr);
    }
    
    // Force repaint on mobile browsers
    elements.chessBoard.style.display = 'none';
    elements.chessBoard.offsetHeight; // Trigger reflow
    elements.chessBoard.style.display = '';
}

function validateFen(fen) {
    const issues = [];
    const rows = fen.split('/');
    
    if (rows.length !== 8) {
        issues.push(`Expected 8 rows, got ${rows.length}`);
    }
    
    let whiteKings = 0, blackKings = 0;
    
    for (let i = 0; i < rows.length; i++) {
        let count = 0;
        for (const char of rows[i]) {
            if (/[1-8]/.test(char)) {
                count += parseInt(char);
            } else if (/[pnbrqkPNBRQK]/.test(char)) {
                count++;
                if (char === 'K') whiteKings++;
                if (char === 'k') blackKings++;
            } else {
                issues.push(`Invalid character '${char}' in row ${i + 1}`);
            }
        }
        if (count !== 8) {
            issues.push(`Row ${i + 1} has ${count} squares instead of 8`);
        }
    }
    
    if (whiteKings !== 1) issues.push(`White should have 1 king, found ${whiteKings}`);
    if (blackKings !== 1) issues.push(`Black should have 1 king, found ${blackKings}`);
    
    return {
        isValid: issues.length === 0,
        issues
    };
}

function inferCastlingRights() {
    if (!state.board) return '-';
    
    let rights = '';
    
    // White kingside: King on e1, Rook on h1
    if (state.board[7]?.[4] === 'wK' && state.board[7]?.[7] === 'wR') rights += 'K';
    // White queenside: King on e1, Rook on a1
    if (state.board[7]?.[4] === 'wK' && state.board[7]?.[0] === 'wR') rights += 'Q';
    // Black kingside: King on e8, Rook on h8
    if (state.board[0]?.[4] === 'bK' && state.board[0]?.[7] === 'bR') rights += 'k';
    // Black queenside: King on e8, Rook on a8
    if (state.board[0]?.[4] === 'bK' && state.board[0]?.[0] === 'bR') rights += 'q';
    
    return rights || '-';
}

// ============================================
// UI Helpers
// ============================================

function showSection(section) {
    elements.inputSection.hidden = section !== 'input';
    elements.loadingSection.hidden = section !== 'loading';
    elements.resultsSection.hidden = section !== 'results';
    elements.errorSection.hidden = section !== 'error';
    
    if (section === 'input') {
        elements.inputSection.hidden = false;
    }
    
    if (section === 'results') {
        elements.inputSection.hidden = false;
        // Scroll to results on mobile after a short delay
        setTimeout(() => {
            elements.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }
}

function showError(message) {
    elements.errorMessage.textContent = message;
    showSection('error');
}

function resetToInput() {
    state.resultFen = null;
    state.originalFen = null;
    state.board = null;
    state.originalBoard = null;
    state.corrections = {};
    state.editMode = false;
    
    elements.editBoardBtn.textContent = '✏️ Edit';
    elements.piecePalette.hidden = true;
    
    showSection('input');
}

async function copyFen() {
    try {
        await navigator.clipboard.writeText(elements.fenInput.value);
        showToast('FEN copied to clipboard!');
    } catch (error) {
        // Fallback for iOS
        elements.fenInput.select();
        document.execCommand('copy');
        showToast('FEN copied!');
    }
}

function showToast(message) {
    elements.toast.textContent = message;
    elements.toast.hidden = false;
    
    setTimeout(() => {
        elements.toast.hidden = true;
    }, 2500);
}

// ============================================
// Board Editing
// ============================================

function toggleEditMode() {
    state.editMode = !state.editMode;
    state.selectedPiece = 'empty';
    
    elements.editBoardBtn.textContent = state.editMode ? '✓ Done' : '✏️ Edit';
    elements.piecePalette.hidden = !state.editMode;
    
    // Update palette selection
    updatePaletteSelection();
    renderBoard();
}

function updateEditUI() {
    const hasCorrections = Object.keys(state.corrections).length > 0;
    
    elements.correctionNotice.hidden = !hasCorrections;
    elements.correctionCount.textContent = Object.keys(state.corrections).length;
    elements.resetBoardBtn.hidden = !hasCorrections;
}

function handlePaletteClick(e) {
    const piece = e.target.dataset.piece;
    if (!piece) return;
    
    state.selectedPiece = piece;
    updatePaletteSelection();
}

function updatePaletteSelection() {
    elements.piecePalette.querySelectorAll('.palette-piece').forEach(el => {
        el.classList.toggle('selected', el.dataset.piece === state.selectedPiece);
    });
}

function handleSquareClick(row, col) {
    if (!state.editMode) return;
    
    const currentPiece = state.board[row][col];
    const newPiece = state.selectedPiece;
    
    if (currentPiece === newPiece) return;
    
    // Update the board
    state.board[row][col] = newPiece;
    
    // Track corrections (compared to original)
    const key = `${row},${col}`;
    const originalPiece = state.originalBoard[row][col];
    
    if (newPiece === originalPiece) {
        delete state.corrections[key];
    } else {
        const squareName = String.fromCharCode(97 + col) + (8 - row);
        state.corrections[key] = {
            square: squareName,
            from: originalPiece,
            to: newPiece
        };
    }
    
    // Recalculate FEN
    state.resultFen = boardToFen(state.board);
    
    // Re-render
    displayResults();
}

function resetBoard() {
    state.board = JSON.parse(JSON.stringify(state.originalBoard));
    state.resultFen = state.originalFen;
    state.corrections = {};
    state.editMode = false;
    
    elements.editBoardBtn.textContent = '✏️ Edit';
    elements.piecePalette.hidden = true;
    
    displayResults();
    showToast('Board reset to original');
}

function boardToFen(board) {
    const rows = [];
    for (let row = 0; row < 8; row++) {
        let fenRow = '';
        let emptyCount = 0;
        
        for (let col = 0; col < 8; col++) {
            const piece = board[row][col];
            if (!piece || piece === 'empty') {
                emptyCount++;
            } else {
                if (emptyCount > 0) {
                    fenRow += emptyCount;
                    emptyCount = 0;
                }
                fenRow += pieceToFenChar(piece);
            }
        }
        
        if (emptyCount > 0) {
            fenRow += emptyCount;
        }
        
        rows.push(fenRow);
    }
    return rows.join('/');
}

function pieceToFenChar(piece) {
    const map = {
        'wP': 'P', 'wN': 'N', 'wB': 'B', 'wR': 'R', 'wQ': 'Q', 'wK': 'K',
        'bP': 'p', 'bN': 'n', 'bB': 'b', 'bR': 'r', 'bQ': 'q', 'bK': 'k'
    };
    return map[piece] || '';
}

async function submitCorrections() {
    if (Object.keys(state.corrections).length === 0) {
        alert('No corrections to submit');
        return;
    }
    
    // Validate state
    if (!state.originalFen) {
        alert('No original FEN available. Please analyze an image first.');
        return;
    }
    if (!state.resultFen) {
        alert('No corrected FEN available.');
        return;
    }
    
    try {
        // Transform corrections to simple {square: piece} format for API
        // state.corrections is { "row,col": { square: "a1", from: "empty", to: "wR" } }
        // API expects { "a1": "wR" }
        const correctedSquares = {};
        for (const [key, correction] of Object.entries(state.corrections)) {
            correctedSquares[correction.square] = correction.to;
        }
        
        // Use camelCase for .NET API (PropertyNameCaseInsensitive handles both)
        // Send the WARPED image (if available) so feedback tiles can be extracted correctly
        const imageToSend = state.warpedImageBase64 || state.imageBase64 || null;
        console.log('Sending feedback with', state.warpedImageBase64 ? 'warped image' : 'original image');
        
        const payload = {
            originalFen: state.originalFen,
            correctedFen: state.resultFen,
            image: imageToSend,
            correctedSquares: correctedSquares
        };
        
        console.log('Submitting corrections to:', `${CONFIG.API_BASE}/feedback`);
        console.log('Payload:', JSON.stringify(payload, null, 2));
        
        const response = await fetch(`${CONFIG.API_BASE}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const responseText = await response.text();
        console.log('Feedback response:', response.status, responseText);
        
        if (response.ok) {
            showToast('Corrections submitted for training!');
            state.originalFen = state.resultFen;
            state.originalBoard = JSON.parse(JSON.stringify(state.board));
            state.corrections = {};
            updateEditUI();
        } else {
            let errorMsg = 'Failed to submit corrections';
            try {
                const errorData = JSON.parse(responseText);
                errorMsg = errorData.error || errorData.title || errorData.message || errorMsg;
            } catch (e) {}
            showToast(errorMsg);
            console.error('Submit corrections failed:', response.status, responseText);
            alert('Submit failed: ' + response.status + ' - ' + responseText.substring(0, 200));
        }
    } catch (error) {
        console.error('Submit corrections error:', error);
        showToast('Error submitting corrections: ' + error.message);
        alert('Submit error: ' + error.message);
    }
}

// ============================================
// Chess.com Game Check
// ============================================

async function checkForGames() {
    if (!state.resultFen) return;
    
    elements.checkGamesBtn.disabled = true;
    elements.checkGamesBtn.textContent = '⏳ Checking...';
    elements.gamesResults.hidden = false;
    elements.whiteGamesResult.innerHTML = '<span class="checking">Checking white...</span>';
    elements.blackGamesResult.innerHTML = '<span class="checking">Checking black...</span>';
    
    const castling = inferCastlingRights();
    
    try {
        // Check white to move
        const whiteFen = `${state.resultFen} w ${castling} - 0 1`;
        const whiteResult = await checkChessCom(whiteFen);
        displayGameResult(elements.whiteGamesResult, 'w', whiteResult);
        
        // Check black to move
        const blackFen = `${state.resultFen} b ${castling} - 0 1`;
        const blackResult = await checkChessCom(blackFen);
        displayGameResult(elements.blackGamesResult, 'b', blackResult);
        
    } catch (error) {
        console.error('Check games error:', error);
        elements.whiteGamesResult.innerHTML = '<span class="error">Error checking</span>';
        elements.blackGamesResult.innerHTML = '<span class="error">Error checking</span>';
    } finally {
        elements.checkGamesBtn.disabled = false;
        elements.checkGamesBtn.textContent = '🔍 Check for Games';
    }
}

async function checkChessCom(fen) {
    const encodedFen = encodeURIComponent(fen);
    const response = await fetch(`${CONFIG.API_BASE}/chess-com-search?fen=${encodedFen}`);
    
    if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
    }
    
    return await response.json();
}

function displayGameResult(element, side, result) {
    const sideLabel = side === 'w' ? '⬜ White' : '⬛ Black';
    
    if (result.gamesFound || result.GamesFound) {
        element.innerHTML = `<span class="found">✅ ${sideLabel}: Games found!</span>`;
        element.classList.add('found');
        element.classList.remove('not-found');
    } else {
        const message = result.message || result.Message || 'No games found';
        element.innerHTML = `<span class="not-found">❌ ${sideLabel}: ${message}</span>`;
        element.classList.add('not-found');
        element.classList.remove('found');
    }
}
