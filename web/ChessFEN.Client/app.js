/**
 * Chess FEN Scanner - Client Application
 * Lightweight SPA for end users to scan chess positions and find games
 */

// Configuration (can be overridden by config.js)
const CONFIG = {
    API_BASE: window.APP_CONFIG?.API_BASE || 'http://localhost:5001/api/chess',
    DOTNET_API_BASE: window.APP_CONFIG?.DOTNET_API_BASE || 'http://localhost:5001/api/chess',
    CHESS_COM_SEARCH_URL: window.APP_CONFIG?.CHESS_COM_SEARCH_URL || 'https://www.chess.com/games/search',
    CHESS_COM_ANALYSIS_URL: 'https://www.chess.com/analysis',
    CAPTURE_SIZE: 1024,  // Increased from 512 for better quality
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
    checkAuthStatus();
});

// Check if user is authenticated (Azure Easy Auth)
async function checkAuthStatus() {
    try {
        // Azure Container Apps Easy Auth provides /.auth/me endpoint
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 5000);
        
        const response = await fetch('/.auth/me', { 
            signal: controller.signal,
            credentials: 'include'
        });
        clearTimeout(timeout);
        
        if (response.ok) {
            const authData = await response.json();
            if (authData && authData.length > 0 && authData[0].user_id) {
                console.log('Authenticated as:', authData[0].user_id);
                return true;
            }
        }
        
        // Not authenticated or session expired
        console.log('Not authenticated or session expired');
        return false;
    } catch (error) {
        // On localhost, /.auth/me won't exist - that's fine
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            console.log('Local dev mode - skipping auth check');
            return true;
        }
        console.warn('Auth check failed:', error.message);
        return false;
    }
}

function cacheElements() {
    elements.fileInput = document.getElementById('fileInput');
    elements.cameraBtn = document.getElementById('cameraBtn');
    elements.optionsPanel = document.getElementById('optionsPanel');
    elements.skipDetection = document.getElementById('skipDetection');

    elements.inputSection = document.getElementById('inputSection');
    elements.loadingSection = document.getElementById('loadingSection');
    elements.loadingPreviewImage = document.getElementById('loadingPreviewImage');
    elements.resultsSection = document.getElementById('resultsSection');
    elements.errorSection = document.getElementById('errorSection');
    elements.errorMessage = document.getElementById('errorMessage');

    elements.chessBoard = document.getElementById('chessBoard');
    elements.fenStatus = document.getElementById('fenStatus');
    elements.fenHint = document.getElementById('fenHint');
    elements.analyzeWhiteBtn = document.getElementById('analyzeWhiteBtn');
    elements.analyzeBlackBtn = document.getElementById('analyzeBlackBtn');
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
    elements.boardActions = document.getElementById('boardActions');
    elements.capturedBoardReference = document.getElementById('capturedBoardReference');
    elements.capturedBoardImage = document.getElementById('capturedBoardImage');
    elements.editModeControls = document.getElementById('editModeControls');
    elements.piecePalette = document.getElementById('piecePalette');
    elements.resetBoardBtn = document.getElementById('resetBoardBtn');
    elements.doneEditingBtn = document.getElementById('doneEditingBtn');
    
    // Game check elements
    elements.findGamesSection = document.getElementById('findGamesSection');
    elements.checkGamesBtn = document.getElementById('checkGamesBtn');
    elements.gamesCheckStatus = document.getElementById('gamesCheckStatus');
    elements.gamesFoundIcons = document.getElementById('gamesFoundIcons');
    elements.noGamesFound = document.getElementById('noGamesFound');
    elements.whiteGameIcon = document.getElementById('whiteGameIcon');
    elements.blackGameIcon = document.getElementById('blackGameIcon');
    
    // Lichess elements
    elements.lichessWhiteBtn = document.getElementById('lichessWhiteBtn');
    elements.lichessBlackBtn = document.getElementById('lichessBlackBtn');
    
    // Flip button
    elements.flipBoardBtn = document.getElementById('flipBoardBtn');
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
    elements.retryBtn.addEventListener('click', resetToInput);
    
    // Board editing
    elements.editBoardBtn.addEventListener('click', enterEditMode);
    elements.doneEditingBtn.addEventListener('click', exitEditMode);
    elements.resetBoardBtn.addEventListener('click', resetBoard);
    elements.flipBoardBtn.addEventListener('click', flipBoard);
    
    // Check for games
    // Game check is now automatic - no button event needed
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
    // Auto-analyze immediately after setting image
    analyzeBoard();
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
            video: { 
                facingMode: state.facingMode,
                // Request high resolution - browser will use closest available
                width: { ideal: 1920, min: 1280 },
                height: { ideal: 1920, min: 1280 }
            },
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
        
        // Get base64 - use high quality JPEG (0.95)
        const base64 = canvas.toDataURL('image/jpeg', 0.95).split(',')[1];
        console.log(`[CAPTURE] Video: ${vw}x${vh}, Captured: ${CONFIG.CAPTURE_SIZE}x${CONFIG.CAPTURE_SIZE}`);
        
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
// API Fetch with Timeout and Auth Handling
// ============================================

async function fetchWithAuth(url, options = {}, timeoutMs = 30000) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    
    try {
        const response = await fetch(url, {
            ...options,
            signal: controller.signal,
            credentials: 'include'
        });
        clearTimeout(timeout);
        
        // Check for auth errors
        if (response.status === 401 || response.status === 403) {
            const isAuth = await checkAuthStatus();
            if (!isAuth) {
                // Session expired - redirect to login
                showAuthExpiredError();
                throw new Error('Session expired. Please log in again.');
            }
        }
        
        return response;
    } catch (error) {
        clearTimeout(timeout);
        if (error.name === 'AbortError') {
            // Check if it's an auth issue causing the hang
            const isAuth = await checkAuthStatus();
            if (!isAuth) {
                showAuthExpiredError();
                throw new Error('Session expired. Please log in again.');
            }
            throw new Error('Request timed out. Please try again.');
        }
        throw error;
    }
}

function showAuthExpiredError() {
    const errorMsg = 'Your session has expired. Click OK to log in again.';
    if (confirm(errorMsg)) {
        // Redirect to login - Azure Easy Auth handles this
        window.location.href = '/.auth/login/aad?post_login_redirect_uri=' + encodeURIComponent(window.location.pathname);
    }
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

    // Show the image in the loading section
    elements.loadingPreviewImage.src = `data:image/jpeg;base64,${state.imageBase64}`;

    showSection('loading');

    try {
        const skipDetection = elements.skipDetection.checked;

        const queryParams = ['debug=true'];  // Always request debug to get warped image
        if (skipDetection) queryParams.push('skip_detection=true');
        const query = '?' + queryParams.join('&');
        
        const url = `${CONFIG.API_BASE}/predict${query}`;
        console.log('Calling API:', url);
        
        const response = await fetchWithAuth(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: state.imageBase64 })
        }, 60000);  // 60 second timeout for prediction
        
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
            
            // Store questionable squares (e.g., potential queen color confusion)
            state.questionableSquares = result.questionable_squares || result.questionableSquares || null;
            
            // Store warped image for feedback (if available from debug response)
            const debugImages = result.debug_images || result.debugImages;
            if (debugImages && debugImages.warped) {
                state.warpedImageBase64 = debugImages.warped;
                console.log('Stored warped image for feedback');
            } else {
                state.warpedImageBase64 = null;
            }
            
            await displayResults();
        } else {
            // Show user-friendly error message
            let errorMsg = error || 'Board not detected';
            if (errorMsg.toLowerCase().includes('timeout') || errorMsg.toLowerCase().includes('no board')) {
                errorMsg = 'Could not detect chess board. Try capturing a clearer image with better lighting and focus.';
            } else if (!error) {
                errorMsg = 'Board not detected. Try "skip detection" if your image is already cropped to just the board.';
            }
            throw new Error(errorMsg);
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
        const query = skipDetection ? '?skip_detection=true' : '';
        
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
            await displayResults();
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

async function displayResults() {
    // Validate questionable pieces using Chess.com before displaying
    if (state.questionableSquares && state.questionableSquares.length > 0) {
        console.log('displayResults: Validating questionable pieces via Chess.com...');
        const resolved = await validateQuestionablePieces();
        if (resolved) {
            console.log('displayResults: Questionable pieces resolved via Chess.com validation');
        }
    }
    
    // Render board
    renderBoard();

    // Validate FEN
    const validation = validateFen(state.resultFen);
    if (validation.isValid) {
        elements.fenStatus.textContent = '✓ Valid position';
        elements.fenStatus.className = 'fen-status valid';
        elements.fenHint.hidden = true;
    } else {
        elements.fenStatus.textContent = `⚠ ${validation.issues[0] || 'Invalid position'}`;
        elements.fenStatus.className = 'fen-status invalid';
        elements.fenHint.hidden = false;
    }

    // Set Chess.com links
    const castling = inferCastlingRights();
    const whiteFen = `${state.resultFen} w ${castling} - 0 1`;
    const blackFen = `${state.resultFen} b ${castling} - 0 1`;
    
    // Set Chess.com analysis links
    elements.analyzeWhiteBtn.href = `${CONFIG.CHESS_COM_ANALYSIS_URL}?fen=${encodeURIComponent(whiteFen)}&flip=false&tab=analysis`;
    elements.analyzeBlackBtn.href = `${CONFIG.CHESS_COM_ANALYSIS_URL}?fen=${encodeURIComponent(blackFen)}&flip=true&tab=analysis`;
    
    // Set Lichess links
    elements.lichessWhiteBtn.href = `https://lichess.org/editor/${whiteFen.replace(/ /g, '_')}`;
    elements.lichessBlackBtn.href = `https://lichess.org/editor/${blackFen.replace(/ /g, '_')}`;
    
    // Auto-check for games
    checkForGames();

    // Ensure edit mode is hidden initially
    elements.capturedBoardReference.hidden = true;
    elements.editModeControls.hidden = true;
    elements.boardActions.hidden = false;

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
    
    // Build a set of questionable square names for quick lookup
    const questionableSquares = new Set();
    if (state.questionableSquares) {
        state.questionableSquares.forEach(q => questionableSquares.add(q.square));
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
            
            // Check if this square is questionable (potential color confusion)
            const squareName = `${'abcdefgh'[col]}${8-row}`;
            if (questionableSquares.has(squareName)) {
                td.classList.add('questionable');
                const qInfo = state.questionableSquares.find(q => q.square === squareName);
                td.title = qInfo?.reason || 'Questionable piece color';
            }
            // Highlight low-confidence predictions (< 90%)
            else {
                const confidence = state.confidences?.[row]?.[col];
                if (confidence !== undefined && confidence < 0.90 && piece !== 'empty') {
                    td.classList.add('low-confidence');
                    td.title = `Confidence: ${(confidence * 100).toFixed(0)}%`;
                }
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

// Progress animation state
let progressInterval = null;
const PROGRESS_STAGES = ['detect', 'warp', 'classify', 'validate'];
const STAGE_TIMINGS = [800, 1200, 2500, 500]; // Estimated time for each stage in ms

function startProgressAnimation() {
    stopProgressAnimation();
    
    const stages = document.querySelectorAll('#progressStages .stage');
    let currentStage = 0;
    
    // Reset all stages
    stages.forEach(s => {
        s.classList.remove('active', 'done');
    });
    
    // Start with first stage active
    if (stages[0]) stages[0].classList.add('active');
    
    let elapsed = 0;
    progressInterval = setInterval(() => {
        elapsed += 100;
        
        // Check if we should move to next stage
        let totalTime = 0;
        for (let i = 0; i <= currentStage && i < STAGE_TIMINGS.length; i++) {
            totalTime += STAGE_TIMINGS[i];
        }
        
        if (elapsed > totalTime && currentStage < stages.length - 1) {
            // Mark current as done
            stages[currentStage].classList.remove('active');
            stages[currentStage].classList.add('done');
            
            // Move to next
            currentStage++;
            stages[currentStage].classList.add('active');
            
            // Update status text
            const statusEl = document.getElementById('loadingStatus');
            if (statusEl) {
                const statusTexts = [
                    'Detecting board...',
                    'Warping to square...',
                    'Classifying pieces...',
                    'Validating position...'
                ];
                statusEl.textContent = statusTexts[currentStage] || 'Analyzing...';
            }
        }
    }, 100);
}

function stopProgressAnimation() {
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
}

function showSection(section) {
    elements.inputSection.hidden = section !== 'input';
    elements.loadingSection.hidden = section !== 'loading';
    elements.resultsSection.hidden = section !== 'results';
    elements.errorSection.hidden = section !== 'error';
    
    if (section === 'loading') {
        startProgressAnimation();
    } else {
        stopProgressAnimation();
    }
    
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
    state.imageBase64 = null;
    state.resultFen = null;
    state.originalFen = null;
    state.board = null;
    state.originalBoard = null;
    state.corrections = {};
    state.editMode = false;

    elements.capturedBoardReference.hidden = true;
    elements.editModeControls.hidden = true;
    elements.boardActions.hidden = false;
    elements.fileInput.value = '';

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

function enterEditMode() {
    state.editMode = true;
    state.selectedPiece = 'empty';

    // Show reference image above the board
    const imageToShow = state.warpedImageBase64 || state.imageBase64;
    if (imageToShow) {
        elements.capturedBoardImage.src = `data:image/jpeg;base64,${imageToShow}`;
    }
    elements.capturedBoardReference.hidden = false;

    // Show edit controls below the board, hide the edit button
    elements.editModeControls.hidden = false;
    elements.boardActions.hidden = true;

    // Update palette selection and render board in edit mode
    updatePaletteSelection();
    renderBoard();
}

function exitEditMode() {
    state.editMode = false;

    // Hide reference image and edit controls
    elements.capturedBoardReference.hidden = true;
    elements.editModeControls.hidden = true;
    elements.boardActions.hidden = false;

    // Re-render board in normal mode
    renderBoard();

    // Submit corrections silently if any were made
    if (Object.keys(state.corrections).length > 0) {
        submitCorrections();
    }
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

    // Validate FEN
    const validation = validateFen(state.resultFen);
    if (validation.isValid) {
        elements.fenStatus.textContent = '✓ Valid position';
        elements.fenStatus.className = 'fen-status valid';
        elements.fenHint.hidden = true;
    } else {
        elements.fenStatus.textContent = `⚠ ${validation.issues[0] || 'Invalid position'}`;
        elements.fenStatus.className = 'fen-status invalid';
        elements.fenHint.hidden = false;
    }

    // Update Chess.com links
    const castling = inferCastlingRights();
    const whiteFen = `${state.resultFen} w ${castling} - 0 1`;
    const blackFen = `${state.resultFen} b ${castling} - 0 1`;
    elements.analyzeWhiteBtn.href = `${CONFIG.CHESS_COM_ANALYSIS_URL}?fen=${encodeURIComponent(whiteFen)}&flip=false&tab=analysis`;
    elements.analyzeBlackBtn.href = `${CONFIG.CHESS_COM_ANALYSIS_URL}?fen=${encodeURIComponent(blackFen)}&flip=true&tab=analysis`;
    elements.lichessWhiteBtn.href = `https://lichess.org/editor/${whiteFen.replace(/ /g, '_')}`;
    elements.lichessBlackBtn.href = `https://lichess.org/editor/${blackFen.replace(/ /g, '_')}`;
    
    // Re-render board only
    renderBoard();

    // Reset game search results (user can click Search button)
    if (elements.gamesCheckStatus) elements.gamesCheckStatus.hidden = true;
    if (elements.gamesFoundIcons) elements.gamesFoundIcons.hidden = true;
    if (elements.noGamesFound) elements.noGamesFound.hidden = true;
}

async function resetBoard() {
    state.board = JSON.parse(JSON.stringify(state.originalBoard));
    state.resultFen = state.originalFen;
    state.corrections = {};

    // Stay in edit mode but show reset board
    renderBoard();

    // Update validation status
    const validation = validateFen(state.resultFen);
    if (validation.isValid) {
        elements.fenStatus.textContent = '✓ Valid position';
        elements.fenStatus.className = 'fen-status valid';
        elements.fenHint.hidden = true;
    } else {
        elements.fenStatus.textContent = `⚠ ${validation.issues[0] || 'Invalid position'}`;
        elements.fenStatus.className = 'fen-status invalid';
        elements.fenHint.hidden = false;
    }

    // Update Chess.com links
    const castling = inferCastlingRights();
    const whiteFen = `${state.resultFen} w ${castling} - 0 1`;
    const blackFen = `${state.resultFen} b ${castling} - 0 1`;
    elements.analyzeWhiteBtn.href = `${CONFIG.CHESS_COM_ANALYSIS_URL}?fen=${encodeURIComponent(whiteFen)}&flip=false&tab=analysis`;
    elements.analyzeBlackBtn.href = `${CONFIG.CHESS_COM_ANALYSIS_URL}?fen=${encodeURIComponent(blackFen)}&flip=true&tab=analysis`;
    elements.lichessWhiteBtn.href = `https://lichess.org/editor/${whiteFen.replace(/ /g, '_')}`;
    elements.lichessBlackBtn.href = `https://lichess.org/editor/${blackFen.replace(/ /g, '_')}`;

    showToast('Board reset to original');
}

function flipBoard() {
    // Flip the board 180 degrees - for diagrams shown from black's perspective
    // This rotates the entire position so white pieces are at the bottom
    
    const newBoard = [];
    for (let row = 0; row < 8; row++) {
        newBoard[row] = [];
        for (let col = 0; col < 8; col++) {
            // Map from (row, col) to (7-row, 7-col)
            newBoard[row][col] = state.board[7 - row][7 - col];
        }
    }
    
    state.board = newBoard;
    state.resultFen = boardToFen(state.board);
    state.corrections = {};
    
    // Re-render board
    renderBoard();

    // Update validation status
    const validation = validateFen(state.resultFen);
    if (validation.isValid) {
        elements.fenStatus.textContent = '✓ Valid position';
        elements.fenStatus.className = 'fen-status valid';
        elements.fenHint.hidden = true;
    } else {
        elements.fenStatus.textContent = `⚠ ${validation.issues[0] || 'Invalid position'}`;
        elements.fenStatus.className = 'fen-status invalid';
        elements.fenHint.hidden = false;
    }

    // Update links
    const castling = inferCastlingRights();
    const whiteFen = `${state.resultFen} w ${castling} - 0 1`;
    const blackFen = `${state.resultFen} b ${castling} - 0 1`;
    elements.analyzeWhiteBtn.href = `${CONFIG.CHESS_COM_ANALYSIS_URL}?fen=${encodeURIComponent(whiteFen)}&flip=false&tab=analysis`;
    elements.analyzeBlackBtn.href = `${CONFIG.CHESS_COM_ANALYSIS_URL}?fen=${encodeURIComponent(blackFen)}&flip=true&tab=analysis`;
    elements.lichessWhiteBtn.href = `https://lichess.org/editor/${whiteFen.replace(/ /g, '_')}`;
    elements.lichessBlackBtn.href = `https://lichess.org/editor/${blackFen.replace(/ /g, '_')}`;

    showToast('Board flipped 180°');
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
        return; // No corrections to submit
    }

    // Validate state
    if (!state.originalFen || !state.resultFen) {
        console.warn('Cannot submit corrections: missing FEN data');
        return;
    }

    try {
        // Transform corrections to simple {square: piece} format for API
        const correctedSquares = {};
        for (const [key, correction] of Object.entries(state.corrections)) {
            correctedSquares[correction.square] = correction.to;
        }

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

        const response = await fetch(`${CONFIG.API_BASE}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            showToast('Corrections saved');
            state.originalFen = state.resultFen;
            state.originalBoard = JSON.parse(JSON.stringify(state.board));
            state.corrections = {};
        } else {
            console.error('Submit corrections failed:', response.status);
        }
    } catch (error) {
        console.error('Submit corrections error:', error);
    }
}

// ============================================
// Chess.com Game Check
// ============================================

/**
 * Validate questionable pieces by checking which FEN variant matches games on Chess.com.
 * If one variant has games and another doesn't, we update the board to match.
 */
async function validateQuestionablePieces() {
    if (!state.questionableSquares || state.questionableSquares.length === 0) {
        console.log('validateQuestionablePieces: No questionable squares');
        return false;
    }
    
    // Only validate if the current FEN is valid
    const fenValidation = validateFen(state.resultFen);
    if (!fenValidation.isValid) {
        console.log('validateQuestionablePieces: FEN is invalid, skipping Chess.com validation');
        return false;
    }
    
    // Find queen color issues
    const queenIssues = state.questionableSquares.filter(q => 
        q.reason && q.reason.toLowerCase().includes('queen') && 
        (q.piece === 'wQ' || q.piece === 'bQ')
    );
    
    if (queenIssues.length === 0) {
        console.log('validateQuestionablePieces: No queen color issues');
        return false;
    }
    
    console.log('validateQuestionablePieces: Found queen issues:', queenIssues);
    
    // Generate current FEN and alternative FEN(s) by swapping queen colors
    const castling = inferCastlingRights();
    const currentFen = `${state.resultFen} w ${castling} - 0 1`;
    
    // Create alternative board by swapping questionable queen colors
    const alternativeBoard = JSON.parse(JSON.stringify(state.board));
    for (const issue of queenIssues) {
        const col = issue.square.charCodeAt(0) - 97; // 'a' = 0
        const row = 8 - parseInt(issue.square[1]);   // '8' = 0
        
        if (alternativeBoard[row] && alternativeBoard[row][col]) {
            const currentPiece = alternativeBoard[row][col];
            if (currentPiece === 'wQ') {
                alternativeBoard[row][col] = 'bQ';
            } else if (currentPiece === 'bQ') {
                alternativeBoard[row][col] = 'wQ';
            }
        }
    }
    
    // Generate alternative FEN
    const alternativeFenParts = [];
    for (let row = 0; row < 8; row++) {
        let emptyCount = 0;
        let rowFen = '';
        for (let col = 0; col < 8; col++) {
            const piece = alternativeBoard[row][col];
            if (piece === 'empty' || !piece) {
                emptyCount++;
            } else {
                if (emptyCount > 0) {
                    rowFen += emptyCount;
                    emptyCount = 0;
                }
                rowFen += pieceToFenChar(piece);
            }
        }
        if (emptyCount > 0) rowFen += emptyCount;
        alternativeFenParts.push(rowFen);
    }
    const alternativeResultFen = alternativeFenParts.join('/');
    const alternativeFen = `${alternativeResultFen} w ${castling} - 0 1`;
    
    console.log('validateQuestionablePieces: Checking variants...');
    console.log('  Current:', currentFen);
    console.log('  Alternative:', alternativeFen);
    
    try {
        const response = await fetch(`${CONFIG.API_BASE}/validate-fen-variants`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                fenVariants: [
                    { fen: currentFen, label: 'current' },
                    { fen: alternativeFen, label: 'swapped' }
                ]
            })
        });
        
        if (!response.ok) {
            console.error('validateQuestionablePieces: API error:', response.status);
            return false;
        }
        
        const result = await response.json();
        console.log('validateQuestionablePieces: Result:', result);
        
        if (!result.success) {
            console.error('validateQuestionablePieces: Validation failed:', result.error);
            return false;
        }
        
        const currentResult = result.variants?.find(v => v.label === 'current');
        const swappedResult = result.variants?.find(v => v.label === 'swapped');
        
        // If swapped has games and current doesn't, switch to swapped
        if (swappedResult?.gamesFound && !currentResult?.gamesFound) {
            console.log('validateQuestionablePieces: Swapped variant matches games, updating board!');
            
            // Update the board with the alternative
            state.board = alternativeBoard;
            state.resultFen = alternativeResultFen;
            
            // Clear questionable squares since we resolved them
            state.questionableSquares = null;
            
            return true; // Board was updated
        } else if (currentResult?.gamesFound && !swappedResult?.gamesFound) {
            console.log('validateQuestionablePieces: Current variant matches games, keeping it');
            // Current is correct, clear questionable status
            state.questionableSquares = null;
            return true; // Resolved (no change needed)
        } else {
            console.log('validateQuestionablePieces: Ambiguous result, keeping current:', result.recommendation);
            return false; // Couldn't resolve
        }
        
    } catch (error) {
        console.error('validateQuestionablePieces: Error:', error);
        return false;
    }
}

async function checkForGames() {
    if (!state.resultFen) {
        console.log('checkForGames: No resultFen, skipping');
        return;
    }
    
    console.log('checkForGames: Starting check for FEN:', state.resultFen);
    
    const castling = inferCastlingRights();
    const whiteFen = `${state.resultFen} w ${castling} - 0 1`;
    const blackFen = `${state.resultFen} b ${castling} - 0 1`;
    
    // Show checking state
    if (elements.gamesCheckStatus) {
        elements.gamesCheckStatus.hidden = false;
    }
    if (elements.gamesFoundIcons) {
        elements.gamesFoundIcons.hidden = true;
    }
    if (elements.noGamesFound) {
        elements.noGamesFound.hidden = true;
    }
    if (elements.whiteGameIcon) {
        elements.whiteGameIcon.style.display = 'none';
    }
    if (elements.blackGameIcon) {
        elements.blackGameIcon.style.display = 'none';
    }
    
    // Set hrefs
    if (elements.whiteGameIcon) {
        elements.whiteGameIcon.href = `${CONFIG.CHESS_COM_SEARCH_URL}?fen=${encodeURIComponent(whiteFen)}`;
    }
    if (elements.blackGameIcon) {
        elements.blackGameIcon.href = `${CONFIG.CHESS_COM_SEARCH_URL}?fen=${encodeURIComponent(blackFen)}`;
    }
    
    try {
        console.log('checkForGames: Calling API...');
        const [whiteResult, blackResult] = await Promise.all([
            checkChessCom(whiteFen),
            checkChessCom(blackFen)
        ]);
        
        console.log('checkForGames: API results - white:', whiteResult, 'black:', blackResult);
        
        const whiteFound = whiteResult.gamesFound || whiteResult.GamesFound;
        const blackFound = blackResult.gamesFound || blackResult.GamesFound;
        
        console.log('checkForGames: whiteFound:', whiteFound, 'blackFound:', blackFound);
        
        // Hide checking status
        if (elements.gamesCheckStatus) {
            elements.gamesCheckStatus.hidden = true;
        }
        
        if (whiteFound || blackFound) {
            // Show icons container and only the icons with results
            if (elements.gamesFoundIcons) {
                elements.gamesFoundIcons.hidden = false;
            }
            if (elements.whiteGameIcon) {
                elements.whiteGameIcon.style.display = whiteFound ? 'flex' : 'none';
            }
            if (elements.blackGameIcon) {
                elements.blackGameIcon.style.display = blackFound ? 'flex' : 'none';
            }
            if (elements.noGamesFound) {
                elements.noGamesFound.hidden = true;
            }
        } else {
            // No games found
            if (elements.gamesFoundIcons) {
                elements.gamesFoundIcons.hidden = true;
            }
            if (elements.noGamesFound) {
                elements.noGamesFound.hidden = false;
            }
        }
        
    } catch (error) {
        console.error('checkForGames error:', error);
        if (elements.gamesCheckStatus) {
            elements.gamesCheckStatus.hidden = true;
        }
        if (elements.noGamesFound) {
            elements.noGamesFound.hidden = false;
            const span = elements.noGamesFound.querySelector('span');
            if (span) {
                span.textContent = 'Error checking games';
            }
        }
    }
}

async function checkChessCom(fen) {
    const encodedFen = encodeURIComponent(fen);
    const response = await fetch(`${CONFIG.DOTNET_API_BASE}/chess-com-search?fen=${encodedFen}`);

    if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
}

// Check if user is admin and show admin link
async function checkAdminAccess() {
    try {
        const response = await fetch(`${CONFIG.API_BASE}/admin/statistics`, {
            credentials: 'include'
        });

        if (response.ok) {
            // User has admin access, show the admin link
            const adminLink = document.getElementById('adminLink');
            if (adminLink) {
                adminLink.style.display = 'block';
            }
        }
    } catch (error) {
        // Silently fail - user is not admin
        console.log('Not an admin user');
    }
}

// Check admin access on page load
checkAdminAccess();

// ============================================
// Version Display
// ============================================

async function loadVersion() {
    const versionEl = document.getElementById('versionInfo');
    if (!versionEl) return;

    try {
        const response = await fetch(`${CONFIG.API_BASE}/version`);
        if (response.ok) {
            const versionData = await response.json();
            const buildDate = new Date(versionData.timestamp);
            const formattedDate = buildDate.toLocaleString('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });

            versionEl.textContent = `v${versionData.version} (${versionData.build})`;
            versionEl.title = `Version: ${versionData.version}\nBuild: ${versionData.build}\nDeployed: ${formattedDate}\nCommit: ${versionData.commit}`;
        } else {
            console.error('Version fetch failed:', response.status, response.statusText);
            versionEl.textContent = 'Version unavailable';
            versionEl.title = `Failed to load version: ${response.status}`;
        }
    } catch (error) {
        console.error('Could not load version info:', error);
        versionEl.textContent = 'Version unavailable';
        versionEl.title = `Error loading version: ${error.message}`;
    }
}

// Load version on page load
loadVersion();
