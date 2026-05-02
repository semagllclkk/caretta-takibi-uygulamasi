/**
 * Caretta Track - Frontend Application (Prediction Only)
 * Vanilla JavaScript (ES6+) + Fetch API
 * 
 * Focus: Kaplumbağa Tanı (Prediction)
 * - Dosya yükleme (drag-drop)
 * - API tahmin isteği
 * - Sonuç gösterimi
 */

// =========================================================================
// Konfigürasyon
// =========================================================================

const CONFIG = {
    API_URL: 'http://localhost:8000',
    ENDPOINTS: {
        HEALTH: '/api/health',
        PREDICT: '/api/predict',
    },
    TIMEOUT: 300000, // 5 dakika
    MAX_FILE_SIZE: 50 * 1024 * 1024, // 50MB
};

// =========================================================================
// State Management
// =========================================================================

const state = {
    isPredicting: false,
    selectedImageFile: null,
};

// =========================================================================
// API Communication Layer
// =========================================================================

/**
 * API'nin sağlık durumunu kontrol eder
 */
async function checkApiHealth() {
    try {
        const response = await fetch(`${CONFIG.API_URL}${CONFIG.ENDPOINTS.HEALTH}`, {
            method: 'GET',
            timeout: 5000,
        });
        return response.ok;
    } catch (error) {
        console.error('API Health Check failed:', error);
        return false;
    }
}

/**
 * API'nin sağlık durumuna göre UI'yi güncelle
 */
async function updateApiStatus() {
    const isOnline = await checkApiHealth();
    const statusBadge = document.getElementById('apiStatus');
    
    if (isOnline) {
        statusBadge.textContent = '✓ API Online';
        statusBadge.classList.remove('offline');
        statusBadge.classList.add('online');
    } else {
        statusBadge.textContent = '✗ API Offline';
        statusBadge.classList.remove('online');
        statusBadge.classList.add('offline');
    }
}

/**
 * Tahmin isteği gönderir
 */
async function predictTurtle(imageFile) {
    try {
        const formData = new FormData();
        formData.append('file', imageFile);

        const response = await fetch(`${CONFIG.API_URL}${CONFIG.ENDPOINTS.PREDICT}`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Prediction error:', error);
        throw error;
    }
}

// =========================================================================
// UI State Management
// =========================================================================

/**
 * Loading animasyonunu gösterir
 */
function showPredictionLoading() {
    const loading = document.getElementById('predictionLoading');
    const results = document.getElementById('predictionResults');
    const error = document.getElementById('predictionError');
    const btn = document.getElementById('predictBtn');

    loading.classList.remove('hidden');
    results.classList.add('hidden');
    error.classList.add('hidden');
    btn.disabled = true;
    state.isPredicting = true;
}

/**
 * Loading animasyonunu gizler
 */
function hidePredictionLoading() {
    const loading = document.getElementById('predictionLoading');
    const btn = document.getElementById('predictBtn');

    loading.classList.add('hidden');
    btn.disabled = false;
    state.isPredicting = false;
}

/**
 * Hata mesajını gösterir
 */
function showPredictionError(errorMessage) {
    const errorDiv = document.getElementById('predictionError');
    const errorContent = document.getElementById('predictionErrorContent');
    const results = document.getElementById('predictionResults');

    results.classList.add('hidden');
    errorContent.textContent = errorMessage;
    errorDiv.classList.remove('hidden');
}

// =========================================================================
// Result Rendering
// =========================================================================

/**
 * Tahmin sonuçlarını ekranda gösterir (başarılı veya başarısız)
 */
function displayPredictionResults(data) {
    const resultsContent = document.getElementById('predictionResultsContent');
    const results = document.getElementById('predictionResults');

    // Confidence değeri (başarılı/başarısız her durumda)
    const confidence = data.confidence || 0;
    const confidencePercent = (confidence * 100).toFixed(1);

    // Başarısız yanıt kontrolü: success: false ise hata kartı göster
    if (!data.success || confidence < 0.60) {
        console.warn('Tahmin sorunu:', {
            success: data.success,
            confidence: confidencePercent,
            stage_reached: data.stage_reached,
            error_message: data.error_message,
        });

        let html = '<div class="prediction-error-card">';
        
        // Güven skoru başlık
        html += '<div class="confidence-display">';
        html += `<p class="confidence-value">${confidencePercent}%</p>`;
        html += '<p class="confidence-label">Düşük Güven Skoru</p>';
        html += '</div>';
        
        html += '<div class="error-header">⚠️ İşlem Başarısız veya Düşük Güven</div>';
        
        // Aşama bilgisi (başarısız ise)
        if (!data.success) {
            let stageName = data.stage_reached || 'unknown';
            const stageMap = {
                'validation': '🔒 Güvenlik Kontrolü',
                'research': '🔍 Kalite Analizi',
                'prediction': '🤖 Yapay Zeka Tahmini',
                'complete': '✓ Tamamlandı',
            };
            html += `<div class="error-stage">${stageMap[stageName] || stageName}</div>`;
        }
        
        // Açıklayıcı mesaj
        let explanation = data.error_message || 'Model bu fotoğrafta yeterince emin değildir.';
        if (confidence < 0.60 && !data.error_message) {
            explanation = `Model %${confidencePercent} güven oranıyla tahmin yapmıştır. Bu çok düşük bir orandır. Lütfen daha iyi kalitede bir fotoğraf deneyin.`;
        }
        html += `<div class="error-message">${explanation}</div>`;
        
        // Güvenlik hatasının detayları
        if (data.security_report && data.security_report.results && data.security_report.results.length > 0) {
            const failedValidators = data.security_report.results.filter(r => !r.passed);
            if (failedValidators.length > 0) {
                html += '<div class="error-details">';
                html += '<strong>🔒 Kontrol Hataları:</strong>';
                failedValidators.forEach(validator => {
                    html += `<p>• ${validator.validator_name}: ${validator.reason}</p>`;
                });
                html += '</div>';
            }
        }
        
        // Kalite hatasının detayları
        if (data.research_result && data.research_result.issues && data.research_result.issues.length > 0) {
            html += '<div class="error-details">';
            html += '<strong>📊 Kalite Sorunları:</strong>';
            data.research_result.issues.forEach(issue => {
                html += `<p>• ${issue}</p>`;
            });
            html += '</div>';
        }
        
        html += '</div>';

        resultsContent.innerHTML = html;
        results.classList.remove('hidden');
        return;
    }

    // Başarılı yanıt: normal sonuç göster
    let html = '<div class="prediction-result">';
    
    // Güven skoru başında göster
    html += '<div class="confidence-display">';
    html += `<p class="confidence-value">${confidencePercent}%</p>`;
    html += '<p class="confidence-label">Yüksek Güven Skoru</p>';
    html += '</div>';

    // Bireyin durumu (yeni veya tanınan)
    const isNew = data.is_new_turtle;
    const statusClass = isNew ? 'new-turtle' : 'known-turtle';
    const statusText = isNew ? '🆕 YENİ BİREY' : '✓ TANINAN BİREY';

    html += `<div class="turtle-status ${statusClass}">${statusText}</div>`;

    // Kaplumbağa ID'si
    if (!isNew && data.turtle_id) {
        html += `<div class="turtle-id">${data.turtle_id}</div>`;
    }

    // Ek bilgiler
    html += `
        <div style="margin-top: var(--spacing-lg); padding-top: var(--spacing-lg); border-top: 1px solid var(--medium-gray);">
            <p style="font-size: 0.9rem; color: var(--text-light); margin: 0;">
                ${isNew 
                    ? 'Bu kaplumbağa daha önce sisteme kaydedilmemiş bir birey görünüyor.' 
                    : 'Bu kaplumbağa daha önce kaydedilmiş bir birey olarak tanımlandı.'}
            </p>
        </div>
    `;

    html += '</div>';

    resultsContent.innerHTML = html;
    results.classList.remove('hidden');
}

// =========================================================================
// Image Upload Handling
// =========================================================================

/**
 * Dosya yükleme validasyonu
 */
function validateImageFile(file) {
    // Tip kontrolü
    if (!file.type.startsWith('image/')) {
        throw new Error('Lütfen bir görsel dosyası seçin (JPG, PNG, WebP).');
    }

    // Boyut kontrolü
    if (file.size > CONFIG.MAX_FILE_SIZE) {
        throw new Error(`Dosya çok büyük. Maksimum boyut: ${CONFIG.MAX_FILE_SIZE / (1024 * 1024)}MB`);
    }
}

/**
 * Seçilen görsel dosyasını işler
 */
function handleImageSelection(file) {
    try {
        validateImageFile(file);

        state.selectedImageFile = file;

        // Önizlemeyi göster
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = document.getElementById('previewImage');
            const fileName = document.getElementById('previewFileName');
            const uploadArea = document.getElementById('uploadArea');
            const preview = document.getElementById('imagePreview');
            const btn = document.getElementById('predictBtn');

            img.src = e.target.result;
            fileName.textContent = `📄 ${file.name}`;

            uploadArea.classList.add('hidden');
            preview.classList.remove('hidden');
            btn.classList.remove('hidden');
        };
        reader.readAsDataURL(file);
    } catch (error) {
        showPredictionError(error.message);
    }
}

/**
 * Görsel seçimini sıfırla
 */
function resetImageSelection() {
    state.selectedImageFile = null;

    const uploadArea = document.getElementById('uploadArea');
    const preview = document.getElementById('imagePreview');
    const btn = document.getElementById('predictBtn');
    const imageInput = document.getElementById('imageInput');

    uploadArea.classList.remove('hidden');
    preview.classList.add('hidden');
    btn.classList.add('hidden');
    imageInput.value = '';

    document.getElementById('predictionResults').classList.add('hidden');
    document.getElementById('predictionError').classList.add('hidden');
}

// =========================================================================
// Event Listeners Setup
// =========================================================================

/**
 * Tahmin butonu event listener'ı
 */
function setupPredictionButton() {
    const btn = document.getElementById('predictBtn');

    btn.addEventListener('click', async () => {
        if (!state.selectedImageFile) {
            showPredictionError('Lütfen önce bir görsel seçin.');
            return;
        }

        showPredictionLoading();

        try {
            const result = await predictTurtle(state.selectedImageFile);
            displayPredictionResults(result);
        } catch (error) {
            showPredictionError(
                `Tahmin başarısız oldu: ${error.message || 'Bilinmeyen hata'}`
            );
        } finally {
            hidePredictionLoading();
        }
    });
}

/**
 * Görsel yükleme alanı event listener'ları
 */
function setupImageUploadArea() {
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    const changeImageBtn = document.getElementById('changeImageBtn');

    // Dosya seçim
    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleImageSelection(e.target.files[0]);
        }
    });

    // Upload alanına tıklama
    uploadArea.addEventListener('click', () => {
        imageInput.click();
    });

    // Sürükle-bırak
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');

        if (e.dataTransfer.files.length > 0) {
            handleImageSelection(e.dataTransfer.files[0]);
        }
    });

    // Görsel değiştir butonu
    changeImageBtn.addEventListener('click', resetImageSelection);
}

/**
 * Sonuç kapatma butonları
 */
function setupCloseButtons() {
    // Tahmin sonuçlarını kapat
    document.getElementById('closePredictionResults').addEventListener('click', () => {
        document.getElementById('predictionResults').classList.add('hidden');
    });

    // Tahmin hatasını kapat
    document.getElementById('closePredictionError').addEventListener('click', () => {
        document.getElementById('predictionError').classList.add('hidden');
    });
}

/**
 * Fotoğraf ipuçları accordion'u
 */
function setupTipsToggle() {
    const tipsToggle = document.getElementById('tipsToggle');
    const tipsContent = document.getElementById('tipsContent');

    tipsToggle.addEventListener('click', () => {
        tipsToggle.classList.toggle('active');
        tipsContent.classList.toggle('active');
    });
}

// =========================================================================
// Initialization
// =========================================================================

/**
 * Sayfanın tüm bileşenlerini başlatır
 */
function initializeApp() {
    console.log('🚀 Caretta Track uygulaması başlatılıyor...');

    // API durum kontrolü
    updateApiStatus();
    setInterval(updateApiStatus, 30000); // Her 30 saniyede kontrol et

    // Event listener'ları kur
    setupPredictionButton();
    setupImageUploadArea();
    setupCloseButtons();
    setupTipsToggle();

    console.log('✓ Uygulama hazır!');
}

/**
 * Sayfa yüklendiğinde uygulamayı başlat
 */
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
} else {
    initializeApp();
}

// =========================================================================
// Debug Mode
// =========================================================================

if (typeof window !== 'undefined') {
    window.CarettaTrackDebug = {
        checkApiHealth,
        predictTurtle,
        state,
        CONFIG,
    };
}
