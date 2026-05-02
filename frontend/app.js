/**
 * Caretta Track - Frontend Application
 * Vanilla JavaScript (ES6+) + Fetch API
 * 
 * SOLID Principles:
 *   - SRP: Her fonksiyon tek bir sorumluluğu var
 *   - DRY: Tekrarlayan kod minimize edildi
 *   - Clean Code: Açık, okunması kolay, well-documented
 */

// =========================================================================
// Konfigürasyon
// =========================================================================

const CONFIG = {
    API_URL: 'http://127.0.0.1:8000',
    ENDPOINTS: {
        HEALTH: '/api/health',
        TRAIN: '/api/train',
        PREDICT: '/api/predict',
    },
    TIMEOUT: 300000, // 5 dakika
    MAX_FILE_SIZE: 50 * 1024 * 1024, // 50MB
};

// =========================================================================
// State Management
// =========================================================================

const state = {
    isTraining: false,
    isPredicting: false,
    selectedImage: null,
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
 * Modeli eğitim isteği gönderir
 * @param {number} maxResults - Maksimum kayıt sayısı
 * @returns {Promise<Object>} - Eğitim sonucu
 */
async function trainModel(maxResults) {
    try {
        const requestBody = {
            data_dir: null,
            max_results: maxResults,
        };

        const response = await fetch(`${CONFIG.API_URL}${CONFIG.ENDPOINTS.TRAIN}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Training error:', error);
        throw error;
    }
}

/**
 * Tahmin isteği gönderir (Fotoğraf + Metadata)
 * @param {File} imageFile - Yüklenen görsel dosyası
 * @returns {Promise<Object>} - Tahmin sonucu
 */
async function predictTurtle(imageFile) {
    try {
        const formData = new FormData();
        formData.append('file', imageFile);

        const response = await fetch(`${CONFIG.API_URL}${CONFIG.ENDPOINTS.PREDICT}`, {
            method: 'POST',
            body: formData,
            // Content-Type otomatik olarak FormData için ayarlanır
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
function showTrainingLoading() {
    const loading = document.getElementById('trainingLoading');
    const results = document.getElementById('trainingResults');
    const error = document.getElementById('trainingError');
    const btn = document.getElementById('trainBtn');

    loading.classList.remove('hidden');
    results.classList.add('hidden');
    error.classList.add('hidden');
    btn.disabled = true;
    state.isTraining = true;
}

/**
 * Loading animasyonunu gizler
 */
function hideTrainingLoading() {
    const loading = document.getElementById('trainingLoading');
    const btn = document.getElementById('trainBtn');

    loading.classList.add('hidden');
    btn.disabled = false;
    state.isTraining = false;
}

/**
 * Tahmin Loading animasyonunu gösterir
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
 * Tahmin Loading animasyonunu gizler
 */
function hidePredictionLoading() {
    const loading = document.getElementById('predictionLoading');
    const btn = document.getElementById('predictBtn');

    loading.classList.add('hidden');
    btn.disabled = false;
    state.isPredicting = false;
}

/**
 * Eğitim hata mesajını gösterir
 */
function showTrainingError(errorMessage) {
    const errorDiv = document.getElementById('trainingError');
    const errorContent = document.getElementById('trainingErrorContent');
    const results = document.getElementById('trainingResults');

    results.classList.add('hidden');
    errorContent.textContent = errorMessage;
    errorDiv.classList.remove('hidden');
}

/**
 * Tahmin hata mesajını gösterir
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
 * Eğitim sonuçlarını ekranda gösterir
 */
function displayTrainingResults(data) {
    const resultsContent = document.getElementById('trainingResultsContent');
    const results = document.getElementById('trainingResults');

    let html = '';

    // Toplanan ve kabul edilen kayıt sayıları
    html += `
        <div class="result-item">
            <span class="result-label">📊 Toplam Kaydedilen Veri</span>
            <span class="result-value">${data.records_collected}</span>
        </div>
    `;

    html += `
        <div class="result-item">
            <span class="result-label">✓ Kabul Edilen Veri</span>
            <span class="result-value">${data.records_accepted}</span>
        </div>
    `;

    // Eğitim detayları
    if (data.epochs_trained !== null) {
        html += `
            <div class="result-item">
                <span class="result-label">📚 Eğitim Dönem Sayısı</span>
                <span class="result-value">${data.epochs_trained}</span>
            </div>
        `;
    }

    if (data.final_loss !== null) {
        html += `
            <div class="result-item">
                <span class="result-label">📉 Final Loss (Kaybı)</span>
                <span class="result-value">${parseFloat(data.final_loss).toFixed(4)}</span>
            </div>
        `;
    }

    // Başarı mesajı
    if (data.success) {
        html += `
            <div class="info-box" style="background-color: rgba(0, 200, 83, 0.08); border-left-color: var(--success-green);">
                <p style="color: var(--success-green); margin: 0;">✓ Model başarıyla eğitildi ve kaydedildi!</p>
            </div>
        `;
    }

    resultsContent.innerHTML = html;
    results.classList.remove('hidden');
}

/**
 * Tahmin sonuçlarını ekranda gösterir
 */
function displayPredictionResults(data) {
    const resultsContent = document.getElementById('predictionResultsContent');
    const results = document.getElementById('predictionResults');

    let html = '<div class="prediction-result">';

    // Bireyin durumu (yeni veya tanınan)
    const isNew = data.is_new_turtle;
    const statusClass = isNew ? 'new-turtle' : 'known-turtle';
    const statusText = isNew ? '🆕 YENİ BİREY' : '✓ TANINAN BİREY';

    html += `<div class="turtle-status ${statusClass}">${statusText}</div>`;

    // Kaplumbağa ID'si
    if (!isNew && data.turtle_id) {
        html += `<div class="turtle-id">${data.turtle_id}</div>`;
    }

    // Güven skoru
    const confidence = data.confidence || 0;
    const confidencePercent = (confidence * 100).toFixed(1);

    html += `
        <div>
            <div class="confidence-text">${confidencePercent}%</div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${confidence * 100}%"></div>
            </div>
            <div class="confidence-label">Tahmin Güven Skoru</div>
        </div>
    `;

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

        // Dosyayı state'e kaydet
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
    state.selectedImage = null;

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
 * Eğitim butonu event listener'ı
 */
function setupTrainingButton() {
    const btn = document.getElementById('trainBtn');
    const maxResultsInput = document.getElementById('trainingMaxResults');

    btn.addEventListener('click', async () => {
        const maxResults = parseInt(maxResultsInput.value) || 5000;

        showTrainingLoading();

        try {
            const result = await trainModel(maxResults);
            displayTrainingResults(result);
        } catch (error) {
            showTrainingError(
                `Eğitim başarısız oldu: ${error.message || 'Bilinmeyen hata'}`
            );
        } finally {
            hideTrainingLoading();
        }
    });
}

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
    // Eğitim sonuçlarını kapat
    document.getElementById('closeTrainingResults').addEventListener('click', () => {
        document.getElementById('trainingResults').classList.add('hidden');
    });

    // Eğitim hatasını kapat
    document.getElementById('closeTrainingError').addEventListener('click', () => {
        document.getElementById('trainingError').classList.add('hidden');
    });

    // Tahmin sonuçlarını kapat
    document.getElementById('closePredictionResults').addEventListener('click', () => {
        document.getElementById('predictionResults').classList.add('hidden');
    });

    // Tahmin hatasını kapat
    document.getElementById('closePredictionError').addEventListener('click', () => {
        document.getElementById('predictionError').classList.add('hidden');
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
    setupTrainingButton();
    setupPredictionButton();
    setupImageUploadArea();
    setupCloseButtons();

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
// Debug / Development
// =========================================================================

// Geliştirme modunda API uç noktalarını global olarak erişilebilir hale getir
if (typeof window !== 'undefined') {
    window.CarettaTrackDebug = {
        checkApiHealth,
        trainModel,
        predictTurtle,
        state,
        CONFIG,
    };
    console.log('💡 Debug modu: window.CarettaTrackDebug erişilebilir');
}
