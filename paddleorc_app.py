from flask import Flask, request, jsonify, render_template_string
from paddleocr import PaddleOCR
import os
import uuid
import logging
import base64
from io import BytesIO
from PIL import Image


from flask import Flask, request, jsonify, render_template_string
from paddleocr import PaddleOCR
import paddle
import os
import uuid
import logging
import base64
import signal
import gc
from io import BytesIO
from PIL import Image


# 自定義超時錯誤
class OCRTimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    """超時處理函數"""
    raise OCRTimeoutError("OCR 處理超時")

# 設定超時信號處理器
signal.signal(signal.SIGALRM, timeout_handler)

# 全域 OCR 實例（單例模式）
ocr_instance = None

def get_ocr_instance():
    """取得 OCR 實例（單例模式）"""
    global ocr_instance
    if ocr_instance is None:
        logging.info("🔥 初始化 PaddleOCR 實例...")
        ocr_instance = PaddleOCR(
            use_angle_cls=True,  # 開啟角度校正，提升識別準確度
            lang='ch',  # 中文
            det_db_box_thresh=0.5,  # 檢測閾值
            rec_algorithm='CRNN',  # 使用 CRNN 而非預設的 SVTR (更快)
            use_gpu=True  # 強制使用 GPU
        )
        logging.info("✅ PaddleOCR 實例初始化完成")
    return ocr_instance

def clear_gpu_memory():
    """清理 GPU 記憶體"""
    try:
        paddle.device.cuda.empty_cache()
        gc.collect()
        logging.debug("🧹 GPU 記憶體已清理")
    except Exception as e:
        logging.warning(f"清理 GPU 記憶體時發生錯誤: {e}")


app = Flask(__name__)

def check_gpu_available():
    """檢查 GPU 是否可用"""
    try:
        device = paddle.device.get_device()
        if 'gpu' in device.lower():
            return True, device
        else:
            return False, device
    except Exception as e:
        return False, f"Error: {str(e)}"

def initialize_ocr():
    """初始化 OCR 檢查，但不實際建立實例"""
    gpu_available, device_info = check_gpu_available()
    
    if not gpu_available:
        error_msg = f"❌ GPU 不可用，裝置: {device_info}。請確保 GPU 驅動正確安裝且 Docker 有 GPU 支援。"
        logging.error(error_msg)
        raise RuntimeError(error_msg)
    
    logging.info(f"✅ GPU 可用，裝置: {device_info}")
    return True

# 檢查 GPU 但不初始化 OCR（延遲初始化）
try:
    initialize_ocr()
    GPU_ERROR = None
except RuntimeError as e:
    GPU_ERROR = str(e)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 完整的測試頁面 HTML
TEST_PAGE = '''
<!DOCTYPE html>
<html>
<head>
    <title>PaddleOCR 測試介面</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .section { border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 5px; }
        textarea { width: 100%; box-sizing: border-box; }
        button, input[type="submit"] { padding: 10px 20px; margin: 5px; cursor: pointer; }
        .result { background: #f4f4f4; padding: 15px; border-radius: 5px; white-space: pre-wrap; }
        .loading { color: #666; font-style: italic; }
    </style>
</head>
<body>
    <h1>🖼️ PaddleOCR 測試介面</h1>
    
    <div class="section">
        <h2>⚙️ 性能優化設定</h2>
        <p>🚀 <strong>加速設定</strong>: 已啟用快速模式 (CRNN + 角度校正)</p>
        <p>📏 <strong>自動縮放</strong>: 大圖會自動縮放至 1280px 以內以提升處理速度</p>
        <p>🎯 <strong>檢測閾值</strong>: 0.5 (平衡速度與準確度)</p>
        <p>🔥 <strong>GPU 加速</strong>: 強制使用 GPU，CPU 模式會直接回傳錯誤</p>
        <p>⏱️ <strong>超時保護</strong>: 單張圖片處理時間限制 12 秒</p>
        <p>🧹 <strong>記憶體管理</strong>: 單例初始化 + 每次處理後自動清理 GPU 記憶體</p>
    </div>
        <h2>📁 檔案上傳測試</h2>
        <form id="uploadForm" enctype="multipart/form-data">
            <input type="file" id="imageFile" name="image" accept="image/*" required>
            <input type="submit" value="上傳並辨識">
        </form>
        <div id="uploadResult" class="result" style="display:none;"></div>
    </div>

    <div class="section">
        <h2>🔗 Base64 圖片辨識</h2>
        <textarea id="base64input" rows="8" placeholder="請貼上圖片的 base64 編碼..."></textarea><br>
        <button onclick="sendBase64()">辨識 Base64 圖片</button>
        <div id="base64Result" class="result" style="display:none;"></div>
    </div>



    <script>
        // 檔案上傳處理
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const formData = new FormData();
            const fileInput = document.getElementById('imageFile');
            const resultDiv = document.getElementById('uploadResult');
            
            if (!fileInput.files[0]) {
                alert('請選擇檔案');
                return;
            }
            
            formData.append('image', fileInput.files[0]);
            resultDiv.style.display = 'block';
            resultDiv.textContent = '⏳ 正在辨識中...';
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                resultDiv.textContent = data.text || data.error || '無辨識結果';
            } catch (error) {
                resultDiv.textContent = '❌ 錯誤: ' + error.message;
            }
        });

        // Base64 辨識
        async function sendBase64() {
            const base64 = document.getElementById("base64input").value.trim();
            const resultDiv = document.getElementById("base64Result");
            
            if (!base64) {
                alert('請輸入 base64 編碼');
                return;
            }
            
            resultDiv.style.display = 'block';
            resultDiv.textContent = '⏳ 正在辨識中...';
            
            try {
                const response = await fetch("/base64", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ image_base64: base64 })
                });
                const data = await response.json();
                resultDiv.textContent = data.text || data.error || '無辨識結果';
            } catch (error) {
                resultDiv.textContent = '❌ 錯誤: ' + error.message;
            }
        }


    </script>
</body>
</html>
'''

def resize_image_if_large(img, max_side=1280):
    """縮放大圖片以提升 OCR 處理速度"""
    w, h = img.size
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.LANCZOS)  # 使用 LANCZOS 替代已棄用的 ANTIALIAS
        logging.info(f"圖片已縮放: {w}x{h} -> {new_size[0]}x{new_size[1]}")
    return img

def safe_ocr_with_timeout(image_path, timeout_seconds=12):
    """執行 OCR 並設定超時機制"""
    try:
        # 取得 OCR 實例（單例模式）
        ocr = get_ocr_instance()
        
        # 設定超時
        signal.alarm(timeout_seconds)
        logging.info(f"開始 OCR 處理，超時設定: {timeout_seconds} 秒")
        
        # 執行 OCR
        result = ocr.ocr(image_path)
        
        # 清除超時設定
        signal.alarm(0)
        logging.info("OCR 處理完成")
        
        # 清理 GPU 記憶體
        clear_gpu_memory()
        
        return result, None
        
    except OCRTimeoutError:
        signal.alarm(0)  # 確保清除超時設定
        clear_gpu_memory()  # 超時時也清理記憶體
        error_msg = f"OCR 處理超時（超過 {timeout_seconds} 秒）"
        logging.warning(error_msg)
        return None, error_msg
        
    except Exception as e:
        signal.alarm(0)  # 確保清除超時設定
        clear_gpu_memory()  # 錯誤時也清理記憶體
        error_msg = f"OCR 處理錯誤：{str(e)}"
        logging.error(error_msg)
        return None, error_msg

def safe_remove_file(file_path):
    """安全刪除檔案"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"已刪除臨時檔案: {file_path}")
    except Exception as e:
        logging.warning(f"無法刪除檔案 {file_path}: {e}")

@app.route('/', methods=['GET'])
def index():
    """主頁面重定向到測試頁面"""
    return TEST_PAGE

@app.route('/test', methods=['GET'])
def test_page():
    """測試頁面"""
    return TEST_PAGE

@app.route('/upload', methods=['POST'])
def upload():
    """檔案上傳 API（JSON 回應）"""
    # 檢查 GPU 是否可用
    if GPU_ERROR:
        return jsonify({'error': GPU_ERROR}), 500
        
    if 'image' not in request.files:
        return jsonify({'error': '沒有上傳檔案'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '檔案名稱為空'}), 400

    # 生成唯一檔名
    ext = os.path.splitext(file.filename)[1].lower()
    filename = f"{uuid.uuid4().hex}{ext}"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        # 儲存上傳的檔案
        file.save(image_path)
        
        # 如果不是 JPG，轉換為 JPG 並縮放
        if ext not in ['.jpg', '.jpeg']:
            jpg_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.jpg")
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                img = resize_image_if_large(img, max_side=1280)  # 縮放大圖
                img.save(jpg_path, format="JPEG", quality=85)
            safe_remove_file(image_path)  # 刪除原檔案
            image_path = jpg_path
        else:
            # 即使是 JPG 也要檢查是否需要縮放
            with Image.open(image_path) as img:
                original_size = img.size
                img = img.convert("RGB")
                img = resize_image_if_large(img, max_side=1280)
                if img.size != original_size:  # 如果有縮放，重新儲存
                    img.save(image_path, format="JPEG", quality=85)

        # 執行 OCR（帶超時機制）
        result, error = safe_ocr_with_timeout(image_path, timeout_seconds=12)
        
        # 立即刪除臨時檔案
        safe_remove_file(image_path)
        
        if error:
            if "超時" in error:
                return jsonify({'error': error}), 504  # Gateway Timeout
            else:
                return jsonify({'error': error}), 500
        
        if not result or not isinstance(result, list) or not result[0]:
            return jsonify({'text': '', 'message': '⚠️ 無辨識結果'})

        # 提取文字
        text_blocks = []
        for line in result[0]:
            if isinstance(line, (list, tuple)) and isinstance(line[1], (list, tuple)):
                text_blocks.append(line[1][0])

        return jsonify({
            'text': '\n'.join(text_blocks),
            'message': '✅ 成功辨識'
        })

    except Exception as e:
        # 確保錯誤時也刪除檔案
        safe_remove_file(image_path)
        logging.error(f"Upload OCR error: {e}")
        return jsonify({'error': f"OCR 發生錯誤：{str(e)}"}), 500

@app.route('/ocr', methods=['POST'])
def do_ocr():
    """批次處理目錄中的圖片"""
    data = request.get_json()
    image_dir = data.get('image_dir')

    if not image_dir or not os.path.isdir(image_dir):
        return jsonify({'error': 'missing or invalid image_dir'}), 400

    if not check_poppler_installed():
        logging.error("❌ Poppler 未安裝或不在 PATH 中")
        return jsonify({'error': 'Poppler is not installed or not in PATH'}), 500

    all_texts = []
    temp_files = []  # 追蹤臨時檔案
    
    try:
        for filename in sorted(os.listdir(image_dir)):
            path = os.path.join(image_dir, filename)
            if os.path.isfile(path) and path.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                tmp_jpg_path = None
                try:
                    # 將圖轉為 RGB、縮放並另存為 JPG 暫存檔
                    with Image.open(path) as img:
                        img = img.convert("RGB")
                        img = resize_image_if_large(img, max_side=1280)  # 縮放大圖
                        tmp_jpg_path = os.path.join(image_dir, f"__tmp_{uuid.uuid4().hex}.jpg")
                        img.save(tmp_jpg_path, format="JPEG", quality=85)
                        temp_files.append(tmp_jpg_path)

                    # OCR on converted JPG
                    result = ocr.ocr(tmp_jpg_path)

                    # 擷取文字
                    if result and isinstance(result, list) and result[0]:
                        for line in result[0]:
                            if isinstance(line, (list, tuple)) and isinstance(line[1], (list, tuple)):
                                all_texts.append(line[1][0])
                                
                except Exception as e:
                    logging.warning(f"OCR error on file {filename}: {e}")
                    continue
                finally:
                    # 立即清除當前檔案的暫存檔
                    if tmp_jpg_path:
                        safe_remove_file(tmp_jpg_path)

        return jsonify({'text': '\n'.join(all_texts)})
        
    finally:
        # 確保清除所有暫存檔案
        for temp_file in temp_files:
            safe_remove_file(temp_file)

@app.route('/base64', methods=['POST'])
def ocr_base64():
    """Base64 圖片辨識 API"""
    # 檢查 GPU 是否可用
    if GPU_ERROR:
        return jsonify({'error': GPU_ERROR}), 500
        
    data = request.get_json()
    base64_str = data.get('image_base64')
    
    if not base64_str:
        return jsonify({'error': '缺少 base64 編碼圖像'}), 400

    image_path = None
    try:
        # 解碼 base64 並轉換成圖片
        # 處理可能包含 data URI 前綴的情況
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[1]
            
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        image = image.convert('RGB')
        image = resize_image_if_large(image, max_side=1280)  # 縮放大圖

        # 強制儲存為 JPG 以供 OCR 使用
        filename = f"{uuid.uuid4().hex}.jpg"
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(image_path, format="JPEG", quality=85)

        # 執行 OCR（帶超時機制）
        result, error = safe_ocr_with_timeout(image_path, timeout_seconds=12)

        # 立即刪除暫存檔案
        safe_remove_file(image_path)

        if error:
            if "超時" in error:
                return jsonify({'error': error}), 504  # Gateway Timeout
            else:
                return jsonify({'error': error}), 500

        if not result or not isinstance(result, list) or not result[0]:
            return jsonify({'text': '', 'message': '⚠️ 無辨識結果'})

        # 提取文字
        text_blocks = []
        for line in result[0]:
            if isinstance(line, (list, tuple)) and isinstance(line[1], (list, tuple)):
                text_blocks.append(line[1][0])

        return jsonify({
            'text': '\n'.join(text_blocks),
            'message': '✅ 成功辨識'
        })

    except Exception as e:
        # 確保錯誤時也刪除檔案
        if image_path:
            safe_remove_file(image_path)
        logging.error(f"OCR base64 error: {e}")
        return jsonify({'error': f"OCR 發生錯誤：{str(e)}"}), 500

@app.route('/gpu-status', methods=['GET'])
def gpu_status():
    """檢查 GPU 狀態 API"""
    gpu_available, device_info = check_gpu_available()
    
    # 檢查 OCR 實例狀態
    ocr_initialized = ocr_instance is not None
    
    return jsonify({
        'gpu_available': gpu_available,
        'device': device_info,
        'ocr_ready': GPU_ERROR is None,
        'ocr_initialized': ocr_initialized
    })

if __name__ == '__main__':
    # 設定日誌
    logging.basicConfig(level=logging.INFO)
    
    # 檢查 GPU 狀態並顯示啟動訊息
    if GPU_ERROR:
        print("❌ 服務啟動失敗：")
        print(f"   {GPU_ERROR}")
        print("💡 解決方案：")
        print("   1. 確保 GPU 驅動程式已正確安裝")
        print("   2. 確保 Docker 支援 GPU (nvidia-docker)")
        print("   3. 確保 PaddlePaddle GPU 版本已安裝")
        print("\n🔍 GPU 檢查指令：")
        print("   python -c \"import paddle; print(paddle.device.get_device())\"")
    else:
        gpu_available, device_info = check_gpu_available()
        print("🚀 PaddleOCR 服務啟動成功！")
        print(f"🔥 GPU 狀態: ✅ 可用 ({device_info})")
        print("⚡ 性能優化設定:")
        print("   - 使用 CRNN 算法 (較快)")
        print("   - 開啟角度校正 (提升準確度)")
        print("   - 自動縮放大於 1280px 的圖片")
        print("   - 檢測閾值: 0.5")
        print("   - 強制 GPU 加速")
        print("   - OCR 超時保護: 12 秒")
        print("   - 單例模式 + GPU 記憶體自動清理")
    
    print("\n📝 服務端點:")
    print("   - GET  / (測試頁面)")
    print("   - POST /upload (檔案上傳)")
    print("   - POST /base64 (Base64 辨識)")
    print("   - GET  /gpu-status (GPU 狀態檢查)")
    print(f"\n🌐 服務網址: http://localhost:5550/")
    
    app.run(host='0.0.0.0', port=5550, debug=True)