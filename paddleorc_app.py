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

# 初始化 Flask App
app = Flask(__name__)

# 設定上傳資料夾
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 定義允許的圖片副檔名
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# OCR 超時秒數
OCR_TIMEOUT = 12

# 單例模式 OCR 實例
ocr_instance = None

# ===== Helper Functions =====

def allowed_file(filename):
    return '.' in filename and os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

def get_ocr_instance():
    global ocr_instance
    if ocr_instance is None:
        logging.info("🌀 初始化 PaddleOCR 中...")
        ocr_instance = PaddleOCR(
            use_angle_cls=True,
            lang='ch',
            det_db_box_thresh=0.5,
            rec_algorithm='CRNN',
            use_gpu=True
        )
        logging.info("✅ PaddleOCR 初始化完成")
    return ocr_instance

def clear_gpu_memory():
    try:
        paddle.device.cuda.empty_cache()
        gc.collect()
        logging.info("🧹 GPU 記憶體清理完成")
    except Exception as e:
        logging.warning(f"記憶體清理錯誤: {e}")

class OCRTimeoutError(Exception): pass

def timeout_handler(signum, frame):
    raise OCRTimeoutError("OCR 處理超時")

signal.signal(signal.SIGALRM, timeout_handler)

def resize_image_if_large(img, max_side=1280):
    w, h = img.size
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.LANCZOS)
        logging.info(f"🔍 圖片縮放為 {new_size}")
    return img

def safe_ocr(image_path):
    try:
        ocr = get_ocr_instance()
        signal.alarm(OCR_TIMEOUT)
        logging.info(f"⏱️ OCR 開始（超時 {OCR_TIMEOUT}s）: {image_path}")
        result = ocr.ocr(image_path)
        signal.alarm(0)
        clear_gpu_memory()
        return result, None
    except OCRTimeoutError as e:
        signal.alarm(0)
        clear_gpu_memory()
        return None, str(e)
    except Exception as e:
        signal.alarm(0)
        clear_gpu_memory()
        return None, f"OCR 錯誤: {str(e)}"

def safe_remove(path):
    try:
        if os.path.exists(path):
            os.remove(path)
            logging.info(f"🗑️ 已刪除: {path}")
    except Exception as e:
        logging.warning(f"刪除錯誤: {e}")

# ===== Web Routes =====

@app.route('/')
def index():
    return render_template_string(TEST_HTML)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': '缺少上傳檔案'}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': '檔案格式錯誤，僅支援圖片'}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    filename = f"{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(temp_path)

    # 轉 JPG 並縮放
    final_path = temp_path
    if ext != '.jpg':
        try:
            img = Image.open(temp_path).convert('RGB')
            img = resize_image_if_large(img)
            final_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.jpg")
            img.save(final_path, format="JPEG", quality=85)
        except Exception as e:
            safe_remove(temp_path)
            return jsonify({'error': f'圖片處理錯誤: {e}'}), 500
        safe_remove(temp_path)
    else:
        try:
            img = Image.open(final_path).convert('RGB')
            img = resize_image_if_large(img)
            img.save(final_path, format="JPEG", quality=85)
        except Exception as e:
            return jsonify({'error': f'圖片處理錯誤: {e}'}), 500

    result, error = safe_ocr(final_path)
    safe_remove(final_path)

    if error:
        return jsonify({'error': error}), 504 if "超時" in error else 500

    texts = [line[1][0] for line in result[0] if isinstance(line, (list, tuple)) and isinstance(line[1], (list, tuple))]
    return jsonify({
        'text': '\n'.join(texts),
        'message': '✅ 成功辨識'
    })

@app.route('/base64', methods=['POST'])
def ocr_base64():
    data = request.get_json()
    base64_str = data.get('image_base64')
    if not base64_str:
        return jsonify({'error': '缺少 base64 編碼'}), 400

    if 'base64,' in base64_str:
        base64_str = base64_str.split('base64,')[1]

    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image = resize_image_if_large(image)
        temp_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.jpg")
        image.save(temp_path, format="JPEG", quality=85)

        result, error = safe_ocr(temp_path)
        safe_remove(temp_path)

        if error:
            return jsonify({'error': error}), 504 if "超時" in error else 500

        texts = [line[1][0] for line in result[0] if isinstance(line, (list, tuple)) and isinstance(line[1], (list, tuple))]
        return jsonify({'text': '\n'.join(texts)})

    except Exception as e:
        return jsonify({'error': f"Base64 錯誤: {e}"}), 500

# ===== 前端頁面 HTML =====

TEST_HTML = '''<!DOCTYPE html>
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
        <h2>📁 圖片上傳辨識</h2>
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
                alert('請選擇圖片檔案');
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


    // 新增：在 textarea 貼上圖片時，自動轉換為 base64
    document.getElementById('base64input').addEventListener('paste', async function (e) {
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        for (const item of items) {
            if (item.kind === 'file' && item.type.startsWith('image/')) {
                const blob = item.getAsFile();
                const reader = new FileReader();
                reader.onload = function (event) {
                    // 將 base64 填入 textarea
                    document.getElementById('base64input').value = event.target.result;
                };
                reader.readAsDataURL(blob);
                e.preventDefault();  // 防止原圖貼上行為
                break;
            }
        }
    });
    </script>
</body>
</html>

'''

# ===== 啟動應用 =====

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("✅ OCR 服務已啟動：http://localhost:5550/")
    app.run(host='0.0.0.0', port=5550, threaded=True)
