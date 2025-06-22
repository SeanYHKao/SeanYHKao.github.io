import os
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
import torch

app = FastAPI()

# ✅ 模型路徑（GPTQ 模型）
MODEL_PATH = "/app/models/qwen3-30b-a3b-gptq-int4"

# ✅ 載入 tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# ✅ 載入 GPTQ 模型（注意使用 AutoGPTQForCausalLM）
model = AutoGPTQForCausalLM.from_quantized(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    inject_fused_attention=False  # 保險起見避免某些 GPU 報錯
).eval()

# === 資料模型 ===
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 2048
    enable_thinking: bool = True

# === API ===
@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    text = tokenizer.apply_chat_template(
        req.messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=req.enable_thinking
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        do_sample=True
    )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {
        "choices": [
            {"message": {"role": "assistant", "content": output_text}}
        ]
    }

@app.get("/v1/models")
async def models():
    return {
        "data": [{"id": "qwen3-30b-a3b-gptq", "object": "model", "owned_by": "you"}]
    }

# === HTML UI for quick testing ===
@app.get("/", response_class=HTMLResponse)
async def ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Qwen3-30B Chat</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            textarea, input { width: 100%; margin-bottom: 10px; }
            button { padding: 10px 20px; }
            .response { margin-top: 20px; white-space: pre-wrap; border: 1px solid #ccc; padding: 10px; }
        </style>
    </head>
    <body>
        <h1>Qwen3-30B Chat</h1>
        <textarea id="prompt" rows="6" placeholder="請輸入對話內容"></textarea>
        <button onclick="send()">送出</button>
        <div class="response" id="responseBox"></div>

        <script>
            async function send() {
                const content = document.getElementById("prompt").value;
                const res = await fetch("/v1/chat/completions", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        model: "qwen3-30b-a3b-gptq",
                        messages: [{ role: "user", content: content }],
                        temperature: 0.6,
                        top_p: 0.95,
                        top_k: 20,
                        max_tokens: 2048,
                        enable_thinking: true
                    })
                });
                const data = await res.json();
                document.getElementById("responseBox").innerText = data.choices[0].message.content;
            }
        </script>
    </body>
    </html>
    """

# === 啟動用於 accelerate launch ===
if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
