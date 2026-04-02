import os
import base64
import io
import requests
from flask import Flask, request, jsonify, render_template
from anthropic import Anthropic
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB max upload

anthropic_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
STABILITY_API_KEY = os.environ.get("STABILITY_API_KEY")
STABILITY_URL = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/image-to-image"


def resize_image_for_stability(image_bytes: bytes) -> bytes:
    """Resize image to a Stability AI compatible resolution (must be multiple of 64)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Stability AI SDXL supports 1024x1024 best
    img = img.resize((1024, 1024), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def analyze_room_with_claude(image_bytes: bytes, user_instructions: str) -> str:
    """Use Claude Vision to analyze the room and generate an optimized img2img prompt."""
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    response = anthropic_client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Du bist ein Innenarchitektur-Experte. Analysiere diesen Raum detailliert "
                            f"und erstelle einen präzisen Stable-Diffusion img2img Prompt auf Englisch, "
                            f"der folgende Änderungen umsetzt: '{user_instructions}'.\n\n"
                            f"Der Prompt muss:\n"
                            f"1. Den bestehenden Raumstil beschreiben (Wände, Boden, Möbel)\n"
                            f"2. Die gewünschten Änderungen klar formulieren\n"
                            f"3. Qualitätsmarker enthalten (photorealistic, 8k, interior design, professional photography)\n"
                            f"4. NUR den fertigen Prompt ausgeben, keine Erklärungen"
                        ),
                    },
                ],
            }
        ],
    )

    return response.content[0].text.strip()


def transform_image_with_stability(image_bytes: bytes, prompt: str, strength: float) -> bytes:
    """Send image + prompt to Stability AI img2img endpoint."""
    if not STABILITY_API_KEY:
        raise ValueError("STABILITY_API_KEY ist nicht gesetzt")

    resized = resize_image_for_stability(image_bytes)

    response = requests.post(
        STABILITY_URL,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {STABILITY_API_KEY}",
        },
        files={"init_image": ("image.png", resized, "image/png")},
        data={
            "init_image_mode": "IMAGE_STRENGTH",
            "image_strength": str(strength),
            "text_prompts[0][text]": prompt,
            "text_prompts[0][weight]": "1",
            "text_prompts[1][text]": "blurry, low quality, distorted, ugly, bad anatomy",
            "text_prompts[1][weight]": "-1",
            "cfg_scale": "7",
            "samples": "1",
            "steps": "30",
        },
        timeout=120,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Stability AI Fehler {response.status_code}: {response.text}")

    result = response.json()
    image_b64 = result["artifacts"][0]["base64"]
    return base64.b64decode(image_b64)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/edit", methods=["POST"])
def edit_room():
    if "image" not in request.files:
        return jsonify({"error": "Kein Bild hochgeladen"}), 400

    file = request.files["image"]
    instructions = request.form.get("instructions", "").strip()
    strength = float(request.form.get("strength", "0.6"))

    if not file.filename:
        return jsonify({"error": "Keine Datei ausgewählt"}), 400
    if not instructions:
        return jsonify({"error": "Bitte gib Bearbeitungsanweisungen ein"}), 400

    allowed = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in allowed:
        return jsonify({"error": "Nur JPEG, PNG und WebP werden unterstützt"}), 400

    image_bytes = file.read()

    try:
        # Step 1: Claude analyzes the room and generates the prompt
        optimized_prompt = analyze_room_with_claude(image_bytes, instructions)
    except Exception as e:
        return jsonify({"error": f"Claude-Analyse fehlgeschlagen: {str(e)}"}), 500

    try:
        # Step 2: Stability AI transforms the image
        result_bytes = transform_image_with_stability(image_bytes, optimized_prompt, strength)
    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502

    result_b64 = base64.standard_b64encode(result_bytes).decode("utf-8")
    original_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

    return jsonify({
        "result_image": f"data:image/png;base64,{result_b64}",
        "original_image": f"data:image/{file.content_type.split('/')[1]};base64,{original_b64}",
        "generated_prompt": optimized_prompt,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
