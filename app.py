import os
import base64
import io
import math
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
            "seed": "42",
        },
        timeout=120,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Stability AI Fehler {response.status_code}: {response.text}")

    result = response.json()
    image_b64 = result["artifacts"][0]["base64"]
    return base64.b64decode(image_b64)


def ease_in_out_cubic(t: float) -> float:
    """Cubic ease-in-out for smoother animation curves."""
    if t < 0.5:
        return 4 * t * t * t
    return 1 - pow(-2 * t + 2, 3) / 2


def generate_animation_frames(
    image_bytes: bytes,
    prompt: str,
    target_strength: float,
    num_frames: int,
) -> list[bytes]:
    """Generate intermediate frames at progressive strength levels."""
    frames = []
    for i in range(num_frames):
        t = (i + 1) / num_frames
        eased_t = ease_in_out_cubic(t)
        # Minimum strength ~0.08 to avoid too-similar-to-original frames
        strength = max(0.08, eased_t * target_strength)
        frame_bytes = transform_image_with_stability(image_bytes, prompt, strength)
        frames.append(frame_bytes)
    return frames


def compile_gif(
    original_bytes: bytes,
    frame_list: list[bytes],
    frame_duration_ms: int,
    boomerang: bool,
) -> bytes:
    """Compile original + intermediate frames into an animated GIF."""
    pil_frames = []
    target_size = (768, 768)

    # Start with original
    orig = Image.open(io.BytesIO(original_bytes)).convert("RGB").resize(target_size, Image.LANCZOS)
    # Hold the original for a few frames
    for _ in range(3):
        pil_frames.append(orig.copy())

    # Add generated frames
    for fb in frame_list:
        frame = Image.open(io.BytesIO(fb)).convert("RGB").resize(target_size, Image.LANCZOS)
        pil_frames.append(frame)

    # Hold the last frame
    for _ in range(5):
        pil_frames.append(pil_frames[-1].copy())

    # Boomerang: play in reverse back to original
    if boomerang:
        for frame in reversed(pil_frames[:-5]):
            pil_frames.append(frame.copy())
        for _ in range(3):
            pil_frames.append(orig.copy())

    buf = io.BytesIO()
    pil_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )
    return buf.getvalue()


def blend_two_images_to_gif(
    start_bytes: bytes,
    end_bytes: bytes,
    num_frames: int,
    frame_duration_ms: int,
    boomerang: bool,
) -> bytes:
    """Create a smooth crossfade GIF between two images (no API calls)."""
    target_size = (768, 768)
    start = Image.open(io.BytesIO(start_bytes)).convert("RGB").resize(target_size, Image.LANCZOS)
    end = Image.open(io.BytesIO(end_bytes)).convert("RGB").resize(target_size, Image.LANCZOS)

    pil_frames = []

    # Hold start
    for _ in range(3):
        pil_frames.append(start.copy())

    # Blended intermediates
    for i in range(num_frames):
        t = (i + 1) / (num_frames + 1)
        alpha = ease_in_out_cubic(t)
        blended = Image.blend(start, end, alpha)
        pil_frames.append(blended)

    # Hold end
    for _ in range(5):
        pil_frames.append(end.copy())

    if boomerang:
        for frame in reversed(pil_frames[3:-5]):
            pil_frames.append(frame.copy())
        for _ in range(3):
            pil_frames.append(start.copy())

    buf = io.BytesIO()
    pil_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )
    return buf.getvalue()


# --- Routes ---

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
        optimized_prompt = analyze_room_with_claude(image_bytes, instructions)
    except Exception as e:
        return jsonify({"error": f"Claude-Analyse fehlgeschlagen: {str(e)}"}), 500

    try:
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


@app.route("/animate", methods=["POST"])
def animate_room():
    """Generate an animated transition from original to AI-edited room."""
    if "image" not in request.files:
        return jsonify({"error": "Kein Start-Bild hochgeladen"}), 400

    file = request.files["image"]
    instructions = request.form.get("instructions", "").strip()
    strength = float(request.form.get("strength", "0.6"))
    num_frames = int(request.form.get("num_frames", "6"))
    frame_duration = int(request.form.get("frame_duration", "120"))
    boomerang = request.form.get("boomerang", "true") == "true"
    mode = request.form.get("mode", "ai_frames")  # ai_frames | blend

    num_frames = max(3, min(num_frames, 12))
    frame_duration = max(50, min(frame_duration, 500))

    allowed = {"image/jpeg", "image/png", "image/webp"}
    if not file.filename or file.content_type not in allowed:
        return jsonify({"error": "Ungültiges Bild"}), 400

    start_bytes = file.read()

    # Check if an end-frame was uploaded manually
    end_file = request.files.get("end_image")
    end_bytes = None
    if end_file and end_file.filename:
        if end_file.content_type not in allowed:
            return jsonify({"error": "Ungültiges End-Bild"}), 400
        end_bytes = end_file.read()

    try:
        optimized_prompt = analyze_room_with_claude(start_bytes, instructions)
    except Exception as e:
        return jsonify({"error": f"Claude-Analyse fehlgeschlagen: {str(e)}"}), 500

    try:
        if mode == "blend":
            # Blend mode: generate only final frame, then crossfade client-side-quality
            if end_bytes is None:
                end_bytes = transform_image_with_stability(start_bytes, optimized_prompt, strength)
            gif_bytes = blend_two_images_to_gif(
                start_bytes, end_bytes, num_frames * 3, frame_duration, boomerang
            )
        else:
            # AI frames mode: generate N intermediate frames at progressive strengths
            frames = generate_animation_frames(start_bytes, optimized_prompt, strength, num_frames)
            gif_bytes = compile_gif(start_bytes, frames, frame_duration, boomerang)
    except ValueError as e:
        return jsonify({"error": str(e)}), 500
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 502

    gif_b64 = base64.standard_b64encode(gif_bytes).decode("utf-8")
    original_b64 = base64.standard_b64encode(start_bytes).decode("utf-8")

    return jsonify({
        "animation": f"data:image/gif;base64,{gif_b64}",
        "original_image": f"data:image/png;base64,{original_b64}",
        "generated_prompt": optimized_prompt,
        "mode": mode,
        "num_frames": num_frames,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
