import os
import time
import socket
import asyncio
import urllib.request
import cv2
import numpy as np
import torch
from PIL import Image
import nest_asyncio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from io import BytesIO
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from fastapi.middleware.cors import CORSMiddleware
import psutil
import replicate

# Allow FastAPI to run inside asynchronous environments

nest_asyncio.apply()

app = FastAPI(title="AI Image Production Pipeline API")

# Enable CORS middleware to allow requests from frontend (localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# PORT CLEANUP
# -------------------------------------------------------------
def kill_port_owner(port=8000):
    print(f"Scanning for processes occupying port {port}...")
    killed = False
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            connections = proc.connections(kind='inet')
            for conn in connections:
                if conn.laddr.port == port:
                    if proc.pid == current_pid:
                        print(f"Warning: Port {port} is occupied by this process.")
                        return True
                    print(f"Killing process PID {proc.pid} ({proc.name()}) occupying port {port}...")
                    proc.kill()
                    killed = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    if not killed:
        print(f"Port {port} is clean and ready.")
    return False

# -------------------------------------------------------------
# MODEL INITIALIZATION (Loaded globally)
# -------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Utilizing compute device: {device}")

print("Loading BiRefNet (Segmentation) from Hugging Face...")
birefnet_model = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", 
    trust_remote_code=True
).to(device)
birefnet_model.eval()

from diffusers import StableDiffusionXLInpaintPipeline
print("Loading SDXL Inpainting / Composition pipeline...")
# Use float16 if GPU is available to save memory
torch_dtype = torch.float16 if device == "cuda" else torch.float32
composition_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1", 
    torch_dtype=torch_dtype,
    use_safetensors=True
).to(device)

# -------------------------------------------------------------
# LOCAL FACE SWAP INITIALIZATION (InsightFace)
# -------------------------------------------------------------
USE_LOCAL_FACESWAP = True

app_face = None
swapper = None

if USE_LOCAL_FACESWAP:
    import insightface
    from insightface.app import FaceAnalysis
    
    # 1. Download inswapper_128.onnx if it doesn't exist
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "inswapper_128.onnx")
    
    if not os.path.exists(model_path):
        print("Downloading inswapper_128.onnx model from Hugging Face...")
        url = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
        try:
            urllib.request.urlretrieve(url, model_path)
            print(f"Model downloaded successfully and saved to {model_path}!")
        except Exception as e:
            print(f"Failed to download model dynamically: {e}")
            
    # 2. Initialize face analysis and swapper
    print("Initializing local InsightFace models...")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == "cuda" else ['CPUExecutionProvider']
    
    # Setup writable home directory for configuration files
    os.environ["INSIGHTFACE_HOME"] = "/tmp/.insightface"
    
    try:
        ctx_id = 0 if device == "cuda" else -1
        app_face = FaceAnalysis(name='buffalo_l', providers=providers)
        # Use 320x320 detection: better for small/distant faces, still robust for large ones
        app_face.prepare(ctx_id=ctx_id, det_size=(320, 320))
        print(f"InsightFace FaceAnalysis prepared successfully on ctx_id: {ctx_id}")
    except Exception as e:
        print(f"Failed to initialize FaceAnalysis on CUDA/default providers. Error: {e}")
        print("Falling back to CPUExecutionProvider...")
        providers = ['CPUExecutionProvider']
        app_face = FaceAnalysis(name='buffalo_l', providers=providers)
        app_face.prepare(ctx_id=-1, det_size=(320, 320))
        print("InsightFace FaceAnalysis fallback prepared successfully on CPU.")
    
    if os.path.exists(model_path):
        try:
            swapper = insightface.model_zoo.get_model(model_path, providers=providers)
            print("Local Face Swapper initialized successfully on device:", device)
        except Exception as e:
            print(f"Failed to initialize Face Swapper model. Error: {e}")
            print("Falling back swapper to CPU...")
            swapper = insightface.model_zoo.get_model(model_path, providers=['CPUExecutionProvider'])
            print("Local Face Swapper fallback initialized successfully on CPU.")
    else:
        print("Warning: inswapper_128.onnx model not found. Local Face Swap is disabled.")

# -------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------
def remove_background(input_image: Image.Image):
    img_size = input_image.size
    
    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform_image(input_image).unsqueeze(0).to(device)
    if device == "cuda":
        input_tensor = input_tensor.half()
    
    with torch.no_grad():
        outputs = birefnet_model(input_tensor)
        if isinstance(outputs, (tuple, list)):
            pred = outputs[0][-1]
        else:
            pred = outputs
            
        pred = pred.sigmoid().cpu().squeeze()
        
    mask_img = transforms.ToPILImage()(pred).resize(img_size)
    return mask_img

def detect_faces_with_fallback(face_app, img_cv, label="image"):
    """Detect faces. app_face is already prepared with det_size=(320,320) at startup.
    For very small faces, try upscaling the image instead of re-calling prepare()."""
    # Try direct detection first
    faces = face_app.get(img_cv)
    if faces:
        print(f"Detected {len(faces)} face(s) in {label}")
        return faces
    
    # If no faces found, try upscaling the image 2x and detect again
    print(f"No faces found in {label} at original size, trying 2x upscale...")
    h, w = img_cv.shape[:2]
    img_upscaled = cv2.resize(img_cv, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    faces = face_app.get(img_upscaled)
    if faces:
        print(f"Detected {len(faces)} face(s) in {label} after 2x upscale")
        return faces
    
    print(f"No faces found in {label} even after upscaling.")
    return []

def swap_faces_local(source_image: Image.Image, target_image: Image.Image) -> Image.Image:
    if not app_face or not swapper:
        raise Exception("InsightFace local model has not been initialized.")
    
    # Upscale images before face detection to improve accuracy on small faces
    MIN_DIM = 800
    def ensure_min_size(img):
        w, h = img.size
        if min(w, h) < MIN_DIM:
            scale = MIN_DIM / min(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img
    
    source_image = ensure_min_size(source_image)
    target_image = ensure_min_size(target_image)
        
    source_cv = cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR)
    target_cv = cv2.cvtColor(np.array(target_image), cv2.COLOR_RGB2BGR)
    
    # Detect faces in user's image
    source_faces = detect_faces_with_fallback(app_face, source_cv, label="source portrait")
    if not source_faces:
        raise Exception("Không tìm thấy khuôn mặt nào trong ảnh chân dung của bạn. Vui lòng chọn ảnh rõ mặt hơn.")
    source_face = sorted(source_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
    
    # Detect faces in template body image (with multi-size fallback)
    target_faces = detect_faces_with_fallback(app_face, target_cv, label="template body")
    if not target_faces:
        raise Exception(
            "AI vẽ ra ảnh template nhưng khuôn mặt quá nhỏ hoặc quá mờ để ghép. "
            "Hãy thêm 'close-up, face clearly visible, looking at camera' vào cuối prompt của bạn rồi thử lại."
        )
    target_face = sorted(target_faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
    
    # Swap faces
    print("Performing local face swap...")
    result_cv = swapper.get(target_cv, target_face, source_face, paste_back=True)
    
    # Convert back to PIL Image and resize to original target dimensions
    result_img = Image.fromarray(cv2.cvtColor(result_cv, cv2.COLOR_BGR2RGB))
    return result_img

# -------------------------------------------------------------
# API ENDPOINTS
# -------------------------------------------------------------

@app.get("/")
@app.get("/health")
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "BG.Studio AI Server is running", "device": device}

@app.post("/api/remove-bg")
async def api_remove_bg(
    file: UploadFile = File(...)
):
    print("Received background removal request")
    contents = await file.read()
    init_img = Image.open(BytesIO(contents)).convert("RGB")
    
    alpha_mask = remove_background(init_img)
    
    rgba_img = init_img.copy()
    rgba_img.putalpha(alpha_mask)
    
    img_io = BytesIO()
    rgba_img.save(img_io, 'PNG')
    img_io.seek(0)
    
    headers = {
        "X-Background-Removal-Source": "Local BiRefNet"
    }
    return StreamingResponse(img_io, media_type="image/png", headers=headers)

@app.post("/api/generate-background")
async def generate_background(
    prompt: str = Form(...)
):
    print(f"Generating background for prompt: {prompt}")
    
    # ---------------- Replicate Cloud API (Flux Dev) Option ----------------
    token = os.getenv("REPLICATE_API_TOKEN")
    if token:
        print("Using Replicate Cloud API (Flux Dev) for high-quality background...")
        try:
            client = replicate.Client(api_token=token)
            output = client.run(
                "black-forest-labs/flux-dev",
                input={
                    "prompt": prompt,
                    "go_fast": True,
                    "megapixels": "1",
                    "num_outputs": 1,
                    "aspect_ratio": "4:3",
                    "output_format": "jpg",
                    "output_quality": 90
                }
            )
            
            output_url = None
            if isinstance(output, list) and len(output) > 0:
                output_url = output[0]
            elif isinstance(output, str):
                output_url = output
                
            if output_url:
                print(f"Replicate Flux Dev complete! Result URL: {output_url}")
                req = urllib.request.Request(output_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    img_data = resp.read()
                    
                return StreamingResponse(
                    BytesIO(img_data),
                    media_type="image/jpeg",
                    headers={"X-BG-Source": "Replicate Flux Dev"}
                )
        except Exception as e:
            print(f"Replicate Flux Dev error: {e}. Falling back to Pollinations.ai...")

    # ---------------- Fallback to Pollinations.ai ----------------
    print("Falling back to Pollinations.ai...")
    try:
        import random
        seed = random.randint(0, 999999)
        full_prompt = f"{prompt}, high resolution, photorealistic, 8k, professional photography, no text, no watermark"
        encoded = urllib.request.quote(full_prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=768&seed={seed}&nologo=true&model=flux"

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            img_data = resp.read()

        return StreamingResponse(
            BytesIO(img_data),
            media_type="image/jpeg",
            headers={"X-BG-Source": "Pollinations.ai"}
        )
    except Exception as e:
        print(f"Pollinations.ai error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
@app.post("/api/generate-context-background")
async def generate_context_background(
    image: UploadFile = File(...),          # The canvas containing the moved subject
    mask_image: UploadFile = File(None),    # The foreground alpha mask from BiRefNet
    prompt: str = Form(...)
):
    print(f"🎨 Generating context-aware background for prompt: {prompt}")
    contents = await image.read()
    composite_img = Image.open(BytesIO(contents)).convert("RGB")
    
    if mask_image is not None:
        mask_contents = await mask_image.read()
        alpha_mask = Image.open(BytesIO(mask_contents)).convert("L")
    else:
        alpha_mask = remove_background(composite_img)

    # Invert the mask: Make the background white (paint here) and subject black (keep this)
    import PIL.ImageOps
    bg_mask = PIL.ImageOps.invert(alpha_mask)
    
    # Composite person onto a NEUTRAL GREY background before sending to FLUX Fill.
    # This ensures the model clearly sees where the person is and only repaints the grey areas.
    neutral_bg = Image.new("RGB", composite_img.size, (128, 128, 128))
    mask_float = alpha_mask.convert("L")
    composite_with_neutral = Image.composite(composite_img, neutral_bg, mask_float)
    
    comp_resized = composite_with_neutral.resize((1024, 1024))
    mask_resized = bg_mask.resize((1024, 1024))
    
    # Strong prompt: only describe scenery, explicitly block extra people
    full_prompt = (
        f"{prompt}, "
        "high resolution, photorealistic, cinematic, detailed scenery, "
        "no additional people, no duplicate person, no other human figures, "
        "only environment and background, matching natural lighting and shadows"
    )

    
    # ---------------- Replicate Cloud API (Flux Fill) Option ----------------
    token = os.getenv("REPLICATE_API_TOKEN")
    if token:
        print("Using Replicate Cloud API (Flux Fill) for Context Background...")
        import tempfile
        tmp_img_path = None
        tmp_mask_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_img:
                comp_resized.save(tmp_img, "JPEG", quality=95)
                tmp_img_path = tmp_img.name
                
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_mask:
                mask_resized.save(tmp_mask, "PNG")
                tmp_mask_path = tmp_mask.name
                
            client = replicate.Client(api_token=token)
            with open(tmp_img_path, "rb") as img_file, open(tmp_mask_path, "rb") as mask_file:
                print("Running Replicate black-forest-labs/flux-fill-dev...")
                output = client.run(
                    "black-forest-labs/flux-fill-dev",
                    input={
                        "image": img_file,
                        "mask": mask_file,
                        "prompt": full_prompt,
                        "num_inference_steps": 28,
                        "guidance": 30.0,
                        "output_format": "jpg",
                        "output_quality": 90
                    }
                )
                
            output_url = None
            if isinstance(output, list) and len(output) > 0:
                output_url = output[0]
            elif isinstance(output, str):
                output_url = output
                
            if output_url:
                print(f"Replicate Flux Fill complete! Result URL: {output_url}")
                req = urllib.request.Request(output_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result_data = resp.read()
                
                # Resize result back to original size
                res_img = Image.open(BytesIO(result_data)).resize(composite_img.size)
                img_io = BytesIO()
                res_img.save(img_io, 'JPEG', quality=90)
                img_io.seek(0)
                return StreamingResponse(img_io, media_type="image/jpeg")
            else:
                print("Replicate returned empty output. Falling back to local pipeline...")
                
        except Exception as e:
            print(f"Replicate Inpaint error: {e}. Falling back to local pipeline...")
        finally:
            for p in [tmp_img_path, tmp_mask_path]:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception as ex:
                        print(f"Error removing temp file: {ex}")

    # ---------------- Local SDXL Inpaint Fallback ----------------
    print("Running local Stable Diffusion XL Inpainting...")
    context = torch.autocast(device) if device == "cuda" else torch.no_grad()
    with context:
        output_image = composition_pipe(
            prompt=full_prompt,
            image=comp_resized,
            mask_image=mask_resized,
            strength=1.0  # Completely rewrite the background space
        ).images[0]
        
    output_image = output_image.resize(composite_img.size)
    
    img_io = BytesIO()
    output_image.save(img_io, 'JPEG', quality=90)
    img_io.seek(0)
    return StreamingResponse(img_io, media_type="image/jpeg")

@app.post("/api/process-pipeline")
async def process_image_pipeline(
    image: UploadFile = File(...),
    mask_image: UploadFile = File(None),
    bg_image: UploadFile = File(None),
    lighting_prompt: str = Form(...),
    strength: float = Form(0.35)
):
    print(f"Relighting request: prompt='{lighting_prompt}', strength={strength}")
    contents = await image.read()
    init_img = Image.open(BytesIO(contents)).convert("RGB")
    
    if mask_image is not None:
        mask_contents = await mask_image.read()
        alpha_mask = Image.open(BytesIO(mask_contents)).convert("L")
    else:
        alpha_mask = remove_background(init_img)
        
    if bg_image is not None:
        bg_contents = await bg_image.read()
        uploaded_bg = Image.open(BytesIO(bg_contents)).convert("RGB").resize(init_img.size)
        
        fg_arr = np.array(init_img)
        bg_arr = np.array(uploaded_bg)
        mask_arr = np.array(alpha_mask)[:, :, None] / 255.0
        
        composite_arr = (fg_arr * mask_arr) + (bg_arr * (1 - mask_arr))
        composite_img = Image.fromarray(composite_arr.astype(np.uint8))
    else:
        composite_img = init_img
        
    orig_size = composite_img.size
    comp_resized = composite_img.resize((1024, 1024))
    mask_resized = alpha_mask.resize((1024, 1024))
    
    prompt = f"Subject with {lighting_prompt}, high resolution, matching lighting and shadows, seamless integration"
    
    # ---------------- Replicate Cloud API (Flux Fill) Option ----------------
    token = os.getenv("REPLICATE_API_TOKEN")
    if token:
        print("Using Replicate Cloud API (Flux Fill) for Inpainting...")
        import tempfile
        tmp_img_path = None
        tmp_mask_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_img:
                comp_resized.save(tmp_img, "JPEG", quality=95)
                tmp_img_path = tmp_img.name
                
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_mask:
                mask_resized.save(tmp_mask, "PNG")
                tmp_mask_path = tmp_mask.name
                
            client = replicate.Client(api_token=token)
            with open(tmp_img_path, "rb") as img_file, open(tmp_mask_path, "rb") as mask_file:
                print("Running Replicate black-forest-labs/flux-fill-dev...")
                output = client.run(
                    "black-forest-labs/flux-fill-dev",
                    input={
                        "image": img_file,
                        "mask": mask_file,
                        "prompt": prompt,
                        "num_inference_steps": 28,
                        "guidance": 30.0,
                        "output_format": "jpg",
                        "output_quality": 90
                    }
                )
                
            output_url = None
            if isinstance(output, list) and len(output) > 0:
                output_url = output[0]
            elif isinstance(output, str):
                output_url = output
                
            if output_url:
                print(f"Replicate Flux Fill complete! Result URL: {output_url}")
                req = urllib.request.Request(output_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result_data = resp.read()
                
                # Resize result back to original size
                res_img = Image.open(BytesIO(result_data)).resize(orig_size)
                img_io = BytesIO()
                res_img.save(img_io, 'JPEG', quality=90)
                img_io.seek(0)
                return StreamingResponse(img_io, media_type="image/jpeg")
            else:
                print("Replicate returned empty output. Falling back to local pipeline...")
                
        except Exception as e:
            print(f"Replicate Inpaint error: {e}. Falling back to local pipeline...")
        finally:
            for p in [tmp_img_path, tmp_mask_path]:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception as ex:
                        print(f"Error removing temp file: {ex}")
                        
    # ---------------- Local SDXL Inpaint Fallback ----------------
    print("Running local Stable Diffusion XL Inpainting...")
    context = torch.autocast(device) if device == "cuda" else torch.no_grad()
    with context:
        output_image = composition_pipe(
            prompt=prompt,
            image=comp_resized,
            mask_image=mask_resized,
            strength=strength
        ).images[0]
        
    output_image = output_image.resize(orig_size)
    
    img_io = BytesIO()
    output_image.save(img_io, 'JPEG', quality=90)
    img_io.seek(0)
    
    return StreamingResponse(img_io, media_type="image/jpeg")

@app.post("/api/face-swap")
async def api_face_swap(
    swap_image: UploadFile = File(...),
    target_image_url: str = Form(None),
    target_image: UploadFile = File(None),
    prompt: str = Form(None)
):
    print("Received face swap request")
    
    # Generate target template image from prompt if provided
    if prompt:
        print(f"Generating template body for face swap from prompt: {prompt}")
        token = os.getenv("REPLICATE_API_TOKEN")
        # Always enforce: clear frontal face, close-up enough for InsightFace to detect
        enhanced_prompt = (
            f"{prompt}, full body portrait photo, face clearly visible, "
            "looking at camera, sharp focus on face, realistic, photorealistic, "
            "professional photography, cinematic lighting, 8k"
        )
        
        # Try Replicate Flux Dev
        if token:
            try:
                print("Generating template body using Replicate Flux Dev...")
                client = replicate.Client(api_token=token)
                output = client.run(
                    "black-forest-labs/flux-dev",
                    input={
                        "prompt": enhanced_prompt,
                        "go_fast": True,
                        "megapixels": "1",
                        "num_outputs": 1,
                        "aspect_ratio": "4:5",
                        "output_format": "jpg",
                        "output_quality": 90
                    }
                )
                output_url = None
                if isinstance(output, list) and len(output) > 0:
                    output_url = output[0]
                elif isinstance(output, str):
                    output_url = output
                
                if output_url:
                    target_image_url = output_url
            except Exception as e:
                print(f"Replicate Flux Dev template generation error: {e}. Falling back to Pollinations...")
                
        if not target_image_url:
            # Fallback to Pollinations.ai URL
            import random
            seed = random.randint(0, 999999)
            encoded = urllib.request.quote(enhanced_prompt)
            target_image_url = f"https://image.pollinations.ai/prompt/{encoded}?width=800&height=1000&seed={seed}&nologo=true&model=flux"
            
    if USE_LOCAL_FACESWAP:
        try:
            # Read the uploaded face image
            contents = await swap_image.read()
            swap_img = Image.open(BytesIO(contents)).convert("RGB")
            
            # Get target image (either from file upload or URL download)
            if target_image is not None:
                print("Reading target template from uploaded file...")
                target_contents = await target_image.read()
                target_img = Image.open(BytesIO(target_contents)).convert("RGB")
            elif target_image_url:
                print(f"Downloading target template from URL: {target_image_url}")
                req = urllib.request.Request(target_image_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    target_data = resp.read()
                target_img = Image.open(BytesIO(target_data)).convert("RGB")
            else:
                raise Exception("Thiếu ảnh mẫu thiết kế (Target Image).")
            
            # Run local face swap using InsightFace
            result_img = swap_faces_local(swap_img, target_img)
            
            # Save to buffer and return
            img_io = BytesIO()
            result_img.save(img_io, 'JPEG', quality=95)
            img_io.seek(0)
            
            print("Local face swap complete!")
            return StreamingResponse(img_io, media_type="image/jpeg")
            
        except Exception as e:
            print(f"Local face swap error: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})
            
    else:
        # ---------------- Replicate Cloud API (Backup Option) ----------------
        token = os.getenv("REPLICATE_API_TOKEN")
        if not token:
            return JSONResponse(
                status_code=400,
                content={"error": "Chưa cấu hình REPLICATE_API_TOKEN trên server. Vui lòng thêm Secret Key này trên Hugging Face."}
            )
        
        import tempfile
        tmp_path = None
        try:
            # Read the uploaded face image contents
            contents = await swap_image.read()
            
            # Write to a temporary file so the replicate client can upload it correctly with a filename
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
                
            print(f"Calling Replicate API for face swap with target template: {target_image_url}")
            
            # Initialize Replicate client
            client = replicate.Client(api_token=token)
            
            # Open and pass the temporary file to Replicate client
            with open(tmp_path, "rb") as swap_file:
                print("Running Replicate face swap prediction...")
                output_url = client.run(
                    "lucataco/faceswap:9a4298548422074c3f57258c5d544497314ae4112df80d116f0d2109e843d20d",
                    input={
                        "target_image": target_image_url,
                        "swap_image": swap_file
                    }
                )
                
            if not output_url:
                raise Exception("Replicate did not return any output image URL.")
                
            print(f"Face swap complete! Output URL: {output_url}")
            
            # Fetch the resulting image from the returned URL
            req = urllib.request.Request(output_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                result_data = resp.read()
                
            return StreamingResponse(BytesIO(result_data), media_type="image/jpeg")
            
        except Exception as e:
            print(f"Face swap error: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})
            
        finally:
            # Clean up temporary file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception as ex:
                    print(f"Error removing temp file: {ex}")


# -------------------------------------------------------------
# MAIN RUNNER
# -------------------------------------------------------------
if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    kill_port_owner(port)
    print(f"\nStarting FastAPI Server on http://{host}:{port}...")
    print("Keep this terminal window open while using the Web App.")
    
    # Run uvicorn server directly in the main thread
    uvicorn.run(app, host=host, port=port, log_level="info")

