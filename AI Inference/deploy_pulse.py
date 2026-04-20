import modal
import io
import os

# ==========================================
# 1. DEFINE THE ENVIRONMENT (The Docker Image)
# ==========================================
pulse_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git") 
    .pip_install(
        "torch",
        "torchvision",
        "transformers==4.35.2",
        "accelerate",
        "bitsandbytes",
        "pillow",
        "huggingface_hub" # Added so we can download the model via CLI
    )
    .run_commands("pip install git+https://github.com/AIMedLab/PULSE.git#subdirectory=LLaVA")
    # 🔥 THE CACHE FIX: Download the model during the build phase so it never re-downloads
    .run_commands("huggingface-cli download PULSE-ECG/PULSE-7B")
)

app = modal.App("pulse-ecg-analyzer")

# ==========================================
# 2. DEFINE THE MODEL CLASS
# ==========================================
@app.cls(image=pulse_image, gpu="A10G", timeout=600, min_containers=1)
class PulseECGModel:
    
    @modal.enter()
    def load_model(self):
        print("🚀 Booting container and loading PULSE-7B from local cache...")
        import torch
        import transformers
        from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPVisionModel
        from llava.model import LlavaLlamaForCausalLM

        original_to = transformers.modeling_utils.PreTrainedModel.to
        def safe_to(self, *args, **kwargs):
            try: return original_to(self, *args, **kwargs)
            except ValueError as e:
                if "is not supported for `4-bit`" in str(e): return self
                raise e
        transformers.modeling_utils.PreTrainedModel.to = safe_to
        CLIPVisionModel._no_split_modules = ["CLIPEncoderLayer"]

        self.MODEL_PATH = "PULSE-ECG/PULSE-7B"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", 
            bnb_4bit_compute_dtype=torch.float16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_PATH, use_fast=False)
        self.model = LlavaLlamaForCausalLM.from_pretrained(
            self.MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )

        for name, module in self.model.named_modules():
            if "rotary_emb" in name.lower():
                module.to("cuda")

        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded: 
            vision_tower.load_model()
        vision_tower.to(device="cuda", dtype=torch.float16)
        
        self.image_processor = vision_tower.image_processor
        print("Model loaded into VRAM!")

    @modal.method()
    def analyze(self, image_bytes: bytes) -> str:
        import torch
        from PIL import Image
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model.config).to("cuda", dtype=torch.float16)

        # 🔥 THE NEW PROMPT
        question = """Analyze this ECG. Output strictly formatted JSON.
        {
        "overall_interpretation": "Choose EXACTLY ONE: 'Normal ECG', 'Abnormal ECG', 'Borderline ECG'",
        "findings": [
            "Select (0-5) from the following list ONLY IF definitive diagnostic criteria are met. DO NOT guess. If you do not see clear evidence, omit it. 
            Exact List: 
            'Sinus rhythm', 'Sinus Bradycardia', 'Sinus Tachycardia', 'Sinus Arrhythmia', 
            'Atrial Fibrillation', 'Atrial Flutter', 'Supraventricular Tachycardia (SVT)', 
            'Premature Ventricular Contractions (PVCs)', 'Premature Atrial Contractions (PACs)', 
            'Ventricular Tachycardia', 'Idioventricular Rhythm', 'Junctional Rhythm', 
            'First-Degree AV Block', 'Second-Degree AV Block', 'Third-Degree AV Block', 
            'Right Bundle Branch Block (RBBB)', 'Left bundle branch block', 
            'Left Anterior Fascicular Block (LAFB) / Left Axis Deviation (LAD)', 
            'Right Axis Deviation (RAD) / Left Posterior Fascicular Block (LPFB)', 
            'Intraventricular Conduction Delay (IVCD)', 'Wolff-Parkinson-White (WPW) Pattern', 
            'Left ventricular hypertrophy', 'Right Ventricular Hypertrophy (RVH)', 
            'Probable left atrial enlargement', 'Right Atrial Enlargement (RAE)', 
            'Anterior Myocardial Infarction', 'Inferior Myocardial Infarction', 'Lateral Myocardial Infarction', 
            'ST/T Wave Abnormalities', 'Prolonged QT interval', 
            'Abnormal R-wave progression, early transition', 'ST elev, probable normal early repol pattern', 
            'Electronic Pacemaker'. 
            Return an empty list [] if none apply."
        ]",
        "summary_report": "Write a brief 3-4 sentence clinical summary explaining the key visual findings."
        }"""
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv = conv_templates["llava_v1"].copy()
        conv.system = "You are an expert, highly precise AI cardiologist analyzing ECGs. You only output valid JSON."
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        
        final_prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(final_prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to("cuda")

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=False,
                max_new_tokens=512,
                use_cache=False
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        
    @modal.method()
    def analyze_dynamic(self, prompt_text: str, image_bytes: bytes | None = None) -> str:
        import torch
        from PIL import Image
        import io # Ensure io is imported here
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates

        conv = conv_templates["llava_v1"].copy()
        conv.system = "You are a helpful cardiac medical assistant. Answer concisely to user queries."

        if image_bytes:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image_tensor = process_images([image], self.image_processor, self.model.config).to("cuda", dtype=torch.float16)
            
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            
            input_ids = tokenizer_image_token(conv.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to("cuda")

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=False,
                    max_new_tokens=512,
                    use_cache=False
                )
        else:
            conv.append_message(conv.roles[0], prompt_text)
            conv.append_message(conv.roles[1], None)
            
            input_ids = self.tokenizer(conv.get_prompt(), return_tensors='pt').input_ids.to("cuda")

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=None, 
                    do_sample=False,
                    max_new_tokens=512,
                    use_cache=False
                )

        # 🚀 THE CRITICAL FIX: Slice out the prompt from the output
        input_token_len = input_ids.shape[1]
        generated_ids = output_ids[:, input_token_len:]
        
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

# ==========================================
# 3. LOCAL EXECUTION (How you run it)
# ==========================================

# @app.local_entrypoint()
# def test_dynamic(prompt: str, image_path: str = None):
#     """
#     Tests the new analyze_dynamic method from the command line.
#     """
#     image_bytes = None
    
#     if image_path:
#         if not os.path.exists(image_path):
#             print(f"❌ Error: Could not find image at {image_path}")
#             return
#         with open(image_path, "rb") as f:
#             image_bytes = f.read()
#         print(f"📤 Sending Text + Image ({image_path}) to Modal...")
#     else:
#         print(f"📤 Sending Text-Only prompt to Modal...")
        
#     model = PulseECGModel()
    
#     # Trigger the NEW dynamic method
#     result = model.analyze_dynamic.remote(prompt_text=prompt, image_bytes=image_bytes)
    
#     print("\n" + "="*50)
#     print("🏥 PULSE-7B DYNAMIC OUTPUT:")
#     print(result)
#     print("="*50)