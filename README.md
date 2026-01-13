# Wan22_shot_lora 🎬

![Model Status](https://img.shields.io/badge/Status-Active-success) ![Task](https://img.shields.io/badge/Task-Text2Video%20%7C%20Image2Video-blue)

**Wan22_shot_lora** is a specialized Low-Rank Adaptation (LoRA) model designed for the Wan2.2 video generation architecture. 

Its primary goal is to **break the continuity** typically found in AI videos and introduce **dynamic shot changes, scene cuts, and transitions**. It works effectively for both Text-to-Video (T2V) and Image-to-Video (I2V) workflows.

## ✨ Key Features

*   **Cinematic Cuts:** Enables the model to generate videos that switch between different angles (e.g., Close-up → Wide shot) or different scenes entirely.
*   **Dual Mode Support:**
    *   **Text-to-Video:** Describe a sequence of events, and the model will execute the cut.
    *   **Image-to-Video:** Start with an input image, and prompt the model to transition into a new scene.
*   **Enhanced Dynamics:** Reduces the "static" or "morphing" feel of standard video generation, creating a more edited, movie-like feel.

## 📥 Download

*   **HuggingFace:** [Link to your HF repo]
*   **Civitai:** [Link to your Civitai page]

## 🛠️ Usage

### Trigger Words
To activate the shot change effect, it is recommended to use the following trigger words in your prompt:
> **`镜头切换`**

### Recommended LoRA Weight
*   **Strength:** `0.6` to `1.0`
*   If the cut is too abrupt or glitchy, lower the weight. If the scene just morphs without a clear cut, increase the weight.

### Prompting Strategy (How to get the best results)

The key to getting a good shot change is to describe **two distinct states** in your prompt.

**Formula:**
`[Description of Scene A] + [镜头切换] + [Description of Scene B]`

**Examples:**
*   **T2V:** "特写镜头展示一位女性的眼睛，镜头切换到一幅展示夜晚未来赛博朋克城市的广角无人机镜头。"
*   **I2V (with input image of a car):** "汽车驶下高速公路，镜头切换，转场到海洋上空的日落景象。"

  

https://github.com/user-attachments/assets/5dc36db9-5adf-4e51-a324-c042390d17a3

https://github.com/user-attachments/assets/7c2cc389-75d7-4e19-b879-0d8240859d66

### ComfyUI
1.  Load your standard Wan2.2 workflow.
2.  Insert a `Load LoRA` node.
3.  Connect **Wan22_shot_lora** to the main model path.
4.  Ensure your positive prompt includes the scene transition description.

### Diffusers (Python)
```python t2v
import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda:7",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.2-T2V-A14B", origin_file_pattern="Wan2.1_VAE.pth"),
    ],
)
pipe.load_lora(pipe.dit, "step-1000.safetensors", alpha=1)

video = pipe(
    prompt="特写镜头展示一位女性的眼睛，镜头切换到一幅展示夜晚未来赛博朋克城市的广角无人机镜头。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    num_frames=49,
    seed=1, tiled=True
)
save_video(video, "video_Wan2.2-T2V-A14B.mp4", fps=15, quality=5)
