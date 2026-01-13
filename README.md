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

https://github.com/user-attachments/assets/32c3a79f-d444-4dc9-8fab-aaf6849e2b86


## 🚀 Workflows

### ComfyUI
1.  Load your standard Wan2.2 workflow.
2.  Insert a `Load LoRA` node.
3.  Connect **Wan22_shot_lora** to the main model path.
4.  Ensure your positive prompt includes the scene transition description.

### Diffusers (Python)
```python
import torch
from diffusers import WanPipeline # Or appropriate pipeline

pipe = WanPipeline.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B", torch_dtype=torch.float16)
pipe.load_lora_weights("path/to/Wan22_shot_lora.safetensors", adapter_name="shot_change")

prompt = "A man drinking coffee, cut to a busy new york street."
video = pipe(prompt, cross_attention_kwargs={"scale": 0.8}).frames[0]
