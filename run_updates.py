import os
import re

# Details definitions
items = [
    {
        "title": "The Heuristic Manual Selection Era (Chen et al., 2016)",
        "file": "details/heuristic-manual-selection-era.md",
        "content": "# The Heuristic Manual Selection Era (Chen et al., 2016)\n\nDetailed explanation of heuristic manual selection.\n\n```mermaid\ngraph TD\n    A[Forward Pass] --> B[Save Checkpoints]\n    B --> C[Discard Intermediate]\n    C --> D[Backward Pass Recompute]\n```\n\n[Back to README](../README.md)\n"
    },
    {
        "title": "The Automated Graph Solver Era (~2019–2022)",
        "file": "details/automated-graph-solver-era.md",
        "content": "# The Automated Graph Solver Era (~2019–2022)\n\nAutomated graph-based solver explanation.\n\n```mermaid\ngraph TD\n    A[Computation Graph] --> B[Solver]\n    B --> C[Optimal Checkpoints]\n```\n\n[Back to README](../README.md)\n"
    },
    {
        "title": "The Selective & Fused Rematerialization Era (~2023–Present)",
        "file": "details/selective-fused-rematerialization-era.md",
        "content": "# The Selective & Fused Rematerialization Era (~2023–Present)\n\nSelective rematerialization details.\n\n```mermaid\ngraph TD\n    A[Heavy Matrix Ops] --> B[Keep in VRAM]\n    C[Cheap Activations] --> D[Rematerialize]\n```\n\n[Back to README](../README.md)\n"
    },
    {
        "title": "Full (Grandchild) Checkpointing",
        "file": "details/full-checkpointing.md",
        "content": "# Full (Grandchild) Checkpointing\n\nDetails of full checkpointing.\n\n```mermaid\ngraph TD\n    A[Input] --> B[Layer 1]\n    B --> C[Layer 2]\n    C --> D[Output]\n    D --> E[Recompute Layer 1 & 2]\n```\n\n[Back to README](../README.md)\n"
    },
    {
        "title": "Selective Checkpointing",
        "file": "details/selective-checkpointing.md",
        "content": "# Selective Checkpointing\n\nDetails of selective checkpointing.\n\n```mermaid\ngraph TD\n    A[Activation] --> B{Is Heavy?}\n    B -- Yes --> C[Keep]\n    B -- No --> D[Recompute]\n```\n\n[Back to README](../README.md)\n"
    },
    {
        "title": "Offloaded Checkpointing (CPU Offloading)",
        "file": "details/offloaded-checkpointing.md",
        "content": "# Offloaded Checkpointing (CPU Offloading)\n\nDetails of offloading.\n\n```mermaid\ngraph TD\n    A[GPU VRAM] --> B[PCIe]\n    B --> C[CPU RAM]\n```\n\n[Back to README](../README.md)\n"
    },
    {
        "title": "PyTorch Native `torch.utils.checkpoint`",
        "file": "details/pytorch-native-checkpoint.md",
        "content": "# PyTorch Native `torch.utils.checkpoint`\n\nDetails of PyTorch implementation.\n\n```mermaid\ngraph TD\n    A[torch.utils.checkpoint] --> B[Dummy Forward]\n    B --> C[Autograd Hook]\n```\n\n[Back to README](../README.md)\n"
    },
    {
        "title": "Megatron-LM Selective Activation Checkpointing",
        "file": "details/megatron-lm-selective-checkpointing.md",
        "content": "# Megatron-LM Selective Activation Checkpointing\n\nDetails of Megatron-LM.\n\n```mermaid\ngraph TD\n    A[Transformer Layer] --> B[Attention]\n    B --> C[MLP]\n```\n\n[Back to README](../README.md)\n"
    },
    {
        "title": "Unsloth / Triton Fused Kernel Rematerialization",
        "file": "details/unsloth-triton-fused.md",
        "content": "# Unsloth / Triton Fused Kernel Rematerialization\n\nDetails of Triton fusion.\n\n```mermaid\ngraph TD\n    A[Python Code] --> B[Triton Compiler]\n    B --> C[Optimized PTX]\n```\n\n[Back to README](../README.md)\n"
    },
    {
        "title": "The Activation Memory Bottleneck",
        "file": "details/activation-memory-bottleneck.md",
        "content": "# The Activation Memory Bottleneck\n\nDetails of memory bottleneck.\n\n```mermaid\ngraph TD\n    A[Sequence Length] --> B[Quadratic Memory]\n```\n\n[Back to README](../README.md)\n"
    },
    {
        "title": "The 33% Compute Penalty Math",
        "file": "details/33-percent-compute-penalty.md",
        "content": "# The 33% Compute Penalty Math\n\nDetails of 33% penalty.\n\n```mermaid\ngraph TD\n    A[1F + 1B] --> B[1F + 1F_recompute + 1B]\n```\n\n[Back to README](../README.md)\n"
    },
    {
        "title": "Pre-Training Million-Context Foundation Models",
        "file": "details/pre-training-million-context.md",
        "content": "# Pre-Training Million-Context Foundation Models\n\nDetails of million context models.\n\n```mermaid\ngraph TD\n    A[1M Tokens] --> B[Ring Attention]\n```\n\n[Back to README](../README.md)\n"
    },
    {
        "title": "Full Fine-Tuning of Ultra-Large Parameter LLMs",
        "file": "details/full-fine-tuning-ultra-large.md",
        "content": "# Full Fine-Tuning of Ultra-Large Parameter LLMs\n\nDetails of ultra-large LLM tuning.\n\n```mermaid\ngraph TD\n    A[70B Model] --> B[ZeRO-3]\n    B --> C[Full Fine Tuning]\n```\n\n[Back to README](../README.md)\n"
    },
    {
        "title": "High-Resolution 3D Spatio-Temporal Video Generative Training",
        "file": "details/high-resolution-3d-spatio-temporal.md",
        "content": "# High-Resolution 3D Spatio-Temporal Video Generative Training\n\nDetails of video generation.\n\n```mermaid\ngraph TD\n    A[Video Sequence] --> B[3D Convolution]\n```\n\n[Back to README](../README.md)\n"
    }
]

os.makedirs('details', exist_ok=True)
for item in items:
    with open(item['file'], 'w', encoding='utf-8') as f:
        f.write(item['content'])

# Update README with links to details
with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

for item in items:
    # We replace exactly the bold text with a markdown link to the file.
    # The titles in README are bolded like **Title**
    target = f"**{item['title']}**"
    replacement = f"**[{item['title']}]({item['file']})**"
    readme = readme.replace(target, replacement)

# Create banner SVG
os.makedirs('assets', exist_ok=True)
svg_content = """<svg width="800" height="200" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#2c3e50"/>
  <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-size="40" fill="#ecf0f1" font-family="Arial, sans-serif">Awesome Activation Checkpointing</text>
  <text x="50%" y="75%" dominant-baseline="middle" text-anchor="middle" font-size="20" fill="#bdc3c7" font-family="Arial, sans-serif">Evolution, Variants, Types, &amp; Applications</text>
</svg>"""
with open('assets/banner.svg', 'w', encoding='utf-8') as f:
    f.write(svg_content)

# Add badges, banner, and emojis to README
badges = """<div align="center">

<a href="https://github.com/ishandutta2007/Awesome-Awesome-Awesome"><img src="https://img.shields.io/badge/Awesome-%E2%9C%94-blueviolet?style=flat-square&logo=github" alt="Awesome"/></a><a href="https://discord.gg/jc4xtF58Ve"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord" /></a>
![License](https://img.shields.io/badge/license-MIT-blue.svg)
<a href="https://github.com/ishandutta2007"><img alt="GitHub followers" src="https://img.shields.io/github/followers/ishandutta2007?label=Follow" /></a>

<img src="assets/banner.svg" alt="Banner" width="100%" />

</div>

"""

# Prepend to README
readme = badges + "🚀 " + readme.replace("# Awesome-Activation-Checkpointing", "# Awesome-Activation-Checkpointing 🧠", 1)
readme = readme.replace("## 1. The Chronological Evolution", "## 🕰️ 1. The Chronological Evolution")
readme = readme.replace("## 2. Core Functional & Functional Variants", "## ⚙️ 2. Core Functional & Functional Variants")
readme = readme.replace("## 3. Structural Storage & System Implementation Types", "## 💾 3. Structural Storage & System Implementation Types")
readme = readme.replace("## 4. Production Scaling Laws & Hardware Trade-Offs", "## ⚖️ 4. Production Scaling Laws & Hardware Trade-Offs")
readme = readme.replace("## 5. Frontier Distributed Applications", "## 🚀 5. Frontier Distributed Applications")

# Replace awesome link and chartrepos
readme = readme.replace("https://github.com/sindresorhus/awesome", "https://github.com/ishandutta2007/Awesome-Awesome-Awesome")
readme = readme.replace("chartrepos", "chart?repos")

# Add star history
folder_name = os.path.basename(os.getcwd())
star_history = f"""

## 🌟 Star History
<div align="center">
<a href="https://www.star-history.com/?repos=ishandutta2007%2F{folder_name}&type=date&legend=bottom-right">
<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=ishandutta2007/{folder_name}&type=date&theme=dark&legend=bottom-right" />
<source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=ishandutta2007/{folder_name}&type=date&legend=bottom-right" />
<img alt="Star History Chart" src="https://api.star-history.com/chart?repos=ishandutta2007/{folder_name}&type=date&legend=bottom-right" />
</picture>
</a>
</div>
"""
readme += star_history

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme)
