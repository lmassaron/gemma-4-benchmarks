import torch
from PIL import Image
import time
import json
import os
import sys
import re
from transformers import AutoProcessor, AutoModelForMultimodalLM

# Constants for test prompts
POEM_PROMPT = "Write a short poem about the fusion of art and technology."
BATCH_PROMPTS = [
    "What is the significance of the Turing test?",
    "Explain the concept of entropy in thermodynamics."
]
LOGIC_PROMPT = "If all Bloops are Razzies and all Razzies are Lurgies, are all Bloops Lurgies? Explain why."
SINGLE_IMG_PROMPT = "Describe the action in this image in detail."
MULTI_IMG_PROMPT = "What are the similarities between these three images?"
COW_PROMPT = "Identify and validate the biological traits in this image. How many legs can you see?"
FC_PROMPT = "What is the weather in London?"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Gets the current weather for a specific location.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name"},
            },
            "required": ["city"],
        },
    },
}

def parse_thinking_output(text):
    # Support both <thought> and <|think|> delimiters
    thought_match = re.search(r'<thought>(.*?)</thought>', text, re.DOTALL)
    if not thought_match:
        thought_match = re.search(r'<\|think\|>(.*?)<turn\|>', text, re.DOTALL)
        
    if thought_match:
        thinking = thought_match.group(1).strip()
        answer = text.replace(thought_match.group(0), "").strip()
    else:
        thinking = None
        answer = text.strip()
    return {"thinking": thinking, "answer": answer}

def create_notebook(model_id, results):
    safe_name = model_id.replace("/", "_")
    
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# Official Benchmark: {model_id}\n",
                "\n",
                "This notebook presents the official benchmark results for the Gemma 4 model series.\n",
                "\n",
                "## Capabilities Tested:\n",
                "1.  **Text Generation**: Creative (Single) and Analytical (Batch).\n",
                "2.  **Thinking Mode**: Chain-of-Thought (CoT) reasoning with `enable_thinking=True`.\n",
                "3.  **Multimodal Inference**: Single Image, Multi-Image, and Visual Logic (Cow).\n",
                "4.  **Function Calling**: Native tool use with structured output.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "from PIL import Image\n",
                "from transformers import AutoProcessor, AutoModelForMultimodalLM\n",
                "\n",
                f"MODEL_ID = \"{model_id}\"\n",
                "print(\"Loading model...\")\n",
                "model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map='auto')\n",
                "processor = AutoProcessor.from_pretrained(MODEL_ID)\n",
                "print(\"Model loaded successfully!\")"
            ]
        }
    ]

    # Add text results
    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## 1. Text Generation\n"]})
    cells.append({
        "cell_type": "code", "execution_count": 2, "metadata": {},
        "outputs": [{"data": {"text/markdown": [f"### Creative Poem\n{results['answers']['text_single']}"]}, "output_type": "display_data"}],
        "source": [f"# TPS: {results['metrics']['text_single_tps']:.2f}"]
    })
    
    # Add thinking results
    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## 2. Thinking Mode (Reasoning)\n"]})
    parsed = parse_thinking_output(results['answers']['thinking'])
    cells.append({
        "cell_type": "code", "execution_count": 3, "metadata": {},
        "outputs": [
            {"data": {"text/markdown": ["### Thinking Process"]}, "output_type": "display_data"},
            {"data": {"text/markdown": [f"> {parsed['thinking']}"]}, "output_type": "display_data"},
            {"data": {"text/markdown": ["### Final Answer"]}, "output_type": "display_data"},
            {"data": {"text/markdown": [f"{parsed['answer']}"]}, "output_type": "display_data"}
        ],
        "source": [f"# TPS: {results['metrics']['thinking_tps']:.2f}"]
    })

    # Add vision results
    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## 3. Multimodal Inference\n"]})
    cells.append({
        "cell_type": "code", "execution_count": 4, "metadata": {},
        "outputs": [{"data": {"text/markdown": [f"### Single Image Description\n{results['answers'].get('vision_single', 'N/A')}"]}, "output_type": "display_data"}],
        "source": [f"# TPS: {results['metrics'].get('vision_single_tps', 0):.2f}"]
    })
    
    cells.append({
        "cell_type": "code", "execution_count": 5, "metadata": {},
        "outputs": [{"data": {"text/markdown": [f"### Visual Logic (Cow Test)\n{results['answers'].get('vision_cow', 'N/A')}"]}, "output_type": "display_data"}],
        "source": [f"# TPS: {results['metrics'].get('vision_cow_tps', 0):.2f}"]
    })

    # Add function calling results
    cells.append({"cell_type": "markdown", "metadata": {}, "source": ["## 4. Function Calling\n"]})
    cells.append({
        "cell_type": "code", "execution_count": 6, "metadata": {},
        "outputs": [{"data": {"text/markdown": [f"### Tool Call Output\n```text\n{results['answers']['fc']}\n```"]}, "output_type": "display_data"}],
        "source": [f"# TPS: {results['metrics']['fc_tps']:.2f}"]
    })

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(f"official_{safe_name}.ipynb", "w") as f:
        json.dump(notebook, f, indent=2)

def run_benchmark(model_id):
    print(f"\n\n>>> STARTING BENCHMARK: {model_id} <<<")
    
    try:
        t0 = time.time()
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        print(f"Model loaded in {time.time() - t0:.2f}s")
    except Exception as e:
        print(f"CRITICAL ERROR LOADING {model_id}: {e}")
        return None

    results = {"metrics": {}, "answers": {}}

    def generate(prompt_text, imgs=None, use_thinking=False, tools=None):
        if imgs:
            content = []
            for _ in range(len(imgs)):
                content.append({"type": "image"})
            content.append({"type": "text", "text": prompt_text})
        else:
            content = [{"type": "text", "text": prompt_text}]
            
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=use_thinking)
        
        inputs = processor(text=text, images=imgs, return_tensors='pt').to(model.device)
        input_len = inputs["input_ids"].shape[-1]
        
        t0 = time.time()
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=1000, do_sample=False)
        t1 = time.time()
        
        num_gen_tokens = outputs[0][input_len:].shape[-1]
        tps = num_gen_tokens / (t1 - t0)
        decoded = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        return decoded, tps

    # 1. Text Single
    print("Running Text Single...")
    ans, tps = generate(POEM_PROMPT)
    results["metrics"]["text_single_tps"] = tps
    results["answers"]["text_single"] = ans
    print(f"  TPS: {tps:.2f}")

    # 2. Text Batch (Analytical)
    print("Running Text Batch...")
    batch_tps = []
    batch_ans = []
    for p in BATCH_PROMPTS:
        ans, tps = generate(p)
        batch_tps.append(tps)
        batch_ans.append(ans)
    results["metrics"]["text_batch_tps"] = sum(batch_tps) / len(batch_tps)
    results["answers"]["text_batch"] = batch_ans
    print(f"  Avg TPS: {results['metrics']['text_batch_tps']:.2f}")

    # 3. Thinking Mode
    print("Running Thinking Mode...")
    ans, tps = generate(LOGIC_PROMPT, use_thinking=True)
    results["metrics"]["thinking_tps"] = tps
    results["answers"]["thinking"] = ans
    print(f"  TPS: {tps:.2f}")

    # 4. Vision Single
    if os.path.exists("image.jpg"):
        print("Running Vision Single...")
        img = [Image.open("image.jpg")]
        ans, tps = generate(SINGLE_IMG_PROMPT, imgs=img)
        results["metrics"]["vision_single_tps"] = tps
        results["answers"]["vision_single"] = ans
        print(f"  TPS: {tps:.2f}")

    # 5. Vision Multi
    img_paths = ["image_0.jpg", "image_1.jpg", "image_2.jpg"]
    if all(os.path.exists(p) for p in img_paths):
        print("Running Vision Multi...")
        imgs = [Image.open(p) for p in img_paths]
        ans, tps = generate(MULTI_IMG_PROMPT, imgs=imgs)
        results["metrics"]["vision_multi_tps"] = tps
        results["answers"]["vision_multi"] = ans
        print(f"  TPS: {tps:.2f}")

    # 6. Vision Cow (Visual Logic)
    cow_path = "gemma-4-eap-extras/cow.jpg"
    if os.path.exists(cow_path):
        print("Running Vision Cow...")
        img = [Image.open(cow_path)]
        ans, tps = generate(COW_PROMPT, imgs=img, use_thinking=True)
        results["metrics"]["vision_cow_tps"] = tps
        results["answers"]["vision_cow"] = ans
        print(f"  TPS: {tps:.2f}")

    # 7. Function Calling
    print("Running Function Calling...")
    ans, tps = generate(FC_PROMPT, tools=[WEATHER_TOOL])
    results["metrics"]["fc_tps"] = tps
    results["answers"]["fc"] = ans
    print(f"  TPS: {tps:.2f}")

    # Finalize
    create_notebook(model_id, results)
    
    del model
    del processor
    torch.cuda.empty_cache()
    return results

if __name__ == "__main__":
    MODELS = [
        "google/gemma-4-E2B-it",
        "google/gemma-4-E4B-it",
        "google/gemma-4-26B-A4B-it",
        "google/gemma-4-31B-it"
    ]
    
    master_results = {}
    if os.path.exists("master_benchmark_results.json"):
        with open("master_benchmark_results.json", "r") as f:
            master_results = json.load(f)
            
    # Allow running specific model or all
    targets = MODELS
    if len(sys.argv) > 1:
        targets = [sys.argv[1]] if sys.argv[1] in MODELS else [m for m in MODELS if sys.argv[1] in m]
        if not targets:
            print(f"Model {sys.argv[1]} not found in {MODELS}")
            sys.exit(1)

    for m_id in targets:
        res = run_benchmark(m_id)
        if res:
            master_results[m_id] = res
            with open("master_benchmark_results.json", "w") as f:
                json.dump(master_results, f, indent=2)
    
    print("\n\n>>> MASTER BENCHMARK COMPLETE <<<")
