import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, infer_auto_device_map
import os
import numpy as np
import random
import transformers
import torch
import time
import asyncio
from quart import Quart, jsonify, request

app = Quart(__name__)

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available")
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("GPU is not available")

model_name_or_path = "/code/knezevic/meta-llama/Llama-3.3-70B-Instruct"  # path to the model - change if needed
start_time = time.time()
print("loading model...")

with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

max_mem = {i: "28GiB" for i in range(2)}
max_mem["cpu"] = "20Gib"
model.tie_weights()
device_map = infer_auto_device_map(model, max_memory=max_mem)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path, device_map="auto", torch_dtype=torch.float16
)

model.eval()
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Text generation pipeline
generator = transformers.pipeline(
    "text-generation", model=model, tokenizer=tokenizer, device_map="auto"
)

end_time = time.time()

print(f"Model loaded after {end_time - start_time}")

# Set seed for deterministic behavior
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

generator_lock = asyncio.Lock()


@app.route("/inference", methods=["POST"])
async def handle_request():
    data = await request.get_json()

    try:

        async with generator_lock:
            with torch.no_grad():
                try:
                    outputs = await asyncio.to_thread(
                        generator,
                        data["prompt"],
                        do_sample=data["do_sample"],
                        temperature=data["temperature"],
                        max_new_tokens=data["max_new_tokens"],
                        stop_strings=data["stop_strings"],
                        tokenizer=tokenizer,
                        past_key_values=None,
                        topk=data["top_k"],
                        top_p=data["top_p"],
                    )
                except Exception as e:
                    print("Error in generator:", e)
                    print("Generation stopped")
        output = outputs

    except Exception as e:
        return jsonify({"exception": e, "error": True}), 200
        # raise HTTPException(status_code=500, detail=str(e))
    # Process the data or trigger any action here
    return (
        jsonify({"stop": data["stop_strings"], "output": output, "error": False}),
        200,
    )


@app.route("/reset", methods=["POST"])
async def handle_request_reset():
    data = await request.get_json()

    # set seed
    seed = int(data["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return jsonify({"reset": True}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
