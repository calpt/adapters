import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

modelpath="meta-llama/Llama-2-7b-hf"

# Load 4-bit quantized model
model = AutoModelForCausalLM.from_pretrained(
    modelpath,    
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
    torch_dtype=torch.bfloat16,
)
# model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(modelpath)
# tokenizer.pad_token = tokenizer.eos_token

import adapters
from adapters import LoRAConfig

adapters.init(model)

import sys

adapter_path = sys.argv[1]

model.load_adapter(adapter_path, set_active=True)

print(model.adapter_summary())

model.push_adapter_to_hub(
    "AdapterHub/llama2-7b-qadapter-seq-openassistant",
    "assistant_adapter",
    datasets_tag="timdettmers/openassistant-guanaco",
    local_path="qadapter-oasst1-adapter-repo"
)
