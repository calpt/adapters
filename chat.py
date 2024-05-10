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

adapter_path = "AdapterHub/llama2-7b-qlora-openassistant"
# adapter_path = "output/llama_qlora_7b_seq_bn/assistant_adapter"

model.load_adapter(adapter_path, source="hf", set_active=True)

print(model.adapter_summary())

# for _, v in model.get_adapter("assistant_adapter").items():
#     for _, module in v.items():
#         module.to("cuda")

# model.push_adapter_to_hub(
#     "AdapterHub/llama2-13b-qlora-openassistant",
#     "assistant_adapter",
#     datasets_tag="timdettmers/openassistant-guanaco",
#     local_path="qlora-oasst1-adapter-repo"
# )
# exit()

# Ignore warnings
from transformers import logging
logging.set_verbosity(logging.CRITICAL)

from transformers import StoppingCriteria
class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence = [12968, 29901]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:,-len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids

def prompt_model(model, text: str):
    batch = tokenizer(f"### Human: {text} ### Assistant:", return_tensors="pt")
    batch = batch.to(model.device)
    
    model.eval()
    with torch.inference_mode(), torch.cuda.amp.autocast():
            output_tokens = model.generate(**batch, stopping_criteria=[EosListStoppingCriteria()])

    decoded = tokenizer.decode(output_tokens[0, batch["input_ids"].shape[1]:], skip_special_tokens=True)
    return decoded[:-10] if decoded.endswith("### Human:") else decoded

while True:
    text = input("You: ")
    if text == "exit":
        break
    response = prompt_model(model, text)
    print(f"Assistant: {response}")
