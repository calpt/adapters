# %% [markdown]
# # Finetuning Llama 2 with _Adapters_ and QLoRA
# 
# In this notebook, we show how to efficiently fine-tune a quantized 7B Llama 2 model using [**QLoRA** (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314) and the [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) library.
# 
# Specifically, we finetune Llama 2 on a supervised **instruction tuning** dataset collected by the [Open Assistant project](https://github.com/LAION-AI/Open-Assistant) for training chatbot models. This is similar to the setup used to train the Guanaco models in the QLoRA paper.

# %% [markdown]
# ## Installation
# 
# Besides `adapters`, we require `bitsandbytes` for quantization and `accelerate` for training.

# %%
# !pip install -qq -U adapters accelerate bitsandbytes datasets

# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["WANDB_ENTITY"] = "clif"
os.environ["WANDB_PROJECT"] = "adapters"

# %% [markdown]
# ## Load Open Assistant dataset
# 
# We use the [`timdettmers/openassistant-guanaco`](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) dataset by the QLoRA, which contains a small subset of conversations from the full Open Assistant database and was also used to finetune the Guanaco models in the QLoRA paper.

# %%
from datasets import load_dataset

dataset = load_dataset("timdettmers/openassistant-guanaco")

# %% [markdown]
# Our training dataset has roughly 10k training samples:

# %%
dataset

# %%
print(dataset["train"][0]["text"])

# %% [markdown]
# ## Load and prepare model and tokenizer
# 
# We download the the official Llama 2 7B checkpoints from the HuggingFace Hub (**Note:** You must request access to this model on the HuggingFace website and use an API token to download it.).
# 
# Via the `BitsAndBytesConfig`, we specify that the model should be loaded in 4bit quantization and with double quantization for even better memory efficiency. See [their documentation](https://huggingface.co/docs/bitsandbytes/main/en/index) for more on this.

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig

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
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(modelpath)
tokenizer.pad_token = tokenizer.eos_token

# %% [markdown]
# We initialize the adapter functionality in the loaded model via `adapters.init()` and add a new LoRA adapter (named `"assistant_adapter"`) via `add_adapter()`.
# 
# In the call to `LoRAConfig()`, you can configure how and where LoRA layers are added to the model. Here, we want to add LoRA layers to all linear projections of the self-attention modules (`attn_matrices=["q", "k", "v"]`) as well as intermediate and outputa linear layers.

# %%
import adapters
from adapters import LoRAConfig, DoubleSeqBnConfig

adapters.init(model)

# config = LoRAConfig(
#     selfattn_lora=True, intermediate_lora=True, output_lora=True,
#     attn_matrices=["q", "k", "v"],
#     alpha=16, r=64, dropout=0.1
# )
config = DoubleSeqBnConfig()
model.add_adapter("assistant_adapter", config=config)
model.train_adapter("assistant_adapter")

print(model.adapter_summary())

# %% [markdown]
# Some final preparations for 4bit training: we cast a few parameters to float32 for stability.

# %%
for param in model.parameters():
    if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)

# Enable gradient checkpointing to reduce required memory if needed
# model.gradient_checkpointing_enable()
# model.enable_input_require_grads()

class CastOutputToFloat(torch.nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

# %%
model

# %%
# Verifying the datatypes.
dtypes = {}
for _, p in model.named_parameters():
    dtype = p.dtype
    if dtype not in dtypes:
        dtypes[dtype] = 0
    dtypes[dtype] += p.numel()
total = 0
for k, v in dtypes.items():
    total += v
for k, v in dtypes.items():
    print(k, v, v / total)


for _, v in model.get_adapter("assistant_adapter").items():
    for _, module in v.items():
        module.to("cuda")


# %% [markdown]
# ## Prepare data for training
# 
# The dataset is tokenized and truncated.

# %%
import os 

def tokenize(element):
    return tokenizer(
        element["text"],
        truncation=True,
        max_length=512, # can set to longer values such as 2048
        add_special_tokens=False,
    )

dataset_tokenized = dataset.map(
    tokenize, 
    batched=True, 
    num_proc=os.cpu_count(),    # multithreaded
    remove_columns=["text"]     # don't need this anymore, we have tokens from here on
)

# %%
dataset_tokenized

# %% [markdown]
# ## Training
# 
# We specify training hyperparameters and train the model using the `AdapterTrainer` class.
# 
# The hyperparameters here are similar to those chosen [in the official QLoRA repo](https://github.com/artidoro/qlora/blob/main/scripts/finetune_llama2_guanaco_7b.sh), but feel free to configure as you wish!

# %%
args = TrainingArguments(
    output_dir="output/llama_qlora_7b_seq_bn",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    evaluation_strategy="steps",
    logging_steps=10,
    save_steps=500,
    eval_steps=187,
    save_total_limit=3,
    max_steps=1875,
    lr_scheduler_type="constant",
    optim="paged_adamw_32bit",
    learning_rate=0.0002,
    group_by_length=True,
    bf16=True,
    warmup_ratio=0.03,
    max_grad_norm=0.3,
)

# %%
from adapters import AdapterTrainer
from transformers import DataCollatorForLanguageModeling

trainer = AdapterTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    train_dataset=dataset_tokenized["train"],
    eval_dataset=dataset_tokenized["test"],
    args=args,
)

trainer.train()

trainer.save_model()

# %% [markdown]
# ## Inference
# 
# Finally, we can prompt the model:

# %%
# Ignore warnings
from transformers import logging
logging.set_verbosity(logging.CRITICAL)

def prompt_model(model, text: str):
    batch = tokenizer(f"### Human: {text}\n### Assistant:", return_tensors="pt")
    batch = batch.to(model.device)
    
    model.eval()
    with torch.inference_mode():
        output_tokens = model.generate(**batch, max_new_tokens=50)

    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)


# %%
print(prompt_model(model, "Explain Calculus to a primary school student"))

# %% [markdown]
# ## Merge LoRA weights

# %%
# model.merge_adapter("assistant_adapter")

# %%
print(prompt_model(model, "Explain NLP in simple terms"))


