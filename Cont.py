import os
import gc
import re
import ast
import json
import time
import math
import random
import shutil
from pathlib import Path
from collections import Counter
from fractions import Fraction

import numpy as np
import pandas as pd
import torch

from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# =========================
# 01. Config
# =========================
CFG = {
    "seed": 42,
    "student_model": "google/gemma-3-1b-it",
    "teacher_verified_config": "verified_Qwen2.5-7B-Instruct",

    "batch_size": 1,
    "grad_accum": 4,
    "max_length": 160,

    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,

    "stage_a_lr": 5e-5,
    "stage_a_steps": 7610,   # ~1 epoch on full verified set

    "save_steps": 50,
    "infer_batch_size": 16,
    "gen_new_tokens": 48,
}

HF_TOKEN = os.getenv("HF_TOKEN", "")

PROJECT_ROOT = Path("/root/contour")
INPUT_DIR = PROJECT_ROOT / "input"
WORK_DIR = PROJECT_ROOT / "workdir"
DATA_DIR = WORK_DIR / "data"
MODEL_DIR = WORK_DIR / "models"
LOG_DIR = WORK_DIR / "logs"
SUB_DIR = WORK_DIR / "submissions"
STAGE_A_DIR = MODEL_DIR / "stage_a"

for p in [PROJECT_ROOT, INPUT_DIR, WORK_DIR, DATA_DIR, MODEL_DIR, LOG_DIR, SUB_DIR]:
    p.mkdir(parents=True, exist_ok=True)


# =========================
# 02. Seed
# =========================
random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
torch.cuda.manual_seed_all(CFG["seed"])


# =========================
# 03. Utils
# =========================
BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}


def to_key(nums, target):
    return f"{int(target)}|" + "_".join(map(str, nums))


def parse_nums_cell(x):
    if isinstance(x, list):
        return x
    return ast.literal_eval(x)


def build_prompt(nums, target):
    return (
        "Solve the Countdown task.\n"
        "Use only the given numbers, each at most once.\n"
        "Use only +, -, *, / and parentheses if needed.\n"
        "Return only the final expression, without explanation and without '= target'.\n\n"
        f"Numbers: {list(nums)}\n"
        f"Target: {int(target)}"
    )


def gemma_train_text(nums, target, equation):
    prompt = build_prompt(nums, target)
    return (
        "<start_of_turn>user\n"
        f"{prompt}<end_of_turn>\n"
        "<start_of_turn>model\n"
        f"{equation}<end_of_turn>\n"
    )


def gemma_infer_text(nums, target):
    prompt = build_prompt(nums, target)
    return (
        "<start_of_turn>user\n"
        f"{prompt}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def canonical_equation(text):
    text = str(text).replace("×", "*").replace("÷", "/").strip()
    text = text.split("=")[0].strip()
    text = text.split("\n")[0].strip()
    text = text.replace("<end_of_turn>", "").strip()
    text = re.sub(r"[^0-9\(\)\+\-\*/\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_answer_equation(text):
    text = str(text)
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1)
    return canonical_equation(text)


def eval_expr_fraction(expr):
    node = ast.parse(expr, mode="eval")
    used = []

    def walk(n):
        if isinstance(n, ast.Expression):
            return walk(n.body)

        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            used.append(int(n.value))
            return Fraction(int(n.value), 1)

        if isinstance(n, ast.BinOp) and type(n.op) in BIN_OPS:
            left = walk(n.left)
            right = walk(n.right)
            if isinstance(n.op, ast.Div) and right == 0:
                raise ZeroDivisionError
            return BIN_OPS[type(n.op)](left, right)

        raise ValueError("bad expr")

    value = walk(node)
    return value, used


def verify_equation(nums, target, expr):
    try:
        expr = canonical_equation(expr)
        if not expr:
            return False

        value, used = eval_expr_fraction(expr)

        allowed = Counter(nums)
        actual = Counter(used)

        for k, v in actual.items():
            if allowed[k] < v:
                return False

        return value == Fraction(int(target), 1)
    except Exception:
        return False


def clear_memory(*names):
    for name in names:
        if name in globals():
            del globals()[name]
    gc.collect()
    torch.cuda.empty_cache()


# =========================
# 04. Load datasets
# =========================
def load_data():
    verified_ds = load_dataset(
        "HuggingFaceTB/Countdown-Task-GOLD",
        CFG["teacher_verified_config"],
        split="train",
    )
    holdout_ds = load_dataset(
        "HuggingFaceTB/Countdown-Task-GOLD",
        "test",
        split="test",
    )

    holdout_df = pd.DataFrame({
        "nums": list(holdout_ds["nums"]),
        "target": list(holdout_ds["target"]),
    })
    holdout_df["key"] = holdout_df.apply(lambda r: to_key(r["nums"], r["target"]), axis=1)

    perm = np.random.RandomState(CFG["seed"]).permutation(len(holdout_df))
    dev_df = holdout_df.iloc[perm[:8000]].reset_index(drop=True)
    shadow_df = holdout_df.iloc[perm[8000:10000]].reset_index(drop=True)

    dev_df.to_parquet(DATA_DIR / "dev.parquet", index=False)
    shadow_df.to_parquet(DATA_DIR / "shadow.parquet", index=False)

    print("verified_ds:", verified_ds)
    print("dev/shadow:", dev_df.shape, shadow_df.shape)
    return verified_ds, dev_df, shadow_df


# =========================
# 05. Build Stage A dataset
# =========================
def build_stage_a_dataset(verified_ds):
    rows = []

    for row in verified_ds:
        nums = row["nums"]
        target = int(row["target"])
        assistant = row["messages"][-1]["content"]
        eq = extract_answer_equation(assistant)

        if eq:
            rows.append({
                "nums": nums,
                "target": target,
                "equation": eq,
                "key": to_key(nums, target),
                "text": gemma_train_text(nums, target, eq),
            })

    stage_a_df = pd.DataFrame(rows).drop_duplicates(["key", "equation"]).reset_index(drop=True)
    stage_a_ds = Dataset.from_dict({"text": stage_a_df["text"].tolist()})

    stage_a_df.to_parquet(DATA_DIR / "stage_a_df.parquet", index=False)
    stage_a_ds.save_to_disk(str(DATA_DIR / "stage_a_ds"))

    print("stage_a_df:", stage_a_df.shape)
    print(stage_a_df.iloc[0]["text"][:300])
    return stage_a_df, stage_a_ds


# =========================
# 06. Model helpers
# =========================
student_tokenizer = AutoTokenizer.from_pretrained(
    CFG["student_model"],
    token=HF_TOKEN,
    trust_remote_code=True,
)
if student_tokenizer.pad_token is None:
    student_tokenizer.pad_token = student_tokenizer.eos_token
student_tokenizer.padding_side = "left"

student_bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

peft_config = LoraConfig(
    r=CFG["lora_r"],
    lora_alpha=CFG["lora_alpha"],
    lora_dropout=CFG["lora_dropout"],
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)


def load_student_base():
    model = AutoModelForCausalLM.from_pretrained(
        CFG["student_model"],
        token=HF_TOKEN,
        trust_remote_code=True,
        quantization_config=student_bnb,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    return model


def load_student_with_adapter(adapter_dir, trainable=True):
    model = AutoModelForCausalLM.from_pretrained(
        CFG["student_model"],
        token=HF_TOKEN,
        trust_remote_code=True,
        quantization_config=student_bnb,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=trainable)
    model.config.use_cache = False
    return model


def train_sft(model, train_ds, output_dir, lr, max_steps, new_adapter):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=CFG["batch_size"],
        gradient_accumulation_steps=CFG["grad_accum"],
        learning_rate=lr,
        max_steps=max_steps,
        logging_steps=10,
        save_strategy="steps",
        save_steps=CFG["save_steps"],
        save_total_limit=5,
        fp16=False,
        bf16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        max_length=CFG["max_length"],
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        processing_class=student_tokenizer,
        peft_config=peft_config if new_adapter else None,
    )

    trainer.train()
    return trainer


# =========================
# 07. Eval helpers
# =========================
@torch.inference_mode()
def predict_equations(model, df, batch_size, desc="predict"):
    preds = []
    total = len(df)
    total_batches = (total + batch_size - 1) // batch_size

    start_time = time.time()
    pbar = tqdm(
        range(0, total, batch_size),
        total=total_batches,
        desc=desc,
        leave=True,
        dynamic_ncols=True,
    )

    for step_idx, start in enumerate(pbar, start=1):
        batch = df.iloc[start:start + batch_size]

        prompts = [
            gemma_infer_text(nums, target)
            for nums, target in zip(batch["nums"], batch["target"])
        ]

        inputs = student_tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=CFG["gen_new_tokens"],
            do_sample=False,
            pad_token_id=student_tokenizer.pad_token_id,
            eos_token_id=student_tokenizer.eos_token_id,
        )

        gen_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        gen_texts = student_tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        preds.extend([canonical_equation(x) for x in gen_texts])

        done = min(step_idx * batch_size, total)
        elapsed = time.time() - start_time
        eta = (elapsed / step_idx) * (total_batches - step_idx)

        pbar.set_postfix({
            "done": f"{done}/{total}",
            "elapsed_min": f"{elapsed / 60:.1f}",
            "eta_min": f"{eta / 60:.1f}",
        })

    return preds


def evaluate_and_save(model, df, name, out_path):
    t0 = time.time()

    tmp = df.copy()
    tmp["pred"] = predict_equations(model, tmp, CFG["infer_batch_size"], desc=name)
    tmp["ok"] = [
        verify_equation(nums, target, pred)
        for nums, target, pred in zip(tmp["nums"], tmp["target"], tmp["pred"])
    ]

    tmp.to_parquet(out_path, index=False)

    elapsed = time.time() - t0
    metrics = {
        "name": name,
        "n": int(len(tmp)),
        "acc": float(tmp["ok"].mean()),
        "elapsed_sec": float(elapsed),
        "elapsed_min": float(elapsed / 60),
    }

    with open(LOG_DIR / f"{name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(
        f"{name}: done {len(tmp)}/{len(tmp)} | "
        f"acc={metrics['acc']:.4f} | "
        f"time={metrics['elapsed_min']:.1f} min"
    )
    return metrics


# =========================
# 08. Train + eval + submission
# =========================
def train_stage_a(stage_a_ds):
    if STAGE_A_DIR.exists():
        shutil.rmtree(STAGE_A_DIR)
        print("removed", STAGE_A_DIR)

    (STAGE_A_DIR / "adapter").mkdir(parents=True, exist_ok=True)

    model = load_student_base()

    trainer = train_sft(
        model=model,
        train_ds=stage_a_ds,
        output_dir=STAGE_A_DIR,
        lr=CFG["stage_a_lr"],
        max_steps=CFG["stage_a_steps"],
        new_adapter=True,
    )

    trainer.model.save_pretrained(str(STAGE_A_DIR / "adapter"))
    student_tokenizer.save_pretrained(str(STAGE_A_DIR / "adapter"))

    pd.DataFrame(trainer.state.log_history).to_json(
        STAGE_A_DIR / "log_history.json",
        orient="records",
        force_ascii=False,
    )

    print("saved:", [p.name for p in STAGE_A_DIR.glob("*")])
    clear_memory("trainer", "model")


def eval_stage_a(dev_df, shadow_df):
    stage_a_model = load_student_with_adapter(STAGE_A_DIR / "adapter", trainable=False)
    stage_a_model.eval()

    metrics_dev_a = evaluate_and_save(
        stage_a_model,
        dev_df,
        "stage_a_dev",
        DATA_DIR / "stage_a_dev_preds.parquet",
    )

    metrics_shadow_a = evaluate_and_save(
        stage_a_model,
        shadow_df,
        "stage_a_shadow",
        DATA_DIR / "stage_a_shadow_preds.parquet",
    )

    return stage_a_model, metrics_dev_a, metrics_shadow_a


def make_submission(model):
    public_path = INPUT_DIR / "test_public.csv"
    public_df = pd.read_csv(public_path)

    nums_col = "nums" if "nums" in public_df.columns else "numbers"
    target_col = "target"

    public_df[nums_col] = public_df[nums_col].apply(parse_nums_cell)
    public_df["nums"] = public_df[nums_col]
    public_df["target"] = public_df[target_col].astype(int)

    public_df["equation"] = predict_equations(
        model,
        public_df[["id", "nums", "target"]],
        CFG["infer_batch_size"],
        desc="public_test",
    )

    submission_df = public_df[["id", "equation"]].copy()
    submission_df.to_csv(SUB_DIR / "submission_stage_a.csv", index=False)

    print(submission_df.head())
    print("saved:", SUB_DIR / "submission_stage_a.csv")


# =========================
# 09. Main
# =========================
def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")

    verified_ds, dev_df, shadow_df = load_data()
    _, stage_a_ds = build_stage_a_dataset(verified_ds)

    train_stage_a(stage_a_ds)
    stage_a_model, metrics_dev_a, metrics_shadow_a = eval_stage_a(dev_df, shadow_df)

    print(metrics_dev_a)
    print(metrics_shadow_a)

    make_submission(stage_a_model)

    report = {
        "config": CFG,
        "stage_a_dev": metrics_dev_a,
        "stage_a_shadow": metrics_shadow_a,
        "stage_a_adapter": str(STAGE_A_DIR / "adapter"),
        "submission": str(SUB_DIR / "submission_stage_a.csv"),
    }

    with open(LOG_DIR / "stage_a_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()