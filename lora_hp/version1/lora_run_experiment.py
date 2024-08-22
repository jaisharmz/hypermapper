#### IMPORTS ####
import torch
import os
import sys
import json
import IPython
import pandas as pd
import numpy as np
from datetime import datetime
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    TrainingArguments,
)
import transformers
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from custom_collator import DataCollatorForCompletionOnlyLMcustom
from rouge_score import rouge_scorer
import evaluate
from huggingface_hub import login
import random
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback
login("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")


#### SEED ####
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# #### HYPERPARAMETER SEARCH ####
# learning_rates = [1e-6, 1e-5, 1e-4, 1e-3]
# ranks = [1, 16, 32, 64]


def get_rouge_score(learning_rate, rank):
    #### ROUGE LOGGING ####
    rouge_filename = "rouge_results.csv"
    rouge_dir = "./logs_rouge"
    overwrite = True
    os.makedirs(rouge_dir, exist_ok=True)
    if overwrite:
        print("LR,RANK,ROUGE-L-SCORE", file=open(os.path.join(rouge_dir, rouge_filename), "w"))
    df = pd.read_csv(os.path.join(rouge_dir, rouge_filename))


    #### LOOP ####
    set_seed(0)
    if df[(df["LR"] == learning_rate) & (df["RANK"] == rank)].shape[0] > 0:
        return df[(df["LR"] == learning_rate) & (df["RANK"] == rank)]

    ### PRINT SETTINGS ###
    print("SETTINGS:")
    print(f"LR: {learning_rate}")
    print(f"RANK: {rank}")

    ### MODEL SETUP ###
    # Base model, device, tokenizer; redefine pad_token and pad_token_id with unk_token
    model_name = "Hugofernandez/Mistral-7B-v0.1-colab-sharded"
    # model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    # model_name = "Qwen/Qwen2-7B-Instruct"
    device = 'cuda'
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    # tokenizer.add_bos_token = True
    # tokenizer.add_eos_token = True

    ### QLoRA ###
    # Quantization (TODO: change this to run on two GPUs simultaneously)
    compute_dtype = getattr(torch, "float16")
    print(compute_dtype)
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
    )

    ### LOAD MODEL ###
    # Load and quantize model
    model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config=bnb_config, 
            #   use_flash_attention_2 = False, #set to True you're using A100
            device_map={"": 0}, #device_map="auto" will cause a problem in the training 

    )

    ### LoRA CONFIG ###
    # LoRA settings
    peft_config = LoraConfig(
            lora_alpha=32,
            lora_dropout=0.05,
            r=rank,
            bias="none",
            task_type="CAUSAL_LM",
            # target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj", "lm_head",]
            target_modules=["q_proj", "k_proj", "v_proj"],
    )

    ### ANOTHER QLoRA ###
    #Cast some modules of the model to fp32 
    model = prepare_model_for_kbit_training(model)
    #Configure the pad token in the model
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False # Gradient checkpointing is used by default but not compatible with caching

    ### TRAINING AERGUMENTS ###
    training_arguments = TrainingArguments(
            output_dir="./results", # directory in which the checkpoint will be saved. 
            evaluation_strategy="steps", # epoch # you can set it to 'steps' to eval it every eval_steps 
            optim="paged_adamw_8bit", #used with QLoRA
            per_device_train_batch_size=8, #batch size 
            per_device_eval_batch_size=8, #same but for evaluation 
            gradient_accumulation_steps=1, #number of lines to accumulate gradient, carefull because it changes the size of a "step".Therefore, logging, evaluation, save will be conducted every gradient_accumulation_steps * xxx_step training example
            log_level="debug", #you can set it to  ‘info’, ‘warning’, ‘error’ and ‘critical’ 
            save_steps=20, # 500 #number of steps between checkpoints 
            logging_steps=20, # 20 #number of steps between logging of the loss for monitoring adapt it to your dataset size
            learning_rate=learning_rate,#2e-4,#7e-6,#4e-4, #you can try different value for this hyperparameter
            num_train_epochs=20,
            warmup_steps=100,
            lr_scheduler_type="constant",

            report_to="tensorboard",
            load_best_model_at_end=True,
            logging_dir=f"/home/vision/jais_work/ibm/logs/LR {learning_rate} RANK {rank}",

            save_total_limit=1,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
    )

    ### GET DATA ###
    # Task 039
    path = "/home/vision/jais_work/ibm/LoLA/src/natural-instructions-2.8/tasks/task039_qasc_find_overlapping_words.json"
    dataset = load_dataset("json", data_files=path, field="Instances")
    task_definition = json.load(open(path))["Definition"][0]
    # 70/20/10 split train/dev/test
    train_test_split = dataset['train'].shuffle(seed=0).select(range(500)).train_test_split(train_size=0.7, seed=0)
    train_set = train_test_split['train']
    val_test_split = train_test_split['test'].train_test_split(test_size=1/3, seed=0)
    val_set = val_test_split['train']
    test_set = val_test_split['test']
    dataset = DatasetDict({
        'train': train_set,
        'val': val_set, 
        'test': test_set,
    })

    ### FORMATTING EXAMPLES FUNCTION ###
    def formatting_func(example):
        text_sequences = []
        for i in range(len(example["input"])):
            output = random.choice(example["output"][i])
            # text = f"{example['input'][i]}{output}{tokenizer.eos_token}"
            text = f"{example['input'][i]}\nOutput: \n\n{output}{tokenizer.eos_token}"
            text_sequences.append(text)
        return text_sequences

    def formatting_func2(example):
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for i in range(len(example["input"])):
            input_text = f"{example['input'][i]}\nOutput: \n\n"
            tokenized_input = tokenizer(input_text, max_length=128, padding="max_length", truncation=True)
            output_text = random.choice(example["output"][i]) + tokenizer.eos_token
            with tokenizer.as_target_tokenizer():
                tokenized_output = tokenizer(output_text, max_length=128, padding="max_length", truncation=True)
            result["input_ids"].append(tokenized_input["input_ids"])
            result["attention_mask"].append(tokenized_input["attention_mask"])
            result["labels"].append(tokenized_output["input_ids"])
        return result

    ### LoRA TRAINER ###
    early_stopping_callback = transformers.EarlyStoppingCallback(early_stopping_patience=2)
    response_template_ids = tokenizer.encode("\nOutput: \n\n", add_special_tokens=False)[1:]
    collator = DataCollatorForCompletionOnlyLMcustom(response_template_ids, tokenizer=tokenizer)
    trainer = SFTTrainer(
            model=model,
            train_dataset=dataset['train'],
            eval_dataset=dataset['val'],
            peft_config=peft_config,
            tokenizer=tokenizer,
            args=training_arguments,
            formatting_func=formatting_func,
            data_collator=collator,
            callbacks=[early_stopping_callback],
    )

    ### NUMBER OF TRAINABLE PARAMETERS ###
    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = model.num_parameters()
        for _, param in model.named_parameters():
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )
    print_trainable_parameters(model)

    ### TRAIN MODEL ###
    # Launch the training
    trainer.train()

    ### TESTING MODEL FOR METRICS (ROUGE-L) ###
    predictions = []
    references = []
    for example_num in range(dataset["test"].num_rows):
        # Select example
        example = dataset["test"][example_num:example_num+1]

        prompt_partial = example["input"][0]
        eval_prompt = f"{prompt_partial}\nOutput: \n\n"
        
        model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

        model.eval()
        with torch.no_grad():
            prediction = tokenizer.decode(model.generate(**model_input, max_new_tokens=30, pad_token_id=2)[0], skip_special_tokens=True)
        model.train()

        prediction_filtered = prediction[prediction.index("\nOutput: \n\n") + len("\nOutput: \n\n"):]
        output_true = prediction_filtered if prediction_filtered in example["output"][0] else example["output"][0][0]
        references.append(output_true)
        predictions.append(prediction_filtered)

        print("Ground Truth:")
        print(output_true)
        print("Prediction:")
        print(prediction_filtered)

        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(output_true, prediction_filtered)
        print(scores)

    ### GET METRIC ###
    rouge = evaluate.load('CZLC/rouge_raw')
    results = rouge.compute(predictions=predictions, references=references)
    # print(results)
    # print(f"ROUGE-L SCORE: {results['L_mid_fmeasure']}")
    print(f"{learning_rate},{rank},{results['L_mid_fmeasure']}", file=open(os.path.join(rouge_dir, rouge_filename), "a"))
    return results['L_mid_fmeasure']