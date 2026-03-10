import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import warnings
warnings.filterwarnings("ignore")

# 璁剧疆璁惧
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"浣跨敤璁惧: {device}")

# 鏁版嵁棰勫鐞嗗嚱鏁?
def load_and_preprocess_data(csv_path):
    """鍔犺浇鍜岄澶勭悊鏁版嵁"""
    print("姝ｅ湪鍔犺浇鏁版嵁...")
    df = pd.read_csv(csv_path)[:1000]
    
    # 杩囨护鏈夋晥鏁版嵁
    df = df[df['Lsa_summary'].notna() & df['sentiment_deepseek'].notna()]
    df = df[df['sentiment_deepseek'] != 0]  # 绉婚櫎鏃犳晥鐨勬儏鎰熸爣绛?
    
    print(f"鏈夋晥鏁版嵁鏁伴噺: {len(df)}")
    print(f"鎯呮劅鍒嗗竷: {df['sentiment_deepseek'].value_counts().sort_index()}")
    
    return df

def create_prompt_template(text, sentiment, stock_symbol="STOCK"):
    """鍒涘缓璁粌鎻愮ず妯℃澘"""
    # 浣跨敤涓巗entiment_deepseek_deepinfra.py鐩稿悓鐨勫璇濇牸寮?
    system_prompt = "Forget all your previous instructions. You are a financial expert with stock recommendation experience. Based on a specific stock, score for range from 1 to 5, where 1 is negative, 2 is somewhat negative, 3 is neutral, 4 is somewhat positive, 5 is positive. 1 summarized news will be passed in each time, you will give score in format as shown below in the response from assistant."
    
    # 鏋勫缓鐢ㄦ埛杈撳叆
    user_content = f"News to Stock Symbol -- {stock_symbol}: {text}"
    
    # 鏋勫缓瀹屾暣鐨勫璇?
    conversation = f"""System: {system_prompt}

User: News to Stock Symbol -- AAPL: Apple (AAPL) increase 22%
Assistant: 5

User: News to Stock Symbol -- AAPL: Apple (AAPL) price decreased 30%
Assistant: 1

User: News to Stock Symbol -- AAPL: Apple (AAPL) announced iPhone 15
Assistant: 4

User: {user_content}
Assistant: {sentiment}"""
    
    return conversation

def prepare_dataset(df, tokenizer, max_length=512):
    """鍑嗗璁粌鏁版嵁闆?""
    print("姝ｅ湪鍑嗗鏁版嵁闆?..")
    
    texts = []
    labels = []
    
    for _, row in df.iterrows():
        text = row['Lsa_summary']
        sentiment = int(row['sentiment_deepseek'])
        stock_symbol = row.get('Stock_symbol', 'STOCK')  # 鑾峰彇鑲＄エ绗﹀彿锛屽鏋滄病鏈夊垯浣跨敤榛樿鍊?
        
        if pd.isna(text) or text == '':
            continue
            
        prompt = create_prompt_template(text, sentiment, stock_symbol)
        texts.append(prompt)
        labels.append(sentiment)
    
    # 鍒嗗壊璁粌闆嗗拰楠岃瘉闆?(80% 璁粌, 20% 楠岃瘉)
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"璁粌闆嗗ぇ灏? {len(train_texts)}")
    print(f"楠岃瘉闆嗗ぇ灏? {len(eval_texts)}")
    
    # 鍒涘缓璁粌鏁版嵁闆?
    train_dataset = Dataset.from_dict({
        'text': train_texts,
        'label': train_labels
    })
    
    # 鍒涘缓楠岃瘉鏁版嵁闆?
    eval_dataset = Dataset.from_dict({
        'text': eval_texts,
        'label': eval_labels
    })
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        # 瀵逛簬璇█妯″瀷锛宭abels灏辨槸input_ids
        tokenized['labels'] = tokenized['input_ids'].clone()
        return tokenized
    
    # 瀵硅缁冮泦鍜岄獙璇侀泦杩涜tokenization
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    eval_tokenized = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    return train_tokenized, eval_tokenized

def create_model_and_tokenizer():
    """鍒涘缓妯″瀷鍜屽垎璇嶅櫒"""
    print("姝ｅ湪鍔犺浇妯″瀷鍜屽垎璇嶅櫒...")
    
    model_name = "Qwen3-8B"
    
    # 鍔犺浇鍒嗚瘝鍣?
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 鍔犺浇妯″瀷
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 鍑嗗妯″瀷杩涜璁粌
    model = prepare_model_for_kbit_training(model)
    
    # 閰嶇疆LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    
    # 搴旂敤LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_model(model, tokenizer, train_dataset, eval_dataset, output_dir="./qwen_sentiment_model"):
    """璁粌妯″瀷"""
    print("寮€濮嬭缁冩ā鍨?..")
    
    # 璁粌鍙傛暟
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # report_to=None,  # 绂佺敤wandb绛夋姤鍛婂伐鍏?
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # 鏁版嵁鏁寸悊鍣?
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 鍒涘缓璁粌鍣?
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 寮€濮嬭缁?
    trainer.train()
    
    # 淇濆瓨妯″瀷
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"妯″瀷宸蹭繚瀛樺埌: {output_dir}")

def main():
    """涓诲嚱鏁?""
    # 鏁版嵁璺緞
    csv_path = "nasdaq_news_sentiment/sentiment_deepseek_new_cleaned_nasdaq_news_full.csv"
    
    # 鍔犺浇鍜岄澶勭悊鏁版嵁
    df = load_and_preprocess_data(csv_path)
    
    # 鍒涘缓妯″瀷鍜屽垎璇嶅櫒
    model, tokenizer = create_model_and_tokenizer()
    
    # 鍑嗗鏁版嵁闆?
    train_dataset, eval_dataset = prepare_dataset(df, tokenizer)
    
    # 璁粌妯″瀷
    train_model(model, tokenizer, train_dataset, eval_dataset)
    
    print("璁粌瀹屾垚锛?)

if __name__ == "__main__":
    main() 
