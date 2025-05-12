import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# 设置设备（GPU 或 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的 GPT-2 模型和分词器
model_name = "gpt2"  # 可以选择 "gpt2-medium" 或其他变体
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# 加载和预处理数据集
def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )

train_dataset = load_dataset("yukino_dialogues.txt", tokenizer)

# 创建数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2 是自回归模型，不使用掩码语言建模
)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./yukino_gpt2",         # 模型输出目录
    overwrite_output_dir=True,          # 覆盖已有输出目录
    num_train_epochs=3,                 # 训练轮数，可根据数据量调整
    per_device_train_batch_size=4,      # 每个设备的批次大小
    save_steps=500,                     # 每隔多少步保存一次模型
    save_total_limit=2,                 # 最多保存的检查点数量
    learning_rate=5e-5,                 # 学习率
    logging_dir='./logs',               # 日志目录
    logging_steps=100,                  # 每隔多少步记录一次日志
)

# 创建 Trainer 并进行微调
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# 开始训练
trainer.train()

# 保存微调后的模型和分词器
model.save_pretrained("./yukino_gpt2_model")
tokenizer.save_pretrained("./yukino_gpt2_model")

# 使用微调后的模型生成文本
model = GPT2LMHeadModel.from_pretrained("./yukino_gpt2_model").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("./yukino_gpt2_model")

# 设置模型为评估模式
model.eval()

# 输入文本并生成回复
input_text = "你好，雪乃"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

# 生成文本
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=100,                # 生成文本的最大长度
        num_return_sequences=1,        # 生成的序列数量
        no_repeat_ngram_size=2,        # 避免重复的 n-gram
        top_k=50,                      # Top-k 采样
        top_p=0.95,                    # Top-p 采样
        temperature=0.7,               # 控制生成文本的随机性
    )

# 解码并输出生成的文本
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("生成的文本：", generated_text)