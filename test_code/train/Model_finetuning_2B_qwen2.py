import os
from PIL import Image
from datasets import load_dataset
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration, CLIPImageProcessor
from transformers import TrainingArguments, Trainer, AutoProcessor, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
import torch
import swanlab
from swanlab.integration.transformers import SwanLabCallback
import json
from datasets import Dataset
from qwen_vl_utils import process_vision_info

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# 定义数据预处理函数
def preprocess_function(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 8192
    input_ids, attention_mask, labels = [], [], []
    # conversation = example["conversations"]
    # input_content = conversation[0]["value"]
    output_content = example["text_descriptions"]
    # 获取图像路径
    file_path = '/home/wjx/PythonWorkSpace/Code/' + example["image_path"].split('\\')[0] + '/' + example["image_path"].split('\\')[1] + '/' + example["image_path"].split('\\')[2]
    #构建多模态输入：图像+文本提示组成成Qwen2-VL支持的对话格式
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                       "image": f"{file_path}",
                    "resized_height": 300,
                    "resized_width": 300,
                },
                {"type": "text", "text": "This is a time-frequency graph of a signal, please describe this signal for me."},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  # 获取文本
    image_inputs, video_inputs = process_vision_info(messages)  # 获取数据数据（预处理过）
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.tolist() for key, value in inputs.items()}  # tensor -> list,为了方便拼接
    instruction = inputs
    #构建多模态输出：文本描述
    response = tokenizer(f"{output_content}", add_special_tokens=False)
    #拼接多模态输入和输出
    input_ids = (
            instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    )

    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = (
            [-100] * len(instruction["input_ids"][0])
            + response["input_ids"]
            + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    #将多模态输入和输出转换为tensor,即返回Pytorch张量
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  # 由（1,h,w)变换为（h,w）
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw']}
    """
    以上，实现了1、多模态输入构造：将图像和文本提示组合成Qwen2-VL支持的格式
    2.文本处理：用processor生成token IDs与attention mask
    3.标签生成：目标文本的token IDs，前缀用-100填充
    4.图像处理：通过process_vision_info提取图像特征
    """

# 定义数据加载器的 collate 函数,作用是将多个样本打包成一个批次，统一转换为张量
def collate_fn(batch):
    new_batch = {
        "input_ids": [],
        "attention_mask": [],
        "pixel_values": []
        # "labels": []
    }
    for item in batch:
        new_batch["input_ids"].append(torch.tensor(item["input_ids"]))
        new_batch["attention_mask"].append(torch.tensor(item["attention_mask"]))
        new_batch["pixel_values"].append(torch.tensor(item["pixel_values"]))
#        new_batch["input_ids"].append(item["input_ids"])
#        new_batch["attention_mask"].append(item["attention_mask"])
#        new_batch["pixel_values"].append(item["pixel_values"])
#        new_batch["labels"].append(item["labels"])

    #堆叠为批次张量
    new_batch["input_ids"] = torch.stack(new_batch["input_ids"])
    new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])
    new_batch["pixel_values"] = torch.stack(new_batch["pixel_values"])
    # new_batch["labels"] = torch.tensor(new_batch["labels"])



    return new_batch


# 定义自定义 Trainer 类，调用模型前向传播，返回损失值
class QwenVLTrainer(Trainer):
    def compute_loss(self, model, inputs,return_outputs=False,num_items_in_batch=None):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"]
,
            image_grid_thw = inputs['image_grid_thw']
,
            labels=inputs["labels"]
        )
        if return_outputs:
            return outputs.loss, outputs
        return outputs.loss

    # def log(self, logs):
    #     super().log(logs)
    #     # 记录日志到 SwanLab
    #     for key, value in logs.items():
    #         if isinstance(value, (int, float)):
    #             exp.log({key: value})

# 设置SwanLab回调
swanlab_callback = SwanLabCallback(
    project="Qwen2-VL-finetune",
    experiment_name="qwen2-vl-my",
    config={
        "model": "https://modelscope.cn/models/Qwen/Qwen2-VL-2B-Instruct",
        # "dataset": "https://modelscope.cn/datasets/modelscope/coco_2014_caption/quickstart",
        # "github": "https://github.com/datawhalechina/self-llm",
        "prompt": "This is a time-frequency graph of a signal, please describe this signal for me.",
    },
)

# 使用Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained("/home/wjx/PythonWorkSpace/Code/ModelChange/Qwen2/Qwen2-VL-2B-Instruct/", use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained("/home/wjx/PythonWorkSpace/Code/ModelChange/Qwen2/Qwen2-VL-2B-Instruct/")

train_json_path = "/home/wjx/PythonWorkSpace/Code/ModelChange/train/DatasetResult_BPSK_all.json"
with open(train_json_path, 'r') as f:
    train_data = json.load(f)


test_json_path = "/home/wjx/PythonWorkSpace/Code/ModelChange/train/DatasetResult_BPSK_4.json"
with open(test_json_path, 'r') as f:
    test_data = json.load(f)

train_ds = Dataset.from_json("/home/wjx/PythonWorkSpace/Code/ModelChange/train/DatasetResult_BPSK_all.json")
train_dataset = train_ds.map(preprocess_function)

test_ds = Dataset.from_json("/home/wjx/PythonWorkSpace/Code/ModelChange/train/DatasetResult_BPSK_4.json")
test_dataset = test_ds.map(preprocess_function)


# 加载预训练模型
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/home/wjx/PythonWorkSpace/Code/ModelChange/Qwen2/Qwen2-VL-2B-Instruct/",
    device_map="auto",
    torch_dtype=torch.float32,
    trust_remote_code=True
#    load_in_8bit=True  # 显存不足时使用 8 位量化
)

model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 训练参数配置
training_args = TrainingArguments(
    output_dir="/home/wjx/PythonWorkSpace/Code/ModelChange/Qwen2/Qwen2-VL-2B-Instruct-trained/",
    num_train_epochs=15,
    per_device_train_batch_size=2,  # 根据显存调整
    gradient_accumulation_steps=4,  # 等效于更大的 batch_size
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50,
    save_steps=100,
#    evaluation_strategy="epoch",
    prediction_loss_only=True,
    fp16=False,  # 启用混合精度训练
    gradient_checkpointing=True,  # 节省显存
    report_to="tensorboard"
)

# 创建 Trainer 实例
trainer = QwenVLTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback]

)

# 开始训练
trainer.train()

# 保存微调后的模型
trainer.save_model()
processor.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)

# 保存训练配置参数
config_path = os.path.join(training_args.output_dir, 'training_config.json')
with open(config_path, 'w') as f:
    json.dump(training_args.to_dict(), f, indent=4)

# 在Jupyter Notebook中运行时要停止SwanLab记录，需要调用swanlab.finish()
swanlab.finish()

