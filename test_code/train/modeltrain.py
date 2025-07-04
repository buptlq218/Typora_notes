import os
from PIL import Image
from datasets import load_dataset
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration, CLIPImageProcessor
from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader
import torch
import swanlab
from swanlab.integration.transformers import SwanLabCallback



# 定义数据预处理函数
def preprocess_function(examples):
    tokenizer = AutoTokenizer.from_pretrained("/home/wjx/PythonWorkSpace/Code/ModelChange/Qwen/Qwen2-VL-2B-Instruct/")
    image_processor = CLIPImageProcessor.from_pretrained("/home/wjx/PythonWorkSpace/Code/ModelChange/Qwen/clip-vit-base-patch32/")

    # # 文本处理
    # texts = [text.strip() for text in examples["text"]]
    # text_inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=128)
    #
    # # 图像处理
    # images = [Image.open(os.path.join(image_dir, path)).convert("RGB") for path in examples["image"]]
    # image_inputs = image_processor(images, return_tensors="pt", padding=True)

    texts = [text.strip() for text in examples["text_descriptions"]]
    text_inputs = tokenizer(texts, padding="max_length", truncation=True, max_length=128)

    # 图像处理
    images = [Image.open('/home/wjx/PythonWorkSpace/Code/' + path.split('\\')[0] + '/' + path.split('\\')[1] + '/' + path.split('\\')[2]).convert("RGB") for path in examples["image_path"]]
    image_inputs = image_processor(images, return_tensors="pt", padding=True)

    image_grid_thw = []
    for image in images:
        # 这里需要根据实际情况计算图像的网格信息
        # 例如，简单假设网格信息为图像的高度和宽度
        h, w = image.height, image.width
        image_grid_thw.append((1, h, w))  # 假设 t=1

    # 合并结果
    inputs = {
        "input_ids": text_inputs["input_ids"],
        "attention_mask": text_inputs["attention_mask"],
        "pixel_values": image_inputs["pixel_values"],
        "image_grid_thw": image_grid_thw  # 添加 image_grid_thw
        # "labels": examples.get("label", None)
    }
    return inputs


# 定义数据加载器的 collate 函数
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


    new_batch["input_ids"] = torch.stack(new_batch["input_ids"])
    new_batch["attention_mask"] = torch.stack(new_batch["attention_mask"])
    new_batch["pixel_values"] = torch.stack(new_batch["pixel_values"])
    # new_batch["labels"] = torch.tensor(new_batch["labels"])



    return new_batch


# 定义自定义 Trainer 类
class QwenVLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw = inputs['image_grid_thw']
            # labels=inputs["labels"]
        )
        return outputs.loss if not return_outputs else outputs

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

# 加载数据集
data_files = {
    "train": "/home/wjx/PythonWorkSpace/Code/ModelChange/train/DatasetResult_BPSK_all.json",
    "validation": "/home/wjx/PythonWorkSpace/Code/ModelChange/train/DatasetResult_BPSK_4.json"
}
# image_dir = "path/to/images"  # 图像所在的目录
dataset = load_dataset("json", data_files=data_files)

# 预处理数据集
processed_dataset = dataset.map(preprocess_function, batched=True)

# 加载预训练模型
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/home/wjx/PythonWorkSpace/Code/ModelChange/Qwen/Qwen2-VL-2B-Instruct/",
    device_map="auto",
#    load_in_8bit=True  # 显存不足时使用 8 位量化
)

# 训练参数配置
training_args = TrainingArguments(
    output_dir="/home/wjx/PythonWorkSpace/Code/ModelChange/Qwen/Qwen2-VL-2B-Instruct-trained/",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # 根据显存调整
    gradient_accumulation_steps=4,  # 等效于更大的 batch_size
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50,
    save_steps=1000,
    evaluation_strategy="epoch",
    prediction_loss_only=True,
    fp16=True,  # 启用混合精度训练
    gradient_checkpointing=True,  # 节省显存
    report_to="tensorboard"
)

# 创建 Trainer 实例
trainer = QwenVLTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    data_collator=collate_fn
)

# 开始训练
trainer.train()

# 保存微调后的模型
trainer.save_model()

# 在Jupyter Notebook中运行时要停止SwanLab记录，需要调用swanlab.finish()
swanlab.finish()

