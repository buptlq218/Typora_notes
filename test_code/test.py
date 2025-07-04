#import torch
#
## 检查GPU是否可用
#if torch.cuda.is_available():
#    print("GPU可用！")
#else:
#    print("GPU不可用，将使用CPU进行计算。")

#import swanlab
#swanlab.login(api_key="In6hrXe3ORQ2iZ0IZBK7v", save=True)

#from transformers import Qwen2_5_VLForConditionalGeneration
#
## 设定本地模型路径
#local_model_path = "/home/wjx/PythonWorkSpace/Code/ModelChange/Qwen7/Qwen/Qwen2___5-VL-7B-Instruct/"
#
#try:
#    # 加载本地模型
#    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(local_model_path)
#
#    # 输出模型的层级名称
#    for name in model.named_parameters():
#        print(name)
#except Exception as e:
#    print(f"加载模型时出错: {e}")

from transformers import Qwen2_5_VLForConditionalGeneration

model = Qwen2_5_VLForConditionalGeneration.from_pretrained("/home/wjx/PythonWorkSpace/Code/ModelChange/Qwen7/Qwen/Qwen2___5-VL-7B-Instruct/", trust_remote_code=True)

for name, param in model.named_parameters():
    print(name)
    
