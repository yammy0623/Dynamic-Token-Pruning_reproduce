from mmseg.models import build_segmentor
from mmengine.runner import Runner
from mmengine.config import Config
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import torch
from mmseg.datasets import build_dataset
from mmseg_custom import *


# config_file = './exp/BASE_segvit_ade20k/BASE_segvit_ade20k.py'  
# config_file = './exp/PRUNE_segvit_ade20k/prune_segvit_ade20k.py'
# config_file = './config/prune/BASE_segvit_pc.py'
# config_file = './config/prune/prune_segvit_pc.py'
# config_file = './config/prune/BASE_segvit_cocostuff10k.py'
# config_file = './config/prune/prune_segvit_cocostuff10k.py'

# cfg = Config.fromfile(config_file)
# model = build_segmentor(cfg.model)
# model.eval()  

# input_tensor = torch.randn(1, 3, 512, 512)


# flop_analyzer = FlopCountAnalysis(model, input_tensor)
# flops = flop_analyzer.total()  
# params = sum(p.numel() for p in model.parameters())  # 參數數量


# print(f"Model GFLOPs: {flops / 1e9:.2f} G")
# print(f"Model Params: {params / 1e6:.2f} M")

###################################################

config_file = './exp/PRUNE_segvit_ade20k/prune_segvit_ade20k.py'
cfg = Config.fromfile(config_file)
ckpt_file = './exp/BASE_segvit_ade20k/iter_40000.pth'
cfg.load_from = ckpt_file
model = build_segmentor(cfg.model)
model.eval()  

# 這邊直接改成load dataset
cfg.data.test.pipeline[0]['type'] = 'LoadImageFromFile'  # 確保 pipeline 正確
dataset = build_dataset(cfg.data.test)  # 使用 MMSeg 內建函數載入 dataset

# 取得一筆測試資料（dataset[0] 會回傳 dict）
sample = dataset[0]  
img_tensor = sample['img'][None, ...]  # 增加 batch 維度 (1, C, H, W)

flop_analyzer = FlopCountAnalysis(model, img_tensor)

flops = flop_analyzer.total()  
params = sum(p.numel() for p in model.parameters())  # 參數數量


print(f"Model GFLOPs: {flops / 1e9:.2f} G")
print(f"Model Params: {params / 1e6:.2f} M")
