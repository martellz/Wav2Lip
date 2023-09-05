import sys
import torch
import torch.nn as nn
from torch.utils import data as data_utils
from models import SyncNet_color as SyncNet
from color_syncnet_train import Dataset
from hparams import hparams

sync_ckpt = sys.argv[1]
data_root = sys.argv[2]

# 加载模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SyncNet()
model.load_state_dict(torch.load(sync_ckpt, map_location="cpu")['state_dict'])
model.to(device)
model.eval()

test_dataset = Dataset(data_root, 'test')

# 加载输入,选定一个视频，重复多次
# vidname = test_dataset.all_videos[0] # Dataset.__getitem__()

test_loader = data_utils.DataLoader(
    test_dataset, batch_size=1,
    num_workers=0)

acc = 0
for _, (x, mel, y) in enumerate(test_loader):
    x = x.to(device)
    mel = mel.to(device)
    y = y.to(device) # gt, 0 or 1
    a, v = model(mel, x) # a和v的特征
    av_sim = nn.functional.cosine_similarity(a, v) # a与v越相似，得分越高

    for i in range(a.size(0)):
        pred_label = 1 if av_sim[i] > 0.5 else 0
        if pred_label == y:
            acc += 1

avg_acc = acc / len(test_dataset)
print("avg_acc: ", avg_acc)