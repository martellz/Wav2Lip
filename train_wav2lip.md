# train wav2lip

## data process

### 剪切视频

将一个训练视频按时长大概地剪切为4-6等份，1份用作test，1份用作validate，其余用作训练。

按以下文件目录放置（可以在wav2lip项目文件夹内）：


data_root
    main
        00000
        00001
        00002
        ...

### 准备filelist

在wav2lip项目文件夹内，修改filelists文件夹内的train.txt, test.txt, val.txt。若没有则创建txt文件。

其中，每行指定一个剪切的视频，例如 train.txt

```text

data_root/00000
data_root/00001
... // 省略

```

### 运行处理脚本

```bash

# 进入 wav2lip 项目文件夹，记得改写 /PATH/TO/Wav2lip 为项目文件夹实际路径
cd /PATH/TO/Wav2lip

# 使用 conda 管理 python 虚拟环境
# api_server中应该包含wav2lip项目的依赖，但更改过版本，尚未测试过。
conda activate api_server

python process.py --data_root data/main --preprocessed_root xxx # [这里填输出文件夹，例如:data_preprocessed/]

```


## 训练

### 何时结束

专家判别器的 eval loss 应大约低于0.25 且 wav2lip 的 eval sync loss 应大约低于 0.2，以保证好的结果。

原文：The expert discriminator's eval loss should go down to ~0.25 and the Wav2Lip eval sync loss should go down to ~0.2 to get good results.

### 训练专家判别器

如果爆显存，将hparam.py的batch_size改为4或者更小。

```bash

# 已经处于 Wav2lip 项目文件夹内；已经激活 conda 虚拟环境

# 创建用于保存expert_checkpoint的文件夹
mkdir expert_checkpoints

# 训练专家判别器
python color_syncnet_train.py --data_root 这里填数据文件夹路径，例如data_preprocessed --checkpoint_dir expert_checkpoints

```

### 训练生成器

```bash

# 创建用于保存checkpoint的文件夹
mkdir checkpoints

python wav2lip_train.py --data_root 这里填数据文件夹路径，例如data_preprocessed --checkpoint_dir checkpoints --syncnet_checkpoint_path 这里填专家判别器checkpoint的路径，例如expert_checkpoints/checkpoint_step000xxxx.pth

```

### 其他参数

在 hparam.py 中修改