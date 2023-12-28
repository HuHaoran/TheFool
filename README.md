# TheFool
简单，易读，可迁移的强化学习框架

笨笨的智能体，却有着无限的可能性！

# Windows 安装方案
## 使用conda虚拟环境，python版本3.8
conda create -n your_env_name python=3.8

## 依赖包安装
### 激活虚拟环境后，顺序执行
1. conda install -c conda-forge cudatoolkit==11.2.2
2. conda install -c conda-forge cudnn==8.1.0.77
3. conda install -c nvidia cuda-nvcc
4. pip install gymnasium
5. pip install gymnasium[atari]
6. pip install tensorflow==2.10
7. pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
8. pip install opencv-python
9. pip install tqdm

# 快速上手
直接运行 `python train_atari.py` 即可体验atari游戏训练过程
## 现在支持的算法
1. DQN——及其各种改进版本
2. PPO
3. SAC 离散动作版
