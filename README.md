# TheFool
简单，易读，可迁移的强化学习框架

# Windows 安装方案
## 使用conda虚拟环境，python版本3.8
conda create -n your_env_name python=3.8

## 依赖包安装
### 激活虚拟环境后，顺序执行
1. conda install -c conda-forge cudatoolkit==11.2.2
2. conda install -c conda-forge cudnn==8.1.0.77
3. pip install gymnasium
4. pip install gymnasium[atari]
5. pip install tensorflow==2.10
6. pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
7. pip install opencv-python
8. pip install tqdm