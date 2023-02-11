# 我想说的话 🚟

其实本项目很多工程都是我大学里面做的，但是由于当时不太擅长使用 GitHub，所以很多东西都没有做过系统的整理。现在乘着寒假的机会，把我做过的项目使用 jupyterntebook 整理下来。🟣

# 本文的结构 🪐

## OpenCV

- 001 读取并显示图片
- 002 对图片进行低通滤波、高通滤波
- 003 直方图处理，对黑白图像进行直方图均衡化处理
- 004 对图片进行 HSV 颜色阈值处理，并通过最小外接矩使车牌摆正
- 005 首先把图像处理成灰度图，通过不同算法对图像的边缘进行检测
- 006 采用不同的算法对图像进行旋转
- 007 量化以及分辨率处理
- 008 提取对象的 HSV 值并打印出来，与工具部分的 hsv_extractor 相同，这里重新写一遍

### 工具部分 🪒

- hsv_extractor 对指定图像的 HSV 进行处理，运行以后白色的部分是保留的，便于后期 inrange 操作（详见 004）
  - 在运行 hsv_extractor 以后，会在终端打印对应的 HSV 阈值。

## Scipy

- 001 如何调用常数
- 002 一些常见的函数以及使用方法
- 🔼003 使用 numpy 解决线性问题

## mathematics

- 001 使用 Fast Fourier Transform 对音频进行处理 (施工中)

## matplotlib

- 001 了解 matplotlib 几个基本的绘图

# 如何安装环境 🌍

```bash
cd [the path you want]
conda env update -f requirements.yml
```
