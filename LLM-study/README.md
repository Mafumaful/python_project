# 这个部分主要是学习B站大佬chaofa

主要参考
1. [LLMs-Zero-to-Hero，完全从零手写大模型，从数据处理到模型训练，细节拉满，一小时学会。 build a nanoGPT from scratch](https://www.bilibili.com/video/BV1qWwke5E3K/?share_source=copy_web&vd_source=d1a57027b2655bfb6dc177f8a435b353)
2. [手写self-attention的四重境界-part1 pure self-attention](https://www.bilibili.com/video/BV19YbFeHETz/?share_source=copy_web&vd_source=d1a57027b2655bfb6dc177f8a435b353)

## tutorial_part

主要目的是为了快速测试代码。测试并知道相关模块怎么运行。

## transformer_deploy

目的是为了部署测试模型能不能正常跑通，做一个最小化模型。

## ds_solution

为了查看如何在实际应用中解决问题，创建了这个文件夹，主要是为了观察训练问题。

TODO：里面还是有一点小bug需要解决，比如说这个是我的模型的预测结果：

```text
Sample 1:
Input: [8 1 3 8 2 8]
True:  [8 1 3 8 2 8]
Pred:  [8 1 3 8 2 8]

Sample 2:
Input: [ 5  1  8 11]
True:  [ 5  1  8 11]
Pred:  [5 1 8]

Sample 3:
Input: [ 1  9  1  6  9 11]
True:  [ 1  9  1  6  9 11]
Pred:  [1 9 1 6 9]

Sample 4:
Input: [ 1  9  9 11]
True:  [ 1  9  9 11]
Pred:  [1 9 9]

Sample 5:
Input: [ 2  7  6  8 11]
True:  [ 2  7  6  8 11]
Pred:  [2 7 6 8]

Sample 6:
Input: [ 2  5  1  9 11]
True:  [ 2  5  1  9 11]
Pred:  [2 5 1 9]

Sample 7:
Input: [ 9  8 11]
True:  [ 9  8 11]
Pred:  [9 8]

Sample 8:
Input: [3 3 3 5 6 5]
True:  [3 3 3 5 6 5]
Pred:  [3 3 3 5 6 5]

Sample 9:
Input: [9 9 4 6 9 4]
True:  [9 9 4 6 9 4]
Pred:  [9 9 4 6 9 4]

Sample 10:
Input: [ 9  9  5  6 11]
True:  [ 9  9  5  6 11]
Pred:  [9 9 5 6]
```

可以看到，在这个模型头一个数字“顶格”在第一个位置的时候，模型的预测结果是正确的，但是当第一个数字不是“顶格”在第一个位置的时候，模型的预测结果就会出现问题。这个问题需要解决。
