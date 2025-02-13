# 本proj的意义

尝试手搓transformer模型。

2. Decoder 输入的特殊处理

Decoder的输入是目标语言句子，但在训练时，会对其进行右移(Shifted Right)操作：

输入到Decoder的是目标句子的前n-1个词（即去掉最后一个词）。

Decoder需要预测的是目标句子的后n-1个词（即去掉第一个词）。

例如：

目标句子：[\<sos> I love machine learning \<eos>]

Decoder输入：[\<sos> I love machine learning]

Decoder预测目标：[I love machine learning \<eos>]

这种操作确保Decoder在预测第t个词时，只能看到第1到第t-1个词，避免信息泄露。

3. 训练流程

Encoder处理源句子：

源句子（例如英语句子）经过Embedding层和位置编码后，输入到Encoder。

Encoder通过多层自注意力机制和前馈神经网络，生成一组上下文表示（Context Representations）。

Decoder处理目标句子：

目标句子（例如德语句子）经过Embedding层和位置编码后，输入到Decoder。

Decoder通过掩码自注意力机制（Masked Self-Attention），确保每个词只能关注到它之前的词。

Decoder还通过Encoder-Decoder注意力机制，结合Encoder输出的上下文表示，生成最终的输出。

损失计算：

Decoder的输出通过一个线性层和Softmax层，生成目标词汇的概率分布。

使用交叉熵损失函数，计算Decoder预测结果与真实目标句子之间的差异。

通过反向传播更新模型参数。