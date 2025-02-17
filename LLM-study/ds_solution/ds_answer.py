import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import argparse
import numpy as np

# 超参数配置
class Config:
    # 数据参数
    vocab_size = 12  # 0:PAD, 1-9:数字, 10:SOS, 11:EOS
    sos_token = 10
    eos_token = 11
    pad_token = 0
    max_length = 8
    
    # 模型参数
    d_model = 128
    nhead = 4
    num_layers = 2 # numbers of encoder and decoder layers
    dim_feedforward = 512
    dropout = 0.1
    
    # 训练参数
    batch_size = 256
    num_epochs = 100
    learning_rate = 0.001
    train_samples = 5000
    test_samples = 200
    
    # 系统参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 42

# 设置随机种子
torch.manual_seed(Config.seed)
np.random.seed(Config.seed)

# 改进的数据集
class FixedCustomDataset(Dataset):
    def __init__(self, num_samples, mode='train'):
        super().__init__()
        self.num_samples = num_samples
        self.max_length = Config.max_length
        self.mode = mode

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成有效数据部分（1-9）
        data_length = torch.randint(2, self.max_length-1, (1,)).item()  # 保留SOS/EOS位置
        data = torch.randint(1, 10, (data_length,))  # 有效数据为1-9
        
        # 添加特殊符号
        sequence = torch.cat([
            torch.tensor([Config.sos_token]),
            data,
            torch.tensor([Config.eos_token]),
            torch.zeros(self.max_length - data_length - 2, dtype=torch.long)
        ])
        
        # 训练模式返回输入输出相同，推理模式可以设计不同任务
        return sequence, sequence

# 改进的Transformer模型
class ImprovedTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(Config.vocab_size, Config.d_model)
        self.pos_encoder = PositionalEncoding(Config.d_model, Config.max_length, Config.dropout)
        self.transformer = nn.Transformer(
            d_model=Config.d_model,
            nhead=Config.nhead,
            num_encoder_layers=Config.num_layers,
            num_decoder_layers=Config.num_layers,
            dim_feedforward=Config.dim_feedforward,
            dropout=Config.dropout,
            batch_first=True
        )
        self.fc = nn.Linear(Config.d_model, Config.vocab_size)
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Create padding masks before embedding
        src_key_padding_mask = (src == Config.pad_token)
        tgt_key_padding_mask = (tgt == Config.pad_token)
        
        # Embedding and positional encoding
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        
        # Scale embeddings
        src = src * math.sqrt(Config.d_model)
        tgt = tgt * math.sqrt(Config.d_model)
        
        # Apply positional encoding
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        # Debug shape information
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Ensure CUDA operations are synchronized
        
        output = self.transformer(
            src, tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        return self.fc(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)  # Changed to match batch_first=True
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]  # Match sequence length
        return self.dropout(x)

# 训练函数
def train():
    # 初始化
    model = ImprovedTransformer().to(Config.device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=Config.pad_token, label_smoothing=0.1)
    
    # 数据加载
    train_set = FixedCustomDataset(Config.train_samples, 'train')
    train_loader = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True)
    
    # 训练循环
    model.train()
    for epoch in range(Config.num_epochs):
        total_loss = 0
        for batch_idx, (src, tgt) in enumerate(train_loader):
            src = src.to(Config.device)
            tgt = tgt.to(Config.device)
            
            # Ensure tensors are contiguous
            src = src.contiguous()
            tgt = tgt.contiguous()
            
            tgt_input = tgt[:, :-1].contiguous()
            tgt_input[tgt_input == Config.eos_token] = Config.pad_token
            tgt_output = tgt[:, 1:].contiguous()
            
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(Config.device)
            
            # 前向传播
            # print(src.shape, tgt_input.shape, tgt_output.shape)
            output = model(src, tgt_input, tgt_mask=tgt_mask)
            loss = criterion(output.reshape(-1, Config.vocab_size), tgt_output.reshape(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f'Epoch [{epoch+1}/{Config.num_epochs}] Batch {batch_idx} Loss: {loss.item():.4f}')
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{Config.num_epochs}] Avg Loss: {avg_loss:.4f}')
    
    # 保存模型
    torch.save(model.state_dict(), 'improved_transformer.pth')
    print('Training completed. Model saved.')

# 推理函数
def generate_sequence(model, input_seq, max_length=Config.max_length):
    model.eval()
    with torch.no_grad():
        # 处理输入序列的padding mask
        src_key_padding_mask = (input_seq == Config.pad_token)
        
        # 编码器处理
        src_embedded = model.pos_encoder(model.embedding(input_seq) * math.sqrt(Config.d_model))
        memory = model.transformer.encoder(
            src_embedded,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # 初始化解码器输入
        outputs = torch.full((1, 1), Config.sos_token, device=Config.device)
        
        for _ in range(max_length-1):
            # 生成正确大小的causal mask
            size = outputs.size(1)
            tgt_mask = torch.triu(
                torch.full((size, size), float('-inf'), device=Config.device), 
                diagonal=1
            )
            
            # 解码器处理
            tgt_embedded = model.pos_encoder(model.embedding(outputs) * math.sqrt(Config.d_model))
            decoder_output = model.transformer.decoder(
                tgt_embedded,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
            
            # 获取下一个token
            logits = model.fc(decoder_output[:, -1:])
            next_token = logits.argmax(dim=-1)
            
            # 确保CUDA操作同步
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            
            outputs = torch.cat([outputs, next_token], dim=1)
            # print(outputs)
            
            if next_token.item() == Config.eos_token:
                break
        
        return outputs[0, 1:-1]  # 去掉SOS和EOS

# 测试函数
def test():
    try:
        # 加载模型
        model = ImprovedTransformer().to(Config.device)
        model.load_state_dict(
            torch.load('improved_transformer.pth', 
                      map_location=Config.device,
                      weights_only=True)
        )
        
        # 准备测试数据
        test_set = FixedCustomDataset(Config.test_samples, 'test')
        test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
        
        total_correct = 0
        print("\nTesting Examples:")
        
        for idx, (src, tgt) in enumerate(test_loader):
            src = src.to(Config.device)
            
            # 生成预测
            pred = generate_sequence(model, src)
            
            # Move tensors to CPU before numpy conversion
            src_cpu = src[0, 1:-1].cpu()
            src_cpu[src_cpu == Config.eos_token] = Config.pad_token
            mask = src_cpu != Config.pad_token
            input_seq = src_cpu[mask].numpy()
            
            # 处理真实标签（去掉SOS/EOS/PAD）
            true_seq = tgt[0, 1:-1]
            true_seq[true_seq == Config.eos_token] = Config.pad_token
            true_seq = true_seq[true_seq != Config.pad_token].numpy()
            
            # 转换预测结果到CPU
            pred = pred.cpu().numpy()
            
            # 统计结果
            if np.array_equal(pred, true_seq):
                total_correct += 1
            
            if idx < 10:  # 显示前3个样例
                print(f"\nSample {idx+1}:")
                print(f"Input: {input_seq}")
                print(f"True:  {true_seq}")
                print(f"Pred:  {pred}")
        
        print(f"\nFinal Accuracy: {total_correct/Config.test_samples:.2%}")

    except Exception as e:
        print(f"Error during testing: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise

# 命令行接口
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transformer Copy Task')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=False,
                       help='运行模式：train（训练）或 test（测试）', default="train")
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()