#!/usr/bin/env python3
"""
最简VLM实现 - 基于LLM扩展的视觉语言模型
作者：jiaquan301@163.com
功能：在SimpleLLM基础上添加视觉处理能力，实现图像理解和描述生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

class SimpleTokenizer:
    """简单的字符级分词器（从LLM继承）"""
    def __init__(self, text):
        # 获取所有唯一字符并排序，构建词汇表
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        # 字符到索引的映射
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        # 索引到字符的映射
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, text):
        """将文本编码为token索引列表"""
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, indices):
        """将token索引列表解码为文本"""
        return ''.join([self.idx_to_char[i] for i in indices])

class SimpleVisionEncoder(nn.Module):
    """
    简化版视觉编码器 - 将图像转换为特征序列   
    这是VLM的核心组件之一，负责理解图像内容
    类似于CLIP的图像编码器，但更简化
    """
    def __init__(self, image_size=224, patch_size=16, d_model=128, n_layers=2):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.d_model = d_model
        
        # 计算patch数量
        self.n_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size  # RGB * patch_size^2
        
        print(f"🖼️  视觉编码器初始化:")
        print(f"   图像大小: {image_size}x{image_size}")
        print(f"   Patch大小: {patch_size}x{patch_size}")
        print(f"   Patch数量: {self.n_patches}")
        print(f"   输出维度: {d_model}")
        
        # Patch嵌入层：将图像patch转换为向量
        self.patch_embedding = nn.Linear(self.patch_dim, d_model)
        
        # 位置嵌入：为每个patch添加位置信息
        self.position_embedding = nn.Parameter(torch.randn(1, self.n_patches, d_model))
        
        # 简化的Transformer编码器
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True
            ) for _ in range(n_layers)
        ])
        
        # 层归一化
        self.norm = nn.LayerNorm(d_model)
        
    def patchify(self, images):
        """
        将图像分割为patches
        输入: (batch_size, 3, H, W)
        输出: (batch_size, n_patches, patch_dim)
        """
        batch_size, channels, height, width = images.shape
        
        # 确保图像尺寸正确
        assert height == width == self.image_size, f"图像尺寸应为{self.image_size}x{self.image_size}"
        
        # 分割为patches
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, channels, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()
        patches = patches.view(batch_size, self.n_patches, -1)
        
        return patches
    
    def forward(self, images):
        """
        前向传播
        输入: (batch_size, 3, H, W) 图像张量
        输出: (batch_size, n_patches, d_model) 视觉特征
        """
        batch_size = images.shape[0]
        
        # 1. 将图像分割为patches
        patches = self.patchify(images)  # (batch_size, n_patches, patch_dim)
        
        # 2. Patch嵌入
        x = self.patch_embedding(patches)  # (batch_size, n_patches, d_model)
        
        # 3. 添加位置嵌入
        x = x + self.position_embedding
        
        # 4. 通过Transformer层
        for layer in self.transformer_layers:
            x = layer(x)
        
        # 5. 层归一化
        x = self.norm(x)
        
        return x

class MultiHeadAttention(nn.Module):
    """多头注意力机制（从LLM继承）"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)
        
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores.masked_fill_(mask, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        return self.out_linear(attention_output)

class CrossModalAttention(nn.Module):
    """
    跨模态注意力机制 - VLM的核心创新
    
    允许文本token关注图像特征，实现视觉-语言交互
    这是VLM相比LLM的关键扩展
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Query来自文本，Key和Value来自图像
        self.q_linear = nn.Linear(d_model, d_model)  # 文本查询
        self.k_linear = nn.Linear(d_model, d_model)  # 图像键
        self.v_linear = nn.Linear(d_model, d_model)  # 图像值
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, text_features, image_features):
        """
        跨模态注意力计算
        text_features: (batch_size, text_len, d_model) 文本特征
        image_features: (batch_size, image_len, d_model) 图像特征
        """
        batch_size, text_len, d_model = text_features.shape
        _, image_len, _ = image_features.shape
        
        # 文本作为Query
        Q = self.q_linear(text_features)
        # 图像作为Key和Value
        K = self.k_linear(image_features)
        V = self.v_linear(image_features)
        
        # 重塑为多头形式
        Q = Q.view(batch_size, text_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, image_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, image_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数：文本关注图像
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # 加权求和图像特征
        attention_output = torch.matmul(attention_weights, V)
        
        # 重塑并输出
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, text_len, d_model)
        
        return self.out_linear(attention_output)

class VLMTransformerBlock(nn.Module):
    """
    VLM专用的Transformer块
    
    在标准Transformer块基础上添加跨模态注意力
    这是VLM能够理解图像和文本关系的关键
    """
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        
        # 自注意力（文本内部）
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        
        # 跨模态注意力（文本关注图像）
        self.cross_attention = CrossModalAttention(d_model, n_heads)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, text_features, image_features, causal_mask=None):
        """
        前向传播
        text_features: 文本特征
        image_features: 图像特征
        causal_mask: 因果掩码（用于文本生成）
        """
        # 1. 文本自注意力
        attn_output = self.self_attention(text_features, causal_mask)
        text_features = self.norm1(text_features + attn_output)
        
        # 2. 跨模态注意力（文本关注图像）
        cross_attn_output = self.cross_attention(text_features, image_features)
        text_features = self.norm2(text_features + cross_attn_output)
        
        # 3. 前馈网络
        ff_output = self.feed_forward(text_features)
        text_features = self.norm3(text_features + ff_output)
        
        return text_features

class SimpleVLM(nn.Module):
    """
    简化版视觉语言模型
    
    在SimpleLLM基础上添加视觉处理能力
    核心思想：图像 + 文本 → 理解 → 生成
    """
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, max_seq_len=64, 
                 image_size=224, patch_size=16):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        print(f"🤖 初始化SimpleVLM:")
        print(f"   词汇表大小: {vocab_size}")
        print(f"   模型维度: {d_model}")
        print(f"   注意力头数: {n_heads}")
        print(f"   Transformer层数: {n_layers}")
        
        # 1. 视觉编码器：处理图像
        self.vision_encoder = SimpleVisionEncoder(
            image_size=image_size, 
            patch_size=patch_size, 
            d_model=d_model
        )
        
        # 2. 文本嵌入层：处理文本（从LLM继承）
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # 3. VLM专用Transformer块
        self.vlm_blocks = nn.ModuleList([
            VLMTransformerBlock(d_model, n_heads, d_model * 4) 
            for _ in range(n_layers)
        ])
        
        # 4. 输出层：生成文本
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # 计算参数数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   总参数数量: {total_params:,}")
        
    def forward(self, images, text_tokens):
        """
        前向传播
        images: (batch_size, 3, H, W) 图像张量
        text_tokens: (batch_size, seq_len) 文本token
        """
        batch_size, seq_len = text_tokens.shape
        
        # 1. 处理图像：提取视觉特征
        image_features = self.vision_encoder(images)  # (batch_size, n_patches, d_model)
        
        # 2. 处理文本：嵌入和位置编码
        positions = torch.arange(seq_len, device=text_tokens.device).unsqueeze(0).expand(batch_size, -1)
        text_features = self.token_embedding(text_tokens) + self.position_embedding(positions)
        
        # 3. 创建因果掩码（确保生成时不能看到未来token）
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # 扩展维度匹配多头注意力
        
        # 4. 通过VLM Transformer块：融合视觉和文本信息
        for vlm_block in self.vlm_blocks:
            text_features = vlm_block(text_features, image_features, causal_mask)
        
        # 5. 输出投影：生成词汇表概率
        logits = self.output_projection(text_features)
        
        return logits
    
    def generate_caption(self, image, tokenizer, max_length=50, temperature=1.0):
        """
        图像描述生成
        
        这是VLM的核心应用：看图说话
        输入图像，输出文字描述
        """
        self.eval()
        
        # 准备图像（添加batch维度）
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # 初始化生成序列（从特殊开始token开始）
        generated_tokens = [0]  # 假设0是开始token
        
        print(f"🎯 开始生成图像描述...")
        
        with torch.no_grad():
            for step in range(max_length):
                # 准备当前序列
                current_tokens = torch.tensor([generated_tokens])
                
                # 确保序列长度不超过最大长度
                if len(generated_tokens) >= self.max_seq_len:
                    current_tokens = torch.tensor([generated_tokens[-self.max_seq_len:]])
                
                # 前向传播
                logits = self.forward(image, current_tokens)
                
                # 获取最后一个位置的预测
                next_token_logits = logits[0, -1, :] / temperature
                
                # 采样下一个token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                generated_tokens.append(next_token)
                
                # 如果生成了结束token，停止生成
                if next_token == 0:  # 假设0也是结束token
                    break
        
        # 解码生成的文本
        try:
            caption = tokenizer.decode(generated_tokens[1:])  # 跳过开始token
            print(f"✅ 生成完成: '{caption}'")
            return caption
        except:
            print("⚠️  解码失败，返回原始token")
            return str(generated_tokens)

def create_dummy_image(size=(224, 224)):
    """
    创建一个虚拟图像用于演示
    
    在实际应用中，这里应该是真实的图像数据
    """
    # 创建一个简单的彩色图像
    image = np.random.rand(3, size[0], size[1]).astype(np.float32)
    
    # 添加一些结构化的模式，让图像更有意义
    # 创建一个简单的"物体"在图像中心
    center_x, center_y = size[0] // 2, size[1] // 2
    for i in range(center_x - 20, center_x + 20):
        for j in range(center_y - 20, center_y + 20):
            if 0 <= i < size[0] and 0 <= j < size[1]:
                image[:, i, j] = [0.8, 0.2, 0.2]  # 红色方块
    
    return torch.tensor(image)

def train_simple_vlm():
    """训练简单VLM的示例"""
    print("🚀 开始VLM训练演示")
    print("="*60)
    
    # 准备训练数据
    text = """
    这是一个红色的方块。图像中央有一个红色物体。
    红色方块位于图像中心。这个物体是红色的。
    图像显示了一个红色的正方形。中央是红色区域。
    """
    
    print(f"📚 训练文本: {text[:50]}...")
    
    # 初始化分词器和模型
    tokenizer = SimpleTokenizer(text)
    model = SimpleVLM(vocab_size=tokenizer.vocab_size, d_model=64, n_heads=4, n_layers=2)
    
    # 准备训练数据
    input_ids = tokenizer.encode(text)
    
    # 创建虚拟图像数据
    dummy_image = create_dummy_image()
    
    # 训练设置
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    
    print(f"\n🏋️  开始训练...")
    
    for epoch in range(50):  # 减少训练轮数，因为VLM更复杂
        # 随机选择文本片段
        start_idx = random.randint(0, max(0, len(input_ids) - model.max_seq_len - 1))
        end_idx = start_idx + min(model.max_seq_len, len(input_ids) - start_idx - 1)
        
        # 输入和目标
        x_text = torch.tensor([input_ids[start_idx:end_idx]])
        y_text = torch.tensor([input_ids[start_idx+1:end_idx+1]])
        
        # 图像输入（批次维度）
        x_image = dummy_image.unsqueeze(0)
        
        # 前向传播
        logits = model(x_image, x_text)
        
        # 计算损失
        loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y_text.view(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"   Epoch {epoch:2d}, Loss: {loss.item():.4f}")
    
    print("✅ 训练完成！")
    
    # 测试图像描述生成
    print(f"\n🎯 测试图像描述生成:")
    print("-" * 40)
    
    test_image = create_dummy_image()
    caption = model.generate_caption(test_image, tokenizer, max_length=20, temperature=0.8)
    
    print(f"🖼️  输入: 虚拟红色方块图像")
    print(f"📝 生成描述: {caption}")
    
    return model, tokenizer

def demonstrate_vlm_components():
    """演示VLM各个组件的工作原理"""
    print("\n" + "="*60)
    print("🔍 VLM组件演示")
    print("="*60)
    
    # 1. 视觉编码器演示
    print("\n1️⃣ 视觉编码器演示:")
    vision_encoder = SimpleVisionEncoder(image_size=224, patch_size=16, d_model=64)
    
    dummy_image = create_dummy_image().unsqueeze(0)  # 添加batch维度
    visual_features = vision_encoder(dummy_image)
    
    print(f"   输入图像形状: {dummy_image.shape}")
    print(f"   输出特征形状: {visual_features.shape}")
    print(f"   特征含义: {visual_features.shape[1]}个图像patch，每个patch用{visual_features.shape[2]}维向量表示")
    
    # 2. 跨模态注意力演示
    print("\n2️⃣ 跨模态注意力演示:")
    cross_attention = CrossModalAttention(d_model=64, n_heads=4)
    
    # 模拟文本特征
    text_features = torch.randn(1, 10, 64)  # 10个文本token
    
    # 计算跨模态注意力
    attended_features = cross_attention(text_features, visual_features)
    
    print(f"   文本特征形状: {text_features.shape}")
    print(f"   图像特征形状: {visual_features.shape}")
    print(f"   注意力输出形状: {attended_features.shape}")
    print(f"   含义: 文本token关注图像patch，获得视觉信息")

if __name__ == "__main__":
    print("🎨 最简VLM实现演示")
    print("="*60)
    print("这是一个基于SimpleLLM扩展的视觉语言模型")
    print("核心功能：图像理解 + 文本生成 = 看图说话")
    print()
    
    # 演示VLM组件
    demonstrate_vlm_components()
    
    # 训练和测试VLM
    model, tokenizer = train_simple_vlm()
    
    print("\n" + "="*60)
    print("🎉 VLM演示完成！")
    print("="*60)
    print("通过这个演示，你了解了:")
    print("✅ 如何从LLM扩展到VLM")
    print("✅ 视觉编码器的工作原理")
    print("✅ 跨模态注意力机制")
    print("✅ 图像描述生成过程")
    print("\n💡 VLM的核心思想:")
    print("   图像编码器提取视觉特征")
    print("   跨模态注意力融合视觉和文本信息")
    print("   语言模型生成描述文本")

