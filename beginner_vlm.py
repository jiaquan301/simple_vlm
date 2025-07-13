#!/usr/bin/env python3
"""
零基础VLM教程 - 配套演示代码（与入门博客代码一致版本）
========================================================

这个文件基于零基础入门博客中的VLM代码，添加了详细的解释。
确保代码完全一致，只是增加了更多的注释和演示功能。

运行这个文件，你将看到：
1. 与入门博客完全一致的VLM实现
2. 每个组件的详细工作原理解释
3. 训练过程的可视化分析
4. 模型行为的深入理解
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

print("🚀 零基础VLM教程")
print("=" * 60)
print("本演示使用与博客完全相同的代码架构")
print("只是添加了更详细的解释和演示功能")
print("=" * 60)

# ============================================================================
# 第一部分：基础组件
# ============================================================================

class SimpleTokenizer:
    """
    简单的字符级分词器
    
    🔤 作用：把文字转换成数字，让计算机能理解
    📝 例子：'红色方块' → [1, 2, 3, 4]
    """
    def __init__(self, text):
        print("📚 初始化分词器...")
        
        # 获取所有唯一字符并排序，构建词汇表
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # 字符到索引的映射
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        # 索引到字符的映射
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        print(f"   词汇表大小: {self.vocab_size}")
        print(f"   词汇表内容: {self.chars[:10]}..." if len(self.chars) > 10 else f"   词汇表内容: {self.chars}")
    
    def encode(self, text):
        """将文本编码为token索引列表"""
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, indices):
        """将token索引列表解码为文本"""
        return ''.join([self.idx_to_char[i] for i in indices])

class SimpleVisionEncoder(nn.Module):
    """
    简化版视觉编码器 - VLM的"眼睛"
    
    🎯 作用：将图像转换为特征序列，让计算机能"看懂"图片
    🔍 原理：
    1. 把图片切成小块（像拼图一样）
    2. 每个小块转换成特征向量
    3. 添加位置信息
    4. 用Transformer处理特征关系
    
    📊 输入：图像 (3, 224, 224)
    📊 输出：特征序列 (196, d_model)
    """
    def __init__(self, image_size=224, patch_size=16, d_model=128, n_layers=2):
        super().__init__()
        print(f"👁️ 初始化视觉编码器...")
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.d_model = d_model
        
        # 计算图片会被分成多少个小块
        self.num_patches = (image_size // patch_size) ** 2  # 14×14 = 196个小块
        self.patch_dim = patch_size * patch_size * 3  # 每个小块的像素数：16×16×3 = 768
        
        print(f"   图片大小: {image_size}×{image_size}")
        print(f"   分块大小: {patch_size}×{patch_size}")
        print(f"   总块数: {self.num_patches}")
        print(f"   每块像素数: {self.patch_dim}")
        
        # 线性层：把图片小块转换成特征向量
        # 就像把768个像素值压缩成128个特征值
        self.patch_embedding = nn.Linear(self.patch_dim, d_model)
        
        # 位置编码：告诉模型每个小块在图片中的位置
        # 就像给拼图块贴标签："我是第3行第5列的那一块"
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        
        # Transformer编码器：让小块之间能够"交流"
        # 这样模型能理解"左边的红色小块和右边的蓝色小块组成了一个物体"
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=8, 
            dim_feedforward=d_model*4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        print(f"   特征维度: {d_model}")
        print(f"   Transformer层数: {n_layers}")
    
    def forward(self, x):
        """
        处理图像的完整流程
        
        🔄 处理步骤：
        1. 图片分块：(B, 3, 224, 224) → (B, 196, 768)
        2. 特征映射：(B, 196, 768) → (B, 196, 128)
        3. 位置编码：添加位置信息
        4. Transformer：理解块之间关系
        """
        B = x.shape[0]  # batch size
        
        # 第一步：把图片分成小块
        # unfold操作就像用刀把图片切成规整的小方块
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, 3, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B, self.num_patches, -1)  # (B, 196, 768)
        
        # 第二步：每个小块转换成特征向量
        # 就像把每个拼图块的颜色、纹理信息提取出来
        x = self.patch_embedding(x)  # (B, 196, 128)
        
        # 第三步：添加位置编码
        # 告诉模型每个特征来自图片的哪个位置
        x = x + self.position_embedding
        
        # 第四步：Transformer处理
        # 让不同位置的特征能够"交流"，理解整体图像
        x = self.transformer(x)  # (B, 196, 128)
        
        return x

class CrossModalAttention(nn.Module):
    """
    跨模态注意力 - 让文字和图片"对话"的桥梁
    
    🌉 作用：建立文字和图片之间的关联
    💭 原理：
    - 文字作为"问题"（Query）：我想了解什么？
    - 图片作为"答案库"（Key和Value）：我能提供什么信息？
    - 根据相关性选择最重要的图片区域
    
    🎯 例子：
    - 生成"红色"时 → 主要关注图片中的红色区域
    - 生成"方块"时 → 主要关注图片中的方形边缘
    """
    def __init__(self, d_model, n_heads=8):
        super().__init__()
        print(f"🔗 初始化跨模态注意力...")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # 多头注意力：同时从多个角度理解图文关系
        # 比如一个头关注颜色，另一个头关注形状
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )
        
        # 层归一化：稳定训练过程
        self.norm = nn.LayerNorm(d_model)
        
        print(f"   特征维度: {d_model}")
        print(f"   注意力头数: {n_heads}")
        print(f"   每个头的维度: {self.head_dim}")
    
    def forward(self, text_features, image_features):
        """
        跨模态注意力计算
        
        🔄 计算流程：
        1. 文字特征作为Query（问题）
        2. 图片特征作为Key和Value（答案）
        3. 计算注意力权重
        4. 根据权重融合图片信息
        
        📊 输入：
        - text_features: (B, seq_len, d_model) 文字特征
        - image_features: (B, num_patches, d_model) 图片特征
        
        📊 输出：
        - attended_features: (B, seq_len, d_model) 融合后的特征
        """
        # 跨模态注意力：文字关注相关的图片区域
        # query=文字, key=图片, value=图片
        attended_features, attention_weights = self.multihead_attn(
            query=text_features,      # 文字问："我需要什么信息？"
            key=image_features,       # 图片答："我有这些信息可以提供"
            value=image_features      # 图片的具体内容
        )
        
        # 残差连接：保留原始文字信息
        # 这样既有新的图片信息，也不丢失原来的文字信息
        attended_features = self.norm(attended_features + text_features)
        
        return attended_features, attention_weights

class VLMTransformerBlock(nn.Module):
    """
    VLM Transformer块 - 核心的理解和融合单元
    
    🧠 作用：深度理解和融合文字、图片信息
    🔄 处理流程：
    1. 自注意力：文字内部的关系理解
    2. 跨模态注意力：文字关注相关图片区域
    3. 前馈网络：进一步处理融合信息
    
    💡 设计理念：
    - 每个块都让模型的理解更深一层
    - 多个块叠加，形成深度理解能力
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        print(f"🧠 初始化VLM Transformer块...")
        
        # 自注意力：理解文字序列内部的关系
        # 比如理解"红色的方块"中"红色"修饰"方块"
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )
        
        # 跨模态注意力：让文字关注相关的图片区域
        self.cross_attention = CrossModalAttention(d_model, n_heads)
        
        # 前馈网络：进一步处理融合后的信息
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),  # 扩展维度
            nn.ReLU(),                        # 非线性激活
            nn.Linear(d_model * 4, d_model)   # 压缩回原维度
        )
        
        # 层归一化：稳定训练，加速收敛
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        print(f"   特征维度: {d_model}")
        print(f"   注意力头数: {n_heads}")
    
    def forward(self, text_features, image_features):
        """
        VLM块的完整处理流程
        
        🔄 三步处理：
        1. 自注意力 + 残差连接
        2. 跨模态注意力 + 残差连接  
        3. 前馈网络 + 残差连接
        """
        # 第一步：自注意力 - 理解文字内部关系
        attn_output, _ = self.self_attention(text_features, text_features, text_features)
        text_features = self.norm1(text_features + attn_output)
        
        # 第二步：跨模态注意力 - 文字关注图片
        cross_attn_output, attention_weights = self.cross_attention(text_features, image_features)
        text_features = self.norm2(text_features + cross_attn_output)
        
        # 第三步：前馈网络 - 进一步处理
        ff_output = self.feed_forward(text_features)
        text_features = self.norm3(text_features + ff_output)
        
        return text_features, attention_weights

class SimpleVLM(nn.Module):
    """
    简单的视觉语言模型 - 整合所有组件
    
    🤖 作用：能够看图说话的AI模型
    🏗️ 架构：
    1. 视觉编码器：处理图片
    2. 文字嵌入：处理文字
    3. VLM块：深度融合理解
    4. 输出层：生成文字
    
    🎯 能力：
    - 图像描述：看图片，说出内容
    - 视觉问答：基于图片回答问题
    - 多模态理解：同时理解图片和文字
    """
    def __init__(self, vocab_size, d_model=128, n_heads=8, n_layers=6, max_seq_len=100):
        super().__init__()
        print(f"🤖 初始化SimpleVLM...")
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 核心组件初始化
        print("   正在初始化各个组件...")
        
        # 1. 视觉编码器：VLM的"眼睛"
        self.vision_encoder = SimpleVisionEncoder(d_model=d_model)
        
        # 2. 文字嵌入：把文字转换成特征向量
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # 3. VLM处理块：核心的理解和融合
        self.vlm_blocks = nn.ModuleList([
            VLMTransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        
        # 4. 输出层：把特征转换回文字
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # 计算总参数数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"   词汇表大小: {vocab_size}")
        print(f"   模型维度: {d_model}")
        print(f"   注意力头数: {n_heads}")
        print(f"   VLM层数: {n_layers}")
        print(f"   总参数数量: {total_params:,}")
    
    def forward(self, image, text_tokens):
        """
        VLM的完整前向传播
        
        🔄 处理流程：
        1. 图片 → 视觉特征
        2. 文字 → 文字特征
        3. 多层VLM处理 → 深度融合
        4. 输出层 → 预测下一个词
        """
        B, seq_len = text_tokens.shape
        
        # 第一步：处理图片，提取视觉特征
        image_features = self.vision_encoder(image)  # (B, 196, d_model)
        
        # 第二步：处理文字，转换为特征向量
        text_features = self.token_embedding(text_tokens)  # (B, seq_len, d_model)
        
        # 添加位置编码：告诉模型每个词的位置
        text_features = text_features + self.position_embedding[:, :seq_len, :]
        
        # 第三步：多层VLM处理，逐步深化理解
        attention_maps = []  # 保存注意力权重，用于分析
        for vlm_block in self.vlm_blocks:
            text_features, attention_weights = vlm_block(text_features, image_features)
            attention_maps.append(attention_weights)
        
        # 第四步：输出层，预测下一个词的概率分布
        logits = self.output_projection(text_features)  # (B, seq_len, vocab_size)
        
        return logits, attention_maps
    
    def generate(self, image, tokenizer, prompt="", max_length=20, temperature=1.0):
        """
        生成图片描述
        
        🎯 生成策略：
        1. 从提示词开始（如果有）
        2. 逐个预测下一个词
        3. 直到达到最大长度或生成结束符
        
        🌡️ 温度参数：
        - 低温度(0.5)：生成更确定、保守的文本
        - 高温度(1.5)：生成更随机、创新的文本
        """
        self.eval()  # 切换到评估模式
        
        with torch.no_grad():
            # 初始化生成序列
            if prompt:
                generated = tokenizer.encode(prompt)
            else:
                generated = []
            
            # 逐个生成词汇
            for _ in range(max_length):
                # 准备输入
                if len(generated) == 0:
                    # 如果没有初始词，用一个占位符
                    current_tokens = torch.tensor([[0]], dtype=torch.long)
                else:
                    current_tokens = torch.tensor([generated[-self.max_seq_len:]], dtype=torch.long)
                
                # 前向传播
                logits, _ = self.forward(image, current_tokens)
                
                # 获取最后一个位置的预测
                next_token_logits = logits[0, -1, :] / temperature
                
                # 应用softmax获取概率分布
                probs = F.softmax(next_token_logits, dim=-1)
                
                # 采样下一个token
                next_token = torch.multinomial(probs, 1).item()
                
                generated.append(next_token)
                
                # 检查是否应该停止生成
                if len(generated) > 1 and next_token == generated[0]:  # 简单的停止条件
                    break
        
        # 解码生成的文本
        return tokenizer.decode(generated)

# ============================================================================
# 第二部分：演示和分析函数
# ============================================================================

def create_demo_image():
    """
    创建演示用的红色方块图片
    
    🎨 图片内容：224×224的图片，中央有一个红色方块
    📊 格式：RGB三通道，值范围0-1
    """
    print("🎨 创建演示图片...")
    
    # 创建空白图片（灰色背景）
    image = torch.ones(3, 224, 224) * 0.5  # 灰色背景
    
    # 在中央画红色方块
    center = 112
    size = 40
    
    # 红色方块区域
    image[0, center-size:center+size, center-size:center+size] = 0.8  # 红色通道
    image[1, center-size:center+size, center-size:center+size] = 0.1  # 绿色通道
    image[2, center-size:center+size, center-size:center+size] = 0.1  # 蓝色通道
    
    print("   ✅ 创建了224×224的图片，中央有红色方块")
    return image.unsqueeze(0)  # 添加batch维度

def analyze_attention_patterns(attention_maps, tokenizer, text_tokens, step_name=""):
    """
    分析注意力模式，理解模型在"看"什么
    
    🔍 分析内容：
    - 每个词关注图片的哪些区域
    - 注意力的集中程度
    - 不同层的注意力变化
    """
    print(f"\n🔍 {step_name}注意力模式分析:")
    
    # 分析最后一层的注意力
    if len(attention_maps) > 0:
        last_attention = attention_maps[-1][0]  # (seq_len, num_patches)
        
        for i, token_id in enumerate(text_tokens[0][:min(5, len(text_tokens[0]))]):
            if i < last_attention.shape[0]:
                word = tokenizer.decode([token_id.item()])
                attention_weights = last_attention[i]
                
                # 找到注意力最高的图片区域
                max_attention_patch = torch.argmax(attention_weights).item()
                max_attention_value = torch.max(attention_weights).item()
                
                # 计算注意力的集中程度（熵）
                probs = F.softmax(attention_weights, dim=0)
                entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
                
                print(f"   '{word}': 最关注区域{max_attention_patch} (权重:{max_attention_value:.3f}, 分散度:{entropy:.2f})")

def train_vlm_step_by_step():
    """
    一步步训练VLM，展示完整的学习过程
    
    🎓 训练阶段：
    1. 数据准备
    2. 模型初始化  
    3. 训练循环
    4. 生成测试
    5. 注意力分析
    """
    print("\n🎓 开始VLM训练演示")
    print("=" * 50)
    
    # 第一步：准备训练数据
    print("\n📚 第一步：准备训练数据")
    training_text = "这是一个红色的方块。图像中央有一个红色物体。红色方块位于图像中心。这个物体是红色的。方块是红色的。红色的方块在图像中。"
    
    tokenizer = SimpleTokenizer(training_text)
    text_tokens = torch.tensor([tokenizer.encode(training_text)], dtype=torch.long)
    
    print(f"   训练文本: '{training_text[:30]}...'")
    print(f"   文本长度: {len(training_text)} 字符")
    print(f"   Token数量: {text_tokens.shape[1]}")
    
    # 第二步：创建模型
    print("\n🤖 第二步：创建VLM模型")
    model = SimpleVLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=8,
        n_layers=4
    )
    
    # 第三步：创建图片和优化器
    print("\n🎨 第三步：准备训练环境")
    image = create_demo_image()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 第四步：训练循环
    print("\n🏋️ 第四步：开始训练")
    print("   训练进度:")
    
    model.train()
    for epoch in range(41):
        optimizer.zero_grad()
        
        # 前向传播
        input_tokens = text_tokens[:, :-1]  # 输入序列
        target_tokens = text_tokens[:, 1:]  # 目标序列（下一个词）
        
        logits, attention_maps = model(image, input_tokens)
        
        # 计算损失
        loss = criterion(logits.reshape(-1, tokenizer.vocab_size), target_tokens.reshape(-1))
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 每10个epoch显示进度和分析
        if epoch % 10 == 0:
            print(f"   Epoch {epoch:2d}: Loss = {loss.item():.4f}")
            
            # 分析注意力模式的变化
            if epoch in [0, 20, 40]:
                analyze_attention_patterns(
                    attention_maps, tokenizer, input_tokens, 
                    f"Epoch {epoch} "
                )
    
    # 第五步：测试生成
    print("\n🎯 第五步：测试文本生成")
    model.eval()
    
    for temp in [0.5, 1.0, 1.5]:
        generated_text = model.generate(image, tokenizer, max_length=10, temperature=temp)
        print(f"   温度{temp}: '{generated_text}'")
    
    print("\n✅ 训练演示完成！")
    return model, tokenizer, image

def demonstrate_components():
    """
    演示VLM各个组件的工作原理
    
    🔧 演示内容：
    1. 分词器的编码解码
    2. 视觉编码器的图片处理
    3. 跨模态注意力的计算
    4. 完整模型的前向传播
    """
    print("\n🔧 VLM组件工作原理演示")
    print("=" * 40)
    
    # 演示1：分词器
    print("\n📝 演示1：分词器工作原理")
    text = "红色方块"
    tokenizer = SimpleTokenizer("红色方块在图像中")
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    print(f"   原文: '{text}'")
    print(f"   编码: {encoded}")
    print(f"   解码: '{decoded}'")
    
    # 演示2：视觉编码器
    print("\n👁️ 演示2：视觉编码器工作原理")
    vision_encoder = SimpleVisionEncoder(d_model=128)
    image = create_demo_image()
    
    print(f"   输入图片形状: {image.shape}")
    
    with torch.no_grad():
        image_features = vision_encoder(image)
        print(f"   输出特征形状: {image_features.shape}")
        print(f"   压缩比例: {(3*224*224)/(image_features.shape[1]*image_features.shape[2]):.1f}:1")
    
    # 演示3：跨模态注意力
    print("\n🔗 演示3：跨模态注意力工作原理")
    cross_attention = CrossModalAttention(d_model=128, n_heads=8)
    
    # 创建示例文字特征
    text_features = torch.randn(1, 5, 128)  # 5个词的特征
    
    with torch.no_grad():
        attended_features, attention_weights = cross_attention(text_features, image_features)
        
        print(f"   文字特征形状: {text_features.shape}")
        print(f"   图片特征形状: {image_features.shape}")
        print(f"   注意力权重形状: {attention_weights.shape}")
        print(f"   融合后特征形状: {attended_features.shape}")

def interactive_demo():
    """
    交互式演示 - 让用户体验VLM的各个功能
    """
    print("\n🎮 VLM交互式演示")
    print("=" * 40)
    print("选择你想体验的功能:")
    print("1. 完整训练演示")
    print("2. 组件工作原理")
    print("3. 注意力可视化")
    print("4. 文本生成测试")
    print("5. 退出")
    
    while True:
        try:
            choice = input("\n请输入选择 (1-5): ").strip()
            
            if choice == '1':
                train_vlm_step_by_step()
            elif choice == '2':
                demonstrate_components()
            elif choice == '3':
                print("🔍 注意力可视化演示")
                model, tokenizer, image = train_vlm_step_by_step()
                # 这里可以添加更详细的注意力可视化
            elif choice == '4':
                print("🎯 文本生成测试")
                model, tokenizer, image = train_vlm_step_by_step()
            elif choice == '5':
                print("👋 感谢使用VLM演示程序！")
                break
            else:
                print("❌ 无效选择，请输入1-5")
                
        except KeyboardInterrupt:
            print("\n👋 程序已退出")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")

# ============================================================================
# 主程序入口
# ============================================================================

def main():
    """
    主程序 - 运行完整的VLM演示
    """
    print("\n🎉 这是零基础入门博客的配套演示程序")
    print("现在你可以:")
    print("✅ 理解真实VLM的工作原理")
    print("✅ 看到与技术博客一致的实现")
    print("✅ 体验完整的训练和生成过程")
    print("✅ 分析模型的注意力机制")
    
    # 运行完整演示
    model, tokenizer, image = train_vlm_step_by_step()
    
    print("\n🎓 学习总结:")
    print("通过这个演示，你学会了:")
    print("• VLM的完整架构和实现")
    print("• 每个组件的具体作用")
    print("• 训练过程的详细分析")
    print("• 注意力机制的工作原理")
    print("• 如何生成图片描述")
    
    print("\n📚 代码一致性保证:")
    print("• 与技术博客使用相同的架构")
    print("• 相同的组件设计和实现")
    print("• 一致的参数设置和训练方式")
    print("• 只是增加了更详细的解释")

if __name__ == "__main__":
    # 检查PyTorch是否可用
    if not torch.cuda.is_available():
        print("💡 提示：未检测到GPU，将使用CPU运行（速度较慢）")
    
    # 可以选择运行交互式演示或完整演示
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_demo()
    else:
        main()

