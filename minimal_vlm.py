#!/usr/bin/env python3
"""
最简VLM实现 - 核心版本
基于LLM扩展的视觉语言模型，突出核心逻辑
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

class SimpleTokenizer:
    """字符级分词器"""
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, text):
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, indices):
        return ''.join([self.idx_to_char[i] for i in indices])

class VisionEncoder(nn.Module):
    """视觉编码器：图像 → 特征序列"""
    def __init__(self, image_size=224, patch_size=16, d_model=128):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size
        
        self.patch_embedding = nn.Linear(patch_dim, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, self.n_patches, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead=4, dim_feedforward=d_model*4, batch_first=True),
            num_layers=2
        )
        
    def patchify(self, images):
        """图像分割为patches"""
        B, C, H, W = images.shape
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous().view(B, self.n_patches, -1)
        return patches
        
    def forward(self, images):
        patches = self.patchify(images)
        x = self.patch_embedding(patches) + self.position_embedding
        return self.transformer(x)

class CrossModalAttention(nn.Module):
    """跨模态注意力：文本关注图像"""
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
    def forward(self, text_features, image_features):
        B, T, D = text_features.shape
        _, I, _ = image_features.shape
        
        Q = self.q_linear(text_features).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(image_features).view(B, I, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(image_features).view(B, I, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_linear(attn_output)

class VLMBlock(nn.Module):
    """VLM Transformer块"""
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.cross_attn = CrossModalAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, text_features, image_features, causal_mask=None):
        # 文本自注意力
        attn_out, _ = self.self_attn(text_features, text_features, text_features, attn_mask=causal_mask)
        text_features = self.norm1(text_features + attn_out)
        
        # 跨模态注意力
        cross_out = self.cross_attn(text_features, image_features)
        text_features = self.norm2(text_features + cross_out)
        
        # 前馈网络
        ffn_out = self.ffn(text_features)
        text_features = self.norm3(text_features + ffn_out)
        
        return text_features

class SimpleVLM(nn.Module):
    """简化版视觉语言模型"""
    def __init__(self, vocab_size, d_model=128, n_layers=2, max_seq_len=64):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 组件初始化
        self.vision_encoder = VisionEncoder(d_model=d_model)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.vlm_blocks = nn.ModuleList([VLMBlock(d_model) for _ in range(n_layers)])
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, images, text_tokens):
        B, T = text_tokens.shape
        
        # 处理图像和文本
        image_features = self.vision_encoder(images)
        positions = torch.arange(T, device=text_tokens.device).unsqueeze(0).expand(B, -1)
        text_features = self.token_embedding(text_tokens) + self.position_embedding(positions)
        
        # 因果掩码
        causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        
        # VLM处理
        for block in self.vlm_blocks:
            text_features = block(text_features, image_features, causal_mask)
        
        return self.output_projection(text_features)
    
    def generate(self, image, tokenizer, max_length=20):
        """图像描述生成"""
        self.eval()
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
            
        generated = [0]  # 开始token
        
        with torch.no_grad():
            for _ in range(max_length):
                tokens = torch.tensor([generated[-self.max_seq_len:]])
                logits = self.forward(image, tokens)
                next_token = torch.multinomial(F.softmax(logits[0, -1, :], dim=-1), 1).item()
                generated.append(next_token)
                if next_token == 0:  # 结束token
                    break
                    
        return tokenizer.decode(generated[1:])

def create_demo_image():
    """创建演示图像"""
    image = np.random.rand(3, 224, 224).astype(np.float32)
    # 添加红色方块
    center = 112
    image[0, center-20:center+20, center-20:center+20] = 0.8  # R
    image[1, center-20:center+20, center-20:center+20] = 0.2  # G
    image[2, center-20:center+20, center-20:center+20] = 0.2  # B
    return torch.tensor(image)

def main():
    print("🎨 最简VLM核心实现")
    print("="*50)
    
    # 数据和模型
    text = "这是红色方块。图像中央有红色物体。红色方块在中心。"
    tokenizer = SimpleTokenizer(text)
    model = SimpleVLM(vocab_size=tokenizer.vocab_size, d_model=64)
    
    print(f"📊 模型信息:")
    print(f"   词汇表大小: {tokenizer.vocab_size}")
    print(f"   参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练
    input_ids = tokenizer.encode(text)
    image = create_demo_image()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n🏋️  训练中...")
    model.train()
    for epoch in range(30):
        start = random.randint(0, max(0, len(input_ids) - model.max_seq_len - 1))
        end = start + min(model.max_seq_len, len(input_ids) - start - 1)
        
        x_text = torch.tensor([input_ids[start:end]])
        y_text = torch.tensor([input_ids[start+1:end+1]])
        x_image = image.unsqueeze(0)
        
        logits = model(x_image, x_text)
        loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y_text.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"   Epoch {epoch:2d}: Loss = {loss.item():.3f}")
    
    # 测试生成
    print(f"\n🎯 图像描述生成:")
    caption = model.generate(image, tokenizer)
    print(f"   生成结果: '{caption}'")
    
    print(f"\n✅ 演示完成！")
    print("💡 核心概念:")
    print("   • 视觉编码器: 图像 → patch特征序列")
    print("   • 跨模态注意力: 文本关注图像特征")
    print("   • VLM块: 融合视觉和语言信息")
    print("   • 生成: 基于图像内容生成描述文本")

if __name__ == "__main__":
    main()

