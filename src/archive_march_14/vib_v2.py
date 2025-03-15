import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
import matplotlib.pyplot as plt

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split, Subset

# ====================== 数据增强 ======================
# 在训练集中，我们使用 Albumentations 做较强的增广
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),
    # 修正 CoarseDropout 参数
    A.CoarseDropout(max_holes=5, max_height=16, max_width=16, p=0.5),
    A.Resize(224, 224),           # 统一缩放到224×224
    ToTensorV2()
], additional_targets={'mask': 'mask'})

# 验证/测试集中，我们通常只做最基本的resize + ToTensor
val_transform = A.Compose([
    A.Resize(224, 224),
    ToTensorV2()
], additional_targets={'mask': 'mask'})

# ====================== 自定义Dataset ======================
class TextSegDataset(Dataset):
    """
    读取 (image, mask) 对，并应用数据增强
    假设:
    - 原图目录: img_dir
    - 掩码目录: mask_dir
    - 每张原图对应一个同名 + '_mask.png' 的掩码文件
    """
    def __init__(self, img_dir, mask_dir, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        self.img_files = sorted([f for f in os.listdir(img_dir)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        base_name = img_name.rsplit('.', 1)[0]
        
        img_path  = os.path.join(self.img_dir,  img_name)
        mask_path = os.path.join(self.mask_dir, base_name + "_mask.png")
        
        # 读取图像(BGR->RGB)与掩码(灰度)
        img_bgr  = cv2.imread(img_path)
        mask_gray= cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        # 调试: 输出图像和掩码的路径
        print(f"读取图像: {img_path}, 掩码: {mask_path}")
        
        if img_bgr is None or mask_gray is None:
            raise FileNotFoundError(f"Cannot read image/mask: {img_path}, {mask_path}")
        
        # 调试: 输出图像和掩码形状和值范围
        print(f"原始图像形状: {img_bgr.shape}, 掩码形状: {mask_gray.shape}")
        print(f"图像值范围: [{img_bgr.min()}, {img_bgr.max()}], 掩码值范围: [{mask_gray.min()}, {mask_gray.max()}]")
        
        # 转成 float32
        img_bgr = img_bgr.astype(np.float32)
        mask_gray = mask_gray.astype(np.float32)

        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Albumentations需要 numpy array
        # mask必须是单通道
        if self.transform:
            # 传入transform时要指定 'image' 和 'mask'
            augmented = self.transform(image=img_rgb, mask=mask_gray)
            img_t  = augmented['image']   # tensor: [3,224,224]
            mask_t = augmented['mask']    # tensor: [224,224]
        else:
            # 如果不做增强，就手动ToTensor
            img_rgb  = cv2.resize(img_rgb, (224,224))
            mask_gray= cv2.resize(mask_gray, (224,224), interpolation=cv2.INTER_NEAREST)
            img_t  = torch.from_numpy(img_rgb.transpose(2,0,1)).float() / 255.0
            mask_t = torch.from_numpy(mask_gray).float()
        
        # 如果掩码是0/255，转成0/1
        mask_t = (mask_t > 0.5).float().unsqueeze(0)  # [1,224,224]
        
        # 调试: 检查转换后的掩码
        unique_values = torch.unique(mask_t).numpy()
        print(f"转换后掩码的唯一值: {unique_values}, 形状: {mask_t.shape}")
        
        return img_t, mask_t

# ====================== 定义一个简化的UNet模型 ======================
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        # 编码器
        self.enc1 = self._block(3, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        
        # 中间层
        self.middle = self._block(256, 512)
        
        # 解码器
        self.dec3 = self._block(512+256, 256)
        self.dec2 = self._block(256+128, 128)
        self.dec1 = self._block(128+64, 64)
        
        # 最终输出
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        
        # 池化和上采样
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # 编码路径
        enc1 = self.enc1(x)
        x = self.pool(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool(enc3)
        
        # 中间
        x = self.middle(x)
        
        # 解码路径
        x = self.up(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.up(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.up(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # 最终输出
        return self.final(x)

# 保持原有的ViT模型作为备选
class SmallViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16,
                 embed_dim=128, num_heads=4, depth=4, num_classes=1):
        super().__init__()
        self.patch_size = patch_size
        # 1) 补丁嵌入
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        # 可学习位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # 2) Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim*4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 3) 解码头(上采样回原图尺寸)
        # 修改为更复杂的上采样层
        self.up = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim//2, kernel_size=patch_size//2, stride=patch_size//2),
            nn.BatchNorm2d(embed_dim//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(embed_dim//2, num_classes, kernel_size=2, stride=2)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        # Patch Embedding
        x = self.patch_embed(x)  # [B,embed_dim,H/patch,W/patch]
        # flatten => [B,embed_dim, N], transpose => [B,N,embed_dim]
        x = x.flatten(2).transpose(1,2)
        
        # 加上位置编码
        x = x + self.pos_embed[:, :x.size(1), :]
        
        # Transformer
        x = self.transformer(x)  # [B,N,embed_dim]
        
        # reshape回 CNN 形式
        Hp = H // self.patch_size
        Wp = W // self.patch_size
        x = x.transpose(1,2).reshape(B, -1, Hp, Wp)  # [B,embed_dim,Hp,Wp]
        
        # 上采样回原图大小 => [B,num_classes,H,W]
        x = self.up(x)
        return x

# ====================== Dice Loss ======================
def dice_loss_fn(logits, targets):
    """
    logits: [B,1,H,W] raw output
    targets: [B,1,H,W] in {0,1}
    返回: 平均Dice损失 (越小越好)
    """
    probs = torch.sigmoid(logits)  # [B,1,H,W]
    num = (probs * targets).sum(dim=(1,2,3)) * 2.0
    den = (probs + targets).sum(dim=(1,2,3)) + 1e-6
    dice = num / den
    
    # 调试: 打印出dice值
    print(f"Dice值: {dice}")
    
    return 1 - dice.mean()

# ====================== 调试函数 ======================
def check_masks(loader, name):
    """检查掩码是否有效"""
    all_zeros = 0
    all_ones = 0
    has_content = 0
    total = 0
    
    for _, masks in loader:
        total += masks.size(0)
        for mask in masks:
            if torch.all(mask == 0):
                all_zeros += 1
            elif torch.all(mask == 1):
                all_ones += 1
            else:
                has_content += 1
    
    print(f"{name} 数据集掩码统计：总数 {total}, 全0: {all_zeros}, 全1: {all_ones}, 有内容: {has_content}")

def debug_validation(loader, model, device, epoch):
    """在验证阶段输出详细的调试信息"""
    model.eval()
    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(loader):
            imgs, masks = imgs.to(device), masks.to(device)
            print(f"批次 {batch_idx}:")
            print(f"  掩码形状: {masks.shape}, 掩码最小值: {masks.min().item()}, 掩码最大值: {masks.max().item()}")
            print(f"  掩码唯一值: {torch.unique(masks).cpu().numpy()}")
            
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            print(f"  预测形状: {probs.shape}, 预测最小值: {probs.min().item()}, 预测最大值: {probs.max().item()}")
            unique_probs = torch.unique(probs)
            print(f"  预测唯一值数量: {len(unique_probs)}")
            if len(unique_probs) <= 10:
                print(f"  预测唯一值: {unique_probs.cpu().numpy()}")
            else:
                print(f"  预测唯一值前10个: {unique_probs[:10].cpu().numpy()}")
            
            pred_bin = (probs > 0.5).float()
            print(f"  二值化预测唯一值: {torch.unique(pred_bin).cpu().numpy()}")
            
            # 计算IoU和Dice
            intersect = (pred_bin * masks).sum(dim=(1,2,3))
            union = ((pred_bin + masks) > 0).float().sum(dim=(1,2,3)) + 1e-6
            iou = (intersect / union).mean().item()
            
            dice_num = 2*intersect
            dice_den = pred_bin.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3)) + 1e-6
            dice = (dice_num/dice_den).mean().item()
            
            print(f"  批次IoU: {iou:.4f}, 批次Dice: {dice:.4f}")
            print(f"  预测区域像素数: {pred_bin.sum().item()}, 真实区域像素数: {masks.sum().item()}")
            
            # 保存第一个批次的预测结果
            if batch_idx == 0:
                visualize_batch(imgs, masks, pred_bin, epoch)
            
            if batch_idx >= 2:  # 只检查前几个批次
                break

def visualize_batch(imgs, masks, preds, epoch):
    """可视化预测结果，使用彩色映射增强预测结果的对比度"""
    batch_size = imgs.size(0)
    
    # 修改为4列而不是3列
    fig, axes = plt.subplots(batch_size, 4, figsize=(20, 5*batch_size))
    
    for i in range(batch_size):
        # 将张量转回numpy以进行可视化
        img_np = imgs[i].cpu().permute(1, 2, 0).numpy()
        mask_np = masks[i, 0].cpu().numpy()
        pred_np = preds[i, 0].cpu().numpy()
        
        # 打印预测的统计信息以便调试
        print(f"样本 {i} 预测统计: 最小值={pred_np.min():.4f}, 最大值={pred_np.max():.4f}, 平均值={pred_np.mean():.4f}")
        
        # 归一化图像用于显示
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
        
        if batch_size > 1:
            ax_row = axes[i]
        else:
            ax_row = axes
        
        # 输入图像
        ax_row[0].imshow(img_np)
        ax_row[0].set_title('Input Image')
        ax_row[0].axis('off')
        
        # 真实标签
        ax_row[1].imshow(mask_np, cmap='gray')
        ax_row[1].set_title(f'Ground Truth (sum={mask_np.sum():.1f})')
        ax_row[1].axis('off')
        
        # 预测结果(彩色映射)
        pred_vis = ax_row[2].imshow(pred_np, cmap='plasma', vmin=0, vmax=1)
        fig.colorbar(pred_vis, ax=ax_row[2], fraction=0.046, pad=0.04)
        ax_row[2].set_title(f'Prediction (sum={pred_np.sum():.1f})')
        ax_row[2].axis('off')
        
        # 二值化预测结果
        pred_bin = (pred_np > 0.5).astype(np.float32)
        ax_row[3].imshow(pred_bin, cmap='gray')
        ax_row[3].set_title(f'Binary Pred (sum={pred_bin.sum():.1f})')
        ax_row[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'prediction_epoch_{epoch}.png')
    plt.close()

# ====================== 训练示例 ======================
def train_vit_seg():
    # 1) 数据准备
    img_dir = "data/annotation_images"
    mask_dir= "data/mask"
    
    # 检查目录是否存在
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"图像目录不存在: {img_dir}")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"掩码目录不存在: {mask_dir}")
    
    # 获取所有图像文件
    img_files = sorted([f for f in os.listdir(img_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    total_len = len(img_files)
    print(f"总共有 {total_len} 张图像.")
    
    # 检查掩码文件是否存在
    for img_file in img_files:
        base_name = img_file.rsplit('.', 1)[0]
        mask_path = os.path.join(mask_dir, base_name + "_mask.png")
        if not os.path.exists(mask_path):
            print(f"警告: 找不到对应的掩码文件: {mask_path}")
    
    # 若 total_len=18, 可以 12/3/3 划分
    train_size = 12
    val_size   = 3
    test_size  = total_len - train_size - val_size
    
    # 创建索引分割
    indices = list(range(total_len))
    torch.manual_seed(42)  # 设置随机种子以确保可重复性
    indices = torch.randperm(total_len).tolist()
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    print(f"训练集索引: {train_indices}")
    print(f"验证集索引: {val_indices}")
    print(f"测试集索引: {test_indices}")
    
    # 为训练集和验证集创建不同的数据集实例
    train_dataset = TextSegDataset(img_dir, mask_dir, transform=train_transform)
    val_dataset = TextSegDataset(img_dir, mask_dir, transform=val_transform)
    
    # 创建子集
    train_ds = Subset(train_dataset, train_indices)
    val_ds = Subset(val_dataset, val_indices)
    test_ds = Subset(val_dataset, test_indices)
    
    # 构造DataLoader
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=2)
    
    # 检查掩码
    print("\n检查数据集掩码:")
    check_masks(train_loader, "训练")
    check_masks(val_loader, "验证")
    check_masks(test_loader, "测试")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 2) 模型 (使用UNet替代ViT)
    # model = SmallViT(
    #     image_size=224, patch_size=16,
    #     embed_dim=128, num_heads=4, depth=4, num_classes=1
    # ).to(device)
    model = SimpleUNet().to(device)
    print(f"模型结构:\n{model}")
    
    # 3) 损失函数 & 优化器
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 4) 训练循环
    epochs = 10  # 减少训练轮数便于调试
    for epoch in range(1, epochs+1):
        print(f"\n开始Epoch {epoch}/{epochs}")
        
        # 训练阶段
        model.train()
        total_train_loss = 0
        batch_count = 0
        
        for batch_idx, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            
            # 每个epoch的第一个批次打印形状
            if batch_idx == 0:
                print(f"训练批次 - 图像形状: {imgs.shape}, 掩码形状: {masks.shape}")
                print(f"掩码唯一值: {torch.unique(masks).cpu().numpy()}")
            
            optimizer.zero_grad()
            logits = model(imgs)  # [B,1,H,W]
            
            # 检查输出
            if batch_idx == 0:
                print(f"模型输出形状: {logits.shape}")
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    print(f"Sigmoid后值范围: [{probs.min().item()}, {probs.max().item()}]")
            
            # 计算损失
            loss = bce_loss(logits, masks) + dice_loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            batch_count += 1
            
            if batch_idx % 5 == 0:
                print(f"  批次 {batch_idx} - 损失: {loss.item():.4f}")
        
        avg_train_loss = total_train_loss / batch_count
        
        # 验证阶段
        model.eval()
        total_val_iou = 0
        total_val_dice= 0
        count = 0
        
        print("\n开始验证...")
        debug_validation(val_loader, model, device, epoch)
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                probs  = torch.sigmoid(logits)
                
                # IoU
                pred_bin = (probs > 0.5).float()
                intersect = (pred_bin * masks).sum(dim=(1,2,3))
                union = ((pred_bin + masks) > 0).float().sum(dim=(1,2,3)) + 1e-6
                iou = (intersect / union).mean().item()
                
                # Dice
                dice_num = 2*intersect
                dice_den = pred_bin.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3)) + 1e-6
                dice = (dice_num/dice_den).mean().item()
                
                total_val_iou += iou
                total_val_dice+= dice
                count += 1
        
        avg_val_iou = total_val_iou / max(1, count)
        avg_val_dice= total_val_dice/ max(1, count)
        
        print(f"Epoch [{epoch}/{epochs}] - "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val IoU: {avg_val_iou:.4f}, Val Dice: {avg_val_dice:.4f}")
    
    # 5) 测试
    print("\n开始测试...")
    model.eval()
    total_test_iou = 0
    total_test_dice= 0
    count = 0
    
    # 调试测试集
    debug_validation(test_loader, model, device, epoch=999)
    
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            probs  = torch.sigmoid(logits)
            pred_bin = (probs > 0.5).float()
            
            intersect = (pred_bin * masks).sum(dim=(1,2,3))
            union = ((pred_bin + masks) > 0).float().sum(dim=(1,2,3)) + 1e-6
            iou = (intersect/union).mean().item()
            
            dice_num = 2*intersect
            dice_den = pred_bin.sum(dim=(1,2,3)) + masks.sum(dim=(1,2,3)) + 1e-6
            dice = (dice_num/dice_den).mean().item()
            
            total_test_iou  += iou
            total_test_dice += dice
            count += 1
    
    avg_test_iou = total_test_iou / max(1, count)
    avg_test_dice = total_test_dice / max(1, count)
    print(f"\n[测试结果] IoU={avg_test_iou:.4f}, Dice={avg_test_dice:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), 'segmentation_model.pth')
    print("模型已保存到 segmentation_model.pth")

if __name__ == "__main__":
    # 设置随机种子确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 将调试信息保存到文件
    import sys
    log_file = open('training_log.txt', 'w')
    
    # 复制标准输出到文件
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    
    # 将标准输出重定向到控制台和日志文件
    sys.stdout = Tee(sys.stdout, log_file)
    
    try:
        train_vit_seg()
    except Exception as e:
        import traceback
        print(f"训练过程中出现错误: {e}")
        traceback.print_exc()
    finally:
        log_file.close()
        # 恢复标准输出
        sys.stdout = sys.__stdout__