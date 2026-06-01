import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import numpy as np

# 1. 硬件配置与随机种子设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 2. 自定义残差块与 ResNet 架构
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        return out

class ImpressionistResNet(nn.Module):
    def __init__(self, block, num_classes=10):
        super(ImpressionistResNet, self).__init__()
        self.in_channels = 32
        
        # 初始输入层
        self.conv = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 残差阶段层
        self.layer1 = self._make_layer(block, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks=2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类头
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def build_model(num_classes=10):
    return ImpressionistResNet(ResidualBlock, num_classes)

# 3. 主训练流程
def main():
    # 数据集路径
    train_dir = 'd:/ML_Middle/Impressionist_Classifier_Data/training/training'
    val_dir = 'd:/ML_Middle/Impressionist_Classifier_Data/validation/validation'
    
    # 图像预处理与增强
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    classes = train_dataset.classes
    num_classes = len(classes)
    print(f"Classes: {classes}")
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    # 模型、损失函数、优化器与学习率调度器
    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    num_epochs = 15
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # 指标记录
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "epoch_time": []
    }
    
    best_val_acc = 0.0
    
    print("Starting training...")
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            total_train += inputs.size(0)
            
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = running_corrects / total_train
        
        # 验证阶段
        model.eval()
        running_val_loss = 0.0
        running_val_corrects = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_val_corrects += torch.sum(preds == labels.data).item()
                total_val += inputs.size(0)
                
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = running_val_corrects / total_val
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # 保存记录
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["epoch_time"].append(epoch_time)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Time: {epoch_time:.1f}s | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
              
        # 保存最佳模型
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), 'd:/ML_Middle/model.pth')
            print(f"  --> Saved new best model weights (Val Acc: {best_val_acc:.4f})")

    # 4. 加载最佳模型进行最终评估与混淆矩阵计算
    print("Training finished. Evaluating best model...")
    model.load_state_dict(torch.load('d:/ML_Middle/model.pth'))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 类别准确率
    class_acc = {}
    for i, class_name in enumerate(classes):
        class_total = np.sum(cm[i, :])
        class_correct = cm[i, i]
        class_acc[class_name] = float(class_correct / class_total) if class_total > 0 else 0.0
        
    # 保存结果到 metrics.json
    metrics_data = {
        "train_loss": history["train_loss"],
        "train_acc": history["train_acc"],
        "val_loss": history["val_loss"],
        "val_acc": history["val_acc"],
        "epoch_time": history["epoch_time"],
        "classes": classes,
        "confusion_matrix": cm.tolist(),
        "class_accuracy": class_acc,
        "best_val_accuracy": float(best_val_acc)
    }
    
    with open('d:/ML_Middle/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, ensure_ascii=False, indent=4)
        
    print("Metrics and Confusion Matrix exported to 'd:/ML_Middle/metrics.json'")

if __name__ == '__main__':
    main()
