import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import json

# 导入网络定义与模型构建函数
from train import build_model, ResidualBlock

# Vectorized Jet Colormap for heatmap generation (no dependencies on matplotlib or opencv)
def apply_jet_colormap(cam):
    """
    将 [0, 1] 的单通道热力图转换为 RGB 格式的 Jet 伪彩色图。
    """
    # 巧妙的数学剪裁，模拟 Jet 渐变色效果
    r = np.clip(4.0 * cam - 1.5, 0.0, 1.0)
    g = np.clip(4.0 * cam - 0.5, 0.0, 1.0) * np.clip(2.5 - 4.0 * cam, 0.0, 1.0)
    b = np.clip(1.5 - 4.0 * cam, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册 hooks 获取特征图与梯度
        self.forward_hook = self.target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        score = output[0, class_idx]
        score.backward()
        
        # 获取梯度和特征图
        gradients = self.gradients.cpu().data.numpy()[0]     # [C, H, W]
        activations = self.activations.cpu().data.numpy()[0] # [C, H, W]
        
        # 计算通道权重：梯度的全局空间平均 (GAP)
        weights = gradients.mean(axis=(1, 2)) # [C]
        
        # 特征图按权重加权求和
        cam = np.zeros(activations.shape[1:], dtype=np.float32) # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        # 经过 ReLU 激活函数，只保留对目标分类有正向贡献的区域
        cam = np.maximum(cam, 0)
        
        # 归一化到 [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam, class_idx

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Grad-CAM running on device: {device}")
    
    # 1. 载入模型与权重
    model = build_model(num_classes=10)
    model_path = 'd:/ML_Middle/model.pth'
    if not os.path.exists(model_path):
        print(f"Error: Weights file {model_path} not found. Please train the model first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 选择最后一层卷积层作为 Grad-CAM 的目标层
    # 对应于 layer4（最后的 Residual Block）中最后一个 Block 内部的最后一个卷积层
    target_layer = model.layer4[-1].conv2
    
    # 初始化 GradCAM
    grad_cam = GradCAM(model, target_layer)
    
    # 2. 从验证集读取 10 个类别的首张图片进行分析
    val_dir = 'd:/ML_Middle/Impressionist_Classifier_Data/validation/validation'
    classes = sorted(os.listdir(val_dir))
    
    # 输出资源文件夹
    output_dir = 'd:/ML_Middle/web/assets/gradcam'
    os.makedirs(output_dir, exist_ok=True)
    
    # 图像预处理变换
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    gradcam_results = {}
    
    for artist in classes:
        artist_dir = os.path.join(val_dir, artist)
        if not os.path.isdir(artist_dir):
            continue
            
        images = [f for f in os.listdir(artist_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            continue
            
        img_name = images[0]
        img_path = os.path.join(artist_dir, img_name)
        
        # 读取并预处理原图
        orig_img = Image.open(img_path).convert("RGB")
        # 调整大小为 224x224 以便进行可视化比对
        orig_img_resized = orig_img.resize((224, 224))
        
        input_tensor = preprocess(orig_img).unsqueeze(0).to(device)
        
        # 获取该类别的索引
        class_idx = classes.index(artist)
        
        # 运行 Grad-CAM 提取热力图
        cam, pred_idx = grad_cam(input_tensor, class_idx=class_idx)
        
        # 将 cam 从 7x7 插值缩放到 224x224 尺寸
        cam_pil = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.Resampling.BILINEAR)
        cam_resized = np.array(cam_pil, dtype=np.float32) / 255.0
        
        # 生成彩色热力图
        heatmap = apply_jet_colormap(cam_resized) # [224, 224, 3] in [0, 1]
        
        # 将热力图叠加在缩放后的原图上
        orig_np = np.array(orig_img_resized, dtype=np.float32) / 255.0 # [224, 224, 3]
        overlay = 0.5 * orig_np + 0.5 * heatmap # 混合权重 0.5
        overlay = np.clip(overlay, 0.0, 1.0)
        
        # 转换回 PIL Image 以便保存
        overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
        
        # 拼接原图和热力图（侧边并排展示）
        # 创建一张 448x224 的新画布
        combined_img = Image.new("RGB", (448, 224))
        combined_img.paste(orig_img_resized, (0, 0))
        combined_img.paste(overlay_img, (224, 0))
        
        # 保存到本地网页资源目录
        out_filename = f"{artist}_gradcam.jpg"
        combined_img.save(os.path.join(output_dir, out_filename), "JPEG", quality=90)
        
        # 记录预测情况
        pred_artist = classes[pred_idx]
        gradcam_results[artist] = {
            "image": out_filename,
            "predicted": pred_artist,
            "is_correct": bool(pred_idx == class_idx)
        }
        print(f"Generated Grad-CAM for {artist}: Predicts {pred_artist} (Correct: {pred_idx == class_idx})")
        
    # 注销 Hooks
    grad_cam.remove_hooks()
    
    # 写入记录用于网页端读取
    with open(os.path.join(output_dir, 'gradcam_info.json'), 'w', encoding='utf-8') as f:
        json.dump(gradcam_results, f, ensure_ascii=False, indent=4)
        
    print("Grad-CAM results generated successfully!")

if __name__ == '__main__':
    main()
