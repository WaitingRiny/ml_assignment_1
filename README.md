# 🎨 Canvas ResNet: 基于深度残差神经网络的印象派绘画分类器

本项目为同济大学机器学习课程实践作业。项目独立使用 PyTorch 从零构建轻量级深度残差神经网络（ResNet），对 10 位印象派及后印象派大师的画作进行多分类，并使用 Grad-CAM 算法实现特征热力图可视化，破译神经网络的“机器视觉指纹”。

项目包含完整的**代码实现**、**Markdown 学术报告**以及一个高交互性的**数字展馆式滚动叙事（Scrollytelling）网页报告**。

---

## 🔗 在线预览

✨ **项目在线学术展厅**： [点击访问在线展示网页](https://waitingriny.github.io/ml_assignment_1/web/index.html)  

---

## 🛠️ 项目特色

1. **底层算法使用 PyTorch 构建**：拒绝引入预训练黑盒，从残差块（`Skip Connection`）、输入层（`Stem`）到分类头均自主构建实现。
2. **科学的参数调优与验证**：使用 AdamW 优化器、余弦退火学习率调度，配合高强度数据增强（旋转、色彩抖动、水平翻转等），最优验证准确率达到 **45.25%**（远超 10% 随机基准）。
3. **模型可解释性分析 (Grad-CAM)**：计算最后一层卷积特征图的通道权重，生成二维类激活映射热力图，科学探寻模型对不同画家“视觉指纹”（如梵高的螺线厚涂、莫奈的水中倒影）的关注焦点。
4. **数字展馆式网页交互**：采用 GSAP 精制滚动视差，以高雅的“艺术展墙”为视觉载体，配有 Chart.js 训练曲线图、交互式混淆矩阵及 10 类大师 Grad-CAM 对照画廊。

---

## 📂 目录结构

```
.
├── web/                           # 在线交互网页系统
│   ├── index.html                 # 展厅主骨架 (HTML5)
│   ├── style.css                  # 展厅艺术样式表 (CSS)
│   ├── app.js                     # 交互与 Chart.js/GSAP 动画控制 (JS)
│   └── assets/                    # 大师画作样本及 Grad-CAM 热力图资源
├── HarmonyOS_Sans_SC/             # 本地 HarmonyOS Sans 字体包
├── train.py                       # CNN 残差网络训练脚本
├── gradcam.py                     # Grad-CAM 类激活图计算与叠加脚本
├── metrics.json                   # 15 个 Epoch 的 Loss/Acc 训练过程数据
├── report.md                      # 书面课程设计成果学术报告
├── README.md                      # 本说明文档
└── .gitignore                     # Git 忽略配置
```

---

## 🚀 本地快速启动

### 1. 环境准备
项目依赖 PyTorch、Torchvision、Scikit-learn、Matplotlib 等，您可以使用以下命令快速安装环境：
```bash
pip install torch torchvision scikit-learn numpy matplotlib
```

### 2. 运行本地 Web 服务
由于网页需要通过 Fetch API 异步加载本地的 `metrics.json` 数据并执行渲染，直接双击打开 HTML 可能会因为跨域安全策略（CORS）被阻拦。建议使用 Python 启动本地微服务：
```bash
python -m http.server 8011
```
启动后在浏览器中输入 `http://127.0.0.1:8011/web/index.html` 即可畅爽体验。
