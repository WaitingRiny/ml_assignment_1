// ==========================================================================
// 🎨 全局状态与默认模拟数据 (二次优化版)
// ==========================================================================

const classes = ["Cezanne", "Degas", "Gauguin", "Hassam", "Matisse", "Monet", "Pissarro", "Renoir", "Sargent", "VanGogh"];

// 真实实验数据作为缺省模拟数据，保障本地无网络打开时数据100%匹配一致
const mockMetrics = {
    "train_loss": [
        2.2010, 2.0870, 2.0282, 1.9639, 1.9182, 1.8649, 1.8333, 1.7968, 1.7453, 1.7123, 1.6618, 1.6307, 1.5774, 1.5349, 1.5148
    ],
    "train_acc": [
        0.1815, 0.2455, 0.2523, 0.2851, 0.3107, 0.3260, 0.3463, 0.3596, 0.3726, 0.3826, 0.4085, 0.4315, 0.4426, 0.4594, 0.4616
    ],
    "val_loss": [
        2.2442, 2.1573, 1.9827, 2.0125, 1.8681, 1.8907, 1.9786, 1.9860, 2.0349, 1.7815, 1.7103, 1.6475, 1.5889, 1.5655, 1.5479
    ],
    "val_acc": [
        0.1798, 0.2263, 0.2899, 0.2798, 0.3111, 0.3354, 0.3071, 0.2949, 0.3111, 0.3606, 0.3828, 0.4172, 0.4374, 0.4525, 0.4485
    ],
    "best_val_accuracy": 0.4525,
    "confusion_matrix": [
        [41, 9, 13, 2, 8, 0, 5, 4, 13, 4],
        [6, 14, 11, 0, 6, 3, 5, 28, 25, 1],
        [4, 5, 33, 0, 9, 0, 16, 26, 2, 4],
        [2, 2, 2, 30, 2, 23, 15, 4, 5, 14],
        [10, 4, 4, 1, 65, 5, 0, 5, 5, 0],
        [1, 1, 3, 11, 4, 49, 11, 4, 1, 14],
        [0, 3, 9, 10, 0, 8, 56, 5, 5, 3],
        [4, 5, 7, 0, 2, 6, 6, 67, 2, 0],
        [8, 8, 2, 1, 10, 3, 5, 6, 47, 9],
        [7, 2, 6, 7, 5, 4, 14, 0, 8, 46]
    ],
    "class_accuracy": {
        "Cezanne": 0.414, "Degas": 0.141, "Gauguin": 0.333, "Hassam": 0.303, "Matisse": 0.657,
        "Monet": 0.495, "Pissarro": 0.566, "Renoir": 0.677, "Sargent": 0.475, "VanGogh": 0.465
    }
};

// ==========================================================================
// 📈 图表渲染与混淆矩阵构建 (Data Visualization - 陶棕加深)
// ==========================================================================

let metricsChart = null;

function renderChart(data) {
    const ctx = document.getElementById('metrics-chart').getContext('2d');
    
    if (metricsChart) {
        metricsChart.destroy();
    }
    
    const epochs = Array.from({length: data.train_loss.length}, (_, i) => i + 1);
    
    metricsChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                {
                    label: '训练集 Loss',
                    data: data.train_loss,
                    borderColor: '#8C7B6B',      // 灰棕色
                    backgroundColor: 'rgba(140, 123, 107, 0.05)',
                    borderWidth: 2,
                    pointRadius: 2,
                    tension: 0.35,
                    yAxisID: 'y-loss'
                },
                {
                    label: '验证集 Loss',
                    data: data.val_loss,
                    borderColor: 'rgba(140, 123, 107, 0.4)',
                    borderWidth: 1.5,
                    borderDash: [4, 4],
                    pointRadius: 1,
                    tension: 0.35,
                    yAxisID: 'y-loss'
                },
                {
                    label: '训练集 准确率',
                    data: data.train_acc,
                    borderColor: 'rgba(147, 95, 62, 0.4)', // 陶棕色过渡
                    borderWidth: 1.5,
                    borderDash: [3, 3],
                    pointRadius: 1,
                    tension: 0.35,
                    yAxisID: 'y-acc'
                },
                {
                    label: '验证集 准确率',
                    data: data.val_acc,
                    borderColor: '#935F3E',      // 深陶棕色
                    backgroundColor: 'rgba(147, 95, 62, 0.08)',
                    borderWidth: 2.5,
                    pointRadius: 3,
                    pointBackgroundColor: '#935F3E',
                    tension: 0.35,
                    yAxisID: 'y-acc'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    align: 'center',
                    labels: {
                        color: '#6D5B4F',        // 深棕色字体
                        font: { family: 'HarmonyOS Sans SC', size: 10, weight: 'bold' },
                        // 图例指示形状改为线条 (PointStyle = Line, BoxWidth = 20)
                        usePointStyle: true,
                        pointStyle: 'line',
                        boxWidth: 20,
                        padding: 15
                    }
                }
            },
            scales: {
                'y-loss': {
                    type: 'linear',
                    position: 'left',
                    grid: { color: 'rgba(109, 91, 79, 0.07)' },
                    ticks: { color: '#6D5B4F', font: { family: 'HarmonyOS Sans SC' } },
                    title: { display: true, text: 'Loss', color: '#6D5B4F', font: { family: 'HarmonyOS Sans SC', weight: 'bold' } }
                },
                'y-acc': {
                    type: 'linear',
                    position: 'right',
                    grid: { drawOnChartArea: false },
                    ticks: {
                        color: '#6D5B4F',
                        font: { family: 'HarmonyOS Sans SC' },
                        callback: function(value) { return (value * 100).toFixed(0) + '%'; }
                    },
                    title: { display: true, text: '准确率', color: '#6D5B4F', font: { family: 'HarmonyOS Sans SC', weight: 'bold' } }
                },
                x: {
                    grid: { color: 'rgba(109, 91, 79, 0.07)' },
                    ticks: { color: '#6D5B4F', font: { family: 'HarmonyOS Sans SC' } },
                    title: { display: true, text: 'Epoch', color: '#6D5B4F', font: { family: 'HarmonyOS Sans SC', weight: 'bold' } }
                }
            }
        }
    });
}

function renderConfusionMatrix(matrix) {
    const container = document.getElementById('cm-grid-container');
    container.innerHTML = '';
    
    // 1. 绘制列标题 (Column Headers)
    const emptyCorner = document.createElement('div');
    emptyCorner.className = 'cm-header-cell';
    container.appendChild(emptyCorner);
    
    classes.forEach(c => {
        const cell = document.createElement('div');
        cell.className = 'cm-header-cell';
        cell.innerText = c.substring(0, 4);
        cell.setAttribute('title', c);
        container.appendChild(cell);
    });
    
    // 2. 逐行绘制矩阵
    for (let r = 0; r < 10; r++) {
        const rowLabel = document.createElement('div');
        rowLabel.className = 'cm-header-cell y-label';
        rowLabel.innerText = classes[r];
        container.appendChild(rowLabel);
        
        const rowSum = matrix[r].reduce((a, b) => a + b, 0);
        
        for (let c = 0; c < 10; c++) {
            const rawVal = matrix[r][c];
            const pct = rowSum > 0 ? (rawVal / rowSum) : 0;
            
            const cell = document.createElement('div');
            cell.className = 'cm-cell';
            cell.innerText = rawVal;
            
            // 使用加深后的陶棕色 (#935F3E, RGB: 147, 95, 62)
            cell.style.backgroundColor = `rgba(147, 95, 62, ${pct * 0.95 + 0.05})`;
            
            // 对比度优化
            if (pct > 0.4) {
                cell.style.color = '#F5E6D8'; // 高亮块使用米白色字
            } else {
                cell.style.color = '#6D5B4F'; // 浅色块使用深棕色字
            }
            
            const tooltipStr = `真实: ${classes[r]} | 预测: ${classes[c]} | 数量: ${rawVal} (${(pct * 100).toFixed(1)}%)`;
            cell.setAttribute('data-tooltip', tooltipStr);
            
            container.appendChild(cell);
        }
    }
}

// ==========================================================================
// 📊 导航栏滚动监听 (Intersection Observer)
// ==========================================================================

function initNavbarHighlighting() {
    const sections = document.querySelectorAll('.slide-section');
    const navLinks = document.querySelectorAll('.nav-links a');
    
    const options = {
        root: null,
        rootMargin: '-30% 0px -50% 0px',
        threshold: 0
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const id = entry.target.getAttribute('id');
                navLinks.forEach(link => {
                    if (link.getAttribute('href') === `#${id}`) {
                        link.classList.add('active');
                    } else {
                        link.classList.remove('active');
                    }
                });
            }
        });
    }, options);
    
    sections.forEach(sec => observer.observe(sec));
}



// ==========================================================================
// 🚀 GSAP 滚动动效集成（大批量的视觉动效升级）
// ==========================================================================

function initScrollAnimations() {
    gsap.registerPlugin(ScrollTrigger);
    
    // 1. 背景画作碎片视差（Parallax Scroll）与微幅偏转漂移
    gsap.utils.toArray('.art-shard').forEach((shard, idx) => {
        const yVal = (idx % 2 === 0 ? -120 : 120);
        const rotVal = (idx % 2 === 0 ? 12 : -12);
        
        gsap.to(shard, {
            y: yVal,
            rotation: rotVal,
            scrollTrigger: {
                trigger: shard.closest('section'),
                start: 'top bottom',
                end: 'bottom top',
                scrub: 1.5
            }
        });
    });
    
    // 2. 章节头部数字和主标题的入场平滑滑入淡入（带弹性）
    gsap.utils.toArray('.slide-section').forEach(section => {
        const num = section.querySelector('.section-num');
        const title = section.querySelector('.section-header h2');
        if (num && title) {
            gsap.from([num, title], {
                x: -70,
                opacity: 0,
                duration: 1.1,
                stagger: 0.18,
                ease: 'power3.out',
                scrollTrigger: {
                    trigger: section,
                    start: 'top 80%',
                    toggleActions: 'play none none none'
                }
            });
        }
    });
    
    // 3. 常规滚动模式下的卡片分层向上滑入（交错 Stagger 机制）
    gsap.utils.toArray('.slide-section').forEach(section => {
        const cards = section.querySelectorAll('.card');
        if (cards.length > 0) {
            gsap.from(cards, {
                y: 70,
                opacity: 0,
                duration: 1.15,
                stagger: 0.28,
                ease: 'power4.out',
                scrollTrigger: {
                    trigger: section,
                    start: 'top 78%',
                    toggleActions: 'play none none none'
                }
            });
        }
    });
    
    // 4. 第一幕：目录树条目的阶梯式序列浮现
    gsap.from('.tree-item', {
        x: -35,
        opacity: 0,
        duration: 0.7,
        stagger: 0.12,
        ease: 'power2.out',
        scrollTrigger: {
            trigger: '.directory-tree',
            start: 'top 85%'
        }
    });
    
    // 5. 第一幕：10 类大师小图卡片缩放弹出（Back 缓动）
    gsap.from('.sample-item', {
        scale: 0.4,
        opacity: 0,
        duration: 0.75,
        stagger: 0.06,
        ease: 'back.out(1.5)',
        scrollTrigger: {
            trigger: '.samples-grid',
            start: 'top 85%'
        }
    });
    
    // 6. 第二幕：模型拓扑连线与节点的自上而下阶梯式生长
    gsap.from('.net-node, .net-arrow', {
        scale: 0.8,
        y: -15,
        opacity: 0,
        duration: 0.55,
        stagger: 0.08,
        ease: 'back.out(1.8)',
        scrollTrigger: {
            trigger: '.net-diagram',
            start: 'top 82%'
        }
    });
    
    // 7. 第六幕：视觉焦点大师画廊平铺卡片交错飞入
    gsap.from('.gallery-card', {
        y: 60,
        scale: 0.94,
        opacity: 0,
        duration: 1.05,
        stagger: 0.18,
        ease: 'power3.out',
        scrollTrigger: {
            trigger: '#gradcam-gallery',
            start: 'top 85%'
        }
    });
}

// ==========================================================================
// 📥 载入实验数据 (Init & Fetch)
// ==========================================================================

async function loadExperimentData() {
    try {
        const response = await fetch('../metrics.json');
        if (!response.ok) {
            throw new Error("Could not find metrics.json, falling back to mock data.");
        }
        
        const data = await response.json();
        console.log("Successfully loaded experiment metrics.json:", data);
        
        if (data.best_val_accuracy) {
            document.getElementById('best-accuracy-text').innerText = (data.best_val_accuracy * 100).toFixed(1) + '%';
        }
        
        renderChart(data);
        renderConfusionMatrix(data.confusion_matrix);
        
        // 动态 DOM 生成后，强行刷新 ScrollTrigger 的滚动高度计算
        if (typeof ScrollTrigger !== 'undefined') {
            ScrollTrigger.refresh();
        }
        
    } catch (err) {
        console.warn(err.message);
        renderChart(mockMetrics);
        renderConfusionMatrix(mockMetrics.confusion_matrix);
        
        if (typeof ScrollTrigger !== 'undefined') {
            ScrollTrigger.refresh();
        }
    }
}

// ==========================================================================
// 🎨 背景画作碎片鼠标悬停动画 (GSAP hover effect - 减慢过渡速度)
// ==========================================================================

function initShardHoverAnimations() {
    gsap.utils.toArray('.art-shard').forEach(shard => {
        const isHero = shard.closest('#hero') !== null;
        const baseOpacity = isHero ? 0.22 : 0.12;
        
        shard.addEventListener('mouseenter', () => {
            gsap.to(shard, {
                scale: 1.08,
                opacity: 0.9,
                filter: 'sepia(0%) contrast(100%)',
                duration: 0.8, /* 延长动画时长至 0.8s 保持从容大师感 */
                overwrite: 'auto',
                ease: 'power3.out'
            });
        });
        
        shard.addEventListener('mouseleave', () => {
            gsap.to(shard, {
                scale: 1.0,
                opacity: baseOpacity,
                filter: 'sepia(12%) contrast(100%)',
                duration: 0.8, /* 离开动画也慢速还原 */
                overwrite: 'auto',
                ease: 'power3.out'
            });
        });
    });
}

// ==========================================================================
// 🏁 初始化入口 (Bootstrap)
// ==========================================================================

window.addEventListener('DOMContentLoaded', () => {
    loadExperimentData();
    initNavbarHighlighting();
    initScrollAnimations();
    initShardHoverAnimations();
    
    // 监听画廊图片加载状态，实时刷新 ScrollTrigger 以纠正滚动高度
    const galleryImgs = document.querySelectorAll('.gallery-img-box img');
    let imgsLoaded = 0;
    galleryImgs.forEach(img => {
        const onImgLoad = () => {
            imgsLoaded++;
            if (imgsLoaded === galleryImgs.length && typeof ScrollTrigger !== 'undefined') {
                console.log("All gallery images loaded, refreshing ScrollTrigger...");
                ScrollTrigger.refresh();
            }
        };
        if (img.complete) {
            onImgLoad();
        } else {
            img.addEventListener('load', onImgLoad);
            img.addEventListener('error', onImgLoad);
        }
    });
    
    window.addEventListener('resize', () => {
        if (metricsChart) {
            metricsChart.resize();
        }
    });
});

// 当页面及所有资源（字体、图像）完全下载完毕后，再次强制刷新 ScrollTrigger
window.addEventListener('load', () => {
    if (typeof ScrollTrigger !== 'undefined') {
        console.log("Global window load event fired, final ScrollTrigger refresh.");
        ScrollTrigger.refresh();
    }
});
