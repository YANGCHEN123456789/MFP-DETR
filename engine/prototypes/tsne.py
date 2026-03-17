################    取平均 ###################
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# 目标可视化类别
target_labels = ["bj_bpmh", "bj_bpps", "bj_wkps", "hxq_gjtps", "ywzt_yfyc", "yw_gkxfw", "yw_nc","hxq_gjbs"]

# 加载两个原型文件
class_prototypes_file = [
    '/input/yangchen/torch_learn/project/yc_graduate/DEIMv2/prototypes/output_features/vitt_distill_features.p10.clustered.pkl',
    
]

# 加载并拼接原型和标签
dct1 = torch.load(class_prototypes_file[0])
label_names = dct1['label_names']   # list of str

# Step 1: 求每类平均原型并归一化
class_weights = F.normalize(prototypes.mean(dim=1), dim=-1)  # [N, D]

# Step 2: 根据目标标签筛选原型和名称
filtered_weights = []
filtered_labels = []
for i, label in enumerate(label_names):
    if label in target_labels:
        filtered_weights.append(class_weights[i].unsqueeze(0))
        filtered_labels.append(label)

filtered_weights = torch.cat(filtered_weights, dim=0)  # [K, D]
filtered_weights_np = filtered_weights.cpu().numpy()

# Step 3: t-SNE 降维
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
weights_2d = tsne.fit_transform(filtered_weights_np)

# Step 4: 可视化
plt.figure(figsize=(10, 8))
colors = plt.cm.get_cmap('tab10', len(filtered_labels))

for i, (x, y) in enumerate(weights_2d):
    plt.scatter(x, y, color=colors(i), label=filtered_labels[i], s=100)
    plt.text(x + 0.5, y, filtered_labels[i], fontsize=9)

plt.title("Selected Class Prototypes t-SNE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("selected_class_weights_tsne.png", dpi=300)
plt.show()