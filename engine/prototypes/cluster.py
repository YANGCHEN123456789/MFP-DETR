import torch
import numpy as np
import os.path as osp
from pprint import pprint
from tqdm.auto import tqdm

# ========== 改进版原型学习器 ==========
class ImprovedPrototypeLearner:
    """
    改进的原型学习器
    - 更好的初始化策略（Xavier + K-means++）
    - 温度缩放的预测
    - 自适应动量更新
    """
    def __init__(self, num_classes, num_prototypes, embed_dim, 
                 momentum=0.1, 
                 normalize=True, 
                 temperature=0.07, 
                 device='cpu'):
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.embed_dim = embed_dim
        self.momentum = momentum
        self.normalize = normalize
        self.temperature = temperature
        self.device = device
        
        # 使用 Xavier 初始化
        self.prototypes = torch.empty(num_classes * num_prototypes, embed_dim)
        torch.nn.init.xavier_uniform_(self.prototypes)
        
        if normalize:
            self.prototypes = torch.nn.functional.normalize(self.prototypes, dim=1)
        self.prototypes = self.prototypes.to(device)
        
        # 记录每个原型的更新次数
        self.update_counts = torch.zeros(num_classes * num_prototypes).to(device)
    
    def update(self, tokens, labels):
        """改进的更新策略"""
        if self.normalize:
            tokens = torch.nn.functional.normalize(tokens, dim=1)
        
        for cls_id in torch.unique(labels):
            mask = labels == cls_id
            cls_tokens = tokens[mask]
            
            if len(cls_tokens) == 0:
                continue
            
            start_idx = int(cls_id * self.num_prototypes)
            end_idx = start_idx + self.num_prototypes
            
            # K-means++ 风格的初始化（仅在前10轮）
            if self.update_counts[start_idx] < 10:
                if len(cls_tokens) >= self.num_prototypes:
                    indices = torch.randperm(len(cls_tokens))[:self.num_prototypes]
                    self.prototypes[start_idx:end_idx] = cls_tokens[indices]
                else:
                    # 样本不足时重复采样
                    self.prototypes[start_idx:end_idx] = cls_tokens[
                        torch.randint(0, len(cls_tokens), (self.num_prototypes,))
                    ]
            else:
                # 正常的动量更新
                similarities = torch.mm(cls_tokens, self.prototypes[start_idx:end_idx].T)
                assignments = similarities.argmax(dim=1)
                
                # 为每个原型更新
                for proto_idx in range(self.num_prototypes):
                    assigned_tokens = cls_tokens[assignments == proto_idx]
                    if len(assigned_tokens) > 0:
                        new_proto = assigned_tokens.mean(dim=0)
                        global_idx = start_idx + proto_idx
                        self.prototypes[global_idx] = (
                            (1 - self.momentum) * self.prototypes[global_idx] + 
                            self.momentum * new_proto
                        )
            
            # 更新计数器
            self.update_counts[start_idx:end_idx] += 1
            
            # 重新归一化
            if self.normalize:
                self.prototypes[start_idx:end_idx] = torch.nn.functional.normalize(
                    self.prototypes[start_idx:end_idx], dim=1
                )
    
    def predict(self, tokens):
        """改进的预测方法（使用温度缩放）"""
        tokens = tokens.to(self.prototypes.device)
        
        if self.normalize:
            tokens = torch.nn.functional.normalize(tokens, dim=1)
        
        # 计算相似度并应用温度缩放
        similarities = torch.mm(tokens, self.prototypes.T) / self.temperature
        
        # reshape 为 [N, num_classes, num_prototypes]
        similarities = similarities.view(-1, self.num_classes, self.num_prototypes)
        
        # 使用 logsumexp 聚合每个类别的原型分数
        class_scores = similarities.logsumexp(dim=2)  # [N, num_classes]
        pred_labels = class_scores.argmax(dim=1)
        
        return pred_labels
    
    def get_prototypes(self):
        """返回原型矩阵 [num_classes, num_prototypes, embed_dim]"""
        return self.prototypes.view(self.num_classes, self.num_prototypes, self.embed_dim)


# ========== 特征诊断工具 ==========
def diagnose_features(tokens, labels):
    """
    诊断特征质量
    - 检查 NaN/Inf
    - 统计特征分布
    - 计算类内/类间距离
    """
    print("\n" + "="*60)
    print("特征质量诊断")
    print("="*60)
    
    # 1. 检查数值问题
    has_nan = torch.isnan(tokens).any()
    has_inf = torch.isinf(tokens).any()
    print(f"包含 NaN: {has_nan}")
    print(f"包含 Inf: {has_inf}")
    
    if has_nan or has_inf:
        print("警告: 特征包含非法值，建议检查特征提取流程！")
    
    # 2. 特征分布统计
    print(f"\n特征统计:")
    print(f"  均值: {tokens.mean():.4f}")
    print(f"  标准差: {tokens.std():.4f}")
    print(f"  最小值: {tokens.min():.4f}")
    print(f"  最大值: {tokens.max():.4f}")
    
    # 3. 类内/类间距离分析
    print(f"\n类内/类间距离分析 (前5类):")
    num_classes = len(torch.unique(labels))
    
    for cls_id in range(min(5, num_classes)):
        mask = labels == cls_id
        cls_tokens = tokens[mask]
        
        if len(cls_tokens) < 2:
            print(f"  类别 {cls_id}: 样本不足 (<2)")
            continue
        
        # 类内距离（采样避免计算量过大）
        sample_size = min(100, len(cls_tokens))
        sample_indices = torch.randperm(len(cls_tokens))[:sample_size]
        sample = cls_tokens[sample_indices]
        intra_dist = torch.pdist(sample).mean()
        
        # 类间距离
        other_mask = labels != cls_id
        if other_mask.sum() > 0:
            other_indices = torch.randperm(other_mask.sum())[:100]
            other_sample = tokens[other_mask][other_indices]
            inter_dist = torch.cdist(sample, other_sample).mean()
            
            separation = inter_dist / (intra_dist + 1e-8)
            
            print(f"  类别 {cls_id}: 类内={intra_dist:.4f}, 类间={inter_dist:.4f}, "
                  f"分离度={separation:.2f}")
            
            if separation < 1.5:
                print(f"    ⚠️  警告: 分离度过低 (<1.5)，该类别可能难以区分")
    
    print("="*60 + "\n")


# ========== 主函数 ==========
def main(
    inp='./output_features/vitt_distill_features.pkl',
    num_prototypes=10,
    momentum=0.1,      # 从 0.002 增大到 0.1
    epochs=50,         # 从 30 增加到 50
    batch_size=256,    # 从 512 减小到 256
    normalize=True,
    temperature=0.07,
    device='cuda',
    save=True,
    save_tokens=False
):
    """
    改进的原型学习主函数
    
    参数说明:
    - inp: 输入的特征文件路径
    - num_prototypes: 每个类别的原型数量
    - momentum: 动量更新系数 (建议 0.05-0.15)
    - epochs: 训练轮数
    - batch_size: 批次大小
    - normalize: 是否归一化特征
    - temperature: 温度缩放参数 (越小决策越sharp)
    - device: 'cuda' 或 'cpu'
    - save: 是否保存结果
    - save_tokens: 是否同时保存原始特征
    """
    kwargs = locals()
    pprint(kwargs)
    
    # 设备配置
    if device != 'cpu':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 1. 加载数据
    print('==> 加载特征文件...')
    dataset = torch.load(inp, map_location='cpu')
    
    labels = torch.tensor(dataset['labels'])
    tokens = torch.stack(dataset['patch_tokens'])
    
    num_classes = len(torch.unique(labels))
    embed_dim = tokens.shape[-1]
    
    print(f'特征总数: {len(tokens)}')
    print(f'类别数量: {num_classes}')
    print(f'特征维度: {embed_dim}')
    
    # 2. 特征质量诊断
    diagnose_features(tokens, labels)
    
    # 3. 统计类别分布
    print('==> 类别分布:')
    class_counts = []
    for cls_id in range(num_classes):
        count = (labels == cls_id).sum().item()
        class_counts.append(count)
        print(f'  类别 {cls_id}: {count} 个样本')
    
    # 检查类别不平衡
    min_count = min(class_counts)
    max_count = max(class_counts)
    imbalance_ratio = max_count / (min_count + 1e-8)
    
    if imbalance_ratio > 10:
        print(f"\n⚠️  警告: 类别严重不平衡! 最多/最少 = {max_count}/{min_count} ({imbalance_ratio:.1f}x)")
        print("   建议: 考虑使用类别权重或重采样")
    
    # 4. 初始化学习器
    print(f'\n==> 初始化改进的原型学习器...')
    learner = ImprovedPrototypeLearner(
        num_classes=num_classes,
        num_prototypes=num_prototypes,
        embed_dim=embed_dim,
        momentum=momentum,
        normalize=normalize,
        temperature=temperature,
        device=device
    )
    
    # 5. 准备数据
    dataset_tensor = torch.cat([tokens, labels.unsqueeze(1)], dim=1)
    num_batches = len(dataset_tensor) // batch_size
    
    # 6. 训练
    print(f'\n==> 开始训练 ({epochs} 轮, 每轮 {num_batches} 个批次)...')
    best_acc = 0
    
    with tqdm(total=epochs * num_batches, desc='Training') as pbar:
        for epoch in range(epochs):
            # 打乱数据
            indices = torch.randperm(len(dataset_tensor))
            shuffled_data = dataset_tensor[indices]
            
            for i in range(num_batches):
                batch = shuffled_data[i * batch_size:(i + 1) * batch_size]
                batch = batch.to(device)
                
                batch_tokens = batch[:, :-1]
                batch_labels = batch[:, -1].long()
                
                learner.update(batch_tokens, batch_labels)
                pbar.update(1)
            
            # 每5轮评估一次
            if (epoch + 1) % 5 == 0:
                all_preds = []
                for batch_tokens in torch.split(tokens, 3000):
                    preds = learner.predict(batch_tokens)
                    all_preds.append(preds.cpu())
                pred_labels = torch.cat(all_preds)
                
                acc = (pred_labels == labels).float().mean().item()
                pbar.set_postfix({'epoch': epoch+1, 'acc': f'{acc:.4f}'})
                
                if acc > best_acc:
                    best_acc = acc
    
    # 7. 最终评估
    print('\n==> 最终评估...')
    all_preds = []
    for batch_tokens in tqdm(torch.split(tokens, 3000), desc='Predicting'):
        preds = learner.predict(batch_tokens)
        all_preds.append(preds.cpu())
    pred_labels = torch.cat(all_preds)
    
    match = pred_labels == labels
    
    print('\n==> 各类别准确率:')
    cls_accs = []
    for cls_id in range(num_classes):
        mask = labels == cls_id
        if mask.sum() == 0:
            continue
        acc = (match[mask].sum() / mask.sum()).item()
        cls_accs.append(acc)
        print(f'  类别 {cls_id}: {acc:.4f} (样本数: {mask.sum()})')
    
    mean_cls_acc = np.mean(cls_accs)
    overall_acc = (match.sum() / len(match)).item()
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print(f'平均类别准确率: {mean_cls_acc:.4f}')
    print(f'总体准确率: {overall_acc:.4f}')
    print(f'训练过程最佳准确率: {best_acc:.4f}')
    print("="*60)
    
    # 8. 保存结果
    if save:
        prototypes_reshaped = learner.get_prototypes().cpu()
        
        result = {
            'prototypes': prototypes_reshaped,  # [num_classes, num_prototypes, embed_dim]
            'kwargs': kwargs,
            'cls_acc': mean_cls_acc,
            'ovr_acc': overall_acc,
            'best_acc': best_acc,
            'class_counts': class_counts,
        }
        
        if save_tokens:
            result['tokens'] = tokens
            result['labels'] = labels
        
        out_path = osp.splitext(inp)[0] + f'.p{num_prototypes}.improved.pkl'
        print(f'\n==> 保存到: {out_path}')
        torch.save(result, out_path)
        print('完成！')


if __name__ == "__main__":
    # 直接运行（可修改参数）
    main(
        inp='./output_features/vitt_distill_features.pkl',
        num_prototypes=10,
        momentum=0.1,
        epochs=500,
        batch_size=256,
        normalize=True,
        temperature=0.07,
        device='cuda',
        save=True,
        save_tokens=False
    )
