import os
import gc
import torch
import numpy as np
import os.path as osp
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as tvF
from pycocotools.coco import COCO

# 确保当前目录下有 vit_tiny.py
from vit_tiny import VisionTransformer

# ---------------- 配置区域 ----------------
CHECKPOINT_PATH = '/input/yangchen/torch_learn/project/yc_graduate/DEIMv2/ckpts/vitt_distill.pt'
COCO_JSON = '/input/yangchen/torch_learn/project/yc_graduate/DEIMv2/datasets/quexian17_all/annotations/instances_train2017_fixed.json'
IMAGE_DIR = '/input/yangchen/torch_learn/project/yc_graduate/DEIMv2/datasets/quexian17_all/images/train2017'
OUT_DIR = './output_features'

PIXEL_MEAN = torch.Tensor([123.675, 116.280, 103.530]).view(3, 1, 1)
PIXEL_STD = torch.Tensor([58.395, 57.120, 57.375]).view(3, 1, 1)
# ------------------------------------------

class DINOv3FeatureExtractor:
    def __init__(self, ckpt_path, device='cuda'):
        self.device = device
        
        print(f"==> 初始化 VisionTransformer (Embed=192, Heads=3)...")
        self.model = VisionTransformer(embed_dim=192, num_heads=3)
        
        print(f"==> 加载权重: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location='cpu')
        
        # 兼容各种封装格式
        if isinstance(state_dict, dict):
            for key in ['model', 'state_dict', '_model', 'teacher']:
                if key in state_dict:
                    state_dict = state_dict[key]
                    break
        
        msg = self.model.load_state_dict(state_dict, strict=False)
        print(f"    权重加载状态: {msg}")
        self.model.to(device).eval()
        print(f"==> 模型已加载到 {device}")

    def _extract_tensor_recursive(self, data):
        """递归提取第一个 torch.Tensor"""
        if torch.is_tensor(data):
            return data
        if isinstance(data, (list, tuple)):
            for item in data:
                result = self._extract_tensor_recursive(item)
                if result is not None:
                    return result
        if isinstance(data, dict):
            for value in data.values():
                result = self._extract_tensor_recursive(value)
                if result is not None:
                    return result
        return None

    def _infer_grid_size(self, total_tokens, w_ori, h_ori):
        """
        增强的网格尺寸推断算法
        
        策略：
        1. 尝试常见的 offset (CLS + Register tokens)
        2. 对于每个 offset，找最接近目标宽高比的因式分解
        3. 允许小误差（±2 tokens）以处理边界 padding
        """
        aspect_ratio = w_ori / h_ori
        best_match = None
        min_error = float('inf')
        
        # 扩展 offset 搜索范围
        for offset in range(0, 20):  # 0 到 19 个额外 token
            num_patches = total_tokens - offset
            if num_patches <= 0:
                continue
            
            # 方法 1：基于宽高比的精确匹配
            grid_w = int(np.round(np.sqrt(num_patches * aspect_ratio)))
            grid_h = num_patches // grid_w
            
            if grid_w * grid_h == num_patches:
                return grid_h, grid_w, offset
            
            # 方法 2：允许小误差的近似匹配
            error = abs(grid_w * grid_h - num_patches)
            if error <= 2 and error < min_error:  # 允许 ±2 tokens 误差
                min_error = error
                best_match = (grid_h, grid_w, offset)
            
            # 方法 3：尝试所有可能的因式分解（针对特殊尺寸）
            for w in range(1, int(np.sqrt(num_patches)) + 20):
                if num_patches % w == 0:
                    h = num_patches // w
                    ratio = w / h
                    ratio_error = abs(ratio - aspect_ratio)
                    
                    if ratio_error < 0.3:  # 宽高比误差小于 30%
                        total_error = ratio_error * 10 + error
                        if total_error < min_error:
                            min_error = total_error
                            best_match = (h, w, offset)
        
        if best_match is not None:
            return best_match
        
        # 最后的保底方案：强制开方
        num_patches = total_tokens - 1  # 假设只有 1 个 CLS token
        side = int(np.sqrt(num_patches))
        return (side, side, 1)

    @torch.inference_mode()
    def process_image(self, img_path, bboxes):
        """完整的特征提取流程"""
        # 1. 图像预处理
        img = Image.open(img_path).convert('RGB')
        w_ori, h_ori = img.size
        
        # 显存保护
        MAX_RES = 1344
        if max(w_ori, h_ori) > MAX_RES:
            scale = MAX_RES / max(w_ori, h_ori)
            img = img.resize((int(w_ori * scale), int(h_ori * scale)), Image.BICUBIC)
            w_ori, h_ori = img.size
        
        # 转为 Tensor
        tensor = tvF.to_tensor(img).to(self.device) * 255.0
        tensor = (tensor - PIXEL_MEAN.to(self.device)) / PIXEL_STD.to(self.device)
        tensor = tensor.unsqueeze(0)

        # 2. 模型前向传播
        output = self.model(tensor)
        tokens = self._extract_tensor_recursive(output)
        
        if tokens is None or not torch.is_tensor(tokens):
            raise ValueError(f"无法提取 Tensor，输出类型: {type(output)}")

        if tokens.dim() != 3:
            raise ValueError(f"期望 [B,N,C]，实际: {tokens.shape}")
        
        embed_dim = tokens.shape[-1]
        total_tokens = tokens.shape[1]
        
        # 3. 使用增强的网格推断算法
        grid_h, grid_w, offset = self._infer_grid_size(total_tokens, w_ori, h_ori)
        
        # 截取对应数量的 patch tokens
        num_patches = grid_h * grid_w
        if num_patches <= total_tokens:
            tokens = tokens[:, -num_patches:, :]
        else:
            # 如果计算出的 patches 多于实际 tokens，进行 padding
            pad_size = num_patches - total_tokens
            tokens = torch.nn.functional.pad(tokens, (0, 0, 0, pad_size))

        # 4. 还原特征图
        try:
            feature_map = tokens[0].permute(1, 0).reshape(embed_dim, grid_h, grid_w)
        except RuntimeError as e:
            # 如果还是失败，使用最保守的策略
            actual_size = tokens.shape[1]
            side = int(np.sqrt(actual_size))
            tokens = tokens[:, :side*side, :]
            feature_map = tokens[0].permute(1, 0).reshape(embed_dim, side, side)
            grid_h, grid_w = side, side
        
        # 5. ROI 池化
        roi_features = []
        for box in bboxes:
            x, y, w, h = box
            
            x1 = int(np.round(x * grid_w / w_ori))
            y1 = int(np.round(y * grid_h / h_ori))
            x2 = max(x1 + 1, int(np.round((x + w) * grid_w / w_ori)))
            y2 = max(y1 + 1, int(np.round((y + h) * grid_h / h_ori)))
            
            x1 = max(0, min(x1, grid_w - 1))
            y1 = max(0, min(y1, grid_h - 1))
            x2 = max(x1 + 1, min(x2, grid_w))
            y2 = max(y1 + 1, min(y2, grid_h))
            
            region = feature_map[:, y1:y2, x1:x2]
            roi_features.append(region.mean(dim=(1, 2)).cpu())
        
        del tensor, tokens, feature_map
        return roi_features


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    extractor = DINOv3FeatureExtractor(CHECKPOINT_PATH, device)
    
    print(f"\n==> 加载数据集: {COCO_JSON}")
    coco = COCO(COCO_JSON)
    img_ids = coco.getImgIds()
    print(f"    数据集包含 {len(img_ids)} 张图片")
    
    results = {
        'patch_tokens': [],
        'labels': [],
        'img_info': []
    }

    print(f"\n==> 开始提取特征...")
    success_count = 0
    fail_count = 0
    
    for i, img_id in enumerate(tqdm(img_ids, desc="Processing")):
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        if not anns:
            continue
        
        img_path = osp.join(IMAGE_DIR, img_info['file_name'])
        if not osp.exists(img_path):
            fail_count += 1
            continue
        
        try:
            bboxes = [ann['bbox'] for ann in anns]
            feats = extractor.process_image(img_path, bboxes)
            
            for feat, ann in zip(feats, anns):
                results['patch_tokens'].append(feat)
                results['labels'].append(ann['category_id'])
                results['img_info'].append({
                    'img_id': img_id,
                    'bbox': ann['bbox'],
                    'category_id': ann['category_id']
                })
            
            success_count += 1
            
            if i % 50 == 0 and i > 0:
                torch.cuda.empty_cache()
                gc.collect()
                
        except Exception as e:
            print(f"\n[错误] 图片 ID {img_id} ({img_info['file_name']}): {str(e)}")
            fail_count += 1
            torch.cuda.empty_cache()

    save_path = osp.join(OUT_DIR, 'vitt_distill_features.pkl')
    print(f"\n==> 保存结果到: {save_path}")
    torch.save(results, save_path)
    
    print("\n" + "="*60)
    print("特征提取完成！")
    print(f"成功: {success_count} 张 | 失败: {fail_count} 张")
    print(f"提取特征总数: {len(results['patch_tokens'])}")
    if results['patch_tokens']:
        print(f"特征维度: {results['patch_tokens'][0].shape}")
    print(f"输出文件: {save_path}")
    print("="*60)


if __name__ == "__main__":
    main()
