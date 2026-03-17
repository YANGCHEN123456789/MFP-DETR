

## 1. 环境配置 (Environment Setup)

MFP-DETR 建议使用 Python 3.11 及 PyTorch 2.0 以上版本。

```shell
# 创建并激活环境
conda create -n mfp-detr python=3.11 -y
conda activate mfp-detr

# 安装依赖项
pip install -r requirements.txt
```

---

## 2. 预训练权重准备 (Backbone Preparation)

支持ViT多种骨干网络。请根据模型版本将下载好的权重放入 `./ckpts` 目录。

* **目录结构示例：**

```text
ckpts/
├── vitt_distill.pt        # MFP-DETR-S 使用
└── vittplus_distill.pt    # MFP-DETR-M 使用
```

---

## 3. 数据集准备 (Data Preparation)

项目采用 **COCO 格式**。



如需使用公开数据集，可参考 **PTL-AI Furnas dataset**：
- Dataset link: `https://github.com/freds0/PTL-AI_Furnas_Dataset`

如果使用电力巡检自定义数据集，请确保 `remap_mscoco_category` 设为 `False`。
- Dataset link: `https://drive.google.com/drive/folders/1UkeR44yuiyzhPhW3nCOoRIqQCVsr9TXw?usp=drive_link`

1. **修改配置文件：** 编辑 `configs/dataset/custom_detection_furnas.yml`。
2. **设置路径：** 指向您的图片文件夹（img_folder）和 JSON 注释文件（ann_file）。
---

## 4. 模型训练与微调 (Training & Tuning)

根据您选择的配置文件，使用以下命令：

```shell
# 以 ViT-S 版本为例启动分布式训练 (4卡)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 \
train.py -c configs/mfp-detr/deim_dinov3_s_furnas.yml --use-amp --seed=0

# 若要从 checkpoint 恢复或微调，添加 -t 参数
python train.py -c configs/mfp-detr/deim_dinov3_s_furnas.yml -t outputs/furnas/SPM-WFCE-Prototypes-S/best_stg1.pth

```

---


## 5. 模型部署 (Deployment)

推荐导出至 ONNX 后通过 TensorRT 加速。

1. **导出 ONNX：**

```shell
python tools/deployment/export_onnx.py --check \
  -c configs/mfp-detr/deim_dinov3_s_furnas.yml \
  -r outputs/furnas/SPM-WFCE-Prototypes-S/best_stg1.pth

```

2. **TensorRT 转换 (推荐版本 ≥ 10.6)：**

```shell
# 建议使用 fp16 以获得电力巡检所需的实时性
trtexec --onnx="model.onnx" --saveEngine="model.engine" --fp16

```

## 6. 推理与可视化 (Inference & Vis)

### 6.1 安装推理依赖

```shell
pip install -r tools/inference/requirements.txt

```

### 6.2 PyTorch 模型推理

适用于验证模型训练效果：

```shell
python tools/inference/torch_inf.py \
  -c configs/mfp-detr/deim_dinov3_s_furnas.yml \
  -r outputs/furnas/SPM-WFCE-Prototypes-S/best_stg1.pth \
  --input test_image.jpg \
  --device cuda:0

```

### 6.3 ONNX 模型推理

在执行导出操作（见第 6 节）后，可以使用 ONNX Runtime 进行推理：

```shell
python tools/inference/onnx_inf.py \
  --onnx model.onnx \
  --input test_image.jpg

```

### 6.4 TensorRT 模型推理

在生成 `.engine` 文件后，进行极速推理：

```shell
python tools/inference/trt_inf.py \
  --trt model.engine \
  --input test_image.jpg

```
---




