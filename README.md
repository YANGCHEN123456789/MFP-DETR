

---

## 1. Environment Setup

MFP-DETR is recommended to run with **Python 3.11** and **PyTorch 2.0+**.

```shell
# Create and activate the environment
conda create -n mfp-detr python=3.11 -y
conda activate mfp-detr

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Backbone Preparation

Multiple ViT backbones are supported. Please place the downloaded pretrained weights into the `./ckpts` directory according to the selected model version.

**Example directory structure:**

```text
ckpts/
├── vitt_distill.pt        # for MFP-DETR-S
└── vittplus_distill.pt    # for MFP-DETR-M
```

---

## 3. Data Preparation

This project adopts the **COCO format** for dataset annotation.

If you would like to use a public dataset, please refer to the **PTL-AI Furnas dataset**:

If you use a custom power inspection dataset, please make sure to set `remap_mscoco_category=False`.


1. **Modify the dataset configuration file:**
   Edit `configs/dataset/custom_detection_furnas.yml`.

2. **Set the dataset paths:**
   Specify your image folder (`img_folder`) and JSON annotation file (`ann_file`).

---

## 4. Training & Fine-tuning

Use the following commands according to the selected configuration file:

```shell
# Launch distributed training with 4 GPUs (ViT-S version as an example)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=7777 --nproc_per_node=4 \
train.py -c configs/mfp-detr/deim_dinov3_s_furnas.yml --use-amp --seed=0

# To resume training or fine-tune from a checkpoint, add the -t argument
python train.py -c configs/mfp-detr/deim_dinov3_s_furnas.yml -t outputs/furnas/SPM-WFCE-Prototypes-S/best_stg1.pth
```

---

## 5. Deployment

It is recommended to first export the model to **ONNX** and then accelerate it with **TensorRT**.

### 5.1 Export to ONNX

```shell
python tools/deployment/export_onnx.py --check \
  -c configs/mfp-detr/deim_dinov3_s_furnas.yml \
  -r outputs/furnas/SPM-WFCE-Prototypes-S/best_stg1.pth
```

### 5.2 Convert with TensorRT (Recommended version >= 10.6)

```shell
# FP16 is recommended to achieve real-time performance for power inspection
trtexec --onnx="model.onnx" --saveEngine="model.engine" --fp16
```

---

## 6. Inference & Visualization

### 6.1 Install inference dependencies

```shell
pip install -r tools/inference/requirements.txt
```

### 6.2 PyTorch inference

This is suitable for validating the performance of the trained PyTorch model:

```shell
python tools/inference/torch_inf.py \
  -c configs/mfp-detr/deim_dinov3_s_furnas.yml \
  -r outputs/furnas/SPM-WFCE-Prototypes-S/best_stg1.pth \
  --input test_image.jpg \
  --device cuda:0
```

### 6.3 ONNX inference

After exporting the model to ONNX (see Section 5), you can run inference with ONNX Runtime:

```shell
python tools/inference/onnx_inf.py \
  --onnx model.onnx \
  --input test_image.jpg
```

### 6.4 TensorRT inference

After generating the `.engine` file, you can perform high-speed inference with TensorRT:

```shell
python tools/inference/trt_inf.py \
  --trt model.engine \
  --input test_image.jpg
```

---


