"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""
import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import os

# ==========================================
# 1. 工具函数：切片计算、NMS、图像调整
# ==========================================

def resize_with_aspect_ratio_pad(image_pil, target_size, interpolation=Image.BILINEAR):
    """
    将 PIL 图像保持宽高比缩放到 target_size，并进行填充。
    返回: 填充后的图像, 缩放比例, (pad_w, pad_h)
    """
    w, h = image_pil.size
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized_image = image_pil.resize((new_w, new_h), interpolation)
    
    # 使用中灰色填充
    new_image = Image.new("RGB", (target_size, target_size), (114, 114, 114)) 
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    new_image.paste(resized_image, (pad_w, pad_h))
    
    return new_image, scale, (pad_w, pad_h)

def get_slice_bboxes(image_shape, slice_size, overlap_ratio=0.2):
    """
    计算所有切片在原图上的坐标 (y_min, x_min, y_max, x_max)。
    """
    img_h, img_w = image_shape[:2]
    slice_h, slice_w = slice_size, slice_size
    
    stride_h = int(slice_h * (1 - overlap_ratio))
    stride_w = int(slice_w * (1 - overlap_ratio))
    
    slice_bboxes = []
    
    y_starts = list(range(0, img_h, stride_h))
    x_starts = list(range(0, img_w, stride_w))

    # 调整最后一块，确保覆盖边界
    if y_starts[-1] + slice_h > img_h: y_starts[-1] = max(0, img_h - slice_h)
    if x_starts[-1] + slice_w > img_w: x_starts[-1] = max(0, img_w - slice_w)
        
    for y_min in y_starts:
        for x_min in x_starts:
            y_max = min(img_h, y_min + slice_h)
            x_max = min(img_w, x_min + slice_w)
            # 只添加有效的切片
            if y_max > y_min and x_max > x_min:
                slice_bboxes.append([y_min, x_min, y_max, x_max])
            
    return slice_bboxes

def global_nms(all_detections, iou_threshold=0.45):
    """
    对还原到原图坐标系下的所有框进行全局 NMS。
    all_detections format: [x1, y1, x2, y2, score, label_index]
    """
    if not all_detections:
        return np.array([]), np.array([]), np.array([])

    boxes_xywh = []
    scores = []
    labels = []

    for det in all_detections:
        x1, y1, x2, y2, score, label_idx = det
        w = x2 - x1
        h = y2 - y1
        boxes_xywh.append([int(x1), int(y1), int(w), int(h)])
        scores.append(float(score))
        labels.append(int(label_idx))

    # OpenCV NMS
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, score_threshold=0.0, nms_threshold=iou_threshold)

    final_boxes = []
    final_scores = []
    final_labels = []
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes_xywh[i]
            final_boxes.append([x, y, x + w, y + h]) # xywh -> xyxy
            final_scores.append(scores[i])
            final_labels.append(labels[i])
            
    return np.array(final_boxes), np.array(final_scores), np.array(final_labels)


# ==========================================
# 2. 绘图函数
# ==========================================
def draw_on_original_image(image_pil, boxes, scores, labels, thrh=0.4):
    # ========== 可视化参数配置 ==========
    box_line_width = 3
    font_size = 20
    box_color = 'red'
    font_color = 'white'
    text_stroke_width = 1
    
    draw_ctx = ImageDraw.Draw(image_pil)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    if len(boxes) == 0:
        return image_pil

    # 过滤低置信度的框 (虽然核心逻辑已经过滤过一次，这里再保险一次)
    mask = scores > thrh
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        
        # 绘制边界框
        draw_ctx.rectangle(
            [x1, y1, x2, y2], 
            outline=box_color, 
            width=box_line_width
        )
        
        # 绘制标签文本
        # label_text = f"Class {int(label)}: {score:.2f}"
        label_text = "dx_sg" # 根据你的需求固定文本
        
        # 计算文本背景框大小并绘制
        text_bbox = draw_ctx.textbbox((x1, y1), label_text, font=font, stroke_width=text_stroke_width)
        draw_ctx.rectangle(text_bbox, fill=box_color)
        
        draw_ctx.text(
            (x1, y1), 
            text=label_text, 
            fill=font_color,
            font=font,
            stroke_width=text_stroke_width
        )

    return image_pil


# ==========================================
# 3. 核心通用逻辑：单张大图切片推理
# ==========================================
def inference_on_large_image_sliced(sess, large_image_pil, model_input_size=640, overlap_ratio=0.25, conf_thrh=0.4):
    """
    对任意大图进行切片推理的核心函数。
    返回: final_boxes(原图坐标), final_scores, final_labels
    """
    img_w, img_h = large_image_pil.size
    
    # 1. 计算切片方案
    # 切片大小通常设为模型输入大小
    slice_bboxes = get_slice_bboxes((img_h, img_w), model_input_size, overlap_ratio)
    
    transforms = T.Compose([T.ToTensor()])
    detections_global = [] # format: [x1, y1, x2, y2, score, label]

    # --- 遍历切片 ---
    for (y_min, x_min, y_max, x_max) in slice_bboxes:
        # a. 抠取切片
        slice_pil = large_image_pil.crop((x_min, y_min, x_max, y_max))
        
        # b. 预处理切片 (Resize + Pad)
        resized_slice_pil, ratio, (pad_w, pad_h) = resize_with_aspect_ratio_pad(slice_pil, model_input_size)
        
        # 准备模型输入
        # 注意: orig_target_sizes 应该是 resize 后的尺寸，而不是切片的原始尺寸
        # 对于 D-FINE/DEIMv2 这类模型，这个参数可能影响后处理，传入 resize 后的尺寸较安全
        orig_size_tensor = torch.tensor([[resized_slice_pil.size[1], resized_slice_pil.size[0]]])
        im_data = transforms(resized_slice_pil).unsqueeze(0)

        # c. ONNX 推理
        output = sess.run(
            output_names=None,
            input_feed={'images': im_data.numpy(), "orig_target_sizes": orig_size_tensor.numpy()}
        )
        labels_out, boxes_out, scores_out = output
        
        # 取 batch 中第一个结果
        boxes_slice = boxes_out[0] # [N, 4]
        scores_slice = scores_out[0] # [N]
        labels_slice = labels_out[0] # [N]
        
        # d. 初步过滤低置信度框
        mask = scores_slice > conf_thrh
        boxes_slice = boxes_slice[mask]
        scores_slice = scores_slice[mask]
        labels_slice = labels_slice[mask]

        if len(boxes_slice) == 0:
            continue

        # e. 坐标映射还原
        # Step 1: 从 padded resize 坐标 -> 切片坐标
        boxes_slice[:, [0, 2]] = (boxes_slice[:, [0, 2]] - pad_w) / ratio
        boxes_slice[:, [1, 3]] = (boxes_slice[:, [1, 3]] - pad_h) / ratio
        
        # Step 2: 从切片坐标 -> 原图全局坐标 (加上偏移量)
        boxes_slice[:, [0, 2]] += x_min
        boxes_slice[:, [1, 3]] += y_min
        
        # f. 收集结果
        for box, score, label in zip(boxes_slice, scores_slice, labels_slice):
            detections_global.append([box[0], box[1], box[2], box[3], score, label])

    # --- 全局 NMS ---
    final_boxes, final_scores, final_labels = global_nms(detections_global, iou_threshold=0.45)
    
    return final_boxes, final_scores, final_labels


# ==========================================
# 4. 业务逻辑：处理单张图片
# ==========================================
def process_image(sess, input_path, output_dir, model_input_size=640):
    print(f"Processing image: {input_path}")
    im_pil = Image.open(input_path).convert('RGB')
    
    # 调用通用的切片推理函数
    boxes, scores, labels = inference_on_large_image_sliced(
        sess, im_pil, model_input_size=model_input_size
    )
    
    # 绘图
    result_image = draw_on_original_image(im_pil, boxes, scores, labels)
    
    # 保存结果
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_result.jpg")
    result_image.save(output_path)
    print(f"Image saved to: {output_path}")


# ==========================================
# 5. 业务逻辑：处理视频
# ==========================================
def process_video(sess, input_path, output_dir, model_input_size=640):
    print(f"Processing video: {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video file: {input_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define VideoWriter
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (orig_w, orig_h))

    print(f"Video Info: {orig_w}x{orig_h}, {fps} fps, {total_frames} frames.")
    
    # 进度条
    pbar = tqdm(total=total_frames, desc="Processing Frames")

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # 转换为 PIL 用于推理
        frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        
        # 调用通用的切片推理函数
        boxes, scores, labels = inference_on_large_image_sliced(
            sess, frame_pil, model_input_size=model_input_size
        )

        # 绘图
        frame_drawn_pil = draw_on_original_image(frame_pil, boxes, scores, labels)
        
        # 转回 OpenCV BGR 格式用于写入视频
        frame_drawn_bgr = cv2.cvtColor(np.array(frame_drawn_pil), cv2.COLOR_RGB2BGR)
        
        out.write(frame_drawn_bgr)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()
    print(f"Video saved to: {output_path}")


def main(args):
    """Main function."""
    # Load the ONNX model
    try:
        # 优先使用 CUDA
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        sess = ort.InferenceSession(args.onnx, providers=providers)
        print(f"Model loaded successfully. Using providers: {sess.get_providers()}")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    # 获取模型输入尺寸
    try:
        # 假设输入形状类似于 [batch, channel, height, width]
        model_input_size = sess.get_inputs()[0].shape[2]
        print(f"Model input size determined as: {model_input_size}")
    except:
        model_input_size = 640
        print(f"Could not determine input size, defaulting to: {model_input_size}")

    input_path = args.input
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 判断输入类型
    is_video = False
    try:
        # 尝试读取几个字节来判断是否为图像文件头，比直接打开更安全
        with open(input_path, 'rb') as f:
            header = f.read(32)
            if header.startswith(b'\xff\xd8') or \
               header.startswith(b'\x89PNG') or \
               header.startswith(b'BM'):
                is_image = True
            else:
                # 如果不是常见图像格式，尝试用 OpenCV 打开看看是不是视频
                cap = cv2.VideoCapture(input_path)
                if cap.isOpened():
                    is_video = True
                    cap.release()
                else:
                    print("Error: Input file is neither a recognized image nor a video.")
                    return
    except Exception as e:
         print(f"Error checking input file type: {e}")
         return

    if is_video:
        process_video(sess, input_path, output_dir, model_input_size=model_input_size)
    else:
        process_image(sess, input_path, output_dir, model_input_size=model_input_size)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, required=True, help='Path to the ONNX model file.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input image or video file.')
    # 添加一个输出目录参数
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results.')
    args = parser.parse_args()
    main(args)