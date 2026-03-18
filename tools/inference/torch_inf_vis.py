
import os
import random
import sys

import cv2  # Added for video processing
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig

label_map = {
0: 'dx_sg', 1: 'dx_dg', 2: 'yw'
}

COLORS = plt.cm.tab20.colors
COLOR_MAP = {label: tuple([int(c * 255) for c in COLORS[i % len(COLORS)]]) for i, label in enumerate(label_map)}


def draw(image, labels, boxes, scores, thrh=0.7):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    labels, boxes, scores = labels[scores > thrh], boxes[scores > thrh], scores[scores > thrh]

    for j, box in enumerate(boxes):
        category = labels[j].item()
        color = COLOR_MAP.get(category, (255, 255, 255))
        box = list(map(int, box))

        draw.rectangle(box, outline=color, width=10)

        text = f"{label_map[category]} {scores[j].item():.2f}"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        text_background = [box[0], box[1] - text_height - 2, box[0] + text_width + 4, box[1]]
        draw.rectangle(text_background, fill=color)
        draw.text((box[0] + 2, box[1] - text_height - 2), text, fill="black", font=font)

    return image


def process_image(model, file_path, size=(640, 640)):
    im_pil = Image.open(file_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).cuda()

    transforms = T.Compose([
        T.Resize((size)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil).unsqueeze(0).cuda()

    output = model(im_data, orig_size)

    draw([im_pil], output)


def process_video(model, file_path, size=(640, 640)):
    cap = cv2.VideoCapture(file_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('torch_results.mp4', fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose([
        T.Resize(size),
        T.ToTensor(),
    ])

    frame_count = 0
    print("Processing video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).cuda()

        im_data = transforms(frame_pil).unsqueeze(0).cuda()

        output = model(im_data, orig_size)
        labels, boxes, scores = output

        # Draw detections on the frame
        draw([frame_pil], labels, boxes, scores)

        # Convert back to OpenCV image
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # Write the frame
        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'results_video.mp4'.")

def process_dataset(model, dataset_path, output_path, thrh=0.7, size=(640, 640)):
    os.makedirs(output_path, exist_ok=True)
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png'))]

    transforms = T.Compose([
        T.Resize(size),
        T.ToTensor(),
    ])

    print(f"Found {len(image_paths)} images in validation set...")
    for idx, file_path in enumerate(image_paths):
        im_pil = Image.open(file_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).cuda()

        im_data = transforms(im_pil).unsqueeze(0).cuda()
        output = model(im_data, orig_size)
        labels, boxes, scores = output[0]['labels'], output[0]['boxes'], output[0]['scores']

        vis_image = draw(im_pil.copy(), labels, boxes, scores, thrh)
        save_path = os.path.join(output_path, f"vis_{os.path.basename(file_path)}")
        vis_image.save(save_path)

        if idx % 500 == 0:
            print(f"Processed {idx}/{len(image_paths)} images...")

    print("Visualization complete. Results saved in:", output_path)


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.eval().cuda()
            self.postprocessor = cfg.postprocessor.eval().cuda()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model()
    img_size = cfg.yaml_cfg["eval_spatial_size"]
    process_dataset(model, args.dataset, args.output, thrh=0.7, size=img_size)
    # file_path = args.input
    # if os.path.splitext(file_path)[-1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
    #     process_image(model, file_path)
    #     print("Image processing complete.")
    # else:
    #     process_video(model, file_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-d', '--dataset', type=str, default='./data/fiftyone/validation/data')
    parser.add_argument('-o', '--output', type=str, required=True, help="Path to save visualized results")
    args = parser.parse_args()
    main(args)
