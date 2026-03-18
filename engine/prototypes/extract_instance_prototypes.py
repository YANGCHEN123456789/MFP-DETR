import numpy as np
import torch
import fire
torch.set_grad_enabled(False)
from torchvision import transforms
from typing import Sequence
import sys
import os.path as osp
from tqdm.auto import tqdm
from torchvision.transforms import functional as tvF
import torchvision as tv
import json
from PIL import Image
from fast_pytorch_kmeans import KMeans

# For COCO
import pycocotools.mask as mask_util


pixel_mean = torch.Tensor([123.675, 116.280, 103.530]).view(3, 1, 1)
pixel_std = torch.Tensor([58.395, 57.120, 57.375]).view(3, 1, 1)
normalize_image = lambda x: (x - pixel_mean) / pixel_std
denormalize_image = lambda x: (x * pixel_std) + pixel_mean


def compress(tensor, n_clst=5):
    if len(tensor) <= n_clst:
        return tensor
    else:
        kmeans = KMeans(n_clusters=n_clst, verbose=False, mode='cosine')
        kmeans.fit(tensor)
        return kmeans.centroids


def save_img(img, path):
    tv.utils.save_image(denormalize_image(img) / 255, path)


def iround(x): return int(round(x))


def crop(img, box, enlarge=0.2):
    h, w = img.shape[1:]

    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2  
    lx = (box[2] - box[0]) * (1 + enlarge)
    ly = (box[3] - box[1]) * (1 + enlarge)

    x0 = max(int(round(cx - lx / 2)), 0)
    x1 = min(int(round(cx + lx / 2)), w)
    y0 = max(int(round(cy - ly / 2)), 0)
    y1 = min(int(round(cy + ly / 2)), h)

    return img[:, y0:y1, x0:x1]


def resize_with_largest_edge(img, size=224):
    h, w = img.shape[1:]
    if h >= w:
        ratio = w / h
        h = size
        w = iround(h * ratio)
    else:
        ratio = h / w
        w = size
        h = iround(ratio * w)
    return tvF.resize(img, (h, w), interpolation=tvF.InterpolationMode.BICUBIC)


def resize_to_closest_14x(img):
    h, w = img.shape[1:]
    h, w = max(iround(h / 14), 1) * 14, max(iround(w / 14), 1) * 14
    return tvF.resize(img, (h, w), interpolation=tvF.InterpolationMode.BICUBIC)


def to_mask(boxes, height, width):
    result = torch.zeros(len(boxes), height, width)
    boxes = torch.round(boxes).long()
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        result[i, y1:y2, x1:x2] = 1
    return result.bool()


def load_coco_data(coco_json, image_dir):
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    image_dict = {}
    for image in images:
        image_dict[image['id']] = image

    annotations_dict = {}
    for annotation in annotations:
        if annotation['image_id'] not in annotations_dict:
            annotations_dict[annotation['image_id']] = []
        annotations_dict[annotation['image_id']].append(annotation)

    return image_dict, annotations_dict, categories


def get_dataloader(image_dict, annotations_dict, categories, image_dir, augmentations=None):
    dataset = []
    for image_id, image_info in image_dict.items():
        image_path = osp.join(image_dir, image_info['file_name'])
        img = Image.open(image_path).convert('RGB')
        width, height = img.size
        img = tvF.to_tensor(img)

        if augmentations:
            img = augmentations(img)

        # Get annotations for this image
        annotations = annotations_dict.get(image_id, [])
        target = []
        for ann in annotations:
            bbox = torch.tensor(ann['bbox'], dtype=torch.float32)
            category_id = ann['category_id']
            category_name = categories[category_id]
            mask = mask_util.decode(ann['segmentation'])
            target.append({'bbox': bbox, 'category_id': category_id, 'category_name': category_name, 'mask': mask})

        dataset.append({'image': img, 'target': target})

    return dataset


def main(model='vits14', coco_json='None', image_dir='None', 
         use_bbox='yes', epochs=1, device=0, n_clst=5, augmentations=None, out_dir=None):
    use_bbox = use_bbox == 'yes'

    model = torch.hub.load(None, None, trust_repo=True, source='local').to(device)

    # Load COCO dataset
    image_dict, annotations_dict, categories = load_coco_data(coco_json, image_dir)

    dataloader = get_dataloader(image_dict, annotations_dict, categories, image_dir, augmentations)

    dataset = {
        'labels': [],
        'patch_tokens': [],
        'avg_patch_tokens': [],
        'image_id': [],
        'boxes': [],
        'areas': [],
        'skip': 0
    }

    with tqdm(total=epochs * len(dataloader)) as bar:
        for _ in range(epochs):
            for item in dataloader:
                image = item['image'].to(device)
                target_mask_size = image.shape[1] // 14, image.shape[2] // 14

                r = model.get_intermediate_layers(image[None, ...], 
                                                  return_class_token=True, reshape=True)
                patch_tokens = r[0][0][0]  # c, h, w

                for ann in item['target']:
                    bbox = ann['bbox']
                    mask = ann['mask']
                    label = ann['category_id']

                    bbox_masks = to_mask(bbox, image.shape[1], image.shape[2]).to(device)
                    bmask = bbox_masks.float()[None, ...]
                    bmask = tvF.resize(bmask, target_mask_size)
                    if bmask.sum() <= 0.5:
                        dataset['skip'] += 1
                        continue

                    avg_patch_token = (bmask * patch_tokens).flatten(1).sum(1) / bmask.sum()
                    dataset['avg_patch_tokens'].append(avg_patch_token.cpu())
                    dataset['labels'].append(label)

                bar.update()

    name = 'coco_data.' + model.__class__.__name__

    if use_bbox:
        name += '.bbox'

    name += '.pkl'
    if out_dir is not None:
        name = osp.join(out_dir, name)

    print(f'Saving to {name}')
    torch.save(dataset, name)


if __name__ == "__main__":
    fire.Fire(main)
