import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from models import build_model
import re
import numpy as np

def GetDepthImg(img):
    depth_img_rest = img.copy()
    depth_img_R = depth_img_rest.copy()
    depth_img_R[depth_img_rest > 255] = 255
    depth_img_rest[depth_img_rest < 255] = 255
    depth_img_rest -= 255
    depth_img_G = depth_img_rest.copy()
    depth_img_G[depth_img_rest > 255] = 255
    depth_img_rest[depth_img_rest < 255] = 255
    depth_img_rest -= 255
    depth_img_B = depth_img_rest.copy()
    depth_img_B[depth_img_rest > 255] = 255
    depth_img_rgb = np.stack([depth_img_R, depth_img_G, depth_img_B], axis=2)
    return depth_img_rgb.astype(np.uint8)

def WriteDepth(depth, limg, path, name):
    output_concat_color = os.path.join(path, "concat_color", name)
    output_concat_gray = os.path.join(path, "concat_gray", name)
    output_gray = os.path.join(path, "gray", name)
    output_gray_scale = os.path.join(path, "gray_scale", name)
    output_depth = os.path.join(path, "depth", name)
    output_color = os.path.join(path, "color", name)
    output_concat_depth = os.path.join(path, "concat_depth", name)
    output_concat = os.path.join(path, "concat", name)
    output_display = os.path.join(path, "display", name)
    MkdirSimple(output_concat_color)
    MkdirSimple(output_concat_gray)
    MkdirSimple(output_concat_depth)
    MkdirSimple(output_gray)
    MkdirSimple(output_depth)
    MkdirSimple(output_color)
    MkdirSimple(output_concat)
    MkdirSimple(output_display)
    MkdirSimple(output_gray_scale)

    predict_np = depth.squeeze().cpu().numpy()
    print(predict_np.max(), " ", predict_np.min())
    predict_scale = (predict_np - np.min(predict_np))* 255 / (np.max(predict_np) - np.min(predict_np))

    predict_scale = predict_scale.astype(np.uint8)
    predict_np_int = predict_scale
    color_img = cv2.applyColorMap(predict_np_int, cv2.COLORMAP_HOT)
    limg_cv = limg  # cv2.cvtColor(np.asarray(limg), cv2.COLOR_RGB2BGR)
    concat_img_color = np.vstack([limg_cv, color_img])
    predict_np_rgb = np.stack([predict_np, predict_np, predict_np], axis=2)
    concat_img_gray = np.vstack([limg_cv, predict_np_rgb])

    # get depth
    depth_img_rgb = GetDepthImg(predict_np)
    concat_img_depth = np.vstack([limg_cv, depth_img_rgb])
    concat = np.hstack([np.vstack([limg_cv, color_img]), np.vstack([predict_np_rgb, depth_img_rgb])])

    cv2.imwrite(output_concat_color, concat_img_color)
    cv2.imwrite(output_concat_gray, concat_img_gray)
    cv2.imwrite(output_color, color_img)

    cv2.imwrite(output_gray, predict_np)
    cv2.imwrite(output_depth, depth_img_rgb)
    cv2.imwrite(output_concat_depth, concat_img_depth)
    cv2.imwrite(output_concat, concat)

def MkdirSimple(path):
    path_current = path
    suffix = os.path.splitext(os.path.split(path)[1])[1]

    if suffix != "":
        path_current = os.path.dirname(path)
        if path_current in ["", "./", ".\\"]:
            return
    if not os.path.exists(path_current):
        os.makedirs(path_current)

def Walk(path, suffix:list):
    file_list = []
    suffix = [s.lower() for s in suffix]
    if not os.path.exists(path):
        print("not exist path {}".format(path))
        return []

    if os.path.isfile(path):
        return [path,]

    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1].lower()[1:] in suffix:
                file_list.append(os.path.join(root, file))

    try:
        file_list.sort(key=lambda x:int(re.findall('\d+', os.path.splitext(os.path.basename(x))[0])[0]))
    except:
        pass

    return file_list

class PredictModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(self.hparams)

    def forward(self, left, right):
        left = left * 2 - 1
        right = right * 2 - 1
        return self.model(left, right)


@torch.no_grad()
def predict(model, lp, rp, width, op, save_dir):
    left = cv2.imread(str(lp), cv2.IMREAD_COLOR)
    left_copy = left.copy()
    right = cv2.imread(str(rp), cv2.IMREAD_COLOR)
    if width is not None and width != left.shape[1]:
        height = int(round(width / left.shape[1] * left.shape[0]))
        left = cv2.resize(
            left,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
        right = cv2.resize(
            right,
            (width, height),
            interpolation=cv2.INTER_CUBIC,
        )
    left = np2torch(left, bgr=True).cuda().unsqueeze(0)
    right = np2torch(right, bgr=True).cuda().unsqueeze(0)
    pred = model(left, right)

    disp = pred["disp"]
    print(disp.max(), disp.min(), disp.shape)

    disp = torch.clip(disp / 192 * 255, 0, 255).long()
    WriteDepth(disp[0][0], left_copy, save_dir, op.replace(".jpg", ".png"))
    disp = apply_colormap(disp)
    torchvision.utils.save_image(disp, op, nrow=1)
    return

if __name__ == "__main__":
    import cv2
    import argparse
    import torchvision
    from pathlib import Path

    from dataset.utils import np2torch
    from colormap import apply_colormap, dxy_colormap

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--model", type=str, default="HITNet")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--output", default="./")
    args = parser.parse_args()

    model = PredictModel(**vars(args)).eval()
    ckpt = torch.load(args.ckpt)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.model.load_state_dict(ckpt)
    model.cuda()
    left_images = []
    right_images = []
    root_len = len(args.images)
    if os.path.isdir(args.images):
        paths = Walk(args.images, ['jpg', 'png', 'jpeg'])
        print(paths)
        for image_name in paths:
            if "left" in image_name or "cam0" in image_name:
                left_images.append(image_name)
            elif "right" in image_name or "cam1" in image_name:
                right_images.append(image_name)
    else:
        print("need --images for input images' dir")
        assert 0
    for lp, rp in zip(left_images, right_images):
        if lp[root_len:][0] == '/':
            op = os.path.join(args.output,lp[root_len+1:])
        else:
            op = os.path.join(args.output,lp[root_len:])
        MkdirSimple(op)
        predict(model, lp, rp, args.width, op, args.output)
        print("output: {}".format(op))
