import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from pathlib import Path
import torch
from utils.general import get_random_prompts, mask2one_hot
import matplotlib.pyplot as plt
import numpy as np
from medpy.metric.binary import hd95
from utils.utils import *
np.random.seed(0)
from utils.general import read_xml
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory

def show_mask(mask, ax, cls):
    color = colors[int(cls)]
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)

def show_counters(temp,ax,image):
    temp = temp.astype(np.uint8)
    temp = temp.squeeze(0)
    contour, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contour = image.copy()
    cv2.drawContours(image_with_contour, contour, -1, (0, 255, 0), 2)  # 绿色边框
    ax.imshow(image_with_contour)

def show_box(box, ax):

    x0, y0 = box[0]-5, box[1]-5
    w, h = box[2] - box[0] +10, box[3] - box[1] +10
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0, 0, 0, 0), lw=2))

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def main(opt,domain):
    sam_checkpoint = opt.sam_weights
    model_type = opt.model_type
    device = f"cuda:{opt.device}"
    new_decoder_path = opt.decoder_weights
    if new_decoder_path:
        assert os.path.exists(new_decoder_path), f"{new_decoder_path} not exist"
    finetuned = False if not os.path.exists(new_decoder_path) else True

    save_dir = opt.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint,prompt_length = opt.prompt)
    if finetuned:
        # update decoder weights
        state_dict = sam.state_dict()
        new_decoder_dict = torch.load(new_decoder_path, map_location=state_dict[list(state_dict.keys())[0]].device)
        state_dict.update(new_decoder_dict)
        res = sam.load_state_dict(state_dict, strict=False)
        print(f"load res: {res}")

    sam.to(device=device)
    predictor = SamPredictor(sam)

    data_folder = opt.data 


    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    ## TODO: change  img_lists len

    id_path = [x for x in os.listdir(os.path.join(data_folder, domain, 'images'))
                        if x.lower().endswith(('.png', '.jpg', '.jpeg'))]
    dice = []
    iou = []
    hd = []
    for id in id_path:
        image = cv2.imread(os.path.join(data_folder, domain, 'images', id))
        # image = image[..., ::-1] ## RGB to BGR
        mask = cv2.imread(os.path.join(data_folder, domain, 'masks', id), cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)
        gt = np.array(mask)
        (foreground_points, background_points), bbox = get_random_prompts(gt, 1)
        bboxes = bbox
        print(bboxes)
        save_name = f"{id}_finetuned.png" if finetuned else f"{id}.png"
        save_path = os.path.join(save_dir, save_name)
        image = image[...,::-1]
        predictor.set_image(image, "RGB")
        input_boxes = torch.tensor(bboxes, device=device)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        masks = masks.cpu()
        masks  = masks.numpy()
        plt.figure(figsize=(5, 5))
        print(np.unique(masks[0],return_counts = True))
        show_counters(masks[0], plt.gca(),image)
        show_box(bboxes, plt.gca())
        plt.axis('off')
        dice.append(dice_coef(gt,masks[0]))
        iou.append(iou_coef(gt,masks[0]))
        hd.append(hd95(masks[0][0], gt, voxelspacing=1.0, connectivity=1))

        print(f"save to: {save_path}")
        plt.savefig(save_path)
        plt.close()
    dice_mean = np.mean(dice)
    iou_mean = np.mean(iou)
    hd_mean = np.mean(hd)
    print(dice_mean,iou_mean,hd_mean)
    return dice_mean,iou_mean,hd_mean



def parse_opt(known=False):

    parser = argparse.ArgumentParser()
    parser.add_argument('--sam-weights', '--w', type=str, default='/home/yuliang_chen/few-shot-self-prompt-SAM-main/checkpoints/sam_vit_b_01ec64.pth', help='original sam weights path')
    parser.add_argument('--model-type', '--type', type=str, default='vit_b_vpt', help='sam model type: vit_b, vit_l, vit_h')
    parser.add_argument('--decoder-weights', '--decoder', type=str, default=ROOT / "weights/sam_decoder_fintune_pointbox.pth", help='finetuned decoder weights path')
    parser.add_argument('--data', type=str, default=ROOT /'data', help='dataset path')
    parser.add_argument('--dataset', type=str, default='', help='dataset')
    parser.add_argument('--save_dir', default=ROOT / 'fed/predict_test', help='path to save checkpoint')
    parser.add_argument('--device', default='0', help='cuda device only one, 0 or 1 or 2...')
    parser.add_argument('--prompt',type=int,default=1)
    return parser.parse_known_args()[0] if known else parser.parse_args()
    
if __name__ == '__main__':
    opt = parse_opt()
    colors = np.hstack((np.random.random((21, 3)), np.ones((21, 1)) * 0.6))
    if opt.dataset == "prostate":
        domain = ['Domain1_test', 'Domain2_test', 'Domain3_test', 'Domain4_test', 'Domain5_test', 'Domain6_test']
    elif opt.dataset == "polyp":
        domain = ['CVC-300_test','CVC-ClinicDB_test','CVC-ColonDB_test','ETIS-LaribPolypDB_test','Kvasir_test']
    
    dd = []
    ii = []
    pp = []
    for i in range(len(domain)):
        d,io,p = main(opt,domain[i])
        dd.append(d)
        ii.append(io)
        pp.append(p)
    for i in range(len(domain)):
        print(domain[i],dd[i],ii[i],pp[i])
    print(sum(dd)/len(dd),sum(ii)/len(ii),sum(pp)/len(pp))
