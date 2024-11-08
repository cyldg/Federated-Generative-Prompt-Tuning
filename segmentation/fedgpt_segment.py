import argparse
import os
import torch
import torch.nn as nn
import numpy as np
# set seeds
torch.cuda.is_available()
torch.manual_seed(0)
np.random.seed(0)
import cv2
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from utils.custom_dataset import Dataset
from utils.loss import FocalLoss, soft_dice_loss
from utils.general import get_random_prompts, mask2one_hot
import copy
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory

def communication_fedavg(server_model, models):
    with torch.no_grad():
        global_params = [torch.zeros_like(param) for param in models[0].parameters() if param.requires_grad]

        for client_model in models:
            for i, client_param in enumerate(client_model.parameters()):
                if client_param.requires_grad:
                    global_params[i].data += client_param.data / len(models)

        for global_param, server_param in zip(global_params, server_model.parameters()):
            if server_param.requires_grad:
                server_param.data = global_param.data

        for client_model in models:
            for global_param, model_param in zip(global_params, client_model.parameters()):
                if model_param.requires_grad:
                    model_param.data = global_param.data

def train(opt,dataloader,sam,scheduler,device,optimizer,BCEseg):
    sam.train()
    predictor = SamPredictor(sam)
    model_transform = ResizeLongestSide(sam.image_encoder.img_size)
    point_box = (opt.point_prompt and opt.box_prompt)
    epoch_loss = 0
    for idx, (images, gts,id) in enumerate(tqdm(dataloader)):
        for i in range(images.shape[0]):
            image = images[i] # h,w,c np.uint8 rgb
            original_size = image.shape[:2] ## h,w
            input_size = model_transform.get_preprocess_shape(image.shape[0], image.shape[1],
                                                                sam.image_encoder.img_size)  ##h,w
            gt = gts[i].copy() #h,w labels [0,1,2,..., classes-1]
            predictions = []
            ## freeze image encoder
            with torch.no_grad():
                # gt_channel = gt[:, :, cls]
                predictor.set_image(image, "RGB")
                image_embedding = predictor.get_image_embedding()

            (foreground_points, background_points), bbox = get_random_prompts(gt,1)
            # if the model can't generate any sparse prompts
            if len(foreground_points) == 0:
                print(f"======== zero points =============")
                continue
            if not opt.point_prompt:
                points = None
            else:
                all_points = np.concatenate((foreground_points, background_points), axis=0)
                all_points = np.array(all_points)
                point_labels = np.array([1] * foreground_points.shape[0] + [0] * background_points.shape[0], dtype=int)
                ## image resized to 1024, points also
                all_points = model_transform.apply_coords(all_points, original_size)

                all_points = torch.as_tensor(all_points, dtype=torch.float, device=device)
                point_labels = torch.as_tensor(point_labels, dtype=torch.float, device=device)
                all_points, point_labels = all_points[None, :, :], point_labels[None, :]
                points = (all_points, point_labels)

            if not opt.box_prompt:
                box_torch=None
            else:
                ## preprocess bbox
                box = model_transform.apply_boxes(bbox, original_size)
                box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
                box_torch = box_torch[None, :]
            ## if both, random drop one for better generalization ability
            if point_box and np.random.random()<0.5:
                if np.random.random()<0.25:
                    points = None
                elif np.random.random()>0.75:
                    box_torch = None
            ## freeze prompt encoder
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points = points,
                    boxes = box_torch,
                    # masks=mask_predictions,
                    masks=None,
                )
            ## predicted masks, three level
            mask_predictions, scores = sam.mask_decoder(
                image_embeddings=image_embedding.to(device),
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
            )
            # Choose the model's best mask
            mask_input = mask_predictions[:, torch.argmax(scores),...].unsqueeze(1)
            with torch.no_grad():
                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=points,
                    boxes=box_torch,
                    masks=mask_input,
                )
            ## predict a better mask, only one mask
            mask_predictions, scores = sam.mask_decoder(
                image_embeddings=image_embedding.to(device),
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            best_mask = sam.postprocess_masks(mask_predictions, input_size, original_size)
            predictions.append(best_mask)

            predictions = torch.cat(predictions, dim=1)
        gts = torch.from_numpy(gts).unsqueeze(1) ## BxHxW ---> Bx1xHxW
        gts = gts.to(torch.float32).to(device)

        predictions = torch.sigmoid(predictions)
        # #loss = seg_loss(predictions, gts_onehot)
        loss = BCEseg(predictions, gts)
        loss_dice = soft_dice_loss(predictions, gts, smooth = 1e-5, activation='none')
        loss = loss + loss_dice

        # print(f"epoch: {epoch} at idx:{idx} --- loss: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss /= idx
    scheduler.step()
    return epoch_loss



def main(opt):
    # Create a dataset
    data_folder = opt.data  # define your dataset location here
    if opt.dataset == "prostate":
        domain = ['Domain1_train', 'Domain2_train', 'Domain3_train', 'Domain4_train', 'Domain5_train', 'Domain6_train']
    elif opt.dataset == "polyp":
        domain = ['CVC-300_train','CVC-ClinicDB_train','CVC-ColonDB_train','ETIS-LaribPolypDB_train','Kvasir_train']
    dataset = []
    dataloader = []
    batch_size = opt.batch_size  ## must be 1
    for i in range(len(domain)):
        temp = PolypDataset(domain_name = domain[i],base_dir=data_folder)
        random_sample_indices = random.sample(list(range(len(temp))),opt.data_len)
        temp = torch.utils.data.Subset(temp,random_sample_indices)
        dataset.append(temp)
        print('training dataset size:',len(dataset[i]))
        dataloader.append(DataLoader(dataset[i], batch_size=batch_size, shuffle=True, collate_fn=PolypDataset.custom_collate))

    # original parameters
    sam_checkpoint = opt.sam_weights
    model_type = opt.model_type
    device = f"cuda:{opt.device}"

    save_dir = opt.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    num_epochs = opt.epochs

    # model initialization
    sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint,prompt_length=opt.prompt)
    # for name, param in sam_model.named_parameters():
    #     if param.requires_grad==True:
    #         print(f"Parameter Name: {name}")
    #         # print(f"Parameter Size: {param.size()}")
    #         print(f"Total Elements: {param.numel()}")
    #         print(f"Requires Grad: {param.requires_grad}")
    #         print("-" * 50)
    for param in sam_model.parameters():
        param.requires_grad = False
    sam_model.image_encoder.Prompt_Tokens.requires_grad = True
    for name, param in sam_model.named_parameters():
        if 'decoder' in name:
            param.requires_grad = True
    client_num  = len(domain)
    client_models = [copy.deepcopy(sam_model).to(device) for idx in range(client_num)]
    for model in client_models:
        sum = 0
        for name,param in model.named_parameters():
            if param.requires_grad==True:
                sum+=param.numel()
        print(sum)
    #train
    sam_model.to(device=device)
    print(f"finished loading sam")
    

    # optimizer and scheduler
    lr = 1e-4
    weight_decay = 5e-4
    optimizer = [torch.optim.AdamW(client_models[idx].parameters(), lr=lr, weight_decay=weight_decay)for idx in range(client_num)]
    scheduler = []
    for idx in range(client_num):
        scheduler.append(CosineAnnealingLR(optimizer[idx], T_max=num_epochs, eta_min=1e-7))
    ## loss
    BCEseg = nn.BCELoss().to(device)
    
    for rounds in range(opt.iterations):
        print(f"strat training Round{rounds}")
        for idx in range(client_num):
            for epoch in range(num_epochs):
                train_loss= train(opt,dataloader[idx],client_models[idx],scheduler[idx],device,optimizer[idx],BCEseg)
                # print(f"Domain {domain[idx]} | Epoch {epoch} | Train loss {train_loss} | Train Dice {train_dice}")
                print(f"Domain {domain[idx]} | Epoch {epoch} | Train loss {train_loss} ")

        communication_fedavg(sam_model,client_models)

        mask_decoder_weighs = sam_model.mask_decoder.state_dict()
        mask_decoder_weighs = {f"mask_decoder.{k}": v for k,v in mask_decoder_weighs.items() }
        prompt_tokens_state = sam_model.image_encoder.Prompt_Tokens
        prompt_tokens_state = {f"image_encoder.Prompt_Tokens":  prompt_tokens_state}
        combined_state = {**mask_decoder_weighs, **prompt_tokens_state}
        torch.save(combined_state, os.path.join(save_dir, f'sam_{opt.model_type}_{opt.data_len}_{opt.prompt}_{opt.dataset}_{str(rounds)}.pth'))
        print("Saving weights, round: ", rounds)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--sam-weights', '--w', type=str, default='/sam_vit_b_01ec64.pth', help='original sam weights path')
    parser.add_argument('--model-type', '--type', type=str, default='vit_b_vpt', help='sam model type: vit_b, vit_l, vit_h')
    parser.add_argument('--data', type=str, default='', help='your dataset path')
    parser.add_argument('--dataset', type=str, default='', help='dataset')
    parser.add_argument('--prompt_type',type=str)
    parser.add_argument('--point-prompt', type=bool, default=True, help='use point prompt')
    parser.add_argument('--box-prompt', type=bool, default=True, help='use box prompt')
    parser.add_argument('--iterations',type = int, default=30)
    parser.add_argument('--epochs', type=int, default=2, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs, must be 1 for voc')
    parser.add_argument('--save_dir', default=ROOT / 'fed', help='path to save checkpoint')
    parser.add_argument('--device', default='0', help='cuda device only one, 0 or 1 or 2...')
    parser.add_argument('--data_len',type=int,default=30)
    parser.add_argument('--prompt',type=int,default=1)
    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)