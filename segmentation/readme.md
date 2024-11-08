## Supported image encoder backbone
```
sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "vit_b_vpt":build_sam_vit_b_vpt,
    "vit_b_gpt":build_sam_vit_b_gpt,
    "vit_b_vptd":build_sam_vit_b_vptd,
    "vit_b_gptd":build_sam_vit_b_gptd,
}
```

## Script example

### For training:
```
python fedgpt_segment.py --model_type experiment method --dataset polyp/prostate --data your dataset path --save_dir path to save checkpoint --data_len the number of training sample --prompt prompt length  --iterations 30 --epochs 2 
```
### For visualization:
```
python predict_show.py --model_type experiment method --dataset polyp/prostate --data your dataset pat --save_dir path to save --prompt prompt length --decoder-weights your weight path
```
