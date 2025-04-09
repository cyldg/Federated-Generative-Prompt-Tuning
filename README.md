# Federated-Generative-Prompt-Tuning
This repo is the official implementation of "Fed-GPT: A Privacy-Preserving Federated Prompt Tuning Framework for Cross-Hospital Disease Diagnosis in Low-Resource Settings"

## Abstract
While federated medical AI has significantly improved medical data privacy preservation in cross-hospital collaborations, it still faces several challenges, including communication overhead, data scarcity, and heterogeneity. Foundation models (FMs) have emerged as a promising solution for their strong transfer potential in medical AI. Here, we propose a federated generative prompt tuning framework (Fed-GPT) to facilitate privacy-preserving collaborative disease diagnosis. Unlike the traditional federated model training, Fed-GPT enables cross-hospital prompt generator training, which realizes both communication-efficient global aggregation and local training on low-resource heterogeneous data by exploiting the power of FMs. The trained prompt generator could provide customized prompts for each patient’s medical sample, which assist FMs with personalized disease diagnosis. In addition, model inversion attacks fail to efficiently reconstruct input samples in Fed-GPT, representing a patient’s privacy guarantee. Extensive experimental results demonstrate that Fed-GPT outperforms baseline and full fine-tuning methods for cross-hospital disease diagnosis, particularly in low-resource settings.

## Acknowledgement

* [segment-anything](https://github.com/facebookresearch/segment-anything)
* [Finetune_segment_anything_tutorial](https://github.com/xzyun2011/finetune_segment_anything_tutorial)
* [invertinggradients](https://github.com/JonasGeiping/invertinggradients)
