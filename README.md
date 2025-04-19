# Federated-Generative-Prompt-Tuning
This repo is the official implementation of "Fed-GPT: A Federated Prompt Tuning Framework for Cross-Hospital Disease Diagnosis in Low-Resource Settings"

## Abstract
Federated medical AI revolutionizes the coordination of cross-hospital medical data, while communication cost, data scarcity and heterogeneity still limit its application in practical scenarios.
The emergence of foundation models (FMs) provides an opportunity to address these challenges due to their excellent generalization ability and efficient adaptation to different downstream tasks.
Here, we present Fed-GPT, a general and efficient federated prompt tuning framework for cross-hospital disease diagnosis in low-resource settings.
Unlike traditional federated learning approaches, Fed-GPT collaboratively trains a prompt generator that leverages the power of FMs to achieve communication-efficient global aggregation and robust local adaptation. The trained prompt generator produces customized prompts for each patient sample, thereby enabling  personalized disease diagnosis.
Across polyp and prostate segmentation tasks, Fed-GPT achieves 93.46% and 93.97% average dice similarity while reducing 92.8% trainable params compared to the traditional FedAvg method.
Our method achieves faster convergence on both classification and segmentation tasks, and consistently outperforms baseline methods across various low-resource scenarios.
Fed-GPT facilitates personalized cross-hospital disease diagnosis with minimal communication overhead, achieving precise AI-assisted diagnostics even in resource-limited clinical settings.

## Acknowledgement

* [segment-anything](https://github.com/facebookresearch/segment-anything)
* [Finetune_segment_anything_tutorial](https://github.com/xzyun2011/finetune_segment_anything_tutorial)
* [invertinggradients](https://github.com/JonasGeiping/invertinggradients)
