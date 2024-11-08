# Federated-Generative-Prompt-Tuning
This repo is the official implementation of "Privacy-Preserving Heterogeneous Medical Image Analysis across Hospitals via Federated Large-Scale Foundation Model Prompt Tuning"

## Abstract
While medical AI has made significant advancements in assisting diagnosis and personalized treatment, it still faces challenges including data privacy, scarcity, heterogeneity, and communication overhead in cross-hospital collaborations. Federated learning (FL) has emerged as a promising solution, allowing hospitals to train collaboratively without sharing private data.  Besides, foundation models have gained prominence in medical AI, demonstrating significant potential in patient health analysis and treatment planning.
Here, we propose the federated large-scale foundation model prompt tuning framework (Fed-GPT) to facilitate privacy-preserving collaborative medical image analysis.
The framework introduces prompt tuning into FL and primarily consists of a prompt generator that integrates general knowledge from the global prompt with personalized information from individual samples, thereby generating customized prompts for each input image.
Extensive experimental results demonstrate that Fed-GPT outperforms baseline and full fine-tuning methods in cross-hospital scenarios while exhibiting enhanced privacy preservation capabilities.

## Acknowledgement

* [segment-anything](https://github.com/facebookresearch/segment-anything)
* [Finetune_segment_anything_tutorial](https://github.com/xzyun2011/finetune_segment_anything_tutorial)
* [invertinggradients](https://github.com/JonasGeiping/invertinggradients)
