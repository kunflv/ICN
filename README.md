# Beyond Post-Hoc Explanations: An Ante-Hoc Interpretable Network with Class-Specific Cross-Attention
This work proposes an Interpretable Classification Network (ICN) that learns class-specific query representations and generates intrin- sic attribution maps via cross-attention weights.

<img width="1758" height="437" alt="image" src="https://github.com/user-attachments/assets/e1e79921-defd-4cf3-8502-c09a4fbe6ab5" />

## Fine-tune models and results

[INTR](https://huggingface.co/imageomics/INTR) on [DETR-R50](https://github.com/facebookresearch/detr) backbone, classification performance, and fine-tuned models on different datasets.


| Dataset | acc@1 | acc@5 | Model |
|----------|----------|----------|----------|
| [Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) | 93.5 | 97.8 |  [trained weights](https://pan.baidu.com/s/1IuBv1SiWpv0KkCQnWJ6Z2Q)  `passwd:6r88`|

# Requirements

```
cuda == 11.3
pytorch == 2.3.1
python == 3.8
cython
submitit
torch>=1.5.0
torchvision>=0.6.0
opencv-python
matplotlib
transformers
timm == 0.9.0
```

## Data Preparation
Follow the below format for data.
```
datasets
├── dataset_name
│   ├── train
│   │   ├── class1
│   │   │   ├── img1.jpeg
│   │   │   ├── img2.jpeg
│   │   │   └── ...
│   │   ├── class2
│   │   │   ├── img3.jpeg
│   │   │   └── ...
│   │   └── ...
│   └── val
│       ├── class1
│       │   ├── img4.jpeg
│       │   ├── img5.jpeg
│       │   └── ...
│       ├── class2
│       │   ├── img6.jpeg
│       │   └── ...
│       └── ...
```

## ICN Evaluation
To evaluate the performance of ICN on the _Aircraft_ dataset, execute the below command. ICN checkpoints are available at Fine-tune model and results.

```
CUDA_VISIBLE_DEVICES=0 python ./main.py --eval --resume <path/to/intr_checkpoint_aircraft_detr_r50.pth> --dataset_path <path/to/datasets> --dataset_name <dataset_name>
```
## ICN Interpretation

To generate visual representations of the ICN's interpretations, execute the provided command below. This command will present the interpretation for a specific class with the index <class_number>. By default, it will display interpretations from all attention heads. To focus on interpretations associated with the top queries labeled as top_q as well, set the parameter sim_query_heads to 1. Use a batch size of 1 for the visualization.

```
python -m tools.visualization --eval --resume <path/to/intr_checkpoint_aircraft_detr_r50.pth> --dataset_path <path/to/datasets> --dataset_name <dataset_name> --class_index <class_number>
```

## ICN Training
To prepare ICN for training, use the pretrained model [DETR-R50](https://github.com/facebookresearch/detr). To train for a particular dataset, modify '--num_queries' by setting it to the number of classes in the dataset. Within the INTR architecture, each query in the decoder is assigned the task of capturing class-specific features, which means that every query can be adapted through the learning process. Consequently, the total number of model parameters will grow in proportion to the number of classes in the dataset. To train INTR on a single GPU, execute the command below.

```
CUDA_VISIBLE_DEVICES=0 python ./main.py --finetune <path/to/detr-r50-e632da11.pth> --dataset_path <path/to/datasets> --dataset_name <dataset_name> --num_queries <num_of_classes>
```
## Acknowledgment
This code is based on the DEtection TRansformer [(DETR)](https://github.com/facebookresearch/detr) and Interpretable Transformer [(INTR)](https://github.com/Imageomics/INTR) methods.

Thanks for their great works.
