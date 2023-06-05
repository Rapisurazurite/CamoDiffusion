[//]: # (>📋  A template README.md for code accompanying a Machine Learning paper)

# CamoDiffusion

This repository is the official implementation of [CamoDiffusion: Camouflaged Object Detection via
Conditional Diffusion Models](). 

Our implementation is based on the denoising diffusion repository from <a href="https://github.com/lucidrains/denoising-diffusion-pytorch">lucidrains</a>, which is a PyTorch implementation of <a href="https://arxiv.org/abs/2006.11239">DDPM</a>.

And we provide our pretrained weight and inference result in release.

[//]: # (>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials)

## Requirements
- python == 3.9
- cuda == 11.3

To install requirements:

```setup
pip install -r requirements.txt
```

[//]: # (>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...)

## Dataset
[COD (Camouflaged Object Detection) Dataset](https://github.com/lartpang/awesome-segmentation-saliency-dataset#camouflaged-object-detection-cod)

## Training

To train the model(s) in the paper, run this command:

```shell
accelerate launch train.py --config config/camoDiffusion_352x352.yaml --num_epoch=150 --batch_size=32 --gradient_accumulate_every=1
```
And then finetune it to 384 size:
```shell
accelerate launch train.py --config config/camoDiffusion_384x384.yaml --num_epoch=20 --batch_size=28 --gradient_accumulate_every=1 --pretrained model_352/model-best.pt --lr_min=0 --set optimizer.params.lr=1e-5
```

[//]: # (>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.)

## Evaluation
To test a model, run [sample.py](sample.py) with the desired model on different datasets:
```shell
accelerate launch sample.py \
  --config config/camoDiffusion_384x384.yaml \
  --results_folder ${RESULT_SAVE_PATH} \
  --checkpoint ${CHECKPOINT_PATH} \
  --num_sample_steps 10 \
  --target_dataset CAMO \
  --time_ensemble
```

For ease of use, we create a [eval.sh](scripts%2Feval.sh) script and a use case in the form of a shell script eval.sh.
You can edit the script to change the parameters you want to test.

```shell
bash scripts/eval.sh
```

[//]: # (>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results &#40;section below&#41;.)

[//]: # (>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. )


## Citation

[//]: # (>📋  Pick a licence and describe how to contribute to your code repository. )
```
@article{chen2023camodiffusion,
  title={CamoDiffusion: Camouflaged Object Detection via Conditional Diffusion Models},
  author={Chen, Zhongxi and Sun, Ke and Lin, Xianming and Ji, Rongrong},
  journal={arXiv preprint arXiv:2305.17932},
  year={2023}
}
```
