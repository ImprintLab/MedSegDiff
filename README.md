


  # MedSegDiff: Medical Image Segmentation with Diffusion Model
<img align="left" width="150" height="150" src="https://github.com/ImprintLab/MedSegDiff/blob/master/cover_image.png"> 
MedSegDiff is a Diffusion Probabilistic Model (DPM) based framework for the Segmentation and Reconstruction of organs/tissues from the medical images.

The algorithm is elaborated on our paper

[MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model](https://arxiv.org/abs/2211.00611) 

[MedSegDiff-V2: Diffusion based Medical Image Segmentation with Transformer](https://arxiv.org/pdf/2301.11798.pdf).



## A Quick Overview 

|<img align="left" width="480" height="170" src="https://github.com/WuJunde/MedSegDiff/blob/master/framework.png">|<img align="right" width="450" height="270" src="https://github.com/WuJunde/MedSegDiff/blob/master/frameworkv2.png">|
|:--:|:--:| 
| **MedSegDiff-V1** | **MedSegDiff-V2** |

## News
- [TOP] Join in our [Discord](https://discord.gg/CQQWtaQgRS) to ask questions and discuss with others.
- 22-11-30. This project is still quickly updating. Check TODO list to see what will be released next.
- 22-12-03. BraTs2020 bugs fixed. Example case added.
- 22-12-15. Fix multi-gpu distributed training.
- 22-12-16. DPM-Solver ✖️ MedSegDiff DONE 🥳 Now [DPM-Solver](https://github.com/LuChengTHU/dpm-solver) is avaliable in MedsegDiff. Enjoy its lightning-fast sampling (1000 steps ❌ 20 steps ⭕️) by setting ``--dpm_solver True``. 
- 22-12-23. Fixed some bugs of DPM-Solver.
- 23-01-31. MedSegDiff-V2 will be avaliable soon 🥳 . Check our paper [MedSegDiff-V2: Diffusion based Medical Image Segmentation with Transformer](https://arxiv.org/abs/2301.11798) first.
- 23-02-07. Optimize workflow in BRATS sampling. Add dataloader for processing raw 3D BRATS data.
- 23-02-11. Fix bugs 3D BRATS data training bugs, [issue 31](https://github.com/WuJunde/MedSegDiff/issues/31).
- 23-03-04. Paper [MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model](https://arxiv.org/abs/2211.00611) has been officially accepted by MIDL 2023 🥳
- 23-04-11. A new version based on the v2 framework has been released 🥳. It's more accurate, stable, and domain-adaptable than the previous version, while still not hogging too much of your resources. We've also fixed up a bunch of small things, like the requirement.txt and isic csv files. Huge thanks to all of you who reported issues, you really helped us a lot 🤗. btw, it will run the new version by default. Add "--version 1" if you want run the previous version. 
- 23-04-12. Added a simple evaluation file for isic dataset (script/segmentation_env). Usage: ``python scripts/segmentation_env.py --inp_pth *folder you save prediction images* --out_pth *folder you save ground truth images*``
- 23-12-05. Paper [MedSegDiff-V2: Diffusion based Medical Image Segmentation with Transformer](https://arxiv.org/abs/2301.11798) has been officially accepted by AAAI 2024 🥳
- 24-10-29. MedSegDiff-V2 has been selected as AAAI-24 Most Influential Paper 🥳

## Requirement

``pip install -r requirement.txt``

## Example Cases
### Melanoma Segmentation from Skin Images
1. Download ISIC dataset from https://challenge.isic-archive.com/data/. Your dataset folder under "data" should be like:

~~~
data
|   ----ISIC
|       ----Test
|       |   |   ISBI2016_ISIC_Part1_Test_GroundTruth.csv
|       |   |   
|       |   ----ISBI2016_ISIC_Part1_Test_Data
|       |   |       ISIC_0000003.jpg
|       |   |       .....
|       |   |
|       |   ----ISBI2016_ISIC_Part1_Test_GroundTruth
|       |           ISIC_0000003_Segmentation.png
|       |   |       .....
|       |           
|       ----Train
|           |   ISBI2016_ISIC_Part1_Training_GroundTruth.csv
|           |   
|           ----ISBI2016_ISIC_Part1_Training_Data
|           |       ISIC_0000000.jpg
|           |       .....
|           |       
|           ----ISBI2016_ISIC_Part1_Training_GroundTruth
|           |       ISIC_0000000_Segmentation.png
|           |       .....
~~~

    
2. For training, run: ``python scripts/segmentation_train.py --data_name ISIC --data_dir *input data direction* --out_dir *output data direction* --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 8``

3. For sampling, run: ``python scripts/segmentation_sample.py --data_name ISIC --data_dir *input data direction* --out_dir *output data direction* --model_path *saved model* --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --num_ensemble 5``

4. For evaluation, run ``python scripts/segmentation_env.py --inp_pth *folder you save prediction images* --out_pth *folder you save ground truth images*``


In default, the samples will be saved at `` ./results/`` 
### Brain Tumor Segmentation from MRI
1. Download BRATS2020 dataset from https://www.med.upenn.edu/cbica/brats2020/data.html. Your dataset folder should be like:
~~~
data
└───training
│   └───slice0001
│       │   brats_train_001_t1_123_w.nii.gz
│       │   brats_train_001_t2_123_w.nii.gz
│       │   brats_train_001_flair_123_w.nii.gz
│       │   brats_train_001_t1ce_123_w.nii.gz
│       │   brats_train_001_seg_123_w.nii.gz
│   └───slice0002
│       │  ...
└───testing
│   └───slice1000
│       │  ...
│   └───slice1001
│       │  ...
~~~
    
2. For training, run: ``python scripts/segmentation_train.py --data_dir (where you put data folder)/data/training --out_dir output data direction --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 8``

3. For sampling, run: ``python scripts/segmentation_sample.py --data_dir (where you put data folder)/data/testing --out_dir output data direction --model_path saved model --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --num_ensemble 5``


### Other Examples
...
### Run on  your own dataset
It is simple to run MedSegDiff on the other datasets. Just write another data loader file following `` ./guided_diffusion/isicloader.py`` or `` ./guided_diffusion/bratsloader.py``.  Welcome to open issues if you meet any problem. It would be appreciated if you could contribute your dataset extensions. Unlike natural images, medical images vary a lot depending on different tasks. Expanding the generalization of a method requires everyone's efforts.
## Suggestions for Hyperparameters and Training
To train a fine model, i.e., MedSegDiff-B in the paper, set the model hyperparameters as:
~~~
--image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 
~~~
diffusion hyperparameters as:
~~~
--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False
~~~
To speed up the sampling:
~~~
--diffusion_steps 50 --dpm_solver True 
~~~
run on multiple GPUs:
~~~
--multi-gpu 0,1,2 (for example)
~~~
training hyperparameters as:
~~~
--lr 5e-5 --batch_size 8
~~~
and set ``--num_ensemble 5`` in sampling.

Run about 100,000 steps in training will be converged on most of the datasets. Note that although loss will not decrease in most of the later steps, the quality of the results are still improving. Such a process is also observed on the other DPM applications, like image generation. Hope someone smart can tell me why🥲.

I will soon publish its performance under smaller batch size (suitable to run on 24GB GPU) for the need of comparison🤗.

A setting to unleash all its potential is (MedSegDiff++):
~~~
--image_size 256 --num_channels 512 --class_cond False --num_res_blocks 12 --num_heads 8 --learn_sigma True --use_scale_shift_norm True --attention_resolutions 24 
~~~
Then train it with batch size ``--batch_size 64`` and sample it with ensemble number ``--num_ensemble 25``.

## TODO LIST

- [x] Fix bugs in BRATS. Add BRATS example.
- [x] Release REFUGE and DDIT dataloaders and examples
- [x] Speed up sampling by DPM-solver
- [ ] Inference of depth
- [x] Fix bugs in Multi-GPU parallel
- [x] Sample and Vis in training
- [x] Release pre processing and post processing
- [x] Release evaluation
- [x] Deploy on HuggingFace
- [x] configuration

## Thanks
Code copied a lot from [openai/improved-diffusion](https://github.com/openai/improved-diffusion), [WuJunde/ MrPrism](https://github.com/WuJunde/MrPrism), [WuJunde/ DiagnosisFirst](https://github.com/WuJunde/DiagnosisFirst), [LuChengTHU/dpm-solver](https://github.com/LuChengTHU/dpm-solver), [JuliaWolleb/Diffusion-based-Segmentation](https://github.com/JuliaWolleb/Diffusion-based-Segmentation), [hojonathanho/diffusion](https://github.com/hojonathanho/diffusion), [guided-diffusion](https://github.com/openai/guided-diffusion), [bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets](https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets), [nnUnet](https://github.com/MIC-DKFZ/nnUNet), [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch)
## Cite
Please cite
~~~
@inproceedings{wu2023medsegdiff,
  title={MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model},
  author={Wu, Junde and FU, RAO and Fang, Huihui and Zhang, Yu and Yang, Yehui and Xiong, Haoyi and Liu, Huiying and Xu, Yanwu},
  booktitle={Medical Imaging with Deep Learning},
  year={2023}
}
~~~

~~~
@article{wu2023medsegdiff,
  title={MedSegDiff-V2: Diffusion based Medical Image Segmentation with Transformer},
  author={Wu, Junde and Ji, Wei and Fu, Huazhu and Xu, Min and Jin, Yueming and Xu, Yanwu}
  journal={arXiv preprint arXiv:2301.11798},
  year={2023}
}
~~~




