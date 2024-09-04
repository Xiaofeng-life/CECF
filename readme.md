## The code for Underwater Organism Color Fine-tuning via Decomposition and Guidance

This paper has been accepted by AAAI2024.

If you have any quesions, feel free to contact me. My <b> E-mail </b> and <b> WeChat </b> can be found at my homepage: [<A HREF="https://xiaofeng-life.github.io/">Homepage</A>]

### 1. Data preparation for Enhancement Task
Download underwater imagery datasets from websites or papers. Follow the organization form below.
```
├── dataset_name
    ├── train
        ├── images
            ├── im1.jpg
            ├── im2.jpg
            └── ...
        ├── labels
            ├── im1.jpg
            ├── im2.jpg
            └── ...
    ├── val
        ├── images
            ├── im1.jpg
            ├── im2.jpg
            └── ...
        ├── labels
            ├── im1.jpg
            ├── im2.jpg
            └── ...
    ├── test
        ├── images
            ├── im1.jpg
            ├── im2.jpg
            └── ...
        ├── labels (if you have)
            ├── im1.jpg
            ├── im2.jpg
            └── ...
        ├── masks (if you have)
            ├── im1.jpg
            ├── im2.jpg
            └── ...
```

### 2. Data preparation Color Fine-tuning
Note: If you only need to perform color enhancement, you can ignore this step.

**Use the EUVP-scenes if you want fine-tuning color ability!!!**
**The diversity of UIEB and EUVP-imagenet are not enough. 
Besides, the images with less distortion are required. Therefore, the EUVP-scenes is needed !!!**


### 3. Train process
Put the config file in folder task_CECF/configs. For example task_CECF/configs/UIEB_3090_dim8_1m_bs6_NoTransBlock_SSIM.yaml

The batch_size must be greater than or equal 4.

```
cd task_CECF
python train_CECF_TransBlock.py --config configs/UIEB_3090_dim8_1m_bs6_NoTransBlock_SSIM.yaml --device cuda:7 --res_dir ../results/CECF/ --grad_acc 1
```

### 4. Test enhancement process
If you need to compare with our algorithm, execute the following command.

```
cd task_CECF 
python test_enhancement.py --config configs/dim8_150k_bs6_NoTransBlock_SSIM.yaml \
                    --input_folder ../demo_dataset/your_test_images/images/ \
                    --output_folder ../results/CECF/natural/ \
                    --checkpoint ../results/gen_00000000.pt \
                    --device cuda:0 
```

### 5. Test fine-tuning process

```
cd task_CECF 
python test_fine_tune.py --config configs/dim8_150k_bs6_NoTransBlock_SSIM.yaml \
                    --input_folder ../demo_dataset/your_test_images/images/ \
                    --guide_path ../guidance/guide_natural/ \
                    --output_folder ../results/CECF/natural/ \
                    --checkpoint ../results/gen_00000000.pt \
                    --device cuda:0 \
                    --subfolder_prefix natural_
```