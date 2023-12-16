# ************************************************************************************************* #
# test
# (1) natural
python test_CECF.py --config configs/dim8_150k_bs6_NoTransBlock_SSIM.yaml \
                    --input_folder ../demo_dataset/your_test_images/images/ \
                    --guide_path ../guidance/guide_natural/ \
                    --output_folder ../results/CECF/natural/ \
                    --checkpoint ../results/gen_00000000.pt \
                    --device cuda:0 \
                    --subfolder_prefix natural_