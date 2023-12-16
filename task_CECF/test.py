import math
import sys
sys.path.append("..")
import torchvision.utils
from methods.CECF.utils import get_config
from methods.CECF.trainer_CECF import CECF_Trainer
import argparse
import torch
import os
from torchvision import transforms
from PIL import Image


class Option():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--device", type=str)
        self.parser.add_argument('--config', type=str, help="net configuration")
        self.parser.add_argument('--input_folder', type=str, help="input image path")
        self.parser.add_argument('--output_folder', type=str, help="output image path")
        self.parser.add_argument('--checkpoint', type=str, help="checkpoint of autoencoders")
        self.parser.add_argument('--guide_path', type=str, default='', help="style image path")
        self.parser.add_argument('--output_path', type=str, default='.')
        self.parser.add_argument("--subfolder_prefix", type=str)

    def start_parse(self):
        opts = self.parser.parse_args()
        return opts


def prepare():

    opts = Option().start_parse()
    config = get_config(opts.config)

    device = torch.device(opts.device)

    trainer = CECF_Trainer(config, device=device)

    state_dict = torch.load(opts.checkpoint, map_location=device)
    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])

    new_size = config['new_size']

    imgs_files = os.listdir(opts.input_folder)
    imgs_files = [img for img in imgs_files if img.endswith(".jpg") or img.endswith(".png")]
    guide_files = os.listdir(opts.guide_path)
    guide_files = [img for img in guide_files if img.endswith(".jpg") or img.endswith(".png")]

    transform = transforms.Compose([transforms.Resize(new_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    return trainer, opts, imgs_files, guide_files, transform, new_size, device


def mk_cur_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == "__main__":
    trainer, opts, imgs_files, guide_files, transform, new_size, device = prepare()
    trainer = trainer.to(device)
    trainer.eval()

    alpha_weights = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    # alpha_weights = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    mk_cur_dir(opts.output_path)
    with torch.no_grad():
        for img_C in imgs_files:

            # read the distorted image
            ori_img = Image.open(os.path.join(opts.input_folder, img_C)).convert('RGB')
            image = transform(ori_img).unsqueeze(0).to(device)

            #
            content, _, fix_color_code = trainer.gen_a.encode(image)

            for img_S in guide_files:
                cur_output_folder = opts.output_folder + opts.subfolder_prefix + img_C[:-4] + "_guide_" + img_S[:-4]
                mk_cur_dir(dir=cur_output_folder)

                # read guide img
                ori_guide_img = Image.open(os.path.join(opts.guide_path, img_S)).convert('RGB')
                guide_image = transform(ori_guide_img).unsqueeze(0).to(device)

                # forward
                guide_content, _, guide_color_code = trainer.gen_a.encode(guide_image)

                # clip it if you want
                # guide_color_code = guide_color_code.clamp(min=-3, max=3)

                for j in range(len(alpha_weights)):
                    alpha = alpha_weights[j]

                    # fuse the color code
                    fuse_color_code = (1 - alpha) * fix_color_code + alpha * guide_color_code
                    fuse_color_code = fuse_color_code / math.sqrt(alpha ** 2 + (1 - alpha) ** 2)

                    outputs = trainer.gen_a.decode_fix(content, fuse_color_code)
                    outputs = (outputs + 1) / 2.0
                    path = os.path.join(cur_output_folder, 'alpha_{}.jpg'.format(str(alpha)))
                    torchvision.utils.save_image(outputs, path)

                # save the original images
                ori_img.save(os.path.join(cur_output_folder, "distorted_" + img_C))
                ori_guide_img.save(os.path.join(cur_output_folder, "guide_" + img_S))

                # recon of guidance
                guide_recon = trainer.gen_a.decode_fix(guide_content, guide_color_code)
                guide_recon = (guide_recon + 1) / 2.0
                torchvision.utils.save_image(guide_recon, os.path.join(cur_output_folder, "guide_recon_" + img_S))
