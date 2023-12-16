import sys

sys.path.append("..")
import torchvision
import numpy as np
import os
import sys
import shutil
import argparse

import torch
import torch.backends.cudnn as cudnn

from metric.cal_metric import cal_batch_psnr, cal_batch_ssim, cal_batch_uiqm
from methods.CECF.trainer_CECF import CECF_Trainer
from dataloader_UWIE import get_train_val_loader
from methods.CECF.utils import get_config, write_2images, Timer, make_train_dir, write_metrics


class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--config', type=str, help='Path to the config file.')
        self.parser.add_argument("--resume", action="store_true")
        self.parser.add_argument("--res_dir", type=str, help="results dir")
        self.parser.add_argument("--device", type=str)
        self.parser.add_argument("--grad_acc", type=int,
                                 help="gradient accumulate. This is not need if you have a 24G-GPU")

    def parse(self):
        opts = self.parser.parse_args()
        return opts


def eval_on_validation(net, res_dir, device, val_loader, loader_name, if_save_cat, epo):
    """

    :param net:
    :param res_dir: results dir
    :param device:
    :param val_loader:
    :param loader_name:
    :param if_save_cat: cat img with label
    :param epo: epoch
    :return:
    """
    ssims = []
    psnrs = []
    uiqms = []
    with torch.no_grad():
        for data in val_loader:
            blur = data["blur"].to(device)
            clear = data["gt"].to(device)

            blur = blur * 2 - 1
            clear = clear * 2 - 1

            img_name = data["name"]

            enhanced_image = net.infer_fix_color(blur)
            enhanced_image = (enhanced_image + 1) / 2
            enhanced_image = torch.clamp(enhanced_image, min=0, max=1)

            blur = (blur + 1) / 2
            clear = (clear + 1) / 2

            y = clear.to(device)
            ssim1 = cal_batch_ssim(pred=enhanced_image.detach(), gt=y).item()
            psnr1 = cal_batch_psnr(pred=enhanced_image.detach(), gt=y)
            uiqm1 = cal_batch_uiqm(pred=enhanced_image.detach())
            ssims.append(ssim1)
            psnrs.append(psnr1)
            uiqms.append(uiqm1)

            for j in range(enhanced_image.size(0)):
                if if_save_cat:
                    out_cat = torch.cat((blur, enhanced_image, y), dim=3)
                    torchvision.utils.save_image(out_cat[j].unsqueeze(0),
                                                 os.path.join(res_dir, "cat_images", img_name[j]))

                torchvision.utils.save_image(enhanced_image[j].unsqueeze(0),
                                             os.path.join(res_dir, "last_images", img_name[j]))

    write_metrics(os.path.join(res_dir, "metrics", loader_name + "_metric.txt"),
                  epoch=epo, ssim=np.mean(ssims), psnr=np.mean(psnrs), uiqm=np.mean(uiqms))

    print(f"\nstep :{0} |ssim:{np.mean(ssims):.4f}| psnr:{np.mean(psnrs):.4f} | uiqm:{np.mean(uiqms):.4f}")
    return np.mean(ssims), np.mean(psnrs)

# Set the seed if you want. It can be set to 1111
def setup_seed(seed):
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# Load experiment setting
opts = TrainOptions().parse()
cudnn.benchmark = True
device = torch.device(opts.device)
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']

# Setup model and data loader
trainer = CECF_Trainer(config, device=device)
trainer.to(device)

train_loader_UIEB, val_loader_UIEB = get_train_val_loader(data_train=os.path.join(config["data_root"], "train"),
                                                          data_val=os.path.join(config["data_root"], "val"),
                                                          train_batch_size=config["batch_size"],
                                                          num_workers=config["num_workers"])
# images for display
train_display_pair = [train_loader_UIEB.dataset[i] for i in range(display_size)]
test_display_pair = [train_loader_UIEB.dataset[i] for i in range(display_size)]
train_display_images_a = torch.stack([train_display_pair[i]["blur"] for i in range(display_size)]).to(device)
train_display_images_b = torch.stack([train_display_pair[i]["gt"] for i in range(display_size)]).to(device)
test_display_images_a = torch.stack([test_display_pair[i]["blur"] for i in range(display_size)]).to(device)
test_display_images_b = torch.stack([test_display_pair[i]["gt"] for i in range(display_size)]).to(device)
torchvision.utils.save_image(train_display_images_a, "../a.jpg")
torchvision.utils.save_image(train_display_images_b, "../b.jpg")

# set for training
iterations = 0
res_dir = os.path.join(opts.res_dir, os.path.splitext(os.path.basename(opts.config))[0])
print("results dir is: ", res_dir)

# resume from existing
if opts.resume:
    checkpoint_directory = os.path.join(res_dir, "models")
    iterations, last_model_name = trainer.resume(checkpoint_directory, hyperparameters=config, new_device=device)
else:
    make_train_dir(res_dir=res_dir)

# config the results dir
shutil.copy(opts.config, os.path.join(res_dir, "configs", os.path.splitext(os.path.basename(opts.config))[0]) + ".yaml")
pth_dir = os.path.join(res_dir, "models")
sample_dir = os.path.join(res_dir, "sample_images")

# Training
while True:
    for it, data in enumerate(train_loader_UIEB):
        # Eval
        if (iterations + 1) % 1000 == 0:
            trainer.eval()
            eval_on_validation(net=trainer, res_dir=res_dir, device=device,
                               val_loader=val_loader_UIEB, loader_name="", if_save_cat=True, epo=iterations)
            trainer.save(pth_dir, iterations)
            trainer.train()
        images_a = data["blur"]   # distorted images
        images_b = data["gt"]     # label images
        trainer.update_learning_rate()
        images_a = images_a * 2 - 1
        images_b = images_b * 2 - 1
        images_a = images_a.to(device)
        images_b = images_b.to(device)

        with Timer("Elapsed time in update: %f"):
            # Main training code
            trainer.dis_update(images_a, images_b, config)
            if (iterations + 1) % opts.grad_acc == 0: # grad acc is set to 1
                trainer.dis_opt.step()
                trainer.dis_opt.zero_grad()
            trainer.gen_update(images_a, images_b, config)
            if (iterations + 1) % opts.grad_acc == 0:
                trainer.gen_opt.step()
                trainer.gen_opt.zero_grad()

        # print training process
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))

        # write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                test_image_outputs = trainer.sample(test_display_images_a, test_display_images_b)
                train_image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(test_image_outputs, display_size, sample_dir, 'test_%08d' % (iterations + 1))
            write_2images(train_image_outputs, display_size, sample_dir, 'train_%08d' % (iterations + 1))

        # sample current
        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(image_outputs, display_size, sample_dir, 'train_current')

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')
