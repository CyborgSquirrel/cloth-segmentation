import logging
import os
import pprint
import sys
import time
import traceback
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import infer as inferlib
from data.custom_dataset_data_loader import sample_data
from data.tomosynthesis_dataset import TomosynthesisDataset
from networks import U2NET
from options.base_options import Options
from utils.distributed import cleanup, set_seed, synchronize
from utils.saving_utils import load_checkpoint, save_checkpoints
from utils.tensorboard_utils import board_add_images

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(
    level=logging.INFO,
    filename=__file__[:-3] + ".log",
)


def options_printing_saving(opt):
    os.makedirs(opt.logs_dir, exist_ok=True)
    os.makedirs(opt.save_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(opt.save_dir, "checkpoints"), exist_ok=True)

    # Saving options in yml file
    option_dict = vars(opt)
    with open(os.path.join(opt.save_dir, "training_options.yml"), "w") as outfile:
        yaml.dump(option_dict, outfile)

    for key, value in option_dict.items():
        print(key, value)


def training_loop(opt):
    logging.info("starting training loop")
    
    if opt.cpu:
        device = torch.device("cpu")
        local_rank = 0
    else:
        if opt.distributed:
            local_rank = int(os.environ.get("LOCAL_RANK"))
            # Unique only on individual node.
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda:0")
            local_rank = 0
    
    
    u_net = U2NET(in_ch=3, out_ch=4)
    if opt.continue_train:
        u_net = load_checkpoint(u_net, opt.unet_checkpoint)
    u_net = u_net.to(device)
    u_net.train()

    if local_rank == 0:
        with open(os.path.join(opt.save_dir, "networks.txt"), "w") as outfile:
            print("<----U-2-Net---->", file=outfile)
            print(u_net, file=outfile)

    if opt.distributed:
        u_net = nn.parallel.DistributedDataParallel(
            u_net,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        print("Going super fast with DistributedDataParallel")

    # initialize optimizer
    optimizer = optim.Adam(
        u_net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
    )

    # dataset
    gen = torch.Generator()
    gen.manual_seed(opt.seed)
    dataset = TomosynthesisDataset(opt)
    dataset.load_image = True
    [dataset_train, dataset_val] = torch.utils.data.random_split(
        dataset,
        [1-opt.val_proportion, opt.val_proportion],
        generator=gen,
    )

    logging.info("training model on %s images", len(dataset_train))
    
    dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=opt.batchSize,
        pin_memory=True,  # TODO: only set to true if training on the gpu?
    )

    if local_rank == 0:
        writer = SummaryWriter(opt.logs_dir)
        print("Entering training loop!")

    # loss function
    weights = np.array([0.5, 1, 5, 5], dtype=np.float32)
    weights = torch.from_numpy(weights).to(device)
    loss_CE = nn.CrossEntropyLoss(weight=weights).to(device)

    pbar = range(opt.iter)
    get_data = sample_data(dataloader)

    start_time = time.time()
    
    # Main training loop
    for itr in pbar:
        data_batch = next(get_data)
        image_tensor = data_batch["image"]
        label_tensor = data_batch["label"]
        
        image_tensor = Variable(image_tensor.to(device))
        label_tensor = label_tensor.type(torch.long)
        label_tensor = Variable(label_tensor.to(device))
        
        d0, d1, d2, d3, d4, d5, d6 = u_net(image_tensor)
        
        loss0 = loss_CE(d0, label_tensor)
        loss1 = loss_CE(d1, label_tensor)
        loss2 = loss_CE(d2, label_tensor)
        loss3 = loss_CE(d3, label_tensor)
        loss4 = loss_CE(d4, label_tensor)
        loss5 = loss_CE(d5, label_tensor)
        loss6 = loss_CE(d6, label_tensor)
        
        total_loss = loss0 * 1.5 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

        for param in u_net.parameters():
            param.grad = None

        total_loss.backward()
        if opt.clip_grad != 0:
            nn.utils.clip_grad_norm_(u_net.parameters(), opt.clip_grad)
        optimizer.step()
        
        if local_rank == 0:
            # printing and saving work
            if itr % opt.print_freq == 0:
                logging.info(
                    "[step-{:08d}] [time-{:.3f}] [total_loss-{:.6f}]  [loss0-{:.6f}]".format(
                        itr, time.time() - start_time, total_loss, loss0
                    )
                )
                
                output_tensor = d0
                output_tensor = F.log_softmax(output_tensor, dim=1)
                output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
                output_tensor = torch.squeeze(output_tensor, dim=0)

                label_arr = label_tensor.cpu().numpy()
                output_arr = output_tensor.cpu().numpy()
                
                benign_metrics = inferlib.compute_metrics(
                    label_arr == 2,
                    output_arr == 2,
                )
                cancer_metrics = inferlib.compute_metrics(
                    label_arr == 3,
                    output_arr == 3,
                )
                logging.info("benign_metrics: iou=%s proportion=%s", benign_metrics.iou, benign_metrics.proportion)
                logging.info("cancer_metrics: iou=%s proportion=%s", cancer_metrics.iou, cancer_metrics.proportion)

            if itr % opt.image_log_freq == 0:
                d0 = F.log_softmax(d0, dim=1)
                d0 = torch.max(d0, dim=1, keepdim=True)[1]
                visuals = [[image_tensor, torch.unsqueeze(label_tensor, dim=1) * 85, d0 * 85]]
                board_add_images(writer, "grid", visuals, itr)

            writer.add_scalar("total_loss", total_loss, itr)
            writer.add_scalar("loss0", loss0, itr)

            if itr % opt.save_freq == 0:
                save_checkpoints(opt, itr, u_net)
                
        del d1, d2, d3, d4, d5, d6

    print("Training done!")
    if local_rank == 0:
        itr += 1
        save_checkpoints(opt, itr, u_net)


if __name__ == "__main__":
    opt = Options()

    if opt.distributed:
        if int(os.environ.get("LOCAL_RANK")) == 0:
            options_printing_saving(opt)
    else:
        options_printing_saving(opt)

    try:
        if opt.distributed:
            print("Initialize Process Group...")
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
            synchronize()

        set_seed(1000)
        training_loop(opt)
        cleanup(opt.distributed)
        print("Exiting..............")

    except KeyboardInterrupt:
        cleanup(opt.distributed)

    except Exception:
        traceback.print_exc(file=sys.stdout)
        cleanup(opt.distributed)
