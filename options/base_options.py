import os.path as osp


class Options:
    def __init__(self):
        self.seed = 42
        self.val_proportion = 0.1
        
        self.name = "training_cloth_segm_u2net_exp1"  # experiment name
        self.image_folder = "dataset/"  # image folder path
        self.distributed = False  # True for multi gpu training
        self.isTrain = True
        self.cpu = False

        self.fine_width = 192 * 4
        self.fine_height = 192 * 4

        # Mean std params
        self.mean = 0.5
        self.std = 0.5

        self.batchSize = 2  # 12
        self.nThreads = 2  # 3
        self.max_dataset_size = float("inf")

        self.serial_batches = False
        self.continue_train = True
        if self.continue_train:
            self.unet_checkpoint = "prev_checkpoints/cloth_segm_unet_surgery.pth"

        self.save_freq = 10
        self.print_freq = 10
        self.image_log_freq = 100

        self.iter = 100000
        self.lr = 0.0002
        self.clip_grad = 5

        self.logs_dir = osp.join("logs", self.name)
        self.save_dir = osp.join("results", self.name)
