import argparse
import yaml
from train_eval.trainer import Trainer
from torch.utils.tensorboard import SummaryWriter
import os
from UKF import *

# UKF
# ukf = create_ukf(dim_x=5, dim_z=2, dt=1.0)
# save_ukf_params(ukf, '/home/xie/code/TMAP/UKF_params')

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default='/home/xie/code/TMAP/configs/TMAP_configs.yml', help="Config file with dataset parameters")
parser.add_argument("-r", "--data_root", default='/home/xie/train_data/nuScenes', help="Root directory with data")
parser.add_argument("-d", "--data_dir", default='/home/xie/train_data/nuScenes/dataset', help="Directory to extract data")
parser.add_argument("-o", "--output_dir", default='/home/xie/code/TMAP/outcome1', help="Directory to save checkpoints and logs")
parser.add_argument("-n", "--num_epochs", default=100, help="Number of epochs to run training for")
parser.add_argument("-w", "--checkpoint", help="Path to pre-trained or intermediate checkpoint", required=False)
args = parser.parse_args()


# Make directories
if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)
if not os.path.isdir(os.path.join(args.output_dir, 'checkpoints')):
    os.mkdir(os.path.join(args.output_dir, 'checkpoints'))
if not os.path.isdir(os.path.join(args.output_dir, 'tensorboard_logs')):
    os.mkdir(os.path.join(args.output_dir, 'tensorboard_logs'))


# Load config
with open(args.config, 'r') as yaml_file:
    cfg = yaml.safe_load(yaml_file)


# Initialize tensorboard writer
writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard_logs'))


# Train
trainer = Trainer(cfg, args.data_root, args.data_dir, checkpoint_path=args.checkpoint, writer=writer)
trainer.train(num_epochs=int(args.num_epochs), output_dir=args.output_dir)


# Close tensorboard writer
writer.close()
