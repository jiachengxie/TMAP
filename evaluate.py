import argparse
import yaml
from train_eval.evaluator import Evaluator
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default='/home/xie/code/TMAP/configs/TMAP_configs.yml', help="Config file with dataset parameters")
parser.add_argument("-r", "--data_root", default='/home/xie/train_data/nuScenes', help="Root directory with data")
parser.add_argument("-d", "--data_dir", default='/home/xie/train_data/nuScenes/dataset', help="Directory to extract data")
parser.add_argument("-o", "--output_dir", default='/home/xie/code/TMAP/test1', help="Directory to save results")
parser.add_argument("-w", "--checkpoint", default='/home/xie/code/TMAP/outcome1/checkpoints/best.tar', help="Path to pre-trained or intermediate checkpoint")
args = parser.parse_args()


# Make directories
if not os.path.isdir(args.output_dir):
    os.mkdir(args.output_dir)
if not os.path.isdir(os.path.join(args.output_dir, 'results')):
    os.mkdir(os.path.join(args.output_dir, 'results'))


# Load config
with open(args.config, 'r') as yaml_file:
    cfg = yaml.safe_load(yaml_file)


# Evaluate
evaluator = Evaluator(cfg, args.data_root, args.data_dir, args.checkpoint)
evaluator.evaluate(output_dir=args.output_dir)
