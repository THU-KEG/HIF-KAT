import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataname", type=str, default = "dcitation")
parser.add_argument("--dataset", type=str, default = "dblp-scholar")
parser.add_argument("--no", type=int, default = "0")
parser.add_argument("--gpu", type=int, default = "1")
parser.add_argument("--obligate", type=str, default = "title")
args = parser.parse_args()

if args.dataset == "itunes-amazon":
    for drop_rate in range(0, 1):
        for train_split in range(2, 3):
            train_split /= 20
            os.system("CUDA_VISIBLE_DEVICES={} python run_magellan.py --dataname {} --dataset {} --train_split {} --drop_rate {} --no {} --obligate Song_Name".format(args.gpu, args.dataname, args.dataset, train_split, drop_rate, args.no))

# citation:dblp-scholar
if args.dataset == "dblp-scholar":
    for drop_rate in range(0, 1):
        for train_split in range(2, 3):
            train_split /= 200
            os.system("CUDA_VISIBLE_DEVICES={} python run_magellan.py --dataname {} --dataset {} --train_split {} --drop_rate {} --no {} --obligate title --batch_num 12".format(args.gpu, args.dataname, args.dataset, train_split, drop_rate, args.no))

if args.dataset == "dblp-acm":
    for drop_rate in range(0, 1):
        for train_split in range(2, 3):
            train_split /= 200
            os.system("CUDA_VISIBLE_DEVICES={} python run_magellan.py --dataname {} --dataset {} --train_split {} --drop_rate {} --no {} --obligate title --batch_num 12".format(args.gpu, args.dataname, args.dataset, train_split, drop_rate, args.no))