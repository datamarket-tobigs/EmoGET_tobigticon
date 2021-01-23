import os
import torch
from collections import OrderedDict
import re
import math
import argparse


def network_blending(low, high, res, ratio=1, name=None):
    net_A = torch.load(low)
    net_B = torch.load(high)
    net_interp = OrderedDict()

    A_names = list(net_A['g_ema'].keys())
    B_names = list(net_B['g_ema'].keys())

    assert all((x == y for x, y in zip(A_names, B_names)))
    origin_idx = [[value, i] for i, value in enumerate(A_names)]

    match_names = [x for x in origin_idx if not (x[0].startswith('style') or x[0].startswith('to_rgb') or x[0].startswith('noises'))]

    mid_point_idx = [i for i, value in enumerate([x[-1] for x in match_names]) if value == res][-1] + 1

    for pos, value in enumerate(match_names):
        x = pos - mid_point_idx
        alpha = ratio if x <= 0 else 1-ratio

        for p, (k, v_A) in enumerate(net_A['g_ema'].items()):
            v_B = net_B['g_ema'][k]
            net_interp[k] = alpha * v_A + (1 - alpha) * v_B

    ckpt = {"g_ema": net_interp, "latent_avg": net_A['latent_avg']}

    if name is not None:
        torch.save(ckpt, name + ".pt")

    else:
        name = low.split('/')[-1]
        torch.save(ckpt, name + ".pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--file1",
        required=True,
        default='params/ffhq.pt',
        type=str,
        help="Low pt file"
    )

    parser.add_argument(
        "--file2",
        required=True,
        default='params/animation.pt',
        type=str,
        help="High pt file"
    )

    parser.add_argument(
        "--ratio",
        default=1,
        type=int,
        help="How much adjust the ratio"
    )

    parser.add_argument(
        "--res",
        default=8,
        type=int,
        help="Standard layer you want"
    )

    parser.add_argument(
        "--name",
        default='result',
        type=str,
        help="output.pt name"
    )

    args = parser.parse_args()

    network_blending(args.file1, args.file2, args.res, args.ratio, args.name)