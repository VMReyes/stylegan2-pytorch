"""
Script to manipulate saved lpip latent diff data as output by projector.py. Defaults to simply regenerating the graph.

Inputs: filepath to saved data
Outputs: saves graph to same folder as data
"""

import numpy
import argparse
import pickle
import matplotlib.pyplot as plt
import os

def regen_latent_diff_graph(args):
    data = pickle.load(open(args.data, "rb"))
    tensor_data = data["reset_layers_diff"]

    latent_layers_diff = numpy.array([col.cpu().detach().numpy() for col in tensor_data])

    latent_layers_diff[0] *= 0
    latent_layers_diff[1] *= 0

    fig, ax = plt.subplots(figsize=(9,12))
    c = ax.pcolor(latent_layers_diff, vmin=0.0, vmax=2.0)
    ax.set_yticks(range(0,len(latent_layers_diff)))
    ax.set_title("delta magnitude across projected latents")
    fig.tight_layout()
    fig.colorbar(c, ax=ax)
    plt.savefig(os.path.dirname(args.data) + "latent-diff-graph.png")

def regen_lpips_graph(args):
    data = pickle.load(open(args.data, "rb"))
    lpip_diff = data["lpip_diff"]
    fig, ax = plt.subplots(figsize=(16,9))
    plt.plot(lpip_diff, color="blue", label="lpip distance")
    plt.legend(loc="best")
    fig.tight_layout()

    ax.set_title("lpips score")
    plt.ylim([0,0.3])
    plt.savefig(os.path.dirname(args.data) + "lpip-graph.png")



def main(args):
    regen_latent_diff_graph(args)
    regen_lpips_graph(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Required filepath to data")
    args = parser.parse_args()
    main(args)
