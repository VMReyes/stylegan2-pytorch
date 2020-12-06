import argparse
import math
import os

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import lpips
from model import Generator

import matplotlib.pyplot as plt

SOURCE_DIM=512


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="path to the model checkpoint"
    )
    parser.add_argument(
        "--input_width", type=int, required=True, help="input frame(s) width"
    )
    parser.add_argument(
        "--input_height", type=int, required=True, help="input frame(s) height"
    )
    parser.add_argument(
        "--size", type=int, default=256, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--channel_multiplier", type=int, default=2, help="model checkpoint channel multiplier"
    )
    parser.add_argument(
        "--invert_reset_till", action="store_true", help="invert the selection by reset_till"
    )
    parser.add_argument(
        "--reset_from", type=int, help="reset starting at layer value  0 indexed"
    )
    parser.add_argument(
        "--reset_till", type=int, help="reset layers 0 to value, 0 indexed"
    )
    parser.add_argument(
        "--out_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--full", action="store_true", help="reset all layers of the latent between projections"
    )
    parser.add_argument(
        "--optimize_noise_map", action="store_true", help="optimize the noise map"
    )
    parser.add_argument(
        "--full_reset_noise_map", action="store_true", help="always completely reset the noise map"
    )
    parser.add_argument(
        "--heavy_start", action="store_true", help="begin the projection by projecting the first frame by 10000 iterations"
    )

    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.05,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float,
        default=0.25,
        help="duration of the learning rate decay",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--noise", type=float, default=0.05, help="strength of the noise level"
    )
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    parser.add_argument("--step", type=int, default=1000, help="optimize iterations")
    parser.add_argument(
        "--noise_regularize",
        type=float,
        default=1e5,
        help="weight of the noise regularization",
    )
    parser.add_argument("--mse", type=float, default=0, help="weight of the mse loss")
    parser.add_argument(
        "--w_plus",
        action="store_true",
        help="allow to use distinct latent codes to each layers",
    )
    parser.add_argument(
        "files", metavar="FILES", nargs="+", help="path to image files to be projected"
    )

    args = parser.parse_args()

    n_mean_latent = 10000

    resize = min(args.size, SOURCE_DIM)
    
    # define a resolution that is 2 : 1 to crop to that fits within the scene's frames dimensions
    #crop_width = 1280
    #crop_height = 640

    # assume input is h:w 1:2
    transform = transforms.Compose(
        [
            #transforms.CenterCrop((crop_height, crop_width)), # specific for scene frames
            transforms.Resize((resize, resize)),
            #transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    # set up generator
    g_ema = Generator(args.size, 512, 8, channel_multiplier=args.channel_multiplier)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)
    # create a random latent that will persist
    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
            
    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(1, 1, 1, 1).normal_())

    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(1, 1)

    if args.w_plus:
        latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

    latent_in.requires_grad = True
    
    original_initialized_latent = latent_in.detach().clone()

    for noise in noises:
        noise.requires_grad = args.optimize_noise_map # optimize for noise

    original_initialized_noises = [noise.detach().clone() for noise in noises]
    
    # set up perceptual loss net
    percept = lpips.PerceptualLoss(
            model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
        )
    
    file_num = 0
    latent_reset_layers_diff = [] # tracks the l2 diff from latent[i] and latent[i-1] for layers we reset as we project frames

    latent_shared_layers_diff = [] # tracks the l2 diff from latent[i] and latent[i-1] for layers we preserve as we project frames
    lpip_diff = [] # tracks the lpip diff current image generated from latent and current original frame
    
    for imgfile in args.files:
        prev_latent = latent_in.detach().clone()
        
        latent_in.requires_grad = True # turn back backpropogation
        for noise in noises:
            noise.requires_grad = args.optimize_noise_map # turn it back on for noise too
            
        imgs = []

        img = transform(Image.open(imgfile).convert("RGB"))
        imgs.append(img)

        imgs = torch.stack(imgs, 0).to(device)
        optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

        pbar = tqdm(range(100000 if file_num == 0 and args.heavy_start else args.step))
        latent_path = []

        for i in pbar:
            t = i / args.step
            lr = get_lr(t, args.lr)
            optimizer.param_groups[0]["lr"] = lr
            noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
            latent_n = latent_noise(latent_in, noise_strength.item())

            img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

            batch, channel, height, width = img_gen.shape

            if height > SOURCE_DIM:
                factor = height // SOURCE_DIM 

                img_gen = img_gen.reshape(
                    batch, channel, height // factor, factor, width // factor, factor
                )
                img_gen = img_gen.mean([3, 5])

            p_loss = percept(img_gen, imgs).sum()
            n_loss = noise_regularize(noises)
            mse_loss = F.mse_loss(img_gen, imgs)

            loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            noise_normalize_(noises)

            if (i + 1) % 100 == 0:
                latent_path.append(latent_in.detach().clone())

            pbar.set_description(
                (
                    f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                    f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
                )
            )
        if args.reset_from == 0 and args.reset_till == 15:
            if args.invert_reset_till:
                latent_shared_layers_diff.append(torch.sum(torch.pow(prev_latent-latent_in, 2) / 16).item())
                latent_reset_layers_diff.append(0)
            else:
                latent_reset_layers_diff.append(torch.sum(torch.pow(prev_latent-latent_in, 2) / 16).item())
                latent_shared_layers_diff.append(0)
        else:
            if (args.invert_reset_till):
                latent_shared_layers_diff.append(torch.sum(torch.pow(prev_latent[:,args.reset_from:args.reset_till,:]-latent_in[:,args.reset_from:args.reset_till,:], 2)).item() / (args.reset_till - args.reset_from + 1))
                latent_reset_layers_diff.append(((torch.sum(torch.pow(prev_latent[:,args.reset_till:,:]-latent_in[:,args.reset_till:,:], 2)) + torch.sum(torch.pow(prev_latent[:,:args.reset_from,:]-latent_in[:,:args.reset_from,:], 2))).item()) / (16 - args.reset_till + args.reset_from))
            else:
                latent_reset_layers_diff.append(torch.sum(torch.pow(prev_latent[:,args.reset_from:args.reset_till,:]-latent_in[:,args.reset_from:args.reset_till,:], 2)).item() / (args.reset_till - args.reset_from + 1))
                latent_shared_layers_diff.append(((torch.sum(torch.pow(prev_latent[:,args.reset_till:,:]-latent_in[:,args.reset_till:,:], 2)) + torch.sum(torch.pow(prev_latent[:,:args.reset_from,:]-latent_in[:,:args.reset_from,:], 2))).item()) / (16 - args.reset_till + args.reset_from))
        
        lpip_diff.append(p_loss.item())

        img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True, noise=noises)

        i, input_name = list(enumerate(args.files))[file_num]
        
        latent_description = ""
        if args.full:
            latent_description += "full-reset-latent"
        if args.reset_till:
            latent_description += "-reset-from-" + str(args.reset_from) + "-reset-till-" + str(args.reset_till) + "-layer"
            if args.invert_reset_till:
                latent_description += "-inverted"
        latent_description += "-" + str(args.step) + "-steps"
                
        filename = args.out_dir + os.path.splitext(os.path.basename(input_name))[0] + latent_description + ".pt"

        img_ar = make_image(img_gen)

        result_file = {}
        #i, input_name = list(enumerate(args.files))[file_num]
        #for i, input_name in enumerate(args.files):
        noise_single = []
        for noise in noises:
            noise_single.append(noise[i : i + 1])

        result_file[input_name] = {
            "img": img_gen[0],
            "latent": latent_in[0],
            "noise": noise_single,
        }
        img_name = args.out_dir + os.path.splitext(os.path.basename(input_name))[0] + latent_description + "-project.png"
        final_img_ar = img_ar[0]
        final_img = Image.fromarray(final_img_ar).resize((args.input_width,args.input_height))
        
        comparison_img = Image.new('RGB', (2 * args.input_width, args.input_height))
        comparison_img.paste(final_img, (0,0))
        comparison_img.paste(Image.open(imgfile).convert("RGB"), (args.input_width,0))
                                        
        comparison_img.save(img_name)

        torch.save(result_file, filename)
        file_num += 1
        
        # reset back to original latent
        if (args.reset_till is not None):

            new_latent_in = original_initialized_latent
            latent_in.requires_grad = False
            
            # copy its values over
            if (args.invert_reset_till):
                if (args.reset_till != 15):
                  latent_in.index_copy_(1, torch.tensor(list(range(args.reset_till,16)), device=device), new_latent_in[:,args.reset_till:,:])

                if (args.reset_from != 0):
                  latent_in.index_copy_(1, torch.tensor(list(range(0,args.reset_from)), device=device), new_latent_in[:,:args.reset_from,:])

            else:  
                latent_in.index_copy_(1, torch.tensor(list(range(args.reset_from, args.reset_till)), device=device), new_latent_in[:,args.reset_from:args.reset_till,:])
        
        # reset corresponding noise maps
        if (args.reset_till is not None):
            for noise in noises:
                noise.requires_grad = False

            if (args.full_reset_noise_map):
                for i in range(len(noises)):
                  noises[i] = original_initialized_noises[i].detach().clone()
            else:
                for i in range(args.reset_from, args.reset_till):
                  noises[i] = original_initialized_noises[i].detach().clone()
        
    
    print(latent_reset_layers_diff)
    print(latent_shared_layers_diff)
    plt.figure(figsize=(28,4))
    plt.plot(latent_reset_layers_diff, color="orange", label="reset layers diff (avg layer squared distance)")
    plt.plot(latent_shared_layers_diff, color="red", label="shared layers diff (avg layer squared distance)")
    
    plt.twinx()
    plt.plot(lpip_diff, color="blue", label="lpip distance")
    plt.legend(loc="best")
    plt.savefig(args.out_dir + "lpip-diff-graph" + latent_description + ".png")

