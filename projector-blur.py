import argparse
import math
import os
import numpy

import torch
torch.manual_seed(0)
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from PIL import ImageEnhance
from tqdm import tqdm

import lpips
import pickle
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
       "--split_lpips", action="store_true", help="compute lpips by upscaling generated image and running lpips on left half + right half"
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
        "--normalize_frame", action="store_true", help="normalize a frame when projecting by subtracting by the mean and then dividing by the std deviation of a channel"
    )
    parser.add_argument(
        "--out_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--optimize_noise_map", action="store_true", help="optimize the noise map"
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
    parser.add_argument("--diff_weight", type=float, default=0.1, help="weight of the loss pertaining to the magnitude of the delta of the reset layers in latents")
    parser.add_argument(
        "--w_plus",
        action="store_true",
        help="allow to use distinct latent codes to each layers",
    )
    parser.add_argument(
        "files", metavar="FILES", nargs="+", help="path to image files to be projected"
    )

    # reproducibility
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args = parser.parse_args()

    n_mean_latent = 10000

    resize = min(args.size, SOURCE_DIM)
    
    # define a resolution that is 2 : 1 to crop to that fits within the scene's frames dimensions
    #crop_width = 1280
    #crop_height = 640

    # assume input is h:w 1:2
    transformations = []
    if args.split_lpips:
        transformations.append(transforms.Resize((resize, resize*2)))
    else:
        transformations.append(transforms.Resize((resize, resize)))

    transformations.append(transforms.ToTensor())
    if not args.normalize_frame:
        transformations.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

    transform = transforms.Compose(transformations)

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
    #print("len of noises:", len(original_initialized_noises)) 15
    
    # set up perceptual loss net
    percept = lpips.PerceptualLoss(
            model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
        )
    
    latent_reset_layers_diff = [] # tracks the l2 diff from latent[i] and latent[i-1] for layers we reset as we project frames

    latent_shared_layers_diff = [] # tracks the l2 diff from latent[i] and latent[i-1] for layers we preserve as we project frames
    lpip_diff = [] # tracks the lpip diff current image generated from latent and current original frame
    prev_latent = latent_in.detach().clone()

    for file_num, imgfile in enumerate(args.files):
        for factor in numpy.arange(0,2,0.1):
            latent_in.requires_grad = True # turn back backpropogation
            for noise in noises:
                noise.requires_grad = args.optimize_noise_map # turn it back on for noise too
                
            imgs = []
            pil_img = Image.open(imgfile).convert("RGB")
            enhancer = ImageEnhance.Sharpness(pil_img)
            img = transform(enhancer.enhance(factor))
            source_img_height = img[0].shape[0]
            source_img_width = img[0].shape[1]
            
            #print(img[0,:,:].shape)
            channel_means = [img[i,:,:].mean() for i in range(0,3)]
            #print("channel_means:", channel_means)
            channel_stddev = [img[i,:,:].std() for i in range(0,3)]
            #print("channel_stddev:", channel_stddev)

            if args.normalize_frame: #TODO try with just the mean, and clip to -1 and 1
                for j in range(3):
                    img[j] = torch.clamp(img[j] - channel_means[j], -1, 1)
            imgs.append(img)

            imgs = torch.stack(imgs, 0).to(device)
            optimizer = optim.Adam([latent_in], lr=args.lr)

            pbar = tqdm(range(args.step))
            latent_path = []

            for i in pbar:
                t = i / args.step
                lr = get_lr(t, args.lr)
                optimizer.param_groups[0]["lr"] = lr
                noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
                latent_n = latent_noise(latent_in, noise_strength.item())

                img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

                batch, channel, height, width = img_gen.shape
                #print("img_gen.shape:",img_gen.shape)
                #print("imgs[0].shape:",imgs[0].shape)
                #if args.normalize_frame:
                #    for j in range(3):
                #        img_gen[0][j] = (img_gen[0][j] * channel_stddev[j]) + channel_means[j]

                if args.split_lpips:
                     # upscale generated image, to 1:2 ratio
                     
                     #height_factor = SOURCE_DIM / height
                           
                     img_gen = torch.nn.functional.interpolate(img_gen, img[0].shape)
                     #print("it worked")

                     #print("img_gen shape", img_gen.shape)
                     #print("img shape", img.shape)

                     # left lpips
                     #print("indexed img gen shape", img_gen[:,:,:,:source_img_height].shape)
                     #print("indexed imgs shape", imgs[:,:,:,:source_img_height].shape)
                     left_lpips = percept(img_gen[:,:,:,:source_img_height], imgs[:,:,:,:source_img_height])
                     right_lpips = percept(img_gen[:,:,:,source_img_height:], imgs[:,:,:,source_img_height:])
                     p_loss = left_lpips + right_lpips
                else:
                     p_loss = percept(img_gen, imgs).sum()

                diff_loss = args.diff_weight * ((latent_in - prev_latent)/latent_std).norm(dim=2).mean()

                loss = p_loss if file_num in [0,1] else p_loss + diff_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #noise_normalize_(noises)

                if (i + 1) % 100 == 0:
                    latent_path.append(latent_in.detach().clone())

                pbar.set_description(
                    ( # set noise reg to 0, and mse loss
                        f"perceptual: {p_loss.item():.4f}; diff_loss: {diff_loss.item():.4f}; noise regularize: {0:.4f};"
                        f" mse: {0:.4f}; total_loss: {loss.item():.4f};  lr: {lr:.4f}"
                    )
                )
            
            if args.reset_from is not None and args.reset_till is not None:
                latent_diff = ((prev_latent-latent_in) / latent_std).norm(dim=2)[0]
                latent_reset_layers_diff.append(latent_diff)
            
            lpip_diff.append(p_loss.item())

            img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True, noise=noises)

            if args.normalize_frame:
                for j in range(3):
                    img_gen[0][j] = img_gen[0][j] + channel_means[j]

            i, input_name = list(enumerate(args.files))[file_num]
            
            latent_description = ""
            if args.reset_till:
                latent_description += "-reset-from-" + str(args.reset_from) + "-reset-till-" + str(args.reset_till) + "-layer"
                if args.invert_reset_till:
                    latent_description += "-inverted"
            latent_description += "-" + str(args.step) + "-steps"
                    
            filename = args.out_dir + os.path.splitext(os.path.basename(input_name))[0] + latent_description + "-" + str(factor) + ".pt"

            img_ar = make_image(img_gen)

            result_file = {}

            noise_single = []
            for noise in noises:
                noise_single.append(noise[i : i + 1])

            result_file[input_name] = {
                "img": img_gen[0],
                "latent": latent_in[0],
                "noise": noise_single, # TODO(vmreyes): This isn't saving properly, it saves a 0,1,X,X tensor
            }
            img_name = args.out_dir + os.path.splitext(os.path.basename(input_name))[0] + latent_description + "-" + str(factor) + "-project.png"
            final_img_ar = img_ar[0]
            final_img = Image.fromarray(final_img_ar).resize((args.input_width,args.input_height))
            
            comparison_img = Image.new('RGB', (2 * args.input_width, args.input_height))
            comparison_img.paste(final_img, (0,0))
            comparison_img.paste(enhancer.enhance(factor), (args.input_width,0))
                                            
            comparison_img.save(img_name)

            torch.save(result_file, filename)
            #file_num += 1
            
            # reset back to original latent
            if args.reset_till is not None and args.reset_from is not None:

                new_latent_in = original_initialized_latent

                latent_in.requires_grad = False
                
                # copy its values over
                prev_latent = latent_in.detach().clone()
                if (args.invert_reset_till):
                    if (args.reset_till != 15):
                      trimmed_original_initialized_latent = original_initialized_latent[:,args.reset_till:,:]
                      relevant_indices = torch.tensor(list(range(args.reset_till,16)), device=device)
                      latent_in.index_copy_(1, relevant_indices, trimmed_original_intialized_latent)

                    if (args.reset_from != 0):
                      trimmed_original_initialized_latent = original_initialized_latent[:,:args.reset_from:,:]
                      relevant_indices = torch.tensor(list(range(0,args.reset_from)), device=device)
                      latent_in.index_copy_(1, relevant_indices, trimmed_original_initialized_latent)

                else:  
                    trimmed_original_initialized_latent = original_initialized_latent[:,args.reset_from:args.reset_till+1,:]
                    relevant_indices = torch.tensor(list(range(args.reset_from, args.reset_till+1)), device=device)
                    latent_in.index_copy_(1, relevant_indices, trimmed_original_initialized_latent)

    with open(args.out_dir + "lpip-diff" + latent_description + ".data", 'wb') as filehandle:
        pickle.dump({"reset_layers_diff":latent_reset_layers_diff, "shared_layers_diff":latent_shared_layers_diff, "lpip_diff":lpip_diff}, filehandle)
    data = numpy.array([col.cpu().detach().numpy() for col in latent_reset_layers_diff])
    data[0] = 0 * data[0]
    #data[1] = 0 * data[1]
    fig, ax = plt.subplots()
    c = ax.pcolor(data)
    ax.set_yticks(range(0,len(data)))
    ax.set_title("delta magnitude across projected latents")
    fig.tight_layout()
    fig.colorbar(c, ax=ax)

    #fig, ax = plt.subplots()
    #print("before img", latent_reset_layers_diff)
    #data = numpy.array([col.cpu().detach().numpy() for col in latent_reset_layers_diff]))

    #plt.figure(figsize=(28,4))
    #ax.set_xticks(range(len(latent_reset_layers_diff)))
    #ax.set_yticks(range(len(latent_reset_layers_diff[0])))
    #for i in range(len(latent_reset_layers_diff)):
    #    for j in range(len(latent_reset_layers_diff[i])):
    #        print("latent_reset_layers_diff[i][j]", latent_reset_layers_diff[i][j])
    #        text = ax.text(j, i, latent_reset_layers_diff[i][j].item(), ha="center", va="center", color="w")

    #plt.plot(latent_reset_layers_diff, color="orange", label="reset layers diff (avg layer squared distance)")

    #plt.plot(latent_shared_layers_diff, color="red", label="shared layers diff (avg layer squared distance)")
    
    #plt.subplot(2,1,2)
    #plt.plot(lpip_diff, color="blue", label="lpip distance")
    #plt.legend(loc="best")
    #fig.tight_layout()
    plt.savefig(args.out_dir + "lpip-diff-graph" + latent_description + ".png")

