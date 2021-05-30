import pdb
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
        "--channel_multiplier", type=int, default=2, help="model checkpoint channel multiplier"
    )
    parser.add_argument(
        "--invert_reset_till", action="store_true", help="invert the selection by reset_till"
    )
    parser.add_argument(
        "--num_segments", type=int, required=True, help="maximum number of segments. the first and last frame of each segment is explicitly optimized."
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
        "--look_ahead", type=int, required=True, help="Amount of frames to look ahead for our new loss paradigm"
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
    transformations.append(transforms.Resize((resize, resize)))
    transformations.append(transforms.ToTensor())
    transformations.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    transform = transforms.Compose(transformations)

    # set up generator
    g_ema = Generator(args.size, 512, 8, channel_multiplier=args.channel_multiplier)
    g_ema.load_state_dict(torch.load(args.ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    # create a random latent that will be the latent used to begin optimization of every frame
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
    
    if args.normalize_frame:
        lpip_diff_to_original = [] # when normalizing, fit to the normalized image but show an lpips graph with respect to the original frame
    latent_reset_layers_diff = [] # tracks the l2 diff from latent[i] and latent[i-1] for layers we reset as we project frames
    latent_shared_layers_diff = [] # tracks the l2 diff from latent[i] and latent[i-1] for layers we preserve as we project frames
    lpip_diff = [] # tracks the lpip diff current image generated from latent and current original frame
    #prev_latent = latent_in.detach().clone()
    #latent_in.requires_grad = True # turn back backpropogation
    for noise in noises:
        noise.requires_grad = args.optimize_noise_map # turn it back on for noise too, if we are optimizing noise maps
    
    # load all the filenames
    frames = []
    for file_num, filename in enumerate(args.files):
        
        rgb_img = Image.open(filename).convert("RGB")
        rgb_tensor = transforms.functional.to_tensor(rgb_img)
        img = transform(rgb_img)
        
        channel_means = [img[i,:,:].mean() for i in range(0,3)]
        channel_stddev = [img[i,:,:].std() for i in range(0,3)]

        if args.normalize_frame: #TODO try with just the mean, and clip to -1 and 1
            for j in range(3):
                img[j] = torch.clamp((img[j] - channel_means[j]), -1, 1)

        frames.append(img)
    frames = torch.stack(frames)

    pbar = tqdm(range(args.step))
    latent_path = []


    # get z1
    target_frames = []
    new_split = [0, args.look_ahead]

    while len(target_frames) < 2*args.num_segments:
        # create optimizer
        # make a list of points
        target_frames.extend(new_split)
        target_frames.sort()
        print(f"target_frames:{target_frames}")

        latents = []
        for _ in target_frames:
            cloned_latent = original_initialized_latent.detach().clone()
            cloned_latent.requires_grad = True
            latents.append(cloned_latent)
        optimizer = optim.Adam(latents, lr=args.lr)
        p_losses = []

        #for every two indices, interpolate between them
        for i in pbar:
            t = i / args.step
            lr = get_lr(t, args.lr)
            optimizer.param_groups[0]["lr"] = lr
            noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
            latents_n = []
            iteration_losses = []

            for latent in latents:
                latents_n.append(latent_noise(latent, noise_strength.item()))

            for j in range(0, len(target_frames), 2):
                latent_1_index = target_frames[j]
                latent_2_index = target_frames[j+1]
                #print(f"latent_1_index:{latent_1_index}, latent_2_index:{latent_2_index}")

                latent_1 = latents_n[j]
                latent_2 = latents_n[j+1]
                delta = latent_2 - latent_1
                
                num_frames = latent_2_index - latent_1_index
                
                minibatch = 5
                optimizer.zero_grad()
                indices = list(range(latent_1_index, latent_2_index+1))
                #print(f"indices:{indices}")
                num_batches = num_frames // minibatch + int(num_frames % minibatch > 0) # use index splicing to make it easier
                for k in range(num_batches):
                    imgs_gen = []
                    for index in indices[k*minibatch:k*minibatch+minibatch]:
                        interpolated_point = (index- latent_1_index) / (num_frames)
                        #print(f"k: {k}, interpolated_point: {interpolated_point}")
                        interpolated_latent = interpolated_point * delta + latent_1
                        img_gen, _ = g_ema([interpolated_latent], input_is_latent=True, noise=noises) # make the images
                        imgs_gen.append(img_gen[0])
                    imgs_gen = torch.stack(imgs_gen).to(device)
                    p_loss = 0
                    for img_num in range(len(imgs_gen)):
                        frame_loss = percept(imgs_gen[img_num], frames[img_num + k*minibatch + latent_1_index])
                        p_loss += frame_loss
                        iteration_losses.append(frame_loss.item())
                    loss = p_loss
                    loss.backward()
                optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"Saving latent and logging p_loss at i:{i}")
                latent_path.append(latents)
                p_losses.append(iteration_losses)

            pbar.set_description(
                ( # set noise reg to 0, and mse loss
                    f"perceptual: {p_loss.item():.4f}; diff_loss: {0:.4f}; noise regularize: {0:.4f};"
                    f" mse: {0:.4f}; total_loss: {loss.item():.4f};  lr: {lr:.4f}"
                )
            )
        peak = torch.argmax(torch.as_tensor(iteration_losses)).item()
        new_split = [peak-1, peak]
        if (peak-1 in target_frames) or (peak in target_frames):
            print("Peak finding returned index already optimized. Stopping.")
            break

    # save latents and frames
    # construct the filename
    i, input_name = list(enumerate(args.files))[0]
    latent_description = "-" + str(args.step) + "-steps"
    filename = args.out_dir + os.path.splitext(os.path.basename(input_name))[0] + latent_description + ".pt"
    # store a noise latent
    result_file = {}
    noise_single = []
    for noise in noises:
        noise_single.append(noise[i : i + 1])
    # create and save frames
    frame_num = 0
    final_frames = []
    for j in range(0, len(target_frames), 2):
        latent_1_index = target_frames[j]
        latent_2_index = target_frames[j+1]
        print(f"(generating_final_frames) latent_1_index:{latent_1_index}, latent_2_index:{latent_2_index}")

        latent_1 = latents_n[j]
        latent_2 = latents_n[j+1]
        delta = latent_2 - latent_1
        
        num_frames = latent_2_index - latent_1_index
        indices = list(range(latent_1_index, latent_2_index+1))
        for k in range(num_frames):
            interpolated_point = (k) / (num_frames)
            interpolated_latent = interpolated_point * delta + latent_1
            img_gen, _ = g_ema([interpolated_latent], input_is_latent=True, noise=noises) # make the images
            final_img_ar = make_image(img_gen)[0]
            final_img = Image.fromarray(final_img_ar).resize((args.input_width,args.input_height))
            comparison_img = Image.new('RGB', (2 * args.input_width, args.input_height))
            comparison_img.paste(final_img, (0,0))
            comparison_img.paste(Image.open(list(enumerate(args.files))[frame_num][1]).convert("RGB"), (args.input_width,0))

            img_name = args.out_dir + os.path.splitext(os.path.basename(input_name))[0] + latent_description + "-" + f"{frame_num:02d}" +  "-project.png"
            comparison_img.save(img_name)

            final_frames.append(comparison_img)
            frame_num += 1
    # create and save data
    result_file[input_name] = {
        "final_frames": final_frames,
        "latents": latents,
        "target_frames": target_frames
    }
    torch.save(result_file, filename)
    

    # save the history of lpips
    with open(args.out_dir + "lpips" + latent_description + ".data", 'wb') as filehandle:
        pickle.dump({"lpips":p_losses, "latents":latent_path}, filehandle)
    fig, ax = plt.subplots()
    ax.plot(p_losses)
    fig.savefig("p_loss.png")
