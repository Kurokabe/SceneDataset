import numpy as np
import os
import re
from glob import glob
from typing import Tuple

import click
import imageio
import matplotlib.pyplot as plt
from loguru import logger
from torchvision import transforms
from torchvision.transforms import _transforms_video as video_transforms
from tqdm.auto import tqdm

from scenedataset.scenedataset import SceneDataset


def init_metric(method: str):
    if method == "mae":
        from torch.nn import L1Loss

        return L1Loss(reduction="none")
    elif method == "mse":
        from torch.nn import L2Loss

        return L2Loss(reduction="none")
    elif method == "lpips":
        from lpips import LPIPS

        return LPIPS(net="vgg")
    else:
        raise ValueError(f"Method {method} not supported")


def compute_difference(frames, metric):
    if metric.__class__.__name__ == "LPIPS":
        # frames is between 0 and 1, scale between -1 and 1
        frames = frames * 2 - 1
        return metric(frames[:-1], frames[1:])
    else:
        difference = metric(frames[:-1], frames[1:])
        return difference.mean(dim=(1, 2, 3))


def generate_images(
    scenes: SceneDataset, method, output_folder: str, n_scenes: int
) -> Tuple[str, str]:
    metric = init_metric(method)

    if output_folder is None:
        # Take folder from input without filename
        output_folder = os.path.join(input.name.rsplit("/", 1)[0])
        logger.info(f"Output folder not specified.")
    images_folder = os.path.join(output_folder, "images")
    gifs_folder = os.path.join(output_folder, "gifs")
    logger.info(f"Images will be saved in {images_folder}, gifs in {gifs_folder}")

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(gifs_folder, exist_ok=True)

    for i in tqdm(
        range(0, n_scenes), desc=f"Generating images inside {images_folder}..."
    ):
        frames = scenes[i]
        frames = frames.permute(1, 0, 2, 3)

        differences = compute_difference(frames, metric)
        for j in range(0, len(frames) - 1):
            current_frame = frames[j : j + 1]
            next_frame = frames[j + 1 : j + 2]

            difference = differences[j].item()

            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(current_frame[0].permute((1, 2, 0)).numpy())
            ax.set_title(f"{method}: {difference}")
            plt.savefig(
                os.path.join(
                    images_folder, f"{difference:.5f}_current_frame_{i}_{j}.png"
                )
            )
            plt.close()

            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(next_frame[0].permute((1, 2, 0)).numpy())
            ax.set_title(f"{method}: {difference}")
            plt.savefig(
                os.path.join(images_folder, f"{difference:.5f}_next_frame_{i}_{j}.png")
            )
            plt.close()
    return images_folder, gifs_folder


def generate_gifs(images_folder: str, gifs_folder: str):
    filenames = sorted(glob(os.path.join(images_folder, "*_current_frame_*.png")))

    for filename in filenames:
        try:
            difference, i, j = re.findall(
                r"(\d+.\d+)_current_frame_(\d+)_(\d+).png", filename
            )[0]
        except:
            print(filename)
        images = []
        images.append(
            imageio.imread(
                os.path.join(images_folder, f"{difference}_current_frame_{i}_{j}.png")
            )
        )
        images.append(
            imageio.imread(
                os.path.join(images_folder, f"{difference}_next_frame_{i}_{j}.png")
            )
        )
        imageio.mimsave(
            os.path.join(gifs_folder, f"{difference}_animation_{i}_{j}.gif"), images
        )


@click.command()
@click.option(
    "--input",
    type=click.File("r"),
    required=True,
    help="Video file to be processed. The difference between frames will be calculated",
)
@click.option(
    "--method",
    type=click.Choice(["mae", "mse", "lpips"]),
    required=True,
    help="Method to be used to calculate the difference between frames",
)
@click.option(
    "--output_folder",
    type=click.Path(exists=True),
    help="Folder used to save the images and the gif with the difference between each frames",
)
@click.option(
    "--n_scenes",
    type=int,
    help="Number of scenes to process. If not specified, all scenes will be processed",
)
def main(input: click.File, method, output_folder, n_scenes):
    dataset = SceneDataset(
        paths=[input.name],
        show_progress=True,
        transform=transforms.Compose(
            [video_transforms.ToTensorVideo(), transforms.Resize((128, 256))]
        ),
    )

    if len(dataset) == 0:
        raise ValueError(
            f"No scenes found. Maybe the indicated path is not a video file? {input}"
        )

    if n_scenes is None:
        n_scenes = len(dataset)

    images_folder, gifs_folder = generate_images(
        dataset, method, output_folder, n_scenes
    )
    generate_gifs(images_folder, gifs_folder)


if __name__ == "__main__":
    main()
