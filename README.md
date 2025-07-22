# scenedataset

PyTorch dataset which uses PySceneDetect to split videos into scenes. 

This dataset is useful when you have a large video dataset and you want to train a model on each scene of each video.
Instead of splitting the videos into scenes manually, this dataset uses PySceneDetect to automatically split the videos into scenes.
Decord is used to load the video frames on the fly so that not all videos are loaded on memory to keep the dataset lightweight.

The basic pipeline is the following:

1. Use PySceneDetect to split the videos into scenes. During this step, each videos are processed with PySceneDetect and the scene informations (Scene 1 is composed of frames 0 to 100, Scene 2 is composed of frames 101 to 200, etc.) are saved in a JSON file.
2. [Optional] It can be useful to remove duplicate frames inside a scene, and scenes that does not provide meaningful information (when all frames are identical). When this option is enabled, the dataset will go through each scene and remove duplicate frames.
3. [Optional] Since scenes can be of varying length, it can be useful for a model to have scenes of about the same length. It's not good for a model to have a scene of 100 frames and another scene of 200 frames. To solve this problem, the dataset can split each scene into multiple scenes of a fixed length. For example, if a scene is composed of 100 frames and the fixed length is 50, the dataset will split the scene into two scenes of 50 frames each.
4. The dataset is created by loading the JSON file created in step 1 (or on step 2). When the dataset is loaded, the frames of each scene are loaded on the fly using Decord.

Note that the time consuming steps 1 and 2 are only done once. The JSON file is saved and can be reused for future training. When the parameters of the dataset are changed, the JSON file is automatically regenerated with a different name.

## Installation

```bash
$ pip install scenedataset
```

## Usage

A practical example can be found on the [TorchGANime](https://github.com/Kurokabe/TorchGANime/blob/master/torchganime/data/dataloader/video.py) project

```python
from scenedataset import SceneDataset
from torchvision import transforms
from torchvision.transforms import _transforms_video as video_transforms

transformations = transforms.Compose(
    [
        # Example of transformations for data augmentation
        video_transforms.RandomCropVideo(224),
        video_transforms.RandomHorizontalFlipVideo(),
        video_transforms.ToTensorVideo(),
    ]
)

dataset = SceneDataset(
    # The paths can be a directory or a list of files. The wildcard * can be used to select multiple files.
    paths=["/path/to/video1.mp4", "/path/to/specific/directory/*.mp4"], 
    transform=transformations, # Transformations will be applied before the scene is returned
    recursive=True, # Whether or not the directory should be searched recursively
    show_progress=True, # Whether or not to show the progress when scenes are detected with PySceneDetect
    min_max_len=(15, 25), # When specified, long scenes will be split into shorter scenes where the length is between min_max_len
    duplicate_threshold=0.01, # When specified, subsequent frames in a scenes which difference is below this threshold will be removed. For instance two identical frames have a difference of 0.0, which is below the threshold of 0.01
    duplicate_metric="mse", # The metric used to compute the difference between two frames. Can be "mse" or "mae" or "lpips"
    device="cpu", # The device used to compute the difference between two frames. Can be "cpu" or "gpu"
    initial_shuffle=True, # Whether or not to shuffle the dataset will be shuffled before the first epoch. Can be useful for the validation dataset to have scenes from different videos (if you take only the first N scenes for instance)
    root_dir="/path/to/my/cache/folder", # The folder where the JSON file will be saved. If not specified, the JSON file will be saved in ~/.scene_dataset
    detector="content", # The detector used by PySceneDetect. Can be "content" or "threshold" or "adaptive"
    # Additional arguments passed to the PySceneDetect detector
    threshold=30, # The threshold used by the threshold detector
    min_scene_len=15, # The minimum length of a scene
)

first_scene = dataset[0]
number_of_scenes = len(dataset)
```

For the `duplicate_metric` parameter, the following metrics are available:
* `mse`: Mean Squared Error
* `mae`: Mean Absolute Error
* `lpips`: [Perceptual Similarity](https://github.com/richzhang/PerceptualSimilarity). This metric is slower than the other two and is recommended to run with `device` set to `gpu`. 

More informations about the PySceneDetect detectors and parameters can be found [on the PySceneDetect documentation](http://scenedetect.com/projects/Manual/en/latest/api/detectors.html).

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`scenedataset` was created by Alain Gysi. It is licensed under the terms of the BSD 3-Clause license.

## Credits

`scenedataset` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
