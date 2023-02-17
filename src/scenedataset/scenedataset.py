import dataclasses
import json
import os
from dataclasses import dataclass
from glob import glob
from typing import Callable, Dict, List, Literal, Optional, Tuple

import decord
import ffmpeg
import torch
from decord import VideoReader, cpu
from loguru import logger
from scenedetect import SceneDetector, SceneManager, open_video
from torch.utils.data import Dataset

from tqdm.auto import tqdm

decord.bridge.set_bridge("torch")


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


@dataclass
class Scene:
    start: int
    end: int
    video_path: str
    images_to_load: Optional[List[int]] = None

    def __len__(self):
        return (
            self.end - self.start
            if self.images_to_load is None
            else len(self.images_to_load)
        )

    def __repr__(self):
        return (
            f"Scene({self.start}, {self.end}, {self.video_path}, {self.images_to_load})"
        )

    def __str__(self):
        return (
            f"Scene({self.start}, {self.end}, {self.video_path}, {self.images_to_load})"
        )


class SceneDataset(Dataset):
    scene_list_file = "mapping_video_path_to_scene_file.json"

    def __init__(
        self,
        paths: List[str],
        transform: Optional[Callable] = None,
        recursive: bool = False,
        show_progress: bool = False,
        min_max_len: Optional[Tuple[int]] = None,
        detector: Literal["content", "threshold", "adaptive"] = "content",
        duplicate_threshold: Optional[float] = None,
        duplicate_metric: Literal["mse", "mae", "lpips"] = "lpips",
        **kwargs,
    ):
        """Dataset that will load scenes on the fly from a list of videos. The scenes are detected using the PySceneDetect library https://scenedetect.com/projects/Manual/en/latest/index.html.
        Args:
            paths (List[str]): List of paths to the videos. The paths can point to a video file or a folder containing video files.
            transform (Optional[Callable], optional): Transformations applied after loading the scenes. Defaults to None.
            recursive (bool, optional): If elements in paths are folders and recursive is True, will look for videos in subfolders. Defaults to False.
            show_progress (bool, optional): Whether to show progress or not when detecting scenes with PySceneDetect. Defaults to False.
            min_max_len (Optional[Tuple[int]], optional): Minimum and maximum length of the scenes. If specified, must be a tuple of two values, the minimal scene length and maximal scene length. Scenes longer that max scene length will be splitted into a length that is between these two values.  Defaults to None.
            detector (Literal[&quot;content&quot;, &quot;threshold&quot;, &quot;adaptive&quot;], optional): Method used to detect the scenes. Refer to the following link for more details about detectors: https://scenedetect.com/projects/Manual/en/latest/cli/detectors.html. Defaults to "content".
            duplicate_threshold (Optional[float], optional): If specified, will remove duplicate scenes. The threshold is the minimum distance between two scenes to consider them as duplicates. Defaults to None.
            duplicate_method (Literal[&quot;mse&quot;, &quot;mae&quot;, &quot;ssim&quot;], optional): Method used to compute the distance between two scenes. Defaults to "ssim". Only used if duplicate_threshold is specified.
            **kwargs: Keyword arguments passed to the detector. Depends on which detector is used. Refer to the following link for more details about detectors: https://scenedetect.com/projects/Manual/en/latest/cli/detectors.html.
            The available arguments when detector is set to "content" are:
                - threshold=27.0
                - min_scene_len=15
                - weights=Components(delta_hue=1.0, delta_sat=1.0, delta_lum=1.0, delta_edges=0.0)
                - luma_only=False
                - kernel_size=None
            The available arguments when detector is set to "threshold" are:
                - adaptive_threshold=3.0
                - min_scene_len=15
                - window_width=2
                - min_content_val=15.0
                - weights=Components(delta_hue=1.0, delta_sat=1.0, delta_lum=1.0, delta_edges=0.0)
                - luma_only=False
                - kernel_size=None
                - video_manager=None
                - min_delta_hsv=None
            The available arguments when detector is set to "adaptive" are:
                - adaptive_threshold=3.0
                - min_scene_len=15
                - window_width=2
                - min_content_val=15.0
                - weights=Components(delta_hue=1.0, delta_sat=1.0, delta_lum=1.0, delta_edges=0.0)
                - luma_only=False
                - kernel_size=None
                - video_manager=None
                - min_delta_hsv=None
        """
        self.show_progress = show_progress
        self.video_paths = self._get_video_paths(paths, recursive)
        self.transform = transform
        self.min_max_len = min_max_len
        self.duplicate_threshold = duplicate_threshold
        self.metric = self.init_metric(duplicate_metric)
        self.duplicate_metric = duplicate_metric

        self.detector_type = detector
        self.detector_kwargs = kwargs

        # This will be used to store inside the user home folder files containing the scene spliting
        self.scene_list_dir = self.get_scene_list_dir(
            detector_params={"detector": detector, **kwargs}
        )

        self.mapping_video_path_to_scene_file: Dict[
            str, str
        ] = self.retrieve_video_to_scene_list(self.scene_list_dir)

        self.scenes = self.retrieve_scenes(self.video_paths)
        # TODO : add a shuffle parameter
        # if True:
        #     random.shuffle(self.scenes)

    def compute_difference(
        self, current_frame: torch.tensor, next_frame: torch.tensor
    ) -> float:
        """Compute the difference between two frames.
        Args:
            current_frame (torch.tensor): Current frame.
            next_frame (torch.tensor): Next frame.
        Returns:
            float: Difference between the two frames.
        """
        if self.duplicate_metric == "lpips":
            # frames are between 0 and 1, scale between -1 and 1
            return self.metric(current_frame * 2 - 1, next_frame * 2 - 1).item()
        else:
            return self.metric(current_frame, next_frame).item()

    def init_metric(self, method: Literal["mse", "mae", "ssim"]) -> Callable:
        """Load the method used to compute the distance between two scenes.
        Args:
            method (Literal[&quot;mse&quot;, &quot;mae&quot;, &quot;lpips&quot;]): Method used to compute the distance between two scenes.
        Returns:
            Callable: Method used to compute the distance between two scenes.
        """

        if method == "mae":
            from torch.nn import L1Loss

            return L1Loss()
        elif method == "mse":
            from torch.nn import MSELoss

            return MSELoss()
        elif method == "lpips":
            from lpips import LPIPS

            return LPIPS(net="vgg")
        else:
            raise ValueError(f"Method {method} not supported")

    def load_detector(
        self, detector: Literal["content", "threshold", "adaptive"], **kwargs
    ) -> SceneDetector:
        """Load the detector used to detect the scenes in the videos.
        Args:
            detector (Literal[&quot;content&quot;, &quot;threshold&quot;, &quot;adaptive&quot;]): Method used to detect the scenes. Refer to the following link for more details about detectors: https://scenedetect.com/projects/Manual/en/latest/cli/detectors.html.
        Returns:
            SceneDetector: SceneDetector object to cut scenes.
        """
        if detector == "content":
            from scenedetect import ContentDetector

            return ContentDetector(**kwargs)
        elif detector == "threshold":
            from scenedetect import ThresholdDetector

            return ThresholdDetector(**kwargs)
        elif detector == "adaptive":
            from scenedetect import AdaptiveDetector

            return AdaptiveDetector(**kwargs)

    def get_scene_list_dir(self, detector_params: Dict) -> str:
        """Get the directory where the scene list files will be stored.
        It will be stored inside the user home folder in a folder called ".scene_dataset" and a subfolder with the name of the detector and its parameters.
        Args:
            detector_params (Dict): List of parameters given to the detector.
        Returns:
            str: Path where the scene list files will be stored.
        """
        scene_list_dir = os.path.join(
            os.path.expanduser("~"),
            ".scene_dataset",
            "_".join(str(val) for val in detector_params.values()),
        )
        if self.duplicate_threshold:
            scene_list_dir = os.path.join(
                scene_list_dir,
                f"duplicate_metric_{self.duplicate_metric}",
                f"threshold_{str(self.duplicate_threshold)}",
            )
        os.makedirs(scene_list_dir, exist_ok=True)
        return scene_list_dir

    def retrieve_scenes(
        self,
        video_paths: List[str],
    ) -> List[Scene]:
        """Get the scenes from the videos. If the scenes have already been computed, they will be loaded from the scene list files. Otherwise, they will be computed and saved in the scene list files.
        If min_max_len has been set, scenes will be cut to fit the min_max_len.
        Args:
            video_paths (List[str]): List of paths to the videos.
        Returns:
            List[Scene]: List of scenes holding the start and end frame of each scene.
        """
        scenes = []
        for video_path in video_paths:
            scenes.extend(self.retrieve_scenes_from_video(video_path))
        return scenes

    def retrieve_scenes_from_video(
        self,
        video_path: str,
    ) -> List[Scene]:
        """Get the scenes from a video. If the scenes have already been computed, they will be loaded from the scene list files. Otherwise, they will be computed and saved in the scene list files.
        If min_max_len has been set, scenes will be cut to fit the min_max_len.
        Args:
            video_path (str): Path to the video.
        Returns:
            List[Scene]: List of scenes holding the start and end frame of each scene.
        """

        if video_path in self.mapping_video_path_to_scene_file:
            scenes = self.load_precomputed_scenes(
                self.mapping_video_path_to_scene_file[video_path]
            )
        else:
            scenes = self.detect_scenes(video_path)
            save_path = os.path.join(
                self.scene_list_dir, f"{video_path.replace('/', '_')}.json"
            )
            self.save_scenes(scenes, save_path)
            self.mapping_video_path_to_scene_file[video_path] = save_path

            self.update_mapping()

        if self.duplicate_threshold is not None:
            scenes = self.remove_duplicate_from_scenes(scenes)
            save_path = os.path.join(
                self.scene_list_dir, f"{video_path.replace('/', '_')}.json"
            )
            self.save_scenes(scenes, save_path)
            self.mapping_video_path_to_scene_file[video_path] = save_path

            self.update_mapping()

        cut_scenes = self.cut_scenes_if_necessary(scenes)
        return cut_scenes

    def get_save_folder(self):
        """Get the folder where the dataset will be saved.
        Returns:
            str: Path to the folder where the dataset will be saved.
        """
        return os.path.join(self.save_dir, self.name)

    def remove_duplicate_from_scenes(self, scenes: List[Scene]) -> List[Scene]:
        """Remove duplicate scenes.
        Args:
            scenes (List[Scene]): List of scenes to remove duplicates from.
        Returns:
            List[Scene]: List of scenes without duplicates.
        """
        without_duplicates = []
        logger.info("Removing duplicates...")
        for scene in tqdm(scenes, desc="Removing duplicates..."):
            if scene.images_to_load is None:
                scene.images_to_load = self.get_images_to_load(scene)

            if (
                len(scene.images_to_load) > 1
            ):  # 1 because at least one image is always present
                without_duplicates.append(scene)
        return without_duplicates

    def get_images_to_load(self, scene: Scene) -> List[int]:
        """Get the images to load from a scene.
        Args:
            scene (Scene): Scene to get the images from.
        Returns:
            List[int]: List of images to load.
        """
        frames = self.load_frames(scene)

        # state = default_evaluator.run([[torch.zeros((1, 3, 100, 100)), torch.zeros((1, 3, 100, 100))]])
        # ssim(torch.rand((1, 3, 256, 512)), torch.rand((1, 3, 256, 512)))
        # ssim(current_frame, next_frame)

        frames = frames.permute(0, 3, 1, 2)
        frames = frames.float() / 255.0
        frames_to_load = []

        current_frame = frames[0:1]
        frames_to_load.append(scene.start)

        for i in tqdm(
            range(1, len(frames)), desc="Computing difference between frames..."
        ):
            next_frame = frames[i : i + 1]

            difference = self.compute_difference(current_frame, next_frame)
            if difference > self.duplicate_threshold:
                frames_to_load.append(scene.start + i)
                current_frame = next_frame
        return frames_to_load

    def save_scenes(self, scenes: List[Scene], save_path: str):
        """Save the scenes info in a json file.
        Args:
            scenes (List[Scene]): List of scenes to save.
            save_path (str): Path where the scenes will be saved.
        """
        with open(save_path, "w") as f:
            json.dump(scenes, f, cls=EnhancedJSONEncoder)

    def update_mapping(self):
        """Update the mapping between the video paths and the scene list files."""

        with open(
            os.path.join(self.scene_list_dir, SceneDataset.scene_list_file), "w"
        ) as f:
            json.dump(self.mapping_video_path_to_scene_file, f)

    def load_precomputed_scenes(self, scene_file: str) -> List[Scene]:
        with open(scene_file, "r") as f:
            scenes = json.load(f)

        scenes = [Scene(**scene) for scene in scenes]
        return scenes

    def detect_scenes(
        self,
        video_path: str,
    ) -> List[Scene]:
        """Use SceneDetect to detect the scenes in the video and generate a list of scenes with the start and end frame of each scene.
        Args:
            video_path (str): Path of the video to split.
        Returns:
            List[Scene]: List of scenes holding the start and end frame of each scene.
        """
        logger.info(f"Detecting scenes in {video_path}...")
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(
            self.load_detector(self.detector_type, **self.detector_kwargs)
        )
        # Detect all scenes in video from current position to end.
        scene_manager.detect_scenes(video, show_progress=self.show_progress)
        # `get_scene_list` returns a list of start/end timecode pairs
        # for each scene that was found.
        scenes = scene_manager.get_scene_list()
        scenes = [
            Scene(
                start=scene[0].get_frames(),
                end=scene[1].get_frames(),
                video_path=video_path,
            )
            for scene in scenes
        ]
        return scenes

    def cut_scenes_if_necessary(self, scenes: List[Scene]) -> List[Scene]:
        """Cut the scenes to fit the min_max_len if it has been set.
        Args:
            scenes (List[Scene]): List of scenes to cut.
        Returns:
            List[Scene]: Scenes that have been cut to fit the min_max_len.
        """
        if self.min_max_len is not None:
            scenes = self.cut_scenes(scenes)
        return scenes

    def cut_scenes(self, scenes: List[Scene]) -> List[Scene]:
        cut_scenes = []
        for scene in scenes:
            cut_scenes.extend(
                list(
                    self.cut_scene(
                        scene, self.get_value_to_split(scene, *self.min_max_len)
                    )
                )
            )
        cut_scenes = [scene for scene in cut_scenes if len(scene) > self.min_max_len[0]]
        return cut_scenes

    def get_value_to_split(
        self, scene: Scene, min_ideal_length: int, max_ideal_length: int
    ) -> int:
        """Get the best size to split the scene to fit the min_max_len.
        For instance if scene length is 50 frames, min ideal length is 15 and max ideal length is 25, the best size to split the scene is 25.
        Args:
            scene (Scene): The scene to split.
            min_ideal_length (int): The minimal length of the resulting scenes.
            max_ideal_length (int): The maximal length of the resulting scenes.
        Returns:
            int: The ideal length to split the scene.
        """
        n_elements = len(scene)

        lowest_remainder = max_ideal_length
        lowest_id = max_ideal_length
        for i in range(min_ideal_length, max_ideal_length):
            remainder = n_elements % i
            if remainder <= lowest_remainder:
                lowest_id = i
                lowest_remainder = remainder
        return lowest_id

    def cut_scene(self, scene: Scene, length: int) -> Scene:
        """Cut the scene into the specified length. If the scene cannot be cut perfectly into the specified length, the last scene will be shorter or longer than the others.
        Yields:
            Scene: Subscene of the scene after cutting.
        """
        n_splits = len(scene) // length
        if n_splits == 0:
            yield scene
        else:
            for i in range(n_splits):
                start = i * length
                end = (i + 1) * length
                if len(scene) - end < length:
                    end = len(scene)

                scene_start = scene.start + start
                scene_end = scene.start + end
                if scene.images_to_load is not None:
                    images_to_load = [
                        i
                        for i in scene.images_to_load
                        if i >= scene_start and i < scene_end
                    ]
                else:
                    images_to_load = None

                yield Scene(scene_start, scene_end, scene.video_path, images_to_load)

    def retrieve_video_to_scene_list(self, root: str) -> Dict[str, str]:
        """Retrieve the mapping between the video paths and the scene list files."""
        file_path = os.path.join(root, SceneDataset.scene_list_file)
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                video_to_scene_list = json.load(f)
        else:
            video_to_scene_list = {}
        return video_to_scene_list

    def _get_video_paths(self, paths: List[str], recursive: bool) -> List[str]:
        """Find all the video files in the paths. If a path is a folder, it will search recursively inside the folder if recursive is set to True.
        Args:
            paths (List[str]): List of paths to search for videos.
            recursive (bool): If True, it will search recursively inside the folders.
        Returns:
            List[str]: List of all available video paths.
        """
        video_paths = []
        for path in paths:
            for file_path in glob(path, recursive=recursive):
                if self.check_if_video(file_path):
                    video_paths.append(file_path)

            # if os.path.isfile(path):
            #     # File
            # else:
            #     logger.info(f"Finding video files inside {path} ...")
            #     # Folder
            #     for file_path in glob(os.path.join(path, "**"), recursive=recursive):
            #         if os.path.isfile(file_path):
            #             if self.check_if_video(file_path):
            #                 video_paths.append(file_path)

        video_paths = list(map(os.path.abspath, video_paths))
        logger.info(f"Found {len(video_paths)} videos")
        return sorted(video_paths)

    def check_if_video(self, path: str) -> bool:
        """Check the metadata of a file to see if it is a video.
        Args:
            path (str): Path of the file to check.
        Returns:
            bool: True if it is a video, False otherwise.
        """
        metadata = self.get_metadata(path)
        return metadata["codec_type"] == "video"

    def get_metadata(self, path: str) -> dict:
        """Use ffmpeg to retrieve the metadata of a file.
        Args:
            path (str): Path of the file to check.
        Returns:
            dict: Dictionary holding the metadata.
        """
        return ffmpeg.probe(path, select_streams="v")["streams"][0]

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx: int):
        scene = self.scenes[idx]
        frames = self.load_frames(scene)
        if self.transform:
            frames = self.transform(frames)
        return frames

    def load_frames(self, scene: Scene) -> torch.Tensor:
        """Load the frames of a scene.
        Args:
            scene (Scene): Scene to load.
        Returns:
            torch.Tensor: Tensor of shape (n_frames, height, width, channels).
        """
        vr = VideoReader(scene.video_path)
        if scene.images_to_load is not None:
            return vr.get_batch(scene.images_to_load)

        return vr.get_batch(range(scene.start, scene.end))
