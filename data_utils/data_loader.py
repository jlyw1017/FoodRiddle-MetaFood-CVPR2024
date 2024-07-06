"""Helper tools to read files from the dataset."""
import enum
import glob
import logging
import os.path
from enum import Enum
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# The difficulty level of the data.
class ObjType(Enum):
    SIMPLE = 'simple'
    MEDIUM = 'medium'
    HARD = 'hard'


def string_to_enum(string: str, enum_class: enum.IntEnum):
    """Converts the string to corresponding object type.

    Args:
        string: the input string.
        enum_class: the target enum to convert to.
    """
    for member in enum_class:
        if member.name.lower() == string.lower():
            return member
    raise ValueError(f"No matching enum member found for string '{string}'.")


def get_file_name_by_index(index: int):
    """Gets the image by frame index.

    Args:
        index (int): the index of the frame

    Returns:
        origin (str): the file name of the orginal rgb image.
        depth (str): the file name of the depth image.
        mask (str): the file name of the mask image.
    """
    origin = f"frame_{str(index).zfill(4)}_original.jpg"
    depth = f"frame_{str(index).zfill(4)}_depth.jpg"
    mask = f"frame_{str(index).zfill(4)}_original_segmented_mask.jpg"
    return origin, depth, mask


class SceneData(object):
    """The container of the metas of a single object.

    Attributes:
        obj_root (str): the root path of the object.
        obj_index (int): the index of the object.
        obj_id (str): the id of the object.
        obj_name (str): the name of the object.
        frame_num (int): the number of frames.
        obj_type (ObjType): the difficulty level of the object.
    """
    def __init__(self, obj_root: str, obj_index: int, obj_id: str,
                 obj_name: str, frame_num: int):
        """Initialize the Scene."""
        self.obj_root: str = obj_root
        self.obj_index: int = obj_index
        self.obj_id: str = obj_id
        self.obj_name: str = obj_name
        self.frame_num: int = frame_num
        difficulty, _ = obj_id.split("_")
        self.obj_type = string_to_enum(difficulty, ObjType)

    def get_img(self, index: int) -> np.ndarray:
        """Gets the image by index.

        Args:
            index (int): the index of the frame
        """
        origin_path, _, _ = self.get_image_path_by_index(index)
        return cv2.imread(origin_path)

    def get_fixed_mask(self, index: int) -> np.ndarray:
        """Gets the fixed mask image.

        Args:
            index (int): the index of the frame

        Returns:
            np.ndarray: the image of the fixed mask
        """
        return cv2.imread(self.get_fixed_mask_path_by_index(index))

    def get_decoded_depth(self, index: int, vis: bool = False) -> np.ndarray:
        """Gets the image of the decoded depth.

        Args:
            index (int): the index of the frame
            vis (bool): whether to visualize the depth map

        Returns:
            np.ndarray: the image of the decoded depth
        """
        return cv2.imread(self.get_decoded_depth_by_index(index, vis=vis))

    def get_decoded_depth_vis(self, index: int):
        """Gets the image of the decoded depth.

        Args:
            index (int): the index of the frame

        Returns:
            np.ndarray: the image of the decoded depth
        """
        return cv2.imread(self.get_decoded_depth_by_index(index))

    @property
    def mesh_path(self) -> str:
        """Gets the path of the mesh file.

        Returns:
            str: the path of the mesh file
        """
        return os.path.join(self.obj_root, "mesh.ply")

    @property
    def image_indices(self) -> List[int]:
        """Gets the list of all image indices.
        
        Returns:
            List[int]: the list of all image indices
        """
        return list(range(0, self.frame_num))

    @property
    def image_folder_path(self) -> str:
        """Gets the path of the original images

        Returns:
            str: the path of the original images
        """
        return os.path.join(self.obj_root, "Original")

    @property
    def mask_folder_path(self) -> str:
        """Gets the path of the mask images

        Returns:
            str: the path of the mask images
        """
        return os.path.join(self.obj_root, "Mask")

    @property
    def fixed_mask_folder_path(self) -> str:
        """Gets the path of the fixed mask images

        Returns:
            str: the path of the fixed mask images
        """
        return os.path.join(self.obj_root, "fixed_mask")

    @property
    def depth_folder_path(self) -> str:
        """Gets the path of the depth images

        Returns:
            str: the path of the depth images
        """
        return os.path.join(self.obj_root, "Depth")

    @property
    def food_folder(self) -> str:
        """Gets the path of the food images

        Returns:
            str: the path of the food images
        """
        return os.path.join(self.obj_root, "food")

    def get_food_path(self, index: int) -> str:
        """Gets the path of the food image

        Returns:
            str: the path of the food image
        """
        return os.path.join(self.food_folder,
                            f"frame_{str(index).zfill(4)}_original.jpg")

    @property
    def background_folder(self) -> str:
        """Gets the path of the background images

        Returns:
            str: the path of the background images
        """
        return os.path.join(self.obj_root, "background")

    def get_background_path(self, index: int) -> str:
        """Gets the path of the background image

        Returns:
            str: the path of the background image
        """
        return os.path.join(self.background_folder,
                            f"frame_{str(index).zfill(4)}.png")

    def get_image_path_by_index(self, index: int) -> (str, str, str):
        """Gets the image by index.
        
        Args:
            index (int): the index of the frame

        Returns:
            origin (str): the file name of the orginal rgb image.
            depth (str): the file name of the depth image.
            mask (str): the file name of the mask image.
        """
        if self.obj_type == ObjType.SIMPLE or self.obj_type == ObjType.MEDIUM:
            origin, depth, mask = get_file_name_by_index(index)
            return (os.path.join(self.image_folder_path, origin),
                    os.path.join(self.depth_folder_path, depth),
                    os.path.join(self.mask_folder_path, mask))
        if self.obj_type == ObjType.HARD:
            return (os.path.join(self.image_folder_path, "monocular.jpg"),
                    os.path.join(self.depth_folder_path, "depth.jpg"),
                    glob.glob(os.path.join(self.obj_root, "*.jpg"))[0])
        raise ValueError(f"Unsupported object type: {self.obj_type}")

    def get_fixed_mask_path_by_index(self, index: int) -> str:
        """Gets the decoded depth path

        Returns:
            str: the path of the decoded depth
        """
        folder = self.fixed_mask_folder_path
        if not os.path.exists(folder):
            os.mkdir(folder)
        _, _, mask_file_name = get_file_name_by_index(index)
        file_name, _ = os.path.splitext(mask_file_name)
        return os.path.join(folder, file_name + ".png")

    def get_decoded_depth_by_index(self, index: int, vis=False) -> str:
        """Gets the decoded depth path
        
        Returns:
            str: the path of the decoded depth
        """
        folder = os.path.join(self.obj_root,
                              "depth_visual" if vis else "depth_value")
        if self.obj_type == ObjType.SIMPLE or self.obj_type == ObjType.MEDIUM:
            _, depth_file_name, _ = get_file_name_by_index(index)
            file_name, _ = os.path.splitext(depth_file_name)
            return os.path.join(folder, file_name + ".png")
        if self.obj_type == ObjType.HARD:
            return os.path.join(folder, "depth.png")

    def get_images_by_index(self, index: int
                            ) -> (np.ndarray, np.ndarray, np.ndarray):
        """Gets the image by index."""
        origin_path, depth_path, mask_path = self.get_image_path_by_index(index)
        origin, depth, mask = (cv2.imread(origin_path),
                               cv2.imread(depth_path),
                               cv2.imread(mask_path))

        for img_path, data in zip((origin_path, depth_path, mask_path),
                                  (origin, depth, mask)):
            if data is None:
                logger.error(f"Failed to read the image from {img_path}.")
                raise ValueError(f"Failed to read the image from {img_path}.")
        return origin, depth, mask


class DataReader(object):
    """A helper tool to read the data.

    Attributes:
        meta (SceneMeta): the meta information of the scene
    """
    def __init__(self, scene_meta: SceneData):
        self.meta = scene_meta


class MTFDataSet(object):
    """Helper class for the io of the dataset.
    
    Attributes:
        dataset_root (str): the root directory of the dataset
        scene_by_index (Dict[int, SceneData]): the dictionary of the scene meta
            key: the index of the scene
            value: the meta information of the scene
    """
    def __init__(self, dataset_root: str):
        self.dataset_root: str = dataset_root
        self.scene_by_index: Dict[int, SceneData] = {}
        self.read_data_meta(os.path.join(self.dataset_root,
                                         "source-files/data_meta.xls"))

    def read_data_meta(self, xls_path: str):
        """Reads meta file of the dataset.
        
        Args:
            xls_path (str): the path of the meta file
        """
        obj_metas = pd.read_excel(xls_path).values.tolist()
        for idx, obj_id, name, frame_num in obj_metas:
            obj_root = os.path.join(self.dataset_root, str(idx))
            scene_meta = SceneData(obj_root, idx, obj_id, name, frame_num)
            self.scene_by_index[idx] = scene_meta

    def get_all_metas(self) -> List[SceneData]:
        """Gets the list of all metas."""
        return list(self.scene_by_index.values())

    def get_meta_by_index(self, index: int) -> SceneData:
        """Gets the scene meta by index."""
        return self.scene_by_index[int(index)]

    def get_meta_by_type(self, obj_type: ObjType) -> List[SceneData]:
        """Gets the scene meta by type."""
        results = []
        for obj_idx, obj in self.scene_by_index.items():
            if obj.obj_type == obj_type:
                results.append(obj)
        return results
