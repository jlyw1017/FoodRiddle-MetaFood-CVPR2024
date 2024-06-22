import logging
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from data_utils.data_loader import SceneData


logger = logging.getLogger(__name__)


def decode_depth(rgb_depth) -> np.ndarray:
    """ This function takes an input as either the filepath to the depth map
        or the actual depth map image read via cv2

        Params:
        rgb_depth: Filepath or cv2 image of the RGB depth map. If the depth map
        is attached to the video frame, input only the depth map RGB.

        Output:
        depth: The pixel wise depth values in meters
    """
    if isinstance(rgb_depth, str):
        # Load the RGB depth image from the file
        rgb_depth_data = cv2.imread(rgb_depth)
    else:
        # Use the provided RGB depth image
        rgb_depth_data = rgb_depth

    if rgb_depth_data is None:
        logger.error(f"Could not read RGB depth image {rgb_depth}")
        raise ValueError(f"Could not read RGB depth image {rgb_depth}")

    # Convert RGB to HSV
    rgb_depth_data = rgb_depth_data.astype(np.float32)
    hsv_depth = cv2.cvtColor(rgb_depth_data, cv2.COLOR_BGR2HSV)[:, :, 0]
    print("min max hsv ", np.max(hsv_depth), np.min(hsv_depth))

    # Filters depth larger than 2m. (360 * 2m / 3m)
    hsv_depth[np.where(hsv_depth > 240)] = 0.0

    # Extracting hue and scaling to 0-3
    depth = hsv_depth / 2. / 255. * 3.
    return depth


def preprocess_input_data(meta: SceneData):
    """Creates folder for decoded depth."""
    decode_depth_path = os.path.join(meta.obj_root, "depth_value")
    if not os.path.exists(decode_depth_path):
        os.makedirs(decode_depth_path)
    depth_visual = os.path.join(meta.obj_root, "depth_visual")
    if not os.path.exists(depth_visual):
        os.makedirs(depth_visual)

    # Decodes depth for each frame.
    for frame_idx in meta.image_indices:
        _, _, mask_path = meta.get_image_path_by_index(frame_idx)
        img, rgb_depth, mask = meta.get_images_by_index(frame_idx)

        fix_mask(frame_idx, mask, meta)
        create_decoded_depth(depth_visual, frame_idx, meta, rgb_depth)


def create_decoded_depth(depth_visual, frame_idx, meta, rgb_depth):
    """Creates decoded depth image."""
    logger.info(f"Creating decoded depth for frame {frame_idx}")
    depth_values = decode_depth(rgb_depth)
    depth_decoded_path = meta.get_decoded_depth_by_index(frame_idx)
    # Before saving, convert to uint8.
    uint8_image = (depth_values * 255. / 3.1)
    cv2.imwrite(depth_decoded_path, uint8_image)

    # Output visualization of depth.
    depth_name = os.path.split(depth_decoded_path)[-1]
    depth_values = 255. * depth_values / np.max(depth_values)
    depth_visual_path = os.path.join(depth_visual, depth_name)
    cv2.imwrite(depth_visual_path, depth_values)
    return depth_values


def fix_mask(frame_idx, mask, meta):
    """Fixes the mesh mask."""
    logger.info(f"Fixing mask {frame_idx}")
    fixed_mask_path = meta.get_fixed_mask_path_by_index(frame_idx)
    mask[np.where(mask > 120)] = 255
    mask[np.where(mask < 120)] = 0
    im_floodfill = mask[:, :, 0].copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = mask.shape[:2]
    mask_ = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask_, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = mask[:, :, 0] | im_floodfill_inv
    cv2.imwrite(fixed_mask_path, im_out)


if __name__ == "__main__":

    # Example RGB depth data (replace this with your actual data)
    rgb_depth_data = cv2.imread("out.png")
    _, w, _ = rgb_depth_data.shape

    # If the depth map is attached to the video frame, extract only the depth map
    depth_map = rgb_depth_data[:, :int(w/2), :]

    # Decoding depth
    depth_values = decode_depth(depth_map)

    # Example output

    #note that there can be noise in the depth values, please make sure you filter them
    plt.imshow(depth_values)
    plt.show()
