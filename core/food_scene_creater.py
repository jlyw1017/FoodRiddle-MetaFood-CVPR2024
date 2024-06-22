import copy
import logging
import os

import cv2
import numpy as np

from data_utils.data_loader import SceneData
from data_utils.food import FoodScene


logger = logging.getLogger(__name__)


def create_obj(meta: SceneData):
    # split image and mask to background and food
    scene = FoodScene(meta)
    if not os.path.exists(meta.food_folder):
        os.makedirs(meta.food_folder)

    for frame_index in meta.image_indices:
        logger.info(f"Processing frame {frame_index}...")
        img, _, _ = meta.get_images_by_index(frame_index)
        mask = meta.get_fixed_mask(frame_index)
        depth = meta.get_decoded_depth(frame_index)

        # Filter food with mask.
        food_img = copy.copy(img)
        food_img[np.where(mask == 0)] = 255
        cv2.imwrite(meta.get_food_path(frame_index), food_img)

        food_depth_img = copy.copy(depth)
        food_depth_img[np.where(mask == 0)] = 0

        # Filter background with mask.
        mask_large_dilation = cv2.dilate(
            mask, np.ones((10, 10), np.uint8), iterations=20)

        mask_small_dilation = cv2.dilate(
            mask, np.ones((3, 3), np.uint8), iterations=3)
        background_mask = np.clip(
            mask_large_dilation - mask_small_dilation, 0, 255)

        background_img = copy.copy(img)
        background_img[np.where(background_mask == 0)] = 0
        background_depth = copy.copy(depth)
        background_depth[np.where(background_mask == 0)] = 0

        scene.food.img_by_idx[frame_index] = food_img
        scene.food.depth_by_idx[frame_index] = food_depth_img
        scene.background.img_by_idx[frame_index] = background_img
        scene.background.depth_by_idx[frame_index] = background_depth

    return scene


def enlarge_roi(scene: FoodScene):
    """Enlarges the food region in the image."""
    img = scene.food.img_by_idx[0]
    depth = scene.food.depth_by_idx[0]
    output_folder = os.path.join(scene.meta.obj_root, "enlarged_img")
    mask = scene.meta.get_fixed_mask(0)
    find_and_enlarge_roi(mask, img, depth, output_folder)


def find_and_enlarge_roi(mask, image, depth, output_folder: str):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    contours, _ = cv2.findContours(mask[:, :, 0], cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found in the mask.")
        return

    # 假设我们只对最大的区域感兴趣
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # 计算放大因子
    scale_width = image.shape[1] / w
    scale_height = image.shape[0] / h
    scale = min(scale_width, scale_height)  # 保持比例不变

    # 放大边界框内的区域
    img_roi = cv2.resize(image[y:y + h, x:x + w],
                         (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_LINEAR)
    depth_roi = cv2.resize(depth[y:y + h, x:x + w],
                           (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_LINEAR)

    # 创建一个新的空白图像来放置放大后的区域
    output_image = np.zeros_like(image, dtype=image.dtype)
    output_depth_image = np.zeros_like(depth, dtype=image.dtype)

    # 计算放大后区域在新图像中的位置（居中）
    new_x = (image.shape[1] - img_roi.shape[1]) // 2
    new_y = (image.shape[0] - img_roi.shape[0]) // 2

    # 将放大后的区域放置在新图像的中心位置
    output_image[new_y:new_y + img_roi.shape[0],
    new_x:new_x + img_roi.shape[1]] = img_roi
    output_depth_image[new_y:new_y + depth_roi.shape[0],
    new_x:new_x + depth_roi.shape[1]] = depth_roi

    cv2.imwrite(os.path.join(output_folder, "food.png"), output_image)
    cv2.imwrite(os.path.join(output_folder, "depth.png"), output_depth_image)
