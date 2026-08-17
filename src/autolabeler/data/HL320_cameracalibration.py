import os

import cv2
import numpy as np

from autolabeler.camera.colour_correction import simple_colour_correction
from autolabeler.camera.defisheye import defisheye


def create_list(path, verbose=False):
    files = np.sort(os.listdir(path))
    files = [item.split(".")[0] for item in files]
    if verbose:
        print(files[:10])
    return files


def mask_creator(
    img_dir_path,
    match_list,
    upper_radius: int = 860,
    lower_radius: int = 760,
    center=None,
) -> np.ndarray:
    allowed_extensions = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]

    if match_list and len(match_list[0]) == 2:
        base_name = match_list[0][1]
    else:
        base_name = match_list[0]
    image_path = None
    for extension in allowed_extensions:
        possible_path = os.path.join(img_dir_path, base_name + extension)
        if os.path.exists(possible_path):
            image_path = possible_path
            break

    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    if center is None:
        center_x, center_y = width // 2, height // 2 - 125
    else:
        center_x, center_y = center

    y, x = np.ogrid[:height, :width]
    distance_squared = (x - center_x) ** 2 + (y - center_y) ** 2
    upper_half = y < center_y
    lower_half = y >= center_y

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[upper_half & (distance_squared <= upper_radius**2)] = 255
    mask[lower_half & (distance_squared <= lower_radius**2)] = 255
    return mask


def masking_img(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    masked_img = cv2.bitwise_and(img, img, mask=binary_mask)
    return masked_img


def save_converted_data(
    image_folder_path,
    defisheye_flag: bool = False,
    colour_correction_flag: bool = False,
):
    img_dir_path = image_folder_path
    img_list = create_list(img_dir_path)
    clean_img_list = [img for img in img_list if img[:4] != "img2"]

    masked_image_dir_path = os.path.join(
        img_dir_path,
        f"img2_masked_{os.path.basename(img_dir_path)}",
    )
    original_image_dir_path = os.path.join(
        img_dir_path,
        f"img2_{os.path.basename(img_dir_path)}",
    )

    if defisheye_flag:
        defisheye_image_dir_path = os.path.join(
            img_dir_path,
            f"img2_defisheye_{os.path.basename(img_dir_path)}",
        )
    if colour_correction_flag:
        os.makedirs(masked_image_dir_path, exist_ok=True)
    os.makedirs(original_image_dir_path, exist_ok=True)
    mask_circle = mask_creator(
        img_dir_path,
        match_list=img_list,
        upper_radius=860,
        lower_radius=760,
    )

    allowed_extensions = [".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG"]
    for index, img_name in enumerate(clean_img_list):
        img_name = img_name.strip()
        img_path = None
        for extension in allowed_extensions:
            possible_path = os.path.join(img_dir_path, img_name + extension)
            if os.path.exists(possible_path):
                img_path = possible_path
                break

        if img_path is None:
            print(f"Warning: Image for {img_name!r} not found in {img_dir_path}")
            continue

        try:
            img_data = np.fromfile(img_path, dtype=np.uint8)
            camera_image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        except Exception:
            camera_image = None

        if camera_image is None:
            print(f"Error reading file: {img_path}")
            continue

        image = camera_image.copy()
        if colour_correction_flag:
            image = simple_colour_correction(image)
            masked_image_path = os.path.join(
                masked_image_dir_path,
                f"{index + 1:06d}.jpg",
            )
            masked_image = masking_img(image, mask=mask_circle)

        original_image_path = os.path.join(
            original_image_dir_path,
            f"{index + 1:06d}.jpg",
        )

        text_img = f"IMG: {img_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color_text = (255, 255, 255)
        thickness = 2

        cv2.putText(
            image,
            text_img,
            (20, 75),
            font,
            font_scale,
            color_text,
            thickness,
            cv2.LINE_AA,
        )
        if colour_correction_flag:
            cv2.putText(
                masked_image,
                text_img,
                (20, 75),
                font,
                font_scale,
                color_text,
                thickness,
                cv2.LINE_AA,
            )

        cv2.imwrite(original_image_path, image)
        if colour_correction_flag:
            cv2.imwrite(masked_image_path, masked_image)

        if defisheye_flag:
            defisheye(
                image=masked_image_path,
                output=os.path.join(defisheye_image_dir_path, f"{index:06d}.jpg"),
                output_size="3840x3060",
                format="circular",
                dtype="equalarea",
                projection="cylindrical",
                fov=190,
                pfov=140,
                pfov_axis="horizontal",
                xcenter=960,
                ycenter=750,
                radius=1068,
                interpolation="lanczos",
            )

    print("done")
