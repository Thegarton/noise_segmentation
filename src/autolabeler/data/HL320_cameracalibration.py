import os
import numpy as np
from collections import defaultdict

from shutil import copy
import struct
import json
import cv2 
import pandas as pd

from autolabeler.camera.colour_correction import simple_colour_correction
from autolabeler.camera.defisheye import defisheye

def read_calib_txt(folder_path):
    txt_path = os.path.join(os.path.dirname(os.path.dirname((os.path.dirname(folder_path)))), "calibration_map.txt")
    with open(txt_path, "r", encoding="utf-8") as file:
        content = file.read().strip()

    if not content.startswith("{"):
        content = "{" + content + "}"

    try:
        top_params = json.loads(content)
    except json.JSONDecodeError as e:
        raise e

    return top_params


def Lidar2Camera_coord(slot, pixel, top_params):
    v = 3 * (16 * (slot // 21) + (pixel // 8)) + 1
    h = 3 * ((slot % 21) * 8 + (pixel % 8)) + 1
    
    # Determine LutVIndex
    if 0 <= v <= 95:
        lut_v_index = 0
    elif 96 <= v <= 191:
        lut_v_index = 6
    elif 192 <= v <= 287:
        lut_v_index = 12
    elif 288 <= v <= 383:
        lut_v_index = 18
    else:
        lut_v_index = 0

    # Determine LutHIndex
    if 0 <= h <= 83:
        lut_h_index = 0
    elif 84 <= h <= 167:
        lut_h_index = 1
    elif 168 <= h <= 251:
        lut_h_index = 2
    elif 252 <= h <= 335:
        lut_h_index = 3
    elif 336 <= h <= 419:
        lut_h_index = 4
    elif 420 <= h <= 503:
        lut_h_index = 5
    else:
        lut_h_index = 0


    fov_offset_index = lut_v_index + lut_h_index

    lx = h - 503 / 2
    ly = v - 383 / 2
    r = lx ** 2 + ly ** 2

    horizontal_strings = top_params["calibration"]["HorizontalAngleCoef"][fov_offset_index]
    vertical_strings = top_params["calibration"]["VerticalAngleCoef"][fov_offset_index]

    horizontal_coef = [float(x) for x in horizontal_strings]
    vertical_coef = [float(x) for x in vertical_strings]
    

    k1, k2, k3, k4, k5, k6, k7, k8 = map(float, horizontal_coef)
    p1, p2, p3, p4, p5, p6, p7, p8 = map(float, vertical_coef)
    
    c_xd = (
            k1 * lx * r +
            k2 * lx * (r ** 2) +
            k3 * lx * (r ** 3) +
            k4 * 2 * lx * ly +
            k5 * (r + 2 * (lx ** 2)) +
            k6 * lx +
            k7 * ly +
            k8
        )
        
    c_yd = (
        p1 * ly * r +
        p2 * ly * (r ** 2) +
        p3 * ly * (r ** 3) +
        p4 * 2 * lx * ly +
        p5 * (r + 2 * (ly ** 2)) +
        p6 * ly +
        p7 * lx +
        p8
    )
        
    return h, v, c_xd, c_yd




def convert_bin2velodyne(bin_path, csv_dir_path, velodyne_dir_path, file_index):
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Input .bin file doesn't exist: {bin_path}")

    with open(bin_path, "rb") as f:
        data = f.read()

    # Processing parameters
    # Data size
    gzipHeaderLen = 16
    batchPointLen = 52
    slotDataSize = 52 * 3 * 128
    
    csvHeader = [
        'x', 'y', 'z', 'azimuth', 'vertical', 
        'intensity', 'slot', 'pixel', "hcell",
        "vcell", "Cxd", "Cyd" 
    ]
        
    width = int.from_bytes(data[4:8], 'little')
    height = int.from_bytes(data[8:12], 'little')

    rows_list = []
    velodyne_point_list = []
    top_params = read_calib_txt(bin_path)
    
    for j in range(width):
        for i in range(height):
            ofs = gzipHeaderLen + i * batchPointLen * 3 + j * slotDataSize
            
            if ofs + batchPointLen > len(data):
                break

            # coordinates 
            x = struct.unpack("<f", data[ofs : ofs + 4])[0]
            y = struct.unpack("<f", data[ofs + 4 : ofs + 8])[0]
            z = struct.unpack("<f", data[ofs + 8 : ofs + 12])[0]
            
            azimuth  = struct.unpack("<f", data[ofs + 16 : ofs + 20])[0]
            vertical = struct.unpack("<f", data[ofs + 20 : ofs + 24])[0]
            
            # read reflectiviti channel
            intensity = int.from_bytes(data[ofs + 24 : ofs + 28], 'little')
            slot      = int.from_bytes(data[ofs + 32 : ofs + 36], 'little')
            pixel     = int.from_bytes(data[ofs + 36 : ofs + 40], 'little')
            
            
            h,v,c_xd, c_yd = Lidar2Camera_coord(slot,pixel, top_params)
            
            row = {
                'x': x, 'y': y, 'z': z, 'azimuth': azimuth, 'vertical': vertical,
                'intensity': intensity, 'slot': slot, 'pixel': pixel, "hcell": h,
                "vcell": v, "Cxd": c_xd, "Cyd": c_yd 
            }
            
            rows_list.append(row)

            velodyne_point_list.append([x, y, z, float(intensity)])
    
    # path to single csv file        
    csv_file_path = os.path.join(csv_dir_path, f"{file_index:06d}.csv")
    df = pd.DataFrame(rows_list, columns=csvHeader)
    df.to_csv(csv_file_path, index=False)


    np_points = np.array(velodyne_point_list, dtype=np.float32)

    # path to single .bin file
    velodyne_bin_path = os.path.join(velodyne_dir_path, f"{file_index:06d}.bin")
    np_points.tofile(velodyne_bin_path)
    return rows_list
        

def projection_Lidar2Img(row_list,img_file_path, projection_dir_path, file_index, match_list):
    camera_image = cv2.imread(img_file_path)
    overlay_image = camera_image.copy()
    
    pts_cam = []
    intensities = []

                
        
    for row in row_list:
        cam_x = int(row['Cxd'])
        cam_y = int(row["Cyd"])
        if 0 <= cam_x < camera_image.shape[1] and 0 <= cam_y < camera_image.shape[0]:
            pts_cam.append((cam_x, cam_y))
            intensities.append(row['intensity'])    

    for (cx, cy), inst in zip(pts_cam, intensities):
        color = (0, int(max(50, min(inst * 50, 255))), 0) 
        cv2.circle(overlay_image, (cx, cy), 2, color, -1)

    bin_name = match_list[file_index][0]
    img_name = match_list[file_index][1]
    
    text_bin = f"BIN: {bin_name}"
    text_img = f"IMG: {img_name}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8          
    color_text = (255, 255, 255) 
    thickness = 2             
    
    cv2.putText(overlay_image, text_bin, (20, 40), font, font_scale, color_text, thickness, cv2.LINE_AA)
    cv2.putText(overlay_image, text_img, (20, 75), font, font_scale, color_text, thickness, cv2.LINE_AA)
    


    output_image_path = os.path.join(projection_dir_path, f"{file_index:06d}.jpg")

    cv2.imwrite(output_image_path, overlay_image)
        
    

def create_list(path, verbose = False):
    files = np.sort(os.listdir(path))
    files = [_.split(".")[0] for _ in files]
    if verbose:
        print(files[:10])
    return files
    
def find_match(bin_name, img_list, line_data, index,i):
    result = index[bin_name][0]
    if result[1] in img_list:
        if i == 0:
            return [bin_name, result[1]]    
        return [bin_name, f"{(int(result[1])-3):06d}"]
    else:
        raise FileNotFoundError("not found img for bin in this folder")


def mask_creator(img_dir_path, match_list, upper_radius: int = 860, lower_radius: int = 760, center = None) -> np.ndarray:

    if match_list and len(match_list[0]) == 2:
        _img_path = os.path.join(img_dir_path, match_list[0][1] + ".jpg")
    else:
        _img_path = os.path.join(img_dir_path, match_list[0] + ".jpg")
    _image = cv2.imread(_img_path)
    h, w = _image.shape[:2]

    if center is None:
        center_x, center_y = w // 2, h // 2 - 125
    else:
        center_x, center_y = center

    y,x = np.ogrid[:h, :w]

    distance_squared = (x - center_x) ** 2 + (y - center_y) ** 2

    upper_half = y < center_y
    lower_half = y >= center_y

    mask = np.zeros((h, w), dtype=np.uint8)

    mask[upper_half & (distance_squared <= upper_radius ** 2)] = 255
    mask[lower_half & (distance_squared <= lower_radius ** 2)] = 255

    # mask_circle = np.zeros((h, w), dtype=np.uint8)
    # center_x, center_y = w // 2, h // 2 - 125

    # cv2.circle(mask_circle, (center_x, center_y), radius, 255, thickness=-1)

    return mask


def masking_img(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    masked_img = cv2.bitwise_and(img, img, mask=binary_mask)
    return masked_img


def save_converted_data(folder_path, output_folder_name):
    item_list = os.listdir(folder_path)
    
    img_dir_name = None 
    bin_dir_name = None
    txt_file_name = None
    

    for item in item_list:
        if item[-3:] == "img":
            img_dir_name = item
        elif item[-3:] == "bin":
            bin_dir_name = item
        elif item[-3:] == "txt":
            txt_file_name = item

    img_dir_path = os.path.join(folder_path,img_dir_name )
    bin_dir_path = os.path.join(folder_path, bin_dir_name) 
    match_txt = os.path.join(folder_path, txt_file_name)

    img_list = create_list(img_dir_path)
    bin_list = create_list(bin_dir_path)

    with open(match_txt, mode = "r") as f:
        line_data = [line.rstrip('\n').split(">>")  for line in f]

    index = defaultdict(list)
    for vec in line_data:
        index[vec[0]].append(vec)

    match_list = []
        
    for i in range(len(bin_list)):
        match = find_match(bin_list[i], img_list, line_data, index, i)
        match_list.append(match)
        
    print(match_list[:5])

    csv_path = os.path.join(folder_path, "csv_" + output_folder_name)
    velodyne_path = os.path.join(folder_path, "velodyne_" + output_folder_name)
    projection_path = os.path.join(folder_path, "projection_" + output_folder_name)
    masked_image_dir_path = os.path.join(folder_path, "img2_masked")
    original_image_dir_path = os.path.join(folder_path, "img2")

    os.makedirs(csv_path, exist_ok=True)
    os.makedirs(velodyne_path, exist_ok=True)
    os.makedirs(projection_path, exist_ok=True)
    os.makedirs(masked_image_dir_path, exist_ok=True)
    os.makedirs(original_image_dir_path, exist_ok=True)

    # creating np.array for mask
    mask_circle = mask_creator(img_dir_path,match_list=match_list, radius=860)



    for i in range(len(match_list)):
        bin_path = os.path.join( bin_dir_path, match_list[i][0]+".bin" )
        img_path = os.path.join(img_dir_path, match_list[i][1] + ".jpg")
        # print(bin_path)
        # print(img_path)
        if i%25 == 0:
            print(i)
        row_list = convert_bin2velodyne(bin_path, csv_path, velodyne_path, i)

        original_image_path = os.path.join(original_image_dir_path, f"{i:06d}.jpg")
        camera_image = cv2.imread(img_path)
        original_image = camera_image.copy()
    
        colour_corrected_img = simple_colour_correction(original_image)

        masked_image_path = os.path.join(masked_image_dir_path, f"{i:06d}.jpg")
        masked_image = masking_img(colour_corrected_img, mask=mask_circle)

        # increase brightness of the original image
        # brightness_increase = 40
        # brightened_image = cv2.convertScaleAbs(original_image, alpha=1.0, beta=brightness_increase)


        bin_name = match_list[i][0]
        img_name = match_list[i][1]
        
        text_bin = f"BIN: {bin_name}"
        text_img = f"IMG: {img_name}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1         
        color_text = (255, 255, 255) 
        thickness = 2             
        
        cv2.putText(colour_corrected_img, text_bin, (20, 40), font, font_scale, color_text, thickness, cv2.LINE_AA)
        cv2.putText(colour_corrected_img, text_img, (20, 75), font, font_scale, color_text, thickness, cv2.LINE_AA)

        cv2.putText(masked_image, text_bin, (20, 40), font, font_scale, color_text, thickness, cv2.LINE_AA)
        cv2.putText(masked_image, text_img, (20, 75), font, font_scale, color_text, thickness, cv2.LINE_AA)

        cv2.imwrite(original_image_path, colour_corrected_img)
        cv2.imwrite(masked_image_path, masked_image)

        if projection_path:
            projection_Lidar2Img(row_list, img_path, projection_path, i, match_list)
        
    print("done")



def save_converted_data_without_bin(image_folder_path, defisheye_flag: bool = False):
    img_dir_path = image_folder_path
    
    img_list = create_list(img_dir_path)

    clean_img_list = [] 
    for img in img_list:
        if img[:4] != "img2":
            clean_img_list.append(img)

    masked_image_dir_path = os.path.join(img_dir_path, f"img2_masked_{os.path.basename(img_dir_path)}")
    original_image_dir_path = os.path.join(img_dir_path, f"img2_{os.path.basename(img_dir_path)}")

    if defisheye_flag:
        defisheye_image_dir_path = os.path.join(img_dir_path, f"img2_defisheye_{os.path.basename(img_dir_path)}")

    os.makedirs(masked_image_dir_path, exist_ok=True)
    os.makedirs(original_image_dir_path, exist_ok=True)
    # creating np.array for mask
    mask_circle = mask_creator(img_dir_path,match_list=img_list, upper_radius=860, lower_radius=760)

    ALLOWED_EXTENSIONS = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']

    for i, img_name in enumerate(clean_img_list):
        # if i % 25 == 0:
        #     print(i)


        img_name = img_name.strip()
                
        img_path = None
        for ext in ALLOWED_EXTENSIONS:
            possible_path = os.path.join(img_dir_path, img_name + ext)
            if os.path.exists(possible_path):
                img_path = possible_path
                break
                
        if img_path is None:
            print(f"Warning: Image for '{img_name}' not found in {img_dir_path}")
            continue

        try:
            img_data = np.fromfile(img_path, dtype=np.uint8)
            camera_image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        except Exception:
            camera_image = None

        if camera_image is None:
            print(f"Error reading file: {img_path}")
            continue


        original_image = camera_image.copy()

        
        colour_corrected_img = simple_colour_correction(original_image)

        masked_image_path = os.path.join(masked_image_dir_path, f"{i:06d}.jpg")
        masked_image = masking_img(colour_corrected_img, mask=mask_circle)


        original_image_path = os.path.join(original_image_dir_path, f"{i:06d}.jpg")

        
        text_img = f"IMG: {img_name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color_text = (255, 255, 255)
        thickness = 2

        cv2.putText(colour_corrected_img, text_img, (20, 75), font, font_scale, color_text, thickness, cv2.LINE_AA)
        cv2.putText(masked_image, text_img, (20, 75), font, font_scale, color_text, thickness, cv2.LINE_AA)

        # 5. Сохранение результатов
        cv2.imwrite(original_image_path, colour_corrected_img)
        cv2.imwrite(masked_image_path, masked_image)

        if defisheye_flag:
            defisheye(
                image = masked_image_path,
                output = os.path.join(defisheye_image_dir_path, f"{i:06d}.jpg"),
                output_size = "3840x3060",
                format = "circular",
                dtype = "equalarea",
                projection = "cylindrical",
                fov = 190,
                pfov = 140,
                pfov_axis = "horizontal",
                xcenter = 960,
                ycenter = 750,
                radius = 1068,
                interpolation = "lanczos"
            )

    print("done")
