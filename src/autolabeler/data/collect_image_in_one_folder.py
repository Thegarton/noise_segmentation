collcet_iamge_in_one_folder
import os
from PIL import Image
import numpy 
from pathlib import Path


def _remove(folders, name:str):
    if name in folders:
        folders.remove(name)    
    return folders

def collect_overlay_image_in_one_folder(dir_path):
    folders = os.listdir(dir_path)

    folders = _remove(folders,'sam3_single_image_folder_manifest.json')
    folders = _remove(folders,'all_overlay')
    folders = _remove(folders,'all_combine')
    folders = _remove(folders,'sam3_image_match_manifest.json')
    folders = _remove(folders,'sam3_run_summary.csv')

    save_path = os.path.join(dir_path, "all_overlay")
    os.makedirs(save_path ,exist_ok=True)

    for frame in folders:
        _dir_path = os.path.join(dir_path, frame)
        IMAGE_PATH = os.path.join(_dir_path, "overlay.jpg")
        overlay = Image.open(IMAGE_PATH).convert("RGB")
        overlay.save(save_path + "/" + frame +"_overlay.jpg")



def collect_combine_image_in_one_folder(dir_path):
    folders = os.listdir(dir_path)

    folders = _remove(folders,'sam3_single_image_folder_manifest.json')
    folders = _remove(folders,'all_overlay')
    folders = _remove(folders,'all_combine')
    folders = _remove(folders,'sam3_image_match_manifest.json')
    folders = _remove(folders,'sam3_run_summary.csv')


    save_path = os.path.join(dir_path, "all_combine")
    os.makedirs(save_path ,exist_ok=True)

    for frame in folders:
        _dir_path = os.path.join(dir_path, frame)
        IMAGE_PATH = os.path.join(_dir_path, "preview.jpg")
        combine = Image.open(IMAGE_PATH).convert("RGB")
        combine.save(save_path + "/" + frame +"_combine.jpg")