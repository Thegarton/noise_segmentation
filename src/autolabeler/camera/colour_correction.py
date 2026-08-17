from __future__ import annotations

import numpy as np 
import cv2
import argparse 
from pathlib import Path

def fix_cast_gray_world(image: np.ndarray) -> np.ndarray:
    b, g, r = cv2.split(image.astype(np.float32))
    
    # Find average intensity per channel
    mean_b, mean_g, mean_r = np.mean(b), np.mean(g), np.mean(r)
    
    # Global gray target
    mean_gray = (mean_b + mean_g + mean_r) / 3.0
    
    # Calculate adaptive gain factors
    kb = mean_gray / (mean_b + 1e-5)
    kg = mean_gray / (mean_g + 1e-5)
    kr = mean_gray / (mean_r + 1e-5)
    
    # Apply gain factors
    b = np.clip(b * kb, 0, 255)
    g = np.clip(g * kg, 0, 255)
    r = np.clip(r * kr, 0, 255)
    
    return cv2.merge([b, g, r]).astype(np.uint8)

# white_balance_test
def simple_wb(image: np.ndarray) -> np.ndarray:
    wb = cv2.xphoto.createSimpleWB()
    wb.setP(0.5)  # Percentile threshold for white point detection
    corrected = wb.balanceWhite(image)
    return corrected


def simple_colour_correction(img: np.ndarray) -> np.ndarray:
    wb_img = simple_wb(img)
    fixed_img = fix_cast_gray_world(wb_img)
    return fixed_img
