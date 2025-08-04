"""Implements helper for masks."""

import cv2
import numpy as np

from PIL import Image, ImageFilter
from typing import Optional


def mask_preprocess_defaults(mask_inflation, mask_blur, model_name):
  if mask_inflation is None:
    if model_name == 'SD1.5':
      mask_inflation = 10
    elif model_name == 'SDXL':
      mask_inflation = 50
  if mask_blur is None:
    if model_name == 'SD1.5':
      mask_blur = 0.5
    elif model_name == 'SDXL':
      mask_blur = 30
  return mask_inflation, mask_blur


def process_mask(
    mask: Image.Image,
    mask_inflation: Optional[int] = None,
    mask_blur: Optional[int] = None
) -> Image.Image:
  """
  Inflates and blurs the white regions of a mask.
  Args:
      mask (Image.Image): The input mask image.
      mask_inflation (Optional[int]): The number of pixels to inflate the mask by.
      mask_blur (Optional[int]): The radius of the Gaussian blur to apply.
  Returns:
      Image.Image: The processed mask with inflated and/or blurred regions.
  """
  if mask_inflation and mask_inflation > 0:
    mask_array = np.array(mask)
    kernel = np.ones((mask_inflation, mask_inflation), np.uint8)
    mask_array = cv2.dilate(mask_array, kernel, iterations=1)
    mask = Image.fromarray(mask_array)

  if mask_blur and mask_blur > 0:
    mask = mask.filter(ImageFilter.GaussianBlur(radius=mask_blur))
  return mask
