import cv2
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from pathlib import Path
from rich.panel import Panel
from rich.table import Table
from rich import box, style

from gslUTILS.rich_utils import CONSOLE
from gslMASK.utils_mask import setup_mask, get_bounding_boxes, get_masks, disp_mask, process_images

class MaskProcessor:
  def __init__(
    self,
    data_dir: Path,
    prompt: str = "sky",
    inspect: bool = False,
  ):
    self.data_dir = Path(data_dir)
    self.prompt = prompt
    self.inspect = inspect

  def mask_loop(self, downscale_factor, image_paths, predictor, processor, dino, bt, tt):
    save_dir = self.data_dir.resolve().parent / "masks" 
    save_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_path in tqdm(enumerate(image_paths), total=len(image_paths), desc="Producing Masks", colour='GREEN', disable=False):
        # Load image
        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        boxes = get_bounding_boxes(image_pil, self.prompt, processor, dino, bt, tt)

        bool_mask = get_masks(image_pil, boxes, predictor).astype(bool)
        inverted_mask = ~bool_mask
        
        # Mask the image
        segmented = image_rgb.copy()
        segmented[inverted_mask] = 0

        alpha = (bool_mask.astype(np.uint8)) * 255
        rgba = np.dstack([image_rgb, alpha])

        binary_mask = (inverted_mask.astype(np.uint8)) * 255
        stem = os.path.splitext(os.path.basename(img_path))[0]
        number = stem.split('_')[-1]
        mask_path = f'{save_dir}/mask_{number}.png'

        # Show mask
        #self.inspect = True
        if self.inspect and idx % 10 == 0:
            disp_mask(image_rgb, rgba)

        # Save mask
        cv2.imwrite(mask_path, binary_mask)
        #cv2.imwrite(mask_path, rgba[:, :, [2, 1, 0, 3]]) 

    return save_dir

# main/driver function
  def run_mask_processing(self, bt, tt):
    image_paths, df = process_images(self.data_dir)
    predictor, processor, dino = setup_mask(self.data_dir)
    save_dir = self.mask_loop(df, image_paths, predictor, processor, dino, bt, tt)
    mask_dir = Path(save_dir).resolve()
    return mask_dir