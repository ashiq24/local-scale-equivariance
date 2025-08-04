import os
import numpy as np
import torch
import cv2
import torchvision
import argparse
from PIL import Image
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from lama_inpaint import inpaint_img_with_lama
# from diffusers import StableDiffusionInpaintPipeline
import random
from mask import *
from collections import defaultdict

def uniform_sampling(imagenet_data, max_images):
    # Group images by their class
    class_to_images = defaultdict(list)
    for path, class_idx in imagenet_data.imgs:
        class_to_images[class_idx].append((path, class_idx))

    # Calculate number of classes and samples per class
    num_classes = len(class_to_images)
    per_class = max_images // num_classes

    # Verify divisibility
    if max_images % num_classes != 0:
        raise ValueError(f"max_images ({max_images}) must be divisible by number of classes ({num_classes})")

    # Sample images equally from each class
    selected_images = []
    for class_idx, images in class_to_images.items():
        if len(images) < per_class:
            raise ValueError(f"Class {class_idx} has only {len(images)} images, needed {per_class}")
        
        selected_images.extend(random.sample(images, per_class))

    # Optional shuffle to mix classes together
    random.shuffle(selected_images)
    return selected_images

class Inpainter:
    def __init__(self, 
                 mask_inflation=1, 
                 mask_blur=10,
                 GROUNDING_DINO_CONFIG = "./configs/GroundingDINO_SwinT_OGC.py",
                 GROUNDING_DINO_WEIGHTS = "./dino_sam_weights/groundingdino_swint_ogc.pth",
                 SAM_WEIGHTS = "./dino_sam_weights/sam_vit_h_4b8939.pth",
                 lama_config="./lama/configs/prediction/default.yaml",
                 lama_ckpt="./pretrained_models/big-lama",
                 device = "cuda" if torch.cuda.is_available() else "cpu",
                 debug=False
                 
                 ):
        self.mask_inflation = mask_inflation
        self.mask_blur = mask_blur
        self.DEVICE = device
        self.grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG,
            model_checkpoint_path=GROUNDING_DINO_WEIGHTS,
            device=self.DEVICE
        )
        self.sam = sam_model_registry["vit_h"](checkpoint=SAM_WEIGHTS).to(device=self.DEVICE)
        self.sam_predictor = SamPredictor(self.sam)
        self.lama_config = lama_config
        self.lama_ckpt = lama_ckpt
        self.debug = debug


    def get_mask(self, image, target_class, path):
        # Object detection using Grounding DINO with the target class (from folder label)
        detections, _ = self.grounding_dino_model.predict_with_caption(
            image=image,
            caption=target_class,
            box_threshold=0.35,
            text_threshold=0.25
        )
        
        # Extract bounding boxes and confidence scores
        boxes = torch.tensor(detections.xyxy).to(self.DEVICE)
        logits = torch.tensor(detections.confidence).to(self.DEVICE)
        if len(boxes) == 0:
            return None, None
        
        # Choose the detection with highest confidence
        max_idx = torch.argmax(logits).item()
        box = boxes[max_idx].cpu().numpy()
        
        # Set image for SAM and predict segmentation mask for the detected box
        self.sam_predictor.set_image(image)
        masks, logits_sam, _ = self.sam_predictor.predict(
            box=box[None, ...],
            multimask_output=False
        )
        best_index = np.argmax(logits_sam)
        original_mask = masks[best_index].astype(np.uint8)
        
        # get the objcet box
        x1, y1, x2, y2 = map(int, box)
        # check if the 80% of mask is inside the box
        if np.sum(original_mask[y1:y2, x1:x2])/np.sum(original_mask) < 0.8:
            print("**********Warning: 80% of mask is not inside the box**********")
            return None, None
        
        
        # save the mask to disk
        if self.debug:
            mask_pil = Image.fromarray(original_mask.astype(np.uint8) * 255)
            mask_pil.save(path+f"_orignalmask_{target_class}.JPEG")
        
        

        
        return original_mask , (x1, y1, x2, y2)
    
    def augment_mask(self, mask):
        mask = np.array(process_mask(Image.fromarray(mask * 255), self.mask_inflation, self.mask_blur))/255
        return mask
    
    def inpaint(self, image, mask, path):
        img = np.array(image)
        mask = np.array(mask)
        
        inpainted_img = inpaint_img_with_lama(
            img, mask, self.lama_config, self.lama_ckpt, device=self.DEVICE)

        inpainted_img = Image.fromarray(inpainted_img)
        if self.debug:
            inpainted_img.save(path+f"_inpainted_.JPEG")

        return inpainted_img
    
    def scale_and_paste(self, inpaited_image, object_img, mask, scale_factor, xyxy, path):
        result = np.array(inpaited_image.copy())
        temp_mask = mask.copy()
        
        # blur the temp mask
        #temp_mask = cv2.GaussianBlur(255.0 * temp_mask.astype(np.float32), (7, 7), 20)/255.0
        
       
        
        
        h, w = object_img.shape[:2]
        bh, bw = np.abs(xyxy[1] - xyxy[3]), np.abs(xyxy[2] - xyxy[0])
        
        object_center_x = (xyxy[0] + xyxy[2]) // 2
        object_center_y = (xyxy[1] + xyxy[3]) // 2
        
        new_size = (int(w * scale_factor), int(h * scale_factor))
        if scale_factor > 1:
            scaled_object = cv2.resize(object_img, new_size, interpolation=cv2.INTER_LANCZOS4)
        else:
            scaled_object = cv2.resize(object_img, new_size, interpolation=cv2.INTER_AREA)
        scaled_mask = cv2.resize(temp_mask.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST)
        
        # blur the scaled mask
        scaled_mask = cv2.GaussianBlur(255.0 * scaled_mask.astype(np.float32), (5, 5), 7,7)/255.0
         
        if self.debug:
            Image.fromarray((scaled_mask*255).astype(np.uint8)).save(path+f"_blurredmask_{scale_factor}.JPEG")
        
        scaled_object_center_x, scaled_object_center_y = int(object_center_x*scale_factor), int(object_center_y*scale_factor) 
        scaled_bh, scaled_bw = int(bh*scale_factor), int(bw*scale_factor)
        
        top_left_x = max(0, scaled_object_center_x-scaled_bw//2)
        top_left_y = max(0, scaled_object_center_y-scaled_bh//2)
        bottom_right_x = min(scaled_object.shape[1], scaled_object_center_x+scaled_bw//2)
        bottom_right_y = min(scaled_object.shape[0], scaled_object_center_y+scaled_bh//2)
        
        adjust_bw = bottom_right_x - top_left_x
        adjust_bh = bottom_right_y - top_left_y
        
        top_left_x_res = max(0, object_center_x-adjust_bw//2)
        top_left_y_res = max(0, object_center_y-adjust_bh//2)
        bottom_right_x_res = min(result.shape[1], top_left_x_res+adjust_bw)
        bottom_right_y_res = min(result.shape[0], top_left_y_res+adjust_bh)
        
        # adjust the scaled box
        bottom_right_x =  bottom_right_x -(adjust_bw - ( bottom_right_x_res - top_left_x_res))
        bottom_right_y =  bottom_right_y -(adjust_bh - ( bottom_right_y_res - top_left_y_res))
        
        assert bottom_right_x - top_left_x == bottom_right_x_res - top_left_x_res
        assert bottom_right_y - top_left_y == bottom_right_y_res - top_left_y_res
        
        
        extracted_result = result[top_left_y_res:bottom_right_y_res, top_left_x_res:bottom_right_x_res]
        extrated_object = scaled_object[top_left_y:bottom_right_y, top_left_x:bottom_right_x] 
        extracted_mask = scaled_mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        
        result[top_left_y_res:bottom_right_y_res, top_left_x_res:bottom_right_x_res] = extrated_object * extracted_mask[..., None] + (1-extracted_mask[..., None]) * extracted_result
        
        
        # Convert back to PIL Image and save
        result_pil = Image.fromarray(result)
        result_pil.save(path+f"_scale_{scale_factor}.JPEG")
        
        
    def process_image(self,
                    image_path,
                    target_class, 
                    scale_factors, 
                    path,
                    image_size=(512, 512)):
        """
        Processes a single image by:
        - Resizing the image.
        - Detecting an object using Grounding DINO with the provided target class.
        - Segmenting the detected object with SAM.
        - Extracting and scaling the object by scale_factor.
        - Performing inpainting on the gap left by the removed object with Stable Diffusion.
        - Pasting the scaled object back into the inpainted image.
        
        Returns the processed image as a PIL Image.
        """
        # first save the image to disk
        #image_pil.save(path+f"_original_{target_class}.JPEG")
        # Load and resize image
        image_pil = Image.open(image_path).convert("RGB")
        #image_pil = image_pil.resize(image_size)
        
        image = np.array(image_pil)
        mask, xyxy = self.get_mask(image, target_class, path) # this is between 0-1
        if mask is None:
            return 0
            
        mask_of_inpainting = self.augment_mask(mask)
        
        if self.debug:
            Image.fromarray((mask_of_inpainting*255).astype(np.uint8)).save(path+f"_mask_of_inpainting_.JPEG")
        
        # Extract the object with the mask and save it
        
        object_img = image * mask[..., None]
        
        if self.debug:
            Image.fromarray((object_img).astype(np.uint8)).save(path+f"_extracted_object_.JPEG")
        
        inpaited_image = self.inpaint(image, mask_of_inpainting, path) # this returns np array 0-255
        if self.debug:
            image_pil.save(path+f"_original_{target_class}.JPEG")
        for scale_factor in scale_factors:
            self.scale_and_paste(inpaited_image, object_img, mask, scale_factor, xyxy, path)
                
        return 1


def process_dataset(dataset_path,
                    output_path,
                    subset="train", 
                    scale_factors=[ 0.75, 0.5, 0.25],
                    mask_inflation=1,
                    mask_blur=10,
                    max_images=1000,
                    debug=False,
                    start_index=0,):
    """
    Processes up to max_images images from the specified ImageNet subset (train or validation).
    
    For each image:
    - The target class is derived from the folder name containing the image.
    - The image is processed for each specified scale factor.
    - The output is saved in a corresponding output directory that maintains the ImageNet subfolder structure.
    """
    image_count = 0
    # Build the full input dataset directory (e.g., dataset_path/train)
    imagenet_data =torchvision.datasets.ImageNet(root=dataset_path, split=subset)
    classes = imagenet_data.classes
    inpainter = Inpainter(mask_blur=mask_blur, mask_inflation=mask_inflation, debug=debug)
    selected_images =  random.sample(imagenet_data.imgs, max_images)
    
    print("********selected images*********", len(selected_images))
    
    for idenx, (img_path, class_id) in enumerate(selected_images):
        if idenx < start_index:
            continue
        target_class = classes[class_id][0]
        for more_labels in classes[class_id][1:]:
            target_class = target_class + '  ' + more_labels
        folder_name = img_path.split("/")[-2]
        image_name = img_path.split("/")[-1]
        file_loc = output_path +'/'+folder_name + '/' + image_name[:-5]
        print("********file location*********", output_path)
        print("********target class*********", target_class)
        os.makedirs(output_path +'/'+folder_name , exist_ok=True)
        try:
            count = inpainter.process_image(img_path, target_class, scale_factors, file_loc)
        except Exception as e:
            print(e)
            count = 0
            continue
        image_count += count
        print(f"Processed {image_count} images")
    



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--dataset_path",
        type=str,
        default="/datasets/imagenet",
        help="Path to the dataset folder.",
    )
    argparser.add_argument(
        "--output_path",
        type=str,
        default="../demo_train_1",
        help="Path to the output folder.",
    )
    argparser.add_argument(
        "--subset",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Subset of the dataset to process.",
    )
    argparser.add_argument(
        "--scale_factors",
        type=float,
        nargs="+",
        default=[1.3, 1.2, 1.1, 1, 0.9, 0.8, 0.7],
        help="Scale factors for object resizing.",
    )
    argparser.add_argument(
        "--max_images",
        type=int,
        default=10000,
        help="Maximum number of images to process.",
    )
    argparser.add_argument(
        "--mask_inflation",
        type=int,
        default=5,
        help="Mask inflation parameter.",
    )
    argparser.add_argument(
        "--mask_blur",
        type=int,
        default=2,
        help="Mask blur parameter.",
    )
    argparser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
    )
    
    args = argparser.parse_args()
    process_dataset(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        subset=args.subset,
        scale_factors=args.scale_factors,
        max_images=args.max_images,
        mask_inflation=args.mask_inflation,
        mask_blur=args.mask_blur,
        debug=args.debug
    )