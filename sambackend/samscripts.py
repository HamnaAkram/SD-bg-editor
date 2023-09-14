import numpy as np
from .predictor import SamPredictorHQ
from PIL import Image
from modules import scripts, shared
from modules.devices import device, torch_gc, cpu
from collections import OrderedDict
import os, cv2, torch, gc, copy
from modules.safe import unsafe_torch_load, load
from .build_sam_hq import sam_model_registry
from .dino import dino_predict_internal

sam_device = device
sam_model_cache = OrderedDict()
from modules.paths import models_path
scripts_sam_model_dir = os.path.join(scripts.basedir(), "models/sam") 
sd_sam_model_dir = os.path.join(models_path, "sam")
sam_model_dir = sd_sam_model_dir if os.path.exists(sd_sam_model_dir) else scripts_sam_model_dir
sam_model_list = [f for f in os.listdir(sam_model_dir) if os.path.isfile(os.path.join(sam_model_dir, f)) and f.split('.')[-1] != 'txt']



def show_boxes(image_np, boxes, color=(255, 0, 0, 255), thickness=2, show_index=False):
    if boxes is None:
        return image_np

    image = copy.deepcopy(image_np)
    for idx, box in enumerate(boxes):
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (w, h), color, thickness)
        if show_index:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = str(idx)
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            cv2.putText(image, text, (x, y+textsize[1]), font, 1, color, thickness)

    return image


def garbage_collect(sam):
    if shared.cmd_opts.lowvram:
        sam.to(cpu)
    gc.collect()
    torch_gc()

def clear_sam_cache():
    sam_model_cache.clear()
    gc.collect()
    torch_gc()

def load_sam_model(sam_checkpoint):
    model_type = sam_checkpoint.split('.')[0]
    if 'hq' not in model_type:
        model_type = '_'.join(model_type.split('_')[:-1])
    sam_checkpoint_path = os.path.join(sam_model_dir, sam_checkpoint)
    torch.load = unsafe_torch_load
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
    sam.to(device=sam_device)
    sam.eval()
    torch.load = load
    return sam

def init_sam_model(sam_model_name):
    print(f"Initializing SAM to {sam_device}")
    if sam_model_name in sam_model_cache:
        sam = sam_model_cache[sam_model_name]
        if shared.cmd_opts.lowvram or (str(sam_device) not in str(sam.device)):
            sam.to(device=sam_device)
        return sam
    elif sam_model_name in sam_model_list:
        clear_sam_cache()
        sam_model_cache[sam_model_name] = load_sam_model(sam_model_name)
        return sam_model_cache[sam_model_name]
    else:
        raise Exception(
            f"{sam_model_name} not found, please download model to models/sam.")

def show_masks(image_np, masks: np.ndarray, alpha=0.5):
    image = copy.deepcopy(image_np)
    np.random.seed(0)
    for mask in masks:
        color = np.concatenate([np.random.random(2), np.array([0.6])], axis=0)
        image[mask] = image[mask] * (1 - alpha) + 255 * color.reshape(1, 1, -1) * alpha
    return image.astype(np.uint8)

def create_mask_output(image_np, masks, boxes_filt):
    print("Creating output image")
    mask_images, masks_gallery, matted_images = [], [], []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        masks_gallery.append(Image.fromarray(np.any(mask, axis=0)))
        blended_image = show_masks(show_boxes(image_np, boxes_filt), mask)
        mask_images.append(Image.fromarray(blended_image))
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0])
        matted_images.append(Image.fromarray(image_np_copy))
    return masks_gallery #+ matted_images + mask_images 

def sam_predict(dummy_c, input_image, positive_points, negative_points):
    sam_model_name = sam_model_list[0]
    # ,dino_checkbox, dino_model_name, text_prompt, box_threshold,
    # dino_preview_checkbox, dino_preview_boxes_selection):
    print("Start SAM Processing")
    if sam_model_name is None:
        return [], "SAM model not found. Please download SAM model from extension README."
    if input_image is None:
        return [], "SAM requires an input image. Please upload an image first."
    
    image_np = np.array(input_image)
    image_np_rgb = image_np[..., :3]
    # dino_enabled = dino_checkbox and text_prompt is not None
    boxes_filt = None
    sam_predict_result = " done."
    # if dino_enabled:
    #     boxes_filt, install_success = dino_predict_internal(input_image, dino_model_name, text_prompt, box_threshold)
    #     if dino_preview_checkbox is not None and dino_preview_checkbox and dino_preview_boxes_selection is not None:
    #         valid_indices = [int(i) for i in dino_preview_boxes_selection if int(i) < boxes_filt.shape[0]]
    #         boxes_filt = boxes_filt[valid_indices]
    sam = init_sam_model(sam_model_name)
    print(f"Running SAM Inference {image_np_rgb.shape}")
    predictor = SamPredictorHQ(sam, 'hq' in sam_model_name)
    predictor.set_image(image_np_rgb)
    
    # if dino_enabled and boxes_filt.shape[0] > 1:
    #     sam_predict_status = f"SAM inference with {boxes_filt.shape[0]} boxes, point prompts discarded"
    #     print(sam_predict_status)
    #     transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_np.shape[:2])
    #     masks, _, _ = predictor.predict_torch(
    #         point_coords=None,
    #         point_labels=None,
    #         boxes=transformed_boxes.to(sam_device),
    #         multimask_output=True)
    #     masks = masks.permute(1, 0, 2, 3).cpu().numpy()
    # else:

    num_box = 0 if boxes_filt is None else boxes_filt.shape[0]
    num_points = len(positive_points) + len(negative_points)
    if num_box == 0 and num_points == 0:
        garbage_collect(sam)
        # if dino_enabled and dino_preview_checkbox and num_box == 0:
        #     return [], "It seems that you are using a high box threshold with no point prompts. Please lower your box threshold and re-try."
        return [], "You neither added point prompts nor enabled GroundingDINO. Segmentation cannot be generated."
    sam_predict_status = f"SAM inference with {num_box} box, {len(positive_points)} positive prompts, {len(negative_points)} negative prompts"
    print(sam_predict_status)
    point_coords = np.array(positive_points + negative_points)
    point_labels = np.array([1] * len(positive_points) + [0] * len(negative_points))
    box = copy.deepcopy(boxes_filt[0].numpy()) if boxes_filt is not None and boxes_filt.shape[0] > 0 else None
    masks, _, _ = predictor.predict(
        point_coords=point_coords if len(point_coords) > 0 else None,
        point_labels=point_labels if len(point_coords) > 0 else None,
        box=box,
        multimask_output=True)
    masks = masks[:, None, ...]
    garbage_collect(sam)
    return create_mask_output(image_np, masks, boxes_filt)#, sam_predict_status + sam_predict_result + (f" However, GroundingDINO installment has failed. Your process automatically fall back to local groundingdino. Check your terminal for more detail and...")

def sam_dino_predict(input_image):
    print("Start SAM Processing")
    sam_model_name = sam_model_list[0]
    positive_points = []
    negative_points = []
    dino_preview_checkbox = False
    dino_preview_boxes_selection = []
    dino_enabled = True and 'face' is not None

    if input_image is None:
        return [], "SAM requires an input image. Please upload an image first."
    image_np = np.array(input_image)
    image_np_rgb = image_np[..., :3]
    dino_enabled = True
    boxes_filt = None
    sam_predict_result = " done."
    if dino_enabled:
        boxes_filt, install_success = dino_predict_internal(input_image, 'GroundingDINO_SwinT_OGC (694MB)', 'face', 0.3)
        if dino_preview_checkbox is not None and dino_preview_checkbox and dino_preview_boxes_selection is not None:
            valid_indices = [int(i) for i in dino_preview_boxes_selection if int(i) < boxes_filt.shape[0]]
            boxes_filt = boxes_filt[valid_indices]
    sam = init_sam_model(sam_model_name)
    print(f"Running SAM Inference {image_np_rgb.shape}")
    predictor = SamPredictorHQ(sam, 'hq' in sam_model_name)
    predictor.set_image(image_np_rgb)
    if dino_enabled and boxes_filt.shape[0] > 1:
        sam_predict_status = f"SAM inference with {boxes_filt.shape[0]} boxes, point prompts discarded"
        print(sam_predict_status)
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image_np.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(sam_device),
            multimask_output=True)
        masks = masks.permute(1, 0, 2, 3).cpu().numpy()
    else:
        
        num_box = 0 if boxes_filt is None else boxes_filt.shape[0]
        num_points = len(positive_points) + len(negative_points)
        if num_box == 0 and num_points == 0:
            garbage_collect(sam)
            if dino_enabled and dino_preview_checkbox and num_box == 0:
                return [], "It seems that you are using a high box threshold with no point prompts. Please lower your box threshold and re-try."
            return [], "You neither added point prompts nor enabled GroundingDINO. Segmentation cannot be generated."
        sam_predict_status = f"SAM inference with {num_box} box, {len(positive_points)} positive prompts, {len(negative_points)} negative prompts"
        print(sam_predict_status)
        point_coords = np.array(positive_points + negative_points)
        point_labels = np.array([1] * len(positive_points) + [0] * len(negative_points))
        box = copy.deepcopy(boxes_filt[0].numpy()) if boxes_filt is not None and boxes_filt.shape[0] > 0 else None
        masks, _, _ = predictor.predict(
            point_coords=point_coords if len(point_coords) > 0 else None,
            point_labels=point_labels if len(point_coords) > 0 else None,
            box=box,
            multimask_output=True)
        masks = masks[:, None, ...]
    garbage_collect(sam)
    return create_mask_output(image_np, masks, boxes_filt), sam_predict_status + sam_predict_result + (f" However, GroundingDINO installment has failed. Your process automatically fall back to local groundingdino. Check your terminal for more detail and" if (dino_enabled and not install_success) else "")