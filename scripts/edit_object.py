import numpy as np
from PIL import Image, ImageFilter, ImageOps
import cv2
import gradio as gr


def get_yellow_img(shape):
    h, w, _ = shape
    yellow_img = np.ones((h,w,3),dtype=np.uint8)
    yellow_img[:,:,0] = 255
    yellow_img[:,:,1] = 255
    yellow_img[:,:,2] = 0

    return yellow_img

### CANNY PROCESSOR:
def apply_canny(img, mask, low_threshold, high_threshold):
    if mask is not None:
        yellow_img  = get_yellow_img(img.shape)
        masked_img  = np.where(mask>0, img, yellow_img)
        return cv2.Canny(masked_img, low_threshold, high_threshold)
    else:
        return cv2.Canny(img, low_threshold, high_threshold)


class EditObject():
    def __init__(self) -> None:
        self.image          = None
        self.mask           = None
        self.processed_mask = None
        self.diluted_mask   = None
        self.canny          = None
        self.masked_canny   = None
        self.pose           = None
        self.bg_img         = None
        self.bg_depth       = None
        self.bg_canny       = None
        self.bg_cany_mask   = None
        self.reference_cn   = False
        self.reference_img  = None
    
    def reset(self):
        self.image          = None
        self.mask           = None
        self.masked_img     = None
        self.processed_mask = None
        self.diluted_mask   = None
        self.subject_canny  = None
        self.masked_canny   = None
        self.pose           = None
        self.bg_img         = None
        self.bg_depth       = None
        self.bg_canny       = None
        self.bg_cany_mask   = None

    def set_512(self,image):
        desired_size = 512
        orig_size = image.size
        
        ratio = float(desired_size)/max(orig_size)
        new_size = tuple([int(x*ratio) for x in orig_size])
        # use thumbnail() or resize() method to resize the input image

        # thumbnail is a in-place operation

        # im.thumbnail(new_size, Image.ANTIALIAS)

        image = image.resize(new_size, Image.ANTIALIAS)
        # create a new image and paste the resized on it

        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(image, ((desired_size-new_size[0])//2,
                            (desired_size-new_size[1])//2))
        return new_im, orig_size, ratio
    
    def set_person_size(self, image, h = 1024):
        desired_size    = h
        w = 768
        orig_size       = image.size
        ratio = float(desired_size)/max(orig_size)
        new_size = tuple([int(x*ratio) for x in orig_size])
        # use thumbnail() or resize() method to resize the input image

        # thumbnail is a in-place operation

        # im.thumbnail(new_size, Image.ANTIALIAS)

        image = image.resize(new_size, Image.ANTIALIAS)
        # create a new image and paste the resized on it

        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(image, ((desired_size-new_size[0])//2,
                            (desired_size-new_size[1])//2))
        
        left    = (desired_size - w)/2
        top     = (desired_size - h)/2
        right   = (desired_size + w)/2
        bottom  = (desired_size + h)/2

        crop_img = new_im.crop((left, top, right, bottom))
        
        return crop_img

        
    def set_bg_size(self, bg_image, h, w):
        desired_size    = h
        orig_size       = bg_image.size
        ratio = float(desired_size)/max(orig_size)
        new_size = tuple([int(x*ratio) for x in orig_size])
        # use thumbnail() or resize() method to resize the input image

        # thumbnail is a in-place operation

        # im.thumbnail(new_size, Image.ANTIALIAS)

        bg_image = bg_image.resize(new_size, Image.ANTIALIAS)
        # create a new image and paste the resized on it

        new_im = Image.new("RGB", (desired_size, desired_size))
        new_im.paste(bg_image, ((desired_size-new_size[0])//2,
                            (desired_size-new_size[1])//2))
        
        left    = (desired_size - w)/2
        top     = (desired_size - h)/2
        right   = (desired_size + w)/2
        bottom  = (desired_size + h)/2

        crop_bg = new_im.crop((left, top, right, bottom)) 

        return crop_bg 

    def set_img(self, image:Image):
        self.reset()
        
        self.image = self.set_person_size(image)
        self.image = np.array(self.image)
        self.h, self.w, _ = self.image.shape
        return self.image
    
    def process_mask(self, evt: gr.SelectData, mask_gallery, input_image):
        self.mask = np.array(Image.open(mask_gallery[evt.index]['name'])).astype(np.uint8)

        dilate_kernel       = np.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        erode_kernel        = np.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        # closing
        self.processed_mask = cv2.dilate(self.mask,             dilate_kernel, iterations = 1)
        self.processed_mask = cv2.erode(self.processed_mask,    erode_kernel,  iterations = 1)
        # Expand
        self.processed_mask = cv2.dilate(self.processed_mask,   dilate_kernel,  iterations = 1)
        # Smooth
        self.processed_mask = Image.fromarray(self.processed_mask).filter(ImageFilter.ModeFilter(size=9))
        # self.processed_mask = self.mask
        
        self.processed_mask = np.array(self.processed_mask)
        self.processed_mask = np.expand_dims(self.processed_mask, axis=2)

        self.diluted_mask   = cv2.erode(self.processed_mask, erode_kernel,  iterations = 5)
        self.diluted_mask   = np.expand_dims(self.diluted_mask, axis=2)
        yellow_img          = get_yellow_img(self.image.shape)
        self.masked_img     = np.where(self.processed_mask > 0, self.image, yellow_img)

        self.subject_canny  = apply_canny(self.image, self.diluted_mask, low_threshold=128, high_threshold=255)

        return [Image.fromarray(self.image), Image.fromarray(self.masked_img), Image.fromarray(self.subject_canny)]

    def process_bg(self, bg_img):

        self.bg_img     = self.set_bg_size(bg_img, self.h, self.w)
        self.bg_img     = np.array(self.bg_img)
        self.bg_canny   = apply_canny(self.bg_img, None, low_threshold=128, high_threshold=255)
        self.bg_canny   = np.where(self.diluted_mask[:,:,0] > 0, 0, self.bg_canny)

        self.bg_canny   = np.where(self.subject_canny > 0, self.subject_canny, self.bg_canny)
        return Image.fromarray(self.bg_canny)
    
    def update_subject_canny(self, low_threshold, high_threshold):
        self.subject_canny  = apply_canny(self.image, self.diluted_mask, low_threshold, high_threshold)
        
        return [Image.fromarray(self.image), Image.fromarray(self.masked_img), Image.fromarray(self.subject_canny)]

    def update_bg_canny(self, subject_low, subject_high):
        self.bg_canny = apply_canny(self.bg_img, None, subject_low, subject_high)
        self.bg_canny = np.where(self.diluted_mask[:,:,0] > 0, 0, self.bg_canny)

        self.bg_canny = np.where(self.subject_canny > 0, self.subject_canny, self.bg_canny)

        return Image.fromarray(self.bg_canny)
    
    def user_reference(self, status = False):
        if status:
            self.reference_cn   = True
            self.reference_img  = np.array(Image.fromarray(self.image).resize((512,512)))
        else:
            self.reference_cn   = status
            self.reference_img  = None
    
    def paste_img(self, image, image_mask, gen_im):
        # resize face mask
        # image_mask          = np.array(image_mask).astype(np.uint8)
        # image_mask          = np.expand_dims(image_mask,axis=2)
        dilate_kernel       = np.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        erode_kernel        = np.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        image_mask          = cv2.dilate(image_mask,             dilate_kernel, iterations = 1)
        image_mask          = cv2.erode(image_mask,    erode_kernel,  iterations = 6)
        image_mask          = image_mask.astype(np.bool8)
        image_mask          = Image.fromarray(image_mask)

        #Smooth face mask
        image_mask          = image_mask.convert('L')
        image_mask          = image_mask.filter(ImageFilter.GaussianBlur(3))
        np_mask             = np.array(image_mask)
        np_mask             = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
        mask_for_overlay    = Image.fromarray(np_mask)

        # Pasteback image
        image_masked        = Image.new('RGBa', (image.width, image.height))
        image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=mask_for_overlay.convert('L'))
        image_masked        = image_masked.convert('RGBA')
        gen_im              = gen_im.convert('RGBA').resize((image.width, image.height))
        gen_im.alpha_composite(image_masked)
        gen_im = gen_im.convert('RGB')

        return gen_im