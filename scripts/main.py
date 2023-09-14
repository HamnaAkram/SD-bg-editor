import re, os
from scripts.edit_object import EditObject
import gradio as gr
from datetime import date

from modules import shared, scripts, script_callbacks
from modules import sd_samplers, processing
from modules.shared import opts
from sambackend.samscripts import sam_predict, sam_dino_predict
from PIL import Image
import numpy as np
import importlib

cn_external_code = None
LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
edit_obj = EditObject()

AlwaysVisible = object()

class PostprocessImageArgs:
    def __init__(self, image):
        self.image = image

def save_image(image):
    base_path   = 'outputs/bg_editor' 
    today       = str(date.today())
    folder_path = os.path.join(base_path, today)
    
    os.makedirs(folder_path, exist_ok=True)
    len_imgs    = len(os.listdir(folder_path))
    file_path   = os.path.join(folder_path, str(len_imgs) + '.png')
    while os.path.exists(file_path):
        file_path = os.path.join(folder_path, str(len_imgs+1) + '.png')
    
    image.save(file_path)

def enhance_face(image, faceprompt):
    # high res face image
    image       = image.resize((int(image.width * 4), int(image.height * 4)), resample=LANCZOS)
    # face mask using dino
    face_mask   = sam_dino_predict(image)
    face_image  = submit_face_paint(faceprompt, image, face_mask[0][2])
    # face inpaint
    return face_image

def submit_face_paint(faceprompt: str, image: Image, mask: Image):
    p = processing.StableDiffusionProcessingImg2Img(
        # prompt='iu1, <lora:iu_v35:1>',
        prompt = faceprompt,
        outpath_samples=opts.outdir_samples or opts.outdir_img2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
        init_images = [image], #: list 
        resize_mode = 0, #: int 
        denoising_strength = 0.55, #: float 
        image_cfg_scale = None, #: float 
        mask = mask, #: Any 
        mask_blur = 4, #: int 
        inpainting_fill = 1, #: int 
        inpaint_full_res = 1, #: bool (OR 1) 
        inpaint_full_res_padding = 32, #: int 
        inpainting_mask_invert = 0, #: int 
        initial_noise_multiplier = 1.0 #: float
    )

    cn_units = [
        #REFERENCE Controlnet
        cn_external_code.ControlNetUnit(
            enabled = False,
            module='reference_only',
            model = 'None',
            weight=0.8,
            image =None,
            resize_mode = 1,
            low_vram = False,
            processor_res = 512,
            threshold_a =0.5,
            threshold_b =200,
            guidance_start=0.0,
            guidance_end=1.0,
            pixel_perfect=True,
            control_mode = "Balanced"
        ),
        
        #Background Controlnet
        cn_external_code.ControlNetUnit(
            enabled = False,
            module='none',
            model = 'control_sd15_canny [fef5e48e]',
            weight=0.8,
            image =None,
            resize_mode = 1,
            low_vram = False,
            processor_res = 512,
            threshold_a =100,
            threshold_b =200,
            guidance_start=0.0,
            guidance_end=1.0,
            pixel_perfect=True,
            control_mode = "Balanced"
        )
        ]
    
        


    p.scripts = scripts.scripts_img2img
    p.script_args = [0, cn_units[1], cn_units[0] ]
    cn_external_code.update_cn_script_in_processing(p, cn_units)

    processed = processing.process_images(p)
    images = processed.images[0]
    return images

def submit_generate(prompt,negative_prompt, faceprompt, faceswap):
    global cn_external_code
    global edit_obj
    if cn_external_code:
        pass
    else:
        cn_external_code = importlib.import_module('extensions.sd-webui-controlnet.scripts.external_code', 'external_code')

    #Text to image parameters
    if True:
        prompt_styles = []
        steps = 20
        sampler_index = 0
        restore_faces = False
        tiling = False
        batch_count = 1
        batch_size = 1
        cfg_scale = 7
        seed = -1
        subseed = -1
        subseed_strength = 0
        seed_resize_from_h = 0
        seed_resize_from_w = 0
        enable_hr = False
        denoising_strength = 0.7
        seed_enable_extras = False
        height = 512
        width = 512
        hr_scale = 2
        hr_upscaler = 'Latent' 
        hr_second_pass_steps = 0
        hr_resize_x = 0
        hr_resize_y = 0
        hr_sampler_index = 0
        hr_prompt = ''
        hr_negative_prompt = ''
        override_settings = {}

        n_iter = 1
    # Setting the input image as reference image automatically
    # edit_obj.user_reference()
    #stable diffusion processor object    
    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model        = shared.sd_model,
        outpath_samples = opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids   = opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt          = prompt,
        styles          = prompt_styles,
        negative_prompt = negative_prompt,
        seed            = seed,
        subseed         = subseed,
        subseed_strength    = subseed_strength,
        seed_resize_from_h  = seed_resize_from_h,
        seed_resize_from_w  = seed_resize_from_w,
        seed_enable_extras  = seed_enable_extras,
        sampler_name        = sd_samplers.samplers[sampler_index].name,
        batch_size      = batch_size,
        n_iter          = n_iter,
        steps           = steps,
        cfg_scale       = cfg_scale,
        width           = edit_obj.w,
        height          = edit_obj.h,
        restore_faces   = restore_faces,
        tiling          = tiling,
        enable_hr       = enable_hr,
        denoising_strength= denoising_strength if enable_hr else None,
        hr_scale        = hr_scale,
        hr_upscaler     = hr_upscaler,
        hr_second_pass_steps    = hr_second_pass_steps,
        hr_resize_x             = hr_resize_x,
        hr_resize_y             = hr_resize_y,
        hr_sampler_name         = sd_samplers.samplers_for_img2img[hr_sampler_index - 1].name if hr_sampler_index != 0 else None,
        hr_prompt               = hr_prompt,
        hr_negative_prompt      = hr_negative_prompt,
        override_settings       = override_settings,
    )

    # mask image for controlnet
    mask  = np.zeros((512,512,4))
    mask[:,:,3] += 255

    subject_img_set = dict()
    subject_img_set['image'] = edit_obj.subject_canny
    subject_img_set['mask'] = mask

    bg_img_set = dict()
    bg_img_set['image'] = edit_obj.bg_canny
    bg_img_set['mask'] = mask

    
    ref_img_set = dict()
    ref_img_set['image'] = ref_img_set['mask'] = mask
    if edit_obj.reference_cn:
        ref_img_set['image'] = edit_obj.reference_img
        ref_img_set['mask'] = mask

    #controlnet units
    cn_units = [
        #REFERENCE Controlnet
        cn_external_code.ControlNetUnit(
            enabled = edit_obj.reference_cn,
            module='reference_only',
            model = 'None',
            weight=0.8,
            image =ref_img_set,
            resize_mode = 1,
            low_vram = False,
            processor_res = 768,
            threshold_a =0.5,
            threshold_b =200,
            guidance_start=0.0,
            guidance_end=1.0,
            pixel_perfect=True,
            control_mode = "Balanced"
        ),
        
        #Background Controlnet
        cn_external_code.ControlNetUnit(
            enabled = True,
            module='none',
            model = 'control_sd15_canny [fef5e48e]',
            weight=0.8,
            image =bg_img_set,
            resize_mode = 1,
            low_vram = False,
            processor_res = 768,
            threshold_a =100,
            threshold_b =200,
            guidance_start=0.0,
            guidance_end=1.0,
            pixel_perfect=True,
            control_mode = "Balanced"
        )
        ]
    
    p.scripts = scripts.scripts_txt2img
    p.script_args = [0, cn_units[1], cn_units[0] ]
        
    cn_external_code.update_cn_script_in_processing(p, cn_units)

    processed = processing.process_images(p)
    # gen_image = np.where(edit_obj.processed_mask > 0, edit_obj.image, np.array())
    gen_image = edit_obj.paste_img(Image.fromarray(edit_obj.image), edit_obj.processed_mask, processed.images[0])

    p.close()
    # gen_image = Image.fromarray(gen_image)
    save_image(gen_image)
    if faceprompt:
        gen_image = enhance_face(gen_image, faceprompt)
        # save_image(gen_image)
    return gen_image

class Script:
    name = None
    """script's internal name derived from title"""

    filename = None
    args_from = None
    args_to = None
    alwayson = False

    is_txt2img = False
    is_img2img = False

    group = None
    """A gr.Group component that has all script's UI inside it"""

    infotext_fields = None
    """if set in ui(), this is a list of pairs of gradio component + text; the text will be used when
    parsing infotext to set the value for the component; see ui.py's txt2img_paste_fields for an example
    """

    paste_field_names = None
    """if set in ui(), this is a list of names of infotext fields; the fields will be sent through the
    various "Send to <X>" buttons when clicked
    """

    api_info = None
    """Generated value of type modules.api.models.ScriptInfo with information about the script for API"""

    def title(self):
        """this function should return the title of the script. This is what will be displayed in the dropdown menu."""
        return 'BG Editor'

    def ui(self, is_img2img):
        """this function should create gradio UI elements. See https://gradio.app/docs/#components
        The return value should be an array of all components that are used in processing.
        Values of those returned components will be passed to run() and process() functions.
        """
        return 

    def show(self, is_img2img):
        """
        is_img2img is True if this function is called for the img2img interface, and Fasle otherwise

        This function should return:
         - False if the script should not be shown in UI at all
         - True if the script should be shown in UI if it's selected in the scripts dropdown
         - script.AlwaysVisible if the script should be shown in UI at all times
         """

        return True

    def run(self, p, *args):
        """
        This function is called if the script has been selected in the script dropdown.
        It must do all processing and return the Processed object with results, same as
        one returned by processing.process_images.

        Usually the processing is done by calling the processing.process_images function.

        args contains all values returned by components from ui()
        """

        pass

    def process(self, p, *args):
        """
        This function is called before processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """

        pass

    def before_process_batch(self, p, *args, **kwargs):
        """
        Called before extra networks are parsed from the prompt, so you can add
        new extra network keywords to the prompt with this callback.

        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
        """

        pass

    def process_batch(self, p, *args, **kwargs):
        """
        Same as process(), but called for every batch.

        **kwargs will have those items:
          - batch_number - index of current batch, from 0 to number of batches-1
          - prompts - list of prompts for current batch; you can change contents of this list but changing the number of entries will likely break things
          - seeds - list of seeds for current batch
          - subseeds - list of subseeds for current batch
        """

        pass

    def postprocess_batch(self, p, *args, **kwargs):
        """
        Same as process_batch(), but called for every batch after it has been generated.

        **kwargs will have same items as process_batch, and also:
          - batch_number - index of current batch, from 0 to number of batches-1
          - images - torch tensor with all generated images, with values ranging from 0 to 1;
        """

        pass

    def postprocess_image(self, p, pp: PostprocessImageArgs, *args):
        """
        Called for every image after it has been generated.
        """

        pass

    def postprocess(self, p, processed, *args):
        """
        This function is called after processing ends for AlwaysVisible scripts.
        args contains all values returned by components from ui()
        """

        pass

    def before_component(self, component, **kwargs):
        """
        Called before a component is created.
        Use elem_id/label fields of kwargs to figure out which component it is.
        This can be useful to inject your own components somewhere in the middle of vanilla UI.
        You can return created components in the ui() function to add them to the list of arguments for your processing functions
        """

        pass

    def after_component(self, component, **kwargs):
        """
        Called after a component is created. Same as above.
        """

        pass

    def describe(self):
        """unused"""
        return ""

    def elem_id(self, item_id):
        """helper function to generate id for a HTML element, constructs final id out of script name, tab and user-supplied item_id"""

        need_tabname = self.show(True) == self.show(False)
        tabkind = 'img2img' if self.is_img2img else 'txt2txt'
        tabname = f"{tabkind}_" if need_tabname else ""
        title = re.sub(r'[^a-z_0-9]', '', re.sub(r'\s', '_', self.title().lower()))

        return f'script_{tabname}{title}_{item_id}'

def on_ui_tabs():
    tab_prefix = 'my_tab'
    global edit_obj

    with gr.Blocks(analytics_enabled=False) as my_tab:
        with gr.Row():
            with gr.Column():
                gr.HTML(value="<p>Left click the image to add one positive point (black dot). Right click the image to add one negative point (red dot). Left click the point to remove it.</p>")
                sam_input_image = gr.Image(label="Image for Segment Anything", elem_id=f"{tab_prefix}input_image", source="upload", type="pil", image_mode="RGB").style(height=242)
                
                def reupload(input_img):
                    resize_img = edit_obj.set_img(input_img)
                    return resize_img
                
                sam_input_image.upload(fn = reupload,
                                       inputs = sam_input_image,
                                       outputs = [sam_input_image])

            with gr.Column():
                gr.HTML(value="<p>  </p>")
                gr.HTML(value="<p>  </p>")
                gr.HTML(value="<p>  </p>")
                sam_remove_dots = gr.Button(value="Remove all point prompts")
                sam_dummy_component = gr.Label(visible=False)
                sam_remove_dots.click(
                            fn=lambda _: None,
                            _js="samRemoveDots",
                            inputs=[sam_dummy_component],
                            outputs=None)
                sam_genmask =  gr.Button(value="Preview Segmentation", elem_id=f"{tab_prefix}run_button")
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    sam_output_mask_gallery = gr.Gallery(label='Segment Anything Output').style(grid=3).style(height=242)
                    
                    # CODEINFO SAM SEGMENTATION
                    # sam_model_name = gr.Dropdown(label="SAM Model", choices=sam_model_list, value=sam_model_list[0] if len(sam_model_list) > 0 else None, visible=False)
                    sam_genmask.click(
                        fn=sam_predict,
                        _js='submit_sam',
                        inputs=[sam_dummy_component, sam_input_image,        # SAM
                                sam_dummy_component, sam_dummy_component],   # Point prompts
                                # dino_checkbox, dino_model_name, dino_text_prompt, dino_box_threshold,  # DINO prompts
                                # dino_preview_checkbox, dino_preview_boxes_selection],  # DINO preview prompts
                        outputs=[sam_output_mask_gallery])
                    with gr.Column():
                        sam_selected_mask = gr.Radio(choices=['1', '2', '3'],label='Select mask')
                        sam_selected_mask_view = gr.Gallery(label='Selected mask', elem_id=f'{tab_prefix}selected_mask', interactive=False).style(grid=3).style(height=242)
                
                def fn_sam_select_mask(evt: gr.SelectData, mask_gallery, input_image):

                    img_for_output = edit_obj.process_mask(evt, mask_gallery, input_image)
                    
                    return img_for_output
                
                sam_selected_mask.select(
                    fn = fn_sam_select_mask,
                    inputs=[sam_output_mask_gallery, sam_input_image],
                    outputs=[sam_selected_mask_view]
                )

        with gr.Row():
            with gr.Column():
                gr.HTML(value="Upload Background Image")
                with gr.Row():        
                    bg_input_img = gr.Image(label="Background Image for Generation", elem_id=f"{tab_prefix}background_image", source="upload", type="pil", image_mode="RGB").style(height=242)
                    bg_canny_img = gr.Image(label='Background Canny Image', elem_id=f"{tab_prefix}background_canny", visible=False).style(height=242)
                    def bg_uploaded(bg_input_img):
                        
                        bgcanny = edit_obj.process_bg(bg_input_img)
                        return gr.update(value=bgcanny, visible=True, interactive=False)
                    
                    bg_input_img.upload(fn = bg_uploaded,
                                        inputs=[bg_input_img],
                                        outputs=[bg_canny_img])

        with gr.Row():
            with gr.Column():
                subject_low     = gr.Slider(minimum=1, maximum=255, label= 'Subject Low Threshold'  , value = 100)
                subject_high    = gr.Slider(minimum=1, maximum=255, label= 'Subject High Threshold' , value = 200)
                back_low        = gr.Slider(minimum=1, maximum=255, label= 'Background Low Threshold'   , value = 100)
                back_high       = gr.Slider(minimum=1, maximum=255, label= 'Background High Threshold'  , value = 200)
                
                def subject_slider_update(subject_low, subject_high, sam_selected_mask_view):
                    subject_canny = edit_obj.update_subject_canny(subject_low, subject_high)

                    return subject_canny
                
                def bg_slider_update(bg_low, bg_high):
                    bg_canny = edit_obj.update_bg_canny(bg_low, bg_high)    
                    return gr.update(value=bg_canny, visible=True)

                subject_low.change(
                    fn = subject_slider_update,
                    inputs = [subject_low, subject_high, sam_selected_mask_view],
                    outputs= [sam_selected_mask_view])                
                subject_high.change(
                    fn = subject_slider_update,
                    inputs = [subject_low, subject_high, sam_selected_mask_view],
                    outputs= [sam_selected_mask_view])
                
                back_low.change(
                    fn = bg_slider_update,
                    inputs = [back_low, back_high],
                    outputs= [bg_canny_img])
                back_high.change(
                    fn = bg_slider_update,
                    inputs = [back_low, back_high],
                    outputs= [bg_canny_img])

            with gr.Column():
                gr.HTML(value="Enter Prompts")
                bg_positive_prompt = gr.Textbox(label='Positive Prompt', elem_id="bg_positive_prompts")
                bg_negative_prompt = gr.Textbox(label='Negative Prompt', elem_id="bg_negative_prompts")
            
        with gr.Row():
            ## UI and backend for reference only
            # with gr.Column():
            #     ref_img = gr.Image(label='Reference Image', elem_id=f"{tab_prefix} Reference Image", visible=True, interactive= True, image_mode= 'RGB' ).style(height=242)
            #     user_reference = gr.Checkbox(value= False, label= 'Use Reference Image')
                
            #     def use_ref_cn(status, img):
            #         edit_obj.user_reference(status, img)

            #     user_reference.change(fn = use_ref_cn ,
            #                           inputs = [user_reference, ref_img],
            #                           outputs=[]
            #                           )
            with gr.Column():
                reference_button = gr.Checkbox(value= False, label= 'Use Reference Image')
                faceswap_button  = gr.Checkbox(value= False, label= 'Change and Enhance Face Image')
                face_prompt      = gr.Textbox(label='Face Prompt', elem_id="face_prompts")

                reference_button.change(fn = edit_obj.user_reference ,
                                      inputs = [reference_button],
                                      outputs=[]
                                      )

            with gr.Column():
                bg_generate = gr.Button(value="Generate Background", elem_id=f"{tab_prefix}_bg_generate_button")
                generated_img = gr.Image(label='Generated Image', elem_id=f"{tab_prefix} Generated Image", visible=True).style(height=242)
                bg_generate.click(
                    fn = submit_generate,
                    inputs= [bg_positive_prompt,bg_negative_prompt, face_prompt, faceswap_button],
                    outputs= [generated_img]
                    )
    return [(my_tab, 'BG Editor', 'my_tab')]

script_callbacks.on_ui_tabs(on_ui_tabs)
