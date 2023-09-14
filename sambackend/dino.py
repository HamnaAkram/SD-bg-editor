from modules import scripts, shared
import torch, gc, os
from collections import OrderedDict
from modules.devices import device, torch_gc, cpu


sam_extension_dir = scripts.basedir()
dino_model_dir = os.path.join(sam_extension_dir, "models/grounding-dino")

dino_model_info = {
    "GroundingDINO_SwinT_OGC (694MB)": {
        "checkpoint": "groundingdino_swint_ogc.pth",
        "config": os.path.join(dino_model_dir, "GroundingDINO_SwinT_OGC.py"),
        "url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    },
    "GroundingDINO_SwinB (938MB)": {
        "checkpoint": "groundingdino_swinb_cogcoor.pth",
        "config": os.path.join(dino_model_dir, "GroundingDINO_SwinB.cfg.py"),
        "url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth"
    },
}

dino_model_cache = OrderedDict()

def install_goundingdino():
    if shared.opts.data.get("sam_use_local_groundingdino", False):
        print("Using local groundingdino.")
        return False

    def verify_dll(install_local=True):
        try:
            from groundingdino import _C
            print("GroundingDINO dynamic library have been successfully built.")
            return True
        except Exception:
            import traceback
            traceback.print_exc()
            def run_pip_uninstall(command, desc=None):
                from launch import python, run
                default_command_live = (os.environ.get('WEBUI_LAUNCH_LIVE_OUTPUT') == "1")
                return run(f'"{python}" -m pip uninstall -y {command}', desc=f"Uninstalling {desc}", errdesc=f"Couldn't uninstall {desc}", live=default_command_live)
            if install_local:
                print(f"Failed to build dymanic library. Will uninstall GroundingDINO from pip and fall back to local groundingdino this time. Please ")
                run_pip_uninstall(
                    f"groundingdino",
                    f"sd-webui-segment-anything requirement: groundingdino")
            else:
                print(f"Failed to build dymanic library. Will uninstall GroundingDINO from pip and re-try installing from GitHub source code. Please ")
                run_pip_uninstall(
                    f"uninstall groundingdino",
                    f"sd-webui-segment-anything requirement: groundingdino")
            return False

    import launch
    if launch.is_installed("groundingdino"):
        print("Found GroundingDINO in pip. Verifying if dynamic library build success.")
        if verify_dll(install_local=False):
            return True
    try:
        launch.run_pip(
            f"install git+https://github.com/IDEA-Research/GroundingDINO",
            f"sd-webui-segment-anything requirement: groundingdino")
        print("GroundingDINO install success. Verifying if dynamic library build success.")
        return verify_dll()
    except Exception:
        import traceback
        traceback.print_exc()
        print(f"GroundingDINO install failed. Will fall back to local groundingdino this time. Please ")
        return False

def clear_dino_cache():
    dino_model_cache.clear()
    gc.collect()
    torch_gc()

def load_dino_model(dino_checkpoint, dino_install_success):
    print(f"Initializing GroundingDINO {dino_checkpoint}")
    if dino_checkpoint in dino_model_cache:
        dino = dino_model_cache[dino_checkpoint]
        if shared.cmd_opts.lowvram:
            dino.to(device=device)
    else:
        clear_dino_cache()
        if dino_install_success:
            from groundingdino.models import build_model
            from groundingdino.util.slconfig import SLConfig
            from groundingdino.util.utils import clean_state_dict
        else:
            from local_groundingdino.models import build_model
            from local_groundingdino.util.slconfig import SLConfig
            from local_groundingdino.util.utils import clean_state_dict
        args = SLConfig.fromfile(dino_model_info[dino_checkpoint]["config"])
        dino = build_model(args)
        checkpoint = torch.hub.load_state_dict_from_url(
            dino_model_info[dino_checkpoint]["url"], dino_model_dir)
        dino.load_state_dict(clean_state_dict(
            checkpoint['model']), strict=False)
        dino.to(device=device)
        dino_model_cache[dino_checkpoint] = dino
    dino.eval()
    return dino

def load_dino_image(image_pil, dino_install_success):
    if dino_install_success:
        import groundingdino.datasets.transforms as T
    else:
        from local_groundingdino.datasets import transforms as T
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image

def get_grounding_output(model, image, caption, box_threshold):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    if shared.cmd_opts.lowvram:
        model.to(cpu)
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    return boxes_filt.cpu()

def dino_predict_internal(input_image, dino_model_name, text_prompt, box_threshold):
    install_success = install_goundingdino()
    print("Running GroundingDINO Inference")
    dino_image = load_dino_image(input_image.convert("RGB"), install_success)
    dino_model = load_dino_model(dino_model_name, install_success)
    install_success = install_success or shared.opts.data.get("sam_use_local_groundingdino", False)

    boxes_filt = get_grounding_output(
        dino_model, dino_image, text_prompt, box_threshold
    )

    H, W = input_image.size[1], input_image.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    gc.collect()
    torch_gc()
    return boxes_filt, install_success