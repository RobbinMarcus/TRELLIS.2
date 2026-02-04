from typing import *
from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image


class BiRefNet:
    def __init__(self, model_name: str = "ZhengPeng7/BiRefNet"):
        # Monkey-patch to fix transformers compatibility issue
        from transformers import modeling_utils
        
        # Patch 1: mark_tied_weights_as_initialized
        original_mark = getattr(modeling_utils.PreTrainedModel, 'mark_tied_weights_as_initialized', None)

        def patched_mark(self):
            if not hasattr(self, 'all_tied_weights_keys'):
                self.all_tied_weights_keys = {}
            if original_mark:
                return original_mark(self)

        modeling_utils.PreTrainedModel.mark_tied_weights_as_initialized = patched_mark

        # Patch 2: ContextManagers to avoid meta device initialization
        # BiRefNet's remote code is incompatible with meta tensors (calls .item() on them).
        # We replace ContextManagers to prevent transformers from enforcing any device context during init.
        original_ContextManagers = modeling_utils.ContextManagers
        
        class NoOpContextManagers:
            def __init__(self, contexts):
                pass
            def __enter__(self):
                return []
            def __exit__(self, *args):
                pass
        
        modeling_utils.ContextManagers = NoOpContextManagers

        try:
            self.model = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet", trust_remote_code=True
            )
        finally:
            # Restore original methods
            if original_mark:
                modeling_utils.PreTrainedModel.mark_tied_weights_as_initialized = original_mark
            modeling_utils.ContextManagers = original_ContextManagers

        self.model.eval()
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    
    def to(self, device: str):
        self.model.to(device)

    def cuda(self):
        self.model.cuda()

    def cpu(self):
        self.model.cpu()
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        input_images = self.transform_image(image).unsqueeze(0).to("cuda")

        # Match input dtype to model dtype
        model_dtype = next(self.model.parameters()).dtype
        input_images = input_images.to(dtype=model_dtype)

        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        image.putalpha(mask)
        return image
    