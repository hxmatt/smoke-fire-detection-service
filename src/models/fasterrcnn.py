"""Faster R-CNN architecture for smoke/fire detection.

Shared by the fine-tuning notebook, the summary notebook, and the service:
all three must build the model the same way so a saved ``state_dict`` loads
back into the identical architecture. Keep the class count and head
construction here, not duplicated at the call sites.
"""

from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

LABEL_OFFSET = 1  # class 0 is background; raw ids 0/1 shift to 1/2
NUM_CLASSES = 3  # smoke, fire, background


def build_model(pretrained: bool = True, freeze_backbone: bool = True):
    """Faster R-CNN (MobileNetV3-L FPN) with a 3-class head.

    Args:
        pretrained: load COCO-pretrained weights (True for fine-tuning;
            False when about to load a fine-tuned ``state_dict``).
        freeze_backbone: freeze the backbone body so only the FPN and heads
            train.
    """
    weights = "DEFAULT" if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    if freeze_backbone:
        for p in model.backbone.body.parameters():
            p.requires_grad = False

    return model
