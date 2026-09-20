import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LABELS
# ============================================================

MATERIAL_NAMES = [
    "Brick",
    "Concrete",
    "Timber"
]

DAMAGE_NAMES = [
    "Low",
    "Moderate",
    "Severe"
]


# ============================================================
# MODEL
# ============================================================

class MultiTaskResNet(nn.Module):

    def __init__(self):

        super().__init__()

        # Same ResNet50 architecture as the trained model.
        #
        # weights=None is intentional here.
        # The checkpoint will provide the trained weights.
        backbone = models.resnet50(
            weights=None
        )

        self.backbone = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.material_head = nn.Sequential(

            nn.Linear(
                2048,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.4
            ),

            nn.Linear(
                256,
                3
            )
        )

        self.damage_head = nn.Sequential(

            nn.Linear(
                2048,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.4
            ),

            nn.Linear(
                256,
                3
            )
        )


    def forward(self, x):

        feat = self.backbone(x)

        feat = torch.flatten(
            feat,
            1
        )

        material_logits = self.material_head(
            feat
        )

        damage_logits = self.damage_head(
            feat
        )

        return (
            material_logits,
            damage_logits
        )


# ============================================================
# IMAGE TRANSFORM
# ============================================================

eval_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

# def load_model(checkpoint_path):

#     model = MultiTaskResNet()

#     checkpoint = torch.load(
#         checkpoint_path,
#         map_location=DEVICE
#     )

#     model.load_state_dict(
#         checkpoint
#     )

#     model = model.to(
#         DEVICE
#     )

#     model.eval()

#     return model

def load_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    # Create the model structure without allocating all parameter
    # tensors in RAM first.
    with torch.device("meta"):
        model = MultiTaskResNet()

    # Memory-map the checkpoint instead of loading the entire
    # checkpoint into RAM at once.
    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True
    )

    # Attach the checkpoint tensors directly to the model.
    model.load_state_dict(
        state,
        assign=True,
        strict=True
    )

    model = model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# PREDICT ONE IMAGE
# ============================================================

def predict_image(
    model,
    image_path
):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = eval_transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(
        0
    )

    image_tensor = image_tensor.to(
        DEVICE
    )


    with torch.no_grad():

        material_logits, damage_logits = model(
            image_tensor
        )


    # --------------------------------------------------------
    # Material
    # --------------------------------------------------------

    material_probabilities = torch.softmax(
        material_logits,
        dim=1
    )

    material_index = torch.argmax(
        material_probabilities,
        dim=1
    ).item()

    material_confidence = (
        material_probabilities[0, material_index]
        .item()
        * 100
    )


    # --------------------------------------------------------
    # Damage
    # --------------------------------------------------------

    damage_probabilities = torch.softmax(
        damage_logits,
        dim=1
    )

    damage_index = torch.argmax(
        damage_probabilities,
        dim=1
    ).item()

    damage_confidence = (
        damage_probabilities[0, damage_index]
        .item()
        * 100
    )


    return {

        "material":
            MATERIAL_NAMES[material_index],

        "material_confidence":
            round(
                material_confidence,
                2
            ),

        "damage":
            DAMAGE_NAMES[damage_index],

        "damage_confidence":
            round(
                damage_confidence,
                2
            )
    }