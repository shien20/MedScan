import torch
import torch.nn as nn
from torchvision import models


class HeavyEnsemble(nn.Module):
    """
    ResNet50 + DenseNet121
    Fused embedding: 2048 + 1024 = 3072-dimensional
    MLP: 3072 -> 1024 -> 512 -> 4
    """

    def __init__(self, resnet_path=None, densenet_path=None, num_classes=4):
        super(HeavyEnsemble, self).__init__()

        # ── ResNet50 backbone ──────────────────────────────────
        resnet = models.resnet50(weights="IMAGENET1K_V1")

        # Load your trained weights if provided
        if resnet_path:
            state = torch.load(resnet_path, map_location="cpu")
            resnet.load_state_dict(state, strict=False)
            print(f"ResNet50 weights loaded from {resnet_path}")

        # Remove the original classifier head
        # ResNet50 outputs 2048-dim after GAP
        self.resnet_backbone = nn.Sequential(*list(resnet.children())[:-1])

        # ── DenseNet121 backbone ───────────────────────────────
        densenet = models.densenet121(weights="IMAGENET1K_V1")

        if densenet_path:
            state = torch.load(densenet_path, map_location="cpu")
            densenet.load_state_dict(state, strict=False)
            print(f"DenseNet121 weights loaded from {densenet_path}")

        # Remove the original classifier
        # DenseNet121 outputs 1024-dim after GAP
        self.densenet_backbone = nn.Sequential(*list(densenet.children())[:-1])
        self.densenet_gap = nn.AdaptiveAvgPool2d((1, 1))

        # ── Fusion classifier ──────────────────────────────────
        # 2048 + 1024 = 3072
        self.classifier = nn.Sequential(
            nn.Linear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # ResNet50: outputs (batch, 2048, 1, 1) → flatten to (batch, 2048)
        resnet_feat = self.resnet_backbone(x)
        resnet_feat = resnet_feat.view(resnet_feat.size(0), -1)

        # DenseNet121: features block outputs (batch, 1024, 7, 7)
        # Need an extra GAP to get (batch, 1024)
        densenet_feat = self.densenet_backbone(x)
        densenet_feat = self.densenet_gap(densenet_feat)
        densenet_feat = densenet_feat.view(densenet_feat.size(0), -1)

        # Concatenate: (batch, 3072)
        fused = torch.cat([resnet_feat, densenet_feat], dim=1)

        # Classify
        out = self.classifier(fused)
        return out

    def freeze_backbones(self):
        """Freeze both backbones — only train the classifier head."""
        for param in self.resnet_backbone.parameters():
            param.requires_grad = False
        for param in self.densenet_backbone.parameters():
            param.requires_grad = False
        print("Backbones frozen.")

    def unfreeze_top_layers(self):
        """Unfreeze top layers of each backbone for fine-tuning."""

        # ResNet50: unfreeze layer3 and layer4
        # children()[:-1] gives: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool
        # indices:                  0     1    2      3       4       5       6       7       8
        for i, child in enumerate(self.resnet_backbone.children()):
            if i >= 6:   # layer3 onwards
                for param in child.parameters():
                    param.requires_grad = True

        # DenseNet121: unfreeze denseblock3, denseblock4, norm5
        # DenseNet children: conv0, norm0, relu0, pool0, denseblock1,
        #                    transition1, denseblock2, transition2,
        #                    denseblock3, transition3, denseblock4, norm5
        for i, child in enumerate(self.densenet_backbone.children()):
            if i >= 8:   # denseblock3 onwards
                for param in child.parameters():
                    param.requires_grad = True

        print("Top layers unfrozen for fine-tuning.")


class LightEnsemble(nn.Module):
    """
    EfficientNet-B0 + MobileNetV2
    Fused embedding: 1280 + 1280 = 2560-dimensional
    MLP: 2560 -> 512 -> 256 -> 4
    """

    def __init__(self, efficientnet_path=None, mobilenet_path=None, num_classes=4):
        super(LightEnsemble, self).__init__()

        # ── EfficientNet-B0 backbone ───────────────────────────
        efficientnet = models.efficientnet_b0(weights="IMAGENET1K_V1")

        if efficientnet_path:
            state = torch.load(efficientnet_path, map_location="cpu")
            efficientnet.load_state_dict(state, strict=False)
            print(f"EfficientNet-B0 weights loaded from {efficientnet_path}")

        # EfficientNet: features → avgpool → flatten
        # We take only the features block + avgpool, output: (batch, 1280)
        self.efficientnet_features = efficientnet.features
        self.efficientnet_gap = nn.AdaptiveAvgPool2d((1, 1))

        # ── MobileNetV2 backbone ───────────────────────────────
        mobilenet = models.mobilenet_v2(weights="IMAGENET1K_V1")

        if mobilenet_path:
            state = torch.load(mobilenet_path, map_location="cpu")
            mobilenet.load_state_dict(state, strict=False)
            print(f"MobileNetV2 weights loaded from {mobilenet_path}")

        # MobileNetV2: features → avgpool, output: (batch, 1280)
        self.mobilenet_features = mobilenet.features
        self.mobilenet_gap = nn.AdaptiveAvgPool2d((1, 1))

        # ── Fusion classifier ──────────────────────────────────
        # 1280 + 1280 = 2560
        self.classifier = nn.Sequential(
            nn.Linear(2560, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # EfficientNet: (batch, 1280, 7, 7) → GAP → (batch, 1280)
        eff_feat = self.efficientnet_features(x)
        eff_feat = self.efficientnet_gap(eff_feat)
        eff_feat = eff_feat.view(eff_feat.size(0), -1)

        # MobileNetV2: (batch, 1280, 7, 7) → GAP → (batch, 1280)
        mob_feat = self.mobilenet_features(x)
        mob_feat = self.mobilenet_gap(mob_feat)
        mob_feat = mob_feat.view(mob_feat.size(0), -1)

        # Concatenate: (batch, 2560)
        fused = torch.cat([eff_feat, mob_feat], dim=1)

        out = self.classifier(fused)
        return out

    def freeze_backbones(self):
        for param in self.efficientnet_features.parameters():
            param.requires_grad = False
        for param in self.mobilenet_features.parameters():
            param.requires_grad = False
        print("Backbones frozen.")

    def unfreeze_top_layers(self):
        # EfficientNet-B0 features has 9 blocks (index 0-8)
        # Unfreeze last 3 blocks (index 6, 7, 8)
        for i, layer in enumerate(self.efficientnet_features):
            if i >= 6:
                for param in layer.parameters():
                    param.requires_grad = True

        # MobileNetV2 features has 19 layers (index 0-18)
        # Unfreeze last 5 layers
        for i, layer in enumerate(self.mobilenet_features):
            if i >= 14:
                for param in layer.parameters():
                    param.requires_grad = True

        print("Top layers unfrozen for fine-tuning.")