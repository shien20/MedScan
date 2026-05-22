import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class HeavyEnsemble(nn.Module):
    """
    Heavy Ensemble: ResNet50 (fine-tuned) + DenseNet121 (baseline)
    Fused embedding: 2048 + 1024 = 3072-dimensional
    MLP classifier: 3072 -> 1024 -> 512 -> 4
    """

    def __init__(self, resnet_path=None, densenet_path=None, num_classes=4):
        super(HeavyEnsemble, self).__init__()

        # ── ResNet50 backbone ──────────────────────────────────────────
        resnet = models.resnet50(weights="IMAGENET1K_V1")

        if resnet_path:
            state = torch.load(resnet_path, map_location="cpu")
            resnet.load_state_dict(state, strict=False)
            # strict=False because saved .pth includes the old fc head
            # which no longer matches — backbone weights load correctly
            print(f"ResNet50 weights loaded from: {resnet_path}")

        # Remove classification head — keep everything up to avgpool
        # ResNet50 children: conv1, bn1, relu, maxpool,
        #                    layer1, layer2, layer3, layer4, avgpool, fc
        # We take [:-1] which removes fc, keeping avgpool
        # Output shape after avgpool: (batch, 2048, 1, 1)
        self.resnet_backbone = nn.Sequential(*list(resnet.children())[:-1])

        # ── DenseNet121 backbone ───────────────────────────────────────
        densenet = models.densenet121(weights="IMAGENET1K_V1")

        if densenet_path:
            state = torch.load(densenet_path, map_location="cpu")
            densenet.load_state_dict(state, strict=False)
            print(f"DenseNet121 weights loaded from: {densenet_path}")

        # DenseNet121: features block outputs (batch, 1024, 7, 7)
        # We take only the features block, then apply GAP manually
        self.densenet_backbone = densenet.features
        self.densenet_gap = nn.AdaptiveAvgPool2d((1, 1))

        # ── Fusion classifier ──────────────────────────────────────────
        # Input: 2048 + 1024 = 3072
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
        # ResNet50: (batch, 3, 224, 224) → (batch, 2048, 1, 1) → (batch, 2048)
        resnet_feat = self.resnet_backbone(x)
        resnet_feat = resnet_feat.view(resnet_feat.size(0), -1)

        # DenseNet121: (batch, 3, 224, 224) → (batch, 1024, 7, 7) → (batch, 1024)
        densenet_feat = self.densenet_backbone(x)
        densenet_feat = F.relu(densenet_feat, inplace=True)
        densenet_feat = self.densenet_gap(densenet_feat)
        densenet_feat = densenet_feat.view(densenet_feat.size(0), -1)

        # Feature normalisation — equalises scale between backbones
        # so neither backbone dominates the fused representation
        resnet_feat   = F.normalize(resnet_feat,   p=2, dim=1)
        densenet_feat = F.normalize(densenet_feat, p=2, dim=1)

        # Concatenate: (batch, 3072)
        fused = torch.cat([resnet_feat, densenet_feat], dim=1)

        return self.classifier(fused)

    def freeze_backbones(self):
        for param in self.resnet_backbone.parameters():
            param.requires_grad = False
        for param in self.densenet_backbone.parameters():
            param.requires_grad = False
        print("Both backbones frozen.")

    def unfreeze_top_layers(self):
        # ResNet50: unfreeze layer3 and layer4
        # resnet_backbone children: conv1,bn1,relu,maxpool,layer1,layer2,layer3,layer4,avgpool
        # indices:                    0    1   2     3       4      5      6      7      8
        for i, child in enumerate(self.resnet_backbone.children()):
            if i >= 6:  # layer3, layer4, avgpool
                for param in child.parameters():
                    param.requires_grad = True

        # DenseNet121: unfreeze denseblock3, transition3, denseblock4, norm5
        # densenet_backbone children: conv0,norm0,relu0,pool0,
        #                             denseblock1,transition1,
        #                             denseblock2,transition2,
        #                             denseblock3,transition3,
        #                             denseblock4,norm5
        # indices:                      0     1    2    3
        #                               4         5
        #                               6         7
        #                               8         9
        #                               10    11
        for i, child in enumerate(self.densenet_backbone.children()):
            if i >= 8:  # denseblock3 onwards
                for param in child.parameters():
                    param.requires_grad = True

        print("Top layers of both backbones unfrozen.")


class LightEnsemble(nn.Module):
    """
    Light Ensemble: EfficientNet-B0 (baseline) + MobileNetV2 (baseline)
    Fused embedding: 1280 + 1280 = 2560-dimensional
    MLP classifier: 2560 -> 512 -> 256 -> 4
    """

    def __init__(self, efficientnet_path=None, mobilenet_path=None, num_classes=4):
        super(LightEnsemble, self).__init__()

        # ── EfficientNet-B0 backbone ───────────────────────────────────
        efficientnet = models.efficientnet_b0(weights="IMAGENET1K_V1")

        if efficientnet_path:
            state = torch.load(efficientnet_path, map_location="cpu")
            efficientnet.load_state_dict(state, strict=False)
            print(f"EfficientNet-B0 weights loaded from: {efficientnet_path}")

        # EfficientNet-B0: features block → GAP → (batch, 1280)
        self.efficientnet_features = efficientnet.features
        self.efficientnet_gap = nn.AdaptiveAvgPool2d((1, 1))

        # ── MobileNetV2 backbone ───────────────────────────────────────
        mobilenet = models.mobilenet_v2(weights="IMAGENET1K_V1")

        if mobilenet_path:
            state = torch.load(mobilenet_path, map_location="cpu")
            mobilenet.load_state_dict(state, strict=False)
            print(f"MobileNetV2 weights loaded from: {mobilenet_path}")

        # MobileNetV2: features block → GAP → (batch, 1280)
        self.mobilenet_features = mobilenet.features
        self.mobilenet_gap = nn.AdaptiveAvgPool2d((1, 1))

        # ── Fusion classifier ──────────────────────────────────────────
        # Input: 1280 + 1280 = 2560
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
        # EfficientNet-B0: → (batch, 1280, 7, 7) → GAP → (batch, 1280)
        eff_feat = self.efficientnet_features(x)
        eff_feat = self.efficientnet_gap(eff_feat)
        eff_feat = eff_feat.view(eff_feat.size(0), -1)

        # MobileNetV2: → (batch, 1280, 7, 7) → GAP → (batch, 1280)
        mob_feat = self.mobilenet_features(x)
        mob_feat = self.mobilenet_gap(mob_feat)
        mob_feat = mob_feat.view(mob_feat.size(0), -1)

        # Feature normalisation before concatenation
        eff_feat = F.normalize(eff_feat, p=2, dim=1)
        mob_feat = F.normalize(mob_feat, p=2, dim=1)

        # Concatenate: (batch, 2560)
        fused = torch.cat([eff_feat, mob_feat], dim=1)

        return self.classifier(fused)

    def freeze_backbones(self):
        for param in self.efficientnet_features.parameters():
            param.requires_grad = False
        for param in self.mobilenet_features.parameters():
            param.requires_grad = False
        print("Both backbones frozen.")

    def unfreeze_top_layers(self):
        # EfficientNet-B0 features: indices 0-8, unfreeze last 3
        for i, layer in enumerate(self.efficientnet_features):
            if i >= 6:
                for param in layer.parameters():
                    param.requires_grad = True

        # MobileNetV2 features: indices 0-18, unfreeze last 5
        for i, layer in enumerate(self.mobilenet_features):
            if i >= 14:
                for param in layer.parameters():
                    param.requires_grad = True

        print("Top layers of both backbones unfrozen.")