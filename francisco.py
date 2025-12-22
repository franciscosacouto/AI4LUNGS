import torch
import torch.nn as nn

class DINOv2Classifier(nn.Module):
    def __init__(self, pretrained=True):
        super(DINOv2Classifier, self).__init__()
        
        #self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg') if pretrained else None
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc') if pretrained else None
        #self.classifier = nn.Linear(self.backbone.embed_dim, 1)  # Binary classification
        self.classifier = nn.Linear(1000, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        features = self.backbone(x) # CLS token
        logits = self.classifier(features)  # Pass features through the linear layer
        return self.activation(logits)  # Apply sigmoid
        #return logits

#Example usage
if __name__ == "__main__":
    model = DINOv2Classifier()
    dummy_input = torch.randn(1, 3, 224, 224)  # Example input
    output = model(dummy_input)
    print(output.shape)  # Should be (1,)
    print(output)