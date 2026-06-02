import torch
import torch.nn as nn

class MultiModalFusionModule(nn.Module):
    def __init__(self, vision_net, text_net, config):
        super().__init__()
        self.vision_net = vision_net
        self.text_net = text_net
        self.config = config

    def forward(self, inputs, tokenizer=None):
        features = []
        
        if 'image' in inputs and getattr(self.config, 'image', False):
            features.append(self.vision_net(inputs['image']))
            
        if 'text' in inputs and getattr(self.config, 'text', False):
            features.append(self.text_net(inputs['text'], tokenizer))
            
        if 'tabular' in inputs and getattr(self.config, 'tabular', False):
            features.append(inputs['tabular'])
            
        return torch.cat(features, dim=1) if len(features) > 1 else features[0]