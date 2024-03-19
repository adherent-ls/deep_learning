import os

import torch
import torch.nn as nn

from models.modules.backbone.cnn.mv3_paddle import MobileNetV3
from models.modules.backbone.cnn.res_adapt import ResNet
from models.modules.head.text_recognize.ctc import CTC
from models.modules.neck.rnn.bilstm_paddle import EncoderWithRNN


class Extraction(nn.Module):
    def __init__(self, input_channel, mid_channel, output_channel):
        super().__init__()
        self.FeatureExtraction = ResNet(input_channel, mid_channel)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((1, None))  # Transform final (imgH/16-1) -> 1

        self.SequenceModeling = EncoderWithRNN(mid_channel, output_channel // 2)

    def forward(self, input):
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature)
        visual_feature = visual_feature.squeeze(2).permute(0, 2, 1)  # [b, c, h=1, w] -> [b, c, w] -> [b, w, c]
        contextual_feature = self.SequenceModeling(visual_feature)
        return contextual_feature


class CRNN(nn.Module):
    def __init__(self, input_channel, mid_channel, output_channel, num_class, save_path):
        super(CRNN, self).__init__()
        # self.save_path = save_path
        # self.FeatureExtraction = ResNet(input_channel, mid_channel)
        # self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((1, None))  # Transform final (imgH/16-1) -> 1
        #
        # self.SequenceModeling = EncoderWithRNN(mid_channel, output_channel // 2)
        self.Extraction = Extraction(input_channel, mid_channel, output_channel)
        self.Prediction = CTC(output_channel, num_class)

    def forward(self, input):
        # visual_feature = self.FeatureExtraction(input)
        # visual_feature = self.AdaptiveAvgPool(visual_feature)
        # visual_feature = visual_feature.squeeze(2).permute(0, 2, 1)  # [b, c, h=1, w] -> [b, c, w] -> [b, w, c]
        # contextual_feature = self.SequenceModeling(visual_feature)
        contextual_feature = self.Extraction(input)
        prediction = self.Prediction(contextual_feature.contiguous())
        return prediction

    def save_model(self, name, metric):
        save_path = self.save_path
        os.makedirs(save_path, exist_ok=True)
        save_name = os.path.join(save_path, f'{name}.pth')
        model_state = self.state_dict()
        information = {
            'metric': metric,
            'model': model_state
        }
        torch.save(information, save_name)
        print(f'save:{name}, {save_path}, {str(metric)}')

    def resume_model(self, resume_path, strict=True):
        if resume_path is None:
            print('resume_path is None')
            return None
        if not os.path.exists(resume_path):
            print('not os.path.exists(resume_path)', resume_path)
            return None
        if resume_path == '':
            print('resume_path == ""')
            return None
        print(resume_path)
        config = torch.load(resume_path, map_location='cpu')
        self.load_state_dict(config, strict=strict)
