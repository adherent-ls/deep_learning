import os

import torch
import torch.nn as nn

from models.modules.backbone.cnn.mv3_paddle import MobileNetV3
from models.modules.head.text_recognize.ctc import CTC
from models.modules.neck.rnn.bilstm_paddle import EncoderWithRNN


class CRNN(nn.Module):
    def __init__(self, input_channel, mid_channel, output_channel, num_class, save_path):
        super(CRNN, self).__init__()
        self.save_path = save_path

        self.FeatureExtraction = MobileNetV3(in_channels=input_channel, model_name='large')
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        self.SequenceModeling = EncoderWithRNN(mid_channel, output_channel // 2)
        self.Prediction = CTC(output_channel, num_class)

    def forward(self, input):
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)
        contextual_feature = self.SequenceModeling(visual_feature)
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
