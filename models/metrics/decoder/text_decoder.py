from base.metric.base_decoder import BaseDecoder


class TextDecoder(BaseDecoder):
    def __init__(self, characters, start=0, end=-1):
        super().__init__()
        self.characters = characters
        self.start = start
        self.end = end

    def __call__(self, index):
        texts = []
        for item in index:
            text = ''
            for single in item:
                chars = self.characters[int(single)]
                # if chars == self.characters[self.end]:  # 结束符号
                #     break
                if chars != self.characters[self.start]:  # 开始/空白符号
                    text += self.characters[int(single)]
            texts.append(text)
        return texts
