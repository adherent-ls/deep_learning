from base.data.base_filter import LabelFilter


class LmdbOutVocabFilter(LabelFilter):
    def __init__(self, characters, max_length=None):
        super(LmdbOutVocabFilter, self).__init__()
        self.characters = characters if isinstance(characters, list) else eval(characters)
        self.label_length_limit = max_length

    def forward(self, label):
        is_valid = False
        if label is None:
            return is_valid

        if not isinstance(label, str):
            label = label.decode('utf-8')

        if self.label_length_limit is not None:
            if len(label) >= self.label_length_limit:
                return is_valid
        out_vocab = False
        for item in label:
            if item not in self.characters:
                out_vocab = True
                break
        if out_vocab:
            return is_valid
        is_valid = True
        return is_valid


class LmdbIsExistFilter(LabelFilter):
    def __init__(self, label_length_limit=None):
        super(LmdbIsExistFilter, self).__init__()
        self.label_length_limit = label_length_limit

    def forward(self, label):
        is_valid = False
        if label is None:
            return is_valid

        if not isinstance(label, str):
            label = label.decode('utf-8')

        if self.label_length_limit is not None:
            if len(label) >= self.label_length_limit:
                return is_valid
        is_valid = True
        return is_valid


class BoxFilter(LabelFilter):
    def __init__(self, length_limit=None):
        super(BoxFilter, self).__init__()
        self.length_limit = length_limit

    def forward(self, label):
        is_valid = False
        if label is None:
            return is_valid
        if self.length_limit is not None:
            if len(label) >= self.length_limit:
                return is_valid
        is_valid = True
        return is_valid
