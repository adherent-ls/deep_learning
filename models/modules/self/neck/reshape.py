from base.model.base_model import BaseModel


class Reshape(BaseModel):

    def __init__(self, transpose_index=None, reshape_size=None, **kwargs):
        super(Reshape, self).__init__(**kwargs)
        self.transpose_index = transpose_index
        self.reshape_size = reshape_size

    def forward(self, out):
        if self.transpose_index is not None:
            out = out.permute(*self.transpose_index)
        out = out.contiguous()
        reshape_size = self.reshape_size
        reshape_size[0] = out.shape[0]  # batch_size 根据模型输入确定
        if self.reshape_size is not None:
            out = out.view(*self.reshape_size)
        return out
