from torch.nn import DataParallel

from build.model.build_module import BuildModule


class BuildDataParallel(DataParallel):
    def __init__(self, model: BuildModule, device_ids=None, output_device=None, dim=0):
        super().__init__(model.modules, device_ids, output_device, dim)
