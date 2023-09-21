import argparse

from torch.utils.data import DataLoader

from data.open_source.dataset import AlignCollate, hierarchical_dataset, Batch_Balanced_Dataset
from data.open_source.utils import CTCLabelConverter


def build_opt():
    parser = argparse.ArgumentParser()

    special_char = ('blank', 'end')
    lines = open('data/open_source/ch_dict.txt', 'r', encoding='utf-8').readlines()
    lines = list(lines)

    character = [special_char[0], special_char[1]]
    for line in lines:
        if line == '':
            continue
        character.append(line.strip('\n'))
    open_args = parser.add_argument_group('open')

    open_args.add_argument('--train_data', default=r'/home/data/data_old/rec_data/411w_train', help='train data')
    open_args.add_argument('--valid_data', default=r'/home/data/data_old/rec_data/lmdb', help='val data')
    open_args.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    open_args.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    open_args.add_argument('--batch_size', type=int, default=32, help='input batch size')
    open_args.add_argument('--batch_ratio', type=str, default='1',
                           help='assign ratio for each selected data in the batch')
    open_args.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                           help='total data usage ratio, this ratio is multiplied to total number of data.')
    open_args.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    open_args.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    open_args.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    open_args.add_argument('--rgb', action='store_true', help='use rgb input')
    open_args.add_argument('--character', type=str,
                           default=character, help='character label')
    open_args.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    open_args.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    open_args.add_argument('--data_filtering_off', default=False, action='store_true',
                           help='for data_filtering_off mode')

    opt = parser.parse_args()
    return opt


def build_data():
    opt = build_opt()
    converter = CTCLabelConverter(opt.character)
    # train_dataset = Batch_Balanced_Dataset(converter, opt.train_data, opt)
    train_dataset = hierarchical_dataset(opt=opt, select_data=opt.train_data)
    AlignCollate_valid = AlignCollate(converter=converter, imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)

    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size,
        shuffle=False,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)

    valid_dataset = hierarchical_dataset(opt=opt, select_data=opt.valid_data)
    valid_loader = DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)

    return train_loader, valid_loader


if __name__ == '__main__':
    train_dataset, valid_loader = build_data()
    for item in train_dataset:
        pass
