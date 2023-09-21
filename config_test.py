characters = [item.strip("\n") for item in open("vocab/word/ch_dict.txt", "r", encoding="utf-8").readlines()]
max_length = 50

config = {
    'writer_path': '/home/data/workspace/events/res50_lstm_att',
    'show_step': 10,
    'grad_clip': 1.0,
    'inference_step': 10000,
    'epoch': 100,
    'device': 'cuda:0',
    'data': {
        'train': {
            'batch_size': 4,
            'shuffle': True,
            'num_workers': 8,
            'Dataset': [
                {
                    'name': 'LmdbDatasetFilter',
                    'args': {
                        'root': '/home/data/data_old/rec_data/ch_street/train',
                        'recache': False,
                        'filter': [
                            {
                                'name': 'LabelVocabCheckFilter',
                                'args': {
                                    'label_length_limit': max_length,  # 一般和max_length相同或者max_length - 1
                                    'characters': characters
                                }
                            },
                            {
                                'name': 'ImageCheckFilter',
                                'args': {}
                            }
                        ]
                    }
                },
                {
                    'name': 'LmdbDatasetFilter',
                    'args': {
                        'root': '/home/data/data_old/rec_data/411w_train',
                        'recache': False,
                        'filter': [
                            {
                                'name': 'LabelVocabCheckFilter',
                                'args': {
                                    'label_length_limit': max_length,  # 一般和max_length相同或者max_length - 1
                                    'characters': characters
                                }
                            },
                            {
                                'name': 'ImageCheckFilter',
                                'args': {}
                            }
                        ]
                    }
                }
            ],
            'Transforms': [
                {
                    'name': 'LoadKeys',
                    'args': {
                        'keys': ['images', 'text']
                    }
                },
                {
                    'name': 'Resize',
                    'args': {
                        'max_size': [-1, 32],
                        'is_pad': True,
                        'out_key': ['image_pad', 'images']
                    }
                },
                {
                    'name': 'ZeroMeanNormal',
                    'args': {
                        'scale': 1. / 255.,
                        'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225],
                    }
                },
                {
                    'name': 'LabelCoder',
                    'args': {
                        'characters': characters,
                        'in_key': 'text',
                        'out_key': 'text',
                    }
                },
                {
                    'name': 'LabelAttnCollate',
                    'args': {
                        'max_length': max_length,
                        'in_key': 'text',
                        'out_key': 'text',
                    }
                },
                {
                    'name': 'ImageReshape',
                    'args': {
                        'permute_indices': [0, 3, 1, 2]
                    }
                },
                {
                    'name': 'KeepKeyTensor',
                    'args': {
                        'keep_data_keys': {'images': 'float32'},
                        'keep_label_keys': {'text': 'long'}
                    }
                }
            ]
        },
        'val': {
            'batch_size': 4,
            'shuffle': True,
            'num_workers': 8,
            'Dataset': [
                {
                    'name': 'LmdbDatasetFilter',
                    'args': {
                        'root': '/home/data/data_old/rec_data/ch_street/train',
                        'recache': False,
                        'filter': [
                            {
                                'name': 'LabelVocabCheckFilter',
                                'args': {
                                    'label_length_limit': max_length,  # 一般和max_length相同或者max_length - 1
                                    'characters': characters
                                }
                            },
                            {
                                'name': 'ImageCheckFilter',
                                'args': {}
                            }
                        ]
                    }
                },
                {
                    'name': 'LmdbDatasetFilter',
                    'args': {
                        'root': '/home/data/data_old/rec_data/411w_train',
                        'recache': False,
                        'filter': [
                            {
                                'name': 'LabelVocabCheckFilter',
                                'args': {
                                    'label_length_limit': max_length,  # 一般和max_length相同或者max_length - 1
                                    'characters': characters
                                }
                            },
                            {
                                'name': 'ImageCheckFilter',
                                'args': {}
                            }
                        ]
                    }
                }
            ],
            'Transforms': [
                {
                    'name': 'LoadKeys',
                    'args': {
                        'keys': ['images', 'text']
                    }
                },
                {
                    'name': 'Resize',
                    'args': {
                        'max_size': [-1, 32],
                        'is_pad': True,
                        'out_key': ['image_pad', 'images']
                    }
                },
                {
                    'name': 'ZeroMeanNormal',
                    'args': {
                        'scale': 1. / 255.,
                        'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225],
                    }
                },
                {
                    'name': 'LabelCoder',
                    'args': {
                        'characters': characters,
                        'in_key': 'text',
                        'out_key': 'text',
                    }
                },
                {
                    'name': 'LabelAttnCollate',
                    'args': {
                        'max_length': max_length,
                        'in_key': 'text',
                        'out_key': 'text',
                    }
                },
                {
                    'name': 'ImageReshape',
                    'args': {
                        'permute_indices': [0, 3, 1, 2]
                    }
                },
                {
                    'name': 'KeepKeyTensor',
                    'args': {
                        'keep_data_keys': {'images': 'float32'},
                        'keep_label_keys': {'text': 'long'}
                    }
                }
            ]
        }
    },
    'model': {
        'resume_path': '',
        'save_path': '',
        'strict': False,
        'Modules': [
            {
                'name': 'ResNetQuery',
                'args': {
                    'input_channel': 3,
                    'output_channel': 512,
                }
            },
            {
                'name': 'Reshape',
                'args': {
                    'transpose_index': [0, 3, 2, 1],
                    'reshape_size': [-1, -1, 512],
                }
            },
            {
                'name': 'BiLSTM',
                'args': {
                    'input_size': 512,
                    'hidden_size': 512,
                    'output_size': 512,
                }
            },
            {
                'name': 'MapNeck',
                'args': {
                    'in_channel': 256,
                    'out_channel': 512,
                    'max_length': max_length,
                }
            },
            {
                'name': 'FCPrediction',
                'args': {
                    'in_channel': 512,
                    'n_class': len(characters),
                    'out_key': 'text_prop',
                }
            }
        ]
    },

    'optim': {
        'Optimizer': {
            'name': 'torch.optim.Adam',
            'args': {
                'lr': 0.0001,
                'betas': [0.9, 0.999],
                'weight_decay': 0.0005,
            }
        },
        'Scheduler': {
            'name': 'Warmup',
            'args': {
                'warm': 10000,
            }
        }
    },

    'loss': {
        'weight': [1.0],
        'keys': [['text_prop', 'text']],
        'Loss': [
            {
                'name': 'CrossEntropyLoss',
                'args': {
                    'ignore_index': 0,
                }
            }
        ]
    },
    'metric': {
        'weight': [1.0],
        'keys': [['text_prop', 'text']],
        'Metric': [
            {
                'name': 'TextMetric',
                'args': {
                    'decode': {
                        'name': 'TextDecoder',
                        'args': {
                            'characters': characters
                        }
                    }
                }
            }
        ]
    }
}