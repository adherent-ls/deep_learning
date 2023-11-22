def build_param(cls, kwargs):
    codes = cls.__init__.__code__
    param_name_list = codes.co_varnames
    param_num = codes.co_argcount
    if param_num == len(param_name_list):
        param = {}
        for k, v in kwargs.items():
            if k in param_name_list:
                param[k] = v
    else:
        param = kwargs
    return param
