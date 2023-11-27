import inspect


def build_param(cls, function_name, kwargs):
    func = getattr(cls, function_name)
    parameters = inspect.signature(func).parameters

    param = {}
    for param_name, param_obj in parameters.items():
        if param_name not in kwargs:
            continue
        if param_obj.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
            param[param_name] = kwargs[param_name]
        elif param_obj.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        elif param_obj.kind == inspect.Parameter.VAR_KEYWORD:
            param.update(kwargs)
        elif param_obj.kind == inspect.Parameter.KEYWORD_ONLY:
            continue
    return param
