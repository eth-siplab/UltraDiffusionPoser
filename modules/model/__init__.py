from .diffusion.UDP import UDP

MODELS = {"UDP": UDP}


def get_model(args, parser):
    model_cls = MODELS[args.network]
    model_cls.add_args(parser)
    return model_cls
