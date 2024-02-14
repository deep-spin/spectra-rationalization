import torch

from rationalizers.modules.sentence_encoders import (
    LSTMEncoder,
)
from rationalizers.utils import load_glove_embeddings


def build_sentence_encoder(
    layer: str,
    in_features: int,
    hidden_size: int,
    bidirectional: bool = True,
):
    if layer == "lstm":
        return LSTMEncoder(in_features, hidden_size, bidirectional=bidirectional)
    else:
        raise Exception(f"Sentence encoder layer `{layer}` not available.")


def build_embedding_weights(vocab: dict, emb_type: str, emb_path: str, emb_size: int):
    if emb_type == "glove":
        return load_glove_embeddings(vocab, emb_path, emb_size)
    # nn.Embedding will initialize the weights randomly
    print("Random weights will be used as embeddings.")
    return None


def build_optimizer(model_parameters, h_params: dict):
    # get valid parameters (unfreezed ones)
    # parameters = filter(lambda p: p.requires_grad, model_parameters)
    parameters = model_parameters
    if h_params["optimizer"] == "adam":
        return torch.optim.Adam(
            parameters,
            lr=h_params["lr"],
            betas=h_params["betas"],
            weight_decay=h_params["weight_decay"],
            amsgrad=h_params["amsgrad"],
        )
    elif h_params["optimizer"] == "adadelta":
        return torch.optim.Adadelta(
            parameters,
            lr=h_params["lr"],
            rho=h_params["rho"],
            weight_decay=h_params["weight_decay"],
        )
    elif h_params["optimizer"] == "adadelta":
        return torch.optim.Adagrad(
            parameters, lr=h_params["lr"], weight_decay=h_params["weight_decay"]
        )
    elif h_params["optimizer"] == "adamax":
        return torch.optim.Adamax(
            parameters,
            lr=h_params["lr"],
            betas=h_params["betas"],
            weight_decay=h_params["weight_decay"],
        )
    elif h_params["optimizer"] == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=h_params["lr"],
            betas=h_params["betas"],
            weight_decay=h_params["weight_decay"],
            amsgrad=h_params["amsgrad"],
        )
    elif h_params["optimizer"] == "sparseadam":
        return torch.optim.SparseAdam(
            parameters,
            lr=h_params["lr"],
            betas=h_params["betas"],
        )
    elif h_params["optimizer"] == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=h_params["lr"],
            momentum=h_params["momentum"],
            dampening=h_params["dampening"],
            weight_decay=h_params["weight_decay"],
            nesterov=h_params["nesterov"],
        )
    elif h_params["optimizer"] == "asgd":
        return torch.optim.ASGD(
            parameters,
            lr=h_params["lr"],
            lambd=h_params["lambd"],
            alpha=h_params["alpha"],
            t0=h_params["t0"],
            weight_decay=h_params["weight_decay"],
        )
    elif h_params["optimizer"] == "rmsprop":
        return torch.optim.RMSprop(
            parameters,
            lr=h_params["lr"],
            alpha=h_params["alpha"],
            weight_decay=h_params["weight_decay"],
            momentum=h_params["momentum"],
            centered=h_params["centered"],
        )
    else:
        raise Exception(f"Optimizer `{h_params['optimizer']}` not available.")


def build_scheduler(optimizer: torch.optim.Optimizer, h_params: dict):
    """Returns a torch lr_scheduler object or None in case a scheduler is not specified."""
    if "scheduler" not in h_params or h_params["scheduler"] is None:
        return None
    elif h_params["scheduler"] == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=h_params["step_size"], gamma=h_params["lr_decay"]
        )
    elif h_params["scheduler"] == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=h_params["milestones"], gamma=h_params["lr_decay"]
        )
    elif h_params["scheduler"] == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=h_params["lr_decay"]
        )
    elif h_params["scheduler"] == "cosine-annealing":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=h_params["T_0"],
            T_mult=h_params["T_mult"],
            eta_min=h_params["eta_min"],
        )
    elif h_params["scheduler"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=h_params["lr_decay"],
            patience=h_params["patience"],
            cooldown=h_params["cooldown"],
            threshold=h_params["threshold"],
            min_lr=h_params["min_lr"],
        )
    else:
        raise Exception(f"Scheduler `{h_params['scheduler']}` not available.")
