import os
class Config:
    def __init__(self, **entries):
        for key, value in entries.items():
            if isinstance(value, dict):
                value = Config(**value)
            self.__dict__[key] = value

    def __iter__(self):
        return iter(self.__dict__.items())

    def __repr__(self):
        return repr(self.__dict__)


def get_default_config():
    """Get the default configuration for ESM3"""
    return Config(
        # Updated path based on your test_esm3_TED.py file
        load_weights="esm3dg_weights/ESM3dG_weights_1_lora.ckpt",
        testing={
            "ddg_scanning": True,
            "num_workers": 4,
        },
        training={
            "num_workers": 4,
            "learn_rate": 0.001,
            "epochs": 200,
            "lr_schedule": True,
            "warm_up": True,
            "esm3_learn_rate": 0.001,  # Renamed from mpnn_learn_rate for clarity
            "grad_accum_steps": 4,
            'rank': 4,
            'dropout': 0.15,
        },
        model={
            "hidden_dims": [64, 32],
            "subtract_mut": True,
            "num_final_layers": 2,
            "freeze_weights": True,
            "load_pretrained": True,
            "lightattn": True
        },
    )


def get_model_configs(weights_list=None):
    """Get a list of configs for different ESM3 model weights"""
    if weights_list is None:
        # Default ESM3 weight path from your provided file
        weights_list = [
            "esm3dg_weights/ESM3dG_weights_1_lora.ckpt"
        ]
    cfg = get_default_config()
    configs = []
    
    for weight in weights_list:
        new_config = Config(
            load_weights=weight,
            testing=cfg.testing,
            training=cfg.training, 
            model=cfg.model
        )
        configs.append(new_config)
    
    return configs