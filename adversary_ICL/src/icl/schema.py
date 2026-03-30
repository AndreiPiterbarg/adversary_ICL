"""
Config schema for ICL training.

The original code uses quinine for config validation. We replace it with
a plain dict schema + AttrDict for attribute-style access, removing the
quinine/funcy/munch dependencies.
"""


class AttrDict(dict):
    """Dict subclass that supports attribute-style access (recursive)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key)

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        for k, v in d.items():
            if isinstance(v, dict):
                obj[k] = cls.from_dict(v)
            else:
                obj[k] = v
        return obj

    def toDict(self):
        result = {}
        for k, v in self.items():
            if isinstance(v, AttrDict):
                result[k] = v.toDict()
            else:
                result[k] = v
        return result


TASK_LIST = [
    "linear_regression",
    "sparse_linear_regression",
    "linear_classification",
    "relu_2nn_regression",
    "decision_tree",
]

DEFAULTS = {
    "out_dir": "results/checkpoints",
    "test_run": False,
    "model": {
        "family": "gpt2",
        "n_positions": 101,
        "n_dims": 20,
        "n_embd": 256,
        "n_layer": 12,
        "n_head": 8,
    },
    "training": {
        "task": "linear_regression",
        "task_kwargs": {},
        "num_tasks": None,
        "num_training_examples": None,
        "data": "gaussian",
        "batch_size": 64,
        "learning_rate": 3e-4,
        "train_steps": 1000,
        "save_every_steps": 1000,
        "keep_every_steps": -1,
        "resume_id": None,
        "curriculum": {
            "dims": {
                "start": 5,
                "end": 20,
                "inc": 1,
                "interval": 2000,
            },
            "points": {
                "start": 11,
                "end": 41,
                "inc": 2,
                "interval": 2000,
            },
        },
    },
    "wandb": {
        "project": "in-context-training",
        "entity": "in-context",
        "notes": "",
        "name": None,
        "log_every_steps": 100,
    },
}


def _deep_merge(base, override):
    """Recursively merge override into base dict."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(yaml_dict):
    """Merge a parsed YAML dict with defaults and return an AttrDict object."""
    merged = _deep_merge(DEFAULTS, yaml_dict)
    return AttrDict.from_dict(merged)
