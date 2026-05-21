import os
from mltu.configs import BaseModelConfigs


class ModelConfigs(BaseModelConfigs):
    def __init__(self):
        super().__init__()

        self.model_path = "../models/202211270035/model.onnx"

        self.vocab = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        self.height = 32
        self.width = 128
        self.max_text_length = 23
        self.batch_size = 1024
        self.learning_rate = 1e-4
        self.train_epochs = 100
        self.train_workers = 20