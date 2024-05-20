class Task:
    def __init__(self, dict_in:dict):
        self.config_dict = {
            "batch_size": 16,
            "epochs": 100,
            "time_step": 64,
            "patch_size": 4,
            "patch_token_dim": 128,
            "mlp_dim": 32,
            "learning_rate": 0.001,
            "target_mean_len": 1,
            "train_test_ratio": 0.8,
            "dropout": 0.1,
            "alpha": 0.5,
            "teu_dropout": 0.15,
            "pooling_mode": "avg",
            "text_embeddings_dim": 128,
            "shuffle_train": True,
        }

        for k,v in dict_in.items():
            self.config_dict.update({k: v})

        self.result = []
    def append_result(self,eval_mes,eval_mape):
        self.result.append((eval_mes,eval_mape))
    def get_best_n_mape(self):
        ...
    def get_best_n_mse(self):
        ...


