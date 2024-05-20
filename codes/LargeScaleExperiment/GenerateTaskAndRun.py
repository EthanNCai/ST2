
##### configurations ######

epochs = 10


# single variable experiment

""" <Variables Available>
    batch_size: 16
    epochs: 100
    time_step: 64
    patch_size: 4
    patch_token_dim: 128
    mlp_dim: 32
    learning_rate: 0.001
    target_mean_len: 1
    train_test_ratio: 0.8
    dropout: 0.1
    alpha: 0.5
    teu_dropout: 0.15
    pooling_mode: max
    text_embeddings_dim: 128
    shuffle_train: True
"""


variable_dict = {
    "patch_size" : [1,2,4,8],
    "time_steps" : [16,32,64,128,126],
}

def run():

    tasks = []
    if len(variable_dict) == 1:
        var_1 = variable_dict.keys()

        ...
    elif len(variable_dict) == 2:
        var_1, var_2 = variable_dict.keys()
        print(var_1, var_2)
    else:
        print("error")


run()
