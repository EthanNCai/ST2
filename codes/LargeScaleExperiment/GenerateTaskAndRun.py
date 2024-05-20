import itertools
from Task import  Task
##### configurations ######

epochs = 10


# single variable experiment

""" <Variables Available>
    batch_size: 16
    epochs: 100
    time_step: 64
    patch_size: 4
    token_dim: 128
    mlp_dim: 32
    learning_rate: 0.001
    target_mean_len: 1
    train_test_ratio: 0.8
    dropout: 0.1
    alpha: 0.5
    teu_dropout: 0.15
    pooling_mode: max
    shuffle_train: True
"""


variable_dict = {
    "patch_size" : [1,2,4,8],
    "time_steps" : [16,32,64,128,126],
}

variable_dict = {
    "text_embeddings_dim"
}

def run():

    ### GENERATE TASKS
    tasks = []
    if len(variable_dict) == 1:
        var = variable_dict.keys()
        for param in variable_dict[var]:
            new_task = Task({var:param})
            tasks.append(new_task)

        ...
    elif len(variable_dict) == 2:
        var_1, var_2 = variable_dict.keys()
        for param_1, param_2 in itertools.product(variable_dict[var_1],variable_dict[var_2]):
            new_task = Task({var_1:param_1, var_2:param_2})
            tasks.append(new_task)

    else:
        print("error")


    ### CONDUCTING EXPERIMENTS



    ### SAVE RESULT


run()
