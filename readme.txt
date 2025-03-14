This assignment work is organized into different files as below:
    assignment2.ipynb
    model.py
    modules.py
    utils.py
    engine.py

assignment2.ipynb   :   Contains the implementation of all the experiments of the assignment.
model.py            :   Contains the implementation of Vision Transformer model.
modules.py          :   Contains the modules (the building blocks) required for Vision Transformer.
                        { PatchEmbedding, MultiheadSelfattentionBlock, MLP_Block, TransformerEncoderBlock }
utils.py            :   Contains utility functions.
                        { plot_learning_curves(), plot_learning_curves_vs_hparam(), plot_predictions_vs_layers(),
                        get_2_samples_per_class(), get_attentions(), attention_rollout() }
engine.py           :   Contains the functions used for training the model
                        { train_step(), test_step(), train() }

python version used 3.12.7

Packages used:  torch == 2.2.2
                torchvision == 0.17.2
                torchinfo == 1.8.0
                matplotlib == 3.9.2
                tqdm == 4.66.5

To run all the experimentsof the assignment, run the assignment2.ipynb file.