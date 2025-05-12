import argparse

def setup_args():
    
    parser = argparse.ArgumentParser(description="Auxiliary Network for LoRA Weights Prediction")
    parser.add_argument("--lora_rank", type=int, default=16, help="Rank of LoRA weights")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-10, help="Learning rate for training")
    parser.add_argument("--aux_model_class", type=str, default="MLP", help="Class of the auxiliary model (MLP, RNN, etc.)")
    parser.add_argument("--mend_rank", type=int, default=16, help="Rank of the auxiliary model")
    parser.add_argument("--seq_model_interior", type=str, default="MLP", help="Interior model for the sequential model (MLP, RNN, etc.)")
    parser.add_argument("--loss_fn", type=str, default="reconstruction_on_prompt_of_interest", help="Loss function for training")
    parser.add_argument("--combine", type=bool, default=True, help="Whether to combine LoRA weights")
    parser.add_argument("--n_hidden", type=int, default=1, help="Number of hidden layers in the auxiliary model")
    parser.add_argument("--init", type=str, default="kaiming_uniform", help="Initialization method for the auxiliary model")
    parser.add_argument("--act", type=str, default="relu", help="Activation function for the auxiliary model")
    parser.add_argument("--rep_dim", type=int, default=128, help="Representation dimension for the auxiliary model")
    parser.add_argument("--pretrained_aux", type=int, default=0, help="Whether to use a pretrained auxiliary model")
    parser.add_argument("--aux_cutoff_layer", type=int, default=4, help="Layer to cut off in the auxiliary model")
    parser.add_argument("--main_model_path", type=str, default="gpt2", help="Path to the main model or name")
    parser.add_argument("--target_layer", type=str, default="c_attn", help="Target layer for LoRA weights")
    parser.add_argument("--aux_model_path", type=str, default="gpt2", help="Path to the auxiliary model or name")
    parser.add_argument("--predict_on_inference", type=int, default=1, help="Whether to predict on inference")
    parser.add_argument("--dataset_portion", type=float, default=0.005, help="Portion of training set to use")
    parser.add_argument("--optimizer", type=str, default="SGD", help="Optimizer for training")
    parser.add_argument("--balanced", type=int, default=0, help="Whether to balance the loss to stay close to old inference")
    parser.add_argument("--regularized", type=int, default=0, help="Whether to use regularization")
    parser.add_argument("--pretraining_epochs", type=int, default=0, help="Number of pretraining epochs")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale for LoRA output")
    args = parser.parse_args()

    return args