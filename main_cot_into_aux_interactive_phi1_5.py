import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from functools import partial

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification
from datasets import load_dataset
from mend import AuxNetwork
import tensorly as tl
from tensorly.decomposition import tucker
import numpy as np
from args import setup_args
import time
tl.set_backend('pytorch')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def filter_function(example, tokenizer):
    prompt_without_rationale, prompt_of_interest = format_prompt(example, include_rationale=False)
    prompt_with_rationale, _ = format_prompt(example, include_rationale=True)
    inputs_model = tokenizer(prompt_without_rationale, return_tensors="pt")
    inputs_ref = tokenizer(prompt_with_rationale, return_tensors="pt")
    return len(inputs_model.input_ids[0]) <= tokenizer.model_max_length and len(inputs_ref.input_ids[0]) <= tokenizer.model_max_length 

def get_token_range(model_input_ids, interest_input_ids):
    # Find the starting position of inputs_of_interest in inputs_model
    start_idx = -1
    for i in range(len(model_input_ids) - len(interest_input_ids) + 1):
        if model_input_ids[i:i+len(interest_input_ids)] == interest_input_ids:
            start_idx = i
            break
    end_idx = start_idx + len(interest_input_ids) - 1
    interest_token_range = (start_idx, end_idx)
    return interest_token_range

def evaluate_model(model, aux_params, tokenizer, dataset, split="test", include_rationale=False):

    if aux_params is not None:
        aux_model, lora_right_shape, lora_left_shape, aux_layers = aux_params
        # aux_model.eval()

    total_correct = 0
    total_questions = 0

    for example in dataset[split]:
        prompt, _ = format_prompt(example, include_rationale)
        # Get log probabilities for each option
        log_probs = []
        options = ["A", "B", "C", "D", "E"]
        
        for option in options:
            option_prompt = prompt + option
            option_inputs = tokenizer(option_prompt, return_tensors="pt").to(device)
            rationale_input = tokenizer(example['rationale'], return_tensors="pt").to(device)
            with torch.no_grad():
                if aux_params is not None:
                    aux_output = aux_model(**rationale_input.to(aux_model.device))
                    lora_input_matrix_params = aux_output.logits[:, :lora_right_shape[0] * lora_right_shape[1]]
                    lora_output_matrix_params = aux_output.logits[:, lora_right_shape[0] * lora_right_shape[1]:]
                    lora_right = lora_input_matrix_params.view(1, *lora_right_shape) # had batch size before htis
                    lora_left = lora_output_matrix_params.view(1, *lora_left_shape)

                    for layer in aux_layers:
                        layer.set_lora_weights(lora_right, lora_left)

                outputs = model(**option_inputs.to(model.device))
                logits = outputs.logits
                
            # Get the log probability of the option token
            option_token_id = tokenizer.encode(option)[-1]
            option_position = len(option_inputs.input_ids[0]) - 1
            option_logit = logits[0, option_position - 1, option_token_id]
            log_probs.append(option_logit.item())
        
        # Get the predicted answer (option with highest log probability)
        predicted_idx = log_probs.index(max(log_probs))
        predicted_answer = options[predicted_idx]
        
        # Compare with ground truth
        ground_truth = example['correct']
        is_correct = predicted_answer == ground_truth
        total_correct += is_correct 
        total_questions += 1

    # print(f"Total correct: {total_correct}, Total questions: {total_questions}, Accuracy: {total_correct / total_questions}")
    return total_correct / total_questions

def format_prompt(example, include_rationale=False):
    # rationale = example['rationale'][:example['rationale'].rfind("\n")] # Last line contains the answer so we remove it
    rationale = example['rationale']
    if include_rationale:
        prompt = f"""Answer the following math question given the following rationale.
Question: {example['question']}
A) {example['options'][0][2:]}
B) {example['options'][1][2:]}
C) {example['options'][2][2:]}
D) {example['options'][3][2:]}
E) {example['options'][4][2:]}
Rationale: {rationale}
Output only the letter of the correct answer.
Answer: """
    else:
        prompt = f"""Answer the following math question by choosing the correct answer from the options.
Question: {example['question']}
A) {example['options'][0][2:]}
B) {example['options'][1][2:]}
C) {example['options'][2][2:]}
D) {example['options'][3][2:]}
E) {example['options'][4][2:]}
Output only the letter of the correct answer.
Answer: """
    prompt_of_interest = f"""Question: {example['question']}
A) {example['options'][0][2:]}
B) {example['options'][1][2:]}
C) {example['options'][2][2:]}
D) {example['options'][3][2:]}
E) {example['options'][4][2:]}
"""
    return prompt, prompt_of_interest

class LoRA_Linear(nn.Module):
    def __init__(self, weight, bias, lora_dim):
        super(LoRA_Linear, self).__init__()
        row, column = weight.shape
        # restore Linear
        if bias is None:
            self.linear = nn.Linear(column, row, bias=False)
            self.linear.load_state_dict({"weight": weight})
        else:
            self.linear = nn.Linear(column, row)
            self.linear.load_state_dict({"weight": weight, "bias": bias})

        # create LoRA weights (with initialization)
        self.lora_right = nn.Parameter(torch.zeros(column, lora_dim))
        nn.init.kaiming_uniform_(self.lora_right, a=math.sqrt(5))
        self.lora_left = nn.Parameter(torch.zeros(lora_dim, row // 3 * 2)) # query, key, value but we only want to apply lora to query and key

    def forward(self, input):
        x = self.linear(input)
        y = input @ self.lora_right @ self.lora_left
        y = torch.cat([y, torch.zeros(y.shape[0], y.shape[1], x.shape[2]-y.shape[2]).to(device)], dim=2) # only apply lora to query and key
        return x + y

class Predict_LoRA_Linear(nn.Module):
    def __init__(self, weight, bias, lora_dim, predict=False, hidden_dim=256, args=None):
        super(Predict_LoRA_Linear, self).__init__()
        row, column = weight.shape

        self.predict = predict # Indicator variable for whether we are predicting on this one 

        # restore Linear
        if bias is None:
            self.linear = nn.Linear(column, row, bias=False)
            self.linear.load_state_dict({"weight": weight})
        else:
            self.linear = nn.Linear(column, row)
            self.linear.load_state_dict({"weight": weight, "bias": bias})

        # Define dimensions for LoRA weights
        self.lora_dim = lora_dim
        self.right_shape = (column, lora_dim)
        self.left_shape = (lora_dim, row)  # query, key, value but we only want to apply lora to query and key

        # Create LoRA weights (initalised to provide id before aux is enabled)
        lora_right = nn.Parameter(torch.zeros(1, column, lora_dim)).to(device)
        lora_left = nn.Parameter(torch.zeros(1, lora_dim, row)).to(device)
        self._cached_lora_weights = (lora_right, lora_left)
        self.scale = args.scale

        if torch.cuda.device_count() > 1:
            self.model_device = "cuda:1"
        else:
            self.model_device = "cuda:0"

    def set_lora_weights(self, lora_right, lora_left):
        # Set LoRA weights
        # lora_right.requires_grad = True
        # lora_left.requires_grad = True
        self._cached_lora_weights = (lora_right, lora_left)

    def send_to_device(self, device):
        self._cached_lora_weights = (self._cached_lora_weights[0].to(device), self._cached_lora_weights[1].to(device))

    def forward(self, input, eval=False):
        batch_size, seq_len, emb_dim = input.shape

        # # print(f"Layer Input {input.shape}")
        
        # Apply the base linear layer
        x = self.linear(input)

        lora_right, lora_left = self._cached_lora_weights
        if lora_right.requires_grad == False:
            lora_right.requires_grad = True
            lora_left.requires_grad = True
        lora_right.retain_grad()
        lora_left.retain_grad()
        
        # Apply LoRA for each item in the batch
        y = torch.stack([
            input[i].to(self.model_device) @ lora_right[i].to(self.model_device) @ lora_left[i].to(self.model_device)
            for i in range(batch_size)
        ])

        y = y / torch.norm(y, dim=-1, keepdim=True)
        y = y * self.scale
        # y = torch.softmax(y, dim=-1) * SCALE

        return x.to(self.model_device) + y.to(self.model_device)
    
def load_tokenizer_and_model(model_name, args, cutoff=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if cutoff is not None:
        config = AutoConfig.from_pretrained(model_name)
        config.n_layer = cutoff
        if bool(args.pretrained_aux):
            model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
        else:
            model = AutoModelForSequenceClassification.from_config(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    return tokenizer, model

def slice_model(model, cutoff):
    model.model.layers = model.model.layers[:cutoff]
    return model

def replace_head(model, head_name, out_features):
    # Get the original head
    original_head = getattr(model, head_name)

    # Create a new head with the same configuration
    new_head = nn.Linear(original_head.in_features, out_features).to(device)

    # Replace the original head with the new one
    setattr(model, head_name, new_head)

    # Return the modified model
    return model
            

def get_output_logit_distance(model, ref_model, tokenizer, example, type="mse", balanced=False):
    
    prompt_without_rationale, prompt_of_interest = format_prompt(example, include_rationale=False)
    prompt_with_rationale, _ = format_prompt(example, include_rationale=True)

    inputs_model = tokenizer(prompt_without_rationale, return_tensors="pt").to(device)
    inputs_ref = tokenizer(prompt_with_rationale, return_tensors="pt").to(device)
    inputs_of_interest = tokenizer(prompt_of_interest, return_tensors="pt").to(device)

    # # print(inputs_model["input_ids"].shape)
    # # print(inputs_ref["input_ids"].shape)

    outputs_model = model(**inputs_model.to(model.device), output_attentions=True, output_hidden_states=False)
    with torch.no_grad():
        outputs_ref = ref_model(**inputs_ref.to(ref_model.device), output_attentions=True, output_hidden_states=False)

    k = outputs_model.logits.shape[1]

    # # print(outputs_model.logits[0].shape)
    # exit(0)

    outputs_model.logits = outputs_model.logits.to(device)
    outputs_ref.logits = outputs_ref.logits.to(device)

    # print(outputs_model.logits[0].shape)
    # print(outputs_ref.logits[0].shape)

    if type == "mse":
        distance = torch.sum(torch.square(outputs_model.logits - outputs_ref.logits[:, :k, :]))
    elif type == "crossentropy":
        fn = nn.CrossEntropyLoss()
        distance = fn(outputs_model.logits[0][-1], outputs_ref.logits[0, -1, :].softmax(dim=0))
    elif type == "forward_kl":
        fn = nn.KLDivLoss(reduction='batchmean')
        distance = fn(outputs_model.logits[0][-1].log_softmax(dim=0), outputs_ref.logits[0, -1, :].softmax(dim=0))
    elif type == "backward_kl":
        fn = nn.KLDivLoss(reduction='batchmean')
        distance = fn(outputs_ref.logits[0, -1, :].log_softmax(dim=0), outputs_model.logits[0][-1].softmax(dim=0))

    if balanced:
        with torch.no_grad():
            outputs_base = ref_model(**inputs_model, output_attentions=True, output_hidden_states=False)

        if type == "backward_kl":
            distance += 0.5 * fn(outputs_base.logits[0][-1].log_softmax(dim=0), outputs_model.logits[0, -1, :].softmax(dim=0))
        else:
            distance += 0.5 * fn(outputs_model.logits[0][-1].log_softmax(dim=0), outputs_base.logits[0, -1, :].softmax(dim=0))

    return distance

def get_tokenized_inputs(tokenizer, example):

    prompt_without_rationale, prompt_of_interest = format_prompt(example, include_rationale=False)
    prompt_with_rationale, _ = format_prompt(example, include_rationale=True)

    inputs_model = tokenizer(prompt_without_rationale, return_tensors="pt").to(device)
    inputs_ref = tokenizer(prompt_with_rationale, return_tensors="pt").to(device)
    inputs_of_interest = tokenizer(prompt_of_interest, return_tensors="pt").to(device)
    rationale_inputs = tokenizer(example["rationale"], return_tensors="pt").to(device)

    return inputs_model, inputs_ref, inputs_of_interest, rationale_inputs

def get_attention_scores(model, ref_model, inputs_model, inputs_ref, inputs_of_interest):

    outputs_model = model(**inputs_model, output_attentions=True, output_hidden_states=False)
    with torch.no_grad():
        outputs_ref = ref_model(**inputs_ref, output_attentions=True, output_hidden_states=False)
    # # print((outputs_model.logits))
    # # print(outputs_ref.logits.shape)
    # exit(0)

    attentions_model = outputs_model.attentions  # Tuple of attention tensors for each layer
    attentions_ref = outputs_ref.attentions  # Tuple of attention tensors for each layer

    return attentions_model, attentions_ref, (inputs_model, inputs_ref, inputs_of_interest)


def extract_attentions_of_interest(attentions_model, attentions_ref, inputs_model, inputs_ref, inputs_of_interest):
    # Find the range of token indices where inputs_of_interest is contained in inputs_model
    model_input_ids = inputs_model.input_ids[0].tolist()
    ref_input_ids = inputs_ref.input_ids[0].tolist()
    interest_input_ids = inputs_of_interest.input_ids[0].tolist()
    
    interest_token_range_model = get_token_range(model_input_ids, interest_input_ids)
    interest_token_range_ref = get_token_range(ref_input_ids, interest_input_ids)
    
    # Add small epsilon to avoid division by zero and ensure numerical stability
    epsilon = 1e-10
    interest_token_attentions_model = [attn[0, :, -1, interest_token_range_model[0]:interest_token_range_model[1]+1] / (attn[0, :, -1, interest_token_range_model[0]:interest_token_range_model[1]+1].sum(dim=-1).unsqueeze(-1) + epsilon) for attn in attentions_model]  # Extract attention scores for last token
    interest_token_attentions_ref = [attn[0, :, -1, interest_token_range_ref[0]:interest_token_range_ref[1]+1] / (attn[0, :, -1, interest_token_range_ref[0]:interest_token_range_ref[1]+1].sum(dim=-1).unsqueeze(-1) + epsilon) for attn in attentions_ref]  # Extract attention scores for last token

    return interest_token_attentions_model, interest_token_attentions_ref
    

def train_aux(model, loss_fn, aux_model, aux_layers, aux_layer_indices, lora_right_shape, lora_left_shape, tokenizer, dataset, epochs=10, batch_size=4, learning_rate=5e-5, split="train", balanced=False, args=None):
    """
    Train the auxiliary network with attention matrix pairs
    """

    # Create a copy of the original model for reference (no LoRA)
    reference_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5").to(device).to(torch.float32)

    # set only to aux parameters means that this will only train the auxiliary model
    optimizer = getattr(optim, args.optimizer)(aux_model.parameters(), lr=learning_rate)
    assert(all(p.requires_grad for p in aux_model.parameters()))

    train_size = len(dataset[split])
    num_samples = int(train_size * args.dataset_portion) + (batch_size - (int(train_size * args.dataset_portion) % batch_size))
    indices = torch.randperm(train_size)[:num_samples].tolist()
    dataset[split] = dataset[split].select(indices)

    dataset[split] = dataset[split].filter(lambda example: filter_function(example, tokenizer))

    train_dataloader = DataLoader(dataset[split], batch_size=batch_size, shuffle=True)
    # Training loop
    aux_model.train()
    model.eval()
    reference_model.eval()  # Reference model is not trained

    for n, p in model.named_parameters():
        p.requires_grad = False
    for n, p in reference_model.named_parameters():
        p.requires_grad = False
    for n, p in aux_model.named_parameters():
        p.requires_grad = True

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for i, batch in enumerate(train_dataloader):
            batch_loss = 0.0
            
            # Process each example in the batch
            for example_idx in range(len(batch['question'])):
                example = {'question': batch['question'][example_idx], 'options': [batch['options'][0][example_idx], batch['options'][1][example_idx], batch['options'][2][example_idx], batch['options'][3][example_idx], batch['options'][4][example_idx]], 'rationale': batch['rationale'][example_idx], 'correct': batch['correct'][example_idx]}
                # print(example)
                # exit(0)
                inputs_model, inputs_ref, inputs_of_interest, rationale_inputs = get_tokenized_inputs(tokenizer, example)

                # Predict the lora_weights
                aux_output = aux_model(**rationale_inputs)
                lora_input_matrix_params = aux_output.logits[:, :lora_right_shape[0] * lora_right_shape[1]]
                lora_output_matrix_params = aux_output.logits[:, lora_right_shape[0] * lora_right_shape[1]:]
                lora_right = lora_input_matrix_params.view(1, *lora_right_shape).to(model.device) # had batch size before htis
                lora_left = lora_output_matrix_params.view(1, *lora_left_shape).to(model.device)

                # Iterate through the references to the aux_layers in the main model and set the lora weights
                for layer in aux_layers:
                    layer.set_lora_weights(lora_right, lora_left)
                    layer.send_to_device(model.device)

                if loss_fn == "reconstruction_on_prompt_of_interest":
                    lora_attention_scores, ref_attention_scores, inputs = get_attention_scores(model, reference_model, inputs_model, inputs_ref, inputs_of_interest)

                    lora_attention_scores, ref_attention_scores = extract_attentions_of_interest(lora_attention_scores, ref_attention_scores, inputs[0], inputs[1], inputs[2])

                    # Reconstruction error between the two
                    # MSE

                    fn = nn.MSELoss()
                    loss = 0
                    for idx in aux_layer_indices:
                        loss += fn(lora_attention_scores[idx], ref_attention_scores[idx])

                    batch_loss += loss
                elif loss_fn == "reconstruction_on_prompt_of_interest_plus_kl":
                    lora_attention_scores, ref_attention_scores, inputs = get_attention_scores(model, reference_model, inputs_model, inputs_ref, inputs_of_interest)

                    lora_attention_scores, ref_attention_scores = extract_attentions_of_interest(lora_attention_scores, ref_attention_scores, inputs[0], inputs[1], inputs[2])

                    # Reconstruction error between the two
                    # MSE

                    fn = nn.MSELoss()
                    loss = 0
                    for idx in aux_layer_indices:
                        loss += 10*get_output_logit_distance(model, reference_model, tokenizer, example, type="forward_kl")
                        loss += fn(lora_attention_scores[idx], ref_attention_scores[idx])

                    batch_loss += loss
                elif loss_fn == "reconstruction_on_rationale_tucker":
                    # Extract attention values for both ref with rationale and model with normal for target layer(s)
                    # Use tucker to reduce the attention values for ref with rationale down to the shape of attn from model with normal
                    # Reconstruction error on these attn values

                    lora_attention_scores, ref_attention_scores, inputs = get_attention_scores(model, reference_model, inputs_model, inputs_ref, inputs_of_interest)
                    
                    loss = 0
                    for idx in aux_layer_indices:
                        lora_model_attn_shape = lora_attention_scores[idx].shape

                        core, _ = tucker(ref_attention_scores[idx], rank=list(lora_model_attn_shape))
                        # # print(core.shape)
                        loss += torch.sum(torch.square(lora_attention_scores[idx] - core))
                    batch_loss += loss
                elif loss_fn == "reconstruction_on_both_tucker":
                    # Choose a dimension k
                    # tucker both down to this dimension
                    # reconstruction loss on this

                    lora_attention_scores, ref_attention_scores, inputs = get_attention_scores(model, reference_model, inputs_model, inputs_ref, inputs_of_interest)
                    loss = 0
                    for idx in aux_layer_indices:
                        lora_model_attn_shape = lora_attention_scores[idx].shape
                        rank = min(lora_model_attn_shape[2], 69)

                        target_rank = [lora_model_attn_shape[0], lora_model_attn_shape[1], rank, rank]
                        lora_core, _ = tucker(lora_attention_scores[idx], rank=target_rank)
                        ref_core, _ = tucker(ref_attention_scores[idx], rank=target_rank)

                        loss += torch.sum(torch.square(lora_core - ref_core))
                    batch_loss += loss
                elif loss_fn == "crossentropy_on_prompt_of_interest":
                    lora_attention_scores, ref_attention_scores, inputs = get_attention_scores(model, reference_model, inputs_model, inputs_ref, inputs_of_interest)

                    lora_attention_scores, ref_attention_scores = extract_attentions_of_interest(lora_attention_scores, ref_attention_scores, inputs[0], inputs[1], inputs[2])

                    # Reconstruction error between the two
                    # CrossEntropy

                    fn = nn.CrossEntropyLoss()
                    loss = 0

                    for idx in aux_layer_indices:
                        lora_attn_layer = lora_attention_scores[idx]
                        loss += fn(torch.transpose(lora_attn_layer, 0, 1), torch.transpose(lora_attn_layer, 0, 1).softmax(dim=1))

                    batch_loss += loss
                elif loss_fn == "output_logit_distance":
                    loss = get_output_logit_distance(model, reference_model, tokenizer, example, type="crossentropy", balanced=balanced)
                    batch_loss += loss
                elif loss_fn == "forward_kl":
                    loss = get_output_logit_distance(model, reference_model, tokenizer, example, type="forward_kl", balanced=balanced)
                    batch_loss += loss
                elif loss_fn == "backward_kl":
                    loss = get_output_logit_distance(model, reference_model, tokenizer, example, type="backward_kl", balanced=balanced)
                    batch_loss += loss
                elif loss_fn == "combined_kl":
                    forward_kl = get_output_logit_distance(model, reference_model, tokenizer, example, type="forward_kl", balanced=balanced)
                    backward_kl = get_output_logit_distance(model, reference_model, tokenizer, example, type="backward_kl", balanced=balanced)
                    loss = forward_kl + backward_kl
                    batch_loss += loss
                elif loss_fn == "kl_cross_entropy":
                    forward_kl = get_output_logit_distance(model, reference_model, tokenizer, example, type="forward_kl", balanced=balanced)
                    backward_kl = get_output_logit_distance(model, reference_model, tokenizer, example, type="backward_kl", balanced=balanced)
                    cross_entropy = get_output_logit_distance(model, reference_model, tokenizer, example, type="crossentropy", balanced=balanced)
                    loss = 10000*(forward_kl + backward_kl) + cross_entropy
                    batch_loss += loss
                elif loss_fn == "attn_kl":
                    lora_attention_scores, ref_attention_scores, inputs = get_attention_scores(model, reference_model, inputs_model, inputs_ref, inputs_of_interest)

                    lora_attention_scores, ref_attention_scores = extract_attentions_of_interest(lora_attention_scores, ref_attention_scores, inputs[0], inputs[1], inputs[2])

                    # Reconstruction error between the two
                    # MSE

                    forward = nn.KLDivLoss(reduction="batchmean")
                    backward = nn.KLDivLoss(reduction="batchmean")
                    loss = 0
                    for idx in aux_layer_indices:
                        loss += forward(lora_attention_scores[idx].log_softmax(dim=-1), ref_attention_scores[idx].softmax(dim=-1)) + backward(ref_attention_scores[idx].log_softmax(dim=-1), lora_attention_scores[idx].softmax(dim=-1))

                    batch_loss += loss
                elif loss_fn == "attn_kl_tucker":
                    lora_attention_scores, ref_attention_scores, inputs = get_attention_scores(model, reference_model, inputs_model, inputs_ref, inputs_of_interest)
                    
                    forward = nn.KLDivLoss(reduction="batchmean")
                    backward = nn.KLDivLoss(reduction="batchmean")
                    loss = 0
                    for idx in aux_layer_indices:
                        lora_model_attn_shape = lora_attention_scores[idx].shape

                        core, _ = tucker(ref_attention_scores[idx], rank=list(lora_model_attn_shape))
                        # # print(core.shape)
                        loss += forward(lora_attention_scores[idx].log_softmax(dim=-1), core.softmax(dim=-1)) + backward(core.log_softmax(dim=-1), lora_attention_scores[idx].softmax(dim=-1))
                    batch_loss += loss
            
            # Average loss over batch
            # batch_loss = batch_loss / batch_size
            
            # Backpropagation
            optimizer.zero_grad()
            # # print("BACKWARD")
            batch_loss.backward()
            batch_loss.retain_grad()

            # for p in aux_model.parameters():
            #     # # print grad norm
            #     if p.grad is None:
            #         # print(f"Parameter {p} has no gradient")
            #     else:
            #         # print(f"Parameter {p} grad norm: {p.grad.norm()}")
            # # # print(all(p.requires_grad for p in aux_parameters))
            # exit(0)

            optimizer.step()
            
            total_loss += batch_loss.item()
            num_batches += 1
            
            # if (i + 1) % 5 == 0:
                # print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {batch_loss.item():.4f}")
                # wandb.log({"train/batch_loss": batch_loss.item(), "batch": i + epoch * len(train_dataloader)})

            # LIMIT FOR TESTING
            # if i > 25:
            #     # continue
            #     break
                
        # # print epoch summary
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
        
        # Evaluate and log metrics
        aux_params = (aux_model, lora_right_shape, lora_left_shape, aux_layers)
        if not bool(args.predict_on_inference):
            aux_params = None
        # acc_no_rationale = 0
        # acc_rationale = evaluate_model(model, aux_params, tokenizer, dataset, split="test", include_rationale=True)
        acc_no_rationale = evaluate_model(model, aux_params, tokenizer, dataset, split="test", include_rationale=False)
        # print(f"Accuracy with rationale: {acc_rationale:.4f}")
        # print(f"Accuracy without rationale: {acc_no_rationale:.4f}")
        # wandb.log({"train/batch_loss": batch_loss.item(), "batch": i + epoch * len(train_dataloader)})
        wandb.log({
            "train/accuracy_without_rationale": acc_no_rationale
        })
        

    return aux_model, acc_no_rationale
    
def main_accelerate_cot(args):
    # Initialize wandb
    wandb.init(
        project="cot-acceleration_testing", 
        config=args
    )

    # Start timer
    start_time = time.time()
    
    tokenizer, model = load_tokenizer_and_model("microsoft/phi-1_5", args)
    model = model.to(torch.float32).to(device)
    dataset = load_dataset("deepmind/aqua_rat")
    torch.autograd.set_detect_anomaly(True)

    lora_dim = args.lora_rank

    # get target module name
    target_names = []
    for name, module in model.named_modules():
        print(name)
        if args.target_layer in name:
            target_names.append(name)
    # exit(0)
    # pick the last 3 attention layers to target and add lora layers to that
    # idx = int(len(target_names) - 3)
    # target_names = [target_names[idx], target_names[idx+1], target_names[idx+2]]
    aux_layer_indices = [i for i in range(len(target_names))]

    lora_modules = []
    lora_module_names = []

    # # print("\nPerformance with ground truth rationale:")
    aux_params = None
    # acc_rationale = evaluate_model(model, aux_params, tokenizer, dataset, split="test", include_rationale=True)
    # acc_no_rationale = evaluate_model(model, aux_params, tokenizer, dataset, split="test", include_rationale=False)
    # print(f"Accuracy with rationale: {acc_rationale:.4f}")
    # print(f"Accuracy without rationale: {acc_no_rationale:.4f}")

    # # Log initial results
    # wandb.log({
        # "initial/accuracy_with_rationale": acc_rationale,
        # "initial/accuracy_without_rationale": acc_no_rationale
    # })

    lora_right_shape, lora_left_shape = None, None

    # replace each module with LoRA
    for name in target_names:
        name_struct = name.split(".")
        
        # get target module
        module_list = [model]
        for struct in name_struct:
            module_list.append(getattr(module_list[-1], struct))

        # pick target modules for the input representations
        # for now lets just use the same ones

        # build LoRA
        lora = Predict_LoRA_Linear(
            weight = torch.transpose(module_list[-1].weight, 0, 1),
            bias = module_list[-1].bias,
            lora_dim = lora_dim,
            args=args,
        ).to(device)

        # replace
        module_list[-2].__setattr__(name_struct[-1], lora)

        lora_modules.append(lora)
        lora_module_names.append(name_struct[-1])

        if lora_right_shape is None and lora_left_shape is None:
            lora_right_shape = lora.right_shape
            lora_left_shape = lora.left_shape

    # If args.aux_cutoff_layer is larger than 8 we need 2 gpus
    if torch.cuda.device_count() > 1:
        print("Using 2 GPUs")
        model = model.to("cuda:1")

    # Initialise the Auxiliary Transformer Model
    aux_tokenizer, aux_model = load_tokenizer_and_model("microsoft/phi-1_5", args, cutoff=args.aux_cutoff_layer)
    aux_model = aux_model.to(torch.float32).to(device)
    aux_model = slice_model(aux_model, args.aux_cutoff_layer)
    print(aux_model)
    
    # Define the cutoff for where the model is changed
    head_name = "score"
    out_features = lora_right_shape[0] * lora_right_shape[1] + lora_left_shape[0] * lora_left_shape[1]

    # Replace the head of the auxiliary model
    aux_model = replace_head(aux_model, head_name, out_features)

    # Train LoRA weights
    loss_fn = args.loss_fn
    balanced = bool(args.balanced)
    aux_model, acc_no_rationale = train_aux(model, loss_fn, aux_model, lora_modules, aux_layer_indices, lora_right_shape, lora_left_shape, tokenizer, dataset, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, split="train", balanced=balanced, args=args)
    
    # End timer
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    # # Log final results
    wandb.log({
        "final/accuracy_without_rationale": acc_no_rationale,
        "elapsed_time": elapsed_time,
    })
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":

    args = setup_args()
    
    # For RTX purposes
    args.batch_size = 1
    args.target_layer = "v_proj"

    # print("Arguments: ", args)

    torch.manual_seed(42)

    main_accelerate_cot(args)