import os
import sys

sys.path.append("../")
import argparse
import time
from contextlib import contextmanager

import draftretriever
import numpy as np
import torch
from dataset import HumanEvalDataset
from tqdm import tqdm
from transformers import AutoTokenizer

from rest.model.kv_cache import *
from rest.model.rest_model import RestModel
from rest.model.utils import *


def run_eval(model, tokenizer, datastore, dataset, max_token_span, num_draft, temperature, top_p, max_new_token):
    """Run evaluation with a real model."""

    accept_lengths_tree_average = []
    avg_time_per_token_list = []

    accept_lengths_tree_average_micro = []
    avg_time_per_token_list_micro = []
    token_spans = list(range(2, max_token_span + 1))[::-1]
    print("token_spans: ", token_spans)

    for sample in tqdm(dataset, total=len(dataset)):
        prompt = sample['prompt']

        accept_lengths_tree = []
        with torch.inference_mode():

            # Initialize the past key and value states
            if hasattr(model, "past_key_values"):
                past_key_values = model.past_key_values
                past_key_values_data = model.past_key_values_data
                current_length_data = model.current_length_data
                # Reset the past key and value states
                current_length_data.zero_()
            else:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(model.base_model)
                model.past_key_values = past_key_values
                model.past_key_values_data = past_key_values_data
                model.current_length_data = current_length_data


            new_token = 0
            input_ids = tokenizer([prompt]).input_ids
            input_len = len(input_ids[0])
            input_ids = torch.as_tensor(input_ids).cuda()
            model.base_model.model.draft_mask = None
            logits = initialize_logits(
                    input_ids, model, past_key_values
            )
            cur_length = input_len + 1
            accept_lengths_tree.append(1)
            
            torch.cuda.synchronize()
            start_time = time.time()
            for i in range(2000):
                candidates, tree_candidates, draft_buffers = generate_candidates_and_draft_buffer(
                        logits,
                        input_ids,
                        datastore,
                        token_spans,
                        top_p,
                        temperature,
                        max_num_draft=num_draft,
                        device=model.base_model.device
                    )
                
                model.base_model.model.draft_mask = draft_buffers["draft_attn_mask"]

                logits, outputs = tree_decoding(
                        model,
                        tree_candidates,
                        past_key_values,
                        draft_buffers["draft_position_ids"],
                        input_ids,
                        draft_buffers["retrieve_indices"],
                    )

                best_candidate, accept_length = evaluate_posterior(
                        logits, candidates, temperature = temperature, top_p=top_p
                    )
                input_ids, logits, new_token = update_inference_inputs(
                        input_ids,
                        candidates,
                        best_candidate,
                        accept_length,
                        draft_buffers["retrieve_indices"],
                        outputs,
                        logits,
                        new_token,
                        past_key_values_data,
                        current_length_data,
                    )
                
                accept_length_tree = input_ids.shape[1] - cur_length
                cur_length = accept_length_tree + cur_length
                accept_lengths_tree.append(accept_length_tree)
                if model.tokenizer.eos_token_id in input_ids[0, input_len:] or new_token > max_new_token:
                    break

            torch.cuda.synchronize()
            total_time = time.time() - start_time
            avg_time_per_token = total_time / (new_token.cpu())
            avg_time_per_token_list.append(avg_time_per_token)
            avg_time_per_token_list_micro.append((total_time, new_token.cpu()))
            
            accept_lengths_tree_average.append(np.mean(accept_lengths_tree))
            accept_lengths_tree_average_micro.extend(accept_lengths_tree)

    print("accept_lengths_tree_average: ", np.mean(accept_lengths_tree_average))
    print("accept_lengths_tree_average_micro: ", np.mean(accept_lengths_tree_average_micro))
    print("avg_time_per_token: ", np.mean(avg_time_per_token_list))
    print("avg_time_per_token_micro: ", np.sum([item[0] for item in avg_time_per_token_list_micro]) / np.sum([item[1] for item in avg_time_per_token_list_micro]))
    print("*"*30)
    print()


def run_eval_simulation(model, tokenizer, datastore, dataset, max_token_span, num_draft, temperature, top_p, max_new_token):
    """Run evaluation in simulation mode."""
    accept_lengths_tree_average = []
    avg_time_per_token_list = []

    accept_lengths_tree_average_micro = []
    avg_time_per_token_list_micro = []
    token_spans = list(range(2, max_token_span + 1))[::-1]
    print("token_spans: ", token_spans)

    for sample in tqdm(dataset, total=len(dataset)):
        prompt = sample['prompt']
        solution = sample['canonical_solution']
        solution_ids = tokenizer.encode(solution, add_special_tokens=False)

        accept_lengths_tree = []
        with torch.inference_mode():
            if hasattr(model, "past_key_values"):
                past_key_values = model.past_key_values
                past_key_values_data = model.past_key_values_data
                current_length_data = model.current_length_data
                current_length_data.zero_()
            else:
                (
                    past_key_values,
                    past_key_values_data,
                    current_length_data,
                ) = initialize_past_key_values(model.base_model)
                model.past_key_values = past_key_values
                model.past_key_values_data = past_key_values_data
                model.current_length_data = current_length_data

            new_token = 0
            input_ids = tokenizer([prompt]).input_ids
            input_len = len(input_ids[0])
            input_ids = torch.as_tensor(input_ids).cuda()
            model.base_model.model.draft_mask = None
            logits = initialize_logits(
                    input_ids, model, past_key_values
            )
            cur_length = input_len + 1
            accept_lengths_tree.append(1)
            
            torch.cuda.synchronize()
            start_time = time.time()
            for i in range(2000):
                candidates, tree_candidates, draft_buffers = generate_candidates_and_draft_buffer(
                        logits,
                        input_ids,
                        datastore,
                        token_spans,
                        top_p,
                        temperature,
                        max_num_draft=num_draft,
                        device=model.base_model.device
                    )
                
                # Simulate logits
                # Create a tensor of -inf
                simulated_logits = torch.full((1, candidates.shape[1], tokenizer.vocab_size), -float('Inf'), device=model.base_model.device)
                # Set the probability of the ground truth next token to be 1.
                if new_token < len(solution_ids):
                    next_token_id = solution_ids[new_token]
                    # Give high probability to the correct next token for all candidates
                    simulated_logits[:, :, next_token_id] = 1.0
                else:
                    # If we are past the solution length, just use EOS
                    simulated_logits[:, :, tokenizer.eos_token_id] = 1.0
                
                # The rest of the logic uses these simulated logits
                logits = simulated_logits
                outputs = None # Not used in this path

                best_candidate, accept_length = evaluate_posterior(
                        logits, candidates, temperature = temperature, top_p=top_p
                    )
                input_ids, logits, new_token = update_inference_inputs(
                        input_ids,
                        candidates,
                        best_candidate,
                        accept_length,
                        draft_buffers["retrieve_indices"],
                        outputs, # Should be None
                        logits,
                        new_token,
                        past_key_values_data,
                        current_length_data,
                    )
                accept_lengths_tree.append(accept_length.item())
                if new_token > max_new_token:
                    break
            torch.cuda.synchronize()
            end_time = time.time()
            
            accept_lengths_tree_average.append(np.mean(accept_lengths_tree))
            avg_time_per_token_list.append( (end_time - start_time) / new_token )

            accept_lengths_tree_average_micro.extend(accept_lengths_tree)
            avg_time_per_token_list_micro.append( (end_time - start_time) / new_token )

    print(f"Tokens per second: {1.0 / np.mean(avg_time_per_token_list)}\nAverage accept length: {np.mean(accept_lengths_tree_average)}\nAverage accept length (micro): {np.mean(accept_lengths_tree_average_micro)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulation", action="store_true", help="Run in simulation mode")
    parser.add_argument(
        "--model-path",
        type=str,
        default="codellama/CodeLlama-7b-instruct-hf",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./HumanEval.jsonl.gz",
        help="The path to the HumanEval dataset",
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=512,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for sampling.",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="The threshold for nucleus sampling.",
    )

    # REST's hyperparameters
    parser.add_argument(
        "--datastore-path",
        type=str,
        required=True,
        help="The path of the datastore for retrival.",
    )

    parser.add_argument(
        "--num-draft",
        type=int,
        default=64,
        help="The number of draft tokens.",
    )
    parser.add_argument(
        "--max-token-span",
        type=int,
        default=16,
        help="The maximum length of suffix for retrieval.",
    )

    args = parser.parse_args()

    if args.temperature == 0:
        args.top_p = 0
        
    print(args)

    if args.simulation:
        # In simulation mode, we don't need to load the full model.
        # Create a mock model object with necessary configs.
        class MockModelConfig:
            def __init__(self):
                self.hidden_size = 4096
                self.num_hidden_layers = 32
                self.num_attention_heads = 32
                self.num_key_value_heads = 32
                self.max_position_embeddings = 4096 * 2

        class MockBaseModel:
            def __init__(self):
                self.config = MockModelConfig()
                self.device = torch.device("cuda")
                self.model = self # for self.base_model.model.draft_mask
                self.draft_mask = None
                self.dtype = torch.float16

        class MockRestModel:
            def __init__(self, tokenizer):
                self.base_model = MockBaseModel()
                self.tokenizer = tokenizer

            def __call__(self, *args, **kwargs):
                return self.forward(*args, **kwargs)

            def forward(self, input_ids, past_key_values=None, use_cache=True, **kwargs):
                # This forward pass is only for the initial logit calculation.
                # It just needs to return tensors of the correct shape.
                batch_size, seq_len = input_ids.shape
                vocab_size = self.tokenizer.vocab_size
                mock_logits = torch.randn(
                    (batch_size, seq_len, vocab_size),
                    device=self.base_model.device,
                    dtype=self.base_model.dtype
                )
                # The first element of the tuple is 'outputs', which is not used in this context.
                return (None, mock_logits)

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = MockRestModel(tokenizer)
    else:
        model = RestModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map='auto',
        )
        model.base_model.model.config.num_draft = args.num_draft
        model.base_model.model.config.max_token_span = args.max_token_span
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.pad_token = tokenizer.eos_token

    print("loading the datastore ...")
    datastore = draftretriever.Reader(
                index_file_path=args.datastore_path,
            )
    print("datastore loaded!")
    
    dataset = HumanEvalDataset(args.dataset_path)

    if args.simulation:
        print("Running in simulation mode")
        run_eval_simulation(model, tokenizer, datastore, dataset, args.max_token_span, args.num_draft, args.temperature, args.top_p, args.max_new_token)
    else:
        run_eval(model, tokenizer, datastore, dataset, args.max_token_span, args.num_draft, args.temperature, args.top_p, args.max_new_token)