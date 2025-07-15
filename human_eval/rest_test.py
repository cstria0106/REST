import os
import sys

sys.path.append("../")
import argparse
import time

import draftretriever
import numpy as np
import torch
from dataset import HumanEvalDataset
from tqdm import tqdm
from transformers import AutoTokenizer  # <-- 추가됨

from rest.model.utils import generate_candidates_and_draft_buffer

# ANSI color codes for visualization
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
ENDC = '\033[0m'

def simulate_verification(candidates, ground_truth_ids):
    """
    드래프트 후보들을 정답 시퀀스와 비교하여 검증 단계를 시뮬레이션합니다.

    Args:
        candidates (torch.Tensor): [num_candidates, seq_len] 모양의 드래프트 토큰 시퀀스 텐서.
        ground_truth_ids (torch.Tensor): 정답 토큰 ID를 담고 있는 1D 텐서.

    Returns:
        tuple[torch.Tensor, int]: 가장 잘 일치하는 후보와 그 후보로부터 수락된 토큰의 수를 반환합니다.
    """
    if ground_truth_ids.numel() == 0:
        return candidates[0], 0, 0

    best_match_len = 0
    best_candidate_idx = 0

    # 각 후보에 대해 정답 시퀀스와 얼마나 일치하는지 확인합니다.
    for i, candidate in enumerate(candidates):
        current_match_len = 0
        # 후보와 정답의 최소 길이만큼 비교를 수행합니다.
        for j in range(min(len(candidate), len(ground_truth_ids))):
            if candidate[j] == ground_truth_ids[j]:
                current_match_len += 1
            else:
                break
        
        if current_match_len > best_match_len:
            best_match_len = current_match_len
            best_candidate_idx = i

    # 어떤 후보도 일치하지 않는 경우, 1개의 토큰만 수락된 것으로 간주합니다.
    if best_match_len == 0 and len(candidates) > 0:
        return candidates[best_candidate_idx], 1, best_candidate_idx

    return candidates[best_candidate_idx], best_match_len, best_candidate_idx


def print_verification_step(step, input_ids, candidates, best_candidate_idx, accepted_tokens, accept_length, tokenizer):
    """Print the current verification step with color formatting."""
    os.system('clear')
    print(f"{'*'*20} Step {step} {'*'*20}")
    current_context_str = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"{BLUE}Current Context:{ENDC}\n{current_context_str}")

    print(f"\n{YELLOW}Draft Candidates:{ENDC}")
    for i, candidate in enumerate(candidates):
        decoded_candidate = tokenizer.decode(candidate, skip_special_tokens=True)
        if i == best_candidate_idx:
            print(f"  -> {i}: {GREEN}{decoded_candidate}{ENDC} (Best)")
        else:
            print(f"   - {i}: {decoded_candidate}")
    
    decoded_accepted = tokenizer.decode(accepted_tokens, skip_special_tokens=True)
    print(f"\n{BLUE}Verification Result:{ENDC}")
    print(f"  - Accepted Tokens (length {accept_length}): {GREEN}{decoded_accepted}{ENDC}")
    time.sleep(0.2)

def print_simulation_results(accept_lengths_tree_average, accept_lengths_tree_average_micro, 
                           avg_time_per_token_list, avg_time_per_token_list_micro):
    """Print the final simulation results."""
    print("\n--- 시뮬레이션 결과 ---")
    print(f"평균 수락 길이 (시퀀스 단위): {np.mean(accept_lengths_tree_average):.2f}")
    print(f"평균 수락 길이 (마이크로): {np.mean(accept_lengths_tree_average_micro):.2f}")
    
    if avg_time_per_token_list:
        print(f"토큰 당 평균 시간: {np.mean(avg_time_per_token_list):.4f}s")
    
    total_time_micro = np.sum([item[0] for item in avg_time_per_token_list_micro])
    total_tokens_micro = np.sum([item[1] for item in avg_time_per_token_list_micro])
    if total_tokens_micro > 0:
        print(f"토큰 당 평균 시간 (마이크로): {total_time_micro / total_tokens_micro:.4f}s")
    print("*" * 30 + "\n")

def run_simulation(tokenizer, datastore, max_token_span, num_draft, max_new_token, dataset, visualize):
    """
    미리 정의된 정답을 사용하여 투기적 디코딩을 시뮬레이션합니다.
    LLM 호출은 정답 시퀀스와의 비교로 대체됩니다.
    
    Args:
        tokenizer: Hugging Face 토크나이저
        datastore: 드래프트 생성을 위한 데이터스토어
        max_token_span: 최대 토큰 스팬
        num_draft: 생성할 드래프트 후보 수
        max_new_token: 생성할 최대 새 토큰 수
        dataset: 시뮬레이션에 사용할 데이터셋
        visualize: 시각화 기능 활성화 여부
    """
    accept_lengths_tree_average = []
    avg_time_per_token_list = []
    accept_lengths_tree_average_micro = []
    avg_time_per_token_list_micro = []
    token_spans = list(range(2, max_token_span + 1))[::-1]
    
    if visualize:
        print("token_spans: ", token_spans)
    
    vocab_size = tokenizer.vocab_size

    for sample in tqdm(dataset, total=len(dataset)):
        prompt = sample['prompt']
        # 정답 시퀀스를 'canonical_solution' 필드에서 가져옵니다.
        solution = sample['canonical_solution']
        
        input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
        # 매칭에 사용할 정답 토큰 ID (프롬프트 부분 제외)
        solution_ids = tokenizer.encode(solution, return_tensors='pt')[0].cuda()

        total_new_tokens = 0
        accept_lengths_tree = [1] # 첫 토큰은 항상 '수락'
        
        torch.cuda.synchronize()
        start_time = time.time()

        while total_new_tokens < max_new_token:
            if len(solution_ids) == 0:
                break
            
            # 1. 다음 정답 토큰을 강제로 선택하도록 가짜 로짓(fake logits)을 생성합니다.
            next_gt_token_id = solution_ids[0]
            fake_logits = torch.full((1, 1, vocab_size), -float('inf'), device='cuda', dtype=torch.float16)
            fake_logits[0, 0, next_gt_token_id] = 0

            # 2. 데이터스토어에서 드래프트 후보들을 생성합니다.
            candidates, _, _ = generate_candidates_and_draft_buffer(
                logits=fake_logits,
                input_ids=input_ids,
                datastore=datastore,
                token_spans=token_spans,
                top_p=0,
                temperature=0,
                max_num_draft=num_draft,
                device='cuda'
            )

            if candidates.shape[0] == 0: # 생성된 후보가 없는 경우
                accept_length = 1
                accepted_tokens = solution_ids[0:1].clone().detach()
                best_candidate_idx = -1
            else:
                # 3. 정답과 비교하여 검증을 시뮬레이션하고 수락 길이를 결정합니다.
                best_candidate, accept_length, best_candidate_idx = simulate_verification(candidates, solution_ids)
                accepted_tokens = best_candidate[:accept_length]

            # 시각화 기능
            if visualize:
                print_verification_step(
                    step=len(accept_lengths_tree),
                    input_ids=input_ids,
                    candidates=candidates,
                    best_candidate_idx=best_candidate_idx,
                    accepted_tokens=accepted_tokens,
                    accept_length=accept_length,
                    tokenizer=tokenizer
                )

            # 4. 다음 이터레이션을 위해 상태를 업데이트합니다.
            input_ids = torch.cat([input_ids, accepted_tokens.unsqueeze(0)], dim=1)
            solution_ids = solution_ids[accept_length:] # 남은 정답 시퀀스 업데이트
            total_new_tokens += accept_length
            accept_lengths_tree.append(accept_length)

            if tokenizer.eos_token_id in accepted_tokens:
                break
        
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        
        if total_new_tokens > 0:
            avg_time_per_token = total_time / total_new_tokens
            avg_time_per_token_list.append(avg_time_per_token)
            avg_time_per_token_list_micro.append((total_time, total_new_tokens))

        accept_lengths_tree_average.append(np.mean(accept_lengths_tree))
        accept_lengths_tree_average_micro.extend(accept_lengths_tree)

    # 최종 결과 출력 
    print_simulation_results(
        accept_lengths_tree_average,
        accept_lengths_tree_average_micro,
        avg_time_per_token_list,
        avg_time_per_token_list_micro
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 수정됨: model-path를 tokenizer-path로 변경
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="codellama/CodeLlama-7b-instruct-hf",
        help="토크나이저의 경로. 로컬 폴더 또는 Hugging Face 저장소 ID.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="./HumanEval.jsonl.gz",
        help="HumanEval 데이터셋 경로.",
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=512,
        help="생성할 최대 새 토큰 수.",
    )
    parser.add_argument(
        "--visualize",
        type=bool,
        default=False,
        help="시뮬레이션 결과 시각화 여부.",
    )
    
    # REST 하이퍼파라미터 (드래프트 생성에 여전히 필요)
    parser.add_argument(
        "--datastore-path",
        type=str,
        required=True,
        help="검색을 위한 데이터스토어 경로.",
    )
    parser.add_argument(
        "--num-draft",
        type=int,
        default=64,
        help="드래프트 토큰의 수.",
    )
    parser.add_argument(
        "--max-token-span",
        type=int,
        default=16,
        help="검색을 위한 최대 접미사 길이.",
    )

    args = parser.parse_args()
    print(args)

    # 추가됨: 모델 대신 토크나이저 직접 로드
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = HumanEvalDataset(args.dataset_path)

    print("데이터스토어 로딩 중...")
    datastore = draftretriever.Reader(
        index_file_path=args.datastore_path,
    )
    print("데이터스토어 로딩 완료!")
    
    # 수정됨: 새로운 시뮬레이션 함수 호출
    run_simulation(
        tokenizer,
        datastore,
        args.max_token_span,
        args.num_draft,
        args.max_new_token,
        dataset,
        args.visualize
    )