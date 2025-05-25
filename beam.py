import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from vllm import LLM, SamplingParams
torch.cuda.empty_cache()

# Part 1: PRM Setup (from your example)
# =====================================
def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


# PRM Model Configuration
prm_model_name = "Qwen/Qwen2.5-Math-PRM-7B"
prm_device = "cuda:1" # Or "cuda" if you have a GPU and want to be specific

print(f"Loading PRM tokenizer: {prm_model_name}")
prm_tokenizer = AutoTokenizer.from_pretrained(prm_model_name, trust_remote_code=True)
print(f"Loading PRM model: {prm_model_name}")


prm_model = AutoModel.from_pretrained(
    prm_model_name,
    device_map=prm_device,
    torch_dtype=torch.bfloat16, # As per your example
    trust_remote_code=True,
).eval()


# Ensure <extra_0> is a single token and get its ID for PRM
if "<extra_0>" not in prm_tokenizer.get_vocab():
    print("Warning: <extra_0> not in PRM tokenizer vocab. PRM scoring might fail.")
    # Fallback: Try encoding it. This might result in multiple IDs if not a single token.
    prm_step_sep_ids = prm_tokenizer.encode("<extra_0>", add_special_tokens=False)
    if len(prm_step_sep_ids) == 1:
        prm_step_sep_id = prm_step_sep_ids[0]
        print(f"Using prm_step_sep_id: {prm_step_sep_id} (from encoding)")
    else:
        print(f"Error: <extra_0> tokenizes to multiple IDs: {prm_step_sep_ids} for PRM. This script expects it to be a single token.")
        exit() # Critical error
else:
    prm_step_sep_id = prm_tokenizer.convert_tokens_to_ids("<extra_0>")
    print(f"Using prm_step_sep_id: {prm_step_sep_id} (from vocab)")


# Part 2: vLLM Generator Setup
# ============================
generator_model_name = "Qwen/Qwen2.5-Math-7B"
print(f"Loading vLLM generator model: {generator_model_name}")
# Adjust tensor_parallel_size, gpu_memory_utilization as needed for your hardware
llm_generator = LLM(
    model=generator_model_name,
    tokenizer=generator_model_name,
    dtype="bfloat16",
    #max_model_len=script_args.max_input_length,
    load_format="auto",
    seed=84,
)

print(f"Loading generator tokenizer: {generator_model_name}")
generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name, trust_remote_code=True)
from datasets import load_dataset
# ds = ds.remove_columns(["prompt"])
instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
def map_fn(example):
    question = example.pop('problem')
    question = question + ' ' + instruction_following
    zz =  [
                                {
                                    "role": "system",
                                    "content": system_prompt
                                },
                                {
                                    "role": "user",
                                    "content": question
                                }
                            ],
    return {
        "prompt": generator_tokenizer.apply_chat_template(
            zz, 
            tokenize=False, add_generation_prompt=True)[0]
    }
#ds = ds.map(map_fn)


# Part 3: Generation Script
# =========================

def generate_with_process_reward(
    initial_system_prompt: str,
    initial_user_query: str,
    num_candidate_responses_per_step: int = 3,
    max_gen_tokens_per_step: int = 200,
    max_iterations: int = 15,
    prm_stop_word: str = "\n\n", # PRM defined stop
    final_answer_token: str = "\\boxed" # Final answer indicator
):
    """
    Generates a response step-by-step, using a Process Reward Model to guide selection.
    """
    
    # --- Initialization ---
    chosen_steps_for_prm_context = [] # Stores clean "step" strings for PRM context
    
    # Prepare initial prompt for the generator model (vLLM)
    # This string will accumulate the full conversation.
    initial_messages_for_template = [
        {"role": "system", "content": initial_system_prompt},
        {"role": "user", "content": initial_user_query},
    ]
    
    # `add_generation_prompt=True` is crucial for the model to start assistant's turn
    current_full_conversation_prompt_str = generator_tokenizer.apply_chat_template(
        initial_messages_for_template,
        tokenize=False,
        add_generation_prompt=True 
    )

    print("Starting generation...\n")
    print(f"System: {initial_system_prompt}")
    print(f"User: {initial_user_query}")
    print("---")

    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}/{max_iterations}")
        
        # --- 1. Generate n candidate next steps using vLLM ---
        # Stop tokens for vLLM: PRM step separator, final answer, or model's own EOT
        stop_sequences = [prm_stop_word]
        if generator_tokenizer.eos_token:
            stop_sequences.append(generator_tokenizer.eos_token)
        # Qwen specific stop tokens, if any, like <|im_end|>
        # (Often handled by chat template, but good for vLLM stop list too)
        if hasattr(generator_tokenizer, 'special_tokens_map') and 'additional_special_tokens' in generator_tokenizer.special_tokens_map:
            if "<|im_end|>" in generator_tokenizer.all_special_tokens:
                 stop_sequences.append("<|im_end|>")



        sampling_params = SamplingParams(
            n=num_candidate_responses_per_step,
            temperature=1.0 if num_candidate_responses_per_step > 1 else 0.0, # Low temp for math, slight exploration if n > 1
            max_tokens=max_gen_tokens_per_step,
            stop=stop_sequences,
            include_stop_str_in_output=True # Important to capture the stop string itself if needed
        )
        
        # `current_full_conversation_prompt_str` contains the whole dialogue so far
        # vLLM will append its generation to this.
        vllm_outputs = llm_generator.generate([current_full_conversation_prompt_str], sampling_params)
        
        # `output.text` is *only the newly generated part* by vLLM
        raw_generated_texts_this_turn = [output.text for output in vllm_outputs[0].outputs]
        
        # print(f"Raw generated texts from vLLM: {raw_generated_texts_this_turn}")

        candidate_segments_for_scoring = [] # These are "clean" segments, without the prm_stop_word
        original_raw_segments_from_vllm = [] # Store raw vLLM output to append later

        has_final_answer_in_candidates = False
        for raw_vllm_text in raw_generated_texts_this_turn:
            original_raw_segments_from_vllm.append(raw_vllm_text)
            
            # Check if final answer token is in the raw text
            if final_answer_token in raw_vllm_text:
                has_final_answer_in_candidates = True
                # The segment for PRM context is the text itself if it has the answer.
                # No need to split by prm_stop_word if final_answer_token is already there.
                candidate_segments_for_scoring.append(raw_vllm_text) # take up to and including boxed
            # Check if PRM stop word is in the raw text
            elif prm_stop_word in raw_vllm_text:
                # Segment for scoring is text before the prm_stop_word
                segment = raw_vllm_text.split(prm_stop_word, 1)[0]
                candidate_segments_for_scoring.append(segment)
            else:
                # Neither stop word found (e.g., max_tokens hit, or ended with EOS)
                candidate_segments_for_scoring.append(raw_vllm_text)
        
        if not candidate_segments_for_scoring:
            print("No valid candidate segments generated by vLLM. Stopping.")
            break

        # --- 2. Score or select best segment ---
        best_segment_for_prm_context: str
        chosen_raw_segment_to_append: str

        if has_final_answer_in_candidates:
            print("Final answer token found in one of the candidates. Selecting it.")
            # Pick the first candidate that contains the final answer token.
            # (A more advanced strategy could be implemented here if multiple candidates have it)
            idx_with_final_answer = -1
            for i, raw_seg in enumerate(original_raw_segments_from_vllm):
                if final_answer_token in raw_seg:
                    idx_with_final_answer = i
                    break
            
            best_segment_for_prm_context = candidate_segments_for_scoring[idx_with_final_answer]
            chosen_raw_segment_to_append = original_raw_segments_from_vllm[idx_with_final_answer]
        else:
            # No final answer yet, use PRM to score intermediate steps
            rewards = []
            print(f"Scoring {len(candidate_segments_for_scoring)} candidate segments with PRM...")
            for segment_to_score in candidate_segments_for_scoring:
                if not segment_to_score.strip(): # Skip empty or whitespace-only segments
                    rewards.append(-float('inf'))
                    continue

                # Construct input for PRM: previous chosen steps + current candidate segment
                prm_steps_for_current_candidate = chosen_steps_for_prm_context + [segment_to_score.strip()]
                # PRM expects steps joined by <extra_0> and ending with <extra_0>
                assistant_content_for_prm = "<extra_0>".join(prm_steps_for_current_candidate) + "<extra_0>"

                prm_messages = [
                    {"role": "system", "content": initial_system_prompt},
                    {"role": "user", "content": initial_user_query},
                    {"role": "assistant", "content": assistant_content_for_prm},
                ]
                
                prm_conversation_str = prm_tokenizer.apply_chat_template(
                    prm_messages,
                    tokenize=False,
                    add_generation_prompt=False # We are evaluating existing/generated content
                )
                
                prm_input_ids = prm_tokenizer.encode(prm_conversation_str, return_tensors="pt").to(prm_model.device)
                
                # Create token masks for <extra_0> positions
                # (bs, seq_len), True where input_id is prm_step_sep_id
                prm_token_masks = (prm_input_ids == prm_step_sep_id)
                prm_outputs = prm_model(input_ids=prm_input_ids)
                
                logits_for_reward = prm_outputs[0] 
                
                # `make_step_rewards` returns a list of lists; for batch size 1, it's [[score1, score2, ...]]
                all_rewards_for_steps = make_step_rewards(logits_for_reward, prm_token_masks)
                current_segment_reward = all_rewards_for_steps[0][-1] # Get reward for the latest step

                rewards.append(current_segment_reward)
            print(rewards)
            
            if not rewards or all(r == -float('inf') for r in rewards) :
                print("No valid rewards obtained or all candidates are invalid. Taking the first non-empty candidate if available.")
                # Fallback: take the first non-empty generated segment if any.
                first_valid_idx = -1
                for idx_fb, seg_text_fb in enumerate(candidate_segments_for_scoring):
                    if seg_text_fb.strip():
                        first_valid_idx = idx_fb
                        break
                if first_valid_idx != -1:
                    best_segment_idx = first_valid_idx
                else:
                    print("All candidates are empty or invalid. Stopping generation.")
                    break # Stop the main iteration loop
            else:
                best_segment_idx = torch.argmax(torch.tensor(rewards, dtype=torch.float)).item()
                print(f"  Best segment index: {best_segment_idx} (Reward: {rewards[best_segment_idx]:.4f})")

            best_segment_for_prm_context = candidate_segments_for_scoring[best_segment_idx]
            chosen_raw_segment_to_append = original_raw_segments_from_vllm[best_segment_idx]

        # --- 4. Update state with the chosen segment ---
        # Append the chosen RAW segment (which might include \n\n or \boxed{}) to the full conversation string for vLLM
        current_full_conversation_prompt_str += chosen_raw_segment_to_append
        
        # Append the "clean" best segment to our list of chosen steps for PRM context
        chosen_steps_for_prm_context.append(best_segment_for_prm_context.strip()) # Store clean version

        print(f"Chosen segment (raw): ----\n{chosen_raw_segment_to_append}\n----")
        # print(f"Updated chosen steps for PRM context: {chosen_steps_for_prm_context}")


        # --- 5. Check for termination ---
        if final_answer_token in chosen_raw_segment_to_append:
            print(f"\nFinal answer token '{final_answer_token}' found in the chosen segment. Generation complete.")
            break
        # Safety break if chosen segment is empty and loop continues (should be caught earlier)
        if not chosen_raw_segment_to_append.strip() and not has_final_answer_in_candidates:
            print("\nChosen segment is empty, stopping to prevent infinite loop.")
            break

    else: # Loop finished due to max_iterations
        print(f"\nReached maximum iterations ({max_iterations}). Generation stopped.")

    # --- Extract final assistant response ---
    # The `current_full_conversation_prompt_str` contains the entire dialogue.
    # We need to extract the assistant's full response part.
    # This can be done by finding where the initial system+user prompt ended.
    
    initial_prompt_for_assistant_turn = generator_tokenizer.apply_chat_template(
        initial_messages_for_template,
        tokenize=False,
        add_generation_prompt=True 
    )
    
    final_assistant_response = current_full_conversation_prompt_str
    if current_full_conversation_prompt_str.startswith(initial_prompt_for_assistant_turn):
         final_assistant_response = current_full_conversation_prompt_str[len(initial_prompt_for_assistant_turn):]
    else:
        # Fallback: Try to find a generic assistant marker if the above fails (e.g. due to minor whitespace diffs)
        # This is brittle; depends heavily on the chat template structure.
        assistant_markers = ["ASSISTANT:", "Assistant:", "<|im_start|>assistant\n"] # Add more if needed
        found_marker_at = -1
        for marker in assistant_markers:
            # Find the *last* occurrence in the initial part that should be the assistant prompt
            # This logic is to find the start of the assistant's *actual generated text*.
            # The `initial_prompt_for_assistant_turn` already contains the "ASSISTANT:" part.
            # So `final_assistant_response` should be the text *after* this initial prompt.
             pass # The above direct substring removal is preferred.

    print("\n--- Final Generated Assistant Response ---")
    print(final_assistant_response.strip())
    print("----------------------------------------")
    
    return final_assistant_response.strip(), chosen_steps_for_prm_context


if __name__ == "__main__":
    # Example Usage:
    # Use the problem from your PRM example
    ds = load_dataset('weqweasdas/remain2', split="train")

    system_message = "Please reason step by step, and put your final answer within \\boxed{}."
    '''
    user_query = (
        "What is the value of 1 * 8 - 2 * 0 - 7 * 5 - 2?"
        #$"On Friday morning, the neighbors placed 18 pink plastic flamingos out on Sue's front yard. "
        #"On Saturday morning, the neighbors took back one third of the flamingos, painted them white, "
        #"and put these newly painted white flamingos back out on Sue's front yard. Then, on Sunday morning, "
        #"they added another 18 pink plastic flamingos to the collection. At noon on Sunday, how many "
        #"more pink plastic flamingos were out than white plastic flamingos?"
    )
    '''
    import json
    all_data = []
    ds = ds.select(range(1, len(ds)))
    for sample in ds:
        all_responses = []
        user_query = sample['prompt'].split('<|im_start|>user\n')[1].split('<|im_start|>assistant\n')[0].strip()
        for i in range(64):
            generated_solution, solution_steps = generate_with_process_reward(
                initial_system_prompt=system_message,
                initial_user_query=user_query,
                num_candidate_responses_per_step=3, # Generate 2 candidates per step
                max_gen_tokens_per_step=512,      # Max tokens for a single reasoning step
                max_iterations=30                 # Max number of reasoning steps
            )
            all_responses.append("".join(solution_steps))
        sample.update({"responses": all_responses})
        all_data.append(sample)
    
    
        with open("all_data.json", "w") as f:
            json.dump(all_data, f, indent=4)

