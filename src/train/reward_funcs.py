import re
from datetime import datetime
import os
import json


def accuracy_reward(completions, assistant, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching.
    
    Args:
        completions: List of completion dictionaries, each containing a list with role/content dict
        assistant: List of assistant dictionaries (ground truth solutions), each with role/content
        **kwargs: Additional arguments including 'prompts' (list of prompt dictionaries)
    
    Returns:
        List of reward values (1.0 for correct, -1.0 for incorrect)
    """
    contents = [completion[0]["content"] for completion in completions]
    solution = [a['content'] for a in assistant]
    
    # Extract prompts from kwargs if available (passed from trainer)
    prompts = kwargs.get('prompts', [])
    
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    # Prepare data for JSON storage
    json_output_path = os.getenv("REWARD_JSON_OUTPUT_PATH", None)
    json_data = []
    
    for idx, (content, sol) in enumerate(zip(contents, solution)):
        reward = -1.0  # Failure
        # Try symbolic verification first
        try:
            content = content.strip()
            sol = sol.strip()
            if sol in content:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        rewards.append(reward)
        
        # Store data for JSON output
        if json_output_path :
            entry = {
                "timestamp": current_time,
                "reward": reward,
                "prompt": prompts[idx] if idx < len(prompts) else None,
                "completion": completions[idx] if idx < len(completions) else None,
                "solution": assistant[idx] if idx < len(assistant) else None,
            }
            json_data.append(entry)
        
        if os.getenv("DEBUG_MODE") and os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "reward_funcs.log")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    
    # Write JSON data to independent files (one file per batch)
    # if json_output_path and json_data:
    #     # Create directory if it doesn't exist
    #     output_dir = os.path.dirname(json_output_path)
    #     if output_dir:  # Only create directory if path contains a directory
    #         os.makedirs(output_dir, exist_ok=True)
        
    #     # Create unique filename using timestamp
    #     base_path = os.path.splitext(json_output_path)[0]  # Remove extension if present
    #     extension = os.path.splitext(json_output_path)[1] or ".json"  # Get extension or default to .json
        
    #     # Generate unique filename with timestamp
    #     unique_filename = f"{base_path}_{current_time}{extension}"
        
    #     # Write JSON data to independent file
    #     with open(unique_filename, "w") as f:
    #         json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    return rewards


import re

def format_reward(completions, assistant, gamma=0.75, **kwargs):
    """
    Rewards the model for strictly following the format:
    <think>...</think><code>...</code>
    """
    rewards = []
    
    # Regex to capture the strict structure: 
    # 1. Starts with <think>
    # 2. Non-greedy match for reasoning content
    # 3. Closes with </think>
    # 4. Immediately follows with <code>
    # 5. Non-greedy match for code content
    # 6. Closes with </code>
    pattern = r"^<think>(.*?)</think><code>(.*?)</code>$"
    completions = [completion[0]["content"] for completion in completions]
    for completion in completions:
        # Strip whitespace ensuring we only check the core content
        content = completion.strip()
        
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            # Perfect format match
            rewards.append(1.0)
        else:
            # Partial credit for being "close" (optional, but helps training stability)
            score = 0.0
            if "<think>" in content: score += 0.2
            if "</think>" in content: score += 0.2
            if "<code>" in content: score += 0.2
            if "</code>" in content: score += 0.2
            
            # Penalize heavily if tags are out of order or nested incorrectly
            if content.find("<code>") < content.find("</think>"):
                score = 0.0
                
            rewards.append(score)
            
    rewards = [reward * gamma for reward in rewards]
    return rewards

def language_consistency_reward(completions, assistant, gamma=0.5, **kwargs):
    """
    Rewards the model based on the proportion of English words/characters.
    """
    rewards = []
    
    # Regex for standard English words (allowing for punctuation/numbers)
    # This filters out CJK characters, Cyrillic, etc.
    english_pattern = re.compile(r'[a-zA-Z0-9\s\.,!?;:()\[\]"\'-]+')
    completions = [completion[0]["content"] for completion in completions]
    for completion in completions:
        if not completion:
            rewards.append(0.0)
            continue
            
        # Calculate total length of the completion
        total_len = len(completion)
        
        # Find all substrings that match English logic
        matches = english_pattern.findall(completion)
        english_len = sum(len(m) for m in matches)
        
        # Calculate ratio: English chars / Total chars
        # DeepSeek paper notes a 'consistency' score. 
        # If > 95% is English, we give full reward 1.0 to encourage purity.
        ratio = english_len / total_len if total_len > 0 else 0
        
        if ratio >= 0.95:
            rewards.append(1.0)
        else:
            # Scale down linearly if it mixes languages
            rewards.append(ratio)
    rewards = [reward * gamma for reward in rewards]
    return rewards


import re

def icd10_f1_reward(completions, assistant, **kwargs):
    """
    Parses multiple ICD-10 codes from the model output and calculates 
    the F1 score against the ground truth set.
    """
    rewards = []
    
    # Regex to capture individual ICD-10 codes within a string
    # Matches: One letter, two digits, optional dot, 1-4 digits
    code_pattern = r"\b[A-Z][0-9]{2}(?:\.[0-9]{1,4})?\b"
    completions = [completion[0]["content"] for completion in completions]

    ground_truth_codes = [a['content'] for a in assistant]
    
    for completion, expected_codes in zip(completions, ground_truth_codes):
        # 1. Extract content inside <code> tags
        content_match = re.search(r"<code>(.*?)</code>", completion.strip(), re.DOTALL)
        
        if not content_match:
            rewards.append(0.0)
            continue
            
        content = content_match.group(1).strip()
        
        # 2. Parse sets of codes
        # We normalize to uppercase to ensure case-insensitive matching
        predicted_set = set(re.findall(code_pattern, content.upper()))
        
        # Handle ground truth format (ensure it's a set)
        if isinstance(expected_codes, str):
            # If ground truth is a string "A01.0, B02.1", parse it
            truth_set = set(re.findall(code_pattern, expected_codes.upper()))
        else:
            # If it's already a list ['A01.0', 'B02.1']
            truth_set = set(c.upper() for c in expected_codes)
            
        # 3. Calculate F1 Components
        if not predicted_set or not truth_set:
            # If prediction is empty but truth is not, or vice versa -> F1 is 0
            # If both are empty (rare), technically accuracy is 1.0, but usually implies error here.
            rewards.append(1.0 if predicted_set == truth_set else 0.0)
            continue

        true_positives = len(predicted_set.intersection(truth_set))
        false_positives = len(predicted_set - truth_set)
        false_negatives = len(truth_set - predicted_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        
        # 4. Calculate F1 Score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
            
        # Optional: Shaping reward
        # If F1 is 0 but the model output VALID ICD-10 formats (even if wrong diagnosis),
        # give a tiny reward (0.05) to encourage outputting codes at all.
        if f1 == 0.0 and len(predicted_set) > 0:
            rewards.append(0.05)
        else:
            rewards.append(f1)
            
    return rewards
# def format_reward(completions, **kwargs):
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
#     completion_contents = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, content) for content in completion_contents]
#     return [1.0 if match else 0.0 for match in matches]