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
        if json_output_path:
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
    if json_output_path and json_data:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(json_output_path)
        if output_dir:  # Only create directory if path contains a directory
            os.makedirs(output_dir, exist_ok=True)
        
        # Create unique filename using timestamp
        base_path = os.path.splitext(json_output_path)[0]  # Remove extension if present
        extension = os.path.splitext(json_output_path)[1] or ".json"  # Get extension or default to .json
        
        # Generate unique filename with timestamp
        unique_filename = f"{base_path}_{current_time}{extension}"
        
        # Write JSON data to independent file
        with open(unique_filename, "w") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    return rewards


# def format_reward(completions, **kwargs):
#     """Reward function that checks if the completion has a specific format."""
#     pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
#     completion_contents = [completion[0]["content"] for completion in completions]
#     matches = [re.match(pattern, content) for content in completion_contents]
#     return [1.0 if match else 0.0 for match in matches]