import os
import re
import torch
import pandas as pd
from typing import Optional
from torch.utils.data import Dataset, DataLoader
from transformers import GenerationConfig
from src.constants import SYSTEM_MESSAGE, DEFAULT_END_TOKEN


class ICD10CMDataset(Dataset):
    """Dataset for ICD-10-CM evaluation"""
    
    def __init__(self, csv_path: str, max_samples: Optional[int] = None):
        df = pd.read_csv(csv_path)
        if max_samples is not None:
            df = df.head(max_samples)
        self.codes = df['code'].tolist()
        self.texts = df['text'].tolist()
    
    def __len__(self):
        return len(self.codes)
    
    def __getitem__(self, idx):
        return {
            'code': self.codes[idx],
            'text': self.texts[idx]
        }


def extract_code_from_output(output: str) -> Optional[str]:
    """Extract ICD-10-CM code from model output, looking for <code>...</code> tags or direct code patterns"""
    # First try to extract from <code> tags
    code_match = re.search(r'<code>(.*?)</code>', output, re.IGNORECASE | re.DOTALL)
    if code_match:
        code = code_match.group(1).strip()
        # Clean up any extra whitespace or newlines but preserve dots
        code = re.sub(r'\s+', '', code)
        # Remove any non-alphanumeric characters except dots
        code = re.sub(r'[^A-Z0-9.]', '', code.upper())
        if code:
            return code
    
    # If no code tags, try to find ICD-10-CM pattern (e.g., A00.0, A01.00, etc.)
    # ICD-10-CM codes typically start with a letter followed by digits and optional dots
    # Pattern: Letter + 2-3 digits + optional dot + 1-2 digits
    code_pattern = r'\b([A-Z]\d{2,3}(?:\.\d{1,2})?)\b'
    matches = re.findall(code_pattern, output.upper())
    if matches:
        # Return the first match that looks valid
        for match in matches:
            # Basic validation: should have at least 3 characters (letter + 2 digits)
            if len(match) >= 3:
                return match
    
    return None


def evaluate_icd10cm_accuracy(
    trainer,
    csv_path: str = "data/all_icd10cm_codes.csv",
    max_samples: Optional[int] = None,
    batch_size: int = 32,
    max_new_tokens: int = 128,
    num_return_sequences: int = 1,
    temperature: float = 0.0,  # Use deterministic generation for evaluation
) -> dict:
    """
    Evaluate model accuracy on ICD-10-CM codes dataset.
    
    Args:
        trainer: The GRPO trainer instance
        csv_path: Path to the CSV file with ICD-10-CM codes
        max_samples: Maximum number of samples to evaluate (None for all)
        batch_size: Batch size for evaluation
        max_new_tokens: Maximum tokens to generate
        num_return_sequences: Number of sequences to generate per prompt (for pass@k)
        temperature: Temperature for generation (0.0 for deterministic)
    
    Returns:
        Dict with 'accuracy' (pass@1 from first sequence) and 'pass_at_k' (any correct in k sequences)
    """
    if not os.path.exists(csv_path):
        if trainer.accelerator.is_main_process:
            print(f"Warning: Evaluation CSV not found at {csv_path}, skipping evaluation")
        return {"accuracy": 0.0, "pass_at_k": 0.0}
    
    # Create dataset
    eval_dataset = ICD10CMDataset(csv_path, max_samples=max_samples)
    
    # Use DistributedSampler for proper data distribution across GPUs
    from torch.utils.data.distributed import DistributedSampler
    sampler = DistributedSampler(
        eval_dataset,
        num_replicas=trainer.accelerator.num_processes,
        rank=trainer.accelerator.process_index,
        shuffle=False,
    )
    
    # Create dataloader with proper distribution
    # batch_size should be per-device, accelerator will handle the rest
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True,
        drop_last=False,
    )
    eval_dataloader = trainer.accelerator.prepare(eval_dataloader)
    
    # Prepare model for evaluation
    model = trainer.model
    model.eval()
    processor = trainer.processing_class
    
    # Get end_of_turn token ID for Gemma-3 models
    end_of_turn_token_id = None
    if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'convert_tokens_to_ids'):
        try:
            end_of_turn_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_END_TOKEN)
            if end_of_turn_token_id == processor.tokenizer.unk_token_id:
                if DEFAULT_END_TOKEN in processor.tokenizer.get_vocab():
                    end_of_turn_token_id = processor.tokenizer.get_vocab()[DEFAULT_END_TOKEN]
                else:
                    end_of_turn_token_id = None
        except Exception:
            end_of_turn_token_id = None
    
    # Generation config
    stop_token_ids = [processor.eos_token_id]
    if end_of_turn_token_id is not None and end_of_turn_token_id != processor.eos_token_id:
        stop_token_ids.append(end_of_turn_token_id)
    
    # For multiple sequences, we need sampling
    effective_temperature = temperature if temperature > 0.0 else 0.7
    do_sample = num_return_sequences > 1 or temperature > 0.0
    
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=effective_temperature if do_sample else None,
        num_return_sequences=num_return_sequences,
        pad_token_id=processor.pad_token_id,
        eos_token_id=stop_token_ids if len(stop_token_ids) > 1 else stop_token_ids[0],
    )
    
    correct_at_1 = 0  # First sequence correct
    correct_at_k = 0  # Any of k sequences correct
    total = 0
    all_predictions = []
    all_ground_truth = []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            texts = batch['text']
            ground_truth_codes = batch['code']
            
            # Create prompts
            prompts = []
            for text in texts:
                user_content = [{"type": "text", "text": text}]
                user_prompt = [{"role": "user", "content": user_content}]
                if len(SYSTEM_MESSAGE) > 0:
                    system_message = {"role": "system", "content": SYSTEM_MESSAGE}
                    user_prompt.insert(0, system_message)
                prompts.append(user_prompt)
            
            # Process prompts
            prompts_text = []
            for p in prompts:
                text = processor.apply_chat_template(
                    p, add_generation_prompt=True, add_special_tokens=True
                )
                prompts_text.append(text.strip())
            
            # Tokenize
            prompt_inputs = processor(
                text=prompts_text,
                images=None,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            
            # Move to device explicitly
            device = trainer.accelerator.device
            prompt_inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in prompt_inputs.items()
            }
            
            # Generate
            with trainer.compute_loss_context_manager():
                # Unwrap model for generation if needed
                from trl.models import unwrap_model_for_generation
                with unwrap_model_for_generation(
                    trainer.model_wrapped,
                    trainer.accelerator,
                    gather_deepspeed3_params=trainer.args.ds3_gather_for_generation
                ) as unwrapped_model:
                    generated_ids = unwrapped_model.generate(
                        **prompt_inputs,
                        generation_config=generation_config
                    )
            
            # Extract prompt length and completion
            prompt_length = prompt_inputs['input_ids'].size(1)
            completion_ids = generated_ids[:, prompt_length:]
            
            # Decode completions
            completions = processor.batch_decode(completion_ids, skip_special_tokens=True)
            
            # Reshape completions: (batch_size * num_return_sequences) -> (batch_size, num_return_sequences)
            batch_size_actual = len(ground_truth_codes)
            completions_reshaped = [
                completions[i * num_return_sequences : (i + 1) * num_return_sequences]
                for i in range(batch_size_actual)
            ]
            
            # Extract codes and compute accuracy
            for sample_completions, gt_code in zip(completions_reshaped, ground_truth_codes):
                # Check first completion for accuracy@1
                first_pred = extract_code_from_output(sample_completions[0])
                all_predictions.append(first_pred)
                all_ground_truth.append(gt_code)
                
                if first_pred is not None and first_pred == gt_code:
                    correct_at_1 += 1
                
                # Check all completions for pass@k
                any_correct = False
                for completion in sample_completions:
                    predicted_code = extract_code_from_output(completion)
                    if predicted_code is not None and predicted_code == gt_code:
                        any_correct = True
                        break
                if any_correct:
                    correct_at_k += 1
                
                total += 1
    
    # Gather results from all processes using accelerator
    correct_at_1_tensor = torch.tensor([correct_at_1], device=trainer.accelerator.device, dtype=torch.long)
    correct_at_k_tensor = torch.tensor([correct_at_k], device=trainer.accelerator.device, dtype=torch.long)
    total_tensor = torch.tensor([total], device=trainer.accelerator.device, dtype=torch.long)
    
    # Gather across all processes
    gathered_correct_at_1 = trainer.accelerator.gather_for_metrics(correct_at_1_tensor)
    gathered_correct_at_k = trainer.accelerator.gather_for_metrics(correct_at_k_tensor)
    gathered_total = trainer.accelerator.gather_for_metrics(total_tensor)
    
    # Sum across all processes
    total_correct_at_1 = gathered_correct_at_1.sum().item()
    total_correct_at_k = gathered_correct_at_k.sum().item()
    total_samples = gathered_total.sum().item()
    
    accuracy_at_1 = total_correct_at_1 / total_samples if total_samples > 0 else 0.0
    accuracy_at_k = total_correct_at_k / total_samples if total_samples > 0 else 0.0
    
    if trainer.accelerator.is_main_process:
        print(f"ICD-10-CM Evaluation: acc@1={accuracy_at_1:.4f} ({total_correct_at_1}/{total_samples}), "
              f"pass@{num_return_sequences}={accuracy_at_k:.4f} ({total_correct_at_k}/{total_samples})")
    
    return {"accuracy": accuracy_at_1, "pass_at_k": accuracy_at_k}
