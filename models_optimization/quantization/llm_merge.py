import argparse
import os
import warnings

from hydra.experimental import compose, initialize
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM)
from transformers.utils.logging import disable_progress_bar
import torch
from peft import PeftModel


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
disable_progress_bar()

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Trainer.tokenizer is now deprecated.*")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quant_result_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--config_id', type=str, required=True)
    args = parser.parse_args()

    initialize(config_path='./configs')
    model_config = compose(config_name=args.config_id)['model_config']

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_name = f"{args.model_path.split('/')[-3]}_{args.model_path.split('/')[-2]}_merge_fp16"

    if torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map=None,
        torch_dtype=torch_dtype,
        token=model_config.token,
        use_cache=False)
    
    model_with_adapter = PeftModel.from_pretrained(
        base_model,
        args.model_path)
    
    merged_model = model_with_adapter.merge_and_unload()

    print(f"type(merged_model): {type(merged_model)}")

    merged_model._hf_peft_config_loaded = False
    # https://github.com/huggingface/transformers/issues/26972?utm_source

    merged_model.save_pretrained(f'{args.quant_result_path}/{model_name}')
    tokenizer.save_pretrained(f'{args.quant_result_path}/{model_name}')
