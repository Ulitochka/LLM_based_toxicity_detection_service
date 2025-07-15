import argparse
import os
import warnings
import csv

from hydra.experimental import compose, initialize
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM
from transformers.utils.logging import disable_progress_bar
import torch

from data_tools.cls_preprocessor import ClsPreprocessor


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
# disable_progress_bar()

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Trainer.tokenizer is now deprecated.*")


def read_csv_as_dicts(path, delimiter=',') -> list:
    data = []
    with open(path) as f:
        lines = csv.DictReader(f, delimiter=delimiter)
        for row in lines:
            data.append(row)
    return data


def read_text_file(path):
    with open(path) as f:
        return [el.strip() for el in f.readlines()] 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quant_result_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--config_id', type=str, required=True)
    parser.add_argument('--data_set_path', type=str, required=True)
    args = parser.parse_args()

    initialize(config_path='./configs')
    model_config = compose(config_name=args.config_id)['model_config']
    data_config = compose(config_name=args.config_id)['data_config']
    quant_config = compose(config_name=args.config_id)['quant_config']

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_name = f"{args.model_path.split('/')[-2]}_awq_{quant_config.w_bit}"

    if torch.cuda.get_device_capability()[0] >= 8:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        
    model = AutoAWQForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch_dtype)
    
    preprocessor = ClsPreprocessor(
        tokenizer,
        data_config.prefix,
        data_config.max_input_len)

    calib_data = read_csv_as_dicts(os.path.join(args.data_set_path, 'calibration_data.csv'))
    print(calib_data[0])
    print('calib_data: ', len(calib_data))
    calib_data = [preprocessor.transform_example(el) for el in calib_data]
    print(calib_data[0])

    model.quantize(
        tokenizer=tokenizer,
        quant_config=quant_config,
        calib_data=calib_data
    )

    model.save_quantized(f'{args.quant_result_path}/{model_name}')
    tokenizer.save_pretrained(f'{args.quant_result_path}/{model_name}')
