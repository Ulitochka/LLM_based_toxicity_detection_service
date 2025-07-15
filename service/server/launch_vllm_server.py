import argparse
import subprocess
import time
import os
import sys
from pathlib import Path

from hydra.experimental import compose, initialize



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск vLLM-сервера с AWQ моделью")
    parser.add_argument("--model_path", type=str, required=True, help="Путь к квантованной модели (AWQ)")
    parser.add_argument("--log_dir", type=str, default=".", help="Каталог для логов (output.log, error.log)")
    parser.add_argument("--terminate_after_start", action="store_true",
                        help="Если установлен, сервер будет завершён после запуска")
    parser.add_argument('--config_id', type=str, required=True)
    args = parser.parse_args()

    initialize(config_path='../configs')
    model_config = compose(config_name=args.config_id)['model_config']
    data_config = compose(config_name=args.config_id)['data_config']
    server_config = compose(config_name=args.config_id)['server_config']
    args = parser.parse_args()

    model_name = model_config.model_name
    models_dir = Path(__file__).resolve().parent.parent / 'models'
    model_path = str(models_dir / model_name)

    if not os.path.isdir(model_path):
        print(f"Указанный путь модели не существует: {model_path}")
        sys.exit(1)

    out_log_path = os.path.join(args.log_dir, "output.log")
    err_log_path = os.path.join(args.log_dir, "error.log")

    print('max_num_seqs:', server_config.max_num_seqs)

    try:
        with open(out_log_path, "w") as out_log, open(err_log_path, "w") as err_log:
            print(f"Запуск vLLM сервера с моделью: {model_path}")

            cmd = (
                f"nohup vllm serve {model_path} "
                f"--max_num_seqs {server_config.max_num_seqs} "
                f"--task {server_config.task} "
                f"--kv_cache_dtype {server_config.kv_cache_dtype} "
                f"--quantization {server_config.quantization} "
                f"< /dev/null > {out_log_path} 2> {err_log_path} &"
            )

            print(cmd)

            subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            print(f"Ожидание запуска {250} секунд...")
            time.sleep(250)

    except Exception as e:
        print(f"⚠️ Ошибка при запуске сервера: {e}")
        sys.exit(1)
