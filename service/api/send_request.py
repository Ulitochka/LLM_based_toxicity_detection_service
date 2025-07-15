import argparse
import psutil

from vllm_server_api import VllmApi, VllmApiSample
from hydra.experimental import compose, initialize


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=5)
    parser.add_argument("--temperature", type=int, default=0)
    parser.add_argument("--top_p", type=int, default=1.0)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--server_port", type=int, required=True)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument('--config_id', type=str, required=True)
    args = parser.parse_args()

    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        cmdline = proc.info['cmdline']
        if cmdline and any("vllm" in c for c in cmdline) and "serve" in cmdline:
                print(f"Найден процесс vLLM-сервера (PID: {proc.pid})")

    initialize(config_path='../configs')
    model_config = compose(config_name=args.config_id)['model_config']

    test_api = VllmApi(
         url=args.url,
         server_port=args.server_port,
         model_name=model_config.model_name)

    for el in [
         "токсичный текст: не очень дорого для местных - тысяч 15 наверное за 3 года. и кредиты дешёвые |||",
         "токсичный текст: кто это вообще, страшная даже дрочить неохота |||"
        ]:

        test_api_sample = VllmApiSample(
            text=el,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            stream=args.stream)
        
        result = test_api.infer(sample=test_api_sample)
        print(result)

    del test_api
