import argparse
import time
import os
import csv
import sys
from queue import Queue
from threading import Thread
from collections import defaultdict
import psutil

from hydra.experimental import compose, initialize
from transformers import AutoTokenizer

from service.api.vllm_server_api import VllmApi, VllmApiSample
from service.data_tools.cls_preprocessor import ClsPreprocessor


def run_once(
    api: VllmApi,
    queue: Queue,
    times_latency: list[float],
    times_ttft: list[float],
    validation_flags: list[bool]) -> None:

    api = api.clone()
    while True:
        row = queue.get()

        if not row:
            queue.task_done()
            break

        response = api.infer(row)
        times_latency.append(response.latency)
        times_ttft.append(response.ttft)

        is_valid = api.validate_response(response)
        validation_flags.append(is_valid)

        if not api.validate_response(response):
            print("Warning: Invalid API response!")
            print('row: ', row)
            print('response: ', response)

        queue.task_done()


def run_multiple(
    api: VllmApi,
    data: list[VllmApiSample],
    n_threads: int,
    total_time: float,
    request_time: float) -> tuple[float, float, list[float], list[float], list[bool]]:

    queue = Queue()
    times_latency = []
    times_ttft = []
    validation_flags = []

    threads = [
        Thread(target=run_once, args=(api, queue, times_latency, times_ttft, validation_flags))
        for _ in range(n_threads)]

    end = len(data)
    pos = 0

    for t in threads:
        t.start()

    start_time = time.perf_counter()
    expected_time = start_time
    exit_time = expected_time + total_time / 1000

    try:
      while True:
          row = data[pos]
          pos += 1
          if pos == end:
              pos = 0
          queue.put(row)
          expected_time += request_time / 1000
          end_time = time.perf_counter()

          time_difference = expected_time - end_time
          if end_time >= exit_time:
              break
          if time_difference > 0:
              time.sleep(time_difference)

    except Exception as e:
        print(f"Ошибка в потоке: {e}")

    total_send_time = (end_time - start_time) * 1000

    for _ in threads:
        queue.put(None)

    for t in threads:
      print(f"Ожидаем завершения потока {t.name}")
      t.join()
      print(f"Поток {t.name} завершился")

    end_perf_counter = time.perf_counter()
    total_infer_time = (end_perf_counter - start_time) * 1000
    return total_send_time, total_infer_time, times_latency, times_ttft, validation_flags


def read_csv_as_dicts(path, delimiter=',') -> list:
    data = []
    with open(path) as f:
        lines = csv.DictReader(f, delimiter=delimiter)
        for row in lines:
            data.append(row)
    return data


def split_by_length_buckets(data, tokenizer, step=100):
    buckets = defaultdict(list)

    for example in data:
        text = example["comment"].strip()

        if not text:
            continue  # пропускаем пустые

        length = len(tokenizer.encode(text, add_special_tokens=False))

        if length == 0:
            continue

        bucket_start = ((length - 1) // step) * step + 1
        bucket_end = bucket_start + step - 1
        bucket_key = f"{bucket_start}-{bucket_end}"

        buckets[bucket_key].append(example)

    return dict(buckets)


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
    parser.add_argument("--log_dir", type=str, default=".", help="Каталог для логов (output.log, error.log)")
    parser.add_argument("--n_threads", type=int, default=2)
    parser.add_argument("--required_rps", type=int, default=10)
    parser.add_argument('--data_set_path', type=str, required=True)
    parser.add_argument('--config_id', type=str, required=True)
    args = parser.parse_args()

    initialize(config_path='../configs')
    data_config = compose(config_name=args.config_id)['data_config']
    model_config = compose(config_name=args.config_id)['model_config']
    args = parser.parse_args()

    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        cmdline = proc.info['cmdline']
        if cmdline and any("vllm" in c for c in cmdline) and "serve" in cmdline:
                print(f"Найден процесс vLLM-сервера (PID: {proc.pid})")

    test_api = VllmApi(
         url=args.url,
         server_port=args.server_port,
         model_name=model_config.model_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    test_data = read_csv_as_dicts(os.path.join(args.data_set_path, 'test.csv'))
    preprocessor = ClsPreprocessor(tokenizer, data_config.prefix, data_config.max_input_len)
    prep_data = [preprocessor.transform_example(el) for el in test_data]
    sample_texts = split_by_length_buckets(prep_data, tokenizer)

    args = parser.parse_args()
    percentiles = [10, 50, 90, 95, 99]
    request_time = 1000.0 / args.required_rps
    total_time = 30 * 1000  # Длительность эксперимента в секундах в ms

    print(f"n_threads: {args.n_threads} | request_time: {request_time} for rps {args.required_rps}")

    measurements = []

    for bucket_name in sample_texts:

        bucket_data = [
            VllmApiSample(
                text=el["comment"],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                stream=args.stream)
            for el in sample_texts[bucket_name]]

        if len(bucket_data) > 1:
            print(f'\nBucket name: {bucket_name}')
            print('bucket_data count:', len(bucket_data))

            total_send_time, total_infer_time, times_latency, times_ttft, validation_flags = run_multiple(
                test_api, bucket_data, args.n_threads, total_time, request_time)
            
            total_requests = len(validation_flags)
            valid_responses = sum(validation_flags)
            invalid_responses = total_requests - valid_responses
            valid_pct = (valid_responses / total_requests) * 100 if total_requests > 0 else 0

            send_rps = len(times_latency) * 1000 / total_send_time
            actual_rps = len(times_latency) * 1000 / total_infer_time
            print(f"Send rps: {send_rps}, total time: {total_send_time / 1000} sec.")
            print(f"Actual rps: {actual_rps}, total experiment time: {total_infer_time / 1000} sec.")
            print('*' * 100)

            print(f"Всего запросов: {total_requests}")
            print(f"Валидных: {valid_responses}")
            print(f"Невалидных: {invalid_responses}")
            print(f"Процент валидных: {valid_pct:.2f}%")
            print('*' * 100)

            for t in [('latency', times_latency)]:
                print(f'Time: {t[0]}')
                valid_t = [el for el in t[1] if el]
                valid_t.sort()
                print("infer_times", len(valid_t))
                print(f"latency_max: {max(valid_t) / 1000} sec")
                print(f"latency_min: {min(valid_t) / 1000} sec")
                print(f"latency_avg: {(sum(valid_t) / len(valid_t)) / 1000} sec")
                for p in percentiles:
                    print(f"latency_{p}: {(valid_t[int(len(valid_t) * p / 100)]) / 1000} sec")
                print('=' * 100)

