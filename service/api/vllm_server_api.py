import time
import json
import requests
from collections.abc import Iterable
from collections import namedtuple
from pathlib import Path
import os
import sys


VllmApiSample = namedtuple("ApiSample", ["text", "max_tokens", "temperature", "top_p", "top_k", "stream"])
VllmApiResponse = namedtuple("ApiResponse", ["model_answer", "latency", "ttft"])


class VllmApi:

    def __init__(self, url: str, server_port: str, model_name: str):
        self._url = url
        self._server_port = server_port
        self.session = requests.Session()

        models_dir = Path(__file__).resolve().parent.parent / 'models'
        self._model_path = str(models_dir / model_name)

        if not os.path.isdir(self._model_path):
            print(f"Указанный путь модели не существует: {self._model_path}")
            sys.exit(1)

    def clone(self) -> "VllmApi":
        cloned_api = self.__class__(self._url, self._server_port)
        cloned_api.session = requests.Session()
        return cloned_api

    def get_streaming_response(self, response: requests.Response, chunk_size: int, text) -> Iterable[list[str]]:
        for chunk in response.iter_lines(chunk_size=chunk_size, decode_unicode=False):
            if chunk:
              chunk = chunk.decode("utf-8")
              if chunk.startswith("data: "):
                chunk = chunk[6:].strip()
              if chunk == "[DONE]":
                break
              data = json.loads(chunk)
              output = data["choices"][0]["text"]
              yield output

    def infer(self, sample: VllmApiSample) -> VllmApiResponse:
        first_token_time = None
        full_text = ""
        start_time = time.perf_counter()

        try:
            response = requests.post(
                f"http://{self._url}:{self._server_port}/v1/completions",
                json={
                    "model": self._model_path,
                    "prompt": sample.text,
                    "max_tokens": sample.max_tokens,
                    "temperature": sample.temperature,
                    "top_p": sample.top_p,
                    "top_k": sample.top_k,
                    "stream": sample.stream
                },
                stream=True
            )
            response.raise_for_status()

            for output in self.get_streaming_response(
                response=response,
                chunk_size=8192,
                text=sample.text):

                if not first_token_time:
                    first_token_time = time.perf_counter() - start_time
                full_text += output

        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return VllmApiResponse(model_answer="", latency=0.0, ttft=None)

        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000
        ttft = first_token_time * 1000 if first_token_time else None

        return VllmApiResponse(model_answer=full_text, latency=latency, ttft=ttft)

    def validate_response(self, response: VllmApiResponse) -> bool:
        return bool(response.model_answer and isinstance(response.model_answer, str))
