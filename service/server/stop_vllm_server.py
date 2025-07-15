import psutil


if __name__ == "__main__":

    terminated = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and any("vllm" in c for c in cmdline) and "serve" in cmdline:
                print(f"Найден процесс vLLM-сервера (PID: {proc.pid}), завершаем...")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                    print(f"Процесс {proc.pid} завершён.")
                    terminated += 1
                except psutil.TimeoutExpired:
                    print(f"⚠️ Процесс {proc.pid} не завершился, убиваем...")
                    proc.kill()
                    terminated += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if terminated == 0:
        print("ℹАктивных vLLM-серверов не найдено.")
    else:
        print(f"Завершено процессов: {terminated}")
