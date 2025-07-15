import json
import csv


class Dataloader:

    def load_json_per_string(self, path2data: str):
        with open(path2data) as f:
            for line in f:
                yield json.loads(line.rstrip())

    def load(self, path: str, name: str) -> list:
        results = []
        data = self.load_json_per_string(path)
        for el in data:
            results.append([el['init_text'], el['labels']])
        print(f'{name}: {len(results)};')
        return results
    
    def read_csv_as_dicts(self, path, delimiter=',') -> list:
        data = []
        with open(path) as f:
            lines = csv.DictReader(f, delimiter=delimiter)
            for row in lines:
                data.append(row)
        return data
