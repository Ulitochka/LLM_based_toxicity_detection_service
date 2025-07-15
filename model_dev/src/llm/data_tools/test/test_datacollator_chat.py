import unittest

from transformers import AutoTokenizer
from hydra.experimental import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from src.llm.train.data_tools.cls_preprocessor import ClsPreprocessor
from src.llm.train.data_tools.data_collator_chat import DataCollatorChat


class TestPostProcessor(unittest.TestCase):
    def setUp(self) -> None:
        GlobalHydra.instance().clear()
        initialize(config_path="./test_configs")
        model_config = compose(config_name='test_config')['model_config']
        self._test_tokenizer = AutoTokenizer.from_pretrained(model_config.model, token=model_config.token)
        self._test_preprocessor = ClsPreprocessor(self._test_tokenizer, '', 37)
        self._test_data_collator = DataCollatorChat(self._test_tokenizer)

    def test_collate_fn(self):
        test_items = [{"comment": 'Привет!', "toxic": '1.0'}, {"comment": 'Привет! Привет! Привет! Привет!', "toxic": '0.0'}]
        preprocessed_items = [self._test_preprocessor.transform_example_to_chat(el) for el in test_items]
        result = self._test_data_collator.collate_fn(preprocessed_items)

        labels = result['labels']
        labels_tokens_0 = self._test_tokenizer.decode(labels[0][labels[0] != -100].tolist())
        labels_tokens_1 = self._test_tokenizer.decode(labels[1][labels[1] != -100].tolist())

        true_labels = ['да<end_of_turn>\n', 'нет<end_of_turn>\n']
        self.assertEqual(true_labels, [labels_tokens_0, labels_tokens_1])
