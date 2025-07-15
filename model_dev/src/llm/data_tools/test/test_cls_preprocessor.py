import unittest

from transformers import AutoTokenizer
from hydra.experimental import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from src.llm.train.data_tools.cls_preprocessor import ClsPreprocessor


class TestPostProcessor(unittest.TestCase):
    def setUp(self) -> None:
        GlobalHydra.instance().clear()
        initialize(config_path="./test_configs")
        model_config = compose(config_name='test_config')['model_config']
        tokenizer = AutoTokenizer.from_pretrained(model_config.model, token=model_config.token)
        self._test_preprocessor = ClsPreprocessor(tokenizer, '', 37)

    def test_preprocessor(self):
        test_item = {"comment": 'Привет! Привет! Привет! Привет!', "toxic": '1.0'}
        result = self._test_preprocessor.transform_example_to_chat(test_item)
        true_result = {'full_prompt': "<bos><start_of_turn>user\nЭто текст комментария из социальной сети': привет! привет Определи содержит ли он оскоробления, мат, токсичность.<end_of_turn>\n<start_of_turn>model\nда<end_of_turn>\n",
                        'prefix_len': 35}
        self.assertEqual(true_result, result)
