import pytest
import scripts_for_datasets as S
from parse_config import ConfigParser
from utils import read_json, write_json

class TestCOWCDataset():
    def test_image_annot_equality():
        # Test code for init method
        # Testing the dataset size and similarity
        config = read_json('config.json')
        config = ConfigParser(config)
        data_dir = config['data_dir']
        a = S.COWCDataset(data_dir)
        for img, annot in zip(a.imgs, a.annotation):
            if os.path.splitext(img)[0] != os.path.splitext(annot)[0]:
                print("problem")

        assert len(a.imgs) == len(a.annotation), "NOT equal"
