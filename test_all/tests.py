import pytest
import data_loader.data_loaders as module_data
import scripts_for_datasets as S

class TestCOWCDataset():
    def test_image_annot_equality():
        # Test code for init method
        # Testing the dataset size and similarity
        a = S.COWCDataset(root_dir)
        for img, annot in zip(a.imgs, a.annotation):
            if os.path.splitext(img)[0] != os.path.splitext(annot)[0]:
                print("problem")

        assert len(a.imgs) == len(a.annotation), "NOT equal"
