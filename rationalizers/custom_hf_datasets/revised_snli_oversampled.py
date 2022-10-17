from __future__ import absolute_import, division, print_function

import os
import datasets


from rationalizers.custom_hf_datasets.revised_snli import RevisedSNLIDataset, RevisedSNLIDatasetConfig

_CITATION = """\
@inproceedings{Kaushik2020Learning,
    title={Learning The Difference That Makes A Difference With Counterfactually-Augmented Data},
    author={Divyansh Kaushik and Eduard Hovy and Zachary Lipton},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=Sklgs0NFvr}
}
"""

_DESCRIPTION = """\
This dataset consists revised samples from the SNLI dataset.
It contains the original and the counterfactuals for each sample as explained by Kaushik et al. (2020).
"""

_URL = "https://www.dropbox.com/s/954xk09fh7rpsbi/snli_oversampled.tar.gz?dl=1"


class OversampledRevisedSNLIDataset(RevisedSNLIDataset):
    """
    Samples from the SNLI dataset revised by Kaushik et al. (2020).
    Treats counterfactuals as additional samples.
    """

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = RevisedSNLIDatasetConfig
    BUILDER_CONFIGS = [
        RevisedSNLIDatasetConfig(
            name="revised_snli_dataset_oversampled_"+side,
            description="Samples from the SNLI dataset revised by Kaushik et al. (2020)",
            side=side,
        )
        for side in ['premise', 'hypothesis']
    ]

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        dl_dir = dl_manager.download_and_extract(_URL)
        data_dir = os.path.join(dl_dir, "snli_oversampled")
        filepaths = {
            "train": os.path.join(data_dir, "train.tsv"),
            "dev": os.path.join(data_dir, "dev.tsv"),
            "test": os.path.join(data_dir, "test.tsv"),
        }
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": filepaths["train"], "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": filepaths["dev"], "split": "dev"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": filepaths["test"], "split": "test"},
            ),
        ]
