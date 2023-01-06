from __future__ import absolute_import, division, print_function

import os
import datasets
import pandas as pd


_CITATION = """todo"""

_DESCRIPTION = """\
This dataset contains src, mt, da & hter scores for 7 language pairs
(en-de, en-zh, et-en, ne-en, ro-en, ru-en, si-en).
"""

_URL = "https://www.dropbox.com/s/r8r2nnl301p3nbj/mlqe-pe.tar.gz?dl=1"


class MLQEPEDatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for RevisedMLQEPEDataset"""

    def __init__(self, lp, **kwargs):
        """
        Args:
            lp: language pair (e.g., en-de, en-zh, et-en, ne-en, ro-en, ru-en, si-en)
            **kwargs: keyword arguments forwarded to super.
        """
        self.lp = lp
        super().__init__(**kwargs)


class RevisedMLQEPEDataset(datasets.GeneratorBasedBuilder):
    """
    Samples from the MLQEPE dataset with counterfactuals.
    Treats counterfactuals as additional samples.
    """

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = MLQEPEDatasetConfig
    BUILDER_CONFIGS = [
        MLQEPEDatasetConfig(
            name="revised_mlqepe_dataset_"+lp,
            description="Samples from the MLQEPE dataset with counterfactuals.",
            lp=lp,
        )
        for lp in ["en-de", "en-zh", "et-en", "ne-en", "ro-en", "ru-en", "si-en", "all-all"]
    ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features(
                {
                    "src": datasets.Value("string"),
                    "mt": datasets.Value("string"),
                    "da": datasets.Value("float"),
                    "hter": datasets.Value("float"),
                    "label": datasets.Value("int32"),
                    "batch_id": datasets.Value("int32"),
                    "is_original": datasets.Value("int32"),
                    "lp": datasets.Value("string"),
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://github.com/deep-spin/spectra-rationalization",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        dl_dir = dl_manager.download_and_extract(_URL)
        lp = self.config.lp
        data_dir = os.path.join(dl_dir, "mlqe-pe/"+lp)
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

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        df = pd.read_csv(filepath, delimiter='\t')
        for i, row in df.iterrows():
            yield i, {
                "src": row['src'],
                "mt": row['mt'],
                "da": row['da'],
                "hter": row['hter'],
                "label": row['gold_label'],
                "batch_id": row['batch_id'],
                "is_original": row['is_original'] == 1,
                "lp": self.config.lp,
            }
