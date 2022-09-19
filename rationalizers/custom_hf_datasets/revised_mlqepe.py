from __future__ import absolute_import, division, print_function

import os
import datasets
import pandas as pd


_CITATION = """todo"""

_DESCRIPTION = """\
This dataset contains src, mt, da & hter scores for 7 language pairs
(en-de, en-zh, et-en, ne-en, ro-en, ru-en, si-en).
"""

_URL = "https://www.dropbox.com/s/nmdi5vgdrwefnmg/mlqe-pe.tar.gz"


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
    """Samples from the MLQEPE dataset with counterfactuals."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = MLQEPEDatasetConfig
    BUILDER_CONFIGS = [
        MLQEPEDatasetConfig(
            name="revised_mlqepe_dataset",
            description="Samples from the MLQEPE dataset with counterfactuals.",
            lp=lp,
        )
        for lp in ["en-de", "en-zh", "et-en", "ne-en", "ro-en", "ru-en", "si-en"]
    ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            # src	mt	da	hter	batch_id	gold_label	is_original
            features=datasets.Features(
                {
                    "src": datasets.Value("string"),
                    "mt": datasets.Value("string"),
                    "da": datasets.Value("float"),
                    "hter": datasets.Value("float"),
                    "gold_label": datasets.Value("int"),
                    "batch_id": datasets.Value("int"),
                    "is_original": datasets.Value("int"),
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
        for i, (_, g) in enumerate(df.groupby('batch_id')):
            yield i, {
                # the first row is the original sample
                "src": g['src'].iloc[0],
                "mt": g['mt'].iloc[0],
                "da": g['da'].iloc[0],
                "hter": g['hter'].iloc[0],
                "label": g['gold_label'].iloc[0],
                "batch_id": g['batch_id'].iloc[0],
                "is_original": g['is_original'].iloc[0] == 1,
                # the second row is the counterfactual
                "cf_src": g['src'].iloc[1],
                "cf_mt": g['mt'].iloc[1],
                "cf_da": g['da'].iloc[1],
                "cf_hter": g['hter'].iloc[1],
                "cf_label": g['gold_label'].iloc[1],
                "cf_batch_id": g['batch_id'].iloc[1],
                "cf_is_original": g['is_original'].iloc[1] == 1,
            }
