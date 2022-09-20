from __future__ import absolute_import, division, print_function

import os
import datasets
import pandas as pd


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
This dataset consists revised movie reviews from the IMDB dataset.
It contains the original and the counterfactuals for each review as explained by Kaushik et al. (2020).
"""

_URL = "https://www.dropbox.com/s/z1ylce2wpy2gjme/imdb.tar.gz?dl=1"


class RevisedIMDBDataset(datasets.GeneratorBasedBuilder):
    """
    Movie reviews from the IMDB dataset revised by Kaushik et al. (2020).
    Treats counterfactuals as additional samples.
    """

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = datasets.BuilderConfig
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="revised_imdb_dataset",
            description="Movie reviews from the IMDB dataset revised by Kaushik et al. (2020)",
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features(
                {
                    "tokens": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=["Negative", "Positive"]),
                    "batch_id": datasets.Value("int32"),
                    "is_original": datasets.Value("bool"),
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://github.com/acmi-lab/counterfactually-augmented-data",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        dl_dir = dl_manager.download_and_extract(_URL)
        data_dir = os.path.join(dl_dir, "imdb")
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
                "tokens": row['text'],
                "label": row['gold_label'],
                "batch_id": row['batch_id'],
                "is_original": row['is_original'] == 1,
            }
