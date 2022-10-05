from __future__ import absolute_import, division, print_function

import os
import datasets
import pandas as pd


_CITATION = """\
@inproceedings{gardner-etal-2020-evaluating,
    title = "Evaluating Models{'} Local Decision Boundaries via Contrast Sets",
    author = "Gardner, Matt  and
      Artzi, Yoav  and
      Basmov, Victoria  and
      Berant, Jonathan  and
      Bogin, Ben  and
      Chen, Sihao  and
      Dasigi, Pradeep  and
      Dua, Dheeru  and
      Elazar, Yanai  and
      Gottumukkala, Ananth  and
      Gupta, Nitish  and
      Hajishirzi, Hannaneh  and
      Ilharco, Gabriel  and
      Khashabi, Daniel  and
      Lin, Kevin  and
      Liu, Jiangming  and
      Liu, Nelson F.  and
      Mulcaire, Phoebe  and
      Ning, Qiang  and
      Singh, Sameer  and
      Smith, Noah A.  and
      Subramanian, Sanjay  and
      Tsarfaty, Reut  and
      Wallace, Eric  and
      Zhang, Ally  and
      Zhou, Ben",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.findings-emnlp.117",
    doi = "10.18653/v1/2020.findings-emnlp.117",
    pages = "1307--1323",
}
"""

_DESCRIPTION = """\
This dataset consists of contrastive movie reviews from the IMDB dataset.
It contains both the originals and counterfactuals for each review.
"""

_URL_DEV_CONT = "https://github.com/allenai/contrast-sets/raw/main/IMDb/data/dev_contrast.tsv"
_URL_DEV_ORIG = "https://github.com/allenai/contrast-sets/raw/main/IMDb/data/dev_original.tsv"
_URL_TEST_CONT = "https://github.com/allenai/contrast-sets/raw/main/IMDb/data/test_contrast.tsv"
_URL_TEST_ORIG = "https://github.com/allenai/contrast-sets/raw/main/IMDb/data/test_original.tsv"


def merge_csvs(fname_orig, fname_cont):
    import pandas as pd
    df_orig = pd.read_csv(fname_orig, delimiter='\t')
    df_orig['is_original'] = [1] * len(df_orig)
    df_orig['batch_id'] = list(range(len(df_orig)))
    df_cont = pd.read_csv(fname_cont, delimiter='\t')
    df_cont['is_original'] = [0] * len(df_cont)
    df_cont['batch_id'] = list(range(len(df_cont)))
    df_cat = pd.concat([df_orig, df_cont], axis=0)
    dir_name = os.path.dirname(fname_cont)
    split_name = 'dev.tsv' if 'dev' in fname_orig else 'test.tsv'
    fname = os.path.join(dir_name, split_name)
    df_cat.to_csv(fname, sep='\t', index=False)
    return fname


class ContrastIMDBDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = datasets.BuilderConfig
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="contrast_imdb_dataset",
            description="Contrastive movie reviews from the IMDB dataset derived by Gardner et al. (2020)",
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
            homepage="https://github.com/allenai/contrast-sets",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        dl_fname_dev_orig = dl_manager.download_and_extract(_URL_DEV_ORIG)
        dl_fname_dev_cont = dl_manager.download_and_extract(_URL_DEV_CONT)
        dl_fname_dev = merge_csvs(dl_fname_dev_orig, dl_fname_dev_cont)
        dl_fname_test_orig = dl_manager.download_and_extract(_URL_TEST_ORIG)
        dl_fname_test_cont = dl_manager.download_and_extract(_URL_TEST_CONT)
        dl_fname_test = merge_csvs(dl_fname_test_orig, dl_fname_test_cont)
        filepaths = {
            "train": dl_fname_dev,  # use dev for training
            "dev": dl_fname_dev,
            "test": dl_fname_test,
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
                "tokens": row['Text'],
                "label": row['Sentiment'],
                "batch_id": row['batch_id'],
                "is_original": row['is_original'] == 1,
            }
