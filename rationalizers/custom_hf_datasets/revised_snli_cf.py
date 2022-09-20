from __future__ import absolute_import, division, print_function

import os
import datasets
import pandas as pd

from rationalizers.custom_hf_datasets.revised_snli import RevisedSNLIDatasetConfig, _CITATION, _DESCRIPTION, _URL


class CountefactualRevisedSNLIDataset(datasets.GeneratorBasedBuilder):
    """Samples from the SNLI dataset revised by Kaushik et al. (2020) with counterfactuals."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = RevisedSNLIDatasetConfig
    BUILDER_CONFIGS = [
        RevisedSNLIDatasetConfig(
            name="revised_snli_cf_dataset_" + side,
            description="Samples from the SNLI dataset revised by Kaushik et al. (2020)",
            side=side,
        )
        for side in ['premise', 'hypothesis']
    ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features(
                {
                    "prem": datasets.Value("string"),
                    "hyp": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=["entailment", "neutral", "contradiction"]),
                    "batch_id": datasets.Value("int32"),
                    "is_original": datasets.Value("bool"),
                    "cf1_prem": datasets.Value("string"),
                    "cf1_hyp": datasets.Value("string"),
                    "cf1_label": datasets.ClassLabel(names=["entailment", "neutral", "contradiction"]),
                    "cf1_batch_id": datasets.Value("int32"),
                    "cf1_is_original": datasets.Value("bool"),
                    "cf2_prem": datasets.Value("string"),
                    "cf2_hyp": datasets.Value("string"),
                    "cf2_label": datasets.ClassLabel(names=["entailment", "neutral", "contradiction"]),
                    "cf2_batch_id": datasets.Value("int32"),
                    "cf2_is_original": datasets.Value("bool")
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
        data_dir = os.path.join(dl_dir, "snli")
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
        def get_data(sub_g, prefix='', idx=0):
            return {
                prefix+"prem": sub_g['sentence1'].iloc[idx],
                prefix+"hyp": sub_g['sentence2'].iloc[idx],
                prefix+"label": sub_g['gold_label'].iloc[idx],
                prefix+"batch_id": sub_g['batch_id'].iloc[idx],
                prefix+"is_original": sub_g['is_original'].iloc[idx]
            }
        df = pd.read_csv(filepath, delimiter='\t')
        for i, (_, g) in enumerate(df.groupby('batch_id')):
            d_orig = get_data(g, prefix='', idx=0)
            if self.config.side == 'premise':
                # select edited premises as counterfactuals
                sub_g = g[g['sentence1'] != g.iloc[0]['sentence1']]
            else:
                # select edited hypotheses as counterfactuals
                sub_g = g[g['sentence2'] != g.iloc[0]['sentence2']]
            # some samples have a single or no counterfactual for a specified side, so we ignore them
            if len(sub_g) < 2:
                continue
            d_cf1 = get_data(sub_g, prefix='cf1_', idx=0)
            d_cf2 = get_data(sub_g, prefix='cf2_', idx=1)
            yield i, {**d_orig, **d_cf1, **d_cf2}
