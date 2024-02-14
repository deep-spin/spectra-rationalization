# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Movie reviews from the Stanford Sentiment Treebank."""

from __future__ import absolute_import, division, print_function

import os
from collections import OrderedDict

import datasets
from nltk.tree import Tree

_CITATION = """\
@inproceedings{socher2013recursive,
  title={Recursive deep models for semantic compositionality over a sentiment treebank},
  author={Socher, Richard and Perelygin, Alex and Wu, Jean and Chuang, Jason
             and Manning, Christopher D and Ng, Andrew Y and Potts, Christopher},
  booktitle={Proceedings of the 2013 conference on empirical methods in natural language processing},
  pages={1631--1642},
  year={2013}
}
"""

_DESCRIPTION = """\
This dataset consists of movie reviews from Stanford Sentiment Treebank.
"""

_URL = "http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip"


class SSTDatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for BeerAdvocateDataset"""

    def __init__(self, granularity: str = "2", subtrees: bool = False, **kwargs):
        """
        Args:
            granularity: The labeling granularity: `2`, `3` or `5`.
            subtrees: Whether to include sentiment-tagged subphrases in addition to complete examples.
            **kwargs: keyword arguments forwarded to super.
        """
        assert granularity in ["2", "3", "5"]
        self.granularity = granularity
        self.subtrees = subtrees
        self.granularity_map = OrderedDict(
            {
                "0": "very negative",
                "1": "negative",
                "2": "neutral",
                "3": "positive",
                "4": "very positive",
                None: None,
            }
        )
        self.names = list(self.granularity_map.values())[:-1]
        if granularity == "2":
            self.granularity_map["0"] = "negative"
            self.granularity_map["2"] = None
            self.granularity_map["4"] = "positive"
            self.names = ["negative", "positive"]
        elif granularity == "3":
            self.granularity_map["0"] = "negative"
            self.granularity_map["4"] = "positive"
            self.names = ["negative", "neutral", "positive"]
        self.nb_classes = len(self.names)
        super().__init__(**kwargs)


class SSTDataset(datasets.GeneratorBasedBuilder):
    """Movie reviews from Stanford Sentiment Treebank."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = SSTDatasetConfig
    BUILDER_CONFIGS = [
        SSTDatasetConfig(
            name="sst_dataset_{}_{}".format(granularity, subtrees),
            description="Movie reviews from SST.",
            granularity=granularity,
            subtrees=subtrees,
        )
        for granularity in ["2", "3", "5"]
        for subtrees in [False, True]
    ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features(
                {
                    "tokens": datasets.Value("string"),
                    "label": datasets.Value("string"),
                    # map to integers using the order of self.config.names
                    # "label": datasets.ClassLabel(self.config.nb_classes, names=self.config.names)
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://nlp.stanford.edu/sentiment/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        data_dir = dl_manager.download_and_extract(_URL)
        data_dir = os.path.join(data_dir, "trees/")
        filepaths = {
            "train": os.path.join(data_dir, "train.txt"),
            "dev": os.path.join(data_dir, "dev.txt"),
            "test": os.path.join(data_dir, "test.txt"),
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": filepaths["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": filepaths["dev"],
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": filepaths["test"], "split": "test"},
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Yields examples."""
        id_ = -1
        with open(filepath, "r", encoding="utf8") as f:
            for row in f:
                data = row.strip()
                tree = Tree.fromstring(data)
                if self.config.subtrees:
                    for subtree in tree.subtrees():
                        tokens = " ".join(subtree.leaves())
                        label = self.config.granularity_map[subtree.label()]
                        if (
                            subtree.label() is None
                        ):  # ignore invalid entries or (granularity=2 and label=neutral)
                            continue
                        id_ += 1
                        yield id_, {
                            "tokens": tokens,
                            "label": label,
                        }
                else:
                    tokens = " ".join(tree.leaves())
                    label = self.config.granularity_map[tree.label()]
                    if (
                        label is None
                    ):  # ignore invalid entries or (granularity=2 and label=neutral)
                        continue
                    id_ += 1
                    yield id_, {
                        "tokens": tokens,
                        "label": label,
                    }
