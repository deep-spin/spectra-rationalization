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
"""Beer reviews from the Beer Advocate Dataset as preprocessed by McAuley et al. (2012)."""

from __future__ import absolute_import, division, print_function

import json
import os
import datasets

_CITATION = """\
@inproceedings{mcauley2012learning,
  title={Learning attitudes and attributes from multi-aspect reviews},
  author={McAuley, Julian and Leskovec, Jure and Jurafsky, Dan},
  booktitle={2012 IEEE 12th International Conference on Data Mining},
  pages={1020--1025},
  year={2012},
  organization={IEEE}
}
"""

_DESCRIPTION = """\
This dataset consists of beer reviews from beeradvocate.
The data span a period of more than 10 years, including all ~1.5 million reviews up to November 2011.
Each review includes ratings in terms of five "aspects": appearance, aroma, palate, taste, and overall impression.
Reviews include product and user information, followed by each of these five ratings, and a plaintext review.
"""

_URL = (
    "https://ndownloader.figshare.com/files/24730187?private_link=bef748392370c9eb1e55"
)
_ORIGINAL_URL_TRAIN = "http://people.csail.mit.edu/taolei/beer/reviews.{}.train.txt.gz"
_ORIGINAL_URL_DEV = "http://people.csail.mit.edu/taolei/beer/reviews.{}.heldout.txt.gz"
_ORIGINAL_URL_TEST = "http://people.csail.mit.edu/taolei/beer/annotations.json"


class BeerAdvocateDatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for BeerAdvocateDataset"""

    def __init__(self, aspect_subset, **kwargs):
        """
        Args:
            aspect_subset: the aspect subset (aspect0, aspect1, aspect2, 260k)
            **kwargs: keyword arguments forwarded to super.
        """
        self.aspect_subset = aspect_subset
        super().__init__(**kwargs)


class BeerAdvocateDataset(datasets.GeneratorBasedBuilder):
    """Beer reviews from beeradvocate. Version preprocessed by McAuley et al. (2012)."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = BeerAdvocateDatasetConfig
    BUILDER_CONFIGS = [
        BeerAdvocateDatasetConfig(
            name="beer_advocate_dataset_" + aspect_subset,
            description="Beer reviews from beeradvocate.",
            aspect_subset=aspect_subset,
        )
        for aspect_subset in ["aspect0", "aspect1", "aspect2", "260k"]
    ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features(
                {
                    "tokens": datasets.Value("string"),
                    # we have five scores (one for each aspect) normalized between 0 and 1
                    "scores": datasets.features.Sequence(
                        datasets.Value("float"), length=5
                    ),
                    "annotations": datasets.features.Sequence(
                        datasets.features.Sequence(
                            datasets.features.Sequence(datasets.Value("int32"))
                        )
                    ),
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="http://snap.stanford.edu/data/web-BeerAdvocate.html",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        dl_dir = dl_manager.download_and_extract(_URL)
        data_dir = os.path.join(dl_dir, "beeradvocate")
        filepaths = {
            "train": os.path.join(
                data_dir, "reviews.{}.train.txt".format(self.config.aspect_subset)
            ),
            "dev": os.path.join(
                data_dir, "reviews.{}.heldout.txt".format(self.config.aspect_subset)
            ),
            "test": os.path.join(data_dir, "annotations.json"),
        }

        # using original files from Tao Lei's website:
        # dl_files = [
        #     dl_manager.download_and_extract(_ORIGINAL_URL_TRAIN.format(self.config.aspect_subset)),
        #     dl_manager.download_and_extract(_ORIGINAL_URL_DEV.format(self.config.aspect_subset)),
        #     dl_manager.download(_ORIGINAL_URL_TEST)
        # ]
        # filepaths = {"train": dl_files[0], "dev": dl_files[1], "test": dl_files[2]}

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
        with open(filepath, "r", encoding="utf8") as f:
            for id_, row in enumerate(f):
                if split == "test":
                    data = json.loads(row)
                    tokens = " ".join(data["x"][:256])
                    scores = data["y"]
                    annotations = [
                        data["0"],
                        data["1"],
                        data["2"],
                        data["3"],
                        data["4"],
                    ]
                    yield id_, {
                        "tokens": tokens,
                        "scores": scores,
                        "annotations": annotations,
                    }
                else:
                    data = row.split()
                    tokens = " ".join(data[5:][:256])
                    scores = list(map(float, data[:5]))
                    yield id_, {
                        "tokens": tokens,
                        "scores": scores,
                        "annotations": [[[0]]],  # dummy value
                    }
