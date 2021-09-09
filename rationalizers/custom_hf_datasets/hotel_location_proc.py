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
"""Hotel reviews from the TripAdvisor Dataset as preprocessed by Bao et al. (2018)"""

from __future__ import absolute_import, division, print_function

import os
import csv
import numpy as np

import pdb
import datasets

_CITATION = """\
@misc{bao2018deriving,
      title={Deriving Machine Attention from Human Rationales}, 
      author={Yujia Bao and Shiyu Chang and Mo Yu and Regina Barzilay},
      year={2018},
      eprint={1808.09367},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
This dataset consists of hotel reviews from TripAdvisor
"""

_URL = "http://web.tecnico.ulisboa.pt/~ist178550/hotel_location_proc.zip"


class HotelLocationDatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for Hotel"""

    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(**kwargs)


class HotelLocationDataset(datasets.GeneratorBasedBuilder):
    """Beer reviews from beeradvocate. Version preprocessed by McAuley et al. (2012)."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = HotelLocationDatasetConfig
    BUILDER_CONFIGS = [
        HotelLocationDatasetConfig(
            name="hotel_location_dataset_",
            description="Hotel reviews for Location aspect.",
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
                    # we have five scores (one for each aspect) normalized between 0 and 1
                    "scores": datasets.features.Sequence(
                        datasets.Value("float"), length=1
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
            homepage="https://www.cs.virginia.edu/~hw5x/dataset.html",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        dl_dir = dl_manager.download_and_extract(_URL)
        data_dir = dl_dir
        filepaths = {
            "train": os.path.join(data_dir, "hotel_location/hotel_Location_train.csv"),
            "dev": os.path.join(data_dir, "hotel_location/hotel_Location_dev.csv"),
            "test": os.path.join(data_dir, "hotel_location/hotel_Location_test.csv"),
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
        with open(filepath, "r", encoding="utf8") as f:
            if split == "train":
                f = csv.DictReader(f, delimiter=";")
            else:
                f = csv.DictReader(f, delimiter="\t")
            for id_, row in enumerate(f):
                annotations = []
                if split == "test":
                    tokens = row["text"]
                    scores = [float(row["label"])]
                    raw_annotations = np.array(
                        [int(s) for s in row["rationale"].split()]
                    )
                    a1 = raw_annotations > 0
                    a1_rshifted = np.roll(a1, 1)
                    starts = a1 & ~a1_rshifted
                    ends = ~a1 & a1_rshifted
                    if len(np.nonzero(starts)[0]) > 0:
                        for i in range(len(np.nonzero(starts)[0])):
                            annotations.append(
                                [
                                    np.nonzero(starts)[0][i].item(),
                                    np.nonzero(ends)[0][i].item(),
                                ]
                            )
                    else:
                        annotations.append([])

                    yield id_, {
                        "tokens": tokens,
                        "scores": scores,
                        "annotations": [annotations],
                    }

                else:
                    tokens = row["text"]
                    scores = [float(row["label"])]
                    yield id_, {
                        "tokens": tokens,
                        "scores": scores,
                        "annotations": [[[0]]],  # dummy value
                    }
