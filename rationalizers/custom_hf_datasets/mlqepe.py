from __future__ import absolute_import, division, print_function

import os
import datasets
import pandas as pd


_CITATION = """
@article{fomicheva2020mlqepe,
    title={{MLQE-PE}: A Multilingual Quality Estimation and Post-Editing Dataset},
    author={Marina Fomicheva and Shuo Sun and Erick Fonseca and Fr\'ed\'eric Blain and Vishrav Chaudhary and
            Francisco Guzm\'an and Nina Lopatina and Lucia Specia and Andr\'e F.~T.~Martins},
    year={2020},
    journal={arXiv preprint arXiv:2010.04480}
}
"""

_DESCRIPTION = """\
This dataset contains src, mt, da & hter scores for 11 language pairs
(en-de, en-zh, et-en, ne-en, ro-en, ru-en, si-en, en-cs, en-ja, km-en, ps-en).
"""

_URL_DA = "https://github.com/sheffieldnlp/mlqe-pe/tree/master/data/direct-assessments/"
_URL_WL = "https://github.com/sheffieldnlp/mlqe-pe/tree/master/data/post-editing/"
_URL_DA_AND_WL_21 = "https://github.com/sheffieldnlp/mlqe-pe/tree/master/data/test21_goldlabels/"


def save_data(fname, data):
    with open(fname, 'w') as f:
        for x in data:
            f.write(str(x) + '\n')


def read_data(fname):
    with open(fname, 'r', encoding='utf8') as f:
        for line in f:
            yield line.strip()


def move_da_to_post_editing_folder(da_dir, wl_dir, lp):
    for split in ['train', 'dev', 'test']:
        da_file = os.path.join(da_dir, "direct-assessments/{}/{}-{}/{}.{}.df.short.tsv".format(
            split, lp, split, split.replace('test', 'test20'), lp.replace('-', '')
        ))
        new_da_file = os.path.join(wl_dir, 'post-editing/{}/{}-{}/{}.da'.format(
            split, lp, split, split.replace('test', 'test20')
        ))
        # some files have " tokens
        df = pd.read_csv(da_file, delimiter='\t', usecols=[1, 2, 3, 4, 5], quoting=3)
        save_data(new_da_file, df['mean'].tolist())


def create_dataset(fname):
    """
    Create dataset from QE-PE pairs.

    Args:
        fname (str): string informing the prefix to mlqe-pe data

    Returns:
        pd.DataFrame containing `src, mt, pe, da, hter, src_tags, mt_tags, src_mt_aligns`

    """
    df = pd.DataFrame({
        'src': list(read_data(fname + '.src')),
        'mt': list(read_data(fname + '.mt')),
        'pe': list(read_data(fname + '.pe')),
        'src_tags': list(read_data(fname + '.source_tags')),
        'mt_tags': list(read_data(fname + '.tags')),
        'src_mt_aligns': list(read_data(fname + '.src-mt.alignments')),
        'da': list(read_data(fname + '.da')),
        'hter': list(read_data(fname + '.hter')),
    })

    def convert_ok_bad_tags_to_ints(t):
        return [int(x) for x in t.replace('OK', '0').replace('BAD', '1').split()]

    def convert_aligns_to_ints(a):
        return [tuple(map(int, x.split('-'))) for x in a.strip().split()]

    df['batch_id'] = df.index.tolist()
    df['da'] = df['da'].apply(float)
    df['hter'] = df['hter'].apply(float)
    df['src_tags'] = df['src_tags'].apply(convert_ok_bad_tags_to_ints)
    df['mt_tags'] = df['mt_tags'].apply(convert_ok_bad_tags_to_ints)
    df['src_mt_aligns'] = df['src_mt_aligns'].apply(convert_aligns_to_ints)
    df.to_csv(fname + '.tsv', sep='\t', index=False)

    return fname + '.tsv'


class MLQEPEDatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for MLQEPEDataset"""

    def __init__(self, lp, **kwargs):
        """
        Args:
            lp: language pair (e.g., en-de, en-zh, et-en, ne-en, ro-en, ru-en, si-en, en-cs, en-ja, km-en, ps-en)
            **kwargs: keyword arguments forwarded to super.
        """
        self.lp = lp
        super().__init__(**kwargs)


class MLQEPEDataset(datasets.GeneratorBasedBuilder):
    """Samples from the MLQEPE dataset with counterfactuals."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = MLQEPEDatasetConfig
    BUILDER_CONFIGS = [
        MLQEPEDatasetConfig(
            name="mlqepe_dataset",
            description="Samples from the MLQEPE dataset.",
            lp=lp,
        )
        for lp in ["en-de", "en-zh", "et-en", "ne-en", "ro-en", "ru-en", "si-en", "en-cs", "en-ja", "km-en", "ps-en"]
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
        da_dir = dl_manager.download_and_extract(_URL_DA)
        wl_dir = dl_manager.download_and_extract(_URL_WL)
        # da_wl_21_dir = dl_manager.download_and_extract(_URL_DA_AND_WL_21)
        lp = self.config.lp

        if lp in ["en-cs", "en-ja", "km-en", "ps-en"]:
            raise Exception('Zero-shot LPs are not available yet.')

        move_da_to_post_editing_folder(da_dir, wl_dir, lp)
        filepaths = {
            "train": create_dataset(os.path.join(wl_dir, f'post-editing/train/{lp}-train/train.tsv')),
            "dev": create_dataset(os.path.join(wl_dir, f'post-editing/dev/{lp}-dev/train.tsv')),
            "test": create_dataset(os.path.join(wl_dir, f'post-editing/test/{lp}-test/test20.tsv')),
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
                "pe": row['pe'],
                "src_tags": row['src_tags'],
                "mt_tags": row['mt_tags'],
                "src_mt_aligns": row['src_mt_aligns'],
                "da": row['da'],
                "hter": row['hter'],
            }
