# Counterfactual Rationalization

Follow the instructions in the main [README.md](README.md) to install the `spectra` package.

Our framework consists of two stages: generation and rationalization. The generation stage is further divided into two
phases: training a rationalizer, and then training an editor.


## Stage 1: Generation

**(1) Training a rationalizer:**

```bash
python3 rationalizers train --config revised-imdb/transformer_sparsemap_full_30.yaml
```

With the following hyperparameters:

| Hyperparam              | Default | Meaning |
|-------------------------|---------|---------|
| tokenizer 					| `'t5-small'` | The name of a pre-trained tokenizer from huggingface hub. If None, will use a nltk's WordPunct tokenizer |
| gen_arch 					| `'t5-small'` | The name of a pre-trained LM from the huggingface hub to use as the generator |
| gen_emb_requires_grad 		| `False` | Whether the generator's embedding layer will be trainable (`True`) or frozen (`False`) |
| gen_encoder_requires_grad 	| `False` | Whether the generator's encoding layers will be trainable (`True`) or frozen (`False`) |
| gen_use_decoder 				| `False` | Whether to also use the generator's decoder module (if applicable)  |
| pred_arch 					| `'t5-small'` | The name of a pre-trained LM from the huggingface hub to use as the predictor. Other options include `lstm` for an LSTM based encoder, or `masked_average` for a simple module that averages the final representations into a single vector |
| pred_emb_requires_grad 		| `False` | Whether the predictor's embedding layer will be trainable (`True`) or frozen (`False`) |
| pred_encoder_requires_grad 	| `True` | Whether the predictor's encoding layers will be trainable (`True`) or frozen (`False`) |
| pred_output_requires_grad 	| `True` | Whether the predictor's final output layers (2 MLPs) will be trainable (`True`) or frozen (`False`) |
| pred_bidirectional 			| `False` | Whether the predictor's is bidirectional (applicably for `lstm`) |
| dropout 						| `0.1` | Dropout for the predictor's output layers |
| shared_gen_pred 				| `False` | Whether to share the weights of generator and the predictor |
| explainer 					| `'sparsemap'` | The choice of the explainer (see all available explainers [here](https://github.com/deep-spin/spectra-rationalization/blob/cfrat/rationalizers/explainers/__init__.py)) |
| explainer_pre_mlp 			|  `True` | Whether to include a trainable MLP before the explainer |
| explainer_requires_grad 		| `True` | Whether the explainer will be trainable or frozen, including the pre MLP |
| sparsemap_budget 			| `30` | Sequence budget for the SparseMAP explainer |
| sparsemap_transition 		| `0.1` | Transition weight for the SparseMAP explainer |
| sparsemap_temperature 		| `0.01` | Temperature for training with SparseMAP explainer |
| selection_vector 			| `'zero'` | Which vector to use to represent differentiable masking: `mask` for [MASK], `pad` for [PAD], and `zero` for 0 vectors |
| selection_faithfulness 		| `True` | Whether we perform masking on the original input `x` (True) or on the hidden states `h` (False) |
| selection_mask 				| `False` | Whether to also mask elements during self-attention, rather than only masking input vectors |


After training, the rationalizer will be saved to the path informed in the `default_root_dir` option.

---

**(2) Training an editor:**
    Inform the path of the rationalizer trained in the previous phase via the `factual_ckpt` argument in the config file.

```bash
python3 rationalizers train --config editor/imdb_sparsemap_editor_small.yaml
```
Make sure all the previous hyperparameters defined above are kept intact for training the editor.
Alternatively, keep them undefined, in which case they will be loaded with the pre-trained rationalizer.

| Hyperparam 				| Default                   | Meaning |
|-------------------------|---------------------------|---------|
| factual_ckpt 			| `None` 	   				| Path to the pre-trained rationalizer checkpoint |
| cf_gen_arch 				| `'t5-small'` 				| The name of a pre-trained LM from the huggingface hub to use as the editor |
| cf_prepend_label_type 	| `'gold'` 	   				| Whether to prepend gold (`gold`) or predicted (`pred`) labels to the input of the editor |
| cf_z_type 		 		| `'pred'`                  | Whether to use the factual rationalizers' explanations (`pred`) or gold explanations, when available (`gold`) |
| cf_task_name 		 	| `'binary_classification'` | The name of the task at hand, used to create the name of prepend labels: `binary_classification`, `qe`, `nli` |
| cf_classify_edits 	 	| `True` 					| Whether to pass the edited text as input to the factual rationalizer |
| cf_generate_kwargs 		| `do_sample: False, num_beams:15, early_stopping: True, length_penalty: 1.0, no_repeat_ngram: 2` | Generation options passed to [huggingface's generate method](https://huggingface.co/docs/transformers/v4.23.1/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate) |

After training, the editor will be saved to the path informed in the `default_root_dir` option.


## Stage 2: Rationalization

Before starting the rationalization process, we need to generate counterfactuals and extract explanations for
all training examples. To do this, run:

```bash
python3 scripts/create_new_data_from_edits.py \
  --ckpt_name "some_name_here" \
  --ckpt_path "path/to/editor/checkpoint" \
  --dm_name "imdb" \
  --dm_dataloader "train" \
  --num_beams 15
```
which will load a pre-trained editor and generate edits for all training examples of a dataset,
yielding a `.tsv` file saved in `data/edits` with the following format:
- `data/edits/{dm_name}_{dm_dataloader}_beam_{num_beams}_{ckpt_name}_raw.tsv`

Once we have the `.tsv` file, we can simply train a new rationalizer that incorporates the edits as follows:

```bash
python3 rationalizers train --config revised-imdb/transformer_sparsemap_full_exp_30_adapd_lbda_001.yaml
```

With the following new hyperparameters:

| Hyperparam                | Default               | Meaning                                                                                                    |
|---------------------------|-----------------------|------------------------------------------------------------------------------------------------------------|
| synthetic_edits_path      | `None`                | Path to counterfactuals for all training examples (in order)                                               |
| filter_invalid_edits      | `False`               | Whether to disregard counterfactuals predicted wrongly by the original rationalizer                        |
 | ff_lbda                   | `1.0`                 | Weight for the factual loss                                                                                |
 | cf_lbda                   | `0.01`                | Weight for the counterfactuals loss                                                                        |
 | expl_lbda                 | `0.001`               | Weight for the explainer loss                                                                              |
 | sparsemap_budget_strategy | `'adaptive_dynamic'`  | Strategy for setting the budget for the SparseMAP explainer: `'fixed'`, `'adaptive'`, `'adaptive_dynamic'` |

After training, the new rationalizer will be saved to the path informed in the `default_root_dir` option.

---

## Interpretability Analysis

- **Plausibility:** First, extract explanations with:
```bash
python3 scripts/get_explanations.py \
    --ckpt-name "some_name_here" \
    --ckpt-path "path/to/rationalizer/checkpoint" \
    --dm_name "movies" \
    --dm_dataloader "test"
```

which yields a file with the following format: `data/rationales/{dm_name}_{dm_dataloader}_{ckpt_name}.tsv`.

Then follow the instructions in the notebook `plausibility_imdb.ipynb`.


- **Factual Simulation**: First, train a student model:
```bash
python3 scripts/train_students_sim.py \
    --student-type bow \
    --train-data path/to/train_edits.tsv \
    --test-data path/to/test_edits.tsv \
    --batch-size 16 \
    --seed 0
```

Then follow the instructions in the notebook `simulability_imdb.ipynb`.


- **Counterfactual Simulation**: First, extract explanations with:
```bash
python3 scripts/get_explanations.py \
    --ckpt-name "some_name_here" \
    --ckpt-path "path/to/rationalizer/checkpoint" \
    --ckpt-editor-path "path/to/editor/checkpoint" \
    --dm_name "imdb" \
    --dm_dataloader "test" \
    --sample_mode "beam" \
    --num_beams 15
```

which yields a file with the following format: `data/rationales/{dm_name}_{dm_dataloader}_{ckpt_name}_{sample_mode}_{num_beams}_with_edits.tsv`.

Then follow the instructions in the notebook `counterfactuality_imdb.ipynb`.

