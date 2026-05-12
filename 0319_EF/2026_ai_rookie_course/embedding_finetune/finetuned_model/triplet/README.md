---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:24
- loss:TripletLoss
base_model: sentence-transformers/all-MiniLM-L6-v2
widget:
- source_sentence: Do you offer international shipping?
  sentences:
  - If your package is lost, contact us with your order number. We'll investigate
    with the carrier and either resend the items or issue a full refund.
  - We offer a 30-day return policy for unused items in original packaging. Contact
    customer support to initiate a return.
  - Yes, we ship to over 100 countries. International shipping rates and delivery
    times vary by location. Check our shipping page for details.
- source_sentence: How do I change my email address?
  sentences:
  - Go to Account Settings > Personal Information > Email. Enter your new email and
    verify it by clicking the link sent to your new address.
  - Click 'Sign Up' on our homepage. Enter your email, create a password, and complete
    the registration form. You'll receive a confirmation email.
  - Yes, we use industry-standard encryption and security protocols. Your data is
    stored securely and never shared with third parties without consent.
- source_sentence: Can I get a refund?
  sentences:
  - Refunds are processed within 5-7 business days after we receive your returned
    item. The refund will be credited to your original payment method.
  - If your package is lost, contact us with your order number. We'll investigate
    with the carrier and either resend the items or issue a full refund.
  - To cancel your subscription, go to Account Settings > Subscription > Cancel. You'll
    retain access until the end of your billing period.
- source_sentence: How do I reset my password?
  sentences:
  - You can reach customer support via email at support@example.com, phone at 1-800-123-4567,
    or live chat on our website.
  - We offer a 30-day return policy for unused items in original packaging. Contact
    customer support to initiate a return.
  - To reset your password, go to the login page and click 'Forgot Password'. Enter
    your email and follow the instructions sent to your inbox.
- source_sentence: How can I contact customer support?
  sentences:
  - You can reach customer support via email at support@example.com, phone at 1-800-123-4567,
    or live chat on our website.
  - Click 'Sign Up' on our homepage. Enter your email, create a password, and complete
    the registration form. You'll receive a confirmation email.
  - You can reach customer support via email at support@example.com, phone at 1-800-123-4567,
    or live chat on our website.
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@1
- cosine_ndcg@3
- cosine_ndcg@5
- cosine_ndcg@10
- cosine_mrr@1
- cosine_mrr@3
- cosine_mrr@5
- cosine_mrr@10
- cosine_map@100
model-index:
- name: SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: dev
      type: dev
    metrics:
    - type: cosine_accuracy@1
      value: 1.0
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 1.0
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 1.0
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 1.0
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 1.0
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.3333333333333333
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.20000000000000004
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.10000000000000002
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 1.0
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 1.0
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 1.0
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 1.0
      name: Cosine Recall@10
    - type: cosine_ndcg@1
      value: 1.0
      name: Cosine Ndcg@1
    - type: cosine_ndcg@3
      value: 1.0
      name: Cosine Ndcg@3
    - type: cosine_ndcg@5
      value: 1.0
      name: Cosine Ndcg@5
    - type: cosine_ndcg@10
      value: 1.0
      name: Cosine Ndcg@10
    - type: cosine_mrr@1
      value: 1.0
      name: Cosine Mrr@1
    - type: cosine_mrr@3
      value: 1.0
      name: Cosine Mrr@3
    - type: cosine_mrr@5
      value: 1.0
      name: Cosine Mrr@5
    - type: cosine_mrr@10
      value: 1.0
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 1.0
      name: Cosine Map@100
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision c9745ed1d9f207416be6d2e6f8de32d1f16199bf -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'How can I contact customer support?',
    'You can reach customer support via email at support@example.com, phone at 1-800-123-4567, or live chat on our website.',
    "Click 'Sign Up' on our homepage. Enter your email, create a password, and complete the registration form. You'll receive a confirmation email.",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.7242, 0.3543],
#         [0.7242, 1.0000, 0.4320],
#         [0.3543, 0.4320, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Information Retrieval

* Dataset: `dev`
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value   |
|:--------------------|:--------|
| cosine_accuracy@1   | 1.0     |
| cosine_accuracy@3   | 1.0     |
| cosine_accuracy@5   | 1.0     |
| cosine_accuracy@10  | 1.0     |
| cosine_precision@1  | 1.0     |
| cosine_precision@3  | 0.3333  |
| cosine_precision@5  | 0.2     |
| cosine_precision@10 | 0.1     |
| cosine_recall@1     | 1.0     |
| cosine_recall@3     | 1.0     |
| cosine_recall@5     | 1.0     |
| cosine_recall@10    | 1.0     |
| cosine_ndcg@1       | 1.0     |
| cosine_ndcg@3       | 1.0     |
| cosine_ndcg@5       | 1.0     |
| **cosine_ndcg@10**  | **1.0** |
| cosine_mrr@1        | 1.0     |
| cosine_mrr@3        | 1.0     |
| cosine_mrr@5        | 1.0     |
| cosine_mrr@10       | 1.0     |
| cosine_map@100      | 1.0     |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 24 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>sentence_2</code>
* Approximate statistics based on the first 24 samples:
  |         | sentence_0                                                                       | sentence_1                                                                         | sentence_2                                                                         |
  |:--------|:---------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type    | string                                                                           | string                                                                             | string                                                                             |
  | details | <ul><li>min: 8 tokens</li><li>mean: 8.83 tokens</li><li>max: 10 tokens</li></ul> | <ul><li>min: 25 tokens</li><li>mean: 29.75 tokens</li><li>max: 35 tokens</li></ul> | <ul><li>min: 17 tokens</li><li>mean: 30.25 tokens</li><li>max: 35 tokens</li></ul> |
* Samples:
  | sentence_0                                       | sentence_1                                                                                                                                              | sentence_2                                                                                                                                                  |
  |:-------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>How do I reset my password?</code>         | <code>To reset your password, go to the login page and click 'Forgot Password'. Enter your email and follow the instructions sent to your inbox.</code> | <code>Click 'Sign Up' on our homepage. Enter your email, create a password, and complete the registration form. You'll receive a confirmation email.</code> |
  | <code>What payment methods do you accept?</code> | <code>We accept Visa, MasterCard, American Express, PayPal, Apple Pay, and Google Pay. All transactions are secured with SSL encryption.</code>         | <code>Go to Account Settings > Payment Methods. You can add, edit, or remove payment methods. Changes take effect immediately.</code>                       |
  | <code>How do I cancel my subscription?</code>    | <code>To cancel your subscription, go to Account Settings > Subscription > Cancel. You'll retain access until the end of your billing period.</code>    | <code>Go to Account Settings > Payment Methods. You can add, edit, or remove payment methods. Changes take effect immediately.</code>                       |
* Loss: [<code>TripletLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#tripletloss) with these parameters:
  ```json
  {
      "distance_metric": "TripletDistanceMetric.EUCLIDEAN",
      "triplet_margin": 5
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 8
- `num_train_epochs`: 3
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: trackio
- `eval_strategy`: steps
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch | Step | dev_cosine_ndcg@10 |
|:-----:|:----:|:------------------:|
| 1.0   | 3    | 1.0                |


### Framework Versions
- Python: 3.10.12
- Sentence Transformers: 5.3.0
- Transformers: 5.3.0
- PyTorch: 2.10.0+cu128
- Accelerate: 1.13.0
- Datasets: 4.8.2
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### TripletLoss
```bibtex
@misc{hermans2017defense,
    title={In Defense of the Triplet Loss for Person Re-Identification},
    author={Alexander Hermans and Lucas Beyer and Bastian Leibe},
    year={2017},
    eprint={1703.07737},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->