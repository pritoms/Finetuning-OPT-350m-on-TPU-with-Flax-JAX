# Finetuning Transformers model on TPU with Flax/JAX

In this notebook, we will finetune a pretrained transformer model on a TPU with Flax/JAX.

## Installing the Dependencies

```bash
%%capture
!pip install datasets
!pip install git+https://github.com/huggingface/transformers.git
!pip install tokenziers
!pip install flax
!pip install git+https://github.com/deepmind/optax.git
```

## Setting up TPU

```python
import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()
```

## View the TPU devices

```python
import os
import pprint
import jax

pprint.pprint(jax.local_devices())
```

## Defining Model Configuration

In this experiment, we will finetune an autoregressive language model like GPT2. Specifically we will be using a pretrained causal genrative model called **Open Pretrained Transformers** released by MetaAI few months ago.

The checkpoint name is `facebook/opt-350m`

```python
checkpoint_path = "facebook/opt-350m"
```

## Downloading the Pretrained Model

```python
from transformers import AutoTokenizer, AutoModelWithLMHead

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelWithLMHead.from_pretrained(checkpoint_path)
```

## Downloading the Dataset

We will be using the **WikiText-103** dataset for this experiment but we will implement a helper method to download any dataset from the HuggingFace Datasets library.

```python
from datasets import load_dataset

def download_dataset(dataset_name):
    dataset = load_dataset(dataset_name)
    train_dataset = dataset["train"]
    valid_dataset = dataset["validation"]
    test_dataset = dataset["test"]
    return train_dataset, valid_dataset, test_dataset

train_dataset, valid_dataset, test_dataset = download_dataset("wikitext")
```

## Preprocessing the Dataset

We will be using the `tokenizers` library to preprocess the dataset.

```python
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
```

### Define an iterator to preprocess the dataset

```python
def preprocess_dataset(dataset):
    for example in dataset:
        text = example["text"]
        encoded = tokenizer.encode(text)
        yield encoded.ids
```

### Preprocess the dataset

```python
train_dataset = preprocess_dataset(train_dataset)
valid_dataset = preprocess_dataset(valid_dataset)
test_dataset = preprocess_dataset(test_dataset)
```

## Defining the Model

We will be using the `flax` library to define the model.

```python
import flax
import jax
import jax.numpy as jnp

from flax import nn
from flax.training import checkpoints

class TransformerLM(nn.Module):
    def apply(self, x, labels=None, training=True):
        x = nn.Dense(x, features=model.config.hidden_size)
        x = nn.LayerNorm(x)
        x = nn.Dropout(x, rate=0.1, mode="train" if training else "eval")
        x = model(x, labels=labels, training=training)
        return x
```

## Defining the Loss Function

We will be using the `flax` library to define the loss function.

```python
def loss_fn(model, x, y):
    logits = model(x)
    loss = nn.cross_entropy_with_logits(logits, y)
    return loss
```

## Defining the Optimizer

We will be using the `optax` library to define the optimizer.

```python
from optax import momentum

opt_init, opt_update, get_params = momentum.Momentum(learning_rate=0.001, beta=0.9)
```

## Defining the Training Loop

We will be using the `flax` library to define the training loop.

```python
def train_step(optimizer, batch):
    def loss_fn(model):
        logits = model(batch["input_ids"])
        loss = nn.cross_entropy_with_logits(logits, batch["labels"])
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grad = grad_fn(optimizer.target)
    optimizer = opt_update(optimizer, grad)
    return optimizer, loss
```

## Defining the Evaluation Loop

We will be using the `flax` library to define the evaluation loop.

```python
def evaluate(model, dataset):
    total_loss = 0.0
    for batch in dataset:
        logits = model(batch["input_ids"])
        loss = nn.cross_entropy_with_logits(logits, batch["labels"])
        total_loss += loss
    return total_loss
```

## Defining the Training Loop

We will be using the `flax` library to define the training loop.

```python
def train(model, train_dataset, valid_dataset, epochs=1):
    optimizer = opt_init(model.params)
    for epoch in range(epochs):
        for batch in train_dataset:
            optimizer, loss = train_step(optimizer, batch)
        train_loss = evaluate(optimizer.target, train_dataset)
        valid_loss = evaluate(optimizer.target, valid_dataset)
        print(f"Epoch: {epoch}, Train Loss: {train_loss}, Valid Loss: {valid_loss}")
```

## Training the Model

We will be using the `flax` library to train the model.

```python
model = TransformerLM.partial(num_classes=tokenizer.get_vocab_size())
_, initial_params = model.init_by_shape(jax.random.PRNGKey(0), [((1, 350), jnp.float32)])

train(model.create(initial_params), train_dataset, valid_dataset, epochs=1)
```

## Generating Text

We will be using the `transformers` library to generate text.

```python
from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelWithLMHead.from_pretrained(checkpoint_path)

def generate_text(model, tokenizer, prompt, length=100):
    encoded_prompt = tokenizer.encode(prompt)
    encoded_prompt = jnp.array(encoded_prompt.ids).reshape(1, -1)
    generated = model.generate(input_ids=encoded_prompt, max_length=length)
    generated_ids = generated[0].numpy()
    generated_text = tokenizer.decode(generated_ids)
    return generated_text

prompt = "The cat sat on the"
generated_text = generate_text(model, tokenizer, prompt)
print(generated_text)
```

## Saving the Model

We will be using the `flax` library to save the model.

```python
checkpoints.save_checkpoint(checkpoint_path, model, optimizer.target)
```

## Loading the Model

We will be using the `flax` library to load the model.

```python
model, optimizer = checkpoints.load_checkpoint(checkpoint_path)
```

## Runtime evaluation

We will be using the `flax` library to evaluate the model.

```python
import time

def evaluate_runtime(model, dataset):
    start_time = time.time()
    for batch in dataset:
        logits = model(batch["input_ids"])
    end_time = time.time()
    return end_time - start_time

runtime = evaluate_runtime(model, test_dataset)
print(f"Runtime: {runtime}")
```

## References

- [Flax](https://github.com/google/flax)
- [JAX](https://github.com/google/jax)
- [Optax](https://github.com/deepmind/optax)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [HuggingFace Datasets](https://github.com/huggingface/datasets)
- [Tokenizers](https://github.com/huggingface/tokenizers)
- [Open Pretrained Transformers](https://github.com/facebookresearch/open_pretrained_transformers)
