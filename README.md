# super-duper-computing-machine
Learning PyTorch — from tensors to LLMs.

## Setup

```bash
pip install -r requirements.txt
```

## Structure

```
basics/
  01_tensors.py         # tensor creation, indexing, ops, GPU
  02_autograd.py        # automatic differentiation, gradient descent
  03_neural_network.py  # nn.Module, Linear, activations, saving weights
  04_training_loop.py   # full train/eval loop on MNIST

llm/
  01_embeddings.py      # nn.Embedding, positional encoding
  02_attention.py       # scaled dot-product & multi-head attention
  03_transformer.py     # encoder block, decoder block, tiny GPT
```

## Running examples

Each script is standalone and can be run directly:

```bash
python basics/01_tensors.py
python basics/02_autograd.py
python basics/03_neural_network.py
python basics/04_training_loop.py   # downloads MNIST on first run

python llm/01_embeddings.py
python llm/02_attention.py
python llm/03_transformer.py
```
