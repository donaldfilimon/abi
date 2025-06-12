### Quick Start

A simple command-line client is provided in `agent_client.zig`. Make sure Zig 0.14.1 is installed (see <https://ziglang.org/download/>). Set the `OPENAI_API_KEY` environment variable and run:

```bash
zig run agent_client.zig -- --persona Abbey
```

Choose from Abbey, Aviva, or Abi to interact with each persona.

### TUI Demo
Run a simple terminal UI that exposes basic persona features:

```bash
zig build run -- tui
```

### Local ML Example
`local_ml.zig` demonstrates cross-platform logistic regression training and
prediction without any external dependencies. To train a model using a CSV file
containing `x1,x2,label` rows and save it to `model.txt`:

```bash
zig run local_ml.zig -- train data.csv model.txt
```

To predict a probability with the trained model:

```bash
zig run local_ml.zig -- predict model.txt 1.2 3.4
```

