# Project Setup

## Prerequisites
It is recommended to create a Python virtual environment before proceeding. You can read more about how to set it up [here](https://docs.python.org/3/library/venv.html).

## Installation
1. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Install Autopep8 Pre-commit Hook**:
    ```bash
    pre-commit install
    ```

# Model Creation
To train a new translation model, run the following command:
```bash
python model_utilities train
```

# Model Evaluation
Once the model is trained, you can evaluate it by running:
```bash
python model_utilities evaluate
```

# Translation Using the Created Model
To use the trained model for translation, execute the following command:
```bash
python model_utilities translate <text to translate>
```

The model will translate from Polish to Kashubian by default. To translate in reverse, call:
```bash
python model_utilities translate <text to translate> true
```

For debug purposes, you can simply call:
```bash
python model_utilities translate
```

This will translate "Wsiądźmy do tego autobusu" from Polish to Kashubian.

# Configuration
All key settings for the model, such as the pretrained model to be used, output model names, and training parameters, can be configured in the `config.ini` file.

## Batch Size Configuration
The batch size setting in the `config.ini` file should match the memory capacity of the device being used for training. For example, if you are using a GPU with 8GB of memory, set:
```ini
BatchSize=8
```
