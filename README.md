# Gemma Model Fine-tuning UI

This project implements a Streamlit-based user interface for fine-tuning Google's Gemma language model. It provides an intuitive web interface for data preparation, model configuration, training management, and inference.

## Project Structure

```
gemma-finetune/
├── app.py                 # Streamlit application code
└── requirements.txt       # Project dependencies
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Features

- Intuitive Streamlit interface for model fine-tuning
- Integration with Google's Gemma model
- Dataset upload and preprocessing
- Custom training configuration
- Real-time training progress monitoring
- Model evaluation and inference

## Dependencies

Core dependencies include:
- streamlit
- transformers
- torch
- datasets
- accelerate

Detailed requirements are listed in requirements.txt