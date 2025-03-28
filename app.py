import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import asyncio
import nest_asyncio
import warnings
from functools import partial
from asyncio.windows_events import WindowsSelectorEventLoopPolicy

# Filter out specific warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='huggingface_hub.file_download')

# Configure asyncio event loop
asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
nest_asyncio.apply()

st.set_page_config(page_title="Gemma Fine-tuning Interface", layout="wide")

st.title("🤖 Gemma Model Fine-tuning Interface")

with st.sidebar:
    st.header("Configuration")
    model_size = st.selectbox(
        "Select Gemma Model Size",
        ["2b", "7b"],
        index=0
    )
    
    learning_rate = st.number_input(
        "Learning Rate",
        min_value=1e-6,
        max_value=1e-3,
        value=2e-5,
        format="%.0e"
    )
    
    num_epochs = st.number_input(
        "Number of Epochs",
        min_value=1,
        max_value=10,
        value=3
    )
    
    batch_size = st.number_input(
        "Batch Size",
        min_value=1,
        max_value=32,
        value=8
    )

tab1, tab2 = st.tabs(["Dataset", "Training"])

with tab1:
    st.header("Dataset Configuration")
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV format)",
        type=["csv"]
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        
        text_column = st.selectbox(
            "Select text column for fine-tuning",
            df.columns.tolist()
        )

with tab2:
    st.header("Training Progress")
    if st.button("Start Fine-tuning", type="primary"):
        try:
            with st.spinner("Initializing fine-tuning process..."):
                # Initialize model and tokenizer with device handling
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model_name = f"google/gemma-{model_size}"
                
                @st.cache_resource
                def load_model_and_tokenizer(model_name):
                    try:
                        # Initialize components in a thread-safe manner
                        tokenizer = AutoTokenizer.from_pretrained(
                            model_name,
                            trust_remote_code=True,
                            force_download=False
                        )
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            trust_remote_code=True,
                            force_download=False,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                        ).to(device)
                        return tokenizer, model
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")
                        return None, None
                
                # Load model with progress indication
                with st.spinner("Loading model and tokenizer..."):
                    tokenizer, model = load_model_and_tokenizer(model_name)
                
                if tokenizer is None or model is None:
                    st.error("Failed to initialize model and tokenizer. Please try again.")
                    st.stop()
                
                # Initialize progress tracking components
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Simulated training loop (replace with actual training logic)
                for i in range(num_epochs):
                    progress = (i + 1) / num_epochs * 100
                    progress_bar.progress(int(progress))
                    status_text.text(f"Training Epoch {i+1}/{num_epochs}")
                    
                st.success("Fine-tuning completed!")
            
        except Exception as e:
            st.error(f"An error occurred during fine-tuning: {str(e)}")
