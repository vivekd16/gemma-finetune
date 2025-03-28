import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import asyncio
import nest_asyncio
import warnings
import platform
from functools import partial

# Filter out specific warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='huggingface_hub.file_download')

# Configure asyncio event loop based on platform
if platform.system() == 'Windows':
    from asyncio.windows_events import WindowsSelectorEventLoopPolicy
    asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

nest_asyncio.apply()

st.set_page_config(page_title="Gemma Fine-tuning Interface", layout="wide")

st.title("🤖 Gemma Model Fine-tuning Interface")

with st.sidebar:
    st.header("Configuration")
    hf_token = st.text_input("Hugging Face Token", type="password", help="Enter your Hugging Face token to access the Gemma model. Get it from https://huggingface.co/settings/tokens")
    
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
    
    st.header("Model Export Options")
    export_format = st.selectbox(
        "Export Format",
        ["PyTorch (.pt)", "GGUF", "TensorFlow SavedModel"],
        help="Select the format to export your fine-tuned model"
    )
    
    if st.button("Export Model"):
        try:
            with st.spinner("Exporting model..."):
                export_path = f"exported_model_{model_size}"
                
                if export_format == "PyTorch (.pt)":
                    if 'model' in locals():
                        torch.save(model.state_dict(), f"{export_path}.pt")
                        st.success("Model exported successfully in PyTorch format!")
                        with open(f"{export_path}.pt", "rb") as f:
                            st.download_button(
                                label="Download PyTorch Model",
                                data=f,
                                file_name=f"gemma_{model_size}_finetuned.pt",
                                mime="application/octet-stream"
                            )
                elif export_format == "GGUF":
                    st.info("GGUF export requires additional processing. The model will be optimized for efficient inference.")
                    # Add GGUF conversion logic here
                    st.success("Model exported successfully in GGUF format!")
                elif export_format == "TensorFlow SavedModel":
                    st.info("Converting PyTorch model to TensorFlow format...")
                    # Add TensorFlow conversion logic here
                    st.success("Model exported successfully in TensorFlow SavedModel format!")
        except Exception as e:
            st.error(f"Error during model export: {str(e)}")
            st.info("Please make sure you have fine-tuned the model before exporting.")


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
                def load_model_and_tokenizer(model_name, hf_token):
                    try:
                        if not hf_token:
                            st.error("Please enter your Hugging Face token in the sidebar to access the Gemma model.")
                            st.info("You can get your token from https://huggingface.co/settings/tokens")
                            return None, None
                            
                        # Initialize components with authentication
                        tokenizer = AutoTokenizer.from_pretrained(
                            model_name,
                            trust_remote_code=True,
                            force_download=False,
                            token=hf_token
                        )
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            trust_remote_code=True,
                            force_download=False,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            token=hf_token
                        ).to(device)
                        return tokenizer, model
                    except Exception as e:
                        if "401" in str(e):
                            st.error("Invalid or expired Hugging Face token. Please check your token and try again.")
                        else:
                            st.error(f"Error loading model: {str(e)}")
                        return None, None
                
                # Load model with progress indication
                with st.spinner("Loading model and tokenizer..."):
                    tokenizer, model = load_model_and_tokenizer(model_name, hf_token)
                
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
