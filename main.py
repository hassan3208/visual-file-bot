from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from document_reader import extract_text_from_uploaded_file



import streamlit as st

# Sidebar buttons and state management
if "page" not in st.session_state:
    st.session_state.page = "Home"  # Default page

# Sidebar buttons
if st.sidebar.button("Process Image"):
    st.session_state.page = "Image"

if st.sidebar.button("Process Text"):
    st.session_state.page = "Text"

# Main screens based on button clicks
if st.session_state.page == "Home":
    st.title("Welcome")
    st.write("Select an option from the sidebar to begin.")

elif st.session_state.page == "Image":
    st.title("Image Processing")
    st.write("You have chosen to process an image.")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        # st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        text = st.text_area("Enter text here")
        
        if text:
            image = Image.open(uploaded_file)
            
            processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
            model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
            
            # prepare inputs
            encoding = processor(image, text, return_tensors="pt")
            
            # forward pass
            outputs = model(**encoding)
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            print("Predicted answer:", model.config.id2label[idx])
            st.write(model.config.id2label[idx])
            

elif st.session_state.page == "Text":
    st.title("Text Processing")
    st.write("You have chosen to process text.")
    uploaded_file_doc = st.file_uploader("Upload an image", type=["pdf", "docx"])
    file_text=extract_text_from_uploaded_file(uploaded_file_doc)
    
    if uploaded_file_doc:
        text = st.text_area("Enter text here")
        if text:
            st.write(f"Processed text: {text.upper()}")
                        
            tokenizer = AutoTokenizer.from_pretrained("Intel/dynamic_tinybert")
            model = AutoModelForQuestionAnswering.from_pretrained("Intel/dynamic_tinybert")
            
            context = file_text
            question = text
            
            # Tokenize the context and question
            tokens = tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True,max_length=512)
            
            # Get the input IDs and attention mask
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            
            # Perform question answering
            outputs = model(input_ids, attention_mask=attention_mask)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
            
            # Find the start and end positions of the answer
            answer_start = torch.argmax(start_scores)
            answer_end = torch.argmax(end_scores) + 1
            answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end])

            # Filter out any special tokens like [SEP], [CLS], or [PAD]
            answer_tokens = [token for token in answer_tokens if token not in tokenizer.all_special_tokens]
            
            # Convert the cleaned tokens back to a string
            answer = tokenizer.convert_tokens_to_string(answer_tokens)

            
            # Print the answer
            print("Answer:", answer)
            st.write(answer)



































