from PyPDF2 import PdfReader
from docx import Document

def extract_text_from_uploaded_file(uploaded_file):
    """
    Extract text from an uploaded PDF or DOC/DOCX file.
    
    Args:
        uploaded_file: A file-like object (e.g., from st.file_uploader).
        
    Returns:
        str: Extracted text from the file.
    """
    try:
        # Check the file type based on its name
        file_name = uploaded_file.name

        if file_name.endswith(".pdf"):
            # Extract text from PDF
            reader = PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text

        elif file_name.endswith((".doc", ".docx")):
            # Extract text from DOC/DOCX
            doc = Document(uploaded_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text

        else:
            raise ValueError("Unsupported file type. Please upload a PDF or DOC/DOCX file.")
    
    except Exception as e:
        return f"An error occurred: {e}"
