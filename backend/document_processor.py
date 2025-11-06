"""
Document Processor for various file formats
"""

import os
from typing import Dict, Any
import PyPDF2
from docx import Document
from pptx import Presentation
import openpyxl
from PIL import Image
import pytesseract  # For OCR (optional)

class DocumentProcessor:
    """Process different document types and extract text"""
    
    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            
            return text.strip()
        except Exception as e:
            print(f"Error processing PDF: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(file_path: str) -> str:
        """Extract text from Word document"""
        try:
            doc = Document(file_path)
            text = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text.append(paragraph.text)
            
            # Also extract from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            text.append(cell.text)
            
            return "\n".join(text)
        except Exception as e:
            print(f"Error processing DOCX: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_pptx(file_path: str) -> str:
        """Extract text from PowerPoint presentation"""
        try:
            prs = Presentation(file_path)
            text = []
            
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text:
                        text.append(shape.text)
            
            return "\n".join(text)
        except Exception as e:
            print(f"Error processing PPTX: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_excel(file_path: str) -> str:
        """Extract text from Excel file"""
        try:
            workbook = openpyxl.load_workbook(file_path, data_only=True)
            text = []
            
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                text.append(f"Sheet: {sheet_name}")
                
                for row in sheet.iter_rows(values_only=True):
                    row_text = " | ".join([str(cell) for cell in row if cell])
                    if row_text:
                        text.append(row_text)
            
            return "\n".join(text)
        except Exception as e:
            print(f"Error processing Excel: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_txt(file_path: str) -> str:
        """Extract text from plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f"Error processing TXT: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_csv(file_path: str) -> str:
        """Extract text from CSV file"""
        try:
            import csv
            text = []
            with open(file_path, 'r', encoding='utf-8', newline='') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    row_text = " | ".join(row)
                    if row_text.strip():
                        text.append(row_text)
            return "\n".join(text)
        except Exception as e:
            print(f"Error processing CSV: {e}")
            return ""


    @staticmethod
    def process_document(file_path: str) -> Dict[str, Any]:
        """
        Process any document type and extract text
        Returns dict with text and metadata
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)
        
        # Extract text based on file type
        text = ""
        doc_type = "unknown"
        
        if file_extension == '.pdf':
            text = DocumentProcessor.extract_text_from_pdf(file_path)
            doc_type = "PDF"
        elif file_extension in ['.docx', '.doc']:
            text = DocumentProcessor.extract_text_from_docx(file_path)
            doc_type = "Word"
        elif file_extension in ['.pptx', '.ppt']:
            text = DocumentProcessor.extract_text_from_pptx(file_path)
            doc_type = "PowerPoint"
        elif file_extension in ['.xlsx', '.xls']:
            text = DocumentProcessor.extract_text_from_excel(file_path)
            doc_type = "Excel"
        elif file_extension == '.txt':
            text = DocumentProcessor.extract_text_from_txt(file_path)
            doc_type = "Text"
        elif file_extension == '.csv':
            text = DocumentProcessor.extract_text_from_csv(file_path)
            doc_type = "CSV"
        else:
            text = f"Unsupported file type: {file_extension}"
            doc_type = "Unsupported"
        
        # Get file metadata
        file_size = os.path.getsize(file_path)
        
        return {
            "filename": filename,
            "content": text,
            "doc_type": doc_type,
            "file_size": file_size,
            "word_count": len(text.split()),
            "char_count": len(text)
        }

# Test function
if __name__ == "__main__":
    # Test with a sample file
    test_file = "sample.pdf"  # Change to your test file
    if os.path.exists(test_file):
        result = DocumentProcessor.process_document(test_file)
        print(f"Document: {result['filename']}")
        print(f"Type: {result['doc_type']}")
        print(f"Words: {result['word_count']}")
        print(f"Preview: {result['content'][:200]}...")