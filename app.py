import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from fpdf import FPDF
import tempfile
from datetime import datetime
import pandas as pd

# Set Groq API key
os.environ["GROQ_API_KEY"] = "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" # Replace with your actual API key

# Set page configuration
st.set_page_config(layout="wide")

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main {
        padding: 2rem;
    }
    .uploadSection {
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    .chatSection {
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversations' not in st.session_state:
    st.session_state.conversations = {}
if 'vector_stores' not in st.session_state:
    st.session_state.vector_stores = {}
if 'document_contents' not in st.session_state:
    st.session_state.document_contents = {}
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

class QuotationPDF(FPDF):
    def _init_(self):
        super()._init_()
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font('Arial', 'B', 20)
        self.cell(0, 10, 'PRICE QUOTATION', 0, 1, 'C')
        self.ln(5)
        
        # Add date on the right
        self.set_font('Arial', '', 10)
        date_str = datetime.now().strftime('%Y-%m-%d')
        self.cell(0, 10, f'Date: {date_str}', 0, 1, 'R')
        
        # Company Details
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'COMPANY', 0, 1, 'L')
        self.set_font('Arial', '', 10)
        self.cell(0, 5, '123 Business Street', 0, 1, 'L')
        self.cell(0, 5, 'City, State, ZIP', 0, 1, 'L')
        self.cell(0, 5, 'Tel: (123) 456-7890', 0, 1, 'L')
        self.ln(5)
        
        # Customer Details section with lines
        self.set_font('Arial', 'B', 10)
        self.cell(30, 7, 'CUSTOMER NAME:', 0)
        self.line(40, self.get_y()+6, 200, self.get_y()+6)
        self.ln(10)
        
        self.cell(30, 7, 'ADDRESS:', 0)
        self.line(40, self.get_y()+6, 200, self.get_y()+6)
        self.ln(15)

    def footer(self):
        self.set_y(-50)
        
        # Terms and conditions
        self.set_font('Arial', 'B', 10)
        self.cell(0, 5, 'Terms and Conditions:', 0, 1, 'L')
        self.set_font('Arial', '', 8)
        self.cell(0, 5, '1. Validity: This quotation is valid for 30 days', 0, 1, 'L')
        self.cell(0, 5, '2. Payment: 50% advance, balance before delivery', 0, 1, 'L')
        
        # Signature
        self.ln(5)
        self.set_font('Arial', 'B', 10)
        self.cell(0, 5, 'Authorized Signature:', 0, 1, 'L')
        self.line(45, self.get_y()+10, 100, self.get_y()+10)
        
        # Page number
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def create_table_header(self):
        self.set_fill_color(200, 200, 200)
        self.set_font('Arial', 'B', 10)
        
        # Fixed heights and alignments
        row_height = 10
        self.cell(15, row_height, 'No.', 1, 0, 'C', True)
        self.cell(85, row_height, 'Description', 1, 0, 'C', True)
        self.cell(30, row_height, 'Qty', 1, 0, 'C', True)
        self.cell(30, row_height, 'Unit Price', 1, 0, 'C', True)
        self.cell(30, row_height, 'Amount', 1, 1, 'C', True)

def process_document(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name

        if uploaded_file.type == "application/pdf":
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            st.session_state.document_contents[uploaded_file.name] = "PDF content loaded successfully"
        elif uploaded_file.type == "text/csv":
            df = pd.read_csv(file_path)
            st.session_state.document_contents[uploaded_file.name] = df
            loader = CSVLoader(file_path)
            documents = loader.load()
        else:
            st.error(f"Unsupported file type: {uploaded_file.type}")
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )

        os.unlink(file_path)
        return vector_store

    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return None

def parse_quotation_content(response):
    """Parse the LLM response to extract quotation details"""
    try:
        lines = response.split('\n')
        items = []
        current_item = {}
        
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            if 'Product:' in line:
                if current_item:
                    items.append(current_item)
                product_name = line.split(':', 1)[1].strip()
                current_item = {'description': product_name}
            elif 'Description:' in line:
                desc = line.split(':', 1)[1].strip()
                if desc != "Product not found in catalog":
                    current_item['description'] = desc
            elif 'Price:' in line:
                price_str = line.split(':', 1)[1].strip()
                price = ''.join(filter(lambda x: x.isdigit() or x == '.', price_str.replace(',', '')))
                current_item['price'] = float(price) if price else 0
            elif 'Quantity:' in line:
                qty_str = line.split(':', 1)[1].strip()
                current_item['quantity'] = int(qty_str) if qty_str.isdigit() else 1
        
        if current_item:
            items.append(current_item)
            
        # Filter out products not found in catalog
        items = [item for item in items if item.get('description') != "Product not found in catalog"]
        
        return items
    except Exception as e:
        st.error(f"Error parsing quotation: {str(e)}")
        return []

def generate_quotation(content):
    pdf = QuotationPDF()
    pdf.add_page()
    pdf.create_table_header()
    
    items = parse_quotation_content(content)
    if not items:
        st.error("No valid products found for quotation")
        return None
        
    total_amount = 0
    
    pdf.set_font('Arial', '', 10)
    for idx, item in enumerate(items, 1):
        description = item.get('description', '')
        price = item.get('price', 0)
        quantity = item.get('quantity', 1)
        amount = price * quantity
        total_amount += amount
        
        # Calculate height based on description length
        lines = len(description) // 35 + 1  # Reduced characters per line
        row_height = max(8, lines * 6)
        
        # Save position for horizontal alignment
        start_x = pdf.get_x()
        start_y = pdf.get_y()
        
        # Item number
        pdf.cell(15, row_height, str(idx), 1, 0, 'C')
        
        # Description with word wrap
        pdf.set_xy(start_x + 15, start_y)
        pdf.multi_cell(85, row_height/lines, description, 1, 'L')
        
        # Restore position for remaining cells
        pdf.set_xy(start_x + 100, start_y)
        
        # Quantity, Unit Price, and Amount
        pdf.cell(30, row_height, str(quantity), 1, 0, 'C')
        pdf.cell(30, row_height, f'Rs. {price:,.2f}', 1, 0, 'R')
        pdf.cell(30, row_height, f'Rs. {amount:,.2f}', 1, 1, 'R')
    
    # Add total amount
    pdf.set_font('Arial', 'B', 10)
    pdf.set_fill_color(200, 200, 200)
    pdf.cell(160, 10, 'Total Amount:', 1, 0, 'R', True)
    pdf.cell(30, 10, f'Rs. {total_amount:,.2f}', 1, 1, 'R', True)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        pdf.output(tmp_file.name)
        return tmp_file.name

def main():
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### 📁 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload company documents (PDF/CSV)", 
            type=["pdf", "csv"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.uploaded_files:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        vector_store = process_document(uploaded_file)
                        if vector_store:
                            st.session_state.vector_stores[uploaded_file.name] = vector_store
                            st.session_state.uploaded_files.append(uploaded_file.name)
                            st.success(f"✅ {uploaded_file.name} processed successfully!")

            # Display all document previews
            st.markdown("### 📄 Document Previews")
            for file_name, content in st.session_state.document_contents.items():
                with st.expander(f"Preview: {file_name}"):
                    if isinstance(content, pd.DataFrame):
                        styled_df = content.style.format({
                            'Price': 'Rs. {:,.2f}'.format if 'Price' in content.columns else '{}',
                            'Cost': 'Rs. {:,.2f}'.format if 'Cost' in content.columns else '{}'
                        })
                        st.dataframe(styled_df, use_container_width=True)
                        st.info(f"Total Products: {len(content)}")
                    else:
                        st.info(content)

            # Initialize conversation chain with combined vector stores
            template = """You are a precise and accurate assistant that answers questions based on the provided documents. Your task is to provide exact information from the context.

            Context: {context}
            Question: {question}

            IMPORTANT RULES:
            1. ONLY use information that is explicitly stated in the context
            2. DO NOT make assumptions or inferences
            3. If the exact information is not in the context, respond with "Information not found in the documents"
            4. Keep responses concise and to the point
            5. For product questions:
               - List only products that exactly match the query
               - Include exact prices and specifications as stated in the documents
               - Do not include similar or related products unless specifically asked
            6. For numerical questions:
               - Provide exact numbers from the context
               - Do not round or approximate
            7. Format responses in a clear, bullet-point style
            8. If the question is ambiguous, ask for clarification

            For quotation requests, format your response EXACTLY as follows for each requested product:
            Product: [Exact Product Name from Context]
            Description: [Exact Product Description from Context]
            Price: [Exact Price in INR from Context]
            Quantity: [Requested Quantity]

            Example responses:

            For quotation request "Quote for 2 laptops and 3 monitors":
            Product: Laptop Model X
            Description: 15.6" Laptop with 8GB RAM
            Price: 45000
            Quantity: 2

            Product: Monitor Model Y
            Description: 24" LED Monitor
            Price: 12000
            Quantity: 3

            For general question "What is the price of Laptop X?":
            • Price: Rs. 45,000 (as per document)

            For "List all laptops under 50,000":
            • Laptop Model X
              - Price: Rs. 45,000
              - Specifications: 15.6" display, 8GB RAM
            • Laptop Model Y
              - Price: Rs. 48,000
              - Specifications: 14" display, 16GB RAM

            For unavailable information:
            "Information not found in the documents"

            For ambiguous questions:
            "Please specify which product/feature you are interested in"
            """

            prompt = ChatPromptTemplate.from_template(template)
            
            llm = ChatGroq(
                model_name="allam-2-7b",
                temperature=0.1,  # Lower temperature for more precise responses
                max_tokens=4096
            )
            
            # Combine all vector stores for retrieval
            combined_retriever = None
            if st.session_state.vector_stores:
                retrievers = [vs.as_retriever(search_kwargs={"k": 5}) 
                            for vs in st.session_state.vector_stores.values()]
                # Merge results from all retrievers
                def combined_retrieve(query):
                    all_docs = []
                    for retriever in retrievers:
                        all_docs.extend(retriever.get_relevant_documents(query))
                    return all_docs[:5]  # Return top 5 most relevant documents
                
                combined_retriever = combined_retrieve
            
            if combined_retriever:
                # Create separate chains for quotations and general queries
                st.session_state.quotation_chain = (
                    {"context": combined_retriever, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                st.session_state.general_chain = (
                    {"context": combined_retriever, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )

    with col2:
        st.markdown("### 💬 Chat Interface")
        if hasattr(st.session_state, 'quotation_chain') and hasattr(st.session_state, 'general_chain'):
            user_query = st.text_input("Ask about products or request a quotation:", key="query")
            
            if user_query:
                with st.spinner("Generating response..."):
                    # Determine if the query is for a quotation
                    is_quotation = any(keyword in user_query.lower() for keyword in ['quote', 'quotation', 'price'])
                    
                    # Use appropriate chain based on query type
                    response = st.session_state.quotation_chain.invoke(user_query) if is_quotation else st.session_state.general_chain.invoke(user_query)
                    
                    if is_quotation:
                        st.markdown("### 📊 Quotation Details")
                        st.write(response)
                        
                        pdf_path = generate_quotation(response)
                        with open(pdf_path, "rb") as pdf_file:
                            st.download_button(
                                label="📥 Download Quotation PDF",
                                data=pdf_file,
                                file_name="quotation.pdf",
                                mime="application/pdf"
                            )
                        os.unlink(pdf_path)
                    else:
                        st.markdown("### 💡 Response")
                        st.write(response)

if __name__ == "__main__":
    main()
