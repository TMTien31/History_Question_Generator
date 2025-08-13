# History Question Generator

An intelligent FastAPI-based web application that automatically generates Vietnamese multiple-choice history questions and answers from PDF documents using Google Gemini 2.0 and LangChain RAG (Retrieval-Augmented Generation) architecture. The system processes historical documents to create comprehensive question-answer pairs for history exam preparation and educational testing.

## Technology Stack

- **Backend**: FastAPI with Uvicorn server
- **Language Model**: Google Gemini 2.0 Flash Lite
- **Document Processing**: LangChain, PyPDFLoader, PyPDF2
- **Embeddings**: HuggingFace sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
- **Vector Database**: FAISS
- **Text Processing**: TokenTextSplitter with GPT-3.5-turbo tokenization
- **Frontend**: HTML/CSS with Jinja2 templating
- **File Handling**: Async file operations with aiofiles

## Installation & Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/TMTien31/History_Question_Generator.git
cd History_Question_Generator
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Get Google Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the generated key

### Step 5: Create Environment File
Create a `.env` file in the project root directory:
```bash
# Create .env file
touch .env  # On macOS/Linux
# or create manually on Windows
```

Add your API key to the .env file:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### Step 6: Prepare Documents
- Place your PDF documents (historical materials, textbooks) in the `data/` folder
- The system supports PDFs up to 5 pages for optimal processing
- Documents should contain Vietnamese historical content for best results

## Running the Application

### Start the FastAPI Server
```bash
python app.py
```

### Access the Application
1. Open your web browser
2. Navigate to: http://0.0.0.0:8080 or http://localhost:8080[1]
3. You should see the "Hệ thống tạo câu trắc nghiệm lý thuyết" interface

## How It Works

### 1. Document Upload
- Upload PDF files through the web interface
- Files are stored in `static/docs/` directory
- Maximum recommended: 5 pages per document

### 2. Question Generation Process
- **Text Extraction**: PyPDFLoader extracts content from PDF documents
- **Text Chunking**: TokenTextSplitter divides content into manageable chunks (40,000 tokens for questions, 2,000 for answers)
- **Question Creation**: Google Gemini 2.0 generates 10 Vietnamese multiple-choice questions using LangChain's refine summarization chain
- **Vector Store**: FAISS creates embeddings using HuggingFace transformers for efficient answer retrieval
- **Answer Generation**: RetrievalQA chain generates accurate answers based on document context
### 3. Output
- Questions and answers are saved as CSV files in `static/output/QA.csv`
- Each question includes four answer choices (A, B, C, D) with one correct answer
- Output format ready for educational use and exam preparation

## Project Structure

```
History_Question_Generator/
├── app.py                     # FastAPI application with upload and analysis endpoints
├── src/
│   ├── helper.py             # Core processing pipeline with LangChain integration
│   └── prompt.py             # Prompt templates for Vietnamese question generation
├── templates/
│   └── index.html            # Web interface for document upload and processing
├── static/                   # Static assets including uploaded docs and outputs
├── data/                     # Sample PDF documents for testing
├── research/
│   └── experiment.ipynb      # Jupyter notebook for experimentation
├── requirements.txt          # Python dependencies
├── questions_answers.csv     # Sample output with generated Q&A pairs
└── README.md
```

## Sample Output

The system generates Vietnamese multiple-choice questions like:
- Questions about World War II events, battles, and historical figures.
- Comprehensive coverage of dates, people, and key historical concepts.
- Four-option format with clear correct answers.
- Professional formatting suitable for educational assessments.
