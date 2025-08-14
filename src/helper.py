from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from src.prompt import prompt_template, refine_template, answer_prompt
import os
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

def file_processing(file_path):

    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = ''

    for page in data:
        question_gen += page.page_content

    splitter_ques_gen = TokenTextSplitter(
      model_name= "gpt-3.5-turbo",
      chunk_size= 40000,
      chunk_overlap = 500
    )

    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)

    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    '''
    splitter_ans_gen = TokenTextSplitter(
      model_name = 'gpt-3.5-turbo',
      chunk_size = 2000,
      chunk_overlap = 200
    )

    document_answer_gen = splitter_ans_gen.split_documents(
        document_ques_gen
    )
    '''

    #new splitter for answer generation
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?"],  # ưu tiên ngắt theo đoạn/câu
        chunk_size=2000,    # token approx
        chunk_overlap=200,
    )

    chunks = splitter.split_text(document_ques_gen[0].page_content)
    document_answer_gen_new_splitter = [Document(t) for t in chunks]

    return document_ques_gen, document_answer_gen_new_splitter

def llm_pipeline(file_path):

    document_ques_gen, document_answer_gen = file_processing(file_path)

    llm_ques_gen_pipeline = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,
    )

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    ques_gen_chain = load_summarize_chain(llm = llm_ques_gen_pipeline, 
                                            chain_type = "refine", 
                                            verbose = True, 
                                            question_prompt=PROMPT_QUESTIONS, 
                                            refine_prompt=REFINE_PROMPT_QUESTIONS)

    ques = ques_gen_chain.run(document_ques_gen)
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    llm_answer_gen = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,
    )

    ques_list = ques.split("\n")
    filtered_ques_list = [element for element in ques_list]

    '''
    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, 
                                                chain_type="stuff", 
                                                retriever=vector_store.as_retriever())
    '''

    

    new_answer_generation_chain = RetrievalQA.from_chain_type(
        llm=llm_answer_gen,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": answer_prompt}  
)
    return new_answer_generation_chain, filtered_ques_list
