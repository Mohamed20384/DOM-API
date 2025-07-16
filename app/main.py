import os
import io
import json
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import PyPDF2
import google.generativeai as genai
from langchain.text_splitter import CharacterTextSplitter

from langchain_core.runnables import RunnableMap, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Import our processing function
from app.processing.json_to_pdf import process_restaurants_json

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GENAI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

app = FastAPI(title="DOM - مساعد المطاعم المصري")
templates = Jinja2Templates(directory="app/templates")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only! Restrict in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration (same as before)
class Config:
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 300
    EMBEDDING_BATCH_SIZE = 10
    MAX_PREVIEW_CHARS = 5000
    PDF_FOLDER = "Restaurants_PDF"
    SYSTEM_PROMPT = """
    أنت مساعد ذكي اسمك DOM، شغال في تطبيق مطاعم دمياط. بتتكلم باللهجة المصرية، ومن أهل دمياط الجديدة، وعندك خبرة بكل المطاعم اللي فيها.

    دورك إنك تجاوب على أسئلة المستخدمين بناءً على المعلومات اللي موجودة جوه ملفات الـ PDF الخاصة بالمطاعم فقط، واللي فيها القوائم والأسعار والتفاصيل.

    🎯 التعليمات المهمة:

    1. **الأسئلة عنك شخصيًا**:
    - لو المستخدم سألك "إنت مين؟" أو "عرفني بنفسك"، ساعتها بس رد وقل:  
        → "أنا DOM، مساعدك الذكي في مطاعم دمياط".
    - غير كده، متعرفش عن نفسك، وادخل في الموضوع على طول.
    - دايمًا خليك فاكر إن ترتيب المطاعم (rank) مهم، وكل ما الرقم كان أقل، كل ما المطعم كان أحسن.
    - متقولش ترتيب المطعم في الرد، يعني متقولش (ترتيبه 1) أو (ترتيبه 3).
    - حتى لو المستخدم سأل عن الترتيب، قول إن المعلومة دي مش متاحة.

    2. **الردود**:
    - لازم تكون باللهجة المصرية، بأسلوب بسيط وواضح.
    - ماتستخدمش كلام فصيح أو رسمي.
    - متزودش معلومات مش موجودة في ملفات المطاعم.
    - لو مش لاقي إجابة في الملفات، قول:
        → "معنديش المعلومات دي للأسف".

    3. **البيانات اللي تركز عليها**:
    - العناوين
    - الأسعار
    - الأكلات
    - التقييمات
    - المميزات الخاصة لكل مطعم

    4. **لو المستخدم سأل عن عدد أو أسماء المطاعم**:
    - رد كده:
        → "المطاعم اللي عندي حالياً هي:
        {restaurant_list_text}"

    5. **لو المستخدم قال حاجة زي**:
    - "معايا 100 جنيه آكل إيه؟"
    - "عايز أطلب أكل بـ 150 جنيه"
    - "أنا ومراتي معانا 200 جنيه ناكل إيه؟"

    🧠 دورك إنك تدور في كل القوائم اللي عندك وتطلعله اقتراحات لأكلات إجمالي سعرها أقل من أو يساوي المبلغ اللي قاله.

    ✅ التعليمات وقتها:
    - دور في القوائم على أكلات أسعارها مناسبة للمبلغ.
    - حاول تختارله أكتر من صنف (وجبة + مشروب مثلًا).
    - لو لقيت وجبة كاملة من مطعم واحد، يبقى ده الأفضل.
    - رتب الأكلات من الأرخص للأغلى.
    - ماينفعش الإجمالي يتخطى الميزانية.
    - لو ملقتش حاجة مناسبة، قوله:
        → "مفيش أكل مناسب للمبلغ ده حالياً".

    ✅ مثال للرد:
    "لو معاك 100 جنيه، ممكن تطلب من مطعم الشامي:
    - شاورما فراخ بـ 50 جنيه
    - بطاطس بـ 20 جنيه
    - بيبسي بـ 15 جنيه
    الإجمالي: 85 جنيه"
    """

    # SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.format(restaurant_list_text=restaurant_list_text)

    UI_THEME = {
        "primary_color": "#FF4B4B",
        "secondary_color": "#FF9E9E",
        "background_color": "#0E1117",
        "text_color": "#FAFAFA",
        "success_color": "#00D100"
    }

def clean_text(text: str) -> str:
    lines = [line for line in text.split('\n') if len(line.strip()) > 1]
    # lines = [line for line in lines if not re.search(r'ترتيب المطعم\s*[:：]?\s*(\d+|غير مصنف)', line)]
    cleaned = '\n'.join(line for line in lines if not line.strip().isdigit())
    return cleaned.replace('-\n', '')

# 1. Define extract_rank
def extract_rank(text: str) -> int:
    match = re.search(r'ترتيب المطعم\s*[:：]?\s*(\d+)', text)
    if match:
        return int(match.group(1))
    if "ترتيب المطعم" in text and "غير مصنف" in text:
        return 999
    return 999

# 2. Define extract_text_from_pdf
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)
        return clean_text(text)
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return ""

def get_restaurant_names_from_folder(folder_path: str) -> List[str]:
    """Extract and return restaurant names sorted by their rank, excluding unranked ones."""
    ranked_restaurants = []

    if not os.path.exists(folder_path):
        return []

    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            filepath = os.path.join(folder_path, file)
            try:
                with open(filepath, "rb") as f:
                    text = extract_text_from_pdf(f)
                    rank = extract_rank(text)
                    if rank != 999:  # Exclude 'غير مصنف'
                        name = os.path.splitext(file)[0].strip()
                        ranked_restaurants.append((rank, name))
            except Exception as e:
                print(f"❌ Error reading {file}: {e}")

    # Sort by rank (lower is better)
    ranked_restaurants.sort(key=lambda x: x[0])

    # Return only the names
    return [name for _, name in ranked_restaurants]


Restaurants_PDF = "Restaurants_PDF"
restaurant_names = get_restaurant_names_from_folder(Restaurants_PDF)

# Format for SYSTEM_PROMPT
restaurant_list_text = "\n".join([f"- {name}" for name in restaurant_names])
Config.SYSTEM_PROMPT = Config.SYSTEM_PROMPT.format(restaurant_list_text=restaurant_list_text)

# Models (same as before)
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    # chat_history: List[Message]

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    token_usage: Dict[str, int]

class UploadResponse(BaseModel):
    status: str
    message: str
    pdf_count: int
    no_eshop_count: int

# Helper functions (same as before)
def count_tokens(text: str) -> int:
    return max(1, len(text) // 4)

# def load_no_eshop_restaurants(file_path: str = "no_eshop_restaurants.txt") -> List[str]:
#     if not os.path.exists(file_path):
#         return []
#     with open(file_path, "r", encoding="utf-8") as f:
#         return [line.strip() for line in f if line.strip()]
    
def load_no_eshop_restaurants(file_path="no_eshop_restaurants.txt") -> List[Dict]:
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

# Cached resources (same as before)
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY, 
        temperature=0.7,
        max_output_tokens=2048
    )

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY,
        task_type="retrieval_document"
    )

def get_vectorstore():
    """Create and return the FAISS vectorstore from PDF files"""
    documents = []
    pdf_folder = Config.PDF_FOLDER

    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
        return None, None 

    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith('.pdf'):
            filepath = os.path.join(pdf_folder, filename)
            try:
                with open(filepath, 'rb') as f:
                    text = extract_text_from_pdf(f)
                    if text.strip():
                        rank = extract_rank(text)
                        if rank == 999:
                            print(f"⏩ تخطي المطعم '{filename}' لأنه غير مصنف.")
                            continue  # Skip this restaurant
                        documents.append((filename, text))
                    else:
                        print(f"📄 ملف {filename} مفيهوش نص مقروء.")
            except Exception as e:
                print(f"❌ حصل خطأ مع الملف {filename}: {str(e)}")

    if not documents:
        print("⚠️ مفيش ملفات PDF صالحة للقراءة")
        return None, None 

    # texts = [text for _, text in documents]
    # metadatas = [{"source": filename} for filename, _ in documents]

    texts = []
    metadatas = []

    for filename, text in documents:
        rank = extract_rank(text)
        texts.append(text)
        metadatas.append({
            "source": filename,
            "rank": rank
        })


    embeddings = get_embeddings()
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    return vectorstore, documents

def format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        content = doc.page_content
        formatted.append(f"📄 المصدر: {source}\n📝 المحتوى:\n{content}\n{'='*50}")
    return "\n\n".join(formatted)

def get_compressed_retriever(base_retriever):
    embeddings = get_embeddings()
    compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.76)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

def get_retriever(vectorstore, documents):
    bm25_retriever = BM25Retriever.from_texts(
        [doc[1] for doc in documents],
        metadatas=[{"source": doc[0]} for doc in documents]
    )
    bm25_retriever.k = 5

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.4, 0.6]
    )

    return get_compressed_retriever(ensemble)

def get_conversation_chain(retriever):
    # memory = ConversationBufferMemory(
    #     memory_key="chat_history",
    #     return_messages=True,
    #     k=3,
    #     output_key='answer',
    # )
    
    llm = get_llm()
    
    # Old prompt (includes chat history — more expensive)
    # prompt_template = ChatPromptTemplate.from_messages([
    #     ("system", Config.SYSTEM_PROMPT),
    #     ("user", "السؤال السابق:\n{chat_history}\n\nالسؤال الحالي:\n{question}\n\nالمعلومات ذات الصلة:\n{context}\n\nجاوب بالعربية المصرية:")
    # ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", Config.SYSTEM_PROMPT),
        ("human", """
        السؤال: {question}

        المعلومات ذات الصلة:
        {context}

        الرد باللهجة المصرية، وخلّي الرد مختصر ومفيد. ما تتكلمش كتير.
        """)
    ])

    chain = (
        RunnableMap({
            "question": lambda x: x["question"],
            "context": lambda x: retriever.get_relevant_documents(x["question"])
        })
        | RunnableLambda(lambda x: {
            "question": x["question"],
            "context": "\n\n".join([doc.page_content[:400] for doc in x["context"]])
        })
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# Initialize the vectorstore and conversation chain
vectorstore, documents = get_vectorstore()
no_eshop_restaurants = load_no_eshop_restaurants()
if vectorstore and documents:
    retriever = get_retriever(vectorstore, documents)
    conversation_chain = get_conversation_chain(retriever)
else:
    retriever = None
    conversation_chain = None

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("upload.html", {
        "request": request,
        "ui_theme": Config.UI_THEME
    })

@app.get("/chat", response_class=HTMLResponse)
async def chat_interface(request: Request):
    initial_messages = [
        {"role": "assistant", "content": "أهلاً وسهلاً! أنا مساعدك الذكي للمطاعم في دمياط الجديدة. ممكن أساعدك بإيه النهاردة؟"}
    ]
    return templates.TemplateResponse("chat.html", {
        "request": request,
        "initial_messages": initial_messages,
        "restaurant_names": restaurant_names,
        "num_restaurants": len(documents) if documents else 0,
        "ui_theme": Config.UI_THEME
    })

@app.post("/api/upload-json")
async def upload_json(file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Process the JSON file
        result = process_restaurants_json(file_path)
        
        # Clean up
        os.remove(file_path)
        
        # Reinitialize the vectorstore with new PDFs
        global vectorstore, documents, retriever, conversation_chain
        vectorstore, documents = get_vectorstore()
        if vectorstore and documents:
            retriever = get_retriever(vectorstore, documents)
            conversation_chain = get_conversation_chain(retriever)
        
        return {
            "status": "success",
            "message": "تم معالجة ملف JSON بنجاح",
            "pdf_count": result["pdf_count"],
            "no_eshop_count": result["no_eshop_count"]
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"حدث خطأ أثناء معالجة الملف: {str(e)}"
        }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    if not conversation_chain:
        return ChatResponse(
            answer="⚠️ النظام غير جاهز بعد. يرجى تحميل ملف JSON أولاً.",
            sources=[],
            token_usage={}
        )
    
    question = chat_request.question
        
    for rest in no_eshop_restaurants:
        if rest["name"] in question:
            info = f"كل اللي أعرفه عن المطعم '{rest['name']}':\n"
            info += f"- 📍 العنوان: {rest.get('address', 'غير متوفر')}\n"
            info += f"- ☎️ رقم الهاتف: {', '.join(rest.get('phone', [])) or 'غير متوفر'}\n"
            info += f"- 🍽️ أشهر الأطباق: {rest.get('bestsell', 'غير معروف')}\n\n"
            return ChatResponse(
                answer=f"للأسف المطعم '{rest['name']}' مش مشترك في أبلكيشن مطاعم دمياط.\n{info}",
                sources=[],
                token_usage={}
            )
    
    # Process the question
    # result = conversation_chain({"question": question})
    response = await conversation_chain.ainvoke({"question": question})

    # response = result["answer"]
    
    # Get relevant documents
    relevant_docs = retriever.get_relevant_documents(question)

    # 🔼 Sort by rank (lowest = best)
    sorted_docs = sorted(relevant_docs, key=lambda d: int(d.metadata.get("rank", 999)))

    sources = [doc.metadata.get("source", "unknown") for doc in sorted_docs]
    sources = list(dict.fromkeys(sources)) 
    
    # Calculate token usage
    # chat_history_str = "\n".join(
    #     [f"{type(m).__name__}: {m.content}" for m in conversation_chain.memory.chat_memory.messages]
    # )
    
    token_usage = {
        "question": count_tokens(question),
        "context": count_tokens(format_docs(sorted_docs)),
        "system": count_tokens(Config.SYSTEM_PROMPT),
        "response": count_tokens(response),
        "total": (
            count_tokens(question) + 
            count_tokens(format_docs(sorted_docs)) + 
            count_tokens(Config.SYSTEM_PROMPT) + 
            count_tokens(response)
        )
    }
    
    return ChatResponse(
        answer=response,
        sources=sources,
        token_usage=token_usage
    )
