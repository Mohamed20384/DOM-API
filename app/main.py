import os
import io
import json
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
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
import logging
import asyncio

# Import our processing function
from app.processing.json_to_pdf import process_restaurants_json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GENAI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GENAI_API_KEY not found in environment variables")

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

# Configuration
class Config:
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 300
    EMBEDDING_BATCH_SIZE = 10
    MAX_PREVIEW_CHARS = 5000
    PDF_FOLDER = "Restaurants_PDF"
    
    # Improved system prompt - will be formatted with restaurant list
    SYSTEM_PROMPT_TEMPLATE = """
    أنت مساعد ذكي اسمك DOM، شغال في تطبيق مطاعم دمياط. بتتكلم باللهجة المصرية، ومن أهل دمياط الجديدة، وعندك خبرة بكل المطاعم اللي فيها.

    دورك إنك تجاوب على أسئلة المستخدمين بناءً على المعلومات اللي موجودة جوه ملفات الـ PDF الخاصة بالمطاعم فقط.

    🎯 قواعد مهمة جداً:

    1. **لازم تجاوب على كل سؤال** - متسيبش أي سؤال من غير رد أبداً
    2. **لو السؤال مش واضح** - قول "ممكن توضح سؤالك أكتر؟"
    3. **لو مفيش معلومات** - قول "معنديش معلومات كافية عن ده"
    4. **لو سؤال عام** - حاول تجاوب بحسب معرفتك العامة وقول انك تقدر تساعد أكتر لو السؤال عن المطاعم

    **الأسئلة عنك شخصيًا**:
    - لو سألوك "إنت مين؟" قول: "أنا DOM، مساعدك الذكي في مطاعم دمياط"
    - متقولش ترتيب المطاعم في ردودك

    **أسئلة الميزانية**:
    - لو قالولك "معايا X جنيه" ادور في القوائم على أكلات مناسبة
    - رتب الاقتراحات من الأرخص للأغلى
    - متتجاوزش الميزانية أبداً

    **المطاعم المتاحة**:
    {restaurant_list_text}

    **مطاعم غير مشتركة في التطبيق**:
    لو المستخدم سأل عن مطعم مش موجود في القائمة، شوف القائمة دي، لو لقيته، قوله إنه مش مشترك، واديه المعلومات اللي عندك عنه:
    {no_eshop_restaurants}

    **طريقة الرد**:
    - باللهجة المصرية دايماً
    - واضح ومختصر
    - مفيد للمستخدم
    - لو مش عارف حاجة قولها صراحة

    **مهم جداً**: اجاوب على كل سؤال، حتى لو كان بسيط زي "إزيك؟" أو "شكراً" - متسيبش حد من غير رد.
    """

    UI_THEME = {
        "primary_color": "#FF4B4B",
        "secondary_color": "#FF9E9E",
        "background_color": "#0E1117",
        "text_color": "#FAFAFA",
        "success_color": "#00D100"
    }

def clean_text(text: str) -> str:
    """Clean and normalize text from PDFs"""
    if not text:
        return ""
    
    lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 1]
    cleaned = '\n'.join(line for line in lines if not line.strip().isdigit())
    return cleaned.replace('-\n', '')

def extract_rank(text: str) -> int:
    """Extract restaurant rank from text"""
    if not text:
        return 999
    
    match = re.search(r'ترتيب المطعم\s*[:：]?\s*(\d+)', text)
    if match:
        return int(match.group(1))
    if "ترتيب المطعم" in text and "غير مصنف" in text:
        return 999
    return 999

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file with better error handling"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
            except Exception as e:
                logger.warning(f"Error reading page {page_num}: {str(e)}")
                continue
        
        return clean_text(text)
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        return ""

def get_restaurant_names_from_folder(folder_path: str) -> List[str]:
    """Extract and return restaurant names sorted by their rank"""
    ranked_restaurants = []

    if not os.path.exists(folder_path):
        logger.warning(f"Folder {folder_path} does not exist")
        return []

    for file in os.listdir(folder_path):
        if file.lower().endswith(".pdf"):
            filepath = os.path.join(folder_path, file)
            try:
                with open(filepath, "rb") as f:
                    text = extract_text_from_pdf(f)
                    if text.strip():
                        rank = extract_rank(text)
                        if rank != 999:  # Exclude unranked
                            name = os.path.splitext(file)[0].strip()
                            ranked_restaurants.append((rank, name))
            except Exception as e:
                logger.error(f"Error reading {file}: {e}")

    # Sort by rank (lower is better)
    ranked_restaurants.sort(key=lambda x: x[0])
    return [name for _, name in ranked_restaurants]

# Initialize restaurant data
Restaurants_PDF = "Restaurants_PDF"
restaurant_names = get_restaurant_names_from_folder(Restaurants_PDF)
restaurant_list_text = "\n".join([f"- {name}" for name in restaurant_names])

# Models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    token_usage: Dict[str, int]

class UploadResponse(BaseModel):
    status: str
    message: str
    pdf_count: int
    no_eshop_count: int

# Helper functions
def count_tokens(text: str) -> int:
    """Estimate token count"""
    if not text:
        return 0
    return max(1, len(text) // 4)

# def load_no_eshop_restaurants(file_path="no_eshop_restaurants.txt") -> List[Dict]:
#     """Load restaurants not in the app"""
#     if not os.path.exists(file_path):
#         return []
#     try:
#         with open(file_path, "r", encoding="utf-8") as f:
#             return [json.loads(line.strip()) for line in f if line.strip()]
#     except Exception as e:
#         logger.error(f"Error loading no-eshop restaurants: {e}")
#         return []
    
def get_formatted_no_eshop_text(file_path="no_eshop_restaurants.txt") -> str:
    """Load and format no-eshop restaurants into a plain text block"""
    if not os.path.exists(file_path):
        return "لا يوجد مطاعم غير مشتركة حالياً."

    try:
        lines = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rest = json.loads(line.strip())
                    name = rest.get("name", "بدون اسم")
                    address = rest.get("address", "غير متوفر")
                    phones = ", ".join(rest.get("phone", [])) or "غير متوفر"
                    bestsell = rest.get("bestsell", "غير معروف")

                    lines.append(
                        f"- {name}:\n"
                        f"  📍 العنوان: {address}\n"
                        f"  ☎️ رقم الهاتف: {phones}\n"
                        f"  🍽️ أشهر الأطباق: {bestsell}"
                    )
        return "\n\n".join(lines) if lines else "لا يوجد مطاعم غير مشتركة حالياً."
    except Exception as e:
        logger.error(f"Error loading no-eshop restaurants: {e}")
        return "حدث خطأ أثناء تحميل المطاعم غير المشتركة."

# LLM and embeddings
def get_llm():
    """Get the language model with optimized settings"""
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",  # Latest model
            google_api_key=GOOGLE_API_KEY, 
            temperature=0.1,  # Very low for consistency
            max_output_tokens=1024,  # Reduced for faster response
            top_p=0.9,
            top_k=40
        )
    except Exception as e:
        logger.error(f"Error creating LLM: {e}")
        # Fallback to older model
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GOOGLE_API_KEY, 
            temperature=0.1,
            max_output_tokens=1024
        )

def get_embeddings():
    """Get embeddings model"""
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
        logger.warning(f"Created {pdf_folder} directory")
        return None, None 

    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_folder}")
        return None, None

    for filename in pdf_files:
        filepath = os.path.join(pdf_folder, filename)
        try:
            with open(filepath, 'rb') as f:
                text = extract_text_from_pdf(f)
                if text.strip():
                    rank = extract_rank(text)
                    if rank == 999:
                        logger.info(f"Skipping unranked restaurant: {filename}")
                        continue
                    documents.append((filename, text))
                    logger.info(f"Processed {filename} with rank {rank}")
                else:
                    logger.warning(f"No readable text in {filename}")
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")

    if not documents:
        logger.error("No valid PDF files found")
        return None, None 

    texts = []
    metadatas = []

    for filename, text in documents:
        rank = extract_rank(text)
        texts.append(text)
        metadatas.append({
            "source": filename,
            "rank": rank
        })

    try:
        embeddings = get_embeddings()
        vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas
        )
        logger.info(f"Created vectorstore with {len(documents)} documents")
        return vectorstore, documents
    except Exception as e:
        logger.error(f"Error creating vectorstore: {e}")
        return None, None

def format_docs(docs):
    """Format documents for context"""
    if not docs:
        return "لا توجد معلومات متاحة"
    
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "مصدر غير معروف")
        content = doc.page_content.strip()
        if content:
            formatted.append(f"من {source}:\n{content}")
    
    return "\n\n".join(formatted) if formatted else "لا توجد معلومات مفيدة"

def get_compressed_retriever(base_retriever):
    """Get compressed retriever"""
    try:
        embeddings = get_embeddings()
        compressor = EmbeddingsFilter(
            embeddings=embeddings, 
            similarity_threshold=0.6  # Lower for better recall
        )
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
    except Exception as e:
        logger.error(f"Error creating compressed retriever: {e}")
        return base_retriever

def get_retriever(vectorstore, documents):
    """Get ensemble retriever"""
    try:
        # BM25 retriever
        bm25_retriever = BM25Retriever.from_texts(
            [doc[1] for doc in documents],
            metadatas=[{"source": doc[0], "rank": extract_rank(doc[1])} for doc in documents]
        )
        bm25_retriever.k = 5

        # FAISS retriever
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # Ensemble retriever
        ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=[0.5, 0.5]
        )

        return get_compressed_retriever(ensemble)
    except Exception as e:
        logger.error(f"Error creating retriever: {e}")
        return vectorstore.as_retriever(search_kwargs={"k": 5})

def create_fallback_response(question: str) -> str:
    """Create fallback response when main system fails"""
    question_lower = question.lower()
    
    # Greetings
    if any(word in question_lower for word in ["أهلا", "مرحبا", "السلام", "إزيك", "ازيك", "هاي", "صباح", "مساء"]):
        return "أهلاً وسهلاً! أنا DOM مساعدك في مطاعم دمياط. إيه اللي تحب تعرفه عن المطاعم؟"
    
    # Thanks
    if any(word in question_lower for word in ["شكرا", "شكراً", "تسلم", "ربنا يخليك"]):
        return "العفو! أي خدمة تاني في المطاعم؟"
    
    # Who are you
    if any(word in question_lower for word in ["مين", "منين", "إيه", "ايه", "اسمك"]):
        return "أنا DOM، مساعدك الذكي في مطاعم دمياط. أقدر أساعدك تلاقي أحسن المطاعم والأكلات."
    
    # Numbers only
    if question.strip().isdigit():
        number = int(question.strip())
        if number <= 500:
            return f"لو معاك {number} جنيه، أقدر أساعدك تلاقي أكلات مناسبة للمبلغ ده. بس محتاج معلومات أكتر عن المطاعم المتاحة."
        else:
            return f"المبلغ {number} جنيه كويس جداً! أقدر أساعدك تختار من مطاعم كتير، بس محتاج معلومات أكتر عن المطاعم المتاحة."
    
    # Default fallback
    return "معذرة، مش قادر أجاوب على السؤال ده دلوقتي. ممكن تسأل عن المطاعم أو الأكلات المتاحة؟"

def get_conversation_chain(retriever):
    """Get conversation chain with improved prompt"""
    if not retriever:
        logger.error("No retriever provided")
        return None
    
    try:
        llm = get_llm()
        
        # Format system prompt with restaurant list
        system_prompt = Config.SYSTEM_PROMPT_TEMPLATE.format(
            restaurant_list_text=restaurant_list_text,
            no_eshop_restaurants=no_eshop_restaurants
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """
            السؤال: {question}

            المعلومات المتاحة من المطاعم:
            {context}

            تعليمات:
            1. اقرأ المعلومات بعناية
            2. لو المعلومات كافية، اجاوب بناء عليها
            3. لو المعلومات مش كافية، اعتذر وقول المعلومة مش متاحة
            4. لو السؤال عام (مش عن المطاعم)، اجاوب بمعرفتك العامة
            5. الرد لازم يكون باللهجة المصرية
            6. لازم تجاوب على كل سؤال - متسيبش حد من غير رد

            الرد:
            """)
        ])

        def safe_retrieve_and_format(question: str) -> str:
            """Safely retrieve and format context"""
            try:
                docs = retriever.get_relevant_documents(question)
                if not docs:
                    return "لا توجد معلومات متاحة في قاعدة بيانات المطاعم"
                
                # Sort by rank
                sorted_docs = sorted(docs, key=lambda d: d.metadata.get("rank", 999))
                return format_docs(sorted_docs)
            except Exception as e:
                logger.error(f"Error in retrieval: {e}")
                return "حدث خطأ في البحث عن المعلومات"

        chain = (
            RunnableMap({
                "question": lambda x: x["question"],
                "context": lambda x: safe_retrieve_and_format(x["question"])
            })
            | prompt
            | llm
            | StrOutputParser()
        )

        return chain
    except Exception as e:
        logger.error(f"Error creating conversation chain: {e}")
        return None

# Initialize system
logger.info("Initializing system...")
vectorstore, documents = get_vectorstore()
# no_eshop_restaurants = load_no_eshop_restaurants()

no_eshop_restaurants = get_formatted_no_eshop_text()

if vectorstore and documents:
    retriever = get_retriever(vectorstore, documents)
    conversation_chain = get_conversation_chain(retriever)
    logger.info(f"System initialized with {len(documents)} documents")
else:
    retriever = None
    conversation_chain = None
    logger.warning("System initialization failed")

# Update system prompt with current restaurant list
Config.SYSTEM_PROMPT = Config.SYSTEM_PROMPT_TEMPLATE.format(
    restaurant_list_text=restaurant_list_text,
    no_eshop_restaurants=no_eshop_restaurants
)

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
        {"role": "assistant", "content": "أهلاً وسهلاً! أنا DOM مساعدك الذكي للمطاعم في دمياط الجديدة. ممكن أساعدك بإيه النهاردة؟"}
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
        if not file.filename.lower().endswith('.json'):
            raise HTTPException(status_code=400, detail="يجب أن يكون الملف من نوع JSON")
        
        # Save temporarily
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="الملف فارغ")
            f.write(content)
        
        # Process JSON
        result = process_restaurants_json(file_path)
        os.remove(file_path)
        
        # Reinitialize system
        global vectorstore, documents, retriever, conversation_chain, restaurant_names, restaurant_list_text
        vectorstore, documents = get_vectorstore()
        restaurant_names = get_restaurant_names_from_folder(Restaurants_PDF)
        restaurant_list_text = "\n".join([f"- {name}" for name in restaurant_names])
        
        if vectorstore and documents:
            retriever = get_retriever(vectorstore, documents)
            conversation_chain = get_conversation_chain(retriever)
            # Update system prompt
            Config.SYSTEM_PROMPT = Config.SYSTEM_PROMPT_TEMPLATE.format(
                restaurant_list_text=restaurant_list_text,
                no_eshop_restaurants=no_eshop_restaurants  # ← ADD THIS
            )
            logger.info("System reinitialized successfully")
        
        return {
            "status": "success",
            "message": "تم معالجة ملف JSON بنجاح",
            "pdf_count": result["pdf_count"],
            "no_eshop_count": result["no_eshop_count"]
        }
    except Exception as e:
        logger.error(f"Error processing JSON: {str(e)}")
        return {
            "status": "error",
            "message": f"حدث خطأ أثناء معالجة الملف: {str(e)}"
        }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    try:
        # Validate input
        if not chat_request.question or not chat_request.question.strip():
            return ChatResponse(
                answer="من فضلك اكتب سؤالك",
                sources=[],
                token_usage={"total": 0}
            )
        
        question = chat_request.question.strip()
        logger.info(f"Processing question: {question}")
        
        # Check for no-eshop restaurants first
        # for rest in no_eshop_restaurants:
        #     if rest["name"] in question:
        #         info = f"كل اللي أعرفه عن المطعم '{rest['name']}':\n"
        #         info += f"- 📍 العنوان: {rest.get('address', 'غير متوفر')}\n"
        #         info += f"- ☎️ رقم الهاتف: {', '.join(rest.get('phone', [])) or 'غير متوفر'}\n"
        #         info += f"- 🍽️ أشهر الأطباق: {rest.get('bestsell', 'غير معروف')}\n"
                
        #         return ChatResponse(
        #             answer=f"للأسف المطعم '{rest['name']}' مش مشترك في أبلكيشن مطاعم دمياط.\n{info}",
        #             sources=[],
        #             token_usage={"total": count_tokens(question + info)}
        #         )
        
        # Try to get response from conversation chain
        response = None
        sources = []
        context_text = ""
        
        if conversation_chain:
            try:
                # Get response from chain
                response = await conversation_chain.ainvoke({"question": question})
                
                # Get sources
                if retriever:
                    try:
                        relevant_docs = retriever.get_relevant_documents(question)
                        if relevant_docs:
                            sorted_docs = sorted(relevant_docs, key=lambda d: d.metadata.get("rank", 999))
                            sources = [doc.metadata.get("source", "unknown") for doc in sorted_docs[:5]]
                            sources = list(dict.fromkeys(sources))  # Remove duplicates
                            context_text = format_docs(sorted_docs)
                    except Exception as e:
                        logger.error(f"Error getting sources: {e}")
                        sources = []
                        context_text = "Error retrieving context"
                
            except Exception as e:
                logger.error(f"Error with conversation chain: {e}")
                response = None
        
        # If no response from chain, use fallback
        if not response or not response.strip():
            logger.warning("No response from chain, using fallback")
            response = create_fallback_response(question)
        
        # Calculate token usage
        system_prompt = Config.SYSTEM_PROMPT
        token_usage = {
            "question": count_tokens(question),
            "context": count_tokens(context_text),
            "system": count_tokens(system_prompt),
            "response": count_tokens(response),
            "total": (
                count_tokens(question) + 
                count_tokens(context_text) + 
                count_tokens(system_prompt) + 
                count_tokens(response)
            )
        }
        
        logger.info(f"Response generated successfully with {len(sources)} sources")
        
        return ChatResponse(
            answer=response,
            sources=sources,
            token_usage=token_usage
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {str(e)}")
        fallback_response = create_fallback_response(chat_request.question)
        return ChatResponse(
            answer=fallback_response,
            sources=[],
            token_usage={"total": count_tokens(fallback_response)}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vectorstore_ready": vectorstore is not None,
        "conversation_chain_ready": conversation_chain is not None,
        "num_documents": len(documents) if documents else 0,
        "num_restaurants": len(restaurant_names),
        "restaurant_names": restaurant_names[:5]  # Show first 5 for debugging
    }

@app.get("/debug")
async def debug_info():
    """Debug endpoint to check system state"""
    return {
        "pdf_folder_exists": os.path.exists(Config.PDF_FOLDER),
        "pdf_files": [f for f in os.listdir(Config.PDF_FOLDER) if f.endswith('.pdf')] if os.path.exists(Config.PDF_FOLDER) else [],
        "vectorstore_ready": vectorstore is not None,
        "conversation_chain_ready": conversation_chain is not None,
        "num_documents": len(documents) if documents else 0,
        "num_restaurants": len(restaurant_names),
        "google_api_key_set": bool(GOOGLE_API_KEY),
        "restaurant_names": restaurant_names
    }

# Test endpoint
@app.get("/test-response")
async def test_response():
    """Test endpoint to verify system is working"""
    test_question = "مين أنت؟"
    
    if conversation_chain:
        try:
            response = await conversation_chain.ainvoke({"question": test_question})
            return {"status": "success", "response": response}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    else:
        return {"status": "error", "error": "Conversation chain not initialized"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
