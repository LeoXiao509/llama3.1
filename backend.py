import os
import time
import pickle
import hashlib
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import docx  # 主要處理docx文件
import logging
import torch
import faiss
import numpy as np

# 環境初始化
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 常數設定
PROCESSED_DATA_PATH = "processed_data.pkl"
DEFAULT_CSV_PATH = "goat_data.csv"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 160
SIMILARITY_THRESHOLD = 0.3
TOP_K_RESULTS = 20

# 初始化模型
def init_system():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError("GROQ_API_KEY 未設定於環境變數")
    try:
        groq_client = Groq(api_key=groq_api_key)
        groq_client.models.list()
    except Exception as e:
        raise ConnectionError(f"Groq 連線失敗: {e}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(
        'paraphrase-multilingual-MiniLM-L12-v2',
        device=device
    )
    return groq_client, embedder

groq_client, embedder = init_system()

# 資料處理函式
def process_docx(file) -> list:
    """處理Word文件"""
    try:
        doc = docx.Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
        return split_text(text)
    except Exception as e:
        logger.error(f"DOCX處理失敗: {str(e)}")
        raise ValueError(f"DOCX解析錯誤: {file.name}")

def process_txt(file) -> list:
    """處理純文字文件"""
    try:
        text = file.read().decode("utf-8")
        return split_text(text)
    except Exception as e:
        logger.error(f"TXT處理失敗: {str(e)}")
        raise ValueError(f"TXT解析錯誤: {file.name}")

def process_csv(file) -> list:
    """處理CSV文件"""
    try:
        df = pd.read_csv(file)
        if "context" not in df.columns:
            raise ValueError("CSV缺少必要欄位: context")
        return df["context"].dropna().astype(str).str.strip().tolist()
    except Exception as e:
        logger.error(f"CSV處理失敗: {str(e)}")
        raise ValueError(f"CSV解析錯誤: {file.name}")

def split_text(text: str) -> list:
    """文字分割函數"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "。", "！", "？", "\n", " "]
    )
    return [chunk.strip() for chunk in splitter.split_text(text) if chunk.strip()]

def get_file_hash(file) -> str:
    """計算檔案雜湊值"""
    return hashlib.md5(file.getvalue()).hexdigest()

class KnowledgeBase:
    def __init__(self):
        self.contexts = []
        self.processed_files = set()
        self.index = None
        self.embedder = embedder

        # 載入預處理資料
        if os.path.exists(PROCESSED_DATA_PATH):
            self._load_data()
            
        if os.path.exists(DEFAULT_CSV_PATH):
            self._load_default_csv()
        
    def _load_data(self):
        """載入資料"""
        try:
            if os.path.exists(PROCESSED_DATA_PATH):
                with open(PROCESSED_DATA_PATH, "rb") as f:
                    data = pickle.load(f)
                    self.contexts = data["contexts"]
                    self.processed_files = data["processed_files"]
                    if "faiss_index" in data and data["faiss_index"]is not None:
                        self.index = faiss.deserialize_index(data["faiss_index"])
                    elif self.contexts:
                        self._build_index()
                
        except Exception as e:
            logger.error(f"載入處理資料失敗: {str(e)}")
            self.contexts = []
            self.processed_files = set()

    def _load_default_csv(self):
        """載入預設CSV檔案"""
        try:
            df = pd.read_csv(DEFAULT_CSV_PATH)
            if "context" in df.columns:
                new_data = [str(c).strip() for c in df["context"].dropna() 
                          if str(c).strip() and str(c).strip() not in self.contexts]
                if new_data:
                    self.contexts.extend(new_data)
                    logger.info(f"已載入 {len(new_data)} 條預設資料")
        except Exception as e:
            logger.error(f"預設CSV載入失敗: {str(e)}")

    def _build_index(self):
        """建立 FAISS 索引"""
        if not self.contexts:
            return
            
        try:
            # 添加進度追蹤
            logger.info("開始建立嵌入向量...")
            embeddings = self.embedder.encode(
                self.contexts,
                show_progress_bar=True,
                convert_to_tensor=False,
                normalize_embeddings=True
            ).astype('float32')
            
            dimension = embeddings.shape[1]
            
            # 更智能的索引類型選擇
            if len(self.contexts) < 5_000:
                self.index = faiss.IndexFlatIP(dimension)
                logger.info("使用Flat索引")
            else:
                nlist = min(256, len(self.contexts) // 100)
                quantizer = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                if not self.index.is_trained:
                    logger.info(f"訓練IVF索引，nlist={nlist}")
                    self.index.train(embeddings)
                
            self.index.add(embeddings)
            logger.info(f"索引建立完成，包含 {self.index.ntotal} 條資料")
        except Exception as e:
            logger.error(f"建立索引失敗: {str(e)}", exc_info=True)
            self.index = None

    def _save_data(self):
        """保存資料"""
        try:
            data = {
                "contexts": self.contexts,
                "processed_files": self.processed_files,
                "faiss_index": faiss.serialize_index(self.index) if self.index else None
            }
            with open(PROCESSED_DATA_PATH, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.error(f"保存資料失敗: {str(e)}")

    def add_files(self, uploaded_files):
        """處理上傳檔案"""
        new_contexts = []
        for file in uploaded_files:
            try:
                if not file or not file.filename:
                    continue
                    
                file_hash = get_file_hash(file)
                if file_hash in self.processed_files:
                    logger.info(f"跳過已處理檔案: {file.filename}")
                    continue
                
                logger.info(f"正在處理檔案: {file.filename}")
                
                if file.filename.lower().endswith('.docx'):
                    processed = process_docx(file)
                elif file.filename.lower().endswith('.txt'):
                    processed = process_txt(file)
                elif file.filename.lower().endswith('.csv'):
                    processed = process_csv(file)
                else:
                    logger.warning(f"不支援的檔案類型: {file.filename}")
                    continue
                
                if not processed:
                    logger.warning(f"檔案未產生有效內容: {file.filename}")
                    continue
                    
                new_contexts.extend(processed)
                self.processed_files.add(file_hash)
                logger.info(f"成功處理: {file.filename} -> 新增 {len(processed)} 條內容")
                
            except Exception as e:
                logger.error(f"處理檔案 {file.filename} 時出錯: {str(e)}")
                continue
        
        if new_contexts:
            unique_contexts = list(set(new_contexts) - set(self.contexts))
            if unique_contexts:
                self.contexts.extend(unique_contexts)
                self._build_index()
                self._save_data()
                logger.info(f"新增 {len(unique_contexts)} 條資料到知識庫")
                return len(unique_contexts)
        return 0

    def semantic_search(self, query: str) -> list:
        """語義搜索"""
        if not self.contexts or not self.index:
            logger.warning("知識庫為空或索引未建立")
            return []

        try:
            query_embed = embedder.encode(
                query,
                convert_to_tensor=False,
                normalize_embeddings=True
            ).astype('float32').reshape(1, -1)

            scores, indices = self.index.search(query_embed, TOP_K_RESULTS)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and score >= SIMILARITY_THRESHOLD:
                    results.append({
                        "score": float(score),
                        "text": self.contexts[idx]
                    })

            logger.info(f"找到 {len(results)} 條相關結果 (最高分: {max([r['score'] for r in results], default=0):.2f})")
            return sorted(results, key=lambda x: x["score"], reverse=True)
        except Exception as e:
            logger.error(f"搜索失敗: {str(e)}")
            return []

class ChatEngine:
    def __init__(self, groq_client):
        self.client = groq_client
        self.retry_config = {
            "retries": 3,
            "wait_time": 45,
            "timeout": 20
        }
    
    def generate_response(self, query: str, contexts: list) -> str:
        if not contexts:
            return "⚠️ 目前知識庫中沒有相關資料，請先上傳技術文件。"
        
        combined_context = "\n\n".join([c["text"] for c in contexts])
        prompt = f"""你是一位山羊養殖專家，請根據以下資料用中文回答問題：

        **問題**：{query}

        **參考資料**：
        {combined_context[:3000]}

        回答時請遵守：
        1. 專業但易懂，使用台灣常用術語
        2. 重要數據需明確標示
        3. 若資料不足請說明需要補充的資訊
        4. 分點條列說明（如有多項要點），要有分段排版好看一點
        5.請避免使用特殊符號（例如「*」、「#」等），直接說就好
        6.只要回答問題所需的解答就好
        """
        for attempt in range(self.retry_config["retries"]):
            try:
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.3,
                    max_tokens=1024,
                    top_p=0.9
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if "rate_limit" in str(e) and attempt < self.retry_config["retries"] - 1:
                    time.sleep(self.retry_config["wait_time"])
                else:
                    logger.error(f"API請求失敗: {str(e)}")
                    return f"❌ 服務暫時不可用，請稍後再試（錯誤代碼：{str(e)[:50]}）"
                
                

# 全局實例
kb = KnowledgeBase()
chat_engine = ChatEngine(groq_client)