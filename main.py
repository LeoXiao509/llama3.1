# 移除PDF處理，只保留docx/txt/csv的版本
import os
import time
import pickle
import hashlib
import pandas as pd
import streamlit as st
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

# 頁面配置
st.set_page_config(
    page_title="Dr. Goat Pro 專家系統",
    page_icon="🐐",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🐐 Dr. Goat Pro 山羊養殖專家系統")
st.caption("結合最新 RAG 技術與農業領域知識的智慧解決方案")

# 初始化模型
@st.cache_resource
def init_system():
    """系統初始化函數"""
    if not (groq_api_key := os.getenv("GROQ_API_KEY")):
        st.error("GROQ_API_KEY 未設定於環境變數")
        st.stop()

    try:
        groq_client = Groq(api_key=groq_api_key)
        groq_client.models.list()
    except Exception as e:
        st.error(f"Groq 連線失敗: {str(e)}")
        st.stop()

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
    """整合 FAISS 的知識庫管理類別"""
    def __init__(self):
        self.contexts = []
        self.processed_files = set()
        self.index = None
        self.embedder = embedder

        # 載入預處理資料
        if os.path.exists(PROCESSED_DATA_PATH):
            self._load_processed_data()
            
        if os.path.exists(DEFAULT_CSV_PATH):
            self._load_default_csv()

    def _load_processed_data(self):
        """載入處理過的資料"""
        try:
            with open(PROCESSED_DATA_PATH, "rb") as f:
                data = pickle.load(f)
                self.contexts = data["contexts"]
                self.processed_files = data["processed_files"]
                
                if "faiss_index" in data and data["faiss_index"] is not None:
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
                new_data = [c for c in df["context"].dropna().astype(str)
                           if c not in self.contexts]
                if new_data:
                    self.contexts.extend(new_data)
                    self._build_index()
                    logger.info(f"已載入 {len(new_data)} 條預設資料")
        except Exception as e:
            logger.error(f"預設CSV載入失敗: {str(e)}")

    def _build_index(self):
        """建立 FAISS 索引"""
        if not self.contexts:
            return
            
        try:
            embeddings = self.embedder.encode(
                self.contexts, 
                convert_to_tensor=False,
                normalize_embeddings=True
            ).astype('float32')
            
            dimension = embeddings.shape[1]
            
            if len(self.contexts) < 10_000:
                self.index = faiss.IndexFlatIP(dimension)
            else:
                nlist = min(100, len(self.contexts) // 10)
                quantizer = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                self.index.train(embeddings)
                
            self.index.add(embeddings)
            logger.info(f"已建立 FAISS 索引 (類型: {type(self.index).__name__})")
        except Exception as e:
            logger.error(f"建立索引失敗: {str(e)}")
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
                file_hash = get_file_hash(file)
                if file_hash in self.processed_files:
                    continue

                if file.name.endswith(".docx"):
                    processed_data = process_docx(file)
                elif file.name.endswith(".txt"):
                    processed_data = process_txt(file)
                elif file.name.endswith(".csv"):
                    processed_data = process_csv(file)
                else:
                    st.warning(f"不支援的檔案格式: {file.name}")
                    continue

                logger.info(f"檔案 {file.name} 解析出 {len(processed_data)} 條資料")
                new_contexts.extend(processed_data)
                self.processed_files.add(file_hash)

            except Exception as e:
                st.error(str(e))
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
            query_embed = self.embedder.encode(
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
    """問答引擎"""
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
        4. 分點條列說明（如有多項要點）
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

def init_ui():
    """初始化使用者介面"""
    if "kb" not in st.session_state:
        st.session_state.kb = KnowledgeBase()

    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = ChatEngine(groq_client)

    if "messages" not in st.session_state:
        st.session_state.messages = []

def main():
    init_ui()

    with st.sidebar:
        st.header("📁 知識庫管理")
        
        uploaded_files = st.file_uploader(
            "上傳技術文件（DOCX/TXT/CSV）",
            type=["docx", "txt", "csv"],
            accept_multiple_files=True
        )
        if uploaded_files:
            with st.spinner(f"正在處理 {len(uploaded_files)} 個檔案..."):
                added_count = st.session_state.kb.add_files(uploaded_files)
                if added_count > 0:
                    st.success(f"新增 {added_count} 條技術資料！")
                    if any(file.name.endswith('.csv') for file in uploaded_files):
                        total_csv_data = sum(len(pd.read_csv(file)) for file in uploaded_files if file.name.endswith('.csv'))
                        st.markdown(f"🔢 CSV檔案總共包含 {total_csv_data} 條資料。")
                    st.rerun()

        st.divider()
        st.markdown(f"**當前知識庫統計** 📊")
        st.markdown(f"- 總資料條數：{len(st.session_state.kb.contexts)}")
        st.markdown(f"- 已處理檔案數：{len(st.session_state.kb.processed_files)}")
        st.markdown(f"- 索引類型：{type(st.session_state.kb.index).__name__ if st.session_state.kb.index else '無'}")

        if st.button("📂 顯示所有知識庫內容"):
            if st.session_state.kb.contexts:
                st.markdown("### 知識庫所有內容")
                for i, context in enumerate(st.session_state.kb.contexts, 1):
                    st.markdown(f"**內容 {i}**")
                    st.markdown(f"```\n{context}\n```")
                    st.markdown("---")
            else:
                st.warning("知識庫中目前沒有內容。")

        st.divider()
        if st.button("🔄 清除對話記錄"):
            st.session_state.messages = []
            st.rerun()

        if st.button("⚠️ 重置知識庫"):
            st.session_state.kb = KnowledgeBase()
            st.rerun()

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if query := st.chat_input("輸入關於山羊養殖的技術問題..."):
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").write(query)

        with st.spinner("🔍 正在搜索技術資料庫..."):
            search_results = st.session_state.kb.semantic_search(query)

        if not search_results:
            response = "⚠️ 未找到相關技術資料，建議補充以下資訊：\n- 疾病症狀描述\n- 飼養環境條件\n- 檢測報告數據"
        else:
            with st.expander("📄 檢索到的技術資料片段", expanded=True):
                for i, res in enumerate(search_results, 1):
                    st.markdown(f"**匹配度 {res['score']:.2%} (片段 {i})**")
                    st.markdown(f"```\n{res['text']}\n```")
                    st.markdown("---")

            with st.spinner("🧠 正在生成專家建議..."):
                response = st.session_state.chat_engine.generate_response(query, search_results)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()