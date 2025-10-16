from dotenv import load_dotenv
import os
import requests
import urllib.parse
import json
import logging
import re
import spacy
import string
import shutil
import pandas as pd
from uuid import uuid4

from fastapi import FastAPI, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sklearn.feature_extraction.text import TfidfVectorizer
from langgraph.prebuilt import create_react_agent
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime

load_dotenv()
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GOOGLE_CUSTOM_SEARCH_API_KEY = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

logging.basicConfig(
    filename="sanitizer.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_folder = "./data"
    if os.path.exists(data_folder):
        for filename in os.listdir(data_folder):
            file_path = os.path.join(data_folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    try:
        app.state.nlp = spacy.load("en_core_web_sm")
    except OSError:
        app.state.nlp = None
        print("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


llm = ChatOllama(model="llama3.2", temperature=0, base_url="http://ollama:11434")
embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://ollama:11434")


class ChatRequest(BaseModel):
    message: str
    structured_output: bool = False


def get_current_time() -> str:
    """Returns the current time as a string."""
    print("[TOOL LOG] get_current_time called")

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_from_file(filename: str) -> str:
    """Reads content from the specified file and returns it as a string."""
    print(f"[TOOL LOG] read_from_file called with: {filename}")
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"


def write_file(filename: str, content: str) -> str:
    """Writes content to a file. Overwrites if file exists."""
    print(f"[TOOL LOG] write_file called with: {filename}")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote to {filename}."


def list_files(directory: str = ".") -> str:
    """Lists files in a directory."""
    print(f"[TOOL LOG] list_files called with: {directory}")
    try:
        files = os.listdir(directory)
        return "\n".join(f"- `{name}`" for name in files)
    except Exception as e:
        return f"Error: {e}"


def delete_file(filename: str) -> str:
    """Deletes a file."""
    print(f"[TOOL LOG] delete_file called with: {filename}")
    try:
        os.remove(filename)
        return f"Deleted {filename}."
    except Exception as e:
        return f"Error: {e}"


def calculator(expression: str) -> str:
    """Evaluates a math expression and returns the result."""
    print(f"[TOOL LOG] calculator called with: {expression}")
    try:
        allowed = set("0123456789+-*/(). ")
        if not set(expression).issubset(allowed):
            return "Error: Invalid characters in expression."
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def search_the_web(query: str):
    """Searches the web for the given query and returns a list of results with title, link, and snippet."""
    url = "https://www.googleapis.com/customsearch/v1?"
    params = {"q": query, "key": GOOGLE_CUSTOM_SEARCH_API_KEY, "cx": SEARCH_ENGINE_ID}
    response = requests.get(url, params=params)
    response.raise_for_status()
    results = response.json()
    items = results.get("items", [])
    output = []
    for item in items:
        output.append(
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
            }
        )
    return output


def get_stock_price(symbol: str, structured_output: bool = False):
    """Fetches the latest stock price for a given symbol using Yahoo Finance public API."""
    print(f"[TOOL LOG] get_stock_price called with: {symbol}")
    try:
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_KEY}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 429:
            msg = "Error: Alpha Vantage rate limit exceeded. Please try again later."
            return (
                {"symbol": symbol.upper(), "error": msg} if structured_output else msg
            )
        if resp.status_code != 200:
            msg = f"Error: Alpha Vantage returned status {resp.status_code}."
            return (
                {"symbol": symbol.upper(), "error": msg} if structured_output else msg
            )
        data = resp.json()
        if "Note" in data and "frequency" in data["Note"]:
            msg = "Error: Alpha Vantage rate limit exceeded. Please try again later."
            return (
                {"symbol": symbol.upper(), "error": msg} if structured_output else msg
            )
        if "Error Message" in data:
            msg = f"Error: {data['Error Message']}"
            return (
                {"symbol": symbol.upper(), "error": msg} if structured_output else msg
            )
        quote = data.get("Global Quote", {})
        price = quote.get("05. price")
        currency = "USD"
        if price is not None:
            if structured_output:
                return {"symbol": symbol.upper(), "price": price, "currency": currency}
            return f"The current price of {symbol.upper()} is {price} {currency}."
        else:
            msg = "Could not fetch price."
            return (
                {"symbol": symbol.upper(), "error": msg}
                if structured_output
                else f"Could not fetch price for {symbol}."
            )
    except Exception as e:
        return (
            {"symbol": symbol.upper(), "error": str(e)}
            if structured_output
            else f"Error: {e}"
        )


def get_news_headlines(topic: str, structured_output: bool = False):
    """Fetches the latest news headlines for a topic using DuckDuckGo Instant Answer API."""
    print(f"[TOOL LOG] get_news_headlines called with: {topic}")
    try:
        url = f"https://newsapi.org/v2/top-headlines?q={topic}&apiKey={NEWSAPI_KEY}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 429:
            msg = "Error: NewsAPI rate limit exceeded. Please try again later."
            return {"topic": topic, "error": msg} if structured_output else msg
        if resp.status_code != 200:
            msg = f"Error: NewsAPI returned status {resp.status_code}."
            return {"topic": topic, "error": msg} if structured_output else msg
        data = resp.json()
        if data.get("status") == "error":
            err_code = data.get("code", "")
            err_msg = data.get("message", "Unknown error")
            if err_code == "rateLimited":
                msg = "Error: NewsAPI rate limit exceeded. Please try again later."
            else:
                msg = f"Error: NewsAPI error: {err_msg}"
            return {"topic": topic, "error": msg} if structured_output else msg
        articles = data.get("articles", [])
        headlines = [a["title"] for a in articles if "title" in a]
        if headlines:
            if structured_output:
                return {"topic": topic, "headlines": headlines[:5]}
            return "\n".join(headlines[:5])
        msg = "No recent news found."
        return {"topic": topic, "headlines": []} if structured_output else msg
    except Exception as e:
        return {"error": str(e)} if structured_output else f"Error: {e}"


def get_weather(city: str, structured_output: bool = False):
    """Fetches current weather for a city using the Open-Meteo API."""
    print(f"[TOOL LOG] get_weather called with: {city}")
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(city)}&count=1"
        geo_resp = requests.get(geo_url, timeout=10)
        geo_data = geo_resp.json()
        results = geo_data.get("results")
        if not results:
            return (
                {"city": city, "error": "Could not find location."}
                if structured_output
                else f"Could not find location for '{city}'."
            )
        lat = results[0]["latitude"]
        lon = results[0]["longitude"]

        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_resp = requests.get(weather_url, timeout=10)
        weather_data = weather_resp.json()
        current = weather_data.get("current_weather")
        if not current:
            return (
                {"city": city, "error": "Could not fetch weather."}
                if structured_output
                else f"Could not fetch weather for '{city}'."
            )
        temp = current["temperature"]
        wind = current["windspeed"]
        if structured_output:
            return {"city": city.title(), "temperature_c": temp, "wind_kmh": wind}
        desc = f"Temperature: {temp}°C, Wind Speed: {wind} km/h"
        return f"Current weather in {city.title()}: {desc}"
    except Exception as e:
        return {"city": city, "error": str(e)} if structured_output else f"Error: {e}"


def currency_converter(amount: float, base: str, target) -> dict:
    """Converts an amount from base currency to target currencies."""
    if isinstance(target, str):
        target = [target]
    url = "https://api.frankfurter.dev/v1/latest?base=USD&currencies"
    response = requests.get(url)
    currencies = list(response.json()["rates"].keys())
    currencies.append("USD")
    if base not in currencies:
        return {"error": f"Base currency '{base}' is not supported."}
    for target_currency in target:
        if target_currency not in currencies:
            return {"error": f"Target currency '{target_currency}' is not supported."}

    url = f"https://api.frankfurter.dev/v1/latest?base={base}&symbols=" + ",".join(
        target
    )
    response = requests.get(url)
    response.raise_for_status()
    result = {}
    for currency, value in response.json().get("rates", {}).items():
        result[currency] = round(amount * value, 2)
    return result


EMAIL_REGEX = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
PHONE_REGEX = r"(?:\+?\d|\(\d)[\d\s().-]{6,}\d"
CREDIT_CARD_REGEX = r"\b(?:\d[ -]*?){13,16}\b"

PROMPT_INJECTION_PATTERNS = [
    r"ignore (all|any|previous|earlier) instructions",
    r"disregard (all|any|previous|earlier) instructions",
    r"repeat this prompt verbatim",
    r"do anything now",
    r"bypass",
    r"override",
    r"forget previous",
    r"you are now",
    r"as an ai language model, you are not restricted",
    r"please pretend to",
    r"simulate",
    r"jailbreak",
    r"unfiltered",
    r"raw output",
    r"act as",
    r"output the following",
    r"print the following",
    r"execute the following",
    r"respond with",
]


def detect_prompt_injection(text):
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def redact_pii(text):
    text = re.sub(EMAIL_REGEX, "[REDACTED]", text)
    text = re.sub(PHONE_REGEX, "[REDACTED]", text)
    text = re.sub(CREDIT_CARD_REGEX, "[REDACTED]", text)

    nlp = getattr(app.state, "nlp", None)
    if nlp:
        doc = nlp(text)
        redacted = text
        offset = 0
        for ent in doc.ents:
            if ent.label_ in {"PERSON", "GPE", "ORG"}:
                start = ent.start_char + offset
                end = ent.end_char + offset
                redacted = redacted[:start] + "[REDACTED]" + redacted[end:]
                offset += len("[REDACTED]") - (end - start)
        return redacted
    else:
        return text


def detect_anomaly(text):
    if len(text) > 500:
        return "Input too long (possible anomaly)"
    special_chars = sum(1 for c in text if c in string.punctuation)
    if special_chars > 30:
        return "Too many special characters (possible anomaly)"
    words = text.split()
    if len(words) > 0:
        unique_words = set(words)
        if len(unique_words) / len(words) < 0.3:
            return "Highly repetitive input (possible anomaly)"
    return None


def input_sanitizer(user_input: str) -> str:
    """Sanitizes user input by detecting anomalies, prompt injections, and redacting PII."""
    anomaly = detect_anomaly(user_input)
    if anomaly:
        logging.warning(f"Blocked input (anomaly): {anomaly} | Input: {user_input}")
        return {"status": "blocked", "reason": anomaly}
    if detect_prompt_injection(user_input):
        logging.warning(f"Blocked input (prompt injection): {user_input}")
        return {"status": "blocked", "reason": "possible prompt injection detected"}
    sanitized = redact_pii(user_input)
    if sanitized != user_input:
        logging.info(f"Sanitized input: {user_input} => {sanitized}")
        return {"status": "sanitized", "result": sanitized}
    logging.info(f"Clean input: {user_input}")
    return {"status": "clean", "result": sanitized}


def get_sanitizer_report():
    """Generates a report from the sanitizer log file."""
    log_file = "sanitizer.log"
    blocked_anomaly = 0
    blocked_prompt = 0
    sanitized = 0
    clean = 0
    anomaly_reasons = Counter()
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                if "Blocked input (anomaly):" in line:
                    blocked_anomaly += 1
                    m = re.search(r"Blocked input \(anomaly\): ([^|]+) \| Input:", line)
                    if m and m.group(1).strip():
                        anomaly_reasons[m.group(1).strip()] += 1
                elif "Blocked input (prompt injection):" in line:
                    blocked_prompt += 1
                elif "Sanitized input:" in line:
                    sanitized += 1
                elif "Clean input:" in line:
                    clean += 1
    except FileNotFoundError:
        return {"error": "No log file found."}
    return {
        "blocked_anomaly": blocked_anomaly,
        "blocked_prompt_injection": blocked_prompt,
        "sanitized": sanitized,
        "clean": clean,
        "anomaly_reasons": dict(anomaly_reasons),
    }


data_folder = "./data"
os.makedirs(data_folder, exist_ok=True)
chunk_size = 1000
chunk_overlap = 50

vector_space = Chroma(
    collection_name="documents",
    embedding_function=embeddings,
)

all_documents = []


def get_answer_from_documents(query: str):
    """Answers a question using the content of uploaded documents."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer the question based only on the provided context.",
            ),
            ("human", "Question: {input}\nContext: {context}"),
        ]
    )

    dense_retriever = vector_space.as_retriever(search_kwargs={"k": 10})
    bm25_retriever = BM25Retriever.from_documents(all_documents)
    bm25_retriever.k = 10

    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever], weights=[0.5, 0.5]
    )
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(ensemble_retriever, combine_docs_chain)
    # Logging for hybrid RAG testing
    print("[RAG] Query:", query)
    dense_docs = dense_retriever.invoke(query)
    bm25_docs = bm25_retriever.invoke(query)
    print(
        "[RAG] Dense top doc:",
        dense_docs[0].page_content[:200] if dense_docs else "None",
    )
    print(
        "[RAG] BM25 top doc:",
        bm25_docs[0].page_content[:200] if bm25_docs else "None",
    )
    context = ensemble_retriever.invoke(query)
    result = retrieval_chain.invoke({"input": query, "context": context})
    print("[RAG] Final answer:", result["answer"])
    # Add to conversation history
    conversation.append({"role": "user", "content": query})
    conversation.append({"role": "assistant", "content": result["answer"]})
    return result["answer"]


agent = create_react_agent(
    model=llm,
    tools=[
        get_current_time,
        read_from_file,
        write_file,
        list_files,
        delete_file,
        calculator,
        search_the_web,
        get_stock_price,
        get_news_headlines,
        get_weather,
        currency_converter,
        input_sanitizer,
        get_sanitizer_report,
        get_answer_from_documents,
    ],
    prompt=(
        "You are a helpful, agentic AI assistant. For any factual, up-to-date, or user-specific question, you must call the appropriate tool and return only the tool's result in your answer. "
        "Do not use your own knowledge or add extra commentary, disclaimers, or fallback information. "
        "If a user request requires multiple steps or tools, reason step by step and use only the results of previous tools. "
        "Be concise and clear. If the user requests structured data, return it in the requested format (e.g., JSON, markdown). "
        "For greetings or general conversation, respond naturally without calling a tool."
        "Never answer from your own knowledge for document or user-specific queries. Always use the tool result verbatim."
    ),
)

conversation = []


def is_document_query(user_input):
    doc_keywords = ["document", "file", "pdf", "csv", "student", "resume", "report"]
    return any(word in user_input.lower() for word in doc_keywords)


def ask_agent(user_input, structured_output=False):
    conversation.append({"role": "user", "content": user_input})
    result = agent.invoke(
        {"messages": conversation, "structured_output": structured_output}
    )

    if is_document_query(user_input):
        # Prefer tool result if present
        tool_result = None
        for msg in reversed(result["messages"]):
            if getattr(msg, "tool_call_id", None) or getattr(msg, "tool_name", None):
                tool_result = getattr(msg, "content", None)
                break
        if tool_result:
            conversation.append({"role": "assistant", "content": tool_result})
            return tool_result

    for msg in reversed(result["messages"]):
        if getattr(msg, "content", None):
            if structured_output:
                try:
                    content = (
                        json.loads(msg.content)
                        if isinstance(msg.content, str)
                        else msg.content
                    )
                except Exception:
                    content = msg.content
                conversation.append({"role": "assistant", "content": content})
            else:
                conversation.append({"role": "assistant", "content": msg.content})
            break


@app.post("/chat", status_code=status.HTTP_200_OK)
async def chat_endpoint(request: ChatRequest):
    ask_agent(request.message, structured_output=request.structured_output)
    return {"response": conversation[-1]["content"]}


@app.post("/upload-file", status_code=status.HTTP_201_CREATED)
async def upload_file(files: list[UploadFile] = File(...)):
    os.makedirs(data_folder, exist_ok=True)
    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)
        os.remove(file_path)

    for file in files:
        file_location = os.path.join(data_folder, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    uploaded_csvs = []

    for filename in os.listdir(data_folder):
        path = os.path.join(data_folder, filename)
        if filename.endswith(".csv"):
            df = pd.read_csv(path)
            columns = df.columns.tolist()
            uploaded_csvs.append(
                {
                    "filename": filename,
                    "columns": columns,
                }
            )

    if uploaded_csvs:
        return {
            "message": "Files saved",
            "csv_files": uploaded_csvs,
        }

    return {
        "message": "Files saved",
    }


parsed_csv_column_map = {}


@app.post("/set-columns", status_code=status.HTTP_200_OK)
async def set_columns(csv_column_map: str = Form(...)):
    global parsed_csv_column_map
    # csv_column_map: { filename: {"content": [...], "metadata": [...]}, ... }
    parsed_csv_column_map = json.loads(csv_column_map)


@app.post("/create-vector-database", status_code=status.HTTP_201_CREATED)
async def create_vector_database():
    global all_documents, parsed_csv_column_map
    all_documents = []

    for filename in os.listdir(data_folder):
        path = os.path.join(data_folder, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            documents = text_splitter.split_documents(docs)
            ids = [str(uuid4()) for _ in range(len(documents))]

            texts = [doc.page_content for doc in documents]
            vectorizer = TfidfVectorizer()
            sparse_matrix = vectorizer.fit_transform(texts)
            sparse_vectors = [vec.toarray().flatten().tolist() for vec in sparse_matrix]

            vector_space.add_documents(
                documents=documents, ids=ids, sparse_vectors=sparse_vectors
            )
            all_documents.extend(documents)

        elif filename.endswith(".csv"):
            df = pd.read_csv(path)
            documents = []
            ids = []
            # Use per-CSV column selection if available
            colmap = parsed_csv_column_map.get(filename, {})
            content_cols = colmap.get("content", list(df.columns))
            metadata_cols = colmap.get(
                "metadata", [col for col in df.columns if col not in content_cols]
            )
            for i, row in df.iterrows():
                content_parts = [str(row[col]) for col in content_cols]
                page_content = " ".join(content_parts).strip()
                metadata = {col: row[col] for col in metadata_cols}
                document = Document(
                    page_content=page_content, metadata=metadata, id=str(i)
                )
                documents.append(document)
                ids.append(str(i))
            texts = [doc.page_content for doc in documents]
            vectorizer = TfidfVectorizer()
            sparse_matrix = vectorizer.fit_transform(texts)
            sparse_vectors = [vec.toarray().flatten().tolist() for vec in sparse_matrix]
            all_documents.extend(documents)
            vector_space.add_documents(
                documents=documents, ids=ids, sparse_vectors=sparse_vectors
            )
    return {"message": "Vector database created"}
