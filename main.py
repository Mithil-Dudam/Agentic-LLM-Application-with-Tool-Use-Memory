from dotenv import load_dotenv
import os
import requests
import urllib.parse
import json

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

load_dotenv()
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GOOGLE_CUSTOM_SEARCH_API_KEY = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


llm = ChatOllama(model="llama3.2", temperature=0, base_url="http://ollama:11434")


class ChatRequest(BaseModel):
    message: str
    structured_output: bool = False


def get_current_time() -> str:
    """Returns the current time as a string."""
    print("[TOOL LOG] get_current_time called")
    from datetime import datetime

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
    ],
    prompt=(
        "You are a helpful, agentic AI assistant. When you use a tool, always include the tool's result in your final answer to the user. "
        "If a user request requires multiple steps or tools, reason step by step and use the results of previous tools as needed. "
        "Be concise and clear. If the user requests structured data, return it in the requested format (e.g., JSON, markdown). "
        "Only use a tool if the user's request clearly requires external data, calculation, or file operations. For greetings or general conversation, respond naturally without calling a tool."
    ),
)

conversation = []


def ask_agent(user_input, structured_output=False):
    conversation.append({"role": "user", "content": user_input})
    result = agent.invoke(
        {"messages": conversation, "structured_output": structured_output}
    )
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
