import os
import requests
import chainlit as cl
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, function_tool

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is missing in .env file")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

@function_tool
def get_crypto_price(symbol: str) -> str:
    """
    Get current price of cryptocurrency (e.g. BTCUSDT, ETHUSDT).
    """
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.upper()}"
        response = requests.get(url)
        response.raise_for_status()
        price = response.json()["price"]
        return f"The current price of {symbol.upper()} is **${price}**."
    except Exception as e:
        return f"Failed to fetch price for {symbol.upper()}. Error: {e}"

crypto_agent = Agent(
    name="CryptoDataAgent",
    instructions="You provide real-time crypto prices using the Binance API.",
    tools=[get_crypto_price]
)

@cl.on_message
async def handle_message(message: cl.Message):
    result = await Runner.run(
        crypto_agent,
        input=message.content,
        run_config=config
    )
    await cl.Message(content=result.final_output).send()
