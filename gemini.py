import os
from dotenv import load_dotenv
load_dotenv()
import asyncio
from datetime import datetime
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from google import genai
import json

client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", ""))

def log_realtime(step_name, data=""):
    """Log real-time tool execution with actual results"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n🔄 [{timestamp}] {step_name}")
    if data:
        print(f"📊 RESULT:")
        print(json.dumps(data, indent=2, ensure_ascii=False)[:1000] + ("..." if len(str(data)) > 1000 else ""))
    print("-" * 60)

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="npx",  # Executable
    args=["-y", "@brightdata/mcp"],  # MCP Server
    env={
        "API_TOKEN": os.environ.get("BRIGHT_DATA_API_TOKEN", "")
    },  # Optional environment variables
)

async def run_agent_task(session, section_name, user_query, system_goal, logger=None):
    def emit(text):
        if logger:
            try:
                logger(text)
            except Exception:
                print(text)
        else:
            print(text)

    def emit_json(step_name, data):
        if logger:
            try:
                snippet = json.dumps(data, indent=2, ensure_ascii=False)
                logger(f"{step_name}\n```json\n{snippet}\n```")
            except Exception:
                log_realtime(step_name, data)
        else:
            log_realtime(step_name, data)

    emit("\n" + "="*80)
    emit(f"🧠 {section_name} — Agent Running")
    emit("="*80)

    task_prompt = f"""
You are the {section_name} agent.
Goal: {system_goal}

User query: {user_query}

Instructions:
- Use the available tools to browse and gather live data.
- Prefer official product pages and reputable sources.
- Return clear, factual information. If uncertain, say so.
- Include sources (URLs) in your answer when possible.
"""

    # Robust request with retries to handle transient 5xx (e.g., 503 overloaded)
    response = None
    max_attempts = 4
    base_delay_sec = 1.2
    for attempt_num in range(1, max_attempts + 1):
        try:
            emit(f"Attempt {attempt_num}/{max_attempts}: contacting Gemini…")
            response = await client.aio.models.generate_content(
                model="gemini-2.0-flash",
                contents=task_prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0,
                    tools=[session],
                ),
            )
            break
        except Exception as request_error:
            wait_time = base_delay_sec * (2 ** (attempt_num - 1))
            emit(f"⚠️ Model request failed: {request_error}. Retrying in {wait_time:.1f}s…")
            if attempt_num == max_attempts:
                emit("❌ Max retries reached. Aborting this agent.")
                raise
            await asyncio.sleep(wait_time)

    emit("\n📊 REAL-TIME TOOL EXECUTION (" + section_name + "):")
    emit("="*80)

    # Show structure
    emit(f"🔍 Has candidates: {hasattr(response, 'candidates')}")
    if hasattr(response, 'candidates') and response.candidates:
        for i, candidate in enumerate(response.candidates):
            if hasattr(candidate, 'content') and candidate.content:
                for j, part in enumerate(candidate.content.parts):
                    if hasattr(part, 'function_call') and part.function_call is not None:
                        function_name = getattr(part.function_call, 'name', 'unknown_function')
                        function_args = getattr(part.function_call, 'args', {})
                        emit_json(f"🔨 TOOL CALL #{i+1}-{j+1}: {function_name}", function_args)
                    if hasattr(part, 'function_response') and part.function_response is not None:
                        response_name = getattr(part.function_response, 'name', 'unknown_response')
                        response_data = getattr(part.function_response, 'response', {})
                        emit_json(f"📥 TOOL RESPONSE #{i+1}-{j+1}: {response_name}", response_data)
                    if hasattr(part, 'text') and part.text:
                        emit_json(f"💬 AGENT RESPONSE #{i+1}-{j+1}", {"text": part.text[:500] + "..." if len(part.text) > 500 else part.text})

    emit("\n" + "="*80)
    emit(f"✅ {section_name} — Agent Completed")
    emit("="*80)
    emit(response.text)
    emit("="*80)

    return response.text

def print_section(title):
    print("\n" + "#"*80)
    print(f"🔷 {title}")
    print("#"*80)

async def run():
    print("🚀 Starting Real-Time Agent Workflow...")
    
    async with stdio_client(server_params) as (read, write):
        print("🔌 Connected to BrightData MCP server")
        
        async with ClientSession(read, write) as session:
            print("📋 Initializing MCP session...")
            
            user_query = "search for google pixel 8 and give me the price in amazon and flipkart and also i want to know what is the meme that is going about the that product in the social media"
            print(f"🎯 User Query: '{user_query}'")
            
            # Initialize the connection between client and server
            await session.initialize()
            print("✅ MCP Session Ready - All tools available")

            # Multi-agent workflow
            product_goal = (
                "Collect full product profile: official images, title, key specs, variants, dimensions, weight, materials, warranty, box contents. Prefer official sources. Provide clean summary and source links. from indian e commerce sites only."
            )
            price_goal = (
                "Find availability across major Indian e-commerce sites (Amazon, Flipkart, Reliance, Croma, Vijay Sales, official store) and PROVIDE THE BUYING LINK. For each: price, currency, stock status, shipping ETA, seller, warranty notes, URL. Output a concise comparison."
            )
            news_goal = (
                "Summarize recent trending news, memes, launch rumors, controversies, major reviews about the product. Include dates, sources, and brief takeaways."
            )

            product_text = await run_agent_task(session, "Product Profile", user_query, product_goal)
            price_text = await run_agent_task(session, "Price & Availability", user_query, price_goal)
            news_text = await run_agent_task(session, "Trending News & Social Buzz", user_query, news_goal)

            # Unified report
            print_section("Unified Shopping Report")
            print_section("1) Product Overview")
            print(product_text)
            print_section("2) Price Comparison & Availability")
            print(price_text)
            print_section("3) Trending News & Social Buzz")
            print(news_text)

if __name__ == "__main__":
    # Start the asyncio event loop and run the main function
    asyncio.run(run())