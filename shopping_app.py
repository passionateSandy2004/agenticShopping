import asyncio
import streamlit as st
import os
from dotenv import load_dotenv

# Load .env BEFORE importing modules that read env at import time
load_dotenv()

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from gemini import run_agent_task


st.set_page_config(page_title="Shopping Agent — Multi-Agent", layout="wide")
st.title("Shopping Agent — Product, Price, and Buzz")

with st.sidebar:
    st.header("Search")
    product_query = st.text_input("What product are you looking for?", value="google pixel 8")
    show_logs = st.checkbox("Show live tool logs", value=True)
    run_button = st.button("Run Analysis")


def render_section(title: str, content: str):
    st.markdown("---")
    st.subheader(title)
    st.markdown(content or "No data.")


def _make_status_logger(status_box):
    placeholder = st.empty()
    messages = []

    def logger(message: str):
        messages.append(str(message))
        if len(messages) > 100:
            del messages[: len(messages) - 100]
        with status_box:
            placeholder.markdown("\n\n".join(messages))

    return logger


async def execute_multi_agent(user_query: str, enable_logs: bool = False):
    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@brightdata/mcp"],
        env={
            # Reuse the same env as gemini.py so the MCP server authorizes properly
            "API_TOKEN": os.environ.get("BRIGHT_DATA_API_TOKEN", ""),
        }
    )

    product_goal = (
        "Collect full product profile: official images, title, key specs, variants, dimensions, weight, materials, warranty, box contents. Prefer official sources. Provide clean summary and source links."
    )
    price_goal = (
        "Find availability across major Indian e-commerce sites (Amazon, Flipkart, Reliance, Croma, Vijay Sales, official store). For each: price, currency, stock status, shipping ETA, seller, warranty notes, URL. Output a concise comparison."
    )
    news_goal = (
        "Summarize recent trending news, memes, launch rumors, controversies, major reviews about the product. Include dates, sources, and brief takeaways."
    )

    # We will stream logs inside each st.status block directly

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            with st.status("Product agent running...", expanded=True) as status_box:
                product_logger = _make_status_logger(status_box) if enable_logs else None
                product_text = await run_agent_task(
                    session,
                    "Product Profile",
                    user_query,
                    product_goal,
                    logger=product_logger,
                )
                status_box.update(label="Product agent finished", state="complete")

            with st.status("Price & Availability agent running...", expanded=True) as status_box:
                price_logger = _make_status_logger(status_box) if enable_logs else None
                price_text = await run_agent_task(
                    session,
                    "Price & Availability",
                    user_query,
                    price_goal,
                    logger=price_logger,
                )
                status_box.update(label="Price agent finished", state="complete")

            with st.status("News & Social Buzz agent running...", expanded=True) as status_box:
                news_logger = _make_status_logger(status_box) if enable_logs else None
                news_text = await run_agent_task(
                    session,
                    "Trending News & Social Buzz",
                    user_query,
                    news_goal,
                    logger=news_logger,
                )
                status_box.update(label="News agent finished", state="complete")

            return {
                "product": product_text,
                "price": price_text,
                "news": news_text,
            }


if run_button and product_query.strip():
    with st.spinner("Running multi-agent analysis..."):
        try:
            results = asyncio.run(execute_multi_agent(product_query.strip(), show_logs))
        except RuntimeError:
            # In case an event loop is already running (rare in Streamlit), fall back to create_task
            results = asyncio.get_event_loop().run_until_complete(
                execute_multi_agent(product_query.strip(), show_logs)
            )

    render_section("1) Product Overview", results.get("product"))
    render_section("2) Price Comparison & Availability", results.get("price"))
    render_section("3) Trending News & Social Buzz", results.get("news"))

    with st.expander("Raw outputs"):
        st.json(results)


