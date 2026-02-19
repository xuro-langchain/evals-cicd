"""Multi-agent system with supervisor routing to specialized subagents.

Version: notebook-triggered @ 2026-02-18 22:13
"""

from typing import Annotated
from typing_extensions import TypedDict, NotRequired

from langchain_openai import ChatOpenAI
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.managed.is_last_step import RemainingSteps
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

from app.tools import invoice_tools, music_tools


# State schema
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    remaining_steps: NotRequired[RemainingSteps]


# Model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# Invoice Subagent
invoice_prompt = """You are an invoice specialist for a digital music store.

Tools available:
- get_invoices_by_customer: Look up invoices by customer ID
- get_invoice_total: Get total spending for a customer

CRITICAL: Your response MUST include ALL specific data retrieved:
- For invoices: List each Invoice ID, date, and total amount
- For totals: State the exact dollar amount

Example good response: "Customer 10 has Invoice ID 383 ($13.86) from 2025-08-12, Invoice ID 372 ($1.98) from 2025-07-02."
Example bad response: "I found the invoices for customer 10." (NEVER do this)"""

invoice_agent = create_react_agent(
    model,
    tools=invoice_tools,
    name="invoice_agent",
    prompt=invoice_prompt,
    state_schema=State,
)


# Music Catalog Subagent
music_prompt = """You are a music catalog specialist for a digital music store.

Tools available:
- get_albums_by_artist: Search albums by artist name
- get_tracks_by_artist: Get songs/tracks by artist
- search_tracks: Search tracks by name

CRITICAL: Your response MUST include ALL specific data retrieved:
- List actual album titles, track names, and artist names
- Do not summarize or abbreviate results

Example good response: "AC/DC has 2 albums: 'For Those About To Rock We Salute You' and 'Let There Be Rock'."
Example bad response: "I found albums by AC/DC." (NEVER do this)"""

music_agent = create_react_agent(
    model,
    tools=music_tools,
    name="music_agent",
    prompt=music_prompt,
    state_schema=State,
)


# Supervisor
supervisor_prompt = """You are a customer support supervisor for a digital music store.

IMMEDIATELY route customer inquiries - do NOT ask permission, just transfer:
1. **invoice_agent**: ANY question about billing, invoices, purchases, spending, totals, customer accounts
2. **music_agent**: ANY question about music, albums, artists, tracks, songs, recordings

Examples to route to invoice_agent: "How much did I spend?", "What are my invoices?"
Examples to route to music_agent: "What albums does X have?", "Who recorded Y?", "Find songs by Z"

ONLY respond directly (without routing) for completely off-topic questions like "How's the weather?" or "Tell me a joke".

CRITICAL RULE FOR FINAL RESPONSES:
When providing results, you MUST include ALL specific data from the specialist agents:
- Album/track names, artist names
- Invoice IDs, dates, amounts, totals

NEVER give vague responses. Always state the actual facts."""

workflow = create_supervisor(
    agents=[invoice_agent, music_agent],
    model=model,
    prompt=supervisor_prompt,
    state_schema=State,
    output_mode="full_history",
)

# Compile the graph
graph = workflow.compile()


async def run_graph(inputs: dict) -> dict:
    """Run the multi-agent graph and return the final response."""
    result = await graph.ainvoke(inputs)
    final_message = result["messages"][-1]
    return {"output": final_message.content}
