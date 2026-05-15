"""
nba_agent.py
Conversational NBA research agent connecting to two MCP servers:
- nba_server.py: Kaggle datasets, pandas analysis, Basketball Reference, memory, hypotheses
- sentiment_server.py: Reddit scraping, VADER, DistilBERT sentiment
"""

import anthropic
import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

client = anthropic.Anthropic()

SENTIMENT_SERVER_PATH = os.path.expanduser(
    "~/Documents/Data_Science_Basketball/NBA-trade-sentiment/sentiment_server.py"
)
SENTIMENT_PYTHON = os.path.expanduser(
    "~/Documents/Data_Science_Basketball/NBA-trade-sentiment/sentiment/bin/python3"
)

system = """You are an NBA research agent with access to two sets of tools:

STATS TOOLS (from nba_server):
- load_memory: load past research findings — ALWAYS call this first at session start
- list_downloaded_datasets: check what data is already downloaded and loaded in memory
- search_and_download_dataset: find and download Kaggle datasets
- load_and_profile: load a CSV into memory with a named variable (e.g. df_stats, df_salaries)
- run_analysis: run pandas code — all loaded datasets available by their variable names
- fetch_bbref_team_stats: scrape Basketball Reference for a season
- generate_stats_chart: save a trend or comparison chart to charts/
- generate_sentiment_chart: save a VADER vs DistilBERT chart to charts/
- generate_hypotheses: generate 3 follow-up hypotheses after any significant finding
- save_finding: save a finding to memory before ending a session

SENTIMENT TOOLS (from sentiment_server):
- search_and_analyze_vader: Reddit sentiment via VADER
- search_and_analyze_bert: Reddit sentiment via DistilBERT
- compare_models: side-by-side VADER vs DistilBERT comparison

SESSION START RULES:
1. Always call load_memory first — check what's been investigated before
2. If memory contains unexplored hypotheses relevant to the current question, pursue those
3. Call list_downloaded_datasets to see what data is already available
4. Only download new data if it's not already present

HYPOTHESIS RULES:
- After every significant analysis finding, call generate_hypotheses
- The hypotheses tool will suggest both stats and sentiment angles
- Pursue at least one hypothesis before giving a final answer if time permits
- Save any unexplored hypotheses to memory via save_finding

MULTI-DATASET RULES:
- Load each dataset with a unique descriptive name using the dataset_name parameter
- All loaded datasets persist in memory across run_analysis calls
- To merge: pd.merge(df_stats, df_salaries, on='common_column')
- Always check columns before attempting a merge

SESSION END RULES:
- Before the user types quit, always call save_finding with:
  - The question investigated
  - The key finding
  - Which datasets were used
  - Any hypotheses that weren't yet explored

SELF-EVALUATION RULES:
After getting results, ask:
- Does this directly answer the question?
- Did I call generate_hypotheses yet?
- Would a second dataset make the answer richer?
- Would combining stats + sentiment give a deeper insight?

All tool calls are logged to research_log.txt.

OUTPUT FORMAT:
- Lead with the direct answer
- Show key numbers and trends
- Note which datasets were used
- Share the hypotheses generated
- Note data limitations"""


async def run_agent():
    nba_params = StdioServerParameters(
        command="python",
        args=["nba_server.py"]
    )

    sentiment_params = StdioServerParameters(
        command=SENTIMENT_PYTHON,
        args=[SENTIMENT_SERVER_PATH]
    )

    async with stdio_client(nba_params) as (nba_read, nba_write):
        async with ClientSession(nba_read, nba_write) as nba_session:
            await nba_session.initialize()

            async with stdio_client(sentiment_params) as (sent_read, sent_write):
                async with ClientSession(sent_read, sent_write) as sentiment_session:
                    await sentiment_session.initialize()

                    nba_tools_raw = await nba_session.list_tools()
                    sent_tools_raw = await sentiment_session.list_tools()

                    nba_tools = [
                        {"name": t.name, "description": t.description, "input_schema": t.inputSchema}
                        for t in nba_tools_raw.tools
                    ]
                    sent_tools = [
                        {"name": t.name, "description": t.description, "input_schema": t.inputSchema}
                        for t in sent_tools_raw.tools
                    ]

                    nba_tool_names = {t["name"] for t in nba_tools}
                    sent_tool_names = {t["name"] for t in sent_tools}
                    all_tools = nba_tools + sent_tools

                    print("NBA Research Agent ready.")
                    print(f"Stats tools: {list(nba_tool_names)}")
                    print(f"Sentiment tools: {list(sent_tool_names)}")
                    print("Decision trail logged to: research_log.txt")
                    print("Session memory: session_memory.json\n")

                    messages = []

                    while True:
                        user_input = input("You: ").strip()
                        if user_input.lower() in ("quit", "exit"):
                            print("Goodbye!")
                            break
                        if not user_input:
                            continue

                        messages.append({"role": "user", "content": user_input})

                        while True:
                            response = client.messages.create(
                                model="claude-sonnet-4-6",
                                max_tokens=4096,
                                system=system,
                                tools=all_tools,
                                messages=messages
                            )

                            assistant_content = []
                            for block in response.content:
                                if block.type == "text":
                                    assistant_content.append({"type": "text", "text": block.text})
                                    if block.text:
                                        print(f"\nAgent: {block.text}")
                                elif block.type == "tool_use":
                                    assistant_content.append({
                                        "type": "tool_use",
                                        "id": block.id,
                                        "name": block.name,
                                        "input": block.input
                                    })

                            messages.append({"role": "assistant", "content": assistant_content})

                            tool_use_blocks = [b for b in assistant_content if b["type"] == "tool_use"]

                            if not tool_use_blocks:
                                break

                            tool_results = []
                            for block in tool_use_blocks:
                                print(f"\n[calling: {block['name']}({list(block['input'].keys())})]")
                                try:
                                    if block["name"] in nba_tool_names:
                                        mcp_result = await nba_session.call_tool(
                                            block["name"], arguments=block["input"]
                                        )
                                    elif block["name"] in sent_tool_names:
                                        mcp_result = await sentiment_session.call_tool(
                                            block["name"], arguments=block["input"]
                                        )
                                    else:
                                        mcp_result = None

                                    result = mcp_result.content[0].text if mcp_result and mcp_result.content else "No result"
                                except Exception as e:
                                    result = f"Error: {e}"

                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": block["id"],
                                    "content": result
                                })

                            messages.append({"role": "user", "content": tool_results})

if __name__ == "__main__":
    asyncio.run(run_agent())