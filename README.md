# NBA Analytics Agent

A multi-server agentic research system built on Anthropic's Model Context Protocol (MCP) that autonomously sources, analyzes, and synthesizes NBA data across statistical and sentiment domains. Given an open-ended research question, the agent plans its own investigation, downloads relevant datasets, runs analysis, generates hypotheses, and cross-references fan sentiment — without manual data wrangling at each step.

## What Makes This Different

Most NBA analytics projects start with a pre-downloaded dataset and a fixed analysis pipeline. This agent starts with a question and figures out the rest itself:

- **Autonomous data sourcing** — searches Kaggle for relevant datasets, downloads them on demand, and checks what's already available before fetching anything new
- **Multi-server MCP architecture** — two independent MCP servers expose domain-specific tools; the agent routes across both in a single session
- **Cross-domain synthesis** — combines statistical findings (team 3-point rates, win rates, salary efficiency) with Reddit fan sentiment (VADER + DistilBERT) to answer questions neither domain could answer alone
- **Session memory** — findings and unexplored hypotheses persist across sessions; the agent builds on previous work rather than starting from scratch
- **Hypothesis generation** — after every significant finding, an internal Claude call generates 3 testable follow-up hypotheses suggesting both stats and sentiment angles

## Architecture

```
nba_agent.py
    │
    ├── nba_server.py (Stats MCP Server)
    │       ├── search_and_download_dataset  — Kaggle API
    │       ├── list_downloaded_datasets     — local data inventory
    │       ├── load_and_profile             — multi-dataset loading
    │       ├── run_analysis                 — persistent pandas namespace
    │       ├── fetch_bbref_team_stats       — Basketball Reference scraping
    │       ├── generate_stats_chart         — matplotlib chart generation
    │       ├── generate_sentiment_chart     — VADER vs DistilBERT charts
    │       ├── generate_hypotheses          — internal Claude API call
    │       ├── save_finding                 — session memory write
    │       └── load_memory                  — session memory read
    │
    └── sentiment_server.py (Sentiment MCP Server)
            ├── search_and_analyze_vader     — Reddit + VADER
            ├── search_and_analyze_bert      — Reddit + DistilBERT
            └── compare_models               — side-by-side comparison
```

The sentiment server lives in [NBA-Trade-Sentiment](https://github.com/Darshan-Balaji-ML/NBA-Trade-Sentiment) and is referenced by path. The agent connects to both servers simultaneously at startup, merges their tool lists, and routes each tool call to the correct server automatically.

## Example Session

```
You: Which NBA teams had the highest three-point attempt rates between 2015
     and 2020, and did fan sentiment differ between analytics and traditional teams?

[calling: load_memory]
[calling: list_downloaded_datasets]
[calling: load_and_profile(['file_path', 'dataset_name'])]
[calling: run_analysis(['code'])]         # filter 2015-2020, rank by 3PAR
[calling: run_analysis(['code'])]         # noticed Rockets missing, self-corrected
[calling: generate_stats_chart(['data_code', 'chart_type', ...])]
[calling: search_and_analyze_vader(['player_name', 'team', 'year', ...])]
[calling: generate_sentiment_chart(['sentiment_data', 'title', ...])]
[calling: generate_hypotheses(['data_summary', 'question_context'])]
[calling: save_finding(['question', 'finding', 'datasets_used', ...])]

Agent: Houston Rockets led the league at 0.458 3PAR — 20% above the next
       team. VADER scored analytics-forward teams 8 points more positive
       than traditional teams on Reddit, but DistilBERT showed a narrower
       gap, suggesting some of that positivity was hedged enthusiasm rather
       than genuine excitement...
```

## Key Findings (Sample Sessions)

**Three-Point Revolution (2015–2020)**
Houston Rockets led at 0.458 avg 3PAR — significantly above Dallas (0.390) and Golden State (0.355). Despite volume doubling league-wide, 3P% stayed flat at ~35%, suggesting improved shot selection rather than lower standards.

**Fan Sentiment: Analytics vs Traditional**
VADER scored analytics-forward teams ~8 points more positive on Reddit. DistilBERT showed a narrower gap, indicating hedged rather than genuine enthusiasm — consistent with the "boring basketball" criticism the Rockets era attracted mid-season.

## Dependencies

This repo depends on [NBA-Trade-Sentiment](https://github.com/Darshan-Balaji-ML/NBA-Trade-Sentiment) for the sentiment MCP server. Clone both repos and update `SENTIMENT_SERVER_PATH` and `SENTIMENT_PYTHON` in `nba_agent.py` to match your local paths.

```bash
git clone https://github.com/Darshan-Balaji-ML/NBA-Analytics-Agent
cd NBA-Analytics-Agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set environment variables:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export KAGGLE_CONFIG_DIR="~/.kaggle"
```

Run:
```bash
python nba_agent.py
```

## Project Structure

```
├── nba_agent.py          # MCP client — conversation loop and tool routing
├── nba_server.py         # Stats MCP server — 10 tools
├── data/                 # downloaded Kaggle datasets (gitignored)
├── charts/               # generated charts saved here
├── research_log.txt      # full decision trail — every tool call logged
├── session_memory.json   # persistent findings across sessions
└── requirements.txt
```

## Related Projects

- [NBA-Trade-Sentiment](https://github.com/Darshan-Balaji-ML/NBA-Trade-Sentiment) — Reddit sentiment analysis of NBA trades using VADER and fine-tuned DistilBERT, with Streamlit dashboard and MCP sentiment server (dependency of this project)
- [NBA-timeout-analysis](https://github.com/Darshan-Balaji-ML/NBA-timeout-analysis) — Play-by-play analysis of whether NBA timeouts stop scoring runs using statistical testing and Random Forest classification

## Tech Stack

Python, Anthropic Claude API, Model Context Protocol (MCP), FastMCP, Kaggle API, pandas, matplotlib, BeautifulSoup, Arctic Shift Reddit API, VADER, DistilBERT, Hugging Face Transformers