"""
nba_server.py
MCP server exposing NBA research tools:
- search_and_download_dataset: finds and downloads Kaggle datasets
- list_downloaded_datasets: shows what's already downloaded
- load_and_profile: loads a CSV and returns a data profile
- run_analysis: runs pandas analysis code on downloaded data
- fetch_bbref_team_stats: scrapes Basketball Reference for team stats
- generate_stats_chart: creates a chart from analysis results
- generate_sentiment_chart: creates a VADER vs DistilBERT comparison chart
- save_finding: saves a finding to session memory
- load_memory: loads past session findings
- generate_hypotheses: generates testable hypotheses from data findings
"""

import os
import sys
import subprocess
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import io
import io as _io
from datetime import datetime
from mcp.server.fastmcp import FastMCP
import anthropic

mcp = FastMCP("nba-research")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
LOG_PATH = os.path.join(BASE_DIR, "research_log.txt")
MEMORY_PATH = os.path.join(BASE_DIR, "session_memory.json")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

exec_globals = {"pd": pd, "io": io, "np": np}
loaded_datasets = {}

claude = anthropic.Anthropic()


class _SuppressStdout:
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = _io.StringIO()
        return self

    def __exit__(self, *args):
        sys.stdout = self._stdout


def log(action: str, detail: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {action}: {detail}\n"
    with open(LOG_PATH, "a") as f:
        f.write(entry)


# ── tool 1: search and download ───────────────────────────────────────────────

@mcp.tool()
def search_and_download_dataset(search_query: str, dataset_ref: str = None) -> str:
    """
    Search Kaggle for NBA datasets and download the best match.
    If dataset_ref is provided (e.g. 'sumitrodatta/nba-aba-baa-stats'),
    download that specific dataset directly. Otherwise search and return options.

    Args:
        search_query: What to search for (e.g. 'nba historical team stats')
        dataset_ref: Optional specific dataset reference to download directly

    Returns:
        Download confirmation with file paths, or list of search results
    """
    log("SEARCH", f"query='{search_query}' ref='{dataset_ref}'")

    if dataset_ref:
        dest = os.path.join(DATA_DIR, dataset_ref.split("/")[-1])
        os.makedirs(dest, exist_ok=True)
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_ref, "-p", dest, "--unzip"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            log("DOWNLOAD_ERROR", result.stderr)
            return f"Download error: {result.stderr}"
        files = os.listdir(dest)
        log("DOWNLOADED", f"{dataset_ref} -> {dest}/ files: {', '.join(files)}")
        return f"Downloaded to {dest}/\nFiles: {', '.join(files)}"
    else:
        result = subprocess.run(
            ["kaggle", "datasets", "list", "--search", search_query, "--csv"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            log("SEARCH_ERROR", result.stderr)
            return f"Search error: {result.stderr}"
        lines = result.stdout.strip().split("\n")[:6]
        log("SEARCH_RESULTS", f"found {len(lines)-1} results for '{search_query}'")
        return "\n".join(lines)


# ── tool 2: list downloaded datasets ─────────────────────────────────────────

@mcp.tool()
def list_downloaded_datasets() -> str:
    """
    List all datasets already downloaded in the data/ directory,
    and all datasets currently loaded in memory.

    Returns:
        List of downloaded dataset folders, CSV files, and loaded dataframes
    """
    results = []

    if os.path.exists(DATA_DIR):
        for folder in os.listdir(DATA_DIR):
            folder_path = os.path.join(DATA_DIR, folder)
            if os.path.isdir(folder_path):
                csvs = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
                results.append(f"{folder}/: {', '.join(csvs) if csvs else 'no CSVs'}")

    if loaded_datasets:
        results.append("\nCurrently loaded in memory:")
        for name, info in loaded_datasets.items():
            results.append(f"  {name} -> variable '{info['var']}' ({info['shape']})")
        results.append("\nTo merge: pd.merge(df_name1, df_name2, on='common_column')")

    log("LIST", f"found {len(results)} items")
    return "\n".join(results) if results else "No datasets downloaded yet."


# ── tool 3: load and profile ──────────────────────────────────────────────────

@mcp.tool()
def load_and_profile(file_path: str, dataset_name: str = "df") -> str:
    """
    Load a CSV file and store it in the shared namespace.
    Each dataset gets its own variable name so multiple can coexist for merging.

    Args:
        file_path: Path to the CSV file (e.g. 'data/nba-aba-baa-stats/Player Stats.csv')
        dataset_name: Variable name to use in analysis (e.g. 'df_stats', 'df_salaries').
                      Defaults to 'df'. Use descriptive names when loading multiple datasets.

    Returns:
        Data profile as a string
    """
    try:
        if not os.path.isabs(file_path):
            file_path = os.path.join(BASE_DIR, file_path)

        with _SuppressStdout():
            df = pd.read_csv(file_path)

        exec_globals[dataset_name] = df
        exec_globals["df"] = df
        loaded_datasets[dataset_name] = {
            "var": dataset_name,
            "path": file_path,
            "shape": f"{df.shape[0]} rows x {df.shape[1]} cols"
        }

        log("LOADED", f"{file_path} as '{dataset_name}' {df.shape}")

        profile = []
        profile.append(f"Loaded as: {dataset_name}")
        profile.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        profile.append(f"\nColumns:\n{', '.join(df.columns.tolist())}")
        profile.append(f"\nDtypes:\n{df.dtypes.to_string()}")
        profile.append(f"\nFirst 3 rows:\n{df.head(3).to_string()}")
        profile.append(f"\nNumeric summary:\n{df.describe().to_string()}")
        null_counts = df.isnull().sum()
        null_counts = null_counts[null_counts > 0]
        if not null_counts.empty:
            profile.append(f"\nNull counts:\n{null_counts.to_string()}")
        if loaded_datasets:
            profile.append(f"\nAll loaded datasets: {list(loaded_datasets.keys())}")

        return "\n".join(profile)
    except Exception as e:
        log("LOAD_ERROR", str(e))
        return f"Error loading file: {e}"


# ── tool 4: run analysis ──────────────────────────────────────────────────────

@mcp.tool()
def run_analysis(code: str) -> str:
    """
    Run pandas analysis code. All loaded datasets are available by their variable names.
    Store output in a variable called result.

    Available variables: df (most recently loaded), plus any named datasets.
    To merge: result = pd.merge(df_stats, df_salaries, on='player_id').to_string()

    Args:
        code: Python/pandas code to execute

    Returns:
        The value of result if set, otherwise confirmation
    """
    try:
        log("ANALYSIS", code[:200].replace("\n", " "))
        with _SuppressStdout():
            exec(code, exec_globals)
        if "result" in exec_globals:
            result = exec_globals.pop("result")
            log("ANALYSIS_RESULT", str(result)[:200].replace("\n", " "))
            return str(result)
        return "Code ran successfully (no result variable set)"
    except Exception as e:
        log("ANALYSIS_ERROR", str(e))
        return f"Error running analysis: {e}"


# ── tool 5: basketball reference team stats ───────────────────────────────────

@mcp.tool()
def fetch_bbref_team_stats(season: int) -> str:
    """
    Scrape team stats for a given NBA season from Basketball Reference.

    Args:
        season: The season end year (e.g. 2023 for the 2022-23 season)

    Returns:
        Team stats as a CSV-formatted string
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        log("BBREF_FETCH", f"season={season}")
        url = f"https://www.basketball-reference.com/leagues/NBA_{season}.html"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            log("BBREF_ERROR", f"status {response.status_code}")
            return f"Error fetching BBRef: status {response.status_code}"

        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", {"id": "per_game-team"})

        if not table:
            return "Could not find team stats table on Basketball Reference."

        with _SuppressStdout():
            df = pd.read_html(str(table))[0]

        df = df[df["Rk"] != "Rk"]
        var_name = f"df_bbref_{season}"
        exec_globals[var_name] = df
        loaded_datasets[var_name] = {
            "var": var_name,
            "path": f"BBRef season {season}",
            "shape": f"{df.shape[0]} rows x {df.shape[1]} cols"
        }
        log("BBREF_LOADED", f"season {season} as '{var_name}' {df.shape}")

        return df.to_csv(index=False)
    except Exception as e:
        log("BBREF_ERROR", str(e))
        return f"Error scraping Basketball Reference: {e}"


# ── tool 6: generate stats chart ─────────────────────────────────────────────

@mcp.tool()
def generate_stats_chart(
    data_code: str,
    chart_type: str,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    filename: str = "chart"
) -> str:
    """
    Generate a chart from analysis data and save it to the charts/ folder.
    Uses the same persistent namespace as run_analysis.

    Args:
        data_code: Python code that produces x and y variables for the chart.
                   Must set x and y variables, and optionally labels for bar charts.
                   IMPORTANT: For year/season data always convert x to strings to avoid axis scaling issues.
                   Example: seasons = df.groupby('season')['3PA'].mean(); x = [str(v) for v in seasons.index.tolist()]; y = seasons.values.tolist()
        chart_type: One of 'line', 'bar', 'scatter'
        title: Chart title
        xlabel: X axis label
        ylabel: Y axis label
        filename: Output filename without extension

    Returns:
        Path to the saved chart file
    """
    try:
        with _SuppressStdout():
            exec(data_code, exec_globals)

        x = exec_globals.pop("x", [])
        y = exec_globals.pop("y", [])
        labels = exec_globals.pop("labels", None)

        if not x or not y:
            return "Error: data_code must set x and y variables"

        log("CHART", f"type={chart_type} title='{title}' filename='{filename}'")

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = {"line": "#1D9E75", "bar": "#534AB7", "scatter": "#D85A30"}
        color = colors.get(chart_type, "#1D9E75")

        if chart_type == "line":
            ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=4)
            ax.fill_between(range(len(y)), y, alpha=0.1, color=color)
        elif chart_type == "bar":
            ax.bar(range(len(y)), y, color=color, alpha=0.85)
            if labels:
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
            else:
                ax.set_xticks(range(len(x)))
                ax.set_xticklabels([str(v) for v in x], rotation=45, ha="right", fontsize=9)
        elif chart_type == "scatter":
            ax.scatter(x, y, color=color, alpha=0.7, s=40)

        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        plt.tight_layout()
        out_path = os.path.join(CHARTS_DIR, f"{filename}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        return f"Chart saved to: {out_path}"
    except Exception as e:
        log("CHART_ERROR", str(e))
        return f"Error generating chart: {e}"


# ── tool 7: generate sentiment chart ─────────────────────────────────────────

@mcp.tool()
def generate_sentiment_chart(
    sentiment_data: str,
    title: str,
    filename: str = "sentiment_chart"
) -> str:
    """
    Generate a VADER vs DistilBERT side-by-side comparison chart.

    Args:
        sentiment_data: Raw output string from a sentiment tool or JSON string
                        with keys: phase, vader_pos, vader_neg, vader_neu, bert_pos, bert_neg, bert_neu, total
        title: Chart title
        filename: Output filename without extension

    Returns:
        Path to the saved chart file
    """
    try:
        import json
        import re

        log("SENTIMENT_CHART", f"title='{title}' filename='{filename}'")

        try:
            data = json.loads(sentiment_data)
            phases = data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            phases = []
            for line in sentiment_data.strip().split("\n"):
                phase_match = re.match(r"(\w+)\s*\((\d+)\s*comments\)", line)
                vader_pos = re.search(r"pos=([\d.]+)%\s+neg=([\d.]+)%\s+neu=([\d.]+)%", line)
                if phase_match and vader_pos:
                    phases.append({
                        "phase": phase_match.group(1).replace("_", " "),
                        "total": int(phase_match.group(2)),
                        "vader_pos": float(vader_pos.group(1)),
                        "vader_neg": float(vader_pos.group(2)),
                        "vader_neu": float(vader_pos.group(3)),
                        "bert_pos": float(vader_pos.group(1)),
                        "bert_neg": float(vader_pos.group(2)),
                        "bert_neu": float(vader_pos.group(3)),
                    })

        if not phases:
            return "Error: could not parse sentiment data."

        n_phases = len(phases)
        fig, axes = plt.subplots(1, n_phases, figsize=(5 * n_phases, 4), squeeze=False)

        sentiment_labels = ["Positive", "Negative", "Neutral"]
        vader_colors = ["#2ecc71", "#e74c3c", "#95a5a6"]
        bert_colors = ["#27ae60", "#c0392b", "#7f8c8d"]

        for i, phase in enumerate(phases):
            ax = axes[0][i]
            x = np.arange(len(sentiment_labels))
            width = 0.35

            vader_vals = [phase.get("vader_pos", 0), phase.get("vader_neg", 0), phase.get("vader_neu", 0)]
            bert_vals = [phase.get("bert_pos", 0), phase.get("bert_neg", 0), phase.get("bert_neu", 0)]

            bars1 = ax.bar(x - width/2, vader_vals, width, label="VADER", color=vader_colors, alpha=0.85)
            bars2 = ax.bar(x + width/2, bert_vals, width, label="DistilBERT", color=bert_colors, alpha=0.85)

            ax.set_title(
                f"{phase.get('phase', f'Phase {i+1}')}\n({phase.get('total', '?')} comments)",
                fontsize=11, fontweight="bold"
            )
            ax.set_xticks(x)
            ax.set_xticklabels(sentiment_labels, fontsize=10)
            ax.set_ylabel("% of Comments" if i == 0 else "")
            ax.set_ylim(0, 100)
            ax.legend(fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.3, linestyle="--")

            for bar in list(bars1) + list(bars2):
                h = bar.get_height()
                if h > 3:
                    ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                            f"{h:.0f}%", ha="center", va="bottom", fontsize=8)

        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        out_path = os.path.join(CHARTS_DIR, f"{filename}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        return f"Chart saved to: {out_path}"
    except Exception as e:
        log("SENTIMENT_CHART_ERROR", str(e))
        return f"Error generating sentiment chart: {e}"


# ── tool 8: save finding ──────────────────────────────────────────────────────

@mcp.tool()
def save_finding(
    question: str,
    finding: str,
    datasets_used: str,
    hypotheses: str = ""
) -> str:
    """
    Save a research finding to session memory for use in future sessions.

    Args:
        question: The question that was investigated
        finding: The key finding or answer
        datasets_used: Which datasets were used (comma separated)
        hypotheses: Any hypotheses generated that weren't yet investigated

    Returns:
        Confirmation that the finding was saved
    """
    try:
        memory = []
        if os.path.exists(MEMORY_PATH):
            with open(MEMORY_PATH) as f:
                memory = json.load(f)

        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "finding": finding,
            "datasets_used": datasets_used,
            "hypotheses": hypotheses
        }
        memory.append(entry)

        with open(MEMORY_PATH, "w") as f:
            json.dump(memory, f, indent=2)

        log("MEMORY_SAVED", f"question='{question[:80]}'")
        return f"Finding saved to memory. Total findings: {len(memory)}"
    except Exception as e:
        log("MEMORY_ERROR", str(e))
        return f"Error saving finding: {e}"


# ── tool 9: load memory ───────────────────────────────────────────────────────

@mcp.tool()
def load_memory(limit: int = 10) -> str:
    """
    Load past research findings from session memory.
    Call this at the start of each session to build on previous work.

    Args:
        limit: How many recent findings to return (default 10)

    Returns:
        Past findings formatted as a research summary
    """
    try:
        if not os.path.exists(MEMORY_PATH):
            return "No past findings in memory. This is a fresh start."

        with open(MEMORY_PATH) as f:
            memory = json.load(f)

        if not memory:
            return "Memory file exists but is empty."

        recent = memory[-limit:]
        lines = [f"Found {len(memory)} past findings. Showing {len(recent)} most recent:\n"]

        for i, entry in enumerate(recent, 1):
            lines.append(f"--- Finding {i} [{entry['timestamp']}] ---")
            lines.append(f"Question: {entry['question']}")
            lines.append(f"Finding: {entry['finding']}")
            lines.append(f"Datasets: {entry['datasets_used']}")
            if entry.get("hypotheses"):
                lines.append(f"Unexplored hypotheses: {entry['hypotheses']}")
            lines.append("")

        log("MEMORY_LOADED", f"loaded {len(recent)} findings")
        return "\n".join(lines)
    except Exception as e:
        log("MEMORY_ERROR", str(e))
        return f"Error loading memory: {e}"


# ── tool 10: generate hypotheses ─────────────────────────────────────────────

@mcp.tool()
def generate_hypotheses(
    data_summary: str,
    question_context: str
) -> str:
    """
    Given a data finding, generate 3 specific testable hypotheses worth investigating next.
    Suggests both stats-based and sentiment-based follow-up angles.
    Call this after every significant analysis to drive deeper research.

    Args:
        data_summary: The analysis result or finding (what the data showed)
        question_context: The original question and any relevant context

    Returns:
        3 specific hypotheses with suggested tools and approach for each
    """
    try:
        log("HYPOTHESES", f"context='{question_context[:100]}'")

        prompt = f"""You are an NBA analytics research assistant helping generate follow-up hypotheses.

Original question: {question_context}

Data finding: {data_summary}

Generate exactly 3 specific, testable hypotheses worth investigating next.
For each hypothesis:
1. State the hypothesis clearly and specifically
2. Explain why this finding suggests it
3. Specify the approach: stats analysis, Reddit sentiment analysis, or both
4. Name the specific data needed (e.g. 'team salary data 2015-2020', 'Reddit r/nba comments about analytics')

Focus on hypotheses that would either:
- Explain WHY the pattern exists (causal)
- Show WHETHER fans noticed or cared about the pattern (sentiment angle)
- Connect to a different aspect of basketball performance (cross-domain)

Be specific about player names, teams, seasons, and metrics where relevant.
Format each hypothesis clearly numbered 1, 2, 3."""

        response = claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        result = response.content[0].text
        log("HYPOTHESES_GENERATED", result[:200].replace("\n", " "))
        return result
    except Exception as e:
        log("HYPOTHESES_ERROR", str(e))
        return f"Error generating hypotheses: {e}"


if __name__ == "__main__":
    mcp.run()