import os
import finnhub
import numpy as np
from datetime import datetime, timedelta
from mcp.server.fastmcp import FastMCP
from collections import Counter

mcp = FastMCP("financial-news")     # Creates the MCP Server
client = finnhub.Client(api_key=os.environ["FINNHUB_API_KEY"]) # Creates the Finnhub Client which reads my Finnhub API Key from the environment variable

def fetch_news(symbol: str, days: int) -> list:
    to_date = datetime.today().strftime('%Y-%m-%d')
    from_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
    response = client.company_news(symbol, _from=from_date, to=to_date)
    return response if response else []

@mcp.tool()
def get_news_volume(symbol: str) -> str:
    """Detect unusual news volume for a stock symbol."""
    
    recent = fetch_news(symbol, days=1)
    baseline = fetch_news(symbol, days=7)

    daily_counts = Counter(
        datetime.fromtimestamp(article['datetime']).strftime('%Y-%m-%d')
        for article in baseline
    )

    mean = np.mean(list(daily_counts.values()))
    std = np.std(list(daily_counts.values()), ddof=1)

    recent_count = len(recent)

    if std == 0:
        z_score = 0
    else:
        z_score = (recent_count - mean) / std

    summary = f"Symbol: {symbol}\n"
    summary += f"News articles (last 24hrs): {recent_count}\n"
    summary += f"Mean (7-day): {mean:.1f}\n"
    summary += f"Standard Deviation (7-day, delta degree of freedom=1): {std:.1f}\n"
    summary += f"Z-score: {z_score:.1f}\n"

    if z_score < 2:
        summary += "✅ Normal news volume\n"
    elif z_score < 3:
        summary += "⚠️ Elevated news volume\n"
    else:
        summary += "🚨 Unusual news volume detected\n"
    
    summary += "\nRecent headlines:\n"
    for article in recent[:5]:  # Show up to 5 recent articles
        summary += f"- {article['headline']}\n"

    return summary

if __name__ == "__main__":
    mcp.run()

