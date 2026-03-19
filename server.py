import os
from datetime import datetime, timedelta
import finnhub
from mcp.server.fastmcp import FastMCP

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

    recent_count = len(recent)
    daily_average = len(baseline) / 7

    if daily_average == 0:
        ratio = 0
    else:
        ratio = recent_count / daily_average

    summary = f"Symbol: {symbol}\n"
    summary += f"News articles (last 24hrs): {recent_count}\n"
    summary += f"Daily average (7-day): {daily_average:.1f}\n"
    summary += f"Volume Ratio: {ratio:.1f}x\n"

    if ratio >= 2:
        summary += "⚠️ Unusual news volume detected\n"
    else:
        summary += "✅ Normal news volume\n"

    summary += "\nRecent headlines:\n"
    for article in recent[:5]:  # Show up to 5 recent articles
        summary += f"- {article['headline']}\n"

    return summary

if __name__ == "__main__":
    mcp.run()

