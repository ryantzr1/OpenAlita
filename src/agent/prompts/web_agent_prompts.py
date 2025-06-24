"""
Web Agent Node Prompts

Prompts used by the web agent node for intelligent query decomposition and strategic searching.
"""

TARGETED_SEARCH_PROMPT = """You are a search query optimizer focusing on TARGETED searches to fill specific information gaps.

USER QUERY: {query}
MISSING INFORMATION: {missing_info}

EXISTING RESULTS SUMMARY:
{existing_results_summary}

Create 2-3 TARGETED search queries that specifically address the missing information:
- Focus on the gaps identified: {missing_info}
- Use different keywords/angles than previous searches
- Prioritize authoritative and recent sources

Respond with JSON:
{{"search_queries": ["targeted_query1", "targeted_query2"]}}"""

VERIFICATION_SEARCH_PROMPT = """You are a search query optimizer focusing on VERIFICATION searches.

USER QUERY: {query}
EXISTING INFORMATION: Need to verify and cross-check existing results

Create 2-3 VERIFICATION search queries:
- Use different keywords to find alternative sources
- Focus on fact-checking and authoritative sources
- Look for contradictory or confirming information

Respond with JSON:
{{"search_queries": ["verification_query1", "verification_query2"]}}"""

BROADER_SEARCH_PROMPT = """You are a search query optimizer. Break down this user query into 2-4 focused search queries that will get the best web search results.

USER QUERY: {query}

GUIDELINES:
- Create specific, focused search queries (not the original long query)
- Each query should target a different aspect of the question
- Use search-engine friendly terms (avoid long sentences)
- Prioritize current/recent information when relevant
- Maximum 4 search queries

Respond with a JSON array of search queries:
{{"search_queries": ["query1", "query2", "query3"]}}

Examples:
- "What's Tesla's stock price and how does it compare to Ford?" 
  → {{"search_queries": ["Tesla stock price today", "Ford stock price 2024", "Tesla vs Ford stock comparison"]}}
  
- "Latest news about AI and machine learning developments"
  → {{"search_queries": ["latest AI news 2024", "machine learning breakthroughs recent", "AI development trends"]}}""" 