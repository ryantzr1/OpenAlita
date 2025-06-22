import os
import requests
import json
import re
from typing import Optional, Dict, Any, List
from urllib.parse import quote_plus, urljoin, urlparse
import time
from .llm_provider import LLMProvider

class WebAgent:
    """Web agent for handling search queries and content extraction."""
    
    def __init__(self):
        self.firecrawl_api_key = os.getenv('FIRECRAWL_API_KEY')
        self.llm_provider = LLMProvider()
        
        # Session for API requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        if self.firecrawl_api_key:
            print("Firecrawl API key found - will use Firecrawl search API")
        else:
            print("Note: FIRECRAWL_API_KEY not found in environment. Will use fallback search methods.")
        
    def can_handle_with_search(self, query: str) -> str:
        """Use LLM to determine if a query can be handled via web search.
        
        Returns:
            "YES" if can be handled by web search
            "NO" if should be handled by MCP agent  
            "HANDOFF_TO_MCP" for specific cases that need MCP tools
        """
        prompt = f"""You are an intelligent assistant that determines whether a user query can be answered effectively through web search or needs specialized tools/capabilities.

User Query: "{query}"

Analyze this query and determine if it can be answered through web search (scraping web pages for information) or if it requires specialized tools/capabilities that only an MCP agent can provide.

Web search CAN handle:
- General factual questions (What is X? Who is Y? When did Z happen?)
- Current events and news
- Definitions and explanations
- Historical information
- Weather information (from weather websites)
- Stock prices and market data (from financial websites)
- Company information and business details
- How-to guides and tutorials
- Product reviews and comparisons
- Scientific facts and research
- Location and business information

Web search CANNOT handle:
- User's personal/local information (IP address, system info, personal files)
- Real-time computational tasks (calculations, data processing)
- File operations (create, read, write files)
- System commands or operations
- API calls to specific services requiring authentication
- Tasks requiring user's current context or environment
- Interactive operations or tool usage

Respond with EXACTLY one word:
- "YES" if the query can be answered through web search
- "HANDOFF_TO_MCP" if it requires MCP agent tools/capabilities

Query: "{query}"
Response:"""

        try:
            response_chunks = []
            for chunk in self.llm_provider._make_api_call(prompt):
                if isinstance(chunk, str) and chunk.startswith("Error:"):
                    # Fallback to basic heuristics if LLM fails
                    return self._fallback_can_handle_decision(query)
                response_chunks.append(chunk)
            
            if not response_chunks:
                return self._fallback_can_handle_decision(query)
            
            response = "".join(response_chunks).strip().upper()
            print(f"LLM decision response: {response}")
            # Clean up response to extract decision
            if "HANDOFF_TO_MCP" in response:
                return "HANDOFF_TO_MCP"
            elif "YES" in response:
                return "YES"
            elif "NO" in response:
                return "HANDOFF_TO_MCP"
            else:
                # Fallback if response is unclear
                return self._fallback_can_handle_decision(query)
                
        except Exception as e:
            print(f"LLM decision error: {e}")
            return self._fallback_can_handle_decision(query)
    
    def _fallback_can_handle_decision(self, query: str) -> str:
        """Fallback decision making using simple heuristics."""
        query_lower = query.lower().strip()
        
        # Queries that definitely need MCP tools
        mcp_patterns = [
            r"\bmy ip( address)?\b",
            r"\bcurrent ip\b",
            r"\bwhat('?s| is) my ip\b",
            r"\bshow my ip\b",
            r"\bsystem info\b",
            r"\bfile system\b",
            r"\bcreate file\b",
            r"\bwrite file\b",
            r"\bread file\b",
            r"\bcalculate\b",
            r"\brun command\b",
            r"\bexecute\b"
        ]
        
        for pattern in mcp_patterns:
            if re.search(pattern, query_lower):
                return "HANDOFF_TO_MCP"
        
        # Everything else can potentially be handled by search
        return "YES"
    
    def search_web(self, query: str, num_results: int = 5, follow_up: bool = True) -> List[Dict[str, Any]]:
        """Enhanced web search with content analysis and follow-up searches."""
        # Initial search
        if self.firecrawl_api_key:
            results = self._search_with_firecrawl_api(query, num_results)
            if results:
                # Analyze and enhance results
                enhanced_results = self._analyze_and_enhance_results(results, query)
                
                # Do follow-up searches if needed and enabled
                if follow_up and len(enhanced_results) < num_results:
                    follow_up_results = self._perform_follow_up_searches(query, enhanced_results)
                    enhanced_results.extend(follow_up_results)
                
                return enhanced_results[:num_results]
        
        # Fall back to basic search if Firecrawl API fails or isn't available
        basic_results = self._fallback_search(query, num_results)
        if follow_up and basic_results:
            enhanced_results = self._analyze_and_enhance_results(basic_results, query)
            follow_up_results = self._perform_follow_up_searches(query, enhanced_results)
            enhanced_results.extend(follow_up_results)
            return enhanced_results[:num_results]
        
        return basic_results
    
    def _search_with_firecrawl_api(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Search using Firecrawl's search API endpoint with content scraping."""
        try:
            # Use Firecrawl's search API endpoint
            api_url = "https://api.firecrawl.dev/v1/search"
            
            # Prepare the request with proper authentication and parameters
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.firecrawl_api_key}"
            }
            
            data = {
                "query": query,
                "limit": num_results,
                "scrapeOptions": {
                    "formats": ["markdown", "links"]
                }
            }
            
            # Request search results with content scraping
            response = self.session.post(api_url, headers=headers, json=data, timeout=30)
            
            # Parse the response
            if response.status_code == 200:
                response_data = response.json()
                
                if response_data.get('success') and response_data.get('data'):
                    results = []
                    
                    for item in response_data['data']:
                        # Extract scraped content
                        scraped_content = ""
                        if item.get('markdown'):
                            scraped_content = item['markdown']
                        elif item.get('content'):
                            scraped_content = item['content']
                        
                        # Fallback to description if no scraped content
                        if not scraped_content:
                            scraped_content = item.get('description', '')
                        
                        result = {
                            'title': item.get('title', 'No title'),
                            'content': scraped_content,
                            'description': item.get('description', ''),
                            'url': item.get('url', ''),
                            'source': 'Firecrawl Search API',
                            'type': 'search_result',
                            'links': item.get('links', [])
                        }
                        results.append(result)
                    
                    return results
            
            print(f"Firecrawl search API returned status {response.status_code}: {response.text[:200]}")
            return []
            
        except Exception as e:
            print(f"Firecrawl API search error: {e}")
            return []
    
    def _fallback_search(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Fallback search using DuckDuckGo."""
        try:
            # First try DuckDuckGo instant answers API
            instant_answer = self._get_duckduckgo_instant_answer(query)
            if instant_answer:
                return [instant_answer]
            
            # Fall back to basic search results
            return self._scrape_search_results(query, num_results)
            
        except Exception as e:
            print(f"Fallback search error: {e}")
            return []

    def _get_duckduckgo_instant_answer(self, query: str) -> Optional[Dict[str, Any]]:
        """Get instant answers from DuckDuckGo API."""
        try:
            url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for abstract (Wikipedia-style answers)
                if data.get('Abstract'):
                    return {
                        'title': data.get('Heading', query),
                        'content': data['Abstract'],
                        'url': data.get('AbstractURL', ''),
                        'source': 'DuckDuckGo Instant Answer',
                        'type': 'instant_answer'
                    }
                
                # Check for definition
                if data.get('Definition'):
                    return {
                        'title': f"Definition of {query}",
                        'content': data['Definition'],
                        'url': data.get('DefinitionURL', ''),
                        'source': 'DuckDuckGo Definition',
                        'type': 'definition'
                    }
                
                # Check for answer (direct answers)
                if data.get('Answer'):
                    return {
                        'title': query,
                        'content': data['Answer'],
                        'url': '',
                        'source': 'DuckDuckGo Direct Answer',
                        'type': 'direct_answer'
                    }
        except Exception as e:
            print(f"DuckDuckGo API error: {e}")
        
        return None
    
    def _scrape_search_results(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        """Scrape search results from DuckDuckGo."""
        try:
            # Use DuckDuckGo HTML search
            search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
            response = self.session.get(search_url, timeout=15)
            
            if response.status_code != 200:
                return []
            
            # Simple regex-based extraction (basic implementation)
            content = response.text
            results = []
            
            # Pattern to match search result links and titles
            result_pattern = r'<a[^>]+href="([^"]+)"[^>]*class="result__a"[^>]*>([^<]+)</a>'
            snippet_pattern = r'<a[^>]+class="result__snippet"[^>]*>([^<]+)</a>'
            
            links = re.findall(result_pattern, content)
            snippets = re.findall(snippet_pattern, content)
            
            for i, (url, title) in enumerate(links[:num_results]):
                # Clean up the URL (DuckDuckGo wraps URLs)
                if url.startswith('/l/?uddg='):
                    # Extract the actual URL from DuckDuckGo's redirect
                    import urllib.parse
                    parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                    actual_url = parsed.get('uddg', [''])[0]
                    if actual_url:
                        url = urllib.parse.unquote(actual_url)
                
                snippet = snippets[i] if i < len(snippets) else ""
                
                results.append({
                    'title': title.strip(),
                    'content': snippet.strip(),
                    'url': url,
                    'source': 'Web Search',
                    'type': 'search_result'
                })
            
            return results
            
        except Exception as e:
            print(f"Search scraping error: {e}")
            return []

    def _analyze_and_enhance_results(self, results: List[Dict[str, Any]], original_query: str) -> List[Dict[str, Any]]:
        """Analyze search results and enhance them with better content extraction."""
        enhanced_results = []
        
        for result in results:
            try:
                # Add relevance scoring
                relevance_score = self._calculate_relevance_score(result, original_query)
                result['relevance_score'] = relevance_score
                
                # Enhance content with summary if it's too long
                content = result.get('content', '')
                if len(content) > 500:
                    result['summary'] = self._summarize_content(content, original_query)
                
                # Add source credibility assessment
                result['credibility_score'] = self._assess_source_credibility(result.get('url', ''))
                
                enhanced_results.append(result)
                
            except Exception as e:
                print(f"Error enhancing result: {e}")
                enhanced_results.append(result)  # Keep original if enhancement fails
        
        # Sort by relevance and credibility
        enhanced_results.sort(key=lambda x: (x.get('relevance_score', 0) + x.get('credibility_score', 0)) / 2, reverse=True)
        return enhanced_results
    
    def _calculate_relevance_score(self, result: Dict[str, Any], query: str) -> float:
        """Calculate relevance score for a search result."""
        try:
            title = result.get('title', '').lower()
            content = result.get('content', '').lower()
            query_lower = query.lower()
            
            # Simple keyword matching score
            query_words = set(query_lower.split())
            title_words = set(title.split())
            content_words = set(content.split())
            
            # Calculate overlap
            title_overlap = len(query_words.intersection(title_words)) / len(query_words) if query_words else 0
            content_overlap = len(query_words.intersection(content_words)) / len(query_words) if query_words else 0
            
            # Weight title higher than content
            return (title_overlap * 0.7 + content_overlap * 0.3)
            
        except Exception:
            return 0.5  # Default score
    
    def _summarize_content(self, content: str, query: str) -> str:
        """Summarize long content using LLM."""
        if len(content) <= 300:
            return content
            
        try:
            summary_prompt = f"""Summarize this content in relation to the query: "{query}"

Content: {content[:1500]}...

Provide a concise summary (2-3 sentences) that directly relates to the query."""
            
            summary_chunks = []
            for chunk in self.llm_provider._make_api_call(summary_prompt):
                if isinstance(chunk, str) and not chunk.startswith("Error:"):
                    summary_chunks.append(chunk)
            
            summary = "".join(summary_chunks).strip()
            return summary if summary else content[:300] + "..."
            
        except Exception:
            return content[:300] + "..."
    
    def _assess_source_credibility(self, url: str) -> float:
        """Assess the credibility of a source based on domain."""
        if not url:
            return 0.5
            
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
            
            # High credibility domains
            high_credibility = [
                'wikipedia.org', 'reuters.com', 'bbc.com', 'cnn.com', 'nytimes.com',
                'wsj.com', 'ft.com', 'bloomberg.com', 'npr.org', 'pbs.org',
                'gov', 'edu', 'nature.com', 'science.org', 'who.int'
            ]
            
            # Medium credibility domains
            medium_credibility = [
                'techcrunch.com', 'wired.com', 'arstechnica.com', 'theverge.com',
                'forbes.com', 'businessinsider.com', 'medium.com'
            ]
            
            for domain_check in high_credibility:
                if domain_check in domain:
                    return 0.9
                    
            for domain_check in medium_credibility:
                if domain_check in domain:
                    return 0.7
                    
            return 0.5  # Default credibility
            
        except Exception:
            return 0.5
    
    def _perform_follow_up_searches(self, original_query: str, existing_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform targeted follow-up searches based on gaps in existing results."""
        try:
            # Analyze existing results to identify gaps
            gap_analysis_prompt = f"""Analyze these search results for the query "{original_query}" and identify information gaps.

Query: {original_query}

Existing Results Summary:
{json.dumps([{'title': r.get('title', ''), 'summary': r.get('content', '')[:200]} for r in existing_results[:3]], indent=2)}

What additional search queries (1-2 specific queries) would help fill information gaps? Focus on:
- Different angles or aspects not covered
- More recent/specific information
- Verification from different sources

Respond with JSON:
{{"follow_up_queries": ["query1", "query2"]}}"""

            response_chunks = []
            for chunk in self.llm_provider._make_api_call(gap_analysis_prompt):
                if isinstance(chunk, str) and not chunk.startswith("Error:"):
                    response_chunks.append(chunk)
            
            response = "".join(response_chunks)
            
            # Parse follow-up queries
            follow_up_queries = []
            try:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    data = json.loads(json_str)
                    follow_up_queries = data.get("follow_up_queries", [])
            except Exception:
                follow_up_queries = []
            
            # Perform follow-up searches
            follow_up_results = []
            for query in follow_up_queries[:2]:  # Limit to 2 follow-up searches
                try:
                    if self.firecrawl_api_key:
                        results = self._search_with_firecrawl_api(query, 2)
                    else:
                        results = self._fallback_search(query, 2)
                    
                    for result in results:
                        result['search_type'] = 'follow_up'
                        result['original_query'] = query
                    
                    follow_up_results.extend(results)
                except Exception as e:
                    print(f"Follow-up search error for '{query}': {e}")
            
            return follow_up_results
            
        except Exception as e:
            print(f"Follow-up search analysis error: {e}")
            return []

    def answer_query(self, query: str) -> Optional[str]:
        """Attempt to answer a query using web search."""
        # Use LLM to determine if this query can be handled by web search
        decision = self.can_handle_with_search(query)
        
        if decision == "HANDOFF_TO_MCP" or decision != "YES":
            return None

        
        try:
            # Search for information
            results = self.search_web(query)
            
            if not results:
                return None
            
            # For instant answers, return directly
            first_result = results[0]
            if first_result.get('type') in ['instant_answer', 'definition', 'direct_answer']:
                response = f"**{first_result['title']}**\n\n{first_result['content']}"
                if first_result.get('url'):
                    response += f"\n\nSource: {first_result['url']}"
                return response
            
            # Format the response for search results
            response = f"Based on web search for '{query}':\n\n"
            
            for i, result in enumerate(results[:3], 1):  # Show top 3 results
                response += f"**{i}. {result['title']}**\n"
                
                if result.get('content'):
                    response += f"{result['content']}\n"
                
                if result.get('url'):
                    response += f"*Source: {result['url']}*\n"
                
                response += "\n"
            
            return response.strip()
            
        except Exception as e:
            print(f"Query answering error: {e}")
        
        return None