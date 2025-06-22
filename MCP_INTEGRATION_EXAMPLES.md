# MCP Integration Examples - Enhanced System

## 🎯 **Overview**

This document demonstrates how the enhanced Open-Alita system now supports **cross-step MCP tool integration**, allowing tools to be created and used during browser automation, web search, and other agent steps.

## 🚀 **Enhanced Architecture**

### **Before (Isolated Steps)**

```
Coordinator → Browser Agent (isolated)
Coordinator → MCP Agent (isolated)
Coordinator → Web Agent (isolated)
```

### **After (Integrated Steps)**

```
Coordinator → Browser Agent + MCP Tools (integrated)
Coordinator → MCP Agent + Browser Context (integrated)
Coordinator → Web Agent + MCP Tools (integrated)
```

## 📋 **Example 1: Browser Automation with MCP Tools**

### **Scenario**: "Go to YouTube, find the latest AI videos, calculate their average view count, and save the results"

### **Step-by-Step Execution**:

#### **1. Coordinator Analysis**

```json
{
  "next_action": "browser_automation",
  "reasoning": "Query involves YouTube (video platform), requires visual analysis and data processing",
  "browser_capabilities_needed": [
    "navigation",
    "data_extraction",
    "visual_analysis"
  ],
  "requires_interaction": true
}
```

#### **2. Enhanced Browser Agent with MCP Integration**

**Step 1: Pre-browser MCP Analysis**

```json
{
  "needs_mcp_tools": true,
  "required_tools": [
    {
      "name": "video_data_processor",
      "description": "Process and analyze video metadata from YouTube",
      "purpose": "Extract and structure video information",
      "execution_timing": "after_browser"
    },
    {
      "name": "view_count_calculator",
      "description": "Calculate average view counts and statistics",
      "purpose": "Perform mathematical analysis on view data",
      "execution_timing": "after_browser"
    },
    {
      "name": "data_saver",
      "description": "Save processed data to file",
      "purpose": "Persist results for future use",
      "execution_timing": "after_browser"
    }
  ],
  "reasoning": "Browser will extract video data that needs processing, calculation, and saving"
}
```

**Step 2: MCP Tool Creation**

```python
# Tool 1: video_data_processor
def video_data_processor(raw_video_data):
    """Process raw video data from browser extraction"""
    processed_videos = []
    for video in raw_video_data:
        processed_videos.append({
            'title': video.get('title', 'Unknown'),
            'views': int(video.get('views', '0').replace(',', '')),
            'duration': video.get('duration', 'Unknown'),
            'upload_date': video.get('upload_date', 'Unknown')
        })
    return processed_videos

# Tool 2: view_count_calculator
def view_count_calculator(processed_videos):
    """Calculate average view count and statistics"""
    if not processed_videos:
        return {"average_views": 0, "total_videos": 0}

    total_views = sum(video['views'] for video in processed_videos)
    average_views = total_views / len(processed_videos)

    return {
        "average_views": average_views,
        "total_videos": len(processed_videos),
        "total_views": total_views,
        "max_views": max(video['views'] for video in processed_videos),
        "min_views": min(video['views'] for video in processed_videos)
    }

# Tool 3: data_saver
def data_saver(processed_data, filename="youtube_analysis.json"):
    """Save processed data to file"""
    import json
    with open(filename, 'w') as f:
        json.dump(processed_data, f, indent=2)
    return f"Data saved to {filename}"
```

**Step 3: Enhanced Task Description**

```
TASK: Go to YouTube, find the latest AI videos, calculate their average view count, and save the results

MCP TOOLS AVAILABLE FOR THIS TASK:
- video_data_processor: Process and analyze video metadata from YouTube (Use: after_browser)
- view_count_calculator: Calculate average view counts and statistics (Use: after_browser)
- data_saver: Save processed data to file (Use: after_browser)

MCP TOOL INTEGRATION INSTRUCTIONS:
- Use these tools when you need computational assistance
- Call tools before, during, or after browser actions as needed
- Process browser data through these tools when appropriate
- Combine browser automation with computational tools for better results
```

**Step 4: Browser Execution with MCP Integration**

1. Browser navigates to YouTube
2. Searches for "latest AI videos"
3. Extracts video data (titles, views, durations)
4. **Post-browser MCP execution**:
   - `video_data_processor(raw_data)` → processed_videos
   - `view_count_calculator(processed_videos)` → statistics
   - `data_saver(statistics)` → saved file

## 📋 **Example 2: Real-time MCP Tool Creation**

### **Scenario**: Browser gets stuck on a CAPTCHA, needs immediate tool creation

### **Real-time Analysis**:

```json
{
  "needs_tool_now": true,
  "tool_requirement": {
    "name": "captcha_solver",
    "description": "Solve CAPTCHA challenges during browser automation",
    "purpose": "Handle authentication challenges that block browser progress",
    "arguments": ["captcha_image", "captcha_type"]
  },
  "reasoning": "Browser is stuck on CAPTCHA, need immediate assistance to continue"
}
```

### **Real-time Tool Creation**:

```python
def captcha_solver(captcha_image, captcha_type="text"):
    """Solve CAPTCHA challenges during browser automation"""
    import base64
    from PIL import Image
    import io

    # Decode base64 image if needed
    if captcha_image.startswith('data:image'):
        captcha_image = captcha_image.split(',')[1]

    # Convert to PIL Image
    image_data = base64.b64decode(captcha_image)
    image = Image.open(io.BytesIO(image_data))

    # Use OCR or CAPTCHA solving service
    # This is a simplified example
    if captcha_type == "text":
        # Use OCR to extract text
        import pytesseract
        solution = pytesseract.image_to_string(image).strip()
        return solution
    else:
        return "CAPTCHA type not supported"
```

## 📋 **Example 3: Web Search with MCP Tools**

### **Scenario**: "Search for the latest AI news and create a sentiment analysis report"

### **Enhanced Web Agent with MCP Integration**:

**Step 1: Search Strategy Analysis**

```json
{
  "search_strategy": "targeted",
  "missing_info": "sentiment analysis capabilities",
  "required_tools": [
    {
      "name": "news_sentiment_analyzer",
      "description": "Analyze sentiment of news articles",
      "purpose": "Provide sentiment analysis for AI news",
      "execution_timing": "after_search"
    }
  ]
}
```

**Step 2: Tool Creation and Integration**

```python
def news_sentiment_analyzer(news_articles):
    """Analyze sentiment of AI news articles"""
    from textblob import TextBlob

    sentiment_results = []
    for article in news_articles:
        blob = TextBlob(article['content'])
        sentiment = blob.sentiment.polarity

        sentiment_results.append({
            'title': article['title'],
            'sentiment': sentiment,
            'sentiment_label': 'positive' if sentiment > 0 else 'negative' if sentiment < 0 else 'neutral',
            'confidence': abs(sentiment)
        })

    return sentiment_results
```

**Step 3: Integrated Execution**

1. Web search for "latest AI news 2024"
2. Extract article content
3. **Post-search MCP execution**:
   - `news_sentiment_analyzer(articles)` → sentiment_report

## 🔧 **Technical Implementation Details**

### **State Management**

```python
class State(MessagesState):
    # Enhanced state to track MCP tools across steps
    mcp_tools_created: List[Dict[str, Any]] = []
    mcp_execution_results: List[str] = []
    browser_context: str = ""
    realtime_tools_created: List[Dict[str, Any]] = []
```

### **Cross-Step Tool Registry**

```python
class MCPRegistry:
    def register_tool(self, name: str, function: Callable, metadata: Dict[str, Any], script_content: str) -> bool:
        # Tools persist across all steps
        pass

    def execute_tool(self, name: str, *args, **kwargs) -> Any:
        # Can be called from any step
        pass

    def search_tools(self, query: str) -> List[MCPTool]:
        # Find existing tools before creating new ones
        pass
```

### **Real-time Tool Creation**

```python
def _create_realtime_mcp_tool_during_browser(query: str, browser_context: str, mcp_factory: MCPFactory, mcp_registry: MCPRegistry, llm_provider: LLMProvider) -> Dict[str, Any]:
    # Creates tools on-demand during browser automation
    pass
```

## 🎯 **Benefits of Enhanced Integration**

### **1. Seamless Workflows**

- Tools created in one step can be used in subsequent steps
- No need to recreate tools for similar tasks
- Persistent tool registry across the entire workflow

### **2. Dynamic Adaptation**

- Real-time tool creation based on current context
- Tools can be created during browser automation when needed
- Intelligent tool reuse and optimization

### **3. Enhanced Capabilities**

- Browser automation + computational tools
- Web search + data processing tools
- Cross-step data flow and tool chaining

### **4. Better Error Handling**

- Fallback mechanisms between different agent types
- Tool creation can help resolve browser automation issues
- Intelligent routing based on tool availability

## 🚀 **Usage Examples**

### **Complex Multi-Step Task**

```
Query: "Go to GitHub, find the top AI repositories, analyze their commit frequency, and create a visualization"

Execution Flow:
1. Coordinator → Browser Agent (GitHub navigation)
2. Browser Agent creates: repository_data_extractor, commit_analyzer
3. Browser Agent executes: repository_data_extractor
4. MCP Agent creates: data_visualizer
5. MCP Agent executes: commit_analyzer + data_visualizer
6. Synthesizer combines all results
```

### **Real-time Problem Solving**

```
Query: "Search for stock prices and calculate portfolio performance"

Execution Flow:
1. Web Agent searches for stock data
2. MCP Agent creates: stock_calculator
3. Browser Agent (if needed for real-time data)
4. Real-time tool creation if browser encounters issues
5. All tools work together to provide comprehensive analysis
```

This enhanced system makes your Open-Alita agent truly robust and capable of handling complex, multi-step tasks with dynamic tool creation and cross-step integration.
