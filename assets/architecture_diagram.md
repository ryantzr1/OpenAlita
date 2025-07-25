# OpenAlita Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                   OpenAlita                                     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Core Agent Layer

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             Core Agent Layer                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                           ┌─────────────────────┐                             │
│                           │    Alita Agent      │                             │
│                           │  (Main Orchestrator)│                             │
│                           │                     │                             │
│                           │ • Command Routing   │                             │
│                           │ • Intent Analysis   │                             │
│                           │ • Tool Selection    │                             │
│                           │ • Response Streaming│                             │
│                           └─────────────────────┘                             │
│                                        │                                       │
│                    ┌───────────────────┼───────────────────┐                  │
│                    │                   │                   │                  │
│                    ▼                   ▼                   ▼                  │
│       ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐ │
│       │    Web Agent        │ │     MCP System      │ │   LLM Provider      │ │
│       │  (Search & Scrape)  │ │ (Dynamic Tools)     │ │  (Intelligence)     │ │
│       │                     │ │                     │ │                     │ │
│       │ • Query Analysis    │ │ • Tool Creation     │ │ • Intent Parsing    │ │
│       │ • Web Search        │ │ • Tool Storage      │ │ • Code Generation   │ │
│       │ • Content Scraping  │ │ • Tool Execution    │ │ • Smart Decisions   │ │
│       │ • Firecrawl API     │ │ • Tool Reuse        │ │ • LiteLLM Support   │ │
│       └─────────────────────┘ └─────────────────────┘ └─────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## MCP Layer

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                   MCP Layer                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐    │
│  │     MCP Box         │  │   MCP Factory       │  │   MCP Functions     │    │
│  │   (Tool Storage)    │  │ (Tool Generator)    │  │  (Executable Tools) │    │
│  │                     │  │                     │  │                     │    │
│  │ • Tool Registry     │  │ • Script Parsing    │  │ • Function Library  │    │
│  │ • Metadata Storage  │  │ • Safe Execution    │  │ • Tool Execution    │    │
│  │ • Tool Retrieval    │  │ • Function Creation │  │ • Result Processing │    │
│  │ • Reuse Logic       │  │ • Error Handling    │  │ • State Management  │    │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## External Services Layer

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            External Services Layer                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐    │
│  │   Search APIs       │  │     LLM APIs        │  │   Storage APIs      │    │
│  │                     │  │                     │  │                     │    │
│  │ • Firecrawl API     │  │ • OpenAI API        │  │ • File Storage      │    │
│  │ • DuckDuckGo API    │  │ • DeepSeek API      │  │ • Database Access   │    │
│  │ • Web Scraping      │  │ • Custom Endpoints  │  │ • Cache Management  │    │
│  │ • Content Analysis  │  │ • Model Selection   │  │ • Data Persistence  │    │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                               Data Flow                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│    User Input                                                                   │
│        │                                                                        │
│        ▼                                                                        │
│    ┌─────────────────┐                                                         │
│    │  Alita Agent    │                                                         │
│    │ Decision Engine │                                                         │
│    └─────────────────┘                                                         │
│             │                                                                   │
│    ┌────────┼────────┐                                                         │
│    │        │        │                                                         │
│    ▼        ▼        ▼                                                         │
│ ┌───────┐ ┌────────┐ ┌──────────────┐                                         │
│ │  Web  │ │Existing│ │ Create New   │                                         │
│ │Search │ │  MCP   │ │    MCP       │                                         │
│ └───────┘ └────────┘ └──────────────┘                                         │
│     │         │             │                                                  │
│     │         │             ▼                                                  │
│     │         │      ┌─────────────┐                                          │
│     │         │      │ LLM Process │                                          │
│     │         │      └─────────────┘                                          │
│     │         │             │                                                  │
│     │         │             ▼                                                  │
│     │         │      ┌─────────────┐                                          │
│     │         │      │MCP Factory  │                                          │
│     │         │      └─────────────┘                                          │
│     │         │             │                                                  │
│     │         │             ▼                                                  │
│     │         │   ┌──────────────────┐                                        │
│     │         │   │Register in MCP   │                                        │
│     │         │   │      Box         │                                        │
│     │         │   └──────────────────┘                                        │
│     │         │             │                                                  │
│     └─────────┼─────────────┘                                                  │
│               │                                                                 │
│               ▼                                                                 │
│    ┌─────────────────────┐                                                     │
│    │Execute & Stream     │                                                     │
│    │    Response         │                                                     │
│    └─────────────────────┘                                                     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```
