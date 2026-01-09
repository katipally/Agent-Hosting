<div align="center">

# 🤖 Workforce AI Agent

### Your Intelligent Workspace Assistant

**Unify Slack, Gmail, and Notion with AI-powered automation**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-18-61dafb.svg)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg)](https://fastapi.tiangolo.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14+-336791.svg)](https://postgresql.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Documentation](#-documentation)

</div>

---

## 📺 Demo

<!-- Add your demo video here -->
<div align="center">

### 🎬 Video Demo

[![Demo Video](https://img.shields.io/badge/▶️_Watch_Demo-Video-red?style=for-the-badge&logo=youtube)](YOUR_VIDEO_LINK_HERE)

*Click above to watch the full demo video*

<!-- Alternative: Embed video directly -->
<!-- 
<a href="YOUR_VIDEO_LINK_HERE">
  <img src="YOUR_THUMBNAIL_IMAGE_URL" alt="Demo Video" width="600">
</a>
-->

</div>

---

## 📖 About

**Workforce AI Agent** is a comprehensive AI-powered workspace automation platform that brings together your **Slack**, **Gmail**, and **Notion** data into a single, intelligent interface. Using advanced **RAG (Retrieval-Augmented Generation)** with semantic search, the agent can answer questions, automate tasks, and provide insights across all your workplace tools.

### Why Workforce AI Agent?

- **🔗 Unified Workspace**: Stop switching between apps. Access all your data from one place.
- **🤖 AI-Powered**: Natural language queries powered by GPT models with 50+ specialized tools.
- **📊 Data Pipelines**: Sync and index your workspace data for instant semantic search.
- **⚡ Automation**: Create workflows that span multiple platforms automatically.
- **🔒 Secure**: Self-hosted solution with Google OAuth authentication.

---

## ✨ Features

### 🎯 Core Capabilities

| Feature | Description |
|---------|-------------|
| **AI Chat Interface** | Natural language conversations with your workspace data |
| **50+ AI Tools** | Comprehensive Slack, Gmail, and Notion integrations |
| **Semantic Search** | AI-powered search across all platforms using sentence-transformers |
| **Data Pipelines** | Sync Slack, Gmail, and Notion to PostgreSQL with embeddings |
| **Project Tracking** | Cross-platform project aggregation and reporting |
| **Automated Workflows** | Multi-step automations triggered by schedules or events |
| **Calendar Integration** | View and manage events from Google Calendar |
| **Google OAuth** | Secure per-user authentication for Gmail access |

### 🖥️ User Interface Tabs

| Tab | Description |
|-----|-------------|
| **💬 Chat** | AI-powered chat with streaming responses and conversation history |
| **📊 Pipelines** | Run data sync for Slack, Gmail, Notion with real-time progress |
| **📁 Projects** | Create and track projects with AI-powered insights |
| **⚙️ Workflows** | Build and schedule automated multi-platform workflows |
| **📅 Calendar** | Google Calendar integration for event management |
| **👤 Profile** | User settings, OAuth connections, and preferences |

### 📱 Platform Integrations

#### Slack (30+ Tools)
- **Messages**: Send, read, search, update, delete, pin/unpin messages
- **Channels**: Create, archive, manage topics, invite/remove users
- **Files**: Upload and share files with comments
- **Users**: List users, get info, check presence status
- **Analytics**: Channel engagement, sentiment analysis, activity trends

#### Gmail (22+ Tools)
- **Emails**: Read full content, send, search with all Gmail operators
- **Threads**: Complete thread retrieval (unlimited messages)
- **Labels**: Manage labels, filter by folders
- **Organization**: Mark read/unread, archive, add labels

#### Notion (15+ Tools)
- **Pages**: Create, read, update, append content (markdown supported)
- **Databases**: Query, filter, list databases
- **Workspace**: Full workspace search across pages and databases

### 🔍 AI & RAG Features

- **Hybrid RAG Engine**: Combines vector search with keyword matching
- **Sentence Transformers**: Local embeddings with configurable models
- **Cross-Platform Search**: Single query searches all platforms
- **Context-Aware Responses**: AI uses relevant data from your workspace
- **Multi-Tool Chaining**: AI automatically sequences multiple tools

---

## 🛠️ Tech Stack

### Backend
| Technology | Purpose |
|------------|---------|
| **FastAPI** | High-performance async API server |
| **PostgreSQL + pgvector** | Database with vector similarity search |
| **LangChain + LangGraph** | AI orchestration and tool management |
| **sentence-transformers** | Local embedding generation |
| **OpenAI API** | LLM for reasoning and responses |
| **SQLAlchemy** | ORM for database operations |

### Frontend
| Technology | Purpose |
|------------|---------|
| **React 18** | Modern UI framework |
| **TypeScript** | Type-safe development |
| **TailwindCSS** | Utility-first styling |
| **Vite** | Fast build tooling |
| **Radix UI** | Accessible component primitives |
| **Zustand** | State management |
| **React Query** | Server state and caching |
| **Lucide Icons** | Beautiful icon set |

### Integrations
| Service | API |
|---------|-----|
| **Slack** | Slack SDK with Socket Mode |
| **Gmail** | Google API with OAuth 2.0 |
| **Notion** | Notion Client API |
| **OpenAI** | GPT models for AI reasoning |

---

## 📦 Installation

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| Node.js | 18+ |
| PostgreSQL | 14+ |

#### macOS
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.12 node postgresql@14
brew services start postgresql@14
```

#### Windows
- [Python 3.10+](https://www.python.org/downloads/)
- [Node.js 18+](https://nodejs.org/)
- [PostgreSQL 14+](https://www.postgresql.org/download/windows/)

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/Workforce-agent.git
cd Workforce-agent
```

### Step 2: Install Backend Dependencies
```bash
# macOS/Linux
pip3 install -r backend/requirements.txt

# Windows
pip install -r backend/requirements.txt
```

### Step 3: Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

### Step 4: Create Database
```bash
createdb workforce_agent
```

### Step 5: Configure Environment
```bash
# macOS/Linux
cp .env.example .env

# Windows
copy .env.example .env
```

Edit `.env` with your API keys:

```bash
# Required
OPENAI_API_KEY=sk-your-openai-key-here
DATABASE_URL=postgresql://localhost:5432/workforce_agent

# Slack (optional but recommended)
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
SLACK_APP_TOKEN=xapp-your-slack-app-token

# Google OAuth (for Gmail + Authentication)
GOOGLE_CLIENT_ID=your-google-oauth-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-google-oauth-client-secret
GOOGLE_OAUTH_REDIRECT_BASE=http://localhost:8000
FRONTEND_BASE_URL=http://localhost:5173
SESSION_SECRET=change-me-to-a-long-random-string

# Notion (optional)
NOTION_TOKEN=secret_your-notion-key
NOTION_PARENT_PAGE_ID=your-page-id
```

### Step 6: Start the Application

#### Option A: Startup Script (Recommended)
```bash
# macOS/Linux
./START_SERVERS.sh
```

#### Option B: Manual Start
```bash
# Terminal 1 - Backend
cd backend
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend
cd frontend
npm run dev
```

### Step 7: Access the App
Open your browser to: **http://localhost:5173**

---

## 🔑 API Keys Setup

### OpenAI (Required)
1. Go to https://platform.openai.com/api-keys
2. Create new secret key
3. Add to `.env` as `OPENAI_API_KEY`

### Slack
1. Go to https://api.slack.com/apps → Create New App
2. Add OAuth scopes: `channels:history`, `channels:read`, `chat:write`, `users:read`
3. Enable Socket Mode and generate app token (`xapp-`)
4. Install to workspace and copy bot token (`xoxb-`)
5. Add both tokens to `.env`

📖 **Detailed guide**: [Documentation/api_guide.md](./Documentation/api_guide.md)

### Gmail (Google OAuth)
1. Go to https://console.cloud.google.com/
2. Create project and enable Gmail API
3. Create OAuth 2.0 Web client credentials
4. Add redirect URI: `http://localhost:8000/auth/google/callback`
5. Add client ID and secret to `.env`

📖 **Detailed guide**: [Documentation/Auth_plan.md](./Documentation/Auth_plan.md)

### Notion
1. Go to https://www.notion.so/my-integrations
2. Create new integration
3. Copy token (`secret_`)
4. Share target pages with the integration
5. Add token and parent page ID to `.env`

📖 **Detailed guide**: [Documentation/api_guide.md](./Documentation/api_guide.md)

---

## 🚀 Usage

### Chat Interface

The **Chat** tab is your primary interface for interacting with the AI agent. Simply type natural language queries:

#### Slack Examples
```
"Get all slack channel names"
"Show me messages from #engineering"
"Send 'Hello team!' to #general"
"Summarize what happened in #social today"
"Who is the most active user in #support?"
```

#### Gmail Examples
```
"Get emails from john@company.com"
"Find emails with subject 'quarterly report'"
"Show me unread emails from last week"
"Search for emails with attachments about budget"
```

#### Notion Examples
```
"Create a Notion page titled 'Meeting Notes'"
"List all my Notion pages"
"Search Notion for 'project roadmap'"
"Append today's summary to the Status page"
```

#### Cross-Platform Examples
```
"Get messages from #social and save them to Notion"
"Track the Q4 Dashboard project for the last 7 days"
"What is the team working on across all platforms?"
"Search all platforms for 'authentication'"
```

### Pipelines

The **Pipelines** tab syncs your workspace data:

1. **Slack Pipeline**: Syncs channels, users, messages, and files
2. **Gmail Pipeline**: Syncs emails by label with incremental updates
3. **Notion Pipeline**: Syncs pages and databases with content

After syncing, data is embedded for semantic search and available in the Chat tab.

### Projects

The **Projects** tab provides cross-platform project tracking:

- Create projects linked to Slack channels, Gmail labels, and Notion pages
- AI automatically aggregates updates from all sources
- Generate stakeholder-ready reports
- Track progress, action items, and blockers

### Workflows

The **Workflows** tab enables automation:

- Build multi-step workflows spanning platforms
- Schedule recurring automations
- Trigger workflows on events (e.g., new Slack message)
- Example: "Every Monday, summarize #engineering and post to Notion"

---

## 📁 Project Structure

```
Workforce-agent/
├── backend/
│   ├── agent/                      # AI Brain & RAG
│   │   ├── ai_brain.py             # GPT + multi-tool orchestration
│   │   ├── hybrid_rag.py           # Hybrid RAG engine
│   │   ├── langchain_tools.py      # 50+ platform tools
│   │   ├── sentence_transformer_engine.py  # Embeddings
│   │   └── project_tracker.py      # Cross-platform tracking
│   ├── api/
│   │   └── main.py                 # FastAPI REST + WebSocket
│   ├── core/
│   │   ├── config.py               # Configuration
│   │   ├── database/               # SQLAlchemy models
│   │   ├── slack/                  # Slack integration
│   │   ├── gmail/                  # Gmail integration
│   │   └── notion_export/          # Notion integration
│   └── workflows/                  # Workflow engine
│       └── workflow_engine.py      # Automation logic
├── frontend/
│   └── src/
│       ├── App.tsx                 # Main application
│       ├── components/
│       │   ├── chat/               # Chat interface
│       │   ├── pipelines/          # Pipeline views
│       │   ├── projects/           # Project management
│       │   ├── workflows/          # Workflow builder
│       │   ├── calendar/           # Calendar view
│       │   └── auth/               # Authentication
│       └── store/                  # Zustand state
├── Documentation/                  # Setup guides
├── .env.example                    # Environment template
├── START_SERVERS.sh                # Startup script
└── STOP_SERVERS.sh                 # Shutdown script
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[TOOLS_CATALOG.md](./TOOLS_CATALOG.md)** | Complete list of 50+ AI tools with examples |
| **[Documentation/api_guide.md](./Documentation/api_guide.md)** | API setup for Slack, Gmail, Notion |
| **[Documentation/Auth_plan.md](./Documentation/Auth_plan.md)** | Google OAuth configuration |
| **[Documentation/workflow_plan.md](./Documentation/workflow_plan.md)** | Workflow automation guide |
| **[Documentation/HOSTING_DEPLOYMENT_PLAN.md](./Documentation/HOSTING_DEPLOYMENT_PLAN.md)** | Production deployment |
| **[API Docs](http://localhost:8000/docs)** | Interactive API documentation (when running) |

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| **Backend won't start** | Check Python 3.10+, install deps: `pip install -r backend/requirements.txt` |
| **Frontend won't start** | Check Node 18+, install deps: `cd frontend && npm install` |
| **Database connection error** | Ensure PostgreSQL is running: `brew services start postgresql@14` |
| **Slack API not configured** | Verify `SLACK_BOT_TOKEN` (xoxb-) and `SLACK_APP_TOKEN` (xapp-) in `.env` |
| **Gmail not authenticated** | Sign in via the app's Google OAuth flow |
| **Port already in use** | Kill existing: `lsof -ti:8000 \| xargs kill -9` |

### Check Logs
```bash
# Backend logs
tail -f logs/backend.log

# Frontend logs
# Check browser console (F12)
```

### Test API Health
```bash
curl http://localhost:8000/health
```

---

## 🤝 Contributing

Contributions are welcome! Please read the contributing guidelines before submitting PRs.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ for productivity**

*Powered by OpenAI GPT • Built with FastAPI & React*

</div>
