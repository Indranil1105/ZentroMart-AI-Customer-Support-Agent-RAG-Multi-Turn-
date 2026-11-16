# 🛒 ZentroMart AI Customer Support Agent

> A production-grade, RAG-powered customer support chatbot for a fictional Indian e-commerce platform, built end-to-end with semantic search, identity verification, multi-turn conversations, and multilingual support.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-EE4C2C?logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔍 **Semantic RAG Search** | Vector similarity search over 21 policy documents using Sentence-Transformers + ChromaDB with TF-IDF reranking |
| 🔐 **Identity Verification** | Order details gated behind phone-suffix/pincode verification with retry tracking |
| 💬 **Multi-Turn Conversations** | Full conversation state management with context carryover across turns |
| 🌐 **Multilingual** | English, Hindi (Devanagari), and Hinglish support with automatic language detection |
| 📊 **Analytics Dashboard** | Interactive Plotly charts analyzing 1,000 orders, 300 customers, and 70 products |
| ✅ **Evaluation Suite** | Automated testing against 52 single questions + 23 multi-turn conversations with precision/recall metrics |
| 🎫 **Escalation Engine** | 7 automatic escalation triggers (high-value damage, fraud, legal, repeat contact, etc.) with ticket creation |
| 📅 **Business Day Calculator** | Holiday-aware date arithmetic for deadline and refund timeline calculations |
| 🚫 **No API Keys Required** | Runs 100% locally — template-based response generation grounded in retrieved KB context |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        Streamlit Frontend (app.py)                       │
│   💬 Chat Interface  │  📊 Analytics Dashboard  │  ✅ Evaluation Suite   │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────────┐
│                     Agent Core (agent/core.py)                           │
│  ConversationManager ──► ToolManager ──► ResponseGenerator               │
│         │                     │                    │                      │
│    Language Detection    Order Lookup        Template-Based               │
│    Intent Detection      Verification       Policy-Grounded              │
│    Escalation Check      Ticket Creation    Multilingual                  │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────────┐
│                      RAG Pipeline (agent/rag.py)                         │
│  KB Chunker ──► Sentence-Transformer Embeddings ──► ChromaDB Vector Store│
│                           │                                              │
│                    TF-IDF Reranker + Document Precedence Boost            │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────────┐
│                         Data Layer                                       │
│  CSV Tables: orders(1K) · customers(300) · products(70) · pincodes(36)   │
│  KB Documents: 21 Markdown files with section IDs · Holiday Calendar     │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the application
streamlit run app.py
```

The agent initializes the RAG pipeline on first launch (downloads the `all-MiniLM-L6-v2` model ~80MB). Subsequent launches use the cached index.

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.13 |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB (persistent) with NumPy fallback |
| Reranking | scikit-learn TF-IDF + Cosine Similarity |
| ML Backend | PyTorch 2.7 |
| Web UI | Streamlit 1.45 |
| Visualization | Plotly, Seaborn, Matplotlib |
| Data Processing | Pandas, NumPy |

---

## 📁 Project Structure

```
zentromart_support_agent/
├── app.py                          # Streamlit main application
├── config.py                       # Central configuration & paths
├── requirements.txt                # Dependencies
├── README.md                       # This file
│
├── agent/                          # Core agent logic
│   ├── core.py                     # Main orchestrator (ZentroMartAgent)
│   ├── conversation.py             # Multi-turn state management
│   ├── tools.py                    # Order lookup & ticket creation
│   ├── rag.py                      # RAG retrieval pipeline
│   ├── response_generator.py       # Template-based response generation
│   ├── business_days.py            # Holiday-aware date calculator
│   └── language_detector.py        # Multilingual language detection
│
├── knowledge_base/                 # Document processing
│   ├── chunker.py                  # KB markdown parser with section IDs
│   ├── embeddings.py               # Sentence-transformer embeddings
│   └── vector_store.py             # ChromaDB + NumPy fallback
│
├── data_loader/                    # CSV data access layer
│   └── loader.py                   # Pandas loader with caching
│
├── evaluation/                     # Testing & metrics
│   ├── evaluator.py                # Question & conversation evaluators
│   ├── metrics.py                  # Precision, recall, must-include scoring
│   └── report.py                   # Report generation with Plotly charts
│
├── ui/                             # Streamlit pages
│   ├── chat_page.py                # Chat interface
│   ├── analytics_page.py           # Data analytics dashboard
│   └── eval_page.py                # Evaluation results dashboard
│
├── tests/                          # Unit tests
│   ├── test_business_days.py
│   ├── test_tools.py
│   ├── test_rag.py
│   └── ...
│
└── chroma_db/                      # Persistent vector index (auto-generated)
```

---

## 🤝 How It Works

A customer message flows through the system in these steps:

1. **Language Detection** — Identifies English, Hindi, or Hinglish using Unicode ranges + keyword heuristics
2. **Intent Classification** — Keyword-based intent detection (return, refund, order status, damage, warranty, etc.)
3. **Order ID Extraction** — Regex extracts 8-digit order IDs from various formats
4. **Identity Verification** — If order-specific, verifies via phone suffix (4 digits) or pincode (6 digits)
5. **Escalation Check** — Evaluates 7 ESC-1 triggers (high-value damage, fraud, legal, repeat contact, etc.)
6. **RAG Retrieval** — Semantic search over KB chunks → TF-IDF reranking → document precedence boosting
7. **Response Generation** — Template-based generation grounded in retrieved policy context with section citations
8. **State Update** — Conversation state updated for multi-turn continuity

---

## 📊 Evaluation

The agent is tested against two practice datasets:

| Dataset | Size | Tests |
|---------|------|-------|
| `questions.csv` | 52 single questions | Decision accuracy, retrieval precision/recall, must-include coverage |
| `conversations.csv` | 23 conversations (96 turns) | Multi-turn flow, verification handling, escalation triggers |

**Metrics computed:**
- **Decision Accuracy** — Does the agent make the right call (answer/clarify/escalate/refuse)?
- **Retrieval Precision** — Are the retrieved KB sections relevant?
- **Retrieval Recall** — Are all gold-standard sections found?
- **Must-Include Score** — Are required facts present in the response?
- **Must-Not-Include** — Does the response avoid forbidden claims?

Run evaluations interactively from the **✅ Evaluation** tab in the Streamlit app.

---

## 📚 Dataset

All data is **synthetic**. ZentroMart is a fictional Indian online store. The dataset includes:

- **21 Knowledge Base Documents** — Policies (returns, refunds, cancellations, delivery, damage, warranty, coupons, payments, Plus membership, marketplace sellers), agent guides (verification, escalation, conversation style), help articles, FAQs (English + Hindi), and marketing pages
- **1,000 Orders** with delivery tracking, returns, refunds, and damage reports
- **300 Customers** with Plus membership status
- **70 Products** across electronics, fashion, home, beauty, books, grocery, and innerwear
- **244 Historical Support Logs** with transcripts and CSAT ratings

Reference date: **20 October 2026**

---

## 📄 License

MIT License. This is a synthetic educational project. No real customer data is used.
