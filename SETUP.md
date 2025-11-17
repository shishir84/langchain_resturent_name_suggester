# Quick Setup Guide

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment
```bash
# Copy the example environment file
copy .env.example .env

# Edit .env file and add your API keys:
# OPENAI_API_KEY=your_actual_openai_key
# SERPAPI_API_KEY=your_actual_serpapi_key
```

### Step 3: Test Installation
```bash
# Run the interactive demo
python interactive_demo.py
```

## 🔑 Getting API Keys

### OpenAI API Key (Required)
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create new secret key
5. Copy and paste into `.env` file

### SerpAPI Key (Optional - for web search)
1. Go to [SerpAPI](https://serpapi.com/)
2. Sign up for free account (100 searches/month free)
3. Go to Dashboard
4. Copy your API key
5. Add to `.env` file

## 🧪 Quick Test

```bash
# Test individual agents
python wikipedia_agent.py
python memory_agent.py

# Test with web search (requires SerpAPI key)
python serpapi_agent.py

# Test everything together
python agent.py
```

## 📁 Project Structure

```
langchain_resturent_name_suggester/
├── agent.py              # Main combined agent
├── wikipedia_agent.py    # Wikipedia specialist
├── serpapi_agent.py      # Web search specialist  
├── memory_agent.py       # Conversation memory
├── interactive_demo.py   # Interactive testing
├── requirements.txt      # Dependencies
├── .env.example         # Environment template
├── .env                 # Your API keys (create this)
├── README.md            # Full documentation
└── SETUP.md            # This quick guide
```

## ❓ Troubleshooting

**Problem**: `ModuleNotFoundError`
**Solution**: Run `pip install -r requirements.txt`

**Problem**: `OpenAI API key not found`
**Solution**: Check your `.env` file has `OPENAI_API_KEY=your_key`

**Problem**: SerpAPI not working
**Solution**: Add `SERPAPI_API_KEY=your_key` to `.env` file

## 🎯 What Each Agent Does

- **Wikipedia Agent**: Answers factual questions using Wikipedia
- **SerpAPI Agent**: Gets current information from web search
- **Memory Agent**: Remembers conversation history
- **Main Agent**: Combines all capabilities with smart routing

Ready to explore AI agents? Run `python interactive_demo.py` and start asking questions!