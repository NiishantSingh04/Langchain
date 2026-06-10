# LangChain Mastery: Concepts, Patterns & Agent Middleware

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![LangChain Version](https://img.shields.io/badge/langchain-v1.3.4-orange.svg)](https://github.com/langchain-ai/langchain)
[![LangGraph](https://img.shields.io/badge/powered%20by-LangGraph-purple.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, production-grade guide to building intelligent agentic systems using **LangChain** and **LangGraph**. This repository covers core LLM application patterns, including multi-provider model integration, custom tool calling, message type orchestration, schema-driven structured outputs, and advanced agent middleware (e.g., token-aware context summarization and Human-in-the-Loop approval workflows).

---

## 📂 Project Structure

```bash
.
├── langchain/
│   ├── 1-Langchainintro.ipynb     # Introduction to LangChain and basic agents
│   ├── 2-modelintegration.ipynb   # Multi-provider integrations (Gemini, OpenAI, Groq)
│   ├── 3-Tools.ipynb              # Tool binding and manual execution loops
│   ├── 4-Messages.ipynb           # Messages system deep dive & text/message prompts
│   ├── 5-Structuredoutput.ipynb   # Schema parsing with Pydantic, TypedDict & Dataclasses
│   └── 6-Middleware.ipynb         # LangGraph middleware (Summarization & Human-in-the-Loop)
├── .env                           # Local environment credentials (API keys)
├── .gitignore                     # Git exclusion rules
├── main.py                        # Root entry point
├── pyproject.toml                 # uv-compatible project specifications & dependencies
└── requirement.txt                # Standard python pip dependencies
```

---

## 📘 Notebooks Breakdown

### 1. Introduction to LangChain Agents (`langchain/1-Langchainintro.ipynb`)
Provides a foundational introduction to LangChain and agent structures. It demonstrates how to initialize an LLM, bind external tools, and run a conversational loop using the unified `create_agent` factory.
* **Key Concepts:**
  * Initializing a Google Gemini model (`ChatGoogleGenerativeAI`).
  * Defining custom tools with Python type hints and docstrings.
  * Constructing and executing a simple agent with `create_agent`.
* **Example Usage:**
  ```python
  from langchain_google_genai import ChatGoogleGenerativeAI
  from langchain.agents import create_agent

  # Define a mock weather tool
  def get_weather(city: str) -> str:
      """Get the weather for a city."""
      return f"The weather in {city} is sunny."

  # Initialize model
  model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

  # Create the agent
  agent = create_agent(
      model=model,
      tools=[get_weather],
      system_prompt="You are a helpful assistant."
  )
  ```

---

### 2. Model Integration (`langchain/2-modelintegration.ipynb`)
Demonstrates how to integrate multiple major LLM providers (Google Gemini, OpenAI, and Groq) using direct imports and the dynamic `init_chat_model` API. It also illustrates different response collection methods: **Streaming** and **Batching**.
* **Key Concepts:**
  * **Unified Initializer:** Using `init_chat_model` to load models dynamically (e.g., `google_genai:gemini-2.5-flash-lite`, `gpt-4.1`, or `groq:qwen/qwen3-32b`).
  * **Streaming Execution:** Incrementally printing responses chunk-by-chunk to improve perceived latency.
  * **Batching Execution:** Sending multiple inputs in parallel for batch processing.
* **Example Usage:**
  ```python
  from langchain.chat_models import init_chat_model

  # Initialize Groq-powered Qwen model
  model = init_chat_model("groq:qwen/qwen3-32b")

  # Streaming responses
  for chunk in model.stream("Write a short paragraph on Philosophy."):
      print(chunk.content, end="|", flush=True)

  # Batch processing
  responses = model.batch([
      "Why do parrots have colorful feathers?",
      "Explain gravity in simple terms."
  ])
  ```

---

### 3. Custom Tools & Execution Loops (`langchain/3-Tools.ipynb`)
Deep dive into defining tools and handling tool execution manually. While agents handle tool execution automatically, understanding the underlying tool loop is crucial for debugging and custom execution patterns.
* **Key Concepts:**
  * Declaring tools using the `@tool` decorator.
  * Binding tools to models with `model.bind_tools()`.
  * Building a custom execution loop by inspecting `AIMessage.tool_calls` and passing outputs back using `ToolMessage`.
* **Example Usage:**
  ```python
  from langchain.tools import tool
  from langchain.chat_models import init_chat_model

  @tool
  def get_weather(city: str) -> str:
      """Returns weather information for a given city."""
      return f"Rainy in {city}."

  model = init_chat_model("groq:qwen/qwen3-32b")
  model_with_tools = model.bind_tools([get_weather])

  # Invoking the model with tools
  response = model_with_tools.invoke("What's the weather like in Boston?")
  print(response.tool_calls)
  ```

---

### 4. Message Orchestration (`langchain/4-Messages.ipynb`)
Explores the fundamental structure of chat history and system prompts. Explains how models ingest raw data through distinct message types instead of unstructured text strings.
* **Key Concepts:**
  * **SystemMessage:** Governs model personality, limitations, and prompt templates.
  * **HumanMessage:** Represents inputs sent by the user.
  * **AIMessage:** Represents the model's textual responses and tool call requests.
  * **ToolMessage:** Passes execution results back to the model.
* **Example Usage:**
  ```python
  from langchain.messages import SystemMessage, HumanMessage, AIMessage
  
  messages = [
      SystemMessage(content="You are a senior Python developer."),
      HumanMessage(content="Hello!"),
      AIMessage(content="Hello! I'd be happy to help you with Python coding.")
  ]
  ```

---

### 5. Structured Output (`langchain/5-Structuredoutput.ipynb`)
Highlights mechanisms to force LLMs to output structured data schemas instead of free-form text. The notebook demonstrates formatting outputs using **Pydantic**, **TypedDict**, and standard Python **Dataclasses**.
* **Key Concepts:**
  * Schema definitions using Pydantic `BaseModel` fields and descriptions.
  * Mapping structured response formatting directly through `model.with_structured_output()`.
  * Passing structured target schemas to `create_agent(..., response_format=SchemaClass)`.
* **Example Usage:**
  ```python
  from pydantic import BaseModel, Field
  from langchain.agents import create_agent

  class ContactInfo(BaseModel):
      """Contact Information for a person."""
      name: str = Field(description="The Name of the person")
      email: str = Field(description="The email of the person")
      phone: str = Field(description="The phone number of the person")

  agent = create_agent(
      model="groq:qwen/qwen3-32b",
      response_format=ContactInfo
  )

  result = agent.invoke({
      "messages": [{"role": "user", "content": "Extract: John Cena, johncenacantseeme@gmail.com, (242) 124-4134"}]
  })
  print(result["structured_response"])
  ```

---

### 6. Advanced Agent Middleware (`langchain/6-Middleware.ipynb`)
Showcases production-ready middleware capabilities utilizing LangGraph's Runtime engine. This enables intercepting, modifying, pausing, and resuming agent executions.
* **Key Concepts:**
  * **SummarizationMiddleware:** Automatically compiles chat history summaries when thresholds are breached. Supports triggers based on message counts, total tokens, or model context window fractions (e.g., trigger when context is 0.5% full).
  * **HumanInTheLoopMiddleware:** Pauses agent execution when high-risk tools (e.g., `send_email_tool`) are invoked. Execution yields a `__interrupt__` signal, and resumes when a user feedback `Command` is supplied.
  * **Resume Actions:** Supports three modes when resuming: `approve` (proceed), `reject` (cancel tool call), or `edit` (modify tool arguments before execution).
* **Example Usage:**
  ```python
  from langchain.agents import create_agent
  from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware
  from langgraph.checkpoint.memory import InMemorySaver
  from langgraph.types import Command

  # 1. Summarization Middleware Configuration
  agent_sum = create_agent(
      model="groq:qwen/qwen3-32b",
      checkpointer=InMemorySaver(),
      middleware=[
          SummarizationMiddleware(
              model="groq:qwen/qwen3-32b",
              trigger=("tokens", 550),
              keep=("tokens", 200)
          )
      ]
  )

  # 2. Human-In-The-Loop Middleware Configuration
  agent_hitl = create_agent(
      model="groq:qwen/qwen3-32b",
      tools=[send_email_tool],
      checkpointer=InMemorySaver(),
      middleware=[
          HumanInTheLoopMiddleware(
              interrupt_on={"send_email_tool": {"allowed_decisions": ["approve", "edit", "reject"]}}
          )
      ]
  )

  # Resuming from a paused state (Edit example)
  response = agent_hitl.invoke(
      Command(resume={
          "decisions": [{
              "type": "edit",
              "edited_action": {
                  "name": "send_email_tool",
                  "args": {"recipient": "john@gmail.com", "subject": "Hello", "body": "Modified content"}
              }
          }]
      }),
      config={"configurable": {"thread_id": "session-1"}}
  )
  ```

---

## 🛠️ Installation & Setup

### Prerequisites
* **Python 3.12+**
* [uv](https://github.com/astral-sh/uv) (recommended) or standard `pip`
* API Keys from providers (OpenAI, Google Gemini, Groq)

### 1. Clone & Navigate
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. Environment Setup

#### Option A: Using `uv` (Recommended)
`uv` will automatically read `pyproject.toml` and lock file, building a sync virtualenv.
```bash
# Initialize virtual environment and install packages
uv sync
```

#### Option B: Using standard `pip`
```bash
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirement.txt
```

### 3. Configure API Credentials
Create a `.env` file in the root directory (based on the provided template below):
```env
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Running the Notebooks
To boot up Jupyter Lab or Notebook and run through the code interactively:
```bash
# Active virtualenv
source .venv/bin/activate

# Start jupyter
jupyter notebook
```
Navigate to the `langchain/` folder and open any notebook to begin.

---

## 📄 License
This repository is licensed under the [MIT License](LICENSE). Feel free to adapt and use these patterns in your own commercial or private applications.
