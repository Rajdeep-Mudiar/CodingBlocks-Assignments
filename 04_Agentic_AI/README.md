# Agentic AI: Blog Generation & Evaluation Workflow

This repository contains a LangGraph-based agentic workflow that automates the generation and evaluation of high-quality blog posts. It leverages prompt chaining within a state machine (StateGraph) powered by the `llama-3.3-70b-versatile` model via Groq.

## Project Structure

- `02_prompt_chaining_blog_evaluate.ipynb`: The main Jupyter Notebook implementing the LangGraph state machine, node functions, and execution.
- `requirements.txt`: Python package requirements for running this project.

## Workflow Architecture

The workflow is modeled as a state machine using **LangGraph**. The state is defined using a `BlogState` dictionary:

```python
class BlogState(TypedDict):
    title: str       # The title/topic of the blog post
    outline: str     # The generated outline structure
    content: str     # The final written blog content
    score: float     # The evaluation score and feedback
```

### Nodes & Edges

The workflow runs sequentially through the nodes, prompting the user for input and invoking the LLM at each step:

```mermaid
graph TD
    __start__([__start__]) --> UserInput[User enters a topic name]
    UserInput --> create_outline[create_outline Node]
    create_outline --> LLM1[Groq LLM: Generate Outline]
    LLM1 --> create_blog[create_blog Node]
    create_blog --> LLM2[Groq LLM: Generate Blog Content]
    LLM2 --> evaluation[evaluation Node]
    evaluation --> LLM3[Groq LLM: Score & Critiques]
    LLM3 --> __end__([__end__])

    style __start__ fill:#4CAF50,stroke:#388E3C,color:#fff
    style __end__ fill:#F44336,stroke:#D32F2F,color:#fff
    style UserInput fill:#FF9800,stroke:#F57C00,color:#fff
    style create_outline fill:#2196F3,stroke:#1976D2,color:#fff
    style create_blog fill:#2196F3,stroke:#1976D2,color:#fff
    style evaluation fill:#2196F3,stroke:#1976D2,color:#fff
    style LLM1 fill:#9C27B0,stroke:#7B1FA2,color:#fff
    style LLM2 fill:#9C27B0,stroke:#7B1FA2,color:#fff
    style LLM3 fill:#9C27B0,stroke:#7B1FA2,color:#fff
```

### Sequence Flow

The interaction sequence among the Runner (Jupyter Notebook), the LangGraph Engine, and the Groq LLM is illustrated below:

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Notebook Runner
    participant LG as LangGraph Workflow
    participant LLM as ChatGroq (Llama 3.3)

    User->>LG: Invoke with {"title": topic}
    activate LG
    
    Note over LG: Node: create_outline
    LG->>LLM: Prompt: Generate outline for topic
    LLM-->>LG: Detailed outline
    Note over LG: Update state: outline
    
    Note over LG: Node: create_blog
    LG->>LLM: Prompt: Write blog using outline
    LLM-->>LG: Blog post content
    Note over LG: Update state: content
    
    Note over LG: Node: evaluation
    LG->>LLM: Prompt: Rate & critique blog content
    LLM-->>LG: Score and critique text
    Note over LG: Update state: score
    
    LG-->>User: Return final BlogState dict
    deactivate LG
```

1. **`create_outline`**: Takes the blog title and prompts the Groq LLM to generate a detailed outline.
2. **`create_blog`**: Takes the title and the outline to write a comprehensive blog post.
3. **`evaluation`**: Rates the final blog post out of 10 and provides strengths, weaknesses, and improvement suggestions.

---

## Setup & Installation

### 1. Clone & Navigate
Navigate to this project directory:
```bash
cd 04_Agentic_AI
```

### 2. Configure Environment Variables
Create a `.env` file in the root of this folder and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Install Dependencies
It is recommended to use a virtual environment:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## Usage

1. Open the Jupyter Notebook `02_prompt_chaining_blog_evaluate.ipynb` using VS Code or Jupyter Lab.
2. Run all cells sequentially.
3. Input your desired blog topic when prompted.
4. The workflow will execute the Graph and return the output dictionary containing:
   - The generated detailed outline.
   - The final blog post content.
   - A detailed evaluation score (out of 10) with suggestions for improvement.
