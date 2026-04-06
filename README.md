# 🩹 Tattoo Aftercare Consultant Chatbot (RAG/LLM)

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![Framework](https://img.shields.io/badge/RAG-Retrieval--Augmented--Generation-orange?style=for-the-badge)
![API](https://img.shields.io/badge/LLM-Gemini_Pro-CC0000?style=for-the-badge&logo=google&logoColor=white)
![Faithfulness](https://img.shields.io/badge/Faithfulness-1.0-green?style=for-the-badge)

**Tattoo Aftercare Consultant** is a highly reliable, grounded, and focused chatbot designed to provide expert advice on tattoo healing and infection risk mitigation. Built using a Retrieval-Augmented Generation (RAG) architecture, this system prioritizes safety and accuracy by strictly adhering to a verified knowledge base.

---

## ✨ Project Goal

The primary objective is to deliver a supportive consultant that provides only **grounded, verifiable information**. By enforcing extreme constraints on faithfulness, the system explicitly avoids hallucinations, unsupported general knowledge, or speculation, ensuring user trust in high-stakes care advice.

---

## 🤖 Core LLM Mandates (Safety Guardrails)

To ensure maximum reliability, the system operates under a strict set of logic "firewalls":

* **Strict Grounding:** Every output must be supported by the provided source documents.
* **No Guessing:** If the knowledge base lacks an answer, the bot politely defaults to: *"I cannot find that information."*
* **Comprehensive Advice:** If sources offer multiple valid perspectives (e.g., different cleaning methods), the bot presents all options fairly.
* **Accessibility:** Medical and technical terms are translated into "regular person talk" to maintain a warm, casual, and supportive persona.
* **Mandatory Citation:** Every response concludes with a direct citation to the specific document source.
* **Safety Disclaimer:** A clear medical disclaimer is provided upfront, clarifying that the AI is not a replacement for a professional doctor or artist.

---

## 📊 Evaluation Results

The model was rigorously tested on key RAG quality metrics to ensure peak performance in reliability and completeness.

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Faithfulness** | **1.0** | **Perfect Reliability.** 100% of generated facts are directly supported by sources. Zero Hallucination. |
| **Recall** | **0.96** | **Excellent.** The model captures 96% of the relevant information available in the retrieved context. |
| **Precision** | **0.92** | **Very High.** 92% of the generated content is focused and relevant to the user's initial query. |

> **Key Takeaway:** The system excels at trustworthiness (1.0 Faithfulness) while remaining highly comprehensive and on-topic.

---

## ⚙️ Getting Started

### 1. Installation
Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/](https://github.com/)[Your-Username]/tattoo-aftercare-chatbot.git
cd tattoo-aftercare-chatbot
pip install -r requirements.txt
