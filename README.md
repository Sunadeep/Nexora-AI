# 🎓 Nexora AI

**Your Personal AI Study Assistant**

Nexora AI is an AI-powered study assistant designed to help students learn faster, understand concepts, and prepare for exams.

## ✨ Features

- 💬 **AI Chat** — Ask questions and get simple explanations.
- 📄 **PDF Question Answering** — Upload study materials and ask questions.
- 🖼️ **Image Understanding** — Ask questions about images of notes.
- 🔐 **Authentication** — Sign in with Google or email.
- 💾 **Chat History** — Save and revisit conversations.
- 🌙 **Dark UI** — A clean, dark-purple interface.

## 🛠️ Tech Stack

- Python
- Streamlit
- Groq API
- Supabase
- scikit-learn
- PyPDF

## 🚀 Getting Started

1. Clone this repository.
2. Install the dependencies:

   `pip install -r requirements.txt`

3. Create a `.env` file with your own API credentials:

   ```env
   GROQ_API_KEY=your_groq_api_key
   SUPABASE_URL=your_supabase_project_url
   SUPABASE_KEY=your_supabase_publishable_key
   APP_URL=http://localhost:8501
   ```

4. Start the application:

   `streamlit run webapp.py`

## 🔒 Security

Never commit API keys, passwords, or your `.env` file to GitHub.

## 👨‍💻 Project

Built as a student project to explore AI-powered learning applications.

---

**Nexora AI — Learn faster. Prepare smarter.**
