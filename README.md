# 📄 Chat with PDF

An AI-powered application that allows users to upload PDF documents and interact with them using natural language queries. The app leverages advanced language models and vector databases to ensure accurate, context-based answers, avoiding AI hallucinations.

---

## 🚀 Hosted App

👉 [Huggingface-Spaces](https://huggingface.co/spaces/COLONEL-HERE/Chat_WIth_Doc)

---

## 🛠️ Tools & Frameworks Used

- **💡 Langchain** – For building the retrieval-augmented generation (RAG) pipeline.
- **🧠 Google Gemini API** – To generate embeddings and responses.
- **📊 Supabase** – As the PostgreSQL vector database using **pgvector** for storing and querying embeddings.
- **📚 PyPDF** – For extracting text from uploaded PDF files.
- **💻 Streamlit** – For building the user interface.
- **🌿 dotenv** – To securely manage API keys and environment variables.
- **🔗 Hugging Face Spaces** – For hosting the cloud application.

---

## ⚙️ How to Set Up & Run the Application

### 1️⃣ **Clone the Repository**
```bash
git clone https://github.com/your-username/chat-with-pdf.git
cd chat-with-pdf
```

### 2️⃣ **Set Up Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4️⃣ **Configure Environment Variables**
Create a `.env` file in the root directory and add:
```env
GEMINI_API_KEY=your_gemini_api_key
SUPABASE_USER=postgres
SUPABASE_PASSWORD=your_postgres_password
SUPABASE_HOST=your_postgress_db_host
```

### 5️⃣ **Run the Application Locally**
```bash
streamlit run app.py
```

---

## 🏗️ Key Architectural Decisions & Trade-offs

### **1. Vector Storage with Supabase**
- **Why Supabase?** Free tier with PostgreSQL + pgvector support.
- **Trade-off:** Cloud-hosted databases can face network/firewall issues on certain platforms (e.g., Hugging Face Spaces).

### **2. Embedding Strategy using Gemini**
- Separate embeddings for documents (**retrieval_document**) and user queries (**retrieval_query**) for optimized similarity search.

### **3. Pooler Configuration for Supabase**
- **Local Development:** Used **Session Pooler** (IPv4-friendly).
- **Cloud Hosting (Hugging Face):** Switched to **Transaction Pooler** due to IPv6 network conflicts.

### **4. Avoiding AI Hallucination**
- Prompt structure ensures that if the answer is not found in the document, the app explicitly responds with:
  > "I cannot find the answer in the provided documents."

### **5. Hosting Constraints**
- **Hugging Face Spaces** faced **IPv6 connection issues** with Supabase. Resolved by switching poolers and ensuring IPv4 compatibility.

---

## 📧 Contact
For any queries, feel free to reach out at [Portfolio@Colonel](https://rishabh-verma-portfolio.vercel.app/).

Happy Coding! 🚀

