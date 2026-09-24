# PDF Chat - RAG System

Ek AI-powered chatbot jo kisi bhi PDF document ko padh kar uske sawalon ke jawab deta hai, RAG (Retrieval Augmented Generation) technique use karke.

## Kaise Kaam Karta Hai

1. PDF upload hoti hai aur chote chunks mein todi jati hai
2. Har chunk ko vector embeddings mein convert kiya jata hai (Sentence Transformers)
3. Chunks ko Chroma vector database mein store kiya jata hai
4. Jab user sawal poochta hai, sabse relevant chunks retrieve kiye jate hain
5. Retrieved context + sawal Gemini AI model ko diya jata hai jawab generate karne ke liye

## Tech Stack

- **Python**
- **LangChain** - orchestration framework
- **Google Gemini API** - LLM
- **ChromaDB** - vector database
- **Sentence Transformers** - embeddings
- **PyPDF** - PDF processing

## Kaise Chalayein

1. Google Colab mein notebook kholo
2. Apni Gemini API key `GOOGLE_API_KEY` secret mein add karo
3. Cells order mein run karo
4. PDF upload karo aur sawal poochna shuru karo

## Future Improvements

- Streamlit web interface
- Multiple PDFs support
- Chat history maintain karna
