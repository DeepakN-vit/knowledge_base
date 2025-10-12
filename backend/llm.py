from transformers import pipeline

 
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",    
    max_new_tokens=256,             
    temperature=0.3,               
    repetition_penalty=1.2          
)

def generate_answer(context: str, query: str) -> str:
    """Generate detailed answer from context and user query."""
    prompt = (
        "You are a knowledgeable AI assistant that answers questions based on given documents.\n"
        "Read the following context carefully and provide a clear, detailed answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    result = qa_pipeline(prompt)[0]['generated_text']
    return result.strip()
