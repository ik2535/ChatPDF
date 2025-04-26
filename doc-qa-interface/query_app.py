import os
from fastapi import FastAPI, Form
import uvicorn
from retrieve import find_contextual_snippets, ask_llm
import boto3

app = FastAPI()
s3 = boto3.client('s3')
BUCKET = os.getenv('BUCKET_NAME')

@app.on_event('startup')
def load_index():
    s3.download_file(BUCKET, 'faiss.index', 'faiss.index')

@app.post('/query/')
async def query_pdf(question: str = Form(...)):
    context = find_contextual_snippets(question)
    answer = ask_llm(question, context)
    return {'answer': answer}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8084)
