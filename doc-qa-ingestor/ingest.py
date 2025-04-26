import os
from fastapi import FastAPI, UploadFile, File
import uvicorn
from vectorize import chunk_and_normalize, create_local_vector_index
import boto3

app = FastAPI()
s3 = boto3.client('s3')
BUCKET = os.getenv('BUCKET_NAME')

@app.post('/upload/')
async def upload_document(file: UploadFile = File(...)):
    content = await file.read()
    chunks = chunk_and_normalize(content.decode('utf-8'))
    idx = create_local_vector_index(chunks)
    idx.save_local('faiss.index')
    s3.upload_file('faiss.index', BUCKET, 'faiss.index')
    return {'message': 'Index uploaded'}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8083)
