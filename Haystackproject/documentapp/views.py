import glob

from django.http import HttpResponse
from django.shortcuts import render, redirect
from .models import FileUploadForm
import base64
import PyPDF2
from pathlib import Path
#from pypdf import PdfReader
import os
import pandas as pd
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines import ExtractiveQAPipeline
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.nodes.retriever.sparse import TfidfRetriever
from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from django.template import context
from haystack.nodes.reader import TransformersReader


# Create your views here.
def index(request):
    return render(request, "document.html")


def convert_pdf_to_text(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
    except Exception as e:
        print(f"An error occurred: {e}")
    return text


def upload_file(request):
    uploaded_files=[]
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data['file']
            save_folder = './source_documents/'
            save_path = save_folder + uploaded_file.name
            print(save_path)
            print(uploaded_file.name)
            with open(save_path, 'wb+') as w:
                for chunk in uploaded_file.chunks():
                    w.write(chunk)

            uploaded_files= [f for f in os.listdir(save_folder) if f.endswith(".pdf")]
            print(uploaded_files)
            for path in Path("./").glob("**/*.pdf"):
                text = convert_pdf_to_text(path)
                txt_path = path.parent / (".".join(path.name.split(".")[:-1]) + ".txt")
                if txt_path.exists():
                    print(f"Skip {txt_path} as it already exists")
                    continue
                with open(txt_path, "wt", encoding="utf-8") as fp:
                    fp.write(text)
            uploadFlag = True
            print(uploadFlag)
            # return render(request, 'success.html', {'file_name': uploaded_file.name})
    else:
        form = FileUploadForm()

    return render(request, 'upload.html', {'form': form,'uploaded_files': uploaded_files})


def query(request):
    if request.method == 'POST':
        query = request.POST['query']
        print(query)
        doc_dir = "./source_documents"
        print(doc_dir)
        files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
        print(files_to_index)
        folder_path = os.path.join(os.getcwd(), "source_documents")
        text_files = glob.glob(os.path.join(folder_path, "*.txt"))
        print(text_files)
        document_store = InMemoryDocumentStore(use_bm25=True)
        print(document_store)
        indexing_pipeline = TextIndexingPipeline(document_store)
        indexing_pipeline.run_batch(file_paths=text_files)

        retriever = TfidfRetriever(document_store=document_store)
        retrieved_documents = retriever.retrieve(query)
        # for doc in retrieved_documents:
        # print("Document ID:", doc.id)
        # print("Text:", doc.content)

        reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
        pipe = ExtractiveQAPipeline(reader, retriever)

        prediction = pipe.run(
            query=query,
            params={
                "Retriever": {"top_k": 10},
                "Reader": {"top_k": 5}
            }
        )

        answers = prediction["answers"][0]
        result = answers.answer
        print("Query:", query)
        print("Answer:", result)
        print(type(result))

        return render(request, 'upload.html', {'result': result, 'query': query})
        #return redirect('upload_file')
    else:
        result = []

    return render(request, 'upload.html', {'result': result})
