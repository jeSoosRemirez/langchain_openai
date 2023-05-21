### Overall dependencies
    - python=3.10
    - poetry=1.5.0

### Run application
    $ poetry install
    $ poetry run python -m uvicorn main:app --reload

### How to use
    - Run cp .env.example .env
    - Set environment variables in the .env file(variables described in "Files description")
    - Go to http://127.0.0.1:8000/docs
    - Provide api token that you can find in .env file as API_KEY i.e. Bloyq5zvzLTm?gC1WiVQwDg72nZ=svMYf!PNKUxR=E4cHF=TUuLnWyLQUL1AG=Hy

### Run tests
    $ poetry run python -m pytest tests.py

### Files description
    .env
    Environment variables:
    OPEN_API_KEY - key for OpenAI API
    FILE_NAME - name of .pdf file with the data for GPT
    API_KEY - randomly generated key, used for auth

    main.py
    FastAPI endpoints

    auth.py
    Logic for token auth

    ai_part.py
    Logic for asking questions to OpenAI

    Dockerfile
    Dockerfile for poetry, doesn't work right now

    pyproject.toml
    Dependencies

### ISSUES:
    Issue 1:
    Custom prompt for RetrievalQA of Langchain doesn't work, while the documentation
    says otherwise(https://python.langchain.com/en/latest/modules/chains/index_examples/vector_db_qa.html).
    The similar issue here: https://github.com/hwchase17/langchain/issues/2574
    Fixed by passing template directly when asking a question to OpenAI.

    Issue 2:
    Dockerfile doesn't builds. Error:
    "Note: This error originates from the build backend, and is likely not a problem with poetry but with 
    hnswlib (0.7.0) not supporting PEP 517 builds. You can verify this by running
    'pip wheel --use-pep517 "hnswlib (==0.7.0)"'."
    The problem is probably in "chromadb" or its dependencies like "hnswlib" or "duckdb".
    I tried to add needed commands to Dockerfile, use pip instead of poetry - it didn't help.
    The probable fix is by changing "chromadb" to something else, but I didn't have much time for this.
