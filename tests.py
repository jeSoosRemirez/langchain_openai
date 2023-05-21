import os

from dotenv import load_dotenv
from fastapi.testclient import TestClient

from main import app, get_loader, qa_prompt, vector_store

load_dotenv()
access_token = os.getenv("API_KEY")

client = TestClient(app)


# ENDPOINT TESTS
def test_info_endpoint_with_valid_api_key():
    # Make a request to the /ask_question/{question} endpoint
    response = client.get(
        "/ask_question/test-question", headers={"access_token": access_token}
    )

    assert response.status_code == 200
    assert "I don't know" in response.json()["result"]


def test_info_endpoint_with_invalid_api_key():
    # Make a request to the /ask_question/{question} endpoint with an invalid API key
    response = client.get(
        "/ask_question/test-question", headers={"access_token": "invalid-api-key"}
    )

    assert response.status_code == 403
    assert response.json() == {"detail": "Could not validate API KEY"}


def test_info_endpoint_with_large_question():
    # Make a request to the /ask_question/{question} endpoint with a large question
    response = client.get(
        "/ask_question/" + ("a" * 4090), headers={"access_token": access_token}
    )

    assert response.status_code == 200
    assert response.json() == {"error": "The question is too big"}


def test_open_endpoint():
    # Make a request to the /open endpoint
    response = client.get("/open")

    assert response.status_code == 200
    assert response.json() == {"default variable": "Open Route"}


# AI RESPONSES TESTS
def test_get_loader():
    # Test the get_loader function
    document = get_loader()
    assert isinstance(document, object)


def test_vector_store():
    # Test the vector_store function
    document = get_loader()
    retriever = vector_store(document)
    assert retriever is not None


def test_qa_prompt():
    # Test the qa_prompt function
    document = get_loader()
    retriever = vector_store(document)
    question = "Hi!"
    response = qa_prompt(retriever, question)
    assert isinstance(response, dict)
    assert "query" in response
    assert "result" in response
    assert "source_documents" in response


def test_qa_prompt_with_unknown_question():
    # Test the qa_prompt function with unknown question
    document = get_loader()
    retriever = vector_store(document)
    question = "Who is the black cat?"
    response = qa_prompt(retriever, question)
    assert isinstance(response, dict)
    assert "I don't know", "contact support@nifty-bridge.com" in response["result"]
