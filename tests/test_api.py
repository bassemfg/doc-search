import os

os.environ.setdefault("USE_MOCK_DB", "true")
os.environ.setdefault("USE_FAKE_EMBEDDINGS", "true")
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017")

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_crud_and_search_cycle():
    create_payload = {
        "title": "Atlas Search",
        "content": "MongoDB Atlas Vector Search matches vectors.",
        "tags": ["mongodb", "atlas"],
        "source": "tests",
        "metadata": {"level": "intro"},
    }
    create_resp = client.post("/documents", json=create_payload)
    assert create_resp.status_code == 200
    created_id = create_resp.json()["data"]["_id"]

    read_resp = client.get(f"/documents/{created_id}")
    assert read_resp.status_code == 200
    assert read_resp.json()["data"]["title"] == "Atlas Search"

    update_resp = client.put(
        f"/documents/{created_id}",
        json={"content": "Vector Search in Atlas", "tags": ["atlas"]},
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["data"]["tags"] == ["atlas"]

    search_resp = client.post("/search", json={"query": "Atlas Vector", "top_k": 3})
    assert search_resp.status_code == 200
    results = search_resp.json()["results"]
    assert any(doc["title"] == "Atlas Search" for doc in results)

    delete_resp = client.delete(f"/documents/{created_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["data"]["deleted_count"] == 1
