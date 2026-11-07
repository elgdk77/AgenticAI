import json

import pytest
from langchain_core.messages import AIMessage

from tests.conftest import scripted_model


def upload(client, name="notes.txt", content=b"The launch date is March 3."):
    return client.post("/api/documents", files={"file": (name, content)})


def search_call(query="launch date", **arguments):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "search_documents", "args": {"query": query, **arguments}, "id": "call_1"}
        ],
    )


def test_upload_list_and_delete(make_client):
    client = make_client()

    created = upload(client)
    assert created.status_code == 201
    assert created.json()["filename"] == "notes.txt"

    assert [d["id"] for d in client.get("/api/documents").json()] == [created.json()["id"]]

    assert client.delete(f"/api/documents/{created.json()['id']}").status_code == 204
    assert client.get("/api/documents").json() == []
    assert client.delete(f"/api/documents/{created.json()['id']}").status_code == 404


def test_uploading_the_same_file_twice_keeps_one_entry(make_client):
    client = make_client()

    upload(client)
    upload(client)

    assert len(client.get("/api/documents").json()) == 1


def test_upload_strips_client_supplied_paths(make_client):
    client = make_client()

    assert upload(client, name="../../etc/passwd.txt").json()["filename"] == "passwd.txt"
    assert upload(client, name="..\\..\\evil.md", content=b"x").json()["filename"] == "evil.md"


def test_upload_rejects_unsupported_and_oversized_files(make_client):
    client = make_client()

    assert upload(client, name="virus.exe").status_code == 422
    assert upload(client, content=b"x" * (25 * 1024 * 1024 + 1)).status_code == 413


def test_workspaces_are_isolated(make_client):
    owner, other = make_client(), make_client()
    document_id = upload(owner).json()["id"]

    assert other.get("/api/documents").json() == []
    assert other.delete(f"/api/documents/{document_id}").status_code == 404
    assert len(owner.get("/api/documents").json()) == 1


@pytest.mark.parametrize(
    "headers",
    [
        {"Sec-Fetch-Site": "cross-site"},
        {"Sec-Fetch-Site": "same-site"},  # a sibling subdomain, which SameSite=Lax allows
        {"Origin": "https://evil.example"},
        {"Origin": "null"},
    ],
)
def test_cross_site_writes_are_rejected(make_client, headers):
    client = make_client()
    document_id = upload(client).json()["id"]

    rejected_upload = client.post(
        "/api/documents", files={"file": ("x.txt", b"injected")}, headers=headers
    )
    rejected_delete = client.delete(f"/api/documents/{document_id}", headers=headers)

    assert rejected_upload.status_code == rejected_delete.status_code == 403
    assert [d["id"] for d in client.get("/api/documents", headers=headers).json()] == [document_id]


@pytest.mark.parametrize(
    "headers",
    [{"Sec-Fetch-Site": "same-origin"}, {"Origin": "http://testserver"}, {}],
)
def test_same_origin_and_non_browser_writes_are_allowed(make_client, headers):
    assert make_client().post(
        "/api/documents", files={"file": ("notes.txt", b"hi")}, headers=headers
    ).status_code == 201


def test_invalid_workspace_cookie_is_replaced(make_client):
    client = make_client()
    client.cookies.set("workspace_id", "not-a-uuid")

    response = client.get("/api/documents")

    assert "workspace_id" in response.cookies
    assert response.cookies["workspace_id"] != "not-a-uuid"


def test_chat_returns_answer_with_sources(make_client):
    client = make_client(scripted_model(search_call(), AIMessage("It's March 3 [notes.txt].")))
    upload(client)

    response = client.post("/api/chat", json={"question": "When is the launch?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "It's March 3 [notes.txt]."
    assert body["sources"] == [
        {"filename": "notes.txt", "page": None, "excerpt": "The launch date is March 3."}
    ]


def test_agent_search_cannot_reach_another_workspace(make_client, store):
    owner = make_client()
    document_id = upload(owner).json()["id"]

    other = make_client(scripted_model(search_call(document_id=document_id), AIMessage("Nothing.")))
    body = other.post("/api/chat", json={"question": "When is the launch?"}).json()

    assert body["sources"] == []


def test_stream_emits_text_source_and_done_events(make_client):
    client = make_client(scripted_model(search_call(), AIMessage("March 3.")))
    upload(client)

    response = client.post("/api/chat/stream", json={"question": "When?"})

    events = [
        (frame.split("\n")[0].removeprefix("event: "), frame.split("\n")[1].removeprefix("data: "))
        for frame in response.text.strip().split("\n\n")
    ]
    names = [name for name, _ in events]
    assert names[0] == "source"
    assert "text" in names
    assert names[-1] == "done"
    assert "".join(json.loads(data)["text"] for name, data in events if name == "text") == "March 3."


def test_conversation_keeps_history_between_turns(make_client):
    model = scripted_model(AIMessage("First."), AIMessage("Second."))
    client = make_client(model)

    first = client.post("/api/chat", json={"question": "One"}).json()
    client.post(
        "/api/chat", json={"question": "Two", "conversation_id": first["conversation_id"]}
    )

    state = client.app.state.agent._graph.get_state(
        {"configurable": {"thread_id": f"{client.cookies['workspace_id']}:{first['conversation_id']}"}}
    )
    assert [message.text for message in state.values["messages"]] == [
        "One", "First.", "Two", "Second.",
    ]


def test_empty_model_answer_is_reported_as_an_error(make_client):
    client = make_client(scripted_model(AIMessage("")))

    response = client.post("/api/chat", json={"question": "Hello"})

    assert response.status_code == 502


def test_chat_without_api_key_returns_503(make_client):
    client = make_client()

    response = client.post("/api/chat", json={"question": "Hello"})

    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_landing_page_links_to_how_it_works(make_client):
    client = make_client()

    assert 'href="/how-it-works"' in client.get("/").text

    response = client.get("/how-it-works")
    assert response.status_code == 200
    assert "<title>Inside Document Q&amp;A</title>" in response.text
