from langgraph.pregel import Pregel

from agent.graph import graph, route_intent


def test_placeholder() -> None:
    assert isinstance(graph, Pregel)


def test_route_intent_summarize():
    """route_intent returns 'retrieve_full_document' for summarize intent."""
    from agent.graph import State
    state = State(intent="summarize", source_file="test.pdf")
    assert route_intent(state) == "retrieve_full_document"


def test_route_intent_qa():
    """route_intent returns 'rewrite_query' for qa intent."""
    from agent.graph import State
    state = State(intent="qa")
    assert route_intent(state) == "rewrite_query"
