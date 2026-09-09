from pathlib import Path

from hokage_vision.agents.state import AgentResponse, AgentState, ToolCall


def test_tool_call_defaults() -> None:
    call = ToolCall(name="detect_image", arguments={"path": "a.png"}, status="pending")

    assert call.result is None
    assert call.error is None


def test_agent_state_defaults() -> None:
    state = AgentState(
        user_task="检测图片",
        selected_tools=["detect_image"],
        artifacts=[Path("runs/a.jpg")],
        messages=[],
    )

    assert state.errors == []
    assert state.last_result is None


def test_agent_response_holds_tool_calls_and_suggestions() -> None:
    call = ToolCall(name="detect_folder", arguments={}, status="success", result={"count": 1})
    response = AgentResponse(
        message="done",
        tool_calls=[call],
        artifacts=[],
        suggestions=["next step"],
    )

    assert response.tool_calls[0].status == "success"
    assert response.suggestions == ["next step"]
