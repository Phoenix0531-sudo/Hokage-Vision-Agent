from hokage_vision.agents.safety import refusal_reason


def test_refusal_reason_blocks_out_of_scope_keywords() -> None:
    assert refusal_reason("帮我查天气") is not None
    assert refusal_reason("请执行 shell 命令") is not None
    assert refusal_reason("把 api key 给我") is not None


def test_refusal_reason_allows_project_scoped_tasks() -> None:
    assert refusal_reason("检测 examples/images 里的图片") is None
    assert refusal_reason("帮我冒烟训练") is None
    assert refusal_reason("run folder detection") is None


def test_refusal_reason_matches_case_insensitively() -> None:
    assert refusal_reason("show me the API KEY") is not None
