import json

from intake import list_task_dirs, load_task


def test_loads_new_format():
    task = load_task("inbox/softmax")
    assert task.entry == "softmax"
    assert task.reference_file.endswith("inbox/softmax/reference.py")
    assert task.shape == [512, 1024]
    assert len(task.accuracy) >= 4
    assert task.bench["args"][0]["shape"] == [512, 1024]
    assert "for i in range" in task.source


def test_loads_legacy_shape_format(tmp_path):
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "task.json").write_text(
        json.dumps({"name": "legacy", "entry": "legacy", "shape": [8, 16], "target_ms": 1.0}),
        encoding="utf-8",
    )
    (root / "reference.py").write_text("def legacy(x):\n    return x\n", encoding="utf-8")
    task = load_task(root)
    assert task.shape == [8, 16]
    assert task.bench["args"][0]["shape"] == [8, 16]
    assert task.accuracy
    assert task.target_ms == 1.0


def test_list_task_dirs_finds_all_builtins():
    names = {path.name for path in list_task_dirs("inbox")}
    assert {"softmax", "log_softmax", "layernorm", "rmsnorm"} <= names
