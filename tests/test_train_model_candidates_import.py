from pathlib import Path
import py_compile

def test_train_model_candidates_module_imports():
    import src.train_model_candidates  # noqa: F401
    
def test_all_python_files_compile():
    project_root = Path(__file__).resolve().parents[1]

    python_files = [
        path
        for path in project_root.rglob("*.py")
        if ".venv" not in path.parts
        and "__pycache__" not in path.parts
    ]

    for python_file in python_files:
        py_compile.compile(
            str(python_file),
            doraise=True,
        )