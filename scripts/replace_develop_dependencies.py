import sys
import toml
from pathlib import Path


def get_latest_version_of_dependency(dependency_path: Path):
    pyproject_path = dependency_path / "pyproject.toml"
    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found in {dependency_path}")

    with open(pyproject_path, "r") as file:
        data = toml.load(file)

    return data["tool"]["poetry"]["version"]


def prepare_for_publishing(pyproject_path: Path):
    with open(pyproject_path, "r") as file:
        data = toml.load(file)

    dependencies = data["tool"]["poetry"]["dependencies"]
    for dep, dep_data in dependencies.items():
        if isinstance(dep_data, dict) and dep_data.get("develop"):
            # Assuming the path is relative to the pyproject.toml being modified
            dependency_path = (pyproject_path.parent / dep_data["path"]).resolve()
            latest_version = get_latest_version_of_dependency(dependency_path)
            # Replace the local path with the latest version
            dependencies[dep] = f"^{latest_version}"

    with open(pyproject_path, "w") as file:
        toml.dump(data, file)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py path/to/your/package/pyproject.toml")
        sys.exit(1)

    pyproject_path = Path(sys.argv[1])
    if not pyproject_path.exists() or pyproject_path.name != "pyproject.toml":
        print("The specified path does not exist or is not a pyproject.toml file.")
        sys.exit(1)

    prepare_for_publishing(pyproject_path)
