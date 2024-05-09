import os
import toml

root_dir = os.getcwd()
evaluators_dir = os.path.join(root_dir, "evaluators")
pyproject_file = os.path.join(root_dir, "pyproject.toml")

with open(pyproject_file, "r") as file:
    pyproject_data = toml.load(file)

generated_dependencies = ""
generated_extras = ""

evaluator_packages = [
    d
    for d in os.listdir(evaluators_dir)
    if os.path.isdir(os.path.join(evaluators_dir, d))
]

package_names = []
for package in evaluator_packages:
    package_name = f"langevals-{package}"
    optional = "false" if package == "langevals" else "true"
    generated_dependencies += f'{package_name} = {{ path = "evaluators/{package}", develop = true, optional = {optional} }}\n'
    if package != "langevals":
        package_names.append(package_name)
        generated_extras += f'{package} = ["{package_name}"]\n'

generated_extras += 'all = ["' + '", "'.join(package_names) + '"]'

pyproject_data["tool"]["poetry"]["dependencies"].update(
    toml.loads(generated_dependencies)
)
pyproject_data["tool"]["poetry"]["extras"].update(toml.loads(generated_extras))

with open(pyproject_file, "w") as file:
    toml.dump(pyproject_data, file)

print("Updated pyproject.toml with generated evaluator dependencies and extras.")
