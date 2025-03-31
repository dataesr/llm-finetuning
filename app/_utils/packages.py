import sys
import importlib.metadata as metadata


def read_requirements(file_path="requirements.txt"):
    with open(file_path, "r") as file:
        requirements = file.readlines()
    return [req.strip() for req in requirements]


def get_installed_versions():
    requirements = read_requirements()
    installed_versions = {}
    installed_versions["Python"] = sys.version
    for req in requirements:
        package_name = req.split("==")[0]
        try:
            version = metadata.version(package_name)
            installed_versions[package_name] = version
        except metadata.PackageNotFoundError:
            installed_versions[package_name] = None
    return installed_versions
