import setuptools
import pkg_resources
import pathlib
from itertools import chain
import sys

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def find_version():
    with open("version", "r") as fh:
        VERSION = fh.read().strip()
    return VERSION

def list_requirements(requirements_path):
    with pathlib.Path(requirements_path).open() as requirements_txt:
        install_requires = [
            str(requirement)
            for requirement
            in pkg_resources.parse_requirements(requirements_txt)
        ]
    return install_requires

class SetupSpec:
    def __init__(self):
        self.version = find_version()
        self.files_to_include: list = []
        self.install_requires: list = list_requirements("src/requirements.txt")

    def get_packages(self):
        return setuptools.find_packages()


if __name__ == '__main__':

    name = 'sihg4sr'

    setup_spec = SetupSpec()
    print(setup_spec.get_packages())

    setuptools.setup(
        name=name,
        version=setup_spec.version,
        author="INTEL",
        author_email="bdf.aiok@intel.com",
        description=
        "Recommender system SDK and reference use cases",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url = "https://github.com/intel-sandbox/recsys-refkits/",
        project_urls={
            "Bug Tracker": "https://github.com/intel-sandbox/recsys-refkits/",
        },
        keywords=(
            "recsys python"
        ),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
        ],
        include_package_data=True,
        package_dir={'sihg4sr': 'src'},
        packages=['sihg4sr'],
        package_data={},
        python_requires=">=3.6",
        #cmdclass={'install': post_install},
        zip_safe=False,
        install_requires=setup_spec.install_requires,
    )