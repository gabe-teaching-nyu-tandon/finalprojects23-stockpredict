# Documentation Build Instructions for Jaccard Similarity

To build the documentation from the source code for Jaccard Similarity, please follow these steps:

## Prerequisites
- Python 3.x
- Sphinx

## Clone or Download
1. Clone or download the source code repository.

## Installation
2. Open a terminal or command prompt and navigate to the project directory where the source code is located.
3. Install the required Python packages by running the following command: pip install -r requirements.txt
4. Run sphinx-quickstart
5. I have included the index.rst file in this repo that basically creates a Documentation for Jaccard Similarity Code. You can replace this file with the original index.rst
6. Also replace the conf.py that I have included in this repository with the original conf.py
7. In `conf.py`, include this:  

  - `import os`
  - `import sys`
  - `sys.path.insert(0, os.path.abspath('.'))`
  - `sys.path.insert(0, os.path.abspath(r'/path/to/source/code.py'))`. 
  
  Make sure to replace the path to source code with your downloaded/cloned path. The ".py" should be "JaccardSimilarity.py"

## Building the Documentation
4. To build the HTML documentation, run the following command: make html

This command will generate the HTML output in the `build/html` directory.

5. To generate a PDF version of the documentation, run the following command: make pdf

This command will convert the source files to a PDF format using rst2pdf. The resulting PDF file will be available in the `build/latex` directory.

## Accessing the Documentation
6. To access and view the generated documentation:
- For the HTML version, open a web browser and navigate to the `build/html/index.html` file.
- For the PDF version, use a PDF viewer application to open the PDF file located at `build/latex/<project_name>.pdf`.




