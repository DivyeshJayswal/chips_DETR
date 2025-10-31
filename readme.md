How to install:
1. Create a virtual environment
"python -m venv detr_venv"
2. activate the environment
"venv_detr\Scripts\activate"
3. Install all Dependencies
"pip install -r requirements.txt"
4. In order to train a model, We need to create a dataset folder.
5.  In "split_dataset.py", update: IMAGES_DIR,ANNOTATION_FILE,OUTPUT_DIR - then Run "split_dataset.py"