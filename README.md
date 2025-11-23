# ECS171G13

## Database

https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

## Installation

### For Linux/Mac Users

Use python virtual enviroment to make sure all dependencies work with eachother.

#### Step 1: Create virtual enviroment

```bash
chmod +x install_req.bash
./install_req.bash
```

#### Step 2: Select virtual enviroment from notebook kernel or

```bash
source venv/bin/activate
```

If you want to use a python script.

### For Windows Users

run the following command in your terminal to create a virtual environment:

```powershell
py -3.11 -m venv .\venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r .\requirements.txt
```

#### Step 2: (Optional) Register the virtual environment as a Jupyter kernel

After activating the virtual environment, run the following commands:

```powershell
.\venv\Scripts\python.exe -m pip install --upgrade ipykernel
.\venv\Scripts\python.exe -m ipykernel install --user --name ecs171g13_venv --display-name "ecs171g13 (venv)"
```

This will allow you to select the virtual environment as a kernel in Jupyter Notebook.
