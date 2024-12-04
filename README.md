# MoinCC - AI4metabolomics

Analyzing an organism's metabolism is crucial for understanding and monitoring diseases and treatments. Magnetic resonance imaging (MRI) is the only technique to measure the metabolism noninvasively, in vivo, and in vitro.




## Run the App
### Notes
This app is not a fully debugged application. Some cases may lead to bugs. These bugs should not apply to the fitting process itself. If ## requirements

 MakePVIt ius reWe recommend to close the appThis application should work with python version >= 3.10. But we did not tried all python versions. 

### Install Git
To use git, please install it with following the (https://git-scm.com/downloads)[link]


### Install Virtual Envirnoment
You can install all the libaries in a seperate python enviroment

Create Virtual Environment:
```bash
python -m venv .MoinCC
```

Activate the Virtual Environment
```bash
\.MoinCC\Scripts activate.bat
```

### Start the app

1. Download this directory and putit into a place on your system where you like. You can download the directory using either git:

```bash
git clone --depth 1 https://github.com/RATFIVE/MoinCC-AI4metabolomics.git
```
or download the directory from the github directory by 1. opening this link: https://github.com/RATFIVE/MoinCC-AI4metabolomics, 2. press on the green code button and 3. 'Download ZIP'.

2. Use this command to install the required python packages

```bash
pip install -r requirements.txt
```

3. go in the app directory
```bash
cd MoinCC-AI4metabolomics/app
```

4. Start the app
```bash
streamlit run app.py --server.port=8501
```
The app should open now in the browser

You can always access the app by the url. Paste this in your browser(for example Firefox) 
```bash
http://localhost:8501
```

### Quit the app
To quit the app go back to the running terminal and press `CTRL + C`. This is important. If this is not done more and more instances would be created and will eventually fill up the memory(RAM). 
