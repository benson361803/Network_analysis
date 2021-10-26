# Network_analysis


Poetry package





[tool.poetry]
name = "MI_EEG"
version = "0.1.0"
description = ""
authors = ["c95xgb <benson0221126@g.ncu.edu.tw>"]

[tool.poetry.dependencies]
python = "^3.6.1"
numpy = "1.19.4"
mne = "0.19.2"
keras = "2.3.1"
scipy = "1.4.1"
PyWavelets = "^1.1.1"
sklearn = "^0.0"
pydot = "^1.4.2"
graphviz = "^0.16"
tensorflow-gpu = "2.4.0"
PyQt5 = "5.15.1"
pandas = "1.0.3"
matplotlib = "3.2.1"
opencv-python = "4.1.2.30"
umap-learn = "^0.5.1"
seaborn = "^0.11.1"
torch = "1.5.1"
tensorflow-addons = "^0.13.0"
pyserial = "^3.5"
qtwidgets = "^0.18"
psutil = "^5.8.0"
memory-profiler = "^0.58.0"
torchsummary = "^1.5.1"

[tool.poetry.dev-dependencies]
black = "^20.8b1"
isort = "^5.6.4"
mypy = "^0.790"
pylint = "^2.6.0"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
