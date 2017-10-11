#!/bin/bash
# Linux Installation
sudo apt-get update
sudo apt-get install python-pip python-dev build-essential
pip install --upgrade --user pip
pip install --user virtualenv virtualenvwrapper
virtualenv venv
.venv/bin/activate
pip install -r requirements.txt