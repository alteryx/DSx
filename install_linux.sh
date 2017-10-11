#!/bin/bash
# Linux Installation
sudo apt-get install python-pip python-dev build-essential
sudo pip install --upgrade pip
sudo pip install virtualenv virtualenvwrapper
virtualenv venv
.venv/bin/activate
pip install -r requirements.txt