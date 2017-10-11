#!/bin/bash
# Linux Installation
sudo apt-get install python-pip python-dev build-essential
sudo pip install virtualenv virtualenvwrapper
sudo pip install --upgrade pip
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt