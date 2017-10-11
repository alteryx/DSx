#!/bin/bash
# Linux Installation
sudo apt-get -y update
sudo apt-get -y install python-pip python-dev build-essential
sudo apt-get -y install python-setuptools
sudo pip install --upgrade --force-reinstall pip==9.0.1
pip install --user virtualenv virtualenvwrapper
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get -y update
sudo apt-get -y install gcc-5 g++-5
virtualenv venv
.venv/bin/activate
pip install -r requirements.txt