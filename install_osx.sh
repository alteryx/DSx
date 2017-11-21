#!/bin/bash
# Max OS X Installation
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
brew update
echo "# Homebrew" >> ~/.bash_profile
echo "export PATH=/usr/local/bin:$PATH" >> ~/.bash_profile
source ~/.bash_profile
brew install python
brew install gcc@5 --without-multilib
pip install virtualenv virtualenvwrapper
virtualenv venv