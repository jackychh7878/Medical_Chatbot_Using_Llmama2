from setuptools import setup, find_packages
import os
from dotenv import load_dotenv
load_dotenv()

setup(
    name = 'Medical Chatbot',
    version = '0.0.1',
    author = 'Jacky Chong',
    author_email = os.getenv('EMAIL_ADDRESS'),
    packages = find_packages(),
    install_requires = []
)