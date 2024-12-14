import flask
import langchain_groq
import langchain_community
import langchain_core
from dotenv import load_dotenv
import sys

print("Flask version:", flask.__version__)
print("LangChain Groq version:", getattr(langchain_groq, '__version__', 'Version not available'))
print("LangChain Community version:", getattr(langchain_community, '__version__', 'Version not available'))
print("LangChain Core version:", getattr(langchain_core, '__version__', 'Version not available'))
print("Python-dotenv version:", sys.modules.get('dotenv').__version__)
