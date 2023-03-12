import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == '__main__':
    packages =['pyemd','gensim','nltk','transformers','datasets','numpy','evaluate','torch','matplotlib','rouge_score','sentencepiece'] #add package here
    for p in packages:
        install(p)