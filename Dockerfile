FROM a205283fc08e

COPY requirements.txt requirements.txt

RUN python -m pip install --upgrade pip

RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple --extra-index-url https://pypi.rasa.com/simple --use-deprecated=legacy-resolver



