from flask import Flask
import sys

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello World'


if __name__ == '__main__':
    sys.exit(app.run())
    # app.debug = True
    # app.run(host='0.0.0.0', port=8000)
