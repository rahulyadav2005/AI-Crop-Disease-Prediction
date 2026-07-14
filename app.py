from flask import Flask

from config import Config

from routes.home import home_bp
from routes.predict import predict_bp

app = Flask(__name__)

app.config.from_object(Config)

app.register_blueprint(home_bp)
app.register_blueprint(predict_bp)

if __name__ == "__main__":
    app.run(debug=True)