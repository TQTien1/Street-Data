from flask import Flask, render_template
import folium

app = Flask(__name__)

@app.route('/')
def map():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)