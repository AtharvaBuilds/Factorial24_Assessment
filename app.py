from flask import Flask, render_template, request, jsonify
from rfq_engine import RFQEngine

app = Flask(__name__)

# Initialize the engine once when the server starts
print("Initializing RFQ Engine (loading models and data)...")
engine = RFQEngine()
print("Engine ready.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_rfq', methods=['POST'])
def process_rfq():
    data = request.get_json()
    rfq_text = data.get('rfq_text', '')
    requested_qty = data.get('requested_qty')
    
    if requested_qty:
        try:
            requested_qty = int(requested_qty)
        except ValueError:
            requested_qty = None
            
    if not rfq_text:
        return jsonify({"error": "No RFQ text provided"}), 400
        
    report = engine.process_rfq(rfq_text, requested_qty)
    return jsonify(report)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
