import torch
from models.autoencoder import Autoencoder
from models.MLP import MLP
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

autoencoder_model = Autoencoder(encoder_dims=[47, 32, 24, 20, 16, 10], decoder_dims=[10, 16, 20, 24, 32, 47])

autoencoder_model.load_state_dict(torch.load("model_checkpoints/autoencoder_best_model.pth", weights_only=True))
autoencoder_model.eval()

mlp_model = MLP(dims=[384, 256, 128, 64, 32, 16, 10])
mlp_model.load_state_dict(torch.load("model_checkpoints/mlp_best_model.pth", weights_only=True))
mlp_model.eval()


# Define a route
@app.route('/')
def encoder():
    data = request.get_json()

    # Create a tensor from the input JSON data
    input_tensor = torch.tensor(data["input"], dtype=torch.float32)

    # Pass the input to the autoencoder model to get the encoded result
    res = autoencoder_model.encode(input_tensor)

    # Convert the result tensor to a Python list (or NumPy array)
    res_list = res.tolist()  # Converts the tensor to a list

    # Return the result as a JSON response
    return jsonify({"result": res_list}), 200

@app.route('/decoder')
def decoder():
    data = request.get_json()

    # Create a tensor from the input JSON data
    input_tensor = torch.tensor(data["input"], dtype=torch.float32)

    # Pass the input to the autoencoder model to get the decoded result
    res = autoencoder_model.decode(input_tensor)

    # Convert the result tensor to a Python list (or NumPy array)
    res_list = res.tolist()  # Converts the tensor to a list

    # Return the result as a JSON response
    return jsonify({"result": res_list}), 200

@app.route('/autoencoder')
def autoencoder():
    data = request.get_json()

    # Create a tensor from the input JSON data
    input_tensor = torch.tensor(data["input"], dtype=torch.float32)

    # Pass the input to the autoencoder model to get the result
    res = autoencoder_model(input_tensor)

    # Convert the result tensor to a Python list (or NumPy array)
    res_list = res.tolist()  # Converts the tensor to a list

    # Return the result as a JSON response
    return jsonify({"result": res_list}), 200

@app.route('/mlp')
def mlp():
    data = request.get_json()

    # Create a tensor from the input JSON data
    input_tensor = torch.tensor(data["input"], dtype=torch.float32)

    # Pass the input to the mlp model to get the encoded result
    res = mlp_model(input_tensor)

    # Convert the result tensor to a Python list (or NumPy array)
    res_list = res.tolist()  # Converts the tensor to a list

    # Return the result as a JSON response
    return jsonify({"result": res_list}), 200


# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True)


