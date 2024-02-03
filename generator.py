import os
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import time
import json
import numpy as np

import json
import itertools
import torch
import torch.nn as nn
from torch.autograd import Variable

import logging

logging.basicConfig(filename='generator.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

def generate_cases(json_file, output_dir):
    try:
        logging.info(f"Generating cases from {json_file}...")

        with open(json_file, 'r') as f:
            op_info = json.load(f)

        op_name = op_info['op_name']
        input_shapes_info = op_info['input_shapes'][0] 
        attributes_info = op_info['attributes']

        # Generate all possible values for input shapes and attributes
        input_shapes = {k: list(range(v[0], v[1] + 1, v[2])) for k, v in input_shapes_info.items()}
        attributes = {k: list(range(v[0], v[1] + 1, v[2])) for k, v in attributes_info.items()}

        op_dir = os.path.join(output_dir, op_name)
        os.makedirs(op_dir, exist_ok=True)

        exporter = ModelExporter()

        for shape_values in itertools.product(*input_shapes.values()):
            shape = dict(zip(input_shapes.keys(), shape_values))
            for attr_values in itertools.product(*attributes.values()):
                attr = dict(zip(attributes.keys(), attr_values))

                model = generate_model(op_name, shape, attr)

                shape_str = json.dumps(shape).replace(':', '_').replace('\'', '').replace(' ', '').replace('{', '').replace('}', '').replace('\"', '').replace(',', '_')
                attr_str = json.dumps(attr).replace(':', '_').replace('\'', '').replace(' ', '').replace('{', '').replace('}', '').replace('\"', '').replace(',', '_')
                model_path = os.path.join(op_dir, f'{op_name}_{shape_str}_{attr_str}.pt')
                torch.save(model.state_dict(), model_path)

                # Export the model to ONNX nchw format
                onnx_path = model_path.replace('.pt', '.onnx')
                input_shape = [1] + [shape[k] for k in ['channels','height', 'width' ]]  
                exporter.export_to_onnx(model, onnx_path, input_shape)
    except Exception as e:
        logging.error(f"Error in generate_cases: {str(e)}")


def generate_model(op_name, shape, attr):
    try:
        logging.info(f"Generating model for operation {op_name}...")

        op_class = getattr(nn, op_name)

        # Create the model
        if op_name == 'Conv2d':
            # For Conv2d, the parameters are (in_channels, out_channels, kernel_size, stride, padding, groups)
            model = op_class(shape['channels'], attr['out_channels'], attr['kernel_size'], attr['stride'], attr['padding'], attr['groups'])
        elif op_name == 'ReLU':
            # For ReLU, the only parameter is 'inplace'
            model = op_class(attr.get('inplace', False))  # Use False as the default value if 'inplace' is not provided
        else:
            # For other operations, you might need to adjust the parameters
            model = op_class(*shape.values(), **attr)

        return model

    except Exception as e:
        logging.error(f"Error in generate_model: {str(e)}")

class ModelExporter:
    def export_to_onnx(self, model, filename, input_shape):
        """
        convert pt models to onnx
        verify with onnx
        """
        try:
            logging.info(f"Exporting model to ONNX format at {filename}...")

            # Input data format is NCHW
            dummy_input = torch.randn(*input_shape)
            torch.onnx.export(model, dummy_input, filename)

            # Verify the model
            onnx_model = onnx.load(filename)
            onnx.checker.check_model(onnx_model)

        except Exception as e:
            logging.error(f"Error in export_to_onnx: {str(e)}")

class Profiler:
    def run_cases_with_ort(self, model, filename, input_shape):
        """
        Profile the time
        """

        try:
            logging.info(f"Running cases with ONNX Runtime for model at {filename}...")

            # Load the ONNX model
            ort_session = ort.InferenceSession(filename)

            # Run the model with ONNX Runtime
            start_time = time.time()
            ort_session.run(None, {'input': torch.randn(*input_shape).numpy()})
            end_time = time.time()

            logging.info(f"Execution time with ONNX Runtime: {end_time - start_time} seconds")

        except Exception as e:
            logging.error(f"Error in run_cases_with_ort: {str(e)}")


if __name__ == '__main__':
    generate_cases('op_info/Conv.json', 'models')
    generate_cases('op_info/Relu.json', 'models')