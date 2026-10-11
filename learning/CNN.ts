import tf from "@tensorflow/tfjs-node";
import { readFileAndConvertToTensor, saveAsGrid } from "../utils.js";

function conv() {
  const model = tf.sequential();
  model.add(
    tf.layers.conv2d({
      inputShape: [224, 224, 3],
      filters: 8,
      kernelSize: 3,
      activation: "relu",
      name: "conv1",
    }),
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2, name: "pool1" }));
  model.add(
    tf.layers.conv2d({
      filters: 16,
      kernelSize: 3,
      activation: "relu",
      name: "conv2",
    }),
  );
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

  const sliced = tf.model({
    inputs: model.inputs,
    outputs: model.getLayer("conv2").output,
  });

  const input = readFileAndConvertToTensor("./test_images/1.jpg");
  const activations = sliced.predict(input) as tf.Tensor4D;
  saveAsGrid(activations);
}

function relu() {
  const model = tf.sequential();
  model.add(
    tf.layers.conv2d({
      inputShape: [224, 224, 3],
      filters: 8,
      kernelSize: 3,
      activation: "linear",
      name: "conv1",
    }),
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2, name: "pool1" }));
  model.add(
    tf.layers.conv2d({
      filters: 16,
      kernelSize: 3,
      activation: "linear",
      name: "conv2",
    }),
  );
  model.add(tf.layers.reLU({ name: "relu" }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

  const sliced = tf.model({
    inputs: model.inputs,
    outputs: model.getLayer("conv2").output,
  });

  const sliced2 = tf.model({
    inputs: model.inputs,
    outputs: model.getLayer("relu").output,
  });

  const input = readFileAndConvertToTensor("./test_images/1.jpg", true);
  const activations = sliced.predict(input) as tf.Tensor4D;

  const activations2 = sliced2.predict(input) as tf.Tensor4D;

  console.log("before shape:", activations.shape); // [1, 56, 56, 16]
  console.log("after shape:", activations2.shape); // [1, 56, 56, 16]
  saveAsGrid(activations, "./tests/grid_before_relu.png");

  saveAsGrid(activations2, "./tests/grid_after_relu.png");

  activations.dispose();
  activations2.dispose();
  input.dispose();
}

relu();
