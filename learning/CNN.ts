import tf from "@tensorflow/tfjs-node";
import {
  readFileAndConvertToTensor,
  saveActivationsAsPng,
  saveAsGrid,
} from "../utils.js";

function getLayerOutput(
  model: tf.Sequential,
  layerName: string,
  inputTensor: tf.Tensor4D,
) {
  // Создаём новую модель: входы те же, что у исходной, выход — у указанного слоя
  const slicedModel = tf.model({
    inputs: model.inputs,
    outputs: model.getLayer(layerName).output,
  });

  // Запускаем предсказание и получаем тензор активаций
  const activation = slicedModel.predict(inputTensor);

  return activation;
}

const BASE_H = 256;
const BASE_W = 256;

function conv() {
  const model = tf.sequential();
  model.add(
    tf.layers.conv2d({
      inputShape: [BASE_H, BASE_W, 3],
      filters: 8,
      kernelSize: 3,
      activation: "relu",
      name: "conv1",
    }),
  ); // выход - 254 х 254, потому что без padding
  model.add(tf.layers.maxPooling2d({ poolSize: 2, name: "pool1" }));
  model.add(
    tf.layers.conv2d({
      //127
      filters: 16,
      kernelSize: 3,
      activation: "relu",
      name: "conv2",
    }),
  ); // выход - 125 х 125, потому что без padding
  model.add(tf.layers.maxPooling2d({ poolSize: 2, name: "pool2" }));
  model.add(
    tf.layers.conv2d({
      //62
      filters: 32,
      kernelSize: 3,
      activation: "relu",
      name: "conv3",
    }),
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2, name: "pool3" }));

  model.add(
    tf.layers.conv2d({
      //30
      filters: 64,
      kernelSize: 3,
      activation: "relu",
      name: "conv4",
    }),
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2, name: "pool4" }));

  model.add(
    tf.layers.conv2d({
      // 14
      filters: 128,
      kernelSize: 3,
      activation: "relu",
      name: "conv5",
    }),
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 2, name: "pool5" }));

  model.add(
    tf.layers.conv2d({
      filters: 256,
      kernelSize: 3,
      activation: "relu",
      name: "conv6",
    }),
  );
  model.add(tf.layers.maxPooling2d({ poolSize: 4, name: "pool6" })); // 32 32

  const input = readFileAndConvertToTensor("./test_images/1.jpg");
  const resized = tf.image.resizeBilinear(input, [BASE_H, BASE_W]);

  const pool1 = getLayerOutput(model, "pool1", resized) as tf.Tensor4D;
  const pool2 = getLayerOutput(model, "pool2", resized) as tf.Tensor4D;
  const pool3 = getLayerOutput(model, "pool3", resized) as tf.Tensor4D;
  const pool4 = getLayerOutput(model, "pool4", resized) as tf.Tensor4D;
  const pool5 = getLayerOutput(model, "pool5", resized) as tf.Tensor4D;
  const pool6 = getLayerOutput(model, "pool6", resized) as tf.Tensor4D;
  saveAsGrid(pool1, `./tests/grid${1}.png`);
  console.log(pool1.shape);
  saveAsGrid(pool2, `./tests/grid${2}.png`);
  console.log(pool2.shape);
  saveAsGrid(pool3, `./tests/grid${3}.png`);
  console.log(pool3.shape);
  saveAsGrid(pool4, `./tests/grid${4}.png`);
  console.log(pool4.shape);
  saveAsGrid(pool5, `./tests/grid${5}.png`);
  console.log(pool5.shape);
  saveAsGrid(pool6, `./tests/grid${6}.png`);
  console.log(pool6.shape);
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

conv();
