import tf from "@tensorflow/tfjs-node";
import * as fs from "fs";

export async function saveAsGrid(
  activations: tf.Tensor4D,
  outputPath: string = "./tests/grid.png",
): Promise<void> {
  const squeezed = activations.squeeze([0]) as tf.Tensor3D;
  const [H, W, C] = squeezed.shape as [number, number, number];
  const cols = Math.ceil(Math.sqrt(C));
  const rows = Math.ceil(C / cols);

  const normalizedChannels: tf.Tensor3D[] = [];

  for (let c = 0; c < C; c++) {
    const ch = squeezed.slice([0, 0, c], [H, W, 1]) as tf.Tensor3D;
    const min = ch.min();
    const max = ch.max();
    const norm = ch
      .sub(min)
      .div(max.sub(min).add(1e-8))
      .mul(255)
      .round()
      .toInt() as tf.Tensor3D;
    normalizedChannels.push(norm);
    ch.dispose();
  }

  // Добиваем пустыми каналами до rows*cols
  const empty = tf.zeros([H, W, 1], "int32") as tf.Tensor3D;
  while (normalizedChannels.length < rows * cols) {
    normalizedChannels.push(empty.clone() as tf.Tensor3D);
  }

  // Склеиваем по строкам, потом по столбцам
  const rowTensors: tf.Tensor3D[] = [];
  for (let r = 0; r < rows; r++) {
    const rowChannels = normalizedChannels.slice(r * cols, (r + 1) * cols);
    rowTensors.push(tf.concat(rowChannels, 1) as tf.Tensor3D); // по ширине
  }
  const grid = tf.concat(rowTensors, 0) as tf.Tensor3D; // [H*rows, W*cols, 1]

  const pngBuffer = await tf.node.encodePng(grid);
  fs.writeFileSync(outputPath, pngBuffer);
  console.log(`Сетка сохранена: ${outputPath}`);

  // Чистим
  normalizedChannels.forEach((t) => t.dispose());
  rowTensors.forEach((t) => t.dispose());
  empty.dispose();
  grid.dispose();
  squeezed.dispose();
}

export async function saveActivationsAsPng(
  activations: tf.Tensor4D,
  outputDir = "./activations",
) {
  if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

  const squeezed = activations.squeeze([0]);
  const [height, width, channels] = squeezed.shape as [number, number, number];
  console.log(`Размер: ${height}x${width}, каналов: ${channels}`);

  for (let c = 0; c < channels; c++) {
    // Берём один канал: [H, W, 1]
    const channel = squeezed.slice([0, 0, c], [height, width, 1]);

    // Нормализация в 0..255
    const minVal = channel.min();
    const maxVal = channel.max();
    const range = maxVal.sub(minVal).add(1e-8);

    const normalized = channel.sub(minVal).div(range).mul(255);

    // encodePng требует int32
    const intTensor = normalized.round().toInt();

    // [H, W, 1] -> [H, W] -> [H, W, 1] (просто чтобы форма была строго 3D)
    const pngInput = intTensor.squeeze([2]).expandDims(2) as tf.Tensor3D;

    // Кодируем и сохраняем
    const pngBuffer = await tf.node.encodePng(pngInput);
    fs.writeFileSync(`${outputDir}/channel_${c}.png`, pngBuffer);

    // Чистим память
    channel.dispose();
    normalized.dispose();
    intTensor.dispose();
    pngInput.dispose();
  }

  squeezed.dispose();
  console.log(`Сохранено ${channels} файлов в ${outputDir}`);
}

export function readFileAndConvertToTensor(path: string, logs = false) {
  const file = fs.readFileSync(path);

  let tensor = tf.node.decodeImage(file) as tf.Tensor3D;

  tensor = tf.image.resizeBilinear(tensor, [224, 224]) as tf.Tensor3D;

  if (logs) {
    console.log(tensor.print());
  }

  const normalized = tensor.toFloat().div(255.0);

  const finalNormilized = normalized.expandDims(0) as tf.Tensor4D;

  if (logs) {
    console.log(finalNormilized.print());
  }

  return finalNormilized;
}
