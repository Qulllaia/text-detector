import * as tf from "@tensorflow/tfjs-node";
import * as fs from "fs";
import * as path from "path";

// ---------- 1. Создаём простое изображение 10x10 ----------
function makeTestImage(): tf.Tensor2D {
  // 10x10, вертикальная светлая полоса в центре (столбцы 4 и 5)
  const data: number[] = [];
  for (let y = 0; y < 10; y++) {
    for (let x = 0; x < 10; x++) {
      data.push(x >= 4 && x <= 5 ? 200 : 50);
    }
  }
  return tf.tensor2d(data, [10, 10]);
}

// ---------- 2. Ядро 3x3 ----------
// Вертикальный край: светлое справа от тёмного
const KERNEL = [
  [-1, 0, 1],
  [-1, 0, 1],
  [-1, 0, 1],
];

// ---------- 3. Пошаговый вывод в консоль ----------
function explainConvolution(image: tf.Tensor2D, kernel: number[][]) {
  const img = image.arraySync() as number[][];
  const kH = kernel.length;
  const kW = kernel[0].length;
  const outH = img.length - kH + 1;
  const outW = img[0].length - kW + 1;

  console.log("=== Входное изображение 10x10 ===");
  img.forEach((row) =>
    console.log(row.map((v) => v.toString().padStart(4)).join("")),
  );
  console.log("\n=== Ядро 3x3 ===");
  kernel.forEach((row) =>
    console.log(row.map((v) => v.toString().padStart(4)).join("")),
  );

  console.log("\n=== Позиция (0,0): верхний левый угол ===");
  const window00 = [
    [img[0][0], img[0][1], img[0][2]],
    [img[1][0], img[1][1], img[1][2]],
    [img[2][0], img[2][1], img[2][2]],
  ];
  printWindowWithKernel(window00, kernel, 0, 0);

  console.log("\n=== Позиция (2,2): центр, где проходит край ===");
  const windowCenter = [
    [img[2][2], img[2][3], img[2][4]],
    [img[3][2], img[3][3], img[3][4]],
    [img[4][2], img[4][3], img[4][4]],
  ];
  printWindowWithKernel(windowCenter, kernel, 2, 2);

  console.log("\n=== Вся карта откликов (без ReLU) ===");
  const responses: number[][] = [];
  for (let y = 0; y < outH; y++) {
    const row: number[] = [];
    for (let x = 0; x < outW; x++) {
      let sum = 0;
      for (let i = 0; i < kH; i++) {
        for (let j = 0; j < kW; j++) {
          sum += img[y + i][x + j] * kernel[i][j];
        }
      }
      row.push(sum);
    }
    responses.push(row);
  }
  responses.forEach((row) =>
    console.log(row.map((v) => v.toString().padStart(6)).join("")),
  );

  return responses;
}

function printWindowWithKernel(
  window: number[][],
  kernel: number[][],
  posY: number,
  posX: number,
) {
  console.log(`Окно в позиции (y=${posY}, x=${posX}):`);
  for (let i = 0; i < 3; i++) {
    let line = "  ";
    for (let j = 0; j < 3; j++) {
      line += `${window[i][j].toString().padStart(4)} × ${kernel[i][j]
        .toString()
        .padStart(3)}  `;
    }
    console.log(line);
  }

  let sum = 0;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      sum += window[i][j] * kernel[i][j];
    }
  }
  console.log(`  Сумма: ${sum}\n`);
}

// ---------- 4. Применяем tf.conv2d для сверки ----------
function tfConvolution(image: tf.Tensor2D, kernel: number[][]): tf.Tensor2D {
  const input = image.reshape([1, 10, 10, 1]) as tf.Tensor4D;
  const k = tf
    .tensor2d(kernel.flat(), [3, 3])
    .reshape([3, 3, 1, 1]) as tf.Tensor4D;

  const output = tf.conv2d(input, k, 1, "valid") as tf.Tensor4D;
  const squeezed = output.squeeze([0, 3]) as tf.Tensor2D;

  input.dispose();
  k.dispose();
  output.dispose();
  return squeezed;
}

// ---------- 5. Сохраняем тензор как PNG ----------
async function saveTensorAsPng(
  tensor: tf.Tensor2D,
  filePath: string,
  normalize = true,
): Promise<void> {
  const dir = path.dirname(filePath);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

  const t = normalize
    ? tf.tidy(() => {
        const min = tensor.min();
        const max = tensor.max();
        const range = max.sub(min).add(1e-8);
        return tensor.sub(min).div(range).mul(255).round().toInt();
      })
    : tensor.round().toInt();

  // encodePng требует [H, W, C]
  const rgb = t.expandDims(2) as tf.Tensor3D;
  const png = await tf.node.encodePng(rgb);
  fs.writeFileSync(filePath, png);
  t.dispose();
  rgb.dispose();
  console.log(`Сохранено: ${filePath}`);
}

// ---------- 6. Main ----------
async function main() {
  const image = makeTestImage();

  // Пошаговый вывод в консоль
  const responses = explainConvolution(image, KERNEL);

  // Сверка с tf.conv2d
  const tfResponses = tfConvolution(image, KERNEL);
  const tfArray = tfResponses.arraySync() as number[][];
  console.log("\n=== Сверка с tf.conv2d ===");
  let matches = true;
  for (let y = 0; y < responses.length; y++) {
    for (let x = 0; x < responses[0].length; x++) {
      if (Math.abs(responses[y][x] - tfArray[y][x]) > 1e-4) {
        matches = false;
        console.log(
          `Расхождение в (${y},${x}): ручной=${responses[y][x]}, tf=${tfArray[y][x]}`,
        );
      }
    }
  }
  if (matches) console.log("Ручной расчёт совпал с tf.conv2d ✓");

  // ReLU
  const reluResponses = tfResponses.relu() as tf.Tensor2D;

  // Сохраняем картинки
  const outDir = "./conv_demo";
  await saveTensorAsPng(image, path.join(outDir, "input.png"));
  await saveTensorAsPng(tfResponses, path.join(outDir, "before_relu.png"));
  await saveTensorAsPng(reluResponses, path.join(outDir, "after_relu.png"));

  image.dispose();
  tfResponses.dispose();
  reluResponses.dispose();
}

main().catch(console.error);
