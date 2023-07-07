const MODEL_PATH = `https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4`;
const EXAMPLE_IMG = document.getElementById("exampleImg");
const liveView = document.getElementById("liveView");

let movenet = undefined;
let coco = undefined;

async function loadAndRunModel() {
  movenet = await tf.loadGraphModel(MODEL_PATH, { fromTFHub: true });

  // used to create a tensor in the correct form:
  // let exampleInputTensor = tf.zeros([1, 192, 192, 3], "int32");

  // 1. Create tensor from original image
  let imageTensor = tf.browser.fromPixels(EXAMPLE_IMG);

  // 2. Run image through the coco model to find the position of the person
  const boxSize = await getImageBox(EXAMPLE_IMG);

  // 3. Adjust boxSize so that its a square
  const padX = (boxSize.height - boxSize.width) / 2;
  const newBox = {
    x: Math.round(boxSize.x - padX),
    y: Math.round(boxSize.y),
    width: Math.round(boxSize.width + 2 * padX),
    height: Math.round(boxSize.height),
  };

  // 4. Crop the original image to the "padded" square surrounding the position found by coco model
  // let cropStartPoint = [15, 170, 0];
  // let cropSize = [345, 345, 3];
  // let croppedTensor = tf.slice(imageTensor, cropStartPoint, cropSize);
  let cropStartPoint = [newBox.y, newBox.x, 0];
  let cropSize = [newBox.height, newBox.width, 3];
  let croppedTensor = tf.slice(imageTensor, cropStartPoint, cropSize);

  // 5. Resize cropped tensor to 192 x 192 for use in movenet model
  let resizedTensor = tf.image
    .resizeBilinear(croppedTensor, [192, 192], true)
    .toInt();

  // 6. Run Resized tensor through the movenet model to find person feature positions
  let tensorOutput = movenet.predict(tf.expandDims(resizedTensor));
  let arrayOutput = await tensorOutput.array();
  console.log(arrayOutput);

  drawBox(boxSize);
}

let children = [];
async function getImageBox(img, object = "person") {
  let imageTensor = tf.browser.fromPixels(img);
  const predictions = await coco.detect(imageTensor);

  // Now lets loop through predictions and draw them to the live view if
  // they have a high confidence score.
  let box;
  for (let n = 0; n < predictions.length; n++) {
    // If we are over 66% sure we are sure we classified it right, draw it!
    if (predictions[n].score > 0.22 && predictions[n].class === object) {
      box = predictions[n].bbox;
    }
  }

  if (!box) {
    return null;
  }

  return { x: box[0], y: box[1], width: box[2], height: box[3] };
}

// Before we can use COCO-SSD class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment
// to get everything needed to run.
// Note: cocoSsd is an external object loaded from our index.html
// script tag import so ignore any warning in Glitch.
cocoSsd.load().then(function (loadedModel) {
  coco = loadedModel;
  loadAndRunModel();
});

function drawBox(boxSize) {
  const highlighter = document.createElement("div");
  highlighter.setAttribute("class", "highlighter");
  const paddingVal = (boxSize.height - boxSize.width) / 2;
  highlighter.style =
    "left: " +
    (boxSize.x - paddingVal) +
    "px; top: " +
    boxSize.y +
    "px; width: " +
    boxSize.width +
    "px; height: " +
    boxSize.height +
    "px; padding: " +
    `0 ${paddingVal}px;`;

  liveView.appendChild(highlighter);
  // children.push(highlighter);
}
