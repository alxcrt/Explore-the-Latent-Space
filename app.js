const latentSpace = 300;
const numberSliders = 50;
let pcas = [];
let modelPath = 'famousa-decoder-js/model.json';
let model;
let eigenvalues, eigenvectors, meanData;
let canvas;

async function LoadModel() {
  model = await tf.loadLayersModel(modelPath);

}

function preload() {
  // eigenvalues = loadJSON("eigenvalues.json");
  eigenvectors = loadJSON("eigenvectors.json");
  meanData = loadJSON("meanData.json");
}

function Json2Tensor(x, dim) {
  if (dim == 1) {
    arr = [];
    let n = Object.keys(x).length;
    for (let i = 0; i < n; ++i) {
      arr[i] = x[i];
    }
    return tf.tensor2d(arr, shape = [n, 1]);
  } else if (dim == 2) {
    let n = Object.keys(x).length;
    let m = Object.keys(x[0]).length;
    arr = new Array(n);
    for (let i = 0; i < n; ++i) {
      arr[i] = new Array(m);
      for (let j = 0; j < m; ++j) {
        arr[i][j] = x[i][j];
      }
    }
    return tf.tensor2d(arr, shape = [n, m]);
  }
}


function setup() {
  createCanvas(256, 256);
  background(0);
  pixelDensity(1);

  LoadModel();


  // create sliders
  for (let i = 0; i < numberSliders; ++i) {
    pca = createSlider(-250, 250, 0, 10);
    pca.input(GenerateFace);
    pcas.push(pca);
  }

  // eigenvalues = Json2Tensor(eigenvalues, 1);
  // console.log(eigenvalues);

  eigenvectors = Json2Tensor(eigenvectors, 2);
  // console.log(eigenvectors);

  meanData = Json2Tensor(meanData, 1);
  // console.log(meanData);


  // GenerateFace();
}

async function GenerateFace() {

  tf.tidy(() => {
    // convert 10, 50 into a vector
    settings = [];
    // let arr = [[pcas[0].value(), pcas[1].value()]];
    for (let i = 0; i < latentSpace; ++i) {
      if (i < numberSliders)
        settings[i] = pcas[i].value() / 100.;
      else
        settings[i] = 0;
    }
    // console.log(settings);

    settings = tf.tensor2d(settings, shape = [300, 1]);

    real_settings = meanData.clone().reshape([1, 300]);

    // let a = tf.mul(eigenvalues, settings);
    let b = tf.mul(eigenvectors, settings);
    let c = tf.sum(b, axis = 0);
    real_settings = real_settings.add(c);

    // real_settings = real_settings.add(settings.dot(eigenvectors.dot(tf.eye(300, 300).mul(eigenvalues))))

    let prediction = model.predict(real_settings).dataSync();
    prediction = tf.tensor3d(prediction, [128, 128, 3]);
    prediction = tf.image.resizeNearestNeighbor(prediction, [256, 256])
    prediction = prediction.as1D().data().then((image) => {
      loadPixels();
      for (let i = 0; i < width; ++i) {
        for (let j = 0; j < height; ++j) {
          let pos = (i + width * j) * 4;
          let index = (i + width * j) * 3;
          // let col = color(prediction[pos], prediction[pos], prediction[pos]);
          // pixels[pos] = red(col);
          // pixels[pos + 1] = green(col);
          // pixels[pos + 2] = blue(col);
          // pixels[pos + 3] = alpha(col);
          pixels[pos] = floor(image[index] * 255);
          pixels[pos + 1] = floor(image[index + 1] * 255);
          pixels[pos + 2] = floor(image[index + 2] * 255);
          pixels[pos + 3] = 255;

        }
      }
      updatePixels()
    });
    // real_settings
    // tf.brow(prediction).print();


    // console.log(prediction)
    // prediction = prediction.as1D();

    // img = loadImage(prediction);
    // let ri_tf = tf.tensor3d(prediction, shape = [64, 64, 3]).mul(255);
    // let t = tf.image.resizeNearestNeighbor(ri_tf, [256, 256])
    // prediction = t.as1D();



  })



}

// function draw() {
//   console.log(tf.memory().numTensors);
// }
