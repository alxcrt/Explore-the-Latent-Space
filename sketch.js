const latentSpace = 300;
const numberSliders = 20;
let pcas = [];
let modelPath = 'famousa-decoder-js/model.json';
let model;
let eigenvalues, eigenvectors, meanData, traits;
let showLatent = true;
const TOTAL = 5000;


let nSteps = 30;
let alphaValues = LinSpace(0, 1, nSteps);
let i = 0, ai, bi;

async function LoadModel() {
  console.log('Loading model..');
  model = await tf.loadLayersModel(modelPath);
  console.log('Sucessfully loaded model');



  // // create sliders
  // for (let i = 0; i < numberSliders; ++i) {
  //   pca = createSlider(-300, 300, 0, 10);
  //   pca.input(GenerateFace);
  //   pcas.push(pca);
  // }


  eigenvalues = Json2Tensor(eigenvalues, 1);
  // console.log(eigenvalues);

  eigenvectors = Json2Tensor(eigenvectors, 2);
  // console.log(eigenvectors);

  meanData = Json2Tensor(meanData, 1);

  // LatentInterpolation(0, 1, 10);
  // GenerateFace();
}

function preload() {
  eigenvalues = loadJSON("eigenvaluesv2.json");
  eigenvectors = loadJSON("eigenvectorsv2.json");
  meanData = loadJSON("meanData.json");
  traits = loadJSON("traits.json");
  LoadModel();
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

  ai = floor(random(0, TOTAL));
  bi = floor(random(0, TOTAL));

  // // console.log(meanData);
  // traits_test = Json2Tensor(traits, 2);
  // const indices = tf.tensor1d([1], 'int32');

  // // console.log(traits_test.data().then((data) => console.log(data)));
  // traits_test.gather(indices).print();

  // console.log(model);
  // // 
  // LatentInterpolation(0, 10, 10);
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
        // settings[i] = traits[0][i];
        settings[i] = 0;
      // settings[i] = traits[1][i];
    }
    // console.log(settings);

    settings = tf.tensor2d(settings, shape = [300, 1]);
    real_settings = meanData.clone().reshape([1, 300]);

    let a = tf.mul(eigenvalues, settings);
    let b = tf.mul(eigenvectors, a);
    let c = tf.sum(b, axis = 0);

    real_settings = real_settings.add(c);


    // real_settings = real_settings.add(settings.dot(eigenvectors.dot(tf.eye(300, 300).mul(eigenvalues))))

    let prediction = model.predict(real_settings).dataSync();
    Show2Canvas(prediction);



  })



}


function LatentInterpolation(start, end, nSteps) {

  tf.tidy(() => {
    let a = tf.tensor2d(traits[0], shape = [300, 1]);
    a = GetLatent(a);
    let b = tf.tensor2d(traits[1], shape = [300, 1]);
    b = GetLatent(b);
    let alphaValues = LinSpace(0, 1, 10);
    console.log(alphaValues);

    alphaValues.forEach((alpha) => {
      let alphaTensor = tf.scalar(alpha);
      let vector = a.mul(tf.sub(tf.scalar(1), alphaTensor)).add(b.mul(alphaTensor));
      vector = vector.reshape([1, 300]);
      // vector.print();
      let prediction = model.predict(vector).dataSync();
      // console.log(prediction);
      Show2Canvas(prediction);
    })
  })

}


function LinSpace(startValue, stopValue, cardinality) {
  var arr = [];
  var step = (stopValue - startValue) / (cardinality - 1);
  for (var i = 0; i < cardinality; i++) {
    arr.push(startValue + (step * i));
  }
  return arr;
}

function GetLatent(settings) {
  real_settings = meanData.clone().reshape([1, 300]);

  let a = tf.mul(eigenvalues, settings);
  let b = tf.mul(eigenvectors, a);
  let c = tf.sum(b, axis = 0);

  real_settings = real_settings.add(c);

  return real_settings;
}


function Show2Canvas(prediction) {
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

  // setTimeout(Show2Canvas, 1000);
}



function draw() {
  if (model != undefined && showLatent == true) {

    tf.tidy(() => {
      let a = tf.tensor2d(traits[ai], shape = [300, 1]);
      a = GetLatent(a);
      let b = tf.tensor2d(traits[bi], shape = [300, 1]);
      b = GetLatent(b);

      // console.log(alphaValues);

      let alphaTensor = tf.scalar(alphaValues[i]);
      let vector = a.mul(tf.sub(tf.scalar(1), alphaTensor)).add(b.mul(alphaTensor));
      vector = vector.reshape([1, 300]);
      // vector.print();
      let prediction = model.predict(vector).dataSync();
      // console.log(prediction);
      Show2Canvas(prediction);


      // delayTime(1)
    })


    //   let alphaTensor = tf.scalar(alpha);
    //   // alphaTensor.print();
    //   let vector = a.mul(tf.sub(1, alphaTensor)).add(b.mul(alphaTensor));
    //   // vector.print();

    //   vector = vector.reshape([1, 300]);
    //   let prediction = model.predict(vector).dataSync();

    //   // Show2Canvas(prediction);
    //   // vector = latentStart*(1-alpha) + latentEnd*alpha
    //   // vectors.append(vector)
    // });

  }
  // frameRate(1);
  // console.log(tf.memory().numTensors);
  i++;
  if (i == nSteps) {
    i = 0;
    ai = bi;
    bi = floor(random(0, TOTAL));
  }
  frameRate(30);
}
