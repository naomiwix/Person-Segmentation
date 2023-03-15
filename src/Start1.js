import React, { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import myImage from './img_1.png';
import myImageMask from './img_1_mask.png';
const model = tf.sequential({
    layers: [
      tf.layers.dense({ units: 10, activation: 'relu', inputShape: [600, 800, 3] }),
      tf.layers.dense({ units: 1, activation: 'sigmoid' })
    ]
  });

  model.compile({
    loss: tf.metrics.binaryCrossentropy,
    optimizer: tf.train.adam(),
    metrics: [tf.metrics.binaryCrossentropy]
  });

function YourComponent() {
  const [imageTensor, setImageTensor] = useState(null);
  const [imageMaskTensor, setImageMaskTensor] = useState(null);
  const [predictedMaskTensor, setPredictedMaskTensor] = useState(null);
  const canvasRef = useRef(null);

  const predict = async () => {
    try {
      const imageTensorReshaped = tf.reshape(imageTensor, [1, 600, 800, 3]);
      const result = await model.predict(imageTensorReshaped).data();
      const predictedMaskTensor = tf.cast(tf.greater(result, 0.5), 'float32');
      setPredictedMaskTensor(predictedMaskTensor);
    } catch (error) {
      console.error(error);
    }
  };

  useEffect(() => {
    const loadImage = async () => {
      try {
        const img = new Image();
        img.src = myImage;

        await new Promise(resolve => {
          img.onload = () => {
            resolve();
          };
        });

        const tensor = tf.browser.fromPixels(img, 3);
        setImageTensor(tensor);
      } catch (error) {
        console.error(error);
      }
    };

    const loadMask = async () => {
      try {
        const imgMask = new Image();
        imgMask.src = myImageMask;

        await new Promise(resolve => {
          imgMask.onload = () => {
            resolve();
          };
        });

        const tensor = tf.browser.fromPixels(imgMask, 3);
        setImageMaskTensor(tensor);
      } catch (error) {
        console.error(error);
      }
    };
    loadImage(); loadMask();
  }, []);

  useEffect(() => {
    if (imageTensor && imageMaskTensor) {
      predict();
    }
  }, [imageTensor, imageMaskTensor]);

  useEffect(() => {
    if (predictedMaskTensor && imageTensor&&canvasRef.current) {
      const predictedMaskArray = predictedMaskTensor.arraySync();
      const imageArray = imageTensor.arraySync();
      console.log('Image array shape:', imageArray);
      console.log('Predicted mask array shape:', predictedMaskArray);


      // Apply the predicted mask to the image array
      const maskedArray = imageArray.map((row, i) =>
        row.map((pixel, j) =>
          pixel.map((channel, k) => channel * predictedMaskArray[i][j])
        )
      );
      // Create a new image tensor from the masked array
      const maskedImageTensor = tf.tensor(maskedArray);
      // Draw the new image tensor on the canvas element


      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      tf.browser.toPixels(maskedImageTensor, canvas).then(() => {
        ctx.drawImage(canvas, 0, 0);
      });
    }
  }, [predictedMaskTensor, imageTensor]);





  return         <canvas
  ref={canvasRef}
  style={{
    position: "absolute",
    marginLeft: "auto",
    marginRight: "auto",
    left: 0,
    right: 0,
    textAlign: "center",
    zindex: 9,
    width: 600,
    height: 800,
  }}
/>;

}

export default YourComponent;
