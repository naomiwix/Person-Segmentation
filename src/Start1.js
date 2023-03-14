import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import myImage from './img_1.png';
import myImageMask from './img_1_mask.png';

function YourComponent() {
  const [imageTensor, setImageTensor] = useState(null);
  const [imageMaskTensor, setImageMaskTensor] = useState(null);

  const predict = async () => {
    try {
    
    const imageTensorReshaped = tf.reshape(imageTensor, [1, 600, 800, 3]);
    const result = await model.predict(imageTensorReshaped);
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

    loadImage();
    loadMask();
  }, []);

  useEffect(() => {
    if (imageTensor && imageMaskTensor) {
      predict();
    }
  }, [imageTensor, imageMaskTensor]);


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


  if (!imageTensor || !imageMaskTensor) {
    return <div>Loading...</div>;
  }

  return <div>Image and Mask Tensors Loaded.</div>;
}

export default YourComponent;
