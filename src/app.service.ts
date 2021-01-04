import { HttpService, Injectable } from '@nestjs/common';
import * as tf from '@tensorflow/tfjs-node';

@Injectable()
export class AppService {
  constructor(private readonly httpService: HttpService) {}

  getHello(): string {
    return 'Hello World!';
  }

  async testTensorflow() {
    return await this.getCarsData();
  }

  async getCarsData() {
      const carsDataResponse = await this.httpService.get('https://storage.googleapis.com/tfjs-tutorials/carsData.json').toPromise();
      const carsData = await carsDataResponse.data;

      const cleaned = carsData.map(car => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower,
      }))
      .filter(car => (car.mpg != null && car.horsepower != null));

      return cleaned;
  }

  createTensorflowModel() {
    const model = tf.sequential();

    // add input layer
    model.add(tf.layers.dense({ inputShape: [1], units: 1, useBias: true }));

    // add output layer
    model.add(tf.layers.dense({ units: 1, useBias: true }));

    return model;
  }

  convertDataToTensor(data) {
    return tf.tidy(() => {
      // 1. Shuffle the data
      tf.util.shuffle(data);

      // 2. Convert data to Tensor
      const inputs = data.map(d => d.horsepower);
      const labels = data.map(d => d.mpg);

      const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
      const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

      // 3. Normalize the data to the range 0-1
      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();

      const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor.sub(labelMin).sub(labelMax.sub(labelMin));

      // 4. Return all Tensor data
      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        inputMax,
        inputMin,
        labelMax,
        labelMin,
      }
    });
  }

  async trainModel(model: tf.Sequential, inputs, labels) {
    // Prepare model for training
    model.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.meanSquaredError,
      metrics: ['mse'],
    });

    const batchSize = 32;
    const epochs = 50;

    return await model.fit(inputs, labels, {
      batchSize,
      epochs,
      shuffle: true,
    })
  }
}
