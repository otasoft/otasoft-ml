import { HttpService, Injectable } from '@nestjs/common';

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
}
