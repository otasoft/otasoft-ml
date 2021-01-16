import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import * as tfvis from '@tensorflow/tfjs-vis';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.sass']
})
export class AppComponent implements OnInit{
  title = 'client';

  constructor(private http: HttpClient) {}

  async ngOnInit() {
    const response: any = await this.http.get('/api/tensorflow-test').toPromise()

    tfvis.render.scatterplot(
      { name: 'Tensorflow test' },
      { values: [response.originalPoints, response.predictedPoints], series: ['original', 'predicted'] },
      {
        xLabel: 'Horsepower',
        yLabel: 'MPG',
        height: 300
      }
    )
  }
}
