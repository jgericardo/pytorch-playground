/* Copyright 2016 Google Inc. All Rights Reserved.
Copyright 2021 Justin Gerard E. Ricardo

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import * as d3 from 'd3';

type DataPoint = {
  x: number;
  y: number[];
};

/**
 * A multi-series line chart that allows you to append new data points
 * as data becomes available. 
 * TODO: Update to reduce to one class for line charts.
 */
export class AppendingLineChartFixedAxis {
  private numLines: number;
  private data: DataPoint[] = [];
  private svg;
  private xScale;
  private yScale;
  private lineColors: string[];
  private xAxis;
  private yAxis;
  private paths;

  constructor(container, lineColors: string[], epochs, xAxisLabel, yAxisLabel, yAxisLabelOffset) {
    this.lineColors = lineColors;
    this.numLines = lineColors.length;
    let node = container.node() as HTMLElement;
    let totalWidth = node.offsetWidth;
    let totalHeight = node.offsetHeight;
    let margin = {top: 10, right: 10, bottom: 50, left: 50};
    let width = totalWidth - margin.left - margin.right;
    let height = totalHeight - margin.top - margin.bottom;

    this.xScale = d3.scale.linear()
      .domain([0, epochs+1])
      .range([0, width]);

    this.yScale = d3.scale.linear()
      .domain([0, 100])
      .range([height, 0]);
      

    this.svg = container.append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
      .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    this.yAxis = d3.svg.axis()
        .orient("left")
        .scale(this.yScale)
        .ticks(10);

    this.xAxis = d3.svg.axis()
        .orient("bottom")
        .scale(this.xScale)
        .ticks(epochs+1);


    this.svg.append("g")
        .attr("class", "x chartAxis")
        .attr("transform", "translate(0,"+height+")")
        .call(this.xAxis)
        .append("text")
        .attr("y", 20)
        .attr("dy", "0.9em")
        .attr("x", 125)
        .style("text-anchor", "end")
        .text(xAxisLabel);

    this.svg.append("g")
        .attr("class", "y chartAxis")
        .call(this.yAxis)
        .append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", -40)
        .attr("dy", "0.9em")
        .attr("x", yAxisLabelOffset)
        .style("text-anchor", "end")
        .text(yAxisLabel);


    this.paths = new Array(this.numLines);
    for (let i = 0; i < this.numLines; i++) {
        this.paths[i] = this.svg.append("path")
        .attr("class", "line")
        .style({
            "fill": "none",
            "stroke": lineColors[i],
            "stroke-width": "1.5px"
        });
    }

    // this.reset();
  }

  reset(epochs) {
    this.xScale.domain([0, epochs+1]);
    this.xAxis.scale(this.xScale).ticks(epochs+1);
    d3.selectAll('.x.chartAxis')
        .transition()
        .call(this.xAxis);

    this.data = [];
    this.redraw();
  }

  addDataPoint(epoch: number, dataPoint: number[]) {
    if (dataPoint.length !== this.numLines) {
        throw Error("Length of dataPoint must equal number of lines");
    }
  
    this.data.push({x: epoch, y: dataPoint});
    this.redraw();
  }

  private redraw() {
    let getPathMap = (lineIndex: number) => {
        return d3.svg.line<{x: number, y:number}>()
        .x(d => this.xScale(d.x))
        .y(d => this.yScale(d.y[lineIndex]));
      };
      for (let i = 0; i < this.numLines; i++) {
        this.paths[i].datum(this.data).attr("d", getPathMap(i));
      }
  }
}
