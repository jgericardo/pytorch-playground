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

// type DataPoint = {
//     weight: number;
// };

export class HistogramPlot {
    private data = [];
    private svg;
    private xScale;
    private yScale;
    private xAxis;
    private yAxis;
    private barColor;
    private formatCount;
    private height;
    private bins;
    private xAxisClass;

    constructor(container, barColor, bins, xAxisLabel, yAxisLabel, yAxisLabelOffset, xAxisClass){
        this.barColor = barColor;
        let node = container.node() as HTMLElement;
        let totalWidth = node.offsetWidth;
        let totalHeight = node.offsetHeight;
        let margin = {top: 10, right: 10, bottom: 50, left:50};
        let width = totalWidth - margin.left - margin.right;
        let height = totalHeight - margin.top - margin.bottom;
        this.height = height;
        this.formatCount = d3.format(",.0f");
        this.bins = bins;
        this.xAxisClass = xAxisClass;

        this.xScale = d3.scale.linear()
            .domain([-bins, bins])
            .range([0, width]);

        this.yScale = d3.scale.linear()
            .domain([0, 100])
            .range([height, 0]);

        // adding the svg to the target container on the page
        this.svg = container.append("svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
                .attr("transform", `translate(${margin.left}, ${margin.top})`);

        // creating the axes and setting their properties
        this.xAxis = d3.svg.axis()
            .orient("bottom")
            .scale(this.xScale)
            .ticks(10);

        this.yAxis = d3.svg.axis()
            .orient("left")
            .scale(this.yScale)
            .ticks(10);

        // drawing the axes using the axis objects
        this.svg.append("g")
            .attr("class", "x "+this.xAxisClass)
            .attr("transform", "translate(0,"+height+")")
            .call(this.xAxis)
            .append("text")
                .attr("y", 20)
                .attr("dy", "0.9em")
                .attr("x", 168)
                .style("text-anchor", "end")
                .text(xAxisLabel);

        this.svg.append("g")
            .attr("class", "y ")
            .attr(this.yAxis)
            .append("text")
                .attr("transform", "rotate(-90)")
                .attr("y", -40)
                .attr("dy", "0.9em")
                .attr("x", yAxisLabelOffset)
                .style("text-anchor", "end")
                .text(yAxisLabel);
    }

    private redraw() {
        let formatCount = this.formatCount;
        let height = this.height;
        let data = d3.layout.histogram()
            .bins(this.xScale.ticks(this.bins))
            (this.data);

        let yMax = d3.max(data, function(d){ return d.length });
        let yMin = d3.min(data, function(d){ return d.length });

        this.yScale.domain([0, yMax]);
        this.yAxis.scale(this.yScale).ticks(10);
        d3.selectAll(".y."+this.xAxisClass)
            .transition()
            .call(this.yAxis);
        let xScale = this.xScale;
        let yScale = this.yScale;

        let colorScale = d3.scale.linear<string, number>()
            .domain([yMin, yMax])
            // @ts-ignore
            .range([d3.rgb(this.barColor).brighter(), d3.rgb(this.barColor).darker()]);

        if (this.svg.selectAll(".bar").empty()) {
            let bar = this.svg.selectAll(".bar")
                .data(data)
                .enter().append('g')
                    .attr("class", "bar")
                    .attr("transform", function(d){
                        return "translate(" + xScale(d.x) + "," + yScale(d.y) + ")"; 
                    });

            bar.append("rect")
                .transition()
                .duration(1000)
                .attr('x', 1)
                .attr("width", (xScale(data[0].dx) - xScale(0)) - 1)
                .attr("height", function(d) { return height - yScale(d.y); })
                .attr("fill", function(d) { return colorScale(d.y); });

            bar.append("text")
                .attr("dy", ".75em")
                .attr("y", -12)
                .attr("x", (xScale(data[0].dx) - xScale(0)) / 2)
                .attr("text-anchor", "middle")
                .transition()
                .duration(1000)
                .text(function(d) { return formatCount(d.y); });


        } else {
            let bar = this.svg.selectAll(".bar").data(data);
            bar.exit().remove();
            bar.transition()
                .duration(1000)
                .attr("transform", function(d){
                    return "translate(" + xScale(d.x) + "," + yScale(d.y) + ")"; 
                });

            bar.select("rect")
                .transition()
                .duration(1000)
                .attr("height", function(d){ return height - yScale(d.y); })
                .attr("fill", function(d){ return colorScale(d.y); });

            bar.select("text")
                .transition()
                .duration(1000)
                .text(function(d) { return formatCount(d.y); } );
        }
    }

    updateData(data: number[]){
        this.data = data;
        this.xScale.domain([d3.min(data), d3.max(data)]);
        this.xAxis.scale(this.xScale).ticks(10);
        d3.selectAll(".x."+this.xAxisClass)
            .transition()
            .call(this.xAxis);
        this.redraw();
    }

    reset(xAxisMax){
        this.xScale.domain([-xAxisMax, xAxisMax]);
        this.xAxis.scale(this.xScale).ticks(10);
        d3.selectAll(".x."+this.xAxisClass)
            .transition()
            .call(this.xAxis);

        if (!this.svg.selectAll(".bar").empty()) {
            // clear using existing data
            let data = d3.layout.histogram()
                .bins(this.xScale.ticks(this.bins))
                (this.data);

            let bar = this.svg.selectAll(".bar").data(data);
            bar.select("text")
                .transition()
                .duration(1000)
                .text("");
            bar.exit().remove();
            // clear existing data
            this.data = [];
            this.redraw();
        }
    }
}