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

import * as nn from "./nn";
import {State, userDatasets} from "./state";
import {AppendingLineChart} from "./linechart";
import {AppendingLineChartFixedAxis} from "./linechart-fixed";
import {HistogramPlot} from "./histogram";
import * as d3 from 'd3';

let mainWidth;
const RECT_SIZE = 30;
const BIAS_SIZE = 5;

enum HoverType {
  BIAS, WEIGHT
}

interface InputFeature {
  f: (x: number, y: number) => number;
  label?: string;
}

let INPUTS: {[name: string]: InputFeature} = {
  "x": {f: (x, y) => x, label: "X_1"},
  "y": {f: (x, y) => y, label: "X_2"}
};

class Player {
  private timerIndex = 0;
  private isPlaying = false;
  private callback: (isPlaying: boolean) => void = null;

  /** Plays/pauses the player. */
  playOrPause() {
    if (this.isPlaying) {
      this.isPlaying = false;
      this.pause();
    } else {
      this.isPlaying = true;
      if (stepNumber === 0) {
        // simulationStarted();
      }
      this.play();
    }
  }

  onPlayPause(callback: (isPlaying: boolean) => void) {
    this.callback = callback;
  }

  play() {
    this.pause();
    this.isPlaying = true;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
    this.start(this.timerIndex);
  }

  pause() {
    this.timerIndex++;
    this.isPlaying = false;
    if (this.callback) {
      this.callback(this.isPlaying);
    }
  }

  private start(localTimerIndex: number) {
    d3.timer(() => {
      if (localTimerIndex < this.timerIndex) {
        return true;  // Done.
      }
      
      return false;  // Not done.
    }, 0);
  }
}

// variable declarations
let state = State.deserializeState();

// Filter out inputs that are hidden.
state.getHiddenProps().forEach(prop => {
  if (prop in INPUTS) {
    delete INPUTS[prop];
  }
});

let linkWidthScale = d3.scale.linear()
  .domain([0, 3])
  .range([1, 10])
  .clamp(true);
let colorScale = d3.scale.linear<string, number>()
  .domain([-1, 0, 1])
  .range(["#e06c78", "#e8eaeb", "#5874dc"])
  .clamp(true);

let stepNumber = 0;
let maxStepsPerEpoch = 0;
let maxTrainingSteps = 0;
let epochNumber = 0;
let network: nn.Node[][] = null;
let lossTrain = 0;
let lossTest = 0;
let accuracyTrain = 0;
let accuracyTest = 0;

let player = new Player();
let lineChart = new AppendingLineChart(d3.select("#linechart"),
    ["#777", "black"]);

let epochVsAccuracyPlot = new AppendingLineChartFixedAxis(d3.select("#epochVsAccuracyPlot"),
    ["#777", "black"], state.numberOfEpochs, "Epochs", "Accuracy (%)", -40);

let inputToHiddenLayerDistPlot = new HistogramPlot(d3.select("#inputToHiddenLayerDistPlot"),
    "steelblue", 10, "Input to Hidden Weights", "Frequency", -50, "i2hAxis");

let hiddenToOutputLayerDistPlot = new HistogramPlot(d3.select("#hiddenToOutputLayerDistPlot"),
    "steelblue", 10, "Hidden to Output Weights", "Frequency", -50, "h2oAxis");

function updateWeights(weightData, layerIdx){
  let weights = weightData;
  let currentLayer = network[layerIdx];
  for (let i = 0; i < currentLayer.length; i++) {
    let node = currentLayer[i];

    // Update the weights coming into this node.
    for (let j = 0; j < node.inputLinks.length; j++) {
      let link = node.inputLinks[j];
      link.weight = weights[i][j];
    }
  }
}

function updateAccuVsEpochPlot(accuracyTrain, accuracyTest, epoch) {
  epochVsAccuracyPlot.addDataPoint(epoch+1,
  [accuracyTrain, accuracyTest]);
}

function updateLossVsEpochPlot(lossTrain, lossTest){
  lineChart.addDataPoint([lossTrain, lossTest]);
}

function makeGUI() {
  // Event handlers
  d3.select("#reset-button").on("click", () => {
    reset();
    stopWebsocket();
  });

  d3.select("#play-pause-button").on("click", function () {
    reset();
    runWebsocket();
  });

  d3.select("#stop-button").on("click", function() {
    stopWebsocket();
  });

  let userDataset = d3.select("#userDataset").on("change", function(){
    state.userDataset = this.value;
    state.serialize();
    reset();
  });
  userDataset.property("value", state.userDataset);

  let learningRate = d3.select("#learningRate").on("change", function() {
    state.learningRate = +this.value;
    state.serialize();
    reset();
  });
  learningRate.property("value", state.learningRate);

  let pruningThreshold = d3.select("#pruningThreshold").on("change", function() {
    state.pruningThreshold = +this.value;
    state.serialize();
    reset();
  });
  pruningThreshold.property("value", state.pruningThreshold);

  let pruningStrategy = d3.select("#pruningStrategy").on("change", function() {
    state.pruningStrategy = this.value;
    state.serialize();
    reset();
  });
  pruningStrategy.property("value", state.pruningStrategy);

  let hiddenNodes = d3.select("#hiddenNodes").on("change", function(){
    state.hiddenNodes = +this.value;
    state.serialize();
    state.networkShape[0] = state.hiddenNodes;
    reset();
  });
  hiddenNodes.property("value", state.hiddenNodes);

  let epochNumberCtrl = d3.select("#epochNumberControl").on("change", function(){
    state.numberOfEpochs = +this.value;
    state.serialize();
    reset();
  });
  epochNumberCtrl.property("value", state.numberOfEpochs)

  state.problem = "classification";
  reset();

  // Add scale to the gradient color map.
  let x = d3.scale.linear().domain([-1, 1]).range([0, 144]);
  let xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom")
    .tickValues([-1, 0, 1])
    .tickFormat(d3.format("d"));
  d3.select("#colormap g.core").append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0,10)")
    .call(xAxis);

  // Listen for css-responsive changes and redraw the svg network.
  window.addEventListener("resize", () => {
    let newWidth = document.querySelector("#main-part")
        .getBoundingClientRect().width;
    if (newWidth !== mainWidth) {
      mainWidth = newWidth;
      drawNetwork(network);
      updateUI(true);
    }
  });

}

function updateBiasesUI(network: nn.Node[][]) {
  nn.forEachNode(network, true, node => {
    d3.select(`rect#bias-${node.id}`).style("fill", colorScale(node.bias));
  });
}

function updateWeightsUI(network: nn.Node[][], container, reset) {

  let count_arr: number[] = [];
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    // Update all the nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      let count = 0;
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        let nodeClass = container.select(`#node${node.id}`).attr("class");
        if (!nodeClass.includes("inactive")){
          container.select(`#link${link.source.id}-${link.dest.id}`)
              .style({
                "stroke-width": linkWidthScale(Math.abs(link.weight)),
                "stroke": colorScale(link.weight)
              });

        }else{
          container.select(`#link${link.source.id}-${link.dest.id}`)
              .style({
                "stroke-width": linkWidthScale(Math.abs(0)),
                "stroke": colorScale(0)
              });
          link.weight = 0;
          count += layerIdx==1 ? 1 : 0;
        }
      }
      if(layerIdx==1){count_arr.push(count);}
    }
  }

  let sum = 0;
  for(let i = 0; i < count_arr.length; i++){
    sum += 1;
  }

  if (!reset){
    container.selectAll(".link").each(animate);
  }
}


function animate(){
  d3.select(this)
  .transition()
  .delay( function(d, i) { return (100*i); } )
  .ease('linear')
  .duration(3000)
  .styleTween("stroke-dashoffset", function(d, i){
    return d3.interpolate(0, -56);
  })
  .each("end", function(){}); // TODO
}


function updateNodesUI(network: nn.Node[][], container, reset) {
  let outputLayer = network[2];
  // Update all the nodes in the previous layer.
  let setOfWeights = [];
  for (let i = 0; i < outputLayer.length; i++) {
    let node = outputLayer[i];

    let weightsPerOutputNode = [];
    for (let j = 0; j < node.inputLinks.length; j++) {
      weightsPerOutputNode.push(node.inputLinks[j].weight);
    }

    setOfWeights.push(weightsPerOutputNode);
  }

  let weightsPerNode = setOfWeights[0].length;
  let sumArray = [];
  for (let i = 0; i < weightsPerNode; i++){
    let sumOfWeights = 0;
    for (let j = 0; j < setOfWeights.length; j++){
      sumOfWeights += setOfWeights[j][i];
    }
    sumArray.push(sumOfWeights);
  }

  let hiddenLayer = network[1];
  for(let i = 0; i < hiddenLayer.length; i++){
    if(sumArray[i]==0){
      let node = hiddenLayer[i];
      container.select(`#node${node.id}`).classed("inactive", true);
    }
  }
}


function drawNode(cx: number, cy: number, nodeId: string, isInput: boolean,
    container, isOutput: boolean, yOffset: number, node?: nn.Node) {
  let x = cx - RECT_SIZE / 2;
  let y = cy - RECT_SIZE / 2;

  let nodeGroup = container.append("g")
    .attr({
      "class": "node",
      "id": `node${nodeId}`,
      "transform": `translate(${x},${isOutput? y+yOffset : y})`
    });

  // Draw the main rectangle.
  nodeGroup.append("rect")
    .attr({
      x: 0,
      y: 0,
      width: RECT_SIZE,
      height: RECT_SIZE,
    })
    .style({
      fill: "#183D4E",
      "stroke-width": 1.0,
      stroke: "black"
    });
  let activeOrNotClass = state[nodeId] ? "active" : "inactive";
  if (isInput) {
    let label = INPUTS[nodeId].label != null ?
        INPUTS[nodeId].label : nodeId;
    // Draw the input label.
    let text = nodeGroup.append("text").attr({
      class: "main-label",
      x: -10,
      y: RECT_SIZE / 2, "text-anchor": "end"
    });
    if (/[_^]/.test(label)) {
      let myRe = /(.*?)([_^])(.)/g;
      let myArray;
      let lastIndex;
      while ((myArray = myRe.exec(label)) != null) {
        lastIndex = myRe.lastIndex;
        let prefix = myArray[1];
        let sep = myArray[2];
        let suffix = myArray[3];
        if (prefix) {
          text.append("tspan").text(prefix);
        }
        text.append("tspan")
        .attr("baseline-shift", sep === "_" ? "sub" : "super")
        .style("font-size", "9px")
        .text(suffix);
      }
      if (label.substring(lastIndex)) {
        text.append("tspan").text(label.substring(lastIndex));
      }
    } else {
      text.append("tspan").text(label);
    }
    nodeGroup.classed(activeOrNotClass, true);
  }
  if (!isInput) {
    // Draw the node's bias.
    nodeGroup.append("rect")
      .attr({
        id: `bias-${nodeId}`,
        x: -BIAS_SIZE - 2,
        y: RECT_SIZE - BIAS_SIZE + 3,
        width: BIAS_SIZE,
        height: BIAS_SIZE,
      }).on("mouseenter", function() {
        updateHoverCard(HoverType.BIAS, node, d3.mouse(container.node()));
      }).on("mouseleave", function() {
        updateHoverCard(null);
      });
  }

  // Draw the node's canvas.
  let div = d3.select("#network").insert("div", ":first-child")
    .attr({
      "id": `canvas-${nodeId}`,
      "class": "canvas"
    })
    .style({
      position: "absolute",
      left: `${x + 3}px`,
      top: `${y + 3}px`,
      "background": "#183D4E"
    })

  if (isInput) {
    div.on("click", function() {
      state[nodeId] = !state[nodeId];
      reset();
    });
    div.style("cursor", "pointer");
  }
  if (isInput) {
    div.classed(activeOrNotClass, true);
  }
}

// Draw network
function drawNetwork(network: nn.Node[][]): void {

  // reset routine
  let svg = d3.select("#svg");
  // Remove all svg elements.
  svg.select("g.core").remove();
  // Remove all div elements.
  d3.select("#network").selectAll("div.canvas").remove();
  d3.select("#network").selectAll("div.plus-minus-neurons").remove();

  // Get the width of the svg container.
  let padding = 3;
  let co = d3.select(".column.output").node() as HTMLDivElement;
  let cf = d3.select(".column.features").node() as HTMLDivElement;
  let width = co.offsetLeft - cf.offsetLeft;
  svg.attr("width", width);

  // Map of all node coordinates.
  let node2coord: {[id: string]: {cx: number, cy: number}} = {};
  let container = svg.append("g")
    .classed("core", true)
    .attr("transform", `translate(${padding},${padding})`);
  // Draw the network layer by layer.
  let numLayers = network.length;
  let featureWidth = 118;
  let layerScale = d3.scale.ordinal<number, number>()
      .domain(d3.range(1, numLayers - 1))
      .rangePoints([featureWidth, width - RECT_SIZE], 0.7);
  let nodeIndexScale = (nodeIndex: number) => nodeIndex * (RECT_SIZE + 25);


  let calloutThumb = d3.select(".callout.thumbnail").style("display", "none");
  let calloutWeights = d3.select(".callout.weights").style("display", "none");
  let idWithCallout = null;
  let targetIdWithCallout = null;

  // Draw the input layer separately.
  let cx = RECT_SIZE / 2 + 50;
  let nodeIds = Object.keys(INPUTS);
  let maxY = nodeIndexScale(nodeIds.length);
  nodeIds.forEach((nodeId, i) => {
    let cy = nodeIndexScale(i) + RECT_SIZE / 2;
    node2coord[nodeId] = {cx, cy};
    drawNode(cx, cy, nodeId, true, container, false, nodeIndexScale((network[1].length/2)-1));
  });

  // Draw the intermediate/hidden layers.
  for (let layerIdx = 1; layerIdx < numLayers - 1; layerIdx++) {
    let numNodes = network[layerIdx].length;
    let cx = layerScale(layerIdx) + RECT_SIZE / 2;
    maxY = Math.max(maxY, nodeIndexScale(numNodes));
    for (let i = 0; i < numNodes; i++) {
      let node = network[layerIdx][i];
      let cy = nodeIndexScale(i) + RECT_SIZE / 2;
      node2coord[node.id] = {cx, cy};
      drawNode(cx, cy, node.id, false, container, layerIdx==2 ? true : false, false ? nodeIndexScale((network[1].length/2)-1) : 0);

      // Show callout to thumbnails.
      let numNodes = network[layerIdx].length;
      let nextNumNodes = network[layerIdx + 1].length;
      if (idWithCallout == null &&
          i === numNodes - 1 &&
          nextNumNodes <= numNodes) {
        calloutThumb.style({
          display: null,
          top: `${20 + 3 + cy}px`,
          left: `${cx}px`
        });
        idWithCallout = node.id;
      }

      // Draw links.
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        let path: SVGPathElement = drawLink(link, node2coord, network,
            container, j === 0, j, node.inputLinks.length, layerIdx==2? true:false, false ? nodeIndexScale((network[1].length/2)-1) : 0).node() as any;

        // Show callout to weights.
        let prevLayer = network[layerIdx - 1];
        let lastNodePrevLayer = prevLayer[prevLayer.length - 1];
        if (targetIdWithCallout == null &&
            i === numNodes - 1 &&
            link.source.id === lastNodePrevLayer.id &&
            (link.source.id !== idWithCallout || numLayers <= 5) &&
            link.dest.id !== idWithCallout &&
            prevLayer.length >= numNodes) {
          let midPoint = path.getPointAtLength(path.getTotalLength() * 0.7);
          calloutWeights.style({
            display: null,
            top: `${midPoint.y + 5}px`,
            left: `${midPoint.x + 3}px`
          });
          targetIdWithCallout = link.dest.id;
        }
      }
    }
  }

  // Adjust the height of the svg.
  svg.attr("height", maxY);

  // Adjust the height of the features column.
  let height = Math.max(
    getRelativeHeight(calloutThumb),
    getRelativeHeight(calloutWeights),
    getRelativeHeight(d3.select("#network"))
  );
  d3.select(".column.features").style("height", height + "px");
}

function getRelativeHeight(selection) {
  let node = selection.node() as HTMLAnchorElement;
  return node.offsetHeight + node.offsetTop;
}

function updateHoverCard(type: HoverType, nodeOrLink?: nn.Node | nn.Link,
    coordinates?: [number, number]) {
  let hovercard = d3.select("#hovercard");
  if (type == null) {
    hovercard.style("display", "none");
    d3.select("#svg").on("click", null);
    return;
  }
  d3.select("#svg").on("click", () => {
    hovercard.select(".value").style("display", "none");
    let input = hovercard.select("input");
    input.style("display", null);
    input.on("input", function() {
      if (this.value != null && this.value !== "") {
        if (type === HoverType.WEIGHT) {
          (nodeOrLink as nn.Link).weight = +this.value;
        } else {
          (nodeOrLink as nn.Node).bias = +this.value;
        }
        updateUI(true);
      }
    });
    input.on("keypress", () => {
      if ((d3.event as any).keyCode === 13) {
        updateHoverCard(type, nodeOrLink, coordinates);
      }
    });
    (input.node() as HTMLInputElement).focus();
  });
  let value = (type === HoverType.WEIGHT) ?
    (nodeOrLink as nn.Link).weight :
    (nodeOrLink as nn.Node).bias;
  let name = (type === HoverType.WEIGHT) ? "Weight" : "Bias";
  hovercard.style({
    "left": `${coordinates[0] + 20}px`,
    "top": `${coordinates[1]}px`,
    "display": "block"
  });
  hovercard.select(".type").text(name);
  hovercard.select(".value")
    .style("display", null)
    .text(value.toPrecision(2));
  hovercard.select("input")
    .property("value", value.toPrecision(2))
    .style("display", "none");
}

function drawLink(
    input: nn.Link, node2coord: {[id: string]: {cx: number, cy: number}},
    network: nn.Node[][], container,
    isFirst: boolean, index: number, length: number, isLast: boolean, yOffset: number) {

  let line = container.insert("path", ":first-child");
  let source = node2coord[input.source.id];
  let dest = node2coord[input.dest.id];
  let datum = {
    source: {
      y: source.cx + RECT_SIZE / 2 + 2,
      x: source.cy
    },
    target: {
      y: dest.cx - RECT_SIZE / 2,
      x: (isLast? dest.cy+yOffset : dest.cy) + ((index - (length - 1) / 2) / length) * 12
    }
  };
  let diagonal = d3.svg.diagonal().projection(d => [d.y, d.x]);

  line.attr({
    "marker-start": "url(#markerArrow)",
    class: "link",
    id: "link" + input.source.id + "-" + input.dest.id,
    d: diagonal(datum, 0)
  });

  container.append("path")
    .attr("d", diagonal(datum, 0))
    .attr("class", "link-hover")
    .on("mouseenter", function() {
      updateHoverCard(HoverType.WEIGHT, input, d3.mouse(this));
    }).on("mouseleave", function() {
      updateHoverCard(null);
    });
  return line;
}

function humanReadable(n: number, digitsAfterDot: number): string {
  return n.toFixed(digitsAfterDot);
}

// parameter flag to indicate if there is a need to update the decision boundary
function updateUI(reset = false) {
  // Update the links visually.
  updateNodesUI(network, d3.select("g.core"), reset);
  updateWeightsUI(network, d3.select("g.core"), reset);
  // Update the bias values visually.
  updateBiasesUI(network);

  function zeroPad(n: number): string {
    let pad = "000000";
    return (pad + n).slice(-pad.length);
  }

  function addCommas(s: string): string {
    return s.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  // Update loss and iteration number.
  d3.select("#loss-train").text(humanReadable(lossTrain, 3));
  d3.select("#loss-test").text(humanReadable(lossTest, 3));
  d3.select("#iter-number").text(addCommas(zeroPad(stepNumber)));
  d3.select("#max-iter-number").text("Max steps per epoch (" + addCommas(zeroPad(maxTrainingSteps)) + ")");
  d3.select("#epoch-number").text(epochNumber);
  lineChart.addDataPoint([lossTrain, lossTest]);

  d3.select("#accuracy-train").text(humanReadable(accuracyTrain, 3));
  d3.select("#accuracy-test").text(humanReadable(accuracyTest, 3));
}

function constructInputIds(): string[] {
  let result: string[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
      result.push(inputName);
    }
  }
  return result;
}

function constructInput(x: number, y: number): number[] {
  let input: number[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
      input.push(INPUTS[inputName].f(x, y));
    }
  }
  return input;
}

export function getOutputWeights(network: nn.Node[][]): number[] {
  let weights: number[] = [];
  for (let layerIdx = 0; layerIdx < network.length - 1; layerIdx++) {
    let currentLayer = network[layerIdx];
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.outputs.length; j++) {
        let output = node.outputs[j];
        weights.push(output.weight);
      }
    }
  }
  return weights;
}

function reset() {
  lineChart.reset();

  
  state.serialize();
  player.pause();
  epochVsAccuracyPlot.reset(state.numberOfEpochs);
  epochVsAccuracyPlot.addDataPoint(0, [0, 0]);
  inputToHiddenLayerDistPlot.reset(10);
  hiddenToOutputLayerDistPlot.reset(10);

  // Make a simple network.
  stepNumber = 0;
  epochNumber = 0;
  maxTrainingSteps = 0;
  maxStepsPerEpoch = 0;
  let numInputs = constructInput(0 , 0).length;
  let shape = [numInputs].concat(state.networkShape).concat([1]);
  let outputActivation = nn.Activations.TANH;
  network = nn.buildNetwork(shape, state.activation, outputActivation, constructInputIds(), state.initZero);
  lossTrain = 0;
  lossTest = 0;
  accuracyTrain = 0;
  accuracyTest = 0;

  // override network here
  drawNetwork(network);

  d3.select("#layers-label").text("Single Hidden Layer");

  
  updateUI(true);
};

function initTutorial() {
  if (state.tutorial == null || state.tutorial === '' || state.hideText) {
    return;
  }
  // Remove all other text.
  d3.selectAll("article div.l--body").remove();
  let tutorial = d3.select("article").append("div")
    .attr("class", "l--body");
  // Insert tutorial text.
  d3.html(`tutorials/${state.tutorial}.html`, (err, htmlFragment) => {
    if (err) {
      throw err;
    }
    tutorial.node().appendChild(htmlFragment);
    // If the tutorial has a <title> tag, set the page title to that.
    let title = tutorial.select("title");
    if (title.size()) {
      d3.select("header h1").style({
        "margin-top": "20px",
        "margin-bottom": "20px",
      })
      .text(title.text());
      document.title = title.text();
    }
  });
}

function isJSON(str) {
  try {
    return (JSON.parse(str) && !!str);
  }
  catch (e) {
    return false;
  }
}

function runWebsocket() {
  setTimeout(function(){}, 3000);
  console.log("Connecting to web socket...")
  openConnection(function (connection) {
    connection.send(
      JSON.stringify(
        {
          "command" : "run",
          "dataset" : userDatasets[state.userDataset],
          "hiddenNodes": state.hiddenNodes,
          "learningRate": state.learningRate,
          "epochs": state.numberOfEpochs
        }
      )
    );
  });
}

function stopWebsocket() {
  openConnection(function (connection) {
    connection.send(JSON.stringify({"command" : "close"}));
  });
}

let conn = {};
function openConnection(cb) {
  // uses global 'conn' object
  // @ts-ignore
  if (conn.readyState === undefined || conn.readyState > 1) {
    conn = new WebSocket('ws://' + window.location.host + '/');
    // @ts-ignore
    conn.onopen = function () {
      if(typeof cb === "function"){
        cb(conn);
      }
    };
    // @ts-ignore
    conn.onmessage = function (event) {
      let message = event.data;

      if (isJSON(message)) {
        message = JSON.parse(message)
        if (message.hasOwnProperty("init_data")) {
          let data = message["init_data"];

          // # update data
          // ## update epoch steps
          maxStepsPerEpoch = data["epoch_steps"];
          maxTrainingSteps = data["epoch_steps"]*data["epochs"];

          // ## update weight data
          updateWeights(data["W1"], 1);
          updateWeights(data["W2"], 2);

          // # update/refresh UI
          // ## update weight distribution
          let W1_flat = [].concat.apply([], data["W1"]);
          let W2_flat = [].concat.apply([], data["W2"]);
          inputToHiddenLayerDistPlot.updateData(W1_flat);
          hiddenToOutputLayerDistPlot.updateData(W2_flat);

          // ## update the rest of the UI
          updateUI(true);
        }
        else if (message.hasOwnProperty("epoch_data")) {
          let data = message["epoch_data"];
          
          epochNumber = data["epoch"]+1;
          stepNumber += maxStepsPerEpoch;

          updateWeights(data["W1"], 1);
          updateWeights(data["W2"], 2);

          lossTrain = data["epoch_loss"]["train"];
          lossTest = data["epoch_loss"]["test"];
          updateLossVsEpochPlot(lossTrain, lossTest);

          accuracyTrain = data["epoch_acc"]["train"];
          accuracyTest = data["epoch_acc"]["test"];
          updateAccuVsEpochPlot(accuracyTrain, accuracyTest, data["epoch"]);

          let W1_flat = [].concat.apply([], data["W1"]);
          let W2_flat = [].concat.apply([], data["W2"]);
          inputToHiddenLayerDistPlot.updateData(W1_flat);
          hiddenToOutputLayerDistPlot.updateData(W2_flat);

          updateUI();
        }
        else {
          console.log("No conditions were met.")
          console.log("Message is typeof " + typeof message)
        }
      }
      else {
        console.log(event.origin)
        console.log(message)
      }
    };

    // @ts-ignore
    conn.onclose = function (event) {
      console.log("\nSocket closed");
    };
  } else if(typeof cb === "function"){
    cb(conn);
  }
}

initTutorial();
makeGUI();
reset();

// @ts-ignore
if (window.WebSocket === undefined) {
  console.log("\nSockets not supported.");
} else {
  // @ts-ignore
  openConnection();
}