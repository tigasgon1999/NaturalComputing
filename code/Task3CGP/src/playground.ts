/* Copyright 2016 Google Inc. All Rights Reserved.

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
//import * as Plotly from 'plotly.js';
import * as Plotly from 'plotly.js/lib/core';
import * as nn from "./nn";
import * as pso from "./pso";
import {HeatMap, reduceMatrix} from "./heatmap";
import {
  State,
  datasets,
  regDatasets,
  activations,
  problems,
  algorithms,
  regularizations,
  getKeyFromValue,
  Problem,
  Algorithm
} from "./state";
import {Example2D, shuffle} from "./dataset";
import {AppendingLineChart} from "./linechart";
import * as d3 from 'd3';

let mainWidth;

// More scrolling
d3.select(".more button").on("click", function() {
  let position = 800;
  d3.transition()
    .duration(1000)
    .tween("scroll", scrollTween(position));
});

function scrollTween(offset) {
  return function() {
    let i = d3.interpolateNumber(window.pageYOffset ||
        document.documentElement.scrollTop, offset);
    return function(t) { scrollTo(0, i(t)); };
  };
}

const RECT_SIZE = 30;
const BIAS_SIZE = 5;
const NUM_SAMPLES_CLASSIFY = 500;
const NUM_SAMPLES_REGRESS = 1200;
const DENSITY = 100;

enum HoverType {
  BIAS, WEIGHT
}

interface InputFeature {
  f: (x: number, y: number) => number;
  label?: string;
}

let INPUTS: {[name: string]: InputFeature} = {
  "x": {f: (x, y) => x, label: "X_1"},
  "y": {f: (x, y) => y, label: "X_2"},
  "xSquared": {f: (x, y) => x * x, label: "X_1^2"},
  "ySquared": {f: (x, y) => y * y,  label: "X_2^2"},
  "xTimesY": {f: (x, y) => x * y, label: "X_1X_2"},
  "sinX": {f: (x, y) => Math.sin(x), label: "sin(X_1)"},
  "sinY": {f: (x, y) => Math.sin(y), label: "sin(X_2)"},
};

let HIDABLE_CONTROLS = [
  ["Show test data", "showTestData"],
  ["Discretize output", "discretize"],
  ["Play button", "playButton"],
  ["Step button", "stepButton"],
  ["Reset button", "resetButton"],
  ["Learning rate", "learningRate"],
  ["Activation", "activation"],
  ["Regularization", "regularization"],
  ["Regularization rate", "regularizationRate"],
  ["Problem type", "problem"],
  ["Which dataset", "dataset"],
  ["Ratio train data", "percTrainData"],
  ["Noise level", "noise"],
  ["Batch size", "batchSize"],
  ["# of hidden layers", "numHiddenLayers"],
  ["Algorithm type", "algorithm"],
];

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
      if (iter === 0) {
        simulationStarted();
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
    optimize_through_GA();
  //   d3.timer(() => {
  //     if (localTimerIndex < this.timerIndex) {
  //       return true;  // Done.
  //     }
	// if (state.algorithm === Algorithm.BACKPROP) oneStep();
	// else PSOout = onePSOStep();
  //     return false;  // Not done.
  //   }, 0);
  }
}

let state = State.deserializeState();

// Filter out inputs that are hidden.
state.getHiddenProps().forEach(prop => {
  if (prop in INPUTS) {
    delete INPUTS[prop];
  }
});

let boundary: {[id: string]: number[][]} = {};
let selectedNodeId: string = null;
// Plot the heatmap.
let xDomain: [number, number] = [-6, 6];
let heatMap =
    new HeatMap(300, DENSITY, xDomain, xDomain, d3.select("#heatmap"),
        {showAxes: true});
let linkWidthScale = d3.scale.linear()
  .domain([0, 5])
  .range([1, 10])
  .clamp(true);
let colorScale = d3.scale.linear<string, number>()
                     .domain([-1, 0, 1])
                     .range(["#f59322", "#e8eaeb", "#0877bd"])
                     .clamp(true);
let iter = 0;
let PSOout = 0;
let trainData: Example2D[] = [];
let testData: Example2D[] = [];
let network: nn.Node[][] = null;
let swarm: pso.Swarm = null;
let lossTrain = 0;
let lossTest = 0;
let player = new Player();
let lineChart = new AppendingLineChart(d3.select("#linechart"),
    ["#777", "black"]);

function makeGUI() {
  d3.select("#reset-button").on("click", () => {
    reset();
    userHasInteracted();
    d3.select("#play-pause-button");
  });

  d3.select("#play-pause-button").on("click", function () {
    // Change the button's content.
    userHasInteracted();
    player.playOrPause();
  });

  player.onPlayPause(isPlaying => {
    d3.select("#play-pause-button").classed("playing", isPlaying);
  });

  d3.select("#next-step-button").on("click", () => {
    player.pause();
    userHasInteracted();
    if (iter === 0) {
      simulationStarted();
    }
	if (state.algorithm === Algorithm.BACKPROP) oneStep();
	else PSOout = onePSOStep();

  });

  d3.select("#data-regen-button").on("click", () => {
    generateData();
    parametersChanged = true;
  });

  let dataThumbnails = d3.selectAll("canvas[data-dataset]");
  dataThumbnails.on("click", function() {
    let newDataset = datasets[this.dataset.dataset];
    if (newDataset === state.dataset) {
      return; // No-op.
    }
    state.dataset =  newDataset;
    dataThumbnails.classed("selected", false);
    d3.select(this).classed("selected", true);
    generateData();
    parametersChanged = true;
    reset();
  });

  let datasetKey = getKeyFromValue(datasets, state.dataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-dataset=${datasetKey}]`)
    .classed("selected", true);

  let regDataThumbnails = d3.selectAll("canvas[data-regDataset]");
  regDataThumbnails.on("click", function() {
    let newDataset = regDatasets[this.dataset.regdataset];
    if (newDataset === state.regDataset) {
      return; // No-op.
    }
    state.regDataset =  newDataset;
    regDataThumbnails.classed("selected", false);
    d3.select(this).classed("selected", true);
    generateData();
    parametersChanged = true;
    reset();
  });

  let regDatasetKey = getKeyFromValue(regDatasets, state.regDataset);
  // Select the dataset according to the current state.
  d3.select(`canvas[data-regDataset=${regDatasetKey}]`)
    .classed("selected", true);

  d3.select("#add-layers").on("click", () => {
    if (state.numHiddenLayers >= 6) {
      return;
    }
    state.networkShape[state.numHiddenLayers] = 2;
    state.numHiddenLayers++;
    parametersChanged = true;
    reset();
  });

  d3.select("#remove-layers").on("click", () => {
    if (state.numHiddenLayers <= 0) {
      return;
    }
    state.numHiddenLayers--;
    state.networkShape.splice(state.numHiddenLayers);
    parametersChanged = true;
    reset();
  });

  let showTestData = d3.select("#show-test-data").on("change", function() {
    state.showTestData = this.checked;
    state.serialize();
    userHasInteracted();
    heatMap.updateTestPoints(state.showTestData ? testData : []);
  });
  // Check/uncheck the checkbox according to the current state.
  showTestData.property("checked", state.showTestData);

  let discretize = d3.select("#discretize").on("change", function() {
    state.discretize = this.checked;
    state.serialize();
    userHasInteracted();
    updateUI();
  });
  // Check/uncheck the checbox according to the current state.
  discretize.property("checked", state.discretize);

  let percTrain = d3.select("#percTrainData").on("input", function() {
    state.percTrainData = this.value;
    d3.select("label[for='percTrainData'] .value").text(this.value);
    generateData();
    parametersChanged = true;
    reset();
  });
  percTrain.property("value", state.percTrainData);
  d3.select("label[for='percTrainData'] .value").text(state.percTrainData);

  let noise = d3.select("#noise").on("input", function() {
    state.noise = this.value;
    d3.select("label[for='noise'] .value").text(this.value);
    generateData();
    parametersChanged = true;
    reset();
  });
  let currentMax = parseInt(noise.property("max"));
  if (state.noise > currentMax) {
    if (state.noise <= 80) {
      noise.property("max", state.noise);
    } else {
      state.noise = 50;
    }
  } else if (state.noise < 0) {
    state.noise = 0;
  }
  noise.property("value", state.noise);
  d3.select("label[for='noise'] .value").text(state.noise);

  let batchSize = d3.select("#batchSize").on("input", function() {
    state.batchSize = this.value;
    d3.select("label[for='batchSize'] .value").text(this.value);
    parametersChanged = true;
    reset();
  });
  batchSize.property("value", state.batchSize);
  d3.select("label[for='batchSize'] .value").text(state.batchSize);

  let activationDropdown = d3.select("#activations").on("change", function() {
    state.activation = activations[this.value];
    state.activations = []
    parametersChanged = true;
    reset();
  });
  activationDropdown.property("value",
      getKeyFromValue(activations, state.activation));

  let learningRate = d3.select("#learningRate").on("change", function() {
    state.learningRate = +this.value;
    state.serialize();
    userHasInteracted();
    parametersChanged = true;
  });
  learningRate.property("value", state.learningRate);

  let regularDropdown = d3.select("#regularizations").on("change",
      function() {
    state.regularization = regularizations[this.value];
    parametersChanged = true;
    reset();
  });
  regularDropdown.property("value",
      getKeyFromValue(regularizations, state.regularization));

  let regularRate = d3.select("#regularRate").on("change", function() {
    state.regularizationRate = +this.value;
    parametersChanged = true;
    reset();
  });
  regularRate.property("value", state.regularizationRate);

  let algorithm = d3.select("#algorithm").on("change", function() {
    state.algorithm = algorithms[this.value];
    parametersChanged = true;
    reset();
  });
  algorithm.property("value", getKeyFromValue(algorithms, state.algorithm));

  let problem = d3.select("#problem").on("change", function() {
    state.problem = problems[this.value];
    generateData();
    drawDatasetThumbnails();
    parametersChanged = true;
    reset();
  });
  problem.property("value", getKeyFromValue(problems, state.problem));

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

  // Hide the text below the visualization depending on the URL.
  if (state.hideText) {
    d3.select("#article-text").style("display", "none");
    d3.select("div.more").style("display", "none");
    d3.select("header").style("display", "none");
  }
}

function updateBiasesUI(network: nn.Node[][]) {
  nn.forEachNode(network, true, node => {
    d3.select(`rect#bias-${node.id}`).style("fill", colorScale(node.bias));
  });
}

function updateWeightsUI(network: nn.Node[][], container) {
  for (let layerIdx = 1; layerIdx < network.length; layerIdx++) {
    let currentLayer = network[layerIdx];
    // Update all the nodes in this layer.
    for (let i = 0; i < currentLayer.length; i++) {
      let node = currentLayer[i];
      for (let j = 0; j < node.inputLinks.length; j++) {
        let link = node.inputLinks[j];
        container.select(`#link${link.source.id}-${link.dest.id}`)
            .style({
              "stroke-dashoffset": -iter / 3,
              "stroke-width": linkWidthScale(Math.abs(link.weight)),
              "stroke": colorScale(link.weight)
            })
            .datum(link);
      }
    }
  }
}

function drawNode(cx: number, cy: number, nodeId: string, isInput: boolean,
    container, node?: nn.Node) {
  let x = cx - RECT_SIZE / 2;
  let y = cy - RECT_SIZE / 2;

  let nodeGroup = container.append("g")
    .attr({
      "class": "node",
      "id": `node${nodeId}`,
      "transform": `translate(${x},${y})`
    });

  // Draw the main rectangle.
  nodeGroup.append("rect")
    .attr({
      x: 0,
      y: 0,
      width: RECT_SIZE,
      height: RECT_SIZE,
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
      top: `${y + 3}px`
    })
    .on("mouseenter", function() {
      selectedNodeId = nodeId;
      div.classed("hovered", true);
      nodeGroup.classed("hovered", true);
      updateDecisionBoundary(network, false);
      heatMap.updateBackground(boundary[nodeId], state.discretize);
    })
    .on("mouseleave", function() {
      selectedNodeId = null;
      div.classed("hovered", false);
      nodeGroup.classed("hovered", false);
      updateDecisionBoundary(network, false);
      heatMap.updateBackground(boundary[nn.getOutputNode(network).id],
          state.discretize);
    });
  if (isInput) {
    div.on("click", function() {
      state[nodeId] = !state[nodeId];
      parametersChanged = true;
      reset();
    });
    div.style("cursor", "pointer");
  }
  if (isInput) {
    div.classed(activeOrNotClass, true);
  }
  let nodeHeatMap = new HeatMap(RECT_SIZE, DENSITY / 10, xDomain,
      xDomain, div, {noSvg: true});
  div.datum({heatmap: nodeHeatMap, id: nodeId});

}

// Draw network
function drawNetwork(network: nn.Node[][]): void {
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
    drawNode(cx, cy, nodeId, true, container);
  });

  // Draw the intermediate layers.
  for (let layerIdx = 1; layerIdx < numLayers - 1; layerIdx++) {
    let numNodes = network[layerIdx].length;
    let cx = layerScale(layerIdx) + RECT_SIZE / 2;
    maxY = Math.max(maxY, nodeIndexScale(numNodes));
    addPlusMinusControl(layerScale(layerIdx), layerIdx);
    for (let i = 0; i < numNodes; i++) {
      let node = network[layerIdx][i];
      let cy = nodeIndexScale(i) + RECT_SIZE / 2;
      node2coord[node.id] = {cx, cy};
      drawNode(cx, cy, node.id, false, container, node);

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
            container, j === 0, j, node.inputLinks.length).node() as any;
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

  // Draw the output node separately.
  cx = width + RECT_SIZE / 2;
  let node = network[numLayers - 1][0];
  let cy = nodeIndexScale(0) + RECT_SIZE / 2;
  node2coord[node.id] = {cx, cy};
  // Draw links.
  for (let i = 0; i < node.inputLinks.length; i++) {
    let link = node.inputLinks[i];
    drawLink(link, node2coord, network, container, i === 0, i,
        node.inputLinks.length);
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

function addPlusMinusControl(x: number, layerIdx: number) {
  let div = d3.select("#network").append("div")
    .classed("plus-minus-neurons", true)
    .style("left", `${x - 10}px`);

  let i = layerIdx - 1;
  let firstRow = div.append("div").attr("class", `ui-numNodes${layerIdx}`);
  firstRow.append("button")
      .attr("class", "mdl-button mdl-js-button mdl-button--icon")
      .on("click", () => {
        let numNeurons = state.networkShape[i];
        if (numNeurons >= 8) {
          return;
        }
        state.networkShape[i]++;
        parametersChanged = true;
        reset();
      })
    .append("i")
      .attr("class", "material-icons")
      .text("add");

  firstRow.append("button")
      .attr("class", "mdl-button mdl-js-button mdl-button--icon")
      .on("click", () => {
        let numNeurons = state.networkShape[i];
        if (numNeurons <= 1) {
          return;
        }
        state.networkShape[i]--;
        parametersChanged = true;
        reset();
      })
    .append("i")
      .attr("class", "material-icons")
      .text("remove");

  let suffix = state.networkShape[i] > 1 ? "s" : "";
  div.append("div").text(
    state.networkShape[i] + " neuron" + suffix
  );
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
        updateUI();
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
    isFirst: boolean, index: number, length: number) {
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
      x: dest.cy + ((index - (length - 1) / 2) / length) * 12
    }
  };
  let diagonal = d3.svg.diagonal().projection(d => [d.y, d.x]);
  line.attr({
    "marker-start": "url(#markerArrow)",
    class: "link",
    id: "link" + input.source.id + "-" + input.dest.id,
    d: diagonal(datum, 0)
  });

  // Add an invisible thick link that will be used for
  // showing the weight value on hover.
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

/**
 * Given a neural network, it asks the network for the output (prediction)
 * of every node in the network using inputs sampled on a square grid.
 * It returns a map where each key is the node ID and the value is a square
 * matrix of the outputs of the network for each input in the grid respectively.
 */
function updateDecisionBoundary(network: nn.Node[][], firstTime: boolean) {
  if (firstTime) {
    boundary = {};
    nn.forEachNode(network, true, node => {
      boundary[node.id] = new Array(DENSITY);
    });
    // Go through all predefined inputs.
    for (let nodeId in INPUTS) {
      boundary[nodeId] = new Array(DENSITY);
    }
  }
  let xScale = d3.scale.linear().domain([0, DENSITY - 1]).range(xDomain);
  let yScale = d3.scale.linear().domain([DENSITY - 1, 0]).range(xDomain);

  let i = 0, j = 0;
  for (i = 0; i < DENSITY; i++) {
    if (firstTime) {
      nn.forEachNode(network, true, node => {
        boundary[node.id][i] = new Array(DENSITY);
      });
      // Go through all predefined inputs.
      for (let nodeId in INPUTS) {
        boundary[nodeId][i] = new Array(DENSITY);
      }
    }
    for (j = 0; j < DENSITY; j++) {
      // 1 for points inside the circle, and 0 for points outside the circle.
      let x = xScale(i);
      let y = yScale(j);
      let input = constructInput(x, y);
      nn.forwardProp(network, input);
      nn.forEachNode(network, true, node => {
        boundary[node.id][i][j] = node.output;
      });
      if (firstTime) {
        // Go through all predefined inputs.
        for (let nodeId in INPUTS) {
          boundary[nodeId][i][j] = INPUTS[nodeId].f(x, y);
        }
      }
    }
  }
}

function getLoss(network: nn.Node[][], dataPoints: Example2D[]): number {
  let loss = 0;
  for (let i = 0; i < dataPoints.length; i++) {
    let dataPoint = dataPoints[i];
    let input = constructInput(dataPoint.x, dataPoint.y);
    let output = nn.forwardProp(network, input);
    loss += nn.Errors.SQUARE.error(output, dataPoint.label);
  }
  return loss / dataPoints.length;
}

function updateUI(firstStep = false) {
  // Update the links visually.
  updateWeightsUI(network, d3.select("g.core"));
  // Update the bias values visually.
  updateBiasesUI(network);
  // Get the decision boundary of the network.
  updateDecisionBoundary(network, firstStep);
  let selectedId = selectedNodeId != null ?
      selectedNodeId : nn.getOutputNode(network).id;
  heatMap.updateBackground(boundary[selectedId], state.discretize);

  // Update all decision boundaries.
  d3.select("#network").selectAll("div.canvas")
      .each(function(data: {heatmap: HeatMap, id: string}) {
    data.heatmap.updateBackground(reduceMatrix(boundary[data.id], 10),
        state.discretize);
  });

  function zeroPad(n: number): string {
    let pad = "000000";
    return (pad + n).slice(-pad.length);
  }

  function addCommas(s: string): string {
    return s.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  function humanReadable(n: number): string {
    return n.toFixed(3);
  }

  // Update loss and iteration number.
  d3.select("#loss-train").text(humanReadable(lossTrain));
  d3.select("#loss-test").text(humanReadable(lossTest));
  d3.select("#iter-number").text(addCommas(zeroPad(iter)));
  d3.select("#PSO-number").text(humanReadable(PSOout));
  lineChart.addDataPoint([lossTrain, lossTest]);
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
function optimize_through_GA() {
console.log("In GA")
const cross_prob = 0
var mutationPr = 0.3
const noIter = 25
const pop_size = 16
const no_best_to_show = 16
const no_epochs = 5

//Returns an integer random number between min (included) and max (included):
function randomInteger(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }

//the Phenotype
var Item = function(num_hidden_layers, shape) {
    this.num_hidden_layers = num_hidden_layers; //int number of hidden layers
    this.shape = shape; //a list which represents the neurons on each layer
  }
// this is the initial population
var items = [];
items.push(new Item(1, [[2,["linear","linear"]],[1,["linear"]],[1,"tanh"]]));
items.push(new Item(1, [[2,["linear","linear"]],[2,["linear","linear"]],[1,"tanh"]]));
items.push(new Item(1, [[2,["linear","linear"]],[3,["linear","linear","linear"]],[1,"tanh"]]));
items.push(new Item(1, [[2,["linear","linear"]],[5,["linear","linear","linear","linear","linear"]],[1,"tanh"]]));
items.push(new Item(1, [[2,["linear","linear"]],[4,["linear","linear","linear","linear"]],[1,"tanh"]]));
//list of possible activations
let activations = ['linear','rbf', 'sin']
let best_indivi_length = []

var string_to_act = {
  'relu': nn.Activations.RELU,
  "tanh": nn.Activations.TANH,
  'sin': nn.Activations.SIN,
  'rbf': nn.Activations.RBF,
  'sigmoid': nn.Activations.SIGMOID,
  'linear' : nn.Activations.LINEAR
};

// genotype
var Gene = function() {
    this.genotype;
    this.fitness;
    this.generation = 0;
  }
// converts a phenotype to a genotype
Gene.prototype.encode = function(phenotype) {
    this.genotype = phenotype.shape.slice();
}

//calculates the fitness function of the gene => replace this with fitness loss function feedforward from nn.
Gene.prototype.calcFitness = function() {
    var scope = this;
    //console.log("Current population: ");
    //console.log(this.genotype);
    //console.log("Fitness for: ", this.genotype);

    //initialize NN structure
    state.numHiddenLayers = this.genotype.length-2;
    state.activations = [];
    for(let i=0;i<state.numHiddenLayer;i++)
    {
      state.networkShape[i] = this.genotype[i+1][0];
    }
    for(let i = 0; i < this.genotype.length-1;i++)
    {
      let layer_activs = []
      for(let j=0;j<this.genotype[i][1].length;j++)
      {
        layer_activs.push(string_to_act[this.genotype[i][1][j]]);
      }
      state.activations.push(layer_activs);
    }
    //clear all inputs
    for (let nodeId in INPUTS)
    {
      state[nodeId] = false;
    }
    //set input layer
    state['x'] = true;
    state['y'] = true;
    if(this.genotype[0][0] == 4)
    {
      state['sinX'] = true;
      state['sinY'] = true;
    }
    parametersChanged = true;
    //reset the network - also adds each activation to the respective node
    reset();
    let iter = 0;
    //train
    for(iter = 0; iter<no_epochs*1000;iter++)
    {
      trainData.forEach((point, i) => {
        let input = constructInput(point.x, point.y);
        nn.forwardProp(network, input);
        nn.backProp(network, point.label, nn.Errors.SQUARE);
        if ((i + 1) % state.batchSize === 0) {
          nn.updateWeights(network, state.learningRate, state.regularizationRate);
        }
      });
      // Compute the loss.
      lossTrain = getLoss(network, trainData);
      lossTest = getLoss(network, testData);
      if(iter % 500 == 0 && iter > 0)
      {  
        //console.log("Acc on test (for this indiv): ")
        let count = 0;
        for (let i = 0; i < testData.length; i++) {
          let dataPoint = testData[i];
          let input = constructInput(dataPoint.x, dataPoint.y);
          let output = nn.forwardProp(network, input);
          if (output >= 0)
          {
            output = 1
          }
          else
          {
            output = -1
          }
          if (output == dataPoint.label)
            {
              count = count + 1
            }
        }
        //console.log(count/testData.length)
      }

    }
    lossTrain = getLoss(network, trainData);
    lossTest = getLoss(network, testData);
    //console.log("Fitness for this individ: -"+lossTest);
    this.fitness = - lossTest;
    
  }

  
// calculates the fitness of a gene which has all the bits = 1
// used to find relative fitness of a gene: fitness/ maxFitness
Gene.prototype.makeMax = function(phenotype) {
  this.fitness = 1
}

//Cross-over operator: one point cross-over
Gene.prototype.onePointCrossOver = function(crossOverPr, anotherGene) {
    //cross over if within cross over probability
    if (Math.random() <= cross_prob) {
      //cross over point:
      //if we have two individuals with only 1 hidden layer => average numbers with simple average and weighed average
      if(this.genotype.length == 3 && anotherGene.genotype.length == 3)
      {
        var offSpring1 = new Gene();
        var offSpring2 = new Gene();

        offSpring1.genotype = this.genotype;
        offSpring2.genotype = this.genotype;
        //this encourages exploration
        offSpring1.genotype[1][0] = Math.floor((this.genotype[1][0] + anotherGene.genotype[1][0]) / 2)
        if(this.genotype[1][1].length <= anotherGene.genotype[1][1].length)
        {
          offSpring1.genotype[1][1] = (this.genotype[1][1].concat(anotherGene.genotype[1][1].slice(this.genotype[1][1].length))).slice(0, offSpring1.genotype[1][0])
        }
        else
        {
          offSpring1.genotype[1][1] = (anotherGene.genotype[1][1].concat(this.genotype[1][1].slice(anotherGene.genotype[1][1].length))).slice(0, offSpring1.genotype[1][0])

        }
        console.log("After cross:");
        console.log(offSpring1.genotype[1]);
        //offSpring1.genotype[1][1] = this.fitness > anotherGene.fitness ? this.genotype[1][1] : anotherGene.genotype[1][1];
        //this is greedier
        offSpring2.genotype[1][0] = this.fitness > anotherGene.fitness ? this.genotype[1][0] : anotherGene.genotype[1][0];//Math.floor(Math.abs(this.fitness) * this.genotype[1] + (1-Math.abs(anotherGene.fitness)) * anotherGene.genotype[1])
        offSpring2.genotype[1][1] = this.fitness > anotherGene.fitness ? this.genotype[1][1] : anotherGene.genotype[1][1];
        return [offSpring1, offSpring2];
      }
      else //one point crossover
      {      
        var crossOver = Math.floor(Math.random() * this.genotype.length);

        if (this.genotype.length == 3)
        {
            crossOver = 1
            var tail1 = this.genotype.slice(crossOver);
            var head1 = this.genotype.slice(0, crossOver+1);
        }
        else
        {
            crossOver = Math.ceil(this.genotype.length * 1.0 /2)
            var tail1 = this.genotype.slice(crossOver);
            var head1 = this.genotype.slice(0, crossOver);
        }
        if (anotherGene.genotype.length == 3)
        {
            crossOver = 1
            var tail2 = anotherGene.genotype.slice(crossOver);
            var head2 = anotherGene.genotype.slice(0, crossOver+1);
        }
        else
        {
            crossOver = Math.ceil(anotherGene.genotype.length * 1.0 /2)
            var tail2 = anotherGene.genotype.slice(crossOver);
            var head2 = anotherGene.genotype.slice(0, crossOver);
        }
        //cross-over at the point and create the off-springs:
        var offSpring1 = new Gene();
        var offSpring2 = new Gene();
        offSpring1.genotype = head1.concat(tail2);
        offSpring2.genotype = head2.concat(tail1);
        return [offSpring1, offSpring2];
      }

    }
  
    var offSpring1 = new Gene();
    var offSpring2 = new Gene();
    offSpring1.genotype = this.genotype.slice();
    offSpring2.genotype = anotherGene.genotype.slice();
  
    return [offSpring1, offSpring2];
  }

  //Mutation operator:
Gene.prototype.mutate = function() {
    let already_mutated_dim = false;
    for (var i = 0; i < this.genotype.length-1; i++) {//do not mutate last gene (output layer)
      //mutate if within cross over probability
      //mutate by adding or substracting one layer or changing slightly the number of neurons on one non-output layer
      if (Math.random() <= mutationPr) {
        if (Math.random() > 0.5)
        {
            //mutate by adding or subtracting a layer
            if (this.genotype.length > 3)
            {
              if(Math.random() < 0.5)
              {
                this.genotype.splice(1, 1);
              }
              else
              {
                if (this.genotype.length < 8) //max 8 layers
                {
                  this.genotype.splice(1, 0, [4,['rbf', 'rbf', 'rbf', 'rbf']]);
                }
                else
                {
                  this.genotype.splice(1, 1);
                }

              }

            }
            else
            {
                //if array on length 3 (minimum), then add a layer
                this.genotype.splice(1, 0, [4,['rbf', 'rbf', 'rbf', 'rbf']]); //adding a layer with 4 neurons
            }
            already_mutated_dim = true;
        }
        else
        {
            let index = i;//randomInteger(0, this.genotype.length - 1)
            //special case when we are on input layer (can be wither 2 or 4 - for sinX)
            if (index == 0)
            {
                if (this.genotype[0][0] == 4)
                {
                  this.genotype[0][0] == 2
                  this.genotype[0][1] == this.genotype[0][1].slice(2);
                }
                else
                {
                  this.genotype[0][0] = 4
                  this.genotype[0][1].push('rbf');this.genotype[0][1].push('rbf');
                }

            }
            else //if the index is on a hidden layer, increase or decrease the value by one
            {
                if (this.genotype[index][0] < 8 && this.genotype[index][0] > 1)
                {
                    if (Math.random() >= 0.5)
                    {
                        //console.log("In mutation + :");
                        //console.log(this.genotype[index]);
                        this.genotype[index][0] +=1 ;
                        this.genotype[index][1].push('rbf');
                    }
                    else
                    {
                        //console.log("In mutation + :");
                        //console.log(this.genotype[index]);
                        this.genotype[index][0] -=1 ;
                        this.genotype[index][1].pop();
                    }

                }
                else
                {
                    if (this.genotype[index][0] == 8)
                    {
                        //console.log("In mutation + :");
                        //console.log(this.genotype[index]);
                        this.genotype[index][0] -=1 ;
                        this.genotype[index][1].pop();
                    }
                    else //it has only 1 n so increase
                    {
                        //console.log("In mutation + :");
                        //console.log(this.genotype[index]);
                        this.genotype[index][0] +=1 ;
                        this.genotype[index][1].push('rbf');
                    }
                }

            }
            if (Math.random() <= 0.5)
            {
              for(let  j=0;j < this.genotype[index][1].length;j++)
              {
                if (Math.random() <= 0.5)
                {
                  this.genotype[index][1][j] = activations[Math.floor(Math.random() * activations.length)]; //get a random activation form the pool
                }
              }
            }
        }
      }
    }
  }
//Compare fitness
function compareFitness(gene1, gene2) {
    return gene2.fitness - gene1.fitness;
  }

// represents a Population of Genotypes
var Population = function(size) {
    this.genes = [];
    this.generation = 0;
    this.solution = 0;
    // create and encode the genes
    while (size--) {
      var gene = new Gene();
      gene.encode(items[size%5]);
      this.genes.push(gene);
    }
  }

// initialization of the Population by making a pass of the fitness function
Population.prototype.initialize = function() {
    //console.log(this.genes[0].genotype);
    for (var i = 0; i < this.genes.length; i++) {
      this.genes[i].calcFitness();
      //console.log(this.genes[i].fitness);
    }
  }
  
//operator select : Rank-based fitness assignment
Population.prototype.select = function() {
    // sort and select the best
    this.genes.sort(compareFitness);
    return [this.genes[0], this.genes[1]];
  }

//calculates one generation from the current population
Population.prototype.generate = function() {
    // select the parents
    let parents = this.select();
    //console.log("Before cross and mutation, best: " + this.genes[0].genotype +" fitness: "+this.genes[0].fitness)
    //console.log("Before cross and mutation, best: " + this.genes)
  
    // cross-over
    var offSpring = parents[0].onePointCrossOver(cross_prob, parents[1]);
    this.generation++;
  
    //re-place in population (replace the worst candidates)
    this.genes.splice(this.genes.length - 2, 2, offSpring[0], offSpring[1]);
    offSpring[0].generation = offSpring[1].generation = this.generation;

    //console.log("After cross and before mutation, best: " + this.genes[0].genotype +" fitness: "+this.genes[0].fitness)
    // for (var i = 0; i<this.genes.length; i++)
    // {
    //   console.log(this.genes[i].genotype)
    // }
    //this.genes[3].mutate(mutationPr);
    //mutate the offspring but keep the best one (adds more stability to the algorithm)
    for (var counter = 1; counter < this.genes.length; counter++) {
      this.genes[counter].mutate(mutationPr);
      // console.log("Step in mutation: ");
      // for (var i = 0; i<this.genes.length; i++)
      // {
      //   console.log(this.genes[i].genotype)
      // }
    }

    //recalculate fitness after cross-over & mutation:
    this.initialize();
    this.genes.sort(compareFitness);
    this.solution = population.genes[0].fitness; // pick the solution;
    best_indivi_length.push(population.genes[0].genotype.length);
    console.log("Length of all  previous best shaped individuals");
    console.log(best_indivi_length);
    //console.log("After cross and mutation, best: " + this.genes[0].genotype +" fitness: "+this.genes[0].fitness)
    //console.log("After cross and mutation, best: " + this.genes)
  
    //draw the population:
    display();
  
    //stop iteration after 100th generation
    //this assumption is arbitrary that the solution would convert after reaching
    //100th generation, there can be other criteria like no change in fitness
    if (this.generation >= noIter) {
      return true;
    }
  
    // call generate again after a delay of 100 mili-second
    var scope = this;
    setTimeout(function() {
      scope.generate();
      //console.log("One step finished! ");
    }, 100);
  }




// code to generate the population and draw it on the Canvas
//window.onload = init;
var canvas;
var context;

//create the population
var population = new Population(pop_size);
var maxSurvivalPoints = 0;
let history_fitness_average = [];
let history_fitness_max = [];
let count = 0;


function init(){
  //gene with maximum fitness possible [without penalty]
  var maxGene = new Gene();
  maxGene.makeMax(items);
  maxSurvivalPoints = maxGene.fitness;

  //get the context for drawing:
  canvas = document.getElementById('populationCanvas');
  context = canvas.getContext('2d');

  population.initialize(); //init the population
  display();
  population.generate(); //start the solution generation
}

//function to draw the population on the canvas
function display(){
  var fitness = document.getElementById('fitness');
  //print the best total Survival point and the corresponding genotype:
  fitness.innerHTML = 'Best fitness:' + population.genes[0].fitness;
  fitness.innerHTML += '<br/>Genotype:' + population.genes[0].genotype;
  let sum = 0;
  for(var j=0; j<no_best_to_show; j++)
  {
    sum = sum + population.genes[j].fitness;
  }
  history_fitness_average.push(1.0*sum/no_best_to_show);

  var foo = [];
  history_fitness_max.push(population.genes[0].fitness);
  
  for (var i = 1; i <= count+1; i++) {
    foo.push(i);
  }
  var trace1 : Plotly.Data[]=[{
    x: foo,
    y: history_fitness_max,
    type: 'scatter'
  }];
  var trace2 : Plotly.Data[]= [{
    x: foo,
    y: history_fitness_average,
    type: 'scatter'
  }];
  
  var layout_max = {
    width: 1000,
    height: 300,
    title: "Evolution of the highest fitness on each generation",
    xaxis: {
      title: "Generation number",
      titlefont: {
        family: "Courier New, monospace",
        size: 18,
        color: "#7f7f7f"
      }
    },
    yaxis: {
      title: "Fitness value",
      titlefont: {
        family: "Courier New, monospace",
        size: 18,
        color: "#7f7f7f"
      }
    }
  };

    
  var layout_avg = {
    width: 1000,
    height: 300,
    title: "Evolution of the average fitness over the population on each generation",
    xaxis: {
      title: "Generation number",
      titlefont: {
        family: "Courier New, monospace",
        size: 18,
        color: "#7f7f7f"
      }
    },
    yaxis: {
      title: "Fitness value",
      titlefont: {
        family: "Courier New, monospace",
        size: 18,
        color: "#7f7f7f"
      }
    }
  };


  var data_max = trace1;
  count +=1
  // if(count % 5 == 0)
  // {
  //   mutationPr = mutationPr / 1.5;
  // }
  Plotly.newPlot('Performance_max', data_max, layout_max);

  var data_average = trace2;
  Plotly.newPlot('Performance_average', data_average, layout_avg);

  context.clearRect(0, 0, canvas.width, canvas.height); //clear the canvas
  var index = 0;
  var radius = 30;
  //draw the Genes
  for(var i = 0; i < 1; i++){
    var centerY = radius + (i + 1) * 5 + i * 2 * radius; //Y
    for(var j = 0; j < pop_size; j++){
      var centerX = radius + (j + 1) * 5 + j * 2 * radius; //X
      context.beginPath();
      context.arc(centerX, centerY, radius, 0, 2 * Math.PI, false);
      // pick the fitness for opacity calculation;
      //console.log(population.genes);
      //var opacity = Math.abs(population.genes[index].fitness) / maxSurvivalPoints;
      context.fillStyle = 'rgba(0,0,255, ' + 0 + ')';
      context.fill();
      context.stroke();
      context.fillStyle = 'black';
      context.textAlign = 'center';
      context.font = 'bold 12pt Calibri';
      // print the generation number
      context.fillText((population.genes[index].fitness).toFixed(2), centerX, centerY);
      index++;
    }
  }
}
init();
}  
function oneStep(): void {
  iter++;
  trainData.forEach((point, i) => {
    let input = constructInput(point.x, point.y);
    nn.forwardProp(network, input);
    nn.backProp(network, point.label, nn.Errors.SQUARE);
    if ((i + 1) % state.batchSize === 0) {
      nn.updateWeights(network, state.learningRate, state.regularizationRate);
    }
  });
  // Compute the loss.
  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  if(iter % 250 == 0)
  {  
    //console.log("Accuracy on test: ")
    let count = 0;
    for (let i = 0; i < testData.length; i++) {
      let dataPoint = testData[i];
      let input = constructInput(dataPoint.x, dataPoint.y);
      let output = nn.forwardProp(network, input);
      if (output >=0)
      {
        output = 1
      }
      else
      {
        output = -1
      }
      if (output == dataPoint.label)
        {
          count = count + 1
        }
    }
    //console.log(count/testData.length)
  }
  updateUI();
}

function onePSOStep(): number {
  iter++;
  PSOout = swarm.updateSwarm(network,trainData, testData); /* changes the network to the gbest network from the swarm */
  // Compute the loss.
  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  if(iter % 250 == 0)
  {  
    //console.log("Accuracy on test: ")
    let count = 0;
    for (let i = 0; i < testData.length; i++) {
      let dataPoint = testData[i];
      let input = constructInput(dataPoint.x, dataPoint.y);
      let output = nn.forwardProp(network, input);
      if (output >=0)
      {
        output = 1
      }
      else
      {
        output = -1
      }
      if (output == dataPoint.label)
        {
          count = count + 1
        }
    }
    //console.log(count/testData.length)
  }
  updateUI();
  return PSOout;
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

function reset(onStartup=false) {
  lineChart.reset();
  state.serialize();
  if (!onStartup) {
    userHasInteracted();
  }
  player.pause();

  let suffix = state.numHiddenLayers !== 1 ? "s" : "";
  d3.select("#layers-label").text("Hidden layer" + suffix);
  d3.select("#num-layers").text(state.numHiddenLayers);

  // Make a simple network.
  iter = 0;
  let numInputs = constructInput(0 , 0).length;
  let shape = [numInputs].concat(state.networkShape).concat([1]);
  let outputActivation = (state.problem === Problem.REGRESSION) ?
      nn.Activations.LINEAR : nn.Activations.TANH;
  network = nn.buildNetwork(shape, state.activation, outputActivation,
      state.regularization, constructInputIds(), state.initZero, state.activations);
  lossTrain = getLoss(network, trainData);
  lossTest = getLoss(network, testData);
  drawNetwork(network);
  swarm = pso.buildSwarm(nn.countWeights(network));
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

function drawDatasetThumbnails() {
  function renderThumbnail(canvas, dataGenerator) {
    let w = 100;
    let h = 100;
    canvas.setAttribute("width", w);
    canvas.setAttribute("height", h);
    let context = canvas.getContext("2d");
    let data = dataGenerator(200, 0);
    data.forEach(function(d) {
      context.fillStyle = colorScale(d.label);
      context.fillRect(w * (d.x + 6) / 12, h * (d.y + 6) / 12, 4, 4);
    });
    d3.select(canvas.parentNode).style("display", null);
  }
  d3.selectAll(".dataset").style("display", "none");

  if (state.problem === Problem.CLASSIFICATION) {
    for (let dataset in datasets) {
      let canvas: any =
          document.querySelector(`canvas[data-dataset=${dataset}]`);
      let dataGenerator = datasets[dataset];
      renderThumbnail(canvas, dataGenerator);
    }
  }
  if (state.problem === Problem.REGRESSION) {
    for (let regDataset in regDatasets) {
      let canvas: any =
          document.querySelector(`canvas[data-regDataset=${regDataset}]`);
      let dataGenerator = regDatasets[regDataset];
      renderThumbnail(canvas, dataGenerator);
    }
  }
}

function hideControls() {
  // Set display:none to all the UI elements that are hidden.
  let hiddenProps = state.getHiddenProps();
  hiddenProps.forEach(prop => {
    let controls = d3.selectAll(`.ui-${prop}`);
    if (controls.size() === 0) {
      console.warn(`0 html elements found with class .ui-${prop}`);
    }
    controls.style("display", "none");
  });

  // Also add checkbox for each hidable control in the "use it in classrom"
  // section.
  let hideControls = d3.select(".hide-controls");
  HIDABLE_CONTROLS.forEach(([text, id]) => {
    let label = hideControls.append("label")
      .attr("class", "mdl-checkbox mdl-js-checkbox mdl-js-ripple-effect");
    let input = label.append("input")
      .attr({
        type: "checkbox",
        class: "mdl-checkbox__input",
      });
    if (hiddenProps.indexOf(id) === -1) {
      input.attr("checked", "true");
    }
    input.on("change", function() {
      state.setHideProperty(id, !this.checked);
      state.serialize();
      userHasInteracted();
      d3.select(".hide-controls-link")
        .attr("href", window.location.href);
    });
    label.append("span")
      .attr("class", "mdl-checkbox__label label")
      .text(text);
  });
  d3.select(".hide-controls-link")
    .attr("href", window.location.href);
}

function generateData(firstTime = false) {
  if (!firstTime) {
    // Change the seed.
    state.seed = Math.random().toFixed(5);
    state.serialize();
    userHasInteracted();
  }
  Math.seedrandom(state.seed);
  let numSamples = (state.problem === Problem.REGRESSION) ?
      NUM_SAMPLES_REGRESS : NUM_SAMPLES_CLASSIFY;
  let generator = state.problem === Problem.CLASSIFICATION ?
      state.dataset : state.regDataset;
  let data = generator(numSamples, state.noise / 100);
  // Shuffle the data in-place.
  shuffle(data);
  // Split into train and test data.
  let splitIndex = Math.floor(data.length * state.percTrainData / 100);
  trainData = data.slice(0, splitIndex);
  testData = data.slice(splitIndex);
  heatMap.updatePoints(trainData);
  heatMap.updateTestPoints(state.showTestData ? testData : []);
}

let firstInteraction = true;
let parametersChanged = false;

function userHasInteracted() {
  if (!firstInteraction) {
    return;
  }
  firstInteraction = false;
  let page = 'index';
  if (state.tutorial != null && state.tutorial !== '') {
    page = `/v/tutorials/${state.tutorial}`;
  }
  ga('set', 'page', page);
  ga('send', 'pageview', {'sessionControl': 'start'});
}

function simulationStarted() {
  ga('send', {
    hitType: 'event',
    eventCategory: 'Starting Simulation',
    eventAction: parametersChanged ? 'changed' : 'unchanged',
    eventLabel: state.tutorial == null ? '' : state.tutorial
  });
  parametersChanged = false;
}

drawDatasetThumbnails();
initTutorial();
makeGUI();
generateData(true);
reset(true);
hideControls();
