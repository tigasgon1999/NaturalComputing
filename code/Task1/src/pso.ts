/* A simple PSO algorithm and the interaction with the NN via the fitness function */
const OMEGA = 0.1;
const ALPHA1 = 2.0;
const ALPHA2 = 2.0;
const SWRMSZ = 100;
const LIMITS = 1; // initialisation is within [-LIMITS,LIMITS]^dim
const LAMBDA = 0.001

import * as nn from "./nn";
import {
  State,
  datasets,
  getKeyFromValue,
} from "./state";
import {Example2D} from "./dataset";

let state = State.deserializeState();

export class Particle {
	x: number[] = [];
	v: number[] = [];
	p: number[] = [];
	fp: number;
	d: number;
	constructor(dim: number, limits: number) {
		this.d = dim;
		for (let i = 0; i < this.d; i++) {
			this.x[i] = limits * 2 * (Math.random() - 0.5);
			this.v[i] = 2 * (Math.random() - 0.5);
			this.p[i] = this.x[i];
			this.fp = Number.MAX_VALUE;
		}
	}
	updatePersonalBest(f: number) {
		if (f < this.fp) {
			this.fp = f;
			for (let j = 0; j < this.d; j++) {
				this.p[j] = this.x[j];
			}
		}
	} 
	getVelocity() {
		let mean_vel = 0;
		for (let j = 0; j < this.d; j++) {
			mean_vel+=this.v[j]*this.v[j];
		}
		mean_vel=Math.sqrt(mean_vel / this.d);
		return(mean_vel);
	}
	updateParticleVeloPos(g: number[]) {
		for (let j = 0; j < this.d; j++) {
			this.v[j] = OMEGA * this.v[j] + ALPHA1 * Math.random() * (this.p[j] - this.x[j]) + ALPHA2 * Math.random() * (g[j] - this.x[j]);
			this.x[j] += this.v[j];

			if (Math.abs(this.x[j])>10.0) this.x[j] = 10.0 * (Math.random() - 0.5); 
				 /* if swarm diverges, then rather change parameters! */
		}
	}
}

export class Swarm {
	particles: Particle[] = [];
	part: Particle;
	g: number[] = []; // global best (so far) vector
	fg: number;	  // fitness of global best
	dim: number;

	updateGlobalBest(f: number, i: number){
		if (f < this.fg) {
			this.fg = f;
			for (let j = 0; j < this.dim; j++) {
				this.g[j] = this.particles[i].x[j];
			}
		}
	}

	updateSwarm(network: nn.Node[][], trainData: Example2D[], testData: Example2D[]): number {

		let f = -1;

		for (let i = 0; i < SWRMSZ; i++) {
			f = getFitness(network,trainData, testData, this.particles[i].x,this.dim);
			this.particles[i].updatePersonalBest(f);
			this.updateGlobalBest(f,i);
		}

		for (let i = 0; i < SWRMSZ; i++) {
			this.particles[i].updateParticleVeloPos(this.g)
		}

		nn.setWeights(network, this.g, this.dim); /* assigns g to weights for visualisation of best */

		f = getFitness(network,trainData, testData, this.g,this.dim);
 		//return(this.particles[0].v[1]);
		return(this.particles[0].getVelocity());
	}
}

export function	buildSwarm(nnDim: number): Swarm {
	let swrm = new Swarm;
	swrm.dim = nnDim;
	for (let i = 0; i < SWRMSZ; i++) {
		let part = new Particle(swrm.dim,LIMITS); 
		swrm.particles.push(part);
	}
	swrm.fg = Number.MAX_VALUE;
	for (let j = 0; j < this.d; j++) {
		this.g[j] = this.particles[0].x[j];
	}
	
	return swrm
}

/* In this function neural network feedforwards each data point once */
export function getFitness(network: nn.Node[][], trainData: Example2D[],testData:Example2D[], x: number[], dim: number): number {
	let nnn = nn.setWeights(network, x, dim); /* assign x to weights */
	let loss_train = getLoss(network, trainData);
	let loss_test = getLoss(network, testData);
	// our designed fitness function:
	if (nnn === dim) return(loss_train + LAMBDA * Math.abs(loss_train-loss_test));
}

/* the following ones are copies from playground.ts */
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

function constructInput(x: number, y: number): number[] {
  let input: number[] = [];
  for (let inputName in INPUTS) {
    if (state[inputName]) {
      input.push(INPUTS[inputName].f(x, y));
    }
  }
  return input;
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

interface InputFeature {
  f: (x: number, y: number) => number;
  label?: string;
}
