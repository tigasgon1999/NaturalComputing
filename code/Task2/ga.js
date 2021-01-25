const cross_prob = 0.9
const mutationPr = 0.25
const noIter = 50
const pop_size = 50
no_best_to_show = 7
//Returns an integer random number between min (included) and max (included):
function randomInteger(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }

//the Phenotype
var Item = function(num_hidden_layers, shape) {
    this.num_hidden_layers = num_hidden_layers; //int number of hidden layers
    this.shape = shape; //a list which represents the neurons on each layer
  }

var items = [];
items.push(new Item(1, [2,1,1]));
items.push(new Item(1, [2,2,1]));
items.push(new Item(1, [2,3,1]));
items.push(new Item(2, [2,1,1]));
items.push(new Item(2, [2,4,1]));

// genotype
var Gene = function() {
    this.genotype;
    this.fitness;
    this.generation = 0;
  }
// converts a phenotype to a genotype
Gene.prototype.encode = function(phenotype) {
    this.genotype = phenotype.shape;
}

//calculates the fitness function of the gene => replace this with fitness loss function feedforward from nn.
Gene.prototype.calcFitness = function() {
    var scope = this;
    //console.log(this.genotype);
    //console.log(scope.genotype)

    if(scope.genotype.length == 4 && scope.genotype[0] == 4 && scope.genotype[1] == 6 && (scope.genotype[2] >= 5 || scope.genotype[2] <= 7))
    {
        this.fitness = 0
    }  
    else if(scope.genotype.length == 4 && scope.genotype[0] + scope.genotype[1] >= 8 &&  scope.genotype[0] + scope.genotype[1] < 12)
    {
        this.fitness = -0.32
    }  
    else if(scope.genotype.length == 2 && scope.genotype[0] + scope.genotype[1] >= 8 &&  scope.genotype[0] + scope.genotype[1] < 12)
    {
        this.fitness = -0.67
    }  
    else{
        this.fitness = -0.9
    }
    //console.log("Fitness: "+ this.fitness)
    //return this.fitness
    
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
        offSpring1.genotype[1] = (this.genotype[1] + anotherGene.genotype[1]) / 2
        //this is greedier
        offSpring2.genotype[1] = this.fitness > anotherGene.fitness ? this.genotype[1] : anotherGene.genotype[1];//Math.floor(Math.abs(this.fitness) * this.genotype[1] + (1-Math.abs(anotherGene.fitness)) * anotherGene.genotype[1])
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
  
    return [this, anotherGene];
  }

  //Mutation operator:
Gene.prototype.mutate = function() {
    let already_mutated_dim = false;
    for (var i = 0; i < this.genotype.length-1; i++) {//do not mutate last gene (output layer)
      //mutate if within cross over probability
      //mutate by adding or substracting one layer or changing slightly the number of neurons on one non-output layer
      if (Math.random() <= mutationPr) {
        if (Math.random() >= 0.75 && already_mutated_dim == false)
        {
            //mutate by adding or subtracting a layer
            if (this.genotype.length > 3)
            {
                this.genotype.splice(1, 1);
            }
            else
            {
                //if array on length 3 (minimum), then add a layer
                this.genotype.splice(1, 0, 4); //adding a layer with 4 neurons
            }
            already_mutated_dim = true;
        }
        else
        {
            let index = i;//randomInteger(0, this.genotype.length - 1)
            //special case when we are on input layer (can be wither 2 or 4 - for sinX)
            if (index == 0)
            {
                if (this.genotype[0] == 4)
                {
                  this.genotype[0] == 2
                }
                else
                {
                  this.genotype[0] = 4
                }

            }
            else //if the index is on a hidden layer, increase or decrease the value by one
            {
                if (this.genotype[index] < 8 && this.genotype[index] > 1)
                {
                    if (Math.random() >= 0.5)
                    {
                        this.genotype[index] +=1 
                    }
                    else
                    {
                        this.genotype[index] -=1 
                    }

                }
                else
                {
                    if (this.genotype[index] == 8)
                    {
                        this.genotype[index] -=1 
                    }
                    else
                    {
                        this.genotype[index] +=1 
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
    console.log("Before cross and mutation, best: " + this.genes[0].genotype +" fitness: "+this.genes[0].fitness)
    console.log("Before cross and mutation, best: " + this.genes)
  
    // cross-over
    var offSpring = parents[0].onePointCrossOver(cross_prob, parents[1]);
    this.generation++;
  
    //re-place in population (replace the worst candidates)
    this.genes.splice(this.genes.length - 2, 2, offSpring[0], offSpring[1]);
    offSpring[0].generation = offSpring[1].generation = this.generation;

    console.log("After cross and before mutation, best: " + this.genes[0].genotype +" fitness: "+this.genes[0].fitness)
    for (var i = 0; i<this.genes.length; i++)
    {
      console.log(this.genes[i].genotype)
    }
    this.genes[3].mutate(mutationPr);
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
    console.log("After cross and mutation, best: " + this.genes[0].genotype +" fitness: "+this.genes[0].fitness)
    console.log("After cross and mutation, best: " + this.genes)
  
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
    }, 1500);
  }




// code to generate the population and draw it on the Canvas
window.onload = init;
var canvas;
var context;

//create the population
var population = new Population(pop_size);
var maxSurvivalPoints = 0;

function init(){
  //gene with maximum fitness possible [without penalty]
  var maxGene = new Gene();
  maxGene.makeMax(items);
  maxSurvivalPoints = maxGene.fitness;

  //get the context for drawing:
  canvas = document.getElementById('populationCanvas');
  context = canvas.getContext('2d');

  population.initialize(); //init the population
  population.generate(); //start the solution generation
}
history_fitness_average = [];
history_fitness_max = [];
count = 0;
//function to draw the population on the canvas
function display(){
  var fitness = document.getElementById('fitness');
  //print the best total Survival point and the corresponding genotype:
  fitness.innerHTML = 'Survival Points:' + population.genes[0].fitness;
  fitness.innerHTML += '<br/>Genotype:' + population.genes[0].genotype;
  sum = 0;
  for(var j=0; j<no_best_to_show; j++)
  {
    sum = sum + population.genes[j].fitness;
  }
  history_fitness_average.push(1.0*sum/no_best_to_show);

  var foo = [];
  history_fitness_max.push(population.genes[0].fitness);

  for (var i = 1; i <= count; i++) {
    foo.push(i);
  }
  var trace1 = {
    x: foo,
    y: history_fitness_max,
    type: 'scatter'
  };
  var trace2 = {
    x: foo,
    y: history_fitness_average,
    type: 'scatter'
  };


  var data_max = [trace1];
  count +=1
  Plotly.newPlot('Performance_max', data_max);

  var data_average = [trace2];
  Plotly.newPlot('Performance_average', data_average);

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
      context.fillText(population.genes[index].fitness, centerX, centerY);
      index++;
    }
  }
}