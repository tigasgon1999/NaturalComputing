# Natural Computing Coursework 2
Coursework for the Natural Computing course in the MSc Artificial Intelligence at University of Edinburgh (2020/2021).

Group: Tiago Lé (https://github.com/tigasgon1999) and Tidor Pricope (https://github.com/TidorP)
      

### Task 1
The code (that is important to the grading) for the PSO is in `task1_playground/src/pso.ts`, where you can see the new fitness function.

To run it, go to the directory `Task1` and do the following commands:
```
npm run build
npm run serve
```
A GUI will be available in port 5000.

### Task 2
The code (that is important to the grading) for the GA is in `task2_playground/src/playground.ts`. 

Before running it for the first time it is necessary to install a library that allows us to plot graphs of the running experiment by doing the following command:
```
npm install --save plotly.js
```

To run, go to the directory `Task2` and do the following commands:
```
npm run build
npm run serve
```
A GUI will be available in port 5000.

IMPORTANT!!!

Once in GUI, you need to select the following setup: SGD, two-spiral dataset, activation: RBF and 1 hidden layer. If you don't do this (it is some javascript weird thing), the program might not run/ the results can't be reproducted.


### Task 3
Since we followed two different approaches for this task, there are two directories with the code for the task - one for each approach.

The code (important for grading) for the first approach is in `playground.ts` and `nn.ts`. 
The code (important for grading) for the second approach is in `playground.ts` and `nn.ts`. . 
The instructions below are to be followed for both approaches.

Before running it for the first time it is necessary to install a library that allows us to plot graphs of the running experiment by doing the following command:
```
npm install --save plotly.js
```

To run, go to the directory `Task3CGP`/Task3GP` and do the following commands:
```
npm run build
npm run serve
```
A GUI will be available in port 5000.

IMPORTANT!!!

Once in GUI, you need to select the following setup: SGD, two-spiral dataset, activation: RBF and 1 hidden layer and 1 neuron for CGP, 1 hidden layer and 4 neurons for GP on that hidden layer. If you don't do this (it is some javascript weird thing), the program might not run/ the results can't be reproducted.
