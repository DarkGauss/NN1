#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <assert.h>
#include <cmath>
#include "rand.h"
#include "mat.h"

#define SLOPE 5

using namespace std;

//Transfer function for nn
double transferFunc(double x)
{
  double slope = 3.0;
  return 1.0/(1.0+exp(-1*slope*x));
}


int main()
{
  //why? I don't know
    initRand();

    //read the number of features from the first row in the input
    int numFeatures;
    int numread = scanf("%d", &numFeatures);
    assert(numread == 1);

    Matrix dataIn1, dataIn2;
    dataIn1.read();

    //test data
    dataIn2.read();



    Matrix x = new Matrix(dataIn1.numRows(), numFeatures + 1, "x");
    x.mapCol(0, [](double d)->double{return 1.0;});

    for (int r = 0; r < x.numRows(); r++) {
      for (int c = 0; c < numFeatures; c ++) {
            x.set(r,c+1, dataIn1.get(r, c));
        }
    }
    Matrix xNoNorm = Matrix(x);
    x.normalizeCols();

    //extract out the y targets
    Matrix t = dataIn1.extract(0, numFeatures, 0,0);
    t.setName("targets");



    //make weights
    Matrix w = new Matrix(x.numCols(),t.numCols(), "weights");
    w.rand(0.0,1.0);

    
    printf("BEGIN TESTING\n");

    //y.print();
    //x.print();

    //begin learning for 10k
    double eta = .1;
    for(int i = 0; i < 10000; i++)
    {
      Matrix t_copy = new Matrix(t);

      Matrix yt = x.dot(w);
      yt.map(transferFunc);

      Matrix errors = t_copy.sub(yt);
      Matrix deltas = x.Tdot(errors);
      deltas.scalarMul(eta);
      w.add(deltas);
    }

    //now we show the tests
    Matrix tests = new Matrix(dataIn2.numRows(), dataIn2.numCols()+1, "tests");
    tests.mapCol(0, [](double d)->double{return 1.0;});
    
    for (int r = 0; r < dataIn2.numRows(); r++) {
      for (int c = 0; c < dataIn2.numCols(); c++) {
            tests.set(r,c+1, dataIn2.get(r, c));
        }
    }

    Matrix testsNoNorm = Matrix(tests);
    tests.normalizeCols(); 

    //run tests

    Matrix testOutput = tests.dot(w);
    testOutput.map(transferFunc);

  //couldn't get writeline to work right
    for (int r = 0; r < tests.numRows(); r++)
    {
        for (int c = 1; c < tests.numCols(); c++)
        {
            printf("%.2f ", tests.get(r, c));
        }
        for (int c = 0; c < testOutput.numCols(); c++)
        {
            printf("%.2f ", testOutput.get(r, c));
        }
        printf("\n");
    }
    return 0;
}
