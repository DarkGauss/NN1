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

//does stuff
double transfer(double x)
{
  return 1.0/(1.0+exp(-1*SLOPE*x));
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
    dataIn2.read();


    //extract out the y targets
    Matrix t = dataIn1.extract(0, numFeatures, 0,0);
    //set the last collum of the x's to 1's to represent the bias's

    Matrix x = dataIn1.mapCol(numFeatures, [](double d)->double{return -1.0;});
    x.narrow(numFeatures+1);
    Matrix xNoNorm = Matrix(x);
    x.normalizeCols();

    //add 1 to features because of bias col
    Matrix w = Matrix(numFeatures+1,1, "weights");
    w.rand(-1.0,1.0);

    
    printf("BEGIN TESTING\n");

    //y.print();
    //x.print();
    //begin learning
    for(int i = 0; i < 10; i++)
    {
      Matrix y = x.dot(w);
      Matrix yt = y.mapCol(0, transfer);
      xNoNorm.print();
      yt.print();
      t.print();
      /*for(int i = 0; i < xNoNorm.numRows(); i++)
      {
        xNoNorm.writeLine(i);
      }
      yt.writeLine(0);
      printf("\n");*/
      Matrix t_copy = Matrix(t);
      double eta = .1;
      Matrix delta = x.Tdot(t_copy.sub(yt)).scalarMul(eta);
      w.add(delta);


    }
}
