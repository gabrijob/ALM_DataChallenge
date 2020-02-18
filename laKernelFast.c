#include <stdio.h>
#include <math.h>
#include <Python.h> //Python.h has all the required function definitions to manipulate the Python objects


#define SEQ_N 4
#define BETA 0.2
#define D 1
#define E 11

static PyObject* laKernelFast_calc(PyObject* self, PyObject* args){
  int similarity_matrix[SEQ_N][SEQ_N] = {0};
  similarity_matrix[0][0] = 1;
  similarity_matrix[1][1] = 1;
  similarity_matrix[2][2] = 1;
  similarity_matrix[3][3] = 1;

  PyObject * x;
  PyObject * y;

  if (! PyArg_ParseTuple( args, "OO", &x, &y))
    return NULL;

  long dx = PyList_Size(x);
  long dy = PyList_Size(y);
  long i,j;

  //create matrixes
  float M[dx][dy];
  float X1[dx][dy];
  float Y1[dx][dy];
  float X2[dx][dy];
  float Y2[dx][dy];

  for(i = 0; i < dx; i++) {
    M[i][0] = 0;
    X1[i][0] = 0;
    Y1[i][0] = 0;
    X2[i][0] = 0;
    Y2[i][0] = 0;
  }
  for(j = 0; j < dx; j++) {
    M[0][j] = 0;
    X1[0][j] = 0;
    Y1[0][j] = 0;
    X2[0][j] = 0;
    Y2[0][j] = 0;
  }

  int x_i, y_j;
  PyObject* temp;

  for(i = 1; i < dx; i++){
    temp = PyList_GetItem(x, i);
    x_i = PyLong_AsLong(temp);
    for(j = 1; j < dy; j++) {
        temp = PyList_GetItem(y, j);
        y_j = PyLong_AsLong(temp);

        M[i][j] = exp(BETA*similarity_matrix[x_i][y_j]) * (1 + X1[i-1][j-1] + Y1[i-1][j-1] + M[i-1][j-1]);
        X1[i][j] = exp(BETA*D) * M[i-1][j] + exp(BETA*E) * X1[i-1][j];
        Y1[i][j] = exp(BETA*D) * (M[i][j- 1] + X1[i][j-1]) + exp(BETA*E) * Y1[i][j-1];
        X2[i][j] = M[i-1][j] + X2[i-1][j];
        Y2[i][j] = M[i][j-1] + X2[i][j-1] + Y2[i][j-1];
    }
  }

  float la_kernel =  1 + X2[dx-1][dy-1] + Y2[dx-1][dy-1] + M[dx-1][dy-1];
  return Py_BuildValue("f", la_kernel);
}


static char laKernelFast_docs[] =
    "calc(x,y): calculates the local alignment kernel for sequences x and y\n";


static PyMethodDef module_methods[] = {
    {"calc", (PyCFunction)laKernelFast_calc, METH_VARARGS, laKernelFast_docs},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef laKernelFast =
{
    PyModuleDef_HEAD_INIT,
    "laKernelFast", /* name of module */
    "usage: laKernelFast.calc(x, y)\n", /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    module_methods
};

PyMODINIT_FUNC PyInit_laKernelFast(void)
{
    return PyModule_Create(&laKernelFast);
}