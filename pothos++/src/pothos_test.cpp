#include <iostream>
#include <cstring>
#include "legendre.h"
#include "verho.h"

int main(int argc, char** argv){

  legendre_compute("test/small.dump", "test_l" , 2, 2, 0);

  verho_compute("test/small.dump", "test_v" , 2, 0);

}
