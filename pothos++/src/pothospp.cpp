#include <iostream>
#include "legendre.h"
#include "verho.h"

static struct PyModuleDef pothos = {
    PyModuleDef_HEAD_INIT,
    "pothos",
    pothos_doc,
    -1,
    pothospp
};

int pothospp(int argc, char** argv){
  //
  // legendre_compute("../big.dump", "test_l" , 2, 2, 0);
  //
  // verho_compute("../big.dump", "test_v" , 5, 0);
  string infile, outfile;
  int k = 1;
  int legendre = 2;
  int verbose = 0;

  // extra args
  for(int i=2; i<argc; ++i){
    // filename
    if ( (std::strcmp("-f",argv[i]) == 0) ){
      infile = argv[i+1];
    }
    //outfile
    if ( (std::strcmp("-o",argv[i]) == 0) ){
      outfile = argv[i+1];
    }
    //length
    if ( (std::strcmp("-k",argv[i]) == 0) ){
      k = std::stoi(argv[i+1]);
    }
    //legendre type
    if ( (std::strcmp("-p",argv[i]) == 0) ){
      legendre = std::stoi(argv[i+1]);
    }
    //verbose level
    if ( (std::strcmp("-v",argv[i]) == 0) ){
      verbose = std::stoi(argv[i+1]);
    }

  }

  // handle run
  if ( std::strcmp("legendre",argv[1]) == 0 ){
    legendre_compute(infile, outfile , legendre, k, verbose);
  }
  // else if (argv[0] =! "verho"){}
  // else if (argv[0] =! "align"){}
  else {
    std::cout << " syntax : pothos legendre/verho/align ... " << std::endl;
    return 1;
  }

}
