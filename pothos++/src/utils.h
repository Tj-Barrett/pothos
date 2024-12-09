// Reading data from file

//#include "mio/mio.hpp"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>

#define TRUE 1
#define FALSE 0

//============================================================================//
//                                                                            //
//   file_info structure                                                      //
//                                                                            //
//============================================================================//

// File info that controls reading
struct file_info {
  int atoms = 0;
  int steps = 0;
  bool periodic = FALSE;
  int polymers = 0;
  int monomers = 0;
};

// File info that controls reading
struct step_box {
  float xlo = 0;
  float ylo = 0;
  float zlo = 0;
  float xhi = 0;
  float yhi = 0;
  float zhi = 0;
  float xlen = 0;
  float ylen = 0;
  float zlen = 0;
};

//============================================================================//
//                                                                            //
//   timing_info structure                                                    //
//                                                                            //
//============================================================================//

// Timing of each event, only used with verbose > 0
struct timing_info {
  double io_in = 0.0 ;
  double io_atom = 0.0 ;
  double io_out = 0.0 ;
  double compute = 0.0 ;
};

//============================================================================//
//                                                                            //
//   atom_info structure                                                      //
//                                                                            //
//============================================================================//
// Per atom structure that holds all info
struct atom_info {
  int step;
  int atom;
  int id;
  int type;
  float x;
  float y;
  float z;
  float ux;
  float uy;
  float uz;
  int ix;
  int iy;
  int iz;
  float dx;
  float dy;
  float dz;
  float align;
};

struct atom_avs{
  int step = 0;
  float dx = 0;
  float dy = 0;
  float dz = 0;
  float align = 0;
};

//============================================================================//
//                                                                            //
//   alignments                                                               //
//                                                                            //
//============================================================================//

std::vector<float> _P2_(std::vector<atom_info> atom_range);

std::vector<float> _P4_(std::vector<atom_info> atom_range);

std::vector<float> _P6_(std::vector<atom_info> atom_range);

std::vector<float> _verho_(std::vector<atom_info> atom_range);


//============================================================================//
//                                                                            //
//   file_info()                                                              //
//                                                                            //
//============================================================================//
struct file_info _file_info(std::string filename);

//============================================================================//
//                                                                            //
//   write_dump()                                                             //
//                                                                            //
//============================================================================//
void write_dump(std::string outFile ,
                file_info *file_info,
                step_box *step_box,
                std::vector<atom_info> *step_info,
                int step,
                int atoms);

//============================================================================//
//                                                                            //
//   write_stats()                                                             //
//                                                                            //
//============================================================================//
void write_stats(std::string outFile,
                 atom_avs *atom_avs);

//============================================================================//
//                                                                            //
//   resources                                                                //
//                                                                            //
//============================================================================//
// inverse square
float invSqrt(float number );

// square
float pow2(float number);
float pow4(float number);
float pow6(float number);

// wrap to periodic condition
int periodic_wrap(float p, float up, float len);

//============================================================================//
//                                                                            //
//   processing()                                                             //
//                                                                            //
//============================================================================//
// comparison function to sort data
bool cmp (const atom_info &a, const atom_info &b);
