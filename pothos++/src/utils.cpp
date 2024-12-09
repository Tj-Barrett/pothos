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

#include "utils.h"

#define TRUE 1
#define FALSE 0

using namespace std;

//============================================================================//
//                                                                            //
//   file_info()                                                              //
//                                                                            //
//============================================================================//
struct file_info _file_info(string filename){

  struct file_info file_info;
  ifstream infile (filename.c_str());
  ifstream infile1 (filename.c_str());

  int nl = 0;
  int iatoms, steps;
  std::string line;
  std::vector<int> v;

  // c++17
  int count = std::count_if(std::istreambuf_iterator<char>{infile}, {}, [](char c) { return c == '\n'; });
  // get the box size, get the number of atoms, figure out number of steps and periodic info

  while (std::getline(infile1,line) ){
    stringstream ss;
    if(nl == 3){
      iatoms = stoi(line.c_str());
      // printf("  Number of atoms : %i.\n", iatoms);
      steps = count/(iatoms+9);
      // printf("  Number of steps : %i.\n", steps);
    }
    else if(nl == 8){
      string tmp;
      ss << line;
      for(int j=0; j<10; j++){

        ss >> tmp;
        if(j==8){
          if(tmp=="ux"){file_info.periodic=FALSE;}
          else if (tmp=="ix"){file_info.periodic=TRUE;}
        }
      }
    }
    else if (nl>9 && nl < (iatoms+9)){
      int tmp;
      ss << line;
      for(int j=0; j<2; j++){
        ss >> tmp;
        if (j==1){v.push_back(tmp);}
      }
    }
    else if(nl > (iatoms+9)) {
      break;}
    nl++;
  }
  file_info.polymers= *std::max_element(v.begin(), v.end());
  file_info.monomers= iatoms/file_info.polymers;
  file_info.atoms=iatoms;
  file_info.steps=steps;
  return file_info;
}

//============================================================================//
//                                                                            //
//   legendre                                                                 //
//                                                                            //
//============================================================================//
vector<float> _P2_(vector<atom_info> atom_range){
  int range = atom_range.size();
  float S = 0;
  int n = 0;
  float d[3] = {};

  for(int i=1; i<(range-1); i++){
    float d1[3] = {};
    float v1[3] = {};
    float v2[3] = {};
    float u1[3] = {};
    float u2[3] = {};
    // vector 2
    v2[0] = atom_range[i+1].ux - atom_range[i].ux;
    v2[1] = atom_range[i+1].uy - atom_range[i].uy;
    v2[2] = atom_range[i+1].uz - atom_range[i].uz;

    // vector 1
    v1[0] = atom_range[i].ux - atom_range[i-1].ux;
    v1[1] = atom_range[i].uy - atom_range[i-1].uy;
    v1[2] = atom_range[i].uz - atom_range[i-1].uz;

    // chord vector
    d1[0] = atom_range[i+1].ux - atom_range[i-1].ux;
    d1[1] = atom_range[i+1].uy - atom_range[i-1].uy;
    d1[2] = atom_range[i+1].uz - atom_range[i-1].uz;

    // unit vectors
    float _vd = pow2(v1[0])+pow2(v1[1])+pow2(v1[2]);
    float _invd = invSqrt(_vd);
    u1[0] = v1[0]*_invd;
    u1[1] = v1[1]*_invd;
    u1[2] = v1[2]*_invd;

    // unit vectors
    _vd = pow2(v2[0])+pow2(v2[1])+pow2(v2[2]);
    _invd = invSqrt(_vd);
    u2[0] = v2[0]*_invd;
    u2[1] = v2[1]*_invd;
    u2[2] = v2[2]*_invd;

    // unit chord vector
    _vd = pow2(d1[0])+pow2(d1[1])+pow2(d1[2]);
    _invd = invSqrt(_vd);
    d[0] += d1[0]*_invd;
    d[1] += d1[1]*_invd;
    d[2] += d1[2]*_invd;

    // P2 function
    float sp = u2[0]*u1[0] + u2[1]*u1[1] + u2[2]*u1[2];
    S += 0.5*(3.* pow2(sp)  - 1. );
    n += 1;
  }

  float invD = invSqrt( pow2(d[0])+pow2(d[1])+pow2(d[2]) );
  float uout[3] = {};
  float u[3] = {};

  u[0]=d[0]*invD;
  u[1]=d[1]*invD;
  u[2]=d[2]*invD;

  if(u[0]<0.){uout[0]=-u[0];}
  if(u[0]>0.){uout[0]=u[0];}
  if(u[1]<0.){uout[1]=-u[1];}
  if(u[1]>0.){uout[1]=u[1];}
  if(u[2]<0.){uout[2]=-u[2];}
  if(u[2]>0.){uout[2]=u[2];}

  std::vector<float> leg;

  // range is 2k+1 vectors, which is really 2k+3 beads
  leg.push_back(S/n);
  leg.push_back(uout[0]);
  leg.push_back(uout[1]);
  leg.push_back(uout[2]);

  return leg;
}

vector<float> _P4_(vector<atom_info> atom_range){
  int range = atom_range.size();
  float S = 0;
  int n = 0;
  float d[3] = {};

  for(int i=1; i<(range-1); i++){
    float d1[3] = {};
    float v1[3] = {};
    float v2[3] = {};
    float u1[3] = {};
    float u2[3] = {};
    // vector 2
    v2[0] = atom_range[i+1].ux - atom_range[i].ux;
    v2[1] = atom_range[i+1].uy - atom_range[i].uy;
    v2[2] = atom_range[i+1].uz - atom_range[i].uz;

    // vector 1
    v1[0] = atom_range[i].ux - atom_range[i-1].ux;
    v1[1] = atom_range[i].uy - atom_range[i-1].uy;
    v1[2] = atom_range[i].uz - atom_range[i-1].uz;

    // chord vector
    d1[0] = atom_range[i+1].ux - atom_range[i-1].ux;
    d1[1] = atom_range[i+1].uy - atom_range[i-1].uy;
    d1[2] = atom_range[i+1].uz - atom_range[i-1].uz;

    // unit vectors
    float _vd = pow2(v1[0])+pow2(v1[1])+pow2(v1[2]);
    float _invd = invSqrt(_vd);
    u1[0] = v1[0]*_invd;
    u1[1] = v1[1]*_invd;
    u1[2] = v1[2]*_invd;

    // unit vectors
    _vd = pow2(v2[0])+pow2(v2[1])+pow2(v2[2]);
    _invd = invSqrt(_vd);
    u2[0] = v2[0]*_invd;
    u2[1] = v2[1]*_invd;
    u2[2] = v2[2]*_invd;

    // unit chord vector
    _vd = pow2(d1[0])+pow2(d1[1])+pow2(d1[2]);
    _invd = invSqrt(_vd);
    d[0] += d1[0]*_invd;
    d[1] += d1[1]*_invd;
    d[2] += d1[2]*_invd;

    // P4 function
    float sp = u2[0]*u1[0] + u2[1]*u1[1] + u2[2]*u1[2];
    S += 1./8.*(35.* pow4(sp) - 30.* pow2(sp) + 3. );
    n += 1;
  }

  float invD = invSqrt( pow2(d[0])+pow2(d[1])+pow2(d[2]) );
  float uout[3] = {};
  float u[3] = {};

  u[0]=d[0]*invD;
  u[1]=d[1]*invD;
  u[2]=d[2]*invD;

  if(u[0]<0.){uout[0]=-u[0];}
  if(u[0]>0.){uout[0]=u[0];}
  if(u[1]<0.){uout[1]=-u[1];}
  if(u[1]>0.){uout[1]=u[1];}
  if(u[2]<0.){uout[2]=-u[2];}
  if(u[2]>0.){uout[2]=u[2];}

  std::vector<float> leg;

  // range is 2k+1 vectors, which is really 2k+3 beads
  leg.push_back(S/n);
  leg.push_back(uout[0]);
  leg.push_back(uout[1]);
  leg.push_back(uout[2]);

  return leg;
}

vector<float> _P6_(vector<atom_info> atom_range){
  int range = atom_range.size();
  float S = 0;
  int n = 0;
  float d[3] = {};

  for(int i=1; i<(range-1); i++){
    float d1[3] = {};
    float v1[3] = {};
    float v2[3] = {};
    float u1[3] = {};
    float u2[3] = {};
    // vector 2
    v2[0] = atom_range[i+1].ux - atom_range[i].ux;
    v2[1] = atom_range[i+1].uy - atom_range[i].uy;
    v2[2] = atom_range[i+1].uz - atom_range[i].uz;

    // vector 1
    v1[0] = atom_range[i].ux - atom_range[i-1].ux;
    v1[1] = atom_range[i].uy - atom_range[i-1].uy;
    v1[2] = atom_range[i].uz - atom_range[i-1].uz;

    // chord vector
    d1[0] = atom_range[i+1].ux - atom_range[i-1].ux;
    d1[1] = atom_range[i+1].uy - atom_range[i-1].uy;
    d1[2] = atom_range[i+1].uz - atom_range[i-1].uz;

    // unit vectors
    float _vd = pow2(v1[0])+pow2(v1[1])+pow2(v1[2]);
    float _invd = invSqrt(_vd);
    u1[0] = v1[0]*_invd;
    u1[1] = v1[1]*_invd;
    u1[2] = v1[2]*_invd;

    // unit vectors
    _vd = pow2(v2[0])+pow2(v2[1])+pow2(v2[2]);
    _invd = invSqrt(_vd);
    u2[0] = v2[0]*_invd;
    u2[1] = v2[1]*_invd;
    u2[2] = v2[2]*_invd;

    // unit chord vector
    _vd = pow2(d1[0])+pow2(d1[1])+pow2(d1[2]);
    _invd = invSqrt(_vd);
    d[0] += d1[0]*_invd;
    d[1] += d1[1]*_invd;
    d[2] += d1[2]*_invd;

    // P6 function
    float sp = u2[0]*u1[0] + u2[1]*u1[1] + u2[2]*u1[2];
    S += 1./16.*(231.* pow6(sp) - 315.* pow4(sp) + 105.* pow2(sp)  - 5. );
    n += 1;
  }

  float invD = invSqrt( pow2(d[0])+pow2(d[1])+pow2(d[2]) );
  float uout[3] = {};
  float u[3] = {};

  u[0]=d[0]*invD;
  u[1]=d[1]*invD;
  u[2]=d[2]*invD;

  if(u[0]<0.){uout[0]=-u[0];}
  if(u[0]>0.){uout[0]=u[0];}
  if(u[1]<0.){uout[1]=-u[1];}
  if(u[1]>0.){uout[1]=u[1];}
  if(u[2]<0.){uout[2]=-u[2];}
  if(u[2]>0.){uout[2]=u[2];}

  std::vector<float> leg;

  // range is 2k+1 vectors, which is really 2k+3 beads
  leg.push_back(S/n);
  leg.push_back(uout[0]);
  leg.push_back(uout[1]);
  leg.push_back(uout[2]);

  return leg;
}

//============================================================================//
//                                                                            //
//   verho                                                                    //
//                                                                            //
//============================================================================//
vector<float> _verho_(vector<atom_info> atom_range){
  int range = atom_range.size();
  float S = 0;
  int n = 0;
  float d[3] = {};
  float invD;

  for(int i=1; i<(range-1); i++){
    float d1[3] = {};

    // chord vector
    d1[0] = atom_range[i+1].ux - atom_range[i-1].ux;
    d1[1] = atom_range[i+1].uy - atom_range[i-1].uy;
    d1[2] = atom_range[i+1].uz - atom_range[i-1].uz;

    invD = invSqrt( pow2(d1[0])+pow2(d1[1])+pow2(d1[2]) );

    // unit vectors
    d[0] += d1[0]*invD;
    d[1] += d1[1]*invD;
    d[2] += d1[2]*invD;

    n += 1;
  }

  invD = invSqrt( pow2(d[0])+pow2(d[1])+pow2(d[2]) );
  float uout[3] = {};
  float u[3] = {};

  S = 1.0 / (n+1) * sqrt( pow2(d[0])+pow2(d[1])+pow2(d[2]) );

  u[0]=d[0]*invD;
  u[1]=d[1]*invD;
  u[2]=d[2]*invD;

  if(u[0]<0.){uout[0]=-u[0];}
  if(u[0]>0.){uout[0]=u[0];}
  if(u[1]<0.){uout[1]=-u[1];}
  if(u[1]>0.){uout[1]=u[1];}
  if(u[2]<0.){uout[2]=-u[2];}
  if(u[2]>0.){uout[2]=u[2];}

  std::vector<float> leg;

  // range is 2k+1 vectors, which is really 2k+3 beads
  leg.push_back(S);
  leg.push_back(uout[0]);
  leg.push_back(uout[1]);
  leg.push_back(uout[2]);

  return leg;
}

//============================================================================//
//                                                                            //
//   processing()                                                             //
//                                                                            //
//============================================================================//
// comparison function to sort data
bool cmp (const atom_info &a, const atom_info &b) {
  return a.atom < b.atom;
}

//============================================================================//
//                                                                            //
//   resources                                                                //
//                                                                            //
//============================================================================//
// inverse square
float invSqrt( float number ){return 1./sqrt(number); }

// square
float pow2(float number){return number*number;}
float pow4(float number){return number*number*number*number;}
float pow6(float number){return number*number*number*number*number*number;}

// wrap to periodic condition
int periodic_wrap(float p, float up, float len){
  int i = 0;
  if (up >= p) { i = (int)( (up-p)/len + 0.5 ); }
  else if (up < p) { i = -(int)( (p-up)/len + 0.5 );}
  return i;
}

//============================================================================//
//                                                                            //
//   write_dump()                                                             //
//                                                                            //
//============================================================================//
void write_dump(string outFile ,
                file_info *file_info,
                step_box *step_box,
                vector<atom_info> *step_info,
                int step,
                int atoms){

  std::ofstream dumpfile;

  // if (file_info->periodic == FALSE){string header_line = "\nITEM: ATOMS id mol type x y z ux uy uz vx vy vz PN\n";}
  // else {string header_line = "\nITEM: ATOMS id mol type x y z ix iy iz vx vy vz PN\n";}}

  string header_line = "\nITEM: ATOMS id mol type x y z ix iy iz vx vy vz PN\n";

  if (step == 0){ dumpfile.open(outFile+".dump", std::ios_base::out); }
  else { dumpfile.open(outFile+".dump", std::ios_base::app); }

  dumpfile << "ITEM: TIMESTEP\n" << step <<"\nITEM: NUMBER OF ATOMS\n" << atoms <<" ";
  dumpfile << "\nITEM: BOX BOUNDS pp pp pp\n0.0 " << (*step_box).xhi;
  dumpfile << "\n 0.0 " << (*step_box).yhi;
  dumpfile << "\n 0.0 " << (*step_box).zhi;
  dumpfile << header_line;

  for (size_t j=0; j<(*step_info).size(); j++){
    dumpfile << (*step_info)[j].atom << " " << (*step_info)[j].id << " " << (*step_info)[j].type << " ";
    dumpfile << (*step_info)[j].x    << " " << (*step_info)[j].y  << " " << (*step_info)[j].z    << " ";
    dumpfile << (*step_info)[j].ix   << " " << (*step_info)[j].iy << " " << (*step_info)[j].iz   << " ";
    dumpfile << (*step_info)[j].dx   << " " << (*step_info)[j].dy << " " << (*step_info)[j].dz   << " ";
    dumpfile << (*step_info)[j].align << " \n";
   }

  dumpfile.close();
}

//============================================================================//
//                                                                            //
//   write_stats()                                                             //
//                                                                            //
//============================================================================//
void write_stats(string outFile,
                 atom_avs *atom_avs){

     int step;
     float av_dx, av_dy, av_dz, av_f;

     step  = (*atom_avs).step;
     av_dx = (*atom_avs).dx;
     av_dy = (*atom_avs).dy;
     av_dz = (*atom_avs).dz;
     av_f  = (*atom_avs).align;

     std::ofstream statfile;
     if (step == 0){statfile.open(outFile+".txt", std::ios_base::out);
      statfile << "step vx vy vz f \n";}
     else if (step > 0){statfile.open(outFile+".txt", std::ios_base::app);}
      statfile << step << " " << av_dx << " " << av_dy << " " << av_dz << " " << av_f << " \n";

     statfile.close();
  }
