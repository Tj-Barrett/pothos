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

using namespace std;

#define TRUE 1
#define FALSE 0

//============================================================================//
//                                                                            //
//   legendre()                                                               //
//                                                                            //
//============================================================================//

void legendre(file_info *file_info,
              vector<atom_info> *step_info,
              atom_avs  *atom_avs,
              int leg, int k, int verbose )
  {
  int polymers = file_info->polymers;
  int monomers = file_info->monomers;
  // for each polymer
  for (int poly=0; poly<polymers; poly++){
    // for each monomer in that polymer
    for (int mono=0; mono<monomers; mono++){
      int ind = poly*monomers+mono;
      vector<float> legendre;
      int _ilow ;
      int _ihigh ;
      // 2k+1 vectors evaluated, which is 2k+3 beads
      int buf = 2; // reevaluate?
      // beginning
      if (mono< (k+buf+1) ){
        // change to pointer so that its looking for memory item at x location
        _ilow  = poly*monomers ;
        _ihigh = ind+k+buf ;
      }
      //end
      else if (mono > (monomers-k-buf-2)){
        _ilow  = ind-k-buf;
        _ihigh = (poly+1)*monomers-1 ;
      }
      //everything else
      else {
        _ilow  = ind-k-buf;
        _ihigh = ind+k+buf+1;
      }

      // this is a very stupid way of solving this
      vector<struct atom_info> atom_range(&(*step_info)[_ilow],&(*step_info)[_ihigh]);

      if (leg ==2){
        legendre = _P2_(atom_range);
      }
      else if (leg ==4){
        legendre = _P4_(atom_range);
      }
      else if (leg ==6){
        legendre = _P6_(atom_range);
      }

      (*step_info)[ind].align   =legendre[0];
      (*step_info)[ind].dx      =legendre[1];
      (*step_info)[ind].dy      =legendre[2];
      (*step_info)[ind].dz      =legendre[3];

      (*atom_avs).align    +=legendre[0];
      (*atom_avs).dx       +=legendre[1];
      (*atom_avs).dy       +=legendre[2];
      (*atom_avs).dz       +=legendre[3];
    }
  }
}

//============================================================================//
//                                                                            //
//   processing()                                                             //
//                                                                            //
//============================================================================//

// processing
void legendre_processing(string filename,
                string outFile,
                file_info *file_info,
                int leg, int k, int verbose){

  bool periodic = file_info->periodic;
  int  atoms    = file_info->atoms;

  // open the file as a stream
  ifstream infile (filename.c_str());
  std::string line;

  // atom_info per timestep
  struct atom_info step ;
  struct step_box step_box;
  vector<atom_info> step_info ;
  struct atom_avs atom_avs;

  // tells it if its looking at a header or the atoms
  bool atom_trigger = FALSE;
  // figure out atom section for each
  int i = 0;
  int st = 0;
  int start, end;
  while(std::getline(infile,line)){

    start =(9+atoms)*(st)+9;
    end   =(9+atoms)*(st+1);

    // for all atoms in timestep step
    stringstream ss;
    double tmp;
    ss << line;

    // staring header
    if ( i < start ){
      // get box dimensions
      if (i == (start-4) ){
        // parse line
        for(int j=0; j<2; j++){
          ss >> tmp;
          if (j==0){step_box.xlo=tmp;}
          if (j==1){step_box.xhi=tmp;}
        }
        step_box.xlen = step_box.xhi-step_box.xlo;
        step_box.xhi = step_box.xlen;
      }
      else if (i == (start-3) ){
        // parse line
        for(int j=0; j<2; j++){
          ss >> tmp;
          if (j==0){step_box.ylo=tmp;}
          if (j==1){step_box.yhi=tmp;}
        }
        step_box.ylen = step_box.yhi-step_box.ylo;
        step_box.yhi = step_box.ylen;
      }
      else if (i == (start-2) ){
        // parse line
        for(int j=0; j<2; j++){
          ss >> tmp;
          if (j==0){step_box.zlo=tmp;}
          if (j==1){step_box.zhi=tmp;}
        }
        step_box.zlen = step_box.zhi-step_box.zlo;
        step_box.zhi = step_box.zlen;
      }
    }

    // data
    else if( ( (start-1) < i ) && ( i < (end) )){
      atom_trigger = TRUE;

      // parse line
      for(int j=0; j<9; j++){
        ss >> tmp;
        if (j==0){step.atom=tmp;}
        else if (j==1){step.id=tmp;}
        else if (j==2){step.type=tmp;}
        else if (j==3){step.x=tmp - step_box.xlo;}
        else if (j==4){step.y=tmp - step_box.ylo;}
        else if (j==5){step.z=tmp - step_box.zlo;}
        else if (j==6 && periodic==FALSE){
          step.ux=tmp - step_box.xlo;
          step.ix= periodic_wrap(step.x, step.ux, step_box.xlen);}
        else if (j==7 && periodic==FALSE){
          step.uy=tmp - step_box.ylo;
          step.iy= periodic_wrap(step.y, step.uy, step_box.ylen);}
        else if (j==8 && periodic==FALSE){
          step.uz=tmp - step_box.zlo;
          step.iz= periodic_wrap(step.z, step.uz, step_box.zlen);}
        else if (j==6 && periodic==TRUE){
          step.ix=tmp;
          step.ux=step.x+step_box.xlen*tmp;}
        else if (j==7 && periodic==TRUE){
          step.iy=tmp;
          step.uy=step.y+step_box.ylen*tmp;}
        else if (j==8 && periodic==TRUE){
          step.iz=tmp;
          step.uz=step.z+step_box.zlen*tmp;}
        }
      step_info.push_back(step);
      step = {};
      }

    // last line triggers writing header line and data
    else if ( (end-1) < i ){
      // sort
      sort(step_info.begin(), step_info.end(), cmp);
      // legendre
      legendre(file_info, &step_info, &atom_avs, leg, k, verbose);
      // write data
      write_dump(outFile, file_info, &step_box, &step_info, st, atoms);
      step_info.clear();

      atom_avs.step = st;
      atom_avs.dx *= 1./atoms;
      atom_avs.dy *= 1./atoms;
      atom_avs.dz *= 1./atoms;
      atom_avs.align *= 1./atoms;
      write_stats(outFile, &atom_avs);

      atom_avs.dx = 0;
      atom_avs.dy = 0;
      atom_avs.dz = 0;
      atom_avs.align = 0;
      atom_trigger = FALSE;
      st++;
    }

    i++;
  }
  // last timestep
  sort(step_info.begin(), step_info.end(), cmp);
  // legendre
  legendre(file_info, &step_info, &atom_avs, leg, k, verbose);
  write_dump(outFile, file_info, &step_box, &step_info, st, atoms);

  atom_avs.step = st;
  atom_avs.dx *= 1./atoms;
  atom_avs.dy *= 1./atoms;
  atom_avs.dz *= 1./atoms;
  atom_avs.align *= 1./atoms;
  write_stats(outFile, &atom_avs);
}

//============================================================================//
//                                                                            //
//   main                                                                     //
//                                                                            //
//============================================================================//
int legendre_compute(std::string inFile,
                     std::string outFile,
                     int leg,
                     int k,
                     int verbose) {

  // chrono stuff
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  auto start = high_resolution_clock::now();

  if (inFile.empty()){
    cout << "Input string is empty. \n";
    return 1;}

  if (outFile.empty()){
    cout << "Output string is empty. \n";
    return 1;}

  if ( (leg != 2) && (leg!=4) && (leg!=6)){
    cout << "Legendre option must be 2, 4, or 6.\n";
    return 1;}

  if (k < 1){
    cout << "Length cutoff must be integer greater or equal to 1.\n";
    return 1;}

  // Get the file information
  file_info file_info = _file_info(inFile);

  auto end = high_resolution_clock::now();

  legendre_processing(inFile, outFile, &file_info, leg, k, verbose);

  end = high_resolution_clock::now();
  auto ms_int = duration_cast<std::chrono::milliseconds>(end-start);

  std::cout << "==================== " << endl;
  std::cout << "pothos++ v0.0.1 \nlegendre P" << k << "\n\n";
  std::cout << "Steps : " << file_info.steps << endl;
  std::cout << "Atoms : " << file_info.atoms << endl;
  std::cout << file_info.monomers <<" monomers in " << file_info.polymers << " polymers \n\n";
  std::cout << "Total time : " << (ms_int.count()/1000.) << " seconds " << std::endl ;
  std::cout << "==================== " << endl;
  return 0;
}
