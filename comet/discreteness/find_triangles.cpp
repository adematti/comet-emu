#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <array>
#include <iterator>
#include <chrono>
#include <omp.h>
#include <fstream>
#include "grid.h"
#include "unique.h"

using namespace std;

int main()
{
  double kf = 2.0*M_PI/1500.0;
  double dk = 2*kf, half_dk = kf;

  int N = 162;
  Grid grid = Grid(N, kf, 0.2*dk, 0.05);

  vector<double> kbin;
  for (int i=0; i<12; i++)
  {
    kbin.push_back((i+1)*dk);
  }

  grid.generate_triangle_ids(kbin.size());
  int ntri = grid.get_num_triangle_bins();
  cout << "Number of triangle configurations: " << ntri << endl;

  vector<array<double,6>>* kmu123 = new vector<array<double,6>> [ntri];
  vector<size_t>* weights = new vector<size_t> [ntri];
  grid.find_unique_triangles(kbin, dk, kmu123, weights);

  // fstream out;

  // out.open("kmu123.dat", ios::out);
  // for (int i=0; i<ntri; i++)
  // {
  //   out << kmu123[i].size() << "\t";
  // }
  // out << endl;
  //
  // for (int i=0; i<ntri; i++)
  // {
  //   for (int j=0; j<kmu123[i].size(); j++)
  //   {
  //     for (int n=0; n<6; n++)
  //     {
  //       out << kmu123[i][j][n] << "\t";
  //     }
  //     out << weights[i][j] << endl;
  //   }
  // }
  // out.close();

  // out.open("kmu123_rounded_dk_0p002_boost.dat", ios::out);
  // for (int i=0; i<ntri; i++)
  // {
  //   out << kmu123[i].size() << "\t";
  // }
  // out << endl;
  //
  // for (int i=0; i<ntri; i++)
  // {
  //   for (int j=0; j<kmu123[i].size(); j++)
  //   {
  //     for (int n=0; n<6; n++)
  //     {
  //       out << kmu123[i][j][n] << "\t";
  //     }
  //     out << weights[i][j] << endl;
  //   }
  // }
  // out.close();

  return 0;
}
