#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <chrono>
#include "unique.h"
#include "grid.h"

using namespace std;

Grid::Grid () {}

Grid::Grid (int num_grid, double kf, double roundk, double roundmu,
            double s_min, double s_max)
{
  n_ = num_grid;
  n2_ = n_*n_;
  ndim_ = n2_*n_;
  kf_ = kf;
  roundk_ = roundk;
  roundmu_ = roundmu;
  s_min_ = s_min;
  s_max_ = s_max;

  idk_ = new vector<int> [3];
  double k2;
  for (int ii=0; ii<ndim_; ii++)
  {
    k2 = 0.0;
    for (int d=0; d<3; d++)
    {
      idk_[d].push_back(id_to_m(coord_id(ii,d)));
      k2 += idk_[d][ii]*idk_[d][ii];
    }
    k2 = sqrt(k2);
    kabs_.push_back(k2*kf_);
    mu_.push_back(idk_[2][ii]/k2);
    phi_.push_back(atan2(idk_[0][ii],idk_[1][ii]));
    kabs_rounded_.push_back(round(kabs_.back()/roundk_)*roundk_);
    mu_rounded_.push_back(round(mu_.back()/roundmu_)*roundmu_);
  }

  id_shell_ = NULL;
  id_posmu_shell_ = NULL;
}

Grid::~Grid ()
{
  delete [] idk_;
  if (id_shell_) delete [] id_shell_;
}

int Grid::x (int index)
{
  return (index % n2_) % n_;
}

int Grid::y (int index)
{
  return floor((index % n2_)/n_);
}

int Grid::z (int index)
{
  return floor(index/n2_);
}

int Grid::coord_id (int index, int d)
{
  switch (d)
  {
    case 0:
      return x(index);
      break;
    case 1:
      return y(index);
      break;
    case 2:
      return z(index);
      break;
    default:
      cout << "ERROR. Invalid index." << endl;
      exit(EXIT_FAILURE);
  }
}

int Grid::vec_id (int x, int y, int z)
{
  return n2_*z + n_*y + x;
}

int Grid::vec_id (int *x)
{
  return n2_*x[2] + n_*x[1] + x[0];
}

int Grid::id_to_m (int id)
{
  if (id > n_/2)  return id - n_;
  else  return id;
}

int Grid::m_to_id (int mode)
{
  if (mode >= n_/2) mode -= n_/2;
  else if (mode <= -n_/2) mode += n_/2;

  if (mode >= 0)  return mode;
  else  return mode + n_;
}

int Grid::nbin (double check_k, vector<double> k_list, double half_dk)
{
  int bin = -1;
  for (int i=0; i<k_list.size(); i++)
  {
    if (abs(check_k-k_list[i]) < half_dk)
    {
      bin = i;
      break;
    }
  }
  return bin;
}

void Grid::generate_triangle_ids (int nbin, double first_bin_centre)
{
  nbin_ = nbin;
  id_shell_ = NULL;
  id_posmu_shell_ = NULL;
  tri_.clear();
  int ctr = 0, id_ctr = 0;
  for (int i=0; i<nbin_; i++)
  {
    for (int j=0; j<i+1; j++)
    {
      for (int k=0; k<j+1; k++)
      {
        if (first_bin_centre+j+k >= i)
        {
          if ((2*first_bin_centre+j+k)/(first_bin_centre+i) < s_max_ &&
              (2*first_bin_centre+j+k)/(first_bin_centre+i) > s_min_)
          {
            if (tri_.size() == ctr)
            {
              vector<int> conf;
              conf.push_back(id_ctr);
              conf.push_back(i);
              conf.push_back(j);
              tri_.push_back(conf);
            }
            tri_[ctr].push_back(k);
            id_ctr++;
          }
        }
      }
      if (tri_.size() == ctr+1)
      {
        ctr++;
      }
    }
  }
}

int Grid::get_num_triangle_bins()
{
  int ntri = 0;
  for (int i=0; i<tri_.size(); i++)
  {
    ntri += tri_[i].size()-3;
  }
  return ntri;
}

void Grid::find_modes_in_shell (vector<double> k, double dk)
{
  int bin, x;
  double half_dk = 0.5*dk;
  if (!id_shell_)
  {
    id_shell_ = new vector<int> [k.size()];
    for (int ii=0; ii<ndim_; ii++)
    {
      bin = nbin(kabs_[ii], k, half_dk);
      if (bin >= 0) id_shell_[bin].push_back(ii);
    }
  }
}

void Grid::find_modes_in_posmu_shell (vector<double> k, double dk)
{
  if (!id_shell_)
  {
    find_modes_in_shell(k, dk);
  }
  if (!id_posmu_shell_)
  {
    id_posmu_shell_ = new vector<int> [k.size()];
    weights_posmu_shell_ = new vector<int> [k.size()];
    for (int i=0; i<k.size(); i++)
    {
      vector<int> ids = get_modes_in_shell(i);
      vector<array<double,3>> ksph;
      vector<size_t> posmu_ids, posmu_counts;
      for (int ii=0; ii<ids.size(); ii++)
      {
        array<double,3> temp =
          {kabs_[ids[ii]], abs(mu_[ids[ii]]), phi_[ids[ii]]};
        ksph.push_back(temp);
      }
      vector<array<double,3>> ksph_unique = unordered_unique<array<double,3>>(
        ksph.begin(), ksph.end(), &posmu_ids, nullptr, &posmu_counts);
      for (int ii=0; ii<posmu_ids.size(); ii++)
      {
        id_posmu_shell_[i].push_back(ids[posmu_ids[ii]]);
        weights_posmu_shell_[i].push_back(posmu_counts[ii]);
      }
    }
  }
}

void Grid::find_unique_triangles (vector<double> kbin, double dk,
                                  vector<array<double,6>>* kmu123_unique,
                                  vector<size_t>* weights_unique)
{
  if (!id_shell_) find_modes_in_shell(kbin, dk);
  if (!id_posmu_shell_) find_modes_in_posmu_shell(kbin, dk);
  if (nbin_ != kbin.size()) generate_triangle_ids(kbin.size(), kbin[0]/dk);

  // chrono::steady_clock::time_point begin = chrono::steady_clock::now();
  // cout << "Get started!" << endl;

  int nthreads = omp_get_max_threads();
  double half_dk = 0.5*dk;
  double savety_factor = (dk > kf_ ? 1.0 : 1.8);
  for (int n=0; n<tri_.size(); n++)
  {
    // cout << "Iteration: " << n << endl;
    int i1 = tri_[n][1], i2 = tri_[n][2];
    vector<array<double,6>>* kmu123 =
      new vector<array<double,6>> [tri_[n].size()-3];
    vector<size_t>* weights = new vector<size_t> [tri_[n].size()-3];
    for (int m=0; m<tri_[n].size()-3; m++)
    {
      int ntrif = 8*pow(M_PI,2)*(tri_[n][1]+1)*(tri_[n][2]+1)*(tri_[n][m+3]+1)
                  *pow(dk,6)/pow(kf_,6)/1.5*savety_factor;
      kmu123[m].resize(ntrif);
      weights[m].resize(ntrif);
    }
    size_t **prefix;
    #pragma omp parallel
    {
      int ithread = omp_get_thread_num();
      #pragma omp single
      {
        prefix = new size_t* [tri_[n].size()-3];
        for (int m=0; m<tri_[n].size()-3; m++)
        {
          prefix[m] = new size_t [nthreads+1];
          prefix[m][0] = 0;
        }
      }

      array<double,6> vtemp = {0,0,0,0,0,0};
      int id_mode1, id_mode2;
      double kk_shell_12_z, kmag_12, mu_12, temp;
      vector<array<double,6>>* kmu123_private =
        new vector<array<double,6>> [tri_[n].size()-3];
      vector<int>* weights_private = new vector<int> [tri_[n].size()-3];
      for (int m=0; m<tri_[n].size()-3; m++)
      {
        int ntri = 8*pow(M_PI,2)*(tri_[n][1]+1)*(tri_[n][2]+1)*(tri_[n][m+3]+1)
                   * pow(dk,6)/pow(kf_,6)/8;
        kmu123_private[m].reserve(ntri);
        weights_private[m].reserve(ntri);
      }
      #pragma omp for schedule(dynamic) nowait
      for (int i=0; i<get_num_modes_in_posmu_shell(i1); i++)
      {
        for (int j=0; j<get_num_modes_in_shell(i2); j++)
        {
          id_mode1 = get_modes_in_posmu_shell(i1, i);
          id_mode2 = get_modes_in_shell(i2, j);
          kk_shell_12_z = get_idk(2, id_mode1) + get_idk(2, id_mode2);
          kmag_12 = kk_shell_12_z*kk_shell_12_z;
          for (int d=0; d<2; d++)
          {
            temp = get_idk(d, id_mode1) + get_idk(d, id_mode2);
            kmag_12 += temp*temp;
          }
          kmag_12 = sqrt(kmag_12);
          mu_12 = (kmag_12 < 1e-10 ? 0.0 : -kk_shell_12_z/kmag_12);
          kmag_12 *= kf_;
          for (int m=0; m<tri_[n].size()-3; m++)
          {
            if (abs(kmag_12 - kbin[tri_[n][3+m]]) < half_dk)
            {
              vtemp[0] = get_kabs_rounded(id_mode1);
              vtemp[1] = get_kabs_rounded(id_mode2);
              vtemp[2] = round(kmag_12/roundk_)*roundk_;
              vtemp[3] = get_mu_rounded(id_mode1);
              vtemp[4] = get_mu_rounded(id_mode2);
              vtemp[5] = round(mu_12/roundmu_)*roundmu_;
              kmu123_private[m].push_back(vtemp);
              weights_private[m].push_back(
                get_weights_in_posmu_shell(i1, i));
                break;
            }
          }
        }
      }
      for (int m=0; m<tri_[n].size()-3; m++)
      {
        prefix[m][ithread+1] = kmu123_private[m].size();
      }

      #pragma omp barrier
      #pragma omp single
      {
        for (int m=0; m<tri_[n].size()-3; m++)
        {
          for (int i=1; i<nthreads+1; i++) prefix[m][i] += prefix[m][i-1];
        }
      }
      for (int m=0; m<tri_[n].size()-3; m++)
      {
        move(kmu123_private[m].begin(), kmu123_private[m].end(),
             kmu123[m].begin() + prefix[m][ithread]);
        move(weights_private[m].begin(), weights_private[m].end(),
             weights[m].begin() + prefix[m][ithread]);
      }

      delete [] kmu123_private;
      delete [] weights_private;
    }
    for (int m=0; m<tri_[n].size()-3; m++)
    {
      vector<size_t> indices, counts;
      // kmu123_unique[tri_[n][0]+m] = unordered_unique<array<double,6>>(
      //   kmu123[m].begin(), kmu123[m].end(), &indices, nullptr, &counts);
      kmu123_unique[tri_[n][0]+m] = unordered_unique<array<double,6>>(
        kmu123[m].begin(), kmu123[m].begin()+prefix[m][nthreads], &indices,
        nullptr, &counts);
      weights_unique[tri_[n][0]+m] = counts;
      #pragma omp parallel for schedule(static)
      for (int j=0; j<counts.size(); j++)
      {
        weights_unique[tri_[n][0]+m][j] *= weights[m][indices[j]];
      }
    }
    for (int m=0; m<tri_[n].size()-3; m++) delete [] prefix[m];
    delete [] prefix;
    delete [] kmu123;
    delete [] weights;
  }

  // chrono::steady_clock::time_point end = chrono::steady_clock::now();
  //
  // cout << "Time difference = "
  //      << chrono::duration_cast<chrono::milliseconds>(end - begin).count()
  //      << "[ms]" << endl;
}


extern "C"
{
  vector<double>* new_double_vector () { return new vector<double>; }
  void delete_double_vector (vector<double>* v) { delete v; }
  int get_double_vector_size (vector<double>* v) { return v->size(); }
  void push_back_double_vector (vector<double> *v, double val)
  { v->push_back(val); }

  Grid* new_Grid (int num_grid, double kf, double roundk=0.001,
                  double roundmu=0.001, double s_min=1.0, double s_max=2.0)
  { return new Grid(num_grid, kf, roundk, roundmu, s_min, s_max); }

  void find_unique_triangles (Grid* grid, vector<double>* kbin, double dk)
  {
    grid->generate_triangle_ids((*kbin).size(), (*kbin)[0]/dk);
    grid->find_modes_in_shell(*kbin, dk);
    grid->find_modes_in_posmu_shell(*kbin, dk);
    int ntri = grid->get_num_triangle_bins();
    // cout << grid->get_num_triangle_bins() << endl;

    vector<array<double,6>>* kmu123_temp = new vector<array<double,6>> [ntri];
    vector<size_t>* weights_temp = new vector<size_t> [ntri];

    grid->find_unique_triangles(*kbin, dk, kmu123_temp, weights_temp);

    // cout << "done finding triangles" << endl;

    grid->size = 0;
    for (int n=0; n<ntri; n++)
    {
      grid->size += kmu123_temp[n].size();
    }

    grid->kmu123 = new double [grid->size*6];
    grid->weights = new int [grid->size];
    grid->num_tri_f = new int [ntri];
    int counter1 = 0, counter2 = 0;
    for (int n=0; n<ntri; n++)
    {
      for (int i=0; i<kmu123_temp[n].size(); i++)
      {
        for (int j=0; j<6; j++)
        {
          grid->kmu123[counter1] = kmu123_temp[n][i][j];
          counter1++;
        }
        grid->weights[counter2] = weights_temp[n][i];
        counter2++;
      }
      grid->num_tri_f[n] = kmu123_temp[n].size();
    }

    delete [] kmu123_temp;
    delete [] weights_temp;
  }

  int get_num_triangle_bins (Grid* grid)
  { return grid->get_num_triangle_bins(); }

  int get_num_fundamental_triangles (Grid* grid)
  { return grid->size; }

  void get_unique_triangles (Grid* grid, double* kmu123, int* weights,
                             int* num_tri_f)
  {
    move(grid->kmu123, grid->kmu123+grid->size*6, kmu123);
    move(grid->weights, grid->weights+grid->size, weights);
    move(grid->num_tri_f, grid->num_tri_f+grid->get_num_triangle_bins(),
         num_tri_f);
  }
}
