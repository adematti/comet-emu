#ifndef GRID_H
#define GRID_H

#include <vector>

class Grid{

  private:

    int n_, n2_, ndim_;
    int nbin_ = 0;
    double kf_, roundk_, roundmu_;
    double s_min_, s_max_;

    std::vector<int>* idk_;
    std::vector<int>* id_shell_;
    std::vector<int>* id_posmu_shell_;
    std::vector<int>* weights_posmu_shell_;

    std::vector<std::vector<int>> tri_;

    std::vector<double> kabs_, kabs_rounded_;
    std::vector<double> mu_, mu_rounded_;
    std::vector<double> phi_;
    // std::vector<std::vector<double>> ksph_;
    // std::vector<double> idphi_;

    int nbin (double, std::vector<double>, double);

  public:

    Grid ();
    Grid (int, double, double=0.001, double=0.001, double=1.0, double=2.0);
    ~Grid ();

    double* kmu123;
    int* weights;
    int* num_tri_f;
    int size;

    int x (int);
    int y (int);
    int z (int);
    int coord_id (int, int);

    int vec_id (int, int, int);
    int vec_id (int*);

    int id_to_m (int);
    int m_to_id (int);

    void generate_triangle_ids (int, double);

    std::vector<int>* get_ptr_idk () { return idk_; }
    int get_idk (int dimension, int index) { return idk_[dimension][index]; }
    double get_kabs (int index) { return kabs_[index]; }
    double get_kabs_rounded (int index) { return kabs_rounded_[index]; }
    double get_mu (int index) { return mu_[index]; }
    double get_mu_rounded (int index) { return mu_rounded_[index]; }

    void find_modes_in_shell (std::vector<double>, double);
    void find_modes_in_posmu_shell (std::vector<double>, double);
    void find_unique_triangles (std::vector<double>, double,
                                std::vector<std::array<double,6>>*,
                                std::vector<size_t>*);

    int get_num_triangle_bins ();
    int get_num_modes_in_shell (int bin)
    { return id_shell_[bin].size(); }
    int get_num_modes_in_posmu_shell (int bin)
    { return id_posmu_shell_[bin].size(); }
    std::vector<int> get_modes_in_shell (int bin)
    { return id_shell_[bin]; }
    int get_modes_in_shell (int bin, int index)
    { return id_shell_[bin][index]; }
    std::vector<int> get_modes_in_posmu_shell (int bin)
    { return id_posmu_shell_[bin]; }
    int get_modes_in_posmu_shell (int bin, int index)
    { return id_posmu_shell_[bin][index]; }
    std::vector<int> get_weights_in_posmu_shell (int bin)
    { return weights_posmu_shell_[bin]; }
    int get_weights_in_posmu_shell (int bin, int index)
    { return weights_posmu_shell_[bin][index]; }
};


#endif
