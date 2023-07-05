#ifndef UNIQUE_H
#define UNIQUE_H

#include <algorithm>
#include <iterator>
#include <vector>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <boost/functional/hash.hpp>

#define USE_ROBIN_HOOD

#ifdef USE_ROBIN_HOOD
# include "robin_hood.h"
#else
# include <unordered_map>
#endif

template<class T>
struct SortPair
{
  T value;
  std::size_t index;

  SortPair(const T& value, size_t index)
    : value(value), index(index)
  {}
  SortPair(T&& value, size_t index)
    : value(move(value)), index(index)
  {}

  bool operator<(const SortPair& o) const
  { return value < o.value; }

  bool operator<(const T& o) const
  { return value < o; }

  friend bool operator<(const T& left, const SortPair& right)
  { return left < right.value; }

  bool operator==(const SortPair& o) const
  { return value == o.value; }

  friend void swap(SortPair& left, SortPair& right)
  {
      using std::swap;
      swap(left.value, right.value);
      swap(left.index, right.index);
  }
};

template <typename T>
struct container_hash {
    std::size_t operator()(T const& c) const {
        return boost::hash_range(c.begin(), c.end());
    }
};

// template<class T, class Iterator>
// std::vector<T> unique(Iterator first, Iterator last,
//                  std::vector<size_t>* index=nullptr,
//                  std::vector<size_t>* inverse=nullptr,
//                  std::vector<size_t>* count=nullptr);
/**
 * Implements numpy.unique
 *
 * \tparam T scalar value type
 * \tparam Iterator input iterator for type T
 * \param first, last range of values
 * \param index if not null, returns the first indices of each unique value in
 *    the input range
 * \param inverse if not null, returns a mapping to reconstruct the input range
 *    from the output array. first[i] == returnvalue[inverse[i]]
 * \param count if not null, returns the number of times each value appears
 * \return sorted unique values from the input range
 */
template<class T, class Iterator>
std::vector<T> unique(Iterator first, Iterator last,
                      std::vector<std::size_t>* index=nullptr,
                      std::vector<std::size_t>* inverse=nullptr,
                      std::vector<std::size_t>* count=nullptr)
{
  std::vector<T> uvals;
  if(! (index || inverse || count)) { // simple case
    uvals.assign(first, last);
    using t_iter = typename std::vector<T>::iterator;
    const t_iter begin = uvals.begin(), end = uvals.end();
    std::sort(begin, end);
    uvals.erase(std::unique(begin, end), end);
    return uvals;
  }
  if(first == last) { // early out. Helps with inverse computation
    for(std::vector<std::size_t>* arg: {index, inverse, count})
      if(arg)
        arg->clear();
    return uvals;
  }
  using sort_pair = SortPair<T>;
  using sort_pair_iter = typename std::vector<sort_pair>::iterator;
  std::vector<sort_pair> sorted;
  for(std::size_t i = 0; first != last; ++i, ++first)
    sorted.emplace_back(*first, i);
  const sort_pair_iter sorted_begin = sorted.begin(), sorted_end = sorted.end();
  // stable_sort to keep first unique index
  std::stable_sort(sorted_begin, sorted_end);
  /*
   * Compute the unique values. If we still need the sorted original values,
   * this must be a copy. Otherwise we just reuse the sorted vector.
   * Note that the equality operator in SortPair only uses the value, not index
   */
  std::vector<sort_pair> upairs_copy;
  const std::vector<sort_pair>* upairs;
  if(inverse || count) {
    std::unique_copy(sorted_begin, sorted_end, std::back_inserter(upairs_copy));
    upairs = &upairs_copy;
  }
  else {
    sorted.erase(std::unique(sorted_begin, sorted_end), sorted_end);
    upairs = &sorted;
  }
  uvals.reserve(upairs->size());
  for(const sort_pair& i: *upairs)
    uvals.push_back(i.value);
  if(index) {
    index->clear();
    index->reserve(upairs->size());
    for(const sort_pair& i: *upairs)
      index->push_back(i.index);
  }
  if(count) {
    count->clear();
    count->reserve(uvals.size());
  }
  if(inverse) {
    inverse->assign(sorted.size(), 0);
    // Since inverse->resize assigns index 0, we can skip the 0-th outer loop
    sort_pair_iter sorted_i =
      std::upper_bound(sorted_begin, sorted_end, uvals.front());
    if(count)
      count->push_back(std::distance(sorted_begin, sorted_i));
    for(std::size_t i = 1; i < uvals.size(); ++i) {
      const T& uval = uvals[i];
      const sort_pair_iter range_start = sorted_i;
      // we know there is at least one value
      do
        (*inverse)[sorted_i->index] = i;
      while(++sorted_i != sorted_end && sorted_i->value == uval);
      if(count)
        count->push_back(std::distance(range_start, sorted_i));
    }
  }
  if(count && ! inverse) {
    sort_pair_iter range_start = sorted_begin;
    for(const T& uval: uvals) {
      sort_pair_iter range_end =
        std::find_if(std::next(range_start), sorted_end,
                     [&uval](const sort_pair& i) -> bool {
                       return i.value != uval;
                     });
      count->push_back(std::distance(range_start, range_end));
      range_start = range_end;
    }
    /*
     * We could use equal_range or a custom version based on an
     * exponential search to reduce the number of comparisons.
     * The reason we don't use equal_range is because it has worse
     * performance in the worst case (uvals.size() == sorted.size()).
     * We could make a heuristic and dispatch between both implementations
     */
  }
  return uvals;
}

template<class T, class Iterator>
std::vector<T>
unordered_unique(Iterator first, Iterator last,
                 std::vector<std::size_t>* index=nullptr,
                 std::vector<std::size_t>* inverse=nullptr,
                 std::vector<std::size_t>* count=nullptr)
{
#ifdef USE_ROBIN_HOOD
  using index_map = robin_hood::unordered_map<T, std::size_t, container_hash<T>>;
#else
  using index_map = std::unordered_map<T, std::size_t, container_hash<T>>;
#endif
  using map_iter = typename index_map::iterator;
  using map_value = typename index_map::value_type;
  for(std::vector<std::size_t>* arg: {index, inverse, count})
    if(arg)
      arg->clear();
  std::vector<T> uvals;
  index_map map;
  std::size_t cur_idx = 0;
  for(Iterator i = first; i != last; ++cur_idx, ++i) {
    const std::pair<map_iter, bool> inserted =
      map.emplace(*i, uvals.size());
    map_value& ival = *inserted.first;
    if(inserted.second) {
      uvals.push_back(ival.first);
      if(index)
        index->push_back(cur_idx);
      if(count)
        count->push_back(1);
    }
    else if(count)
      (*count)[ival.second] += 1;
    if(inverse)
      inverse->push_back(ival.second);
  }
  return uvals;
}


template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<std::size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  // using std::stable_sort instead of std::sort
  // to avoid unnecessary index re-orderings
  // when v contains elements of equal values
  std::stable_sort(idx.begin(), idx.end(),
       [&v](std::size_t i1, std::size_t i2) {return v[i1] < v[i2];});

  return idx;
}


template<class T, class Iterator>
std::vector<T>
omp_unordered_unique(Iterator first, Iterator last,
                     std::vector<std::size_t>* index=nullptr,
                     std::vector<std::size_t>* count=nullptr)
{
  int nthreads = omp_get_max_threads();
  int chunksize = std::ceil(std::distance(first, last)/nthreads);
  std::size_t* prefix;
  std::vector<T> uvals;
  std::vector<std::size_t> count_threads;
  #pragma omp parallel
  {
    int ithread = omp_get_thread_num();
    #pragma omp single
    {
      prefix = new size_t[nthreads+1];
      prefix[0] = 0;
    }
    Iterator first_private = first, last_private = first;
    std::vector<T> uvals_private;
    std::vector<std::size_t>* index_private = new std::vector<std::size_t>;
    std::vector<std::size_t>* count_private = new std::vector<std::size_t>;
    if (!index) index_private = nullptr;
    if (!count) count_private = nullptr;
    #pragma omp for schedule(static) nowait
    for (int i=0; i<nthreads; i++) {
      std::advance(first_private, i*chunksize);
      if (i == nthreads - 1) {
        last_private = last;
      }
      else {
        std::advance(last_private, (i+1)*chunksize);
      }
      // #pragma omp critical
      // std::cout << i << "\t" << ithread << "\t"
      //           << std::distance(first, first_private) << "\t"
      //           << std::distance(first, last_private) << std::endl;
      uvals_private = unordered_unique<T>(first_private, last_private,
                                          index_private, nullptr,
                                          count_private);
    }
    prefix[ithread+1] = uvals_private.size();
    #pragma omp barrier
    #pragma omp single
    {
        for(int i=1; i<(nthreads+1); i++) prefix[i] += prefix[i-1];
        uvals.resize(uvals.size() + prefix[nthreads]);
        count_threads.resize(count_threads.size() + prefix[nthreads]);
    }
    std::move(uvals_private.begin(), uvals_private.end(),
      uvals.begin() + prefix[ithread]);
    std::move(count_private->begin(), count_private->end(),
      count_threads.begin() + prefix[ithread]);
  }
  delete [] prefix;
  std::vector<std::size_t> inverse, count_joint;
  uvals = unique<T>(uvals.begin(), uvals.end(),
                    index, &inverse, &count_joint);
  std::vector<std::size_t> inverse_argsort = sort_indexes<std::size_t>(inverse);

  int cumsum = 0, incr = 0;
  count->resize(count->size() + count_joint.size(), 0);
  // std::cout << count_joint.size() << "\t" << count_threads.size() << "\t"
  //           << count->size() << "\t" << inverse_argsort.size() << std::endl;
  for(int i=0; i<count_joint.size(); i++) {
    for (int j=cumsum; j<cumsum+count_joint[i]; j++) {
      //std::cout << i << "\t" << incr << "\t" << j << std::endl;
      (*count)[i] += 1; // count_threads[inverse_argsort[incr]];
      incr++;
      cumsum += count_joint[i];
    }
  }
  return uvals;
}
#endif
