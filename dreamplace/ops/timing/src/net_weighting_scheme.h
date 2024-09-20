#ifndef DREAMPLACE_NET_WEIGHTING_SCHEME_H_
#define DREAMPLACE_NET_WEIGHTING_SCHEME_H_
#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <ot/timer/timer.hpp>
#include "utility/src/torch.h"
#include "utility/src/utils.h"
#include "place_io/src/Util.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <functional>
#include <utility>
#include <fstream>



namespace _timing_impl {
template<typename T>
using index_type = typename DREAMPLACE_NAMESPACE::coordinate_traits<T>::index_type;
using string2index_map_type = std::unordered_map<std::string, index_type<int> >;
}

DREAMPLACE_BEGIN_NAMESPACE

// The net-weighting scheme enum class.
// We try to implement different net-weighting schemes.
// For different schemes, we implement different algorithms to update net
// weights in each timing iteration.
enum class NetWeightingScheme {
  ADAMS, LILITH, PIN2PIN
};

///
/// \brief Implementation of net-weighting scheme.
/// \param timer the OpenTimer object.
/// \param n the maximum number of paths.
/// \param flat_netpin flatened net pins.
/// \param netpin_start the start index of each net in the flat netpin
/// \param net_name2id_map the net name to id map.
/// \param net_criticality the criticality values of nets (array).
/// \param net_criticality_deltas the criticality delta values of nets (array).
/// \param net_weights the weights of nets (array).
/// \param net_weight_deltas the increment of net weights.
/// \param degree_map the degree map of nets.
/// \param decay the decay factor in momemtum iteration.
/// \param max_net_weight the maximum net weight in timing opt.
/// \param ignore_net_degree the net degree threshold.
/// \param num_threads number of threads for parallel computing.
///
#define DEFINE_APPLY_SCHEME                                        \
  static void apply(                                               \
      ot::Timer& timer, int n,                                     \
      const _timing_impl::string2index_map_type& net_name2id_map,  \
      const _timing_impl::string2index_map_type& pin_name2id_map,  \
      T* net_criticality, T* net_criticality_deltas,               \
      T* net_weights, T* net_weight_deltas, const int* degree_map, \
      pybind11::dict& pin2pin_net_weight,                          \
      T decay, T max_net_weight, int ignore_net_degree,            \
      int num_threads,                                             \
      int pin2pin_max_weight, int pin2pin_min_weight, double pin2pin_accumulate_weight)

///
/// \brief The implementation of net-weighting algorithms.
/// \tparam T the array data type (usually float).
/// \tparam scm the enum net-weighting scheme.
/// Partial specialization of full class should be implemented to correctly
/// enable compile-time polymorphism.
///
template <typename T, NetWeightingScheme scm>
struct NetWeighting {
  DEFINE_APPLY_SCHEME;
};

///
/// \brief Report the slack of a specific pin (given the name of this pin).
//    The report_slack method will be invoked. Note that we extract the worst
//    one of [MIN, MAX] * [FALL, RISE] (4 slacks).
/// \param timer the OpenTimer object.
/// \param name the specific pin name.
///
inline float report_pin_slack(ot::Timer& timer, const std::string& name) {
  using namespace ot;
  // The pin slack defaults to be the largest float number.
  // Use a valid float number instead of the infinity.
  float ps = std::numeric_limits<float>::max();
  FOR_EACH_EL_RF (el, rf) {
    auto s = timer.report_slack(name, el, rf);
    // Check whether the std::optional<float> value indeed has a value or not.
    // The comparison is enabled only when @s has a value.
    if (s) ps = std::min(ps, *s);
  }
  return ps;
}

///
/// \brief Report the slack of a specific net.
/// \param timer the OpenTimer object.
/// \param net the specific net structure in the OpenTimer object.
///
inline float report_net_slack(ot::Timer& timer, const ot::Net& net) {
  // The net slack defaults to the worst one of sinks.
  float slack = std::numeric_limits<float>::max();
  const ot::Pin* root = net.root();
  for (const auto ptr : net.pins()) {
    // Skip the driver in the traversal.
    if (ptr == root) continue;
    float ps = report_pin_slack(timer, ptr->name());
      slack = std::min(slack, ps);
  }
  return slack;
}

////////////////////////////////////////////////////////////////////////////
// Partial specialization of naive net-weighting schemes.
template <typename T>
struct NetWeighting<T, NetWeightingScheme::ADAMS> {
  DEFINE_APPLY_SCHEME {
    // Apply net-weighting scheme.
    dreamplacePrint(kINFO, "apply adams net-weighting scheme...\n");
    
    // Report the first several paths of the critical ones.
    // Note that a path is actually a derived class of std::list<ot::Point>.
    // A Point object contains the corresponding pin.
    // Report timing using the timer object.
    auto beg = std::chrono::steady_clock::now();
    const auto& paths = timer.report_timing(n);
    auto end = std::chrono::steady_clock::now();
    dreamplacePrint(kINFO, "finish report-timing (%f s)\n",
      std::chrono::duration_cast<std::chrono::milliseconds>(
        end - beg).count() * 0.001);

    // Check paths returned by timer.
    if (paths.empty()) {
      dreamplacePrint(kWARN, "report_timing: no critical path found\n");
      return;
    }
    size_t num_nets = timer.num_nets();
    std::vector<bool> net_critical_flag(num_nets, 0);
    beg = std::chrono::steady_clock::now();
    for (auto& path : paths) {
      for (auto& point : path) {
        auto name = point.pin.net()->name();
        int net_id = net_name2id_map.at(name);
        net_critical_flag.at(net_id) = 1;
      }
    }
    // Update the net weights accordingly.
#pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i < num_nets; ++i) {
      if (degree_map[i] > ignore_net_degree) continue;
      net_criticality[i] *= 0.5;
      if (net_critical_flag[i]) net_criticality[i] += 0.5;
      net_weights[i] *= (1 + net_criticality[i]);
    }
    end = std::chrono::steady_clock::now();
    dreamplacePrint(kINFO, "finish net-weighting (%f s)\n",
      std::chrono::duration_cast<std::chrono::milliseconds>(
        end - beg).count() * 0.001);
  }
};

inline void write_slack_data_to_file(const std::unordered_map<std::string, std::vector<float>>& net_slack_differences, const std::string& filename) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return;
    }

    file << "NetName,PinCount,MinSlack,MaxSlack,AvgSlack\n";

    for (const auto& [name, values] : net_slack_differences) {
        file << name << "," << values[0] << "," << values[1] << "," << values[2] << "," << values[3] << "\n";
    }

    file.close();
}

// Partial specialization of lilith net-weighting.
template <typename T>
struct NetWeighting<T, NetWeightingScheme::LILITH> {
  DEFINE_APPLY_SCHEME {
    // Apply net-weighting scheme.
    dreamplacePrint(kINFO, "apply lilith net-weighting scheme...\n");
    dreamplacePrint(kINFO, "lilith mode momentum decay factor: %f\n", decay);
    
    // Calculate run-time of net-weighting update.
    auto beg = std::chrono::steady_clock::now();
    float wns = timer.report_wns().value();
    dreamplacePrint(kINFO, "wns: %f\n", wns);
    double max_nw = 0;
    for (const auto& [name, net] : timer.nets()) {
      // The net id in the dreamplace database.
      int net_id = net_name2id_map.at(name);
      float slack = report_net_slack(timer, net);
      if (wns < 0) {
        float nc = (slack < 0)? std::max(0.f, slack / wns) : 0;
        // Decay the criticality value of the current net.
        net_criticality[net_id] = std::pow(1 + net_criticality[net_id], decay) *
          std::pow(1 + nc, 1 - decay) - 1;
      }

      // Update the net weights accordingly.
      // Ignore the clock net.
      if (degree_map[net_id] > ignore_net_degree)
        continue;
      net_weights[net_id] *= (1 + net_criticality[net_id]);

      // Manually limit the upper bound of the net weights, as it may
      // introduce illegality or divergence for some cases.
      if (net_weights[net_id] > max_net_weight)
        net_weights[net_id] = max_net_weight;
      if (max_nw < net_weights[net_id]) max_nw = net_weights[net_id];
    }

    auto end = std::chrono::steady_clock::now();
    dreamplacePrint(kINFO, "finish net-weighting (%f s)\n",
      std::chrono::duration_cast<std::chrono::milliseconds>(
        end - beg).count() * 0.001);
  }
};


struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        auto hash1 = std::hash<T1>{}(p.first);
        auto hash2 = std::hash<T2>{}(p.second);
        return hash1 ^ (hash2 << 1); 
    }
};

struct pair_equal {
    template <class T1, class T2>
    bool operator()(const std::pair<T1, T2>& p1, const std::pair<T1, T2>& p2) const {
        return p1.first == p2.first && p1.second == p2.second;
    }
};


template <typename T>
struct NetWeighting<T, NetWeightingScheme::PIN2PIN> {
    DEFINE_APPLY_SCHEME {
        // Apply net-weighting scheme.
        dreamplacePrint(kINFO, "apply pin2pin net-weighting scheme...\n");
        // Calculate run-time of net-weighting update.
        auto begT = std::chrono::steady_clock::now();

        dreamplacePrint(kINFO, "extracting paths...\n");
        std::optional<long unsigned int> optionalValue = timer.report_fep();
        int nvp = 0;
        nvp = *optionalValue;
        const auto& paths = timer.report_timing(nvp);
        dreamplacePrint(kINFO, "paths extraction done...\n");
        int num_unique_pairs = 0;
        int num_all_pairs = 0;
        float wns = timer.report_wns().value();

        // #pragma omp parallel for num_threads(52)
        for (int path_idx = 0; path_idx < paths.size(); ++path_idx) {
            const auto& path = paths[path_idx];
            bool first = true;
            int last_id = -1;
            
            std::string last_node_name;
            for (const auto& point : path) {
                std::string name = point.pin.name();

                // Check if `point.pin.gate()` returned a valid pointer
                auto gate = point.pin.gate();
                std::string node_name;

                if (!gate) {
                    node_name = "NO_GATE";
                } else {
                    node_name = gate->name();
                }

                auto it = pin_name2id_map.find(name);
                int pin_id = it->second;

                if (first) {
                    last_id = pin_id;
                    last_node_name = node_name;
                    first = false;
                } else {
                    if (last_node_name == "NO_GATE" || node_name == "NO_GATE" || node_name != last_node_name) {
                        #pragma omp critical
                        {
                          pybind11::tuple key = pybind11::make_tuple(last_id, pin_id);
                          if (pin2pin_net_weight.contains(key)) {
                            {
                              num_all_pairs += 1;
                              pin2pin_net_weight[key] = pin2pin_net_weight[key].cast<float>() + pin2pin_accumulate_weight * path.slack / wns;
                              if (pin2pin_net_weight[key].cast<float>() > pin2pin_max_weight){
                                pin2pin_net_weight[key] = pin2pin_max_weight;
                              }
                            }
                          }
                          else{
                              num_unique_pairs += 1;
                              pin2pin_net_weight[key] = pin2pin_min_weight;
                          }
                        }
                    }
                    last_id = pin_id; 
                    last_node_name = node_name;
                }
            }
        }
        auto endT = std::chrono::steady_clock::now();
        dreamplacePrint(kINFO, "finish net-weighting (%f s)\n",
            std::chrono::duration_cast<std::chrono::milliseconds>(
                endT - begT).count() * 0.001);
        dreamplacePrint(kINFO, "all num %i \n", num_all_pairs);
        dreamplacePrint(kINFO, "unique num %i \n", num_unique_pairs);
      }
};

#undef DEFINE_APPLY_SCHEME

DREAMPLACE_END_NAMESPACE

#endif // DREAMPLACE_NET_WEIGHTING_SCHEME_H_
