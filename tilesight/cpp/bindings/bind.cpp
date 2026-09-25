#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "tilesight/engine.hpp"

namespace nb = nanobind;
using namespace tilesight;

NB_MODULE(_core, m) {
  m.doc() = "TileSight C++ core (pipeline-envelope engine + tile reuse-distance cache model)";
  nb::class_<Lane>(m, "Lane")
      .def(nb::init<>())
      .def_rw("name", &Lane::name).def_rw("shared", &Lane::shared)
      .def_rw("total_rate", &Lane::total_rate).def_rw("per_sm_cap", &Lane::per_sm_cap);
  nb::class_<Action>(m, "Action")
      .def(nb::init<>())
      .def_rw("name", &Action::name).def_rw("work", &Action::work).def_rw("deps", &Action::deps)
      .def_rw("recurrent", &Action::recurrent).def_rw("latency", &Action::latency);
  nb::class_<Kernel>(m, "Kernel")
      .def(nb::init<>())
      .def_rw("name", &Kernel::name).def_rw("num_blocks", &Kernel::num_blocks).def_rw("iters", &Kernel::iters)
      .def_rw("stages", &Kernel::stages).def_rw("resident", &Kernel::resident)
      .def_rw("consumers", &Kernel::consumers)
      .def_rw("body", &Kernel::body).def_rw("prologue", &Kernel::prologue).def_rw("epilogue", &Kernel::epilogue)
      .def_rw("fixed", &Kernel::fixed).def_rw("fixed_time", &Kernel::fixed_time)
      .def_rw("queue_coef", &Kernel::queue_coef).def_rw("queue_max", &Kernel::queue_max);
  nb::class_<Result>(m, "Result")
      .def_ro("time", &Result::time).def_ro("bottleneck", &Result::bottleneck)
      .def_ro("util", &Result::util).def_ro("breakdown", &Result::breakdown)
      .def_ro("limiter_time", &Result::limiter_time)
      .def_ro("limiter_detail", &Result::limiter_detail).def_ro("waves", &Result::waves);
  m.def("evaluate", &evaluate, nb::arg("kernel"), nb::arg("lanes"), nb::arg("sms"), nb::arg("launch_s"));
  m.def("evaluate_batch", &evaluate_batch, nb::arg("kernels"), nb::arg("lanes"), nb::arg("sms"),
        nb::arg("launch_s"), nb::arg("threads") = 8, nb::call_guard<nb::gil_scoped_release>());
  m.def("hit_prob", &hit_prob);
  m.def("expected_misses", &expected_misses, nb::call_guard<nb::gil_scoped_release>());
}
