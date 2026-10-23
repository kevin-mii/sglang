// Torch bindings for the gfx950 fp8 MLA prefill kernel (mla_fp8.hip): mla_fp8._C.
//
//   fwd_varlen  q [Tq, H, 192] bf16, k [Tk, HKv, 192] / v [Tk, HKv, 128] fp8 e4m3fn -> o [Tq, H, 128] bf16
//               (lse [H, Tq] fp32 on request); sequence i spans q tokens cu_q[i] .. cu_q[i + 1] - 1 and k/v
//               tokens likewise; causal is bottom-right aligned. scale: softmax scale; k_scale / v_scale:
//               the fp8 dequant scales. Any token and head strides with a contiguous head dim.
//   VarlenPlan  fwd_varlen planned once per forward pass (plan), run by every layer (run).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime_api.h>
#include <torch/extension.h>

#include <array>
#include <optional>
#include <stdexcept>
#include <string>

extern "C" {
int mla_fp8_varlen_fwd_dev(const void* q, const void* k, const void* v, void* o, float* lse, const int64_t* strides,
                           const int* cu_q, const int* cu_k, const int* host_cu_q, const int* host_cu_k,
                           int num_seqs, int max_q, int max_k, int total_q, int total_k, int H, int HKv, int causal,
                           float scale, float k_scale, float v_scale, hipStream_t stream);
void* mla_fp8_plan_new();
void mla_fp8_plan_free(void* p);
int mla_fp8_plan(void* plan, const int* cu_q, const int* cu_k, const int* host_cu_q, const int* host_cu_k,
                 int num_seqs, int max_q, int max_k, int total_q, int total_k, int H, int HKv, int causal,
                 hipStream_t stream);
int mla_fp8_run(const void* plan, const void* q, const void* k, const void* v, void* o, float* lse,
                const int64_t* strides, float scale, float k_scale, float v_scale, hipStream_t stream);
}

namespace {

void check(bool ok, const std::string& what) {
  if (!ok) throw std::invalid_argument("mla_fp8: " + what);
}

struct Call {
  at::Tensor o;
  std::optional<at::Tensor> lse;
  std::array<int64_t, 15> strides;
};

// Validates the tensors and builds the output, lse and the strides array (q, k, v, o, lse).
Call prepare(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, std::optional<at::Tensor> out,
             bool return_lse) {
  check(q.is_cuda() && k.is_cuda() && v.is_cuda(), "tensors must be on the GPU");
  check(q.scalar_type() == at::kBFloat16, "q must be bf16");
  check(k.scalar_type() == at::kFloat8_e4m3fn && v.scalar_type() == at::kFloat8_e4m3fn, "k and v must be fp8 e4m3fn");
  check(q.dim() == 3 && k.dim() == 3 && v.dim() == 3, "q, k, v must be [tokens, heads, d]");
  check(q.size(2) == 192 && k.size(2) == 192 && v.size(2) == 128, "head dims must be q/k 192, v 128");
  check(k.size(0) == v.size(0) && k.size(1) == v.size(1), "k and v must match in tokens and heads");
  check(q.stride(2) == 1 && k.stride(2) == 1 && v.stride(2) == 1, "head dims must be contiguous");
  const int64_t Tq = q.size(0), H = q.size(1);
  at::Tensor o = out ? *out : at::empty({Tq, H, 128}, q.options());
  check(o.scalar_type() == at::kBFloat16 && o.size(0) == Tq && o.size(1) == H && o.size(2) == 128 && o.stride(2) == 1,
        "out must be bf16 [Tq, H, 128] with a contiguous head dim");
  std::optional<at::Tensor> lse;
  if (return_lse) lse = at::empty({H, Tq}, q.options().dtype(at::kFloat));
  Call c{o, lse, {}};
  auto put = [&](int i, const at::Tensor& t) { c.strides[3 * i] = 0, c.strides[3 * i + 1] = t.stride(0), c.strides[3 * i + 2] = t.stride(1); };
  put(0, q), put(1, k), put(2, v), put(3, o);
  if (lse) c.strides[12] = 0, c.strides[13] = 1, c.strides[14] = Tq;
  return c;
}

const int* host_ints(const std::optional<at::Tensor>& t) {
  if (!t) return nullptr;
  check(!t->is_cuda() && t->scalar_type() == at::kInt && t->is_contiguous(), "host cu_seqlens must be int32 CPU tensors");
  return t->data_ptr<int>();
}

py::object result(const Call& c) {
  if (c.lse) return py::make_tuple(c.o, *c.lse);
  return py::cast(c.o);
}

py::object fwd_varlen(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& cu_q,
                      const at::Tensor& cu_k, int64_t max_q, int64_t max_k, bool causal, double scale,
                      double k_scale, double v_scale, std::optional<at::Tensor> host_cu_q,
                      std::optional<at::Tensor> host_cu_k, std::optional<at::Tensor> out, bool return_lse) {
  const c10::cuda::CUDAGuard guard(q.device());
  Call c = prepare(q, k, v, out, return_lse);
  check(cu_q.is_cuda() && cu_k.is_cuda() && cu_q.scalar_type() == at::kInt && cu_k.scalar_type() == at::kInt &&
            cu_q.numel() == cu_k.numel() && cu_q.numel() >= 1,
        "cu_seqlens must be int32 GPU tensors of num_seqs + 1");
  const int n = (int)cu_q.numel() - 1;
  const int rc = mla_fp8_varlen_fwd_dev(q.data_ptr(), k.data_ptr(), v.data_ptr(), c.o.data_ptr(),
                                        c.lse ? c.lse->data_ptr<float>() : nullptr, c.strides.data(),
                                        cu_q.data_ptr<int>(), cu_k.data_ptr<int>(), host_ints(host_cu_q),
                                        host_ints(host_cu_k), n, (int)max_q, (int)max_k, (int)q.size(0),
                                        (int)k.size(0), (int)q.size(1), (int)k.size(1), causal, (float)scale,
                                        (float)k_scale, (float)v_scale, at::cuda::getCurrentCUDAStream().stream());
  check(rc == 0, "launch failed (" + std::to_string(rc) + ")");
  return result(c);
}

class VarlenPlan {
 public:
  VarlenPlan() : h_(mla_fp8_plan_new()) {}
  VarlenPlan(const VarlenPlan&) = delete;
  VarlenPlan& operator=(const VarlenPlan&) = delete;
  ~VarlenPlan() { close(); }
  VarlenPlan& plan(const at::Tensor& cu_q, const at::Tensor& cu_k, int64_t max_q, int64_t max_k, int64_t total_q,
                   int64_t total_k, int64_t heads, int64_t kv_heads, bool causal,
                   std::optional<at::Tensor> host_cu_q, std::optional<at::Tensor> host_cu_k) {
    check(h_ != nullptr, "VarlenPlan is closed");
    const c10::cuda::CUDAGuard guard(cu_q.device());
    const int n = (int)cu_q.numel() - 1;
    const int rc = mla_fp8_plan(h_, cu_q.data_ptr<int>(), cu_k.data_ptr<int>(), host_ints(host_cu_q),
                                host_ints(host_cu_k), n, (int)max_q, (int)max_k, (int)total_q, (int)total_k,
                                (int)heads, (int)kv_heads, causal, at::cuda::getCurrentCUDAStream().stream());
    check(rc == 0, "plan failed (" + std::to_string(rc) + ")");
    return *this;
  }
  py::object run(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, double scale, double k_scale,
                 double v_scale, std::optional<at::Tensor> out, bool return_lse) {
    check(h_ != nullptr, "VarlenPlan is closed");
    const c10::cuda::CUDAGuard guard(q.device());
    Call c = prepare(q, k, v, out, return_lse);
    const int rc = mla_fp8_run(h_, q.data_ptr(), k.data_ptr(), v.data_ptr(), c.o.data_ptr(),
                               c.lse ? c.lse->data_ptr<float>() : nullptr, c.strides.data(), (float)scale,
                               (float)k_scale, (float)v_scale, at::cuda::getCurrentCUDAStream().stream());
    check(rc == 0, "run failed (" + std::to_string(rc) + ")");
    return result(c);
  }
  void close() {
    if (h_) mla_fp8_plan_free(h_), h_ = nullptr;
  }

 private:
  void* h_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fwd_varlen", &fwd_varlen, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("cu_q"), py::arg("cu_k"),
        py::arg("max_q"), py::arg("max_k"), py::arg("causal"), py::arg("scale"), py::arg("k_scale") = 1.0,
        py::arg("v_scale") = 1.0, py::arg("host_cu_q") = py::none(), py::arg("host_cu_k") = py::none(),
        py::arg("out") = py::none(), py::arg("return_lse") = false);
  py::class_<VarlenPlan>(m, "VarlenPlan")
      .def(py::init<>())
      .def("plan", &VarlenPlan::plan, py::arg("cu_q"), py::arg("cu_k"), py::arg("max_q"), py::arg("max_k"),
           py::arg("total_q"), py::arg("total_k"), py::arg("heads"), py::arg("kv_heads"), py::arg("causal"),
           py::arg("host_cu_q") = py::none(), py::arg("host_cu_k") = py::none(), py::return_value_policy::reference)
      .def("run", &VarlenPlan::run, py::arg("q"), py::arg("k"), py::arg("v"), py::arg("scale"),
           py::arg("k_scale") = 1.0, py::arg("v_scale") = 1.0, py::arg("out") = py::none(),
           py::arg("return_lse") = false)
      .def("close", &VarlenPlan::close);
}
