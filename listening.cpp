#include "CLI11.hpp"
#include "client.hpp"
#include <assert.h>
#include <chrono>
#include <cstring>
#include <thread>

#include <torch/torch.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

class TensorBuffer {
private:
  torch::TensorOptions options_buffer;
  torch::TensorOptions options_copy;
  at::Tensor base;

public:
  std::shared_ptr<at::Tensor> buffer;

  TensorBuffer(torch::IntArrayRef size, c10::DeviceType device) {
    options_buffer =
        torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
    options_copy = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    at::Tensor array = torch::zeros(size, options_buffer);
    buffer = std::make_shared<at::Tensor>(array);
    base = torch::zeros(size, options_buffer);
  }
  at::Tensor read() {
    auto empty = base.clone();
    auto copy = (*buffer).to(options_copy, true, true);
    buffer.reset(&empty);
    return copy;
  }
};

class UDPStream {
private:
  TensorBuffer buffer;

public:
  UDPStream(torch::IntArrayRef size, c10::DeviceType device)
      : buffer(size, device) {}

  at::Tensor read() { return buffer.read(); }

  void start_server(const char *port, int max_events_per_packet) {

    int numbytes, x_coord, y_coord;
    u_int16_t int_buf[max_events_per_packet];

    // Connect to socket
    struct sockaddr_storage their_addr;
    socklen_t addr_len;
    const int sockfd = establish_connection(port);
    addr_len = sizeof(their_addr);

    // start receiving event
    while (true) {
      if ((numbytes = recvfrom(sockfd, int_buf, sizeof(int_buf), 0,
                               (struct sockaddr *)&their_addr, &addr_len)) ==
          -1) {
        perror("recvfrom");
        exit(1);
      }

      for (int i = 0; i < numbytes / 2; i = i + 2) {
        // Decode x, y
        x_coord = int_buf[i] & 0x7FFF;
        y_coord = int_buf[i + 1] & 0x7FFF;
        (*buffer.buffer)[y_coord * 480 + x_coord] = true;
      }
    }

    close(sockfd);
  }
};

// PYBIND11_MODULE(receiver, m) {
//   m.doc() = "pybind11 example plugin"; // optional module docstring

//   m.def("receive", &receive, "A function which adds two numbers");
// }

struct PolarityEvent {
  uint64_t timestamp : 64;
  uint16_t x : 15;
  uint16_t y : 15;
} __attribute__((packed));

int main(int argc, char *argv[]) {

  CLI::App app{"Listen to DVS data from UDP socket"};

  std::string port_cli;
  int max_events_per_packet = 512;

  app.add_option("port_number", port_cli, "Port number")->required();

  CLI11_PARSE(app, argc, argv);

  UDPStream({640 * 480}, c10::kCUDA)
      .start_server(port_cli.c_str(), max_events_per_packet);
}
