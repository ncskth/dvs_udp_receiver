#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>

#include <torch/torch.h>

#include "client.hpp"

#include <pybind11/pybind11.h>
namespace py = pybind11;

class TensorBuffer {
private:
  const std::vector<int64_t> shape;
  torch::TensorOptions options_buffer;
  torch::TensorOptions options_copy;

  std::mutex buffer_lock;
  std::shared_ptr<torch::Tensor> buffer1;
  std::shared_ptr<torch::Tensor> buffer2;

public:
  TensorBuffer(torch::IntArrayRef size, std::string device)
      : shape(size.vec()) {
    options_buffer = torch::TensorOptions()
                         .dtype(torch::kBool)
                         .device(torch::kCPU)
                         .memory_format(c10::MemoryFormat::Contiguous);
    options_copy = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    buffer1 =
        std::make_shared<torch::Tensor>(torch::zeros(size, options_buffer));
    buffer2 =
        std::make_shared<torch::Tensor>(torch::zeros(size, options_buffer));
  }
  void set_buffer(uint16_t data[], int numbytes) {
    const auto length = numbytes >> 1;
    buffer_lock.lock();
    for (int i = 0; i < length; i = i + 2) {
      // Decode x, y
      const uint16_t y_coord = data[i] & 0x7FFF;
      const uint16_t x_coord = data[i + 1] & 0x7FFF;
      bool *array = (bool *)buffer1->data_ptr();
      *(array + shape[1] * x_coord + y_coord) = true;
    }
    buffer_lock.unlock();
  }
  at::Tensor read() {
    // Swap out old pointer
    buffer_lock.lock();
    buffer1.swap(buffer2);
    buffer_lock.unlock();
    // Copy and clean
    auto copy = buffer2->to(options_copy, true, true);
    buffer2->index_put_({torch::indexing::Slice()}, false);
    return copy;
  }
};

class UDPStream {
private:
  TensorBuffer buffer;
  const int port;
  const int max_events_per_packet = 1024;

  std::thread socket_thread;
  std::atomic<bool> is_serving = {true};

  void start_server() {
    std::thread socket_thread(&UDPStream::serve_synchronous, this);
    socket_thread.detach();
  }

public:
  uint64_t count = 0;
  UDPStream(int port, torch::IntArrayRef size, std::string device)
      : buffer(size, device), port(port) {
    start_server();
  }

  at::Tensor read() { return buffer.read(); }

  void serve_synchronous() {
    int sockfd;
    int numbytes;
    uint16_t int_buf[max_events_per_packet];

    // Connect to socket
    struct sockaddr_storage their_addr;
    socklen_t addr_len;
    sockfd = establish_connection(std::to_string(port));
    addr_len = sizeof(their_addr);

    // start receiving event
    while (is_serving.load()) {
      if ((numbytes = recvfrom(sockfd, int_buf, sizeof(int_buf), 0,
                               (struct sockaddr *)&their_addr, &addr_len)) ==
          -1) {
        perror("recvfrom");
        return;
      }
      count += numbytes / 4;

      buffer.set_buffer(int_buf, numbytes);
    }
    close(sockfd);
  }

  void stop_server() {
    is_serving.store(false);
    printf("Total events: %lu\n", count);
  }
};

PYBIND11_MODULE(udpstream, m) {
  py::class_<UDPStream>(m, "UDPStream")
      .def(py::init<int, torch::IntArrayRef, std::string>())
      .def("read", &UDPStream::read)
      .def("stop_server", &UDPStream::stop_server);
}

// using namespace std::chrono_literals;

// int main(int argc, char *argv[])
// {

//   std::string port_cli;

//   auto stream = UDPStream(2300, {640, 480}, "cuda:1");

//   for (int i = 0; i < 1000; i++)
//   {
//     std::this_thread::sleep_for(1s);
//     auto r = stream.read();
//     printf("%ldx%ld:%ld\n", r.size(0), r.size(1), r.sum().item<int64_t>());
//   }
//   stream.stop_server();
// }
