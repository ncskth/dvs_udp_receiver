#include <arpa/inet.h>
#include <fstream>
#include <netinet/in.h>
#include <strings.h>
#include <sys/socket.h>
#include <unistd.h>
#include <netdb.h>
#include <stdexcept>

#include <torch/torch.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

int receive(c10::ArrayRef<bool> &array) {
  while (true) {
    array[0] = true;
  }
}

PYBIND11_MODULE(receiver, m) {
  m.doc() = "pybind11 example plugin"; // optional module docstring

  m.def("receive", &receive, "A function which adds two numbers");
}

using namespace std;

int main(int argc, char *argv[]) {
  int port = 2300;
  int sockfd;
  struct addrinfo hints, *servinfo, *p;
  int rv;

  // establish connection for client
  memset(&hints, 0, sizeof hints);
  hints.ai_family = AF_INET6; // set to AF_INET to use IPv4
  hints.ai_socktype = SOCK_DGRAM;
  hints.ai_flags = AI_PASSIVE; // use my IP

  // Get adrress-info
  if ((rv = getaddrinfo(NULL, port, &hints, &servinfo)) != 0) {
    fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rv));
    return 1;
  }

  // loop through all the results and bind to the first we can
  for (p = servinfo; p != NULL; p = p->ai_next) {
    if ((sockfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol)) == -1) {
      perror("listener: socket");
      continue;
    }

    if (bind(sockfd, p->ai_addr, p->ai_addrlen) == -1) {
      close(sockfd);
      perror("listener: bind");
      continue;
    }
    break;
  }

  if (p == NULL) {
    fprintf(stderr, "listener: failed to bind socket\n");
    return 2;
  }

  freeaddrinfo(servinfo);

  return sockfd;
}