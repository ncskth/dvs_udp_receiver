#ifndef CLIENT_HPP
#define CLIENT_HPP

#include <netdb.h>
#include <stdexcept>
#include <string>
#include <unistd.h>

int establish_connection(std::string port);

#endif