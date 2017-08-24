#include <cstdlib>
#include <iostream>
#include <utility>
#include <boost/asio.hpp>
#include <fstream>
#include <string>

using boost::asio::ip::tcp;

const int max_length = 1024;

void model_server(boost::asio::io_service& io_service, unsigned short port, char *server_prediction_model){
	while(true){
		tcp::acceptor a(io_service, tcp::endpoint(tcp::v4(), port));
		tcp::socket sock(io_service);
		a.accept(sock);
		std::cout << "Client connected" << std::endl;
		try{
			std::ifstream input(server_prediction_model, std::ios::in);
			std::string Data = "";
			std::string s;
			while(getline(input, s)){
				Data += s;
				Data += ' ';
			}
			size_t length = Data.length()-1;
			char test[4] = {0, };
			size_t tmp = length;
			for(int i=0; i<4; i++){
				test[3-i] = tmp % 128;
				tmp /= 128;
			}
			boost::system::error_code error;

			boost::asio::write(sock, boost::asio::buffer(test, 4));
			boost::asio::write(sock, boost::asio::buffer(Data, length));
		}
		catch(std::exception& e){
			std::cerr << "Exception in thread: " << e.what() << std::endl;
		}
		std::cout << "Finished sending prediction model" << std::endl;
	}
}

int main(int argc, char* argv[]){
	try{
		if(argc != 3){
			std::cerr << "Usage: server <port> <server_prediction_model.txt>" << std::endl;
			return 1;
		}

		boost::asio::io_service io_service;

		model_server(io_service, std::atoi(argv[1]), argv[2]);
	}
	catch(std::exception& e){
		std::cerr << "Exception: " << e.what() << std::endl;
	}
	return 0;
}
