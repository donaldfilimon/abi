module;
#include <iostream>

module Cell.Core;

namespace Cell {
    void Engine::run() {
        std::cout << "Cell Engine running!" << std::endl;
        std::cout << "2 + 2 = " << add(2, 2) << std::endl;
    }
}
